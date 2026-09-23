# SPDX-License-Identifier: Apache-2.0
"""GLM-Image stage factories.

Each factory wraps the matching sglang stage: sglang's own component loaders
build the modules, its stage object holds the logic, and the omni payload is
converted to a sglang ``Req`` and back around every call. The loaders matter
beyond convenience -- they resolve the DiT through sglang's ModelRegistry, and
only that implementation carries the rotary_emb and kv_caches interface the
denoising stage drives.

Text-to-image only. The image-conditioned path additionally runs the DiT to
fill a reference KV cache, which the payload cannot carry between stages.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageBeforeDenoisingStage,
    GlmImageDecodingStage,
)
from sglang.multimodal_gen.runtime.server_args.server_args import (
    ServerArgs,
    set_global_server_args,
)

from sglang_omni.models.glm_image import constants as C
from sglang_omni.models.glm_image.config import DENOISING_STAGE
from sglang_omni.models.glm_image.checkpoint import load_json, resolve_checkpoint
from sglang_omni.models.glm_image.hf_config import make_runtime_config
from sglang_omni.models.glm_image.payload_types import GLMImageState
from sglang_omni.scheduling.pipeline_state import load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(*, field: str, name: str) -> torch.dtype:
    if name not in _TORCH_DTYPES:
        raise ValueError(
            f"GLM-Image {field} must be one of {', '.join(_TORCH_DTYPES)}, got {name!r}"
        )
    return _TORCH_DTYPES[name]


# ===== sglang runtime bootstrap =====


@lru_cache(maxsize=None)
def _bootstrap(model_path: str):
    """Stand up the sglang globals a wrapped stage needs, once per process.

    ``PipelineStage.__init__`` reads ``get_global_server_args()``, and
    ``DenoisingStage.__init__`` reaches further into
    ``pipeline_config.dit_config``, so the architecture has to be merged in
    before any stage object is constructed.
    """
    paths = resolve_checkpoint(model_path)
    config = make_runtime_config(model_path)

    pipeline_config = GlmImagePipelineConfig()
    pipeline_config.dit_config.update_model_arch(
        {k: v for k, v in config.transformer.items() if not k.startswith("_")}
    )

    server_args = ServerArgs(model_path=str(paths.root))
    server_args.pipeline_config = pipeline_config
    # Offload defaults to on for the text encoder, and upstream relies on a
    # component residency manager to bring a component back before use. These
    # stages run without one, so every component stays resident.
    server_args.text_encoder_cpu_offload = False
    server_args.image_encoder_cpu_offload = False
    server_args.vae_cpu_offload = False
    server_args.dit_cpu_offload = False
    set_global_server_args(server_args)
    return paths, config, server_args


@lru_cache(maxsize=None)
def _init_parallel() -> None:
    """Stand up one-process parallel groups, which sglang's DiT layers need.

    Its ColumnParallelLinear and USPAttention read the TP and SP groups while
    being constructed, and the forward calls get_sp_world_size() unguarded.
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@lru_cache(maxsize=None)
def _load_component(model_path: str, name: str):
    """Load one checkpoint component through sglang's own loader.

    The library and class come from model_index.json, so the DiT resolves
    through sglang's ModelRegistry rather than to the diffusers class: only
    sglang's carries the rotary_emb and kv_caches interface DenoisingStage
    drives. Cached per name, so stages sharing a process share the module --
    the scheduler in particular must be one instance, because conditioning
    configures its resolution-dependent sigma schedule and denoising reads it
    back off the Req.
    """
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        PipelineComponentLoader,
    )

    _init_parallel()
    paths, _, server_args = _bootstrap(model_path)
    library, architecture = load_json(paths.model_index)[name]
    module, _ = PipelineComponentLoader.load_component(
        component_name=name,
        component_model_path=str(getattr(paths, name)),
        transformers_or_diffusers=library,
        server_args=server_args,
        component_architecture=architecture,
    )
    return module


class _PipelineView:
    """The three attributes ComponentResidencyPipeline asks of a pipeline.

    It is a Protocol, so this satisfies it structurally. One view per process
    collects stages and modules as the factories build them.
    """

    def __init__(self) -> None:
        self.modules: dict[str, object] = {}
        self._stage_name_mapping: dict[str, object] = {}
        self.component_residency_strategies: dict[str, object] = {}


_PIPELINE_VIEW = _PipelineView()


def _attach_residency(stage, stage_name: str, server_args, **modules):
    """Give a stage sglang's residency manager.

    DenoisingStage._manage_dit_use_site dereferences the manager with no None
    check, unlike every other call site. With the offload flags off it resolves
    ResidentStrategy per component and moves nothing.
    """
    from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
        get_global_component_residency_manager,
    )

    _PIPELINE_VIEW.modules.update(modules)
    _PIPELINE_VIEW._stage_name_mapping[stage_name] = stage
    stage.set_component_residency_manager(
        get_global_component_residency_manager(_PIPELINE_VIEW, server_args)
    )
    return stage


# ===== payload <-> Req conversion =====


def _to_req(
    state: GLMImageState,
    *,
    num_inference_steps: int = C.DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = C.DEFAULT_GUIDANCE_SCALE,
) -> Req:
    """Build the sglang request a wrapped stage expects from omni state.

    Zero means "the request did not ask", so the stage's own default wins.
    GlmImageSamplingParams refuses a non-positive step count outright, and
    validate() derives do_classifier_free_guidance from the scale, so both
    have to be real values by the time the Req is built.
    """
    sampling_params = GlmImageSamplingParams(
        prompt=state.prompt,
        width=state.width or None,
        height=state.height or None,
        seed=state.seed if state.seed is not None else 42,
        num_outputs_per_prompt=state.num_outputs,
        num_inference_steps=state.num_inference_steps or num_inference_steps,
        guidance_scale=state.guidance_scale or guidance_scale,
    )
    req = Req(sampling_params=sampling_params)
    # Fields the earlier stages already produced. Req delegates unknown names
    # to sampling_params, so only set what the stage actually reads.
    for name in (
        "prior_token_id",
        "prompt_embeds",
        "negative_prompt_embeds",
        "latents",
        "raw_latent_shape",
        "timesteps",
        "target_size",
        "crop_coords",
        "prior_token_drop_cond",
        "prior_token_drop_uncond",
    ):
        value = getattr(state, name, None)
        if value is not None:
            setattr(req, name, value)
    return req


def _from_req(req: Req, state: GLMImageState) -> GLMImageState:
    """Copy back the fields the stage produced."""
    for name in (
        "prior_token_id",
        "prompt_embeds",
        "negative_prompt_embeds",
        "latents",
        "raw_latent_shape",
        "timesteps",
        "target_size",
        "crop_coords",
        "prior_token_drop_cond",
        "prior_token_drop_uncond",
        "width",
        "height",
        "requested_width",
        "requested_height",
    ):
        value = getattr(req, name, None)
        if value is not None:
            setattr(state, name, value)
    return state


def _run(stage, server_args, **req_defaults):
    """Wrap one sglang stage as an omni compute function.

    ``req_defaults`` carries the stage's FactoryArgs knobs into the Req, which
    is where the wrapped sglang logic reads them from.
    """

    @torch.inference_mode()
    def compute(payload):
        state = load_state(payload, GLMImageState)
        req = _to_req(state, **req_defaults)
        # DenoisingStage reads batch.scheduler, not self.scheduler, and the
        # instance carries the schedule conditioning configured on it.
        scheduler = getattr(stage, "scheduler", None)
        if scheduler is not None:
            req.scheduler = scheduler
        return store_state(payload, _from_req(stage.forward(req, server_args), state))

    return compute


# ===== stage factories =====


def create_ar_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_concurrency: int = 1,
) -> SimpleScheduler:
    """AR stage: the VLM turns the prompt into prior tokens."""
    _resolve_dtype(field="dtype", name=dtype)
    resolve_concrete_device(device, gpu_id)
    _, _, server_args = _bootstrap(model_path)

    stage = GlmImageAR(
        processor=_load_component(model_path, "processor"),
        vision_language_encoder=_load_component(model_path, "vision_language_encoder"),
    )
    return SimpleScheduler(_run(stage, server_args), max_concurrency=max_concurrency)


def create_before_denoising_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    num_inference_steps: int = C.DEFAULT_NUM_INFERENCE_STEPS,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    """Conditioning stage: ByT5 glyph embeds, initial noise, timestep schedule."""
    _resolve_dtype(field="dtype", name=dtype)
    resolve_concrete_device(device, gpu_id)
    _, _, server_args = _bootstrap(model_path)

    # The DiT is here for its config only on the text-to-image path; the cache
    # hands back the same module the denoising stage runs, not a second copy.
    stage = GlmImageBeforeDenoisingStage(
        tokenizer=_load_component(model_path, "tokenizer"),
        text_encoder=_load_component(model_path, "text_encoder"),
        vae=_load_component(model_path, "vae"),
        transformer=_load_component(model_path, "transformer"),
        scheduler=_load_component(model_path, "scheduler"),
    )

    return SimpleScheduler(
        _run(stage, server_args, num_inference_steps=num_inference_steps),
        max_concurrency=max_concurrency,
    )


def create_denoising_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    guidance_scale: float = C.DEFAULT_GUIDANCE_SCALE,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    """Denoising stage: the DiT sampling loop with classifier-free guidance."""
    _resolve_dtype(field="dtype", name=dtype)
    resolve_concrete_device(device, gpu_id)
    _, _, server_args = _bootstrap(model_path)

    transformer = _load_component(model_path, "transformer")
    stage = _attach_residency(
        DenoisingStage(
            transformer=transformer,
            scheduler=_load_component(model_path, "scheduler"),
        ),
        DENOISING_STAGE,
        server_args,
        transformer=transformer,
    )
    return SimpleScheduler(
        _run(stage, server_args, guidance_scale=guidance_scale),
        max_concurrency=max_concurrency,
    )


def create_decode_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    max_concurrency: int = 1,
) -> SimpleScheduler:
    """Decode stage: VAE decode, then crop back to the requested canvas."""
    _resolve_dtype(field="dtype", name=dtype)
    resolve_concrete_device(device, gpu_id)
    _, _, server_args = _bootstrap(model_path)

    stage = GlmImageDecodingStage(vae=_load_component(model_path, "vae"))

    @torch.inference_mode()
    def compute(payload):
        state = load_state(payload, GLMImageState)
        output_batch = stage.forward(_to_req(state), server_args)
        payload = store_state(payload, state)
        # TODO(you): omni has no image output contract yet -- every model here
        # emits modality="audio" or "text", and sglang_omni/utils has no image
        # payload helper. Decide the response shape (PNG bytes? base64?) and
        # write output_batch.output into payload.data accordingly.
        payload.data.update(modality="image")
        return payload

    return SimpleScheduler(compute, max_concurrency=max_concurrency)
