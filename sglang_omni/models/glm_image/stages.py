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
import time
from functools import lru_cache

import torch

from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.glmimage import GlmImageSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
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
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.image_payload import image_pixels_payload

from typing import Any

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
    """
    Load one checkpoint component through sglang's own loader.
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


def _to_device(value, device):
    """Put a carried tensor back on the device the stage computes on.

    store_state moves every tensor to CPU so a payload can cross a process
    boundary, and upstream stages assume their inputs are already placed.
    Conditioning hands prompt_embeds on as a one-element list, which the DiT
    unwraps itself, so the container has to survive the move.
    """
    if device is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        return type(value)(_to_device(item, device) for item in value)
    return value


# Everything a stage produces for a later one. All of it is tensor-shaped, so
# the terminal stage clears it: the completion message to the coordinator goes
# through plain msgpack, which cannot pack a Tensor even on CPU.
_CARRIED_FIELDS = (
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
)


def _clear_carried(state: GLMImageState) -> GLMImageState:
    for name in _CARRIED_FIELDS:
        setattr(state, name, None)
    return state


def _to_req(
    state: GLMImageState,
    *,
    device=None,
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
        seed=state.seed if state.seed is not None else C.DEFAULT_SEED,
        num_outputs_per_prompt=state.num_outputs,
        num_inference_steps=state.num_inference_steps or num_inference_steps,
        guidance_scale=state.guidance_scale or guidance_scale,
    )
    req = Req(sampling_params=sampling_params)
    # Req delegates unknown names to sampling_params, so only set what the
    # stage actually reads.
    for name in _CARRIED_FIELDS:
        value = getattr(state, name, None)
        if value is not None:
            setattr(req, name, _to_device(value, device))
    return req


def _record_usage(state: GLMImageState) -> GLMImageState:
    """Count tokens where they first appear, so no stage has to know its place.

    Timing is left to StageProfiler, which reports per stage rather than as one
    sum and drains the device queue so a stage is charged for its own kernels.
    """
    if not state.completion_tokens and state.prior_token_id is not None:
        state.completion_tokens = int(state.prior_token_id.numel())
    if not state.prompt_tokens:
        # Req defaults prompt_embeds to [], and stages before conditioning
        # carry that empty list rather than None.
        embeds = state.prompt_embeds
        if isinstance(embeds, (list, tuple)):
            embeds = embeds[0] if embeds else None
        if embeds is not None and embeds.dim() >= 2:
            state.prompt_tokens = int(embeds.shape[-2])
    return state


# ===== stage factories =====
def create_srt_ar_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 1,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
):
    """Returns OmniScheduler for the GLM-Image AR engine."""
    from sglang_omni.models.glm_image.engine_builder import (
        GLMImageEngineBuilder,
    )

    return GLMImageEngineBuilder(
        max_concurrency=max_concurrency,
        tp_rank=tp_rank,
        tp_size=tp_size,
        nccl_port=nccl_port,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_dit_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    num_inference_steps: int = C.DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = C.DEFAULT_GUIDANCE_SCALE,
    max_concurrency: int = 1,
) -> SimpleScheduler:
    """Conditioning, DiT sampling and VAE decode of one request in one stage."""
    from sglang_omni.models.glm_image.diffusion import GLMImageDiffusion

    _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    # sglang multimodal_gen places modules on npu:{LOCAL_RANK}, not on our device;
    # drop once each stage process sees only its own card as index 0.
    os.environ["LOCAL_RANK"] = str(device.index)
    _, _, server_args = _bootstrap(model_path)

    transformer = _load_component(model_path, "transformer")
    scheduler = _load_component(model_path, "scheduler")
    vae = _load_component(model_path, "vae")
    diffusion = GLMImageDiffusion(
        before_denoising=GlmImageBeforeDenoisingStage(
            tokenizer=_load_component(model_path, "tokenizer"),
            text_encoder=_load_component(model_path, "text_encoder"),
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        ),
        denoising=_attach_residency(
            DenoisingStage(transformer=transformer, scheduler=scheduler),
            DENOISING_STAGE,
            server_args,
            transformer=transformer,
        ),
        decoding=GlmImageDecodingStage(vae=vae),
        server_args=server_args,
    )

    @torch.inference_mode()
    def compute(payload):
        state = load_state(payload, GLMImageState)
        req = _to_req(
            state,
            device=device,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        output_batch = diffusion.run(req)
        _record_usage(state)
        payload = store_state(payload, _clear_carried(state))
        payload.data.update(
            image_pixels_payload(output_batch.output, source_hint="GLM-Image"),
            usage=build_usage(state),
        )
        if state.request_started_at:
            # Same wording as sglang's log_generation_timer, and the same span:
            # from the pipeline taking the request to the output being ready.
            logger.info(
                "Pixel data generated successfully in %.2f seconds",
                time.time() - state.request_started_at,
            )
        return payload

    return SimpleScheduler(compute, max_concurrency=max_concurrency)
