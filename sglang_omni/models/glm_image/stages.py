# SPDX-License-Identifier: Apache-2.0
"""GLM-Image stage factories.

Each factory wraps the matching sglang stage: the model modules are loaded
with ``from_pretrained`` here, sglang's stage object holds the logic, and the
omni payload is converted to a sglang ``Req`` and back around every call.

Text-to-image only. The image-conditioned path additionally runs the DiT to
fill a reference KV cache, which the payload cannot carry between stages.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from types import SimpleNamespace

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
from sglang_omni.models.glm_image.checkpoint import resolve_checkpoint
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
    set_global_server_args(server_args)
    return paths, config, server_args


@lru_cache(maxsize=None)
def _load_scheduler(scheduler_path: str):
    """One scheduler instance shared by conditioning and denoising.

    Conditioning configures the sigma schedule on it (resolution-dependent mu)
    and denoising reads it back off the Req, so a second instance would sample
    on an unconfigured schedule. Its step index is mutable state, which holds
    only while both stages stay at max_concurrency=1 in one process.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(scheduler_path)


class _TransformerConfigView:
    """The DiT's config for a stage that reads it but never calls the module.

    T2I conditioning only reads in_channels, num_layers and patch_size off the
    transformer; I2I also runs it to fill the reference KV cache (upstream
    stage lines 1305-1334) and needs the real module instead.
    """

    def __init__(self, transformer_config: dict, dtype: torch.dtype):
        # hasattr(config, "sample_size") must stay False: upstream falls back
        # to 128, and GLM-Image checkpoints do not ship the key.
        self.config = SimpleNamespace(
            **{k: v for k, v in transformer_config.items() if not k.startswith("_")}
        )
        self.dtype = dtype


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
    from transformers import GlmImageForConditionalGeneration, GlmImageProcessor

    torch_dtype = _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    paths, _, server_args = _bootstrap(model_path)

    processor = GlmImageProcessor.from_pretrained(str(paths.processor))
    vlm = (
        GlmImageForConditionalGeneration.from_pretrained(
            str(paths.vision_language_encoder), torch_dtype=torch_dtype
        )
        .to(device)
        .eval()
    )
    stage = GlmImageAR(processor=processor, vision_language_encoder=vlm)
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
    from diffusers import AutoencoderKL
    from transformers import AutoTokenizer, T5EncoderModel

    torch_dtype = _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    paths, config, server_args = _bootstrap(model_path)

    tokenizer = AutoTokenizer.from_pretrained(str(paths.tokenizer))
    text_encoder = (
        T5EncoderModel.from_pretrained(str(paths.text_encoder), torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )
    vae = (
        AutoencoderKL.from_pretrained(str(paths.vae), torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )
    scheduler = _load_scheduler(str(paths.scheduler))

    stage = GlmImageBeforeDenoisingStage(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=_TransformerConfigView(config.transformer, torch_dtype),
        scheduler=scheduler,
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
    from diffusers import GlmImageTransformer2DModel

    torch_dtype = _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    paths, _, server_args = _bootstrap(model_path)

    transformer = (
        GlmImageTransformer2DModel.from_pretrained(
            str(paths.transformer), torch_dtype=torch_dtype
        )
        .to(device)
        .eval()
    )
    scheduler = _load_scheduler(str(paths.scheduler))

    stage = DenoisingStage(transformer=transformer, scheduler=scheduler)
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
    from diffusers import AutoencoderKL

    torch_dtype = _resolve_dtype(field="dtype", name=dtype)
    device = resolve_concrete_device(device, gpu_id)
    paths, _, server_args = _bootstrap(model_path)

    vae = (
        AutoencoderKL.from_pretrained(str(paths.vae), torch_dtype=torch_dtype)
        .to(device)
        .eval()
    )
    stage = GlmImageDecodingStage(vae=vae)

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
