# SPDX-License-Identifier: Apache-2.0
"""GLM-Image stage factories.

The AR stage runs the vision-language model on sglang's srt engine (see
engine_builder.py). The DiT stage builds sglang's own conditioning, denoising
and decoding stages and runs them back to back on one Req (see diffusion.py).
Its modules come from sglang's component loaders, which resolve the DiT
through sglang's ModelRegistry; only that implementation carries the
rotary_emb and kv_caches interface DenoisingStage drives.

Text-to-image only: the AR request builder carries no source image.
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
    get_global_server_args,
)

from sglang_omni.models.glm_image import constants as C
from sglang_omni.models.glm_image.checkpoint import load_json, resolve_checkpoint
from sglang_omni.models.glm_image.payload_types import GLMImageState
from sglang_omni.scheduling.pipeline_state import build_usage, load_state, store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.image_payload import image_pixels_payload

from typing import Any

logger = logging.getLogger(__name__)


# ===== sglang runtime bootstrap =====


@lru_cache(maxsize=None)
def _bootstrap(
    model_path: str,
    *,
    num_gpus: int,
    tp_size: int,
    sp_degree: int,
    ulysses_degree: int,
    ring_degree: int,
) -> ServerArgs:
    """Stand up the sglang globals a wrapped stage needs, once per process.

    ``PipelineStage.__init__`` reads ``get_global_server_args()``, and
    ``DenoisingStage.__init__`` reaches further into
    ``pipeline_config.dit_config``, so the architecture has to be merged in
    before any stage object is constructed. ``tp_size`` is the DiT's own TP
    degree; ``num_gpus`` is every rank of the stage.
    """
    paths = resolve_checkpoint(model_path)
    transformer_config = load_json(paths.transformer / "config.json")

    pipeline_config = GlmImagePipelineConfig()
    pipeline_config.dit_config.update_model_arch(
        {k: v for k, v in transformer_config.items() if not k.startswith("_")}
    )

    # CFG and data parallelism stay off: sglang would otherwise turn CFG
    # parallel on by itself once num_gpus > 1. Autocast is off for GLM-Image,
    # but sglang resolves that from pipeline_config while constructing
    # ServerArgs, and the config below is only attached afterwards.
    server_args = ServerArgs(
        model_path=str(paths.root),
        num_gpus=num_gpus,
        tp_size=tp_size,
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        cfg_parallel_degree=1,
        dp_size=1,
        disable_autocast=True,
    )
    server_args.pipeline_config = pipeline_config
    # Offload defaults to on for the text encoder, and upstream relies on a
    # component residency manager to bring a component back before use. These
    # stages run without one, so every component stays resident.
    server_args.text_encoder_cpu_offload = False
    server_args.image_encoder_cpu_offload = False
    server_args.vae_cpu_offload = False
    server_args.dit_cpu_offload = False
    set_global_server_args(server_args)
    return server_args


@lru_cache(maxsize=None)
def _init_parallel(
    *,
    tp_rank: int,
    tp_size: int,
    device: torch.device,
    nccl_port: int | None,
    tp_dit: int,
    sp_degree: int,
    ulysses_degree: int,
    ring_degree: int,
) -> None:
    """Stand up the distributed groups sglang's DiT layers need, once per process.

    Its ColumnParallelLinear and USPAttention read the TP and SP groups while
    being constructed, so this has to run before the first component loads.
    ``tp_rank`` / ``tp_size`` are this rank and the stage's rank count.
    """
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    os.environ["RANK"] = str(tp_rank)
    # sglang multimodal_gen places modules on npu:{LOCAL_RANK}, not on our device;
    # drop once each stage process sees only its own card as index 0.
    os.environ["LOCAL_RANK"] = str(device.index)
    os.environ["WORLD_SIZE"] = str(tp_size)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    master_port = str(nccl_port) if nccl_port is not None else C.DEFAULT_MASTER_PORT
    os.environ["MASTER_PORT"] = master_port

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_dit, 
        sp_size=sp_degree, 
        cfg_degree=1,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,    
    )


@lru_cache(maxsize=None)
def _load_component(model_path: str, name: str):
    """
    Load one checkpoint component through sglang's own loader.
    """
    from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
        PipelineComponentLoader,
    )

    paths = resolve_checkpoint(model_path)
    server_args = get_global_server_args()
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

# Key of the denoising stage in sglang's component residency manager.
DENOISING_STAGE = "denoising_stage"


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
    device: torch.device,
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
    # store_state moved the AR prior to CPU for the hop between processes.
    if state.prior_token_id is not None:
        req.prior_token_id = state.prior_token_id.to(device)
    return req


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
    num_inference_steps: int = C.DEFAULT_NUM_INFERENCE_STEPS,
    guidance_scale: float = C.DEFAULT_GUIDANCE_SCALE,
    max_concurrency: int = 1,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    sp_degree: int = 1,
    ulysses_degree: int | None = None,
    ring_degree: int = 1,
) -> SimpleScheduler:
    """Conditioning, DiT sampling and VAE decode of one request in one stage.

    ``tp_size`` is the stage's rank count; ``sp_degree`` of them go to sequence
    parallelism and the DiT's own TP degree is what remains.
    """
    from sglang_omni.models.glm_image.diffusion import GLMImageDiffusion

    device = resolve_concrete_device(device, gpu_id)

    if sp_degree > tp_size or tp_size % sp_degree != 0:
        raise ValueError(
            f"tp_size ({tp_size}) must be >= and divisible by sp_degree ({sp_degree})"
        )
    tp_dit = tp_size // sp_degree
    if ulysses_degree is None:
        ulysses_degree = sp_degree // ring_degree
    if ulysses_degree * ring_degree != sp_degree:
        raise ValueError(
            f"sp_degree ({sp_degree}) must equal ring_degree ({ring_degree}) "
            f"* ulysses_degree ({ulysses_degree})"
        )

    _init_parallel(
        tp_rank=tp_rank,
        tp_size=tp_size,
        device=device,
        nccl_port=nccl_port,
        tp_dit=tp_dit,
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )
    server_args = _bootstrap(
        model_path,
        num_gpus=tp_size,
        tp_size=tp_dit,
        sp_degree=sp_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
    )

    # sglang applies tp_size to every TP-capable module, ByT5 included; it
    # falls back to a broken HF T5 when the heads do not split evenly.
    num_heads_text_enc = (
        server_args.pipeline_config.text_encoder_configs[0].arch_config.num_heads
    )
    num_heads_dit = server_args.pipeline_config.dit_config.arch_config.num_attention_heads
    if num_heads_text_enc % tp_dit != 0 or num_heads_dit % tp_dit != 0:
        raise ValueError(
            f"DiT TP degree {tp_dit} must divide the text encoder's "
            f"{num_heads_text_enc} heads and the DiT's {num_heads_dit} heads"
        )

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
        # The completion message to the coordinator goes through plain msgpack,
        # which cannot pack a Tensor even on CPU.
        state.prior_token_id = None
        payload = store_state(payload, state)
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
