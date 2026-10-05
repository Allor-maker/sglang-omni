# SPDX-License-Identifier: Apache-2.0
"""GLM-Image: AR prior tokens, conditioning, DiT sampling, and VAE decode."""

from typing import ClassVar

from pydantic import Field

from sglang_omni.config import (
    EngineArgs,
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    PlacementConfig,
    StageConfig,
)
from sglang_omni.models.glm_image import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.glm_image"

GLM_IMAGE_AR = "glm_image_ar"
GLM_IMAGE_BEFORE_DENOISING_STAGE = "glm_image_before_denoising_stage"
DENOISING_STAGE = "denoising_stage"
DECODE_STAGE = "decode"


def _dit_stages(*, process: str, gpu: int) -> list[StageConfig]:
    return [
        StageConfig(
            name=GLM_IMAGE_BEFORE_DENOISING_STAGE,
            process=process,
            factory_path=f"{_PKG}.stages.create_before_denoising_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                num_inference_steps=C.DEFAULT_NUM_INFERENCE_STEPS,
                max_concurrency=1,
            ),
            gpu=gpu,
            next=DENOISING_STAGE,
        ),
        StageConfig(
            name=DENOISING_STAGE,
            process=process,
            factory_path=f"{_PKG}.stages.create_denoising_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                guidance_scale=C.DEFAULT_GUIDANCE_SCALE,
                max_concurrency=1,
            ),
            gpu=gpu,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process=process,
            factory_path=f"{_PKG}.stages.create_decode_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_concurrency=1,
            ),
            gpu=gpu,
            terminal=True,
        ),
    ]


def _srt_stages(
    *, dit_gpu: int, kv_cache_bytes: str | None = None
) -> list[StageConfig]:
    # The srt engine sets up its own torch.distributed world, and the DiT stack
    # sets up another in _init_parallel; separate processes keep them apart.
    return [
        EngineStageConfig(
            name=GLM_IMAGE_AR,
            process="glm_image_ar",
            factory_path=f"{_PKG}.stages.create_srt_ar_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_concurrency=1,
            ),
            engine=EngineArgs(kv_cache_bytes=kv_cache_bytes),
            gpu=0,
            next=GLM_IMAGE_BEFORE_DENOISING_STAGE,
        ),
        *_dit_stages(process="glm_image_dit", gpu=dit_gpu),
    ]


def _dual_npu_stages() -> list[StageConfig]:
    return _srt_stages(dit_gpu=1)


def _single_npu_stages() -> list[StageConfig]:
    # Without a fixed KV pool srt sizes it from a fraction of the card and
    # leaves the DiT too little; one request at 2048x2048 needs ~4.4k tokens.
    return _srt_stages(dit_gpu=0, kv_cache_bytes="4GiB")


def _hf_ar_stages() -> list[StageConfig]:
    return [
        StageConfig(
            name=GLM_IMAGE_AR,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_hf_ar_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_concurrency=1,
            ),
            gpu=0,
            next=GLM_IMAGE_BEFORE_DENOISING_STAGE,
        ),
        *_dit_stages(process="pipeline", gpu=0),
    ]


class GLMImagePipelineConfig(PipelineConfig):
    """srt AR on NPU 0, conditioning + DiT + VAE on NPU 1."""

    architecture: ClassVar[str] = "GlmImagePipeline"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        GLM_IMAGE_AR: EngineStageConfig,
    }

    stages: list[StageConfig] = Field(default_factory=_dual_npu_stages)

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Conditioning hands the denoiser state the payload does not carry:
        # today the DiT config it reads off the transformer module, and with
        # I2I the 30-layer reference KV cache, which no tensor_cpu hop can
        # move. Drop the edge once both travel in the payload.
        return frozenset({(GLM_IMAGE_BEFORE_DENOISING_STAGE, DENOISING_STAGE)})


class GLMImageSingleNPUPipelineConfig(GLMImagePipelineConfig):
    """Both processes on NPU 0; the srt KV pool is capped to leave room for the DiT."""

    stages: list[StageConfig] = Field(default_factory=_single_npu_stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )


class GLMImageHFARPipelineConfig(GLMImagePipelineConfig):
    """Previous layout: HF generate() AR and the DiT stages in one process on NPU 0."""

    # Empty, so a rebuilt config validates the HF AR stage as a plain StageConfig.
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {}

    stages: list[StageConfig] = Field(default_factory=_hf_ar_stages)


EntryClass = GLMImagePipelineConfig

Variants = {
    "default": GLMImagePipelineConfig,
    "single-npu": GLMImageSingleNPUPipelineConfig,
    "hf-ar": GLMImageHFARPipelineConfig,
}
