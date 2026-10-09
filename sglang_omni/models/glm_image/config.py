# SPDX-License-Identifier: Apache-2.0
"""GLM-Image: AR prior tokens, conditioning, DiT sampling, and VAE decode."""

from typing import Any, ClassVar

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
GLM_IMAGE_DIT = "glm_image_dit"


def _stage_gpu_set(stage: StageConfig) -> set[int]:
    gpu = stage.gpu
    if gpu is None:
        return set()
    return set(gpu) if isinstance(gpu, list) else {gpu}


def _reject_shared_tp_gpus(stages: list[StageConfig]) -> None:
    """
    Refuse multi-rank AR and DiT stages placed on the same cards.
    """
    by_name = {stage.name: stage for stage in stages}
    ar, dit = by_name.get(GLM_IMAGE_AR), by_name.get(GLM_IMAGE_DIT)
    if ar is None or dit is None or ar.tp_size == 1 or dit.tp_size == 1:
        return
    shared = _stage_gpu_set(ar) & _stage_gpu_set(dit)
    if shared:
        raise ValueError(
            f"{GLM_IMAGE_AR} (tp_size={ar.tp_size}) and {GLM_IMAGE_DIT} "
            f"(tp_size={dit.tp_size}) share GPUs {sorted(shared)}. Two multi-rank "
            "stages on the same cards deadlock at startup: each rank waits for its "
            "peers inside the per-card startup lock the other stage holds. Place "
            'them on disjoint cards or run one of them on a single rank.'
        )


def _dit_stages(*, process: str, gpu: int | list[int], tp_size: int = 1, sp_degree: int = 1) -> list[StageConfig]:
    return [
        StageConfig(
            name=GLM_IMAGE_DIT,
            process=process,
            factory_path=f"{_PKG}.stages.create_dit_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                num_inference_steps=C.DEFAULT_NUM_INFERENCE_STEPS,
                guidance_scale=C.DEFAULT_GUIDANCE_SCALE,
                max_concurrency=1,
                sp_degree=sp_degree,
                ulysses_degree=None,
                ring_degree=1,
            ),
            gpu=gpu,
            tp_size=tp_size,
            terminal=True,
        ),
    ]


def _srt_stages(
    *, ar_gpus: list[int], dit_gpus: list[int], dit_sp_degree: int = 1, kv_cache_bytes: str | None = None
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
            tp_size = len(ar_gpus),
            gpu=ar_gpus,
            next=GLM_IMAGE_DIT,
        ),
        *_dit_stages(
            process="glm_image_dit", 
            tp_size=len(dit_gpus), 
            gpu=dit_gpus, 
            sp_degree=dit_sp_degree
        ),
    ]


def _dual_npu_stages() -> list[StageConfig]:
    return _srt_stages(ar_gpus=[0], dit_gpus=[1])


def _single_npu_stages() -> list[StageConfig]:
    # Without a fixed KV pool srt sizes it from a fraction of the card and
    # leaves the DiT too little; one request at 2048x2048 needs ~4.4k tokens.
    return _srt_stages(ar_gpus=[0], dit_gpus=[0], kv_cache_bytes="4GiB")


class GLMImagePipelineConfig(PipelineConfig):
    """srt AR on NPU 0, conditioning + DiT + VAE on NPU 1."""

    architecture: ClassVar[str] = "GlmImagePipeline"

    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        GLM_IMAGE_AR: EngineStageConfig,
    }

    stages: list[StageConfig] = Field(default_factory=_dual_npu_stages)

    def model_post_init(self, __context: Any = None) -> None:
        super().model_post_init(__context)
        _reject_shared_tp_gpus(self.stages)


class GLMImageSingleNPUPipelineConfig(GLMImagePipelineConfig):
    """Both processes on NPU 0; the srt KV pool is capped to leave room for the DiT."""

    stages: list[StageConfig] = Field(default_factory=_single_npu_stages)
    placement: PlacementConfig = Field(
        default_factory=lambda: PlacementConfig(
            require_memory_fraction_for_colocation=False
        )
    )


EntryClass = GLMImagePipelineConfig

Variants = {
    "default": GLMImagePipelineConfig,
    "single-npu": GLMImageSingleNPUPipelineConfig,
}
