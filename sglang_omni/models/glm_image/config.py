# SPDX-License-Identifier: Apache-2.0
"""GLM-Image: AR prior tokens, conditioning, DiT sampling, and VAE decode."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig
from sglang_omni.models.glm_image import constants as C
from sglang_omni.platforms import current_platform

_PKG = "sglang_omni.models.glm_image"

GLM_IMAGE_AR = "glm_image_ar"
GLM_IMAGE_BEFORE_DENOISING_STAGE = "glm_image_before_denoising_stage"
DENOISING_STAGE = "denoising_stage"
DECODE_STAGE = "decode"


class GLMImagePipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "GlmImagePipeline"

    stages: list[StageConfig] = [
        StageConfig(
            name=GLM_IMAGE_AR,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_ar_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_concurrency=1,
            ),
            gpu=0,
            next=GLM_IMAGE_BEFORE_DENOISING_STAGE,
        ),
        StageConfig(
            name=GLM_IMAGE_BEFORE_DENOISING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_before_denoising_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                num_inference_steps=C.DEFAULT_NUM_INFERENCE_STEPS,
                max_concurrency=1,
            ),
            gpu=0,
            next=DENOISING_STAGE,
        ),
        StageConfig(
            name=DENOISING_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_denoising_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                guidance_scale=C.DEFAULT_GUIDANCE_SCALE,
                max_concurrency=1,
            ),
            gpu=0,
            next=DECODE_STAGE,
        ),
        StageConfig(
            name=DECODE_STAGE,
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_decode_executor",
            factory=FactoryArgs(
                device=current_platform.device_type,
                dtype="bfloat16",
                max_concurrency=1,
            ),
            gpu=0,
            terminal=True,
        ),
    ]

    @classmethod
    def process_local_edges(cls) -> frozenset[tuple[str, str]]:
        # Conditioning hands the denoiser state the payload does not carry:
        # today the DiT config it reads off the transformer module, and with
        # I2I the 30-layer reference KV cache, which no tensor_cpu hop can
        # move. Drop the edge once both travel in the payload.
        return frozenset({(GLM_IMAGE_BEFORE_DENOISING_STAGE, DENOISING_STAGE)})


EntryClass = GLMImagePipelineConfig
