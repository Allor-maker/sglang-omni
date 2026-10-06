from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder
from sglang_omni.models.glm_image.checkpoint import resolve_checkpoint as resolve_root
from transformers import AutoProcessor
from typing import Any
from sglang_omni.platforms import current_platform

from sglang_omni.models.glm_image.sglang_request_builder import make_glm_image_ar_adapters

class GLMImageEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "GLM-Image"
    model_arch_override = "GlmImageForConditionalGeneration"
    context_length = 8192

    def __init__(
            self, 
            max_concurrency,
            tp_rank: int = 0,
            tp_size: int = 1,
            nccl_port: int | None = None,
        ):
        self.max_concurrency = max_concurrency
        self.paths = None
        self.processor = None
        
        tp_rank = int(tp_rank)
        tp_size = int(tp_size)

        if tp_size <= 0:
            raise ValueError(
                f"tp_size must be positive; got {tp_size}"
            )
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError(
                f"tp_rank={tp_rank} is out of range "
                f"for tp_size={tp_size}"
            )
        if tp_size > 1 and nccl_port is None:
            raise ValueError("TP requires nccl_port")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.nccl_port = nccl_port
        
    def resolve_checkpoint(self, model_path):
        self.paths = resolve_root(model_path=model_path)
        return str(self.paths.vision_language_encoder)

    def adjust_overrides(self, overrides: dict[str, Any]) -> None:
        overrides["tp_size"] = self.tp_size

    def infra_kwargs(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "nccl_port": self.nccl_port,
        }

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        del checkpoint_dir
        self.processor = AutoProcessor.from_pretrained(
            str(self.paths.processor)
        )
    
    def generation_defaults(self, *, dtype: str) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "max_running_requests": self.max_concurrency,
            "disable_cuda_graph": False,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "dtype": dtype,
            "tokenizer_path": str(self.paths.processor),
            "sampling_backend": "pytorch"
        }

        if current_platform.is_npu():
            defaults["attention_backend"] = "ascend"

        return defaults
    
    def make_model_runner(self, model_worker, output_proc):
        from sglang_omni.model_runner.base import (
            ModelRunner,
        )

        return ModelRunner(model_worker, output_proc)
 
    def make_adapters(self, model):

        return make_glm_image_ar_adapters(
            processor=self.processor,
            image_start_token_id=model.config.image_start_token_id,
            image_end_token_id=model.config.image_end_token_id,
            vocab_size=model.config.text_config.vision_vocab_size,
        )
