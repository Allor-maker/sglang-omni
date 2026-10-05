from sglang_omni.scheduling.engine_factory import SGLangGenerationEngineBuilder
from sglang_omni.models.glm_image.checkpoint import resolve_checkpoint as resolve_root

from typing import Any
from sglang_omni.platforms import current_platform

class GLMImageEngineBuilder(SGLangGenerationEngineBuilder):
    model_name = "GLM-Image"
    model_arch_override = "GlmImageForConditionalGeneration"
    context_length = 8192

    def __init__(self, max_concurrency):
        self.max_concurrency = max_concurrency
        self.paths = None

    def resolve_checkpoint(self, model_path):
        self.paths = resolve_root(model_path=model_path)
        return str(self.paths.vision_language_encoder)


    
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

        def request_builder(payload):
            raise NotImplementedError("Not implement for GLM-Image yet")

        def request_adapter(data):
            raise NotImplementedError("Not implement for GLM-Image yet")
            
        return (request_builder, request_adapter)
        
    