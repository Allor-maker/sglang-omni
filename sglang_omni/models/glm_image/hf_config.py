

from .checkpoint import resolve_checkpoint, load_json

from typing import Any
from dataclasses import dataclass, field


import logging

@dataclass
class GLMImageRuntimeConfig:
    transformer: dict[str, Any] = field(default_factory=dict)
    
def load_glm_image_config(model_path: str) -> GLMImageRuntimeConfig:
    """read transformer/config.json, vae/config.json,  from a checkpoint."""

    paths = resolve_checkpoint(model_path=model_path)
    transformer_config = load_json(paths.transformer / "config.json")
    
    return GLMImageRuntimeConfig(
        transformer=transformer_config,
    )

def make_runtime_config(
    model_path: str,
) -> GLMImageRuntimeConfig:
    """Load a config, applying command-line overrides."""
    config = load_glm_image_config(model_path)
    return config
