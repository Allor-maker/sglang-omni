# SPDX-License-Identifier: Apache-2.0
"""GLMImage checkpoint path and weight loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_source


@dataclass(frozen=True)
class CheckpointPaths:
    root: Path
    text_encoder: Path
    tokenizer: Path
    vae: Path
    vision_language_encoder: Path
    processor: Path
    transformer: Path
    scheduler: Path
    model_index: Path


@lru_cache(maxsize=None)
def _download_once(model_path: str) -> str:
    return _resolve_source(model_path)


def resolve_checkpoint(model_path: str | Path) -> CheckpointPaths:
    root = Path(_download_once(str(Path(model_path).expanduser()))).expanduser()

    model_index_path = root / "model_index.json"

    if not model_index_path.exists():
        raise ValueError("No model_index.json file is downloaded")
    paths = CheckpointPaths(
        root=root,
        text_encoder=root / "text_encoder",
        tokenizer=root / "tokenizer",
        vae=root / "vae",
        vision_language_encoder=root / "vision_language_encoder",
        processor=root / "processor",
        transformer=root / "transformer",
        scheduler=root / "scheduler",
        model_index=root / "model_index.json"
    )

    return paths

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value




__all__ = [
    "CheckpointPaths",
    "load_json",
    "resolve_checkpoint",
]
