# SPDX-License-Identifier: Apache-2.0
"""Turn an image-generation request into GLM-Image pipeline state."""

from __future__ import annotations

from typing import Any

from sglang_omni.models.glm_image import constants as C
from sglang_omni.models.glm_image.payload_types import GLMImageState
from sglang_omni.proto import StagePayload

_MAX_PROMPT_CHARS = 4096


def _as_non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GLM-Image requires a non-empty {field}")
    if len(value) > _MAX_PROMPT_CHARS:
        raise ValueError(
            f"GLM-Image {field} is limited to {_MAX_PROMPT_CHARS} characters"
        )
    return value


def _parse_size(value: Any) -> tuple[int, int]:
    """Parse OpenAI's ``WIDTHxHEIGHT`` size, or fall back to the square default."""
    if value in (None, "", "auto"):
        return C.DEFAULT_IMAGE_SIZE, C.DEFAULT_IMAGE_SIZE
    if not isinstance(value, str) or "x" not in value:
        raise ValueError(f"GLM-Image size must look like '1024x1024', got {value!r}")
    width_text, _, height_text = value.partition("x")
    try:
        width, height = int(width_text), int(height_text)
    except ValueError as exc:
        raise ValueError(
            f"GLM-Image size must look like '1024x1024', got {value!r}"
        ) from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"GLM-Image size must be positive, got {value!r}")
    if width > C.MAX_IMAGE_SIZE or height > C.MAX_IMAGE_SIZE:
        raise ValueError(
            f"GLM-Image size must not exceed {C.MAX_IMAGE_SIZE} per side, got {value!r}"
        )
    return width, height


def _parse_positive_int(value: Any, field: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GLM-Image {field} must be an integer, got {value!r}")
    if value <= 0:
        raise ValueError(f"GLM-Image {field} must be positive, got {value}")
    return value


def _parse_seed(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GLM-Image seed must be an integer, got {value!r}")
    return value


def build_glm_image_state(payload: StagePayload) -> GLMImageState:
    """Build the entry-stage state from an image-generation request."""
    request = payload.request
    metadata = request.metadata or {}
    image_params = metadata.get("image_params")
    if not isinstance(image_params, dict):
        raise ValueError("GLM-Image requires a /v1/images/generations request")

    width, height = _parse_size(image_params.get("size"))
    guidance_scale = image_params.get("guidance_scale")
    if guidance_scale is not None and not isinstance(guidance_scale, (int, float)):
        raise ValueError(
            f"GLM-Image guidance_scale must be a number, got {guidance_scale!r}"
        )

    return GLMImageState(
        prompt=_as_non_empty_string(request.inputs, "prompt"),
        width=width,
        height=height,
        # Both stay at the user's canvas; the AR stage rounds width and height
        # up to the D32 grid and decode crops back to these.
        requested_width=width,
        requested_height=height,
        seed=_parse_seed(image_params.get("seed")),
        num_outputs=_parse_positive_int(image_params.get("n"), "n", 1),
        # Zero means "not requested", so the stage default applies.
        num_inference_steps=_parse_positive_int(
            image_params.get("num_inference_steps"), "num_inference_steps", 0
        ),
        guidance_scale=float(guidance_scale or 0.0),
    )


__all__ = ["build_glm_image_state"]
