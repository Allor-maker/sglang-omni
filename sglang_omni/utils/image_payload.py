# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for image payloads."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch

DEFAULT_IMAGE_MEDIA_TYPE = "image/png"


def _to_uint8_hwc(image: Any) -> np.ndarray:
    """Normalize a decoded image to a contiguous HxWxC uint8 array."""
    if isinstance(image, torch.Tensor):
        image = image.detach().float().cpu().numpy()
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"image payload takes one image at a time, got batch of {array.shape[0]}"
            )
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a CHW or HWC image, got shape {array.shape}")
    # Decoders emit channel-first; a trailing 1/3/4 means it is already HWC.
    if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = (
            np.clip(array.astype(np.float32) * 255.0, 0, 255).round().astype(np.uint8)
        )
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.ascontiguousarray(array)


def image_pixels_payload(
    image: Any,
    *,
    modality: str | None = "image",
    source_hint: str = "image",
) -> dict[str, Any]:
    """Serialize a decoded image into the relay payload format.

    Raw pixels rather than an encoded file, mirroring the audio helper: the
    container format belongs to the endpoint, which is where response_format
    is read.
    """
    try:
        array = _to_uint8_hwc(image)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported {source_hint} image output type: {type(image)}"
        ) from exc
    payload: dict[str, Any] = {
        "image_pixels": array.tobytes(),
        "image_pixels_shape": list(array.shape),
        "image_pixels_dtype": "uint8",
    }
    if modality is not None:
        payload["modality"] = modality
    return payload


def encode_image(pixels: Any, image_format: str = "png") -> tuple[bytes, str]:
    """Encode an HxWxC uint8 array, returning ``(bytes, media_type)``."""
    from PIL import Image

    array = _to_uint8_hwc(pixels)
    mode = "RGBA" if array.shape[-1] == 4 else "RGB"
    buffer = io.BytesIO()
    pillow_format = {"png": "PNG", "jpeg": "JPEG", "jpg": "JPEG", "webp": "WEBP"}.get(
        image_format.lower()
    )
    if pillow_format is None:
        raise ValueError(f"unsupported image format: {image_format!r}")
    if pillow_format == "JPEG" and mode == "RGBA":
        array = array[..., :3]
        mode = "RGB"
    Image.fromarray(array, mode=mode).save(buffer, format=pillow_format)
    media_type = "image/jpeg" if pillow_format == "JPEG" else f"image/{image_format}"
    return buffer.getvalue(), media_type
