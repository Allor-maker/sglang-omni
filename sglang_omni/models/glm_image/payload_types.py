# SPDX-License-Identifier: Apache-2.0
"""GLM-Image pipeline state.

Mirrors the subset of sglang's ``Req`` that has to survive the hop between
stage processes. Tensors travel as CPU copies; everything the wrapped sglang
stage rebuilds for itself stays out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class GLMImageState(DeclarativeStateBase):
    """Request fields shared by the AR, conditioning, denoising and decode stages."""

    # --- request parameters ---
    prompt: str = wire("", codec="str")

    width: int = wire(0, codec="int")
    height: int = wire(0, codec="int")
    requested_width: int = wire(0, codec="int")
    requested_height: int = wire(0, codec="int")
    seed: int | None = None
    num_outputs: int = wire(1, codec="int")
    num_inference_steps: int = wire(0, codec="int")
    guidance_scale: float = wire(0.0, codec="float")

    # --- produced by the AR stage ---
    prior_token_id: Any | None = wire(None, codec="tensor_cpu")

    # --- produced by the conditioning stage ---
    prompt_embeds: Any | None = wire(None, codec="tensor_cpu")
    negative_prompt_embeds: Any | None = wire(None, codec="tensor_cpu")
    latents: Any | None = wire(None, codec="tensor_cpu")
    timesteps: Any | None = wire(None, codec="tensor_cpu")
    target_size: Any | None = wire(None, codec="tensor_cpu")
    crop_coords: Any | None = wire(None, codec="tensor_cpu")
    prior_token_drop_cond: Any | None = wire(None, codec="tensor_cpu")
    prior_token_drop_uncond: Any | None = wire(None, codec="tensor_cpu")

    # note: kv_caches is deliberately absent. The I2I reference cache is 30
    # layers of K/V and cannot cross a process boundary through tensor_cpu;
    # when I2I lands, conditioning and denoising share one process and the
    # cache is handed over in memory (see PipelineConfig.process_local_edges).
