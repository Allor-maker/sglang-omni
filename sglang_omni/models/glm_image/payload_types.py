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
    """Request fields shared by the AR stage and the diffusion (DiT) stage."""

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
    # Wall-clock epoch seconds, not perf_counter: the value is compared in the
    # DiT stage, which runs in a different process from the AR entry stage.
    request_started_at: float = wire(0.0, codec="float")

    # --- produced by the AR stage ---
    prior_token_id: Any | None = wire(None, codec="tensor_cpu")
