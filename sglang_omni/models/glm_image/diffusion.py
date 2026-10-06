# SPDX-License-Identifier: Apache-2.0
"""Conditioning, DiT sampling and VAE decode as one call per request.

This is the work sglang's distributed GLM-Image denoiser does in one worker:
the three stages pass the same Req along in memory, so nothing is converted
between them, and the scheduler conditioning configures stays private to the
request until it is decoded.
"""

from __future__ import annotations

import logging
from typing import Any

from sglang.multimodal_gen.runtime.utils.perf_logger import StageProfiler

logger = logging.getLogger(__name__)


class GLMImageDiffusion:
    """Runs sglang's GLM-Image conditioning, denoising and decoding stages."""

    def __init__(
        self,
        *,
        before_denoising: Any,
        denoising: Any,
        decoding: Any,
        server_args: Any,
    ) -> None:
        self.before_denoising = before_denoising
        self.denoising = denoising
        self.decoding = decoding
        self.server_args = server_args

    def run(self, req: Any) -> Any:
        """Take a Req carrying the AR prior tokens, return sglang's OutputBatch."""
        # DenoisingStage reads batch.scheduler, not self.scheduler, and the
        # instance carries the schedule conditioning configures on it.
        req.scheduler = self.denoising.scheduler
        req = self._forward(self.before_denoising, req)
        req = self._forward(self.denoising, req)
        return self._forward(self.decoding, req)

    def _forward(self, stage: Any, batch: Any) -> Any:
        with StageProfiler(
            type(stage).__name__, logger, metrics=None, log_stage_start_end=True
        ):
            return stage.forward(batch, self.server_args)
