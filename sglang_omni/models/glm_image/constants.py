# SPDX-License-Identifier: Apache-2.0
"""GLM-Image constants and server-side defaults."""

# The AR grid is D32: both dimensions are rounded up to a multiple of this
# before token generation, and the decoded image is cropped back afterwards.
GLM_IMAGE_RESOLUTION_ALIGNMENT = 32

# Server defaults; a request may override either.
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.5
