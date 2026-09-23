# SPDX-License-Identifier: Apache-2.0
"""GLM-Image constants and server-side defaults."""

# The AR grid is D32: both dimensions are rounded up to a multiple of this
# before token generation, and the decoded image is cropped back afterwards.
GLM_IMAGE_RESOLUTION_ALIGNMENT = 32

# Server defaults; a request may override either.
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.5

DEFAULT_IMAGE_SIZE = 1024
# From the GlmImageTransformer2DModel docstring: pos_embed_max_size 128 gives
# 128 * vae_scale_factor * patch_size. GLM-Image checkpoints omit that key and
# position with RoPE, so treat this as a guard rather than a model limit.
MAX_IMAGE_SIZE = 2048
