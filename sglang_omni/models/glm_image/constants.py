# SPDX-License-Identifier: Apache-2.0
"""GLM-Image constants and server-side defaults."""

# The AR grid is D32: both dimensions are rounded up to a multiple of this
# before token generation, and the decoded image is cropped back afterwards.
GLM_IMAGE_RESOLUTION_ALIGNMENT = 32

# Glyph embedding budget. ByT5 tokenizes bytes, so this is a byte count, not
# a word count -- CJK text spends roughly three of these per character.
MAX_SEQUENCE_LENGTH = 1024

# Server defaults; a request may override either.
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.5
