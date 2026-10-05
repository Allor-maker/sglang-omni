from sglang.srt.managers.schedule_batch import Req, MultimodalInputs
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang.srt.multimodal.processors.glm_image import GlmImageProcessor
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import GlmImageAR
from sglang.multimodal_gen.configs.sample.glmimage import align_glm_image_resolution

from sglang_omni.proto import StagePayload

from sglang_omni.scheduling.sglang_backend import SGLangARRequestData
from sglang_omni.scheduling.pipeline_state import store_state

from sglang_omni.models.glm_image.request_builders import build_glm_image_state
import sglang_omni.models.glm_image.constants as C

import torch
from dataclasses import dataclass

from sglang_omni.models.glm_image.payload_types import GLMImageState

import logging
logger = logging.getLogger(__name__)

from types import SimpleNamespace
import time

@dataclass
class GLMImageARRequestData(SGLangARRequestData):
    state: GLMImageState | None = None
    generation_shape: tuple[int, int, int] | None = None


def make_glm_image_ar_adapters(processor, image_start_token_id, image_end_token_id, vocab_size):

    proc = SimpleNamespace(
        IMAGE_START_TOKEN_ID=image_start_token_id,
        IMAGE_END_TOKEN_ID=image_end_token_id
    )

    def request_builder(
            payload: StagePayload,
        )-> GLMImageARRequestData:

        state = build_glm_image_state(payload=payload)
        state.request_started_at = time.time()
        requested_width, requested_height = state.requested_width, state.requested_height
        prompt = state.prompt

        width, height = align_glm_image_resolution(requested_width, requested_height)
        if (width, height) != (requested_width, requested_height):
            logger.warning(
                "GLM-Image requires dimensions divisible by %s; adjusted "
                "runtime resolution from %sx%s to %sx%s",
                C.GLM_IMAGE_RESOLUTION_ALIGNMENT,
                requested_width,
                requested_height,
                width,
                height,
            )
        state.width, state.height = width, height
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            target_h=height,
            target_w=width,
            return_dict=True,
            return_tensors="pt",
        )

        image_grid_thw = inputs.get("image_grid_thw")
        input_ids = inputs["input_ids"][0]
        max_new_tokens, large_image_offset, token_h, token_w = (
            GlmImageAR._compute_generation_params(
                image_grid_thw=image_grid_thw,
                is_text_to_image=True,
            )
        )
        mrope_positions, mrope_position_delta = GlmImageProcessor._compute_glm_image_mrope_positions(
            proc,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        mm_inputs = MultimodalInputs(
            mm_items=[],
        )
        mm_inputs.mrope_positions = mrope_positions
        mm_inputs.mrope_position_delta = mrope_position_delta

        seed = state.seed if state.seed is not None else C.DEFAULT_SEED

        sampling_params = SamplingParams(
            **GlmImageAR._external_ar_sampling_params(max_new_tokens, seed),
        )
        sampling_params.normalize(tokenizer=None)
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=input_ids.tolist(),
            sampling_params=sampling_params,
            vocab_size=vocab_size,
        )

        req.multimodal_inputs = mm_inputs

        return GLMImageARRequestData(
                state=state,
                stage_payload=payload,
                req=req,
                input_ids=input_ids,
                generation_shape=(large_image_offset, token_h, token_w),
            )

    def result_adapter(data):
        state = data.state
        payload = data.stage_payload
        offset, token_h, token_w = data.generation_shape
        output_ids = list(data.output_ids or [])
        actual_output_len = len(output_ids)
        expected_output_len = offset + token_w * token_h

        if actual_output_len < expected_output_len:
            raise RuntimeError(
                "GLM-Image AR returned too few output_ids: "
                f"got {actual_output_len}, need at least {expected_output_len} "
                f"(large_image_offset={offset}, "
                f"token_h={token_h}, token_w={token_w})."
            )
        prior_token_ids_d32 = torch.tensor(
            output_ids[offset : offset + token_h * token_w],
            dtype=torch.long,
        )

        state.prior_token_id = GlmImageAR._upsample_token_ids(prior_token_ids_d32, token_h, token_w)

        state.prompt_tokens = len(data.input_ids)
        state.completion_tokens = actual_output_len
        logger.info("[GlmImageAR] finished in %.4f seconds", time.time() - state.request_started_at)

        return store_state(payload, state)
    


    return (request_builder, result_adapter)
    