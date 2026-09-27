"""TinyChat inference backend for AWQ-quantized NVILA models.

Wraps the TinyChat NVILA implementation from mit-han-lab/llm-awq (W4A16 LLM + SmoothQuant
W8A8 vision tower) behind the same `generate_content` interface as the Hugging Face model.
Checkpoints are produced by `scripts/awq/quantize_nvila.sh`.

TinyChat is imported lazily: importing it patches torch weight-init functions globally.
"""

import os
from typing import Any, List, Optional, Union

import torch
from transformers import AutoConfig, GenerationConfig

from llava.utils.logging import logger

QUANT_LLM_FILENAME = "llm-w4-g128-v2.pt"
SMOOTH_SCALE_FILENAME = "smooth-scale.pt"
# SmoothQuant migration strength used by tinychat/nvila_demo.py.
SMOOTH_ALPHA = 0.3


def resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download

    return snapshot_download(model_path)


def default_quant_dir(model_path: str) -> str:
    return os.path.join("runs", "awq", os.path.basename(os.path.normpath(model_path)))


class TinyChatNVILA:
    def __init__(
        self,
        model_path: str,
        quant_dir: Optional[str] = None,
        device: str = "cuda:0",
        max_seq_len: int = 2048,
    ) -> None:
        import tinychat.utils.constants
        from awq.quantize import smooth_lm
        from tinychat.models.nvila_qwen2 import NVILAQwen2
        from tinychat.models.qwen2 import Qwen2ForCausalLM
        from tinychat.modules import QuantSiglipEncoder, make_quant_attn, make_quant_norm
        from tinychat.utils.load_quant import load_awq_model
        from tinychat.utils.tune import device_warmup, tune_llava_patch_embedding

        quant_dir = quant_dir or default_quant_dir(model_path)
        quant_llm_path = os.path.join(quant_dir, QUANT_LLM_FILENAME)
        smooth_scale_path = os.path.join(quant_dir, SMOOTH_SCALE_FILENAME)
        for path in (quant_llm_path, smooth_scale_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{path} not found; run scripts/awq/quantize_nvila.sh first")

        tinychat.utils.constants.max_seq_len = max_seq_len
        model_path = resolve_model_path(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.resume_path = model_path
        model = NVILAQwen2(config, False).half()

        # Vision tower: fold SmoothQuant scales into LayerNorms, then swap in the W8A8 encoder.
        smooth_lm(model.vision_tower, torch.load(smooth_scale_path), SMOOTH_ALPHA)
        vision_model = model.vision_tower.vision_tower.vision_model
        vision_model.encoder = QuantSiglipEncoder(vision_model.encoder)

        # LLM: W4A16 weights with fused attention and norm kernels.
        model.llm = Qwen2ForCausalLM(model.llm_cfg).half()
        model.llm = load_awq_model(model.llm, quant_llm_path, 4, 128, device)
        make_quant_attn(model.llm, device, True)
        make_quant_norm(model.llm)
        model.llm.cpu()
        model.llm.resize_token_embeddings(len(model.tokenizer))

        self.model = model.cuda().eval()
        self.device = device
        device_warmup(device)
        tune_llava_patch_embedding(self.model.vision_tower, device=device)
        logger.info(f"Loaded TinyChat NVILA from {model_path} with AWQ weights from {quant_dir}")

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def config(self):
        return self.model.config

    @property
    def default_generation_config(self) -> GenerationConfig:
        # Mirrors tinychat.utils.conversation_utils.gen_params.
        return GenerationConfig(
            max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95, top_k=50, repetition_penalty=1.1
        )

    @staticmethod
    def _to_gen_params(generation_config: GenerationConfig):
        from tinychat.utils.conversation_utils import gen_params

        # AttributeDict does not support deepcopy; its values are all scalars or an empty dict.
        params = type(gen_params)(list(gen_params.items()))
        params.n_predict = generation_config.max_new_tokens or params.n_predict
        params.temp = generation_config.temperature if generation_config.do_sample else 0.0
        params.top_p = generation_config.top_p
        params.top_k = generation_config.top_k
        params.repeat_penalty = generation_config.repetition_penalty
        return params

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[Any] = None,
        streamer: Optional[Any] = None,
    ) -> str:
        """Generate a response; if given, `streamer` (a TextIteratorStreamer) receives text deltas."""
        if response_format is not None:
            raise NotImplementedError("response_format is not supported by the TinyChat backend")

        from tinychat.stream_generators.NVILA_stream_gen import NVILAStreamGenerator
        from tinychat.utils.prompt_templates import NVILAPrompter, get_stop_token_ids

        # prepare_media rewrites the conversation text with one media token per image tile / video frame.
        conversation = [{"from": "human", "value": prompt}]
        media, media_config = self.model.prepare_media(conversation)
        prompter = NVILAPrompter()
        prompter.insert_prompt(conversation[0]["value"])

        outputs = NVILAStreamGenerator(
            self.model,
            self._to_gen_params(generation_config or self.default_generation_config),
            prompter.model_input,
            media or None,
            media_config if media else None,
            start_pos=0,
            device=self.device,
            stop_token_ids=get_stop_token_ids("nvila"),
            quant_llm=True,
        )

        text = streamed = ""
        try:
            for output in outputs:
                text = output["text"]
                # Hold back incomplete multi-byte characters until they decode fully.
                if streamer is not None and text.startswith(streamed) and not text.endswith("�"):
                    streamer.on_finalized_text(text[len(streamed) :])
                    streamed = text
        except SystemExit as e:
            # NVILAStreamGenerator calls exit() when sampling hits Inf/NaN probabilities.
            raise RuntimeError("TinyChat generation produced invalid probabilities") from e
        finally:
            if streamer is not None:
                streamer.on_finalized_text(text[len(streamed) :] if text.startswith(streamed) else "", stream_end=True)
        return text.strip()
