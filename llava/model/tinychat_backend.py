"""TinyChat inference backend for AWQ-quantized NVILA models.

Wraps the TinyChat NVILA implementation from mit-han-lab/llm-awq (W4A16 LLM + SmoothQuant
W8A8 vision tower) behind the same `generate_content` interface as the Hugging Face model.
Checkpoints are produced by `scripts/awq/quantize_nvila.sh`.

TinyChat is imported lazily: importing it patches torch weight-init functions globally.
"""

import os
import time
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, GenerationConfig

from llava.utils.generation_stats import summarize_generation
from llava.utils.logging import logger
from llava.utils.prefix_cache import CachedMediaEncoder, reusable_prefix_length
from llava.utils.stop_strings import truncate_at_stop
from llava.utils.tokenizer import as_conversation

QUANT_LLM_FILENAME = "llm-w4-g128-v2.pt"
SMOOTH_SCALE_FILENAME = "smooth-scale.pt"
# SmoothQuant migration strength used by tinychat/nvila_demo.py.
SMOOTH_ALPHA = 0.3
ROLES = {"system": "system", "human": "user", "gpt": "assistant"}


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
        max_seq_len: int = 8192,
        prefix_caching: bool = False,
    ) -> None:
        import tinychat.utils.constants

        # Sizes the preallocated KV cache; must be set before the TinyChat model modules are imported.
        tinychat.utils.constants.max_seq_len = max_seq_len

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
        self.kv_max_seq_len = next(m.kv_max_seq_len for m in self.model.llm.modules() if hasattr(m, "kv_max_seq_len"))
        self.prefix_caching = False
        # Embeddings of the positions currently held in the LLM's KV cache (see generate_content).
        self._prefix_embeds = None
        if prefix_caching:
            self.enable_prefix_caching()
        device_warmup(device)
        tune_llava_patch_embedding(self.model.vision_tower, device=device)
        logger.info(f"Loaded TinyChat NVILA from {model_path} with AWQ weights from {quant_dir}")

    def enable_prefix_caching(self) -> None:
        """Reuse work across calls that share a prompt prefix, e.g. successive chat turns:
        the KV cache of the shared prefix and the embeddings of recently seen media."""
        self.prefix_caching = True
        self.model.encoders = {
            name: encoder if isinstance(encoder, CachedMediaEncoder) else CachedMediaEncoder(encoder)
            for name, encoder in self.model.encoders.items()
        }

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def config(self):
        return self.model.config

    @property
    def default_generation_config(self) -> GenerationConfig:
        # Mirrors tinychat.utils.conversation_utils.gen_params (its repetition penalty is unused for NVILA).
        return GenerationConfig(max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.95, top_k=50)

    def _embed_prompt(self, text_prompt: str, media: Dict[str, List[torch.Tensor]], media_config) -> torch.Tensor:
        """Prompt embeddings [seq, hidden], built the same way as TinyChat's NVILA `stream_gen`."""
        input_ids = torch.as_tensor([self.tokenizer(text_prompt)["input_ids"]], device=self.device)
        if not media:
            return self.model.llm.model.embed_tokens(input_ids)[0]

        image_token_id = self.tokenizer.media_token_ids["image"]
        image_positions = (input_ids[0] == image_token_id).nonzero()
        if len(image_positions) == 1 and self.config.image_aspect_ratio == "dynamic":
            # A single dynamic-resolution image is split into tiles, each with its own media token.
            position = image_positions[0].item()
            newline = self.tokenizer.encode("\n", add_special_tokens=False)
            tiles = torch.as_tensor([(newline + [image_token_id] + newline) * len(media["image"])], device=self.device)
            input_ids = torch.cat([input_ids[:, :position], tiles, input_ids[:, position + 1 :]], dim=1)
        inputs_embeds, _, _ = self.model._embed(input_ids, media, media_config, None, attention_mask=None)
        return inputs_embeds[0]

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[Any] = None,
        streamer: Optional[Any] = None,
        stop: Optional[List[str]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a response; if given, `streamer` (a TextIteratorStreamer) receives text deltas.

        `prompt` is a single-turn prompt or a conversation (see `llava.utils.tokenizer.as_conversation`).
        If given, `stats` is filled with token counts and latency metrics. With `prefix_caching`, the KV
        cache of the longest prefix shared with the previous call is reused, so only the rest is prefilled.
        """
        start = time.perf_counter()
        if response_format is not None:
            raise NotImplementedError("response_format is not supported by the TinyChat backend")

        from tinychat.stream_generators.llava_stream_gen import prepare_logits_processor
        from tinychat.utils.prompt_templates import get_stop_token_ids

        # prepare_media replaces media parts with media tokens in the conversation text.
        conversation = as_conversation(prompt)
        media, media_config = self.model.prepare_media(conversation)
        text_prompt = self.tokenizer.apply_chat_template(
            [{"role": ROLES[m["from"]], "content": m["value"].strip()} for m in conversation],
            add_generation_prompt=True,
            tokenize=False,
        )
        embeds = self._embed_prompt(text_prompt, media, media_config)
        prompt_length = embeds.shape[0]

        generation_config = generation_config or self.default_generation_config
        max_new_tokens = min(generation_config.max_new_tokens or 512, self.kv_max_seq_len - prompt_length)
        if max_new_tokens <= 0:
            raise ValueError(f"The prompt ({prompt_length} tokens) does not fit in the KV cache ({self.kv_max_seq_len})")
        temperature = generation_config.temperature if generation_config.do_sample else 0.0
        greedy = temperature < 1e-5 or generation_config.top_p < 1e-8
        logits_processor = prepare_logits_processor(
            temperature, 1.0, generation_config.top_p, generation_config.top_k or 0
        )
        stop_token_ids = set(get_stop_token_ids("nvila") + [self.tokenizer.eos_token_id])

        reused = 0
        if self.prefix_caching and self._prefix_embeds is not None:
            reused = reusable_prefix_length(self._prefix_embeds, embeds)
        # Positions from `reused` on are about to be overwritten.
        self._prefix_embeds = None
        chunk_prefilling = reused > 0

        sampled_ids, token_times = [], []
        text = streamed = ""
        try:
            # Prefill the uncached part of the prompt (attending to the cached prefix), then decode.
            logits = self.model.llm(None, reused, embeds[None, reused:], chunk_prefilling)
            for step in range(max_new_tokens):
                if step > 0:
                    token_embeds = self.model.llm.model.embed_tokens(
                        torch.as_tensor([[sampled_ids[-1]]], device=self.device)
                    )
                    logits = self.model.llm(None, prompt_length + step - 1, token_embeds, chunk_prefilling)
                scores = logits_processor(None, logits[:, -1, :])[0]
                if greedy:
                    token = int(torch.argmax(scores))
                else:
                    probs = torch.softmax(scores.float(), dim=-1)
                    if not torch.isfinite(probs).all():
                        raise RuntimeError("TinyChat generation produced invalid probabilities")
                    token = int(torch.multinomial(probs, num_samples=1))
                sampled_ids.append(token)
                token_times.append(time.perf_counter())
                if token in stop_token_ids:
                    break

                text, stopped = truncate_at_stop(self.tokenizer.decode(sampled_ids, skip_special_tokens=True), stop)
                # Hold back incomplete multi-byte characters until they decode fully.
                if streamer is not None and text.startswith(streamed) and not text.endswith("\ufffd"):
                    streamer.on_finalized_text(text[len(streamed) :])
                    streamed = text
                if stopped:
                    break
        finally:
            if streamer is not None:
                streamer.on_finalized_text(text[len(streamed) :] if text.startswith(streamed) else "", stream_end=True)

        if self.prefix_caching:
            # The KV cache holds the prompt and every sampled token except the last one (never fed back).
            fed_ids = torch.as_tensor(sampled_ids[:-1], dtype=torch.long, device=self.device)
            self._prefix_embeds = torch.cat([embeds, self.model.llm.model.embed_tokens(fed_ids)])
        if stats is not None:
            stats.update(
                summarize_generation(
                    prompt_length,
                    len(sampled_ids),
                    start,
                    token_times[0] if token_times else None,
                    token_times[-1] if token_times else None,
                    time.perf_counter(),
                ),
                cached_tokens=reused,
            )
        return text.strip()
