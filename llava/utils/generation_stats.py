import time
from typing import Any, Dict, Optional

import torch
from transformers import StoppingCriteria

__all__ = ["GenerationTimer", "summarize_generation"]


def summarize_generation(
    prompt_tokens: int,
    completion_tokens: int,
    start: float,
    first_token: Optional[float],
    last_token: Optional[float],
    end: float,
) -> Dict[str, Any]:
    """Token counts and latency metrics; `start` is when the request began processing (media included)."""
    stats = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": None if first_token is None else first_token - start,
        "decode_tokens_per_s": None,
        "total_s": end - start,
    }
    # The first token comes from prefill; decode speed covers the remaining tokens.
    if completion_tokens > 1 and last_token > first_token:
        stats["decode_tokens_per_s"] = (completion_tokens - 1) / (last_token - first_token)
    return stats


class GenerationTimer(StoppingCriteria):
    """Records when each token is generated; never stops generation.

    Hugging Face `generate` calls stopping criteria once per generated token.
    """

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.token_times = []

    def reset_tokens(self) -> None:
        self.token_times = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self.token_times.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def summary(self, prompt_tokens: int) -> Dict[str, Any]:
        first_token, last_token = (self.token_times[0], self.token_times[-1]) if self.token_times else (None, None)
        return summarize_generation(
            prompt_tokens, len(self.token_times), self.start, first_token, last_token, time.perf_counter()
        )
