from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer, StoppingCriteria

__all__ = ["truncate_at_stop", "StopStringFilter", "StopStringsCriteria"]


def truncate_at_stop(text: str, stop: Optional[List[str]]) -> Tuple[str, bool]:
    """Cut `text` at the earliest stop string. Returns the kept text and whether a stop string was found."""
    positions = [text.find(s) for s in stop or [] if s]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


class StopStringFilter:
    """Filter streamed text so that stop strings (and anything after them) are never emitted.

    Text that could be the start of a stop string is held back until it is disambiguated.
    """

    def __init__(self, stop: Optional[List[str]]) -> None:
        self.stop = [s for s in stop or [] if s]
        self.buffer = ""
        self.stopped = False

    def feed(self, text: str) -> str:
        if self.stopped:
            return ""
        self.buffer += text
        kept, self.stopped = truncate_at_stop(self.buffer, self.stop)
        if self.stopped:
            self.buffer = ""
            return kept
        holdback = self._partial_stop_length(self.buffer)
        emitted, self.buffer = self.buffer[: len(self.buffer) - holdback], self.buffer[len(self.buffer) - holdback :]
        return emitted

    def flush(self) -> str:
        emitted, self.buffer = ("" if self.stopped else self.buffer), ""
        return emitted

    def _partial_stop_length(self, text: str) -> int:
        """Length of the longest suffix of `text` that is a proper prefix of some stop string."""
        for length in range(min(len(text), max((len(s) for s in self.stop), default=1) - 1), 0, -1):
            suffix = text[-length:]
            if any(s.startswith(suffix) for s in self.stop):
                return length
        return 0


class StopStringsCriteria(StoppingCriteria):
    """Stop once the generated text contains a stop string.

    Only tokens generated after the first call are checked, so the prompt (which may be filled
    with placeholder ids when generating from a KV cache) never triggers a stop.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, stop: List[str]) -> None:
        self.tokenizer = tokenizer
        self.stop = stop
        self.prompt_length = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1] - 1
        texts = self.tokenizer.batch_decode(input_ids[:, self.prompt_length :], skip_special_tokens=True)
        return torch.tensor([truncate_at_stop(text, self.stop)[1] for text in texts], device=input_ids.device)
