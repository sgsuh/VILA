"""Helpers for reusing work across generate calls that share a prompt prefix (e.g. chat turns)."""

import hashlib
from collections import OrderedDict
from typing import Any, Dict, List

import torch

__all__ = ["CachedMediaEncoder", "common_prefix_length", "reusable_prefix_length"]


def common_prefix_length(cached: torch.Tensor, new: torch.Tensor) -> int:
    """Number of leading positions whose embeddings are identical; both tensors are [seq, hidden].

    Prompts are compared as embeddings (media included), so no token-to-media bookkeeping is needed.
    """
    length = min(cached.shape[0], new.shape[0])
    mismatches = (cached[:length] != new[:length]).any(dim=-1).nonzero()
    return int(mismatches[0]) if len(mismatches) else length


def reusable_prefix_length(cached: torch.Tensor, new: torch.Tensor) -> int:
    """Length of the cached prefix worth reusing for the prompt `new`.

    At least one prompt position is always recomputed to produce the next-token logits. A prefix
    shorter than the remaining suffix is not reused: prefilling the whole prompt in one pass is
    cheaper than attending from a long suffix to a short cache.
    """
    length = min(common_prefix_length(cached, new), new.shape[0] - 1)
    return length if length >= new.shape[0] - length else 0


def _media_key(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha1(f"{tuple(tensor.shape)}{tensor.dtype}".encode() + data).hexdigest()


class CachedMediaEncoder:
    """Wraps a media encoder (e.g. `BasicImageEncoder`) to reuse the embeddings of recently seen media.

    Each media tensor is encoded independently unless a config (e.g. S2 `block_sizes`) or extra
    arguments couple them, in which case the call is passed through uncached.
    """

    def __init__(self, encoder: Any, capacity: int = 64) -> None:
        object.__setattr__(self, "encoder", encoder)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "cache", OrderedDict())

    def __getattr__(self, name: str) -> Any:
        return getattr(self.encoder, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self.encoder, name, value)

    def __call__(self, media: List[torch.Tensor], config: Dict[str, Any], **kwargs) -> List[torch.Tensor]:
        if config or kwargs:
            return self.encoder(media, config, **kwargs)
        keys = [_media_key(tensor) for tensor in media]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        if missing:
            first_index = {key: keys.index(key) for key in missing}
            outputs = self.encoder([media[first_index[key]] for key in missing], config)
            for key, output in zip(missing, outputs):
                self.cache[key] = output
        for key in keys:
            self.cache.move_to_end(key)
        results = [self.cache[key] for key in keys]
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return results
