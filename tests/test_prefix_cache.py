import torch

from llava.utils.prefix_cache import CachedMediaEncoder, common_prefix_length, reusable_prefix_length
from llava.utils.stop_strings import StopStringsCriteria


def test_common_prefix_length():
    cached = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    assert common_prefix_length(cached, cached.clone()) == 6
    assert common_prefix_length(cached, cached[:4]) == 4
    changed = cached.clone()
    changed[3, 1] = -1
    assert common_prefix_length(cached, changed) == 3
    assert common_prefix_length(cached[:0], cached) == 0


class CharTokenizer:
    """Token id i decodes to chr(i)."""

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["".join(chr(i) for i in seq.tolist()) for seq in sequences]


def test_stop_criteria_ignores_prompt():
    criteria = StopStringsCriteria(CharTokenizer(), ["ab"])
    prompt = [ord("a")] * 5  # ends with "a": must not combine with a generated "b"
    ids = torch.tensor([prompt + [ord("b")]])
    assert not criteria(ids, None).any()
    ids = torch.tensor([prompt + [ord("b"), ord("a"), ord("b")]])
    assert criteria(ids, None).all()


def test_reusable_prefix_length():
    cached = torch.arange(20, dtype=torch.float16).reshape(10, 2)
    assert reusable_prefix_length(cached, cached.clone()) == 9  # the last position is always recomputed
    longer = torch.cat([cached, torch.full((4, 2), -1.0, dtype=torch.float16)])
    assert reusable_prefix_length(cached, longer) == 10
    much_longer = torch.cat([cached, torch.full((30, 2), -1.0, dtype=torch.float16)])
    assert reusable_prefix_length(cached, much_longer) == 0  # prefix shorter than the new suffix


class CountingEncoder:
    def __init__(self):
        self.calls = []
        self.end_tokens = "\n"

    def __call__(self, media, config, **kwargs):
        self.calls.append(len(media))
        return [tensor * 2 for tensor in media]


def test_cached_media_encoder():
    inner = CountingEncoder()
    encoder = CachedMediaEncoder(inner, capacity=2)
    a, b, c = torch.ones(2), torch.zeros(2), torch.full((2,), 3.0)
    assert [t.tolist() for t in encoder([a, b, a], {})] == [[2, 2], [0, 0], [2, 2]]
    assert inner.calls == [2]  # duplicates encoded once
    encoder([b, a], {})
    assert inner.calls == [2]  # all cached
    encoder([c], {})
    encoder([b], {})
    assert inner.calls == [2, 1, 1]  # capacity 2 evicted the least recently used entry (b)
    encoder([a], {"block_sizes": [None]})
    assert inner.calls[-1] == 1  # configs couple media, so the call passes through
    encoder.end_tokens = None
    assert inner.end_tokens is None
