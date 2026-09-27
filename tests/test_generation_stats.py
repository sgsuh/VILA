import pytest

from llava.utils.generation_stats import summarize_generation


def test_summarize_generation():
    stats = summarize_generation(100, 11, start=0.0, first_token=0.5, last_token=1.5, end=2.0)
    assert stats["prompt_tokens"] == 100
    assert stats["completion_tokens"] == 11
    assert stats["ttft_s"] == pytest.approx(0.5)
    assert stats["decode_tokens_per_s"] == pytest.approx(10.0)
    assert stats["total_s"] == pytest.approx(2.0)


def test_summarize_single_or_no_token():
    assert summarize_generation(10, 1, 0.0, 0.2, 0.2, 0.3)["decode_tokens_per_s"] is None
    stats = summarize_generation(10, 0, 0.0, None, None, 0.3)
    assert stats["ttft_s"] is None and stats["decode_tokens_per_s"] is None
