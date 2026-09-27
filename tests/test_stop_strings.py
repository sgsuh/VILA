from llava.utils.stop_strings import StopStringFilter, truncate_at_stop


def test_truncate_at_earliest_stop():
    assert truncate_at_stop("a END b STOP", ["STOP", "END"]) == ("a ", True)
    assert truncate_at_stop("no stop here", ["STOP"]) == ("no stop here", False)
    assert truncate_at_stop("text", None) == ("text", False)
    assert truncate_at_stop("text", [""]) == ("text", False)


def stream(chunks, stop):
    stop_filter = StopStringFilter(stop)
    return "".join(stop_filter.feed(chunk) for chunk in chunks) + stop_filter.flush()


def test_filter_stops_across_chunks():
    assert stream(["Hello ", "wor", "ld. EN", "D more"], ["END"]) == "Hello world. "


def test_filter_releases_false_prefix():
    assert stream(["value E", "N", "d"], ["END"]) == "value ENd"


def test_filter_without_stop_passes_everything():
    assert stream(["a", "b", "c"], []) == "abc"


def test_filter_ignores_text_after_stop():
    stop_filter = StopStringFilter(["X"])
    assert stop_filter.feed("abXcd") == "ab"
    assert stop_filter.feed("more") == ""
    assert stop_filter.flush() == ""
