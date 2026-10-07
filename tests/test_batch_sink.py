import pytest

from loguru import BatchSink, logger


class ListSink:
    """In-memory sink recording sends, failing on the given 1-based send attempts."""

    def __init__(self, fail_on=()):
        self.received = []
        self.calls = 0
        self.fail_on = set(fail_on)
        self.errors = {}

    def write(self, message):
        self.calls += 1
        if self.calls in self.fail_on:
            error = ValueError("send failed at attempt %d" % self.calls)
            self.errors[self.calls] = error
            raise error
        self.received.append(str(message))


def make_adapter(sink, begins=None, completions=None, on_batch_end=None):
    if on_batch_end is None:
        on_batch_end = lambda ok, err: completions.append((ok, err))  # noqa: E731
    return BatchSink(
        sink,
        on_batch_begin=lambda: begins.append(True) if begins is not None else None,
        on_batch_end=on_batch_end,
    )


def test_batch_second_event_fails():
    sink = ListSink(fail_on={2})
    completions = []
    adapter = make_adapter(sink, completions=completions)

    adapter.begin_batch()
    adapter.write("event-1")
    adapter.write("event-2")
    adapter.write("event-3")
    with pytest.raises(ValueError) as exc_info:
        adapter.end_batch()

    # Send count: each of the three events was attempted exactly once.
    assert sink.calls == 3
    # Successful events were not re-sent after the failure.
    assert sink.received == ["event-1", "event-3"]
    # Completion count and first failure reported to the end hook.
    assert completions == [(2, sink.errors[2])]
    # The original sink error is the one propagating out of end_batch().
    assert exc_info.value is sink.errors[2]


def test_end_hook_failure_does_not_mask_sink_error():
    sink = ListSink(fail_on={2})
    hook_error = RuntimeError("end hook boom")

    def on_batch_end(ok, err):
        assert (ok, err) == (2, sink.errors[2])
        raise hook_error

    adapter = make_adapter(sink, on_batch_end=on_batch_end)
    adapter.begin_batch()
    adapter.write("event-1")
    adapter.write("event-2")
    adapter.write("event-3")
    with pytest.raises(ValueError) as exc_info:
        adapter.end_batch()

    # The original sink error is raised, the hook failure stays as context.
    assert exc_info.value is sink.errors[2]
    assert exc_info.value.__context__ is hook_error


def test_end_hook_failure_without_sink_error():
    sink = ListSink()
    hook_error = RuntimeError("end hook boom")

    def on_batch_end(ok, err):
        assert (ok, err) == (3, None)
        raise hook_error

    adapter = make_adapter(sink, on_batch_end=on_batch_end)
    adapter.begin_batch()
    adapter.write("event-1")
    adapter.write("event-2")
    adapter.write("event-3")
    with pytest.raises(RuntimeError) as exc_info:
        adapter.end_batch()

    assert exc_info.value is hook_error
    assert sink.received == ["event-1", "event-2", "event-3"]


def test_normal_batch():
    sink = ListSink()
    begins = []
    completions = []
    adapter = make_adapter(sink, begins=begins, completions=completions)

    assert not adapter.in_batch
    with adapter.batch() as batch:
        assert batch is adapter
        assert adapter.in_batch
        adapter.write("event-1")
        adapter.write("event-2")
        adapter.write("event-3")
    assert not adapter.in_batch

    assert begins == [True]
    assert completions == [(3, None)]
    assert sink.calls == 3
    assert sink.received == ["event-1", "event-2", "event-3"]

    # end_batch() returns the completion count on success.
    adapter.begin_batch()
    adapter.write("event-4")
    assert adapter.end_batch() == 1


def test_batch_state_errors():
    adapter = BatchSink(ListSink())
    with pytest.raises(RuntimeError, match="no active batch"):
        adapter.end_batch()
    adapter.begin_batch()
    with pytest.raises(RuntimeError, match="already active"):
        adapter.begin_batch()
    adapter.end_batch()


def test_batch_context_manager_body_exception_not_masked():
    sink = ListSink(fail_on={1})
    completions = []
    adapter = make_adapter(sink, completions=completions)
    body_error = KeyError("body")

    with pytest.raises(KeyError) as exc_info:
        with adapter.batch():
            adapter.write("event-1")
            raise body_error

    assert exc_info.value is body_error
    assert completions == [(0, sink.errors[1])]
    assert not adapter.in_batch


def test_logger_per_record_sending_unchanged():
    sink = ListSink()
    adapter = BatchSink(sink)
    handler_id = logger.add(adapter, format="{message}")

    logger.info("one")
    logger.info("two")
    logger.remove(handler_id)

    # Outside of a batch, the logger keeps sending record by record.
    assert sink.calls == 2
    assert sink.received == ["one\n", "two\n"]


def test_logger_batch_with_second_sink_failure():
    sink = ListSink(fail_on={2})
    completions = []
    adapter = make_adapter(sink, completions=completions)
    handler_id = logger.add(adapter, format="{message}")

    with pytest.raises(ValueError) as exc_info:
        with adapter.batch():
            logger.info("first")
            logger.info("second")
            logger.info("third")
    logger.remove(handler_id)

    assert exc_info.value is sink.errors[2]
    assert sink.calls == 3
    assert sink.received == ["first\n", "third\n"]
    assert completions == [(2, sink.errors[2])]


def test_logger_batch_with_file_sink(tmp_path):
    path = tmp_path / "batch.log"
    file = path.open("w", encoding="utf8")
    completions = []
    adapter = make_adapter(file, completions=completions)
    handler_id = logger.add(adapter, format="{message}")

    with adapter.batch():
        logger.info("line-1")
        logger.info("line-2")
    logger.remove(handler_id)  # Also stops the adapter, closing the file.

    assert completions == [(2, None)]
    assert path.read_text(encoding="utf8") == "line-1\nline-2\n"


def test_invalid_sink_type():
    with pytest.raises(TypeError, match="Cannot batch to objects of type"):
        BatchSink(object())
