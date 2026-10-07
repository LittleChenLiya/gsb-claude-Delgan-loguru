import pytest

from loguru import BatchSink, logger


class BatchFixture:
    """Small in-memory fixture recording sends, attempts and hook calls."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.attempts = []
        self.sent = []
        self.started = 0
        self.completed = []

    def function(self, message):
        self.attempts.append(message)
        if message in self.fail_on:
            raise ValueError("sink exploded on %r" % message)
        self.sent.append(message)

    def on_batch_start(self):
        self.started += 1

    def on_batch_end(self, succeeded, first_error):
        self.completed.append((succeeded, first_error))

    def make_sink(self, **kwargs):
        kwargs.setdefault("on_batch_start", self.on_batch_start)
        kwargs.setdefault("on_batch_end", self.on_batch_end)
        return BatchSink(self.function, **kwargs)


def test_batch_second_of_three_events_fails():
    fixture = BatchFixture(fail_on={"event-2"})
    sink = fixture.make_sink()

    with pytest.raises(ValueError, match="sink exploded on 'event-2'") as exc_info:
        sink.send_batch(["event-1", "event-2", "event-3"])

    # The failing event stopped the batch: the third event was never attempted and
    # the first (successful) event was not sent again.
    assert fixture.attempts == ["event-1", "event-2"]
    assert fixture.sent == ["event-1"]

    # The end hook ran exactly once, carrying the success count and the first failure.
    assert fixture.started == 1
    assert len(fixture.completed) == 1
    succeeded, first_error = fixture.completed[0]
    assert succeeded == 1
    assert first_error is exc_info.value

    # The original sink error is the one propagated, unmodified.
    assert str(exc_info.value) == "sink exploded on 'event-2'"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_batch_end_hook_failure_does_not_mask_sink_error():
    fixture = BatchFixture(fail_on={"event-1"})

    def broken_on_batch_end(succeeded, first_error):
        raise RuntimeError("hook exploded")

    sink = fixture.make_sink(on_batch_end=broken_on_batch_end)

    with pytest.raises(ValueError, match="sink exploded on 'event-1'") as exc_info:
        sink.send_batch(["event-1", "event-2"])

    # The sink's original error wins over the end hook failure.
    assert str(exc_info.value) == "sink exploded on 'event-1'"
    assert fixture.sent == []


def test_batch_end_hook_failure_without_sink_error():
    fixture = BatchFixture()
    hook_calls = []

    def broken_on_batch_end(succeeded, first_error):
        hook_calls.append((succeeded, first_error))
        raise RuntimeError("hook exploded")

    sink = fixture.make_sink(on_batch_end=broken_on_batch_end)

    with pytest.raises(RuntimeError, match="hook exploded"):
        sink.send_batch(["event-1", "event-2"])

    assert fixture.sent == ["event-1", "event-2"]
    assert hook_calls == [(2, None)]


def test_batch_all_events_succeed():
    fixture = BatchFixture()
    sink = fixture.make_sink()

    sink.send_batch(["event-1", "event-2", "event-3"])

    assert fixture.attempts == ["event-1", "event-2", "event-3"]
    assert fixture.sent == ["event-1", "event-2", "event-3"]
    assert fixture.started == 1
    assert fixture.completed == [(3, None)]


def test_batch_without_hooks():
    sent = []
    sink = BatchSink(sent.append)
    sink.send_batch(["event-1", "event-2"])
    assert sent == ["event-1", "event-2"]


def test_logger_per_message_sending_unchanged():
    fixture = BatchFixture()
    sink = fixture.make_sink()

    handler_id = logger.add(sink, format="{message}")
    logger.info("message-1")
    logger.info("message-2")
    logger.remove(handler_id)

    # Regular logging goes through "write()" one message at a time, without hooks.
    assert fixture.sent == ["message-1\n", "message-2\n"]
    assert fixture.started == 0
    assert fixture.completed == []


def test_batch_report_written_to_tmp_path(tmp_path):
    fixture = BatchFixture(fail_on={"event-2"})
    sink = fixture.make_sink()

    with pytest.raises(ValueError, match="sink exploded on 'event-2'"):
        sink.send_batch(["event-1", "event-2", "event-3"])

    report = tmp_path / "batch_report.txt"
    succeeded, first_error = fixture.completed[0]
    report.write_text(
        "attempts=%d sent=%d succeeded=%d error=%s\n"
        % (len(fixture.attempts), len(fixture.sent), succeeded, first_error),
        encoding="utf8",
    )

    content = report.read_text(encoding="utf8")
    assert content == "attempts=2 sent=1 succeeded=1 error=sink exploded on 'event-2'\n"
