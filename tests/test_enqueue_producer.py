"""Targeted tests for producer attribution of enqueued messages.

When ``enqueue=True``, each emitted message is stamped at emission time with a
``ProducerStamp(producer_id, sequence)`` (see ``loguru._handler``):

- ``producer_id`` comes from ``logger.bind(producer_id=...)`` and falls back to the
  id of the emitting thread; the arrival order (or arrival time) of messages in the
  writer thread is never used to guess the producer.
- ``sequence`` is a local, per-producer counter which is committed only after the
  record has been successfully serialized and enqueued. A failed serialization
  consumes no number and leaves no "sent" receipt behind.

A few record examples, with two producers "A" and "B" whose messages happen to
arrive interleaved (B first). Each producer keeps its own contiguous sequence::

    producer_b.info("b1")  # -> ProducerStamp(producer_id='B', sequence=1)
    producer_a.info("a1")  # -> ProducerStamp(producer_id='A', sequence=1)
    producer_b.info("b2")  # -> ProducerStamp(producer_id='B', sequence=2)
    producer_a.info("a2")  # -> ProducerStamp(producer_id='A', sequence=2)

and the handler receipts then read::

    handler.send_receipts == (
        ProducerStamp(producer_id='B', sequence=1),
        ProducerStamp(producer_id='A', sequence=1),
        ProducerStamp(producer_id='B', sequence=2),
        ProducerStamp(producer_id='A', sequence=2),
    )

Receipts store plain ``(producer_id, sequence)`` values only: they never retain the
record, its ``extra`` dict or any other bound context value.
"""

import json
import threading

import pytest

from loguru import logger
from loguru._handler import ProducerStamp


def get_handler(handler_id):
    return logger._core.handlers[handler_id]


def test_two_producers_reverse_arrival_order(writer):
    handler_id = logger.add(writer, format="{message}", enqueue=True)
    handler = get_handler(handler_id)

    producer_a = logger.bind(producer_id="A")
    producer_b = logger.bind(producer_id="B")

    # Messages of producer "B" arrive before (and interleaved with) the ones of
    # producer "A": attribution must follow the producer, not the arrival order.
    producer_b.info("b1")
    producer_a.info("a1")
    producer_b.info("b2")
    producer_a.info("a2")
    logger.complete()

    expected = [
        ProducerStamp(producer_id="B", sequence=1),
        ProducerStamp(producer_id="A", sequence=1),
        ProducerStamp(producer_id="B", sequence=2),
        ProducerStamp(producer_id="A", sequence=2),
    ]
    assert [message.producer for message in writer.written] == expected
    assert list(handler.send_receipts) == expected

    # Each producer is individually recognizable through its own contiguous sequence.
    assert [s.sequence for s in expected if s.producer_id == "A"] == [1, 2]
    assert [s.sequence for s in expected if s.producer_id == "B"] == [1, 2]


def test_failed_serialization_leaves_no_receipt_and_sequence_stays_contiguous(writer):
    handler_id = logger.add(
        writer, format="{extra[needed]} {message}", enqueue=True, catch=False
    )
    handler = get_handler(handler_id)

    ok = logger.bind(producer_id="P", needed="ok")
    broken = logger.bind(producer_id="P")

    ok.info("first")
    with pytest.raises(ValueError, match="Failed to format log record"):
        broken.info("boom")
    ok.info("second")
    logger.complete()

    expected = [
        ProducerStamp(producer_id="P", sequence=1),
        ProducerStamp(producer_id="P", sequence=2),
    ]
    # The failed record was neither sent nor stamped...
    assert [str(message) for message in writer.written] == ["ok first\n", "ok second\n"]
    assert [message.producer for message in writer.written] == expected
    # ...it left no "sent" receipt behind...
    assert list(handler.send_receipts) == expected
    # ...and it consumed no sequence number: the next success is contiguous.
    assert handler._producer_sequences == {"P": 2}


def test_failed_dynamic_formatter_leaves_no_receipt(writer):
    def formatter(record):
        if record["extra"].get("explode"):
            raise RuntimeError("cannot serialize this record")
        return "{message}\n"

    handler_id = logger.add(writer, format=formatter, enqueue=True, catch=False)
    handler = get_handler(handler_id)

    producer = logger.bind(producer_id="Q")
    producer.info("one")
    with pytest.raises(RuntimeError, match="cannot serialize this record"):
        producer.bind(explode=True).info("two")
    producer.info("three")
    logger.complete()

    expected = [
        ProducerStamp(producer_id="Q", sequence=1),
        ProducerStamp(producer_id="Q", sequence=2),
    ]
    assert [message.producer for message in writer.written] == expected
    assert list(handler.send_receipts) == expected


def test_receipt_does_not_store_context_values(writer):
    handler_id = logger.add(writer, format="{message}", enqueue=True)
    handler = get_handler(handler_id)

    context_value = {"secret": object()}
    payload = object()
    logger.bind(producer_id="P", context=context_value, payload=payload).info("hello")
    logger.complete()

    (receipt,) = handler.send_receipts
    assert receipt == ProducerStamp(producer_id="P", sequence=1)
    # The receipt is a bare (producer_id, sequence) pair: it retains neither the
    # record nor any bound "extra" / context value.
    assert receipt._fields == ("producer_id", "sequence")
    assert not hasattr(receipt, "record")
    assert context_value not in receipt
    assert context_value["secret"] not in receipt
    assert payload not in receipt
    # The stamp travels with the message itself (the queue pickles messages, so the
    # received stamp is an equal copy of the emitted one).
    assert writer.written[0].producer == receipt


def test_serialized_output_format_stays_compatible(writer):
    logger.add(writer, format="{message}", enqueue=True, serialize=True)
    logger.bind(producer_id="P").info("hello")
    logger.complete()

    (line,) = writer.written
    payload = json.loads(line)
    assert set(payload) == {"text", "record"}
    assert set(payload["record"]) == {
        "elapsed",
        "exception",
        "extra",
        "file",
        "function",
        "level",
        "line",
        "message",
        "module",
        "name",
        "process",
        "template",
        "thread",
        "time",
    }
    assert payload["record"]["extra"] == {"producer_id": "P"}
    assert payload["record"]["message"] == "hello"
    assert payload["text"] == "hello\n"


def test_default_producer_falls_back_to_emitting_thread(writer):
    handler_id = logger.add(writer, format="{message}", enqueue=True)
    handler = get_handler(handler_id)

    logger.info("one")
    logger.info("two")
    logger.complete()

    expected = [
        ProducerStamp(producer_id=threading.get_ident(), sequence=1),
        ProducerStamp(producer_id=threading.get_ident(), sequence=2),
    ]
    assert [message.producer for message in writer.written] == expected
    assert list(handler.send_receipts) == expected


def test_non_enqueued_message_has_no_producer_stamp(writer):
    handler_id = logger.add(writer, format="{message}", enqueue=False)
    handler = get_handler(handler_id)

    logger.bind(producer_id="P").info("plain")

    assert writer.written[0].producer is None
    assert handler.send_receipts == ()
