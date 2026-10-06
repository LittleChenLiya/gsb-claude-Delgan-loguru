"""Producer identifiers and local sequence numbers for enqueued messages.

The sequence number is attributed by the producer (the process that serialized
the message) and is committed only after the message has been handed over to
the queue. It is never recomputed from the arrival order or the arrival time on
the writer side. The report only records ``(producer_id, sequence)`` receipts,
never the record itself nor its context ("extra") values.
"""

import json
import pickle

import pytest

from loguru import logger
from loguru._enqueue import EnqueueBook, EnqueueReceipt
from loguru._handler import Message


class Unserializable:
    def __reduce__(self):
        raise pickle.PicklingError("cannot cross the queue boundary")


# A few small record examples shared by the fixtures below. They only use basic
# types so that a minimal stand-in record can travel through multiprocessing
# queues in tests.
SAMPLE_RECORDS = [
    {
        "message": "first",
        "level": "INFO",
        "extra": {"event": "startup"},
    },
    {
        "message": "second",
        "level": "WARNING",
        "extra": {"event": "retry"},
    },
    {
        "message": "third",
        "level": "ERROR",
        "extra": {"event": "failure"},
    },
]


def make_record(message="message", extra=None):
    return {
        "message": message,
        "level": {"name": "INFO", "no": 20},
        "extra": {} if extra is None else dict(extra),
    }


def first_handler():
    return next(iter(logger._core.handlers.values()))


def craft_enqueued(text, producer_id, sequence, record=None):
    """Build a message as it would be stamped by a (possibly remote) producer."""
    message = Message(text, producer_id=producer_id, sequence=sequence)
    message.record = make_record() if record is None else record
    return message


def test_two_fake_producers_arrive_in_reverse_order(writer):
    """Each message keeps the sequence given by its own producer.

    Messages are pushed to the handler queue in reverse inter-producer order
    (producer "B" is seen before earlier messages of producer "A"), as could
    happen after a delay or a retry: attribution must not be based on arrival
    order or arrival time.
    """
    logger.add(writer, enqueue=True, format="{message}")
    handler = first_handler()

    queue = handler._queue

    a2 = craft_enqueued("A-2\n", "producer-a", 2)
    a1 = craft_enqueued("A-1\n", "producer-a", 1)
    b1 = craft_enqueued("B-1\n", "producer-b", 1)
    a3 = craft_enqueued("A-3\n", "producer-a", 3)

    # Reverse arrival: B-1 and A-3 reach the writer before A-1/A-2 did.
    queue.put(b1)
    queue.put(a3)
    queue.put(a2)
    queue.put(a1)

    logger.complete()
    logger.remove()

    seen = [(m.producer_id, m.sequence, str(m)) for m in writer.written]
    assert seen == [
        ("producer-b", 1, "B-1\n"),
        ("producer-a", 3, "A-3\n"),
        ("producer-a", 2, "A-2\n"),
        ("producer-a", 1, "A-1\n"),
    ]

    # Per-producer sequences remain independently recognizable and contiguous.
    sequences = {}
    for producer_id, sequence, _ in seen:
        sequences.setdefault(producer_id, []).append(sequence)
    assert sorted(sequences["producer-a"]) == [1, 2, 3]
    assert sorted(sequences["producer-b"]) == [1]


def test_failed_serialization_leaves_no_receipt_and_next_sequence_is_contiguous(
    writer, capsys
):
    logger.add(writer, enqueue=True, catch=True, format="{message}")
    handler = first_handler()

    logger.info("one")
    logger.bind(payload=Unserializable()).info("dropped")
    logger.info("two")

    logger.complete()
    receipts = handler.enqueue_receipts()
    logger.remove()

    assert writer.read() == "one\ntwo\n"

    received = [(m.producer_id, m.sequence) for m in writer.written]
    producer_id = received[0][0]
    assert received == [(producer_id, 1), (producer_id, 2)]
    assert receipts == [
        EnqueueReceipt(producer_id, 1),
        EnqueueReceipt(producer_id, 2),
    ]

    out, err = capsys.readouterr()
    assert out == ""
    assert "PicklingError: cannot cross the queue boundary" in err


def test_receipts_record_only_producer_id_and_sequence(writer):
    logger.add(writer, enqueue=True, format="{message}")
    handler = first_handler()

    logger.bind(event="secret-context").info("first")
    logger.bind(event="other-context").info("second")
    logger.complete()

    receipts = handler.enqueue_receipts()
    logger.remove()

    assert len(receipts) == 2
    assert all(isinstance(r, EnqueueReceipt) for r in receipts)
    assert receipts[0]._fields == ("producer_id", "sequence")
    producer_id = receipts[0].producer_id
    assert [r.sequence for r in receipts] == [1, 2]
    assert all(r.producer_id == producer_id for r in receipts)

    # The report must not carry the record or its context values.
    for receipt in receipts:
        assert not hasattr(receipt, "record")
        assert not hasattr(receipt, "context")
        assert not hasattr(receipt, "extra")
        assert not hasattr(receipt, "time")
    report = repr(receipts)
    assert "secret-context" not in report
    assert "other-context" not in report


def test_serialized_and_plain_output_stay_compatible(writer):
    """The text and the JSON payload keep their existing format and keys."""
    serialized = []

    def json_sink(message):
        serialized.append(message)

    logger.add(writer, enqueue=False, format="{message}")
    logger.add(json_sink, enqueue=True, serialize=True, format="{message}")

    logger.info("compat")
    logger.complete()

    json_handler = next(
        h for h in logger._core.handlers.values() if h._serialize and h._enqueue
    )
    receipts = json_handler.enqueue_receipts()
    logger.remove()

    assert writer.read() == "compat\n"
    assert isinstance(writer.written[0], Message)
    # Messages written by a non-enqueued sink carry no producer attribution.
    assert writer.written[0].producer_id is None
    assert writer.written[0].sequence is None

    payload = json.loads(serialized[0])
    assert set(payload.keys()) == {"text", "record"}
    assert payload["text"] == "compat\n"
    assert "producer_id" not in payload["record"]
    assert "sequence" not in payload["record"]
    assert payload["record"]["message"] == "compat"

    # The enqueued Message itself is stamped, without affecting the payload.
    assert serialized[0].sequence == 1
    assert serialized[0].producer_id == receipts[0].producer_id


def test_enqueue_book_sequences_per_producer():
    book = EnqueueBook()
    producer_a = "fake-producer-a"
    producer_b = "fake-producer-b"

    assert book.reserve(producer_a) == 1
    # A failed serialization never commits, so reserving again yields the same
    # number and no receipt exists.
    book.retract(producer_a, 1)
    assert book.receipts() == []
    assert book.reserve(producer_a) == 1

    receipt_a1 = book.commit(producer_a, 1)
    assert receipt_a1 == EnqueueReceipt(producer_a, 1)

    # The other producer has an independent, local sequence.
    assert book.reserve(producer_b) == 1
    receipt_b1 = book.commit(producer_b, 1)
    assert receipt_b1 == EnqueueReceipt(producer_b, 1)

    assert book.reserve(producer_a) == 2
    book.commit(producer_a, 2)

    assert book.receipts() == [
        EnqueueReceipt(producer_a, 1),
        EnqueueReceipt(producer_b, 1),
        EnqueueReceipt(producer_a, 2),
    ]
    # Committing an out-of-order or duplicated number is a programming error.
    with pytest.raises(AssertionError):
        book.commit(producer_a, 2)
    with pytest.raises(AssertionError):
        book.commit(producer_a, 4)


def test_local_book_identity_is_contiguous_after_failure(writer):
    logger.add(writer, enqueue=True, format="{message}")
    handler = first_handler()
    book = handler._enqueue_book

    producer_id = book.producer_id
    assert isinstance(producer_id, str) and producer_id

    logger.info("a")
    logger.bind(broken=Unserializable()).info("dropped")
    logger.bind(broken=Unserializable()).info("dropped again")
    logger.info("b")
    logger.complete()

    receipts = handler.enqueue_receipts()
    logger.remove()

    assert receipts == [
        EnqueueReceipt(producer_id, 1),
        EnqueueReceipt(producer_id, 2),
    ]
    assert [m.sequence for m in writer.written] == [1, 2]
    assert all(m.producer_id == producer_id for m in writer.written)


def test_message_roundtrip_preserves_stamping():
    record = make_record("sample", SAMPLE_RECORDS[0]["extra"])
    message = craft_enqueued("first\n", "producer-a", 7, record)

    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(message, protocol))
        assert restored == "first\n"
        assert restored.producer_id == "producer-a"
        assert restored.sequence == 7
        assert restored.record["message"] == "sample"
        assert restored.record["extra"] == {"event": "startup"}


def test_sample_record_examples():
    assert [r["message"] for r in SAMPLE_RECORDS] == [
        "first",
        "second",
        "third",
    ]
    stamped = [
        craft_enqueued(
            r["message"] + "\n",
            "producer-a",
            index,
            make_record(r["message"], r["extra"]),
        )
        for index, r in enumerate(SAMPLE_RECORDS, start=1)
    ]
    assert [(m.producer_id, m.sequence, str(m)) for m in stamped] == [
        ("producer-a", 1, "first\n"),
        ("producer-a", 2, "second\n"),
        ("producer-a", 3, "third\n"),
    ]
    for message, example in zip(stamped, SAMPLE_RECORDS):
        assert message.record["extra"] == example["extra"]
