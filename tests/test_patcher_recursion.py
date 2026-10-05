"""Tests for the explicit reentrancy protection of patchers.

A patcher which logs again through the very same chain of patchers must not
recurse endlessly: the reentrant call is diverted to a fixed bypass (patchers
skipped, full record not exposed to them, controlled diagnostic marker set).
The guard is scoped to the calling context (a ``ContextVar``), so other
threads keep logging normally and the state is restored even on exceptions.

Run locally with:

    python -m pytest tests/test_patcher_recursion.py -v

The module can also be executed directly to run a two-thread demonstration:

    python tests/test_patcher_recursion.py
"""

import threading

import pytest

if __name__ == "__main__" and __package__ is None:
    # Allow running this file directly as a two-thread demonstration, even
    # when Loguru is not installed in the current environment.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from loguru._logger import patcher_context


def make_record_sink():
    """Return the list collecting the full record of each logged message."""
    records = []
    logger.add(lambda message: records.append(message.record), format="{message}")
    return records


def test_reentrant_core_patcher_is_bypassed(writer):
    records = make_record_sink()
    patched_messages = []

    def patcher(record):
        # This would recurse endlessly without the guard: the inner call goes
        # through the very same chain (the core patcher itself).
        patched_messages.append(record["message"])
        record["extra"]["patched"] = True
        if record["message"] == "outer":
            logger.info("inner")

    logger.configure(patcher=patcher)

    logger.info("outer")

    # The patcher ran exactly once: the reentrant record did not leak into it.
    assert patched_messages == ["outer"]
    # Both messages were emitted exactly once (the inner one first, since the
    # patcher runs before the outer record is emitted), the outer record
    # being processed exactly once.
    assert [record["message"] for record in records] == ["inner", "outer"]
    # The inner record went through the fixed bypass: not patched, but marked
    # with the controlled diagnostic flag.
    assert records[1]["extra"].get("patched") is True
    assert "patcher_recursion_bypassed" not in records[1]["extra"]
    assert records[0]["extra"].get("patched") is None
    assert records[0]["extra"]["patcher_recursion_bypassed"] is True


def test_reentrant_patch_chain_is_bypassed(writer):
    records = make_record_sink()
    patched_messages = []
    holder = {}

    def patcher(record):
        patched_messages.append(record["message"])
        if record["message"] == "outer":
            holder["logger"].info("inner")

    patched_logger = logger.patch(patcher)
    holder["logger"] = patched_logger

    patched_logger.info("outer")

    assert patched_messages == ["outer"]
    assert [record["message"] for record in records] == ["inner", "outer"]
    assert records[0]["extra"]["patcher_recursion_bypassed"] is True


def test_chain_identity_follows_patcher_composition(writer):
    # Two loggers built with the exact same patcher composition share the same
    # chain identity, hence the same reentrancy protection.
    records = make_record_sink()
    holder = {}

    def patcher(record):
        if record["message"] == "outer":
            # A distinct Logger object, but the very same patcher composition.
            holder["other"].info("inner")

    holder["other"] = logger.patch(patcher)
    logger.patch(patcher).info("outer")

    assert [record["message"] for record in records] == ["inner", "outer"]
    assert records[0]["extra"]["patcher_recursion_bypassed"] is True


def test_distinct_chain_is_not_blocked(writer):
    records = make_record_sink()
    calls = []
    holder = {}

    def patcher_a(record):
        calls.append("a")
        if record["message"] == "outer":
            holder["logger_b"].info("inner")

    def patcher_b(record):
        calls.append("b")
        record["extra"]["from_b"] = True

    holder["logger_b"] = logger.patch(patcher_b)
    logger.patch(patcher_a).info("outer")

    # The inner call used a different chain, so its patcher ran normally.
    assert calls == ["a", "b"]
    assert [record["message"] for record in records] == ["inner", "outer"]
    assert records[0]["extra"]["from_b"] is True
    assert "patcher_recursion_bypassed" not in records[0]["extra"]


def test_mutual_recursion_terminates(writer):
    records = make_record_sink()
    holder = {}

    def patcher_a(record):
        if record["message"] == "start":
            holder["logger_b"].info("via-b")

    def patcher_b(record):
        if record["message"] == "via-b":
            holder["logger_a"].info("via-a")

    holder["logger_a"] = logger.patch(patcher_a)
    holder["logger_b"] = logger.patch(patcher_b)

    holder["logger_a"].info("start")

    # A -> B -> A: the third call re-enters chain "A" and is bypassed, so the
    # mutual recursion terminates instead of overflowing the stack. Records
    # are emitted innermost-first.
    assert [record["message"] for record in records] == ["via-a", "via-b", "start"]
    assert records[0]["extra"]["patcher_recursion_bypassed"] is True


def test_other_thread_is_not_blocked(writer):
    logger.add(writer, format="{message}")
    calls = []
    entered = threading.Event()
    release = threading.Event()

    def patcher(record):
        calls.append(record["message"])
        if record["message"] == "main":
            entered.set()
            # Keep the guard active in this thread while the other one logs.
            assert release.wait(timeout=10)

    patched_logger = logger.patch(patcher)

    def worker():
        assert entered.wait(timeout=10)
        # Same chain, but a different thread: must be patched normally.
        patched_logger.info("worker")
        release.set()

    thread = threading.Thread(target=worker)
    thread.start()
    patched_logger.info("main")
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert sorted(calls) == ["main", "worker"]
    assert writer.read() == "worker\nmain\n"


def test_guard_is_released_after_patcher_exception(writer):
    logger.add(writer, format="{message}")
    calls = []
    state = {"fail": True}

    def patcher(record):
        calls.append(record["message"])
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("boom")

    patched_logger = logger.patch(patcher)

    with pytest.raises(RuntimeError, match="boom"):
        patched_logger.info("first")

    # The guard was restored despite the exception: logging works again.
    patched_logger.info("second")

    assert calls == ["first", "second"]
    assert writer.read() == "second\n"
    assert patcher_context.get() is None


def test_plain_logger_is_not_guarded(writer):
    logger.add(writer, format="{message}")

    logger.info("plain")

    assert writer.read() == "plain\n"
    # No protection state was installed for a logger without any patcher.
    assert patcher_context.get() is None


def two_thread_demo():
    """Two-thread example: a reentrant patcher plus a concurrent worker.

    The main thread logs through a patcher which (mistakenly) logs again
    through the same patched logger: the inner record is bypassed once instead
    of recursing endlessly. Meanwhile, another thread logging through the very
    same chain is not blocked and gets patched normally.
    """
    import sys

    logger.remove()
    logger.add(sys.stderr, format="{thread.name} | {extra} | {message}")

    entered = threading.Event()
    release = threading.Event()
    holder = {}

    def patcher(record):
        record["extra"]["patched"] = True
        if record["message"] == "from-main":
            entered.set()
            release.wait(timeout=10)
            # Reentrant call through the very same chain: fixed bypass.
            holder["logger"].info("reentrant-from-patcher")

    patched_logger = logger.patch(patcher)
    holder["logger"] = patched_logger

    def worker():
        entered.wait(timeout=10)
        patched_logger.info("from-worker")
        release.set()

    thread = threading.Thread(target=worker, name="worker")
    thread.start()
    patched_logger.info("from-main")
    thread.join(timeout=10)
    logger.remove()


if __name__ == "__main__":
    two_thread_demo()
