"""Tests for explicit reentrancy protection of patchers.

When a patcher (or a sink called by a patcher) attempts to log a new message while the same
patcher chain is already running, the nested call must be skipped with a controlled diagnostic
instead of recursing forever. The protection is per call-context (thread-local), so it does not
block normal logging from another thread, and it is cleaned up even if a patcher raises.
"""

import threading

import pytest

from loguru import logger


def test_reentrant_patcher_is_skipped(writer, capsys):
    calls = []

    def patcher(record):
        calls.append("patcher")
        logger.info("nested")  # reentrant: same chain already running

    logger.remove()
    logger.configure(patcher=patcher)
    logger.add(writer, format="{message}")

    logger.info("outer")

    # The patcher ran once, the outer record was processed once, the nested one was skipped.
    assert calls == ["patcher"]
    assert writer.read() == "outer\n"

    # A controlled diagnostic was emitted, and it does not leak the record.
    captured = capsys.readouterr()
    assert "Logging error in Loguru Patcher" in captured.err
    assert "infinite recursion" in captured.err
    assert "nested" not in captured.err
    assert "outer" not in captured.err


def test_reentrant_patcher_does_not_block_other_thread(writer):
    def patcher(record):
        logger.info("nested")  # reentrant in every thread

    logger.remove()
    logger.configure(patcher=patcher)
    logger.add(writer, format="{message}")

    barrier = threading.Barrier(2)
    errors = []

    def worker(name):
        barrier.wait()
        try:
            logger.info("outer-%s" % name)
        except Exception as exc:
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("a",))
    t2 = threading.Thread(target=worker, args=("b",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert not errors

    # Each thread processed its own outer record once; the nested record was skipped everywhere.
    assert "outer-a\n" in writer.read()
    assert "outer-b\n" in writer.read()
    assert "nested\n" not in writer.read()


def test_logging_remains_available_after_patcher_error(writer):
    state = {"fail": True}

    def patcher(record):
        if state["fail"]:
            state["fail"] = False
            raise RuntimeError("boom")

    logger.remove()
    logger.configure(patcher=patcher)
    logger.add(writer, format="{message}")

    with pytest.raises(RuntimeError, match="boom"):
        logger.info("will-fail")

    # The reentrancy guard was cleaned up despite the exception, so logging works again.
    logger.info("after-recovery")
    assert writer.read() == "after-recovery\n"


def test_plain_logger_without_patcher_is_not_protected(writer, capsys):
    logger.remove()
    logger.configure(patcher=None)
    logger.add(writer, format="{message}")

    logger.info("plain")

    assert writer.read() == "plain\n"
    captured = capsys.readouterr()
    assert "Logging error in Loguru Patcher" not in captured.err


def test_chain_identity_follows_real_patcher_composition(writer):
    def patcher_f1(record):
        # Log through a differently-patched logger: its chain is (f2,), not (f1,), so it must
        # not be considered reentrant and must run normally.
        logger.patch(patcher_f2).info("from-f1")

    def patcher_f2(record):
        record["extra"]["f2"] = True

    logger.remove()
    logger.configure(patcher=None)
    logger.add(writer, format="{message}")

    logger.patch(patcher_f1).info("outer")

    # The outer record was processed once, and the nested log through the different chain ran.
    assert writer.read() == "from-f1\nouter\n"


def test_reentrant_patched_logger_is_skipped(writer):
    calls = []

    def patcher(record):
        calls.append("patcher")
        # Re-log through the same patched composition: chain is (patcher,) again, so it is
        # reentrant and must be skipped.
        logger.patch(patcher).info("nested")

    logger.remove()
    logger.configure(patcher=None)
    logger.add(writer, format="{message}")

    logger.patch(patcher).info("outer")

    assert calls == ["patcher"]
    assert writer.read() == "outer\n"
