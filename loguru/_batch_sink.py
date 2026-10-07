"""A synchronous sink adapter adding explicit batch begin/end hooks.

This module implements :class:`BatchSink`, an opt-in adapter which wraps an existing
synchronous sink (a callable or a file-like object) and adds batch boundaries on top of
it. Outside of a batch, messages are forwarded one by one exactly as before. Within an
explicit batch, each message is still sent immediately and attempted exactly once: if a
send fails, the failure is recorded and the batch continues without re-sending the
events that already succeeded. When the batch ends, the completion hook is invoked with
the number of successfully sent events and the first failure (if any), and the original
sink error is re-raised so it is never masked by a failing completion hook.
"""

from contextlib import contextmanager


class _BatchState:
    __slots__ = ("attempted", "succeeded", "first_failure")

    def __init__(self):
        self.attempted = 0
        self.succeeded = 0
        self.first_failure = None


class BatchSink:
    """A synchronous sink adapter providing begin/end hooks for explicit small batches.

    Parameters
    ----------
    sink
        The underlying synchronous sink: a callable accepting the message, or a
        file-like object with a ``write()`` method.
    on_batch_begin
        An optional callable invoked without arguments each time a batch begins.
    on_batch_end
        An optional callable invoked each time a batch ends. It receives two arguments:
        the number of successfully sent events and the first failure that occurred
        during the batch (or ``None`` if all events were sent successfully).
    """

    def __init__(self, sink, on_batch_begin=None, on_batch_end=None):
        write = getattr(sink, "write", None)
        if callable(write):
            self._send = write
        elif callable(sink):
            self._send = sink
        else:
            raise TypeError("Cannot batch to objects of type '%s'" % type(sink).__name__)
        self._sink = sink
        self._on_batch_begin = on_batch_begin
        self._on_batch_end = on_batch_end
        self._state = None

    @property
    def in_batch(self):
        """Tell whether a batch is currently active.

        Returns
        -------
        bool
            ``True`` if a batch has been begun and not yet ended.
        """
        return self._state is not None

    def write(self, message):
        """Send a message to the underlying sink.

        Outside of a batch, the message is forwarded immediately and errors propagate
        unchanged (per-record behavior is not modified). Within a batch, the message is
        attempted exactly once: a failure is recorded (keeping the first one) instead of
        being raised, so that the remaining events can still be attempted and the events
        already sent are never re-sent.

        Parameters
        ----------
        message
            The message to send.
        """
        state = self._state
        if state is None:
            self._send(message)
            return
        state.attempted += 1
        try:
            self._send(message)
        except Exception as error:
            if state.first_failure is None:
                state.first_failure = error
        else:
            state.succeeded += 1

    def flush(self):
        """Flush the underlying sink if it supports flushing."""
        flush = getattr(self._sink, "flush", None)
        if callable(flush):
            flush()

    def stop(self):
        """Stop the underlying sink if it supports being stopped or closed."""
        stop = getattr(self._sink, "stop", None)
        if callable(stop):
            stop()
            return
        close = getattr(self._sink, "close", None)
        if callable(close):
            close()

    def begin_batch(self):
        """Begin a new explicit batch of events.

        Raises
        ------
        RuntimeError
            If a batch is already active (batches cannot be nested).
        """
        if self._state is not None:
            raise RuntimeError("A batch is already active, nested batches are not supported")
        self._state = _BatchState()
        if self._on_batch_begin is not None:
            self._on_batch_begin()

    def end_batch(self):
        """End the current batch and report its completion.

        The ``on_batch_end`` hook (if any) is invoked with the number of successfully
        sent events and the first failure of the batch (or ``None``).

        Returns
        -------
        int
            The number of events successfully sent during the batch.

        Raises
        ------
        RuntimeError
            If no batch is currently active.
        Exception
            The first failure raised by the underlying sink during the batch, if any.
            This original sink error takes precedence: if the ``on_batch_end`` hook also
            fails, the hook error is chained as ``__context__`` and never masks the
            original sink error. If only the hook failed, the hook error is raised.
        """
        state = self._state
        if state is None:
            raise RuntimeError("There is no active batch to end")
        self._state = None
        if self._on_batch_end is not None:
            try:
                self._on_batch_end(state.succeeded, state.first_failure)
            except Exception:
                if state.first_failure is not None:
                    # The original sink error must not be masked by the failing hook:
                    # it is re-raised here, with the hook error as its "__context__".
                    raise state.first_failure
                raise
        if state.first_failure is not None:
            raise state.first_failure
        return state.succeeded

    @contextmanager
    def batch(self):
        """Create a context manager wrapping a batch (begin on enter, end on exit).

        If the wrapped block raises, the batch is still ended (so the completion hook
        runs), but batch finalization errors are discarded so they cannot mask the
        in-flight exception.

        Yields
        ------
        BatchSink
            The adapter itself, so events can be sent with ``write()`` in the block.
        """
        self.begin_batch()
        try:
            yield self
        except BaseException:
            try:
                self.end_batch()
            except Exception:
                pass  # Do not mask the in-flight exception with batch finalization errors.
            raise
        else:
            self.end_batch()
