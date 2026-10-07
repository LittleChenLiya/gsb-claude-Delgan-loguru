import inspect
import logging
import weakref

from ._asyncio_loop import get_running_loop, get_task_loop


class StreamSink:
    """A sink that writes log messages to a stream object.

    Parameters
    ----------
    stream
        A stream object that supports write operations.
    """

    def __init__(self, stream):
        self._stream = stream
        self._flushable = callable(getattr(stream, "flush", None))
        self._stoppable = callable(getattr(stream, "stop", None))
        self._completable = inspect.iscoroutinefunction(getattr(stream, "complete", None))

    def write(self, message):
        """Write a message to the stream.

        Parameters
        ----------
        message
            The message to write.
        """
        self._stream.write(message)
        if self._flushable:
            self._stream.flush()

    def stop(self):
        """Stop the stream if it supports the stop operation."""
        if self._stoppable:
            self._stream.stop()

    def tasks_to_complete(self):
        """Return list of tasks that need to be completed.

        Returns
        -------
        list
            List of tasks to complete.
        """
        if not self._completable:
            return []
        return [self._stream.complete()]


class StandardSink:
    """A sink that writes log messages using the standard logging module.

    Parameters
    ----------
    handler
        A logging handler instance.
    """

    def __init__(self, handler):
        self._handler = handler

    def write(self, message):
        """Write a message using the standard logging handler.

        Parameters
        ----------
        message
            The message to write.
        """
        if message.record["level"].no < self._handler.level:
            return
        raw_record = message.record
        message = str(message)
        exc = raw_record["exception"]
        record = logging.getLogger().makeRecord(
            raw_record["name"],
            raw_record["level"].no,
            raw_record["file"].path,
            raw_record["line"],
            message,
            (),
            (exc.type, exc.value, exc.traceback) if exc else None,
            raw_record["function"],
            {"extra": raw_record["extra"]},
        )

        # By default, the standard logging module will format the exception and assign it to the
        # "exc_text" attribute. Then, the formatted exception will be automatically appended to the
        # message when the record is formatted. This is a problem, because that would cause the
        # exception to be duplicated in the log message, since it's also formatted by Loguru. To
        # avoid this, we set "exc_text" to a simple newline character, which will end the message.
        if exc:
            record.exc_text = "\n"

        record.levelname = raw_record["level"].name
        self._handler.handle(record)

    def stop(self):
        """Close the logging handler."""
        self._handler.close()

    def tasks_to_complete(self):
        """Return list of tasks that need to be completed.

        Returns
        -------
        list
            Empty list as standard sink has no async tasks.
        """
        return []


class AsyncSink:
    """A sink that handles asynchronous logging operations.

    Parameters
    ----------
    function
        The async function to execute.
    loop
        The event loop to use.
    error_interceptor
        An interceptor for handling errors.
    """

    def __init__(self, function, loop, error_interceptor):
        self._function = function
        self._loop = loop
        self._error_interceptor = error_interceptor
        self._tasks = weakref.WeakSet()

    def write(self, message):
        """Asynchronously write a message.

        Parameters
        ----------
        message
            The message to write.
        """
        try:
            loop = self._loop or get_running_loop()
        except RuntimeError:
            return

        coroutine = self._function(message)
        task = loop.create_task(coroutine)

        def check_exception(future):
            if future.cancelled() or future.exception() is None:
                return
            if not self._error_interceptor.should_catch():
                raise future.exception()
            self._error_interceptor.print(message.record, exception=future.exception())

        task.add_done_callback(check_exception)
        self._tasks.add(task)

    def stop(self):
        """Cancel all pending tasks."""
        for task in self._tasks:
            task.cancel()

    def tasks_to_complete(self):
        """Return list of tasks that need to be completed.

        Returns
        -------
        list
            List of tasks to complete.
        """
        # To avoid errors due to "self._tasks" being mutated while iterated, the
        # "tasks_to_complete()" method must be protected by the same lock as "write()" (which
        # happens to be the handler lock). However, the tasks must not be awaited while the lock is
        # acquired as this could lead to a deadlock. Therefore, we first need to collect the tasks
        # to complete, then return them so that they can be awaited outside of the lock.
        return [self._complete_task(task) for task in self._tasks]

    async def _complete_task(self, task):
        """Complete a single task.

        Parameters
        ----------
        task
            The task to complete.
        """
        loop = get_running_loop()
        if get_task_loop(task) is not loop:
            return
        try:
            await task
        except Exception:
            pass  # Handled in "check_exception()"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tasks"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tasks = weakref.WeakSet()


class CallableSink:
    """A sink that executes a callable function for each log message.

    Parameters
    ----------
    function
        The function to call for each message.
    """

    def __init__(self, function):
        self._function = function

    def write(self, message):
        """Write a message by calling the function.

        Parameters
        ----------
        message
            The message to pass to the function.
        """
        self._function(message)

    def stop(self):
        """Stop the sink (no-op for callable sink)."""
        pass

    def tasks_to_complete(self):
        """Return list of tasks that need to be completed.

        Returns
        -------
        list
            Empty list as callable sink has no tasks.
        """
        return []


class BatchSink:
    """A synchronous sink adapter adding begin/end hooks around explicit batches.

    The adapter wraps a callable sink. Messages written individually through
    :meth:`write` are forwarded one by one, exactly like a regular callable sink.
    Explicit batches sent through :meth:`send_batch` are delimited by the
    ``on_batch_start`` and ``on_batch_end`` hooks, the latter receiving the number of
    successfully delivered messages and the first error raised by the sink (if any).

    Parameters
    ----------
    function
        The callable invoked for each message.
    on_batch_start
        An optional callable invoked without argument before a batch is sent.
    on_batch_end
        An optional callable invoked as ``on_batch_end(succeeded, first_error)`` once a
        batch has been processed, where ``succeeded`` is the number of messages
        successfully delivered and ``first_error`` is the first exception raised by the
        sink (or ``None`` if the whole batch succeeded).
    """

    def __init__(self, function, on_batch_start=None, on_batch_end=None):
        self._function = function
        self._on_batch_start = on_batch_start
        self._on_batch_end = on_batch_end

    def write(self, message):
        """Write a single message, forwarded as-is to the wrapped callable.

        Parameters
        ----------
        message
            The message to pass to the function.
        """
        self._function(message)

    def send_batch(self, messages):
        """Send an explicit batch of messages, guarded by the begin/end hooks.

        Messages are forwarded one by one. As soon as one message fails, the remaining
        ones are skipped and messages already delivered are not sent again. The end
        hook is always executed and receives the count of successful deliveries along
        with the first error encountered. If the end hook itself fails while the sink
        already failed, the sink error is the one propagated, so that the original
        cause is not masked by the hook failure.

        Parameters
        ----------
        messages
            An iterable of messages forming the batch.

        Raises
        ------
        Exception
            The first error raised by the wrapped callable, if any. Otherwise, the
            error raised by the end hook, if any.
        """
        if self._on_batch_start is not None:
            self._on_batch_start()

        succeeded = 0
        first_error = None
        for message in messages:
            try:
                self._function(message)
            except Exception as error:
                first_error = error
                break
            succeeded += 1

        hook_error = None
        if self._on_batch_end is not None:
            try:
                self._on_batch_end(succeeded, first_error)
            except Exception as error:
                hook_error = error

        if first_error is not None:
            raise first_error
        if hook_error is not None:
            raise hook_error

    def stop(self):
        """Stop the sink (no-op for batch sink)."""
        pass

    def tasks_to_complete(self):
        """Return list of tasks that need to be completed.

        Returns
        -------
        list
            Empty list as batch sink is fully synchronous.
        """
        return []
