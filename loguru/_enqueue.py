import os
import threading
import uuid
from collections import namedtuple


EnqueueReceipt = namedtuple("EnqueueReceipt", ("producer_id", "sequence"))


class EnqueueBook:
    """Per-producer monotonic sequence numbers for enqueued messages.

    A sequence number is reserved before a message is handed over to the queue
    and it is committed only once the message has been successfully serialized
    and enqueued. If serialization fails, the tentative number is rolled back so
    that the next successfully enqueued message reuses it and the sequence of
    delivered receipts stays contiguous.

    The book intentionally records only the opaque producer identifier and the
    sequence number: it never stores the record, its context ("extra" values) or
    the arrival time. Attribution is thus always based on the producer that
    serialized the message rather than on the order in which it reached the
    sink.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pid = None
        self._producer_id = None
        self._counters = {}
        self._receipts = []

    def _ensure_owner(self):
        # A book inherited from a parent process (through "fork") must start a
        # fresh identity instead of reusing and extending the parent's ledger.
        pid = os.getpid()
        if self._pid != pid:
            self._pid = pid
            self._producer_id = "%d-%s" % (pid, uuid.uuid4().hex)
            self._counters = {}
            self._receipts = []

    @property
    def producer_id(self):
        """Opaque identifier of the producer running in the current process."""
        with self._lock:
            self._ensure_owner()
            return self._producer_id

    def reserve(self, producer_id=None):
        """Return the sequence number the next enqueued message would use.

        The number is not recorded until "commit()" is called, so a message
        whose serialization fails leaves no gap in the sequence.
        """
        with self._lock:
            self._ensure_owner()
            if producer_id is None:
                producer_id = self._producer_id
            return self._counters.get(producer_id, 0) + 1

    def commit(self, producer_id, sequence):
        """Record the receipt of a message that was successfully serialized."""
        with self._lock:
            expected = self._counters.get(producer_id, 0) + 1
            assert sequence == expected, (
                "Cannot commit sequence %d for producer %r, expected %d"
                % (sequence, producer_id, expected)
            )
            self._counters[producer_id] = sequence
            receipt = EnqueueReceipt(producer_id, sequence)
            self._receipts.append(receipt)
            return receipt

    def retract(self, producer_id, sequence):
        """Discard a tentative number after a failed serialization.

        No receipt was recorded for the number, so this is intentionally a
        no-op: the next successful message simply reserves the same number
        again, which keeps the committed sequence contiguous.
        """

    def receipts(self):
        """Return a copy of the receipts for messages actually enqueued."""
        with self._lock:
            return list(self._receipts)
