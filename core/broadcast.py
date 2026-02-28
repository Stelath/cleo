"""Single-producer, multi-consumer fan-out hub for streaming data."""

import queue
import threading
from typing import TypeVar

T = TypeVar("T")


class BroadcastHub:
    """Thread-safe fan-out: one producer publishes items to N subscriber queues.

    Subscribers receive a bounded queue.  When a subscriber's queue is full the
    oldest item is silently dropped so a slow consumer never blocks the producer.
    """

    def __init__(self, maxsize: int = 64):
        self._lock = threading.Lock()
        self._subscribers: dict[int, queue.Queue] = {}
        self._next_id = 0
        self._maxsize = maxsize

    def subscribe(self) -> tuple[int, queue.Queue]:
        """Register a new consumer. Returns (subscriber_id, queue)."""
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            q: queue.Queue = queue.Queue(maxsize=self._maxsize)
            self._subscribers[sid] = q
            return sid, q

    def unsubscribe(self, sid: int) -> None:
        """Remove a consumer by ID."""
        with self._lock:
            self._subscribers.pop(sid, None)

    def publish(self, item) -> None:
        """Send *item* to every subscriber queue (drop-oldest on overflow)."""
        with self._lock:
            for q in self._subscribers.values():
                try:
                    q.put_nowait(item)
                except queue.Full:
                    # Drop oldest, then enqueue
                    try:
                        q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(item)
                    except queue.Full:
                        pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)
