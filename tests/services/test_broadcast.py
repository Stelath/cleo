"""Tests for services.media.broadcast.BroadcastHub."""

import queue
import threading

from services.media.broadcast import BroadcastHub


def test_single_subscriber_receives_all():
    hub = BroadcastHub()
    sid, q = hub.subscribe()

    hub.publish("a")
    hub.publish("b")
    hub.publish("c")

    assert q.get_nowait() == "a"
    assert q.get_nowait() == "b"
    assert q.get_nowait() == "c"

    hub.unsubscribe(sid)


def test_multiple_subscribers():
    hub = BroadcastHub()
    sid1, q1 = hub.subscribe()
    sid2, q2 = hub.subscribe()

    hub.publish(42)

    assert q1.get_nowait() == 42
    assert q2.get_nowait() == 42

    hub.unsubscribe(sid1)
    hub.unsubscribe(sid2)


def test_unsubscribe_stops_delivery():
    hub = BroadcastHub()
    sid, q = hub.subscribe()

    hub.publish("before")
    hub.unsubscribe(sid)
    hub.publish("after")

    assert q.get_nowait() == "before"
    assert q.empty()


def test_drop_oldest_on_overflow():
    hub = BroadcastHub(maxsize=2)
    sid, q = hub.subscribe()

    hub.publish(1)
    hub.publish(2)
    hub.publish(3)  # should drop 1

    assert q.get_nowait() == 2
    assert q.get_nowait() == 3
    assert q.empty()

    hub.unsubscribe(sid)


def test_subscriber_count():
    hub = BroadcastHub()
    assert hub.subscriber_count == 0

    sid1, _ = hub.subscribe()
    assert hub.subscriber_count == 1

    sid2, _ = hub.subscribe()
    assert hub.subscriber_count == 2

    hub.unsubscribe(sid1)
    assert hub.subscriber_count == 1

    hub.unsubscribe(sid2)
    assert hub.subscriber_count == 0


def test_thread_safety():
    hub = BroadcastHub(maxsize=256)
    results = {i: [] for i in range(4)}
    barrier = threading.Barrier(5)  # 4 subscribers + 1 producer

    def consumer(idx):
        sid, q = hub.subscribe()
        barrier.wait()
        for _ in range(100):
            try:
                results[idx].append(q.get(timeout=2))
            except queue.Empty:
                break
        hub.unsubscribe(sid)

    threads = [threading.Thread(target=consumer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()

    barrier.wait()
    for i in range(100):
        hub.publish(i)

    for t in threads:
        t.join(timeout=5)

    # Each subscriber should have received all 100 items
    for idx in range(4):
        assert len(results[idx]) == 100
