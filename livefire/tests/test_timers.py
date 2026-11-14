import time

import pytest

from livefire.common.timers import StageTimer


def test_percentiles():
    t = StageTimer()
    for i in range(1, 101):
        t.record("a", i)
    s = t.summary()["a"]
    assert s["n"] == 100
    assert s["median_ms"] == pytest.approx(50.5)
    assert s["p95_ms"] == pytest.approx(95.05)
    assert s["p99_ms"] == pytest.approx(99.01)
    t.reset()
    assert t.summary() == {}


def test_stage_measures_sleep_and_syncs():
    t = StageTimer()
    calls = []
    with t.stage("s", device_sync=lambda: calls.append(1)):
        time.sleep(0.01)
    assert t.summary()["s"]["median_ms"] >= 9
    assert len(calls) == 2
