"""Lifecycle transition table (P0-010)."""

import pytest

from livefire.app.lifecycle import ALLOWED, IllegalTransition, Lifecycle, State


def test_happy_path_and_pause():
    lc = Lifecycle()
    for s in (State.CALIBRATING, State.ACQUIRING, State.READY, State.SIMULATING, State.PAUSED, State.SIMULATING):
        lc.transition(s)
    lc.set_degraded(True)
    assert lc.label == "SIMULATING+DEGRADED"
    lc.transition(State.PAUSED)
    assert lc.degraded
    lc.transition(State.ERROR)
    assert not lc.degraded
    lc.transition(State.UNINITIALIZED)


def test_every_illegal_transition_raises():
    for src in State:
        for dst in State:
            lc = Lifecycle()
            lc.state = src
            if dst in ALLOWED[src]:
                lc.transition(dst)
                assert lc.state is dst
            else:
                with pytest.raises(IllegalTransition):
                    lc.transition(dst)
                assert lc.state is src


def test_specific_illegal_and_degraded_flag():
    lc = Lifecycle()
    with pytest.raises(IllegalTransition):
        lc.transition(State.SIMULATING)
    with pytest.raises(IllegalTransition):
        lc.set_degraded(True)
    lc.transition(State.CALIBRATING)
    with pytest.raises(IllegalTransition):
        lc.transition(State.READY)
