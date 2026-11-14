"""Application lifecycle state machine (PRD 5.1)."""

from __future__ import annotations

from enum import Enum


class State(Enum):
    """Lifecycle states. DEGRADED is a flag on the machine (shown with SIMULATING), not a state."""

    UNINITIALIZED = "UNINITIALIZED"
    CALIBRATING = "CALIBRATING"
    ACQUIRING = "ACQUIRING"
    READY = "READY"
    SIMULATING = "SIMULATING"
    PAUSED = "PAUSED"
    ERROR = "ERROR"


class IllegalTransition(RuntimeError):
    """Raised when a lifecycle transition is not in the allowed table."""


S = State
ALLOWED: dict[State, frozenset[State]] = {
    S.UNINITIALIZED: frozenset({S.CALIBRATING, S.ERROR}),
    S.CALIBRATING: frozenset({S.ACQUIRING, S.ERROR}),
    S.ACQUIRING: frozenset({S.READY, S.ERROR}),
    S.READY: frozenset({S.SIMULATING, S.ERROR}),
    S.SIMULATING: frozenset({S.PAUSED, S.ERROR}),
    S.PAUSED: frozenset({S.SIMULATING, S.ACQUIRING, S.ERROR}),  # PAUSED -> ACQUIRING: user re-acquire request
    S.ERROR: frozenset({S.UNINITIALIZED}),
}


class Lifecycle:
    """State machine with an explicit allowed-transition table. ``degraded`` is only meaningful while
    SIMULATING or PAUSED (it is cleared on any other state). Not thread-safe."""

    def __init__(self) -> None:
        self.state = State.UNINITIALIZED
        self.degraded = False
        self.history: list[State] = [self.state]

    def transition(self, to: State) -> None:
        """Move to ``to``; raises ``IllegalTransition`` if the table does not allow it."""
        if to not in ALLOWED[self.state]:
            raise IllegalTransition(f"{self.state.value} -> {to.value} is not allowed")
        self.state = to
        if to not in (State.SIMULATING, State.PAUSED):
            self.degraded = False
        self.history.append(to)

    def set_degraded(self, flag: bool) -> None:
        """Set the DEGRADED flag (allowed only while SIMULATING or PAUSED)."""
        if flag and self.state not in (State.SIMULATING, State.PAUSED):
            raise IllegalTransition(f"cannot degrade in {self.state.value}")
        self.degraded = flag

    @property
    def label(self) -> str:
        """HUD label, e.g. ``SIMULATING+DEGRADED``."""
        return self.state.value + ("+DEGRADED" if self.degraded else "")
