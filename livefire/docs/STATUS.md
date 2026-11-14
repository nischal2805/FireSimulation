# STATUS

| Field | Value |
| --- | --- |
| Phase | P1 in progress (Gate 0 approved 2026-11-14) |
| Last passing commit | 13425f0 (`make test`: 249 passed, 4 slow deselected; CPU) |
| Gate 1 | not started |

## Completed tasks
- P0 (all), Gate 0 report `docs/reports/phase-0.md`
- P1-001 live camera sources, P1-002 click-to-ignite (synthetic), P1-003 ChArUco calibration, P1-004 depth spike
  (ADR-0005 proposed) — implemented and reviewed
- P1-005 acquisition/fusion — implemented, review pending

## Blockers
- Nothing has run on the user's RTX 4060 / Windows yet (GPU, cameras, window).
- Monocular depth accuracy risk (see `docs/HANDOVER.md` section 4).

## Active ADRs
- ADR-0001 (proposed), ADR-0002, ADR-0003, ADR-0004, ADR-0005 (proposed)

## Next task
- Review P1-005, then P1-006 live mode (`docs/tasks/P1-plan.md`). Handover notes: `docs/HANDOVER.md`.
