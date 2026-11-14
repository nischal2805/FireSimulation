# STATUS

| Field | Value |
| --- | --- |
| Phase | P0 complete in code; Gate 0 awaiting the user's GPU run and approval |
| Last passing commit | 63829e8 (`make test`: 163 passed, 3 slow deselected; CPU) |
| Gate 0 | recommended PASSED_WITH_APPROVED_DEVIATION, pending items in `docs/reports/phase-0.md` |

## Completed tasks
- P0-001 .. P0-013 and P0-008b (see `docs/tasks/`), final whole-branch review and fix wave
- Gate 0 report: `docs/reports/phase-0.md`

## In progress
- none

## Blockers
- No GPU in the development container. GPU validation (`make test-slow`, `make bench-spike`, `make bench` with
  `device: "cuda:0"`) must run on the user's RTX 4060 (Windows 11).
- Literature full-text fetch blocked (search-only citations).

## Active ADRs
- ADR-0001 (proposed): Warp primary, Taichi fallback; decided by the 4060 spike run
- ADR-0002 MAC grid, ADR-0003 energy split, ADR-0004 combustion-intensity field

## Next task
- User: GPU run and Gate 0 approval. Then P1 task planning (live cameras, calibration, depth, click-to-ignite).
