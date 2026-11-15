# STATUS

| Field | Value |
| --- | --- |
| Phase | P1 code complete (Gate 1 pending the user's physical runs); P2 and P3 in progress |
| Last passing commit | f00fc65 area (`make test`: 305 passed, 2 skipped, 1 xfailed; CPU, measured before P1-007 fix commit) |
| Gate 1 | NOT evaluated: needs real cameras + RTX 4060 (see docs/reports/phase-1.md, docs/RUNBOOK_P1.md) |

## Completed tasks
- P0 (all), Gate 0 approved. P1-001..P1-007 implemented and reviewed; P1-008 report written (`docs/reports/phase-1.md`).
- P2-001 radiation (modes 0/1/2) reviewed; on this branch.

## In progress
- P2-002 (radiation ADR-0006, audit; worktree branch worktree-agent-af7a64019b66f7883), P3-001 (segmentation backends).

## Blockers
- Nothing has run on the user's RTX 4060 / Windows yet (GPU, cameras, window). Gates 1-3 cannot be passed without it.
- Known deviation: cloth merges into table under geometric clustering (strict xfail; resolved by P3).

## Active ADRs
- ADR-0001 (proposed), ADR-0002, ADR-0003, ADR-0004, ADR-0005 (proposed)

## Next
- Finish P2-002..P2-006 and P3-001..P3-006 (`docs/tasks/P2-plan.md`, `P3-plan.md`).
