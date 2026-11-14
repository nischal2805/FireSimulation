# CLAUDE.md (LiveFire Previs)

Read `docs/PRD.md` (the full spec) and `docs/STATUS.md` at the start of every session. This file holds only the short rules; do not copy the PRD into it.

## Rules

- All fire is virtual. Never control real heat, flame or ignition.
- No hand-written CUDA C++. Use Warp or Taichi (chosen in P0), PyTorch, Triton, TensorRT or shaders.
- Follow the geometry contract (PRD section 4): world right-handed, metres, +Z up; camera OpenCV; `T_world_from_camera`; Z-depth; invalid depth = 0.0 with a valid mask.
- Do not silently change the thesis, coordinate conventions, public interfaces, metric definitions or invariants. Use an ADR (`docs/decisions/`) and update the PRD change log.
- Work only on the current phase and task file. Build the smallest testable vertical slice first. Keep a runnable baseline before optimising.
- Research is time-boxed: at most 3 candidates per decision, about half a day each, scored by PRD 11.6, recorded in an ADR.
- Never fabricate benchmarks, citations or test results. Mark mocks. A number in a report must come from a committed CSV.
- If a test fails, find the root cause before weakening it. Gate outcomes: PASSED, PASSED_WITH_APPROVED_DEVIATION, FAILED. Ask the user before reducing scope.
- Finish each task with: summary, files changed, exact commands, test results, benchmarks, limitations, remaining dependencies, reproduction steps.
- Keep `docs/STATUS.md` current: phase, last passing commit, completed tasks, blockers, active ADRs, next task.

## Common commands

`make setup`, `make test`, `make calibrate`, `make run`, `make bench`, `make eval`, `make plots`

## Ask the user only for

Physical actions (cameras, calibration board, second GPU machine), contradictory requirements, thesis-breaking decisions, scope reductions, license or cost questions.
