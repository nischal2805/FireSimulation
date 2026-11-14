# STATUS

| Field | Value |
| --- | --- |
| Phase | P0 (contracts and synthetic vertical slice), in progress |
| Last passing commit | P0-010 (see git log) |
| Gate 0 | not started |

## Completed tasks
- P0-001..P0-009 (skeleton/config/timers, geometry, contracts/materials, GPU spike, synthetic scene, snapshot builder, surface fire model, gas solver and coupled sim, volume renderer and compositor)
- P0-008b: combustion-intensity field for visible flames (ADR-0004)
- P0-013: literature review (search-only)
- P0-010: app lifecycle, scheduler, HUD, record and replay (`make run` -> `out/demo/composite.mp4`, offline mode on CPU)

## In progress
- none

## Blockers
- No GPU in the development container (`nvidia-smi` not found); GPU numbers need a run on the RTX 4060 machine. CPU runs are labelled `gpu=none, framework=warp-cpu` and are far below real time.
- GitHub push returns 403 (commits stay local).
- Literature full-text fetch is blocked (search-only summaries).

## Active ADRs
- ADR-0001 (proposed): GPU framework; pending `make bench-spike` with device cuda:0 on the RTX 4060
- ADR-0002, ADR-0003, ADR-0004

## Next task
- P0-011
