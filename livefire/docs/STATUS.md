# STATUS

| Field | Value |
| --- | --- |
| Phase | P0 (contracts and synthetic vertical slice), in progress |
| Last passing commit | P0-004 (see git log) |
| Gate 0 | not started |

## Completed tasks
- Setup: repo folder, CLAUDE.md, AGENTS.md, docs/PRD.md, docs/STATUS.md, docs/{decisions,tasks,reports}/
- P0-001: repo skeleton, uv env, config loader, timers, logger (14 tests, ruff clean)
- P0-002: geometry module (cameras, transforms, depth, AABB, VoxelGrid) + 12 tests (26 total, ruff clean)
- P0-003: interface Protocols, Frame/DepthResult/MaterialPrediction/SceneSnapshot/FireStateHost, MaterialTable + tests
- P0-004: GPU spike (Warp vs Taichi) kernels, tests, CPU CSV; ADR-0001 proposed (draft recommends Warp, Taichi fallback)

## In progress
- none

## Blockers
- Development container has no NVIDIA GPU (`nvidia-smi` not found). GPU kernels can only be validated on CPU backends here; GPU benchmarks need a run on the user's RTX 4060 machine.

## Active ADRs
- ADR-0001 (proposed): GPU framework; pending `make bench-spike` with device cuda:0 on the RTX 4060

## Next task
- P0-005
