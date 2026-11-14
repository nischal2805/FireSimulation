# ADR-0002: Gas velocity on a staggered (MAC) grid; cell-centred accessor

Date: 2026-11-14. Status: **accepted** (plan ruling R4; implemented in P0-008).

## Context

PRD 6.2 specifies `FireState.vel` as a cell-centred `[G,G,G,3]` field and PRD 8.4 asks for a pressure projection
with solids as boundaries. A collocated (cell-centred) velocity with a naive projection has the well-known
checkerboard pressure null space and needs a Rhie-Chow style correction; solid-obstacle boundaries are also
awkward (flux through a face is not a stored quantity). The renderer and the replay digest only need a
cell-centred velocity. Interface Protocols (`FireStateView`, `FireStateHost`) must not change (CLAUDE.md).

## Candidates (source, date, license, size)

1. Collocated grid, cell-centred velocity, projection with face-interpolated velocities. Source: textbook
   (needs a Rhie and Chow 1983 style correction). No code dependency.
2. Staggered MAC grid: normal velocity on cell faces, scalars cell-centred (Harlow and Welch 1965; Bridson, "Fluid Simulation for Computer Graphics", 2015). No code dependency.
3. Finite-volume with a ready PDE framework: rejected (new dependency, not Warp, PRD 0.2 time-box).

## Method (data, metrics, hardware, commands)

Candidate 2 was implemented (`src/livefire/sim/gas.py`) and measured: `uv run pytest tests/test_gas.py -q -s`
(CPU, Warp 1.19.0, no GPU; numbers labelled `gpu=none, framework=warp-cpu`). Measured values are inlined below.

## Results (table, CSV links)

No benchmark CSV (correctness-only decision). Measured in tests (G=32, random face velocity, dt=1/60):
projection reduces `||div u||_2` over fluid cells by 39.8x (5 iterations), 95.6x (10), 224x (20), 688x (60);
faces touching solids/walls are exactly 0 after projection. Zero-wind symmetry (G=32, 60 steps, hot Gaussian blob at
the x-y centre): centre-of-mass drift 4e-5 cells (limit 0.25), mirror asymmetry 1.4e-5 relative L2 (limit 1e-3).
Cost on CPU (single-threaded Warp): about 0.4 ms per red-black half-sweep at G=32, about 59 ms per coupled step at
60 iterations. Vorticity confinement is applied as `f = eps_vc * h * (N x omega)` [m/s^2] (N = normalised gradient
of `|omega|`): an artistic, grid-scaled strength per unit time, not a physical coefficient.

## Scores (11.6)

Not scored with the full PRD 11.6 table (single dominant candidate, time-box); qualitative: correctness of the
projection and boundary exactness favour MAC; memory cost is +3 face arrays of ~(G+1)G^2 floats (about 0.4 MB each
at G=32, about 13 MB at G=96), negligible.

## Decision and fallback

- Storage: `u (G+1,G,G)`, `v (G,G+1,G)`, `w (G,G,G+1)` float32 face normal velocities; `T_g`, `soot`, `p`
  cell-centred `(G,G,G)`; `solid` from the snapshot. Face `(i,j,k)` of `u` sits at `roi_min + h*(i, j+.5, k+.5)`,
  analogous for `v`, `w`. Faces on domain walls and faces touching a solid cell are fixed (velocity 0, except the
  `-x` wall which carries the wind inflow); the top face `z = roi_max` is open (ghost cell `p = 0`).
- Accessor: `GasSolver.velocity_cell_centered() -> (G,G,G,3)` float32 = average of the two bounding faces per axis.
  `FireStateHost.vel` and the public shape do not change; consumers never see the staggering.
- Pressure is stored kinematically (`p / rho_g`); `state_host().p` multiplies by `rho_g` (Pa).
- Fallback: if red-black Gauss-Seidel proves too slow at G=96 on the GPU, add multigrid-preconditioned CG (PRD 8.4)
  on the same MAC discretisation; the storage does not change.

## Consequences

- Projection is exact on face fluxes (divergence of the stored faces), so "no flux through solids" is exactly 0.
- Semi-Lagrangian advection of face velocities needs three kernels (one per component) instead of one.
- The cell-centred velocity is a derived, lossy view (it cannot reproduce the face field); the digest uses it.
- Red-black GS colouring is not mirror-symmetric (parity flips under reflection with even G); measured
  mirror-asymmetry after 60 steps is ~1e-5 relative L2 (test threshold 1e-3). The grid size must be even.
- Soot is a mass concentration in kg/m^3 (`soot += soot_kg / h^3`, PRD 8.3; controller ruling). The
  `FireStateView` docstring in `interfaces.py` was corrected from "dimensionless mass fraction".
