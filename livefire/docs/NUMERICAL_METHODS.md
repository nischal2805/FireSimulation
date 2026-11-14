# Numerical methods

Spec: `docs/PRD.md` section 8 (8.1 parameter policy, 8.2 surface model, 8.3 ignition and combustion, 8.4 gas
solver, 8.5 material table, 8.6 invariants). This file records the decisions and controller rulings that the code
relies on. Parameter values live in YAML, not here.

## Decisions (ADRs)

- **ADR-0001** GPU framework: NVIDIA Warp primary, Taichi fallback. Status *proposed* until the GPU run on the
  user's machine. Everything here runs on the Warp CPU device in this container.
- **ADR-0002** Gas velocity on a staggered MAC grid, with a cell-centred accessor for export.
- **ADR-0003** Combustion energy split (`f_feedback` to the surface, `1 - f_feedback` to the gas), surface
  re-radiation, and temperature clamps with counted clamp events.
- **ADR-0004** Combustion-intensity field `flame` [W/m^3] for visible flames.

## Controller rulings

These are not in the PRD and are the code's behaviour.

- **Edge-flux conduction.** Each neighbour pair contributes one symmetric flux
  `P_ij = G_ij (T_j - T_i)` [W], with `G_ij = kd_ij * s / d_ij`. Here `kd_ij` is the harmonic mean of `k_m * delta_m`
  over the two elements, `d_ij` is their distance, and `s` is the surfel spacing. Then
  `dT_i = dt * sum_j P_ij / (A_i * C''_i)`. Neighbour lists must be symmetric, and conduction uses neighbours within
  the same `obj_id` only (ruling R6 in the plan approval). On a uniform grid with `A = s^2` and `d = s` this reduces to PRD 8.2's
  `(k delta / s^2) * sum (T_j - T_i)`. Energy is conserved exactly in the sum over edges.
- **Stability and sub-stepping.** `dt_cond_max = 0.5 / max_i(sum_j G_ij / (A_i C''_i))`. The convective limit
  `dt_conv_max` is computed the same way from `h_c`. `SurfaceFireModel.substeps_for(dt)` takes
  `ceil(dt / min(dt_cond_max, dt_conv_max))` sub-steps. The sub-step count is reported in the stats.
- **Gas sampling and injection offset.** Gas temperature is sampled at, and heat and soot are injected at,
  `x_i + n_i * h_gas`, i.e. one gas cell off the element along its normal. This keeps elements on solid cells
  coupled to fluid (`SurfaceFireModel.sample_points()`).
- **Hole policy (occlusion).** `view_depth` is the splat of surface elements. Where no element covers a pixel but
  the accumulated measured depth is valid and lies beyond the ROI exit along that ray, `view_depth` takes that
  measured depth (known-clear background). Otherwise the pixel is a hole: `view_hole_mask = True` and flames are
  suppressed there.
- **Ignition.** `ignite(point, radius)` sets `T_s = T_pilot`, `E = t_delay` and `burning = 1` for flammable,
  unburnt elements within the radius (R7). Lit elements burn until burnt out, so the thermal step cannot extinguish
  them. The default radius is `ignite_radius` in `configs/sim.yaml`.
- **Flame reference.** `flame_ref` (`configs/render.yaml`, 3e4 W/m^3, retuned in P0-008b) sets the renderer's
  flame emission, `T_vis = max(T_g, T_amb + dT_flame (1 - exp(-flame / flame_ref)))`, and the active-box threshold.
  The gas solver does not use it.
- **Gas pressure storage.** Pressure is stored kinematically as `p / rho_g` (m^2/s^2). `FireSimulation.state_host()`
  exports it in Pa.

## Parameter files

- `configs/sim.yaml`: surface model, gas solver and combustion-intensity parameters, in the PRD 8.1 format
  (`value`, `unit`, `range`, `status`). `status` is `physical_estimate` or `artistic`.
- `configs/materials.yaml`: per-material table in the PRD 8.5 format (fuel quantities are zero for non-flammable
  classes). Read through `sim/materials.py`, which derives `m_res = m0 * char_yield`.
- `configs/grid.yaml`: ROI (`roi_min`, `roi_max`), gas grid `G`, surfel spacing, neighbour radius and cap.

## Invariants and tests

The PRD 8.6 invariants are tested in `tests/test_surface_invariants.py`, `tests/test_gas.py` and
`tests/test_coupled.py`. See `docs/TEST_STRATEGY.md` for the mapping. The resolution-sensitivity invariant is
reported and does not gate in P0 (R9).
