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
- **Flame reference.** `flame_ref` (`configs/render.yaml` key `flame_ref`; retuned in commit 9e24909, see ADR-0004) sets the renderer's
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

## Parameter audit

PRD 8.1 / P2R8. Every `configs/sim.yaml` entry has a unit and a status; every material has a per-group
`status_detail` in `configs/materials.yaml` (checked by `tests/test_thermal_audit.py::test_sim_yaml_audit` and
`tests/test_materials.py`). `physical_estimate` here means an order of magnitude from memory, to be verified before
it is cited; nothing in the repository is a measurement. `artistic` values carry no physical claim.

| Parameter group | Where | Unit | Status | Source note |
| --- | --- | --- | --- | --- |
| Ambient `T_amb`, gas `rho_g`, `c_p`, `g` | sim.yaml | K, kg/m^3, J/(kg K), m/s^2 | physical_estimate | standard room air values, from memory |
| Convection `h_c` | sim.yaml | W/(m^2 K) | physical_estimate | natural-convection order of magnitude (5-25), from memory, verify |
| Feedback split `f_feedback`, `T_pilot`, `ignite_radius`, `ignition_decay`, `eps_m`, `T_s_min`, `T_s_max` | sim.yaml | -, K, m, 1/s, kg/m^2, K | artistic | tuned for look (ADR-0003) |
| Gas solver `eps_vc`, `k_cool`, `k_soot`, `T_g_max`, `cfl_max`, `n_iters`, `wind_speed`, `wind_noise` | sim.yaml | -, 1/s, K, m/s | artistic | tuned; clamps counted (ADR-0003) |
| Flame field `k_flame` | sim.yaml | 1/s | artistic | ADR-0004 |
| Radiation `rad_mode`, `eps_f`, `T_rad_flame`, `rad_R`, `rad_r_min`, `rad_cell`, `rad_max_emitters` | sim.yaml | -, -, K, m, m, m, - | artistic | ADR-0006; flame emissivity and temperature are look parameters, the geometry (cos, 1/r^2) is textbook view-factor form |
| Material thermal group (`rho`, `c`, `k`, `delta`) | materials.yaml | kg/m^3, J/(kg K), W/(m K), m | physical_estimate for paper, cloth, wood, metal, glass, ceramic; artistic for plastic, other_nonflammable | handbook order of magnitude from memory, verify; plastic is a generic stand-in |
| Material ignition group (`T_ign`, `t_delay`) | materials.yaml | K, s | artistic (all flammables) | `T_ign` is an order of magnitude from memory (verify), but `t_delay` is tuned, so the group is artistic |
| Material combustion group (`m0_areal`, `mdot_ref`, `dH`, `chi`, `char_yield`, `y_soot`, `consumed_when_burnt`) | materials.yaml | kg/m^2, kg/(m^2 s), J/kg, -, -, kg/kg | artistic (all flammables) | `m0_areal` and `dH` are order-of-magnitude values, but `mdot_ref`, `chi`, `char_yield`, `y_soot` are tuned, so the group is artistic |
| Material radiation group (`eps`) | materials.yaml | - | physical_estimate (artistic for other_nonflammable) | emissivity of non-metals about 0.9 and of metal about 0.3, from memory, verify |

## Invariants and tests

The PRD 8.6 invariants are tested in `tests/test_surface_invariants.py`, `tests/test_gas.py` and
`tests/test_coupled.py`. See `docs/TEST_STRATEGY.md` for the mapping. The resolution-sensitivity invariant is
reported and does not gate in P0 (R9).
