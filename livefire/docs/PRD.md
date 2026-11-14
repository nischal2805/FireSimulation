# LiveFire Previs: Technical PRD v2 (Build Spec for Claude Code)

Single source of truth for building the project. Read all of it, then read `CLAUDE.md` (rules) and `docs/STATUS.md` (where we are). v2 supersedes v1: it is graphics-first, contract-driven and vertical-slice-first.

---

## 0. Execution contract (read first)

**Mission:** build a live, two-camera, material-aware virtual-fire previs system, evaluate it rigorously, produce a demo and a benchmark report.

### 0.1 Authority and change control

- This PRD defines requirements. ADRs (`docs/decisions/`) record approved technical decisions. Task files (`docs/tasks/`) define the scope of each work session.
- You must not silently change: the thesis (section 1), the coordinate conventions (section 4), public interfaces (section 9), metric definitions (section 11), numerical invariants (section 8.6). Changing any of these needs an ADR that states why, and an entry in the change log at the end of this file.
- Everything else is yours to improve. If research or measurement shows a better design, adopt it via an ADR, as long as the thesis still holds.

### 0.2 How to research (time-boxed, not open-ended)

- Search the web for current options (papers, arXiv, GitHub, Hugging Face, release notes) only inside the phases that need a decision (depth in P1, segmentation and materials in P3, solver and render techniques in P2 and P4). The candidate lists in this PRD are seeds from memory and may be outdated; find newer ones.
- Per decision: shortlist **at most 3** candidates (fewer when one is clearly dominant), time-box each spike to about half a day, run all on the same recorded inputs and hardware, score with the rule in 11.6, write an ADR, keep the runner-up as a config-selectable fallback.
- A working integrated pipeline always beats a perfect model comparison. Never block the vertical slice on research.

### 0.3 Before, during, after every task

**Before:** read this PRD, `docs/STATUS.md`, the task file, relevant ADRs and the existing code. Inspect the repo for existing implementations before writing new ones. Write a short plan: files to change, tests, acceptance criteria. Flag any ambiguity that could produce incompatible implementations. Do not implement future-phase features.

**During:** build the smallest complete, testable vertical slice. Keep CPU/GPU transfers, tensor ownership and synchronization explicit. Never substitute mock results for real ones without marking the mock in code and output. Do not fabricate benchmarks, citations or test outcomes. Use deterministic synthetic fixtures for geometry and numerics. Keep a runnable baseline before any optimisation.

**After:** report implementation summary, files changed, exact commands run, test results (including failures), benchmark results, known limitations, remaining integration dependencies, reproduction steps. A task is complete only when its acceptance tests pass.

### 0.4 Failure handling and gates

- If a test fails, find the root cause before weakening the test or changing a threshold.
- If an optimisation does not improve the measured workload, keep the validated baseline and document the failed experiment.
- **Gate outcomes are one of:** `PASSED` (all criteria met), `PASSED_WITH_APPROVED_DEVIATION` (user approved a reduced-scope baseline; the failed criterion stays in the report), `FAILED`. An ADR explaining a failure does not turn it into a pass. Ask the user before choosing a reduced scope.
- Blockers (hardware access, licenses, missing datasets, dependency conflicts): report the blocker with the smallest actionable resolution and continue with unblocked work.

### 0.5 Escalate to the user only for

Physical actions (cameras, calibration board, running on the second GPU machine), contradictory requirements, decisions that would break the thesis, reduced-scope approvals, license or cost questions.

### 0.6 Hard rules

- All fire is virtual. Never add code that controls real heat, flame or ignition.
- No hand-written CUDA C++. Python GPU frameworks (NVIDIA Warp, Taichi), PyTorch, Triton, TensorRT, ONNX Runtime or GLSL/Slang shaders are fine. Ask first if raw CUDA seems unavoidable.
- Never state a performance or accuracy number unless a committed script produced it.
- All tunables live in YAML under `configs/`.
- Seed all randomness. Benchmarks and evals use recorded clips and manifests so they repeat.
- Keep `docs/STATUS.md` current (phase, last passing commit, completed tasks, blockers, active ADRs, next task). Read it at the start of every session.

---

## 1. Thesis and product

**Product:** a system that takes live video from two cameras looking at the same static real scene, acquires its 3D geometry and the materials of its objects from those live frames, simulates fire spreading through the objects on the GPU, and composites the fire over both live camera views in real time. Use case: film previsualization.

**Thesis (what the project must demonstrate):**

> Material-aware fire on a real scene, driven by **live camera streams with no offline scan**, with **one 3D fire state shared across two views**, rendered at interactive rates on a consumer laptop GPU, with measured latency and a documented parallel-computing speed-up.

**Meaning of "live" (precise):**

- Scene geometry and materials are acquired from live camera frames at start-up in an **Acquisition mode** (target 15 s or less, no separate scan or reconstruction tool).
- In **Simulation mode** the composited output uses the current live camera frames every display frame; the fire simulation runs on the frozen acquired snapshot.
- If the scene changes (detected by a change monitor, or by hotkey), the system pauses and re-acquires. It never silently rewrites the world under a burning object.
- A clean background plate (section 7.4 level 2) is an optional enhancement and must not be a hidden dependency of the core demo.

**Prior art (read in full in P0 and extend):**

- FieryGS (arXiv 2605.00177): 3D Gaussian Splatting scene, multimodal-LLM material inference, combustion and charring, rendered in the splat scene; appears offline.
- WildFireGS (arXiv 2608.11100): wildfire simulation on semantic Gaussian-splat forest scenes.
- Wong et al. 2025, Fire 8(4):132 (doi 10.3390/fire8040132): AR fire-dynamics platform for evacuation studies.
- Background: Stam "Stable Fluids" (1999); Fedkiw, Stam, Jensen "Visual Simulation of Smoke" (2001); Nguyen, Fedkiw, Jensen "Physically Based Modeling and Animation of Fire" (2002).

P0 task: `docs/literature.md` with a wide search (live, real-time, streaming, multi-view, camera-based, AR, depth-based, fire, smoke, combustion, material-aware) and a verdict on novelty. If a live system exists, narrow the thesis to what remains open and record it.

**Fidelity scope:** previs-grade plausibility, not predictive combustion. Geometry never breaks or collapses. Burnt parts are shown through charring, ash and erosion overlays (section 7.4).

---

## 2. Constraints and environment

| Item | Value |
| --- | --- |
| Primary GPU | RTX 4060 laptop, assume 8 GB VRAM (verify) |
| Secondary GPU | RTX 5050 laptop, separate machine, used for comparison benchmarks |
| Cameras | Two views of one static scene; two USB webcams, or webcam plus phone (Galaxy S25 Ultra) via USB/IP camera. Test phone latency early |
| Scene | Static indoor objects, ROI about 3 m across. People or objects moving are out of scope in v1 |
| Language | Python 3.11 exactly (pin it; lock dependencies). Native or shader hot paths only if profiling proves the need |
| OS | Unknown (Windows or Linux). Keep code portable; ask only if blocked |
| Deadline | Unknown. Phases are ordered by importance; see cut order in section 12 |

---

## 3. Minimum viable demo vs stretch

**MVP (must exist for the thesis):** calibrated two-camera real-scene geometry; material-dependent simulated fire; one shared 3D fire state; two consistent composited views; interactive frame rate; GPU performance study with scaling data; manual material override.

**Stretch (only after MVP passes):** automatic material recognition polish, relighting with shadows, background-plate burn-away, embers, wind, detailed smoke, sparse-grid study, second-GPU comparison, learned components.

**Deferred unless proven necessary:** learned geometry replacing calibration, learned image harmonization, neural gas solvers, CUDA IPC multiprocess comparisons, full SAM-family video tracking for the static scene (try keyframe segmentation first).

---

## 4. Geometry contract (mandatory for every module)

| Item | Convention |
| --- | --- |
| World frame | Right-handed, metres, +Z up, origin at the calibration board centre on the table plane |
| Camera frame | OpenCV: +X right, +Y down, +Z forward |
| Pose | `T_world_from_camera` (4x4, maps camera coordinates to world). Name every transform `T_<to>_from_<from>` |
| Intrinsics | `K` (3x3, pixels) valid for the exact image resolution it is stored with; distortion `dist` (OpenCV model). Images are undistorted once; downstream code uses the undistorted `K` |
| Depth | Z-depth along the camera +Z axis in metres (not ray distance). Invalid pixels hold `0.0` with `valid=False` (never NaN) |
| ROI | Axis-aligned box in world frame, `roi_min`, `roi_max`, set at calibration (default 3 m cube) |
| Voxel grid | Logical index `(ix, iy, iz)`; voxel centre = `roi_min + (i + 0.5) * h`; `h = roi_size / N`. Physical memory layout is framework-internal; all modules use accessor helpers in `common/geometry.py` |
| Versions | Every calibration has `calib_version`; every snapshot records the `calib_version` it was built from. Mismatch is an error |

Back-projection from pixel `(u, v)` with Z-depth `d`:

```
p_c = d * inv(K) @ [u, v, 1]^T
p_w = T_world_from_camera @ [p_c, 1]^T
```

**Required tests (P0, no AI, no camera):** project known 3D points into both synthetic cameras and back-project them (round trip error under 1e-4 m); use asymmetric poses and off-axis points to catch reversed transforms; check Z-depth vs ray-distance conversion; check voxel index/centre round trips at grid corners; check that flipping a transform direction fails a test.

---

## 5. System architecture

### 5.1 Three subsystems and a lifecycle

```text
 A. ACQUISITION (live frames, startup / re-acquire)
    capture -> calibrate/verify -> depth -> segment -> fuse -> materials -> validate
        produces: immutable, versioned SceneSnapshot

 B. SIMULATION (fixed timestep, independent of camera frames)
    SceneSnapshot + ignition events + material table -> surface fuel + gas solver -> FireState

 C. RENDERING (display rate)
    live camera frames + SceneSnapshot + FireState + camera poses -> two composited views
```

Lifecycle: `UNINITIALIZED -> CALIBRATING -> ACQUIRING -> READY -> SIMULATING <-> PAUSED`, plus `DEGRADED` (running with a reduced-quality mode, shown in HUD) and `ERROR`.

- In v1 geometry is frozen once `READY`. Material overrides are allowed without re-acquisition.
- A change monitor compares live frames against the snapshot's expected appearance in static regions; a large change moves to `PAUSED` and asks to re-acquire.
- Perception results never overwrite geometry or material state of a running simulation.

### 5.2 Two operating modes

- **Acquisition mode:** perception models resident; priority is depth, segmentation, fusion, materials. Lower render FPS is acceptable.
- **Simulation mode:** geometry frozen; perception models released from VRAM where possible; priority is simulation and rendering. This is what makes 8 GB workable.

### 5.3 Scheduling

- Capture thread: grabs frames with minimal buffering, timestamps on arrival, keeps only the newest.
- Simulation: fixed `dt_sim` (default 1/60 s, configurable, conservative). Max `max_substeps_per_frame` per displayed frame. If behind real time, drop to the degradation ladder (5.6), never use variable `dt`. Report **simulation real-time factor** (simulated seconds per wall second).
- Rendering: driven by display and camera availability; may render more often than the sim advances (interpolating nothing: it just re-composites the latest `FireState` snapshot over the newest camera frame).
- Perception (acquisition mode only): async worker publishing results through an explicit ownership protocol (producer writes a fresh buffer, publishes by atomic swap, consumer holds a reference until done; GPU results carry a CUDA event the consumer waits on).

### 5.4 Memory and VRAM budget

Before choosing models, measure and record a VRAM table: each backend's peak allocated and peak reserved VRAM under a defined workload, plus the integrated pipeline's total device usage via `nvidia-smi` and framework counters. Budget the sim (surface elements, gas grid with double buffering and solver scratch), render targets, camera textures, and models separately.

### 5.5 Latency measurement (definitions)

Use one monotonic host clock (`time.perf_counter_ns`) for CPU timestamps and GPU events for GPU durations, with explicit synchronization at measurement boundaries. CUDA events alone cannot measure end-to-end latency.

| Quantity | Definition |
| --- | --- |
| `t_arrive` | Host timestamp when a camera frame reaches the CPU |
| Capture-to-display latency | `t_present - t_arrive`, plus a one-time glass-to-glass offset measured with an on-screen timer or LED test for each camera, reported separately |
| Frame age at use | Age of the camera frame when composited |
| Simulation lag | Wall time minus simulated time since start |
| Delivered FPS | Presented frames per second, with p95 and p99 frame time and dropped-frame rate |
| GPU stage times | CUDA events per stage (sim, render, composite) |

### 5.6 Degradation ladder (never silent)

In order: lower volume-render resolution; fewer ray-march samples; disable relight and shadows; reduce nonessential refreshes; reduce gas-grid resolution through a validated state transition; simplified smoke. The active mode is shown in the HUD and written to every benchmark row.

---

## 6. Scene representation

### 6.1 Static scene (immutable `SceneSnapshot`)

Built once in acquisition from fused geometry. Contents:

```python
@dataclass(frozen=True)
class SceneSnapshot:
    version: int; calib_version: int; roi: "AABB"; h_gas: float
    cameras: dict[int, "CameraCalibration"]
    # surface fuel elements (N_e elements, SoA arrays on GPU)
    pos: "f32[N_e,3]"; normal: "f32[N_e,3]"; area: "f32[N_e]"
    obj_id: "i32[N_e]"       # -1 = unknown/background sentinel
    material_id: "u8[N_e]"; material_conf: "f32[N_e]"
    neighbors: "i32[N_e,K]"  # K nearest elements within radius r_nb, -1 padded
    # coarse obstacle mask for the gas solver
    solid_mask: "u8[G,G,G]"; solid_fraction: "f32[G,G,G]"
    # per-view synthetic depth buffer of the reconstructed surfaces (occlusion)
    view_depth: dict[int, "f32[H,W]"]; view_hole_mask: dict[int, "bool[H,W]"]
    objects: "ObjectRegistry"  # world objects with per-camera observations
```

Rationale: a surface seen by two cameras is not a solid volume, and a 2.3 cm voxel is thicker than a sheet of paper. Fuel therefore lives on **surface elements** with thin-shell properties (areal quantities), while heat, gas and smoke live on a **coarse 3D grid**. Surface elements come from the fused TSDF (marching cubes or surface-voxel sampling at about 1 to 2 cm spacing; the agent chooses by measurement) and are re-sampled at READY only.

An **ObjectRegistry** holds world objects. Per-camera segmentation IDs are observations, not global IDs. Objects are merged across views using the fused 3D proxy (project one view's instance into the other and compare) with appearance as a tiebreaker.

### 6.2 Dynamic state (`FireState`, double-buffered where needed)

```python
@dataclass
class FireState:
    # per surface element
    T_s: "f32[N_e]"          # surface temperature, K
    ignition_exposure: "f32[N_e]"  # accumulated seconds above threshold
    m_fuel: "f32[N_e]"       # areal fuel mass, kg/m^2
    char: "f32[N_e]"         # 0..1
    removed: "f32[N_e]"      # 0..1 visual burn-away
    burning: "u8[N_e]"
    # gas grid [G,G,G]
    T_g: "f32[...]"; vel: "f32[...,3]"; soot: "f32[...]"; p: "f32[...]"
    flame: "f32[...]"        # combustion intensity, W/m^3 (heat-release-rate density proxy, ADR-0004)
    t_sim: float; step: int
```

Solid and gas temperatures are separate fields. For each field the agent documents units, memory layout, ownership, and initialization in `docs/ARCHITECTURE.md`.

### 6.3 Resolution reality check (grid 3 m ROI)

| N | Voxel width |
| --- | --- |
| 64 | 4.69 cm |
| 96 | 3.13 cm |
| 128 | 2.34 cm |
| 192 | 1.56 cm |

Default gas grid is `G=96` and the benchmark sweeps 64 to 192. As a **baseline for an ablation** (not the default), also implement an all-voxel fuel variant to measure what the surface-element hybrid buys in correctness and cost.

---

## 7. Perception, reconstruction and rendering requirements

### 7.1 Capture and calibration

- Two cameras, low-latency capture, newest-frame-wins, timestamps and skew logged.
- ChArUco calibration for intrinsics and extrinsics; held-out validation (images not used for fitting). Report RMS and median reprojection error separately; target RMS under 1 px on held-out images.
- Camera study (P1): baseline and overlap. Define minimum shared visible area and depth-validity coverage required before fusion is accepted. Fail loudly with guidance if the placement is poor.
- Research option (time-boxed, stretch): feed-forward multi-view geometry (DUSt3R, MASt3R, VGGT or newer) as calibration cross-check. The board path stays the reference.

### 7.2 Depth and fusion (acquisition)

- Output metric Z-depth per view plus confidence and valid mask. Seeds to evaluate (max 3): Depth Anything V2 metric, Video Depth Anything, Depth Pro, UniDepth V2, Metric3D V2, MoGe, two-view stereo using the calibrated pair, or a hybrid (monocular depth scale-anchored by two-view triangulation on matches).
- Because the scene is static, accumulate depth over the acquisition window (robust median or confidence-weighted average) to remove flicker and cost.
- Fuse into a TSDF or log-odds occupancy; then extract surface elements. Do not decay valid frozen geometry after READY.
- Report: metric MAE in cm, AbsRel, surface completeness, edge accuracy, valid coverage, per-view scale consistency.

### 7.3 Segmentation, objects and materials (acquisition)

- Segmentation seeds (max 3): open-vocabulary detector plus a SAM-family prompt on keyframes (SAM 2 or newer), a closed-set instance segmenter, or a panoptic model. Start with keyframes; add tracking only if keyframes prove insufficient.
- Material recognition is a **prior for plausible simulation**, not physical identification. Approaches to compare (max 3): category-to-material lookup (category from the detector, mapped through a stored table), appearance zero-shot (CLIP/SigLIP family), small VLM with a constrained prompt, or DINO-feature probe trained on collected data; fusion of category prior and appearance is allowed.
- Interface returns a distribution, a source and a review flag:

```python
@dataclass
class MaterialPrediction:
    class_probabilities: "f32[n_materials]"; category_hint: str | None
    confidence: float; source: Literal["model", "category_prior", "manual_override"]
    requires_review: bool
```

- Uncertain predictions are visibly marked in the UI and can be corrected before ignition. Overrides are stored on the world object and survive re-acquisition when the object is matched.
- Classes (v1): paper, cloth, wood, plastic, metal, glass, ceramic, other_nonflammable.
- Dataset protocol: split by **physical object identity** (no photos of the same object in train and test), and evaluate on a held-out scene and lighting setup. Report macro-F1, confusion matrix and confidence calibration, not just top-1.

### 7.4 Rendering and compositing

Per view, per displayed frame, in linear light:

1. **Occlusion depth** comes from the snapshot's `view_depth` (rendered from the frozen reconstruction), not from noisy per-frame depth. Pixels in `view_hole_mask` (no reconstruction) use a conservative policy: draw flames only where the ray is known to be unoccluded; show the hole fraction in the HUD. Soften depth edges with a small feather. Transparent and thin objects are listed as known limitations.
2. **Flame volume** ray-march through gas `T_g` and `soot`: emission from a blackbody ramp, absorption and scattering from soot, adaptive steps, early exit on opacity. Evaluate low-resolution volume rendering with bilateral upsampling.
3. **Flame appearance model** (separate from the solver; it never alters simulation state): combines gas temperature, local combustion intensity, distance to burning surface and procedural turbulent detail. All procedural noise is evaluated in **world space** (3D noise at world position plus time), never per camera in screen space, so both views agree.
4. **Fire-lit surroundings:** per-pixel illumination from flame emission using the snapshot's normals; flicker from the appearance model. Start with a gather over a coarse emission grid; shadows are stretch.
5. **Charring and burn-away** (three levels):

| Level | Behaviour | Status |
| --- | --- | --- |
| 1 | Charring, darkening, edge erosion, ash tint; consumed paper and cloth become a dark ash-coloured residue, not invisible | MVP |
| 2 | Reveal a previously captured clean background plate where an object was consumed | Stretch, optional capture |
| 3 | Multi-view-consistent background completion | Research extension |

Surface elements splat `char` and `removed` into per-view buffers by projection with a depth tolerance against `view_depth`, then filter temporally and across mask edges to avoid shimmer. Ordinary inpainting is not used for removal.

6. **Smoke:** soot volume darkens and tints; rises above flames. Minimal smoke stays in MVP; detailed smoke is stretch.
7. **Camera match:** convert to linear light, estimate exposure (lock at start or per frame; evaluate both), add layers, bloom, tone curve, convert back. Match camera noise level.

---

## 8. Simulation numerical methods

Reference design. Improve via ADR; keep the behaviours and the invariants in 8.6.

### 8.1 Units and parameter policy

Every parameter has a **unit, valid range, default, and `status`**: `physical_estimate` (order-of-magnitude value from memory or literature, verify before citing) or `artistic` (tuned for look, no physical claim). Do not present artistic values as measured properties.

### 8.2 Surface-element thermal model (per element i)

```
C''_m dT_i/dt = q_conv + q_cond + q_rad + q_comb        [W/m^2]
q_conv = h_c (T_g(x_i) - T_i)                           T_g sampled trilinearly
q_cond = (k_m * delta_m / s^2) * sum_j (T_j - T_i)      j in neighbors, s = mean element spacing
q_rad  = sum over burning b within R: eps_f * sigma * T_b^4 * A_b * cos(th_i) cos(th_b) / (pi r^2)   (r >= r_min)
q_comb = chi * dH_m * mdot''_i                          only while burning
C''_m  = rho_m * c_m * delta_m                          areal heat capacity [J/(m^2 K)]
```

Radiation is a cost risk. Implement and compare, in this order: (1) local only (conv + cond), (2) coarse radiative-emission grid with interpolation, (3) bounded gather over burning elements, (4) optional visibility-aware radiation. Pick by visual effect and measured GPU cost; the simplest sufficient option wins.

Explicit conduction stability: require `dt <= C'' s^2 / (4 k delta)` (or implicit/clamped update); the solver checks and sub-steps.

### 8.3 Ignition, combustion, char

```
if T_i >= T_ign_m:  E_i += dt              (ignition_exposure)
else:               E_i = max(0, E_i - decay * dt)
ignite when E_i >= t_delay_m and m_fuel_i > 0
user ignition sets T_i to T_pilot and E_i to t_delay_m for elements in a radius

burning:  mdot''_i = mdot_ref_m                         [kg/(m^2 s)]  (v1 constant; flux feedback is stretch)
          m_fuel_i = max(m_res_m, m_fuel_i - mdot''_i * dt),   m_res_m = char_yield_m * m0''_m
char_i    = (m0'' - m_fuel_i) / (m0'' * (1 - char_yield_m))     clamped 0..1
burnt out when m_fuel_i <= m_res_m + eps  -> burning = 0
removed_i = char_i if consumed_when_burnt else 0         (visual only)
heat to gas:  dT_g(cell) = sum_i A_i q_comb_i * dt / (rho_g c_p V_cell)
soot to gas:  d(soot)(cell) = sum_i y_s_m * mdot''_i * A_i * dt / V_cell      [kg/m^3]
```

Non-flammable materials never consume fuel and never ignite.

### 8.4 Gas solver (grid G^3, solids as obstacles)

- Semi-Lagrangian (or MacCormack) advection of `vel`, `T_g`, `soot`; clamp `|u| dt / h` and log violations.
- Buoyancy (Boussinesq): `a_z = g (T_g - T_amb) / T_amb`.
- Vorticity confinement with a tunable strength; cooling `dT_g/dt = -k_cool (T_g - T_amb)`; soot decay.
- Wind: inflow boundary velocity plus low-amplitude noise (seeded). Zero wind must give no directional bias.
- Pressure projection with solids as boundaries. Start with a simple validated solver (Jacobi or red-black Gauss-Seidel) and add multigrid-preconditioned CG only if profiling shows it wins.
- Solver quality is measured, not assumed: report divergence norm `||div u||_2` before and after projection, residual convergence, boundary penetration, and stability. A faster solver that leaves the field more divergent is rejected.

### 8.5 Material table (`configs/materials.yaml`), units explicit

Values below are placeholders to be tuned. Status is mandatory per field group.

```yaml
# units: m0_areal [kg/m^2], delta [m], rho [kg/m^3], c [J/(kg K)], k [W/(m K)],
#        T_ign [K], t_delay [s], mdot_ref [kg/(m^2 s)], dH [J/kg], chi [-],
#        char_yield [-], y_soot [kg/kg], eps [-]
paper:   {status: physical_estimate, flammable: true,  m0_areal: 0.08, delta: 1.0e-4, rho: 800, c: 1340, k: 0.1,  T_ign: 505, t_delay: 0.2, mdot_ref: 0.012, dH: 1.4e7, chi: 0.6, char_yield: 0.0,  y_soot: 0.01, eps: 0.9, consumed_when_burnt: true}
cloth:   {status: physical_estimate, flammable: true,  m0_areal: 0.20, delta: 4.0e-4, rho: 500, c: 1300, k: 0.06, T_ign: 520, t_delay: 0.4, mdot_ref: 0.008, dH: 1.6e7, chi: 0.6, char_yield: 0.05, y_soot: 0.02, eps: 0.9, consumed_when_burnt: true}
wood:    {status: physical_estimate, flammable: true,  m0_areal: 3.0,  delta: 5.0e-3, rho: 600, c: 1700, k: 0.15, T_ign: 570, t_delay: 1.5, mdot_ref: 0.010, dH: 1.3e7, chi: 0.6, char_yield: 0.15, y_soot: 0.015, eps: 0.9, consumed_when_burnt: false}
plastic: {status: artistic,          flammable: true,  m0_areal: 0.5,  delta: 1.0e-3, rho: 1000, c: 1900, k: 0.2, T_ign: 600, t_delay: 1.0, mdot_ref: 0.010, dH: 2.2e7, chi: 0.6, char_yield: 0.05, y_soot: 0.08, eps: 0.9, consumed_when_burnt: false}
metal:   {status: physical_estimate, flammable: false, rho: 7800, c: 500, k: 50, delta: 2.0e-3, eps: 0.3}
glass:   {status: physical_estimate, flammable: false, rho: 2500, c: 840, k: 1.0, delta: 3.0e-3, eps: 0.9}
ceramic: {status: physical_estimate, flammable: false, rho: 2400, c: 900, k: 1.5, delta: 5.0e-3, eps: 0.9}
other_nonflammable: {status: artistic, flammable: false, rho: 1500, c: 1000, k: 0.5, delta: 3.0e-3, eps: 0.9}
```

### 8.6 Invariants (automated tests, non-negotiable)

| Invariant | Expected |
| --- | --- |
| Fuel bounds | `m_res <= m_fuel <= m0` always |
| Char and removed | Within 0..1 |
| No spontaneous fire | No burning without a source or sufficient transferred heat |
| Non-flammables | Never consume fuel, never ignite |
| Zero wind | No directional bias in spread or flame lean |
| Heat conduction | Smooths temperature; stable; conserves energy within tolerance in an isolated test |
| Projection | Reduces divergence by a stated factor |
| Replay | Same seed and inputs give identical state within a stated tolerance |
| Long run (10 min simulated) | No NaN, Inf or unbounded growth |
| Resolution sensitivity | Ignition time of a reference sheet changes by less than 15% between G=64 and G=128 (tune until true) |

---

## 9. Interface contracts

Define typed `Protocol`s with shapes, units, coordinate convention, device placement (CPU or GPU), contiguity, synchronization requirement, ownership, error behaviour and the `calib_version` they correspond to. Required interfaces:

`CameraSource`, `CameraCalibration`, `DepthBackend`, `SegmentationBackend`, `MaterialBackend`, `SceneReconstructor`, `SceneSnapshot`, `SimulationBackend`, `RendererBackend`, `ReplayController`, `BenchmarkRecorder`.

Example (depth backend must state all of this in its docstring):

```python
class DepthBackend(Protocol):
    """Input: undistorted rgb uint8[H,W,3] on GPU or CPU (declare). Output DepthResult:
    z_depth_m f32[H,W] contiguous, GPU-resident; valid bool[H,W]; conf f32[H,W] in [0,1]
    (state how calibrated); invalid pixels = 0.0. Thread-safe. fp16 allowed.
    Tied to the calib_version of its input frame. Raises DepthError on model failure."""
    def warmup(self, shape: tuple[int, int]) -> None: ...
    def infer(self, frame: Frame) -> DepthResult: ...
```

Every backend reports `latency_ms`, `peak_vram_allocated_mb`, `peak_vram_reserved_mb` through the shared timer.

---

## 10. Test strategy

1. **Unit tests** for geometry (section 4), config loading, interfaces, kernels on tiny inputs.
2. **Deterministic synthetic fixture (no AI):** a procedural scene (table plane, boxes, a thin sheet, a cloth-like patch) with analytic two-camera rendering giving exact depth, masks, poses and materials. Everything in P0 uses it. Also a Blender-generated set later for realism.
3. **Numerical tests:** the invariants in 8.6; solver divergence tests; conduction energy tests.
4. **Integration tests:** acquisition on the synthetic fixture produces a `SceneSnapshot` that matches the known geometry within tolerance; both views render the same fire state at the known 3D positions.
5. **Replay tests:** recorded inputs plus events reproduce the same output.
6. **Benchmark tests:** fixed workloads, repeated trials, results to CSV (Appendix C).
7. **Eval scripts** for depth, segmentation, materials, consistency, spread behaviour.

---

## 11. Metrics and evaluation

### 11.1 Targets (stretch; record achieved targets separately after the P1 baseline, keep the originals)

| ID | Metric | Definition and target |
| --- | --- | --- |
| M1 | Delivered frame rate | 2 views, 720p, G=96 to 128 on RTX 4060: 30 fps median target; report p95 and p99 frame time and dropped-frame rate |
| M2 | Latency | `t_present - t_arrive` per 5.5, target 100 ms or less; glass-to-glass offset reported separately |
| M2b | Simulation real-time factor | 1.0 target at the configured `dt_sim` |
| M3 | Acquisition time | From start to READY (calibrated, aligned, renderable) 15 s or less |
| M4 | Geometry accuracy | Metric MAE in cm against ruler-measured and Blender ground truth in the ROI; plus AbsRel, completeness, edge accuracy |
| M5 | Geometry stability | Depth flicker on static pixels AND accuracy reported together |
| M6 | Fusion alignment | Distance between independent landmarks (ChArUco corners, AprilTags with known positions) as reconstructed vs known, median under 3 cm |
| M7 | Cross-view consistency | Independent test: known 3D landmarks and a synthetic flame at a known position projected into both views; pixel error vs analytic projection, median under 6 px at 720p. Do not use fused depth as the reference |
| M8 | Material classification | Object-disjoint split, held-out scene; macro-F1, confusion matrix, confidence calibration; target 85% top-1 on the labelled set |
| M9 | Spread behaviour | Randomized geometry, distances, wind and held-out scenarios; check **monotonicity**: swapping the material of the same element with the same geometry orders time-to-ignite as the material parameters imply (lower `T_ign`, lower `C''` ignites sooner). Scripted paper then cloth then wood is a demo script, not a validation |
| M10 | Speed-up study | Fixed workload, repeated trials, quality constraints held (divergence and visual metrics unchanged within tolerance), full-pipeline ablations, scaling plots (grid size, views, dense vs surface hybrid), both GPUs |

### 11.2 Ablations (each answers one question)

Single vs dual view; hand-labelled vs inferred materials; uniform vs material-dependent combustion; all-voxel vs surface-hybrid; synchronous vs asynchronous scheduling; full vs reduced volume-render resolution; with and without world-space flame detail; local radiation vs gather.

### 11.3 Datasets and manifests

`data/` is gitignored but every dataset has a manifest with file hashes, license, capture conditions, and a deterministic regeneration script where possible. Contents: calibration captures; static real-scene clips (5+ scenes, 6+ objects each, tape-measured distances); synthetic Blender scenes with ground truth; the material set (about 100 objects, several photos each, object-disjoint splits); scripted scenarios; self-shot or properly licensed reference fire footage (no copyrighted footage in the repo).

### 11.4 Benchmark rules

Fixed recorded inputs, warm-up frames discarded, at least 5 repeats, median and p95, power mode and GPU clocks recorded, nothing else running. Plots read only from CSVs.

### 11.5 Reporting

Every number in the report comes from a committed CSV or eval output.

### 11.6 Decision rule for model/technique choices

Score 0 to 5: accuracy on this project's data (35), latency and VRAM on the RTX 4060 (30), robustness (20), implementation risk (10), license of code, weights and datasets for demo and distribution (5). Hard limits (exceeds budget or VRAM, license forbids use) eliminate a candidate. Ties go to the simpler option. Record in the ADR.

---

## 12. Phases, tasks and gates (graphics-first, vertical slice first)

Each phase ends with a runnable artifact, tests, a report in `docs/reports/phase-N.md`, and an updated `docs/STATUS.md`. Split each phase into task files `docs/tasks/PN-NNN-*.md`.

### P0: contracts and synthetic vertical slice (no AI models, no cameras)

Tasks: repo skeleton, pinned environment, config loader, shared timers; geometry module and tests (section 4); procedural synthetic two-camera scene; surface-element extraction from the synthetic scene into a `SceneSnapshot`; minimal fire (local conduction and convection, ignition, combustion, char); gas solver baseline; volume renderer with snapshot-based occlusion; two-view composite on synthetic frames; HUD, timers, record and replay; `CLAUDE.md` and `docs/STATUS.md`; extraction of architecture, numerical methods, coordinate conventions, test strategy and performance budget sections into `docs/` stubs that point back to this PRD (no duplicated text).
Also: GPU framework spike (Warp vs Taichi) on two kernels, ADR.
**Gate 0:** a synthetic two-camera scene renders the same shared fire state correctly in both views; geometry and invariant tests pass; baseline benchmark CSV committed; `make test` green.

### P1: real-camera geometry

Tasks: capture and sync, latency measurement, ChArUco calibration with held-out validation, camera baseline/overlap study, depth (up to 3 candidates, ADR), depth accumulation, fusion, surface-element extraction on real data, snapshot freeze, change monitor, manual material tagging, overlay of the reconstructed geometry on live frames as an alignment check.
**Gate 1:** reconstructed geometry aligns with a measured real scene from both cameras (M4, M6 measured); fire runs on real geometry with manual materials; baseline performance recorded; originals and achieved targets recorded separately.

### P2: simulation quality

Tasks: tune thermal model and units; radiation options compared; gas solver quality metrics; invariants and resolution-sensitivity test; long-run stability; uniform vs material-dependent ablation; scripted and randomized scenarios.
**Gate 2:** all invariants pass; M9 passes; documented numerical behaviour.

### P3: automatic scene understanding

Tasks: segmentation (keyframes first), object registry and cross-view association, material recognition (up to 3 approaches, ADR), `MaterialPrediction` with review UI, object-disjoint evaluation, VRAM and mode switching between acquisition and simulation.
**Gate 3:** scene acquired and labelled with minimal manual intervention within M3; M8 measured.

### P4: visual fidelity and performance

Tasks: flame appearance model, relight, charring levels 1 (and 2 if time), exposure matching, profiling (Nsight), optimisations, degradation ladder, ablations, both GPUs.
**Gate 4:** M1, M2, M2b, M7, M10 measured and tabulated; demo-quality visuals; limitations written.

### P5: final evaluation and delivery

Tasks: held-out tests, scripted demo, recorded video, final report from CSVs, optional paper draft if the novelty verdict supports it.
**Gate 5:** demo runs three times in a row; `make bench` and `make plots` reproduce report figures from a clean checkout.

### Cut order if time is short

Drop: sparse-grid study, second-GPU comparison, detailed smoke, background-plate burn-away, learned look enhancements, wind. Keep a minimal smoke and flame model. Never cut: calibration, benchmarks, geometry contract and tests, material-dependent combustion, the independent consistency metric.

---

## 13. Demo specification

Table with a paper sheet, cloth, wood block, plastic bottle, metal cup, ceramic mug **and a glass**. Two cameras 60 to 90 degrees apart. Screen shows both composited views and the HUD.

Script (about 3 minutes): start system and show acquisition reach READY; show detected materials with uncertain ones flagged; correct one by click; ignite the paper; fire spreads through cloth, wood chars slowly, metal, glass and ceramic do not burn; both views show the same fire; toggle wind; toggle relight; toggle single-view vs two-view with HUD numbers; replay a recorded run and show one scaling plot.
Fallbacks: manual material tags if perception misfires; single-view if a camera drops; always keep a recorded good run.

---

## 14. Risks and fallbacks

| Risk | Mitigation |
| --- | --- |
| Depth noise or scale drift | Static-scene accumulation, calibration scale anchor, two-view triangulation refinement |
| Two cameras do not reveal enough geometry | Baseline/overlap study, hole policy, conservative occlusion, report hole fraction |
| One GPU shared by perception and sim | Acquisition vs simulation modes, release models after READY |
| Wrong materials | Prior-not-truth interface, review flags, click override |
| Cameras unsynchronised | Static scene, timestamp matching, skew logged |
| Phone camera latency | Test early; use two webcams if needed |
| Framework choice wrong | P0 spike, thin kernel interface |
| Flames look like glowing fog | Separate appearance model, world-space noise, reference footage, time budget in P4 |
| Novelty narrower than hoped | P0 literature verdict; narrow the claim |
| Agent loses context across sessions | `docs/STATUS.md`, task files, ADRs |
| Scope creep | MVP vs stretch (section 3), gates, cut order |

---

## 15. Repository layout and tooling

```text
livefire/
  AGENTS.md                 # shared agent rules (short; points here)
  CLAUDE.md                 # same core rules for Claude Code (short)
  README.md  pyproject.toml  Makefile  # setup calibrate run test bench eval plots
  configs/                  # camera.yaml grid.yaml materials.yaml app.yaml
  src/livefire/
    common/                 # geometry.py, timers, config, logging
    capture/  perception/{depth,segment,material}/  fusion/  scene/  sim/  render/  app/
  tests/  eval/  bench/{results,plots,traces}/  data/
  docs/
    PRD.md  ARCHITECTURE.md  NUMERICAL_METHODS.md  COORDINATE_CONVENTIONS.md
    TEST_STRATEGY.md  PERFORMANCE_BUDGET.md  STATUS.md  literature.md
    decisions/  tasks/  reports/
```

Conventions: Python 3.11 pinned and locked, type hints, `ruff`, `pytest`, one logical change per commit, perception backends swappable by config.
`AGENTS.md` and `CLAUDE.md` carry the short shared rules (section 0.6 and pointers). Do not duplicate this PRD into them.

---

## Appendix A: Seed reading (verify and replace with newer work)

Stam 1999; Fedkiw et al. 2001; Nguyen et al. 2002. Depth: Depth Anything V2, Video Depth Anything, Depth Pro, UniDepth V2, Metric3D V2, MoGe. Multi-view: DUSt3R, MASt3R, VGGT. Segmentation: SAM 2 and successors, EfficientSAM, EdgeTAM, Grounding DINO, OWLv2, Florence-2. Materials: Materials in Context (MINC), CLIP/SigLIP, DINOv2/v3. Scene-fire: FieryGS, WildFireGS, Wong et al. 2025.

## Appendix B: ADR template

```markdown
# ADR-XXXX: <title>
Date, status (proposed / accepted / superseded)
## Context   ## Candidates (source, date, license, size)   ## Method (data, metrics, hardware, commands)
## Results (table, CSV links)   ## Scores (11.6)   ## Decision and fallback   ## Consequences
```

## Appendix C: Benchmark CSV schema

`run_id, date, git_sha, workload_version, gpu, driver, cuda, os, python, framework, power_mode, scene, grid_g, surface_elements, views, resolution, mode, degradation_level, stage, variant, timing_method, warmup_frames, samples, median_ms, p95_ms, p99_ms, delivered_fps, rt_factor, peak_vram_alloc_mb, peak_vram_reserved_mb, notes`

## Change log

- v2: graphics-first phases; geometry contract; frozen-snapshot architecture; surface-element hybrid; numerical methods with units; burn-away levels; latency definitions; revised metrics; agent execution contract.
- v2.1 (P0-008): ADR-0002 (accepted): gas velocity is stored staggered (MAC) internally; the public `FireState.vel` stays cell-centred `[G,G,G,3]`, exposed through an accessor; grid size must be even. ADR-0003 (accepted): combustion energy split `f_feedback` surface / `1 - f_feedback` gas, surface re-radiation, counted clamps on `T_s`, `T_g` and velocity (PRD 8.2, 8.4). `configs/sim.yaml` gains the gas section.
- v2.2 (P0-008b): ADR-0004 (accepted): new passive gas-grid field `flame` (combustion intensity, W/m^3) fed by the released power of burning elements, advected and decayed with `k_flame`; the renderer derives a visual temperature `T_vis = max(T_g, T_amb + dT_flame (1 - exp(-flame / flame_ref)))` for emission (PRD 6.2, 7.4). `FireStateView`/`FireStateHost` gain `flame`; `configs/sim.yaml` gains `k_flame`, `configs/render.yaml` gains `dT_flame`, `flame_ref`.
