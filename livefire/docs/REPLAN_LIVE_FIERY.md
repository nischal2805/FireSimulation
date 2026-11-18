# LiveFire: replan and handoff (2026-11-18)

**Start here if you are a new Claude session (the user is switching Claude accounts).** This document holds
everything: what the project is, what happened in the GPU-machine session of 2026-11-14..18, what the user now wants,
the brainstormed approaches for each part, the new roadmap, and the exact environment. Read it fully, then
`docs/PRD.md` (still the formal spec; see section 9 for the parts this replan changes), `CLAUDE.md` (rules),
`docs/STATUS.md`.

All fire is virtual. Nothing here controls real heat or flame.

---

## 1. The project in one paragraph

LiveFire Previs (repo `nischal2805/FireSimulation`, project folder `livefire/`, Python 3.11, uv, NVIDIA Warp on the GPU)
takes live video from one or two **fixed** cameras looking at a **still** real scene, builds the scene's 3D geometry
and materials at start-up, simulates fire spreading through the objects on the GPU, and composites the fire over the
live camera views in real time. Use case: film previsualisation.

## 2. What the user wants now (their words, clarified over the session)

The user showed FieryGS (ICLR 2026, https://pku-vcl-geometry.github.io/FieryGS/, code
https://github.com/PKU-VCL-Geometry/FieryGS) as the **quality bar**, and was explicit that they do **not** want to
recreate FieryGS (FieryGS is offline: multi-view photos -> COLMAP -> 3D Gaussian Splatting -> GPT-4o materials ->
volumetric combustion -> offline render; no runtimes or real-time claims are published). What they want:

1. **Real scenes, not synthetic.** Point a fixed camera (tripod is fine) at a real still scene, live or as a recording
   from a fixed camera (the `file` camera type already plays a clip as if it were live).
2. **No printed calibration board.** The ChArUco board workflow is rejected for normal use.
3. **Click anywhere on the image -> fire starts exactly there** on the real object (e.g. click a table -> the table
   catches fire at that spot).
4. **The fire behaves like fire on the object**: flames attached to and licking the burning surface (not a plume
   floating in the scene, which is what the current renderer looks like), spreading across the object and to
   neighbours, **material-dependent** (wood, paper, cloth, plastic burn differently; metal/glass/ceramic do not).
5. **Things burn away**: char front spreading with glowing edges, paper/cloth/wood actually burning through, ash,
   smoke rising.
6. **Fire lights the scene** (orange flicker on nearby surfaces), like FieryGS.
7. **Live / interactive** frame rate on the RTX 4060 laptop.
8. Materials from a **local** vision-language model is fine; the user also said they can get an API key
   ("I can get API or anything you want").
9. **Still scene now; moving (handheld) shots later.**

The user is impatient with process detours and synthetic-only demos: show results on **real footage**.

## 3. Environment (the user's machine)

| Item | Value |
| --- | --- |
| Machine | Windows 11, RTX 4060 Laptop GPU 8 GB (driver 610.88, CUDA UMD 13.3), 16 cores, 32 GB RAM |
| Shells | PowerShell and Git Bash. **No `make`, no `gh` CLI** (install `winget install GitHub.cli` + `gh auth login` to open PRs) |
| Python | uv 0.10.8, Python 3.11 (pinned). System nvcc 11.8 is irrelevant (torch/Warp bring their own runtimes) |
| Main checkout | `E:\el 7th sem\FireSimulation` (on the old branch `agent/inspiring-maxwell-5od2va`; run `git checkout main && git pull`) |
| Install | from `livefire/`: `uv sync --all-groups --all-extras`. **Plain `uv sync` removes torch/transformers/taichi** (extras) |
| Device | `configs/app.yaml: device` is committed as `"cpu"` (cloud CPU agents need it). On this machine set `device: "cuda:0"` locally and **do not commit it** |
| uv cache | `C:\Users\nisch\AppData\Local\uv\cache` on C:, project on E: -> hardlinks fail, uv copies (slow). Optional: `setx UV_CACHE_DIR E:\uv-cache` |
| Model weights | `livefire/data/models` (gitignored HF cache): Depth Anything V2 (metric indoor + relative small), Grounding DINO tiny, SAM 2.1 tiny, Mask2Former, Depth Pro (being downloaded 2026-11-18). **SAM 3 not downloaded: gated** (user must accept at https://huggingface.co/facebook/sam3, then `uv run hf auth login` and `uv run hf download facebook/sam3 --cache-dir data/models`) |
| Cameras | Two Android phones running the "IP Webcam" app: `http://10.12.60.26:8080/video` (cam 0), `http://10.12.60.76:8080/video` (cam 1), 1280x720 ~28-30 fps. LAN IPs may change. Phones went black once (screen lock / lights) |
| Real test clip | `E:\el 7th sem\FireSimulation\test.mp4` (854x480, 10.6 s, handheld, bed + desk + laptop; **camera moves**, so only single frames are usable for now) |
| Calibration store | `E:\el 7th sem\FireSimulation\calib_data` (`calib_v001` active). **Invalid now**: phones/board moved after calibration (see 4.6) |
| Tests | fast suite: `uv run pytest -m "not slow" -q`; parallel: `uv run --with pytest-xdist pytest -m "not slow" -q -n 8 --dist loadfile` (~5 min on GPU) |
| Lint | `uv run ruff check src tests bench eval` |

## 4. What happened in this session (chronological, with evidence)

### 4.1 Setup and first GPU test run (2026-11-14/15)
- Dependencies installed from `uv.lock` (torch 2.11.0+cu128, Warp 1.19.0, OpenCV 5.0.0, taichi 1.7.4); torch and Warp
  both see the RTX 4060.
- First fast-suite run on CPU: 246 passed, 1 failed (camera reconnect, see below).
- The cloud agent's work was merged into `main` (`e1dc40f`): P0, P1-001..007, P2-001/002, P3-001/002 + unverified WIP
  (P2-003, P3-002 fix round). See `docs/GPU_TEST_HANDOFF.md` for that agent's own state description.

### 4.2 Fast suite on GPU and four fixes (commit `ca5c4c4`, on `main`)
GPU fast suite (8 xdist workers): 355 passed, 4 failed. Root causes and fixes:

| Failure | Root cause | Fix |
| --- | --- | --- |
| `test_failed_is_time_based_and_keeps_retrying` | `opencv_source._open` retried a failed URL open **without** a timeout (~5.4 s per attempt on Windows) | untimed fallback only when OpenCV rejects the timeout props (`cv2.error`) |
| `test_placement_good_rig_passes_and_angle_is_analytic` | arccos round-off: 60 deg -> 59.99999999999999 crossed a band edge (fails on CPU too) | `eval/placement.py` rounds the angle to 1e-6 |
| `test_replay_tolerance_branch` | test assumed device CPU; GPU replay correctly uses tolerance mode | expectation follows the configured device |
| `test_energy_budget_p2r3` | WIP had raised `T_rad_flame` 1100 -> 1400 K (unreviewed, for demo spread); emitters then radiate more than their heat release (`rad_deficit_J > 0`) | **user chose to revert to 1100 K** (ADR-0006 value). Consequence: less spread between objects (quick M9: target ignited at 2.1 s at 1400 K, not within 5 s at 1100 K) |

Also: `test_m9_quick_runs` (slow) expected the wrong file prefix (`quick_` is correct); `bench/p2_radiation_compare.py`
labelled GPU runs as `cpu` and overwrote the committed CPU CSVs (fixed: GPU runs write `radiation_gpu_*`,
`radiation_effect_gpu_*`).

GPU CSVs committed: `bench/results/spike_gpu_2026-11-15.csv` (Warp 2.4-9x faster than Taichi on every kernel ->
supports ADR-0001 Warp), `baseline_gpu_2026-11-15.csv`, `radiation_gpu_2026-11-15.csv` (mode 2 radiation 2.0-5.5 ms per
step), `depth_spike_gpu_2026-11-15*.csv` (synthetic only, DA2 small ~51-54 ms, ~224 MB), `segment_spike_gpu_2026-11-15*`
(Grounding DINO+SAM2 335-460 ms ~1.0-1.15 GB; Mask2Former 151-164 ms; SAM 3 skipped: gated).
Not committed (stale, produced at 1400 K): `eval/results/radiation_effect_gpu_2026-11-15.csv`, `eval/results/quick_*`.

### 4.3 Performance work (commits `a15f40c`, `dd6a4cc`, on `main`)
Profiling showed the GPU did little of the work:
- **Renderer**: the GPU ray march was ~6 ms of ~200-300 ms per view; the rest was host NumPy (full-res compositing,
  sRGB conversion, char/ash splats, per-pixel ray/box intersection). Moved to Warp kernels
  (`render/composite.py`: CSR splat, gap fill, fused composite; box intersection inside `march_kernel`). Output
  matches the old pipeline within 1 LSB (`test_device_composite_matches_host_reference`).
- **Gas step**: under `wp.config.deterministic = RUN_TO_RUN` (needed for GPU replay), **every kernel containing an
  atomic is wrapped in memsets and blocking device-to-host copies**, and in Warp **`arr[i] += x` compiles to an
  atomic**. Single-writer updates now use `a[i] = a[i] + x`; `divergence_k` no longer accumulates norms with one global
  atomic per cell (norms computed on demand in `divergence_norms()`); RBGS reads one packed `uint8` face mask instead
  of seven `int32` flags (`rbgs_mask_k`, bit-identical to `rbgs_k`, tested). CPU results bit-identical to before; GPU
  deterministic run to run; vs old GPU code differences are float32 rounding (FMA contraction).
- Result (`bench/results/baseline_gpu_2026-11-18.csv` vs `..._2026-11-15.csv`, G=128): sim step 81.5 -> 23.9 ms,
  gas step 76.3 -> 21.2 ms, render view 204 -> 25 ms, **two-view frame 513 -> 64 ms (~15.5 fps)**. Target (PRD M1)
  is 30 fps = 33 ms. Largest remaining cost: 60 RBGS pressure iterations (~0.5 ms each at G=128).
- The standalone `projection` bench row (45 ms) now includes the on-demand norm readback; the in-step projection is
  inside `gas_step`.

Full GPU fast suite after these commits: 365 passed, 1 failed. The failure
(`test_startup_with_one_camera_never_delivering_then_reacquire_with_both`) passes 3/3 alone: the test gives the
pipeline only 2 steps to see a late camera's first frame; flaky under 8-worker load. Not fixed.

### 4.4 Synthetic demos
`python -m livefire.app run --scene table_v1 --out out/demo_offline --grid 64 --offline --seconds 10 --ignite
"-0.075,0.0,0.1@0.3:0.08"` renders a 10 s two-view video (before the perf work it ran at 0.05x real time). The user's
verdict on the look: a tall orange column **"flame flying on a scene"**, not an object on fire. This is the main
visual problem to solve.

### 4.5 Calibration (separate session, branch `agent/jovial-poincare-0a30c0`, commit `065a1c5`, not pushed)
The user printed the ChArUco board (7x5 squares, 35 mm), calibrated both phones: intrinsics held-out RMS 0.33 / 0.55 px,
extrinsics reprojection 0.44 / 0.42 px, cams 1.09 / 1.00 m from the board, 43 deg apart (below the recommended 60-90).
That branch only changes `configs/cameras.yaml` to:
```yaml
cameras:
  - {cam_id: 0, type: url, source: "http://10.12.60.26:8080/video", width: 1280, height: 720, fps: 30, backend: auto}
  - {cam_id: 1, type: url, source: "http://10.12.60.76:8080/video", width: 1280, height: 720, fps: 30, backend: auto}
```
The calibration session later stopped with an SSL certificate error from its API connection (environment issue).

### 4.6 First real live runs (2026-11-18) and diagnosis
`uv run python -m livefire.app live --cameras <that cameras.yaml> --calib-root "E:\el 7th sem\FireSimulation\calib_data" --out out/live`
- Both phones connected (~28-30 fps, skew ~6-16 ms).
- Run 1: lifecycle CALIBRATING -> ACQUIRING -> ERROR `DepthError: anchor inlier ratio 0.410 below minimum 0.500`.
  The board then lay in a clear plastic tray (raised a few cm, glare broke ChArUco detection: 13 corners / none).
  Acquisition ran ~107 s before failing.
- After the tray was removed: verification on fresh frames showed the board **55-57 px (cam0, 20-24 corners) and
  74-124 px (cam1)** from where `calib_v001` puts it (limit 2 px): the phones or the board moved after calibration
  (likely when the phones went black and were touched). Run 2 failed the same way (inlier 0.432) after ~29 s.
- Depth diagnostic (`da2_metric_small`, single frames, 0.6x0.5 m anchor patch, 3 % threshold): inliers 0.63 (cam0) /
  0.57 (cam1), i.e. **borderline even when it passes**; the relative model failed both. A single failing frame stops
  the whole live start-up (acquisition aborts on the first `DepthError`).
- Conclusions: (a) any calibration that depends on nothing moving is fragile for this user; (b) monocular depth
  anchored to a plane is the weak link (it was risk 1 in `docs/HANDOVER.md`); (c) the board workflow is rejected by the
  user anyway. This motivated the replan.

### 4.7 Board-free geometry spike (in progress at writing time)
Script: Depth Pro (`apple/DepthPro-hf` via `transformers`, metric depth + focal length from one image) on the first
frame of `test.mp4`, then RANSAC of the dominant plane in the lower half. Result: see section 11 (filled in if the run
finished before the handoff; otherwise re-run it, the script is reproduced there).

## 5. Current architecture (what exists and is reusable)

Read `docs/ARCHITECTURE.md` for detail. Key modules (all under `livefire/src/livefire/`):
- `capture/`: OpenCV USB/URL/file sources (newest-frame-wins, reconnect), recorder. Reuse as is.
- `calib/`: ChArUco board, intrinsics, extrinsics, versioned store. Keep as a **fallback and as ground truth for
  evaluation**, not as the normal path.
- `perception/depth/`: depth backends (DA2 metric/relative, oracle) + table-plane anchoring (`anchor.py`).
- `perception/segment/`: Grounding DINO + SAM 2, SAM 3 (gated), Mask2Former backends (P3-001; ADR-0007 proposed).
- `fusion/`: depth accumulation, Warp TSDF, surfel extraction, table plane + clustering, cross-view association.
- `scene/`: `SceneSnapshot` (frozen geometry: surface elements `pos/normal/area/obj_id/material`, `view_depth`,
  `view_hole_mask`, solid mask on the gas grid), acquisition, synthetic scene.
- `sim/`: surface-element thermal/combustion model (`surface.py`), MAC gas solver (`gas.py`, RBGS projection),
  radiation modes 0/1/2 (`radiation.py`), coupled stepping (`coupled.py`), materials (`configs/materials.yaml`).
- `render/`: ray-marched volume (`volume.py`), GPU compositor (`compositor.py`, `composite.py`), char/removed overlay.
- `app/`: `run` (offline/real-time synthetic), `interactive` (click-to-ignite controller, synthetic), `live` (full
  lifecycle: calibrate-load -> acquire -> READY -> simulate on live frames, HUD, change monitor), `capture`, `calib`.

## 6. Brainstorm: approaches per subsystem

Scoring follows PRD 11.6 in spirit (quality, speed on 8 GB, license, integration effort). At most 3 candidates per
decision get a time-boxed spike and an ADR (CLAUDE.md rule).

### 6.1 Camera intrinsics, pose and world frame without a board (decides everything downstream)
| # | Approach | How | Pros | Cons |
| --- | --- | --- | --- | --- |
| A | **Single-image metric geometry**: Depth Pro (Apple), MoGe-2 (Microsoft), UniDepthV2 | one frame -> metric depth + focal length (MoGe also normals); world frame from RANSAC of the supporting plane (+Z = plane normal, origin = plane point under the image centre or under the first click) | works with ONE fixed camera; no board; seconds at setup; frozen geometry so per-frame cost is zero | metric scale error typically several % to ~10 %; thin/transparent objects; only the visible side |
| B | **Two-view feed-forward reconstruction**: VGGT, MASt3R (DUSt3R family) | both fixed cameras' frames -> relative pose + dense point maps (+ intrinsics) in one pass | gives the camera pair geometry with no board; denser, more consistent geometry than two single-view depths | 2 cameras required; VRAM (VGGT-1B fp16 needs checking on 8 GB); licenses must be checked (some checkpoints are non-commercial); scale still needs an anchor |
| C | Classic features: SIFT / SuperPoint+LightGlue -> essential matrix -> triangulation | match the two views, recover pose, triangulate | well understood, light | needs texture and baseline; scale unknown; intrinsics from EXIF or A |
| D | Keep ChArUco (existing) | as built | most accurate when nothing moves | rejected by the user for normal use; fragile (4.6) |

Scale anchor options: the metric model's own scale; one typed measurement (e.g. table width) in a setup dialog; a known
object (A4 sheet, credit card) detected in view; average of both cameras (B). **Recommendation:** milestone 1 uses
**A (single fixed camera)**; two cameras later via **B**, with A as a per-view cross-check and D kept for evaluation
ground truth (M4/M6 metrics). Spike Depth Pro vs MoGe-2 vs UniDepthV2 on real frames (ADR-0008).

### 6.2 Scene geometry for simulation
| Option | Notes |
| --- | --- |
| Depth -> point cloud -> surface elements (existing `fusion` + `scene/surfels`) | reuse; feed it the new depth; per-object surfels from segmentation masks. **Recommended.** |
| 3D Gaussian Splatting (FieryGS style) | needs many views / a moving camera and minutes of training; good for the later "moving shots" stage, not for a still fixed camera |
| Mesh from depth (Poisson / TSDF mesh) | useful for relighting normals and for thickness; optional |

Known limitation: a single fixed camera sees only the front surfaces; give objects an assumed thickness (extrude
along the view ray or along -normal) so the gas solver sees solids and burning has volume.

### 6.3 Objects and segmentation
- **Click-driven**: on click, run SAM 2 with the click as a point prompt -> mask of the clicked object -> ignite
  only that object's surface elements near the click. Immediate, robust, matches the user's interaction. **Do first.**
- **Automatic at setup**: SAM 2 automatic masks or Grounding DINO + SAM 2 (existing P3-001 backends) -> objects for
  materials and spread.
- SAM 3 (text-concept prompts) once the user gets gated access.

### 6.4 Materials
| Option | Pros | Cons |
| --- | --- | --- |
| Local VLM on object crops (Qwen2.5-VL-3B/7B, Moondream, Florence-2) -> label from a fixed list + confidence | user-approved, free, offline | 7B tight on 8 GB next to the sim: run at setup, then unload (PRD acquisition/simulation modes) |
| CLIP / SigLIP zero-shot over material prompts | very fast, small | weaker on fine material distinctions |
| GPT-4o (or similar) API | best quality (FieryGS uses it, ~$0.55/scene) | needs a key and network; user said they can get one |

Map labels to `configs/materials.yaml`; flag low confidence; keep the manual override (P3-004). **Recommendation:**
local VLM first with CLIP as the cheap fallback; API as an optional backend behind the same interface (ADR-0009).

### 6.5 Fire look (the biggest visible gap)
Current: flames are emission from the gas temperature / `flame` field on a coarse grid (G=96-128 over a 3 m ROI ->
2-3 cm cells), ray-marched; at that resolution flames become a soft column above the object.

| Approach | How | Pros | Cons |
| --- | --- | --- | --- |
| a. Surface-attached flame field | emit flame density **from burning surface elements** (per-element burn rate -> flame height), advect it with the gas velocity for a short lifetime, add **world-space curl noise** for licking detail (PRD 7.4.3); render with a sharper transfer function and a bright core | stays attached to the object; physically driven by the existing sim; consistent across views | needs tuning; still limited by grid resolution |
| b. Higher local resolution | a fine flame grid (e.g. 128^3 over a box of ~0.5-1 m) around the active fire, or a narrow band near burning surfaces | sharper flames | more memory/time; needs the solver speed-ups (6.8) |
| c. Particle / sprite flames | spawn particles at burning elements, advect in the gas velocity, render as additive blackbody sprites (or flipbook sprites) | cheap, very fire-like, art-directable | less physical; risk of "video game" look; screen-space sprites must still be placed in world space for two views |
| d. Learned fire (video diffusion) | generate fire appearance | none for live | far too slow; out of scope |

**Recommendation:** a + c hybrid first (attached flame field driven by the surface model, plus particle flames for the
licking tips), b only if needed after measuring (ADR-0010). Reference look: FieryGS kitchen/campfire examples.

### 6.6 Char, burn-through and what appears behind
- Level 1 (exists): char darkening, ash tint from `char` / `removed` per element.
- Add: **burn front glow** (ember edge: elements near the burning/char boundary or with high `T_s` glow orange-red),
  smouldering after flames die.
- **Burn-through reveal** (PRD level 2, currently "stretch"; the user wants it): when an element is `removed`, the
  pixels behind it must show what was behind the object. Options: (i) per-object **inpainted removal plate**
  computed once at setup with a local inpainting model (LaMa is fast; SD-inpaint is slower) using the object mask;
  (ii) the other camera's view (two cameras); (iii) clean background plate captured before placing the object (needs
  the user to do it); (iv) dark charred hole with glowing edges (cheapest, previs-acceptable). **Recommendation:**
  (iv) immediately, then (i). Geometry never collapses in the sim (PRD); burn-away stays a visual layer.

### 6.7 Smoke and fire light
- Smoke: soot field exists; render with ambient scattering and grey-brown tint, more soot output, rising and dispersing.
- **Fire light on the live image** (PRD 7.4.4): per pixel, irradiance from a coarse emission grid (or the list of
  burning elements) using the snapshot normals and depth; inverse-square falloff, flicker from the flame field;
  additive in linear light, then tone map. Shadows are a stretch goal.

### 6.8 Performance (PRD M1: 30 fps two views at 720p, G=96-128)
Now 64 ms per two-view frame at G=128 (15.5 fps). Options: multigrid-preconditioned CG or a geometric multigrid for
pressure instead of 60 RBGS iterations (largest win; numerics change -> ADR); CUDA graph capture of the whole sim step
(launch overhead); compute the active box on the GPU (removes 3 field downloads per view per frame); render at
`volume_scale` 0.5 with bilateral upsampling; one camera halves render cost.

### 6.9 Interaction and app flow
Setup (a few seconds): grab frame(s) -> geometry (6.1) -> world frame -> objects (6.3) -> materials (6.4) ->
snapshot -> READY. Then: click -> ray-cast `view_depth` -> SAM click mask -> ignite; keys for material override,
reset, record. Change monitor: re-acquire if the camera or scene moves. Same lifecycle as `app/live.py`; add a
"no-board" calibration source.

### 6.10 Moving camera (later, user-confirmed)
Needs per-frame camera tracking (SLAM or VGGT/MASt3R-style tracking) and a persistent reconstruction (3DGS fits
here, FieryGS-style). Out of scope until the still-scene product works.

## 7. Roadmap (new)

Each milestone ends with something the user can **see on real footage**. Keep the existing rules: task file in
`docs/tasks/`, tests, ADR for choices, numbers only from committed CSVs, never weaken tests.

| # | Milestone | Done when |
| --- | --- | --- |
| **M1** | **Board-free single-camera setup** (ADR-0008: Depth Pro vs MoGe-2 vs UniDepthV2 on real frames). New calibration source: intrinsics + metric depth + plane -> world frame -> `SceneSnapshot` | from one still clip of a real table (fixed camera), the reconstructed surfaces overlay the image correctly (`o` overlay), table plane found automatically, no board |
| **M2** | **Click-to-ignite on a real clip** | `live` (or a new `scene` command) on a still clip: click on the table -> fire starts at that pixel and spreads over the table's surface elements; replay works |
| **M3** | **Fire look v1**: surface-attached flames + world-space noise + particle tips, ember front, char, dark burn-through holes, smoke | side by side with the FieryGS kitchen/campfire stills, the user agrees it reads as "the object is on fire" |
| **M4** | **Fire light** on the live image + exposure match | orange flicker on nearby real surfaces |
| **M5** | **Objects + materials**: SAM 2 click masks, automatic objects, local VLM materials (ADR-0009), manual override | wood/paper/cloth burn differently, metal/glass don't, without manual tagging |
| **M6** | **Two fixed cameras without a board** (ADR on VGGT vs MASt3R vs features) | one shared fire, consistent in both views |
| **M7** | **Performance**: pressure solver ADR, CUDA graphs, GPU active box | >= 25-30 fps on the demo scene, measured CSV |
| **M8** | Burn-through reveal with inpainted plates | burnt paper shows the table underneath |
| later | Moving camera | tracking + persistent reconstruction |

Suggested first concrete steps for the next session:
1. Finish/repeat the Depth Pro spike (section 11) and add MoGe-2 (`uv run --with git+https://github.com/microsoft/MoGe.git`)
   and UniDepthV2; ask the user to record **a still clip from a fixed phone** of a table with a few objects
   (`uv run python -m livefire.app capture --record DIR` with a one-camera `cameras.yaml`, or any phone video on a tripod).
2. Implement M1 as a new calibration/geometry source that produces the same `CalibrationSet`-like camera + plane data
   the acquisition expects, so `fusion`, `scene`, `sim` and `render` stay unchanged.
3. Then M2 on that clip, then M3.

## 8. Decisions already made (do not reopen without the user)
- Keep the LiveFire live thesis; FieryGS is inspiration and quality bar, not a base to copy.
- No printed board in normal use; tripods fine; still scenes now, moving shots later.
- Local VLM for materials is fine; API keys available if needed.
- `T_rad_flame` = 1100 K (ADR-0006); spread must come from physics/tuning, not from breaking energy accounting.
- Warp is the GPU framework (spike data supports ADR-0001; mark it accepted).
- `RUN_TO_RUN` determinism stays on (GPU replay depends on it); write single-writer updates without `+=` on arrays.

## 9. PRD / ADR changes this replan requires
- PRD section 1 / 3 / 7.1: board-free geometry moves from "deferred (learned geometry replacing calibration)" to the
  core path; ChArUco becomes fallback + evaluation ground truth. Record in the PRD change log with an ADR (ADR-0008).
- PRD 7.4 level 2 (burn-through reveal) moves from stretch toward MVP (user requirement), starting with dark holes.
- PRD 7.4.3 flame appearance model and 7.4.4 fire light move earlier (they were P4).
- Single fixed camera becomes a first-class mode (PRD assumed two views; P1R1 already allows single-view fallback).
- ADRs to write: 0008 board-free geometry, 0009 materials model, 0010 flame rendering, 0011 pressure solver;
  update 0001 (accepted, GPU data), 0005 (superseded or narrowed by 0008), 0007 (re-score from the GPU spike CSV).

## 10. Repository state at handoff
- `main` = `dd6a4cc` on GitHub (contains everything above except the calibration branch and this document's commit,
  which is pushed right after). Old branches `agent/inspiring-maxwell-5od2va`, `agent/magical-clarke-xah78d`,
  `worktree-agent-af7a64019b66f7883`, `agent/project-dependencies-setup-47f41e`, `agent/p4-performance` are all
  ancestors of / merged into `main`. GitHub's default branch is still `agent/inspiring-maxwell-5od2va` (the user may
  switch it to `main` in repo settings).
- Not on GitHub: branch `agent/jovial-poincare-0a30c0` (only the `cameras.yaml` change shown in 4.5).
- Local only: `calib_data` (invalid now), `out/` run folders, model weights.
- Gates: Gate 0 approved; Gates 1-3 **not passed**. P2-003 (scenarios/M9) still unfinished; P2-004..006 and
  P3-003..006 not started (now re-prioritised by section 7).

## 11. Board-free geometry spike result (Depth Pro, 2026-11-18)

Result on the first frame of `test.mp4` (854x480, real room: bed, pillow, headboard, desk with laptop), RTX 4060,
fp16, measured by the script below (not a committed CSV; re-run to reproduce):

| Quantity | Value |
| --- | --- |
| Latency per frame | 1039 ms (setup-time only, acceptable) |
| Peak VRAM | 3767 MB (must be unloaded before the simulation starts) |
| Estimated focal length | 760 px (horizontal FOV 58.6 deg) |
| Depth p5 / p50 / p95 | 1.21 / 1.69 / 2.04 m (plausible for the room) |
| Largest plane in the lower half (1 cm band) | 9.5 % of pixels: the bed surface; camera 0.80 m above it |

Visual check: the depth map is sharp and plausible (laptop, desk legs, pillow and headboard edges crisp). The plane
picked was the bed, not the desk (the desk top is smaller in this view), so M1 needs plane selection: the largest
near-horizontal plane, or the plane under the user's first click. No ground truth was available, so metric accuracy is
**not** measured yet: next, a still clip of a table with a tape-measured distance (or the old ChArUco board as ground
truth) and the same run for MoGe-2 and UniDepthV2. Verdict so far: viable as the board-free path for M1.

Script:
`bench/depthpro_board_free_spike.py` (one frame -> Depth Pro metric depth + focal length -> back-projection -> RANSAC of
the dominant plane in the lower half of the image, 1 cm inlier band; prints latency, VRAM, focal/FOV, depth percentiles,
plane coverage and camera height; writes an image with the plane highlighted next to the depth map). Run from
`livefire/`:
`uv run python bench/depthpro_board_free_spike.py "E:/el 7th sem/FireSimulation/test.mp4" out/spike_depthpro.jpg`
(on a still clip of a table from a fixed phone the result is far more meaningful than on `test.mp4`). Compare MoGe-2 and
UniDepthV2 the same way for ADR-0008.

## 12. Prompt for the new Claude session (paste this)

> I'm continuing the LiveFire project on my Windows 11 + RTX 4060 laptop. Repo `nischal2805/FireSimulation`, branch
> `main`, project folder `livefire/`. First read `livefire/docs/REPLAN_LIVE_FIERY.md` completely (it has the whole
> history, what I want, the brainstormed approaches and the roadmap), then `docs/PRD.md`, `CLAUDE.md`,
> `docs/STATUS.md`. Set `configs/app.yaml` to `device: "cuda:0"` locally (don't commit it) and install with
> `uv sync --all-groups --all-extras`. Then start roadmap milestone M1 (board-free single-camera setup on a real still
> clip). I want FieryGS-quality fire on real objects, live, with no printed board: click on an object and it catches
> fire there, spreads, chars, burns through, smokes and lights the scene. Show me results on real footage, not
> synthetic scenes.
