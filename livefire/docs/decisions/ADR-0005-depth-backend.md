# ADR-0005: Monocular depth backend and table-plane anchoring

Date: 2026-11-14. Status: **proposed** (draft by implementer, P1-004). Pending: the user's recorded clips and RTX 4060
timings (no GPU, camera or clips exist in the development container). The controller finalises the decision.

## Context

PRD 7.2 / P1R3: per-view metric Z-depth from a monocular network, scale-anchored to the calibrated table plane
`z = 0` (world frame, PRD 4). Requirements: fits 8 GB VRAM, >= 2 fps at 720p on an RTX 4060 class GPU, usable license,
loads through Hugging Face `transformers` or a pip package without custom CUDA builds, runs on CPU here for tests.
PRD 0.2: at most 3 candidates, time-boxed. Torch lives in the optional `depth` extra (uv sources: CPU wheels on Linux,
CUDA 12.8 wheels on Windows); core code imports without it.

## Candidates (source, date, license, size)

Seeds searched: Depth Anything V2 (metric, relative), Depth Anything 3, MoGe(-2), UniDepth V2, Depth Pro. Facts
marked (HF API) were read from `https://huggingface.co/api/models/<id>` in this container; (search) = seen only in web
search snippets, not verified at the source.

| # | Candidate | Weights (HF) / size | License | Loading | Status |
| --- | --- | --- | --- | --- | --- |
| A | Depth Anything V2 Small, metric-indoor (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, 2024-08) | 99.2 MB safetensors, ~25 M params | Model card has **no license tag** (HF API). Upstream README (raw.githubusercontent.com/DepthAnything/Depth-Anything-V2) states: Small model Apache-2.0, Base/Large/Giant CC-BY-NC-4.0; it does not name the metric fine-tunes explicitly, so Apache-2.0 for this one is **presumed, to verify** | `transformers` (`AutoModelForDepthEstimation`) | spiked |
| B | Depth Anything V2 Small, relative (`depth-anything/Depth-Anything-V2-Small-hf`, 2024-07) | 99.2 MB | Apache-2.0 (HF API tag + upstream README) | `transformers` | spiked |
| C | Depth Anything 3 (`depth-anything/DA3-SMALL` 137 MB, `DA3METRIC-LARGE` 1337 MB; 2025-11) | see left | Apache-2.0 (HF API tags); Nested-Giant variant is CC-BY-NC-4.0 (search) | pip `depth-anything-3` 0.1.1 only (HF API: no `transformers` integration; `transformers` main has no `depth_anything_3` module) | **not run**: package requires xformers, open3d, moviepy, pycolmap, `numpy<2` (PyPI metadata); importing the API pulls the video/GS export stack (moviepy missing on a `--no-deps` install). Too heavy for the time box; best next candidate if A/B prove inadequate on the user's clips |

Eliminated without running: **Depth Pro** (`apple/DepthPro-hf`, 1.9 GB class model; HF license tag `apple-amlr`, a research-use
license, so the PRD 11.6 license hard limit applies; license wording not read here). **MoGe**: the upstream `main` is now
MoGe-3 whose `pyproject.toml` depends on git builds including FlexGEMM (custom CUDA build); MoGe-2 ViT-S weights
(`Ruicheng/moge-2-vits-normal`, 141 MB, MIT per HF API) were not tried (older code tag needed). **UniDepth V2**: not
checked (time box). **Marigold** (`prs-eth/marigold-depth-v1-1`, openrail++, diffusion, multi-step): too slow for 2 fps.

## Method (data, metrics, hardware, commands)

- Code: `src/livefire/perception/depth/{base,anchor,hf_backends,oracle}.py`, `configs/depth.yaml`,
  `bench/depth_spike.py`. Anchoring (`anchor.py`): analytic Z-depth of `z = 0` inside a world-XY table polygon,
  RANSAC (seeded) + least squares in the model's native space: metric `z = s * raw`; disparity `raw = a / z + b`.
  Rejects (`DepthError`) below `min_inlier_ratio` (0.5), `min_pixels` (400), non-positive scale, degenerate plane span.
  Inlier threshold: `inlier_thresh_rel` = 0.03 of plane depth (metric) / of the table-disparity p5-p95 span (disparity).
  Variants in the CSV: **strict** = `configs/depth.yaml`; **lenient** = threshold 0.08, min ratio 0.3 (diagnostic only;
  not adopted).
- Datasets: (a) synthetic `table_v1` camera 0 (sanity only: flat-shaded, out of the models' domain; exact GT);
  (b) TUM RGB-D `fr3/long_office_household`, CC BY 4.0, https://cvg.cit.tum.de/data/datasets/rgbd-dataset (intrinsics
  fx 535.4, fy 539.2, cx 320.1, cy 247.6 from the TUM calibration page; depth scale 5000), 10 frames with a dominant plane
  covering >= 15 % of GT-valid pixels. **Caveat:** the anchor plane is the dominant plane RANSAC-fitted to each frame's *GT*
  depth (stand-in for the calibrated table plane), its inlier convex hull is the table region, and that plane is not
  always a table (floor/desk). This favours anchoring; the project's plane comes from calibration. (c) the user's clips:
  harness implemented (`--clips data/clips`, `cam<N>.mp4` + active calibration), **no clips available: not run**.
- Metrics: MAE (m) and AbsRel over GT-valid pixels, valid coverage (anchored-valid / GT-valid), latency of `infer_raw`
  (network only; anchor time reported separately in `notes`), peak VRAM (empty on CPU). Manifest with source URL and
  license: `bench/results/depth_spike_cpu_2026-11-14_manifest.json`.
- Hardware: container, 4 CPU cores, no GPU, torch 2.15.0+cpu, Python 3.11, `input_size` 518. **CPU numbers only**,
  labelled `gpu=none, framework=torch-2.15.0-cpu`. The machine was shared with other jobs while measuring, so latency
  p95 contains contention spikes. fp16-on-CUDA and VRAM code paths exist and are **untested** (no GPU).
- Commands: `uv sync --all-groups --extra depth`; `uv run pytest tests/test_depth_anchor.py tests/test_depth_backends.py`;
  `make depth-spike` (= `uv run python bench/depth_spike.py`).

## Results (table, CSV links)

CSV: `bench/results/depth_spike_cpu_2026-11-14.csv` (Appendix C schema; accuracy in `notes`), per-frame:
`bench/results/depth_spike_cpu_2026-11-14_frames.csv`. All CPU.

TUM, 10 frames, 640x480 input (median / p95 latency over 10 frames after 1 warm-up; MAE/AbsRel averaged over frames
that were accepted):

| Backend | anchor | frames accepted | MAE m | AbsRel | table MAE m | unanchored MAE m / AbsRel | latency median / p95 ms (CPU) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A da2_metric_small | strict | 0/10 | - | - | - | 0.360 / 0.192 | 593 / 3192 |
| A da2_metric_small | lenient | 8/10 | 0.323 | 0.125 | 0.197 | 0.360 / 0.192 | 593 / 3192 |
| B da2_relative_small | strict | 2/10 | 0.095 | 0.046 | 0.068 | n/a (disparity) | 525 / 557 |
| B da2_relative_small | lenient | 9/10 | 0.255 | 0.103 | 0.149 | n/a | 525 / 557 |

Synthetic table_v1 cam0 (1280x720, sanity only): oracle (MOCK) metric / disparity anchored MAE 0.6 mm / 3.3 mm
(anchoring maths verified end to end); A strict rejected (inlier 0.48), lenient MAE 0.79 m; B strict MAE 0.85 m;
table-region MAE 0.016 m (B) and 0.052 m (A, lenient). CPU latency 705-808 ms at 1280x720. Anchor fit time 20-60 ms.
Valid coverage 0.97-1.00 everywhere (the networks produce dense output).

Interpretation: a single plane fixes global scale (and shift) well on the table itself (cm-level) but not the models'
non-affine error elsewhere; MAE grows to 0.3-0.7 m on frames where the plane is a small part of a deep scene. Strict
acceptance (3 % inliers, 50 %) is too tight for these networks on real data; that threshold is a tuning decision for the
user's clips (the lenient column is diagnostic).

## Scores (11.6, draft, 0-5 per criterion, weighted to 100)

| Criterion (weight) | A metric-small | B relative-small | C DA3 (not run) |
| --- | --- | --- | --- |
| Accuracy on project data (35) | 2 (14) | 3 (21) | n/a |
| Latency and VRAM on RTX 4060 (30) | 4 (24), provisional: 25 M params, no GPU timing | 4 (24), provisional | n/a |
| Robustness (20) | 2 (8): 0/10 strict | 3 (12): 2/10 strict, 9/10 lenient | n/a |
| Implementation risk (10) | 5 (10) | 5 (10) | 1 (2) |
| License (5) | 3 (3): presumed Apache-2.0, unverified | 5 (5) | 4 (4) |
| **Total** | **59** | **72** | not scored (not run) |

The accuracy and robustness scores rest on 10 frames of one indoor sequence with GT-fitted planes; the latency score on
CPU timing plus parameter count, not on a 4060 measurement. They are drafts.

## Decision and fallback

Proposed (not final): primary **B `da2_relative_small`** (scale+shift anchoring, Apache-2.0 verified, smallest and
simplest), config-selectable fallback **A `da2_metric_small`** (`configs/depth.yaml: backend`). If neither meets the
accuracy target on the user's clips, the next spike is C (DA3) in a separate environment or MoGe-2, plus P1 fusion across
two views (depth accumulation) to average errors. Hard limits: none triggered for A/B.

## Consequences

- `depth` extra (torch CPU on Linux, CUDA 12.8 on Windows) is optional; `OracleDepthBackend` (MOCK) covers tests.
- Anchoring is per view and per call; static scene means it can run once per acquisition window and be reused.
- `DepthBackend.infer` returns metric depth only for metric backends; relative backends need `AnchoredDepthBackend`.
- To finalise: run `make depth-spike` on the RTX 4060 (`device: cuda:0`) with the user's clips in `data/clips`, tune
  `anchor.inlier_thresh_rel`/`min_inlier_ratio`, then confirm the license of A at its source.
