## Task 4: P1-004 Depth backends spike and ADR-0005

**Files:** `src/livefire/perception/depth/{base.py,anchor.py,hf_backends.py,oracle.py}`, `bench/depth_spike.py`,
`docs/decisions/ADR-0005-depth-backend.md`, `tests/test_depth_anchor.py`, `tests/test_depth_backends.py`,
`pyproject.toml` (`depth` extra + uv sources), `configs/depth.yaml`.

- Time-boxed research (PRD 0.2, max 3 candidates, web search for 2025-2026 options) for monocular depth that fits
  8 GB, runs at >= 2 fps at 720p on an RTX 4060 class GPU and has a usable license; seeds: Depth Anything V2
  (metric-indoor and relative), Depth Anything 3, MoGe-2, UniDepth V2, Depth Pro. Pick <= 3 that load through
  Hugging Face `transformers` or a pip package without custom CUDA builds.
- `DepthBackend` implementations per PRD 9 (warmup, infer -> `DepthResult` host NumPy per P0 ruling, latency and
  VRAM properties, fp16 on CUDA, model download cached under `data/models/`), selected by `configs/depth.yaml`.
  `OracleDepthBackend` (`# MOCK:` ground truth from the synthetic renderer, optional seeded noise/bias) for tests.
- `anchor.py` (P1R3): given a calibrated camera, the table-region mask (board footprint expanded, or a polygon in
  world coordinates) and a raw depth/disparity map, compute analytic plane depth for table pixels and fit
  scale (metric) or scale+shift in the model's native space (relative/disparity) with RANSAC + least squares;
  return anchored Z-depth, inlier ratio and residual statistics; reject (raise `DepthError`) below a configurable
  inlier ratio.
- `bench/depth_spike.py`: on (a) the synthetic scene rendered with the P0 renderer (sanity only), (b) a small public
  RGB-D sample with a permissive license (e.g. a few frames of TUM RGB-D or another dataset whose license allows
  research use; record license and source URL in a manifest; download into `data/`, gitignored) evaluated with
  table/floor-plane anchoring, and (c) the user's recorded clips when present: metric MAE, AbsRel, valid coverage,
  latency, peak VRAM; CSV in `bench/results/`. CPU timings labelled CPU.
- ADR-0005 scored by PRD 11.6, status **proposed** until the user's clips and GPU timings are in.
- Tests: anchoring recovers a known scale (and shift) from a synthetic plane with outliers within 1 %; rejection on
  low inliers; Oracle backend contract (shapes, invalid = 0.0, no NaN); HF backends tested only if the `depth` extra
  and weights are available (skip otherwise), on one small image, output contract only.

**Acceptance:** anchoring tests pass; spike CSV committed for what could be run; ADR drafted.

