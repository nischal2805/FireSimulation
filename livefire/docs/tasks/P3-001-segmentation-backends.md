## Task 1: P3-001 Segmentation backends (keyframes), oracle, spike and ADR-0007

**Files:** `src/livefire/perception/segment/{base.py,oracle.py,hf_backends.py,factory.py}` (new),
`configs/segment.yaml` (new), `pyproject.toml` (`perception` extra), `uv.lock`, `bench/p3_segment_spike.py` (new),
`docs/decisions/ADR-0007-segmentation.md` (new, proposed), `tests/test_segment.py` (new), Makefile `segment-spike`
target + README rows (`uv sync --extra perception`).

- `configs/segment.yaml`: `backend: oracle|gdino_sam2|sam3|mask2former` (default `gdino_sam2`), `model_cache_dir:
  data/models`, `fp16_on_cuda: true`, `keyframes: 1`, `vocabulary: [paper sheet, cloth, towel, wooden block, plastic
  bottle, metal cup, ceramic mug, drinking glass, book, cardboard box, table]` (text prompts), `box_threshold:
  {value: 0.30, unit: "-", range: [0.05, 0.9], status: artistic}`, `text_threshold {0.25, ...}`,
  `mask_threshold {0.5, ...}`, `min_mask_area_px: 400`, `max_instances: 32`, per-backend `hf_id`s from P3R1,
  `oracle: {seed: 1234, erode_px: 1, drop_prob: 0.0, permute_ids: true}`.
- `base.py`: `SegmentationResult` construction helpers (masks -> `instance_ids` int32 with -1 background, overlapping
  masks resolved by higher score), `segment_extra_available()`, `create_segmentation_backend(name=None, cfg=None,
  device=None)`.
- `oracle.py`: `OracleSegmentationBackend(calib, scene=None, ...)` (`# MOCK:`), mirroring
  `perception/depth/oracle.py::OracleDepthBackend`: renders `obj_id` with `scene/synthetic.py::render_view`, drops the
  room (-1), optionally erodes masks and permutes ids per camera with a seeded permutation (so per-camera ids differ
  from world ids, exercising P3R3), categories from the scene's object names mapped to vocabulary words (e.g.
  `paper_sheet -> paper sheet`), scores 1.0.
- `hf_backends.py`: `GDinoSam2Backend`, `Sam3Backend`, `Mask2FormerBackend` (lazy imports; `warmup`, `segment`,
  `release` (P3R6), metrics via `MetricsMixin`); output contract per `SegmentationBackend` docstring; `SegmentationError`
  on model failure; category = vocabulary phrase (GDINO, SAM 3) or COCO label (Mask2Former).
- `bench/p3_segment_spike.py [--backends ...] [--clips DIR]`: inputs (a) `table_v1` renders from both synthetic
  cameras (sanity, flat shading), (b) the user's recorded clips when present with hand-labelled masks (the script
  documents the label format: one PNG of instance ids per keyframe + a YAML of categories), (c) optionally a few
  COCO val2017 images (annotations CC BY 4.0; record the per-image license in a manifest) - skip if not fetched.
  Metrics: per-object mask IoU (Hungarian matching), recall at IoU 0.5, category accuracy, latency (median/p95 of
  5 repeats after warm-up), peak VRAM -> `bench/results/segment_spike_cpu_<date>.csv`. Run the ungated candidates on
  CPU here; `sam3` rows say `skipped: gated` unless access exists.
- ADR-0007 (Appendix B): candidates (source, date, license, size), method, CSV links, PRD 11.6 scores with GPU
  latency/VRAM "pending user run", decision **proposed** (expected `gdino_sam2` primary, geometric clustering fallback),
  consequences.
- Tests (fast, no weights): oracle on both `table_v1` cameras: every visible object has an instance, IoU vs ground
  truth >= 0.95 with `erode_px: 1`, ids permuted per camera and consistent with the stored permutation, background
  -1, contract validation passes, `calib_version` copied; factory selects by name and raises a clear error for an
  unknown name; `SegmentationResult` helper resolves overlaps by score; core imports without torch (subprocess test
  with `torch` blocked via `sys.modules["torch"] = None`); real backends: skip unless the extra and cached weights
  exist, then one image -> contract only.

**Acceptance:** fast tests green; spike CSV committed for the candidates that ran; ADR-0007 proposed.

