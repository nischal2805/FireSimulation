# ADR-0007: Keyframe instance segmentation backend

Date: 2026-11-15. Status: **proposed** (draft by implementer, P3-001). Pending: the user's recorded clips with
hand-labelled keyframes (none exist in the development container), RTX 4060 latency/VRAM ("pending user run"), and the
user's decision on the gated SAM 3 weights. The controller finalises the decision.

## Context

PRD 7.3 / P3R1: per-camera instance masks on keyframes (no tracking), with a category label that feeds the material
prior. P1 finds objects by geometric clustering only and cannot separate a flat cloth from the table. Requirements: fits
8 GB VRAM (models are loaded only while ACQUIRING and released on READY, P3R6), loads through `transformers` without
custom CUDA builds, runs on CPU here for tests, usable licence. PRD 0.2: at most 3 candidates. Torch lives in the optional
`perception` extra (torch, torchvision, transformers, pillow; torchvision is required by the SAM 2 image processor);
core code imports without it and no test needs weights.

## Candidates (source, date, license, size)

Facts marked (HF API) were read from `https://huggingface.co/api/models/<id>` in this container on 2026-11-15.

| # | Candidate | Weights (HF) / size | License | Status |
| --- | --- | --- | --- | --- |
| 1 | `gdino_sam2`: Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`, created 2023-09, updated 2024-05) + SAM 2.1 Hiera-tiny (`facebook/sam2.1-hiera-tiny`, 2024-09) | 689 MB + 156 MB safetensors (HF tree API) | Apache-2.0 both (HF API tag and card `license:` line); ungated | measured (CPU) |
| 2 | `sam3`: `facebook/sam3` (2025-11) | not readable (gated) | HF tag `license:other` (Meta SAM License, not read: card is gated); `gated: manual` | **not measured here: gated**, needs user licence acceptance + HF token; VRAM unverified (a third-party note says 16 GB recommended, not verified): possible hard limit (PRD 11.6) |
| 3 | `mask2former`: `facebook/mask2former-swin-tiny-coco-instance` (2022-12) | 190 MB safetensors (HF tree API) | HF tag and card say `license: other` only. Upstream code repo `facebookresearch/Mask2Former` LICENSE is MIT (portions Apache-2.0/MIT, read from the repo README). The weights were trained on COCO (annotations CC BY 4.0, images mixed Flickr licences). Weight licence therefore **presumed MIT, not verified at the source** | measured (CPU) |

## Method (data, metrics, hardware, commands)

- Code: `src/livefire/perception/segment/{base,oracle,hf_backends,factory}.py`, `configs/segment.yaml`,
  `bench/p3_segment_spike.py`, `tests/test_segment.py`. Backends take a `Frame` and return `SegmentationResult`
  (per-camera ids, -1 background); overlapping masks are resolved by score; the vocabulary is the text prompt.
- Inputs: (a) `table_v1` renders from both synthetic cameras (sanity only: flat Lambert shading, out of the models'
  domain; GT = exact `obj_id`; categories through `category_map`); (b) the user's labelled clips: harness implemented
  (format in the script docstring), **not run: no clips**; (c) 6 COCO val2017 images (annotations CC BY 4.0,
  https://cocodataset.org/#termsofuse; selected: 2-8 non-crowd instances of area >= 2 % of the image, >= 2 categories;
  per-image Flickr licence in the manifest, several are NonCommercial/NoDerivs: images are used for local measurement
  only, kept in gitignored `data/`, and not redistributed). **Caveat:** COCO's ground-truth objects (person, bed,
  refrigerator, teddy bear, ...) are not in the project vocabulary, so `gdino_sam2` cannot be scored fairly on them; the
  zero recall there reflects the prompt vocabulary, not the model's mask quality. COCO category accuracy is not computed.
- Metrics: Hungarian matching on mask IoU; mean IoU over GT (misses = 0), recall at IoU 0.5, category accuracy among
  matches, latency = `segment` only (median / p95 of 5 timed calls after warm-up), VRAM (CUDA only).
- Hardware: container, 4 CPU cores, no GPU, torch 2.15.0+cpu, `device: cpu`. **CPU numbers only**, labelled
  `gpu=none`. The machine was shared with other jobs while measuring (p95 contains contention spikes); fp16-on-CUDA and
  VRAM paths are untested (no GPU). Thresholds are the untuned `configs/segment.yaml` defaults (box 0.30, text 0.25,
  mask 0.5).
- Commands: `uv sync --all-groups --all-extras`; `uv run pytest tests/test_segment.py`; `make segment-spike
  SEGMENT_ARGS=--fetch-coco` (= `uv run python bench/p3_segment_spike.py --fetch-coco`). The committed CSV merges two
  runs of the same day (the first run could not load SAM 2 because torchvision was missing; `gdino_sam2` was re-run
  after adding it to the extra).

## Results (table, CSV links)

CSV (Appendix C, accuracy in `notes`): `bench/results/segment_spike_cpu_2026-11-15.csv`; per image:
`bench/results/segment_spike_cpu_2026-11-15_frames.csv`; manifest (candidate status, COCO licences):
`bench/results/segment_spike_cpu_2026-11-15_manifest.json`. All CPU.

| Backend | Input | images | mean IoU | recall@0.5 | cat. acc. | latency median / p95 ms (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| gdino_sam2 | synthetic table_v1 (2 cams, 1280x720) | 2 | 0.594 | 0.60 | 0.17 | 7911 / 8628 |
| gdino_sam2 | COCO subset (vocabulary mismatch, see caveat) | 6 | 0.003 | 0.00 | n/a | 7283 / 27348 |
| mask2former | synthetic table_v1 | 2 | 0.000 | 0.00 | n/a | 933 / 4735 |
| mask2former | COCO subset | 6 | 0.510 | 0.53 | n/a | 10359 / 11993 |
| sam3 | - | - | - | - | - | skipped: gated |

Reading: on the flat synthetic render, `gdino_sam2` finds 3 of 5 objects (table, paper, cloth/block depending on camera)
but labels are often wrong (flat shading, no texture); Mask2Former finds nothing on it (no COCO classes, out of domain).
On real COCO photos Mask2Former segments its COCO classes well (recall 0.53 at IoU 0.5 for 2.7 objects/image on average,
many extra instances: precision was not scored) and shows the closed-set limit: table-top materials (paper sheet, cloth,
wooden block) have no COCO class. The CPU latencies (8-10 s per image at native resolution) are not representative of
an RTX 4060; the live budget (M3: <= 15 s to READY, includes depth and fusion) needs a GPU measurement.

## Scores (11.6, draft, 0-5 per criterion, weighted to 100)

| Criterion (weight) | gdino_sam2 | mask2former | sam3 |
| --- | --- | --- | --- |
| Accuracy on project data (35) | 2 (14): 0.59 mIoU on a flat render, no real project data | 1 (7): cannot see table-top classes | not scored |
| Latency and VRAM on RTX 4060 (30) | 3 (18), provisional: two models, 845 MB of weights on disk, GPU timing **pending user run** | 4 (24), provisional: 190 MB of weights on disk, GPU timing **pending user run** | not scored (VRAM unverified, 16 GB reported) |
| Robustness (20) | 2 (8): threshold-sensitive open vocabulary | 1 (4): closed set | not scored |
| Implementation risk (10) | 4 (8) | 5 (10) | 2 (4): gated access |
| License (5) | 5 (5): Apache-2.0 verified | 3 (3): presumed MIT, unverified | 1 (1): Meta SAM License, gated, not read |
| **Total** | **53** | **48** | not scored (not run) |

The scores rest on two flat renders and six off-vocabulary photos; they are drafts, not evidence that either model works
on the user's table. They will change with the labelled clips.

## Decision and fallback

Proposed (not final): primary **`gdino_sam2`** (open vocabulary is the only candidate that can in principle name paper,
cloth and wood objects; the category feeds the material prior; Apache-2.0, ungated), fallback always available: the P1R4
**geometric clustering** (`segmenters=None`). `mask2former` stays selectable (`configs/segment.yaml: backend`) but is not
recommended: closed COCO set. `sam3` is not evaluated: it needs the user to accept Meta's licence and provide an HF token
(escalation, PRD 0.5/11.6), and its VRAM must be confirmed on the 4060 before use. Hard limits: none triggered for
`gdino_sam2`.

## Consequences

- `perception` extra (torch + torchvision CPU wheels on Linux, CUDA 12.8 on Windows) is optional; `OracleSegmentationBackend`
  (MOCK, `# MOCK:`) covers every fast test; real-backend tests skip unless weights are cached.
- Additive helpers (no PRD 9 change): `result_from_masks`, `validate_segmentation_result`, `create_segmentation_backend`,
  `release()` on the real backends (P3R6), and the `perception` extra.
- Oracle with `erode_px: 1` has IoU 0.94-0.99 against ground truth for objects of 4000+ px (not >= 0.95 for every object;
  the exact mask, `erode_px: 0`, is bit-identical); the 249 px edge-on box of camera 0 drops to IoU 0.37.
- To finalise: run `make segment-spike` on the RTX 4060 (`device: cuda:0`) with labelled keyframes in `data/clips`, tune
  `box_threshold`/`text_threshold`, measure VRAM and load time, decide on SAM 3 with the user.
