# P3 implementation plan (automatic scene understanding: segmentation, objects, materials, modes)

Spec: `docs/PRD.md` (authority; P3 = section 12 "P3: automatic scene understanding", 7.3, 6.1 ObjectRegistry,
5.2 modes, 5.4 VRAM table, M3 and M8 in 11.1, 11.3 datasets, 11.6 decision rule, 0.2 research rules, Appendix B
ADR template). Rules: `CLAUDE.md`. Context: `docs/HANDOVER.md` (P1R4 geometric clustering, default material `paper`
flagged for review; cloth merges into the table without segmentation), `docs/tasks/P1-plan.md` (P1R4-P1R6).
**Gate 3:** the scene is acquired and labelled with minimal manual intervention within M3 (<= 15 s to READY on the
RTX 4060); M8 measured (macro-F1, confusion matrix, calibration on an object-disjoint, held-out-scene split).

## Global Constraints (bind every task)

- Everything in the P0 and P1 plans' Global Constraints still holds (`docs/tasks/P0-plan.md`, `docs/tasks/P1-plan.md`):
  Python 3.11 + uv, geometry contract (PRD 4), YAML tunables in `configs/` (`{value, unit, range, status}` for
  physical/artistic numbers), seeded randomness, mocks marked `# MOCK:` in code and labelled in output, ruff clean
  (`uv run ruff check src tests bench eval`), docstrings with shapes/units/frame/device/ownership.
- **No GPU, no camera, no display here.** Everything must be verifiable headless with the synthetic scene
  (`table_v1`, `scene/synthetic.py::render_view` gives exact per-pixel `obj_id`), recorded clips (`VideoFileSource`)
  and oracle backends. Run fast tests with `uv run pytest -m "not slow" -q` from `livefire/`; never run the full suite;
  slow tests by node id only, long jobs in the background.
- **ML is optional.** New optional extra `perception = ["torch", "transformers", "pillow"]` in `pyproject.toml`
  (same uv torch sources as the `depth` extra: CPU wheels on Linux, cu128 on Windows). Core code and every fast test
  must import and run without torch: import model libraries lazily inside backend constructors; availability helper
  like `perception/depth/hf_backends.py::depth_extra_available`. **Tests never require downloaded weights:** contract
  tests use oracle/fake backends (`# MOCK:`); real-model tests are `@pytest.mark.skipif(weights not cached)` and
  touch one small image. Real-model accuracy/latency comes only from bench scripts writing CSVs. Weights cache under
  `data/models/` (gitignored), path from config. Hugging Face and PyPI are reachable from this container (checked
  2026-11-15), so CPU spikes may download ungated weights; gated weights need the user's HF token and license
  acceptance (escalate, PRD 0.5).
- User machine: Windows 11 + RTX 4060 (8 GB). pathlib only, no POSIX-only calls. Every new command goes in both the
  Makefile and the README "Windows (no make)" table; do not edit the Makefile `eval` stub.
- Interfaces: `SegmentationBackend`, `MaterialBackend` (`common/interfaces.py`), `SegmentationResult`
  (`perception/types.py`), `MaterialPrediction` and `MATERIAL_CLASSES` (`perception/material/types.py`),
  `ObjectRegistry`/`WorldObject` (`scene/snapshot.py`), errors `SegmentationError`/`MaterialError`
  (`common/errors.py`) are PRD 9 contracts: implement them, do not change them. Additive helpers (new protocols,
  new optional kwargs with defaults that keep today's behaviour) are allowed and named in the task's ADR.
- Backends report `latency_ms`, `peak_vram_allocated_mb`, `peak_vram_reserved_mb` (reuse
  `perception/depth/base.py::MetricsMixin`; move it to a shared `perception/base.py` only if you update the depth
  imports in the same commit and keep `perception.depth.base.MetricsMixin` importable).
- Per task: copy the brief to `docs/tasks/<P3-ID>-<slug>.md`; commit `<P3-ID>: <title>` with trailers
  `Co-Authored-By: <your model name> <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01FvsiHx5LCL7MYc9rf6PQvv`; `git add` only the task's own files; do not
  push; do not edit `docs/STATUS.md` (controller-owned). Numbers only from committed CSVs; CPU numbers labelled CPU.
  Do not silently change PRD contracts: ADR + PRD change-log entry (only Task 6 edits the PRD change log).
- Outputs: bench CSVs `bench/results/<name>_cpu_<YYYY-MM-DD>.csv` (Appendix C schema via `livefire.common.bench_csv`
  for timing/VRAM rows); eval CSVs `eval/results/<name>_<YYYY-MM-DD>.csv`; P3 eval scripts `eval/p3_*.py`.
- `tests/test_acquire.py` must stay green unchanged, including its strict `xfail`
  `test_cloth_separated_from_table` (that test covers the geometric path; P3 adds a separate passing test for the
  segmentation path).

## Controller rulings (made at plan time)

- P3R1 **Candidates (time-boxed research done at plan time, PRD 0.2; HF API checked 2026-11-15).** Segmentation
  (keyframes only, no tracking, PRD 3/7.3):
  1. `gdino_sam2`: Grounding DINO tiny `IDEA-Research/grounding-dino-tiny` (open-vocabulary boxes from a text
     vocabulary; Apache-2.0, ungated) + SAM 2.1 `facebook/sam2.1-hiera-tiny` box prompts (Apache-2.0, ungated; `-small`
     as a config option). Gives a category label per instance (feeds the material prior). Primary candidate.
  2. `sam3`: `facebook/sam3` (transformers `Sam3Model`/`Sam3Processor`, text-concept prompts, one model). Gated
     (manual approval) under Meta's SAM License (`license:other`), VRAM unverified (a third-party note says 16 GB is
     recommended): license and VRAM are possible hard limits (PRD 11.6); runs only when the user has accepted the
     license and set an HF token. Ask the user before relying on it.
  3. `mask2former`: `facebook/mask2former-swin-tiny-coco-instance` (closed-set COCO instances; HF tag
     `license:other`, verify the weight license in the ADR).
  Fallback always available: the P1R4 geometric clustering (`segmenters=None`).
  Materials (prior, not identification):
  1. `category_prior`: lookup table category -> class distribution (`configs/material.yaml`), no model.
  2. `siglip2`: zero-shot `google/siglip2-base-patch16-224` (Apache-2.0, ungated) on the masked object crop with
     per-class prompt ensembles; fused with the category prior (P3R4).
  3. `dinov2_probe`: `facebook/dinov2-small` (Apache-2.0) frozen features + a logistic-regression probe trained on the
     user's object-disjoint training split (needs the user dataset; pending user run).
  Weights' licenses are re-verified from the model cards/LICENSE files in the ADRs; transformers 5.x in `uv.lock`
  provides `GroundingDinoForObjectDetection`, `Sam2Model`, `Sam3Model`, `Siglip2Model`, `Dinov2Model`,
  `Mask2FormerForUniversalSegmentation` (verified importable).
- P3R2 **Keyframes.** Per camera, segment `segment.keyframes` frames (default 1: the frame nearest the middle of the
  acquisition window) after depth accumulation; instance ids are per-camera observation ids (PRD 6.1), never world ids.
- P3R3 **Cross-view association via the fused 3D proxy.** Each surfel is projected into each camera
  (`common/geometry.py::project`); it is visible if its Z agrees with the accumulated depth within
  `assoc.visibility_tol_m`; it takes that camera's instance id at the projected pixel (3x3 majority). Per camera,
  instances become surfel sets; world objects = connected components of the bipartite graph linking instances of
  different cameras whose surfel sets have IoU >= `assoc.min_iou` (union-find). If an instance links to two
  candidates, the higher IoU wins and mean-colour (CIELAB of the mask pixels) distance breaks ties within
  `assoc.iou_tie`. A surfel's final object = the world object of its majority instance over visible cameras; surfels
  with no instance keep the geometric label (P1R4); instance-labelled surfels override the table label (fixes the
  low-profile cloth). Table = obj 0 always; world object ids are renumbered 1.. by area.
  `WorldObject.observations = {cam_id: instance_id}`.
- P3R4 **MaterialPrediction rules.** Fusion is log-linear: `p ∝ p_prior^w * p_app^(1-w)` (`material.fusion_weight`,
  default 0.5; prior uniform when no category hint), renormalised to sum 1 (float32). `confidence` = top-1 probability;
  `source` = `"category_prior"` if no appearance model ran, else `"model"`; `requires_review = confidence <
  review.min_confidence (0.6) or (top1 - top2) < review.min_margin (0.15) or the top class is flammable and its
  confidence < review.min_conf_flammable (0.75)` (wrong "non-flammable" on a flammable object is the costly error).
  Per-object prediction = confidence-weighted mean over the object's keyframe observations in all cameras. The table
  stays `other_nonflammable`, `requires_review=False`. With no material backend the P1 default (`paper`, review)
  applies.
- P3R5 **Overrides survive re-acquisition (reconciles HANDOVER section 6 with PRD 7.3).** The live `SceneSnapshot`
  arrays stay frozen; the controller keeps recording `override` replay events. In addition each override is written
  to an `OverrideStore` (`<run dir>/overrides.yaml`, also loadable by path) with the object's 3D descriptor:
  centroid, AABB and the set of occupied 2 cm voxels of its surfels. During a re-acquisition, `acquire_with_report`
  matches new objects to stored overrides by voxel IoU >= `overrides.min_iou` (0.5; ties: nearest centroid) and builds
  the new snapshot with the override already applied (element `material_id` and the `WorldObject.material` =
  `manual_override`, confidence 1, no review). Unmatched stored overrides are listed in the acquisition report and the
  HUD; they are never applied to a different object.
- P3R6 **Modes and VRAM (PRD 5.2, 5.4, P1R6).** Perception models are owned by a `PerceptionModels` holder; models
  load on entering ACQUIRING and are released on READY (drop references, `gc.collect()`, `torch.cuda.empty_cache()`
  on CUDA), and the release is verified (CUDA: `torch.cuda.memory_allocated` back within 64 MB of the pre-load value;
  CPU: weakrefs to the model modules are dead). Release capability is an additive `Releasable` protocol
  (`release() -> None`), not a change to the PRD 9 protocols.
- P3R7 **Live mode dependency.** `app/live.py` (P1-006) is written concurrently. P3 Tasks 1-3 only touch the
  acquisition path (`scene/acquire.py`) and are independent of it. Task 5's live wiring starts only after P1-006 is
  committed (controller checks); if it is missing, Task 5 delivers everything else and lists the live wiring as open.
- P3R8 **M8 data.** No public material dataset with a clearly permissive license is assumed. The evaluation harness is
  verified on a deterministic synthetic fixture (`# MOCK:` labelled procedural patches); real M8 numbers come from the
  user's dataset captured with the Task 6 tool (PRD 11.3 protocol) and are "pending user run" until it exists. A
  public dataset may be added only if its license explicitly allows this use (record it in the manifest), time-boxed.

---

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

## Task 2: P3-002 Object registry with cross-view association, integrated into acquisition

**Files:** `src/livefire/fusion/associate.py` (new), `src/livefire/scene/acquire.py` (optional kwargs),
`configs/acquisition.yaml` (`segment`/`assoc` blocks), `tests/test_associate.py` (new),
`tests/test_acquire_segmented.py` (new).

- `configs/acquisition.yaml` additions under `acquisition`: `assoc: {visibility_tol_m: {value: 0.02, unit: "m", range:
  [0.002, 0.1], status: artistic}, min_iou: {value: 0.3, unit: "-", range: [0.05, 0.95], status: artistic},
  iou_tie: 0.05, min_instance_surfels: 10}`.
- `associate.py`: `label_surfels(pos, cals, accumulated, segmentations, cfg) -> dict[int, NDArray[int32]]`
  (per-camera instance id per surfel, -1 when not visible / background, P3R3 visibility rule) and
  `associate_objects(per_cam_labels, geometric_labels, rgb_stats, cfg) -> AssociationResult` (`labels` (N,) world obj
  id, `observations: dict[obj, dict[cam, inst]]`, `categories: dict[obj, list[str]]`, `report`), implementing the
  union-find merge, IoU tie-break by CIELAB mean colour, table override rule and renumbering of P3R3.
- `acquire_with_report(..., segmenters: Mapping[int, SegmentationBackend] | SegmentationBackend | None = None)`:
  `None` keeps the P1 path exactly. Before editing, record a golden digest (sha256 of `obj_id`, `pos`, `material_id`
  bytes) of the existing two-camera synthetic acquisition fixture into the new test file as a constant, with the
  commit it was taken at; the test asserts `segmenters=None` still produces it. With segmenters: after accumulation,
  segment the P3R2 keyframe(s) per camera (stage `segment`), extract and cluster as today, then `label_surfels` +
  `associate_objects` replace the labels before `bin_surfels` (stage `associate`); categories go into
  `WorldObject.name`/`MaterialPrediction.category_hint`. This task adds only the `segmenters` kwarg;
  `material_backend` (Task 3) and `overrides` (Task 4) are added by those tasks, same pattern (default None = today).
- Report additions: per world object `observations`, `categories`, `source: segmentation|geometric`, unlabelled
  surfel fraction; timings `segment`, `associate`.
- Tests: `associate` unit tests on hand-built two-camera surfel sets: permuted ids merge correctly; an instance seen
  only in one camera becomes its own object; two instances overlapping one instance -> higher IoU wins, colour breaks a
  tie; invisible surfels never take an id. Integration (`tests/test_acquire_segmented.py`, same fixture style and noise
  as `tests/test_acquire.py`: Oracle depth + sigma 1 % + 3 % per-view scale; `OracleSegmentationBackend` with permuted
  ids): 4 objects + table; **cloth separated from the table** with > 90 % membership; every object's observations map
  to the permuted per-camera ids; single-camera acquisition with segmentation works; `segmenters=None` gives the same
  golden digest recorded before the change; timings include `segment` and `associate`.

**Acceptance:** fast tests green (slow ones by node id if any); `tests/test_acquire.py` unchanged and green.

## Task 3: P3-003 Material recognition backends and MaterialPrediction fusion in acquisition

**Files:** `src/livefire/perception/material/{prior.py,oracle.py,hf_backends.py,probe.py,fuse.py,factory.py}` (new),
`configs/material.yaml` (new), `src/livefire/scene/acquire.py` (wire `material_backend`), `bench/p3_material_spike.py`
(new), `tests/test_material_backends.py` (new), Makefile `material-spike` + README row.

- `configs/material.yaml`: `backend: prior|siglip2|siglip2_prior|dinov2_probe|oracle` (default `siglip2_prior`),
  `model_cache_dir`, `fp16_on_cuda`, `category_priors: {paper sheet: {paper: 0.8, cloth: 0.1, plastic: 0.1}, cloth:
  {cloth: 0.85, paper: 0.05, plastic: 0.1}, towel: ..., wooden block: {wood: 0.85, plastic: 0.1,
  other_nonflammable: 0.05}, plastic bottle: ..., metal cup: ..., ceramic mug: ..., drinking glass: ..., book: ...,
  cardboard box: ..., table: {other_nonflammable: 1.0}}` (each row sums to 1; loader validates), `prompts: {paper:
  ["a photo of paper", "a sheet of paper"], ... one list per MATERIAL_CLASSES entry}`, `fusion_weight {0.5, "-",
  [0, 1], artistic}`, `review: {min_confidence 0.6, min_margin 0.15, min_conf_flammable 0.75}` (all `{value, unit,
  range, status: artistic}`), `crop: {pad_frac: 0.1, mask_background: gray}`, `probe: {path: data/models/
  material_probe.npz, C: 1.0}`, `oracle: {seed, confidence: 0.9, flip_prob: 0.0}`.
- `prior.py`: `CategoryPriorBackend` (`MaterialBackend`; `source="category_prior"`; unknown category -> uniform over
  classes with `requires_review=True`). `oracle.py`: `OracleMaterialBackend` (`# MOCK:` reads the synthetic ground
  truth for the mask's majority obj_id; seeded flips for tests). `hf_backends.py`: `SigLIP2MaterialBackend` (masked,
  padded crop; prompt-ensemble text embeddings cached at warmup; softmax over class logits with the model's logit
  scale), `release()`. `probe.py`: `DinoV2ProbeBackend` (frozen CLS features + NumPy/scipy logistic regression saved as
  `.npz`; `fit(features, labels)` and `predict`), plus `extract_features(...)`. `fuse.py`: `fuse(prior, appearance,
  w) -> np.ndarray`, `review_flag(p, cfg) -> bool`, `make_prediction(...) -> MaterialPrediction`,
  `aggregate(predictions, weights) -> MaterialPrediction` (P3R4).
- `acquire_with_report(material_backend=...)`: for every non-table world object and each camera that observes it,
  build the mask from the keyframe instance (`observations`), predict, aggregate per P3R4, write the snapshot's
  `material_id`/`material_conf` and `WorldObject.material`; stage `materials`; report per object
  `class_probabilities`, `confidence`, `source`, `requires_review`. Without a material backend: P1 default unchanged.
- `bench/p3_material_spike.py [--dataset manifest.yaml]`: runs `prior`, `siglip2`, `siglip2_prior` (and `dinov2_probe`
  when a trained probe exists) on a dataset manifest (Task 6 format) or, when none exists, on the synthetic fixture
  only (labelled `# MOCK: synthetic fixture`); latency, VRAM, top-1, macro-F1 -> `bench/results/
  material_spike_cpu_<date>.csv`. ADR-0008 is written in Task 6 when M8 numbers exist.
- Tests (fast, no weights): prior table rows sum to 1 and every class name is valid (loader rejects a bad row);
  fusion math (uniform prior returns the appearance distribution; `w=1` returns the prior; output float32 sums to 1);
  review rule truth table (high-confidence non-flammable -> no review; 0.7 paper -> review via `min_conf_flammable`;
  small margin -> review); aggregation weights; `MaterialPrediction` contract holds for every backend output;
  acquisition with `OracleSegmentationBackend` + `OracleMaterialBackend` (flip 0) gives every object its ground-truth
  material and `source="model"`; with `CategoryPriorBackend` the cloth gets `cloth` from its category; with a flipped
  oracle the wrong object is flagged `requires_review` when its confidence is below thresholds; probe `fit/predict`
  round trip on a separable synthetic feature set (accuracy 1.0); SigLIP2 real test skip-if-unavailable, one crop,
  contract only.

**Acceptance:** fast tests green; spike CSV committed (synthetic fixture rows labelled MOCK; real data rows only if a
dataset exists).

## Task 4: P3-004 Review UI and overrides surviving re-acquisition

**Files:** `src/livefire/scene/overrides.py` (new `OverrideStore`), `src/livefire/scene/acquire.py` (wire
`overrides`), `src/livefire/app/interactive.py` (review features), `src/livefire/app/window.py` (draw review markers;
thin), `tests/test_review_overrides.py` (new), README key table rows.

- Review UI (pure controller, tested headless): `InteractiveController.review_queue() -> list[int]` (objects with
  `requires_review` and no override, sorted by confidence ascending); key `n` selects the next object in the queue
  (HUD "review k/n: object <id> <category> <material> <conf>"); key `a` accepts the selected object's predicted
  material (records an `override` event with the predicted material, so it becomes `manual_override` and leaves the
  queue); keys `1..8` keep their P1 meaning (override). `hud_lines()` adds "N objects need review" while the queue is
  non-empty. `review_markers(cam) -> list[(u, v, text, needs_review)]` projects each object's surfel centroid into the
  view (skips behind-camera/off-image) for the window to draw (red for review, green otherwise). Keys must not clash
  with existing ones (`space + - o v r s q 1-8`).
- `OverrideStore` (P3R5): `add(obj_id, material, snapshot)`, `save(path)`, `load(path)`, `match(snapshot) ->
  dict[new_obj_id, material]` and `unmatched`. YAML with descriptors (centroid, AABB, voxel keys as int triples at
  `overrides.voxel_m 0.02`, `overrides.min_iou 0.5` in `configs/acquisition.yaml`), pathlib only.
  `InteractiveController` writes to the store on every override (optional `override_store` constructor kwarg,
  default None = P1 behaviour). `acquire_with_report(..., overrides: OverrideStore | None = None)` applies matched
  overrides before the snapshot is frozen and reports unmatched ones; `finish_reacquire` keeps the controller
  `overrides` dict in sync with snapshot objects whose `source == "manual_override"`.
- Tests (fast): review queue order, `n`/`a` behaviour and recorded events, HUD line, markers project to the analytic
  centroid pixel within 2 px on `table_v1`; `OverrideStore` save/load round trip; re-acquisition of `table_v1` with
  oracle backends and a seeded small pose jitter of the scene objects (<= 1 cm) re-applies an override to the matched
  object only; a deleted object's override is reported unmatched and not applied to its neighbour; the overridden
  object in the new snapshot has `source == "manual_override"` and its elements' `material_id`; replay of a session
  containing an `a` accept reproduces the digest; `override_store=None` leaves `tests/test_interactive.py` green.

**Acceptance:** fast tests green.

## Task 5: P3-005 Acquisition/simulation modes, VRAM release, VRAM table and live-mode wiring

**Files:** `src/livefire/app/modes.py` (new: `PerceptionModels`, `ModeManager`, `Releasable`), `bench/p3_vram_table.py`
(new), `src/livefire/app/live.py` (wiring only, P3R7), `tests/test_modes.py` (new), Makefile `vram-table` + README
rows.

- Modes (P3R6): `PerceptionModels(depth_factory, seg_factory, material_factory)` with `load()`, `release()`,
  `loaded`, and `memory_report() -> dict` (per backend peak alloc/reserved MB; 0 on CPU); `ModeManager(lifecycle,
  models)` with `enter_acquisition()` (load, ACQUIRING), `enter_simulation()` (release + verify, READY), recording
  timings and memory before/after in a dict for the HUD and logs. Release time is part of M3 (Task 6).
- Live wiring (only if P1-006 is committed, P3R7): `live` CLI gains `--segment <backend>` and `--material <backend>`
  (default from configs; `none` = geometric/P1 default); acquisition and `r` re-acquire go through `ModeManager` and
  pass the session `OverrideStore`; HUD shows the review count. Test with the P1-006 synthetic-clip harness and
  oracle backends (`tests/test_modes.py`, do not edit `tests/test_live_pipeline.py`).
- `bench/p3_vram_table.py`: PRD 5.4 table. Fixed workload: one 1280x720 keyframe per backend (depth, each segmentation
  candidate, each material candidate) loaded alone (peak allocated/reserved via `torch.cuda` counters, reset per
  backend), the sim at G=96 with `table_v1` (Warp `wp.get_mempool_used_mem_high`), the renderer for two 720p views, and
  the integrated pipeline in acquisition mode and after `enter_simulation()` (framework counters + `nvidia-smi
  --query-gpu=memory.used` when available) -> `bench/results/vram_<gpu|cpu>_<date>.csv` (Appendix C columns). On this
  CPU-only container the script runs end to end with `gpu=none` and VRAM columns 0, notes "pending user run".
- Tests (fast): `ModeManager` with fake `# MOCK:` backends: load on acquisition, release on READY (weakrefs dead),
  reload on re-acquire, lifecycle transitions legal (`app/lifecycle.py`); with torch available, a tiny CPU
  `torch.nn.Module` backend is garbage-collected after `release()`; `bench/p3_vram_table.py --quick` writes a CSV;
  live wiring test if applicable.

**Acceptance:** fast tests green; VRAM CSV (CPU, pending GPU) committed; live wiring done or listed as open.

## Task 6: P3-006 M8 evaluation harness, dataset protocol, M3 timing, ADR-0008, runbook and Phase 3 report

**Files:** `src/livefire/perception/material/dataset.py` (new: manifest schema, loader, split), `eval/p3_materials.py`
(new: M8), `eval/p3_make_fixture.py` (new: synthetic fixture), `eval/p3_acquisition_time.py` (new: M3),
`src/livefire/app/material_capture_cli.py` (new: dataset capture tool) + wiring in `app/__main__.py`,
`docs/decisions/ADR-0008-material-recognition.md` (new), `docs/RUNBOOK_P3.md` (new), `docs/reports/phase-3.md` (new),
`docs/PRD.md` (change log v2.4 only), `tests/test_eval_p3.py` (new), Makefile `eval-materials`, `eval-acq-time`,
`capture-materials` + README rows.

- Dataset protocol (PRD 11.3, 7.3): `data/materials/manifest.yaml` (`data/` gitignored; the schema and an example are
  documented in `docs/RUNBOOK_P3.md`): `dataset_id, license, capture: {date, cameras, lighting, notes}, objects:
  [{object_id, label (MATERIAL_CLASSES), category, scene_ids, photos: [{path, sha256, scene_id, lighting, mask_path?}]}]`.
  `dataset.py`: `load_manifest(path)` (validates labels, hashes, unique object ids), `split(manifest, heldout_scene,
  seed, val_frac) -> {train, val, test}` **by object identity** (no object in two splits) with the held-out scene used
  only for test (objects appearing in the held-out scene go to test); raises if any object leaks.
- Capture tool `python -m livefire.app capture-materials --cam N --out data/materials --scene S --lighting L`: live
  view; the user types the object id and label, presses `c` to save frames (box-drawn crop optional), writes photos
  and updates the manifest with sha256. Headless-testable core (`add_photo(...)`) with the window layer thin.
- `eval/p3_make_fixture.py`: deterministic `# MOCK:` fixture (procedural textures per class, 8 classes x 12 objects x
  4 photos, 2 scenes/lightings) with a manifest, for harness tests only (its numbers are never reported as M8).
- `eval/p3_materials.py --manifest PATH --backend NAME [--heldout-scene S]`: trains the probe on train (if the backend
  needs it), fits an optional temperature on val, predicts on test; writes `eval/results/m8_<backend>_<date>.csv`
  (per photo: object_id, label, predicted, confidence, probabilities), `m8_summary_<date>.csv` (top-1, macro-F1,
  per-class F1, ECE with 10 bins, Brier score, review rate, error rate among non-reviewed predictions),
  `m8_confusion_<backend>_<date>.csv`, and a reliability-diagram PNG from the CSV only (`bench/plots.py` style).
  Target 85 % top-1 (PRD M8).
- `eval/p3_acquisition_time.py [--clips DIR] [--backends ...]`: M3 from start to READY = capture window + depth +
  accumulation + TSDF + extraction + segmentation + association + materials + snapshot (+ `enter_simulation()`
  release), per stage median of >= 3 runs on recorded clips (synthetic clips with oracle backends here, CPU-labelled;
  user clips with real backends on the GPU) -> `eval/results/m3_acquisition_<date>.csv`; flags whether <= 15 s.
- ADR-0008 (materials, Appendix B): candidates P3R1, method (manifest, split, metrics), results from the CSVs that
  exist (fixture rows marked MOCK and excluded from scoring), PRD 11.6 scores, decision **proposed** until the user's
  dataset M8 numbers exist; fallback `category_prior`. PRD change log `v2.4 (P3): ADR-0007 segmentation, ADR-0008
  materials, additive acquire kwargs (segmenters, material_backend, overrides), OverrideStore, Releasable protocol,
  perception extra.`
- `docs/RUNBOOK_P3.md`: install the extra; (optional) accept the SAM 3 license and set `HF_TOKEN`; run the spikes on
  the GPU; record clips; capture the material dataset (>= 100 objects target, several photos each, 2+ scenes and
  lighting setups, physical-object-disjoint); run `eval-materials`, `eval-acq-time`, `vram-table`; which files to send
  back. Windows commands for each.
- `docs/reports/phase-3.md`: Gate 3 table (scene acquired and labelled with minimal intervention: number of manual
  actions in the synthetic oracle run and in the user run; M3; M8) quoting committed CSVs only; recommended outcome.
  **Pending user run** list, one command each: segmentation and material spikes on the 4060 (ADR-0007/0008 scores),
  VRAM table on the 4060, M3 on real clips with real backends, M8 on the user dataset, SAM 3 if the license is
  accepted, live-mode review/override flow on real cameras.
- Tests (`tests/test_eval_p3.py`, fast): manifest validation rejects a wrong hash, unknown label and duplicate object
  id; split puts no object in two splits and keeps the held-out scene in test only; metrics on hand-made predictions
  (macro-F1, confusion matrix, ECE) equal hand-computed values; `eval/p3_materials.py` on a 2-class tiny fixture with
  the oracle backend writes all CSVs; `eval/p3_acquisition_time.py --quick` writes a CSV with all stage columns;
  capture-tool core appends a photo with the correct sha256.

**Acceptance:** fast tests green; harness proven on the fixture; M3 (CPU, synthetic) CSV committed; ADR-0008 proposed;
report lists every unmeasured item as pending user run.
