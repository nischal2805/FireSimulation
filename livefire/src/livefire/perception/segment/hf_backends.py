"""Hugging Face ``transformers`` segmentation backends (optional ``perception`` extra: torch + transformers + pillow).

Candidates (ADR-0007): ``GDinoSam2Backend`` (Grounding DINO boxes from a text vocabulary + SAM 2.1 box prompts),
``Sam3Backend`` (one model, text-concept prompts; gated weights) and ``Mask2FormerBackend`` (closed-set COCO
instances). Weights are cached under ``data/models`` (``configs/segment.yaml: model_cache_dir``). On CUDA the networks
run in fp16; masks are converted to host NumPy bool. ``torch``/``transformers`` are imported lazily (inside the
constructors) so ``import livefire`` works without the extra. Keyframes only (no tracking). Input ``Frame``: rgb uint8
(H, W, 3) undistorted sRGB, host; output ``SegmentationResult`` host NumPy, caller-owned, per-camera ids.
Thread-safe (calls are serialised). ``release()`` drops the model references (P3R6).
"""

from __future__ import annotations

import gc

import numpy as np

from livefire.common.errors import SegmentationError
from livefire.perception.segment.base import (
    MetricsMixin,
    dedup_masks,
    default_device,
    load_segment_config,
    model_cache_dir,
    result_from_masks,
    scalar,
    segment_extra_available,
)
from livefire.perception.types import SegmentationResult

__all__ = ["GDinoSam2Backend", "Mask2FormerBackend", "Sam3Backend", "segment_extra_available"]


class _HFBase(MetricsMixin):
    """Shared plumbing: config, device, lock, ``warmup``, ``release`` and the timed/guarded ``segment``."""

    name = "hf"

    def __init__(self, cfg: dict | None, device: str | None) -> None:
        super().__init__()
        if not segment_extra_available():
            raise SegmentationError("the 'perception' extra is not installed (uv sync --extra perception)")
        import torch

        self._torch = torch
        self.cfg = cfg or load_segment_config()
        self.device = device or default_device()
        self._half = bool(self.cfg.get("fp16_on_cuda", True)) and str(self.device).startswith("cuda")
        self._cache = str(model_cache_dir(self.cfg))
        self.box_threshold = scalar(self.cfg["box_threshold"])
        self.text_threshold = scalar(self.cfg["text_threshold"])
        self.mask_threshold = scalar(self.cfg["mask_threshold"])
        self.min_area_px = int(self.cfg["min_mask_area_px"])
        self.max_instances = int(self.cfg["max_instances"])
        self.vocabulary = [str(v) for v in self.cfg["vocabulary"]]
        self.mask_dedup_iou = scalar(self.cfg["mask_dedup_iou"])
        self.local_files_only = bool(self.cfg.get("local_files_only", False))
        self._released = False
        self._models: list = []

    def _to_device(self, model):
        model.eval().to(self.device)
        return model.half() if self._half else model

    def warmup(self, shape: tuple[int, int]) -> None:
        """One segmentation of a synthetic (H, W) image (allocator and kernel warm-up); serialised with ``segment``."""
        self._check_alive()
        h, w = int(shape[0]), int(shape[1])
        img = np.zeros((h, w, 3), np.uint8)
        img[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (200, 180, 40)  # a bright block so a detector has something
        with self._lock:
            self._segment_np(img, 0, warm_box=(w * 0.25, h * 0.25, w * 0.75, h * 0.75))

    def _check_alive(self) -> None:
        if self._released:
            raise SegmentationError("backend released")

    def segment(self, frame) -> SegmentationResult:
        """Instance segmentation of ``frame.rgb`` (``SegmentationError`` on model failure)."""
        self._check_alive()
        try:
            with self._lock:
                self._check_alive()
                return self._timed(lambda: self._segment_np(np.ascontiguousarray(frame.rgb), frame.calib_version))
        except SegmentationError:
            raise
        except Exception as e:
            raise SegmentationError(f"{self.name} inference failed: {e}") from e

    def _segment_np(self, rgb: np.ndarray, calib_version: int, warm_box=None) -> SegmentationResult:
        """Segment one image; ``warm_box`` (x0, y0, x1, y1) is only used by ``warmup`` to force a second-stage call."""
        raise NotImplementedError

    def release(self) -> None:
        """Drop the model references and free the allocator cache (P3R6). Afterwards ``segment``/``warmup`` raise
        ``SegmentationError("backend released")``."""
        with self._lock:
            self._released = True
        self._models.clear()
        for k in [k for k, v in vars(self).items() if k.startswith(("_m_", "_proc"))]:
            setattr(self, k, None)
        gc.collect()
        if str(self.device).startswith("cuda"):
            self._torch.cuda.empty_cache()

    def _load(self, cls, hf_id: str, *, processor=False):
        try:
            kw = {"cache_dir": self._cache, "local_files_only": self.local_files_only}
            return cls.from_pretrained(hf_id, **kw)
        except Exception as e:
            raise SegmentationError(f"could not load {'processor' if processor else 'model'} {hf_id}: {e}") from e

    def _canon_category(self, label: str) -> str:
        """Vocabulary phrase for a detector label: exact; else the longest phrase contained in the label; else, for a
        truncated label (its tokens a contiguous token run of a phrase, e.g. "paper" or "wooden"), the shortest such
        phrase; else the label itself (lower case)."""
        lab = label.strip().lower()
        if lab in self.vocabulary:
            return lab
        hits = [v for v in self.vocabulary if v in lab]
        if hits:
            return max(hits, key=len)
        toks = lab.split()
        if toks:
            parts = [v for v in self.vocabulary if any(v.split()[i : i + len(toks)] == toks
                                                      for i in range(len(v.split()) - len(toks) + 1))]
            if parts:
                return min(parts, key=len)
        return lab


class GDinoSam2Backend(_HFBase):
    """Grounding DINO (open-vocabulary boxes from ``vocabulary``) + SAM 2.1 box-prompted masks. Category = vocabulary
    phrase; score = detector box score. Boxes with an empty label are dropped. One SAM 2 call per image (all boxes)."""

    name = "gdino_sam2"

    def __init__(self, cfg: dict | None = None, device: str | None = None) -> None:
        super().__init__(cfg, device)
        try:
            from transformers import AutoProcessor, GroundingDinoForObjectDetection, Sam2Model, Sam2Processor
        except ImportError as e:
            raise SegmentationError(f"transformers lacks the Grounding DINO / SAM 2 classes needed by gdino_sam2: {e}; "
                                    "update the 'perception' extra (uv sync --extra perception)") from e

        b = self.cfg["backends"]["gdino_sam2"]
        self.detector_hf_id, self.sam_hf_id = b["detector_hf_id"], b["sam_hf_id"]
        self._proc_det = self._load(AutoProcessor, self.detector_hf_id, processor=True)
        self._m_det = self._to_device(self._load(GroundingDinoForObjectDetection, self.detector_hf_id))
        self._proc_sam = self._load(Sam2Processor, self.sam_hf_id, processor=True)
        self._m_sam = self._to_device(self._load(Sam2Model, self.sam_hf_id))
        self._models = [self._m_det, self._m_sam]
        self._text = ". ".join(v.lower() for v in self.vocabulary) + "."
        self.max_boxes = int(self.cfg.get("max_boxes", self.max_instances))

    def _segment_np(self, rgb: np.ndarray, calib_version: int, warm_box=None) -> SegmentationResult:
        torch = self._torch
        from PIL import Image

        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        with torch.inference_mode():
            inp = self._proc_det(images=img, text=self._text, return_tensors="pt").to(self.device)
            if self._half:
                inp["pixel_values"] = inp["pixel_values"].half()
            out = self._m_det(**inp)
            det = self._proc_det.post_process_grounded_object_detection(
                out, inp["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
                target_sizes=[(h, w)],
            )[0]
            labels = det.get("text_labels", det["labels"])
            keep = [i for i, lab in enumerate(labels) if str(lab).strip()]
            keep = sorted(keep, key=lambda i: -float(det["scores"][i]))[: self.max_boxes]  # top-k by score
            if keep:
                boxes = det["boxes"][keep].float().cpu()
                scores = det["scores"][keep].float().cpu().numpy()
                cats = [self._canon_category(str(labels[i])) for i in keep]
            elif warm_box is not None:  # warm-up only: force the SAM 2 stage
                boxes = torch.tensor([list(warm_box)], dtype=torch.float32)
                scores, cats = np.array([0.5]), ["warmup"]
            else:
                return result_from_masks(np.zeros((0, h, w), bool), [], [], calib_version)
            p = min(max(self.mask_threshold, 1e-4), 1 - 1e-4)
            logit = float(np.log(p / (1.0 - p)))  # probability threshold -> SAM 2 mask-logit threshold
            sin = self._proc_sam(images=img, input_boxes=[boxes.tolist()], return_tensors="pt").to(self.device)
            if self._half:
                sin["pixel_values"] = sin["pixel_values"].half()
            sout = self._m_sam(**sin, multimask_output=False)
            masks = self._proc_sam.post_process_masks(sout.pred_masks.float().cpu(), sin["original_sizes"],
                                                      mask_threshold=logit, binarize=True)[0]
        m = masks.reshape(-1, h, w).numpy().astype(bool)  # (n_boxes, 1, H, W) -> (n_boxes, H, W)
        k = dedup_masks(m, scores, self.mask_dedup_iou)  # duplicate boxes give near-identical masks
        m, scores, cats = m[k], scores[k], [cats[i] for i in k]
        return result_from_masks(m, scores, cats, calib_version, min_area_px=self.min_area_px,
                                 max_instances=self.max_instances)


class Sam3Backend(_HFBase):
    """SAM 3 (``Sam3Model``) with one text-concept prompt per vocabulary phrase; category = the prompting phrase.
    Gated weights (Meta SAM License): needs an accepted licence and an HF token (``HF_TOKEN``)."""

    name = "sam3"

    def __init__(self, cfg: dict | None = None, device: str | None = None) -> None:
        super().__init__(cfg, device)
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as e:
            raise SegmentationError(f"transformers lacks Sam3Model/Sam3Processor: {e}; update the 'perception' extra "
                                    "(uv sync --extra perception)") from e

        self.hf_id = self.cfg["backends"]["sam3"]["hf_id"]
        self._proc = self._load(Sam3Processor, self.hf_id, processor=True)
        self._m_sam3 = self._to_device(self._load(Sam3Model, self.hf_id))
        self._models = [self._m_sam3]

    def _segment_np(self, rgb: np.ndarray, calib_version: int, warm_box=None) -> SegmentationResult:
        torch = self._torch
        from PIL import Image

        h, w = rgb.shape[:2]
        img = Image.fromarray(rgb)
        all_m, all_s, all_c = [], [], []
        with torch.inference_mode():
            for phrase in self.vocabulary:
                inp = self._proc(images=img, text=phrase, return_tensors="pt").to(self.device)
                if self._half:
                    inp["pixel_values"] = inp["pixel_values"].half()
                out = self._m_sam3(**inp)
                res = self._proc.post_process_instance_segmentation(
                    out, threshold=self.box_threshold, mask_threshold=self.mask_threshold, target_sizes=[(h, w)]
                )[0]
                n = len(res["scores"])
                if n:
                    all_m.append(res["masks"].cpu().numpy().astype(bool).reshape(n, h, w))
                    all_s += [float(s) for s in res["scores"].float().cpu()]
                    all_c += [phrase] * n
        masks = np.concatenate(all_m) if all_m else np.zeros((0, h, w), bool)
        if len(masks):  # the same object can answer several phrases
            k = dedup_masks(masks, all_s, self.mask_dedup_iou)
            masks, all_s, all_c = masks[k], [all_s[i] for i in k], [all_c[i] for i in k]
        return result_from_masks(masks, all_s, all_c, calib_version, min_area_px=self.min_area_px,
                                 max_instances=self.max_instances)


class Mask2FormerBackend(_HFBase):
    """Mask2Former Swin-T COCO instance segmentation (closed set, 80 things classes); instance score cut =
    ``backends.mask2former.score_threshold``. Category = COCO label name
    (``config.id2label``); score = instance score. Table-top objects without a COCO class are not found."""

    name = "mask2former"

    def __init__(self, cfg: dict | None = None, device: str | None = None) -> None:
        super().__init__(cfg, device)
        try:
            from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
        except ImportError as e:
            raise SegmentationError(f"transformers lacks Mask2Former: {e}; update the 'perception' extra "
                                    "(uv sync --extra perception)") from e

        b = self.cfg["backends"]["mask2former"]
        self.hf_id = b["hf_id"]
        self.score_threshold = scalar(b.get("score_threshold", self.cfg["box_threshold"]))
        self._proc = self._load(AutoImageProcessor, self.hf_id, processor=True)
        self._m_m2f = self._to_device(self._load(Mask2FormerForUniversalSegmentation, self.hf_id))
        self._models = [self._m_m2f]
        self._id2label = {int(k): str(v) for k, v in self._m_m2f.config.id2label.items()}

    def _segment_np(self, rgb: np.ndarray, calib_version: int, warm_box=None) -> SegmentationResult:
        torch = self._torch
        from PIL import Image

        h, w = rgb.shape[:2]
        with torch.inference_mode():
            inp = self._proc(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
            if self._half:
                inp["pixel_values"] = inp["pixel_values"].half()
            out = self._m_m2f(**inp)
            res = self._proc.post_process_instance_segmentation(
                out, threshold=self.score_threshold, mask_threshold=self.mask_threshold, target_sizes=[(h, w)],
                return_binary_maps=True,
            )[0]
        info = res["segments_info"]
        seg = res["segmentation"]
        if not info or seg.ndim != 3:
            return result_from_masks(np.zeros((0, h, w), bool), [], [], calib_version)
        masks = seg.cpu().numpy().astype(bool)
        scores = [float(s["score"]) for s in info]
        cats = [self._id2label.get(int(s["label_id"]), str(s["label_id"])) for s in info]
        return result_from_masks(masks, scores, cats, calib_version, min_area_px=self.min_area_px,
                                 max_instances=self.max_instances)
