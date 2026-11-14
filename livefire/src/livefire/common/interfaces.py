"""Interface contracts (PRD 9): typed Protocols between subsystems.

Common conventions (apply to every Protocol below unless overridden):
  * Frame: world right-handed, metres, +Z up; camera OpenCV (+X right, +Y down, +Z forward);
    transforms ``T_<to>_from_<from>``; depth is Z-depth in metres, invalid = 0.0 (never NaN).
  * calib_version: every input/output tied to a calibration states it; a mismatch with the
    snapshot or frame is an error (``CalibVersionError``), never silently accepted.
  * Backends expose ``latency_ms`` (last/rolling call, ms), ``peak_vram_allocated_mb`` and
    ``peak_vram_reserved_mb`` (0.0 on CPU) through the shared timer.
  * "Device" means the device selected by ``configs/app.yaml: device`` (``cpu`` or ``cuda:N``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from livefire.capture.types import Frame
from livefire.common.geometry import AABB, CameraCalibration
from livefire.perception.material.types import MaterialPrediction
from livefire.perception.types import DepthResult, SegmentationResult
from livefire.scene.snapshot import SceneSnapshot
from livefire.sim.state import FireStateHost

# CameraCalibration is the Task 2 frozen dataclass (re-exported; no separate Protocol).
# Contract: K (3,3) px for stored resolution (undistorted), dist (5,), T_world_from_camera
# (4,4) camera->world float64, host memory, immutable; calib_version identifies the solve.
__all__ = [
    "BenchmarkRecorder", "CameraCalibration", "CameraSource", "DepthBackend", "FireStateView",
    "MaterialBackend", "RendererBackend", "ReplayController", "SceneReconstructor",
    "SegmentationBackend", "SimulationBackend",
]  # fmt: skip


class _BackendMetrics(Protocol):
    @property
    def latency_ms(self) -> float:
        """Latency of the last call in ms (shared timer; CUDA-synchronised on GPU)."""
        ...

    @property
    def peak_vram_allocated_mb(self) -> float:
        """Peak allocated device memory in MB (0.0 on CPU)."""
        ...

    @property
    def peak_vram_reserved_mb(self) -> float:
        """Peak reserved device memory in MB (0.0 on CPU)."""
        ...


@runtime_checkable
class CameraSource(Protocol):
    """Source of undistorted frames for one camera.

    Output: ``Frame`` with rgb uint8 (H,W,3) undistorted sRGB, host memory, C-contiguous;
    ``seq`` increases per camera; ``calib_version`` is the calibration used to undistort.
    Newest-frame-wins: ``latest()`` returns the newest frame (or None before the first) and
    never blocks or queues; older frames are dropped. Thread-safe; the source owns its
    buffers and hands each Frame over read-only (consumers copy before mutating). No
    cross-camera sync guarantee beyond ``t_arrive_ns`` timestamps. Raises ``RuntimeError`` on
    device failure in ``start``; ``latest`` returns None rather than raising."""

    @property
    def cam_id(self) -> int: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def latest(self) -> Frame | None: ...


@runtime_checkable
class DepthBackend(_BackendMetrics, Protocol):
    """Input: undistorted rgb uint8[H,W,3] (host or device; the backend declares and copies).
    Output ``DepthResult``: z_depth_m f32[H,W] contiguous, Z-depth metres along camera +Z, on
    device when the backend is GPU (host copy via ``.cpu()`` semantics is the caller's job);
    valid bool[H,W]; conf f32[H,W] in [0,1] (state how calibrated in the implementation);
    invalid pixels = 0.0. Thread-safe. fp16 allowed internally; output f32. Synchronised:
    results are ready on return. Ownership: result arrays belong to the caller. Tied to the
    calib_version of its input frame (copied to the result). Raises DepthError on model
    failure."""

    def warmup(self, shape: tuple[int, int]) -> None: ...

    def infer(self, frame: Frame) -> DepthResult: ...


@runtime_checkable
class SegmentationBackend(_BackendMetrics, Protocol):
    """Input: Frame (rgb uint8[H,W,3] undistorted sRGB, host or device as declared).
    Output ``SegmentationResult``: instance_ids int32[H,W] (-1 background), categories and
    scores per instance id, host memory, C-contiguous; ids are per-camera observations, not
    world obj_ids. Thread-safe; synchronised on return; caller owns the result. Image space
    only (no world frame). Tied to the input frame's calib_version (copied to the result).
    Raises SegmentationError on model failure."""

    def warmup(self, shape: tuple[int, int]) -> None: ...

    def segment(self, frame: Frame) -> SegmentationResult: ...


@runtime_checkable
class MaterialBackend(_BackendMetrics, Protocol):
    """Input: Frame plus a boolean mask bool[H,W] (host) selecting one object's pixels, and an
    optional category hint. Output ``MaterialPrediction``: class_probabilities f32[8] in the
    fixed class order (paper, cloth, wood, plastic, metal, glass, ceramic,
    other_nonflammable) summing to 1, source, confidence, requires_review. Host memory,
    caller-owned, synchronised on return, thread-safe. A prior for plausible simulation, not
    physical identification. Tied to the frame's calib_version (the mask is in that frame's
    pixel space). Raises MaterialError on model failure."""

    def predict(
        self, frame: Frame, mask: np.ndarray, category_hint: str | None = None
    ) -> MaterialPrediction: ...


@runtime_checkable
class SceneReconstructor(_BackendMetrics, Protocol):
    """Fuses per-camera depth (+ segmentation + materials) into an immutable ``SceneSnapshot``.
    Inputs: frames and DepthResults per cam_id (all with the same calib_version as
    ``cameras``), segmentation per cam_id, ROI ``AABB`` (world metres), gas cell size.
    Output: ``SceneSnapshot`` host NumPy arrays, read-only, world frame, caller-owned; built
    at acquisition/READY only (not per frame). Synchronous. Records ``calib_version``; any
    input mismatch raises CalibVersionError; insufficient/invalid depth raises
    ReconstructionError."""

    def reconstruct(
        self,
        frames: dict[int, Frame],
        depths: dict[int, DepthResult],
        segmentations: dict[int, SegmentationResult],
        cameras: dict[int, CameraCalibration],
        roi: AABB,
    ) -> SceneSnapshot: ...


@runtime_checkable
class FireStateView(Protocol):
    """Read-only view of the simulation's device-resident fields, for the renderer. Fields are
    arrays on the simulation device (framework tensors; opaque ``Any`` here), world frame,
    contiguous, valid until the next ``SimulationBackend.step``. Shapes: T_g (G,G,G) K,
    soot (G,G,G) smoke density (dimensionless mass fraction), char (N_e,) 0..1, removed (N_e,) 0..1,
    h_gas float metres per cell, roi AABB (world metres). The renderer must never write them.
    Synchronisation: the caller guarantees the sim stream has completed before reading. Tied
    to the snapshot's calib_version through the reset snapshot."""

    @property
    def T_g(self) -> Any: ...

    @property
    def soot(self) -> Any: ...

    @property
    def char(self) -> Any: ...

    @property
    def removed(self) -> Any: ...

    @property
    def h_gas(self) -> float: ...

    @property
    def roi(self) -> AABB: ...


@runtime_checkable
class SimulationBackend(_BackendMetrics, Protocol):
    """Fire simulation on the configured device. ``reset(snapshot)`` uploads the snapshot
    (host, read-only, world frame, metres) to device mirrors the backend owns and resets all
    state; the snapshot is never mutated. ``ignite(point_world (3,) float, radius_m)``
    ignites elements within the radius and returns the number lit. ``step(n_steps)`` advances
    exactly n_steps of fixed ``dt_sim`` seconds (deterministic for a given seed); returns
    after the device work is complete. ``state_host()`` returns a ``FireStateHost`` copy
    (caller-owned, units K, kg/m^2, m/s, Pa). ``t_sim`` [s] and ``step_count`` count
    simulated time/steps. ``set_material(obj_id, material_id)`` re-assigns an object's
    material before/while running. Not thread-safe (single owner). Bound to the
    snapshot's calib_version; calling before reset raises RuntimeError; invalid arguments
    raise ValueError."""

    @property
    def t_sim(self) -> float: ...

    @property
    def step_count(self) -> int: ...

    def reset(self, snapshot: SceneSnapshot) -> None: ...

    def ignite(self, point_world: np.ndarray, radius_m: float) -> int: ...

    def step(self, n_steps: int) -> None: ...

    def state_host(self) -> FireStateHost: ...

    def set_material(self, obj_id: int, material_id: int) -> None: ...


@runtime_checkable
class RendererBackend(_BackendMetrics, Protocol):
    """Renders the fire for one camera and composites it over the live frame. Inputs: cam_id
    (key into the snapshot's cameras), ``Frame`` (rgb uint8[H,W,3] sRGB, camera frame of
    that cam, calib_version must equal the snapshot's) and a ``FireStateView`` (device
    fields, read-only: the renderer never writes simulation state). Output: ndarray uint8
    (H,W,3) sRGB composited, host, C-contiguous, caller-owned; synchronised on return. The
    renderer owns its own device mirrors of the snapshot geometry. Not thread-safe per
    instance. Raises CalibVersionError on mismatch, KeyError for unknown cam_id."""

    def render(self, cam_id: int, frame: Frame, fire: FireStateView) -> np.ndarray: ...


@runtime_checkable
class ReplayController(Protocol):
    """Deterministic replay of a recorded session (snapshot + ignition events + seed).
    ``load(path)`` reads a recording (host files, caller-supplied path; raises
    FileNotFoundError / ValueError on bad content); ``seek(step)`` re-runs the simulation
    from reset to ``step`` (fixed dt_sim, same seed), ``digest()`` returns the sha256 of
    ``FireStateHost.digest()`` at the current step for equality checks. Single owner, not
    thread-safe, synchronous. The recording stores the snapshot's calib_version; replay
    against a different calibration raises CalibVersionError."""

    def load(self, path: str) -> None: ...

    def seek(self, step: int) -> None: ...

    def digest(self) -> str: ...


@runtime_checkable
class BenchmarkRecorder(Protocol):
    """Records timing/VRAM samples to CSV (Appendix C schema). ``record(name, value, unit,
    **labels)`` appends one host-side row (floats, SI-ish units as named; labels such as
    gpu, framework, calib_version are strings/ints); ``flush(path)`` writes the CSV (caller
    path, overwrites) and returns the row count. Thread-safe; synchronous; the recorder owns
    its rows. CPU runs must label gpu=none. Raises OSError on write failure."""

    def record(self, name: str, value: float, unit: str, **labels: str | float) -> None: ...

    def flush(self, path: str) -> int: ...
