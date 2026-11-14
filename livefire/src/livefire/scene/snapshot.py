"""Immutable scene snapshot (PRD 6.1), object registry and calib-version checks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, fields

import numpy as np

from livefire.capture.types import Frame
from livefire.common.errors import CalibVersionError
from livefire.common.geometry import AABB, CameraCalibration
from livefire.perception.material.types import MATERIAL_CLASSES, MaterialPrediction


@dataclass
class WorldObject:
    """A world object: global id, name, material prior and per-camera observation ids."""

    obj_id: int
    name: str
    material: MaterialPrediction
    observations: dict[int, int] = field(default_factory=dict)  # cam_id -> per-camera instance id


__all__ = ["CalibVersionError", "ObjectRegistry", "SceneSnapshot", "WorldObject", "assert_calib_version"]


class ObjectRegistry:
    """World objects keyed by ``obj_id``. Overrides are stored on the object (PRD 7.3)."""

    def __init__(self, objects: list[WorldObject] | None = None) -> None:
        self._objects: dict[int, WorldObject] = {}
        for o in objects or []:
            self.add(o)

    def add(self, obj: WorldObject) -> None:
        if obj.obj_id in self._objects:
            raise ValueError(f"duplicate obj_id {obj.obj_id}")
        self._objects[obj.obj_id] = obj

    def __getitem__(self, obj_id: int) -> WorldObject:
        return self._objects[obj_id]

    def __contains__(self, obj_id: int) -> bool:
        return obj_id in self._objects

    def __len__(self) -> int:
        return len(self._objects)

    def __iter__(self) -> Iterator[WorldObject]:
        return iter(self._objects.values())

    def override_material(self, obj_id: int, material_name: str) -> None:
        """Set a one-hot manual override (source ``manual_override``, confidence 1)."""
        if material_name not in MATERIAL_CLASSES:
            raise ValueError(f"unknown material {material_name!r}; expected one of {MATERIAL_CLASSES}")
        obj = self._objects[obj_id]
        p = np.zeros(len(MATERIAL_CLASSES), dtype=np.float32)
        p[MATERIAL_CLASSES.index(material_name)] = 1.0
        obj.material = MaterialPrediction(p, obj.material.category_hint, 1.0, "manual_override", False)


# name -> (dtype, trailing-shape spec); "N" = N_e, "G" = grid, K = neighbours
@dataclass(frozen=True, eq=False)
class SceneSnapshot:
    """Immutable static scene (PRD 6.1). All arrays are host NumPy, made read-only in
    ``__post_init__``; sim and render backends build their own device mirrors once at READY
    and own them. Frame: world (metres, +Z up). ``surfel_spacing`` [m] is an additive field
    (needed by conduction). Every camera's ``calib_version`` must equal ``calib_version``
    (else ``CalibVersionError``)."""

    version: int
    calib_version: int
    roi: AABB
    h_gas: float
    cameras: dict[int, CameraCalibration]
    pos: np.ndarray  # f32 (N,3) m
    normal: np.ndarray  # f32 (N,3) unit
    area: np.ndarray  # f32 (N,) m^2
    obj_id: np.ndarray  # i32 (N,), -1 unknown
    material_id: np.ndarray  # u8 (N,)
    material_conf: np.ndarray  # f32 (N,)
    neighbors: np.ndarray  # i32 (N,K), -1 padded
    solid_mask: np.ndarray  # u8 (G,G,G)
    solid_fraction: np.ndarray  # f32 (G,G,G)
    view_depth: dict[int, np.ndarray]  # f32 (H,W) per cam, z-depth m, 0 = hole
    view_hole_mask: dict[int, np.ndarray]  # bool (H,W) per cam
    objects: ObjectRegistry
    surfel_spacing: float

    def __post_init__(self) -> None:
        def chk(name: str, a: np.ndarray, dtype: type, shape: tuple[int | None, ...]) -> None:
            if not isinstance(a, np.ndarray) or a.dtype != dtype:
                raise ValueError(f"{name}: expected {np.dtype(dtype)} ndarray, got {getattr(a, 'dtype', type(a))}")
            if a.ndim != len(shape) or any(s is not None and s != d for s, d in zip(shape, a.shape)):
                raise ValueError(f"{name}: expected shape {shape}, got {a.shape}")

        n = self.pos.shape[0] if isinstance(self.pos, np.ndarray) and self.pos.ndim == 2 else -1
        chk("pos", self.pos, np.float32, (n, 3))
        chk("normal", self.normal, np.float32, (n, 3))
        chk("area", self.area, np.float32, (n,))
        chk("obj_id", self.obj_id, np.int32, (n,))
        chk("material_id", self.material_id, np.uint8, (n,))
        chk("material_conf", self.material_conf, np.float32, (n,))
        chk("neighbors", self.neighbors, np.int32, (n, None))
        if self.neighbors.size and (self.neighbors.max() >= n or self.neighbors.min() < -1):
            raise ValueError("neighbors: indices must be in [-1, N_e)")
        g = self.solid_mask.shape[0] if self.solid_mask.ndim == 3 else -1
        chk("solid_mask", self.solid_mask, np.uint8, (g, g, g))
        chk("solid_fraction", self.solid_fraction, np.float32, (g, g, g))
        if set(self.view_depth) != set(self.view_hole_mask):
            raise ValueError("view_depth and view_hole_mask must have the same camera ids")
        for cid, d in self.view_depth.items():
            chk(f"view_depth[{cid}]", d, np.float32, (None, None))
            chk(f"view_hole_mask[{cid}]", self.view_hole_mask[cid], np.bool_, d.shape)
        if not self.surfel_spacing > 0 or not self.h_gas > 0:
            raise ValueError("surfel_spacing and h_gas must be positive")
        for cid, cam in self.cameras.items():
            if cam.calib_version != self.calib_version:
                raise CalibVersionError(
                    f"camera {cid} calib_version {cam.calib_version} != snapshot {self.calib_version}"
                )
        for f in fields(self):
            v = getattr(self, f.name)
            for a in v.values() if isinstance(v, dict) else [v]:
                if isinstance(a, np.ndarray):
                    a.flags.writeable = False


def assert_calib_version(snapshot: SceneSnapshot, frame: Frame) -> None:
    """Raise ``CalibVersionError`` if ``frame.calib_version`` differs from the snapshot's."""
    if frame.calib_version != snapshot.calib_version:
        raise CalibVersionError(
            f"frame calib_version {frame.calib_version} != snapshot {snapshot.calib_version}"
        )
