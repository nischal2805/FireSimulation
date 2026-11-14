"""Procedural synthetic two-camera scene "table_v1" (analytic ray casting, NumPy, host).

World frame: right-handed, metres, +Z up. Primitives are axis-aligned boxes (slab method) and
finite axis-aligned rectangles (two-sided or one-sided). ``render_view`` casts one ray per pixel
(pixel centres, OpenCV camera frame) and returns Z-depth (metres along camera +Z), not ray
distance. Invalid pixels (miss) hold depth 0.0, valid False, obj_id -1, black RGB.
All arrays are host NumPy and owned by the caller. Deterministic for a given seed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from livefire.common.config import load_yaml
from livefire.common.geometry import CameraCalibration, look_at, make_K, ray_dirs_camera
from livefire.sim.materials import MaterialTable

F64 = NDArray[np.float64]

LIGHT_DIR = np.array([0.3, -0.4, 1.0]) / np.linalg.norm([0.3, -0.4, 1.0])  # towards the light
LIGHT_INTENSITY = 0.8
AMBIENT = 0.2
_EPS = 1e-9


@dataclass(frozen=True)
class Box:
    """Axis-aligned box, world metres. ``lo``/``hi`` (3,)."""

    obj_id: int
    name: str
    material: str
    albedo: tuple[float, float, float]
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]

    def intersect(self, o: F64, d: F64) -> tuple[F64, F64]:
        """Rays origin o (3,), directions d (N,3) -> (t (N,) inf on miss, normal (N,3))."""
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        dd = np.where(np.abs(d) < 1e-30, 1e-30, d)
        a, b = (lo - o) / dd, (hi - o) / dd
        tmin, tmax = np.minimum(a, b), np.maximum(a, b)
        axis = np.argmax(tmin, axis=1)
        tn = tmin[np.arange(len(d)), axis]
        tf = tmax.min(axis=1)
        hit = (tn <= tf) & (tn > _EPS)
        n = np.zeros_like(d)
        n[np.arange(len(d)), axis] = -np.sign(d[np.arange(len(d)), axis])
        return np.where(hit, tn, np.inf), n

    def distance(self, p: F64) -> F64:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        q = np.maximum(lo - p, p - hi)
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = -q.max(axis=1)
        return np.where(q.max(axis=1) > 0, outside, inside)

    def areas(self) -> tuple[F64, list[tuple[int, float]]]:
        s = np.asarray(self.hi) - np.asarray(self.lo)
        faces = [(ax, sg) for ax in range(3) for sg in (-1.0, 1.0)]
        return np.array([s[(ax + 1) % 3] * s[(ax + 2) % 3] for ax, _ in faces]), faces

    def sample(self, n: int, rng: np.random.Generator) -> tuple[F64, F64, float]:
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        areas, faces = self.areas()
        fi = rng.choice(len(faces), size=n, p=areas / areas.sum())
        pts = lo + rng.random((n, 3)) * (hi - lo)
        nrm = np.zeros((n, 3))
        for k, (ax, sg) in enumerate(faces):
            m = fi == k
            pts[m, ax] = hi[ax] if sg > 0 else lo[ax]
            nrm[m, ax] = sg
        return pts, nrm, float(areas.sum())


@dataclass(frozen=True)
class Rect:
    """Finite axis-aligned rectangle in plane ``p[axis] == offset``; ``lo``/``hi`` are the
    extents along the two remaining axes (in increasing axis order). Two-sided unless
    ``normal_sign`` is +1/-1 (one-sided, facing +/- axis)."""

    obj_id: int
    name: str
    material: str
    albedo: tuple[float, float, float]
    axis: int
    offset: float
    lo: tuple[float, float]
    hi: tuple[float, float]
    two_sided: bool = True
    normal_sign: float = 1.0

    @property
    def _other(self) -> tuple[int, int]:
        return tuple(a for a in range(3) if a != self.axis)  # type: ignore[return-value]

    def intersect(self, o: F64, d: F64) -> tuple[F64, F64]:
        ax = self.axis
        da = d[:, ax]
        ok = np.abs(da) > 1e-12
        t = np.where(ok, (self.offset - o[ax]) / np.where(ok, da, 1.0), np.inf)
        hit = ok & (t > _EPS)
        if not self.two_sided:
            hit &= da * self.normal_sign < 0  # ray must oppose the facing direction
        for k, a in enumerate(self._other):
            c = o[a] + t * d[:, a]
            hit &= (c >= self.lo[k]) & (c <= self.hi[k])
        n = np.zeros_like(d)
        n[:, ax] = -np.sign(da) if self.two_sided else self.normal_sign
        return np.where(hit, t, np.inf), n

    def distance(self, p: F64) -> F64:
        dn = p[:, self.axis] - self.offset
        e2 = np.zeros(len(p))
        for k, a in enumerate(self._other):
            e = np.maximum(self.lo[k] - p[:, a], p[:, a] - self.hi[k])
            e2 += np.maximum(e, 0.0) ** 2
        return np.sqrt(e2 + dn**2)

    def area(self) -> float:
        return float((self.hi[0] - self.lo[0]) * (self.hi[1] - self.lo[1]))

    def sample(self, n: int, rng: np.random.Generator) -> tuple[F64, F64, float]:
        pts = np.empty((n, 3))
        pts[:, self.axis] = self.offset
        for k, a in enumerate(self._other):
            pts[:, a] = self.lo[k] + rng.random(n) * (self.hi[k] - self.lo[k])
        nrm = np.zeros((n, 3))
        nrm[:, self.axis] = self.normal_sign
        return pts, nrm, self.area()


Primitive = Box | Rect


@dataclass
class SyntheticScene:
    """Set of analytic primitives. Objects 0..N-1 are real objects; the room is obj_id -1."""

    name: str
    objects: list[Primitive]  # obj_id >= 0, in id order
    room: list[Primitive] = field(default_factory=list)  # obj_id == -1, material "none"

    def objects_table(self) -> dict[int, tuple[str, str]]:
        """obj_id -> (name, material) for the real objects (room excluded)."""
        return {o.obj_id: (o.name, o.material) for o in self.objects}

    def _get(self, obj_id: int) -> Primitive:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        raise KeyError(f"unknown obj_id {obj_id}")

    def distance_to_object(self, obj_id: int, p: F64) -> F64:
        """Euclidean distance (m, >= 0) from world points p (N,3) to the surface of the object."""
        return self._get(obj_id).distance(np.atleast_2d(np.asarray(p, dtype=np.float64)))

    def sample_surface(
        self, obj_id: int, n: int, rng: np.random.Generator
    ) -> tuple[F64, F64, float]:
        """Seeded area-uniform surface samples: (points (n,3) m, outward unit normals (n,3),
        total surface area m^2). Rect normals follow the facing direction (+axis for two-sided)."""
        return self._get(obj_id).sample(n, rng)


def build_table_v1() -> SyntheticScene:
    """The table_v1 scene (world metres)."""
    objs: list[Primitive] = [
        Box(0, "table", "other_nonflammable", (0.45, 0.33, 0.22), (-0.8, -0.5, -0.05), (0.8, 0.5, 0.0)),
        Box(1, "wood_block", "wood", (0.55, 0.38, 0.2), (0.125, 0.07, 0.0), (0.275, 0.17, 0.08)),
        Box(2, "metal_box", "metal", (0.6, 0.6, 0.62), (-0.34, 0.16, 0.0), (-0.26, 0.24, 0.08)),
        # paper: plane y = 0, spans x and z
        Rect(3, "paper_sheet", "paper", (0.85, 0.85, 0.82), 1, 0.0, (-0.15, 0.0), (0.0, 0.2)),
        # cloth: plane z = 0.003, spans x and y, upward-facing
        Rect(4, "cloth_patch", "cloth", (0.6, 0.15, 0.15), 2, 0.003, (-0.2, -0.25), (0.05, -0.02),
             two_sided=False, normal_sign=1.0),
    ]  # fmt: skip
    g = (0.5, 0.5, 0.5)
    room: list[Primitive] = [
        Rect(-1, "room", "none", g, 2, -0.75, (-2.5, -2.5), (2.5, 2.5)),
        Rect(-1, "room", "none", g, 0, 2.5, (-2.5, -0.75), (2.5, 3.0)),
        Rect(-1, "room", "none", g, 0, -2.5, (-2.5, -0.75), (2.5, 3.0)),
        Rect(-1, "room", "none", g, 1, 2.5, (-2.5, -0.75), (2.5, 3.0)),
        Rect(-1, "room", "none", g, 1, -2.5, (-2.5, -0.75), (2.5, 3.0)),
    ]  # fmt: skip
    # Walls: lo/hi along the two remaining axes in increasing axis order (x-walls: y,z; y-walls: x,z).
    return SyntheticScene("table_v1", objs, room)


def validate_materials(scene: SyntheticScene, table: MaterialTable | None = None) -> None:
    """Raise if any object material name is not in the MaterialTable (room excluded)."""
    table = table or MaterialTable.load()
    for o in scene.objects:
        table.material_id(o.material)


def load_cameras(path_name: str = "camera") -> dict[int, CameraCalibration]:
    """Build the synthetic rig from ``configs/<path_name>.yaml`` (cam0 -> id 0, cam1 -> id 1)."""
    cfg = load_yaml(path_name)
    K = make_K(cfg["fx"], cfg["fy"], cfg["cx"], cfg["cy"])
    out: dict[int, CameraCalibration] = {}
    for i, (_, c) in enumerate(sorted(cfg["cameras"].items())):
        out[i] = CameraCalibration(
            cam_id=i,
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            K=K,
            dist=np.asarray(cfg["dist"], dtype=np.float64),
            T_world_from_camera=look_at(c["position"], c["target"]),
            calib_version=int(cfg["calib_version"]),
        )
    return out


@dataclass(frozen=True)
class SyntheticView:
    """One rendered view, all (H, W[, C]) host arrays, caller-owned.

    rgb uint8 sRGB; rgb_linear float32 linear light; depth float32 Z-depth m (0.0 invalid);
    valid bool; obj_id int32 (-1 background/room/miss); normal_world float32 unit (zeros on miss).
    """

    rgb: NDArray[np.uint8]
    rgb_linear: NDArray[np.float32]
    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    obj_id: NDArray[np.int32]
    normal_world: NDArray[np.float32]


def srgb_encode(lin: NDArray[np.floating]) -> NDArray[np.float64]:
    """Linear [0,1] -> sRGB [0,1] (IEC 61966-2-1)."""
    lin = np.clip(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def render_view(
    scene: SyntheticScene,
    calib: CameraCalibration,
    seed: int | None = None,
    noise_sigma: float = 0.0,
) -> SyntheticView:
    """Ray-cast ``scene`` from ``calib``. Lambert (directional light + ambient) in linear light,
    then sRGB encode. Optional Gaussian noise (std ``noise_sigma``, linear light) seeded by
    ``seed`` (None -> 0). Pure function: two calls with equal arguments are bitwise equal."""
    h, w = calib.height, calib.width
    R = calib.T_world_from_camera[:3, :3]
    o = calib.center_world
    dcam = ray_dirs_camera(calib).reshape(-1, 3)  # z component 1 -> t is Z-depth
    d = dcam @ R.T
    n_pix = h * w
    best_t = np.full(n_pix, np.inf)
    best_n = np.zeros((n_pix, 3))
    best_id = np.full(n_pix, -1, dtype=np.int32)
    best_alb = np.zeros((n_pix, 3))
    for prim in [*scene.room, *scene.objects]:
        t, n = prim.intersect(o, d)
        m = t < best_t
        best_t[m], best_n[m], best_id[m] = t[m], n[m], prim.obj_id
        best_alb[m] = prim.albedo
    valid = np.isfinite(best_t)
    lam = np.clip(best_n @ LIGHT_DIR, 0.0, None)
    lin = best_alb * (AMBIENT + LIGHT_INTENSITY * lam)[:, None]
    lin[~valid] = 0.0
    if noise_sigma > 0:
        rng = np.random.default_rng(0 if seed is None else seed)
        lin = np.clip(lin + rng.normal(0.0, noise_sigma, lin.shape), 0.0, 1.0)
        lin[~valid] = 0.0
    lin32 = lin.astype(np.float32)
    rgb = np.rint(srgb_encode(lin32) * 255.0).astype(np.uint8)
    depth = np.where(valid, best_t, 0.0).astype(np.float32)
    return SyntheticView(
        rgb=rgb.reshape(h, w, 3),
        rgb_linear=lin32.reshape(h, w, 3),
        depth=depth.reshape(h, w),
        valid=valid.reshape(h, w),
        obj_id=np.where(valid, best_id, -1).astype(np.int32).reshape(h, w),
        normal_world=np.where(valid[:, None], best_n, 0.0).astype(np.float32).reshape(h, w, 3),
    )


def _obj_palette(obj: NDArray[np.int32]) -> NDArray[np.uint8]:
    pal = np.array(
        [[60, 60, 60], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48]],
        dtype=np.uint8,
    )
    return pal[np.clip(obj + 1, 0, len(pal) - 1)]


def main(argv: list[str] | None = None) -> None:
    """CLI: ``python -m livefire.scene.synthetic --out DIR`` writes rgb, depth (16-bit PNG, mm)
    and obj_id colour PNGs per camera."""
    import cv2

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--out", default="out/synthetic")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = build_table_v1()
    for cid, calib in load_cameras().items():
        v = render_view(scene, calib, seed=0)
        cv2.imwrite(str(out / f"cam{cid}_rgb.png"), v.rgb[..., ::-1])
        mm = np.clip(np.rint(v.depth * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / f"cam{cid}_depth_mm.png"), mm)
        cv2.imwrite(str(out / f"cam{cid}_obj_id.png"), _obj_palette(v.obj_id)[..., ::-1])
        print(f"cam{cid}: wrote to {out}; valid {v.valid.mean():.3f}")


if __name__ == "__main__":
    main()
