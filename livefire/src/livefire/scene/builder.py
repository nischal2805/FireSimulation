"""Snapshot builder (PRD 6.1): surface elements, obstacles and per-view occlusion from depth views."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from livefire.common.config import load_yaml
from livefire.common.geometry import AABB, CameraCalibration, VoxelGrid
from livefire.perception.material.types import MATERIAL_CLASSES, MaterialPrediction
from livefire.scene.snapshot import ObjectRegistry, SceneSnapshot, WorldObject
from livefire.scene.splat import view_occlusion
from livefire.scene.surfels import bin_surfels, build_neighbors, view_points
from livefire.sim.materials import MaterialTable

THICK_DELTA = 1e-3  # m; objects with delta >= this are obstacles


@dataclass(frozen=True)
class ViewInput:
    """One camera's fused-input view (world frame calibration, Z-depth, per-pixel global obj_id)."""

    calib: CameraCalibration
    depth: NDArray[np.float32]
    valid: NDArray[np.bool_]
    obj_id: NDArray[np.int32]
    rgb: NDArray[np.uint8] | None = None


def manual_material(name: str) -> MaterialPrediction:
    """One-hot manual-override prediction (confidence 1)."""
    p = np.zeros(len(MATERIAL_CLASSES), dtype=np.float32)
    p[MATERIAL_CLASSES.index(name)] = 1.0
    return MaterialPrediction(p, None, 1.0, "manual_override", False)


def build_snapshot(
    views: dict[int, ViewInput],
    materials: dict[int, MaterialPrediction],
    grid_cfg: dict[str, Any],
    version: int = 1,
    names: dict[int, str] | None = None,
    material_table: MaterialTable | None = None,
) -> SceneSnapshot:
    """Build a ``SceneSnapshot`` from per-camera views and ``obj_id -> MaterialPrediction``."""
    table = material_table or MaterialTable.load()
    roi = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
    G = int(grid_cfg["G"])
    s = float(grid_cfg["surfel_spacing"])
    grid = VoxelGrid(roi, G)
    cams = {cid: v.calib for cid, v in views.items()}
    versions = {c.calib_version for c in cams.values()}
    if len(versions) != 1:
        raise ValueError(f"cameras disagree on calib_version: {versions}")

    vps = [view_points(v.calib, v.depth, v.valid, v.obj_id, roi) for v in views.values()]
    sf = bin_surfels(vps, roi, s)
    n = len(sf.pos)
    unknown = sorted(set(np.unique(sf.obj_id).tolist()) - set(materials))
    if unknown:
        raise ValueError(f"no material for obj_ids {unknown}")
    mat_id = np.zeros(n, dtype=np.uint8)
    mat_conf = np.zeros(n, dtype=np.float32)
    for o, mp in materials.items():
        m = sf.obj_id == o
        mat_id[m] = mp.material_id
        mat_conf[m] = mp.confidence
    nbrs = build_neighbors(sf.pos, sf.obj_id, float(grid_cfg["neighbor_radius_factor"]) * s,
                           int(grid_cfg["neighbor_k_max"]))  # fmt: skip

    # obstacles: thick objects only
    thick = table.delta[mat_id] >= THICK_DELTA
    solid_mask = np.zeros((G, G, G), dtype=np.uint8)
    area_sum = np.zeros(G**3)
    idx = grid.index_of(sf.pos[thick].astype(np.float64))
    idx = np.clip(idx, 0, G - 1)
    np.add.at(area_sum, grid.linear_index(idx), sf.area[thick].astype(np.float64))
    solid_mask.reshape(-1)[grid.linear_index(idx)] = 1
    solid_fraction = np.where(solid_mask.reshape(-1) > 0, np.minimum(1.0, area_sum / grid.h**2), 0.0)
    solid_fraction = solid_fraction.reshape(G, G, G).astype(np.float32)

    vd, vh = {}, {}
    for cid, v in views.items():
        vd[cid], vh[cid] = view_occlusion(v.calib, sf.pos, sf.area, v.depth, v.valid, roi, sf.normal)

    reg = ObjectRegistry()
    for o in sorted(materials):
        reg.add(WorldObject(o, (names or {}).get(o, f"object_{o}"), materials[o],
                            {cid: o for cid in views}))  # fmt: skip
    return SceneSnapshot(
        version=version, calib_version=versions.pop(), roi=roi, h_gas=float(grid.h), cameras=cams,
        pos=sf.pos, normal=sf.normal, area=sf.area, obj_id=sf.obj_id, material_id=mat_id,
        material_conf=mat_conf, neighbors=nbrs, solid_mask=solid_mask, solid_fraction=solid_fraction,
        view_depth=vd, view_hole_mask=vh, objects=reg, surfel_spacing=s,
    )  # fmt: skip


def snapshot_from_synthetic(scene_name: str = "table_v1") -> SceneSnapshot:
    """Render the synthetic scene and build its snapshot (ground-truth materials)."""
    from livefire.scene.synthetic import build_table_v1, load_cameras, render_view

    if scene_name != "table_v1":
        raise ValueError(f"unknown scene {scene_name!r}")
    scene = build_table_v1()
    views = {}
    for cid, calib in load_cameras().items():
        r = render_view(scene, calib, seed=0)
        views[cid] = ViewInput(calib, r.depth, r.valid, r.obj_id, r.rgb)
    tab = scene.objects_table()
    mats = {o: manual_material(m) for o, (_, m) in tab.items()}
    return build_snapshot(views, mats, load_yaml("grid"), names={o: nm for o, (nm, _) in tab.items()})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Build a scene snapshot from the synthetic scene.")
    ap.add_argument("--scene", default="table_v1")
    ap.add_argument("--out", default="out/snapshot.npz")
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    snap = snapshot_from_synthetic(args.scene)
    dt = time.perf_counter() - t0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: getattr(snap, k) for k in ("pos", "normal", "area", "obj_id", "material_id",
                                            "material_conf", "neighbors", "solid_mask", "solid_fraction")}  # fmt: skip
    for cid in snap.view_depth:
        arrays[f"view_depth_{cid}"] = snap.view_depth[cid]
        arrays[f"view_hole_mask_{cid}"] = snap.view_hole_mask[cid]
    np.savez_compressed(out, version=snap.version, calib_version=snap.calib_version,
                        h_gas=snap.h_gas, surfel_spacing=snap.surfel_spacing, **arrays)  # fmt: skip
    print(f"snapshot: {len(snap.pos)} elements, solid cells {int(snap.solid_mask.sum())}, "
          f"build {dt:.2f} s -> {out}")  # fmt: skip


if __name__ == "__main__":
    main()
