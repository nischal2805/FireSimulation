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
from livefire.scene.surfels import Surfels, ViewPoints, bin_surfels, build_neighbors, view_points
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
    roi = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
    s = float(grid_cfg["surfel_spacing"])
    vps = [view_points(v.calib, v.depth, v.valid, v.obj_id, roi) for v in views.values()]
    sf = bin_surfels(vps, roi, s)
    return snapshot_from_surfels(sf, views, materials, grid_cfg, version, names, material_table)


def snapshot_from_surfels(
    sf: Surfels,
    views: dict[int, ViewInput],
    materials: dict[int, MaterialPrediction],
    grid_cfg: dict[str, Any],
    version: int = 1,
    names: dict[int, str] | None = None,
    material_table: MaterialTable | None = None,
    observations: dict[int, dict[int, int]] | None = None,
) -> SceneSnapshot:
    """Assemble a ``SceneSnapshot`` from already-binned ``Surfels`` (world frame, metres) and the per-camera
    measured views used for the occlusion depth (``views[c].depth/valid`` are the measured depth, e.g. the accumulated
    acquisition depth; ``obj_id``/``rgb`` are unused here). ``observations`` maps obj_id -> {cam_id: per-camera id}
    (default: every camera observes every object under the same id)."""
    cams = {cid: v.calib for cid, v in views.items()}
    versions = {c.calib_version for c in cams.values()}
    if len(versions) != 1:
        raise ValueError(f"cameras disagree on calib_version: {versions}")
    roi = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
    s = float(grid_cfg["surfel_spacing"])
    vd, vh = {}, {}
    for cid, v in views.items():
        vd[cid], vh[cid] = view_occlusion(v.calib, sf.pos, sf.area, v.depth, v.valid, roi, sf.normal, s)
    obs = {o: {cid: o for cid in views} for o in materials} if observations is None else observations
    return _assemble_snapshot(sf, cams, versions.pop(), vd, vh, materials, grid_cfg, version, names, material_table, obs)


def _assemble_snapshot(
    sf: Surfels,
    cams: dict[int, CameraCalibration],
    calib_version: int,
    view_depth: dict[int, NDArray[np.float32]],
    view_hole_mask: dict[int, NDArray[np.bool_]],
    materials: dict[int, MaterialPrediction],
    grid_cfg: dict[str, Any],
    version: int,
    names: dict[int, str] | None,
    material_table: MaterialTable | None,
    observations: dict[int, dict[int, int]],
) -> SceneSnapshot:
    """Shared tail of ``snapshot_from_surfels`` and ``snapshot_from_primitives``: material lookup, symmetric
    neighbours, thick-object obstacles (``delta >= THICK_DELTA``), registry. Host NumPy, world frame, metres."""
    table = material_table or MaterialTable.load()
    roi = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
    G = int(grid_cfg["G"])
    s = float(grid_cfg["surfel_spacing"])
    grid = VoxelGrid(roi, G)
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

    reg = ObjectRegistry()
    for o in sorted(materials):
        reg.add(WorldObject(o, (names or {}).get(o, f"object_{o}"), materials[o], observations.get(o, {})))
    return SceneSnapshot(
        version=version, calib_version=calib_version, roi=roi, h_gas=float(grid.h), cameras=cams,
        pos=sf.pos, normal=sf.normal, area=sf.area, obj_id=sf.obj_id, material_id=mat_id,
        material_conf=mat_conf, neighbors=nbrs, solid_mask=solid_mask, solid_fraction=solid_fraction,
        view_depth=view_depth, view_hole_mask=view_hole_mask, objects=reg, surfel_spacing=s,
    )  # fmt: skip


def snapshot_from_primitives(
    objects: list[Any],
    grid_cfg: dict[str, Any],
    seed: int,
    material_table: MaterialTable | None = None,
    samples_per_cell: int = 64,
) -> SceneSnapshot:
    """Camera-free ``SceneSnapshot`` of analytic primitives (P2R11): ``Box``/``Rect`` of ``scene.synthetic`` (or any
    object with ``obj_id, name, material, sample(n, rng) -> (points (n,3) m, outward unit normals (n,3), area m^2)``).

    Each object is sampled with ``samples_per_cell * area / surfel_spacing^2`` area-uniform points (seeded by
    ``seed`` and ``obj_id``; at least ``samples_per_cell`` >= 4 per ``surfel_spacing`` cell on average), binned with
    ``bin_surfels`` (``ViewPoints`` footprint = area / n, all ``has_normal``), neighbours and obstacles as in
    ``snapshot_from_surfels`` (shared code). ``cameras={}``, ``view_depth={}``, ``calib_version=0``; materials are
    the objects' ``material`` names as manual overrides. World frame, metres, host NumPy; the ROI is ``grid_cfg``
    (needs ``G, roi_min, roi_max, surfel_spacing, neighbor_radius_factor, neighbor_k_max``)."""
    if samples_per_cell < 4:
        raise ValueError("samples_per_cell must be >= 4")
    roi = AABB(np.asarray(grid_cfg["roi_min"], float), np.asarray(grid_cfg["roi_max"], float))
    s = float(grid_cfg["surfel_spacing"])
    vps = []
    for o in objects:
        if o.obj_id < 0:
            raise ValueError("primitive obj_id must be >= 0")
        rng = np.random.default_rng([int(seed), int(o.obj_id)])
        _, _, area = o.sample(1, rng)
        n = max(16, int(np.ceil(samples_per_cell * area / s**2)))
        pts, nrm, area = o.sample(n, rng)
        keep = roi.contains(pts)
        vps.append(ViewPoints(pts[keep], nrm[keep], np.full(int(keep.sum()), o.obj_id, np.int32),
                              np.full(int(keep.sum()), area / n), np.ones(int(keep.sum()), bool)))  # fmt: skip
    sf = bin_surfels(vps, roi, s)
    materials = {o.obj_id: manual_material(o.material) for o in objects}
    present = set(sf.obj_id.tolist())
    missing = [o.name for o in objects if o.obj_id not in present]
    if missing:
        raise ValueError(f"objects produced no surface elements (outside the ROI or too small): {missing}")
    names = {o.obj_id: o.name for o in objects}
    return _assemble_snapshot(sf, {}, 0, {}, {}, materials, grid_cfg, 1, names, material_table, {})


def snapshot_from_synthetic(scene_name: str = "table_v1", grid_override: dict[str, Any] | None = None) -> SceneSnapshot:
    """Render the synthetic scene and build its snapshot (ground-truth materials). ``grid_override`` updates
    keys of ``configs/grid.yaml`` (e.g. ``{"G": 32}`` for a down-scaled gas grid)."""
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
    return build_snapshot(views, mats, {**load_yaml("grid"), **(grid_override or {})}, names={o: nm for o, (nm, _) in tab.items()})


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
