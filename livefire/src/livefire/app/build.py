"""Shared scene/simulation construction for Runner.setup and replay (one code path, same snapshot)."""

from __future__ import annotations

from typing import Any

import numpy as np

from livefire.common.config import Params
from livefire.common.geometry import CameraCalibration
from livefire.scene.builder import ViewInput, build_snapshot, manual_material
from livefire.scene.snapshot import SceneSnapshot
from livefire.scene.synthetic import build_table_v1, load_cameras, render_view
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

SCENES = ("table_v1",)


def check_scene(name: str) -> None:
    """Raise ValueError for an unknown scene name (call before building anything)."""
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; available: {list(SCENES)}")


def load_synthetic_calibration(scene: str) -> tuple[Any, dict[int, CameraCalibration]]:
    """P0 CALIBRATING: the synthetic scene description and its camera calibrations."""
    check_scene(scene)
    return build_table_v1(), load_cameras()


def acquire_snapshot(scene_obj: Any, cams: dict[int, CameraCalibration], grid_cfg: dict[str, Any]) -> tuple[
    SceneSnapshot, dict[int, np.ndarray]
]:
    """P0 ACQUIRING: render the synthetic views (seed 0) and build the snapshot with ground-truth materials.
    ``grid_cfg`` is the full grid config (grid.yaml with any override). Returns ``(snapshot, {cam_id: rgb})``."""
    views, rgb = {}, {}
    for cid, calib in cams.items():
        r = render_view(scene_obj, calib, seed=0)
        views[cid] = ViewInput(calib, r.depth, r.valid, r.obj_id, r.rgb)
        rgb[cid] = r.rgb
    tab = scene_obj.objects_table()
    snap = build_snapshot(views, {o: manual_material(m) for o, (_, m) in tab.items()}, grid_cfg,
                          names={o: nm for o, (nm, _) in tab.items()})  # fmt: skip
    return snap, rgb


def build_sim(snap: SceneSnapshot, materials: MaterialTable, sim_params: Params, app_cfg: dict[str, Any]) -> FireSimulation:
    """The FireSimulation for ``snap`` (device from ``app_cfg``)."""
    return FireSimulation(snap, materials, sim_params, app_cfg)
