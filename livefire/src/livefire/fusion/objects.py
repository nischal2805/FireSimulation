"""Table plane fit and geometric object clustering (P1R4).

Frame: world, metres (+Z up, table plane z = 0). Inputs are fused surface samples (see ``extract.py``); outputs are
per-sample integer labels owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from livefire.common.config import load_yaml
from livefire.common.errors import ReconstructionError
from livefire.common.log import get_logger

F64 = NDArray[np.float64]
log = get_logger(__name__)

TABLE_ID = 0
DROPPED = -2  # label of samples in components below the minimum object size


def _val(x: Any) -> Any:
    return x["value"] if isinstance(x, dict) and "value" in x else x


@dataclass(frozen=True)
class PlaneFit:
    """Plane ``n . p = d`` (unit ``n`` with ``n_z > 0``, world frame). ``z_dev_m`` is the largest deviation of the fitted
    plane from ``z = 0`` over the XY extent of its inliers; ``agrees`` is ``z_dev_m <= plane_tol_m``."""

    n: F64
    d: float
    n_inliers: int
    z_dev_m: float
    tilt_deg: float
    agrees: bool


def fit_table_plane(
    pos: F64, normal: F64, footprint: F64, cfg: dict[str, Any], seed: int | None = None
) -> PlaneFit:
    """RANSAC (area-weighted inlier count) over up-facing samples, then a least-squares refit on the inliers.
    Warns (logging) if the plane disagrees with ``z = 0`` by more than ``plane_tol_m``; raises ``ReconstructionError`` if
    there is no up-facing surface at all."""
    pc = cfg["plane"]
    tol, inl = float(_val(pc["plane_tol_m"])), float(_val(pc["inlier_m"]))
    cand = np.flatnonzero(normal[:, 2] >= float(pc["min_normal_z"]))
    if len(cand) < 3:
        raise ReconstructionError("no up-facing surface samples: cannot fit the table plane")
    rng = np.random.default_rng(int(pc["seed"]) if seed is None else seed)
    P, A = pos[cand], footprint[cand]
    best, best_score = None, -1.0
    for _ in range(int(pc["ransac_iters"])):
        s = rng.choice(len(cand), 3, replace=False)
        n = np.cross(P[s[1]] - P[s[0]], P[s[2]] - P[s[0]])
        nl = np.linalg.norm(n)
        if nl < 1e-12:
            continue
        n = n / nl * (1.0 if n[2] >= 0 else -1.0)
        if n[2] < float(pc["min_normal_z"]):
            continue
        d = float(n @ P[s[0]])
        score = float(A[np.abs(P @ n - d) <= inl].sum())
        if score > best_score:
            best, best_score = (n, d), score
    if best is None:
        raise ReconstructionError("RANSAC found no plane candidate")
    n, d = best
    for _ in range(3):  # least-squares refit on inliers
        m = np.abs(P @ n - d) <= inl
        Q, w = P[m], A[m]
        c = (Q * w[:, None]).sum(0) / w.sum()
        _u, _s, vt = np.linalg.svd((Q - c) * np.sqrt(w)[:, None], full_matrices=False)
        n = vt[2] * (1.0 if vt[2][2] >= 0 else -1.0)
        d = float(n @ c)
    m = np.abs(P @ n - d) <= inl
    lo, hi = P[m, :2].min(0), P[m, :2].max(0)
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    z_dev = float(np.abs((d - corners @ n[:2]) / n[2]).max())
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1, 1))))
    fit = PlaneFit(n, d, int(m.sum()), z_dev, tilt, z_dev <= tol)
    if not fit.agrees:
        log.warning("fitted table plane deviates from z = 0 by %.1f mm (tol %.1f mm, tilt %.2f deg): "
                    "calibration or anchoring is off", 1e3 * z_dev, 1e3 * tol, tilt)
    return fit


def cluster_objects(
    pos: F64, normal: F64, footprint: F64, plane: PlaneFit, cfg: dict[str, Any], spacing: float
) -> NDArray[np.int32]:
    """Label every sample: ``0`` = table, ``1..`` = objects (largest area first), ``-2`` = dropped.

    Table = up-facing samples within ``inlier_m`` of the plane, plus components lying at or below the plane (the table
    skirt: topmost point within ``inlier_m`` or area-weighted mean height <= 0). The remaining samples are linked when closer than ``max(radius_factor * spacing, 2 * voxel_size)`` (the neighbour rule);
    connected components with area (sum of footprints) below ``min_object_area_m2`` are dropped."""
    pc, cc = cfg["plane"], cfg["cluster"]
    inl = float(_val(pc["inlier_m"]))
    sd = pos @ plane.n - plane.d  # signed distance above the plane
    lab = np.full(len(pos), DROPPED, dtype=np.int32)
    on_plane = (np.abs(sd) <= inl) & (normal @ plane.n >= float(pc["min_normal_z"]))
    lab[on_plane] = TABLE_ID
    rest = np.flatnonzero(~on_plane)
    if not len(rest):
        return lab
    # tie the link radius to the TSDF voxel size so coarse voxels do not fragment objects
    radius = max(float(cc["radius_factor"]) * spacing, 2.0 * float(_val(cfg["voxel_size"])))
    pairs = cKDTree(pos[rest]).query_pairs(radius, output_type="ndarray")
    k = len(rest)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(k, k))
    ncomp, comp = connected_components(g, directed=False)
    area = np.bincount(comp, footprint[rest], ncomp)
    top = np.full(ncomp, -np.inf)
    np.maximum.at(top, comp, sd[rest])
    mean_sd = np.bincount(comp, footprint[rest] * sd[rest], ncomp) / np.maximum(area, 1e-12)
    min_area = float(_val(cc["min_object_area_m2"]))
    order = [c for c in np.argsort(-area, kind="stable") if area[c] >= min_area]
    nxt = 1
    for c in order:
        sel = rest[comp == c]
        if top[c] <= inl or mean_sd[c] <= 0.0:
            lab[sel] = TABLE_ID  # skirt
        else:
            lab[sel] = nxt
            nxt += 1
    return lab


def load_acquisition_config() -> dict[str, Any]:
    """``configs/acquisition.yaml: acquisition`` as a dict."""
    return load_yaml("acquisition")["acquisition"]
