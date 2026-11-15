"""Cross-view object association through the fused 3D proxy (P3R3; PRD 6.1, 7.3).

Per-camera segmentation ids are *observations* (PRD 6.1), never world ids. The fused surfels are the shared 3D proxy:

1. ``label_surfels``: every surfel is projected into every view; it is *visible* there when its Z agrees with the
   accumulated depth within ``assoc.visibility_tol_m`` and it takes the instance id at the projected pixel (3x3
   majority). Invisible surfels and background pixels get -1.
2. ``associate_objects``: per view an instance is a surfel set. Instances of *different* views are linked when their
   surfel-set IoU (on the surfels labelled in both views, with at least ``assoc.min_instance_surfels`` shared) is >=
   ``assoc.min_iou``; of several candidates in one view the higher IoU wins, and the CIELAB mean
   colour distance breaks ties within ``assoc.iou_tie`` (a link needs both ends to pick each other, so one
   instance cannot pull two competitors into one object). World objects are the connected components (union-find).
   A surfel's object is the world object of its majority instance over the views that see it (a tie: the object
   with fewer surfels, since the usual failure is one mask covering two objects); surfels without any
   instance keep the geometric label (P1R4, ``fusion/objects.py``); instance-labelled surfels override the table
   label (the low-profile cloth). Table = object 0; objects are renumbered 1.. by area.

"View" ids are opaque ints: one per camera for a single keyframe, one per (camera, keyframe) with several keyframes
(the caller maps them back to cameras). Frames: world metres (+Z up) for positions, image pixels for the labels.
Everything is host NumPy and the caller owns the results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from livefire.common.geometry import CameraCalibration, project
from livefire.fusion.accumulate import AccumulatedDepth
from livefire.fusion.objects import TABLE_ID
from livefire.perception.types import SegmentationResult

I32 = NDArray[np.int32]
UNLABELLED = -1


def _val(x: Any) -> Any:
    return x["value"] if isinstance(x, dict) and "value" in x else x


def _majority_window(ids: NDArray[np.int32], u: NDArray, v: NDArray, win: int) -> NDArray[np.int32]:
    """Most frequent id in the ``win`` x ``win`` pixel window around ``(u, v)`` (centre wins ties; window clipped)."""
    h, w = ids.shape
    r = win // 2
    offs = [(0, 0)] + [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if (dy, dx) != (0, 0)]
    cand = np.stack([ids[np.clip(v + dy, 0, h - 1), np.clip(u + dx, 0, w - 1)] for dy, dx in offs], axis=1)
    counts = (cand[:, :, None] == cand[:, None, :]).sum(axis=2)
    best = counts.argmax(axis=1)  # first maximum = centre-most on ties
    return cand[np.arange(len(cand)), best].astype(np.int32)


def label_surfels(
    pos: NDArray,
    cals: Mapping[int, CameraCalibration],
    accumulated: Mapping[int, AccumulatedDepth],
    segmentations: Mapping[int, SegmentationResult],
    cfg: dict[str, Any],
) -> dict[int, I32]:
    """Per-view instance id of every surfel (P3R3 visibility rule).

    ``pos`` (N,3) world m; ``cals``/``accumulated``/``segmentations`` keyed by view id (same keys; ``accumulated`` is the
    depth used for fusion, shape = the calibration's (H, W)); ``cfg`` = ``configs/acquisition.yaml: acquisition``
    (reads ``assoc.visibility_tol_m`` and ``segment.majority_window_px``, default 3). Returns ``{view: int32 (N,)}``:
    the instance id at the projected pixel, -1 where the surfel is not visible in that view (outside the image,
    invalid depth, or its Z differs from the accumulated depth by more than the tolerance) or sits on background.
    Pure, caller owns the result."""
    pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
    tol = float(_val(cfg["assoc"]["visibility_tol_m"]))
    win = int(cfg.get("segment", {}).get("majority_window_px", 3))
    out: dict[int, I32] = {}
    for view, seg in segmentations.items():
        cal, acc = cals[view], accumulated[view]
        ids = seg.instance_ids
        if ids.shape != (cal.height, cal.width) or acc.depth.shape != ids.shape:
            raise ValueError(f"view {view}: segmentation {ids.shape}, depth {acc.depth.shape}, calibration "
                             f"{(cal.height, cal.width)} disagree")  # fmt: skip
        uv, z, ok = project(cal, pos)
        u, v = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
        ok &= (u >= 0) & (u < cal.width) & (v >= 0) & (v < cal.height)
        uc, vc = np.where(ok, u, 0), np.where(ok, v, 0)
        ok &= acc.valid[vc, uc] & (np.abs(acc.depth[vc, uc].astype(np.float64) - z) <= tol)
        lab = np.full(len(pos), UNLABELLED, dtype=np.int32)
        if ok.any():
            lab[ok] = _majority_window(ids, uc[ok], vc[ok], win)
        out[view] = lab
    return out


def instance_colors(rgb: NDArray[np.uint8], instance_ids: NDArray[np.int32]) -> dict[int, NDArray[np.float64]]:
    """Mean CIELAB colour (3,) of the pixels of each instance (ids >= 0). ``rgb`` uint8 (H, W, 3) sRGB."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    ids = instance_ids.reshape(-1)
    m = ids >= 0
    if not m.any():
        return {}
    k = int(ids[m].max()) + 1
    cnt = np.bincount(ids[m], minlength=k)
    mean = np.stack([np.bincount(ids[m], lab.reshape(-1, 3)[m, c], k) for c in range(3)], axis=1)
    return {int(i): mean[i] / cnt[i] for i in np.flatnonzero(cnt)}


@dataclass(frozen=True, eq=False)
class AssociationResult:
    """World objects from cross-view association.

    ``labels`` int32 (N,): world object id per surfel (0 = table, 1.. by descending area; -2 stays -2 from the geometric
    labels). ``instance_labelled`` bool (N,): the surfel's object came from an instance vote (False = geometric label).
    ``observations``: obj -> ``{view: instance id}``; ``categories``: obj -> categories of its instances (most frequent
    first; empty for geometric objects); ``report``: JSON-friendly dict (``links`` with IoUs, per-object ``source``
    ``segmentation|geometric``, surfel count and ``unlabelled_fraction``)."""

    labels: I32
    instance_labelled: NDArray[np.bool_]
    observations: dict[int, dict[int, int]]
    categories: dict[int, list[str]]
    report: dict[str, Any] = field(default_factory=dict)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _pick(
    cands: list[tuple[tuple[int, int], float]], src: tuple[int, int], iou_tie: float,
    colors: Mapping[int, Mapping[int, NDArray]],
) -> tuple[int, int]:  # fmt: skip
    """Best candidate: highest IoU; within ``iou_tie`` of it, the smallest CIELAB distance to ``src`` (if known)."""
    top = max(i for _, i in cands)
    near = [(n, i) for n, i in cands if top - i <= iou_tie]
    cs = colors.get(src[0], {}).get(src[1])
    if len(near) > 1 and cs is not None:
        def dist(n: tuple[int, int]) -> float:
            cn = colors.get(n[0], {}).get(n[1])
            return float(np.linalg.norm(cs - cn)) if cn is not None else np.inf

        return min(near, key=lambda t: (dist(t[0]), -t[1], t[0]))[0]
    return max(near, key=lambda t: (t[1], -t[0][0], -t[0][1]))[0]


def associate_objects(
    per_cam_labels: Mapping[int, I32],
    geometric_labels: NDArray,
    rgb_stats: Mapping[int, Mapping[int, NDArray]],
    cfg: dict[str, Any],
    *,
    categories: Mapping[int, Mapping[int, str]] | None = None,
    area: NDArray | None = None,
    pos: NDArray | None = None,
) -> AssociationResult:
    """Merge per-view instances into world objects (module docstring, P3R3).

    ``per_cam_labels`` ``{view: int32 (N,)}`` from ``label_surfels``; ``geometric_labels`` int32 (N,) from
    ``cluster_objects`` (0 table, 1.. objects, -2 dropped); ``rgb_stats`` ``{view: {instance: mean CIELAB (3,)}}``
    (``instance_colors``) for the colour tie-break; ``cfg`` = acquisition config (``assoc`` block); optional
    ``categories`` ``{view: {instance: label}}`` (identifies the table instance via ``assoc.table_categories``) and
    ``area`` (N,) m^2 (object ranking, default: surfel counts) and ``pos`` (N,3) m (nearest-surfel fill, default: the
    geometric component's majority object). Instances below ``assoc.min_instance_surfels`` surfels
    are ignored (their surfels keep the geometric label). The table is the world object containing an instance of a
    table category; without one, the largest object whose surfels are mostly (``assoc.table_overlap``) geometric table
    and that holds at least ``assoc.table_cover`` of all geometric-table surfels; if neither exists the geometric table
    stays obj 0. Surfels of geometric table stay table unless an instance voted otherwise; unlabelled surfels of a
    geometric object take the object of the nearest instance-labelled surfel of the same geometric component (a
    component with no instance at all remains its own object). Pure, caller owns the result."""
    ac = cfg["assoc"]
    min_iou, iou_tie = float(_val(ac["min_iou"])), float(ac["iou_tie"])
    min_n = int(ac["min_instance_surfels"])
    table_cats = {str(c).lower() for c in ac.get("table_categories", ["table"])}
    geo = np.asarray(geometric_labels, dtype=np.int32)
    n = len(geo)
    w = np.ones(n) if area is None else np.asarray(area, dtype=np.float64)
    views = sorted(per_cam_labels)
    colors = rgb_stats

    # nodes = (view, instance) with enough surfels
    size: dict[tuple[int, int], int] = {}
    for vw in views:
        lab = per_cam_labels[vw]
        ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
        for i, c in zip(ids.tolist(), cnt.tolist(), strict=True):
            if c >= min_n:
                size[(vw, int(i))] = int(c)
    nodes = sorted(size)
    index = {nd: k for k, nd in enumerate(nodes)}

    # IoU between instances of different views, on the surfels labelled (= visible, non-background) in both views:
    # an object seen edge-on by one camera shares only a sliver of its surface with the other view, so the IoU of the
    # full surfel sets would be tiny for a correct pair while the co-visible IoU stays near 1 (a wrong pair stays ~0).
    iou: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
    for ai, va in enumerate(views):
        for vb in views[ai + 1 :]:
            la, lb = per_cam_labels[va], per_cam_labels[vb]
            both = (la >= 0) & (lb >= 0)
            if not both.any():
                continue
            sa = dict(zip(*(x.tolist() for x in np.unique(la[both], return_counts=True)), strict=True))
            sb = dict(zip(*(x.tolist() for x in np.unique(lb[both], return_counts=True)), strict=True))
            pairs, inter = np.unique(np.stack([la[both], lb[both]], 1), axis=0, return_counts=True)
            for (ia, ib), c in zip(pairs.tolist(), inter.tolist(), strict=True):
                a, b = (va, int(ia)), (vb, int(ib))
                if a in size and b in size and c >= min_n:  # a few shared surfels are leakage, not evidence
                    iou[(a, b)] = c / (sa[ia] + sb[ib] - c)

    # mutual best pick per (instance, other view), then union-find
    cand: dict[tuple[tuple[int, int], int], list[tuple[tuple[int, int], float]]] = {}
    for (a, b), x in iou.items():
        if x >= min_iou:
            cand.setdefault((a, b[0]), []).append((b, x))
            cand.setdefault((b, a[0]), []).append((a, x))
    pick = {key: _pick(c, key[0], iou_tie, colors) for key, c in cand.items()}
    uf = _UnionFind(len(nodes))
    links = []
    for (a, b), x in sorted(iou.items()):
        if x >= min_iou and pick.get((a, b[0])) == b and pick.get((b, a[0])) == a:
            uf.union(index[a], index[b])
            links.append({"a": list(a), "b": list(b), "iou": float(x)})
    root = np.array([uf.find(k) for k in range(len(nodes))], dtype=np.int64)
    comps = sorted(set(root.tolist()))
    comp_of = {nd: comps.index(int(root[index[nd]])) for nd in nodes}  # node -> component number 0..C-1
    ncomp = len(comps)

    # per-surfel vote: the component holding the majority of the views' instances
    votes = np.full((n, len(views)), -1, dtype=np.int64)
    for j, vw in enumerate(views):
        look = {i: comp_of[(vw, i)] for (v2, i) in nodes if v2 == vw}
        if look:
            lab = per_cam_labels[vw]
            lut = np.full(int(lab.max(initial=-1)) + 2, -1, dtype=np.int64)
            for i, c in look.items():
                if i < len(lut):
                    lut[i] = c
            votes[:, j] = np.where(lab >= 0, lut[np.clip(lab, 0, None)], -1)
    counts = np.stack([(votes == c).sum(1) for c in range(ncomp)], 1) if ncomp else np.zeros((n, 0), np.int64)
    voted = counts.max(1, initial=0) > 0
    # equal vote counts (e.g. two cameras disagree): the component with the smaller surfel extent wins, since the usual
    # failure is one mask covering two objects
    extent = (counts > 0).sum(0) / (n + 1.0)
    vote_comp = np.where(voted, (counts - 0.5 * extent).argmax(1) if ncomp else 0, -1)

    # which components are the table
    comp_cats: dict[int, list[str]] = {c: [] for c in range(ncomp)}
    comp_obs: dict[int, dict[int, int]] = {c: {} for c in range(ncomp)}
    for nd in nodes:
        c = comp_of[nd]
        comp_obs[c].setdefault(nd[0], nd[1])
        cat = (categories or {}).get(nd[0], {}).get(nd[1])
        if cat is not None:
            comp_cats[c].append(str(cat))
    table_comps = {c for c, cs in comp_cats.items() if any(x.lower() in table_cats for x in cs)}
    geo_table = geo == TABLE_ID
    if not table_comps and ncomp and geo_table.any():
        cover = float(_val(ac.get("table_cover", 0.5)))
        overlap = float(_val(ac.get("table_overlap", 0.5)))
        best = None
        for c in range(ncomp):
            m = vote_comp == c
            if m.sum() == 0:
                continue
            on_t = float(geo_table[m].mean())
            share = float((m & geo_table).sum() / geo_table.sum())
            if on_t >= overlap and share >= cover and (best is None or m.sum() > best[1]):
                best = (c, int(m.sum()))
        if best is not None:
            table_comps = {best[0]}

    # temporary world ids: table = TABLE_ID, other components 1.., geometric-only objects after them
    temp = np.full(n, -3, dtype=np.int64)
    non_table = [c for c in range(ncomp) if c not in table_comps]
    cid = {c: 1 + k for k, c in enumerate(non_table)}
    for c in table_comps:
        cid[c] = TABLE_ID
    for c, t in cid.items():
        temp[vote_comp == c] = t
    next_id = 1 + len(non_table)
    labelled = temp >= 0
    geo_only: dict[int, int] = {}  # geometric label -> temp id of an object with no instance at all
    for g in np.unique(geo[geo >= 1]).tolist():
        sel = np.flatnonzero((geo == g) & ~labelled)
        if not len(sel):
            continue
        anchor = np.flatnonzero((geo == g) & labelled)
        if len(anchor):
            if pos is not None:  # nearest instance-labelled surfel of the same geometric component
                _, nn = cKDTree(pos[anchor]).query(pos[sel])
                temp[sel] = temp[anchor[nn]]
            else:  # no positions: the component's majority object
                temp[sel] = np.bincount(temp[anchor][temp[anchor] >= 0]).argmax()
        else:
            geo_only[int(g)] = next_id
            temp[sel] = next_id
            next_id += 1
    temp[(geo == TABLE_ID) & ~labelled] = TABLE_ID
    dropped = (geo < 0) & ~labelled  # geometric noise components stay dropped unless an instance claimed them
    temp[dropped] = geo[dropped]

    # renumber by area (table fixed at 0)
    ids = sorted(set(temp[temp >= 0].tolist()) - {TABLE_ID})
    areas = {t: float(w[temp == t].sum()) for t in ids}
    order = sorted(ids, key=lambda t: (-areas[t], t))
    new_of = {t: 1 + k for k, t in enumerate(order)}
    new_of[TABLE_ID] = TABLE_ID
    lut = np.full(max(next_id, 1) + 1, -1, dtype=np.int32)
    for t, o in new_of.items():
        lut[t] = o
    out = np.where(temp >= 0, lut[np.clip(temp, 0, None)], temp).astype(np.int32)

    observations: dict[int, dict[int, int]] = {}
    cats: dict[int, list[str]] = {}
    for c, t in cid.items():
        o = new_of.get(t)
        if o is None:
            continue
        obs = observations.setdefault(o, {})
        for vw, i in comp_obs[c].items():
            obs.setdefault(vw, i)
        cats.setdefault(o, [])
        cats[o] += comp_cats[c]
    cats = {o: _by_frequency(cs) for o, cs in cats.items()}
    for o in set(new_of.values()):
        observations.setdefault(o, {})
        cats.setdefault(o, [])
    inst_labelled = np.zeros(n, dtype=bool)
    inst_labelled[labelled] = True
    rep_obj = {}
    for o in sorted(set(out[out >= 0].tolist())):
        m = out == o
        rep_obj[int(o)] = {
            "source": "segmentation" if observations.get(o) else "geometric",
            "n_surfels": int(m.sum()),
            "unlabelled_fraction": float((~inst_labelled[m]).mean()),
        }
    report = {"n_instances": len(nodes), "n_components": ncomp, "links": links,
              "table_components": sorted(table_comps), "objects": rep_obj}  # fmt: skip
    return AssociationResult(out, inst_labelled, observations, cats, report)


def _by_frequency(items: Sequence[str]) -> list[str]:
    """Unique items, most frequent first (ties: first seen)."""
    seen: dict[str, int] = {}
    for x in items:
        seen[x] = seen.get(x, 0) + 1
    return sorted(seen, key=lambda k: -seen[k])
