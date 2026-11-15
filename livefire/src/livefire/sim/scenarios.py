"""Scripted and randomized fire-spread scenarios and the M9 monotonicity helpers (P2-003, PRD 11.1 M9, P2R9/P2R11/P2R12).

Scenarios are analytic primitives (``scene.synthetic`` ``Box``/``Rect`` plus the oriented sheet ``OrientedRect``)
turned into a camera-free ``SceneSnapshot`` by ``scene.builder.snapshot_from_primitives`` over the small ROI of
``configs/scenarios.yaml`` (0.96 m cube, G=32), then run with the coupled ``FireSimulation`` (wind needs the gas
solver). World frame, metres, +Z up; times are simulated seconds; host NumPy; everything is seeded.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from livefire.common.config import Params, config_dir, load_params, load_yaml, load_yaml_text
from livefire.perception.material.types import MATERIAL_CLASSES
from livefire.scene.builder import snapshot_from_primitives
from livefire.scene.synthetic import Box, Rect
from livefire.sim.coupled import FireSimulation
from livefire.sim.materials import MaterialTable

F64 = NDArray[np.float64]
_FLAMMABLE = ("paper", "cloth", "wood", "plastic")
_UNIFORM_FIELDS = ("rho", "c", "k", "delta", "eps", "m0_areal", "T_ign", "t_delay", "mdot_ref", "dH", "chi",
                   "char_yield", "y_soot")  # fmt: skip


@dataclass(frozen=True)
class OrientedRect:
    """Finite rectangle ``origin + a u + b v`` (``a`` in [0, width], ``b`` in [0, height]) with a fixed outward unit
    normal; same ``sample`` contract as ``scene.synthetic.Rect``. ``origin``/``u``/``v``/``normal`` (3,) world metres
    / unit vectors; ``u``, ``v``, ``normal`` orthonormal."""

    obj_id: int
    name: str
    material: str
    origin: tuple[float, float, float]
    u: tuple[float, float, float]
    v: tuple[float, float, float]
    width: float
    height: float
    normal: tuple[float, float, float]

    def sample(self, n: int, rng: np.random.Generator) -> tuple[F64, F64, float]:
        """Seeded area-uniform points (n,3) m, outward normals (n,3), total area m^2."""
        a = rng.random(n) * self.width
        b = rng.random(n) * self.height
        pts = np.asarray(self.origin) + a[:, None] * np.asarray(self.u) + b[:, None] * np.asarray(self.v)
        return pts, np.tile(np.asarray(self.normal, float), (n, 1)), float(self.width * self.height)


@dataclass(frozen=True)
class Scenario:
    """One spread scenario. ``objects``: primitives with their own ``material``; ``ignite_point`` (3,) m and
    ``ignite_radius`` m: the emitter ignition; ``wind_speed`` m/s (``sim.yaml wind_speed``); ``target_obj``: obj_id
    whose material M9 swaps; ``emitter_obj``: obj_id whose elements within ``ignite_radius`` of ``ignite_point`` are lit
    (only that object, so the ignition sphere never lights the target directly); ``seed``: snapshot sampling seed; ``split``/``meta``: bookkeeping (``meta`` holds the drawn
    ``gap_m``, ``angle_deg``, ``size_*_m`` of randomized scenarios)."""

    name: str
    objects: list[Any]
    ignite_point: tuple[float, float, float]
    ignite_radius: float
    wind_speed: float
    target_obj: int
    seed: int
    split: str = "scripted"
    meta: dict[str, float] = field(default_factory=dict)
    emitter_obj: int = 0


def scenario_grid_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Grid config of the scenario ROI: ``cfg['grid']`` plus the neighbour settings of ``configs/grid.yaml``."""
    g = load_yaml("grid")
    return {"neighbor_radius_factor": g["neighbor_radius_factor"], "neighbor_k_max": g["neighbor_k_max"], **cfg["grid"]}


def _primitive(oid: int, d: dict[str, Any]) -> Box | Rect:
    alb = (0.5, 0.5, 0.5)
    if d["kind"] == "box":
        return Box(oid, d["name"], d["material"], alb, tuple(d["lo"]), tuple(d["hi"]))
    if d["kind"] == "rect":
        return Rect(oid, d["name"], d["material"], alb, int(d["axis"]), float(d["offset"]), tuple(d["lo"]),
                    tuple(d["hi"]), two_sided=False, normal_sign=float(d["normal_sign"]))  # fmt: skip
    raise ValueError(f"unknown primitive kind {d['kind']!r}")


def scripted_scenarios(cfg: dict[str, Any]) -> list[Scenario]:
    """The scripted scenarios of ``cfg['scripted']`` (demo_chain, two_sheets_parallel, sheet_over_block). Demo
    scripts, not a validation (PRD 11.1 M9)."""
    out = []
    for k, s in enumerate(cfg["scripted"]):
        objs = [_primitive(i, d) for i, d in enumerate(s["objects"])]
        names = [o.name for o in objs]
        ig = s["ignite"]
        out.append(Scenario(s["name"], objs, tuple(float(x) for x in ig["point"]), float(ig["radius_m"]),
                            float(s["wind_m_s"]), names.index(s["target"]), seed=1000 + k, split="scripted"))  # fmt: skip
    return out


def random_scenarios(cfg: dict[str, Any], split: str) -> list[Scenario]:
    """Seeded randomized pairs (emitter paper sheet in the plane y = 0 facing +y, target sheet facing the emitter,
    gap/angle/sizes/wind drawn uniformly from the ``random`` ranges). ``split`` ``dev`` uses seeds
    ``seed_dev + i``, ``heldout`` ``seed_heldout + i`` (i < n_<split>); the same ``(cfg, split)`` always gives the
    same scenarios. The target is built as paper; M9 swaps its material."""
    r = cfg["random"]
    if split not in ("dev", "heldout"):
        raise ValueError("split must be 'dev' or 'heldout'")
    n, base = int(r[f"n_{split}"]), int(r[f"seed_{split}"])
    out = []
    for i in range(n):
        seed = base + i
        rng = np.random.default_rng(seed)
        gap = float(rng.uniform(*r["gap_m"]))
        ang = float(rng.uniform(*r["angle_deg"]))
        s_e, s_t = (float(rng.uniform(*r["size_m"])) for _ in range(2))
        wind = float(rng.uniform(*r["wind_m_s"]))
        zc = 0.10 + 0.5 * max(s_e, s_t)  # both sheets centred at height zc (bottom edge of the larger at 0.10 m)
        emitter = Rect(0, "emitter_sheet", "paper", (0.5, 0.5, 0.5), 1, 0.0, (-s_e / 2, zc - s_e / 2),
                       (s_e / 2, zc + s_e / 2), two_sided=False, normal_sign=1.0)  # fmt: skip
        th = np.deg2rad(ang)
        u = (float(np.cos(th)), float(np.sin(th)), 0.0)
        target = OrientedRect(1, "target_sheet", "paper", (-s_t / 2, gap, zc - s_t / 2), u, (0.0, 0.0, 1.0), s_t, s_t,
                              (float(np.sin(th)), -float(np.cos(th)), 0.0))  # fmt: skip
        out.append(Scenario(f"{split}_{i:02d}", [emitter, target], (0.0, 0.0, zc), float(r["ignite_radius_m"]), wind, 1,
                            seed, split, {"gap_m": gap, "angle_deg": ang, "size_emitter_m": s_e, "size_target_m": s_t}))  # fmt: skip
    return out


def sim_params_with(base: Params | None = None, **over: float) -> Params:
    """``base`` (default ``configs/sim.yaml``) with value overrides (the declared range is widened to include a value
    outside it)."""
    p = base if base is not None else load_params("sim")
    e = copy.deepcopy({k: p.meta(k) for k in p.keys()})  # noqa: SIM118
    for k, v in over.items():
        e[k]["value"] = v
        e[k]["range"] = [min(e[k]["range"][0], v), max(e[k]["range"][1], v)]
    return Params(e)


def run_scenario(
    sc: Scenario,
    target_material: str | None,
    params: Params,
    app_cfg: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    table: MaterialTable | None = None,
    t_max_s: float | None = None,
    stop_on_ignition: bool = True,
) -> dict[str, Any]:
    """Run ``sc`` coupled (``FireSimulation``, device from ``app_cfg``). The snapshot is built once from the
    scenario's own materials; ``target_material`` (None = keep) is then assigned to the target object with
    ``FireSimulation.set_material`` before the ignition (so the geometry, emitter and gas obstacles are identical for
    every material). ``params['wind_speed']`` is overridden by ``sc.wind_speed``. The ignition detector reads the
    target's ``burning`` flags after every step; diagnostics every ``dt_check_steps``. The run ends at the first
    target ignition (``stop_on_ignition``), at ``t_max_s`` (default ``cfg['t_max_s']``), or when the fire is extinct
    (nothing burning, ``max T_g`` below the lowest ``T_ign`` and no flammable element at or above its ``T_ign``:
    nothing can ignite any more).

    Returns ``{t_ignite_s (None = never), ignited, end ('ignited'|'extinct'|'t_max'), t_end_s, steps, n_lit,
    burnt_fraction {obj name: area fraction ever burning}, max_T_g, stats {coupled stats()}}``."""
    cfg = cfg or load_yaml("scenarios")
    table = table or MaterialTable.load()
    t_max = float(cfg["t_max_s"] if t_max_s is None else t_max_s)
    check = int(cfg["dt_check_steps"])
    snap = snapshot_from_primitives(sc.objects, scenario_grid_cfg(cfg), sc.seed, table)
    p = sim_params_with(params, wind_speed=sc.wind_speed)
    sim = FireSimulation(snap, table, p, app_cfg)
    tgt = snap.obj_id == sc.target_obj
    if target_material is not None:
        sim.set_material(sc.target_obj, MATERIAL_CLASSES.index(target_material))
    n_lit = ignite_object_patch(sim, snap, sc)
    if n_lit == 0:
        raise RuntimeError(f"scenario {sc.name}: the ignition lit no element")
    mid = sim.surface.material_id
    t_ign_el = np.where(table.flammable[mid], table.T_ign[mid], np.inf)
    t_ign_min = float(np.min(t_ign_el))
    ever = sim.surface.burning.numpy() > 0
    max_tg, t_ignite, end = 0.0, None, "t_max"
    n_steps = round(t_max / sim.dt_sim)
    k = 0
    while k < n_steps:
        sim.step(1)
        k += 1
        burning = sim.surface.burning.numpy() > 0
        if t_ignite is None and burning[tgt].any():
            t_ignite = k * sim.dt_sim
            if stop_on_ignition:
                ever |= burning
                end = "ignited"
                break
        if k % check == 0:
            ever |= burning
            tg_now = float(sim.gas.T_g.numpy().max())
            max_tg = max(max_tg, tg_now)
            if not burning.any() and tg_now < t_ign_min and not (sim.surface.T_s.numpy() >= t_ign_el).any():
                end = "extinct"
                break
    ever |= sim.surface.burning.numpy() > 0
    max_tg = max(max_tg, float(sim.gas.T_g.numpy().max()))
    area = np.asarray(snap.area, np.float64)
    frac = {}
    for o in snap.objects:
        m = snap.obj_id == o.obj_id
        frac[o.name] = float(area[ever & m].sum() / area[m].sum())
    return {"t_ignite_s": t_ignite, "ignited": t_ignite is not None, "end": end, "t_end_s": k * sim.dt_sim,
            "steps": k, "n_lit": int(n_lit), "burnt_fraction": frac, "max_T_g": max_tg, "stats": sim.stats()}  # fmt: skip


def ignite_object_patch(sim: FireSimulation, snap: Any, sc: Scenario) -> int:
    """Light the flammable elements of object ``sc.emitter_obj`` within ``sc.ignite_radius`` [m] of ``sc.ignite_point``
    (one element at a time through the public ``SurfaceFireModel.ignite``); returns the number lit."""
    d = np.linalg.norm(snap.pos.astype(np.float64) - np.asarray(sc.ignite_point, float), axis=1)
    n = 0
    for i in np.flatnonzero((snap.obj_id == sc.emitter_obj) & (d <= sc.ignite_radius)):
        n += sim.surface.ignite(snap.pos[i].astype(np.float64), 1e-9)
    return n


def ordered_pairs(table: MaterialTable, names: tuple[str, ...] = _FLAMMABLE) -> list[tuple[str, str]]:
    """Ordered pairs ``(A, B)`` of flammable materials for which A must ignite no later than B (P2R9): all of
    ``T_ign``, ``C'' = rho c delta``, ``t_delay``, ``k delta`` of A are ``<=`` those of B, ``eps_A >= eps_B``,
    with at least one strict. Computed from the table, so retuning it updates the assertions."""
    keys = lambda i: np.array([table.T_ign[i], table.C_areal[i], table.t_delay[i], table.k[i] * table.delta[i]])
    out = []
    for a in names:
        for b in names:
            if a == b:
                continue
            ia, ib = table.material_id(a), table.material_id(b)
            ka, kb = keys(ia), keys(ib)
            no_worse = bool(np.all(ka <= kb) and table.eps[ia] >= table.eps[ib])
            strict = bool(np.any(ka < kb) or table.eps[ia] > table.eps[ib])
            if no_worse and strict:
                out.append((a, b))
    return out


def unordered_pairs(table: MaterialTable, names: tuple[str, ...] = _FLAMMABLE) -> list[tuple[str, str]]:
    """Unordered material pairs (sorted by name order in ``names``) that ``ordered_pairs`` orders in neither
    direction (reported, not asserted)."""
    ordered = set(ordered_pairs(table, names))
    return [(a, b) for i, a in enumerate(names) for b in names[i + 1 :] if (a, b) not in ordered and (b, a) not in ordered]


def monotonicity_violations(
    times: dict[str, float | None], pairs: list[tuple[str, str]], dt: float
) -> list[tuple[str, str]]:
    """Pairs ``(A, B)`` violated by ``times`` (material -> first-ignition time [s], None = never = +inf): a
    violation is ``t_A > t_B + dt`` (``dt`` = ``dt_sim``). A pair with a material missing from ``times`` is skipped."""
    inf = float("inf")
    t = {m: (inf if v is None else float(v)) for m, v in times.items()}
    return [(a, b) for a, b in pairs if a in t and b in t and t[a] > t[b] + dt]


def uniform_table(entries: dict[str, dict[str, Any]] | None = None) -> tuple[MaterialTable, int]:
    """Uniform-material table (P2R12): the four flammable classes are replaced by one entry holding the arithmetic
    mean of their parameters (``consumed_when_burnt: false``, status artistic); the 8-class order and the other four
    classes are unchanged. ``entries`` = parsed ``materials.yaml`` (default: the committed file). Returns
    ``(table, uniform_id)``; ``uniform_id`` is the ``paper`` id (any of the four flammable ids is the uniform class)."""
    if entries is None:
        entries = load_yaml_text((config_dir() / "materials.yaml").read_text())
    ent = copy.deepcopy(entries)
    uni: dict[str, Any] = {f: float(np.mean([entries[m][f] for m in _FLAMMABLE])) for f in _UNIFORM_FIELDS}
    uni.update(flammable=True, consumed_when_burnt=False, status="artistic")
    for m in _FLAMMABLE:
        ent[m] = dict(uni)
    return MaterialTable(ent), MATERIAL_CLASSES.index("paper")
