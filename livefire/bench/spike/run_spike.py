"""GPU framework spike runner: Warp vs Taichi on advection and conduction gather.

Usage: python bench/spike/run_spike.py [--device cpu|cuda:0] [--out CSV]
Device defaults to configs/app.yaml `device`. CPU numbers are labelled gpu=none, framework=<fw>-<ver>-cpu.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common
import numpy as np

from livefire.common.config import load_yaml

ADVECT_G = (64, 96, 128)
CONDUCTION_N = (20_000, 100_000)
K = 16
DT_ADV, ROI_SIZE = 1.0 / 60.0, 3.0  # s; m (default ROI edge 3.0 m, h = ROI/G)
DT_COND = 0.01
WARMUP, REPEATS, SAMPLES = 5, 5, 20
WORKLOAD = "spike-v1"


def _row(base: dict, fw: str, device: str, **kw: object) -> dict:
    return {**base, "framework": f"{fw}-cpu" if device == "cpu" else fw, **kw}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: configs/app.yaml device)")
    ap.add_argument("--out", default=None, help="CSV path (default bench/results/spike_<dev>_<date>.csv)")
    ap.add_argument("--frameworks", default="warp,taichi")
    args = ap.parse_args(argv)
    cfg = load_yaml("app")
    device = args.device or cfg.get("device", "cpu")
    seed = int(cfg.get("seed", 1234))
    today = datetime.date.today().isoformat()
    dev_tag = "cpu" if device == "cpu" else "gpu"
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "results" / f"spike_{dev_tag}_{today}.csv"

    base = {
        "run_id": f"spike-{dev_tag}-{today}", "date": today, "git_sha": common.git_sha(),
        "workload_version": WORKLOAD, **common.gpu_info(device), **common.env_info(),
        "scene": "spike_synthetic_random", "views": 0, "mode": "spike", "degradation_level": 0,
        "timing_method": "perf_counter_ns+sync", "warmup_frames": WARMUP,
    }  # fmt: skip
    rows: list[dict] = []
    fws = args.frameworks.split(",")
    cold_dir = tempfile.mkdtemp(prefix="warp_cold_cache_")

    for fw in fws:
        if fw == "warp":
            import warp_kernels as wk

            wk.init(device, cold_cache_dir=cold_dir)
            ver, tag = wk.version(), f"warp-{wk.version()}"
            mk_adv = lambda s, v, h: wk.WarpAdvect(device, s, v, DT_ADV, h)
            mk_con = lambda t, c, n, g: wk.WarpConduction(device, t, c, n, g, DT_COND)
            extra = "kernel cache redirected to empty dir (cold JIT)"
        elif fw == "taichi":
            import taichi_kernels as tk

            tk.init(device, cold_cache=True)
            ver, tag = tk.version(), f"taichi-{tk.version()}"
            mk_adv = lambda s, v, h: tk.TaichiAdvect(s, v, DT_ADV, h)
            mk_con = lambda t, c, n, g: tk.TaichiConduction(t, c, n, g, DT_COND)
            extra = "offline cache off (cold JIT)"
        else:
            raise SystemExit(f"unknown framework {fw}")
        _ = ver
        for g in ADVECT_G:
            s, v = common.make_advect_inputs(g, seed)
            r = mk_adv(s, v, ROI_SIZE / g)
            t = common.time_runner(r.run, r.sync, WARMUP, REPEATS, SAMPLES)
            rows.append(_row(base, tag, device, grid_g=g, stage="advect_rk2", variant="rk2_trilinear_clamp",
                             samples=int(t["n"]), median_ms=round(t["median_ms"], 4),
                             p95_ms=round(t["p95_ms"], 4), p99_ms=round(t["p99_ms"], 4),
                             notes=f"first_launch_ms={t['first_launch_ms']:.1f} (incl. JIT; {extra}); "
                                   f"repeat_medians_ms=[{t['rep_median_min']:.3f},{t['rep_median_max']:.3f}]; "
                                   f"dt=1/60 h=3/G"))  # fmt: skip
            print(rows[-1]["framework"], "advect G=", g, f"{t['median_ms']:.3f} ms", flush=True)
        for n in CONDUCTION_N:
            T, C, nb, gij = common.make_conduction_inputs(n, K, seed)
            r = mk_con(T, C, nb, gij)
            t = common.time_runner(r.run, r.sync, WARMUP, REPEATS, SAMPLES)
            rows.append(_row(base, tag, device, surface_elements=n, stage="conduction_gather", variant=f"K={K}",
                             samples=int(t["n"]), median_ms=round(t["median_ms"], 4),
                             p95_ms=round(t["p95_ms"], 4), p99_ms=round(t["p99_ms"], 4),
                             notes=f"first_launch_ms={t['first_launch_ms']:.1f} (incl. JIT; {extra}); "
                                   f"repeat_medians_ms=[{t['rep_median_min']:.3f},{t['rep_median_max']:.3f}]; "
                                   f"neighbour window +-256, dt=0.01"))  # fmt: skip
            print(rows[-1]["framework"], "conduction N=", n, f"{t['median_ms']:.3f} ms", flush=True)
    if device == "cpu":
        for r in rows:
            r["notes"] += f"; CPU only, cores={__import__('os').cpu_count()}"
    common.write_rows(out, rows)
    print("wrote", out)
    _ = np
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
