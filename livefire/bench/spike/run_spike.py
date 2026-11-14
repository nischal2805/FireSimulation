"""GPU framework spike runner: Warp vs Taichi on advection and conduction gather.

Usage: python bench/spike/run_spike.py [--device cpu|cuda:0] [--out CSV]
Device defaults to configs/app.yaml `device`. CPU numbers are labelled gpu=none, framework=<fw>-<ver>-cpu.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spike_common as common

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


def _backend(fw: str, device: str, cold_dir: str):
    """Return (framework tag, advect factory, conduction factory, cache note, peak-vram fn, cold kernels)."""
    if fw == "warp":
        import warp_kernels as wk

        wk.init(device, cold_cache_dir=cold_dir)
        return (
            f"warp-{wk.version()}",
            lambda s, v, h: wk.WarpAdvect(device, s, v, DT_ADV, h),
            lambda t, c, n, g: wk.WarpConduction(device, t, c, n, g, DT_COND),
            "kernel cache redirected to empty dir",
            lambda: wk.peak_vram_mb(device),
            # one module holds both kernels: only the very first launch compiles
            {"advect"},
        )
    if fw == "taichi":
        import taichi_kernels as tk

        tk.init(device, cold_cache=True)
        return (
            f"taichi-{tk.version()}",
            lambda s, v, h: tk.TaichiAdvect(s, v, DT_ADV, h),
            lambda t, c, n, g: tk.TaichiConduction(t, c, n, g, DT_COND),
            "offline cache off",
            lambda: None,  # no clean peak-memory API in Taichi 1.7.4 (none found)
            {"advect", "conduction"},
        )
    raise SystemExit(f"unknown framework {fw}")


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
    results = Path(__file__).resolve().parents[1] / "results"
    out = Path(args.out) if args.out else results / f"spike_{dev_tag}_{today}.csv"

    info = common.gpu_info(device)
    clocks = info.pop("clocks")
    base = {
        "run_id": f"spike-{dev_tag}-{today}", "date": today, "git_sha": common.git_sha(),
        "workload_version": WORKLOAD, **info, **common.env_info(),
        "scene": "spike_synthetic_random", "views": 0, "mode": "spike", "degradation_level": 0,
        "timing_method": "perf_counter_ns+sync", "warmup_frames": WARMUP,
    }  # fmt: skip
    extra_note = f"CPU only, cores={os.cpu_count()}" if device == "cpu" else f"clocks: {clocks}"
    rows: list[dict] = []
    cold_dir = tempfile.mkdtemp(prefix="warp_cold_cache_")

    for fw in args.frameworks.split(","):
        try:
            tag, mk_adv, mk_con, cache_note, peak_mb, cold_kernels = _backend(fw, device, cold_dir)
        except ImportError as e:
            print(f"SKIPPING framework {fw!r}: cannot import it ({e}). For taichi run `uv sync --all-groups "
                  "--all-extras`.", flush=True)
            continue
        cases = [("advect", g) for g in ADVECT_G] + [("conduction", n) for n in CONDUCTION_N]
        for kernel, size in cases:
            if kernel == "advect":
                s, v = common.make_advect_inputs(size, seed)
                runner = mk_adv(s, v, ROI_SIZE / size)
                ids = {"grid_g": size, "stage": "advect_rk2", "variant": "rk2_trilinear_clamp"}
                what = f"dt=1/60 h=3/G G={size}"
            else:
                runner = mk_con(*common.make_conduction_inputs(size, K, seed))
                ids = {"surface_elements": size, "stage": "conduction_gather", "variant": f"K={K}"}
                what = "neighbour window +-256 dt=0.01 (MOCK random graph)"
            t = common.time_runner(runner.run, runner.sync, WARMUP, REPEATS, SAMPLES)
            if kernel in cold_kernels:
                cold_kernels.discard(kernel)
                launch = f"cold JIT first launch ({cache_note})"
            else:
                launch = "warm first launch (kernel already compiled in this process)"
            vram = peak_mb()
            notes = [
                f"first_launch_ms={t['first_launch_ms']:.1f}: {launch}",
                f"repeat_medians_ms=[{t['rep_median_min']:.3f},{t['rep_median_max']:.3f}]",
                what,
                extra_note,
            ]
            if device != "cpu":
                notes.append(
                    "peak_vram_alloc_mb = Warp mempool high-water (process-cumulative; reserved not available)"
                    if vram is not None
                    else "peak VRAM not recorded (no clean Taichi API)"
                )
            rows.append(_row(
                base, tag, device, **ids, samples=int(t["n"]), median_ms=round(t["median_ms"], 4),
                p95_ms=round(t["p95_ms"], 4), p99_ms=round(t["p99_ms"], 4),
                peak_vram_alloc_mb="" if vram is None else round(vram, 1), notes="; ".join(notes),
            ))  # fmt: skip
            common.write_rows(out, rows)  # incremental: results survive a crash in a later framework/case
            print(rows[-1]["framework"], kernel, size, f"{t['median_ms']:.3f} ms", flush=True)
    if not rows:
        print("no framework could be benchmarked; nothing written")
        return 1
    common.write_rows(out, rows)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
