"""CLI: ``python -m livefire.app run ...`` and ``python -m livefire.app replay <run_dir>``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "calib":  # calibration tool has its own sub-parser
        from livefire.app.calib_cli import main as calib_main

        return calib_main(raw[1:])
    ap = argparse.ArgumentParser(prog="livefire.app", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the synthetic scene and record a composite video")
    r.add_argument("--scene", default="table_v1")
    r.add_argument("--seconds", type=float, default=8.0, help="wall s (real-time) or simulated s (--offline)")
    r.add_argument("--frames", type=int, default=None, help="exact number of output frames (overrides --seconds)")
    r.add_argument("--out", default="out/demo")
    r.add_argument("--grid", type=int, default=None, help="gas grid G override (default: configs/grid.yaml)")
    r.add_argument("--fps", type=float, default=None)
    r.add_argument("--offline", action="store_true", help="step the sim a fixed count per frame (not real-time)")
    r.add_argument("--ignite", action="append", default=[], metavar="x,y,z@t[:r]")
    r.add_argument("--override", action="append", default=[], metavar="obj_id=material@t")
    p = sub.add_parser("replay", help="re-run a recorded run (sim only) and compare the final digest")
    p.add_argument("run_dir")
    p.add_argument("--tolerance", action="store_true",
                   help="compare final arrays with rtol=1e-5 atol=1e-6 instead of the exact digest (default on GPU)")
    p.add_argument("--render", action="store_true", help="also render the final two views to PNG")
    i = sub.add_parser("interactive", help="interactive viewer: click to ignite (synthetic scene)")
    i.add_argument("--scene", default="table_v1")
    i.add_argument("--grid", type=int, default=None, help="gas grid G override (default: configs/grid.yaml)")
    i.add_argument("--out", default="out/interactive", help="directory for screenshots (and the record)")
    i.add_argument("--record", action="store_true", help="record the session (events, config, digest) to --out")
    i.add_argument("--max-width", type=int, default=None, help="window width limit px (default: screen size)")
    i.add_argument("--max-height", type=int, default=None, help="window height limit px (default: screen size)")
    i.add_argument("--frames", type=int, default=None, help="exit after N frames (smoke testing)")
    lv = sub.add_parser("live", help="live cameras: acquire the scene, then click to ignite on the live feed")
    lv.add_argument("--cameras", default=None, help="cameras yaml (default configs/cameras.yaml)")
    lv.add_argument("--calib", default=None, help="calibration root (default data/calib, active version)")
    lv.add_argument("--depth", default=None, help="depth backend name from configs/depth.yaml")
    lv.add_argument("--grid", type=int, default=None, help="gas grid G override")
    lv.add_argument("--out", default="out/live", help="directory for screenshots")
    lv.add_argument("--max-width", type=int, default=None)
    lv.add_argument("--max-height", type=int, default=None)
    lv.add_argument("--frames", type=int, default=None, help="exit after N frames (smoke testing)")
    from livefire.app.capture_cli import add_subparsers, run_command

    add_subparsers(sub)
    argv = raw
    for i in range(len(argv) - 1, 0, -1):  # "--ignite -0.075,..." would be read as a flag: glue to "--ignite=..."
        if argv[i - 1] in ("--ignite", "--override"):
            argv[i - 1 : i + 1] = [f"{argv[i - 1]}={argv[i]}"]
    a = ap.parse_args(argv)

    if a.cmd in ("capture", "latency-timer"):
        return run_command(a)
    if a.cmd == "live":
        from livefire.app.live import run_live

        return run_live(a.cameras, a.calib, a.depth, a.grid, Path(a.out), a.max_width, a.max_height, a.frames)
    if a.cmd == "interactive":
        from livefire.app.window import run_interactive

        return run_interactive(a.scene, a.grid, Path(a.out), a.max_width, a.max_height, a.frames, a.record)

    from livefire.app.replay import parse_ignite, parse_override, replay
    from livefire.app.runner import RunConfig, Runner

    if a.cmd == "run":
        events = [parse_ignite(s) for s in a.ignite] + [parse_override(s) for s in a.override]
        cfg = RunConfig(a.scene, a.seconds, a.frames, Path(a.out), a.grid, a.offline, a.fps, events)
        s = Runner(cfg).run()
        st = s["stages"]
        print(f"run: {s['frames']} frames in {s['wall_s']:.1f} s wall, sim t {s['t_sim']:.3f} s, rt {s['rt_factor']:.3f}x, "
              f"frame median {st['frame']['median_ms']:.0f} ms p95 {st['frame']['p95_ms']:.0f} ms, "
              f"degradation L{s['final_degradation_level']}  [gpu={s['gpu']}, {s['framework']}]")
        print(f"digest {s['digest']}\noutput {Path(a.out) / 'composite.mp4'}")
        return 0
    res = replay(Path(a.run_dir), render=a.render, force_tolerance=a.tolerance)
    print(f"replay: {res['steps']} steps, {res['mode']} {'MATCH' if res['ok'] else 'MISMATCH'} ({res['detail']})")
    if not res["ok"]:
        print(f"expected {res['expected']}\nactual   {res['actual']}", file=sys.stderr)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
