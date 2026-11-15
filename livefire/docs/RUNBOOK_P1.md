# P1 runbook: from a printed board to the Gate 1 numbers (Windows 11, RTX 4060, two cameras)

All fire is virtual: nothing here heats or ignites anything real. Run every command from the `livefire/` folder in a
terminal (PowerShell is fine). Each command has a `make` shortcut on Linux/macOS; on Windows use the `uv run ...` form
shown here (no bash is needed anywhere). If a step fails, the message says what to change; the "Troubleshooting" list at
the end covers the common cases. Nothing in this runbook has been run on real cameras yet: the tools were tested on
synthetic data with known answers, so please report anything that behaves oddly.

**What Gate 1 needs from you** (PRD 11.1, 12):

| Metric | What it is | Target | Tool |
| --- | --- | --- | --- |
| M6 fusion alignment | board corners reconstructed vs their known world positions | median error under 3 cm | `eval/alignment.py` |
| M4 geometry accuracy | reconstructed distance vs tape-measured distance, MAE | reported in cm (no fixed target in the PRD) | `measure` + `eval/geometry_mae.py` |
| M2 latency | `t_present - t_arrive` (PRD target 100 ms or less; the tool reports p50/p95/p99 and uses p95 as its pass/fail rule); glass-to-glass offset reported separately | see PRD 11.1 | `eval/latency.py` |
| Camera placement | angle, shared table area, depth coverage | pass the thresholds in `configs/placement.yaml` (untuned defaults) | `eval/placement.py` |

Plus a few screenshots of live mode with the geometry overlay (`o`) and a fire on your real objects.

## 1. Print the board and measure a square

```powershell
uv run python -m livefire.app calib board --out board.png --dpi 300
```

Print `board.png` on A4 **landscape at 100 % scale** ("actual size", not "fit to page"). Mount it flat on a stiff, flat
surface (foam board or a clipboard with no curl). With a ruler or calipers measure the side of one black square in
metres (for example 34.8 mm = `0.0348`) and write it into `configs/calibration.yaml` as `board.square_length_m`. Every
later number scales with this measurement, so measure across several squares and divide (for example 7 squares / 7).

## 2. Place the cameras

- Two cameras, **60 to 90 degrees apart** as seen from the table centre, **1 to 1.5 m** from the table centre, both
  looking at the same table area, high enough to see over your objects (30 to 60 degrees above the table).
- The table area you care about (default 1.6 m by 1.2 m around the board centre, `fusion_roi` in
  `configs/acquisition.yaml`) should be inside both views. Smaller scenes work: set the region later with `--region`.
- Fix the cameras so they cannot move (tripod, clamp). **Moving a camera after calibration means recalibrating.**
- Do not change camera resolution, zoom or focus mode after calibration. Lock autofocus and auto-exposure if the camera
  allows it.

## 3. Phone as an IP camera (optional second camera)

1. Install an IP camera app: Android "IP Webcam" (stream `http://<phone-ip>:8080/video`) or DroidCam
   (`http://<phone-ip>:4747/video`). Phone and PC must be on the same Wi-Fi (5 GHz is better).
2. Start the server in the app, set video resolution to 1280x720 and the lowest-latency quality setting, lock focus and
   exposure, and note the IP address it shows.
3. Edit `configs/cameras.yaml`: replace the placeholder `192.168.1.50` with that IP. The USB camera is `type: usb`,
   `source: 0` (device index; try 1 or 2 if you have several), `fourcc: MJPG` for 720p at 30 fps.
4. Check both feeds: `uv run python -m livefire.app capture`. You should see two live panes with fps near 30 and frame
   age well under 100 ms. `q` quits. Keep the **cam_id** numbers: calibration and live mode use them.

## 4. Calibrate intrinsics (once per camera)

```powershell
uv run python -m livefire.app calib intrinsics --cam 0
uv run python -m livefire.app calib intrinsics --cam 1 --source "http://<phone-ip>:8080/video"
```

Hold the board in front of the camera and move it over the whole image with varied tilt (also near the corners and at
different distances). Captures are taken automatically when the board is still; `s` forces one, `f` finishes (at least 20
views). Aim for a held-out RMS below 1 px (the report prints it). Repeat for every camera.

## 5. Calibrate extrinsics (the world frame)

Lay the board **flat on the table** where both cameras see it fully, then:

```powershell
uv run python -m livefire.app calib extrinsics --cams "0,1"
```

World origin = board centre, +X along the board's long side, +Z up, the table is the plane `z = 0`. The result is
`data/calib/calib_vNNN.yaml` and becomes active. **Leave the board where it is** until step 7 is finished: M6 compares
against the board's known position, so it must not move between this step and the acquisition below.

## 6. Put objects on the table

Put a few real objects around the board, inside the area both cameras see: a block of wood, a metal object, a sheet of
paper, a box. Variety of heights and a few objects standing 5 to 20 cm tall make the evaluation meaningful. From now on
**do not touch the cameras or the scene** until step 8 is done.

## 7. Acquire once, then run the placement study and M6

```powershell
uv run python -m livefire.app measure --acquire-only
```

Allow a minute or so (depth inference time depends on your machine; no GPU timing has been measured yet). It prints the bundle path, for example
`out/eval/acq_20261120_101500/acquisition.npz`, and writes `view_cam0.png`, `view_cam1.png`, `depth_cam0.png` and
`depth_cam1.png` next to it. Use `--depth da2_relative_small` to try the other depth model. Use that path for everything
that follows (shown below as `<BUNDLE>`).

**Placement study** (PRD 7.1):

```powershell
uv run python eval/placement.py --bundle <BUNDLE>
```

It prints the camera angle, distance, visible table fraction per camera, valid-depth coverage and the area seen by both
cameras, and exits with code 1 and concrete guidance ("move one camera ...") when a threshold of
`configs/placement.yaml` is violated. If the table is smaller than the default area, add
`--region="-0.5,-0.4,0.5,0.4"` (x0,y0,x1,y1 in metres from the board centre; note the `=` because of the minus sign). The
thresholds are untuned defaults: if it fails but the live overlay looks right, send the CSV and we will tune them. Fix
real problems first (angle, overlap, lighting) and re-run steps 7 onward.

**M6 alignment**:

```powershell
uv run python eval/alignment.py --bundle <BUNDLE> --label desk_scene1
```

It detects the board corners in the saved views, reconstructs each one through the frozen depth and prints the median and
p95 error in cm (target: median under 3 cm). Output: `eval/results/alignment_desk_scene1.csv` (summary) and
`alignment_desk_scene1_landmarks.csv` (every corner). Limitation: depth is anchored to the table plane over the board
footprint, so board corners on the table are not fully independent of the reconstruction. For an independent check
add landmarks that are **above the table**: note a few points of known world position (for example tape marks on the
top of a box whose height you measured; x,y from a ruler relative to the board centre: +X along the board's long side to the right and +Y toward the top edge of the printed page, when you read the board with its long side horizontal; +Z up)
and read their pixels from `view_cam0.png` / `view_cam1.png` in any image viewer. Put them in `landmarks.csv`:

```text
name,x,y,z,cam,u,v
box_corner_a,0.120,0.080,0.095,0,742,388
box_corner_a,0.120,0.080,0.095,1,611,402
```

and run `uv run python eval/alignment.py --bundle <BUNDLE> --landmarks landmarks.csv --label desk_scene1_elevated`.

## 8. M4 geometry accuracy: measure with a tape

```powershell
uv run python -m livefire.app measure --bundle <BUNDLE>
```

A window shows both frozen views. For each pair: **left click point A** on a real surface point, **click point B**, then
**type the distance you measured with a tape or ruler in centimetres** and press Enter. A, B may be in different views.
`u` undoes the last pair, `s` saves, `q` or Esc quits; pairs are saved after every change. Guidance:

- Take **at least 10 pairs**, 20 is better: a few short (5 to 10 cm), several medium (20 to 40 cm), a few long (60 cm or more).
- Use clear points: object corners and edges on top faces, marks on the table. Measure along a straight line between the
  two exact points you click (the distance in 3-D, not along the table when the points differ in height).
- Mix horizontal, vertical and diagonal distances, and use both cameras.
- A click on a "hole" (no depth) is rejected with a message; click a little to the side.

On quit it prints the MAE and the path of `pairs.csv`. Then:

```powershell
uv run python eval/geometry_mae.py --pairs out/eval/measure_<time>/pairs.csv --label desk_scene1
```

Output: `eval/results/geometry_mae_desk_scene1.csv` with MAE, median, p95, max, bias, RMSE and mean relative error (cm and %).

## 9. Live mode and M2 latency

```powershell
uv run python -m livefire.app live
```

Wait for READY/SIMULATING. Right click an object and press `1` to `8` to set its real material (1 paper, 2 cloth, 3 wood,
4 plastic, 5 metal, 6 glass, 7 ceramic, 8 non-flammable), left click an object to start a virtual fire, press `o` to
show the geometry overlay (it should hug the real objects), `s` for a screenshot. **Let it run for at least 60 s with a
fire burning and both cameras shown** (no window dragging, no other heavy programs), then press `q`. It prints the run
folder `out/live/live_<time>/` containing `latency.csv`, `events.jsonl`, `summary.json`, screenshots.

```powershell
uv run python eval/latency.py out/live/live_<time>/latency.csv --label rtx4060_dual
```

Output `eval/results/latency_rtx4060_dual.csv`: per scope (`all`, `cam0`, `cam1`) capture-to-display p50/p95/p99/max
(PRD target 100 ms or less; the tool's verdict column uses p95 against 100 ms), frame age, composite time, frame-time percentiles, delivered fps, dropped-frame rate
and simulation lag. The first 30 frames and non-simulating states are ignored (`--skip-frames`, `--states`).

### Glass-to-glass offset (measured separately, once per camera)

M2 starts at the host timestamp when a frame reaches the PC. The delay inside the camera and the transport before that
(exposure, USB or Wi-Fi) is the glass-to-glass offset and is reported separately, never added into M2.

1. Terminal A: `uv run python -m livefire.app latency-timer` (a window with a large millisecond counter; keep it visible).
2. Aim the camera under test at that window (screen filling a good part of the image; for the phone, hold it in front
   of the monitor), terminal B: `uv run python -m livefire.app capture --show-timer`.
3. The capture window shows the live counter and, inside the camera image, the counter as the camera saw it. Take a
   screenshot (Win+Shift+S) and read both numbers: **offset = live counter minus counter seen in the camera image**
   (counter wraps at 100 000 ms). Repeat at least 5 times per camera and write the numbers down. Do this for each camera.
4. Add them to the latency report:

```powershell
uv run python eval/latency.py out/live/live_<time>/latency.csv --label rtx4060_dual --glass "0=95,102,98,101,97" --glass "1=210,190,205,220,198"
```

The CSV then has `g2g_offset_ms` (median per camera) next to, not inside, the M2 columns.

## 10. Which files to send back

Send these (zip the folders if easier):

1. `eval/results/*.csv`: placement, alignment (summary and landmarks), geometry_mae, latency.
2. `out/eval/acq_*/`: `acquisition.npz`, `view_cam*.png`, `depth_cam*.png` of the scene you evaluated (lets us re-run
   every offline evaluation and check the numbers).
3. `out/eval/measure_*/pairs.csv` (your tape pairs, with the clicked pixels and reconstructed points).
4. `out/live/live_*/`: `latency.csv`, `summary.json`, `events.jsonl` and the screenshots (with the overlay on and a fire).
5. `data/calib/calib_vNNN.yaml` and its `calib_vNNN/manifest.json` (the active calibration).
6. `configs/cameras.yaml` and `configs/calibration.yaml` (with your measured square) and the GPU/driver version
   (`nvidia-smi`).
7. A photo of the rig (camera positions) and a sentence on the lighting.

## Troubleshooting

- `no calibration in data/calib`: run steps 4 and 5; the live and evaluation tools use the active calibration.
- `calibration is 1280x720, camera delivers ...`: the camera resolution changed since calibration; set it back or
  recalibrate.
- `placement ... FAILED`: read the guidance lines; usually move a camera so both see the table centre 60-90 degrees apart.
- `no landmarks: the board was not found`: the board must be flat, fully visible in at least one view, well lit and
  unmoved since step 5; re-acquire. With the board removed, use `--landmarks` instead.
- High M6 or M4 errors: check the square size measurement (every error scales with it), recalibrate the camera with
  the worst reprojection error, and compare `depth_cam*.png` for holes or streaks.
- `cannot open a window` in `measure`: no display was found; use `--from-csv` (pairs from pixel coordinates).
