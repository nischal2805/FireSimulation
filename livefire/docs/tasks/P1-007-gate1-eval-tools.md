# P1-007 Gate 1 evaluation tools and camera placement study (task brief, copied from the controller plan)

## Task 7: P1-007 Gate 1 evaluation tools and camera placement study

**Files:** `eval/{alignment.py,geometry_mae.py,placement.py,latency.py}`, `src/livefire/app/measure_cli.py`,
`docs/RUNBOOK_P1.md`, `tests/test_eval_p1.py`.

- M6 alignment (`eval/alignment.py`): with the board (or AprilTags at known positions) visible after acquisition,
  compare reconstructed landmark positions (from the fused geometry along each landmark's pixel ray) with known
  world positions; median/p95 error CSV.
- M4 geometry MAE (`measure` CLI): the user clicks pairs of points in a view (snapshot back-projection) and types the
  tape-measured distance; the tool stores the pairs and reports MAE in cm; also AbsRel/completeness on synthetic.
- Placement study (`placement.py`, PRD 7.1): from calibration + first acquisition: angle between cameras, shared
  visible table area fraction, depth-valid coverage per view; fail loudly with concrete guidance if below thresholds
  (`placement.min_shared_fraction`, `min_valid_coverage`, angle outside 40-100 deg).
- Latency eval: parse the live per-frame CSV into M2 statistics; glass-to-glass procedure doc.
- `docs/RUNBOOK_P1.md`: print board, measure squares, place cameras (60-90 deg apart, 1-1.5 m from table), phone IP
  camera setup, calibrate intrinsics per camera, extrinsics, run live, tag materials, click to ignite, run evals,
  which files to send back.
- Tests on synthetic data for each eval (known answers).

**Acceptance:** tests pass; runbook complete.

