## Task 9: P0-009 Volume renderer and two-view compositor

**Files:** `src/livefire/render/volume.py` (ray march kernel), `src/livefire/render/blackbody.py` (LUT),
`src/livefire/render/overlay.py` (char/removed splat into per-view buffers), `src/livefire/render/compositor.py`
(`Compositor` implementing `RendererBackend`), `configs/render.yaml`, `tests/test_render.py`.

- Per view, per pixel of a `volume_scale` resolution buffer (default 0.5): ray from the camera centre through the
  pixel centre in world space (use geometry helpers; K scaled for the reduced resolution), intersect the ROI AABB,
  march from `t_near` to `min(t_far, t_occ)` where `t_occ` = `view_depth` converted to ray distance
  (`zdepth_to_raydist`) minus a feather band `feather_m`; hole pixels march nothing. Fixed step `step_factor * h`,
  early exit when transmittance < `t_min`. Sampling of T_g and soot is trilinear in world space.
- Emission: blackbody LUT (Planck spectrum integrated against CIE 1931 2-degree CMFs, converted to linear sRGB, each
  colour normalised to unit luminance), 64 entries from 800 K to 3000 K; radiance
  `L = emission_scale * lut(T) * clamp((T - T_emit_min) / (T_emit_ref - T_emit_min), 0, inf) ^ emission_power`.
  Soot: absorption `sigma_a = k_abs * soot`, single-scatter ambient term `soot_albedo * ambient`. Emission is
  absorbed by soot along the ray. All noise (none in P0) would be world-space: no screen-space effects.
- Overlay: splat `char` and `removed` from elements into each view (disc splat like Task 6, accepted only where
  `|z_elem - view_depth| < depth_tol`); composite base = `frame_linear * (1 - char_dark * char)` then mix towards
  `ash_color` by `removed`.
- Composite: sRGB -> linear, base overlay, `out = base * T_vol + L_vol`, fixed exposure `exposure`, simple
  tone curve `x / (1 + x)` applied only to the added light? Use: `out = base * T + tonemap(L)` where
  `tonemap(L) = L / (1 + L)`; clamp; linear -> sRGB uint8. Upsample the volume buffers (T, L) bilinearly to full res.
- `configs/render.yaml`: `volume_scale 0.5`, `step_factor 0.5`, `t_min 0.01`, `feather_m 0.01 m`, `emission_scale 4.0`,
  `T_emit_min 700 K`, `T_emit_ref 1400 K`, `emission_power 2.0`, `k_abs 2000 m^2/kg`, `soot_albedo 0.2`,
  `ambient 0.1`, `char_dark 0.85`, `ash_color [0.12, 0.11, 0.10]`, `depth_tol 0.02 m`, `exposure 1.0`; all artistic,
  `{value, unit, range, status}` form (lists allowed as value).
- Tests: **cross-view consistency (M7 analogue on the synthetic rig):** for 5 seeded 3D points above the table that
  are visible from both cameras, set a Gaussian hot blob (T_g peak 1600 K, sigma 2 h) at the point on an otherwise
  T_amb field with no soot, render each view at volume_scale 1.0, compute the luminance-weighted centroid of `L_vol`
  and compare with `project(calib, point)`: median error < 6 px at 720p (report the actual median and max);
  **occlusion:** a blob placed just behind the wood block as seen from cam0 (choose the point by casting the cam0
  ray through the block centre and going 0.1 m past the far face) while visible from cam1: cam0 luminance inside a
  15 px radius of its projection is < 5 % of cam1's peak region sum; holes: forcing a hole mask suppresses emission;
  blackbody LUT: 1000 K is reddish (R > G > B), 3000 K has B/R higher than 1000 K; compositing round trip: no fire,
  no char -> output equals input frame within 1 LSB.

**Acceptance:** tests pass; `python -m livefire.render.compositor --demo` writes two composited PNGs with a
prescribed blob to `out/`.

