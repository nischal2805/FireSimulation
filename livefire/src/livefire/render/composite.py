"""Device-side compositing kernels (NVIDIA Warp) for ``Compositor.render`` (PRD 7.4).

Same maths as the host reference (``OverlayMap.splat`` and the NumPy pipeline in ``compositor.py``):

* ``splat_kernel``: per pixel, mean of (char, removed) over the pixel's accepted elements (CSR list built once
  per camera by ``OverlayMap``).
* ``fill_kernel``: isolated gaps filled by the normalised 5x5 box convolution of ``OverlayMap.splat``.
* ``composite_kernel``: base = linear(frame) darkened by char and mixed to ash by removed; bilinear upsample of the
  reduced-resolution volume (OpenCV ``INTER_LINEAR`` sampling: half-pixel centres, replicated border);
  ``out = base * T_vol + tonemap(exposure * L_vol)``; sRGB encode to uint8.

Rounding of the final sRGB value is half away from zero (NumPy ``rint`` is half to even), so outputs can differ
from the host reference by at most 1 LSB on exact ties.
"""

from __future__ import annotations

import warp as wp

from livefire.render import overlay

wp.config.quiet = True

FILL_RADIUS = wp.constant(overlay.FILL_RADIUS)


@wp.kernel
def splat_kernel(
    start: wp.array(dtype=wp.int32),  # (H*W+1,) CSR row pointers per pixel
    elem: wp.array(dtype=wp.int32),  # accepted element index per CSR entry
    inv_count: wp.array(dtype=wp.float32),  # (H*W,)
    char: wp.array(dtype=wp.float32),  # (N_e,)
    removed: wp.array(dtype=wp.float32),
    w: int,
    out: wp.array2d(dtype=wp.vec2),  # (H,W) (char, removed)
):
    v, u = wp.tid()
    p = v * w + u
    sc = float(0.0)  # noqa: UP018 (Warp: mutable variable)
    sr = float(0.0)  # noqa: UP018 (Warp: mutable variable)
    for k in range(start[p], start[p + 1]):
        e = elem[k]
        sc += char[e]
        sr += removed[e]
    out[v, u] = wp.vec2(sc, sr) * inv_count[p]


@wp.kernel
def fill_kernel(
    src: wp.array2d(dtype=wp.vec2),
    fill: wp.array2d(dtype=wp.uint8),
    den: wp.array2d(dtype=wp.float32),  # blur of the coverage mask (OverlayMap._den)
    h: int,
    w: int,
    out: wp.array2d(dtype=wp.vec2),
):
    v, u = wp.tid()
    if fill[v, u] == wp.uint8(0):
        out[v, u] = src[v, u]
        return
    acc = wp.vec2(0.0, 0.0)
    for dv in range(-FILL_RADIUS, FILL_RADIUS + 1):
        for du in range(-FILL_RADIUS, FILL_RADIUS + 1):
            vv = v + dv
            uu = u + du
            if vv >= 0 and vv < h and uu >= 0 and uu < w:  # constant (zero) border, as cv2.BORDER_CONSTANT
                acc += src[vv, uu]
    n = float((2 * FILL_RADIUS + 1) * (2 * FILL_RADIUS + 1))
    out[v, u] = (acc / n) / wp.max(den[v, u], 1.0e-6)


@wp.func
def _src_coord(x: int, scale: float, n: int):
    """OpenCV INTER_LINEAR source index and weight for destination index x (replicated border)."""
    s = (float(x) + 0.5) * scale - 0.5
    i0 = int(wp.floor(s))
    f = s - float(i0)
    if i0 < 0:
        i0 = 0
        f = 0.0
    if i0 >= n - 1:
        i0 = n - 1
        f = 0.0
    return i0, f


@wp.func
def _srgb_u8(x: float):
    c = wp.clamp(x, 0.0, 1.0)
    s = float(0.0)  # noqa: UP018 (Warp: mutable variable)
    if c <= 0.0031308:
        s = 12.92 * c
    else:
        s = 1.055 * wp.pow(c, 1.0 / 2.4) - 0.055
    return wp.uint8(int(wp.round(s * 255.0)))


@wp.kernel
def composite_kernel(
    rgb: wp.array3d(dtype=wp.uint8),  # (H,W,3) sRGB frame
    s2l: wp.array(dtype=wp.float32),  # (256,) sRGB -> linear
    cr: wp.array2d(dtype=wp.vec2),  # (H,W) splatted (char, removed)
    vol_T: wp.array2d(dtype=wp.float32),  # (Hs,Ws) transmittance
    vol_L: wp.array2d(dtype=wp.vec3),  # (Hs,Ws) radiance
    has_vol: int,  # 0: no active volume (T = 1, L = 0)
    sx: float,  # Ws / W
    sy: float,  # Hs / H
    char_dark: float,
    ash: wp.vec3,
    exposure: float,
    out: wp.array3d(dtype=wp.uint8),
):
    v, u = wp.tid()
    c = cr[v, u]
    base = wp.vec3(s2l[int(rgb[v, u, 0])], s2l[int(rgb[v, u, 1])], s2l[int(rgb[v, u, 2])])
    base = base * (1.0 - char_dark * c[0])
    base = base + (ash - base) * c[1]
    T = float(1.0)  # noqa: UP018 (Warp: mutable variable)
    L = wp.vec3(0.0, 0.0, 0.0)
    if has_vol != 0:
        hs = vol_T.shape[0]
        ws = vol_T.shape[1]
        x0, fx = _src_coord(u, sx, ws)
        y0, fy = _src_coord(v, sy, hs)
        x1 = wp.min(x0 + 1, ws - 1)
        y1 = wp.min(y0 + 1, hs - 1)
        T = (vol_T[y0, x0] * (1.0 - fx) + vol_T[y0, x1] * fx) * (1.0 - fy) + (
            vol_T[y1, x0] * (1.0 - fx) + vol_T[y1, x1] * fx) * fy
        L = (vol_L[y0, x0] * (1.0 - fx) + vol_L[y0, x1] * fx) * (1.0 - fy) + (
            vol_L[y1, x0] * (1.0 - fx) + vol_L[y1, x1] * fx) * fy
    Le = L * exposure
    o = base * T + wp.cw_div(Le, Le + wp.vec3(1.0, 1.0, 1.0))
    out[v, u, 0] = _srgb_u8(o[0])
    out[v, u, 1] = _srgb_u8(o[1])
    out[v, u, 2] = _srgb_u8(o[2])
