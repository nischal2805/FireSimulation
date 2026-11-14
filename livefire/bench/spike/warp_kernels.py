"""NVIDIA Warp implementations of the spike kernels. Semantics: see numpy_ref.py.

Device is a Warp device string ("cpu" or "cuda:0"). Each runner owns its device arrays (uploaded once in
__init__); `run()` launches asynchronously, `sync()` blocks, `result()` copies the output to a NumPy array.
"""

from __future__ import annotations

import numpy as np
import warp as wp

wp.config.quiet = True

# --- kernel begin
@wp.func
def _corner(p: float, g: int):
    q = wp.clamp(p, 0.0, float(g - 1))
    i0 = wp.min(int(q), g - 2)
    return i0, q - float(i0)


@wp.func
def _sample_s(f: wp.array3d(dtype=wp.float32), p: wp.vec3, g: int):
    i, tx = _corner(p[0], g)
    j, ty = _corner(p[1], g)
    k, tz = _corner(p[2], g)
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@wp.func
def _sample_v(f: wp.array3d(dtype=wp.vec3), p: wp.vec3, g: int):
    i, tx = _corner(p[0], g)
    j, ty = _corner(p[1], g)
    k, tz = _corner(p[2], g)
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@wp.kernel
def advect_kernel(
    scalar: wp.array3d(dtype=wp.float32),
    vel: wp.array3d(dtype=wp.vec3),
    out: wp.array3d(dtype=wp.float32),
    dt: float,
    h: float,
    g: int,
):
    i, j, k = wp.tid()
    x = wp.vec3(float(i), float(j), float(k))
    c = dt / h
    x_mid = x - 0.5 * c * vel[i, j, k]
    u_mid = _sample_v(vel, x_mid, g)
    x_back = x - c * u_mid
    out[i, j, k] = _sample_s(scalar, x_back, g)


@wp.kernel
def conduction_kernel(
    temp: wp.array(dtype=wp.float32),
    cap: wp.array(dtype=wp.float32),
    neighbors: wp.array2d(dtype=wp.int32),
    gij: wp.array2d(dtype=wp.float32),
    out: wp.array(dtype=wp.float32),
    dt: float,
    kmax: int,
):
    i = wp.tid()
    ti = temp[i]
    acc = float(0.0)  # noqa: UP018 (Warp needs an explicit mutable float)
    for s in range(kmax):
        j = neighbors[i, s]
        if j >= 0:
            acc += gij[i, s] * (temp[j] - ti)
    out[i] = ti + dt * acc / cap[i]
# --- kernel end


class _Runner:
    device: str

    def sync(self) -> None:
        """Block until all work on the device finished (no-op on CPU)."""
        wp.synchronize_device(self.device)


class WarpAdvect(_Runner):
    """Advection runner. Inputs host f32 arrays: scalar (G,G,G), vel (G,G,G,3)."""

    def __init__(self, device: str, scalar: np.ndarray, vel: np.ndarray, dt: float, h: float) -> None:
        self.device, self.dt, self.h, self.g = device, float(dt), float(h), scalar.shape[0]
        self.scalar = wp.array(scalar, dtype=wp.float32, device=device)
        self.vel = wp.array(vel, dtype=wp.vec3, device=device)
        self.out = wp.zeros(scalar.shape, dtype=wp.float32, device=device)

    def run(self) -> None:
        """Launch one advection step (asynchronous on CUDA)."""
        wp.launch(
            advect_kernel,
            dim=self.scalar.shape,
            inputs=[self.scalar, self.vel, self.out, self.dt, self.h, self.g],
            device=self.device,
        )

    def result(self) -> np.ndarray:
        """Output field as a NumPy array (syncs)."""
        self.sync()
        return self.out.numpy()


class WarpConduction(_Runner):
    """Conduction gather runner. temp, cap (N,) f32; neighbors (N,K) i32 (-1 pad); gij (N,K) f32."""

    def __init__(
        self, device: str, temp: np.ndarray, cap: np.ndarray, neighbors: np.ndarray, gij: np.ndarray, dt: float
    ) -> None:
        self.device, self.dt, self.k = device, float(dt), neighbors.shape[1]
        self.temp = wp.array(temp, dtype=wp.float32, device=device)
        self.cap = wp.array(cap, dtype=wp.float32, device=device)
        self.nb = wp.array(neighbors, dtype=wp.int32, device=device)
        self.gij = wp.array(gij, dtype=wp.float32, device=device)
        self.out = wp.zeros(temp.shape, dtype=wp.float32, device=device)

    def run(self) -> None:
        """Launch one conduction step (asynchronous on CUDA)."""
        wp.launch(
            conduction_kernel,
            dim=self.temp.shape[0],
            inputs=[self.temp, self.cap, self.nb, self.gij, self.out, self.dt, self.k],
            device=self.device,
        )

    def result(self) -> np.ndarray:
        """Output temperatures as a NumPy array (syncs)."""
        self.sync()
        return self.out.numpy()


def init(device: str, cold_cache_dir: str | None = None) -> None:
    """Initialise Warp. `cold_cache_dir` redirects the kernel cache so first-launch time includes compilation."""
    if cold_cache_dir is not None:
        wp.config.kernel_cache_dir = cold_cache_dir
    wp.init()
    if device != "cpu":
        wp.get_device(device)  # raises if absent


def version() -> str:
    """Installed warp-lang version."""
    return wp.config.version
