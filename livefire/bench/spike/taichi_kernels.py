"""Taichi implementations of the spike kernels. Semantics: see numpy_ref.py.

`init(device)` must be called once per process before building runners ("cpu" or "cuda:0"; Taichi picks the
CUDA device itself, so use CUDA_VISIBLE_DEVICES to choose a GPU). Runners own their ndarrays.
"""

import numpy as np
import taichi as ti


# --- kernel begin
@ti.func
def _corner(p: ti.f32, g: ti.i32):
    q = ti.math.clamp(p, 0.0, ti.cast(g - 1, ti.f32))
    i0 = ti.min(ti.cast(q, ti.i32), g - 2)
    return i0, q - ti.cast(i0, ti.f32)


@ti.func
def _sample_s(f: ti.types.ndarray(dtype=ti.f32, ndim=3), p, g: ti.i32):
    i, tx = _corner(p[0], g)
    j, ty = _corner(p[1], g)
    k, tz = _corner(p[2], g)
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@ti.func
def _sample_v(f: ti.types.ndarray(dtype=ti.math.vec3, ndim=3), p, g: ti.i32):
    i, tx = _corner(p[0], g)
    j, ty = _corner(p[1], g)
    k, tz = _corner(p[2], g)
    c00 = f[i, j, k] * (1.0 - tx) + f[i + 1, j, k] * tx
    c10 = f[i, j + 1, k] * (1.0 - tx) + f[i + 1, j + 1, k] * tx
    c01 = f[i, j, k + 1] * (1.0 - tx) + f[i + 1, j, k + 1] * tx
    c11 = f[i, j + 1, k + 1] * (1.0 - tx) + f[i + 1, j + 1, k + 1] * tx
    return (c00 * (1.0 - ty) + c10 * ty) * (1.0 - tz) + (c01 * (1.0 - ty) + c11 * ty) * tz


@ti.kernel
def advect_kernel(
    scalar: ti.types.ndarray(dtype=ti.f32, ndim=3),
    vel: ti.types.ndarray(dtype=ti.math.vec3, ndim=3),
    out: ti.types.ndarray(dtype=ti.f32, ndim=3),
    dt: ti.f32,
    h: ti.f32,
    g: ti.i32,
):
    for i, j, k in out:
        x = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32), ti.cast(k, ti.f32)])
        c = dt / h
        x_mid = x - 0.5 * c * vel[i, j, k]
        u_mid = _sample_v(vel, x_mid, g)
        x_back = x - c * u_mid
        out[i, j, k] = _sample_s(scalar, x_back, g)


@ti.kernel
def conduction_kernel(
    temp: ti.types.ndarray(dtype=ti.f32, ndim=1),
    cap: ti.types.ndarray(dtype=ti.f32, ndim=1),
    neighbors: ti.types.ndarray(dtype=ti.i32, ndim=2),
    gij: ti.types.ndarray(dtype=ti.f32, ndim=2),
    out: ti.types.ndarray(dtype=ti.f32, ndim=1),
    dt: ti.f32,
    kmax: ti.i32,
):
    for i in out:
        ti_ = temp[i]
        acc = 0.0
        for s in range(kmax):
            j = neighbors[i, s]
            if j >= 0:
                acc += gij[i, s] * (temp[j] - ti_)
        out[i] = ti_ + dt * acc / cap[i]
# --- kernel end

_ARCH: str | None = None


def init(device: str, cold_cache: bool = False) -> None:
    """Initialise Taichi once per process. `cold_cache` disables the offline kernel cache (timing JIT)."""
    global _ARCH
    arch = "cpu" if device == "cpu" else "cuda"
    if _ARCH is not None:
        if _ARCH != arch:
            raise RuntimeError(f"taichi already initialised for {_ARCH}")
        return
    ti.init(arch=ti.cpu if arch == "cpu" else ti.cuda, offline_cache=not cold_cache, log_level=ti.WARN)
    _ARCH = arch


def sync() -> None:
    """Block until all Taichi work finished."""
    ti.sync()


def version() -> str:
    """Installed taichi version."""
    return ".".join(str(v) for v in ti.__version__) if isinstance(ti.__version__, tuple) else str(ti.__version__)


class TaichiAdvect:
    """Advection runner. Inputs host f32 arrays: scalar (G,G,G), vel (G,G,G,3)."""

    def __init__(self, scalar: np.ndarray, vel: np.ndarray, dt: float, h: float) -> None:
        self.dt, self.h, self.g = float(dt), float(h), scalar.shape[0]
        self.scalar = ti.ndarray(ti.f32, shape=scalar.shape)
        self.scalar.from_numpy(scalar)
        self.vel = ti.Vector.ndarray(3, ti.f32, shape=scalar.shape)
        self.vel.from_numpy(vel)
        self.out = ti.ndarray(ti.f32, shape=scalar.shape)

    def run(self) -> None:
        """Launch one advection step (asynchronous on CUDA)."""
        advect_kernel(self.scalar, self.vel, self.out, self.dt, self.h, self.g)

    def sync(self) -> None:
        """Block until finished."""
        sync()

    def result(self) -> np.ndarray:
        """Output field as a NumPy array (syncs, copies)."""
        return self.out.to_numpy()


class TaichiConduction:
    """Conduction gather runner. temp, cap (N,) f32; neighbors (N,K) i32 (-1 pad); gij (N,K) f32."""

    def __init__(
        self, temp: np.ndarray, cap: np.ndarray, neighbors: np.ndarray, gij: np.ndarray, dt: float
    ) -> None:
        self.dt, self.k = float(dt), neighbors.shape[1]
        self.temp, self.cap = ti.ndarray(ti.f32, temp.shape), ti.ndarray(ti.f32, cap.shape)
        self.nb, self.gij = ti.ndarray(ti.i32, neighbors.shape), ti.ndarray(ti.f32, gij.shape)
        self.temp.from_numpy(temp)
        self.cap.from_numpy(cap)
        self.nb.from_numpy(neighbors)
        self.gij.from_numpy(gij)
        self.out = ti.ndarray(ti.f32, temp.shape)

    def run(self) -> None:
        """Launch one conduction step (asynchronous on CUDA)."""
        conduction_kernel(self.temp, self.cap, self.nb, self.gij, self.out, self.dt, self.k)

    def sync(self) -> None:
        """Block until finished."""
        sync()

    def result(self) -> np.ndarray:
        """Output temperatures as a NumPy array (syncs, copies)."""
        return self.out.to_numpy()
