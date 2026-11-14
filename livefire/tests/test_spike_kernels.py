"""Spike kernels (Warp, Taichi) vs the NumPy reference, on CPU."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "spike"))

import numpy_ref
import spike_common as common

ADV_CASES = [(1 / 60, 3.0 / 32), (0.5, 3.0 / 32)]  # second case: displacements of several cells, hits clamping


def _backends():
    try:
        import taichi  # noqa: F401
    except ImportError:
        return ["warp", pytest.param("taichi", marks=pytest.mark.skip(reason="taichi not importable"))]
    return ["warp", "taichi"]


def _adv(name, s, v, dt, h):
    if name == "warp":
        import warp_kernels as wk

        wk.init("cpu")
        return wk.WarpAdvect("cpu", s, v, dt, h)
    import taichi_kernels as tk

    tk.init("cpu")
    return tk.TaichiAdvect(s, v, dt, h)


def _con(name, T, C, nb, gij, dt):
    if name == "warp":
        import warp_kernels as wk

        wk.init("cpu")
        return wk.WarpConduction("cpu", T, C, nb, gij, dt)
    import taichi_kernels as tk

    tk.init("cpu")
    return tk.TaichiConduction(T, C, nb, gij, dt)


@pytest.mark.parametrize("name", _backends())
@pytest.mark.parametrize("dt,h", ADV_CASES)
def test_advection_matches_numpy(name, dt, h):
    s, v = common.make_advect_inputs(32, 1234)
    ref = numpy_ref.advect(s, v, dt, h)
    r = _adv(name, s, v, dt, h)
    r.run()
    out = r.result()
    assert out.shape == ref.shape and out.dtype == np.float32
    assert np.max(np.abs(out - ref)) < 1e-4


@pytest.mark.parametrize("name", _backends())
def test_conduction_matches_numpy(name):
    # Absolute 1e-5 is below 1 float32 ulp for values >= 128 (ulp(512 K) = 6.1e-5), so the brief's absolute
    # threshold is applied to a temperature-excess field in [0, 50) K (ulp <= 3.8e-6). The physical 293-793 K
    # field is checked with a relative bound (a few ulp).
    T, C, nb, gij = common.make_conduction_inputs(5000, 16, 1234, t_lo=0.0, t_span=50.0)
    assert (nb == -1).any()
    ref = numpy_ref.conduction(T, C, nb, gij, 0.01)
    r = _con(name, T, C, nb, gij, 0.01)
    r.run()
    assert np.max(np.abs(r.result() - ref)) < 1e-5
    T, C, nb, gij = common.make_conduction_inputs(5000, 16, 1234)
    ref = numpy_ref.conduction(T, C, nb, gij, 0.01)
    r = _con(name, T, C, nb, gij, 0.01)
    r.run()
    assert np.max(np.abs(r.result() - ref) / ref) < 5e-7


@pytest.mark.parametrize("name", _backends())
def test_repeat_runs_bitwise_identical(name):
    s, v = common.make_advect_inputs(32, 7)
    r = _adv(name, s, v, 1 / 60, 3.0 / 32)
    r.run()
    a = r.result().copy()
    r.run()
    assert np.array_equal(a, r.result())
    T, C, nb, gij = common.make_conduction_inputs(5000, 16, 7)
    c = _con(name, T, C, nb, gij, 0.01)
    c.run()
    a = c.result().copy()
    c.run()
    assert np.array_equal(a, c.result())


def test_advection_reference_properties():
    g = 16
    s = np.full((g, g, g), 3.0, np.float32)
    v = np.zeros((g, g, g, 3), np.float32)
    assert np.array_equal(numpy_ref.advect(s, v, 0.1, 0.1), s)  # zero velocity: identity
    v[..., 0] = 1.0  # uniform +x flow, linear ramp in x: shifts by dt/h cells (clamped at edge)
    ramp = np.broadcast_to(np.arange(g, dtype=np.float32)[:, None, None], (g, g, g)).copy()
    out = numpy_ref.advect(ramp, v, 0.5, 1.0)
    assert np.allclose(out[1:, 0, 0], ramp[1:, 0, 0] - 0.5, atol=1e-5)
    assert out[0, 0, 0] == 0.0


def test_csv_columns_verbatim():
    prd = (Path(__file__).resolve().parents[1] / "docs" / "PRD.md").read_text()
    line = next(ln for ln in prd.splitlines() if ln.startswith("`run_id, date"))
    assert common.COLUMNS == [c.strip() for c in line.strip("`").split(",")]


def test_taichi_kernel_accepts_numpy_directly():
    pytest.importorskip("taichi")
    import taichi_kernels as tk

    tk.init("cpu")
    s, v = common.make_advect_inputs(32, 1234)
    out = np.zeros_like(s)
    tk.advect_kernel(s, v, out, 1 / 60, 3.0 / 32, 32)
    assert np.max(np.abs(out - numpy_ref.advect(s, v, 1 / 60, 3.0 / 32))) < 1e-4
