"""threers-fem through Python: FEM against beam theory, BEM against a sphere."""

import math

import pytest

import threers

pytestmark = pytest.mark.skipif(threers.FemModel is None, reason="built without `fem`")


def test_cantilever_matches_beam_theory():
    # 10 x 1 x 1, E = 1000, tip load 1: FL^3/3EI = 4, plus a little shear.
    m = threers.FemModel.box_grid((0, 0, 0), (10, 1, 1), (20, 2, 2))
    m.set_material(1000.0, 0.3)
    m.fix_box((-1, -1, -1), (0, 2, 2))
    m.force_box((10, -1, -1), (11, 2, 2), (0, -1, 0))
    r = m.solve()
    assert 3.9 < r.max_displacement < 4.2
    assert len(r.displacement) == 3 * m.node_count
    assert r.reaction[1] == pytest.approx(1.0, abs=1e-9)


def test_modes_and_heat_on_a_built_in_shape():
    names = [n for n, _ in threers.FemModel.shapes()]
    assert "bracket" in names
    m = threers.FemModel.shape("beam", 0.5)
    m.set_material(200e3, 0.3, 7.85e-9)
    m.fix_box((-1, -1, -1), (0, 2, 2))
    f = m.modes(2).frequencies
    # Euler-Bernoulli: 0.5596 sqrt(EI / rho A L^4) for a 10 mm steel bar.
    want = 0.5596 * math.sqrt(200e3 / 12 / (7.85e-9 * 1e4))
    assert f[0] == pytest.approx(want, rel=0.03)
    m.clear()
    m.temperature_box((-1, -1, -1), (0, 2, 2), 100.0)
    m.temperature_box((10, -1, -1), (11, 2, 2), 0.0)
    h = m.solve_heat(1.0)
    assert h.max == pytest.approx(100.0) and h.min == pytest.approx(0.0, abs=1e-9)


def test_errors_are_value_errors():
    m = threers.FemModel.box_grid((0, 0, 0), (1, 1, 1), (2, 2, 2))
    m.set_material(1.0, 0.3)
    m.fix_box((5, 5, 5), (6, 6, 6))  # selects nothing
    with pytest.raises(ValueError, match="support"):
        m.solve()
    with pytest.raises(ValueError):
        threers.FemModel.shape("teapot", 1.0)


def test_sphere_capacitance_and_radiation():
    s = threers.FemSurface.sphere(1.0, 3)
    assert s.capacitance() == pytest.approx(4 * math.pi, rel=0.03)
    # Pulsating sphere, dp/dn = 1 at k = 1: p = 1 / (i - 1) on the surface.
    k = 1.0
    f = k * 343.0 / (2 * math.pi)
    rho = 1.204
    v = [1.0 / (k * 343.0 * rho)] * s.triangle_count  # dp/dn = i w rho v
    r = s.radiate(f, [0.0] * s.triangle_count, v)
    # velocity_im = 1/(w rho) gives dp/dn = i w rho (i / (w rho)) = -1.
    want = -1 / complex(-1, 1)
    assert abs(r.pressure[0] - want) < 0.05 * abs(want)
    far = r.field([(10.0, 0.0, 0.0)])[0]
    # A monopole falls off as 1/r.
    assert abs(far) == pytest.approx(abs(want) / 10, rel=0.05)
