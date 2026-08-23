"""Smoke tests — run after `maturin develop`."""

import threers


def test_version():
    assert threers.__version__
    assert threers.version() == threers.__version__


def test_vector3():
    v = threers.Vector3(3, 4, 0)
    assert abs(v.length() - 5.0) < 1e-5


def test_tween():
    if threers.Tween is None:
        return
    t = threers.Tween(0.0, 10.0, 1.0, easing="linear")
    t.update(0.5)
    assert abs(t.value() - 5.0) < 1e-4


def test_physics():
    if threers.PhysicsWorld is None:
        return
    w = threers.PhysicsWorld()
    i = w.add_ball(0.5, y=3.0)
    for _ in range(60):
        w.step(1 / 60)
    assert w.body_translation(i).y < 3.0
