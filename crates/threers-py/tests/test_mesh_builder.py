"""MeshBuilder and Geometry.report: the Rust builder with Python arguments."""

import pytest

import threers
from threers import Geometry, MeshBuilder, Vector3


def face_y(g, t):
    p, i = g.positions(), g.index()
    v = [p[i[t * 3 + k] * 3: i[t * 3 + k] * 3 + 3] for k in range(3)]
    u = [v[1][j] - v[0][j] for j in range(3)]
    w = [v[2][j] - v[0][j] for j in range(3)]
    return u[2] * w[0] - u[0] * w[2]


@pytest.mark.parametrize("order", [[0, 1, 2, 3], [3, 2, 1, 0]])
def test_facing_decides_the_side_whatever_the_order(order):
    corners = [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)]
    m = MeshBuilder()
    ids = [m.vertex(corners[i]) for i in order]
    g = m.facing((0, 1, 0)).quad(*ids).build()
    assert g.triangle_count() == 2
    assert face_y(g, 0) > 0 and face_y(g, 1) > 0


def test_grid_takes_a_function_or_a_list_and_ribbon_takes_vector3s():
    from_fn = MeshBuilder().grid(10, 10, 4, 4, lambda x, z: x * 0.5).build()
    heights = [(c / 4 - 0.5) * 10 * 0.5 for r in range(5) for c in range(5)]
    from_list = MeshBuilder().grid(10, 10, 4, 4, heights).build()
    assert from_fn.positions() == from_list.positions()
    left = [Vector3(i, 0, 0) for i in range(3)]
    right = [(i, 0, 1) for i in range(3)]
    strip = MeshBuilder().ribbon(right, left).build()
    assert strip.triangle_count() == 4
    assert face_y(strip, 0) > 0


def test_append_merges_and_the_report_calls_it_solid():
    box = Geometry.box(1, 1, 1)
    m = MeshBuilder().append(box, position=(5, 0, 0)).append(box, position=(-5, 0, 0), rotation=(0, 1.0, 0))
    g = m.build()
    assert g.triangle_count() == box.triangle_count() * 2
    xs = g.positions()[0::3]
    assert abs(max(xs) - 5.5) < 1e-4
    report = g.report()
    assert report["is_solid"], report
    assert Geometry.plane(1, 1).report()["open_edges"] == 4


def test_mistakes_are_named_where_they_are_made():
    m = MeshBuilder()
    a = m.vertex((0, 0, 0))
    with pytest.raises(ValueError, match="vertex 7 does not exist"):
        m.triangle(a, a, 7)
    with pytest.raises(ValueError, match="expected 9 heights"):
        MeshBuilder().grid(1, 1, 2, 2, [1, 2, 3])
    with pytest.raises(TypeError, match="Vector3"):
        MeshBuilder().vertex("nope")
    with pytest.raises(ValueError, match="16 numbers"):
        MeshBuilder().append(Geometry.box(), matrix=[1, 2, 3])
    with pytest.raises(ZeroDivisionError):
        MeshBuilder().grid(1, 1, 1, 1, lambda x, z: 1 / 0)


def test_a_built_mesh_draws():
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        renderer = threers.HeadlessRenderer(32, 32)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")
    scene = threers.Scene()
    scene.set_background(0x000000)
    scene.add_mesh(MeshBuilder().grid(4, 4, 2, 2).build(), threers.Material.basic(color=0xFF0000))
    camera = threers.PerspectiveCamera(50, 1, 0.1, 100)
    camera.set_position(0, 5, 0.01)
    camera.look_at(0, 0, 0)
    rgba = renderer.render_rgba(scene, camera)
    centre = (16 * 32 + 16) * 4
    assert rgba[centre] > 200 and rgba[centre + 1] < 40, rgba[centre:centre + 4]
    assert "MeshBuilder(" in repr(MeshBuilder())
