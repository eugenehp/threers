"""Quads: polygon access, quad-aware wireframes and edges, loaders that keep
quads, and the quad view / viewport compositing controls — run after
`maturin develop`."""

from pathlib import Path

import pytest

import threers
from threers import Geometry, Material

REPO = Path(__file__).resolve().parents[3]
W = H = 64
BG = 0x204060
BG_PX = (32, 64, 96)


def make_renderer(w=W, h=H):
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        return threers.HeadlessRenderer(w, h)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")


def px(rgba, x, y, w=W):
    i = (y * w + x) * 4
    return tuple(rgba[i : i + 3])


def near(a, b, tol=3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def square():
    """Two triangles over the square -1..1, with no polygon grouping."""
    return Geometry.from_arrays(
        [-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], indices=[0, 1, 2, 0, 2, 3]
    )


def test_grid_geometries_carry_their_quads():
    plane = Geometry.plane(2, 2, 3, 2)
    assert plane.polygon_counts() == [2] * 6
    assert plane.polygon_count() == 6
    assert plane.quad_count() == 6
    assert all(len(loop) == 4 for loop in plane.polygon_loops())
    # 3 x 2 cells: 3 * 3 horizontal + 4 * 2 vertical edges, no diagonals.
    assert len(plane.polygon_edges()) == 17
    assert all(isinstance(e, tuple) and len(e) == 2 for e in plane.polygon_edges())

    assert Geometry.box(1, 1, 1).quad_count() == 6
    assert Geometry.quad([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 4, 2).quad_count() == 8
    sphere = Geometry.sphere(1, 8, 4)
    assert 0 < sphere.quad_count() < sphere.polygon_count(), "quads, plus pole triangles"

    assert len(plane.positions()) == plane.vertex_count() * 3
    assert all(i < plane.vertex_count() for i in plane.polygon_loops()[0])
    assert len(plane.index()) == plane.triangle_count() * 3


def test_set_polygons_groups_triangles_into_quads():
    sq = square()
    assert sq.polygon_counts() is None
    assert sq.polygon_count() == 2
    assert sq.quad_count() == 0
    sq.set_polygons([2])
    assert sq.polygon_counts() == [2]
    assert sq.quad_count() == 1
    assert sorted(sq.polygon_loops()[0]) == [0, 1, 2, 3]
    assert len(sq.polygon_edges()) == 4
    with pytest.raises(ValueError, match="sum to 3 triangles, the geometry has 2"):
        sq.set_polygons([3])
    with pytest.raises(ValueError, match="at least 1"):
        sq.set_polygons([0, 2])
    assert sq.polygon_counts() == [2], "unchanged after a rejected call"


def test_wireframe_and_edges_follow_polygon_outlines():
    def segs(g):
        return g.vertex_count() // 2

    assert segs(Geometry.wireframe(Geometry.plane(1, 1))) == 4, "quad outline, no diagonal"
    assert segs(Geometry.wireframe(square())) == 5, "two triangles share the diagonal"
    assert segs(Geometry.wireframe(Geometry.plane(1, 1, 3, 2))) == 17
    assert segs(Geometry.edges(Geometry.box(1, 1, 1))) == 12
    assert segs(Geometry.edges(Geometry.sphere(1, 16, 8), 90)) == 0
    with pytest.raises(ValueError, match="threshold_angle"):
        Geometry.edges(Geometry.box(), 200)
    with pytest.raises(TypeError):
        Geometry.wireframe(Material.basic())


def test_line_segments_and_quad_wireframes_render_without_diagonals():
    r = make_renderer()
    cam = threers.OrthographicCamera(-1.25, 1.25, 1.25, -1.25, 0.1, 10)
    cam.set_position(0, 0, 5)
    cam.look_at(0, 0, 0)

    def centre_hit(s):
        rgba = r.render_rgba(s, cam)
        return any(not near(px(rgba, x, y), BG_PX) for y in range(30, 35) for x in range(30, 35))

    def edge_hit(s):
        rgba = r.render_rgba(s, cam)
        # The square's left edge sits at x = -1, pixel 6.4.
        return any(not near(px(rgba, x, 32), BG_PX) for x in range(4, 10))

    def scene(build):
        s = threers.Scene()
        s.set_background(BG)
        build(s)
        return s

    red = Material.line(color=0xFF0000)
    wire = Material.basic(color=0xFFFFFF, wireframe=True)

    quad_lines = scene(lambda s: s.add_line_segments(Geometry.wireframe(Geometry.plane(2, 2)), red))
    assert edge_hit(quad_lines) and not centre_hit(quad_lines), "quad outline only"
    assert centre_hit(scene(lambda s: s.add_line_segments(Geometry.wireframe(square()), red)))

    quad_wire = scene(lambda s: s.add_mesh(Geometry.plane(2, 2), wire))
    assert edge_hit(quad_wire) and not centre_hit(quad_wire), "wireframe material: no diagonal"
    assert centre_hit(scene(lambda s: s.add_mesh(square(), wire))), "two triangles: diagonal"
    grouped = square()
    grouped.set_polygons([2])
    assert not centre_hit(scene(lambda s: s.add_mesh(grouped, wire))), "set_polygons removes it"

    rgba = r.render_rgba(quad_lines, cam)
    assert any(px(rgba, x, 32)[0] > 200 and px(rgba, x, 32)[1] < 40 for x in range(4, 10))


def test_line_segments_validation_get_geometry_and_meshes():
    s = threers.Scene()
    with pytest.raises(ValueError, match=r"Material\.line"):
        s.add_line_segments(Geometry.wireframe(Geometry.box()), Material.basic())
    with pytest.raises(ValueError, match="add_line_segments"):
        s.add_mesh(Geometry.box(), Material.line())
    with pytest.raises(ValueError, match="even vertex count"):
        s.add_line_segments(Geometry.from_arrays([0, 0, 0, 1, 0, 0, 0, 1, 0]), Material.line())
    with pytest.raises(TypeError, match="does not apply"):
        Material.line(roughness=0.5)
    assert Material.line(color=0x00FF00, opacity=0.5, depth_test=False).kind == "line"
    line = s.add_line_segments(Geometry.edges(Geometry.box()), Material.line())
    assert s.get_geometry(line).vertex_count() == 24
    with pytest.raises(ValueError, match=r"Material\.line"):
        s.set_material(line, Material.basic())
    s.set_material(line, Material.line(color=0xFF00FF))
    m = s.add_mesh(Geometry.plane(1, 1, 2, 2), Material.basic())
    with pytest.raises(ValueError, match="only applies to line segments"):
        s.set_material(m, Material.line())
    assert s.get_geometry(m).quad_count() == 4
    assert s.meshes(m) == [m]
    g = s.add_group()
    with pytest.raises(ValueError, match="no geometry"):
        s.get_geometry(g)
    assert s.meshes(g) == []


def test_loaders_keep_quads(tmp_path):
    s = threers.Scene()
    (tmp_path / "quad.obj").write_text("v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    obj = s.load(tmp_path / "quad.obj")
    assert s.get_geometry(obj).polygon_counts() == [2]
    assert s.get_geometry(obj).quad_count() == 1

    ply = "\n".join(
        [
            "ply", "format ascii 1.0", "element vertex 4", "property float x",
            "property float y", "property float z", "element face 1",
            "property list uchar int vertex_indices", "end_header",
            "0 0 0", "1 0 0", "1 1 0", "0 1 0", "4 0 1 2 3", "",
        ]
    )
    assert s.get_geometry(s.load_bytes(ply.encode(), "ply")).quad_count() == 1

    fbx = s.load(REPO / "src/loaders/fbx/testdata/blender.fbx")
    parts = s.meshes(fbx)
    assert len(parts) == 2, "both meshes of the file"
    assert all(s.get_geometry(p).quad_count() == 6 for p in parts), "cube faces stay quads"
    assert s.meshes(fbx) == parts, "the same handles again"
    s.set_position(parts[0], 0, 3, 0)

    usd = s.load(REPO / "src/loaders/usd/testdata/polygons.usda")
    (faces,) = s.meshes(usd)
    geo = s.get_geometry(faces)
    assert geo.polygon_counts() == [2, 1, 3], "a quad, a triangle and a pentagon"
    assert [len(loop) for loop in geo.polygon_loops()] == [4, 3, 5]


def test_render_quad_view_layout_border_backgrounds_margin():
    r = make_renderer(200, 160)
    s = threers.Scene()
    s.set_background(0x000000)
    s.add_mesh(Geometry.box(1, 1, 1), Material.normal())
    cam = threers.PerspectiveCamera(45, 1, 0.1, 100)
    cam.set_position(3, 2.5, 4)
    cam.look_at(0, 0, 0)
    rgba = r.render_quad_view(
        s,
        cam,
        format="rgba",
        layout=["side", "front", "perspective", "top"],
        gap=4,
        border=6,
        gap_color=0xFF00FF,
        backgrounds={"top": 0xFF0000, "front": 0x00FF00, "side": 0x0000FF, "perspective": 0xFFFF00},
        margin=0.5,
    )

    def p(x, y):
        return px(rgba, x, y, 200)

    assert near(p(2, 2), (255, 0, 255)), "border is the gap colour"
    assert near(p(10, 10), (0, 0, 255)), "top-left is side"
    assert near(p(190, 10), (0, 255, 0)), "top-right is front"
    assert near(p(10, 150), (255, 255, 0)), "bottom-left is perspective"
    assert near(p(190, 150), (255, 0, 0)), "bottom-right is top"

    for kwargs, match in [
        ({"layout": ["top", "front", "side"]}, "4 pane names"),
        ({"layout": ["top", "top", "side", "front"]}, "twice"),
        ({"layout": ["top", "front", "side", "iso"]}, "unknown pane"),
        ({"border": -1}, "border"),
        ({"backgrounds": {"side": 0x1000000}}, r"backgrounds\['side'\]"),
        ({"backgrounds": {"iso": 0}}, "unknown pane 'iso'"),
    ]:
        with pytest.raises(ValueError, match=match):
            r.render_quad_view(s, cam, **kwargs)


def test_auto_clear_off_and_clear_compose_viewports():
    r = make_renderer()
    s = threers.Scene()
    s.set_background(BG)
    s.add_mesh(Geometry.box(4, 4, 1), Material.basic(color=0xFF0000))
    cam = threers.OrthographicCamera()
    cam.set_position(0, 0, 5)
    assert r.auto_clear()
    r.set_auto_clear(False)
    r.clear(0x00FF00)
    r.set_viewport(0, 0, W / 2, H)  # left half, bottom-left origin
    rgba = r.render_rgba(s, cam)
    assert near(px(rgba, 10, 32), (255, 0, 0)), "left half is the scene"
    assert near(px(rgba, 54, 32), (0, 255, 0)), "right half keeps the clear colour"
    r.set_auto_clear(True)
    r.reset_viewport()
    assert near(px(r.render_rgba(s, cam), 54, 32), (255, 0, 0)), "whole frame again"
    with pytest.raises(ValueError, match="clear: color"):
        r.clear(0x1000000)
    with pytest.raises(ValueError, match="alpha"):
        r.clear(0, 2)


def test_infer_quads():
    sq = square()
    assert sq.infer_quads() == 1
    assert sq.quad_count() == 1
    assert sq.infer_quads() == 0, "nothing left to pair"
    plane = Geometry.plane(3, 3, 3, 3)
    soup = Geometry.from_arrays(plane.positions(), plane.index())
    assert soup.quad_count() == 0
    assert soup.infer_quads() == 9
    assert soup.quad_count() == 9
    with pytest.raises(ValueError, match="max_face_angle"):
        sq.infer_quads(200)
    with pytest.raises(ValueError, match="max_shape_angle"):
        sq.infer_quads(max_shape_angle=-1)


def test_raycast_reports_the_quad_that_was_hit():
    s = threers.Scene()
    h = s.add_mesh(Geometry.plane(2, 2, 2, 2), Material.basic())
    down = (0, 0, -1)

    def hit(x, y):
        hits = s.raycast((x, y, 5), down)
        assert len(hits) == 1
        return hits[0]

    first = hit(0.3, 0.6)  # off both diagonals of its cell
    assert first["handle"] == h
    assert abs(first["distance"] - 5) < 1e-4
    assert abs(first["point"][0] - 0.3) < 1e-4 and abs(first["point"][2]) < 1e-4
    cell = [hit(x, y) for x, y in [(0.15, 0.5), (0.5, 0.15), (0.85, 0.5), (0.5, 0.85)]]
    assert len({c["face_index"] for c in cell}) == 2, "both triangles were hit"
    assert all(c["polygon_index"] == first["polygon_index"] for c in cell), "one quad"
    assert hit(-0.3, -0.6)["polygon_index"] != first["polygon_index"]
    assert s.raycast(threers.Vector3(5, 5, 5), down) == []

    s.set_position(h, 0, 0, -1)
    assert abs(hit(0.3, 0.6)["distance"] - 6) < 1e-4, "transforms are applied"
    plain = threers.Scene()
    plain.add_mesh(square(), Material.basic())
    assert plain.raycast((0.2, 0.6, 5), down)[0]["polygon_index"] is None

    cam = threers.OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
    cam.set_position(0, 0, 10)
    cam.look_at(0, 0, 0)
    (picked,) = s.raycast_from_camera(cam, 0.15, 0.3)
    assert abs(picked["point"][0] - 0.3) < 1e-3 and abs(picked["point"][1] - 0.6) < 1e-3
    persp = threers.PerspectiveCamera(50, 1, 0.1, 100)
    persp.set_position(0, 0, 5)
    persp.look_at(0, 0, 0)
    assert s.raycast_from_camera(persp, 0, 0)[0]["handle"] == h

    with pytest.raises(ValueError, match="direction"):
        s.raycast((0, 0, 5), (0, 0, 0))
    with pytest.raises(ValueError, match="far must be > near"):
        s.raycast((0, 0, 5), down, near=2, far=1)
    with pytest.raises(ValueError, match="finite"):
        s.raycast_from_camera(persp, float("nan"), 0)
    with pytest.raises(TypeError, match="PerspectiveCamera or OrthographicCamera"):
        s.raycast_from_camera("cam", 0, 0)


def test_subdivide_refines_polygons_into_quads():
    box = Geometry.box(1, 1, 1)
    assert box.subdivide().quad_count() == 24, "each of 6 quads -> 4"
    assert box.subdivide(2).quad_count() == 96
    assert box.quad_count() == 6, "the source is untouched"
    assert box.subdivide(0).quad_count() == 6

    def max_abs(g):
        return max(abs(v) for v in g.positions())

    assert max_abs(box.subdivide(2)) < 0.5, "a cube cage shrinks toward a sphere"
    assert square().subdivide().quad_count() == 6, "two triangles -> 3 quads each"
    # An open quad keeps its border (sharp_boundary, the default); the flag is
    # passed through to the core's boundary rule.
    assert abs(max_abs(Geometry.plane(2, 2).subdivide(1)) - 1) < 1e-5, "border held"
    assert Geometry.plane(2, 2).subdivide(1, sharp_boundary=False).quad_count() == 4
    with pytest.raises(ValueError, match="levels"):
        box.subdivide(7)
