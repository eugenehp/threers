"""Geometry, materials, handles, cameras, loaders and renderer settings —
run after `maturin develop`."""

import math
import struct
from pathlib import Path

import pytest

import threers
from threers import Geometry, Material

REPO = Path(__file__).resolve().parents[3]
W = H = 64
BG = 0x204060
BG_PX = (32, 64, 96)


@pytest.fixture
def renderer():
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        return threers.HeadlessRenderer(W, H)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")


def px(rgba, x, y, w=W):
    i = (y * w + x) * 4
    return tuple(rgba[i : i + 3])


def near(a, b, tol=3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def camera():
    c = threers.PerspectiveCamera(50, 1, 0.1, 100)
    c.set_position(0, 0, 4)
    c.look_at(0, 0, 0)
    return c


def scene():
    s = threers.Scene()
    s.set_background(BG)
    return s


# ---- geometry ----------------------------------------------------------------


def test_every_geometry_factory_builds_triangles():
    geoms = {
        "box": Geometry.box(1, 2, 3),
        "plane": Geometry.plane(1, 1, 2, 3),
        "sphere": Geometry.sphere(),
        "cylinder": Geometry.cylinder(0.5, 1, 2, 16, 2, open_ended=True),
        "cone": Geometry.cone(),
        "torus": Geometry.torus(1, 0.3, 8, 24, arc=math.pi),
        "torus_knot": Geometry.torus_knot(),
        "ring": Geometry.ring(),
        "circle": Geometry.circle(1, 12),
        "capsule": Geometry.capsule(0.5, 1),
        "lathe": Geometry.lathe([0, -1, 1, 0, 0, 1], segments=16),
        "lathe_pairs": Geometry.lathe([(0, -1), (1, 0)]),
        "tube": Geometry.tube([(0, 0, 0), threers.Vector3(1, 1, 0), (2, 0, 0)], 32, radius=0.1),
        "icosahedron": Geometry.icosahedron(1, 1),
        "octahedron": Geometry.octahedron(),
        "tetrahedron": Geometry.tetrahedron(),
        "dodecahedron": Geometry.dodecahedron(),
        "quad": Geometry.quad([-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1], 2, 2),
        "from_arrays": Geometry.from_arrays([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    }
    for name, g in geoms.items():
        assert g.vertex_count() > 0, name
        assert g.triangle_count() > 0, name
    assert geoms["box"].triangle_count() == 12
    assert geoms["plane"].triangle_count() == 2 * 2 * 3
    assert geoms["from_arrays"].triangle_count() == 1
    quad = Geometry.from_arrays([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0], indices=[0, 1, 2, 2, 1, 3])
    assert quad.triangle_count() == 2
    assert "triangles=2" in repr(quad)


@pytest.mark.parametrize(
    "call, exc, match",
    [
        (lambda: Geometry.box(-1), ValueError, "width"),
        (lambda: Geometry.sphere(1, 2), ValueError, "width_segments"),
        (lambda: Geometry.plane(1, 1, 0), ValueError, "width_segments"),
        (lambda: Geometry.plane(1, 1, 5000), ValueError, "width_segments"),
        (lambda: Geometry.torus(arc=0), ValueError, "arc"),
        (lambda: Geometry.torus_knot(p=0), ValueError, "p and q"),
        (lambda: Geometry.ring(2, 1), ValueError, "inner_radius"),
        (lambda: Geometry.lathe([0, 0]), ValueError, "at least 2"),
        (lambda: Geometry.lathe([0, 0, 1]), ValueError, "whole number"),
        (lambda: Geometry.lathe([(0, 0), (1, 2, 3)]), ValueError, "expected 2"),
        (lambda: Geometry.lathe(5), TypeError, "sequence"),
        (lambda: Geometry.tube([0, 0, 0, math.nan, 1, 1]), ValueError, "non-finite"),
        (lambda: Geometry.icosahedron(1, 99), ValueError, "detail"),
        (lambda: Geometry.from_arrays([0, 0, 0, 1, 0, 0, 0, 1, 0], [0, 1, 5]), ValueError, "out of range"),
        (lambda: Geometry.from_arrays([0, 0]), ValueError, "multiple of 3"),
        (lambda: Geometry.from_arrays([0, 0, 0, 1, 0, 0, 0, 1, 0], normals=[0, 0, 1]), ValueError, "normals"),
    ],
)
def test_geometry_rejects_bad_input(call, exc, match):
    with pytest.raises(exc, match=match):
        call()


# ---- materials ---------------------------------------------------------------


def test_material_factories_set_kind_and_validate():
    for k in ["basic", "standard", "physical", "lambert", "phong", "normal"]:
        assert getattr(Material, k)().kind == k
    Material.standard(color=0x336699, roughness=0.3, metalness=1, emissive=0x110000, side="double")
    Material.physical(clearcoat=1, transmission=0.5, ior=1.33, thickness=0.2)
    Material.phong(specular=0xFFFFFF, shininess=80)
    Material.basic(opacity=0.5, transparent=True, wireframe=True)
    for call, exc, match in [
        (lambda: Material.basic(roughness=0.5), TypeError, "'roughness' does not apply"),
        (lambda: Material.lambert(shininess=1), TypeError, "'shininess' does not apply"),
        (lambda: Material.normal(color=0xFFFFFF), TypeError, "'color' does not apply"),
        (lambda: Material.standard(colour=0xFFFFFF), TypeError, "unexpected keyword argument 'colour'"),
        (lambda: Material.standard(roughness="rough"), TypeError, "roughness"),
        (lambda: Material.standard(roughness=2), ValueError, r"roughness must be in \[0, 1\]"),
        (lambda: Material.standard(color=0x1000000), ValueError, "0xRRGGBB"),
        (lambda: Material.standard(side="both"), ValueError, "side"),
        (lambda: Material.physical(ior=0.5), ValueError, "ior"),
    ]:
        with pytest.raises(exc, match=match):
            call()


def test_add_mesh_renders_each_material_kind(renderer):
    cam = camera()

    def centre(mat, light=False):
        s = scene()
        s.add_mesh(Geometry.sphere(1), mat)
        if light:
            s.add_directional_light(0xFFFFFF, 3, (0, 0, 5))
        return px(renderer.render_rgba(s, cam), W // 2, H // 2)

    basic = centre(Material.basic(color=0xFF0000))
    assert basic[0] > 240 and basic[1] < 10 and basic[2] < 10, basic
    assert centre(Material.normal())[2] > 200, "normal facing +z is blue"
    for k in ["standard", "physical", "lambert", "phong"]:
        make = getattr(Material, k)
        assert sum(centre(make(color=0x00FF00))) < 6, f"{k} unlit is black"
        lit = centre(make(color=0x00FF00), light=True)
        assert lit[1] > 100 and lit[1] > lit[0] + 50, (k, lit)
    with pytest.raises(TypeError):
        scene().add_mesh(Material.basic(), Geometry.box())


# ---- handles -----------------------------------------------------------------


def test_handles_move_hide_rematerial_and_remove(renderer):
    cam = camera()
    s = scene()
    h = s.add_mesh(Geometry.box(1.5, 1.5, 1.5), Material.basic(color=0xFFFFFF))

    def centre():
        return px(renderer.render_rgba(s, cam), W // 2, H // 2)

    assert sum(centre()) > 700
    s.set_position(h, 10, 0, 0)
    assert near(centre(), BG_PX), "moved out of view"
    s.set_position(h, 0, 0, 0)
    s.set_scale(h, 0.01, 0.01, 0.01)
    assert near(centre(), BG_PX), "scaled to nothing"
    s.set_scale(h, 1, 1, 1)
    s.set_rotation(h, 0, math.pi / 4, 0)
    s.set_visible(h, False)
    assert near(centre(), BG_PX), "hidden"
    s.set_visible(h, True)
    s.set_material(h, Material.basic(color=0x0000FF))
    blue = centre()
    assert blue[2] > 240 and blue[0] < 10, blue
    p = s.get_position(h)
    assert (p.x, p.y, p.z) == (0, 0, 0)

    g = s.add_group()
    s.set_parent(h, g)
    s.set_position(g, 10, 0, 0)
    assert near(centre(), BG_PX), "moved with its group"
    with pytest.raises(ValueError, match="cycle"):
        s.set_parent(g, h)
    s.set_parent(h)
    assert not near(centre(), BG_PX), "back at the root"

    assert s.object_count() == 2
    s.remove(h)
    assert s.object_count() == 1
    assert near(centre(), BG_PX), "removed"
    with pytest.raises(ValueError, match="removed"):
        s.set_position(h, 0, 0, 0)
    with pytest.raises(ValueError, match="bad object handle"):
        s.set_visible(99, True)
    with pytest.raises(ValueError, match="finite"):
        s.set_rotation(g, math.nan, 0, 0)
    with pytest.raises(ValueError, match="no mesh"):
        s.set_material(g, Material.basic())
    assert s.add_group() == 2, "handles stay stable after a removal"


# ---- cameras -----------------------------------------------------------------


def test_orthographic_camera(renderer):
    with pytest.raises(ValueError, match="left != right"):
        threers.OrthographicCamera(1, 1)
    with pytest.raises(ValueError, match="near < far"):
        threers.OrthographicCamera(-1, 1, 1, -1, 5, 1)
    s = scene()
    s.add_mesh(Geometry.box(1, 1, 1), Material.basic(color=0xFFFFFF))
    cam = threers.OrthographicCamera(-2, 2, 2, -2, 0.1, 100)
    cam.set_position(0, 0, 5)
    cam.look_at(0, 0, 0)
    rgba = renderer.render_rgba(s, cam)
    # A unit box in a 4-unit view covers the middle quarter: x 24..40.
    assert sum(px(rgba, 32, 32)) > 700
    assert sum(px(rgba, 26, 32)) > 700, "inside the box edge"
    assert near(px(rgba, 20, 32), BG_PX), "outside it — no perspective growth"
    cam.set_frustum(-0.25, 0.25, 0.25, -0.25)
    assert sum(px(renderer.render_rgba(s, cam), 2, 2)) > 700, "zoomed in"
    assert renderer.render_png(s, cam)[:4] == b"\x89PNG"
    with pytest.raises(TypeError, match="PerspectiveCamera or OrthographicCamera"):
        renderer.render_rgba(s, "camera")


# ---- renderer ----------------------------------------------------------------


def test_size_tone_mapping_and_shadow_switch(renderer):
    s = scene()
    s.add_mesh(Geometry.sphere(1), Material.standard(color=0xFFFFFF))
    s.add_directional_light(0xFFFFFF, 1, (0, 0, 5))
    cam = camera()

    renderer.set_tone_mapping("aces", 1.5)
    assert renderer.tone_mapping() == ("aces", 1.5)
    renderer.set_size(48, 32)
    assert renderer.size() == (48, 32)
    assert len(renderer.render_rgba(s, cam)) == 48 * 32 * 4
    assert renderer.tone_mapping() == ("aces", 1.5), "kept across set_size"
    renderer.set_size(W, H)
    renderer.set_tone_mapping("none")
    plain = sum(px(renderer.render_rgba(s, cam), W // 2, H // 2))
    renderer.set_tone_mapping("linear", 0.25)
    dim = sum(px(renderer.render_rgba(s, cam), W // 2, H // 2))
    assert dim < plain * 0.6, (dim, plain)
    renderer.set_tone_mapping("none")
    with pytest.raises(ValueError, match="unknown tone mapping"):
        renderer.set_tone_mapping("filmic")
    with pytest.raises(ValueError, match="exposure"):
        renderer.set_tone_mapping("linear", -1)
    with pytest.raises(ValueError, match="set_size"):
        renderer.set_size(0, 10)

    # A box casting onto a ground plane.
    sh = scene()
    ground = sh.add_mesh(Geometry.plane(10, 10), Material.standard(color=0xFFFFFF))
    sh.set_rotation(ground, -math.pi / 2, 0, 0)
    sh.set_receive_shadow(ground, True)
    box = sh.add_mesh(Geometry.box(1, 1, 1), Material.standard(color=0xFFFFFF), 0, 1, 0)
    sh.set_cast_shadow(box, True)
    sh.add_directional_light(0xFFFFFF, 2, (5, 10, 0), cast_shadow=True)
    top = threers.PerspectiveCamera(50, 1, 0.1, 100)
    top.set_position(0, 8, 2)
    top.look_at(0, 0, 0)
    assert renderer.shadows_enabled()
    on = renderer.render_rgba(sh, top)
    renderer.set_shadows_enabled(False)
    off = renderer.render_rgba(sh, top)
    renderer.set_shadows_enabled(True)
    diffs = [
        sum(px(off, x, y)) - sum(px(on, x, y)) for y in range(H) for x in range(W)
    ]
    assert min(diffs) >= -6, "shadows only ever darken"
    assert max(diffs) > 150, f"the box's shadow darkens the ground: {max(diffs)}"


# ---- loaders -----------------------------------------------------------------

OBJ = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3\nf 2 4 3\n"
STL = (
    "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\n"
    "vertex 0 1 0\nendloop\nendfacet\nendsolid t\n"
)
PLY = "\n".join(
    [
        "ply", "format ascii 1.0", "element vertex 3", "property float x",
        "property float y", "property float z", "element face 1",
        "property list uchar int vertex_indices", "end_header",
        "0 0 0", "1 0 0", "0 1 0", "3 0 1 2", "",
    ]
)
GLB = REPO / "src/loaders/testdata/sphere_plain.glb"
USDA = REPO / "src/loaders/usd/testdata/apple.usda"
USDC = REPO / "src/loaders/usd/testdata/apple.usdc"


def lit_scene():
    s = scene()
    s.add_ambient_light(0xFFFFFF, 0.5)
    s.add_directional_light(0xFFFFFF, 2, (1, 2, 3))
    return s


def test_load_files(tmp_path):
    s = lit_scene()
    glb = s.load(GLB)
    for name, text in [("tri.obj", OBJ), ("tri.stl", STL), ("tri.ply", PLY)]:
        (tmp_path / name).write_text(text)
        s.load(str(tmp_path / name), material=Material.basic(color=0xFFFFFF))
    usd = s.load(USDA)
    assert s.object_count() == 2 + 5
    assert usd == glb + 4


def test_loaded_glb_renders(renderer):
    s = lit_scene()
    s.load(GLB)
    c = px(renderer.render_rgba(s, camera()), W // 2, H // 2)
    assert not near(c, BG_PX) and sum(c) > 30, c


def test_load_bytes_and_external_gltf_buffers(tmp_path):
    s = lit_scene()
    before = s.object_count()
    s.load_bytes(GLB.read_bytes(), "glb")
    s.load_bytes(OBJ.encode(), "obj")
    s.load_bytes(STL.encode(), "STL")
    s.load_bytes(PLY.encode(), "ply")
    s.load_bytes(USDC.read_bytes(), "usd")
    assert s.object_count() == before + 5

    # A .gltf whose triangle lives in a separate .bin next to it.
    (tmp_path / "tri data.bin").write_bytes(struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0))
    import json

    gltf = json.dumps(
        {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 0]}
            ],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "buffers": [{"byteLength": 36, "uri": "tri%20data.bin"}],
        }
    )
    (tmp_path / "tri.gltf").write_text(gltf)
    s.load(tmp_path / "tri.gltf")
    with pytest.raises(ValueError, match="external buffer"):
        s.load_bytes(gltf.encode(), "gltf")


def test_loaders_reject_bad_input(tmp_path):
    s = threers.Scene()
    with pytest.raises(FileNotFoundError):
        s.load(tmp_path / "missing.glb")
    with pytest.raises(ValueError, match="unknown model format 'dwg'"):
        s.load(tmp_path / "model.dwg")
    with pytest.raises(ValueError, match="no extension"):
        s.load(tmp_path / "noext")
    with pytest.raises(ValueError, match="glTF"):
        s.load_bytes(b"nope", "glb")
    with pytest.raises(ValueError, match="USD"):
        s.load_bytes(b"nope", "usd")
    with pytest.raises(ValueError, match="no geometry"):
        s.load_bytes(b"", "obj")
    with pytest.raises(ValueError, match="unknown model format"):
        s.load_bytes(OBJ.encode(), "dwg")
    with pytest.raises(TypeError):
        s.load(GLB, material=Geometry.box())
    assert s.object_count() == 0
