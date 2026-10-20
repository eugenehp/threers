"""Lights — run after `maturin develop`."""

import math

import pytest

import threers

W = H = 64


@pytest.fixture(scope="module")
def renderer():
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        return threers.HeadlessRenderer(W, H)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")


def px(rgba, x, y):
    i = (y * W + x) * 4
    return tuple(rgba[i : i + 3])


def box_scene():
    """A unit box at the origin seen from (3, 0, 3): its +x face fills the
    right of the image and its +z face the left."""
    scene = threers.Scene()
    scene.set_background(0x204060)
    scene.add_box(1, 1, 1, color=0xFF6633)
    camera = threers.PerspectiveCamera(50, 1, 0.1, 100)
    camera.set_position(3, 0, 3)
    camera.look_at(0, 0, 0)
    return scene, camera


RIGHT = (W // 2 + 8, H // 2)  # +x face
LEFT = (W // 2 - 8, H // 2)  # +z face


def test_no_lights_renders_standard_black(renderer):
    scene, camera = box_scene()
    rgba = renderer.render_rgba(scene, camera)
    assert sum(px(rgba, *RIGHT)) < 6, "box is black (three.js semantics)"
    assert sum(px(rgba, 1, 1)) > 60, "background is not"


def test_directional_from_x_lights_the_x_face(renderer):
    scene, camera = box_scene()
    assert scene.add_directional_light(0xFFFFFF, 3, position=(5, 0, 0)) == 1
    rgba = renderer.render_rgba(scene, camera)
    lit, unlit = px(rgba, *RIGHT), px(rgba, *LEFT)
    assert lit[0] > 100, lit
    assert lit[0] > lit[2], f"keeps the box's orange: {lit}"
    assert sum(lit) > sum(unlit) + 100, (lit, unlit)


def test_ambient_alone_is_uniform(renderer):
    scene, camera = box_scene()
    scene.add_ambient_light(0xFFFFFF, 1)
    rgba = renderer.render_rgba(scene, camera)
    a, b = px(rgba, *RIGHT), px(rgba, *LEFT)
    assert sum(a) > 30, a
    assert all(abs(x - y) <= 2 for x, y in zip(a, b)), (a, b)


@pytest.mark.parametrize(
    "add",
    [
        lambda s: s.add_point_light(0xFFFFFF, 40, position=(3, 0, 0)),
        lambda s: s.add_spot_light(
            0xFFFFFF, 40, position=threers.Vector3(3, 0, 0), target=(0, 0, 0), angle=0.6, penumbra=0.2
        ),
        lambda s: s.add_hemisphere_light(0xFFFFFF, 0x444444, 2),
    ],
    ids=["point", "spot", "hemisphere"],
)
def test_other_lights_light_the_box(renderer, add):
    scene, camera = box_scene()
    add(scene)
    assert sum(px(renderer.render_rgba(scene, camera), *RIGHT)) > 60


def test_light_count_and_handles():
    scene = threers.Scene()
    assert scene.light_count() == 0
    assert scene.add_box() == 0
    assert scene.add_ambient_light() == 1
    assert scene.add_directional_light(0xFFFFFF, 2, (1, 2, 3), cast_shadow=True) == 2
    assert scene.add_point_light(0xFF0000, 5, (0, 2, 0), distance=10, decay=1) == 3
    assert scene.add_spot_light() == 4
    assert scene.add_hemisphere_light(0x87CEEB, 0x362907, 0.5) == 5
    assert scene.light_count() == 5
    assert scene.object_count() == 6
    scene.set_position(3, 0, 5, 0)
    p = scene.get_position(3)
    assert (p.x, p.y, p.z) == (0, 5, 0)
    scene.remove(3)
    assert scene.light_count() == 4


@pytest.mark.parametrize(
    "call, exc, match",
    [
        (lambda s: s.add_ambient_light(0x1000000), ValueError, "0xRRGGBB"),
        (lambda s: s.add_ambient_light(intensity=-1), ValueError, "intensity"),
        (lambda s: s.add_directional_light(intensity=math.nan), ValueError, "intensity"),
        (lambda s: s.add_directional_light(position=(0, 0, 0)), ValueError, "origin"),
        (lambda s: s.add_directional_light(position=(math.inf, 0, 0)), ValueError, "finite"),
        (lambda s: s.add_directional_light(position="up"), TypeError, "Vector3"),
        (lambda s: s.add_point_light(distance=-2), ValueError, "distance"),
        (lambda s: s.add_point_light(decay=-1), ValueError, "decay"),
        (lambda s: s.add_spot_light(position=(0, 0, 0)), ValueError, "differ"),
        (lambda s: s.add_spot_light(angle=2), ValueError, "angle"),
        (lambda s: s.add_spot_light(penumbra=2), ValueError, "penumbra"),
        (lambda s: s.add_hemisphere_light(ground_color=0x1000000), ValueError, "ground_color"),
    ],
)
def test_lights_reject_invalid_input(call, exc, match):
    scene = threers.Scene()
    with pytest.raises(exc, match=match):
        call(scene)
    assert scene.light_count() == 0
    assert scene.object_count() == 0
