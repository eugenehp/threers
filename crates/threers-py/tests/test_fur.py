"""Fur and hair from Python — run after `maturin develop`."""

import threers


def _lit(rgba: bytes) -> int:
    return sum(1 for i in range(0, len(rgba), 4) if rgba[i] + rgba[i + 1] + rgba[i + 2] > 30)


def _scene():
    scene = threers.Scene()
    scene.set_background(0x000000)
    scene.add_ambient_light(0xFFFFFF, 1.5)
    camera = threers.PerspectiveCamera(40, 1, 0.1, 50)
    camera.set_position(0, 0, 5)
    camera.look_at(0, 0, 0)
    return scene, camera


def test_fur_material_options():
    fur = threers.FurMaterial(length=0.3, shells=24, gravity=(0, -0.5, 0), placement="surface", cell=0.03)
    assert abs(fur.length - 0.3) < 1e-6
    assert fur.shells == 24
    try:
        threers.FurMaterial(placement="nowhere")
    except ValueError:
        pass
    else:
        raise AssertionError("a bad placement was accepted")


def test_fur_and_hair_draw_past_the_surface():
    if threers.HeadlessRenderer is None:
        return
    renderer = threers.HeadlessRenderer(96, 96)
    renderer.set_msaa(4)
    for kind in ("fur", "hair"):
        scene, camera = _scene()
        fur = threers.FurMaterial(length=0.4, density=50, tip_color=0xFFE0B0, wind=(0.4, 0, 0), flutter=0.1)
        sphere = threers.Geometry.sphere(1.0, 32, 16)
        if kind == "fur":
            handle = scene.add_fur(sphere, fur)
        else:
            handle = scene.add_hair(sphere, fur, count=3000, segments=6, width=0.02)
        assert handle >= 0
        scene.step_fur(0.0)
        a = renderer.render_rgba(scene, camera)
        # The bare sphere is ~21 px in radius here; the fur reaches past it.
        assert _lit(a) > 3.14 * 21 * 21, kind
        scene.step_fur(0.7, 0.7)
        b = renderer.render_rgba(scene, camera)
        assert a != b, f"{kind}: the wind moved nothing"


def test_stroke_fur_moves_the_coat_and_it_springs_back():
    if threers.HeadlessRenderer is None:
        return
    renderer = threers.HeadlessRenderer(96, 96)
    scene, camera = _scene()
    fur = threers.FurMaterial(length=0.4, density=50, tip_color=0xFFE0B0)
    h = scene.add_fur(threers.Geometry.sphere(1.0, 32, 16), fur)
    for f in range(3):
        scene.step_fur(f / 60)
    before = renderer.render_rgba(scene, camera)
    scene.stroke_fur(h, (0.7, 0.7, 0.4), radius=0.4, push=(-1.5, 0, 0), part=1.0)
    scene.step_fur(3 / 60)
    stroked = renderer.render_rgba(scene, camera)
    assert stroked != before
    for f in range(60):
        scene.step_fur((4 + f) / 60)
    assert renderer.render_rgba(scene, camera) == before
