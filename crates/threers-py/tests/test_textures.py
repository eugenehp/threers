"""Textures, map slots and MeshPhysicalMaterial's options — run after
`maturin develop`."""

import struct
import zlib

import pytest

import threers
from threers import Geometry, Material, Texture

W = H = 32


@pytest.fixture
def renderer():
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        return threers.HeadlessRenderer(W, H)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")


def centre(renderer, mat, x=W // 2, y=H // 2):
    """The pixel at (x, y) of a unit-facing plane filling the view."""
    s = threers.Scene()
    s.set_background(0x000000)
    s.add_mesh(Geometry.plane(2, 2), mat)
    cam = threers.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
    cam.set_position(0, 0, 5)
    cam.look_at(0, 0, 0)
    rgba = renderer.render_rgba(s, cam)
    i = (y * W + x) * 4
    return tuple(rgba[i : i + 3])


def texel(r, g, b, a=255, color_space="srgb"):
    return Texture(1, 1, bytes([r, g, b, a]), color_space)


def write_png(path, w, h, rgba):
    rows = b"".join(b"\0" + rgba[y * w * 4 : (y + 1) * w * 4] for y in range(h))

    def chunk(kind, data):
        c = kind + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


# ---- Texture -----------------------------------------------------------------


def test_texture_defaults_follow_three_js():
    t = Texture(2, 1, bytes(8))
    assert (t.width, t.height, t.color_space) == (2, 1, "srgb")
    assert (t.wrap_s, t.wrap_t) == ("clamp", "clamp")
    assert (t.mag_filter, t.min_filter, t.mipmap_filter) == ("linear", "linear", "linear")
    assert (t.offset, t.repeat, t.rotation, t.center) == ((0, 0), (1, 1), 0, (0, 0))
    assert t.channel == 0
    assert t.flip_y


def test_texture_properties_round_trip():
    t = Texture(1, 1, bytes(4), "linear")
    t.wrap_s, t.wrap_t = "repeat", "mirror"
    t.mag_filter, t.min_filter, t.mipmap_filter = "nearest", "nearest", None
    t.offset, t.repeat, t.rotation, t.center = (0.25, 0.5), (2, 3), 0.5, (0.5, 0.5)
    t.channel = 2
    t.color_space = "srgb"
    assert (t.wrap_s, t.wrap_t) == ("repeat", "mirror")
    assert (t.mag_filter, t.min_filter, t.mipmap_filter) == ("nearest", "nearest", None)
    assert (t.offset, t.repeat, t.center) == ((0.25, 0.5), (2, 3), (0.5, 0.5))
    assert t.rotation == pytest.approx(0.5)
    assert (t.channel, t.color_space) == (2, "srgb")


@pytest.mark.parametrize(
    "make, match",
    [
        (lambda: Texture(2, 2, bytes(15)), "takes 16 bytes"),
        (lambda: Texture(0, 1, b""), "takes 0 bytes"),
        (lambda: Texture(1, 1, bytes(4), "rec709"), "color_space"),
    ],
)
def test_texture_rejects_bad_input(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_texture_rejects_bad_properties():
    t = Texture(1, 1, bytes(4))
    with pytest.raises(ValueError, match="wrap"):
        t.wrap_s = "border"
    with pytest.raises(ValueError, match="filter"):
        t.min_filter = "cubic"
    with pytest.raises(ValueError, match="channel"):
        t.channel = 4


def test_texture_from_png(tmp_path):
    p = tmp_path / "t.png"
    write_png(p, 2, 3, bytes(range(24)))
    t = Texture.from_file(str(p), "linear")
    assert (t.width, t.height, t.color_space) == (2, 3, "linear")
    with pytest.raises(ValueError):
        Texture.from_file(str(tmp_path / "missing.png"))
    (tmp_path / "bad.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="PNG"):
        Texture.from_file(str(tmp_path / "bad.png"))


# ---- map slots ---------------------------------------------------------------


def test_set_map_fills_and_empties_a_slot():
    m = Material.physical()
    assert m.get_map("sheenColorMap") is None
    m.set_map("sheenColorMap", Texture(3, 2, bytes(24)))
    got = m.get_map("sheenColorMap")
    assert (got.width, got.height) == (3, 2)
    m.set_map("sheenColorMap", None)
    assert m.get_map("sheenColorMap") is None


def test_every_physical_slot_is_settable():
    m = Material.physical()
    for slot in [
        "map", "alphaMap", "normalMap", "aoMap", "emissiveMap", "roughnessMap",
        "metalnessMap", "lightMap", "bumpMap", "displacementMap",
        "clearcoatMap", "clearcoatRoughnessMap", "clearcoatNormalMap",
        "iridescenceMap", "iridescenceThicknessMap", "sheenColorMap",
        "sheenRoughnessMap", "specularIntensityMap", "specularColorMap",
        "transmissionMap", "thicknessMap", "anisotropyMap",
    ]:
        m.set_map(slot, texel(255, 255, 255))
        assert m.get_map(slot) is not None, slot


def test_set_map_rejects_a_slot_the_material_lacks():
    with pytest.raises(ValueError, match="no 'roughnessMap'"):
        Material.basic().set_map("roughnessMap", texel(0, 0, 0))
    with pytest.raises(ValueError, match="no 'clearcoatMap'"):
        Material.standard().set_map("clearcoatMap", None)


def test_a_texture_edit_after_set_map_does_not_leak_in():
    # three.js shares the texture object; Python gets a copy-on-write handle,
    # so the material keeps the state it was given.
    t = texel(255, 0, 0)
    m = Material.basic()
    m.set_map("map", t)
    t.wrap_s = "repeat"
    assert m.get_map("map").wrap_s == "clamp"


# ---- rendering ---------------------------------------------------------------


def test_map_tints_the_surface(renderer):
    m = Material.basic(color=0xFFFFFF)
    m.set_map("map", texel(255, 0, 0))
    assert centre(renderer, m) == (255, 0, 0)


def test_color_space_decides_how_the_texel_is_read(renderer):
    srgb = Material.basic(color=0xFFFFFF)
    srgb.set_map("map", texel(128, 128, 128, color_space="srgb"))
    linear = Material.basic(color=0xFFFFFF)
    linear.set_map("map", texel(128, 128, 128, color_space="linear"))
    # sRGB 128 decodes and re-encodes to itself; linear 0.502 encodes to 188.
    assert centre(renderer, srgb)[0] == pytest.approx(128, abs=1)
    assert centre(renderer, linear)[0] == pytest.approx(188, abs=1)


def test_offset_and_wrap_move_the_image(renderer):
    # Two texels, red left and blue right, sampled nearest.
    t = Texture(2, 1, bytes([255, 0, 0, 255, 0, 0, 255, 255]))
    t.mag_filter = t.min_filter = "nearest"
    m = Material.basic(color=0xFFFFFF)
    m.set_map("map", t)
    left, right = W // 4, 3 * W // 4
    assert centre(renderer, m, x=left) == (255, 0, 0)
    assert centre(renderer, m, x=right) == (0, 0, 255)
    # Half a turn of the image along u, repeating: the halves swap.
    t.offset, t.wrap_s = (0.5, 0), "repeat"
    m.set_map("map", t)
    assert centre(renderer, m, x=left) == (0, 0, 255)
    assert centre(renderer, m, x=right) == (255, 0, 0)


def test_alpha_map_cuts_with_alpha_test_free_transparency(renderer):
    m = Material.basic(color=0xFFFFFF, transparent=True)
    m.set_map("alphaMap", texel(0, 0, 0, color_space="linear"))
    assert centre(renderer, m) == (0, 0, 0)


# ---- MeshPhysicalMaterial ----------------------------------------------------


def test_physical_takes_every_three_js_scalar():
    m = Material.physical(
        specular_intensity=0.5,
        specular_color=0xFF8040,
        sheen=1,
        sheen_color=0x808080,
        sheen_roughness=0.3,
        iridescence=1,
        iridescence_ior=1.8,
        iridescence_thickness_range=(200, 800),
        anisotropy=0.6,
        anisotropy_rotation=1.2,
        attenuation_color=0x40A0FF,
        attenuation_distance=0.5,
        dispersion=0.2,
        transmission=1,
        thickness=0.4,
    )
    assert m.kind == "physical"


@pytest.mark.parametrize(
    "kwargs, err, match",
    [
        ({"sheen": 2}, ValueError, "sheen"),
        ({"iridescence_ior": 3}, ValueError, "iridescence_ior"),
        ({"iridescence_thickness_range": (-1, 400)}, ValueError, "iridescence_thickness_range"),
        ({"attenuation_distance": -1}, ValueError, "attenuation_distance"),
        ({"specular_color": 0x1000000}, ValueError, "specular_color"),
    ],
)
def test_physical_validates_its_scalars(kwargs, err, match):
    with pytest.raises(err, match=match):
        Material.physical(**kwargs)


def test_physical_only_options_are_rejected_elsewhere():
    with pytest.raises(TypeError, match="'sheen' does not apply"):
        Material.standard(sheen=1)


def test_specular_intensity_scales_the_highlight(renderer):
    # Matte black dielectric under a light straight on: only the specular
    # lobe shows, and three.js scales F0 and F90 by specularIntensity.
    def lit(intensity):
        s = threers.Scene()
        s.set_background(0x000000)
        s.add_directional_light(0xFFFFFF, 3, (0, 0, 1))
        s.add_mesh(
            Geometry.plane(2, 2),
            Material.physical(color=0x000000, roughness=0.5, specular_intensity=intensity),
        )
        cam = threers.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
        cam.set_position(0, 0, 5)
        cam.look_at(0, 0, 0)
        rgba = renderer.render_rgba(s, cam)
        i = (H // 2 * W + W // 2) * 4
        return rgba[i]

    full, half, off = lit(1.0), lit(0.5), lit(0.0)
    assert full > half > off == 0


# ---- MeshToonMaterial ----------------------------------------------------------


def test_toon_gradient_map_tints_each_light(renderer):
    # A light of intensity π straight on, through a one-texel red ramp: three.js
    # gives irradiance × ramp ÷ π = pure red on a white toon surface.
    def lit(gradient):
        m = Material.toon(color=0xFFFFFF)
        assert m.kind == "toon"
        m.set_map("gradientMap", gradient)
        s = threers.Scene()
        s.set_background(0x000000)
        s.add_directional_light(0xFFFFFF, 3.14159265, (0, 0, 1))
        s.add_mesh(Geometry.plane(2, 2), m)
        cam = threers.OrthographicCamera(-1, 1, 1, -1, 0.1, 10)
        cam.set_position(0, 0, 5)
        cam.look_at(0, 0, 0)
        rgba = renderer.render_rgba(s, cam)
        i = (H // 2 * W + W // 2) * 4
        return tuple(rgba[i : i + 3])

    red = texel(255, 0, 0, color_space="linear")
    red.mag_filter = red.min_filter = "nearest"
    assert lit(red) == (255, 0, 0)
    # No map: three.js's two tones, all of a light on the lit side.
    assert lit(None) == (255, 255, 255)


def test_toon_takes_lambert_like_options_only():
    Material.toon(color=0x123456, emissive=0x010203, opacity=0.5, side="double")
    with pytest.raises(TypeError, match="'roughness' does not apply"):
        Material.toon(roughness=0.5)
