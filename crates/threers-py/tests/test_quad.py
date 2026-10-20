"""Quad geometry and quad view — run after `maturin develop`."""

import struct
import zlib

import pytest

import threers

CORNERS = [-1, 0, -1, 1, 0, -1, 1, 0, 1, -1, 0, 1]


def test_add_quad_adds_an_object():
    scene = threers.Scene()
    assert scene.add_quad(CORNERS, color=0x33AA55) == 0
    pts = [threers.Vector3(0, 0, 0), (1, 0, 0), (1, 1, 0), threers.Vector3(0, 1, 0)]
    assert scene.add_quad(pts, 0xFFFFFF, width_segments=4, height_segments=2) == 1
    assert scene.object_count() == 2


@pytest.mark.parametrize(
    "corners, kwargs, exc, match",
    [
        ([1, 2, 3], {}, ValueError, "12 numbers or 4 points"),
        (CORNERS[:11] + [float("nan")], {}, ValueError, "non-finite"),
        ([0] * 12, {}, ValueError, "degenerate"),
        (CORNERS, {"width_segments": 0}, ValueError, "width_segments"),
        ([(0, 0, 0), (1, 0, 0), "x", (0, 1, 0)], {}, TypeError, "corner 2"),
        (42, {}, TypeError, "sequence"),
    ],
)
def test_add_quad_rejects_bad_input(corners, kwargs, exc, match):
    scene = threers.Scene()
    with pytest.raises(exc, match=match):
        scene.add_quad(corners, **kwargs)
    assert scene.object_count() == 0


def decode_png(data):
    """Minimal decoder for 8-bit RGBA non-interlaced PNGs (what encode_png writes)."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    off, idat = 8, b""
    while off < len(data):
        (length,) = struct.unpack(">I", data[off : off + 4])
        kind = data[off + 4 : off + 8]
        chunk = data[off + 8 : off + 8 + length]
        if kind == b"IHDR":
            width, height, depth, ctype = struct.unpack(">IIBB", chunk[:10])
            assert (depth, ctype) == (8, 6)
        elif kind == b"IDAT":
            idat += chunk
        off += 12 + length
    raw = zlib.decompress(idat)
    stride = width * 4
    out = bytearray(stride * height)
    for y in range(height):
        ft = raw[y * (stride + 1)]
        line = raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)]
        for i in range(stride):
            a = out[y * stride + i - 4] if i >= 4 else 0
            b = out[(y - 1) * stride + i] if y > 0 else 0
            c = out[(y - 1) * stride + i - 4] if y > 0 and i >= 4 else 0
            v = line[i]
            if ft == 1:
                v += a
            elif ft == 2:
                v += b
            elif ft == 3:
                v += (a + b) >> 1
            elif ft == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                v += a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            out[y * stride + i] = v & 0xFF
    return width, height, bytes(out)


def make_renderer(w, h):
    if threers.HeadlessRenderer is None:
        pytest.skip("built without `headless`")
    try:
        return threers.HeadlessRenderer(w, h)
    except RuntimeError as e:
        pytest.skip(f"no GPU adapter: {e}")


def test_render_quad_view_png_has_four_panes():
    r = make_renderer(200, 160)
    scene = threers.Scene()
    scene.set_background(0x2050A0)
    scene.add_box(1, 1, 1, color=0xFF6633)
    camera = threers.PerspectiveCamera(45, 1, 0.1, 100)
    camera.set_position(3, 2.5, 4)
    camera.look_at(0, 0, 0)

    png = r.render_quad_view(scene, camera, gap=4, gap_color=0xFF00FF)
    width, height, rgba = decode_png(png)
    assert (width, height) == (200, 160)

    def px(x, y):
        i = (y * width + x) * 4
        return tuple(rgba[i : i + 3])

    # Panes are 98 x 78 with a 4 px gap at x 98..101 and y 78..81 (top-left origin).
    gap = px(99, 79)
    assert gap[0] > 200 and gap[1] < 60 and gap[2] > 200, gap
    for (x, y), (cx, cy) in [
        ((5, 5), (49, 39)),
        ((195, 5), (151, 39)),
        ((5, 155), (49, 121)),
        ((195, 155), (151, 121)),
    ]:
        bg = px(x, y)
        assert bg != gap and bg[2] > bg[0], f"pane background at {x},{y}: {bg}"
        assert px(cx, cy) != bg, f"box visible in pane centred at {cx},{cy}"

    raw = r.render_quad_view(scene, camera, maximized="front", format="rgba")
    assert len(raw) == 200 * 160 * 4
    i = (79 * 200 + 99) * 4
    assert tuple(raw[i : i + 3]) != gap  # one pane: no gap down the middle

    with pytest.raises(ValueError, match="unknown pane"):
        r.render_quad_view(scene, camera, maximized="diagonal")
