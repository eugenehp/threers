//! Enhanced metafiles: records of a type and a size, the header first.

use crate::gdi::{Gdi, Obj, Output};
use crate::{colorref, dib, Drawing, Error, R};

pub fn is_emf(b: &[u8]) -> bool {
    let r = R(b);
    b.len() >= 88 && r.u32(0) == 1 && r.u32(40) == 0x464D_4520
}

fn utf16(b: &[u8]) -> String {
    let units: Vec<u16> = b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    String::from_utf16_lossy(&units)
}

fn points16(r: R, at: usize, n: usize) -> Vec<(f32, f32)> {
    (0..n).map(|i| (f32::from(r.i16(at + 4 * i)), f32::from(r.i16(at + 4 * i + 2)))).collect()
}

fn points32(r: R, at: usize, n: usize) -> Vec<(f32, f32)> {
    (0..n).map(|i| (r.i32(at + 8 * i) as f32, r.i32(at + 8 * i + 4) as f32)).collect()
}

pub fn read(bytes: &[u8]) -> Result<Drawing, Error> {
    let h = R(bytes);
    // the frame, in hundredths of a millimetre; the reference device
    let frame = [h.i32(24), h.i32(28), h.i32(32), h.i32(36)].map(|v| v as f32);
    let (dev_px, dev_mm) = ((h.i32(72) as f32, h.i32(76) as f32), (h.i32(80) as f32, h.i32(84) as f32));
    let mm_per_dev = if dev_px.0 > 0.0 && dev_mm.0 > 0.0 { (dev_mm.0 / dev_px.0, dev_mm.1 / dev_px.1) } else { (0.2646, 0.2646) };
    let pt_per_mm = 72.0 / 25.4;
    let mut width = (frame[2] - frame[0]) / 100.0 * pt_per_mm;
    let mut height = (frame[3] - frame[1]) / 100.0 * pt_per_mm;
    let mut origin = (frame[0] / 100.0 / mm_per_dev.0, frame[1] / 100.0 / mm_per_dev.1);
    if width <= 0.0 || height <= 0.0 {
        // no frame: the bounds, in device pixels
        let b = [h.i32(8), h.i32(12), h.i32(16), h.i32(20)].map(|v| v as f32);
        width = (b[2] - b[0] + 1.0) * mm_per_dev.0 * pt_per_mm;
        height = (b[3] - b[1] + 1.0) * mm_per_dev.1 * pt_per_mm;
        origin = (b[0], b[1]);
    }
    let mut g = Gdi::new(Output {
        scale: (mm_per_dev.0 * pt_per_mm, mm_per_dev.1 * pt_per_mm),
        origin,
        dev_per_mm: (1.0 / mm_per_dev.0, 1.0 / mm_per_dev.1),
    });

    let mut at = 0;
    while at + 8 <= bytes.len() {
        let kind = h.u32(at);
        let size = h.u32(at + 4) as usize;
        if size < 8 || at + size > bytes.len() {
            if size < 8 {
                return Err(Error::Truncated);
            }
            break;
        }
        let r = R(&bytes[at..at + size]);
        let p = |i: usize| r.i32(8 + 4 * i) as f32;
        match kind {
            14 => break,
            9 => g.dc.win_ext = (p(0), p(1)),
            10 => g.dc.win_org = (p(0), p(1)),
            11 => g.dc.vp_ext = (p(0), p(1)),
            12 => g.dc.vp_org = (p(0), p(1)),
            17 => g.dc.map_mode = r.u32(8),
            19 => g.dc.even_odd = r.u32(8) != 2,
            22 => g.dc.text_align = r.u32(8),
            24 => g.dc.text_color = colorref(r.u32(8)),
            33 => g.save(),
            34 => g.restore(r.i32(8)),
            35 => g.dc.xform = [r.f32(8), r.f32(12), r.f32(16), r.f32(20), r.f32(24), r.f32(28)],
            36 => {
                let m = [r.f32(8), r.f32(12), r.f32(16), r.f32(20), r.f32(24), r.f32(28)];
                let w = g.dc.xform;
                let mul = |a: [f32; 6], b: [f32; 6]| {
                    [
                        a[0] * b[0] + a[1] * b[2],
                        a[0] * b[1] + a[1] * b[3],
                        a[2] * b[0] + a[3] * b[2],
                        a[2] * b[1] + a[3] * b[3],
                        a[4] * b[0] + a[5] * b[2] + b[4],
                        a[4] * b[1] + a[5] * b[3] + b[5],
                    ]
                };
                g.dc.xform = match r.u32(32) {
                    1 => [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    2 => mul(m, w),
                    3 => mul(w, m),
                    _ => m,
                };
            }
            37 => g.select(r.u32(8)),
            38 => {
                let style = r.u32(12);
                g.objects.insert(r.u32(8), Obj::Pen { color: colorref(r.u32(24)), width: r.i32(16) as f32, null: style & 0xF == 5 });
            }
            95 => {
                let style = r.u32(28);
                g.objects.insert(r.u32(8), Obj::Pen { color: colorref(r.u32(40)), width: r.u32(32) as f32, null: style & 0xF == 5 });
            }
            39 => {
                let style = r.u32(12);
                let color = if style == 1 { None } else { Some(colorref(r.u32(16))) };
                g.objects.insert(r.u32(8), Obj::Brush { color });
            }
            40 => {
                g.objects.remove(&r.u32(8));
            }
            82 => {
                let lf = 12;
                g.objects.insert(
                    r.u32(8),
                    Obj::Font {
                        height: r.i32(lf) as f32,
                        width: r.i32(lf + 4) as f32,
                        weight: r.u32(lf + 16),
                        italic: r.u8(lf + 20) != 0,
                        family: utf16(r.slice(lf + 28, 64)).split('\0').next().unwrap_or("").to_string(),
                        angle: r.i32(lf + 8) as f32 / 10.0,
                    },
                );
            }
            27 => g.move_to(p(0), p(1)),
            54 => g.line_to(p(0), p(1)),
            43 => g.rect(p(0), p(1), p(2), p(3), None),
            44 => g.rect(p(0), p(1), p(2), p(3), Some((p(4), p(5)))),
            42 => g.ellipse(p(0), p(1), p(2), p(3)),
            45..=47 => {
                let k = match kind {
                    45 => 0,
                    46 => 1,
                    _ => 2,
                };
                g.arc(k, [p(0), p(1), p(2), p(3)], (p(4), p(5)), (p(6), p(7)));
            }
            // the 16-bit polys: bounds, count, points
            85..=89 => {
                let n = r.u32(24) as usize;
                let pts = points16(r, 28, n);
                match kind {
                    85 => g.beziers(&pts, false),
                    86 => g.polygons(&[pts]),
                    87 => g.polyline(&pts, false),
                    88 => g.beziers(&pts, true),
                    _ => g.polyline(&pts, true),
                }
            }
            2..=6 => {
                let n = r.u32(24) as usize;
                let pts = points32(r, 28, n);
                match kind {
                    2 => g.beziers(&pts, false),
                    3 => g.polygons(&[pts]),
                    4 => g.polyline(&pts, false),
                    5 => g.beziers(&pts, true),
                    _ => g.polyline(&pts, true),
                }
            }
            // poly-polys: bounds, polys, points, counts, then the points
            7 | 8 | 90 | 91 => {
                let n = r.u32(24) as usize;
                let total = r.u32(28) as usize;
                let counts: Vec<usize> = (0..n.min(10000)).map(|i| r.u32(32 + 4 * i) as usize).collect();
                let small = kind >= 90;
                let base = 32 + 4 * n;
                let all = if small { points16(r, base, total) } else { points32(r, base, total) };
                let mut polys = Vec::new();
                let mut k = 0;
                for c in counts {
                    polys.push(all.get(k..(k + c).min(all.len())).unwrap_or(&[]).to_vec());
                    k += c;
                }
                if kind == 8 || kind == 91 {
                    g.polygons(&polys);
                } else {
                    for p in polys {
                        g.polyline(&p, false);
                    }
                }
            }
            59 => g.begin_path(),
            60 => g.end_path(),
            61 => g.close_figure(),
            62 => g.paint_path(true, false),
            63 => g.paint_path(true, true),
            64 => g.paint_path(false, true),
            67 | 68 => g.path = None,
            84 | 83 => {
                // EMRTEXT: the reference point, the characters, their advances
                let t = 36;
                let (x, y) = (r.i32(t) as f32, r.i32(t + 4) as f32);
                let n = r.u32(t + 8) as usize;
                let off = r.u32(t + 12) as usize;
                let text = if kind == 84 { utf16(r.slice(off, 2 * n)) } else { g.bytes(r.slice(off, n)) };
                let dx_off = r.u32(t + 36) as usize;
                let advance = (dx_off > 0 && dx_off + 4 * n <= size).then(|| (0..n).map(|i| r.i32(dx_off + 4 * i) as f32).sum::<f32>());
                let (x, y) = if g.dc.text_align & 1 != 0 { g.dc.cur } else { (x, y) };
                g.text(x, y, &text, advance);
            }
            81 => {
                // STRETCHDIBITS
                let (x, y) = (p(4), p(5));
                let (w, hh) = (r.i32(72) as f32, r.i32(76) as f32);
                let (bmi, cbmi, bits, cbits) = (r.u32(48) as usize, r.u32(52) as usize, r.u32(56) as usize, r.u32(60) as usize);
                if let Some(b) = dib(r.slice(bmi, cbmi), r.slice(bits, cbits)) {
                    g.image(x, y, w, hh, b);
                }
            }
            76 | 77 => {
                // BITBLT, STRETCHBLT with a bitmap
                let (x, y, w, hh) = (p(4), p(5), p(6), p(7));
                let (bmi, cbmi, bits, cbits) = (r.u32(84) as usize, r.u32(88) as usize, r.u32(92) as usize, r.u32(96) as usize);
                if cbmi > 0
                    && let Some(b) = dib(r.slice(bmi, cbmi), r.slice(bits, cbits))
                {
                    g.image(x, y, w, hh, b);
                } else if r.u32(40) == 0x00F0_0021 {
                    // PATCOPY: the brush over the rectangle
                    let pen = std::mem::replace(&mut g.dc.pen, Obj::Pen { color: [0; 4], width: 0.0, null: true });
                    g.rect(x, y, x + w, y + hh, None);
                    g.dc.pen = pen;
                }
            }
            80 => {
                // SETDIBITSTODEVICE
                let (x, y) = (p(4), p(5));
                let (w, hh) = (r.i32(40) as f32, r.i32(44) as f32);
                let (bmi, cbmi, bits, cbits) = (r.u32(48) as usize, r.u32(52) as usize, r.u32(56) as usize, r.u32(60) as usize);
                if let Some(b) = dib(r.slice(bmi, cbmi), r.slice(bits, cbits)) {
                    g.image(x, y, w, hh, b);
                }
            }
            _ => {}
        }
        at += size;
    }
    Ok(Drawing { width, height, ops: g.ops })
}
