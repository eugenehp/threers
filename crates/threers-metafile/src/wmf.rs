//! Windows metafiles of 16 bits: an optional placeable header (the
//! picture's box and its units an inch), the header, then records of a
//! size in words and a function, their parameters often last first.

use crate::gdi::{Gdi, Obj, Output};
use crate::{colorref, dib, Drawing, Error, R};

const PLACEABLE: u32 = 0x9AC6_CDD7;

pub fn is_wmf(b: &[u8]) -> bool {
    let r = R(b);
    r.u32(0) == PLACEABLE || (b.len() >= 18 && matches!(r.u16(0), 1 | 2) && r.u16(2) == 9 && matches!(r.u16(4), 0x100 | 0x300))
}

pub fn read(bytes: &[u8]) -> Result<Drawing, Error> {
    let h = R(bytes);
    let (start, bbox, inch) = if h.u32(0) == PLACEABLE {
        let b = [h.i16(6), h.i16(8), h.i16(10), h.i16(12)].map(f32::from);
        (22, Some(b), f32::from(h.u16(14)).max(1.0))
    } else {
        (0, None, 1440.0)
    };
    let header_words = h.u16(start + 2) as usize;
    let mut at = start + 2 * header_words;

    // a first pass for the window, when there's no box to go by
    let mut win: Option<[f32; 4]> = None;
    {
        let mut a = at;
        let mut org = (0.0, 0.0);
        while a + 6 <= bytes.len() {
            let size = 2 * h.u32(a) as usize;
            let f = h.u16(a + 4);
            if size < 6 {
                break;
            }
            match f {
                0x020B => org = (f32::from(h.i16(a + 8)), f32::from(h.i16(a + 6))),
                0x020C => {
                    win = Some([org.0, org.1, f32::from(h.i16(a + 8)), f32::from(h.i16(a + 6))]);
                    break;
                }
                0 => break,
                _ => {}
            }
            a += size;
        }
    }
    let (width, height, origin, scale) = match (bbox, win) {
        (Some(b), _) => {
            let (w, hh) = ((b[2] - b[0]).abs() / inch * 72.0, (b[3] - b[1]).abs() / inch * 72.0);
            (w, hh, (b[0].min(b[2]), b[1].min(b[3])), (72.0 / inch, 72.0 / inch))
        }
        (None, Some(w)) => {
            let (ww, hh) = (w[2].abs() / 1440.0 * 72.0, w[3].abs() / 1440.0 * 72.0);
            ((ww).max(1.0), hh.max(1.0), (w[0], w[1]), (72.0 / 1440.0 * w[2].signum(), 72.0 / 1440.0 * w[3].signum()))
        }
        _ => (100.0, 100.0, (0.0, 0.0), (1.0, 1.0)),
    };
    let mut g = Gdi::new(Output { scale, origin, dev_per_mm: (inch / 25.4, inch / 25.4) });
    // logical coordinates map straight onto the box: the window is the box
    g.dc.map_mode = 1;
    let mut slots: Vec<bool> = Vec::new();
    let mut win_org = (0.0f32, 0.0f32);
    let mut win_ext: Option<(f32, f32)> = None;
    let fit = |g: &mut Gdi, org: (f32, f32), ext: Option<(f32, f32)>| {
        // the window drawn into the picture's box
        if let (Some(e), Some(b)) = (ext, bbox) {
            g.dc.map_mode = 7;
            g.dc.win_org = org;
            g.dc.win_ext = e;
            g.dc.vp_org = (b[0].min(b[2]), b[1].min(b[3]));
            g.dc.vp_ext = ((b[2] - b[0]).abs(), (b[3] - b[1]).abs());
        } else {
            g.dc.win_org = org;
        }
    };

    while at + 6 <= bytes.len() {
        let size = 2 * h.u32(at) as usize;
        let f = h.u16(at + 4);
        if size < 6 {
            return Err(Error::Truncated);
        }
        if at + size > bytes.len() {
            break;
        }
        let r = R(&bytes[at + 6..at + size]);
        let w = |i: usize| f32::from(r.i16(2 * i));
        let mut add = |g: &mut Gdi, o: Obj| {
            let i = slots.iter().position(|u| !u).unwrap_or(slots.len());
            if i == slots.len() {
                slots.push(true);
            } else {
                slots[i] = true;
            }
            g.objects.insert(i as u32, o);
        };
        match f {
            0 => break,
            0x020B => {
                win_org = (w(1), w(0));
                fit(&mut g, win_org, win_ext);
            }
            0x020C => {
                win_ext = Some((w(1), w(0)));
                fit(&mut g, win_org, win_ext);
            }
            0x0103 => {}
            0x0106 => g.dc.even_odd = r.u16(0) != 2,
            0x012E => g.dc.text_align = u32::from(r.u16(0)),
            0x0209 => g.dc.text_color = colorref(r.u32(0)),
            0x001E => g.save(),
            0x0127 => g.restore(i32::from(r.i16(0))),
            0x02FA => {
                let style = r.u16(0);
                add(&mut g, Obj::Pen { color: colorref(r.u32(6)), width: w(1), null: style & 0xF == 5 });
            }
            0x02FC => {
                let style = r.u16(0);
                add(&mut g, Obj::Brush { color: if style == 1 { None } else { Some(colorref(r.u32(2))) } });
            }
            0x02FB => {
                let name: String = r.slice(18, 32).iter().take_while(|&&c| c != 0).map(|&c| c as char).collect();
                add(&mut g, Obj::Font { height: w(0), width: w(1), weight: u32::from(r.u16(8)), italic: r.u8(10) != 0, family: name, angle: w(2) / 10.0 });
            }
            // a palette, a region, a pattern brush: objects that take a slot
            0x00F7 | 0x06FF | 0x01F9 | 0x0142 => add(&mut g, Obj::Brush { color: Some([128, 128, 128, 255]) }),
            0x012D => g.select(u32::from(r.u16(0))),
            0x01F0 => {
                let i = r.u16(0) as usize;
                g.objects.remove(&(i as u32));
                if i < slots.len() {
                    slots[i] = false;
                }
            }
            0x0214 => g.move_to(w(1), w(0)),
            0x0213 => g.line_to(w(1), w(0)),
            0x041B => g.rect(w(3), w(2), w(1), w(0), None),
            0x061C => g.rect(w(5), w(4), w(3), w(2), Some((w(1), w(0)))),
            0x0418 => g.ellipse(w(3), w(2), w(1), w(0)),
            0x0817 | 0x0830 | 0x081A => {
                // arc, chord, pie: end, start, box (bottom right top left)
                let k = match f {
                    0x0817 => 0,
                    0x0830 => 1,
                    _ => 2,
                };
                g.arc(k, [w(7), w(6), w(5), w(4)], (w(3), w(2)), (w(1), w(0)));
            }
            0x0324 | 0x0325 => {
                let n = r.u16(0) as usize;
                let pts: Vec<(f32, f32)> = (0..n).map(|i| (w(1 + 2 * i), w(2 + 2 * i))).collect();
                if f == 0x0324 {
                    g.polygons(&[pts]);
                } else {
                    g.polyline(&pts, false);
                }
            }
            0x0538 => {
                let n = r.u16(0) as usize;
                let counts: Vec<usize> = (0..n).map(|i| r.u16(2 + 2 * i) as usize).collect();
                let mut k = 1 + n;
                let mut polys = Vec::new();
                for c in counts {
                    polys.push((0..c).map(|i| (w(k + 2 * i), w(k + 2 * i + 1))).collect());
                    k += 2 * c;
                }
                g.polygons(&polys);
            }
            0x0521 => {
                let n = r.u16(0) as usize;
                let text = g.bytes(r.slice(2, n));
                let p = 2 + n.div_ceil(2) * 2;
                let (y, x) = (f32::from(r.i16(p)), f32::from(r.i16(p + 2)));
                g.text(x, y, &text, None);
            }
            0x0A32 => {
                let (y, x) = (w(0), w(1));
                let n = r.u16(4) as usize;
                let opts = r.u16(6);
                let s = if opts & 0x6 != 0 { 16 } else { 8 };
                let text = g.bytes(r.slice(s, n));
                let dx_at = s + n.div_ceil(2) * 2;
                let advance = (r.0.len() >= dx_at + 2 * n).then(|| (0..n).map(|i| f32::from(r.i16(dx_at + 2 * i))).sum::<f32>());
                g.text(x, y, &text, advance);
            }
            0x0F43 => {
                // STRETCHDIB: rop, usage, source box, destination box, the DIB
                let (dh, dw, dy, dx) = (w(7), w(8), w(9), w(10));
                let info = r.slice(22, r.0.len());
                if let Some(b) = dib_inline(info) {
                    g.image(dx, dy, dw, dh, b);
                }
            }
            0x0B41 => {
                let (dh, dw, dy, dx) = (w(6), w(7), w(8), w(9));
                if let Some(b) = dib_inline(r.slice(20, r.0.len())) {
                    g.image(dx, dy, dw, dh, b);
                }
            }
            0x0940 => {
                let (hh, ww, dy, dx) = (w(4), w(5), w(6), w(7));
                if let Some(b) = dib_inline(r.slice(16, r.0.len())) {
                    g.image(dx, dy, ww, hh, b);
                }
            }
            _ => {}
        }
        at += size;
    }
    Ok(Drawing { width, height, ops: g.ops })
}

/// A DIB with its bits straight after its header and palette.
fn dib_inline(b: &[u8]) -> Option<(u32, u32, Vec<u8>)> {
    let r = R(b);
    let header = r.u32(0) as usize;
    let count = if header == 12 { r.u16(10) } else { r.u16(14) };
    let used = if header == 12 { 0 } else { r.u32(32) as usize };
    let compression = if header == 12 { 0 } else { r.u32(16) };
    let pal = if count <= 8 { if used > 0 { used } else { 1 << count } } else { 0 } * if header == 12 { 3 } else { 4 };
    let extra = if compression == 3 && header == 40 { 12 } else { 0 };
    let bits = header + extra + pal;
    dib(r.slice(0, bits), r.slice(bits, b.len()))
}
