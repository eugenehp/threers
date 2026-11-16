//! TrueType glyphs (`glyf`): read in font units, then scaled to 26.6
//! pixels, hinted or not, as `FT_Load_Glyph` gives them.

use crate::fixed::{mul_fix, pix_round};
use crate::sfnt::{Error, Face, Reader};
use crate::size::Size;

/// The tag of an on-curve point (`FT_CURVE_TAG_ON`).
pub const TAG_ON: u8 = 1;

/// A glyph's outline in 26.6 pixels: points, their tags, and the index of
/// each contour's last point.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Outline {
    pub points: Vec<(i64, i64)>,
    pub tags: Vec<u8>,
    pub contours: Vec<u16>,
}

/// A loaded glyph: its outline, and its advance and left bearing in 26.6.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Glyph {
    pub outline: Outline,
    pub advance: i64,
    pub left_bearing: i64,
}

/// A glyph as `glyf` has it, before scaling.
#[derive(Clone, Debug)]
pub(crate) enum RawGlyph<'a> {
    Empty,
    Simple {
        bbox: [i16; 4],
        points: Vec<(i32, i32)>,
        flags: Vec<u8>,
        contours: Vec<u16>,
        instructions: &'a [u8],
    },
    Composite {
        bbox: [i16; 4],
        components: Vec<Component>,
        instructions: &'a [u8],
    },
}

/// A component of a composite glyph.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Component {
    pub flags: u16,
    pub gid: u16,
    pub arg1: i32,
    pub arg2: i32,
    /// The 2x2 transform in 16.16, row by row (xx, xy, yx, yy).
    pub transform: Option<[i64; 4]>,
}

pub(crate) const ARG_1_AND_2_ARE_WORDS: u16 = 0x0001;
pub(crate) const ARGS_ARE_XY_VALUES: u16 = 0x0002;
pub(crate) const ROUND_XY_TO_GRID: u16 = 0x0004;
pub(crate) const WE_HAVE_A_SCALE: u16 = 0x0008;
pub(crate) const MORE_COMPONENTS: u16 = 0x0020;
pub(crate) const WE_HAVE_AN_X_AND_Y_SCALE: u16 = 0x0040;
pub(crate) const WE_HAVE_A_TWO_BY_TWO: u16 = 0x0080;
pub(crate) const WE_HAVE_INSTRUCTIONS: u16 = 0x0100;
pub(crate) const USE_MY_METRICS: u16 = 0x0200;
pub(crate) const SCALED_COMPONENT_OFFSET: u16 = 0x0800;
pub(crate) const UNSCALED_COMPONENT_OFFSET: u16 = 0x1000;

/// An F2Dot14 as 16.16.
fn f2dot14(v: i16) -> i64 {
    i64::from(v) * 4
}

/// The glyph's `glyf` entry, parsed.
pub(crate) fn parse(face: &Face, gid: u16) -> Result<RawGlyph<'_>, Error> {
    let data = face.glyph_data(gid);

    if data.len() < 10 {
        return Ok(RawGlyph::Empty);
    }

    let r = Reader::new(data);
    let n = r.i16(0)?;
    let bbox = [r.i16(2)?, r.i16(4)?, r.i16(6)?, r.i16(8)?];

    if n >= 0 {
        let n = n as usize;
        let mut contours = Vec::with_capacity(n);

        for i in 0..n {
            contours.push(r.u16(10 + 2 * i)?);
        }

        let count = contours.last().map_or(0, |&e| e as usize + 1);
        let ins_len = r.u16(10 + 2 * n)? as usize;
        let instructions = r.slice(12 + 2 * n, ins_len)?;
        let mut at = 12 + 2 * n + ins_len;
        let mut flags = Vec::with_capacity(count);

        while flags.len() < count {
            let f = r.u8(at)?;
            at += 1;
            flags.push(f);

            if f & 8 != 0 {
                let repeat = r.u8(at)?;
                at += 1;

                for _ in 0..repeat {
                    flags.push(f);
                }
            }
        }

        flags.truncate(count);
        let mut points = vec![(0i32, 0i32); count];
        let mut v = 0i32;

        for (i, &f) in flags.iter().enumerate() {
            if f & 0x02 != 0 {
                let d = i32::from(r.u8(at)?);
                at += 1;
                v += if f & 0x10 != 0 { d } else { -d };
            } else if f & 0x10 == 0 {
                v += i32::from(r.i16(at)?);
                at += 2;
            }

            points[i].0 = v;
        }

        v = 0;

        for (i, &f) in flags.iter().enumerate() {
            if f & 0x04 != 0 {
                let d = i32::from(r.u8(at)?);
                at += 1;
                v += if f & 0x20 != 0 { d } else { -d };
            } else if f & 0x20 == 0 {
                v += i32::from(r.i16(at)?);
                at += 2;
            }

            points[i].1 = v;
        }

        return Ok(RawGlyph::Simple { bbox, points, flags, contours, instructions });
    }

    let mut components = Vec::new();
    let mut at = 10;
    let mut have_instructions = false;

    loop {
        let flags = r.u16(at)?;
        let gid = r.u16(at + 2)?;
        at += 4;

        let (arg1, arg2) = if flags & ARG_1_AND_2_ARE_WORDS != 0 {
            let a = if flags & ARGS_ARE_XY_VALUES != 0 {
                (i32::from(r.i16(at)?), i32::from(r.i16(at + 2)?))
            } else {
                (i32::from(r.u16(at)?), i32::from(r.u16(at + 2)?))
            };
            at += 4;
            a
        } else {
            let a = if flags & ARGS_ARE_XY_VALUES != 0 {
                (i32::from(r.i8(at)?), i32::from(r.i8(at + 1)?))
            } else {
                (i32::from(r.u8(at)?), i32::from(r.u8(at + 1)?))
            };
            at += 2;
            a
        };

        let transform = if flags & WE_HAVE_A_SCALE != 0 {
            let s = f2dot14(r.i16(at)?);
            at += 2;
            Some([s, 0, 0, s])
        } else if flags & WE_HAVE_AN_X_AND_Y_SCALE != 0 {
            let t = [f2dot14(r.i16(at)?), 0, 0, f2dot14(r.i16(at + 2)?)];
            at += 4;
            Some(t)
        } else if flags & WE_HAVE_A_TWO_BY_TWO != 0 {
            // xx, yx, xy, yy in the file
            let (xx, yx, xy, yy) = (f2dot14(r.i16(at)?), f2dot14(r.i16(at + 2)?), f2dot14(r.i16(at + 4)?), f2dot14(r.i16(at + 6)?));
            at += 8;
            Some([xx, xy, yx, yy])
        } else {
            None
        };

        have_instructions |= flags & WE_HAVE_INSTRUCTIONS != 0;
        components.push(Component { flags, gid, arg1, arg2, transform });

        if flags & MORE_COMPONENTS == 0 {
            break;
        }
    }

    let instructions = if have_instructions {
        let len = r.u16(at)? as usize;
        r.slice(at + 2, len)?
    } else {
        &[]
    };

    Ok(RawGlyph::Composite { bbox, components, instructions })
}

/// `FT_Vector_Transform` of a 16.16 matrix (xx, xy, yx, yy).
pub(crate) fn transform(p: (i64, i64), m: [i64; 4]) -> (i64, i64) {
    (mul_fix(p.0, m[0]) + mul_fix(p.1, m[1]), mul_fix(p.0, m[2]) + mul_fix(p.1, m[3]))
}

/// `FT_Hypot` of two 16.16 values (`FT_Vector_Length`), for the scale of a
/// component's offset.
pub(crate) fn hypot(a: i64, b: i64) -> i64 {
    crate::trig::vector_length(a, b)
}

/// An unhinted outline in 26.6, before the move to the origin, with its
/// four phantom points last.
fn load_unhinted(face: &Face, size: &Size, gid: u16, depth: u32) -> Result<Outline, Error> {
    if depth > 16 {
        return Err(Error::Unsupported("composite glyphs nested too deep"));
    }

    let (xs, ys) = (size.metrics.x_scale, size.metrics.y_scale);
    let (advance, lsb) = face.h_metrics(gid);
    let raw = parse(face, gid)?;

    let x_min = match &raw {
        RawGlyph::Empty => 0,
        RawGlyph::Simple { bbox, .. } | RawGlyph::Composite { bbox, .. } => i64::from(bbox[0]),
    };

    // the phantom points in font units: origin and advance
    let pp1 = x_min - i64::from(lsb);
    let pp2 = pp1 + i64::from(advance);

    match raw {
        RawGlyph::Empty => Ok(Outline {
            points: vec![(mul_fix(pp1, xs), 0), (mul_fix(pp2, xs), 0), (0, 0), (0, 0)],
            tags: vec![0; 4],
            contours: Vec::new(),
        }),
        RawGlyph::Simple { points, flags, contours, .. } => {
            let mut outline = Outline { points: Vec::with_capacity(points.len() + 4), tags: Vec::new(), contours };

            for (p, f) in points.iter().zip(&flags) {
                outline.points.push((mul_fix(i64::from(p.0), xs), mul_fix(i64::from(p.1), ys)));
                outline.tags.push(f & TAG_ON);
            }

            outline.points.extend([(mul_fix(pp1, xs), 0), (mul_fix(pp2, xs), 0), (0, 0), (0, 0)]);
            outline.tags.extend([0; 4]);
            Ok(outline)
        }
        RawGlyph::Composite { components, .. } => {
            let mut outline = Outline::default();
            let mut phantom = [(mul_fix(pp1, xs), 0), (mul_fix(pp2, xs), 0), (0, 0), (0, 0)];

            for c in components {
                let mut sub = load_unhinted(face, size, c.gid, depth + 1)?;
                let n = sub.points.len() - 4;
                let sub_phantom: Vec<(i64, i64)> = sub.points.split_off(n);

                if c.flags & USE_MY_METRICS != 0 {
                    phantom.copy_from_slice(&sub_phantom);
                }

                if let Some(m) = c.transform {
                    for p in &mut sub.points {
                        *p = transform(*p, m);
                    }
                }

                let (dx, dy) = if c.flags & ARGS_ARE_XY_VALUES != 0 {
                    let (mut x, mut y) = (i64::from(c.arg1), i64::from(c.arg2));

                    if let Some(m) = c.transform
                        && c.flags & SCALED_COMPONENT_OFFSET != 0 && c.flags & UNSCALED_COMPONENT_OFFSET == 0 {
                            x = mul_fix(x, hypot(m[0], m[2]));
                            y = mul_fix(y, hypot(m[1], m[3]));
                        }

                    (mul_fix(x, xs), mul_fix(y, ys))
                } else {
                    // match a point of the glyph so far with one of the component
                    let (k, l) = (c.arg1 as usize, c.arg2 as usize);
                    let base = outline.points.get(k).ok_or(Error::Unsupported("a component anchor past the points"))?;
                    let own = sub.points.get(l).ok_or(Error::Unsupported("a component anchor past the points"))?;
                    (base.0 - own.0, base.1 - own.1)
                };

                let offset = outline.points.len() as u16;

                for p in &mut sub.points {
                    *p = (p.0 + dx, p.1 + dy);
                }

                outline.contours.extend(sub.contours.iter().map(|e| e + offset));
                outline.points.extend(sub.points);
                outline.tags.extend(sub.tags[..n].iter().copied());
            }

            outline.points.extend(phantom);
            outline.tags.extend([0; 4]);
            Ok(outline)
        }
    }
}

impl Glyph {
    /// `FT_Load_Glyph( FT_LOAD_NO_HINTING )`: the glyph scaled.
    pub fn load(face: &Face, size: &Size, gid: u16) -> Result<Glyph, Error> {
        let mut outline = load_unhinted(face, size, gid, 0)?;
        let n = outline.points.len() - 4;
        let pp1 = outline.points[n].0;
        let advance = outline.points[n + 1].0 - pp1;
        outline.points.truncate(n);
        outline.tags.truncate(n);

        if pp1 != 0 {
            for p in &mut outline.points {
                p.0 -= pp1;
            }
        }

        let left_bearing = outline.points.iter().map(|p| p.0).min().unwrap_or(0);
        Ok(Glyph { outline, advance, left_bearing })
    }
}

impl Glyph {
    /// `FT_Load_Glyph` with the default flags: the glyph hinted by its
    /// instructions at the size `hinter` was made for.
    pub fn load_hinted(face: &Face, size: &Size, hinter: &mut crate::hint::Hinter, gid: u16) -> Result<Glyph, Error> {
        crate::hint::load_hinted(face, size, hinter, gid)
    }
}

/// The hinted advance: the phantom points' distance on whole pixels.
pub(crate) fn hinted_advance(pp1: i64, pp2: i64) -> i64 {
    pix_round(pp2 - pp1)
}

/// `FT_Orientation`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation {
    /// Filled on the right of a contour, clockwise outlines.
    TrueType = 0,
    /// Filled on the left.
    PostScript = 1,
    None = 2,
}

impl Outline {
    /// `FT_Outline_Get_Orientation`: the sign of the area of all contours,
    /// on coordinates shifted to 14 bits.
    pub fn orientation(&self) -> Orientation {
        if self.points.is_empty() || self.contours.is_empty() {
            return Orientation::TrueType;
        }

        let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);

        for &(x, y) in &self.points {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }

        if x0 == x1 || y0 == y1 {
            return Orientation::None;
        }

        let msb = |v: u64| 63 - v.leading_zeros() as i64;
        let xshift = (msb(x1.unsigned_abs() | x0.unsigned_abs()) - 14).max(0);
        let yshift = (msb(y1.unsigned_abs() | y0.unsigned_abs()) - 14).max(0);
        let mut area: i64 = 0;
        let mut first = 0usize;

        for &last in &self.contours {
            let last = last as usize;
            let mut prev = (self.points[last].0 >> xshift, self.points[last].1 >> yshift);

            for &(x, y) in &self.points[first..=last] {
                let cur = (x >> xshift, y >> yshift);
                area = area.wrapping_add((cur.1 - prev.1).wrapping_mul(cur.0 + prev.0));
                prev = cur;
            }

            first = last + 1;
        }

        match area {
            a if a > 0 => Orientation::PostScript,
            a if a < 0 => Orientation::TrueType,
            _ => Orientation::None,
        }
    }
}

/// A step of an outline walked as `FT_Outline_Decompose` walks it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    MoveTo((i64, i64)),
    LineTo((i64, i64)),
    /// A quadratic curve: its control point and end.
    ConicTo((i64, i64), (i64, i64)),
}

impl Outline {
    /// `FT_Outline_Decompose` with no shift: each contour from its first
    /// on-curve point (the last point when the first is off it, else the
    /// middle of the two), off-curve points in a row split at their
    /// middles, closed with a line back to the start. `None` for an outline
    /// FreeType rejects.
    pub fn decompose(&self) -> Option<Vec<Segment>> {
        let mut out = Vec::new();
        let tags: Vec<u8> = self.tags.iter().map(|t| t & 3).collect();
        let mut first = 0usize;

        for &last in &self.contours {
            let last = usize::from(last);

            if last < first || last >= self.points.len() {
                return None;
            }

            let mut limit = last;
            let mut start = self.points[first];
            let v_last = self.points[last];
            let mut point = first;

            match tags[first] {
                2 => return None,
                0 => {
                    if tags[last] == 1 {
                        start = v_last;
                        limit -= 1;
                    } else {
                        start = ((start.0 + v_last.0) / 2, (start.1 + v_last.1) / 2);
                    }

                    // the walk begins one before the first point
                    point = point.wrapping_sub(1);
                }
                _ => {}
            }

            out.push(Segment::MoveTo(start));
            let mut closed = false;

            while point != limit {
                point = point.wrapping_add(1);

                match tags[point] {
                    1 => out.push(Segment::LineTo(self.points[point])),
                    0 => {
                        let mut control = self.points[point];

                        loop {
                            if point < limit {
                                point += 1;
                                let vec = self.points[point];

                                match tags[point] {
                                    1 => {
                                        out.push(Segment::ConicTo(control, vec));
                                        break;
                                    }
                                    0 => {
                                        let middle = ((control.0 + vec.0) / 2, (control.1 + vec.1) / 2);
                                        out.push(Segment::ConicTo(control, middle));
                                        control = vec;
                                    }
                                    _ => return None,
                                }
                            } else {
                                out.push(Segment::ConicTo(control, start));
                                closed = true;
                                break;
                            }
                        }

                        if closed {
                            break;
                        }
                    }
                    _ => return None,
                }
            }

            if !closed {
                out.push(Segment::LineTo(start));
            }

            first = last + 1;
        }

        Some(out)
    }
}

/// `FT_Vector_NormLen`: the vector made a unit one in 16.16 and its
/// length, by FreeType's integer Newton iterations.
fn vector_norm_len(v: &mut (i64, i64)) -> u32 {
    let (x_, y_) = (v.0 as i32, v.1 as i32);
    let (mut x, mut y) = (x_ as u32, y_ as u32);
    let (mut sx, mut sy) = (1i64, 1i64);

    if x_ < 0 {
        x = 0u32.wrapping_sub(x);
        sx = -1;
    }

    if y_ < 0 {
        y = 0u32.wrapping_sub(y);
        sy = -1;
    }

    if x == 0 {
        if y > 0 {
            v.1 = sy * 0x10000;
        }

        return y;
    } else if y == 0 {
        if x > 0 {
            v.0 = sx * 0x10000;
        }

        return x;
    }

    let mut l = if x > y { x.wrapping_add(y >> 1) } else { y.wrapping_add(x >> 1) };
    let mut shift = 31 - (31 - l.leading_zeros() as i32);
    shift -= 15 + i32::from(l >= (0xAAAA_AAAAu32 >> shift));

    if shift > 0 {
        x <<= shift;
        y <<= shift;
        l = if x > y { x.wrapping_add(y >> 1) } else { y.wrapping_add(x >> 1) };
    } else {
        x >>= -shift;
        y >>= -shift;
        l >>= -shift;
    }

    let mut b = 0x10000i32.wrapping_sub(l as i32);
    let (xi, yi) = (x as i32, y as i32);
    let (mut u, mut w);

    loop {
        u = xi.wrapping_add(xi.wrapping_mul(b) >> 16) as u32;
        w = yi.wrapping_add(yi.wrapping_mul(b) >> 16) as u32;
        let mut z = (u.wrapping_mul(u).wrapping_add(w.wrapping_mul(w)) as i32).wrapping_neg() / 0x200;
        z = z.wrapping_mul((0x10000i32.wrapping_add(b)) >> 8) / 0x10000;
        b = b.wrapping_add(z);

        if z <= 0 {
            break;
        }
    }

    v.0 = if sx < 0 { -i64::from(u) } else { i64::from(u) };
    v.1 = if sy < 0 { -i64::from(w) } else { i64::from(w) };

    let mut l = 0x10000u32.wrapping_add((u.wrapping_mul(x).wrapping_add(w.wrapping_mul(y)) as i32 / 0x10000) as u32);

    if shift > 0 {
        l = (l.wrapping_add(1 << (shift - 1))) >> shift;
    } else {
        l <<= -shift;
    }

    l
}

impl Outline {
    /// `FT_Outline_Embolden`: each contour pushed out by half the strength
    /// along the bisectors of its corners. `false` where FreeType refuses
    /// (contours with no orientation).
    pub fn embolden(&mut self, strength: i64) -> bool {
        use crate::fixed::{mul_div, mul_fix};

        let (xs, ys) = (strength / 2, strength / 2);

        if xs == 0 && ys == 0 {
            return true;
        }

        let orientation = self.orientation();

        if orientation == Orientation::None {
            return self.contours.is_empty();
        }

        let truetype = orientation == Orientation::TrueType;
        let mut last: i64 = -1;

        for c in 0..self.contours.len() {
            let first = last + 1;
            last = i64::from(self.contours[c]);
            let mut l_in: i64 = 0;
            let (mut inv, mut anchor) = ((0i64, 0i64), (0i64, 0i64));
            let mut l_anchor: i64 = 0;
            let (mut i, mut j, mut k) = (last, first, -1i64);
            let next = |n: i64| if n < last { n + 1 } else { first };

            while j != i && i != k {
                let mut out;
                let l_out;

                if j != k {
                    let (pj, pi) = (self.points[j as usize], self.points[i as usize]);
                    out = (pj.0 - pi.0, pj.1 - pi.1);
                    l_out = i64::from(vector_norm_len(&mut out));

                    if l_out == 0 {
                        j = next(j);
                        continue;
                    }
                } else {
                    out = anchor;
                    l_out = l_anchor;
                }

                if l_in != 0 {
                    if k < 0 {
                        k = i;
                        anchor = inv;
                        l_anchor = l_in;
                    }

                    let mut d = mul_fix(inv.0, out.0) + mul_fix(inv.1, out.1);
                    let mut shift;

                    if d > -0xF000 {
                        d += 0x10000;
                        shift = (inv.1 + out.1, inv.0 + out.0);

                        if truetype {
                            shift.0 = -shift.0;
                        } else {
                            shift.1 = -shift.1;
                        }

                        let mut q = mul_fix(out.0, inv.1) - mul_fix(out.1, inv.0);

                        if truetype {
                            q = -q;
                        }

                        let l = l_in.min(l_out);

                        shift.0 = if mul_fix(xs, q) <= mul_fix(l, d) { mul_div(shift.0, xs, d) } else { mul_div(shift.0, l, q) };
                        shift.1 = if mul_fix(ys, q) <= mul_fix(l, d) { mul_div(shift.1, ys, d) } else { mul_div(shift.1, l, q) };
                    } else {
                        shift = (0, 0);
                    }

                    while i != j {
                        let p = &mut self.points[i as usize];
                        p.0 += xs + shift.0;
                        p.1 += ys + shift.1;
                        i = next(i);
                    }
                } else {
                    i = j;
                }

                inv = out;
                l_in = l_out;
                j = next(j);
            }
        }

        true
    }
}

impl Outline {
    /// `FT_Outline_Transform` by a 16.16 matrix `[xx, xy, yx, yy]`, as
    /// `FT_Load_Glyph` applies the face's `FT_Set_Transform`.
    pub fn apply_matrix(&mut self, m: [i64; 4]) {
        for p in &mut self.points {
            *p = transform(*p, m);
        }
    }
}
