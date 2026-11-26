//! CFF outlines (`CFF `, Type 2 charstrings): a glyph's contours in font
//! units, its points on the curve or cubic controls, as FreeType's CFF
//! driver builds them unhinted; and its box, as HarfBuzz takes it.

use crate::sfnt::Reader;

/// A point in font units, 16.16.
pub(crate) type Point = (i64, i64);

/// A glyph's contours: points with their tags (1 on the curve, 2 a cubic
/// control, as FreeType tags them) and each contour's last point.
#[derive(Clone, Debug, Default)]
pub(crate) struct Path {
    pub points: Vec<Point>,
    pub tags: Vec<u8>,
    pub contours: Vec<u16>,
}

/// An INDEX: where its items start, and how many there are.
#[derive(Clone, Copy)]
struct Index {
    count: usize,
    off_size: usize,
    offsets: usize,
    data: usize,
}

impl Index {
    fn read(r: Reader, at: usize) -> Option<(Index, usize)> {
        let count = r.u16(at).ok()? as usize;
        if count == 0 {
            return Some((Index { count: 0, off_size: 1, offsets: at + 2, data: at + 2 }, at + 2));
        }
        let off_size = r.u8(at + 2).ok()? as usize;
        if !(1..=4).contains(&off_size) {
            return None;
        }
        let offsets = at + 3;
        let data = offsets + (count + 1) * off_size - 1;
        let ix = Index { count, off_size, offsets, data };
        let end = data + ix.offset(r, count)?;
        Some((ix, end))
    }

    fn offset(&self, r: Reader, i: usize) -> Option<usize> {
        let at = self.offsets + i * self.off_size;
        let mut v = 0usize;
        for k in 0..self.off_size {
            v = v << 8 | r.u8(at + k).ok()? as usize;
        }
        Some(v)
    }

    /// Item `i`: its start and end.
    fn get(&self, r: Reader, i: usize) -> Option<(usize, usize)> {
        if i >= self.count {
            return None;
        }
        Some((self.data + self.offset(r, i)?, self.data + self.offset(r, i + 1)?))
    }
}

/// A DICT's operators and their operands (as numbers).
fn dict(r: Reader, start: usize, end: usize) -> Vec<(u16, Vec<f64>)> {
    let mut out = Vec::new();
    let mut ops: Vec<f64> = Vec::new();
    let mut i = start;
    while i < end {
        let Ok(b) = r.u8(i) else { break };
        match b {
            0..=21 => {
                let op = if b == 12 {
                    i += 1;
                    1200 + u16::from(r.u8(i).unwrap_or(0))
                } else {
                    u16::from(b)
                };
                out.push((op, std::mem::take(&mut ops)));
                i += 1;
            }
            28 => {
                ops.push(f64::from(r.i16(i + 1).unwrap_or(0)));
                i += 3;
            }
            29 => {
                ops.push(f64::from(r.u32(i + 1).unwrap_or(0) as i32));
                i += 5;
            }
            30 => {
                // a real: nibbles
                let mut s = String::new();
                i += 1;
                'n: while i < end {
                    let byte = r.u8(i).unwrap_or(0xFF);
                    i += 1;
                    for nib in [byte >> 4, byte & 15] {
                        match nib {
                            0..=9 => s.push((b'0' + nib) as char),
                            10 => s.push('.'),
                            11 => s.push('E'),
                            12 => s.push_str("E-"),
                            14 => s.push('-'),
                            15 => break 'n,
                            _ => {}
                        }
                    }
                }
                ops.push(s.parse().unwrap_or(0.0));
            }
            32..=246 => {
                ops.push(f64::from(i32::from(b) - 139));
                i += 1;
            }
            247..=250 => {
                ops.push(f64::from((i32::from(b) - 247) * 256 + i32::from(r.u8(i + 1).unwrap_or(0)) + 108));
                i += 2;
            }
            251..=254 => {
                ops.push(f64::from(-(i32::from(b) - 251) * 256 - i32::from(r.u8(i + 1).unwrap_or(0)) - 108));
                i += 2;
            }
            _ => i += 1,
        }
    }
    out
}

fn op(d: &[(u16, Vec<f64>)], code: u16) -> Option<&[f64]> {
    d.iter().find(|(o, _)| *o == code).map(|(_, v)| v.as_slice())
}

/// The subroutine bias for a count of them.
fn bias(count: usize) -> i64 {
    if count < 1240 {
        107
    } else if count < 33900 {
        1131
    } else {
        32768
    }
}

/// A face's CFF table, read enough to load glyphs.
pub(crate) struct Cff<'a> {
    r: Reader<'a>,
    charstrings: Index,
    global: Index,
    /// Local subroutines: one set, or one a font dict (CID-keyed).
    locals: Vec<Option<Index>>,
    /// CID-keyed: the font dict of each glyph (FDSelect).
    fd_select: Option<usize>,
}

impl<'a> Cff<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Option<Cff<'a>> {
        let r = Reader::new(data);
        let hdr = r.u8(2).ok()? as usize;
        let (_, after_names) = Index::read(r, hdr)?;
        let (top, after_top) = Index::read(r, after_names)?;
        let (_, after_strings) = Index::read(r, after_top)?;
        let (global, _) = Index::read(r, after_strings)?;
        let (ts, te) = top.get(r, 0)?;
        let td = dict(r, ts, te);
        let charstrings = Index::read(r, *op(&td, 17)?.first()? as usize)?.0;
        if op(&td, 1206).and_then(|v| v.first()).is_some_and(|&t| t != 2.0) {
            return None;
        }
        let private = |size: f64, at: f64| -> Option<Index> {
            let (at, size) = (at as usize, size as usize);
            let pd = dict(r, at, at + size);
            let subrs = op(&pd, 19)?.first()?;
            Some(Index::read(r, at + *subrs as usize)?.0)
        };
        let (locals, fd_select) = match (op(&td, 1236), op(&td, 1237)) {
            (Some(fda), Some(fds)) => {
                let fd_array = Index::read(r, *fda.first()? as usize)?.0;
                let locals = (0..fd_array.count)
                    .map(|i| {
                        let (s, e) = fd_array.get(r, i)?;
                        let fd = dict(r, s, e);
                        let p = op(&fd, 18)?;
                        private(*p.first()?, *p.get(1)?)
                    })
                    .collect();
                (locals, Some(*fds.first()? as usize))
            }
            _ => (vec![op(&td, 18).and_then(|p| private(*p.first()?, *p.get(1)?))], None),
        };
        Some(Cff { r, charstrings, global, locals, fd_select })
    }

    /// The font dict glyph `gid` takes its subroutines from.
    fn fd(&self, gid: u16) -> usize {
        let Some(at) = self.fd_select else { return 0 };
        let r = self.r;
        match r.u8(at).unwrap_or(0) {
            0 => r.u8(at + 1 + gid as usize).unwrap_or(0) as usize,
            3 => {
                let n = r.u16(at + 1).unwrap_or(0) as usize;
                for k in 0..n {
                    let rec = at + 3 + 3 * k;
                    let (first, fd) = (r.u16(rec).unwrap_or(0), r.u8(rec + 2).unwrap_or(0));
                    let next = r.u16(rec + 3).unwrap_or(u16::MAX);
                    if gid >= first && gid < next {
                        return fd as usize;
                    }
                }
                0
            }
            _ => 0,
        }
    }

    /// Glyph `gid`'s outline, or `None` where its charstring cannot be read.
    pub(crate) fn path(&self, gid: u16) -> Option<Path> {
        let (s, e) = self.charstrings.get(self.r, gid as usize)?;
        let mut m = Machine { cff: self, local: self.locals.get(self.fd(gid)).copied().flatten(), stack: Vec::new(), stems: 0, width_done: false, x: 0.0, y: 0.0, path: Path::default(), open: false, transient: [0.0; 32], depth: 0, ended: false };
        m.run(s, e)?;
        m.close();
        Some(m.path)
    }
}

/// The charstring interpreter's state.
struct Machine<'c, 'a> {
    cff: &'c Cff<'a>,
    local: Option<Index>,
    stack: Vec<f64>,
    stems: usize,
    width_done: bool,
    x: f64,
    y: f64,
    path: Path,
    open: bool,
    transient: [f64; 32],
    depth: u32,
    ended: bool,
}

/// 16.16 font units as FreeType's CFF engine keeps them unhinted: whole
/// units (by a scale of 1/64, rounded, then down).
pub(crate) fn whole_unit(v: i64) -> i64 {
    crate::fixed::mul_fix(v, 0x400) >> 10
}

fn fixed(v: f64) -> i64 {
    (v * 65536.0).round() as i64
}

impl Machine<'_, '_> {
    /// The width, where the operator's arguments are one too many.
    fn width(&mut self, even: bool) {
        if !self.width_done {
            if (self.stack.len() % 2 == 1) == even && !self.stack.is_empty() {
                self.stack.remove(0);
            }
            self.width_done = true;
        }
    }

    fn close(&mut self) {
        if !self.open {
            return;
        }
        // FreeType's builder: a last point on the first is dropped
        let first = self.path.contours.last().map_or(0, |&c| c as usize + 1);
        let n = self.path.points.len();
        // (the same once in whole units, as FreeType compares them)
        let whole = |p: Point| (whole_unit(p.0), whole_unit(p.1));
        if n >= first + 2 && whole(self.path.points[n - 1]) == whole(self.path.points[first]) && self.path.tags[n - 1] == 1 {
            self.path.points.pop();
            self.path.tags.pop();
        }
        if self.path.points.len() > first {
            self.path.contours.push((self.path.points.len() - 1) as u16);
        }
        self.open = false;
    }

    /// A new contour begins (its first point once something is drawn from it).
    fn move_to(&mut self, dx: f64, dy: f64) {
        self.close();
        self.x += dx;
        self.y += dy;
    }

    fn begin(&mut self) {
        if !self.open {
            self.path.points.push((fixed(self.x), fixed(self.y)));
            self.path.tags.push(1);
            self.open = true;
        }
    }

    fn line_to(&mut self, dx: f64, dy: f64) {
        // a line going nowhere (in FreeType's CFF engine's device space, at
        // 1/64 unhinted): skipped, as there
        let ds = |v: f64| crate::fixed::mul_fix(fixed(v), 0x400);
        if ds(self.x + dx) == ds(self.x) && ds(self.y + dy) == ds(self.y) {
            self.x += dx;
            self.y += dy;
            return;
        }
        self.begin();
        self.x += dx;
        self.y += dy;
        self.path.points.push((fixed(self.x), fixed(self.y)));
        self.path.tags.push(1);
    }

    fn curve_to(&mut self, d: [f64; 6]) {
        self.begin();
        for (k, tag) in [(0, 2), (2, 2), (4, 1)] {
            self.x += d[k];
            self.y += d[k + 1];
            self.path.points.push((fixed(self.x), fixed(self.y)));
            self.path.tags.push(tag);
        }
    }

    fn run(&mut self, start: usize, end: usize) -> Option<()> {
        let r = self.cff.r;
        self.depth += 1;
        if self.depth > 10 {
            return None;
        }
        let mut i = start;
        while i < end && !self.ended {
            let b = r.u8(i).ok()?;
            i += 1;
            match b {
                28 => {
                    self.stack.push(f64::from(r.i16(i).ok()?));
                    i += 2;
                }
                32..=246 => self.stack.push(f64::from(i32::from(b) - 139)),
                247..=250 => {
                    self.stack.push(f64::from((i32::from(b) - 247) * 256 + i32::from(r.u8(i).ok()?) + 108));
                    i += 1;
                }
                251..=254 => {
                    self.stack.push(f64::from(-(i32::from(b) - 251) * 256 - i32::from(r.u8(i).ok()?) - 108));
                    i += 1;
                }
                255 => {
                    self.stack.push(f64::from(r.u32(i).ok()? as i32) / 65536.0);
                    i += 4;
                }
                // stems
                1 | 3 | 18 | 23 => {
                    self.width(true);
                    self.stems += self.stack.len() / 2;
                    self.stack.clear();
                }
                // hint and counter masks: their bytes passed over
                19 | 20 => {
                    self.width(true);
                    self.stems += self.stack.len() / 2;
                    self.stack.clear();
                    i += self.stems.div_ceil(8);
                }
                21 => {
                    self.width(true);
                    let (dx, dy) = (*self.stack.first()?, *self.stack.get(1)?);
                    self.move_to(dx, dy);
                    self.stack.clear();
                }
                22 | 4 => {
                    self.width(false);
                    let d = *self.stack.first()?;
                    if b == 22 { self.move_to(d, 0.0) } else { self.move_to(0.0, d) }
                    self.stack.clear();
                }
                5 => {
                    for p in self.stack.clone().as_chunks::<2>().0.iter() {
                        self.line_to(p[0], p[1]);
                    }
                    self.stack.clear();
                }
                6 | 7 => {
                    // alternating horizontal and vertical lines
                    let mut horizontal = b == 6;
                    for v in self.stack.clone() {
                        if horizontal { self.line_to(v, 0.0) } else { self.line_to(0.0, v) }
                        horizontal = !horizontal;
                    }
                    self.stack.clear();
                }
                8 => {
                    for c in self.stack.clone().as_chunks::<6>().0.iter() {
                        self.curve_to([c[0], c[1], c[2], c[3], c[4], c[5]]);
                    }
                    self.stack.clear();
                }
                24 => {
                    // curves, then a line
                    let s = self.stack.clone();
                    let n = s.len().saturating_sub(2) / 6;
                    for c in s.as_chunks::<6>().0.iter().take(n) {
                        self.curve_to([c[0], c[1], c[2], c[3], c[4], c[5]]);
                    }
                    if s.len() >= 2 {
                        self.line_to(s[s.len() - 2], s[s.len() - 1]);
                    }
                    self.stack.clear();
                }
                25 => {
                    // lines, then a curve
                    let s = self.stack.clone();
                    let lines = s.len().saturating_sub(6) / 2;
                    for p in s.as_chunks::<2>().0.iter().take(lines) {
                        self.line_to(p[0], p[1]);
                    }
                    if s.len() >= 6 {
                        let c = &s[s.len() - 6..];
                        self.curve_to([c[0], c[1], c[2], c[3], c[4], c[5]]);
                    }
                    self.stack.clear();
                }
                26 => {
                    // vvcurveto: dx1? {dya dxb dyb dyc}+
                    let mut s = self.stack.clone();
                    let mut dx1 = 0.0;
                    if s.len() % 4 == 1 {
                        dx1 = s.remove(0);
                    }
                    for c in s.as_chunks::<4>().0.iter() {
                        self.curve_to([dx1, c[0], c[1], c[2], 0.0, c[3]]);
                        dx1 = 0.0;
                    }
                    self.stack.clear();
                }
                27 => {
                    // hhcurveto: dy1? {dxa dxb dyb dxc}+
                    let mut s = self.stack.clone();
                    let mut dy1 = 0.0;
                    if s.len() % 4 == 1 {
                        dy1 = s.remove(0);
                    }
                    for c in s.as_chunks::<4>().0.iter() {
                        self.curve_to([c[0], dy1, c[1], c[2], c[3], 0.0]);
                        dy1 = 0.0;
                    }
                    self.stack.clear();
                }
                30 | 31 => {
                    // vhcurveto / hvcurveto: alternating, the last with an extra delta
                    let s = self.stack.clone();
                    let mut horizontal = b == 31;
                    let mut k = 0;
                    while k + 4 <= s.len() {
                        let last = s.len() - k == 5;
                        let extra = if last { s[k + 4] } else { 0.0 };
                        if horizontal {
                            self.curve_to([s[k], 0.0, s[k + 1], s[k + 2], extra, s[k + 3]]);
                        } else {
                            self.curve_to([0.0, s[k], s[k + 1], s[k + 2], s[k + 3], extra]);
                        }
                        horizontal = !horizontal;
                        k += 4;
                    }
                    self.stack.clear();
                }
                10 | 29 => {
                    let n = self.stack.pop()? as i64;
                    let subrs = if b == 10 { self.local? } else { self.cff.global };
                    let (s, e) = subrs.get(r, (n + bias(subrs.count)) as usize)?;
                    self.run(s, e)?;
                }
                11 => break,
                14 => {
                    // endchar (its accent form, of a standard encoding, not drawn)
                    self.width(true);
                    self.close();
                    self.ended = true;
                    self.stack.clear();
                }
                12 => {
                    let b2 = r.u8(i).ok()?;
                    i += 1;
                    self.escape(b2)?;
                }
                _ => self.stack.clear(),
            }
        }
        self.depth -= 1;
        Some(())
    }

    /// The two-byte operators: flex and arithmetic.
    fn escape(&mut self, b: u8) -> Option<()> {
        let s = self.stack.clone();
        let pop = |m: &mut Self| m.stack.pop().unwrap_or(0.0);
        match b {
            35 => {
                // flex: two curves, the depth ignored
                if s.len() >= 12 {
                    self.curve_to([s[0], s[1], s[2], s[3], s[4], s[5]]);
                    self.curve_to([s[6], s[7], s[8], s[9], s[10], s[11]]);
                }
                self.stack.clear();
            }
            34 => {
                // hflex: dx1 dx2 dy2 dx3 dx4 dx5 dx6
                if s.len() >= 7 {
                    self.curve_to([s[0], 0.0, s[1], s[2], s[3], 0.0]);
                    self.curve_to([s[4], 0.0, s[5], -s[2], s[6], 0.0]);
                }
                self.stack.clear();
            }
            36 => {
                // hflex1: dx1 dy1 dx2 dy2 dx3 dx4 dx5 dy5 dx6
                if s.len() >= 9 {
                    self.curve_to([s[0], s[1], s[2], s[3], s[4], 0.0]);
                    let dy6 = -(s[1] + s[3] + s[7]);
                    self.curve_to([s[5], 0.0, s[6], s[7], s[8], dy6]);
                }
                self.stack.clear();
            }
            37 => {
                // flex1: the last point's other coordinate from the start
                if s.len() >= 11 {
                    let dx: f64 = s[0] + s[2] + s[4] + s[6] + s[8];
                    let dy: f64 = s[1] + s[3] + s[5] + s[7] + s[9];
                    let (d6x, d6y) = if dx.abs() > dy.abs() { (s[10], -dy) } else { (-dx, s[10]) };
                    self.curve_to([s[0], s[1], s[2], s[3], s[4], s[5]]);
                    self.curve_to([s[6], s[7], s[8], s[9], d6x, d6y]);
                }
                self.stack.clear();
            }
            3 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(f64::from(u8::from(a != 0.0 && b2 != 0.0)));
            }
            4 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(f64::from(u8::from(a != 0.0 || b2 != 0.0)));
            }
            5 => {
                let a = pop(self);
                self.stack.push(f64::from(u8::from(a == 0.0)));
            }
            9 => {
                let a = pop(self);
                self.stack.push(a.abs());
            }
            10 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(a + b2);
            }
            11 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(a - b2);
            }
            12 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(if b2 == 0.0 { 0.0 } else { a / b2 });
            }
            14 => {
                let a = pop(self);
                self.stack.push(-a);
            }
            15 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(f64::from(u8::from(a == b2)));
            }
            18 => {
                pop(self);
            }
            20 => {
                let (i, v) = (pop(self) as usize, pop(self));
                if let Some(t) = self.transient.get_mut(i) {
                    *t = v;
                }
            }
            21 => {
                let i = pop(self) as usize;
                let v = self.transient.get(i).copied().unwrap_or(0.0);
                self.stack.push(v);
            }
            22 => {
                let (v2, v1, b2, a) = (pop(self), pop(self), pop(self), pop(self));
                self.stack.push(if v1 <= v2 { a } else { b2 });
            }
            23 => self.stack.push(0.5),
            24 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(a * b2);
            }
            26 => {
                let a = pop(self);
                self.stack.push(a.max(0.0).sqrt());
            }
            27 => {
                let a = *self.stack.last()?;
                self.stack.push(a);
            }
            28 => {
                let (b2, a) = (pop(self), pop(self));
                self.stack.push(b2);
                self.stack.push(a);
            }
            29 => {
                let i = pop(self);
                let n = self.stack.len();
                let v = if i < 0.0 { self.stack.last().copied() } else { n.checked_sub(1 + i as usize).and_then(|k| self.stack.get(k).copied()) };
                self.stack.push(v.unwrap_or(0.0));
            }
            30 => {
                let (j, n) = (pop(self) as i64, pop(self) as usize);
                let len = self.stack.len();
                if n > 0 && n <= len {
                    let part = &mut self.stack[len - n..];
                    let j = j.rem_euclid(n as i64) as usize;
                    part.rotate_right(j);
                }
            }
            _ => self.stack.clear(),
        }
        Some(())
    }
}

/// A glyph's box in font units as HarfBuzz takes it from its outline (its
/// curves' control points included): x min, y min, x max, y max; `None`
/// for a glyph with no outline.
pub(crate) fn bounds(p: &Path) -> Option<(f64, f64, f64, f64)> {
    if p.points.is_empty() {
        return None;
    }
    let f = |v: i64| v as f64 / 65536.0;
    let mut b = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for &(x, y) in &p.points {
        b = (b.0.min(f(x)), b.1.min(f(y)), b.2.max(f(x)), b.3.max(f(y)));
    }
    Some(b)
}
