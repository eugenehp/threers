//! The GDI device context both formats drive: the coordinate mapping, the
//! selected pen, brush and font, the current point and path; drawing calls
//! become [`Op`]s in points.

use std::collections::HashMap;

use crate::{Op, Rgba, Seg};

#[derive(Clone, Debug)]
pub enum Obj {
    Pen { color: Rgba, width: f32, null: bool },
    Brush { color: Option<Rgba> },
    Font { height: f32, width: f32, weight: u32, italic: bool, family: String, angle: f32 },
}

#[derive(Clone, Debug)]
pub struct Dc {
    pub win_org: (f32, f32),
    pub win_ext: (f32, f32),
    pub vp_org: (f32, f32),
    pub vp_ext: (f32, f32),
    pub map_mode: u32,
    /// World transform `[m11, m12, m21, m22, dx, dy]`.
    pub xform: [f32; 6],
    pub pen: Obj,
    pub brush: Obj,
    pub font: Obj,
    pub text_color: Rgba,
    pub text_align: u32,
    pub even_odd: bool,
    pub cur: (f32, f32),
}

impl Default for Dc {
    fn default() -> Self {
        Dc {
            win_org: (0.0, 0.0),
            win_ext: (1.0, 1.0),
            vp_org: (0.0, 0.0),
            vp_ext: (1.0, 1.0),
            map_mode: 1,
            xform: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            pen: Obj::Pen { color: [0, 0, 0, 255], width: 0.0, null: false },
            brush: Obj::Brush { color: Some([255, 255, 255, 255]) },
            font: Obj::Font { height: 12.0, width: 0.0, weight: 400, italic: false, family: "Arial".into(), angle: 0.0 },
            text_color: [0, 0, 0, 255],
            text_align: 0,
            even_odd: true,
            cur: (0.0, 0.0),
        }
    }
}

/// Where device units go: points, from the picture's top left.
pub struct Output {
    /// Points per device unit, each axis.
    pub scale: (f32, f32),
    /// The device point at the picture's top left.
    pub origin: (f32, f32),
    /// Device units a millimetre (for the metric map modes).
    pub dev_per_mm: (f32, f32),
}

pub struct Gdi {
    pub dc: Dc,
    pub saved: Vec<Dc>,
    pub objects: HashMap<u32, Obj>,
    pub path: Option<Vec<Seg>>,
    pub ops: Vec<Op>,
    pub out: Output,
}

/// The stock objects (`GetStockObject`), by index.
pub fn stock(i: u32) -> Option<Obj> {
    Some(match i {
        0 => Obj::Brush { color: Some([255, 255, 255, 255]) },
        1 => Obj::Brush { color: Some([192, 192, 192, 255]) },
        2 => Obj::Brush { color: Some([128, 128, 128, 255]) },
        3 => Obj::Brush { color: Some([64, 64, 64, 255]) },
        4 => Obj::Brush { color: Some([0, 0, 0, 255]) },
        5 => Obj::Brush { color: None },
        6 => Obj::Pen { color: [255, 255, 255, 255], width: 0.0, null: false },
        7 => Obj::Pen { color: [0, 0, 0, 255], width: 0.0, null: false },
        8 => Obj::Pen { color: [0, 0, 0, 255], width: 0.0, null: true },
        10..=17 => Obj::Font { height: 12.0, width: 0.0, weight: 400, italic: false, family: "Arial".into(), angle: 0.0 },
        _ => return None,
    })
}

impl Gdi {
    pub fn new(out: Output) -> Gdi {
        Gdi { dc: Dc::default(), saved: Vec::new(), objects: HashMap::new(), path: None, ops: Vec::new(), out }
    }

    /// Logical units through the world transform and the window and
    /// viewport into device units.
    fn device(&self, x: f32, y: f32) -> (f32, f32) {
        let m = &self.dc.xform;
        let (x, y) = (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]);
        let d = &self.dc;
        match d.map_mode {
            // anisotropic, isotropic
            7 | 8 => {
                let sx = if d.win_ext.0 != 0.0 { d.vp_ext.0 / d.win_ext.0 } else { 1.0 };
                let sy = if d.win_ext.1 != 0.0 { d.vp_ext.1 / d.win_ext.1 } else { 1.0 };
                let (sx, sy) = if d.map_mode == 8 {
                    let s = sx.abs().min(sy.abs());
                    (s * sx.signum(), s * sy.signum())
                } else {
                    (sx, sy)
                };
                ((x - d.win_org.0) * sx + d.vp_org.0, (y - d.win_org.1) * sy + d.vp_org.1)
            }
            // the metric and imperial modes: y up
            2..=6 => {
                let mm = match d.map_mode {
                    2 => 0.1,
                    3 => 0.01,
                    4 => 0.254,
                    5 => 0.0254,
                    _ => 25.4 / 1440.0,
                };
                (
                    (x - d.win_org.0) * mm * self.out.dev_per_mm.0 + d.vp_org.0,
                    -(y - d.win_org.1) * mm * self.out.dev_per_mm.1 + d.vp_org.1,
                )
            }
            _ => (x - d.win_org.0 + d.vp_org.0, y - d.win_org.1 + d.vp_org.1),
        }
    }

    /// A logical point in points.
    pub fn pt(&self, x: f32, y: f32) -> (f32, f32) {
        let (dx, dy) = self.device(x, y);
        ((dx - self.out.origin.0) * self.out.scale.0, (dy - self.out.origin.1) * self.out.scale.1)
    }

    /// A logical length (along x) in points.
    pub fn len(&self, l: f32) -> f32 {
        let (a, _) = self.pt(0.0, 0.0);
        let (b, _) = self.pt(l, 0.0);
        let (_, c) = self.pt(0.0, l);
        let (_, d) = self.pt(0.0, 0.0);
        ((b - a).abs() + (c - d).abs()) / 2.0
    }

    fn pen(&self) -> Option<(Rgba, f32)> {
        match &self.dc.pen {
            Obj::Pen { null: false, color, width } => Some((*color, self.len(*width).max(0.35))),
            _ => None,
        }
    }

    fn brush(&self) -> Option<Rgba> {
        match &self.dc.brush {
            Obj::Brush { color } => *color,
            _ => None,
        }
    }

    pub fn select(&mut self, index: u32) {
        let obj = if index & 0x8000_0000 != 0 { stock(index & 0x7FFF_FFFF) } else { self.objects.get(&index).cloned() };
        match obj {
            Some(o @ Obj::Pen { .. }) => self.dc.pen = o,
            Some(o @ Obj::Brush { .. }) => self.dc.brush = o,
            Some(o @ Obj::Font { .. }) => self.dc.font = o,
            None => {}
        }
    }

    pub fn save(&mut self) {
        self.saved.push(self.dc.clone());
    }

    pub fn restore(&mut self, rel: i32) {
        let n = if rel < 0 { (-rel) as usize } else { self.saved.len().saturating_sub(rel as usize).max(1) };
        for _ in 0..n.min(self.saved.len()) {
            if let Some(d) = self.saved.pop() {
                self.dc = d;
            }
        }
    }

    fn emit(&mut self, segs: Vec<Seg>, fill: bool, stroke: bool) {
        if segs.is_empty() {
            return;
        }
        if let Some(p) = &mut self.path {
            p.extend(segs);
            return;
        }
        let fill = if fill { self.brush() } else { None };
        let stroke = if stroke { self.pen() } else { None };
        if fill.is_some() || stroke.is_some() {
            self.ops.push(Op::Path { segs, fill, stroke, even_odd: self.dc.even_odd });
        }
    }

    fn mapped(&self, pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
        pts.iter().map(|&(x, y)| self.pt(x, y)).collect()
    }

    pub fn move_to(&mut self, x: f32, y: f32) {
        self.dc.cur = (x, y);
        if self.path.is_some() {
            let q = self.pt(x, y);
            if let Some(p) = &mut self.path {
                p.push(Seg::Move(q.0, q.1));
            }
        }
    }

    pub fn line_to(&mut self, x: f32, y: f32) {
        let a = self.pt(self.dc.cur.0, self.dc.cur.1);
        let b = self.pt(x, y);
        self.dc.cur = (x, y);
        if let Some(p) = &mut self.path {
            if p.is_empty() {
                p.push(Seg::Move(a.0, a.1));
            }
            p.push(Seg::Line(b.0, b.1));
            return;
        }
        self.emit(vec![Seg::Move(a.0, a.1), Seg::Line(b.0, b.1)], false, true);
    }

    pub fn polyline(&mut self, pts: &[(f32, f32)], to: bool) {
        if pts.is_empty() {
            return;
        }
        let mut segs = Vec::with_capacity(pts.len() + 1);
        let mapped = self.mapped(pts);
        if to {
            let c = self.pt(self.dc.cur.0, self.dc.cur.1);
            if self.path.as_ref().is_none_or(|p| p.is_empty()) {
                segs.push(Seg::Move(c.0, c.1));
            }
            segs.extend(mapped.iter().map(|&(x, y)| Seg::Line(x, y)));
        } else {
            segs.push(Seg::Move(mapped[0].0, mapped[0].1));
            segs.extend(mapped[1..].iter().map(|&(x, y)| Seg::Line(x, y)));
        }
        self.dc.cur = *pts.last().unwrap();
        self.emit(segs, false, true);
    }

    pub fn polygons(&mut self, polys: &[Vec<(f32, f32)>]) {
        let mut segs = Vec::new();
        for poly in polys {
            let m = self.mapped(poly);
            if let Some(&(x, y)) = m.first() {
                segs.push(Seg::Move(x, y));
                segs.extend(m[1..].iter().map(|&(x, y)| Seg::Line(x, y)));
                segs.push(Seg::Close);
            }
        }
        self.emit(segs, true, true);
    }

    /// Béziers: `pts` in threes after the start (`to`: from the current point).
    pub fn beziers(&mut self, pts: &[(f32, f32)], to: bool) {
        let m = self.mapped(pts);
        let (start, rest) = if to {
            (self.pt(self.dc.cur.0, self.dc.cur.1), &m[..])
        } else {
            match m.split_first() {
                Some((s, r)) => (*s, r),
                None => return,
            }
        };
        let mut segs = Vec::new();
        if !to || self.path.as_ref().is_none_or(|p| p.is_empty()) {
            segs.push(Seg::Move(start.0, start.1));
        }
        for c in rest.chunks_exact(3) {
            segs.push(Seg::Cubic(c[0].0, c[0].1, c[1].0, c[1].1, c[2].0, c[2].1));
        }
        if let Some(&p) = pts.last() {
            self.dc.cur = p;
        }
        self.emit(segs, false, true);
    }

    pub fn rect(&mut self, l: f32, t: f32, r: f32, b: f32, rounded: Option<(f32, f32)>) {
        let (a, c) = (self.pt(l, t), self.pt(r, b));
        let (rx, ry) = rounded.map_or((0.0, 0.0), |(w, h)| (self.len(w) / 2.0, self.len(h) / 2.0));
        self.emit(crate::round_rect(a.0, a.1, c.0, c.1, rx, ry), true, true);
    }

    pub fn ellipse(&mut self, l: f32, t: f32, r: f32, b: f32) {
        let (a, c) = (self.pt(l, t), self.pt(r, b));
        self.emit(crate::ellipse(a.0, a.1, c.0, c.1), true, true);
    }

    /// Arc (0), chord (1) or pie (2).
    pub fn arc(&mut self, kind: u8, bx: [f32; 4], start: (f32, f32), end: (f32, f32)) {
        let (a, c) = (self.pt(bx[0], bx[1]), self.pt(bx[2], bx[3]));
        let (s, e) = (self.pt(start.0, start.1), self.pt(end.0, end.1));
        let pts = crate::arc_points(a.0, a.1, c.0, c.1, s, e);
        let mut segs = vec![Seg::Move(pts[0].0, pts[0].1)];
        segs.extend(pts[1..].iter().map(|&(x, y)| Seg::Line(x, y)));
        if kind == 2 {
            segs.push(Seg::Line((a.0 + c.0) / 2.0, (a.1 + c.1) / 2.0));
        }
        if kind > 0 {
            segs.push(Seg::Close);
        }
        self.emit(segs, kind > 0, true);
    }

    pub fn begin_path(&mut self) {
        self.path = Some(Vec::new());
    }

    pub fn close_figure(&mut self) {
        if let Some(p) = &mut self.path {
            p.push(Seg::Close);
        }
    }

    /// The path made: filled, stroked or both.
    pub fn paint_path(&mut self, fill: bool, stroke: bool) {
        if let Some(segs) = self.path.take() {
            let fill = if fill { self.brush() } else { None };
            let stroke = if stroke { self.pen() } else { None };
            if !segs.is_empty() && (fill.is_some() || stroke.is_some()) {
                self.ops.push(Op::Path { segs, fill, stroke, even_odd: self.dc.even_odd });
            }
        }
    }

    pub fn end_path(&mut self) {
        // kept for FILLPATH and friends
    }

    /// Single-byte text in the font there is now.
    pub fn bytes(&self, b: &[u8]) -> String {
        ansi(b, matches!(&self.dc.font, Obj::Font { family, .. } if crate::symbol::is_symbol(family)))
    }

    /// Text at a logical point, `advance` its logical width when known.
    pub fn text(&mut self, x: f32, y: f32, text: &str, advance: Option<f32>) {
        let text = text.trim_end_matches('\0');
        if text.trim().is_empty() {
            return;
        }
        let Obj::Font { height, width: wide, weight, italic, family, angle } = self.dc.font.clone() else { return };
        // a negative height is the characters'; a positive one the cell's
        let size = if height < 0.0 { self.len(-height) } else { self.len(height) * 0.8 }.max(1.0);
        // a width asked for (a tall bracket kept as narrow as a short one):
        // how it compares with the characters' own, on average
        let stretch = if wide.abs() > 0.0 { (self.len(wide.abs()) / (size * average_width(&family))).clamp(0.1, 10.0) } else { 1.0 };
        let (mut px, mut py) = self.pt(x, y);
        let align = self.dc.text_align;
        let width = advance.map(|a| self.len(a)).unwrap_or(size * 0.5 * text.chars().count() as f32);
        match align & 6 {
            6 => px -= width / 2.0,
            2 => px -= width,
            _ => {}
        }
        match align & 24 {
            24 => {}
            8 => py -= size * 0.22,
            _ => py += size * 0.8,
        }
        // the Symbol font's own encoding (an equation's Greek, operators,
        // bracket pieces) as the characters it draws
        let text = if crate::symbol::is_symbol(&family) { crate::symbol::to_unicode(text) } else { text.to_string() };
        self.ops.push(Op::Text {
            x: px,
            y: py,
            text,
            size,
            color: self.dc.text_color,
            bold: weight >= 600,
            italic,
            family,
            angle,
            stretch,
        });
    }

    /// A bitmap stretched into a logical rectangle.
    pub fn image(&mut self, x: f32, y: f32, w: f32, h: f32, bmp: (u32, u32, Vec<u8>)) {
        let (a, b) = (self.pt(x, y), self.pt(x + w, y + h));
        self.ops.push(Op::Image {
            x: a.0.min(b.0),
            y: a.1.min(b.1),
            w: (b.0 - a.0).abs(),
            h: (b.1 - a.1).abs(),
            width: bmp.0,
            height: bmp.1,
            rgba: bmp.2,
        });
    }
}

/// A family's average character width, in ems (what a logical font's width
/// is measured against).
fn average_width(family: &str) -> f32 {
    let f = family.to_lowercase();
    if crate::symbol::is_symbol(&f) {
        0.591
    } else if f.contains("times") {
        0.401
    } else if f.contains("courier") {
        0.6
    } else {
        0.441
    }
}

/// Single-byte text, in Windows' code page 1252 (Symbol's own bytes are
/// left as they are: see `symbol`).
pub fn ansi(bytes: &[u8], symbol: bool) -> String {
    const HIGH: [char; 32] = [
        '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8D}', 'Ž', '\u{8F}', '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™',
        'š', '›', 'œ', '\u{9D}', 'ž', 'Ÿ',
    ];
    bytes.iter().map(|&b| if !symbol && (0x80..0xA0).contains(&b) { HIGH[b as usize - 0x80] } else { b as char }).collect()
}

#[cfg(test)]
mod text_tests {
    use super::ansi;

    #[test]
    fn a_tall_bracket_keeps_the_width_its_font_asks_for() {
        // an equation's: 434 high, as wide as 320 high ones (189)
        let mut g = super::Gdi::new(super::Output { scale: (1.0, 1.0), origin: (0.0, 0.0), dev_per_mm: (1.0, 1.0) });
        g.dc.font = super::Obj::Font { height: -434.0, width: 189.0, weight: 400, italic: false, family: "Symbol".into(), angle: 0.0 };
        g.text(0.0, 0.0, "(", None);
        let crate::Op::Text { stretch, .. } = &g.ops[0] else { panic!() };
        assert!((stretch - 320.0 / 434.0).abs() < 0.01, "{stretch}");
        // none asked for: as the font has it
        g.dc.font = super::Obj::Font { height: -320.0, width: 0.0, weight: 400, italic: false, family: "Symbol".into(), angle: 0.0 };
        g.text(0.0, 0.0, "(", None);
        assert!(matches!(&g.ops[1], crate::Op::Text { stretch, .. } if *stretch == 1.0));
    }

    #[test]
    fn single_byte_text_is_code_page_1252() {
        assert_eq!(ansi(b"J\x88\xD7", false), "Jˆ×");
        assert_eq!(ansi(b"\x88", true), "\u{88}");
    }
}
