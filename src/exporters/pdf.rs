//! PDF drawings: an [`SvgDocument`] written as a PDF page ([`write_pdf`]),
//! or several as one file ([`write_pdf_pages`]), vector all the way down.
//!
//! The drawing is read as `drawing.rs` reads it for every format, so what the
//! SVG shows the PDF shows: the view box fitted `xMidYMid meet` to a page the
//! size of the document's, paths as moves, lines and cubics (arcs converted),
//! the same fill rules, opacities, caps, joins, miter limit and dashes. PDF
//! has all of those natively, so nothing is flattened: a circle stays four
//! Béziers and a stroke is stroked by the viewer.
//!
//! Text uses the standard fonts every viewer has, so none is embedded:
//! Helvetica, or Courier for a family naming `mono` or `courier`, or Times for
//! one naming `serif` or `times` but not `sans`. It is encoded as
//! WinAnsi (Latin-1 and the usual typographic extras); other characters print
//! as `?`. The anchor is placed with the font's real advance widths, from
//! tables generated from the URW AFM files (`scripts/afm_widths.py`), so a
//! centred label is centred.
//!
//! The output is deterministic, with no creation date and no random ID, so
//! the same drawing is the same bytes. Content streams are deflated unless
//! [`PdfOptions::compress`] is off.
//!
//! ```
//! use threers::exporters::{write_pdf, SvgDocument, SvgStyle, SvgUnit};
//!
//! let mut doc = SvgDocument::new(100.0, 50.0, SvgUnit::Millimeters);
//! doc.circle([25.0, 25.0], 8.0, SvgStyle::fill([200, 52, 52]));
//! let pdf = write_pdf(&doc);
//! assert!(pdf.starts_with(b"%PDF-1.4"));
//! assert!(pdf.ends_with(b"%%EOF\n"));
//! ```

use std::collections::BTreeMap;
use std::fmt::Write;

use super::drawing::{self, Affine, Item, Resolved, Seg};
use super::pdf_fonts;
use super::svg::{SvgDocument, SvgFillRule, SvgLineCap, SvgLineJoin, SvgTextAnchor};

/// How [`write_pdf_pages_with`] writes.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfOptions {
    /// The document title; the first page's [`SvgDocument::title`] when `None`.
    pub title: Option<String>,
    /// Deflate the content streams.
    pub compress: bool,
}

impl Default for PdfOptions {
    fn default() -> Self {
        PdfOptions { title: None, compress: true }
    }
}

/// One drawing as a one-page PDF.
pub fn write_pdf(doc: &SvgDocument) -> Vec<u8> {
    write_pdf_pages_with(std::slice::from_ref(doc), &PdfOptions::default())
}

/// Each drawing a page, in order, each page the size of its drawing.
pub fn write_pdf_pages(pages: &[SvgDocument]) -> Vec<u8> {
    write_pdf_pages_with(pages, &PdfOptions::default())
}

pub fn write_pdf_pages_with(pages: &[SvgDocument], options: &PdfOptions) -> Vec<u8> {
    let mut res = Resources::default();
    let contents: Vec<(String, [f64; 2])> = pages.iter().map(|d| page_content(d, &mut res)).collect();

    let mut w = ObjWriter::new();
    // 1: catalog, 2: page tree, 3: resources, 4: info, then each page and its content
    let n_fixed = 4;
    let page_id = |k: usize| n_fixed + 1 + 2 * k;
    let kids: Vec<String> = (0..pages.len()).map(|k| format!("{} 0 R", page_id(k))).collect();
    w.object(1, "<< /Type /Catalog /Pages 2 0 R >>");
    w.object(2, &format!("<< /Type /Pages /Kids [{}] /Count {} >>", kids.join(" "), pages.len()));
    w.object(3, &res.dict());
    let title = options.title.clone().or_else(|| pages.first().and_then(|p| p.title.clone()));
    let mut info = String::from("<< /Producer (threers)");
    if let Some(t) = &title {
        let _ = write!(info, " /Title {}", text_string(t));
    }
    info.push_str(" >>");
    w.object(4, &info);
    for (k, (content, size)) in contents.iter().enumerate() {
        let id = page_id(k);
        w.object(
            id,
            &format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {} {}] /Resources 3 0 R /Contents {} 0 R >>",
                num(size[0]),
                num(size[1]),
                id + 1
            ),
        );
        w.stream(id + 1, content.as_bytes(), options.compress);
    }
    w.finish(n_fixed + 2 * pages.len(), 1, 4)
}

/// The fonts and graphics states the pages use, shared by all of them.
#[derive(Default)]
struct Resources {
    fonts: BTreeMap<&'static str, usize>,
    /// (fill alpha, stroke alpha) in thousandths, by name index.
    states: BTreeMap<(u16, u16), usize>,
}

impl Resources {
    fn font(&mut self, f: Font) -> String {
        let n = self.fonts.len();
        format!("/F{}", self.fonts.entry(f.base_name()).or_insert(n))
    }

    fn state(&mut self, fill: f64, stroke: f64) -> String {
        let key = ((fill * 1000.0).round() as u16, (stroke * 1000.0).round() as u16);
        let n = self.states.len();
        format!("/GS{}", self.states.entry(key).or_insert(n))
    }

    fn dict(&self) -> String {
        let mut d = String::from("<<");
        if !self.fonts.is_empty() {
            d.push_str(" /Font <<");
            for (name, k) in &self.fonts {
                let _ = write!(d, " /F{k} << /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>");
            }
            d.push_str(" >>");
        }
        if !self.states.is_empty() {
            d.push_str(" /ExtGState <<");
            for ((ca, sa), k) in &self.states {
                let _ = write!(d, " /GS{k} << /Type /ExtGState /ca {} /CA {} >>", num(*ca as f64 / 1000.0), num(*sa as f64 / 1000.0));
            }
            d.push_str(" >>");
        }
        d.push_str(" >>");
        d
    }
}

/// A page's content stream and its size in points.
fn page_content(doc: &SvgDocument, res: &mut Resources) -> (String, [f64; 2]) {
    let inches = drawing::page_inches(doc);
    let size = [inches[0] * 72.0, inches[1] * 72.0];
    let mut out = String::new();
    let Some(view) = drawing::viewport(doc, size) else {
        return (out, size);
    };
    // PDF's y points up from the bottom of the page.
    let base = drawing::compose(&[1.0, 0.0, 0.0, -1.0, 0.0, size[1]], &view);
    let _ = writeln!(out, "{} cm", matrix(&base));
    drawing::walk(doc, &mut |item| match item {
        Item::Shape { ctm, style, segs } => shape(&mut out, res, &ctm, &style, &segs),
        Item::Text { ctm, style, pos, text, size, anchor, rotation, font_family } => {
            text_run(&mut out, res, &ctm, &style, pos, text, size, anchor, rotation, font_family)
        }
    });
    (out, size)
}

fn shape(out: &mut String, res: &mut Resources, ctm: &Affine, style: &Resolved, segs: &[Seg]) {
    let (fill, stroke) = (style.fills(), style.strokes());
    if !fill && !stroke {
        return;
    }
    out.push_str("q\n");
    if *ctm != drawing::IDENTITY {
        let _ = writeln!(out, "{} cm", matrix(ctm));
    }
    paint_state(out, res, style, fill, stroke);
    for s in segs {
        let _ = match s {
            Seg::Move(p) => writeln!(out, "{} {} m", num(p[0]), num(p[1])),
            Seg::Line(p) => writeln!(out, "{} {} l", num(p[0]), num(p[1])),
            Seg::Cubic(a, b, p) => writeln!(out, "{} {} {} {} {} {} c", num(a[0]), num(a[1]), num(b[0]), num(b[1]), num(p[0]), num(p[1])),
            Seg::Close => writeln!(out, "h"),
        };
    }
    let even_odd = style.fill_rule == SvgFillRule::EvenOdd;
    out.push_str(match (fill, stroke, even_odd) {
        (true, true, false) => "B\n",
        (true, true, true) => "B*\n",
        (true, false, false) => "f\n",
        (true, false, true) => "f*\n",
        _ => "S\n",
    });
    out.push_str("Q\n");
}

/// Colours, opacities and the stroke's parameters, for what is painted.
fn paint_state(out: &mut String, res: &mut Resources, style: &Resolved, fill: bool, stroke: bool) {
    let fill_alpha = if fill { style.fill_opacity } else { 1.0 };
    let stroke_alpha = if stroke { style.stroke_opacity } else { 1.0 };
    if fill_alpha < 1.0 || stroke_alpha < 1.0 {
        let _ = writeln!(out, "{} gs", res.state(fill_alpha, stroke_alpha));
    }
    if let (true, Some(c)) = (fill, style.fill) {
        let _ = writeln!(out, "{} rg", rgb(c));
    }
    if let (true, Some(c)) = (stroke, style.stroke) {
        let _ = writeln!(out, "{} RG", rgb(c));
        let cap = match style.line_cap {
            SvgLineCap::Butt => 0,
            SvgLineCap::Round => 1,
            SvgLineCap::Square => 2,
        };
        let join = match style.line_join {
            SvgLineJoin::Miter => 0,
            SvgLineJoin::Round => 1,
            SvgLineJoin::Bevel => 2,
        };
        let _ = writeln!(out, "{} w {cap} J {join} j {} M", num(style.stroke_width), num(style.miter_limit));
        if !style.dash.is_empty() {
            let d: Vec<String> = style.dash.iter().map(|v| num(*v)).collect();
            let _ = writeln!(out, "[{}] 0 d", d.join(" "));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn text_run(
    out: &mut String,
    res: &mut Resources,
    ctm: &Affine,
    style: &Resolved,
    pos: [f64; 2],
    text: &str,
    size: f64,
    anchor: SvgTextAnchor,
    rotation: f64,
    family: Option<&str>,
) {
    let (fill, stroke) = (style.fills(), style.strokes());
    if !fill && !stroke {
        return;
    }
    let font = Font::for_family(family);
    let bytes = win_ansi(text);
    let width = font.width(&bytes) * size;
    let shift = match anchor {
        SvgTextAnchor::Start => 0.0,
        SvgTextAnchor::Middle => -width / 2.0,
        SvgTextAnchor::End => -width,
    };
    // Text space is y-up and the user space y-down: flip, then SVG's
    // rotate(r x y), then the anchor's shift along the baseline.
    let (sin, cos) = rotation.to_radians().sin_cos();
    let tm = [cos, sin, sin, -cos, pos[0] + cos * shift, pos[1] + sin * shift];
    out.push_str("q\n");
    if *ctm != drawing::IDENTITY {
        let _ = writeln!(out, "{} cm", matrix(ctm));
    }
    paint_state(out, res, style, fill, stroke);
    let mode = match (fill, stroke) {
        (true, true) => 2,
        (false, true) => 1,
        _ => 0,
    };
    let name = res.font(font);
    let _ = writeln!(out, "BT {name} {} Tf {mode} Tr {} Tm {} Tj ET", num(size), matrix(&tm), literal(&bytes));
    out.push_str("Q\n");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Font {
    Helvetica,
    Times,
    Courier,
}

impl Font {
    fn for_family(family: Option<&str>) -> Font {
        let f = family.unwrap_or("").to_ascii_lowercase();
        if f.contains("mono") || f.contains("courier") {
            Font::Courier
        } else if (f.contains("serif") || f.contains("times")) && !f.contains("sans") {
            Font::Times
        } else {
            Font::Helvetica
        }
    }

    fn base_name(self) -> &'static str {
        match self {
            Font::Helvetica => "Helvetica",
            Font::Times => "Times-Roman",
            Font::Courier => "Courier",
        }
    }

    /// The advance of WinAnsi `bytes` in ems.
    fn width(self, bytes: &[u8]) -> f64 {
        let table = match self {
            Font::Helvetica => &pdf_fonts::HELVETICA,
            Font::Times => &pdf_fonts::TIMES,
            Font::Courier => &pdf_fonts::COURIER,
        };
        bytes.iter().map(|&b| if b >= 32 { table[b as usize - 32] as f64 } else { 0.0 }).sum::<f64>() / 1000.0
    }
}

/// WinAnsiEncoding's codes 128 to 159, as Unicode (0 where unassigned).
const WIN_ANSI_HIGH: [u16; 32] = [
    0x20AC, 0, 0x201A, 0x0192, 0x201E, 0x2026, 0x2020, 0x2021, 0x02C6, 0x2030, 0x0160, 0x2039, 0x0152, 0, 0x017D, 0, 0, 0x2018, 0x2019,
    0x201C, 0x201D, 0x2022, 0x2013, 0x2014, 0x02DC, 0x2122, 0x0161, 0x203A, 0x0153, 0, 0x017E, 0x0178,
];

/// `text` in WinAnsiEncoding: ASCII and Latin-1 as themselves, the
/// typographic extras at 128 to 159, anything else (controls included) `?`.
fn win_ansi(text: &str) -> Vec<u8> {
    text.chars()
        .map(|c| match c as u32 {
            u @ 32..=126 | u @ 160..=255 => u as u8,
            u => WIN_ANSI_HIGH.iter().position(|&h| h != 0 && h as u32 == u).map_or(b'?', |k| 128 + k as u8),
        })
        .collect()
}

/// A literal string: parentheses and backslashes escaped, bytes outside
/// printable ASCII in octal.
fn literal(bytes: &[u8]) -> String {
    let mut s = String::from("(");
    for &b in bytes {
        match b {
            b'(' | b')' | b'\\' => {
                s.push('\\');
                s.push(b as char);
            }
            32..=126 => s.push(b as char),
            _ => {
                let _ = write!(s, "\\{b:03o}");
            }
        }
    }
    s.push(')');
    s
}

/// A text string for the document information: literal when it is
/// printable ASCII, else UTF-16BE with its byte order mark.
fn text_string(t: &str) -> String {
    if t.bytes().all(|b| (32..=126).contains(&b)) {
        return literal(t.as_bytes());
    }
    let mut s = String::from("<FEFF");
    for u in t.encode_utf16() {
        let _ = write!(s, "{u:04X}");
    }
    s.push('>');
    s
}

/// A number as PDF writes them: no exponent, at most six decimals, trailing
/// zeros and `-0` dropped.
fn num(v: f64) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" || s.is_empty() { "0".into() } else { s.to_string() }
}

fn matrix(m: &Affine) -> String {
    m.iter().map(|v| num(*v)).collect::<Vec<_>>().join(" ")
}

fn rgb(c: [u8; 3]) -> String {
    c.iter().map(|v| num((*v as f64 / 255.0 * 10000.0).round() / 10000.0)).collect::<Vec<_>>().join(" ")
}

/// Objects written in id order with their offsets kept for the xref table.
struct ObjWriter {
    out: Vec<u8>,
    offsets: Vec<usize>,
}

impl ObjWriter {
    fn new() -> Self {
        // the comment's high bytes mark the file binary for transfer tools
        ObjWriter { out: b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n".to_vec(), offsets: Vec::new() }
    }

    fn begin(&mut self, id: usize) {
        debug_assert_eq!(id, self.offsets.len() + 1, "objects are written in id order");
        self.offsets.push(self.out.len());
        self.out.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
    }

    fn object(&mut self, id: usize, body: &str) {
        self.begin(id);
        self.out.extend_from_slice(body.as_bytes());
        self.out.extend_from_slice(b"\nendobj\n");
    }

    fn stream(&mut self, id: usize, data: &[u8], compress: bool) {
        self.begin(id);
        let (data, filter) = if compress { (crate::utils::png::zlib(data), " /Filter /FlateDecode") } else { (data.to_vec(), "") };
        self.out.extend_from_slice(format!("<< /Length {}{filter} >>\nstream\n", data.len()).as_bytes());
        self.out.extend_from_slice(&data);
        self.out.extend_from_slice(b"\nendstream\nendobj\n");
    }

    fn finish(mut self, count: usize, root: usize, info: usize) -> Vec<u8> {
        debug_assert_eq!(count, self.offsets.len());
        let xref = self.out.len();
        let mut t = format!("xref\n0 {}\n0000000000 65535 f \n", count + 1);
        for o in &self.offsets {
            let _ = writeln!(t, "{o:010} 00000 n ");
        }
        let _ = write!(t, "trailer\n<< /Size {} /Root {root} 0 R /Info {info} 0 R >>\nstartxref\n{xref}\n%%EOF\n", count + 1);
        self.out.extend_from_slice(t.as_bytes());
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exporters::{SvgNode, SvgPath, SvgStyle, SvgUnit};

    fn content(doc: &SvgDocument) -> String {
        let pdf = write_pdf_pages_with(std::slice::from_ref(doc), &PdfOptions { title: None, compress: false });
        String::from_utf8_lossy(&pdf).into_owned()
    }

    #[test]
    fn numbers_have_no_exponent_and_no_negative_zero() {
        assert_eq!(num(1.5), "1.5");
        assert_eq!(num(-0.0000001), "0");
        assert_eq!(num(1e-7), "0");
        assert_eq!(num(123456789.0), "123456789");
        assert_eq!(num(2.0 / 3.0), "0.666667");
    }

    #[test]
    fn the_page_is_the_drawing_in_points_with_y_flipped() {
        let doc = SvgDocument::new(210.0, 297.0, SvgUnit::Millimeters);
        let c = content(&doc);
        assert!(c.contains("/MediaBox [0 0 595.275591 841.889764]"), "{c}");
        // 1 mm = 72 / 25.4 pt, y from the top
        assert!(c.contains("2.834646 0 0 -2.834646 0 841.889764 cm"), "{c}");
    }

    #[test]
    fn the_xref_points_at_every_object() {
        let mut doc = SvgDocument::new(50.0, 50.0, SvgUnit::Points);
        doc.circle([25.0, 25.0], 10.0, SvgStyle::fill([255, 0, 0]));
        let pdf = write_pdf_pages(&[doc.clone(), doc]);
        // offsets are bytes, and the header's marker and the deflated streams are not UTF-8
        let rfind = |needle: &[u8]| pdf.windows(needle.len()).rposition(|w| w == needle).unwrap();
        let ascii = |from: usize| pdf[from..].iter().take_while(|b| b.is_ascii() && **b != 0).map(|&b| b as char).collect::<String>();
        let xref = rfind(b"\nxref\n") + 1;
        let start = rfind(b"startxref\n") + 10;
        assert_eq!(ascii(start).lines().next().unwrap(), xref.to_string());
        for (k, line) in ascii(xref).lines().skip(3).take_while(|l| l.ends_with(" n ")).enumerate() {
            let at: usize = line[..10].parse().unwrap();
            assert!(pdf[at..].starts_with(format!("{} 0 obj", k + 1).as_bytes()), "object {} at {at}", k + 1);
        }
        let text = String::from_utf8_lossy(&pdf);
        assert!(text.contains("/Count 2"));
    }

    #[test]
    fn styles_become_operators() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        doc.circle([5.0, 5.0], 2.0, SvgStyle::fill([255, 0, 0]).with_opacity(0.5).with_fill_rule(SvgFillRule::EvenOdd));
        doc.line([0.0, 0.0], [10.0, 10.0], SvgStyle::stroke([0, 0, 255], 0.25).with_dash(&[1.0, 0.5]));
        doc.rect([1.0, 1.0], [2.0, 2.0], 0.0, SvgStyle { fill: Some(crate::exporters::SvgPaint::None), ..Default::default() });
        let c = content(&doc);
        assert!(c.contains("/GS0 gs\n1 0 0 rg\n"), "{c}");
        assert!(c.contains("f*\n"));
        assert!(c.contains("0 0 1 RG\n0.25 w 1 J 1 j 4 M\n[1 0.5] 0 d\n"), "{c}");
        assert!(c.contains("/ExtGState << /GS0 << /Type /ExtGState /ca 0.5 /CA 1 >> >>"), "{c}");
        // nothing painted, nothing written
        assert_eq!(c.matches("\nq\n").count(), 2);
    }

    #[test]
    fn text_is_anchored_with_the_fonts_widths() {
        let mut doc = SvgDocument::new(100.0, 100.0, SvgUnit::None);
        doc.children.push(SvgNode::Text {
            pos: [50.0, 20.0],
            text: "AV (é)".into(),
            size: 10.0,
            anchor: SvgTextAnchor::Middle,
            rotation: 0.0,
            font_family: None,
            style: SvgStyle::default(),
        });
        let c = content(&doc);
        // A 667 + V 667 + space 278 + ( 333 + é 556 + ) 333 = 2834 / 1000 em
        assert!(c.contains("BT /F0 10 Tf 0 Tr 1 0 0 -1 35.83 20 Tm (AV \\(\\351\\)) Tj ET"), "{c}");
        assert!(c.contains("/BaseFont /Helvetica /Encoding /WinAnsiEncoding"));
    }

    #[test]
    fn win_ansi_maps_the_typographic_extras() {
        assert_eq!(win_ansi("€–—’ü✓"), vec![128, 150, 151, 146, 252, b'?']);
        assert_eq!(Font::for_family(Some("DejaVu Sans Mono")), Font::Courier);
        assert_eq!(Font::for_family(Some("serif")), Font::Times);
        assert_eq!(Font::for_family(Some("sans-serif")), Font::Helvetica);
    }

    #[test]
    fn titles_outside_ascii_are_utf16() {
        assert_eq!(text_string("a(b)"), "(a\\(b\\))");
        assert_eq!(text_string("Ω"), "<FEFF03A9>");
    }

    #[test]
    fn the_same_drawing_is_the_same_bytes() {
        let mut doc = SvgDocument::new(10.0, 10.0, SvgUnit::None);
        let mut p = SvgPath::new();
        p.move_to([1.0, 1.0]).arc([5.0, 5.0], 3.0, 0.0, 270.0);
        doc.path(p, SvgStyle::stroke([0, 0, 0], 0.5));
        assert_eq!(write_pdf(&doc), write_pdf(&doc));
    }
}
