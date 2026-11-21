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
//! Text is set in the face [`PdfOptions::text`] gives its family, DejaVu Sans
//! unless told otherwise, embedded as a TrueType subset (only the glyphs
//! used, Identity-H, with a `ToUnicode` map so it copies and searches as
//! text): any character the face has, kerned, laid out glyph for glyph as
//! the PNG is (`fonts.rs`), so the two agree.
//!
//! [`PdfText::Standard`] uses the standard fonts every viewer has instead, so
//! none is embedded and the file is smaller: Helvetica, or Courier for a
//! family naming `mono` or `courier`, or Times for one naming `serif` or
//! `times` but not `sans`, encoded as WinAnsi (Latin-1 and the usual
//! typographic extras; other characters print as `?`), anchored with the
//! fonts' real advance widths from the URW AFM files
//! (`scripts/afm_widths.py`).
//!
//! Gradients are shading patterns (stop opacities a luminosity soft mask of
//! the same gradient in grey), an element's or group's `opacity` a
//! transparency group, a clip `W n`, and an image an image XObject with its
//! alpha as a soft mask.
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

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::Arc;

use super::drawing::{self, Affine, Item, Paint, Resolved, Seg};
use super::fonts::{self, DrawingFonts, Face};
use super::pdf_fonts;
use super::svg::{SvgDocument, SvgFillRule, SvgGradient, SvgGradientKind, SvgLineCap, SvgLineJoin, SvgTextAnchor};
use crate::utils::png::PngImage;

/// How [`write_pdf_pages_with`] writes.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfOptions {
    /// The document title; the first page's [`SvgDocument::title`] when `None`.
    pub title: Option<String>,
    /// Deflate the content streams.
    pub compress: bool,
    /// How text is set.
    pub text: PdfText,
}

impl Default for PdfOptions {
    fn default() -> Self {
        PdfOptions { title: None, compress: true, text: PdfText::Embedded(DrawingFonts::new()) }
    }
}

/// How [`write_pdf_pages_with`] sets text.
#[derive(Debug, Clone, PartialEq)]
pub enum PdfText {
    /// In these faces, each embedded as the subset of it the file uses.
    Embedded(DrawingFonts),
    /// In the standard Helvetica, Times and Courier, not embedded; WinAnsi
    /// characters only.
    Standard,
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
    let mut res = Resources { text: options.text.clone(), ..Default::default() };
    let contents: Vec<(String, [f64; 2])> = pages.iter().map(|d| page_content(d, &mut res)).collect();

    let mut w = ObjWriter::new();
    // 1: catalog, 2: page tree, 3: resources, 4: info, then each page and its
    // content, then what the resources refer to
    let n_fixed = 4;
    let page_id = |k: usize| n_fixed + 1 + 2 * k;
    res.first_extra = n_fixed + 2 * pages.len() + 1;
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
    let count = res.write_objects(&mut w, options.compress);
    w.finish(count, 1, 4)
}

/// The fonts and graphics states the pages use, shared by all of them.
#[derive(Default)]
struct Resources {
    text: PdfText,
    /// Standard fonts, by name index.
    fonts: BTreeMap<&'static str, usize>,
    /// Embedded faces, in order of first use: the glyphs used and the
    /// character each stands for.
    embedded: Vec<(Arc<Face>, BTreeMap<u16, char>)>,
    /// (fill alpha, stroke alpha) in thousandths, by name index.
    states: BTreeMap<(u16, u16), usize>,
    /// Shading patterns: a gradient and the matrix from its space to the
    /// default space of the stream that paints it.
    patterns: Vec<(SvgGradient, Affine)>,
    /// Soft masks of gradients' stop opacities, set where the gradient's
    /// user space is the current one.
    masks: Vec<SvgGradient>,
    /// Transparency groups' content streams.
    forms: Vec<String>,
    images: Vec<Arc<PngImage>>,
    /// The id the objects after the pages start at.
    first_extra: usize,
}

/// Where each kind of object after the pages starts.
struct Layout {
    patterns: usize,
    masks: usize,
    forms: usize,
    images: usize,
    end: usize,
}

impl Default for PdfText {
    fn default() -> Self {
        PdfText::Embedded(DrawingFonts::new())
    }
}

/// Objects an embedded face takes: the Type 0 font, its CID font, the
/// descriptor, the font file and the `ToUnicode` map.
const OBJECTS_PER_FACE: usize = 5;

impl Resources {
    /// An embedded face's resource name, noting the glyphs it shows.
    fn embedded_font(&mut self, face: &Arc<Face>, glyphs: impl Iterator<Item = (u16, char)>) -> String {
        let k = match self.embedded.iter().position(|(f, _)| Arc::ptr_eq(f, face)) {
            Some(k) => k,
            None => {
                self.embedded.push((face.clone(), BTreeMap::new()));
                self.embedded.len() - 1
            }
        };
        for (g, c) in glyphs {
            self.embedded[k].1.entry(g).or_insert(c);
        }
        format!("/E{k}")
    }

    fn layout(&self) -> Layout {
        let patterns = self.first_extra + OBJECTS_PER_FACE * self.embedded.len();
        let masks = patterns + self.patterns.len();
        let forms = masks + 2 * self.masks.len();
        let images = forms + self.forms.len();
        Layout { patterns, masks, forms, images, end: images + 2 * self.images.len() }
    }

    fn pattern(&mut self, g: &SvgGradient, m: Affine) -> String {
        let k = self.patterns.iter().position(|(h, n)| h == g && *n == m).unwrap_or_else(|| {
            self.patterns.push((g.clone(), m));
            self.patterns.len() - 1
        });
        format!("/P{k}")
    }

    fn mask(&mut self, g: &SvgGradient) -> String {
        let k = self.masks.iter().position(|h| h == g).unwrap_or_else(|| {
            self.masks.push(g.clone());
            self.masks.len() - 1
        });
        format!("/M{k}")
    }

    fn image(&mut self, image: &PngImage) -> String {
        let k = self.images.iter().position(|i| **i == *image).unwrap_or_else(|| {
            self.images.push(Arc::new(image.clone()));
            self.images.len() - 1
        });
        format!("/I{k}")
    }

    /// Writes the objects the resource dictionary refers to, from
    /// `first_extra` on; returns the number of objects in the file.
    fn write_objects(&self, w: &mut ObjWriter, compress: bool) -> usize {
        let layout = self.layout();
        let mut id = self.first_extra;
        for (face, used) in &self.embedded {
            let gids: BTreeSet<u16> = used.keys().copied().collect();
            let file = fonts::subset(face, &gids).unwrap_or_default();
            let upem = face.units_per_em();
            let scale = |v: f64| num((v * 1000.0 / upem).round());
            // a subset's name is tagged with six letters of its own
            let tag: String = {
                let mut h: u32 = 2166136261;
                for g in &gids {
                    h = (h ^ *g as u32).wrapping_mul(16777619);
                }
                (0..6).map(|k| (b'A' + ((h >> (k * 5)) % 26) as u8) as char).collect()
            };
            let name = format!("{tag}+{}", face.postscript_name);
            let head = face.table(b"head").unwrap_or_default();
            let i16_at = |t: &[u8], at: usize| t.get(at..at + 2).map_or(0, |b| i16::from_be_bytes([b[0], b[1]])) as f64;
            let bbox: Vec<String> = [36, 38, 40, 42].iter().map(|&at| scale(i16_at(head, at))).collect();
            let os2 = face.table(b"OS/2");
            let cap = os2.filter(|t| t.len() >= 90).map_or(face.ttf.ascender as f64, |t| i16_at(t, 88));
            let widths: Vec<String> = gids.iter().map(|&g| format!("{g} [{}]", scale(face.ttf.advance(g) as f64))).collect();
            let (font, cid, desc, ff, tu) = (id, id + 1, id + 2, id + 3, id + 4);
            w.object(
                font,
                &format!("<< /Type /Font /Subtype /Type0 /BaseFont /{name} /Encoding /Identity-H /DescendantFonts [{cid} 0 R] /ToUnicode {tu} 0 R >>"),
            );
            w.object(
                cid,
                &format!(
                    "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{name} /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor {desc} 0 R /CIDToGIDMap /Identity /DW 0 /W [{}] >>",
                    widths.join(" ")
                ),
            );
            w.object(
                desc,
                &format!(
                    "<< /Type /FontDescriptor /FontName /{name} /Flags 32 /FontBBox [{}] /ItalicAngle 0 /Ascent {} /Descent {} /CapHeight {} /StemV 80 /FontFile2 {ff} 0 R >>",
                    bbox.join(" "),
                    scale(face.ttf.ascender as f64),
                    scale(face.ttf.descender as f64),
                    scale(cap)
                ),
            );
            w.stream_with(ff, &file, compress, &format!(" /Length1 {}", file.len()));
            w.stream(tu, to_unicode(used).as_bytes(), compress);
            id += OBJECTS_PER_FACE;
        }
        for (k, (g, m)) in self.patterns.iter().enumerate() {
            w.object(layout.patterns + k, &format!("<< /PatternType 2 /Shading {} /Matrix [{}] >>", shading(g, false), matrix(m)));
        }
        for (k, g) in self.masks.iter().enumerate() {
            let (gs, form) = (layout.masks + 2 * k, layout.masks + 2 * k + 1);
            w.object(gs, &format!("<< /Type /ExtGState /SMask << /Type /Mask /S /Luminosity /G {form} 0 R >> >>"));
            let content = match &g.transform {
                Some(t) => format!("{} cm /S0 sh\n", matrix(t)),
                None => "/S0 sh\n".into(),
            };
            let extra = format!(
                " /Type /XObject /Subtype /Form /BBox [{BIG}] /Group << /S /Transparency /CS /DeviceGray >> /Resources << /Shading << /S0 {} >> >>",
                shading(g, true)
            );
            w.stream_with(form, content.as_bytes(), compress, &extra);
        }
        for (k, content) in self.forms.iter().enumerate() {
            let extra = format!(" /Type /XObject /Subtype /Form /BBox [{BIG}] /Group << /S /Transparency /CS /DeviceRGB >> /Resources 3 0 R");
            w.stream_with(layout.forms + k, content.as_bytes(), compress, &extra);
        }
        for (k, img) in self.images.iter().enumerate() {
            let (id, mask) = (layout.images + 2 * k, layout.images + 2 * k + 1);
            let px = img.rgba.as_chunks::<4>().0;
            let rgb: Vec<u8> = px.iter().flat_map(|p| [p[0], p[1], p[2]]).collect();
            let alpha: Vec<u8> = px.iter().map(|p| p[3]).collect();
            let head = |cs: &str| format!(" /Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /{cs} /BitsPerComponent 8 /Interpolate true", img.width, img.height);
            w.stream_with(id, &rgb, compress, &format!("{} /SMask {mask} 0 R", head("DeviceRGB")));
            w.stream_with(mask, &alpha, compress, &head("DeviceGray"));
        }
        layout.end - 1
    }

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
        if !self.fonts.is_empty() || !self.embedded.is_empty() {
            d.push_str(" /Font <<");
            for (name, k) in &self.fonts {
                let _ = write!(d, " /F{k} << /Type /Font /Subtype /Type1 /BaseFont /{name} /Encoding /WinAnsiEncoding >>");
            }
            for k in 0..self.embedded.len() {
                let _ = write!(d, " /E{k} {} 0 R", self.first_extra + OBJECTS_PER_FACE * k);
            }
            d.push_str(" >>");
        }
        let layout = self.layout();
        if !self.states.is_empty() || !self.masks.is_empty() {
            d.push_str(" /ExtGState <<");
            for ((ca, sa), k) in &self.states {
                let _ = write!(d, " /GS{k} << /Type /ExtGState /ca {} /CA {} >>", num(*ca as f64 / 1000.0), num(*sa as f64 / 1000.0));
            }
            for k in 0..self.masks.len() {
                let _ = write!(d, " /M{k} {} 0 R", layout.masks + 2 * k);
            }
            d.push_str(" >>");
        }
        if !self.patterns.is_empty() {
            d.push_str(" /Pattern <<");
            for k in 0..self.patterns.len() {
                let _ = write!(d, " /P{k} {} 0 R", layout.patterns + k);
            }
            d.push_str(" >>");
        }
        if !self.forms.is_empty() || !self.images.is_empty() {
            d.push_str(" /XObject <<");
            for k in 0..self.forms.len() {
                let _ = write!(d, " /X{k} {} 0 R", layout.forms + k);
            }
            for k in 0..self.images.len() {
                let _ = write!(d, " /I{k} {} 0 R", layout.images + 2 * k);
            }
            d.push_str(" >>");
        }
        d.push_str(" >>");
        d
    }
}

/// A form's bounding box: everything a drawing could reach.
const BIG: &str = "-100000 -100000 100000 100000";

/// A page's content stream and its size in points.
fn page_content(doc: &SvgDocument, res: &mut Resources) -> (String, [f64; 2]) {
    let inches = drawing::page_inches(doc);
    let size = [inches[0] * 72.0, inches[1] * 72.0];
    let Some(view) = drawing::viewport(doc, size) else {
        return (String::new(), size);
    };
    // PDF's y points up from the bottom of the page.
    let base = drawing::compose(&[1.0, 0.0, 0.0, -1.0, 0.0, size[1]], &view);
    let mut page = Page { res, doc, base, bufs: vec![format!("{} cm\n", matrix(&base))], frames: Vec::new() };
    drawing::walk(doc, &mut |item| page.item(item));
    while !page.frames.is_empty() {
        page.item(Item::Pop);
    }
    (page.bufs.swap_remove(0), size)
}

/// A page being written: its content and the transparency groups open in it.
struct Page<'a> {
    res: &'a mut Resources,
    doc: &'a SvgDocument,
    /// The view box to the page's default space.
    base: Affine,
    /// The page's content, then each open group's.
    bufs: Vec<String>,
    /// Each open layer: its opacity, whether it clips, whether it is a group.
    frames: Vec<(f64, bool, bool)>,
}

/// How one pass of painting a shape or text is done.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    Fill,
    Stroke,
    Both,
}

impl Page<'_> {
    fn out(&mut self) -> &mut String {
        self.bufs.last_mut().unwrap()
    }

    /// The transform from the view box to the default space of the stream
    /// being written: the page's, or a group's own (which is the view box).
    fn container(&self) -> Affine {
        if self.bufs.len() > 1 { drawing::IDENTITY } else { self.base }
    }

    fn item(&mut self, item: Item) {
        match item {
            Item::Shape { ctm, style, segs } => self.shape(&ctm, &style, &segs),
            Item::Text { ctm, style, pos, text, size, anchor, rotation, font_family } => {
                self.text_run(&ctm, &style, pos, text, size, anchor, rotation, font_family)
            }
            Item::Image { ctm, origin, size, image } => {
                let name = self.res.image(image);
                let mut s = String::from("q\n");
                if ctm != drawing::IDENTITY {
                    let _ = writeln!(s, "{} cm", matrix(&ctm));
                }
                let place = [size[0], 0.0, 0.0, -size[1], origin[0], origin[1] + size[1]];
                let _ = writeln!(s, "{} cm {name} Do\nQ", matrix(&place));
                self.out().push_str(&s);
            }
            Item::Push { ctm, opacity, clip } => {
                let clipped = clip.is_some();
                if let Some((segs, rule)) = clip {
                    let moved: Vec<Seg> = segs.iter().map(|s| transform_seg(s, &ctm)).collect();
                    let mut s = String::from("q\n");
                    path(&mut s, &moved);
                    s.push_str(if rule == SvgFillRule::EvenOdd { "W* n\n" } else { "W n\n" });
                    self.out().push_str(&s);
                }
                let group = opacity < 1.0;
                if group {
                    self.bufs.push(String::new());
                }
                self.frames.push((opacity, clipped, group));
            }
            Item::Pop => {
                let Some((opacity, clipped, group)) = self.frames.pop() else { return };
                if group {
                    let content = self.bufs.pop().unwrap_or_default();
                    self.res.forms.push(content);
                    let k = self.res.forms.len() - 1;
                    let gs = self.res.state(opacity, opacity);
                    let s = if clipped { format!("{gs} gs /X{k} Do\nQ\n") } else { format!("q {gs} gs /X{k} Do Q\n") };
                    self.out().push_str(&s);
                } else if clipped {
                    self.out().push_str("Q\n");
                }
            }
        }
    }

    /// The passes painting `style` takes: one, unless a gradient's soft mask
    /// would also mask the other paint.
    fn passes(&self, style: &Resolved) -> Vec<Pass> {
        let (fill, stroke) = (style.fills(), style.strokes());
        let masked = |p: Option<Paint>| matches!(p, Some(Paint::Gradient(k)) if self.doc.gradients.get(k).is_some_and(has_alpha));
        match (fill, stroke) {
            (true, true) if masked(style.fill) || masked(style.stroke) => vec![Pass::Fill, Pass::Stroke],
            (true, true) => vec![Pass::Both],
            (true, false) => vec![Pass::Fill],
            (false, true) => vec![Pass::Stroke],
            _ => Vec::new(),
        }
    }

    fn shape(&mut self, ctm: &Affine, style: &Resolved, segs: &[Seg]) {
        let even_odd = style.fill_rule == SvgFillRule::EvenOdd;
        for pass in self.passes(style) {
            let mut s = String::from("q\n");
            if *ctm != drawing::IDENTITY {
                let _ = writeln!(s, "{} cm", matrix(ctm));
            }
            self.paint_state(&mut s, ctm, style, pass);
            path(&mut s, segs);
            s.push_str(match (pass, even_odd) {
                (Pass::Both, false) => "B\n",
                (Pass::Both, true) => "B*\n",
                (Pass::Fill, false) => "f\n",
                (Pass::Fill, true) => "f*\n",
                (Pass::Stroke, _) => "S\n",
            });
            s.push_str("Q\n");
            self.out().push_str(&s);
        }
    }

    /// Colours or patterns, opacities, soft masks and the stroke's
    /// parameters, for a pass.
    fn paint_state(&mut self, s: &mut String, ctm: &Affine, style: &Resolved, pass: Pass) {
        let (fill, stroke) = (pass != Pass::Stroke, pass != Pass::Fill);
        let fill_alpha = if fill { style.fill_opacity } else { 1.0 };
        let stroke_alpha = if stroke { style.stroke_opacity } else { 1.0 };
        if fill_alpha < 1.0 || stroke_alpha < 1.0 {
            let _ = writeln!(s, "{} gs", self.res.state(fill_alpha, stroke_alpha));
        }
        if let (true, Some(p)) = (fill, style.fill) {
            self.set_paint(s, ctm, p, false);
        }
        if let (true, Some(p)) = (stroke, style.stroke) {
            self.set_paint(s, ctm, p, true);
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
            let _ = writeln!(s, "{} w {cap} J {join} j {} M", num(style.stroke_width), num(style.miter_limit));
            if !style.dash.is_empty() {
                let d: Vec<String> = style.dash.iter().map(|v| num(*v)).collect();
                let _ = writeln!(s, "[{}] 0 d", d.join(" "));
            }
        }
    }

    /// A colour, or a gradient's pattern (and soft mask, for stop opacities).
    fn set_paint(&mut self, s: &mut String, ctm: &Affine, paint: Paint, stroking: bool) {
        match paint {
            Paint::Color(c) => {
                let _ = writeln!(s, "{} {}", rgb(c), if stroking { "RG" } else { "rg" });
            }
            Paint::Gradient(k) => {
                let Some(g) = self.doc.gradients.get(k) else { return };
                let to_container = drawing::compose(&self.container(), ctm);
                let m = match &g.transform {
                    Some(t) => drawing::compose(&to_container, t),
                    None => to_container,
                };
                let name = self.res.pattern(g, m);
                let _ = writeln!(s, "{} {name} {}", if stroking { "/Pattern CS" } else { "/Pattern cs" }, if stroking { "SCN" } else { "scn" });
                if has_alpha(g) {
                    let _ = writeln!(s, "{} gs", self.res.mask(g));
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn text_run(
        &mut self,
        ctm: &Affine,
        style: &Resolved,
        pos: [f64; 2],
        text: &str,
        size: f64,
        anchor: SvgTextAnchor,
        rotation: f64,
        family: Option<&str>,
    ) {
        let passes = self.passes(style);
        if passes.is_empty() {
            return;
        }
        // what the run shows, how wide it is, and the font resource it uses
        let face = match &self.res.text {
            PdfText::Standard => None,
            PdfText::Embedded(fonts) => Some(fonts.face(family)),
        };
        let (shown, width, name) = match face {
            None => {
                let font = Font::for_family(family);
                let bytes = win_ansi(text);
                let width = font.width(&bytes) * size;
                (literal(&bytes), width, self.res.font(font))
            }
            Some(face) => {
                let run = fonts::layout(&face, text);
                let name = self.res.embedded_font(&face, run.glyphs.iter().map(|&(g, c, _)| (g, c)));
                (tj_array(&face, &run), run.advance * size / face.units_per_em(), name)
            }
        };
        let shift = match anchor {
            SvgTextAnchor::Start => 0.0,
            SvgTextAnchor::Middle => -width / 2.0,
            SvgTextAnchor::End => -width,
        };
        // Text space is y-up and the user space y-down: flip, then SVG's
        // rotate(r x y), then the anchor's shift along the baseline.
        let (sin, cos) = rotation.to_radians().sin_cos();
        let tm = [cos, sin, sin, -cos, pos[0] + cos * shift, pos[1] + sin * shift];
        let op = if shown.starts_with('[') { "TJ" } else { "Tj" };
        for pass in passes {
            let mut s = String::from("q\n");
            if *ctm != drawing::IDENTITY {
                let _ = writeln!(s, "{} cm", matrix(ctm));
            }
            self.paint_state(&mut s, ctm, style, pass);
            let mode = match pass {
                Pass::Both => 2,
                Pass::Stroke => 1,
                Pass::Fill => 0,
            };
            let _ = writeln!(s, "BT {name} {} Tf {mode} Tr {} Tm {shown} {op} ET", num(size), matrix(&tm));
            s.push_str("Q\n");
            self.out().push_str(&s);
        }
    }
}

/// Path construction operators for `segs`.
fn path(s: &mut String, segs: &[Seg]) {
    for seg in segs {
        let _ = match seg {
            Seg::Move(p) => writeln!(s, "{} {} m", num(p[0]), num(p[1])),
            Seg::Line(p) => writeln!(s, "{} {} l", num(p[0]), num(p[1])),
            Seg::Cubic(a, b, p) => writeln!(s, "{} {} {} {} {} {} c", num(a[0]), num(a[1]), num(b[0]), num(b[1]), num(p[0]), num(p[1])),
            Seg::Close => writeln!(s, "h"),
        };
    }
}

fn transform_seg(s: &Seg, m: &Affine) -> Seg {
    match *s {
        Seg::Move(p) => Seg::Move(drawing::apply(m, p)),
        Seg::Line(p) => Seg::Line(drawing::apply(m, p)),
        Seg::Cubic(a, b, p) => Seg::Cubic(drawing::apply(m, a), drawing::apply(m, b), drawing::apply(m, p)),
        Seg::Close => Seg::Close,
    }
}

/// Whether a gradient has a stop less than opaque.
fn has_alpha(g: &SvgGradient) -> bool {
    g.stops.iter().any(|s| s.opacity < 1.0)
}

/// A gradient as a shading dictionary: its colours, or, for `alpha`, its
/// stop opacities as grey. Padded at both ends; SVG's degenerate cases (a
/// zero-length vector, a zero radius) paint the last stop.
fn shading(g: &SvgGradient, alpha: bool) -> String {
    let stops = drawing::stops(&g.stops);
    let value = |st: &super::svg::SvgStop| -> String {
        if alpha {
            num(st.opacity)
        } else {
            st.color.iter().map(|c| num((*c as f64 / 255.0 * 10000.0).round() / 10000.0)).collect::<Vec<_>>().join(" ")
        }
    };
    let cs = if alpha { "/DeviceGray" } else { "/DeviceRGB" };
    let constant = |v: String| format!("<< /FunctionType 2 /Domain [0 1] /C0 [{v}] /C1 [{v}] /N 1 >>");
    let function = match stops.as_slice() {
        [] => constant(if alpha { "0".into() } else { "0 0 0".into() }),
        [only] => constant(value(only)),
        _ => {
            let mut st = stops.clone();
            if st[0].offset > 0.0 {
                st.insert(0, super::svg::SvgStop { offset: 0.0, ..st[0] });
            }
            if st[st.len() - 1].offset < 1.0 {
                st.push(super::svg::SvgStop { offset: 1.0, ..st[st.len() - 1] });
            }
            let parts: Vec<String> = st
                .windows(2)
                .map(|w| format!("<< /FunctionType 2 /Domain [0 1] /C0 [{}] /C1 [{}] /N 1 >>", value(&w[0]), value(&w[1])))
                .collect();
            let bounds: Vec<String> = st[1..st.len() - 1].iter().map(|s| num(s.offset)).collect();
            let encode = vec!["0 1"; parts.len()].join(" ");
            format!("<< /FunctionType 3 /Domain [0 1] /Functions [{}] /Bounds [{}] /Encode [{encode}] >>", parts.join(" "), bounds.join(" "))
        }
    };
    let last = stops.last().map(value).unwrap_or_else(|| if alpha { "0".into() } else { "0 0 0".into() });
    let (kind, coords, function) = match &g.kind {
        SvgGradientKind::Linear { start, end } if start != end => (2, format!("{} {} {} {}", num(start[0]), num(start[1]), num(end[0]), num(end[1])), function),
        SvgGradientKind::Linear { start, .. } => (2, format!("{} {} {} {}", num(start[0]), num(start[1]), num(start[0] + 1.0), num(start[1])), constant(last)),
        SvgGradientKind::Radial { center, radius, focus } if *radius > 0.0 => {
            let f = drawing::radial_focus(*center, *radius, *focus);
            (3, format!("{} {} 0 {} {} {}", num(f[0]), num(f[1]), num(center[0]), num(center[1]), num(*radius)), function)
        }
        SvgGradientKind::Radial { center, .. } => (3, format!("{} {} 0 {} {} 1", num(center[0]), num(center[1]), num(center[0]), num(center[1])), constant(last)),
    };
    format!("<< /ShadingType {kind} /ColorSpace {cs} /Coords [{coords}] /Function {function} /Extend [true true] >>")
}

/// A laid-out run as a `TJ` array: glyph IDs in hex, broken where kerning
/// moves the pen, by thousandths of an em the other way.
fn tj_array(face: &Face, run: &fonts::Run) -> String {
    let upem = face.units_per_em();
    let mut s = String::from("[<");
    let mut pen = 0.0;
    for &(g, _, x) in &run.glyphs {
        let adjust = x - pen;
        if adjust.abs() > 1e-9 {
            let _ = write!(s, "> {} <", num(-adjust * 1000.0 / upem));
        }
        let _ = write!(s, "{g:04X}");
        pen = x + face.ttf.advance(g) as f64;
    }
    s.push_str(">]");
    s
}

/// The `ToUnicode` CMap of an Identity-H font: each glyph's character.
fn to_unicode(used: &BTreeMap<u16, char>) -> String {
    let mut s = String::from(
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n",
    );
    let pairs: Vec<(&u16, &char)> = used.iter().collect();
    for chunk in pairs.chunks(100) {
        let _ = writeln!(s, "{} beginbfchar", chunk.len());
        for (g, c) in chunk {
            let units: Vec<String> = c.encode_utf16(&mut [0; 2]).iter().map(|u| format!("{u:04X}")).collect();
            let _ = writeln!(s, "<{g:04X}> <{}>", units.join(""));
        }
        s.push_str("endbfchar\n");
    }
    s.push_str("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n");
    s
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
        self.stream_with(id, data, compress, "");
    }

    /// A stream with more entries in its dictionary.
    fn stream_with(&mut self, id: usize, data: &[u8], compress: bool, extra: &str) {
        self.begin(id);
        let (data, filter) = if compress { (crate::utils::png::zlib(data), " /Filter /FlateDecode") } else { (data.to_vec(), "") };
        self.out.extend_from_slice(format!("<< /Length {}{filter}{extra} >>\nstream\n", data.len()).as_bytes());
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
        let pdf = write_pdf_pages_with(std::slice::from_ref(doc), &PdfOptions { compress: false, text: PdfText::Standard, ..Default::default() });
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

    fn label(text: &str, anchor: SvgTextAnchor) -> SvgDocument {
        let mut doc = SvgDocument::new(100.0, 100.0, SvgUnit::None);
        doc.children.push(SvgNode::Text { pos: [50.0, 20.0], text: text.into(), size: 10.0, anchor, rotation: 0.0, font_family: None, style: SvgStyle::default() });
        doc
    }

    #[test]
    fn embedded_text_is_a_subset_with_unicode_and_kerning() {
        let pdf = write_pdf_pages_with(&[label("AV ж€", SvgTextAnchor::Start)], &PdfOptions { compress: false, ..Default::default() });
        let c = String::from_utf8_lossy(&pdf);
        assert!(c.contains("/Subtype /Type0"), "{c}");
        assert!(c.contains("+DejaVuSans /Encoding /Identity-H"));
        assert!(c.contains("/FontFile2"));
        let face = fonts::bundled();
        let g = |ch: char| face.ttf.glyph_id(ch).unwrap();
        // AV kerned apart by the face's own pair value, in thousandths the other way
        let kern = -(face.ttf.kern(g('A'), g('V')) as f64) * 1000.0 / face.units_per_em();
        assert!(c.contains(&format!("[<{:04X}> {} <{:04X}", g('A'), num(kern), g('V'))), "{c}");
        // every glyph shown maps back to its character
        assert!(c.contains(&format!("<{:04X}> <0436>", g('ж'))));
        assert!(c.contains(&format!("<{:04X}> <20AC>", g('€'))));
    }

    #[test]
    fn embedded_text_is_anchored_as_the_png_lays_it_out() {
        let face = fonts::bundled();
        let run = fonts::layout(&face, "Wavy");
        let width = run.advance * 10.0 / face.units_per_em();
        let pdf = write_pdf_pages_with(&[label("Wavy", SvgTextAnchor::End)], &PdfOptions { compress: false, ..Default::default() });
        let c = String::from_utf8_lossy(&pdf);
        assert!(c.contains(&format!("1 0 0 -1 {} 20 Tm", num(50.0 - width))), "{c}");
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
