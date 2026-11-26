//! A drawing written out: a PDF page (the standard fonts, the bitmaps as
//! images), drawings as a PDF's pages ([`pdf_pages`]), or an SVG.

use std::fmt::Write;

use crate::{Drawing, Op, Rgba, Seg};

fn pdf_font(bold: bool, italic: bool, family: &str) -> usize {
    let f = family.to_lowercase();
    if f == "symbol" {
        return 12;
    }
    let base = if f.contains("times") || f.contains("roman") || f.contains("serif") && !f.contains("sans") || f.contains("georgia") || f.contains("cambria") {
        4
    } else if f.contains("courier") || f.contains("mono") || f.contains("consolas") {
        8
    } else {
        0
    };
    base + bold as usize + 2 * italic as usize
}

const FONTS: [&str; 13] = [
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    "Symbol",
];

/// Bytes as a PDF string's body, escaped.
fn pdf_escape(bytes: &[u8]) -> String {
    let mut out = String::new();
    for &b in bytes {
        match b {
            b'(' | b')' | b'\\' => {
                out.push('\\');
                out.push(b as char);
            }
            0x20..=0x7E => out.push(b as char),
            _ => {
                let _ = write!(out, "\\{b:03o}");
            }
        }
    }
    out
}

/// Text in WinAnsi, PDF-escaped.
fn pdf_text(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        let b = match c {
            '(' | ')' | '\\' => {
                out.push('\\');
                c as u32
            }
            '\u{20AC}' => 0x80,
            '\u{2018}' => 0x91,
            '\u{2019}' => 0x92,
            '\u{201C}' => 0x93,
            '\u{201D}' => 0x94,
            '\u{2022}' => 0x95,
            '\u{2013}' => 0x96,
            '\u{2014}' => 0x97,
            c if (c as u32) < 0x20 => continue,
            c if (c as u32) <= 0xFF => c as u32,
            _ => '?' as u32,
        };
        if (0x20..0x7F).contains(&b) {
            out.push(char::from_u32(b).unwrap());
        } else {
            let _ = write!(out, "\\{b:03o}");
        }
    }
    out
}

fn rgb(c: Rgba) -> String {
    format!("{:.3} {:.3} {:.3}", f32::from(c[0]) / 255.0, f32::from(c[1]) / 255.0, f32::from(c[2]) / 255.0)
}

fn segs_pdf(out: &mut String, segs: &[Seg]) {
    for s in segs {
        let _ = match *s {
            Seg::Move(x, y) => writeln!(out, "{x:.2} {y:.2} m"),
            Seg::Line(x, y) => writeln!(out, "{x:.2} {y:.2} l"),
            Seg::Cubic(a, b, c, d, e, f) => writeln!(out, "{a:.2} {b:.2} {c:.2} {d:.2} {e:.2} {f:.2} c"),
            Seg::Close => writeln!(out, "h"),
        };
    }
}

/// A page's content stream, its images, and the standard fonts it uses.
fn page_content<'a>(d: &'a Drawing, fonts_used: &mut [bool; 13]) -> (String, Vec<(u32, u32, &'a [u8])>) {
    let h = d.height.max(1.0);
    let mut content = String::new();
    // y down, as the drawing is
    let _ = writeln!(content, "1 0 0 -1 0 {h:.2} cm");
    let mut images: Vec<(u32, u32, &[u8])> = Vec::new();
    for op in &d.ops {
        match op {
            Op::Path { segs, fill, stroke, even_odd } => {
                segs_pdf(&mut content, segs);
                if let Some(f) = fill {
                    let _ = writeln!(content, "{} rg", rgb(*f));
                }
                if let Some((c, lw)) = stroke {
                    let _ = writeln!(content, "{} RG {lw:.2} w 1 J 1 j", rgb(*c));
                }
                let star = if *even_odd { "*" } else { "" };
                let paint = match (fill.is_some(), stroke.is_some()) {
                    (true, true) => format!("B{star}"),
                    (true, false) => format!("f{star}"),
                    (false, true) => "S".into(),
                    _ => "n".into(),
                };
                let _ = writeln!(content, "{paint}");
            }
            Op::Text { x, y, text, size, color, bold, italic, family, angle, stretch } => {
                let f = pdf_font(*bold, *italic, family);
                fonts_used[f] = true;
                let (s, c) = (angle.to_radians().sin(), angle.to_radians().cos());
                // the text matrix turns y up again (and by the angle)
                let _ = writeln!(
                    content,
                    "BT {} rg /F{f} {size:.2} Tf {:.2} Tz {c:.4} {:.4} {:.4} {:.4} {x:.2} {y:.2} Tm ({}) Tj ET",
                    rgb(*color),
                    stretch * 100.0,
                    -s,
                    -s,
                    -c,
                    // (the Symbol font in its own encoding)
                    if f == 12 { pdf_escape(&crate::symbol::from_unicode(text)) } else { pdf_text(text) }
                );
            }
            Op::Image { x, y, w: iw, h: ih, width, height, rgba } => {
                let _ = writeln!(content, "q {iw:.2} 0 0 {:.2} {x:.2} {:.2} cm /Im{} Do Q", -ih, y + ih, images.len());
                images.push((*width, *height, rgba));
            }
        }
    }
    (content, images)
}

/// A stream object's dictionary and bytes, compressed (`/FlateDecode`).
fn flate_stream(dict: &str, data: &[u8]) -> Vec<u8> {
    let z = miniz_oxide::deflate::compress_to_vec_zlib(data, 6);
    let mut b = format!("<< {dict} /Filter /FlateDecode /Length {} >>\nstream\n", z.len()).into_bytes();
    b.extend_from_slice(&z);
    b.extend_from_slice(b"\nendstream");
    b
}

/// Drawings as the pages of one PDF, each page its drawing's size.
pub fn pdf_pages(pages: &[Drawing]) -> Vec<u8> {
    let mut fonts_used = [false; 13];
    let contents: Vec<(String, Vec<(u32, u32, &[u8])>)> = pages.iter().map(|d| page_content(d, &mut fonts_used)).collect();
    let n = pages.len().max(1);
    let mut pdf: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    let obj = |pdf: &mut Vec<u8>, offsets: &mut Vec<usize>, body: &[u8]| {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", offsets.len()).as_bytes());
        pdf.extend_from_slice(body);
        pdf.extend_from_slice(b"\nendobj\n");
    };
    pdf.extend_from_slice(b"%PDF-1.4\n%\xE2\xE3\xCF\xD3\n");
    // 1 catalog, 2 pages, then each page and its content, the fonts, the images
    let font_ids: Vec<(usize, usize)> = (0..13).filter(|&f| fonts_used[f]).enumerate().map(|(k, f)| (f, 3 + 2 * n + k)).collect();
    let mut next_image = 3 + 2 * n + font_ids.len();
    obj(&mut pdf, &mut offsets, b"<< /Type /Catalog /Pages 2 0 R >>");
    let kids: Vec<String> = (0..n).map(|i| format!("{} 0 R", 3 + 2 * i)).collect();
    obj(&mut pdf, &mut offsets, format!("<< /Type /Pages /Kids [{}] /Count {n} >>", kids.join(" ")).as_bytes());
    let mut image_ids = Vec::new();
    for (i, (content, images)) in contents.iter().enumerate() {
        let d = &pages[i];
        let (w, h) = (d.width.max(1.0), d.height.max(1.0));
        let mut res = String::from("<< /Font << ");
        for (f, id) in &font_ids {
            let _ = write!(res, "/F{f} {id} 0 R ");
        }
        res.push_str(">> /XObject << ");
        for k in 0..images.len() {
            // each image and its soft mask
            let _ = write!(res, "/Im{k} {next_image} 0 R ");
            image_ids.push(images[k]);
            next_image += 2;
        }
        res.push_str(">> >>");
        obj(&mut pdf, &mut offsets, format!("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.2} {h:.2}] /Resources {res} /Contents {} 0 R >>", 4 + 2 * i).as_bytes());
        obj(&mut pdf, &mut offsets, &flate_stream("", content.as_bytes()));
    }
    if pages.is_empty() {
        obj(&mut pdf, &mut offsets, b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>");
        obj(&mut pdf, &mut offsets, b"<< /Length 0 >>\nstream\n\nendstream");
    }
    for (f, _) in &font_ids {
        // Symbol keeps its own encoding
        let enc = if *f == 12 { "" } else { " /Encoding /WinAnsiEncoding" };
        obj(&mut pdf, &mut offsets, format!("<< /Type /Font /Subtype /Type1 /BaseFont /{}{enc} >>", FONTS[*f]).as_bytes());
    }
    let first_image = 3 + 2 * n + font_ids.len();
    for (i, (iw, ih, px)) in image_ids.iter().enumerate() {
        let rgb: Vec<u8> = px.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        let alpha: Vec<u8> = px.chunks_exact(4).map(|p| p[3]).collect();
        let dict = format!("/Type /XObject /Subtype /Image /Width {iw} /Height {ih} /ColorSpace /DeviceRGB /BitsPerComponent 8 /SMask {} 0 R", first_image + 2 * i + 1);
        obj(&mut pdf, &mut offsets, &flate_stream(&dict, &rgb));
        let dict = format!("/Type /XObject /Subtype /Image /Width {iw} /Height {ih} /ColorSpace /DeviceGray /BitsPerComponent 8");
        obj(&mut pdf, &mut offsets, &flate_stream(&dict, &alpha));
    }
    let xref = pdf.len();
    let mut tail = format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len() + 1);
    for o in &offsets {
        let _ = writeln!(tail, "{o:010} 00000 n ");
    }
    let _ = write!(tail, "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n", offsets.len() + 1);
    pdf.extend_from_slice(tail.as_bytes());
    pdf
}

impl Drawing {
    /// One page the drawing's size.
    pub fn to_pdf(&self) -> Vec<u8> {
        pdf_pages(std::slice::from_ref(self))
    }

    /// An SVG the drawing's size (bitmaps as embedded BMPs).
    pub fn to_svg(&self) -> String {
        let mut s = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{w:.2}pt\" height=\"{h:.2}pt\" viewBox=\"0 0 {w:.2} {h:.2}\">\n",
            w = self.width,
            h = self.height
        );
        let hex = |c: Rgba| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]);
        for op in &self.ops {
            match op {
                Op::Path { segs, fill, stroke, even_odd } => {
                    let mut d = String::new();
                    for seg in segs {
                        let _ = match *seg {
                            Seg::Move(x, y) => write!(d, "M{x:.2} {y:.2}"),
                            Seg::Line(x, y) => write!(d, "L{x:.2} {y:.2}"),
                            Seg::Cubic(a, b, c, e, f, g) => write!(d, "C{a:.2} {b:.2} {c:.2} {e:.2} {f:.2} {g:.2}"),
                            Seg::Close => write!(d, "Z"),
                        };
                    }
                    let fill = fill.map_or("none".to_string(), hex);
                    let stroke = stroke.map_or(String::from("stroke=\"none\""), |(c, w)| format!("stroke=\"{}\" stroke-width=\"{w:.2}\"", hex(c)));
                    let rule = if *even_odd { " fill-rule=\"evenodd\"" } else { "" };
                    let _ = writeln!(s, "<path d=\"{d}\" fill=\"{fill}\" {stroke}{rule}/>");
                }
                Op::Text { x, y, text, size, color, bold, italic, family, angle, stretch } => {
                    let esc = text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
                    let _ = writeln!(
                        s,
                        "<text x=\"{x:.2}\" y=\"{y:.2}\" font-family=\"{}\" font-size=\"{size:.2}\" fill=\"{}\"{}{} transform=\"translate({x:.2} {y:.2}) rotate({:.2}) scale({stretch:.3} 1) translate({:.2} {:.2})\">{esc}</text>",
                        if family.eq_ignore_ascii_case("symbol") { "serif".to_string() } else { family.replace('"', "") },
                        hex(*color),
                        if *bold { " font-weight=\"bold\"" } else { "" },
                        if *italic { " font-style=\"italic\"" } else { "" },
                        -angle,
                        -x,
                        -y
                    );
                }
                Op::Image { x, y, w, h, width, height, rgba } => {
                    let _ = writeln!(s, "<image x=\"{x:.2}\" y=\"{y:.2}\" width=\"{w:.2}\" height=\"{h:.2}\" preserveAspectRatio=\"none\" href=\"data:image/bmp;base64,{}\"/>", base64(&bmp(*width, *height, rgba)));
                }
            }
        }
        s.push_str("</svg>\n");
        s
    }
}

/// A 32-bit top-down BMP.
fn bmp(w: u32, h: u32, rgba: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(54 + rgba.len());
    let size = 54 + rgba.len() as u32;
    b.extend_from_slice(b"BM");
    b.extend_from_slice(&size.to_le_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(&54u32.to_le_bytes());
    b.extend_from_slice(&40u32.to_le_bytes());
    b.extend_from_slice(&(w as i32).to_le_bytes());
    b.extend_from_slice(&(-(h as i32)).to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&32u16.to_le_bytes());
    b.extend_from_slice(&[0; 24]);
    for p in rgba.chunks_exact(4) {
        b.extend_from_slice(&[p[2], p[1], p[0], p[3]]);
    }
    b
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len() * 4 / 3 + 4);
    for c in data.chunks(3) {
        let n = u32::from(c[0]) << 16 | u32::from(*c.get(1).unwrap_or(&0)) << 8 | u32::from(*c.get(2).unwrap_or(&0));
        for k in 0..4 {
            if k <= c.len() {
                s.push(T[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}
