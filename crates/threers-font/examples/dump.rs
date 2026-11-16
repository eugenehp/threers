//! Answers the commands of kicad-rs's `tools/font_oracle` in its format,
//! so the two can be compared line by line: open, size, metrics, glyph,
//! char.

use std::io::{BufRead, Write};

use threers_font::{Face, Glyph, Hinter, Size};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    let mut face: Option<Face> = None;
    let mut size: Option<Size> = None;
    let mut hinter: Option<Hinter> = None;
    let config = threers_font::fontconfig::Config::load(std::path::Path::new(
        &std::env::var("FONTCONFIG_FILE").unwrap_or_else(|_| "/opt/homebrew/etc/fonts/fonts.conf".to_string()),
    ));
    let mut fonts: Option<Vec<threers_font::fontconfig::FontEntry>> = None;

    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        let mut words = line.split_whitespace();

        match words.next() {
            Some("open") => {
                // the path, which may have spaces, then the face index
                let rest = line[5..].trim_end();
                let (path, index) = rest.rsplit_once(' ').map_or((rest, 0), |(p, i)| (p, i.parse().unwrap_or(0)));
                face = std::fs::read(path).ok().and_then(|d| Face::new(d, index).ok());

                match &face {
                    Some(f) => writeln!(out, "ok {} glyphs", f.num_glyphs).unwrap(),
                    None => writeln!(out, "error").unwrap(),
                }
            }
            Some("size") => {
                let h: i64 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                let dpi: u32 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                size = face.as_ref().map(|f| Size::char_size(f, 0, h, dpi, 0));
                hinter = match (&face, &size) {
                    (Some(f), Some(s)) => Some(Hinter::new(f, s)),
                    _ => None,
                };
                writeln!(out, "0").unwrap();
            }
            Some("metrics") => {
                if let Some(s) = &size {
                    let m = s.metrics;
                    writeln!(out, "{} {} {} {} {} {} {} {}", m.x_ppem, m.y_ppem, m.x_scale, m.y_scale, m.ascender, m.descender, m.height, m.max_advance).unwrap();
                }
            }
            Some("shape") => {
                if let (Some(f), Some(s)) = (&face, &size) {
                    match threers_font::shape(f, s, &line[6..]) {
                        Ok(glyphs) => {
                            for g in glyphs {
                                writeln!(out, "{} {} {} {} {} {}", g.glyph, g.cluster, g.x_advance, g.y_advance, g.x_offset, g.y_offset).unwrap();
                            }
                        }
                        Err(e) => writeln!(out, "unsupported {}", e.0).unwrap(),
                    }
                }
            }
            Some("fonts") => {
                let list = fonts.get_or_insert_with(|| threers_font::fontconfig::scan(&config.dirs));

                for f in list.iter() {
                    writeln!(
                        out,
                        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|0|{}",
                        f.path.display(),
                        f.index,
                        f.families.join(";"),
                        f.style,
                        if f.weight_range.is_some() { -1.0 } else { f.weight },
                        f.slant,
                        if f.width_range.is_some() { -1.0 } else { f.width },
                        if f.english { 0 } else { 2 },
                        f.version,
                        u8::from(f.symbol),
                        u8::from(f.variable),
                        u8::from(f.decorative)
                    )
                    .unwrap();
                }
            }
            Some("match") => {
                let mut w = line[6..].splitn(3, ' ');
                let bold = w.next() == Some("1");
                let italic = w.next() == Some("1");
                let family = w.next().unwrap_or("");
                let mut p = threers_font::fontconfig::find_pattern(family, bold, italic);
                config.substitute(&mut p);
                threers_font::fontconfig::default_substitute(&mut p, "en");
                write!(out, "families:").unwrap();

                for f in p.families() {
                    write!(out, " [{f}]").unwrap();
                }

                writeln!(out).unwrap();
                writeln!(out, "lang en").unwrap();
                let list = fonts.get_or_insert_with(|| threers_font::fontconfig::scan(&config.dirs));

                if let Some(f) = threers_font::fontconfig::font_match(list, &p) {
                    writeln!(out, "font {}|{}|{}|{}", f.path.display(), f.index, f.families.first().map_or("", String::as_str), f.style).unwrap();
                }
            }
            Some("decompose") => {
                let gid: u16 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                let bold = words.next() == Some("1");

                if let (Some(f), Some(s), Some(h)) = (&face, &size, hinter.as_mut()) {
                    let loaded = Glyph::load_hinted(f, s, h, gid).ok().map(|mut g| {
                        if bold {
                            g.outline.embolden(1 << 6);
                        }
                        g
                    });

                    if let Some(segments) = loaded.and_then(|g| g.outline.decompose()) {
                        for seg in segments {
                            match seg {
                                threers_font::glyph::Segment::MoveTo(p) => writeln!(out, "M {} {}", p.0, p.1).unwrap(),
                                threers_font::glyph::Segment::LineTo(p) => writeln!(out, "L {} {}", p.0, p.1).unwrap(),
                                threers_font::glyph::Segment::ConicTo(c, p) => writeln!(out, "Q {} {} {} {}", c.0, c.1, p.0, p.1).unwrap(),
                            }
                        }
                    }
                }
            }
            Some("char") => {
                let c: u32 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                writeln!(out, "{}", face.as_ref().map_or(0, |f| f.glyph_index(c))).unwrap();
            }
            Some("glyph") => {
                let gid: u16 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);
                let flags: i32 = words.next().and_then(|w| w.parse().ok()).unwrap_or(0);

                match (&face, &size) {
                    (Some(f), Some(s)) => match if flags & 2 == 0 { Glyph::load_hinted(f, s, hinter.as_mut().expect("a size"), gid) } else { Glyph::load(f, s, gid) } {
                        Ok(g) => {
                            writeln!(out, "advance {} 0 lsb {} format 1869968492", g.advance, g.left_bearing).unwrap();

                            for (p, t) in g.outline.points.iter().zip(&g.outline.tags) {
                                writeln!(out, "{} {} {}", p.0, p.1, t).unwrap();
                            }

                            write!(out, "contours").unwrap();

                            for c in &g.outline.contours {
                                write!(out, " {c}").unwrap();
                            }

                            writeln!(out).unwrap();
                            writeln!(out, "orientation {}", g.outline.orientation() as i32).unwrap();
                        }
                        Err(e) => writeln!(out, "error {e}").unwrap(),
                    },
                    _ => writeln!(out, "error no face").unwrap(),
                }
            }
            _ => {}
        }

        writeln!(out, "end").unwrap();
        out.flush().unwrap();
    }
}
