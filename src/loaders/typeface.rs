//! typeface.js fonts — the JSON `FontLoader` reads and `TextGeometry` draws
//! with — and three.js r165's `Font.generateShapes`.
//!
//! A glyph's outline is a string of commands (`m x y`, `l x y`,
//! `q cpx cpy x y` with the end point first, `b …` likewise) in font units;
//! `resolution` is the em size. Glyphs are laid out left to right by their
//! advance `ha`, and a newline drops by the bounding box height plus the
//! underline thickness — exactly as three.js does, so the shapes match.

use super::json;
use crate::curves::shapes::{Shape, ShapePath};
use std::collections::HashMap;

/// One glyph: its advance and its outline commands, split on single spaces as
/// three.js splits them.
#[derive(Debug, Clone, PartialEq)]
pub struct TypefaceGlyph {
    pub ha: f64,
    pub outline: Vec<String>,
}

/// A typeface.js font (`FontLoader.parse(json)` / `new Font(json)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Font {
    pub family_name: String,
    pub resolution: f64,
    pub bounding_box_y_min: f64,
    pub bounding_box_y_max: f64,
    pub underline_thickness: f64,
    pub glyphs: HashMap<String, TypefaceGlyph>,
}

/// JavaScript's `Number(string)` for the tokens of an outline string.
fn js_number(s: &str) -> f64 {
    let t = s.trim();
    if t.is_empty() {
        return 0.0;
    }
    t.parse::<f64>().unwrap_or(f64::NAN)
}

fn num(v: Option<&json::Value>) -> f64 {
    match v {
        Some(json::Value::Number(n)) => *n,
        Some(json::Value::String(s)) => js_number(s),
        _ => f64::NAN,
    }
}

impl Font {
    /// Read typeface.js JSON.
    pub fn parse(src: &str) -> Result<Font, String> {
        let root = json::parse(src).map_err(|e| format!("typeface JSON: {e}"))?;
        Self::from_json(&root)
    }

    fn from_json(root: &json::Value) -> Result<Font, String> {
        let glyphs_v = root
            .get("glyphs")
            .and_then(|g| g.as_object())
            .ok_or("typeface JSON: no `glyphs`")?;
        let mut glyphs = HashMap::with_capacity(glyphs_v.len());
        for (ch, g) in glyphs_v {
            let outline = g
                .get("o")
                .and_then(|o| o.as_str())
                .map(|o| o.split(' ').map(str::to_owned).collect())
                .unwrap_or_default();
            glyphs.insert(
                ch.clone(),
                TypefaceGlyph {
                    ha: num(g.get("ha")),
                    outline,
                },
            );
        }
        let bb = root.get("boundingBox");
        Ok(Font {
            family_name: root
                .get("familyName")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
            resolution: num(root.get("resolution")),
            bounding_box_y_min: num(bb.and_then(|b| b.get("yMin"))),
            bounding_box_y_max: num(bb.and_then(|b| b.get("yMax"))),
            underline_thickness: num(root.get("underlineThickness")),
            glyphs,
        })
    }

    /// One `ShapePath` per drawn character (three.js's `createPaths`).
    /// Characters with no glyph fall back to `?`; with neither, they are
    /// skipped (three.js logs an error and then throws).
    pub fn create_paths(&self, text: &str, size: f64) -> Vec<ShapePath> {
        let scale = size / self.resolution;
        let line_height =
            (self.bounding_box_y_max - self.bounding_box_y_min + self.underline_thickness) * scale;
        let mut paths = Vec::new();
        let mut offset_x = 0.0;
        let mut offset_y = 0.0;
        let mut buf = [0u8; 4];
        for ch in text.chars() {
            if ch == '\n' {
                offset_x = 0.0;
                offset_y -= line_height;
                continue;
            }
            let key: &str = ch.encode_utf8(&mut buf);
            let Some(glyph) = self.glyphs.get(key).or_else(|| self.glyphs.get("?")) else {
                log::error!(
                    "Font: character \"{ch}\" does not exist in font family {}",
                    self.family_name
                );
                continue;
            };
            paths.push(glyph_path(glyph, scale, offset_x, offset_y));
            offset_x += glyph.ha * scale;
        }
        paths
    }

    /// `font.generateShapes(text, size)`: every character's `toShapes()`.
    pub fn generate_shapes(&self, text: &str, size: f64) -> Vec<Shape> {
        self.create_paths(text, size)
            .iter()
            .flat_map(|p| p.to_shapes(false))
            .collect()
    }
}

fn glyph_path(glyph: &TypefaceGlyph, scale: f64, ox: f64, oy: f64) -> ShapePath {
    let mut path = ShapePath::new();
    let o = &glyph.outline;
    let mut i = 0;
    let next = |i: &mut usize| -> f64 {
        let v = o.get(*i).map_or(f64::NAN, |s| js_number(s));
        *i += 1;
        v
    };
    while i < o.len() {
        let action = o[i].as_str();
        i += 1;
        match action {
            "m" => {
                let x = next(&mut i) * scale + ox;
                let y = next(&mut i) * scale + oy;
                path.move_to(x, y);
            }
            "l" => {
                let x = next(&mut i) * scale + ox;
                let y = next(&mut i) * scale + oy;
                path.line_to(x, y);
            }
            "q" => {
                let cpx = next(&mut i) * scale + ox;
                let cpy = next(&mut i) * scale + oy;
                let cpx1 = next(&mut i) * scale + ox;
                let cpy1 = next(&mut i) * scale + oy;
                path.quadratic_curve_to(cpx1, cpy1, cpx, cpy);
            }
            "b" => {
                let cpx = next(&mut i) * scale + ox;
                let cpy = next(&mut i) * scale + oy;
                let cpx1 = next(&mut i) * scale + ox;
                let cpy1 = next(&mut i) * scale + oy;
                let cpx2 = next(&mut i) * scale + ox;
                let cpy2 = next(&mut i) * scale + oy;
                path.bezier_curve_to(cpx1, cpy1, cpx2, cpy2, cpx, cpy);
            }
            _ => {}
        }
    }
    path
}
