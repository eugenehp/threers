use super::ExtrudeGeometry;
use crate::core::{BufferAttribute, BufferGeometry};
use crate::curves::Shape;
use crate::math::Vector3;

pub struct TextGeometry;

/// One glyph's outline + its post-glyph advance.
pub struct Glyph {
    pub shape: Shape,
    /// X advance after this glyph (next glyph's origin offset).
    pub advance: f32,
}

impl TextGeometry {
    /// Lay out a sequence of caller-supplied glyph shapes left-to-right and
    /// extrude them to `depth`. Mirrors three.js's `TextGeometry`. Font
    /// parsing stays out of the crate — the caller (or a downstream FontLoader)
    /// is responsible for producing `Shape`s from glyph outlines.
    pub fn new(glyphs: &[Glyph], depth: f32, curve_segments: usize) -> BufferGeometry {
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut uvs: Vec<f32> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let mut x_offset = 0.0_f32;

        for g in glyphs {
            let glyph_geom = ExtrudeGeometry::new(&g.shape, depth, curve_segments);
            let base_vert = (positions.len() / 3) as u32;
            if let Some(p) = glyph_geom.get_attribute("position") {
                // Translate every position by (x_offset, 0, 0).
                for chunk in p.array.chunks_exact(3) {
                    positions.extend_from_slice(&[chunk[0] + x_offset, chunk[1], chunk[2]]);
                }
            }
            if let Some(n) = glyph_geom.get_attribute("normal") {
                normals.extend_from_slice(&n.array);
            }
            if let Some(u) = glyph_geom.get_attribute("uv") {
                uvs.extend_from_slice(&u.array);
            }
            if let Some(idx) = &glyph_geom.index {
                for i in idx {
                    indices.push(i + base_vert);
                }
            }
            x_offset += g.advance;
        }

        let mut out = BufferGeometry::new();
        out.set_attribute("position", BufferAttribute::new(positions, 3));
        if !normals.is_empty() {
            out.set_attribute("normal", BufferAttribute::new(normals, 3));
        }
        if !uvs.is_empty() {
            out.set_attribute("uv", BufferAttribute::new(uvs, 2));
        }
        if !indices.is_empty() {
            out.set_index(indices);
        }
        let _ = Vector3::ZERO; // silence unused-import lint
        out
    }
}

/// Options of three.js r165's `TextGeometry`: its own defaults on top of
/// [`ExtrudeOptions`](super::ExtrudeOptions)'s.
#[derive(Debug, Clone, PartialEq)]
pub struct TextOptions {
    /// Em size. Default 100 (`Font.generateShapes`'s default).
    pub size: f64,
    /// Extrusion depth. Default 50.
    pub depth: Option<f64>,
    /// The pre-r163 name of `depth`, used when `depth` is not given.
    pub height: Option<f64>,
    pub curve_segments: usize,
    pub steps: usize,
    /// Default false (unlike `ExtrudeGeometry`).
    pub bevel_enabled: bool,
    /// Default 10.
    pub bevel_thickness: f64,
    /// Default 8.
    pub bevel_size: f64,
    pub bevel_offset: f64,
    pub bevel_segments: usize,
}

impl Default for TextOptions {
    fn default() -> Self {
        Self {
            size: 100.0,
            depth: None,
            height: None,
            curve_segments: 12,
            steps: 1,
            bevel_enabled: false,
            bevel_thickness: 10.0,
            bevel_size: 8.0,
            bevel_offset: 0.0,
            bevel_segments: 3,
        }
    }
}

impl TextOptions {
    /// The [`ExtrudeOptions`](super::ExtrudeOptions) three.js hands
    /// `ExtrudeGeometry`.
    pub fn extrude_options(&self) -> super::ExtrudeOptions {
        super::ExtrudeOptions {
            curve_segments: self.curve_segments,
            steps: self.steps,
            depth: self.depth.or(self.height).unwrap_or(50.0),
            bevel_enabled: self.bevel_enabled,
            bevel_thickness: self.bevel_thickness,
            bevel_size: Some(self.bevel_size),
            bevel_offset: self.bevel_offset,
            bevel_segments: self.bevel_segments,
            extrude_path: None,
        }
    }
}

impl TextGeometry {
    /// three.js r165 `new TextGeometry(text, { font, … })`: the font's shapes
    /// for `text` extruded — buffer for buffer what three.js builds.
    pub fn from_font(
        font: &crate::loaders::typeface::Font,
        text: &str,
        options: &TextOptions,
    ) -> BufferGeometry {
        let shapes = font.generate_shapes(text, options.size);
        super::ExtrudeGeometry::from_shapes(&shapes, &options.extrude_options())
    }
}
