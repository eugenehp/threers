//! Text anchored in the world, drawn facing the camera at a fixed pixel size.

use super::{
    Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, MarkView, PrepareContext,
};
use crate::captions::CaptionFont;
use std::collections::HashMap;

/// Pixels per em glyphs are rasterized at by default before conversion to a
/// distance field ([`TextLabels::raster_size`]). The field is
/// resolution-independent within a few times this either way, which covers
/// label sizes from about 8 to 96 pixels.
const RASTER: f32 = 48.0;
const ATLAS_WIDTH: u32 = 1024;

/// How far the distance field reaches outside and inside the outline, in
/// raster pixels, for glyphs rasterized at `raster` pixels per em: a sixth of
/// an em. It bounds the widest halo.
fn spread_for(raster: f32) -> i32 {
    ((raster / 6.0).round() as i32).max(2)
}

/// What the glyph atlas holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GlyphField {
    /// One signed distance per texel: rounds corners a little when text is
    /// drawn much larger than [`TextLabels::raster_size`].
    #[default]
    Sdf,
    /// Multi-channel signed distance fields from the outlines: corners stay
    /// sharp at any size, so a smaller raster serves. Needs a TrueType face;
    /// glyphs from the bitmap face fall back to a single distance.
    Msdf,
}

/// An icon drawn beside a label's text, from images added with
/// [`TextLabels::add_icon`] — together one symbol, placed and culled as one.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelIcon {
    pub name: String,
    /// Height in pixels; the width follows the image.
    pub size: f32,
    /// sRGB, straight alpha: icons are masks, drawn in this colour.
    pub color: [u8; 4],
    /// Pixels between the icon and the text.
    pub gap: f32,
}

impl LabelIcon {
    pub fn new(name: impl Into<String>, size: f32) -> Self {
        Self {
            name: name.into(),
            size,
            color: [255, 255, 255, 255],
            gap: 4.0,
        }
    }
}

/// Where the text sits relative to its anchor point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LabelAnchor {
    /// Centred on the point.
    #[default]
    Center,
    /// To the right of the point, vertically centred.
    Right,
    /// To the left of the point.
    Left,
    /// Above the point, horizontally centred.
    Above,
    /// Below the point.
    Below,
}

/// One label.
#[derive(Debug, Clone, PartialEq)]
pub struct TextLabel {
    pub position: [f64; 3],
    pub text: String,
    /// sRGB, straight alpha.
    pub color: [u8; 4],
    /// Labels that collide keep the one with the higher priority.
    pub priority: f32,
    pub anchor: LabelAnchor,
    /// Extra offset in pixels, x right and y down.
    pub offset: [f32; 2],
    /// Set the text along this line instead of at `position` — a road's or
    /// a river's name. Each glyph follows the line's direction on screen,
    /// centred on its length, reading left to right whichever way the line
    /// was drawn. Hidden when the line is shorter on screen than the text or
    /// bends more than [`TextLabels::max_bend`] between two glyphs.
    pub path: Option<Vec<[f64; 3]>>,
    /// An icon to the left of the text. Ignored along a path.
    pub icon: Option<LabelIcon>,
}

impl TextLabel {
    pub fn new(position: [f64; 3], text: impl Into<String>) -> Self {
        Self {
            position,
            text: text.into(),
            color: [255, 255, 255, 255],
            priority: 0.0,
            anchor: LabelAnchor::Center,
            offset: [0.0, 0.0],
            path: None,
            icon: None,
        }
    }

    /// With an icon to the left of the text.
    pub fn with_icon(mut self, icon: LabelIcon) -> Self {
        self.icon = Some(icon);
        self
    }

    /// A label set along `path`.
    pub fn along(path: Vec<[f64; 3]>, text: impl Into<String>) -> Self {
        let mut l = Self::new(path.first().copied().unwrap_or_default(), text);
        l.path = Some(path);
        l
    }
}

/// Where a path label's glyph sits: its distance along the text's baseline
/// to its centre, and its rect relative to that centre on the line.
#[derive(Debug, Clone, Copy)]
struct PathGlyph {
    along: f32,
    rect: [f32; 4],
}

/// Where a glyph's field sits in the atlas and how it is placed at its
/// raster size: (x, y, w, h, offset_x, offset_y), offsets including the spread.
type GlyphSlot = (u32, u32, u32, u32, i32, i32);

/// What collision was last run for: view-projection, viewport, origin, and
/// the settings that size or choose boxes (padding, collision, stickiness,
/// max bend), as bits.
type ViewKey = ([f32; 16], (u32, u32), [f64; 3], [u32; 4]);

/// One glyph or icon of a label's layout: rect, uv, colour, halo scale.
type Quad = ([f32; 4], [f32; 4], [u8; 4], f32);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GlyphKey {
    /// A character from font `n` of the chain.
    Char(u8, char),
    Icon(String),
}

/// Glyphs packed into one RGBA texture as distance fields: three channels
/// multi-channel or the same distance, the fourth the true distance.
struct Atlas {
    raster: f32,
    spread: i32,
    msdf: bool,
    height: u32,
    pixels: Vec<u8>,
    /// `None` for glyphs with no ink.
    glyphs: HashMap<GlyphKey, Option<GlyphSlot>>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
    dirty: bool,
}

impl Atlas {
    fn new(raster: f32, msdf: bool) -> Self {
        Self {
            raster,
            spread: spread_for(raster),
            msdf,
            height: 128,
            pixels: vec![0; (ATLAS_WIDTH * 128 * 4) as usize],
            glyphs: HashMap::new(),
            shelf_x: 0,
            shelf_y: 0,
            shelf_h: 0,
            dirty: true,
        }
    }

    /// Shelf-pack a `w × h` field; its corner. Grows the atlas as needed.
    fn pack(&mut self, w: u32, h: u32, texels: &[[u8; 4]]) -> (u32, u32) {
        // One pixel of gutter so linear filtering never reads a neighbour.
        if self.shelf_x + w + 1 > ATLAS_WIDTH {
            self.shelf_y += self.shelf_h + 1;
            self.shelf_x = 0;
            self.shelf_h = 0;
        }
        while self.shelf_y + h + 1 > self.height {
            self.height *= 2;
            self.pixels
                .resize((ATLAS_WIDTH * self.height * 4) as usize, 0);
        }
        let (x, y) = (self.shelf_x, self.shelf_y);
        for row in 0..h {
            let dst = (((y + row) * ATLAS_WIDTH + x) * 4) as usize;
            let src = &texels[(row * w) as usize..((row + 1) * w) as usize];
            self.pixels[dst..dst + 4 * w as usize].copy_from_slice(bytemuck::cast_slice(src));
        }
        self.shelf_x += w + 1;
        self.shelf_h = self.shelf_h.max(h);
        self.dirty = true;
        (x, y)
    }

    /// Character `c` of `font` (number `index` in the chain), adding it on
    /// first sight.
    fn ensure(&mut self, font: &mut CaptionFont, index: u8, c: char) {
        let key = GlyphKey::Char(index, c);
        if self.glyphs.contains_key(&key) {
            return;
        }
        let spread = self.spread;
        let field = if self.msdf {
            font.outline(c, self.raster)
                .and_then(|o| super::msdf::msdf(&o, spread))
        } else {
            None
        };
        let slot = match field {
            Some(f) => {
                let (x, y) = self.pack(f.width, f.height, &f.texels);
                Some((x, y, f.width, f.height, f.offset_x, f.offset_y))
            }
            None => {
                let g = font.glyph(c, self.raster).clone();
                if g.is_blank() {
                    None
                } else {
                    let w = g.width + 2 * spread as u32;
                    let h = g.height + 2 * spread as u32;
                    let texels: Vec<[u8; 4]> =
                        distance_field(&g.coverage, g.width, g.height, spread)
                            .into_iter()
                            .map(|v| [v; 4])
                            .collect();
                    let (x, y) = self.pack(w, h, &texels);
                    Some((x, y, w, h, g.offset_x - spread, g.offset_y - spread))
                }
            }
        };
        self.glyphs.insert(key, slot);
    }

    /// An icon from its alpha mask, adding it on first sight.
    fn ensure_icon(&mut self, name: &str, w: u32, h: u32, alpha: &[u8]) {
        let key = GlyphKey::Icon(name.to_owned());
        if self.glyphs.contains_key(&key) {
            return;
        }
        let spread = self.spread;
        let texels: Vec<[u8; 4]> = distance_field(alpha, w, h, spread)
            .into_iter()
            .map(|v| [v; 4])
            .collect();
        let (fw, fh) = (w + 2 * spread as u32, h + 2 * spread as u32);
        let (x, y) = self.pack(fw, fh, &texels);
        self.glyphs
            .insert(key, Some((x, y, fw, fh, -spread, -spread)));
    }
}

/// Squared distance to the nearest `true` cell along one row — Felzenszwalb
/// and Huttenlocher's lower envelope of parabolas.
fn edt_1d(f: &[f32], out: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32))
                / (2.0 * q as f32 - 2.0 * p as f32);
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                // k == 0 and the new parabola dominates everywhere.
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
            break;
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let d = q as f32 - p as f32;
        *o = d * d + f[p];
    }
}

/// Squared Euclidean distance from every cell to the nearest set cell.
fn edt_2d(set: &[bool], w: usize, h: usize) -> Vec<f32> {
    const BIG: f32 = 1e20;
    let mut grid: Vec<f32> = set.iter().map(|&s| if s { 0.0 } else { BIG }).collect();
    let n = w.max(h);
    let mut f = vec![0.0; n];
    let mut out = vec![0.0; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0; n + 1];
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        edt_1d(&f[..h], &mut out[..h], &mut v, &mut z);
        for y in 0..h {
            grid[y * w + x] = out[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..y * w + w]);
        edt_1d(&f[..w], &mut out[..w], &mut v, &mut z);
        grid[y * w..y * w + w].copy_from_slice(&out[..w]);
    }
    grid
}

/// An 8-bit coverage mask as a padded signed distance field: 128 on the
/// outline, more inside, `spread` pixels out reaching 0.
fn distance_field(coverage: &[u8], w: u32, h: u32, spread: i32) -> Vec<u8> {
    let pw = (w as i32 + 2 * spread) as usize;
    let ph = (h as i32 + 2 * spread) as usize;
    let mut inside = vec![false; pw * ph];
    for y in 0..h as usize {
        for x in 0..w as usize {
            inside[(y + spread as usize) * pw + x + spread as usize] =
                coverage[y * w as usize + x] >= 128;
        }
    }
    let outside: Vec<bool> = inside.iter().map(|&b| !b).collect();
    let to_inside = edt_2d(&inside, pw, ph);
    let to_outside = edt_2d(&outside, pw, ph);
    to_inside
        .iter()
        .zip(&to_outside)
        .map(|(&di, &dout)| {
            // Positive outside, negative inside; a half-pixel shift puts the
            // outline between the last inside and first outside cell.
            let d = if di > 0.0 {
                di.sqrt() - 0.5
            } else {
                -(dout.sqrt() - 0.5)
            };
            let v = 0.5 - d / (2.0 * spread as f32);
            (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
        })
        .collect()
}

/// Labels: text at points in the world, always facing the camera and always
/// `font_size` pixels tall, with overlapping labels culled by priority.
///
/// Glyphs are signed distance fields in a shared atlas, so one rasterization
/// serves every size and the edge stays sharp under the pixel snapping the
/// shader does. Any [`CaptionFont`] works; the built-in one is an 8×8 bitmap
/// face and looks it, so give it a TrueType face for real text.
///
/// Collision runs on the CPU when the view changes: labels are projected,
/// sorted by priority and placed greedily, each kept only if its box clears
/// every box already placed. Labels do not depth-test by default — a label is
/// an annotation, not a surface — set `style.depth_test` to hide them behind
/// geometry.
pub struct TextLabels {
    pub common: MarkCommon,
    /// The face, then its fallbacks in order: a character is drawn from the
    /// first that has it.
    fonts: Vec<CaptionFont>,
    labels: Vec<TextLabel>,
    /// Pixels per em glyphs are rasterized at before becoming distance
    /// fields: larger is sharper for large text and costs atlas space.
    /// 48 by default; [`GlyphField::Msdf`] stays sharp from about 32.
    pub raster_size: f32,
    /// Single or multi-channel distance fields.
    pub glyph_field: GlyphField,
    /// Alpha masks added with [`add_icon`](Self::add_icon).
    icons: HashMap<String, (u32, u32, Vec<u8>)>,
    /// Text height in pixels: one em.
    pub font_size: f32,
    /// Outline around the text, in pixels, for legibility over busy maps.
    /// At most `font_size / 6`.
    pub halo_width: f32,
    pub halo_color: [u8; 4],
    /// Cull overlapping labels.
    pub collision: bool,
    /// Extra space kept clear around each label, in pixels.
    pub padding: f32,
    /// Collide against this index instead of a private one, so labels and
    /// icons of every layer that shares it keep clear of each other. Start
    /// each frame with [`CollisionIndex::begin_frame`](super::CollisionIndex::begin_frame).
    pub collision_index: Option<super::SharedCollision>,
    /// Added to the priority of a label that showed last time it was
    /// placed, so a view that moves a little keeps its labels instead of
    /// swapping near-equals back and forth. 0 places on priority alone.
    pub stickiness: f32,
    /// The sharpest turn, in radians, a path label may take between two
    /// neighbouring glyphs.
    pub max_bend: f32,
    /// Per label: its glyphs' places along the line, for path labels, and
    /// the text's length in pixels.
    path_layout: Vec<Option<(Vec<PathGlyph>, f32)>>,
    atlas: Atlas,
    atlas_gpu: Option<(u32, wgpu::Texture, wgpu::BindGroup)>,
    /// Per label: glyph rows `start..end`, and the box around the text in
    /// pixels relative to the anchor (x0, y0, x1, y1).
    layout: Vec<(usize, usize, [f32; 4])>,
    /// Per glyph row: its field's texel size relative to a text glyph's, so
    /// the halo is as wide around an icon as around a letter.
    halo_scale: Vec<f32>,
    layout_dirty: bool,
    /// The `font_size` the layout was made at.
    laid_out_size: f32,
    visible: Vec<bool>,
    last_view: Option<ViewKey>,
    /// The shared index's generation this layer last placed into.
    placed_generation: Option<u64>,
    /// Marks this layer's boxes in a shared index.
    owner: u64,
    anchors: Column,
    rects: Column,
    uvs: Column,
    colors: Column,
    states: Column,
}

impl std::fmt::Debug for TextLabels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextLabels")
            .field("labels", &self.labels.len())
            .field("font_size", &self.font_size)
            .finish()
    }
}

impl TextLabels {
    pub fn new(font: CaptionFont) -> Self {
        let mut common = MarkCommon::default();
        common.style.depth_test = false;
        Self {
            common,
            fonts: vec![font],
            labels: Vec::new(),
            raster_size: RASTER,
            glyph_field: GlyphField::Sdf,
            icons: HashMap::new(),
            halo_scale: Vec::new(),
            font_size: 14.0,
            halo_width: 0.0,
            halo_color: [0, 0, 0, 200],
            collision: true,
            padding: 2.0,
            atlas: Atlas::new(RASTER, false),
            atlas_gpu: None,
            layout: Vec::new(),
            layout_dirty: true,
            laid_out_size: 0.0,
            visible: Vec::new(),
            last_view: None,
            placed_generation: None,
            owner: super::collision::next_owner(),
            collision_index: None,
            stickiness: 0.0,
            max_bend: 0.8,
            path_layout: Vec::new(),
            anchors: Column::new(ColumnFormat::PositionF64),
            rects: Column::float(4),
            uvs: Column::float(4),
            colors: Column::new(ColumnFormat::Unorm8x4),
            states: Column::float(4),
        }
    }

    pub fn set_labels(&mut self, labels: Vec<TextLabel>) {
        self.labels = labels;
        self.layout_dirty = true;
    }

    pub fn labels(&self) -> &[TextLabel] {
        &self.labels
    }

    /// Draw characters the face lacks from `font` — and from further
    /// fallbacks, in the order added. Kerning applies within one face.
    pub fn add_fallback(&mut self, font: CaptionFont) {
        self.fonts.push(font);
        self.reset_atlas();
    }

    /// An icon for [`TextLabel::icon`]: a `width × height` alpha mask, one
    /// byte a pixel, 128 and over inside. Kept as a distance field, so it
    /// scales, takes the label's halo, and is drawn in the icon's colour.
    pub fn add_icon(&mut self, name: impl Into<String>, width: u32, height: u32, alpha: &[u8]) {
        assert_eq!(alpha.len(), (width * height) as usize, "one byte a pixel");
        let name = name.into();
        self.atlas.glyphs.remove(&GlyphKey::Icon(name.clone()));
        self.icons.insert(name, (width, height, alpha.to_vec()));
        self.layout_dirty = true;
    }

    /// Start the glyph atlas over: the next layout fills it again.
    fn reset_atlas(&mut self) {
        self.atlas = Atlas::new(
            self.raster_size.max(4.0),
            self.glyph_field == GlyphField::Msdf,
        );
        self.layout_dirty = true;
    }

    /// The face in the chain that draws `c`: the first that has it, else
    /// the first.
    fn font_for(&self, c: char) -> u8 {
        self.fonts.iter().position(|f| f.has_glyph(c)).unwrap_or(0) as u8
    }

    /// Which labels survived collision in the last frame drawn.
    pub fn visible(&self) -> &[bool] {
        &self.visible
    }

    fn relayout(&mut self) {
        let raster = self.atlas.raster;
        let s = self.font_size / raster;
        let metrics = self.fonts[0].metrics(raster);
        let (ascent, descent) = (metrics.ascent * s, metrics.descent * s);
        // Pass 1: make sure every glyph is in the atlas, so uvs are final.
        for li in 0..self.labels.len() {
            let chars: Vec<char> = self.labels[li].text.chars().collect();
            for c in chars {
                let f = self.font_for(c);
                self.atlas.ensure(&mut self.fonts[f as usize], f, c);
            }
            if let Some(icon) = &self.labels[li].icon {
                if let Some((w, h, alpha)) = self.icons.get(&icon.name) {
                    self.atlas.ensure_icon(&icon.name, *w, *h, alpha);
                }
            }
        }
        let ah = self.atlas.height as f32;
        let aw = ATLAS_WIDTH as f32;
        let uv_of = |x: u32, y: u32, w: u32, h: u32| {
            [
                x as f32 / aw,
                y as f32 / ah,
                (x + w) as f32 / aw,
                (y + h) as f32 / ah,
            ]
        };
        let mut anchors = Vec::new();
        let mut rects = Vec::new();
        let mut uvs = Vec::new();
        let mut colors = Vec::new();
        self.layout.clear();
        self.path_layout.clear();
        self.halo_scale.clear();
        for l in &self.labels {
            let start = anchors.len();
            let mut quads: Vec<Quad> = Vec::new();
            let mut pen = 0.0f32;
            // The icon first, its middle on the text's.
            let icon = l
                .icon
                .as_ref()
                .filter(|_| l.path.is_none())
                .and_then(|icon| {
                    let (iw, ih, _) = self.icons.get(&icon.name)?;
                    let slot = self.atlas.glyphs.get(&GlyphKey::Icon(icon.name.clone()))?;
                    Some((icon, *iw, *ih, (*slot)?))
                });
            if let Some((icon, iw, ih, (x, y, w, h, ox, oy))) = icon {
                let k = icon.size / ih.max(1) as f32;
                let mid = (descent - ascent) * 0.5;
                quads.push((
                    [
                        ox as f32 * k,
                        mid - icon.size * 0.5 + oy as f32 * k,
                        w as f32 * k,
                        h as f32 * k,
                    ],
                    uv_of(x, y, w, h),
                    icon.color,
                    k / s,
                ));
                pen = (iw as f32 * k + icon.gap) / s;
            }
            let mut prev: Option<(u8, char)> = None;
            for c in l.text.chars() {
                let f = self.font_for(c);
                let font = &mut self.fonts[f as usize];
                if let Some((pf, p)) = prev {
                    if pf == f {
                        pen += font.kerning(p, c, raster);
                    }
                }
                if let Some(Some((x, y, w, h, ox, oy))) =
                    self.atlas.glyphs.get(&GlyphKey::Char(f, c)).copied()
                {
                    quads.push((
                        [
                            (pen + ox as f32) * s,
                            oy as f32 * s,
                            w as f32 * s,
                            h as f32 * s,
                        ],
                        uv_of(x, y, w, h),
                        l.color,
                        1.0,
                    ));
                }
                pen += font.advance(c, raster);
                prev = Some((f, c));
            }
            let width = pen * s;
            let (dx, dy) = match l.anchor {
                LabelAnchor::Center => (-width * 0.5, (ascent - descent) * 0.5),
                LabelAnchor::Right => (0.0, (ascent - descent) * 0.5),
                LabelAnchor::Left => (-width, (ascent - descent) * 0.5),
                LabelAnchor::Above => (-width * 0.5, -descent),
                LabelAnchor::Below => (-width * 0.5, ascent),
            };
            let (dx, dy) = (dx + l.offset[0], dy + l.offset[1]);
            // The box: the text's line, and the icon where it is taller.
            let (mut top, mut bottom) = (dy - ascent, dy + descent);
            if let Some((icon, _, _, _)) = icon {
                let mid = dy + (descent - ascent) * 0.5;
                top = top.min(mid - icon.size * 0.5);
                bottom = bottom.max(mid + icon.size * 0.5);
            }
            if l.path.is_some() {
                // Each glyph about its own centre on the baseline, the text
                // centred vertically on the line; placement moves and turns
                // them along it.
                let mid = (ascent - descent) * 0.5;
                let mut glyphs = Vec::with_capacity(quads.len());
                for (r, uv, color, halo) in &quads {
                    let centre = r[0] + r[2] * 0.5;
                    glyphs.push(PathGlyph {
                        along: centre,
                        rect: [r[0] - centre, r[1] + mid, r[2], r[3]],
                    });
                    anchors.push(l.position);
                    rects.extend_from_slice(&[r[0] - centre, r[1] + mid, r[2], r[3]]);
                    uvs.extend_from_slice(uv);
                    colors.push(*color);
                    self.halo_scale.push(*halo);
                }
                self.path_layout.push(Some((glyphs, width)));
            } else {
                for (r, uv, color, halo) in quads {
                    anchors.push(l.position);
                    rects.extend_from_slice(&[r[0] + dx, r[1] + dy, r[2], r[3]]);
                    uvs.extend_from_slice(&uv);
                    colors.push(color);
                    self.halo_scale.push(halo);
                }
                self.path_layout.push(None);
            }
            self.layout
                .push((start, anchors.len(), [dx, top, dx + width, bottom]));
        }
        self.anchors.set_positions(&anchors);
        self.rects.set_f32(&rects);
        self.uvs.set_f32(&uvs);
        self.colors.set_rgba(&colors);
        // Keep the last placement when the labels are the same in number, so
        // `stickiness` still knows which ones were showing.
        if self.visible.len() != self.labels.len() {
            self.visible = vec![true; self.labels.len()];
        }
        self.last_view = None;
        self.layout_dirty = false;
        self.laid_out_size = self.font_size;
    }

    /// Decide which labels show in `view`. Public so placement can be tested
    /// and inspected without a GPU.
    pub fn place(&mut self, view: &MarkView) {
        if self.atlas.raster != self.raster_size.max(4.0)
            || self.atlas.msdf != (self.glyph_field == GlyphField::Msdf)
        {
            self.reset_atlas();
        }
        if self.layout_dirty || self.font_size != self.laid_out_size {
            self.relayout();
            // New labels: whatever was placed for the old ones is stale.
            self.last_view = None;
        }
        let settings = [
            self.padding.to_bits(),
            u32::from(self.collision),
            self.stickiness.to_bits(),
            self.max_bend.to_bits(),
        ];
        let key = (
            view.view_proj,
            view.viewport,
            view.origin.position,
            settings,
        );
        let shared = self.collision_index.clone();
        let generation = shared
            .as_ref()
            .map_or(0, |s| s.lock().unwrap().generation());
        // A cleared shared index needs this layer's boxes back even when
        // nothing here moved.
        if self.last_view == Some(key)
            && self.states.len() == self.anchors.len()
            && Some(generation) == self.placed_generation
        {
            return;
        }
        if let Some(s) = &shared {
            s.lock()
                .unwrap()
                .reclaim(self.owner, self.placed_generation);
        }
        self.last_view = Some(key);
        self.placed_generation = Some(generation);
        let n = self.labels.len();
        let (vw, vh) = (view.viewport.0 as f32, view.viewport.1 as f32);
        let was = std::mem::take(&mut self.visible);
        let score = |i: usize| {
            let bonus = if was.get(i).copied().unwrap_or(false) {
                self.stickiness
            } else {
                0.0
            };
            self.labels[i].priority + bonus
        };
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            score(b)
                .partial_cmp(&score(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut private = super::CollisionIndex::new();
        let mut shared_guard = shared.as_ref().map(|s| s.lock().unwrap());
        let index: &mut super::CollisionIndex = match shared_guard.as_deref_mut() {
            Some(g) => g,
            None => &mut private,
        };
        self.visible = vec![false; n];
        let pad = self.padding;
        let owner = self.owner;
        let mut angles = vec![0.0f32; self.anchors.len()];
        let mut moved: Vec<(usize, [f64; 3])> = Vec::new();
        for i in order {
            if let Some((glyphs, width)) = self.path_layout.get(i).and_then(|p| p.as_ref()) {
                let path = self.labels[i].path.as_deref().unwrap_or(&[]);
                let start = self.layout[i].0;
                let Some(placed) = place_on_path(view, path, glyphs, *width, self.max_bend) else {
                    continue;
                };
                let boxes: Vec<[f32; 4]> = placed
                    .iter()
                    .zip(glyphs)
                    .map(|(&(p, a, _), g)| {
                        let (sn, cs) = a.sin_cos();
                        let (hw, hh) = (g.rect[2] * 0.5, g.rect[3] * 0.5);
                        let c = [g.rect[0] + hw, g.rect[1] + hh];
                        let cx = p[0] + c[0] * cs - c[1] * sn;
                        let cy = p[1] + c[0] * sn + c[1] * cs;
                        let ex = hw * cs.abs() + hh * sn.abs() + pad;
                        let ey = hw * sn.abs() + hh * cs.abs() + pad;
                        [cx - ex, cy - ey, cx + ex, cy + ey]
                    })
                    .collect();
                if boxes
                    .iter()
                    .any(|r| r[2] < 0.0 || r[3] < 0.0 || r[0] > vw || r[1] > vh)
                {
                    continue;
                }
                if self.collision {
                    if boxes.iter().any(|r| index.collides(*r)) {
                        continue;
                    }
                    for r in &boxes {
                        index.insert_owned(owner, *r);
                    }
                }
                for (k, &(_, a, w)) in placed.iter().enumerate() {
                    angles[start + k] = a;
                    moved.push((start + k, w));
                }
                self.visible[i] = true;
                continue;
            }
            let Some(p) = view.project(self.labels[i].position) else {
                continue;
            };
            let b = self.layout[i].2;
            let r = [
                p[0] + b[0] - pad,
                p[1] + b[1] - pad,
                p[0] + b[2] + pad,
                p[1] + b[3] + pad,
            ];
            if r[2] < 0.0 || r[3] < 0.0 || r[0] > vw || r[1] > vh {
                continue;
            }
            if self.collision && !index.try_insert_owned(owner, r) {
                continue;
            }
            self.visible[i] = true;
        }
        drop(shared_guard);
        for (row, w) in moved {
            self.anchors.update_positions(row, &[w]);
        }
        let mut states = vec![0.0f32; self.anchors.len() * 4];
        for (i, &(start, end, _)) in self.layout.iter().enumerate() {
            let v = if self.visible[i] { 1.0 } else { 0.0 };
            for g in start..end {
                states[g * 4] = v;
                states[g * 4 + 1] = i as f32;
                states[g * 4 + 2] = angles[g];
                states[g * 4 + 3] = self.halo_scale.get(g).copied().unwrap_or(1.0);
            }
        }
        self.states.set_f32(&states);
    }
}

/// A path label's glyph this frame: screen point, angle, world anchor.
type PlacedGlyph = ([f32; 2], f32, [f64; 3]);

/// Where each of a path label's glyphs goes this frame: its screen point,
/// its angle (radians, y down) and its world anchor — or `None` when the path
/// is behind the eye, too short for the text, or bends too sharply.
fn place_on_path(
    view: &MarkView,
    path: &[[f64; 3]],
    glyphs: &[PathGlyph],
    width: f32,
    max_bend: f32,
) -> Option<Vec<PlacedGlyph>> {
    if path.len() < 2 || glyphs.is_empty() {
        return None;
    }
    let mut screen: Vec<[f32; 2]> = Vec::with_capacity(path.len());
    for p in path {
        let s = view.project(*p)?;
        screen.push([s[0], s[1]]);
    }
    let seg = |a: [f32; 2], b: [f32; 2]| ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    let mut cum = vec![0.0f32];
    for w in screen.windows(2) {
        cum.push(cum.last().unwrap() + seg(w[0], w[1]));
    }
    let total = *cum.last().unwrap();
    if total < width {
        return None;
    }
    // Read left to right: walk the line from whichever end puts the text's
    // middle heading right.
    let at = |d: f32| -> (usize, f32) {
        let k = cum.partition_point(|&c| c <= d).clamp(1, screen.len() - 1) - 1;
        let len = (cum[k + 1] - cum[k]).max(1e-6);
        (k, ((d - cum[k]) / len).clamp(0.0, 1.0))
    };
    let (km, _) = at(total * 0.5);
    let forward = screen[km + 1][0] >= screen[km][0];
    let start = (total - width) * 0.5;
    let mut out = Vec::with_capacity(glyphs.len());
    let mut last: Option<f32> = None;
    for g in glyphs {
        let d = if forward {
            start + g.along
        } else {
            total - start - g.along
        };
        let (k, t) = at(d);
        let (a, b) = (screen[k], screen[k + 1]);
        let p = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let mut angle = (b[1] - a[1]).atan2(b[0] - a[0]);
        if !forward {
            angle += std::f32::consts::PI;
        }
        if let Some(prev) = last {
            let mut turn = (angle - prev).abs() % std::f32::consts::TAU;
            if turn > std::f32::consts::PI {
                turn = std::f32::consts::TAU - turn;
            }
            if turn > max_bend {
                return None;
            }
        }
        last = Some(angle);
        let (wa, wb) = (path[k], path[k + 1]);
        let t = t as f64;
        let world = [
            wa[0] + (wb[0] - wa[0]) * t,
            wa[1] + (wb[1] - wa[1]) * t,
            wa[2] + (wb[2] - wa[2]) * t,
        ];
        out.push((p, angle, world));
    }
    Some(out)
}

impl Mark for TextLabels {
    fn kind(&self) -> MarkKind {
        MarkKind::Glyph
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        if self.layout_dirty {
            // Not laid out yet; `prepare` does it before anything is drawn.
            self.labels
                .iter()
                .map(|l| l.text.len())
                .sum::<usize>()
                .min(1)
        } else {
            self.anchors.len()
        }
    }
    fn kind_uniforms(&self) -> KindUniforms {
        // The halo in distance-field units, which the shader compares
        // against; the field reaches half a unit outside the outline.
        let halo = (self.halo_width.max(0.0) * self.atlas.raster / self.font_size.max(1.0))
            / (2.0 * self.atlas.spread as f32);
        (
            // y: one em in pixels, which the depth lift is measured in;
            // z: 1 for multi-channel fields.
            [
                halo.min(0.5),
                self.font_size.max(1.0),
                if self.atlas.msdf { 1.0 } else { 0.0 },
                0.0,
            ],
            [0.0; 4],
            super::points::srgb_u8_to_linear(self.halo_color),
        )
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        self.place(ctx.view);
        let stale = self
            .atlas_gpu
            .as_ref()
            .is_none_or(|(h, _, _)| *h != self.atlas.height);
        if stale {
            let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers label atlas"),
                size: wgpu::Extent3d {
                    width: ATLAS_WIDTH,
                    height: self.atlas.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("threers label atlas bind group"),
                layout: &ctx.resources.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&ctx.resources.linear_sampler),
                    },
                ],
            });
            self.atlas_gpu = Some((self.atlas.height, tex, bg));
            self.atlas.dirty = true;
        }
        if self.atlas.dirty {
            let (_, tex, _) = self.atlas_gpu.as_ref().unwrap();
            ctx.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.atlas.pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ATLAS_WIDTH * 4),
                    rows_per_image: Some(self.atlas.height),
                },
                wgpu::Extent3d {
                    width: ATLAS_WIDTH,
                    height: self.atlas.height,
                    depth_or_array_layers: 1,
                },
            );
            self.atlas.dirty = false;
        }
    }
    fn release_gpu(&mut self) {
        self.atlas_gpu = None;
        super::release_columns(self);
    }
    fn gpu_bytes(&mut self) -> u64 {
        let atlas = self
            .atlas_gpu
            .as_ref()
            .map_or(0, |(_, t, _)| t.width() as u64 * t.height() as u64 * 4);
        super::column_bytes(self) + atlas
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![
            &mut self.anchors,
            &mut self.rects,
            &mut self.uvs,
            &mut self.colors,
            &mut self.states,
        ]
    }
    fn extra_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.atlas_gpu.as_ref().map(|(_, _, bg)| bg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marks::Origin;

    fn ortho_view() -> MarkView {
        // Identity view-projection over a 200×100 viewport: x and y in
        // [-1, 1] map straight to the screen.
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        MarkView {
            view_proj: m,
            projection_scale: 1.0,
            viewport: (200, 100),
            origin: Origin::ZERO,
            light_direction: [0.0, 1.0, 0.0],
            eye: [0.0, 0.0, 1.0, 0.0],
            tone_mapping: (crate::ToneMapping::None, 1.0),
            reversed_z: false,
        }
    }

    #[test]
    fn distance_field_is_half_on_the_outline() {
        // A 4×4 solid square.
        let spread = spread_for(RASTER);
        let field = distance_field(&[255; 16], 4, 4, spread);
        let w = 4 + 2 * spread as usize;
        let at = |x: usize, y: usize| field[y * w + x];
        let c = spread as usize;
        assert!(at(c + 1, c + 1) > 128, "inside is above half");
        assert!(at(0, 0) < 20, "far outside is near zero");
        // Just inside and just outside the left edge straddle 128.
        assert!(at(c, c + 2) >= 128 && at(c - 1, c + 2) < 128);
    }

    /// With a shared index begun each frame, a still view places nothing;
    /// a layer that changes re-places without colliding with its own old
    /// boxes, and the next frame settles the draw order again.
    #[test]
    fn a_shared_index_skips_placement_while_nothing_changes() {
        let index = crate::marks::CollisionIndex::new().shared();
        let layer = |text: &str| {
            let mut l = TextLabels::new(CaptionFont::builtin());
            l.collision_index = Some(index.clone());
            l.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], text)]);
            l
        };
        let (mut a, mut b) = (layer("AAAA"), layer("BBBB"));
        let view = ortho_view();
        let frame = |a: &mut TextLabels, b: &mut TextLabels| {
            let cleared = index.lock().unwrap().begin_frame(&view);
            a.place(&view);
            b.place(&view);
            cleared
        };
        assert!(frame(&mut a, &mut b));
        assert_eq!((a.visible(), b.visible()), (&[true][..], &[false][..]));
        assert_eq!(index.lock().unwrap().len(), 1);
        // Still: kept, and nobody placed again — no second copy of a's box.
        assert!(!frame(&mut a, &mut b));
        assert_eq!(index.lock().unwrap().len(), 1);
        // a changes where it stands: its old box must not hide it.
        a.set_labels(vec![TextLabel::new([0.0, 0.0, 0.0], "AAAAA")]);
        assert!(!frame(&mut a, &mut b));
        assert_eq!(a.visible(), &[true]);
        assert!(frame(&mut a, &mut b));
        assert_eq!((a.visible(), b.visible()), (&[true][..], &[false][..]));
        // a moves out of b's way. It shows — its old box is not in its way —
        // and the frame after, b gets its turn.
        a.set_labels(vec![TextLabel::new([0.8, 0.8, 0.0], "AAAA")]);
        assert!(!frame(&mut a, &mut b));
        assert_eq!(a.visible(), &[true]);
        assert!(frame(&mut a, &mut b));
        assert_eq!((a.visible(), b.visible()), (&[true][..], &[true][..]));
    }

    #[test]
    fn a_new_font_size_lays_the_labels_out_again() {
        let mut l = TextLabels::new(CaptionFont::builtin());
        // Side by side, clear of each other at 14 px.
        l.set_labels(vec![
            TextLabel::new([-0.3, 0.0, 0.0], "AB"),
            TextLabel::new([0.3, 0.0, 0.0], "CD"),
        ]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[true, true]);
        l.font_size = 40.0;
        l.place(&ortho_view());
        assert_eq!(
            l.visible(),
            &[true, false],
            "three times the size, they overlap"
        );
    }

    #[test]
    fn overlapping_labels_keep_the_higher_priority() {
        let mut l = TextLabels::new(CaptionFont::builtin());
        let mut a = TextLabel::new([0.0, 0.0, 0.0], "AAAA");
        a.priority = 1.0;
        let mut b = TextLabel::new([0.01, 0.0, 0.0], "BBBB");
        b.priority = 2.0;
        // Far away on screen: no collision with either.
        let c = TextLabel::new([0.8, 0.8, 0.0], "C");
        l.set_labels(vec![a, b, c]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[false, true, true]);
        l.collision = false;
        l.last_view = None;
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[true, true, true]);
    }

    #[test]
    fn stickiness_keeps_a_shown_label_against_a_slightly_better_rival() {
        let mut l = TextLabels::new(CaptionFont::builtin());
        l.stickiness = 0.5;
        let label = |x: f64, text: &str, p: f32| {
            let mut t = TextLabel::new([x, 0.0, 0.0], text);
            t.priority = p;
            t
        };
        l.set_labels(vec![label(0.0, "AAAA", 1.0), label(0.01, "BBBB", 0.9)]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[true, false]);
        // B edges ahead by less than the bonus: A keeps its place.
        l.set_labels(vec![label(0.0, "AAAA", 1.0), label(0.01, "BBBB", 1.3)]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[true, false], "sticky");
        // By more than the bonus, B takes it.
        l.set_labels(vec![label(0.0, "AAAA", 1.0), label(0.01, "BBBB", 1.6)]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[false, true]);
    }

    #[test]
    fn labels_off_screen_or_behind_are_hidden() {
        let mut l = TextLabels::new(CaptionFont::builtin());
        l.set_labels(vec![TextLabel::new([5.0, 0.0, 0.0], "far right")]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[false]);
    }

    #[test]
    fn path_labels_follow_the_line_and_read_left_to_right() {
        let glyph_xs = |l: &TextLabels| -> (Vec<f64>, Vec<f32>) {
            let pos = l.anchors.positions_f64(None).unwrap();
            let st: Vec<f32> = bytemuck::pod_collect_to_vec(l.states.bytes());
            (
                pos.iter().map(|p| p[0]).collect(),
                st.chunks(4).map(|s| s[2]).collect(),
            )
        };
        let mut l = TextLabels::new(CaptionFont::builtin());
        l.font_size = 10.0;
        // Across the 200 px view, left to right, then drawn right to left.
        for path in [
            vec![[-0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
            vec![[0.9, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.9, 0.0, 0.0]],
        ] {
            l.set_labels(vec![TextLabel::along(path, "ROAD")]);
            l.place(&ortho_view());
            assert_eq!(l.visible(), &[true]);
            let (xs, angles) = glyph_xs(&l);
            assert_eq!(xs.len(), 4);
            assert!(xs.windows(2).all(|w| w[1] > w[0]), "left to right: {xs:?}");
            for a in angles {
                let a = a.rem_euclid(std::f32::consts::TAU);
                assert!(
                    !(1e-3..=std::f32::consts::TAU - 1e-3).contains(&a),
                    "upright: {a}"
                );
            }
        }
        // Up the screen: a quarter turn.
        l.set_labels(vec![TextLabel::along(
            vec![[0.0, 0.9, 0.0], [0.0, -0.9, 0.0]],
            "UP",
        )]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[true]);
        let (_, angles) = glyph_xs(&l);
        assert!(
            (angles[0].abs() - std::f32::consts::FRAC_PI_2).abs() < 1e-3,
            "{angles:?}"
        );
        // Too short for the text, and too sharp a turn: hidden.
        l.set_labels(vec![TextLabel::along(
            vec![[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
            "LONG NAME",
        )]);
        l.place(&ortho_view());
        assert_eq!(l.visible(), &[false]);
        l.set_labels(vec![TextLabel::along(
            vec![[-0.3, 0.3, 0.0], [0.0, -0.3, 0.0], [0.3, 0.3, 0.0]],
            "VVVVVVVV",
        )]);
        l.place(&ortho_view());
        assert_eq!(
            l.visible(),
            &[false],
            "the V bends too sharply under the text"
        );
    }

    #[test]
    fn each_glyph_row_records_its_label() {
        let mut l = TextLabels::new(CaptionFont::builtin());
        l.set_labels(vec![
            TextLabel::new([0.0, -0.5, 0.0], "ab"),
            TextLabel::new([0.0, 0.5, 0.0], "c d"),
        ]);
        l.place(&ortho_view());
        let states: Vec<f32> = bytemuck::pod_collect_to_vec(l.states.bytes());
        // Space has no ink: 2 + 2 glyphs.
        assert_eq!(states.len(), 16);
        assert_eq!(
            states.chunks(4).map(|s| s[1]).collect::<Vec<_>>(),
            [0.0, 0.0, 1.0, 1.0]
        );
    }
}
