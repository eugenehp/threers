//! Glyph rasterization for captions.
//!
//! Two faces are available:
//!
//! - [`CaptionFont::builtin`] — a bundled 8×8 bitmap face covering printable
//!   ASCII. Costs under a kilobyte, needs no assets, and works identically on
//!   native and `wasm32`. Latin-1 accented letters fold to their base letter
//!   (`é` → `e`) so European text stays readable; anything else draws as a
//!   hollow box.
//! - [`CaptionFont::ui`] / [`CaptionFont::try_sf_compact`] — SF Compact on macOS
//!   when installed; [`CaptionFont::builtin`] elsewhere. [`CaptionPainter::new`](super::CaptionPainter::new)
//!   uses this by default.
//! - [`CaptionFont::from_ttf_bytes`] — a real TrueType face, rasterized from
//!   its outlines with 4× supersampled analytic coverage. Use this when you
//!   care about typography or need glyphs outside ASCII.
//!
//! Rasterized glyphs are cached per (character, size), so re-drawing the same
//! caption across many frames costs a memcpy.

use std::collections::HashMap;

use crate::curves::Curve2;
use crate::loaders::{TtfError, TtfFont};

/// A rasterized glyph: an 8-bit coverage mask plus placement relative to the
/// pen position on the baseline.
#[derive(Clone, Debug, Default)]
pub struct GlyphBitmap {
    /// Mask width in pixels.
    pub width: u32,
    /// Mask height in pixels.
    pub height: u32,
    /// Row-major coverage, `0` = transparent, `255` = fully inside the glyph.
    pub coverage: Vec<u8>,
    /// X offset from the pen position to mask column 0.
    pub offset_x: i32,
    /// Y offset from the baseline to mask row 0 (negative = above baseline).
    pub offset_y: i32,
    /// How far to advance the pen after drawing, in pixels.
    pub advance: f32,
}

impl GlyphBitmap {
    /// Whether the glyph has no ink (a space, or an unmapped codepoint).
    pub fn is_blank(&self) -> bool {
        self.width == 0 || self.height == 0 || self.coverage.is_empty()
    }
}

/// Vertical metrics for one type size, in pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FontMetrics {
    /// Height of the tallest glyph above the baseline.
    pub ascent: f32,
    /// Depth below the baseline (positive).
    pub descent: f32,
    /// Recommended extra leading between lines.
    pub line_gap: f32,
}

impl FontMetrics {
    /// Baseline-to-baseline distance with no extra leading.
    pub fn line_height(&self) -> f32 {
        self.ascent + self.descent + self.line_gap
    }
}

enum Face {
    Builtin,
    Ttf(Box<TtfFace>),
    Sdf(std::sync::Arc<SdfGlyphs>),
    Bitmap(std::sync::Arc<super::glyphs::BitmapGlyphs>),
}

/// A TrueType face; for a variable font, also the bytes to read it again at
/// another optical size or weight.
struct TtfFace {
    base: std::sync::Arc<TtfFont>,
    source: Option<VariableSource>,
    /// Where the face came from, to make its other styles from.
    family: Family,
    /// Bold made by growing the outlines: the font has no weight to give.
    synthetic_bold: bool,
    /// Italic made by slanting the outlines: the family has no italic.
    oblique: bool,
    /// The CSS weight and italic this face was asked for.
    style: (u16, bool),
}

/// A face's family: its upright file (or the face itself, parsed by the
/// caller) and its italic file if it has one.
#[derive(Clone)]
struct Family {
    upright: FamilyFile,
    italic: Option<std::sync::Arc<[u8]>>,
}

#[derive(Clone)]
enum FamilyFile {
    Bytes(std::sync::Arc<[u8]>),
    Parsed(std::sync::Arc<TtfFont>),
}

/// The slant of a synthetic italic, x per y: Skia's, about 14°.
const OBLIQUE_SKEW: f32 = 0.25;

/// How far a synthetic bold grows each outline, in pixels at `size`: half
/// the stroke Skia fills it with — a 24th of the size at 9 pixels and
/// below, a 32nd at 36 and above, linearly between.
fn synthetic_bold_outset(size: f32) -> f32 {
    let t = ((size - 9.0) / 27.0).clamp(0.0, 1.0);
    size * (1.0 / 24.0 + t * (1.0 / 32.0 - 1.0 / 24.0)) * 0.5
}

/// What drawing a face changes in its outlines: the slant, and how far
/// they grow (`outset`, a synthetic bold's).
#[derive(Clone, Copy, Default)]
struct Synthesis {
    skew: f32,
    outset: f32,
    /// Fit the outline to the pixel grid vertically ([`Hinting::Slight`]).
    hint: bool,
    /// How far the outline grows out in x and y, in pixels, about where it
    /// stands ([`CaptionFont::set_font_smoothing`]).
    darken: [f32; 2],
}

/// How far macOS's font smoothing grows a glyph's outline at `size` pixels
/// per em, in x and y: CoreText's stem darkening, fitted to Chrome's canvas
/// text on macOS at 8, 16, 32 and 64 pixels — across, nothing below about
/// 6.5 pixels and 0.0105 a pixel more for each after, to 0.36; up and down,
/// 0.02 and 0.0055 a pixel, to 0.3. CoreText grows a glyph up, not down:
/// [`dilate_contours`] then lifts it by the vertical amount.
fn stem_darkening(size: f32) -> [f32; 2] {
    [(0.0105 * (size - 6.5)).clamp(0.0, 0.36), (0.0055 * size + 0.02).min(0.3)]
}

/// Grow closed device-space `contours` (y down) by `amount` in x and y —
/// each point moved out along the bisector of its edges' normals — then
/// lift them by `amount`'s y, so the foot stays and the top grows twice.
fn dilate_contours(contours: &mut [Vec<(f32, f32)>], amount: [f32; 2]) {
    let area = |c: &[(f32, f32)]| c.windows(2).map(|w| w[0].0 * w[1].1 - w[1].0 * w[0].1).sum::<f32>();
    // Ink is on the same side of every contour's travel (non-zero
    // winding): the side the largest, an outer one, encloses.
    let Some(largest) = contours.iter().map(|c| area(c)).max_by(|a, b| a.abs().total_cmp(&b.abs())) else {
        return;
    };
    let out = if largest > 0.0 { 1.0 } else { -1.0 };
    let unit = |a: (f32, f32), b: (f32, f32)| {
        let d = (b.0 - a.0, b.1 - a.1);
        let l = (d.0 * d.0 + d.1 * d.1).sqrt();
        if l > 0.0 { (d.0 / l, d.1 / l) } else { (0.0, 0.0) }
    };
    for c in contours.iter_mut() {
        let closed = c.len() > 1 && c.first() == c.last();
        let n = if closed { c.len() - 1 } else { c.len() };
        if n < 3 {
            continue;
        }
        let pts = c[..n].to_vec();
        for i in 0..n {
            let (a, b) = (unit(pts[(i + n - 1) % n], pts[i]), unit(pts[i], pts[(i + 1) % n]));
            // (dy, -dx): out of a contour whose area is positive.
            let bis = (a.1 + b.1, -(a.0 + b.0));
            let l = (bis.0 * bis.0 + bis.1 * bis.1).sqrt();
            if l > 0.0 {
                c[i] = (pts[i].0 + out * bis.0 / l * amount[0], pts[i].1 + out * bis.1 / l * amount[1]);
            }
        }
        if closed {
            c[n] = c[0];
        }
        // Its foot stays: grown up, not down.
        for p in c.iter_mut() {
            p.1 -= amount[1];
        }
    }
}

/// How a TrueType face is fitted to the pixel grid.
///
/// [`None`](Hinting::None), the default, draws the outline as designed, as
/// CoreText does — and the same on every platform, which burned-in captions
/// want. [`Slight`](Hinting::Slight) is FreeType's "light" hinting, the
/// one most Linux desktops and Chrome there use: vertical only, so the
/// baseline, x-height, cap height and descender each land on a pixel edge
/// and their horizontal strokes come out as solid rows rather than two
/// half-grey ones. Small text reads crisper; the shapes move by up to half a
/// pixel to do it, and nothing changes horizontally — advances and spacing
/// are the outline's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Hinting {
    #[default]
    None,
    Slight,
}

/// A face's alignment zones, in font units: where its flat-topped and
/// round-topped letters reach.
#[derive(Debug, Clone, Copy)]
struct BlueZones {
    /// Flat and round x-height: tops of `x` and `o`.
    x: Option<(f32, f32)>,
    /// Flat and round cap height: tops of `H` and `O`.
    cap: Option<(f32, f32)>,
    /// The round letters' dip below the baseline: bottom of `o`.
    base_shoot: Option<f32>,
    /// The descender: bottom of `p`.
    descender: Option<f32>,
    /// The lowercase ascender: top of `h`.
    ascender: Option<f32>,
}

impl BlueZones {
    fn of(font: &TtfFont) -> Self {
        let extent = |c: char| -> Option<(f32, f32)> {
            let gid = *font.cmap.get(&(c as u32))?;
            let g = font.glyphs.get(gid as usize)?;
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for curve in g.shape.outline.curve_path.curves.iter() {
                for k in 0..=8 {
                    let y = curve.get_point(k as f32 / 8.0).y;
                    lo = lo.min(y);
                    hi = hi.max(y);
                }
            }
            (lo <= hi).then_some((lo, hi))
        };
        let pair = |flat: char, round: char| Some((extent(flat)?.1, extent(round)?.1));
        Self {
            x: pair('x', 'o'),
            cap: pair('H', 'O'),
            base_shoot: extent('o').map(|e| e.0),
            descender: extent('p').map(|e| e.0),
            ascender: extent('h').map(|e| e.1),
        }
    }
}

/// A piecewise-linear map of font-unit heights to pixel heights that puts
/// each zone on a pixel edge — FreeType's light autohinter's vertical fit,
/// in its own 26.6 fixed point, since the zones sit on rounding boundaries
/// often enough (Arial's cap height at 18 pixels is 13.5 exactly) that a
/// float lands on the other side of them.
///
/// First a uniform scale, so the round x-height lands on a whole pixel,
/// rounding up from three eighths (`(scaled + 40) & ~63`); then each zone
/// whose overshoot is at most three quarters of a pixel snaps its flat
/// height to a pixel and its overshoot to none (under half a pixel) or half
/// a pixel; a taller zone is left as scaled. Heights between zones
/// interpolate; beyond the last, they keep the scaled slope.
struct VerticalWarp {
    knots: Vec<(f32, f32)>,
    slope: f32,
}

impl VerticalWarp {
    fn new(font: &TtfFont, size: f32) -> Self {
        // FreeType's fixed-point helpers: 16.16 scales, 26.6 positions.
        let mul_fix = |a: i64, b: i64| -> i64 {
            let p = a * b;
            if p >= 0 { (p + 0x8000) >> 16 } else { -((-p + 0x8000) >> 16) }
        };
        let mul_div = |a: i64, b: i64, c: i64| -> i64 {
            let p = a * b;
            if p >= 0 { (p + c / 2) / c } else { -((-p + c / 2) / c) }
        };
        let round = |v: i64| (v + 32) & !63;
        let upem = font.units_per_em.max(1) as i64;
        let ppem = (size * 64.0).round() as i64;
        let mut scale = (ppem * 65536 + upem / 2) / upem;
        let zones = BlueZones::of(font);
        if let Some((_, shoot)) = zones.x {
            let scaled = mul_fix(shoot.round() as i64, scale);
            let fitted = (scaled + 40) & !63;
            if scaled > 0 && fitted > 0 {
                scale = mul_div(scale, fitted, scaled);
            }
        }
        let px = |v: i64| v as f32 / 64.0;
        let mut knots = vec![(0.0, 0.0)];
        let mut zone = |flat: f32, shoot: Option<f32>| {
            let r = mul_fix(flat.round() as i64, scale);
            let Some(shoot) = shoot.filter(|&s| s != flat) else {
                knots.push((flat, px(round(r))));
                return;
            };
            let s = mul_fix(shoot.round() as i64, scale);
            let dist = s - r;
            if dist.abs() > 48 {
                return;
            }
            let fit = round(r);
            let over = match dist.abs() {
                d if d < 32 => 0,
                d if d < 48 => 32,
                _ => 64,
            } * dist.signum();
            knots.push((flat, px(fit)));
            knots.push((shoot, px(fit + over)));
        };
        if let Some((flat, shoot)) = zones.x {
            zone(flat, Some(shoot));
        }
        if let Some((flat, shoot)) = zones.cap {
            zone(flat, Some(shoot));
        }
        if let Some(a) = zones.ascender {
            zone(a, None);
        }
        if let Some(s) = zones.base_shoot.filter(|&s| s < 0.0) {
            zone(0.0, Some(s));
        }
        if let Some(d) = zones.descender.filter(|&d| d < 0.0) {
            zone(d, None);
        }
        knots.sort_by(|a, b| a.0.total_cmp(&b.0));
        knots.dedup_by(|a, b| a.0 == b.0);
        // Never folding the outline over: each knot at or above the last.
        for i in 1..knots.len() {
            knots[i].1 = knots[i].1.max(knots[i - 1].1);
        }
        // Pixels per font unit: 16.16 scale to 26.6 positions.
        Self { knots, slope: scale as f32 / 65536.0 / 64.0 }
    }

    /// Font-unit height `y` in pixels, up.
    fn map(&self, y: f32) -> f32 {
        let k = &self.knots;
        let (first, last) = (k[0], k[k.len() - 1]);
        if y <= first.0 {
            return first.1 + (y - first.0) * self.slope;
        }
        if y >= last.0 {
            return last.1 + (y - last.0) * self.slope;
        }
        let i = k.partition_point(|p| p.0 <= y) - 1;
        let (a, b) = (k[i], k[i + 1]);
        a.1 + (y - a.0) / (b.0 - a.0) * (b.1 - a.1)
    }
}

struct VariableSource {
    bytes: std::sync::Arc<[u8]>,
    /// The `wght` it resolved to, when the font has the axis.
    wght: Option<f32>,
    /// The `opsz` axis's range, when the font has one.
    opsz: Option<(f32, f32)>,
    /// The face at each optical size used so far, by `opsz × 4`.
    sized: std::sync::Mutex<HashMap<u32, std::sync::Arc<TtfFont>>>,
}

impl TtfFace {
    fn fixed(font: TtfFont) -> Self {
        let base = std::sync::Arc::new(font);
        let family = Family { upright: FamilyFile::Parsed(base.clone()), italic: None };
        Self { base, source: None, family, synthetic_bold: false, oblique: false, style: (400, false) }
    }

    /// The face of `family` at CSS `weight`, italic or not: the family's
    /// own where it has them, synthesized where not — bold when asked for
    /// 600 or more of a face that is lighter and has no weight axis, an
    /// oblique when there is no italic file.
    fn styled(family: &Family, weight: u16, italic: bool) -> Result<Self, TtfError> {
        let file = match (&family.italic, italic) {
            (Some(bytes), true) => FamilyFile::Bytes(bytes.clone()),
            _ => family.upright.clone(),
        };
        let mut face = match file {
            FamilyFile::Bytes(bytes) => Self::variable(bytes, weight)?,
            FamilyFile::Parsed(font) => {
                Self { base: font, source: None, family: family.clone(), synthetic_bold: false, oblique: false, style: (400, false) }
            }
        };
        face.family = family.clone();
        let has_weights = face.source.as_ref().is_some_and(|s| s.wght.is_some());
        face.synthetic_bold = weight >= 600 && !has_weights && face.base.weight_class < 600;
        face.oblique = italic && family.italic.is_none();
        face.style = (weight, italic);
        Ok(face)
    }

    /// The outline changes this face draws with, at `size`.
    fn synthesis(&self, size: f32) -> Synthesis {
        Synthesis {
            skew: if self.oblique { OBLIQUE_SKEW } else { 0.0 },
            outset: if self.synthetic_bold { synthetic_bold_outset(size) } else { 0.0 },
            hint: false,
            darken: [0.0; 2],
        }
    }

    /// `bytes` at CSS `weight`, its optical size following the type size.
    fn variable(bytes: std::sync::Arc<[u8]>, weight: u16) -> Result<Self, TtfError> {
        let (axes, instances) = TtfFont::variation_info(&bytes);
        if axes.is_empty() {
            let mut face = Self::fixed(TtfFont::parse(&bytes)?);
            face.family.upright = FamilyFile::Bytes(bytes);
            face.style = (weight, false);
            return Ok(face);
        }
        let wght = crate::loaders::wght_for_css_weight(&axes, &instances, weight as f32);
        let opsz = axes.iter().find(|a| &a.tag == b"opsz").map(|a| (a.min, a.max));
        let mut at = Vec::new();
        if let Some(w) = wght {
            at.push((*b"wght", w));
        }
        let base = std::sync::Arc::new(TtfFont::parse_variation(&bytes, &at)?);
        let family = Family { upright: FamilyFile::Bytes(bytes.clone()), italic: None };
        let source = VariableSource { bytes, wght, opsz, sized: Default::default() };
        Ok(Self { base, source: Some(source), family, synthetic_bold: false, oblique: false, style: (weight, false) })
    }

    /// The face for type `size` pixels per em: with an `opsz` axis, read at
    /// that optical size, as CoreText and CSS's `font-optical-sizing: auto`
    /// do. SF Compact's display cut is visibly tighter than its text cut, and
    /// drawing a 20-pixel line in the text one sets it 5% wide.
    fn at(&self, size: f32) -> std::sync::Arc<TtfFont> {
        let Some(src) = &self.source else { return self.base.clone() };
        let Some((lo, hi)) = src.opsz else { return self.base.clone() };
        let opsz = size.clamp(lo, hi);
        let key = (opsz * 4.0).round() as u32;
        let mut sized = src.sized.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(f) = sized.get(&key) {
            return f.clone();
        }
        let mut at = vec![(*b"opsz", key as f32 / 4.0)];
        if let Some(w) = src.wght {
            at.push((*b"wght", w));
        }
        let font = TtfFont::parse_variation(&src.bytes, &at)
            .map(std::sync::Arc::new)
            .unwrap_or_else(|_| self.base.clone());
        sized.insert(key, font.clone());
        font
    }
}

/// Glyphs given as signed distance fields rather than outlines — MapLibre's
/// glyph PBFs: each a field of `width + 2·buffer` by `height + 2·buffer`
/// bytes at `size` pixels per em, the edge at 192 and about 32 per pixel
/// inside it.
#[derive(Debug, Clone, Default)]
pub struct SdfGlyphs {
    /// Pixels per em the fields were made at (24 for MapLibre's).
    pub size: f32,
    /// Pixels of field around each glyph's box (3 for MapLibre's).
    pub buffer: u32,
    /// Pixels from the baseline up to where `top` is measured from (26 for
    /// MapLibre's: a glyph's box top is `top + 26` above the baseline).
    pub top_origin: f32,
    pub glyphs: HashMap<u32, SdfGlyph>,
}

/// One distance-field glyph.
#[derive(Debug, Clone, PartialEq)]
pub struct SdfGlyph {
    pub width: u32,
    pub height: u32,
    /// Pixels from the pen to the box's left.
    pub left: i32,
    /// The box's top, relative to [`SdfGlyphs::top_origin`].
    pub top: i32,
    pub advance: f32,
    /// `(width + 2·buffer) × (height + 2·buffer)` bytes.
    pub field: Vec<u8>,
}

/// `c` from a distance-field face at `size`: the field resampled to the
/// size and turned to coverage, a pixel wide at the edge.
fn raster_sdf(set: &SdfGlyphs, c: char, size: f32, shift: f32) -> GlyphBitmap {
    let blank = |advance: f32| GlyphBitmap { width: 0, height: 0, coverage: Vec::new(), offset_x: 0, offset_y: 0, advance };
    let Some(g) = set.glyphs.get(&(c as u32)) else { return blank(0.0) };
    let k = size / set.size.max(1.0);
    let advance = g.advance * k;
    if g.width == 0 || g.height == 0 {
        return blank(advance);
    }
    let b = set.buffer as f32;
    let (fw, fh) = (g.width + 2 * set.buffer, g.height + 2 * set.buffer);
    if g.field.len() < (fw * fh) as usize {
        return blank(advance);
    }
    let top = g.top as f32 + set.top_origin;
    // The output box: the glyph's at this size, a pixel of margin around.
    let x0 = (g.left as f32 * k + shift).floor() as i32 - 1;
    let y0 = (-top * k).floor() as i32 - 1;
    let w = (g.width as f32 * k).ceil() as u32 + 3;
    let h = (g.height as f32 * k).ceil() as u32 + 3;
    let sample = |x: f32, y: f32| -> f32 {
        // Bilinear, texel centres at +0.5, clamped to the field.
        let (x, y) = ((x - 0.5).clamp(0.0, fw as f32 - 1.0), (y - 0.5).clamp(0.0, fh as f32 - 1.0));
        let (ix, iy) = (x.floor() as u32, y.floor() as u32);
        let (fx, fy) = (x - ix as f32, y - iy as f32);
        let at = |cx: u32, cy: u32| g.field[(cy.min(fh - 1) * fw + cx.min(fw - 1)) as usize] as f32;
        let top_row = at(ix, iy) * (1.0 - fx) + at(ix + 1, iy) * fx;
        let bottom_row = at(ix, iy + 1) * (1.0 - fx) + at(ix + 1, iy + 1) * fx;
        top_row * (1.0 - fy) + bottom_row * fy
    };
    let mut coverage = Vec::with_capacity((w * h) as usize);
    for oy in 0..h {
        for ox in 0..w {
            // The output pixel's centre in pen space, then in the field.
            let px = (x0 + ox as i32) as f32 + 0.5 - shift;
            let py = (y0 + oy as i32) as f32 + 0.5;
            let v = sample(px / k - g.left as f32 + b, py / k + top + b);
            // Field units to output pixels: 32 a field pixel, `k` output
            // pixels a field pixel.
            let d = (v - 192.0) / 32.0 * k;
            coverage.push(((d + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8);
        }
    }
    GlyphBitmap { width: w, height: h, coverage, offset_x: x0, offset_y: y0, advance }
}

/// A font face plus its rasterized-glyph cache.
///
/// ```
/// use threers::captions::CaptionFont;
/// let mut font = CaptionFont::builtin();
/// let g = font.glyph('A', 32.0);
/// assert!(!g.is_blank());
/// assert!(font.glyph(' ', 32.0).is_blank());
/// ```
pub struct CaptionFont {
    face: Face,
    /// Keyed by codepoint and quantized size (¼-pixel steps).
    cache: HashMap<(u32, u32, u32), GlyphBitmap>,
    /// Glyphs grown by an outline radius, by codepoint, size, phase and
    /// radius (sixteenths of a pixel).
    grown: HashMap<(u32, u32, u32, u32), GlyphBitmap>,
    /// The optical size glyphs are drawn at regardless of their pixel size;
    /// `None` follows the size. See [`set_optical_size`](Self::set_optical_size).
    optical_size: Option<f32>,
    /// See [`set_hinting`](Self::set_hinting).
    hinting: Hinting,
    /// See [`set_font_smoothing`](Self::set_font_smoothing).
    font_smoothing: bool,
}

impl Default for CaptionFont {
    fn default() -> Self {
        Self::builtin()
    }
}

impl std::fmt::Debug for CaptionFont {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CaptionFont")
            .field(
                "face",
                &match self.face {
                    Face::Builtin => "builtin",
                    Face::Ttf(_) => "truetype",
                    Face::Sdf(_) => "distance fields",
                    Face::Bitmap(_) => "bitmaps",
                },
            )
            .field("cached_glyphs", &self.cache.len())
            .finish()
    }
}

impl CaptionFont {
    /// The bundled 8×8 bitmap face — no assets required.
    pub fn builtin() -> Self {
        Self {
            face: Face::Builtin,
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: None,
            hinting: Hinting::None,
            font_smoothing: false,
        }
    }

    /// Use an already-parsed TrueType face, as it is: a variable font at
    /// the location it was parsed at, at every size.
    pub fn from_ttf(font: TtfFont) -> Self {
        Self {
            face: Face::Ttf(Box::new(TtfFace::fixed(font))),
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: None,
            hinting: Hinting::None,
            font_smoothing: false,
        }
    }

    /// Glyphs given as distance fields ([`SdfGlyphs`]): MapLibre's glyph
    /// PBFs, a face for the characters a TrueType face lacks.
    pub fn from_sdf_glyphs(glyphs: std::sync::Arc<SdfGlyphs>) -> Self {
        Self {
            face: Face::Sdf(glyphs),
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: None,
            hinting: Hinting::None,
            font_smoothing: false,
        }
    }

    /// Glyphs given as images ([`BitmapGlyphs`](super::glyphs::BitmapGlyphs)):
    /// a pre-packed font atlas, or glyphs another renderer drew. Drawn at
    /// other sizes by resampling.
    pub fn from_bitmap_glyphs(glyphs: std::sync::Arc<super::glyphs::BitmapGlyphs>) -> Self {
        Self {
            face: Face::Bitmap(glyphs),
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: None,
            hinting: Hinting::None,
            font_smoothing: false,
        }
    }

    /// The bitmap face, if this is one.
    pub fn bitmap_glyphs(&self) -> Option<&std::sync::Arc<super::glyphs::BitmapGlyphs>> {
        match &self.face {
            Face::Bitmap(b) => Some(b),
            _ => None,
        }
    }

    /// Parse a `.ttf` and use it as the caption face.
    ///
    /// Accepts the same subset [`TtfFont`] does: TrueType `glyf` outlines with
    /// a format-4 `cmap`. OpenType/CFF (`.otf`) is not supported.
    ///
    /// A variable font is read at its regular weight (CSS `400`), not at its
    /// default instance — which is the designer's choice of master and can be
    /// anything; SF Compact's is Black — and at the optical size of each type
    /// size drawn.
    pub fn from_ttf_bytes(bytes: &[u8]) -> Result<Self, TtfError> {
        Self::from_ttf_bytes_weight(bytes, 400)
    }

    /// [`from_ttf_bytes`](Self::from_ttf_bytes) at a CSS weight (100–900).
    /// A static font ignores it.
    pub fn from_ttf_bytes_weight(bytes: &[u8], weight: u16) -> Result<Self, TtfError> {
        Ok(Self {
            face: Face::Ttf(Box::new(TtfFace::variable(bytes.into(), weight)?)),
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: None,
            hinting: Hinting::None,
            font_smoothing: false,
        })
    }

    /// The same family at CSS `weight` (100–900), italic or not.
    ///
    /// Real where the family has it: a variable font's weight axis, an
    /// italic file ([`try_sf_compact`](Self::try_sf_compact) knows SF
    /// Compact's). Synthesized where not, as browsers do: bold, asked for
    /// at 600 or more of a lighter face, grows the outlines by a 48th to a
    /// 64th of the size each side and widens the advances to match; italic
    /// slants them by a quarter. `None` for a face without outlines.
    pub fn styled(&self, weight: u16, italic: bool) -> Option<Self> {
        let Face::Ttf(face) = &self.face else { return None };
        Some(Self {
            face: Face::Ttf(Box::new(TtfFace::styled(&face.family, weight, italic).ok()?)),
            cache: HashMap::new(),
            grown: HashMap::new(),
            optical_size: self.optical_size,
            hinting: self.hinting,
            font_smoothing: self.font_smoothing,
        })
    }

    /// [`styled`](Self::styled) at another weight, as italic as this face.
    pub fn with_weight(&self, weight: u16) -> Option<Self> {
        let (_, italic) = self.style()?;
        self.styled(weight, italic)
    }

    /// The CSS weight and italic a TrueType face was asked for; `None` for
    /// the bitmap face and distance-field glyphs.
    pub fn style(&self) -> Option<(u16, bool)> {
        match &self.face {
            Face::Ttf(face) => Some(face.style),
            _ => None,
        }
    }

    /// The CSS weight a TrueType face was asked for; `None` for any other.
    pub fn weight(&self) -> Option<u16> {
        self.style().map(|s| s.0)
    }

    /// SF Compact when the system ships it (macOS), otherwise
    /// [`builtin`](Self::builtin).
    ///
    /// This is the preferred UI face for burned-in labels and data graphics.
    pub fn ui() -> Self {
        Self::try_sf_compact().unwrap_or_else(Self::builtin)
    }

    /// Load Apple's SF Compact from a well-known install path, at its
    /// Regular weight.
    pub fn try_sf_compact() -> Option<Self> {
        Self::try_sf_compact_weight(400)
    }

    /// [`try_sf_compact`](Self::try_sf_compact) at a CSS weight (100–900).
    pub fn try_sf_compact_weight(weight: u16) -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            const PATHS: &[&str] = &[
                "/System/Library/Fonts/SFCompact.ttf",
                "/System/Library/Fonts/SFCompactRounded.ttf",
            ];
            for path in PATHS {
                let Ok(bytes) = std::fs::read(path) else { continue };
                if let Ok(mut font) = Self::from_ttf_bytes_weight(&bytes, weight) {
                    // Its italic is a file of its own.
                    if let (Face::Ttf(face), Ok(italic)) =
                        (&mut font.face, std::fs::read(path.replace(".ttf", "Italic.ttf")))
                    {
                        face.family.italic = Some(italic.into());
                    }
                    return Some(font);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = weight;
        }
        None
    }

    /// Whether this face is the built-in bitmap one.
    pub fn is_builtin(&self) -> bool {
        matches!(self.face, Face::Builtin)
    }

    /// Draw every glyph at optical size `size` (pixels per em), whatever
    /// its pixel size — or, with `None`, each at its own, the default.
    ///
    /// For a raster made larger than it will be shown: a label atlas
    /// rasterized at 48 pixels for text shown at 14 wants the shapes of a
    /// 14-pixel optical size (sturdier, more open) scaled up, not the
    /// display cut. Returns whether anything changed — only a face with an
    /// `opsz` axis has optical sizes — clearing the cached glyphs if so.
    pub fn set_optical_size(&mut self, size: Option<f32>) -> bool {
        let has_axis = matches!(&self.face, Face::Ttf(f) if f.source.as_ref().is_some_and(|s| s.opsz.is_some()));
        if !has_axis || self.optical_size == size {
            return false;
        }
        self.optical_size = size;
        self.clear_cache();
        true
    }

    /// Fit glyphs to the pixel grid from now on, or not; see [`Hinting`].
    /// Clears the cached glyphs when it changes.
    pub fn set_hinting(&mut self, hinting: Hinting) {
        if self.hinting != hinting {
            self.hinting = hinting;
            self.clear_cache();
        }
    }

    /// The hinting glyphs are drawn with.
    pub fn hinting(&self) -> Hinting {
        self.hinting
    }

    /// Draw TrueType glyphs heavier, as macOS does with font smoothing on
    /// (what a browser's canvas text there shows): each outline grown by
    /// CoreText's stem darkening, up to 0.3 of a pixel a side below 72
    /// pixels per em. Advances and the outline's place stay. Clears the
    /// cached glyphs when it changes.
    pub fn set_font_smoothing(&mut self, on: bool) {
        if self.font_smoothing != on {
            self.font_smoothing = on;
            self.clear_cache();
        }
    }

    pub fn font_smoothing(&self) -> bool {
        self.font_smoothing
    }

    fn darkening(&self, size: f32) -> [f32; 2] {
        if self.font_smoothing { stem_darkening(size) } else { [0.0; 2] }
    }

    /// Drop every cached glyph raster.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.grown.clear();
    }

    /// Vertical metrics at `size` pixels per em.
    pub fn metrics(&self, size: f32) -> FontMetrics {
        match &self.face {
            Face::Builtin => FontMetrics {
                // The 8-row cell puts the baseline under row 6.
                ascent: size * (BUILTIN_ASCENT_ROWS as f32 / BUILTIN_CELL as f32),
                descent: size * (1.0 / BUILTIN_CELL as f32),
                line_gap: 0.0,
            },
            Face::Sdf(_) => FontMetrics {
                ascent: size * 0.8,
                descent: size * 0.2,
                line_gap: 0.0,
            },
            Face::Bitmap(b) => {
                let k = size / b.size.max(1e-6);
                FontMetrics { ascent: b.ascent * k, descent: b.descent * k, line_gap: 0.0 }
            }
            Face::Ttf(face) => {
                let font = &face.base;
                let upem = font.units_per_em.max(1) as f32;
                let scale = size / upem;
                // Fall back to a conventional 80/20 split for fonts with a
                // blank hhea.
                if font.ascender == 0 && font.descender == 0 {
                    FontMetrics {
                        ascent: size * 0.8,
                        descent: size * 0.2,
                        line_gap: 0.0,
                    }
                } else {
                    FontMetrics {
                        ascent: font.ascender as f32 * scale,
                        descent: (-(font.descender as f32)) * scale,
                        line_gap: font.line_gap as f32 * scale,
                    }
                }
            }
        }
    }

    /// Rasterize `c` at `size` pixels per em, caching the result.
    pub fn glyph(&mut self, c: char, size: f32) -> &GlyphBitmap {
        self.glyph_at(c, size, 0)
    }

    /// The glyph rasterized for one of [`SUBPIXEL_PHASES`] horizontal positions
    /// within a pixel.
    ///
    /// Text advances are fractional — a 13.7-pixel advance is normal — and
    /// rounding the pen to a whole pixel for each glyph throws that fraction
    /// away every time. The gaps then alternate: at 13.7 the glyphs land on 0,
    /// 14, 27, 41, 55, which is 14, 13, 14, 14 apart. That unevenness is what
    /// makes rendered text look wobbly next to the same text from a real
    /// renderer, and it is not fixable by compositing — the glyph has to be
    /// *rasterized* for the position it actually sits at.
    ///
    /// So the fraction is quantized to quarters of a pixel and the glyph is
    /// rasterized once per phase. Quarter-pixel steps are below what anyone can
    /// see and keep the cache four entries a glyph rather than unbounded.
    pub fn glyph_at(&mut self, c: char, size: f32, phase: usize) -> &GlyphBitmap {
        let gid = self.gid(c);
        self.glyph_of(gid, size, phase)
    }

    /// The same, by the index [`Self::gid`] gives.
    ///
    /// This is the one that a shaped run asks for: shaping rewrites glyph *ids*, and the
    /// glyph it produces for a joined Arabic letter has no codepoint to ask by.
    ///
    /// The cache is keyed on the id rather than the character, which also merges the
    /// entries for two codepoints that a font maps to one glyph.
    pub fn glyph_of(&mut self, gid: u32, size: f32, phase: usize) -> &GlyphBitmap {
        let size = size.max(1.0);
        let phase = phase % SUBPIXEL_PHASES;
        let key = (gid, (size * 4.0).round() as u32, phase as u32);
        if !self.cache.contains_key(&key) {
            let shift = phase as f32 / SUBPIXEL_PHASES as f32;
            // The three char-keyed faces index by codepoint, so the id converts back;
            // `gid` is the identity for them and `from_u32` cannot fail.
            let c = char::from_u32(gid).unwrap_or('\u{FFFD}');
            let bitmap = match &self.face {
                Face::Builtin => raster_builtin(c, size, shift),
                Face::Ttf(face) => raster_ttf(
                    &face.at(self.optical_size.unwrap_or(size)),
                    gid,
                    size,
                    shift,
                    self.optical_size.unwrap_or(size),
                    Synthesis { hint: self.hinting == Hinting::Slight, darken: self.darkening(size), ..face.synthesis(size) },
                ),
                Face::Sdf(set) => raster_sdf(set, c, size, shift),
                Face::Bitmap(set) => set.raster(c, size, shift),
            };
            self.cache.insert(key, bitmap);
        }
        &self.cache[&key]
    }

    /// `c` at [`glyph_at`](Self::glyph_at)'s size and phase, grown by
    /// `radius` pixels — an outline's shape — from the glyph's own outline.
    ///
    /// Growing the antialiased coverage mask instead can only guess where in
    /// each edge pixel the edge lies; against the true offset outline that
    /// was 0.2 off in coverage along the edge on average and up to 0.7. From
    /// the outline, each pixel's signed distance to it gives the exact edge,
    /// rounded corners and all. `None` for a face without outlines (the
    /// bitmap face, distance-field glyphs), which grow from the mask.
    pub fn grown_glyph_at(&mut self, c: char, size: f32, phase: usize, radius: f32) -> Option<&GlyphBitmap> {
        let gid = self.gid(c);
        self.grown_glyph_of(gid, size, phase, radius)
    }

    /// The same, by the index [`Self::gid`] gives.
    pub fn grown_glyph_of(
        &mut self,
        gid: u32,
        size: f32,
        phase: usize,
        radius: f32,
    ) -> Option<&GlyphBitmap> {
        let Face::Ttf(face) = &self.face else { return None };
        let size = size.max(1.0);
        let phase = phase % SUBPIXEL_PHASES;
        let key = (gid, (size * 4.0).round() as u32, phase as u32, (radius.max(0.0) * 16.0).round() as u32);
        if !self.grown.contains_key(&key) {
            let font = face.at(self.optical_size.unwrap_or(size));
            let shift = phase as f32 / SUBPIXEL_PHASES as f32;
            let syn = Synthesis { hint: self.hinting == Hinting::Slight, darken: self.darkening(size), ..face.synthesis(size) };
            let mut bitmap = raster_grown(
                &font,
                gid,
                size,
                shift + syn.outset,
                key.3 as f32 / 16.0 + syn.outset,
                self.optical_size.unwrap_or(size),
                syn,
            );
            bitmap.advance += 2.0 * syn.outset;
            self.grown.insert(key, bitmap);
        }
        Some(&self.grown[&key])
    }

    /// The kerning between two characters at `size` pixels.
    ///
    /// Zero for the built-in face, which is a fixed-width bitmap and has no
    /// pairs to adjust. A TrueType face's pairs come from `kern` or `GPOS`.
    pub fn kerning(&self, left: char, right: char, size: f32) -> f32 {
        self.kerning_of(self.gid(left), self.gid(right), size)
    }

    /// The index this face stores glyphs by.
    ///
    /// A **glyph id** for a TrueType or PostScript face, and the **codepoint** for the
    /// three char-keyed faces — the built-in bitmap, a distance field and a bitmap set
    /// have no glyph ids to have. One number, whose meaning is the face's, which is what
    /// lets one layout path serve all four: a shaper rewrites a run of glyph ids, and a
    /// face that cannot be shaped simply reports the identity.
    ///
    /// A `CaptionFont` holds exactly one face, so the two meanings never meet.
    #[must_use]
    pub fn gid(&self, c: char) -> u32 {
        match &self.face {
            Face::Ttf(face) => u32::from(
                face.base.cmap.get(&(c as u32)).copied().unwrap_or(0),
            ),
            Face::Builtin | Face::Sdf(_) | Face::Bitmap(_) => c as u32,
        }
    }

    /// Where each glyph of a run attaches, in **pixels** at `size`.
    ///
    /// `glyphs` must be in logical order, as [`Self::shape_run`] returns them. A face
    /// without mark positioning — which is most Latin faces, and all three of the
    /// character-keyed ones — reports nothing attached.
    #[must_use]
    pub fn attachments(&self, glyphs: &[u32], size: f32) -> Vec<crate::loaders::mark::Attachment> {
        let blank = || vec![crate::loaders::mark::Attachment::default(); glyphs.len()];
        let Face::Ttf(face) = &self.face else { return blank() };
        let font = face.at(self.optical_size.unwrap_or(size));
        if font.gpos.is_empty() {
            return blank();
        }
        // The anchors' own variation store, which is `GDEF`'s — the same one the kern
        // values vary through, read here from the kept bytes at the font's location.
        let store = crate::loaders::gdef_store(&font.gdef, &font.coords);
        let Some(marks) =
            crate::loaders::mark::Marks::parse(&font.gpos, &font.gdef, store.as_ref())
        else {
            return blank();
        };
        let ids: Vec<u16> = glyphs.iter().map(|g| u16::try_from(*g).unwrap_or(0)).collect();
        let scale = size.max(1.0) / font.units_per_em.max(1) as f32;
        marks
            .attach(&ids)
            .into_iter()
            .map(|a| crate::loaders::mark::Attachment {
                to: a.to,
                dx: a.dx * scale,
                dy: a.dy * scale,
            })
            .collect()
    }

    /// One run of characters shaped into glyph ids, as `(id, source character index)`.
    ///
    /// `script` is the OpenType tag, and it matters: a feature tag means different lookups
    /// under different scripts, and taking the union draws the wrong glyph. The characters
    /// must be in **logical** order — joining asks what precedes and follows a letter.
    ///
    /// A face with no glyph ids reports the identity, which is what makes one layout path
    /// serve all four face kinds: the built-in bitmap face cannot be shaped and does not
    /// need to be, since nothing it can draw joins.
    #[must_use]
    pub fn shape_run(&self, chars: &[char], script: &[u8; 4], size: f32) -> Vec<(u32, usize)> {
        let Face::Ttf(face) = &self.face else {
            return chars.iter().enumerate().map(|(i, c)| (*c as u32, i)).collect();
        };
        let font = face.at(self.optical_size.unwrap_or(size));
        let gsub = crate::loaders::shape::gsub::Gsub::parse(&font.gsub);
        crate::loaders::shape::shape(
            // A character the face does not cover becomes glyph **0**, not nothing. The
            // shaper this came from drops it, which is right there — it sits behind a
            // fallback cascade that will try the next face. A `CaptionFont` is one face,
            // so dropping it would make the text silently vanish where it used to draw a
            // notdef box. Glyph 0 is in no lookup's coverage, so shaping leaves it alone.
            |c| Some(font.cmap.get(&(c as u32)).copied().unwrap_or(0)),
            gsub.as_ref(),
            script,
            chars,
        )
        .into_iter()
        .map(|s| (u32::from(s.glyph), s.cluster))
        .collect()
    }

    /// Whether this face can be shaped: whether [`Self::gid`] means a glyph id.
    #[must_use]
    pub fn is_shapeable(&self) -> bool {
        matches!(&self.face, Face::Ttf(_))
    }

    /// Whether two glyphs of a run that share a cell take the **maximum** of their
    /// coverage rather than its union.
    ///
    /// True for a face whose letters join, where both glyphs cover the same stroke and a
    /// union darkens every joint. The built-in face is Latin, so it takes the union. See
    /// [`crate::loaders::overlap`] for the rule and how it was read.
    pub fn composites_run(&self) -> bool {
        match &self.face {
            Face::Ttf(f) => f.base.composites_run,
            // The built-in face is Latin, and an SDF or bitmap face has no table to ask.
            Face::Builtin | Face::Sdf(_) | Face::Bitmap(_) => false,
        }
    }

    /// The advance of one glyph, by the index [`Self::gid`] gives.
    pub fn advance_of(&mut self, gid: u32, size: f32) -> f32 {
        let size = size.max(1.0);
        match &self.face {
            Face::Builtin => size,
            Face::Ttf(face) => {
                let optical = self.optical_size.unwrap_or(size);
                let font = face.at(optical);
                font.glyphs.get(gid as usize).map_or(0.0, |g| {
                    (g.advance_width as f32 + font.tracking_at(optical)) * size
                        / font.units_per_em.max(1) as f32
                        + 2.0 * face.synthesis(size).outset
                })
            }
            Face::Sdf(set) => {
                set.glyphs.get(&gid).map_or(0.0, |g| g.advance * size / set.size.max(1.0))
            }
            Face::Bitmap(set) => {
                set.glyphs.get(&gid).map_or(0.0, |g| g.advance * size / set.size.max(1e-6))
            }
        }
    }

    /// The kerning between two glyphs, by the index [`Self::gid`] gives.
    ///
    /// `left` and `right` are in **logical** order, which for a reversed run is the
    /// reverse of what is on the screen — see [`super::order::pair`].
    pub fn kerning_of(&self, left: u32, right: u32, size: f32) -> f32 {
        let Face::Ttf(face) = &self.face else {
            return 0.0;
        };
        let font = face.at(self.optical_size.unwrap_or(size));
        if font.kerning.is_empty() && font.gpos_kerning.is_empty() {
            return 0.0;
        }
        let narrow = |g: u32| u16::try_from(g).unwrap_or(0);
        let units = font.kern(narrow(left), narrow(right));
        if units == 0.0 {
            return 0.0;
        }
        units * size.max(1.0) / font.units_per_em.max(1) as f32
    }

    /// Whether this face has its own glyph for `c` rather than a stand-in —
    /// what a chain of fallback faces asks of each in turn.
    pub fn has_glyph(&self, c: char) -> bool {
        match &self.face {
            // Anything it cannot draw comes back as the missing-glyph box.
            Face::Builtin => builtin_rows(c).is_some_and(|rows| rows != TOFU),
            Face::Ttf(face) => face.base.cmap.contains_key(&(c as u32)),
            Face::Sdf(set) => set.glyphs.contains_key(&(c as u32)),
            Face::Bitmap(set) => set.glyphs.contains_key(&(c as u32)),
        }
    }

    /// `c`'s outline at `size` pixels per em, flattened, in pixels with y
    /// down and the pen at the origin: per contour, per curve, the points from
    /// its start to its end. What a multi-channel distance field needs to find
    /// corners. `None` for the bitmap face, which has no outline.
    pub fn outline(&self, c: char, size: f32) -> Option<Vec<Vec<Vec<[f32; 2]>>>> {
        let Face::Ttf(face) = &self.face else {
            return None;
        };
        let font = face.at(self.optical_size.unwrap_or(size));
        let gid = font.cmap.get(&(c as u32)).copied().unwrap_or(0);
        let glyph = font.glyphs.get(gid as usize)?;
        let scale = size.max(1.0) / font.units_per_em.max(1) as f32;
        // Slanted for a synthetic italic, and moved over by a synthetic
        // bold's growth as its raster is; the growth itself is not an
        // outline — see [`synthetic_outset`](Self::synthetic_outset).
        let syn = face.synthesis(size);
        let mut curves = flatten_curves(glyph, scale, syn.skew);
        for p in curves.iter_mut().flatten().flatten() {
            p[0] += syn.outset;
        }
        Some(curves)
    }

    /// How far a synthetic bold grows each outline at `size`, in pixels;
    /// 0 for a face drawn as it is. What a caller building its own shapes
    /// from [`outline`](Self::outline) — a distance field, say — grows them
    /// by to match this face's rasters.
    pub fn synthetic_outset(&self, size: f32) -> f32 {
        match &self.face {
            Face::Ttf(face) => face.synthesis(size.max(1.0)).outset,
            _ => 0.0,
        }
    }

    /// Pen advance for `c` at `size` pixels, from the metrics alone — the
    /// same as the glyph's raster carries, without rasterizing it.
    pub fn advance(&mut self, c: char, size: f32) -> f32 {
        let gid = self.gid(c);
        self.advance_of(gid, size)
    }

    /// Total advance of `text` at `size` pixels, including `letter_spacing`
    /// between glyphs.
    pub fn measure(&mut self, text: &str, size: f32, letter_spacing: f32) -> f32 {
        let mut width = 0.0;
        let mut count = 0usize;
        let mut previous: Option<super::order::Placed> = None;
        // In drawing order, not as typed: composed, reordered and mirrored by
        // [`super::order::visual`]. Measuring the typed characters instead counts a
        // combining accent as a letter of its own, which puts every centred line off by
        // the width of a floating diaeresis. This must stay in step with the glyph walk
        // in `raster`, which asks the same function.
        // Resolved first, so the immutable borrow of `self` ends before the loop needs
        // a mutable one for the glyph cache.
        let run = super::order::visual(text, self, size);
        for placed in run {
            // Kerning is part of the width. Measuring without it and drawing
            // with it puts every centred or right-aligned line off by the sum of
            // its pairs — the text would sit a little to one side, by an amount
            // that changes with the words.
            if let Some(p) = previous {
                let (l, r) = super::order::pair(p, placed);
                width += self.kerning_of(l, r, size);
            }
            width += self.advance_of(placed.gid, size);
            previous = Some(placed);
            // Letter spacing is between **letters**. A mark's advance is zero and it sits
            // on the letter before it, so charging it a gap would push the word apart —
            // and the raster walk skips it the same way, which is what keeps a centred
            // line centred.
            count += usize::from(placed.attach.is_none());
        }
        if count > 1 {
            width += letter_spacing * (count - 1) as f32;
        }
        width
    }
}

// ---------------------------------------------------------------------------
// Built-in 8×8 bitmap face
// ---------------------------------------------------------------------------

/// How many horizontal positions within a pixel a glyph is rasterized for.
///
/// Four — quarter-pixel steps. Fewer brings back the uneven spacing this exists
/// to remove; more costs cache entries for a difference nobody can see.
pub const SUBPIXEL_PHASES: usize = 4;

/// Rows per glyph cell (also the em size in font units).
const BUILTIN_CELL: u32 = 8;
/// Rows of the cell that sit above the baseline; row 7 is the descender.
const BUILTIN_ASCENT_ROWS: u32 = 7;

/// Rasterize a built-in glyph, computing each pixel's coverage exactly.
///
/// The source is a bitmap, so every lit cell is an axis-aligned rectangle in
/// device space and a pixel's coverage is the *exact* sum of its overlaps with
/// them — no sampling needed. The previous version counted 4×4 sample centres
/// per pixel, which quantizes coverage to seventeen levels and steps visibly
/// along a diagonal; this gives all 256 and is cheaper at large sizes, because
/// the work is per overlapping cell rather than per sample.
fn raster_builtin(c: char, size: f32, shift: f32) -> GlyphBitmap {
    let scale = size / BUILTIN_CELL as f32;
    let advance = BUILTIN_CELL as f32 * scale;
    let rows = builtin_rows(c);
    let Some(rows) = rows else {
        return GlyphBitmap {
            advance,
            ..Default::default()
        };
    };
    if rows.iter().all(|&r| r == 0) {
        return GlyphBitmap {
            advance,
            ..Default::default()
        };
    }

    // Cell-space ink bounds, so the mask is no bigger than the glyph.
    let mut min_col = BUILTIN_CELL;
    let mut max_col = 0u32;
    let mut min_row = BUILTIN_CELL;
    let mut max_row = 0u32;
    for (y, &bits) in rows.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        min_row = min_row.min(y as u32);
        max_row = max_row.max(y as u32);
        for x in 0..BUILTIN_CELL {
            if bits & (1 << x) != 0 {
                min_col = min_col.min(x);
                max_col = max_col.max(x);
            }
        }
    }

    let x0 = min_col as f32 * scale + shift;
    let y0 = min_row as f32 * scale;
    let px0 = x0.floor() as i32;
    let py0 = y0.floor() as i32;
    let px1 = ((max_col + 1) as f32 * scale + shift).ceil() as i32;
    let py1 = ((max_row + 1) as f32 * scale).ceil() as i32;
    let width = (px1 - px0).max(1) as u32;
    let height = (py1 - py0).max(1) as u32;

    // Exact area coverage. Lit cell (cx, cy) covers
    // `[cx*scale + shift, (cx+1)*scale + shift) × [cy*scale, (cy+1)*scale)`, so
    // a pixel's coverage is the sum over lit cells of the overlap in x times the
    // overlap in y. Both are one-dimensional interval intersections, which makes
    // the whole thing exact rather than sampled.
    let mut coverage = vec![0u8; (width * height) as usize];
    let overlap = |a0: f32, a1: f32, b0: f32, b1: f32| (a1.min(b1) - a0.max(b0)).max(0.0);
    for (cy, &bits) in rows.iter().enumerate() {
        if bits == 0 {
            continue;
        }
        let cell_top = cy as f32 * scale;
        let cell_bottom = cell_top + scale;
        // Only the pixel rows this cell reaches.
        let first_row = ((cell_top.floor() as i32) - py0).max(0) as u32;
        let last_row = (((cell_bottom.ceil() as i32) - py0).max(0) as u32).min(height);
        for cx in 0..BUILTIN_CELL {
            if bits & (1 << cx) == 0 {
                continue;
            }
            let cell_left = cx as f32 * scale + shift;
            let cell_right = cell_left + scale;
            let first_col = ((cell_left.floor() as i32) - px0).max(0) as u32;
            let last_col = (((cell_right.ceil() as i32) - px0).max(0) as u32).min(width);
            for py in first_row..last_row {
                let top = (py0 + py as i32) as f32;
                let dy = overlap(top, top + 1.0, cell_top, cell_bottom);
                if dy <= 0.0 {
                    continue;
                }
                for px in first_col..last_col {
                    let left = (px0 + px as i32) as f32;
                    let dx = overlap(left, left + 1.0, cell_left, cell_right);
                    if dx <= 0.0 {
                        continue;
                    }
                    let slot = &mut coverage[(py * width + px) as usize];
                    // Cells never overlap each other, so the areas add. Saturate
                    // rather than wrap: rounding can push a full pixel to 256.
                    *slot = slot.saturating_add((dx * dy * 255.0).round() as u8);
                }
            }
        }
    }

    GlyphBitmap {
        width,
        height,
        coverage,
        offset_x: px0,
        // The baseline sits BUILTIN_ASCENT_ROWS down the cell.
        offset_y: py0 - (BUILTIN_ASCENT_ROWS as f32 * scale).round() as i32,
        advance,
    }
}

/// The 8 row bitmaps for `c` (bit 0 = leftmost pixel), or `None` when the
/// codepoint has no built-in glyph.
fn builtin_rows(c: char) -> Option<[u8; 8]> {
    let folded = fold_to_ascii(c);
    let code = folded as u32;
    if (0x20..=0x7E).contains(&code) {
        return Some(FONT8X8[(code - 0x20) as usize]);
    }
    match code {
        // Control characters and the ones layout handles itself draw nothing.
        0x09 | 0x0A | 0x0D | 0xA0 => Some([0; 8]),
        // Everything else gets a "missing glyph" box, as fonts conventionally do.
        _ => Some(TOFU),
    }
}

/// Hollow box drawn for codepoints the built-in face cannot render.
const TOFU: [u8; 8] = [0x00, 0x3E, 0x22, 0x22, 0x22, 0x22, 0x3E, 0x00];

/// Map a Latin-1 / Latin-Extended-A letter to its unaccented ASCII base so
/// European text stays legible in the built-in face. Load a TrueType face for
/// real diacritics.
fn fold_to_ascii(c: char) -> char {
    match c {
        'À'..='Å' => 'A',
        'Æ' => 'A',
        'Ç' => 'C',
        'È'..='Ë' => 'E',
        'Ì'..='Ï' => 'I',
        'Ð' => 'D',
        'Ñ' => 'N',
        'Ò'..='Ö' | 'Ø' => 'O',
        'Ù'..='Ü' => 'U',
        'Ý' => 'Y',
        'ß' => 's',
        'à'..='å' => 'a',
        'æ' => 'a',
        'ç' => 'c',
        'è'..='ë' => 'e',
        'ì'..='ï' => 'i',
        'ñ' => 'n',
        'ò'..='ö' | 'ø' => 'o',
        'ù'..='ü' => 'u',
        'ý' | 'ÿ' => 'y',
        'Š' => 'S',
        'š' => 's',
        'Ž' => 'Z',
        'ž' => 'z',
        'Œ' => 'O',
        'œ' => 'o',
        // Typographic punctuation the ASCII cell can stand in for.
        '\u{2018}' | '\u{2019}' => '\'',
        '\u{201C}' | '\u{201D}' => '"',
        '\u{2013}' | '\u{2014}' => '-',
        '\u{2026}' => '.',
        _ => c,
    }
}

/// Public-domain 8×8 fixed face for `0x20..=0x7E`. Each entry is 8 rows,
/// top to bottom; within a row, bit 0 is the leftmost pixel.
#[rustfmt::skip]
const FONT8X8: [[u8; 8]; 95] = [
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // ' '
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00], // '!'
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '"'
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00], // '#'
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00], // '$'
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00], // '%'
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00], // '&'
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00], // '\''
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00], // '('
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00], // ')'
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00], // '*'
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00], // '+'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ','
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00], // '-'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00], // '.'
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00], // '/'
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00], // '0'
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00], // '1'
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00], // '2'
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00], // '3'
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00], // '4'
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00], // '5'
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00], // '6'
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00], // '7'
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00], // '8'
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00], // '9'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00], // ':'
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06], // ';'
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00], // '<'
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00], // '='
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00], // '>'
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00], // '?'
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00], // '@'
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00], // 'A'
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00], // 'B'
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00], // 'C'
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00], // 'D'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00], // 'E'
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00], // 'F'
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00], // 'G'
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00], // 'H'
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // 'I'
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00], // 'J'
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00], // 'K'
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00], // 'L'
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00], // 'M'
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00], // 'N'
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00], // 'O'
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00], // 'P'
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00], // 'Q'
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00], // 'R'
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00], // 'S'
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // 'T'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00], // 'U'
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // 'V'
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00], // 'W'
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00], // 'X'
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00], // 'Y'
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00], // 'Z'
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00], // '['
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00], // '\\'
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00], // ']'
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00], // '^'
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF], // '_'
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00], // '`'
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00], // 'a'
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00], // 'b'
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00], // 'c'
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00], // 'd'
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00], // 'e'
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00], // 'f'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F], // 'g'
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00], // 'h'
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // 'i'
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E], // 'j'
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00], // 'k'
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00], // 'l'
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00], // 'm'
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00], // 'n'
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00], // 'o'
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F], // 'p'
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78], // 'q'
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00], // 'r'
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00], // 's'
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00], // 't'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00], // 'u'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00], // 'v'
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00], // 'w'
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00], // 'x'
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F], // 'y'
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00], // 'z'
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00], // '{'
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00], // '|'
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00], // '}'
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00], // '~'
];

// ---------------------------------------------------------------------------
// TrueType outline rasterization
// ---------------------------------------------------------------------------

/// Sub-scanlines per pixel row when filling outlines.
/// Vertical samples per pixel row when scan-converting an outline.
///
/// Horizontal coverage is exact — [`add_span`] gives the end pixels their
/// fractional area — so only the vertical axis is sampled, and four samples
/// quantize it to five levels. That shows as stepping along any nearly
/// horizontal edge, which in practice means the top and bottom of every round
/// letter. Sixteen is the usual choice and costs nothing at runtime: glyphs are
/// rasterized once per size and phase, then cached.
const SUBSAMPLES: usize = 16;

/// Rasterize `c` from `font` at `size` pixels per em.
/// `optical`: the size whose tracking applies (see [`CaptionFont::set_optical_size`]);
/// `syn`, a synthetic bold or italic's changes to the outline.
fn raster_ttf(font: &TtfFont, gid: u32, size: f32, shift: f32, optical: f32, syn: Synthesis) -> GlyphBitmap {
    if syn.outset > 0.0 {
        // Grown about the outline, then moved over by the growth so the
        // left side bearing stays; the advance takes both sides.
        let mut g = raster_grown(font, gid, size, shift + syn.outset, syn.outset, optical, syn);
        g.advance += 2.0 * syn.outset;
        return g;
    }
    let upem = font.units_per_em.max(1) as f32;
    let scale = size / upem;
    let Some(glyph) = font.glyphs.get(gid as usize) else {
        return GlyphBitmap::default();
    };
    let advance = (glyph.advance_width as f32 + font.tracking_at(optical)) * scale;

    // Flatten the outline into device-space contours (y grows downward, so the
    // font's y axis is negated).
    // The subpixel phase is applied to the outline itself, so the coverage is
    // exact for the position the glyph will actually sit at — shifting a
    // finished mask could only resample it, which is what blurs subpixel-placed
    // text in a naive implementation.
    let warp = syn.hint.then(|| VerticalWarp::new(font, size));
    let mut contours = flatten_contours(glyph, scale, shift, syn.skew, warp.as_ref());
    if syn.darken != [0.0; 2] {
        dilate_contours(&mut contours, syn.darken);
    }
    let edges: Vec<Edge> = contours
        .iter()
        .flat_map(|c| c.windows(2).map(|w| Edge::new(w[0], w[1])))
        .filter(|e| e.y0 != e.y1)
        .collect();
    if edges.is_empty() {
        return GlyphBitmap {
            advance,
            ..Default::default()
        };
    }

    let (mut min_x, mut min_y) = (f32::MAX, f32::MAX);
    let (mut max_x, mut max_y) = (f32::MIN, f32::MIN);
    for e in &edges {
        min_x = min_x.min(e.x_at_y0.min(e.x_at_y1));
        max_x = max_x.max(e.x_at_y0.max(e.x_at_y1));
        min_y = min_y.min(e.y0.min(e.y1));
        max_y = max_y.max(e.y0.max(e.y1));
    }
    let px0 = min_x.floor() as i32;
    let py0 = min_y.floor() as i32;
    let width = ((max_x.ceil() as i32) - px0).max(1) as u32;
    let height = ((max_y.ceil() as i32) - py0).max(1) as u32;
    // Guard against pathological outlines blowing up memory.
    if width > 8192 || height > 8192 {
        return GlyphBitmap {
            advance,
            ..Default::default()
        };
    }

    let mut coverage = vec![0u8; (width * height) as usize];
    let mut row = vec![0f32; width as usize];
    let mut crossings: Vec<(f32, i32)> = Vec::new();

    for py in 0..height {
        row.iter_mut().for_each(|v| *v = 0.0);
        for s in 0..SUBSAMPLES {
            let y = (py0 + py as i32) as f32 + (s as f32 + 0.5) / SUBSAMPLES as f32;
            crossings.clear();
            for e in &edges {
                if let Some(x) = e.x_at(y) {
                    crossings.push((x - px0 as f32, e.winding));
                }
            }
            if crossings.len() < 2 {
                continue;
            }
            crossings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            // Non-zero winding: a span is inside while the accumulated winding
            // is not zero (this is what TrueType's contour directions encode).
            let mut winding = 0;
            let mut span_start = 0.0f32;
            for &(x, dir) in &crossings {
                if winding == 0 {
                    span_start = x;
                }
                winding += dir;
                if winding == 0 {
                    add_span(&mut row, span_start, x);
                }
            }
        }
        for px in 0..width as usize {
            let v = (row[px] / SUBSAMPLES as f32 * 255.0).round();
            coverage[(py * width) as usize + px] = v.clamp(0.0, 255.0) as u8;
        }
    }

    GlyphBitmap {
        width,
        height,
        coverage,
        offset_x: px0,
        offset_y: py0,
        advance,
    }
}

/// `c` grown by `radius`: per pixel, its centre's signed distance to the
/// outline (negative inside) as a ramp one pixel wide at `radius`, which for
/// a straight edge is exactly the pixel's share of the grown shape.
fn raster_grown(font: &TtfFont, gid: u32, size: f32, shift: f32, radius: f32, optical: f32, syn: Synthesis) -> GlyphBitmap {
    let scale = size / font.units_per_em.max(1) as f32;
    let Some(glyph) = font.glyphs.get(gid as usize) else {
        return GlyphBitmap::default();
    };
    let advance = (glyph.advance_width as f32 + font.tracking_at(optical)) * scale;
    let warp = syn.hint.then(|| VerticalWarp::new(font, size));
    let mut contours = flatten_contours(glyph, scale, shift, syn.skew, warp.as_ref());
    if syn.darken != [0.0; 2] {
        dilate_contours(&mut contours, syn.darken);
    }
    let segments: Vec<((f32, f32), (f32, f32))> = contours
        .iter()
        .flat_map(|c| c.windows(2).map(|w| (w[0], w[1])))
        .filter(|(a, b)| a != b)
        .collect();
    if segments.is_empty() {
        return GlyphBitmap { advance, ..Default::default() };
    }
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for &(a, _) in &segments {
        min_x = min_x.min(a.0);
        max_x = max_x.max(a.0);
        min_y = min_y.min(a.1);
        max_y = max_y.max(a.1);
    }
    let reach = radius + 1.0;
    let x0 = (min_x - reach).floor() as i32;
    let y0 = (min_y - reach).floor() as i32;
    let width = ((max_x + reach).ceil() as i32 - x0).max(1) as u32;
    let height = ((max_y + reach).ceil() as i32 - y0).max(1) as u32;
    if width > 8192 || height > 8192 {
        return GlyphBitmap { advance, ..Default::default() };
    }
    // Two by two samples a pixel, each a ramp half a pixel wide: one sample
    // at the centre gets a straight edge exactly but a corner, where the
    // nearest point turns, up to a third of a pixel wrong.
    const N: u32 = 2;
    let mut acc = vec![0f32; (width * height) as usize];
    let mut crossings: Vec<(f32, i32)> = Vec::new();
    for sy in 0..height * N {
        let y = y0 as f32 + (sy as f32 + 0.5) / N as f32;
        // Inside by non-zero winding along this row of samples.
        crossings.clear();
        for &(a, b) in &segments {
            if (a.1 <= y) != (b.1 <= y) {
                let t = (y - a.1) / (b.1 - a.1);
                crossings.push((a.0 + t * (b.0 - a.0), if b.1 > a.1 { 1 } else { -1 }));
            }
        }
        crossings.sort_by(|p, q| p.0.total_cmp(&q.0));
        let mut k = 0;
        let mut winding = 0;
        // The segments a sample on this row can be within `reach` of.
        let near_row: Vec<((f32, f32), (f32, f32))> = segments
            .iter()
            .copied()
            .filter(|(a, b)| a.1.min(b.1) - reach <= y && a.1.max(b.1) + reach >= y)
            .collect();
        for sx in 0..width * N {
            let x = x0 as f32 + (sx as f32 + 0.5) / N as f32;
            while k < crossings.len() && crossings[k].0 < x {
                winding += crossings[k].1;
                k += 1;
            }
            let mut d2 = f32::MAX;
            for &(a, b) in &near_row {
                let (dx, dy) = (b.0 - a.0, b.1 - a.1);
                let t = (((x - a.0) * dx + (y - a.1) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                let (ex, ey) = (a.0 + t * dx - x, a.1 + t * dy - y);
                d2 = d2.min(ex * ex + ey * ey);
            }
            let d = d2.sqrt();
            let signed = if winding != 0 { -d } else { d };
            let v = ((radius - signed) * N as f32 + 0.5).clamp(0.0, 1.0);
            acc[((sy / N) * width + sx / N) as usize] += v;
        }
    }
    let k = 1.0 / (N * N) as f32;
    let coverage = acc.iter().map(|v| (v * k * 255.0).round() as u8).collect();
    GlyphBitmap { width, height, coverage, offset_x: x0, offset_y: y0, advance }
}

/// Accumulate horizontal coverage for the span `[x0, x1)` into `row`, giving
/// the partially-covered end pixels their exact fractional area.
fn add_span(row: &mut [f32], x0: f32, x1: f32) {
    let w = row.len() as f32;
    let x0 = x0.clamp(0.0, w);
    let x1 = x1.clamp(0.0, w);
    if x1 <= x0 {
        return;
    }
    let i0 = x0.floor() as usize;
    let i1 = x1.floor() as usize;
    if i0 >= row.len() {
        return;
    }
    if i0 == i1 {
        row[i0] += x1 - x0;
        return;
    }
    row[i0] += (i0 + 1) as f32 - x0;
    let full_end = i1.min(row.len());
    for cell in row.iter_mut().take(full_end).skip(i0 + 1) {
        *cell += 1.0;
    }
    if i1 < row.len() {
        row[i1] += x1 - i1 as f32;
    }
}

/// One flattened outline segment in device space.
struct Edge {
    y0: f32,
    y1: f32,
    x_at_y0: f32,
    x_at_y1: f32,
    /// `+1` when the edge runs downward, `-1` upward — the non-zero winding sign.
    winding: i32,
}

impl Edge {
    fn new(a: (f32, f32), b: (f32, f32)) -> Self {
        Self {
            y0: a.1,
            y1: b.1,
            x_at_y0: a.0,
            x_at_y1: b.0,
            winding: if b.1 > a.1 { 1 } else { -1 },
        }
    }

    /// X where the edge crosses horizontal line `y`, using a half-open y range
    /// so a shared vertex is counted exactly once.
    fn x_at(&self, y: f32) -> Option<f32> {
        let (top, bottom) = if self.y0 < self.y1 {
            (self.y0, self.y1)
        } else {
            (self.y1, self.y0)
        };
        if y < top || y >= bottom {
            return None;
        }
        let t = (y - self.y0) / (self.y1 - self.y0);
        Some(self.x_at_y0 + t * (self.x_at_y1 - self.x_at_y0))
    }
}

/// Flatten a glyph's curves into closed device-space contours.
///
/// [`crate::loaders::TtfFont`] stores every contour of a glyph in one `Path`,
/// starting each with a `move_to`. `move_to` adds no curve, so a contour break
/// shows up as a gap between one curve's end point and the next one's start.
///
/// `skew` slants it: x moves by `skew` times the height above the baseline.
/// `warp`, when hinting, maps heights onto the pixel grid instead of scaling
/// them.
fn flatten_contours(
    glyph: &crate::loaders::TtfGlyph,
    scale: f32,
    shift: f32,
    skew: f32,
    warp: Option<&VerticalWarp>,
) -> Vec<Vec<(f32, f32)>> {
    let mut contours: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut current: Vec<(f32, f32)> = Vec::new();
    let mut last: Option<(f32, f32)> = None;

    // Device space: x scaled, y negated (font y is up, raster y is down).
    let to_device = |p: crate::math::Vector2| {
        let y = warp.map_or(p.y * scale, |w| w.map(p.y));
        ((p.x + skew * p.y) * scale + shift, -y)
    };

    for curve in glyph.shape.outline.curve_path.curves.iter() {
        let start = to_device(curve.get_point(0.0));
        let end = to_device(curve.get_point(1.0));
        let breaks = match last {
            Some(prev) => (prev.0 - start.0).abs() > 1e-4 || (prev.1 - start.1).abs() > 1e-4,
            None => true,
        };
        if breaks {
            if current.len() > 2 {
                close_contour(&mut current);
                contours.push(std::mem::take(&mut current));
            } else {
                current.clear();
            }
            current.push(start);
        }
        // Straight segments need no subdivision; curves get enough segments for
        // their on-screen size.
        let steps = if is_straight(curve.as_ref()) {
            1
        } else {
            let span = (end.0 - start.0).abs().max((end.1 - start.1).abs());
            (span.sqrt() * 2.0).ceil().clamp(3.0, 24.0) as usize
        };
        for i in 1..=steps {
            current.push(to_device(curve.get_point(i as f32 / steps as f32)));
        }
        last = Some(end);
    }
    if current.len() > 2 {
        close_contour(&mut current);
        contours.push(current);
    }
    contours
}

/// [`flatten_contours`] keeping each curve's points apart, so the joints
/// between curves — where a glyph's corners are — stay visible.
fn flatten_curves(glyph: &crate::loaders::TtfGlyph, scale: f32, skew: f32) -> Vec<Vec<Vec<[f32; 2]>>> {
    let mut contours: Vec<Vec<Vec<[f32; 2]>>> = Vec::new();
    let mut current: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut last: Option<[f32; 2]> = None;
    let to_device = |p: crate::math::Vector2| [(p.x + skew * p.y) * scale, -p.y * scale];
    for curve in glyph.shape.outline.curve_path.curves.iter() {
        let start = to_device(curve.get_point(0.0));
        let end = to_device(curve.get_point(1.0));
        let breaks = match last {
            Some(prev) => (prev[0] - start[0]).abs() > 1e-4 || (prev[1] - start[1]).abs() > 1e-4,
            None => true,
        };
        if breaks && !current.is_empty() {
            contours.push(std::mem::take(&mut current));
        }
        let steps = if is_straight(curve.as_ref()) {
            1
        } else {
            let span = (end[0] - start[0]).abs().max((end[1] - start[1]).abs());
            (span.sqrt() * 2.0).ceil().clamp(3.0, 24.0) as usize
        };
        let points: Vec<[f32; 2]> = (0..=steps)
            .map(|i| to_device(curve.get_point(i as f32 / steps as f32)))
            .collect();
        if (end[0] - start[0]).abs() + (end[1] - start[1]).abs() > 1e-6 || steps > 1 {
            current.push(points);
        }
        last = Some(end);
    }
    if !current.is_empty() {
        contours.push(current);
    }
    contours
}

/// Whether the curve's midpoint lies on the chord between its endpoints — true
/// for line segments, false for the quadratics TrueType uses. Compared in font
/// units, where a 1e-3 tolerance is far below one pixel.
fn is_straight(curve: &dyn Curve2) -> bool {
    let a = curve.get_point(0.0);
    let b = curve.get_point(1.0);
    let mid = curve.get_point(0.5);
    let cx = (a.x + b.x) * 0.5;
    let cy = (a.y + b.y) * 0.5;
    (mid.x - cx).abs() < 1e-3 && (mid.y - cy).abs() < 1e-3
}

/// Repeat the first point so `windows(2)` yields the closing edge.
fn close_contour(points: &mut Vec<(f32, f32)>) {
    if let Some(&first) = points.first() {
        if points.last() != Some(&first) {
            points.push(first);
        }
    }
}

#[cfg(test)]
mod tests {
    /// A kerned pair measures narrower than its glyphs do apart.
    ///
    /// And the built-in face, which is fixed-width, measures the same either
    /// way — so this is checking that kerning is applied where a font has it and
    /// not invented where it does not.
    ///
    /// Skipped where the font is not installed.
    #[test]
    fn kerning_narrows_a_measured_pair() {
        let mut builtin = CaptionFont::builtin();
        let apart = builtin.advance('A', 20.0) + builtin.advance('V', 20.0);
        assert!(
            (builtin.measure("AV", 20.0, 0.0) - apart).abs() < 1e-3,
            "the built-in face has no pairs to kern"
        );

        let Ok(bytes) = std::fs::read("/System/Library/Fonts/Supplemental/Arial Bold.ttf") else {
            eprintln!("skipping: Arial Bold not installed");
            return;
        };
        let Ok(mut font) = CaptionFont::from_ttf_bytes(&bytes) else {
            return;
        };
        let size = 32.0;
        let apart = font.advance('A', size) + font.advance('V', size);
        let together = font.measure("AV", size, 0.0);
        assert!(
            together < apart - 0.5,
            "AV measured {together}, no tighter than {apart} apart"
        );
        // A pair with no kerning measures exactly its advances.
        let loose = font.advance('o', size) + font.advance('o', size);
        assert!(
            (font.measure("oo", size, 0.0) - loose).abs() < 1e-3,
            "oo should need no adjustment"
        );
        // And the adjustment is the font's, in pixels at this size.
        let expected = font.kerning('A', 'V', size);
        assert!(
            expected < 0.0,
            "AV kerning should be negative, got {expected}"
        );
        assert!(
            ((together - apart) - expected).abs() < 1e-3,
            "measure applied {} where the pair says {expected}",
            together - apart
        );
    }

    /// Coverage is the exact area, so ink is conserved whatever the scale.
    ///
    /// A lit cell of a bitmap glyph occupies `scale × scale` pixels of area, so
    /// the coverage summed over the mask must come to `lit_cells × scale² × 255`
    /// — a property of computing the area rather than sampling it. The previous
    /// 4×4 sampler could only ever approximate this, and at awkward scales it
    /// was out by several percent.
    #[test]
    fn builtin_coverage_conserves_ink_at_any_scale() {
        for size in [7.3f32, 11.0, 13.7, 19.5, 31.25] {
            let scale = size / BUILTIN_CELL as f32;
            let rows = builtin_rows('H').expect("H has a glyph");
            let lit: u32 = rows.iter().map(|r| r.count_ones()).sum();
            let expected = lit as f32 * scale * scale * 255.0;

            let glyph = raster_builtin('H', size, 0.0);
            let total: f32 = glyph.coverage.iter().map(|v| *v as f32).sum();
            // Within the rounding of one byte per pixel.
            let slack = (glyph.coverage.len() as f32) * 0.5 + 1.0;
            assert!(
                (total - expected).abs() <= slack,
                "size {size}: ink {total} vs exact {expected} (slack {slack})"
            );
        }
    }

    /// At a whole-pixel scale every pixel is fully in or fully out.
    ///
    /// Cells land exactly on pixel boundaries, so there is no partial coverage
    /// to report. Anything between 0 and 255 here would mean the area
    /// computation had leaked across an edge.
    #[test]
    fn builtin_coverage_is_binary_when_cells_align() {
        let glyph = raster_builtin('H', BUILTIN_CELL as f32 * 3.0, 0.0);
        for &v in &glyph.coverage {
            assert!(
                v == 0 || v == 255,
                "partial coverage {v} at an integer scale"
            );
        }
    }

    /// A subpixel phase moves the ink without changing how much there is.
    ///
    /// This is the property that makes phases usable: the same glyph at a
    /// different quarter-pixel offset is the same shape, so the total coverage must
    /// match while the distribution differs. If phases changed the amount of
    /// ink, text would visibly pulse as words moved.
    #[test]
    fn a_phase_shifts_the_glyph_without_changing_its_weight() {
        let size = 13.7f32;
        let base = raster_builtin('H', size, 0.0);
        let base_ink: u32 = base.coverage.iter().map(|v| *v as u32).sum();
        let mut differed = false;
        for phase in 1..SUBPIXEL_PHASES {
            let shifted = raster_builtin('H', size, phase as f32 / SUBPIXEL_PHASES as f32);
            let ink: u32 = shifted.coverage.iter().map(|v| *v as u32).sum();
            let slack = (base.coverage.len().max(shifted.coverage.len()) as u32) + 255;
            assert!(
                base_ink.abs_diff(ink) <= slack,
                "phase {phase}: ink {ink} vs {base_ink}"
            );
            if shifted.coverage != base.coverage {
                differed = true;
            }
        }
        assert!(
            differed,
            "no phase changed the raster, so phases do nothing"
        );
    }

    /// Each phase moves the glyph by its own fraction of a pixel.
    ///
    /// This is the measurement that says subpixel positioning works at all. The
    /// centroid of a glyph's coverage sits wherever the shape does, to a
    /// fraction of a pixel — a stem whose edge falls two thirds across a pixel
    /// puts two thirds of the ink there. So stepping the phase must step the
    /// centroid by a quarter of a pixel each time, and a renderer that rounded
    /// the position would show the same centroid four times and then a jump.
    #[test]
    fn stepping_the_phase_steps_the_glyph_by_a_quarter_pixel() {
        let size = 16.0f32;
        let centroid = |phase: usize| {
            let g = raster_builtin('H', size, phase as f32 / SUBPIXEL_PHASES as f32);
            let mut weight = 0.0f32;
            let mut moment = 0.0f32;
            for row in 0..g.height {
                for col in 0..g.width {
                    let v = g.coverage[(row * g.width + col) as usize] as f32;
                    // Absolute position, so the mask's own origin cannot hide a
                    // shift by moving with it.
                    moment += (g.offset_x + col as i32) as f32 * v;
                    weight += v;
                }
            }
            moment / weight
        };
        let steps: Vec<f32> = (0..SUBPIXEL_PHASES).map(centroid).collect();
        for (i, pair) in steps.windows(2).enumerate() {
            let moved = pair[1] - pair[0];
            let expected = 1.0 / SUBPIXEL_PHASES as f32;
            assert!(
                (moved - expected).abs() < 0.06,
                "phase {i} to {}: centroid moved {moved}, expected {expected}",
                i + 1
            );
        }
    }

    /// The cache keeps phases apart.
    ///
    /// Keyed on the character and size alone, the first phase rasterized would
    /// be returned for every position — which is the bug this replaced, wearing
    /// a cache.
    #[test]
    fn the_glyph_cache_distinguishes_phases() {
        let mut font = CaptionFont::builtin();
        let a = font.glyph_at('H', 13.7, 0).coverage.clone();
        let b = font.glyph_at('H', 13.7, 2).coverage.clone();
        let a_again = font.glyph_at('H', 13.7, 0).coverage.clone();
        assert_ne!(a, b, "two phases gave the same mask");
        assert_eq!(
            a, a_again,
            "the cache returned something else the second time"
        );
    }

    use super::*;

    #[test]
    fn builtin_metrics_split_the_cell_at_the_baseline() {
        let font = CaptionFont::builtin();
        let m = font.metrics(32.0);
        assert_eq!(m.ascent, 28.0); // 7/8 of the cell
        assert_eq!(m.descent, 4.0); // 1/8 of the cell
        assert_eq!(m.line_height(), 32.0);
    }

    #[test]
    fn builtin_glyphs_have_ink_and_sit_above_the_baseline() {
        let mut font = CaptionFont::builtin();
        let a = font.glyph('A', 32.0).clone();
        assert!(!a.is_blank());
        assert!(a.coverage.iter().any(|&v| v > 200), "expected solid pixels");
        assert_eq!(a.advance, 32.0);
        // 'A' has no descender, so its mask ends at or above the baseline.
        assert!(a.offset_y + a.height as i32 <= 1, "offset_y={}", a.offset_y);

        // 'g' descends below the baseline.
        let g = font.glyph('g', 32.0).clone();
        assert!(g.offset_y + g.height as i32 > 0, "expected a descender");
    }

    /// The advance from the metrics is the one the raster carries, for every
    /// face, at sizes on and off the optical-size axis, and for characters
    /// the face lacks.
    #[test]
    fn advances_without_rasterizing_match_the_rasters() {
        let mut faces = vec![CaptionFont::builtin()];
        if let Some(sf) = CaptionFont::try_sf_compact() {
            faces.push(sf);
        }
        for font in &mut faces {
            for size in [9.0, 14.0, 19.5, 33.0] {
                for c in ['A', 'g', ' ', '7', '\u{2603}'] {
                    let from_metrics = font.advance(c, size);
                    let from_raster = font.glyph(c, size).advance;
                    assert!((from_metrics - from_raster).abs() < 1e-4, "{c:?} at {size}: {from_metrics} against {from_raster}");
                }
            }
        }
    }

    /// A face set to an optical size draws that size's glyphs at any pixel
    /// size: SF Compact's 14-pixel text cut, rasterized at 48, is the 14-pixel
    /// glyph scaled up — wider set than the 48-pixel display cut — and its
    /// advances are the 14-pixel ones scaled. Skipped without SF Compact.
    #[test]
    fn an_optical_size_holds_at_any_pixel_size() {
        let Some(mut font) = CaptionFont::try_sf_compact() else { return };
        let display = font.advance('a', 48.0);
        let text = font.advance('a', 14.0) * 48.0 / 14.0;
        assert!(text > display + 0.1, "text cut {text}, display cut {display}");
        let display_ink: u32 = font.glyph('a', 48.0).coverage.iter().map(|&v| v as u32).sum();
        assert!(font.set_optical_size(Some(14.0)));
        assert!(!font.set_optical_size(Some(14.0)), "no change the second time");
        assert!((font.advance('a', 48.0) - text).abs() < 1e-3);
        assert!((font.glyph('a', 48.0).advance - text).abs() < 1e-3);
        let text_ink: u32 = font.glyph('a', 48.0).coverage.iter().map(|&v| v as u32).sum();
        assert_ne!(text_ink, display_ink, "the text cut's shape");
        // A face without the axis has nothing to change.
        assert!(!CaptionFont::builtin().set_optical_size(Some(14.0)));
    }

    fn dejavu() -> CaptionFont {
        CaptionFont::from_ttf_bytes(include_bytes!("../openscad/DejaVuSans.ttf")).unwrap()
    }

    fn ink(g: &GlyphBitmap) -> f32 {
        g.coverage.iter().map(|&v| v as f32 / 255.0).sum()
    }

    /// The leftmost and rightmost inked columns, in pixels from the pen.
    fn ink_columns(g: &GlyphBitmap) -> (i32, i32) {
        let cols: Vec<u32> = (0..g.width)
            .filter(|&x| (0..g.height).any(|y| g.coverage[(y * g.width + x) as usize] > 127))
            .collect();
        (g.offset_x + cols[0] as i32, g.offset_x + *cols.last().unwrap() as i32)
    }

    /// A static face asked for bold grows its outlines — by a 48th to a 64th
    /// of the size each side, as Skia's fake bold — keeps its left side
    /// bearing, and is set wider by the growth on both sides.
    #[test]
    fn a_static_face_synthesizes_bold() {
        let mut regular = dejavu();
        let mut bold = regular.styled(700, false).unwrap();
        assert_eq!(bold.style(), Some((700, false)));
        let size = 24.0;
        let outset = synthetic_bold_outset(size);
        assert!((outset - size / 56.0).abs() < size / 200.0, "{outset}");
        let (r, b) = (regular.glyph('H', size).clone(), bold.glyph('H', size).clone());
        // An H's stems and bar, each grown on both sides: well over a tenth
        // more ink.
        assert!(ink(&b) > ink(&r) * 1.1, "bold {} against {}", ink(&b), ink(&r));
        assert!((b.advance - r.advance - 2.0 * outset).abs() < 1e-4);
        assert!((bold.advance('H', size) - b.advance).abs() < 1e-4);
        assert_eq!(ink_columns(&b).0, ink_columns(&r).0, "the left side bearing stays");
        // Asked for anything under 600, it is the regular.
        let mut medium = regular.styled(500, false).unwrap();
        assert_eq!(ink(medium.glyph('H', size)), ink(&r));
    }

    /// A face that is bold already is not made bolder. Skipped without
    /// Arial Bold.
    #[test]
    fn a_bold_face_is_not_emboldened_again() {
        let Ok(bytes) = std::fs::read("/System/Library/Fonts/Supplemental/Arial Bold.ttf") else { return };
        let mut face = CaptionFont::from_ttf_bytes(&bytes).unwrap();
        let mut bold = face.styled(700, false).unwrap();
        assert_eq!(ink(face.glyph('H', 24.0)), ink(bold.glyph('H', 24.0)));
    }

    /// Without an italic, italic slants: the top of an `l` sits a quarter of
    /// its height right of its foot, and the advance is unchanged.
    #[test]
    fn a_face_without_an_italic_slants() {
        let mut upright = dejavu();
        let mut italic = upright.styled(400, true).unwrap();
        let size = 40.0;
        let (u, i) = (upright.glyph('l', size).clone(), italic.glyph('l', size).clone());
        assert_eq!(u.advance, i.advance);
        let centre = |g: &GlyphBitmap, row: u32| -> f32 {
            let (mut sum, mut w) = (0.0, 0.0);
            for x in 0..g.width {
                let v = g.coverage[(row * g.width + x) as usize] as f32;
                sum += v * x as f32;
                w += v;
            }
            sum / w
        };
        let (top, foot) = (1, i.height - 2);
        let lean = (centre(&i, top) - centre(&i, foot)) / (foot - top) as f32;
        assert!((lean - OBLIQUE_SKEW).abs() < 0.03, "lean {lean}");
        let upright_lean = (centre(&u, 1) - centre(&u, u.height - 2)) / (u.height - 3) as f32;
        assert!(upright_lean.abs() < 0.02, "{upright_lean}");
    }

    /// SF Compact's italic is its own file, not the upright slanted: its
    /// glyphs differ from the synthetic oblique of the upright. Skipped
    /// without SF Compact.
    #[test]
    fn sf_compact_italic_is_its_own_design() {
        let Some(sf) = CaptionFont::try_sf_compact() else { return };
        let mut real = sf.styled(400, true).unwrap();
        let bytes = std::fs::read("/System/Library/Fonts/SFCompact.ttf").unwrap();
        let mut slanted = CaptionFont::from_ttf_bytes(&bytes).unwrap().styled(400, true).unwrap();
        let (r, s) = (real.glyph('a', 40.0).clone(), slanted.glyph('a', 40.0).clone());
        assert!(r.coverage != s.coverage || r.advance != s.advance);
        // And both weights of it are real ones.
        let mut bold = real.with_weight(700).unwrap();
        assert!(ink(bold.glyph('a', 40.0)) > ink(&r) * 1.2);
    }

    /// Rows over half covered: (pixels above the baseline the first one's
    /// top is, the last one's bottom).
    fn solid_rows(g: &GlyphBitmap) -> (i32, i32) {
        let rows: Vec<i32> = (0..g.height)
            .filter(|&y| (0..g.width).any(|x| g.coverage[(y * g.width + x) as usize] > 128))
            .map(|y| y as i32)
            .collect();
        (-(g.offset_y + rows[0]), -(g.offset_y + rows[rows.len() - 1] + 1))
    }

    /// Slight hinting puts DejaVu's zones where FreeType's light autohinter
    /// does, at every size from 8 to 28 pixels: per size, the rows over half
    /// covered for `x`, `H`, `o` and `p` — top above the baseline, bottom —
    /// as `FT_LOAD_TARGET_LIGHT | FT_LOAD_FORCE_AUTOHINT` renders them
    /// (FreeType 2 from Homebrew). Arial, Verdana, Georgia and Trebuchet
    /// matched it the same way; Times differed on one row of 126, the top of
    /// a 9-pixel `o`, where FreeType's stem fitting goes beyond zones.
    #[test]
    fn slight_hinting_matches_freetype_light() {
        let mut font = dejavu();
        let plain = solid_rows(&font.glyph('x', 15.0).clone());
        font.set_hinting(Hinting::Slight);
        let freetype: [(u32, [(i32, i32); 4]); 21] = [
            (8, [(5, 0), (7, 0), (5, 0), (5, 0)]),
            (9, [(5, 0), (7, 0), (5, 0), (5, -2)]),
            (10, [(6, 0), (8, 0), (6, 0), (6, -2)]),
            (11, [(6, 0), (8, 0), (6, 0), (6, -2)]),
            (12, [(7, 0), (9, 0), (7, 0), (7, -3)]),
            (13, [(7, 0), (9, 0), (7, 0), (7, -3)]),
            (14, [(8, 0), (10, 0), (8, 0), (8, -3)]),
            (15, [(9, 0), (12, 0), (9, 0), (9, -3)]),
            (16, [(9, 0), (12, 0), (9, 0), (9, -3)]),
            (17, [(10, 0), (13, 0), (10, 0), (10, -4)]),
            (18, [(10, 0), (13, 0), (10, 0), (10, -4)]),
            (19, [(11, 0), (14, 0), (11, 0), (11, -4)]),
            (20, [(11, 0), (14, 0), (11, 0), (11, -4)]),
            (21, [(12, 0), (16, 0), (12, 0), (12, -4)]),
            (22, [(12, 0), (16, 0), (12, 0), (12, -4)]),
            (23, [(13, 0), (17, 0), (13, 0), (13, -5)]),
            (24, [(14, 0), (18, 0), (14, 0), (14, -5)]),
            (25, [(14, 0), (18, 0), (14, 0), (14, -5)]),
            (26, [(15, 0), (20, 0), (15, 0), (15, -6)]),
            (27, [(15, 0), (20, 0), (15, 0), (15, -6)]),
            (28, [(16, 0), (21, 0), (16, 0), (16, -6)]),
        ];
        for (px, expected) in freetype {
            for (c, want) in ['x', 'H', 'o', 'p'].into_iter().zip(expected) {
                let got = solid_rows(&font.glyph(c, px as f32).clone());
                assert_eq!(got, want, "{c} at {px} px");
            }
        }
        // Unhinted, the 15-pixel x-height (8.2 pixels) is its own; hinted,
        // FreeType's 9.
        assert_eq!(plain.0, 8);
        assert_eq!(font.hinting(), Hinting::Slight);
    }

    /// Hinting moves nothing sideways: advances are the outline's.
    #[test]
    fn hinting_keeps_the_advances() {
        let mut font = dejavu();
        let plain: Vec<f32> = "Hamburg".chars().map(|c| font.glyph(c, 13.0).advance).collect();
        font.set_hinting(Hinting::Slight);
        let hinted: Vec<f32> = "Hamburg".chars().map(|c| font.glyph(c, 13.0).advance).collect();
        assert_eq!(plain, hinted);
    }

    #[test]
    fn space_is_blank_but_still_advances() {
        let mut font = CaptionFont::builtin();
        let space = font.glyph(' ', 24.0).clone();
        assert!(space.is_blank());
        assert_eq!(space.advance, 24.0);
    }

    #[test]
    fn unmapped_codepoints_draw_a_box_and_accents_fold() {
        assert_eq!(fold_to_ascii('é'), 'e');
        assert_eq!(fold_to_ascii('Ü'), 'U');
        assert_eq!(fold_to_ascii('\u{2019}'), '\'');
        // Folded letters render as their base glyph, not as tofu.
        assert_eq!(builtin_rows('é'), builtin_rows('e'));
        assert_eq!(builtin_rows('\u{4E2D}'), Some(TOFU));
    }

    #[test]
    fn measure_matches_summed_advances_plus_tracking() {
        let mut font = CaptionFont::builtin();
        assert_eq!(font.measure("abc", 16.0, 0.0), 48.0);
        assert_eq!(font.measure("abc", 16.0, 2.0), 48.0 + 4.0);
        assert_eq!(font.measure("a", 16.0, 2.0), 16.0);
        assert_eq!(font.measure("", 16.0, 2.0), 0.0);
    }

    #[test]
    fn a_decomposed_accent_measures_as_one_letter() {
        // The gap this closes. `"e" + U+0301` is one letter and was measured as two: the
        // base plus a floating accent, so a centred caption sat half an accent to one
        // side. Measured on the built-in face, where every advance is the size, so the
        // numbers say exactly how many glyphs were counted.
        let mut font = CaptionFont::builtin();
        assert_eq!(font.measure("e\u{0301}", 16.0, 0.0), font.measure("é", 16.0, 0.0));
        assert_eq!(font.measure("Noe\u{0308}l", 16.0, 0.0), font.measure("Noël", 16.0, 0.0));
        // And the letter spacing counts letters, not characters: four letters, three gaps.
        assert_eq!(font.measure("Noe\u{0308}l", 16.0, 2.0), 4.0 * 16.0 + 3.0 * 2.0);
    }

    #[test]
    fn a_right_to_left_line_measures_the_same_either_way() {
        // Reordering must not change a line's width: the same glyphs with the same
        // advances, in a different order. If this differs, the kerning pairs are being
        // taken visually rather than logically.
        let mut font = CaptionFont::builtin();
        let hebrew = font.measure("\u{05D0}\u{05D1}\u{05D2}", 16.0, 0.0);
        let reversed = font.measure("\u{05D2}\u{05D1}\u{05D0}", 16.0, 0.0);
        assert_eq!(hebrew, reversed);
    }

    #[test]
    fn a_mark_with_no_composite_still_measures() {
        // `q` has no precomposed form with a diaeresis, so it stays two characters and
        // must not be dropped or doubled.
        let mut font = CaptionFont::builtin();
        let two = font.measure("q\u{0308}", 16.0, 0.0);
        let one = font.measure("q", 16.0, 0.0);
        assert!(two > one, "the mark vanished");
    }

    #[test]
    fn glyphs_are_cached_per_size() {
        let mut font = CaptionFont::builtin();
        font.glyph('A', 16.0);
        font.glyph('A', 16.0);
        assert_eq!(font.cache.len(), 1);
        font.glyph('A', 32.0);
        assert_eq!(font.cache.len(), 2);
        font.clear_cache();
        assert_eq!(font.cache.len(), 0);
    }

    #[test]
    fn spans_accumulate_exact_fractional_coverage() {
        let mut row = vec![0.0f32; 4];
        add_span(&mut row, 0.5, 2.25);
        assert!((row[0] - 0.5).abs() < 1e-6);
        assert!((row[1] - 1.0).abs() < 1e-6);
        assert!((row[2] - 0.25).abs() < 1e-6);
        assert_eq!(row[3], 0.0);
        // Clamped at both ends, and empty spans are no-ops.
        let mut row = vec![0.0f32; 2];
        add_span(&mut row, -3.0, 9.0);
        assert!((row[0] - 1.0).abs() < 1e-6);
        assert!((row[1] - 1.0).abs() < 1e-6);
        add_span(&mut row, 1.0, 1.0);
        assert!((row[1] - 1.0).abs() < 1e-6);
    }
}
