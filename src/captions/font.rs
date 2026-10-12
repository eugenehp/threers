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
    Ttf(Box<TtfFont>),
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
        }
    }

    /// Use an already-parsed TrueType face.
    pub fn from_ttf(font: TtfFont) -> Self {
        Self {
            face: Face::Ttf(Box::new(font)),
            cache: HashMap::new(),
        }
    }

    /// Parse a `.ttf` and use it as the caption face.
    ///
    /// Accepts the same subset [`TtfFont`] does: TrueType `glyf` outlines with
    /// a format-4 `cmap`. OpenType/CFF (`.otf`) is not supported.
    pub fn from_ttf_bytes(bytes: &[u8]) -> Result<Self, TtfError> {
        Ok(Self::from_ttf(TtfFont::parse(bytes)?))
    }

    /// SF Compact when the system ships it (macOS), otherwise
    /// [`builtin`](Self::builtin).
    ///
    /// This is the preferred UI face for burned-in labels and data graphics.
    pub fn ui() -> Self {
        Self::try_sf_compact().unwrap_or_else(Self::builtin)
    }

    /// Load Apple's SF Compact from a well-known install path.
    pub fn try_sf_compact() -> Option<Self> {
        #[cfg(target_os = "macos")]
        {
            const PATHS: &[&str] = &[
                "/System/Library/Fonts/SFCompact.ttf",
                "/System/Library/Fonts/SFCompactRounded.ttf",
            ];
            for path in PATHS {
                let bytes = std::fs::read(path).ok()?;
                if let Ok(font) = Self::from_ttf_bytes(&bytes) {
                    return Some(font);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = ();
        }
        None
    }

    /// Whether this face is the built-in bitmap one.
    pub fn is_builtin(&self) -> bool {
        matches!(self.face, Face::Builtin)
    }

    /// Drop every cached glyph raster.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
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
            Face::Ttf(font) => {
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
        let size = size.max(1.0);
        let phase = phase % SUBPIXEL_PHASES;
        let key = (c as u32, (size * 4.0).round() as u32, phase as u32);
        if !self.cache.contains_key(&key) {
            let shift = phase as f32 / SUBPIXEL_PHASES as f32;
            let bitmap = match &self.face {
                Face::Builtin => raster_builtin(c, size, shift),
                Face::Ttf(font) => raster_ttf(font, c, size, shift),
            };
            self.cache.insert(key, bitmap);
        }
        &self.cache[&key]
    }

    /// The kerning between two characters at `size` pixels.
    ///
    /// Zero for the built-in face, which is a fixed-width bitmap and has no
    /// pairs to adjust, and zero for a TrueType face whose kerning lives in
    /// `GPOS` rather than `kern`.
    pub fn kerning(&self, left: char, right: char, size: f32) -> f32 {
        let Face::Ttf(font) = &self.face else {
            return 0.0;
        };
        if font.kerning.is_empty() {
            return 0.0;
        }
        let gid = |c: char| font.cmap.get(&(c as u32)).copied().unwrap_or(0);
        let units = font.kern(gid(left), gid(right));
        if units == 0 {
            return 0.0;
        }
        units as f32 * size.max(1.0) / font.units_per_em.max(1) as f32
    }

    /// Pen advance for `c` at `size` pixels, without keeping the raster.
    pub fn advance(&mut self, c: char, size: f32) -> f32 {
        self.glyph(c, size).advance
    }

    /// Total advance of `text` at `size` pixels, including `letter_spacing`
    /// between glyphs.
    pub fn measure(&mut self, text: &str, size: f32, letter_spacing: f32) -> f32 {
        let mut width = 0.0;
        let mut count = 0usize;
        let mut previous: Option<char> = None;
        for c in text.chars() {
            // Kerning is part of the width. Measuring without it and drawing
            // with it puts every centred or right-aligned line off by the sum of
            // its pairs — the text would sit a little to one side, by an amount
            // that changes with the words.
            if let Some(p) = previous {
                width += self.kerning(p, c, size);
            }
            width += self.advance(c, size);
            previous = Some(c);
            count += 1;
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
fn raster_ttf(font: &TtfFont, c: char, size: f32, shift: f32) -> GlyphBitmap {
    let upem = font.units_per_em.max(1) as f32;
    let scale = size / upem;
    let gid = font.cmap.get(&(c as u32)).copied().unwrap_or(0);
    let Some(glyph) = font.glyphs.get(gid as usize) else {
        return GlyphBitmap::default();
    };
    let advance = glyph.advance_width as f32 * scale;

    // Flatten the outline into device-space contours (y grows downward, so the
    // font's y axis is negated).
    // The subpixel phase is applied to the outline itself, so the coverage is
    // exact for the position the glyph will actually sit at — shifting a
    // finished mask could only resample it, which is what blurs subpixel-placed
    // text in a naive implementation.
    let contours = flatten_contours(glyph, scale, shift);
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
fn flatten_contours(
    glyph: &crate::loaders::TtfGlyph,
    scale: f32,
    shift: f32,
) -> Vec<Vec<(f32, f32)>> {
    let mut contours: Vec<Vec<(f32, f32)>> = Vec::new();
    let mut current: Vec<(f32, f32)> = Vec::new();
    let mut last: Option<(f32, f32)> = None;

    // Device space: x scaled, y negated (font y is up, raster y is down).
    let to_device = |p: crate::math::Vector2| (p.x * scale + shift, -p.y * scale);

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
        assert!(expected < 0.0, "AV kerning should be negative, got {expected}");
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
            assert!(v == 0 || v == 255, "partial coverage {v} at an integer scale");
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
        assert!(differed, "no phase changed the raster, so phases do nothing");
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
        assert_eq!(a, a_again, "the cache returned something else the second time");
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
