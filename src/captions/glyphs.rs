//! Glyphs as images: a face made of bitmaps rather than outlines — a
//! pre-packed font atlas, glyphs drawn by some other renderer (deck.gl's
//! `FontRenderer`), or a TrueType face rasterized once at a design size —
//! and signed distance fields made from such bitmaps (Mapbox's TinySDF).

use std::collections::HashMap;

use super::CaptionFont;

/// One glyph: its metrics at the face's size, and its image.
///
/// The glyph's box is `width` wide from the pen and runs from `ascent`
/// above the baseline to `descent` below it. The image holds the box with
/// its top-left corner at `(left, top)`, and may reach past it (a distance
/// field's margin).
#[derive(Debug, Clone, PartialEq)]
pub struct BitmapGlyph {
    pub advance: f32,
    pub width: f32,
    pub ascent: f32,
    pub descent: f32,
    pub image_width: u32,
    pub image_height: u32,
    /// `image_width × image_height` bytes: coverage, or a distance field
    /// for a face made with [`BitmapGlyphs::to_sdf`].
    pub alpha: Vec<u8>,
    pub left: i32,
    pub top: i32,
}

/// How a face's images are distance fields: TinySDF's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SdfSettings {
    /// Pixels of field around each glyph's box.
    pub buffer: u32,
    /// Pixels the field spans from the edge each way.
    pub radius: f32,
    /// Where the edge sits in the field's range, 0 to 1 (0.25: at
    /// `255 · 0.75`).
    pub cutoff: f32,
}

impl SdfSettings {
    /// deck.gl's defaults: a 4-pixel buffer, radius 12, cutoff 0.25.
    pub const DECK_GL: Self = Self { buffer: 4, radius: 12.0, cutoff: 0.25 };

    /// The field value on a glyph's edge, 0 to 1.
    pub fn edge(&self) -> f32 {
        1.0 - self.cutoff
    }
}

/// A face of [`BitmapGlyph`]s drawn at `size` pixels per em.
#[derive(Debug, Clone, Default)]
pub struct BitmapGlyphs {
    pub size: f32,
    /// The face's own ascent and descent at `size` (deck.gl's
    /// `measure()` without a character): what lines are centred by.
    pub ascent: f32,
    pub descent: f32,
    pub glyphs: HashMap<u32, BitmapGlyph>,
    /// Set when the images are distance fields.
    pub sdf: Option<SdfSettings>,
    /// Drawn as a canvas's `fillText` draws text, measured by its
    /// `measureText` ([`from_font`](Self::from_font)): deck.gl then keeps
    /// each measured box as the glyph's frame, ink past it unseen, where a
    /// face drawn by a custom renderer is framed by its whole image.
    pub canvas: bool,
    /// For a canvas face, each glyph's image drawn a quarter, a half and
    /// three quarters of a pixel right of the pen (`left` and `top` as the
    /// glyph's own): a canvas places glyphs at quarter pixels, so a glyph
    /// whose pen lands between pixels is drawn there rather than moved.
    pub phases: HashMap<u32, [BitmapGlyph; 3]>,
}

/// A glyph's place in a pre-packed atlas image, as deck.gl's font atlas
/// JSON gives it: the box's top-left `x`, `y` (fractions truncate, as a
/// canvas's `getImageData` does), its size, the baseline `anchor_y` below
/// the box's top, and the advance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AtlasFrame {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub anchor_y: f32,
    pub advance: f32,
}

impl BitmapGlyphs {
    /// Glyphs cut from an atlas image (`width × height` RGBA, coverage in
    /// alpha) at `frames`, a face of `size` pixels per em with the given
    /// line ascent and descent.
    pub fn from_atlas(
        rgba: &[u8],
        width: u32,
        height: u32,
        frames: impl IntoIterator<Item = (char, AtlasFrame)>,
        size: f32,
        ascent: f32,
        descent: f32,
    ) -> Self {
        let mut glyphs = HashMap::new();
        for (c, f) in frames {
            let (x0, y0) = (f.x.max(0.0) as u32, f.y.max(0.0) as u32);
            let w = (f.width.max(0.0) as u32).min(width.saturating_sub(x0));
            let h = (f.height.max(0.0) as u32).min(height.saturating_sub(y0));
            let mut alpha = Vec::with_capacity((w * h) as usize);
            for y in y0..y0 + h {
                for x in x0..x0 + w {
                    alpha.push(rgba.get(((y * width + x) * 4 + 3) as usize).copied().unwrap_or(0));
                }
            }
            glyphs.insert(
                c as u32,
                BitmapGlyph {
                    advance: f.advance,
                    width: f.width,
                    ascent: f.anchor_y,
                    descent: f.height - f.anchor_y,
                    image_width: w,
                    image_height: h,
                    alpha,
                    left: 0,
                    top: 0,
                },
            );
        }
        Self { size, ascent, descent, glyphs, sdf: None, canvas: false, phases: HashMap::new() }
    }

    /// `chars` of `font` rasterized at `size` pixels per em — what a
    /// browser canvas's `fillText` gives deck.gl, measured as its
    /// `measureText` measures: the outline's bounds (its points', as a
    /// TrueType glyph's box is), the box `right − left` wide from the pen
    /// with the ascent and descent ceiled (deck.gl's arithmetic), the face's
    /// line ascent and descent for the whole. Faces without outlines are
    /// measured by their images.
    pub fn from_font(font: &mut CaptionFont, chars: impl IntoIterator<Item = char>, size: f32) -> Self {
        let m = font.metrics(size);
        let mut glyphs = HashMap::new();
        let mut phases = HashMap::new();
        for c in chars {
            let g = font.glyph(c, size).clone();
            let advance = font.advance(c, size);
            let bounds = font.outline(c, size).and_then(|o| {
                let mut pts = o.iter().flatten().flatten().peekable();
                pts.peek()?;
                Some(pts.fold([f32::MAX, f32::MIN, f32::MAX, f32::MIN], |b, p| {
                    [b[0].min(p[0]), b[1].max(p[0]), b[2].min(p[1]), b[3].max(p[1])]
                }))
            });
            let (ascent, descent, width) = match bounds {
                // No ink, or ink whose top is on the baseline (an
                // underscore): `measureText`'s ascent is 0, which deck.gl
                // takes for no box, falling back to the advance and its
                // default 0.9 and 0.3 em.
                _ if g.is_blank() => (size * 0.9, size * 0.3, advance),
                Some([_, _, 0.0, _]) => (size * 0.9, size * 0.3, advance),
                // measureText's `actualBoundingBoxRight - actualBoundingBoxLeft`,
                // the left counted leftwards from the pen.
                Some([x0, x1, y0, y1]) => ((-y0).ceil(), y1.ceil(), (x1 + x0).ceil()),
                None => (
                    (-g.offset_y as f32).max(0.0).ceil(),
                    (g.offset_y + g.height as i32).max(0) as f32,
                    (g.offset_x + g.width as i32).max(0) as f32,
                ),
            };
            let glyph = BitmapGlyph {
                advance,
                width,
                ascent,
                descent,
                image_width: g.width,
                image_height: g.height,
                alpha: g.coverage.clone(),
                // The image's corner relative to the box's (the pen at
                // the baseline): the box's corner within the image.
                left: -g.offset_x,
                top: (-(g.offset_y as f32) - ascent).round() as i32,
            };
            let phase = |font: &mut CaptionFont, k: usize| {
                let g = font.glyph_at(c, size, k);
                BitmapGlyph {
                    image_width: g.width,
                    image_height: g.height,
                    alpha: g.coverage.clone(),
                    left: -g.offset_x,
                    top: (-(g.offset_y as f32) - ascent).round() as i32,
                    ..glyph.clone()
                }
            };
            phases.insert(c as u32, [phase(font, 1), phase(font, 2), phase(font, 3)]);
            glyphs.insert(c as u32, glyph);
        }
        Self { size, ascent: m.ascent.ceil(), descent: m.descent.ceil(), glyphs, sdf: None, canvas: true, phases }
    }

    /// Every glyph as a signed distance field ([`tiny_sdf`]), `buffer`
    /// pixels around its box.
    pub fn to_sdf(&self, settings: SdfSettings) -> Self {
        let b = settings.buffer as i32;
        let glyphs = self
            .glyphs
            .iter()
            .map(|(&c, g)| {
                let (w, h) = (g.image_width + 2 * settings.buffer, g.image_height + 2 * settings.buffer);
                let mut padded = vec![0u8; (w * h) as usize];
                for y in 0..g.image_height {
                    let src = (y * g.image_width) as usize;
                    let dst = ((y + settings.buffer) * w + settings.buffer) as usize;
                    padded[dst..dst + g.image_width as usize].copy_from_slice(&g.alpha[src..src + g.image_width as usize]);
                }
                let field = tiny_sdf(&padded, w, h, settings.radius, settings.cutoff);
                (
                    c,
                    BitmapGlyph {
                        image_width: w,
                        image_height: h,
                        alpha: field,
                        left: g.left + b,
                        top: g.top + b,
                        ..g.clone()
                    },
                )
            })
            .collect();
        Self { glyphs, sdf: Some(settings), canvas: false, phases: HashMap::new(), ..self.clone() }
    }

    /// `c` at `size` pixels per em, resampled to coverage — for
    /// [`CaptionFont::from_bitmap_glyphs`].
    pub(crate) fn raster(&self, c: char, size: f32, shift: f32) -> super::font::GlyphBitmap {
        let k = size / self.size.max(1e-6);
        let blank = |advance: f32| super::font::GlyphBitmap {
            width: 0,
            height: 0,
            coverage: Vec::new(),
            offset_x: 0,
            offset_y: 0,
            advance,
        };
        let Some(g) = self.glyphs.get(&(c as u32)) else { return blank(0.0) };
        let advance = g.advance * k;
        if g.image_width == 0 || g.image_height == 0 {
            return blank(advance);
        }
        // The image's corner from the pen at the baseline, at the face's size.
        let (ix, iy) = (-(g.left as f32), -g.ascent - g.top as f32);
        let x0 = (ix * k + shift).floor() as i32 - 1;
        let y0 = (iy * k).floor() as i32 - 1;
        let w = (g.image_width as f32 * k).ceil() as u32 + 3;
        let h = (g.image_height as f32 * k).ceil() as u32 + 3;
        let at = |x: i32, y: i32| -> f32 {
            if x < 0 || y < 0 || x >= g.image_width as i32 || y >= g.image_height as i32 {
                0.0
            } else {
                g.alpha[(y as u32 * g.image_width + x as u32) as usize] as f32
            }
        };
        let mut coverage = Vec::with_capacity((w * h) as usize);
        for oy in 0..h {
            for ox in 0..w {
                // The output pixel's centre in the image, bilinear.
                let px = ((x0 + ox as i32) as f32 + 0.5 - shift) / k - ix - 0.5;
                let py = ((y0 + oy as i32) as f32 + 0.5) / k - iy - 0.5;
                let (fx, fy) = (px.floor(), py.floor());
                let (tx, ty) = (px - fx, py - fy);
                let (x, y) = (fx as i32, fy as i32);
                let v = (at(x, y) * (1.0 - tx) + at(x + 1, y) * tx) * (1.0 - ty)
                    + (at(x, y + 1) * (1.0 - tx) + at(x + 1, y + 1) * tx) * ty;
                let v = match self.sdf {
                    // Field to coverage: about a pixel of ramp at this size.
                    Some(s) => {
                        let per_px = 255.0 / s.radius / k.max(1e-6);
                        ((v - 255.0 * s.edge()) / per_px + 0.5).clamp(0.0, 1.0) * 255.0
                    }
                    None => v,
                };
                coverage.push(v.round().clamp(0.0, 255.0) as u8);
            }
        }
        super::font::GlyphBitmap { width: w, height: h, coverage, offset_x: x0, offset_y: y0, advance }
    }
}

const INF: f64 = 1e20;

/// Mapbox's TinySDF: a signed distance field from coverage (`w × h` bytes,
/// 0 outside, 255 inside, between on the edge), `255 − 255 · (d / radius +
/// cutoff)` with `d` the signed distance in pixels to the edge, positive
/// outside — so the edge is at `255 · (1 − cutoff)`.
pub fn tiny_sdf(alpha: &[u8], w: u32, h: u32, radius: f32, cutoff: f32) -> Vec<u8> {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    let mut outer = vec![INF; n];
    let mut inner = vec![0.0f64; n];
    for i in 0..n {
        let a = alpha[i] as f64 / 255.0;
        if a == 0.0 {
            continue;
        }
        if a == 1.0 {
            outer[i] = 0.0;
            inner[i] = INF;
        } else {
            // A pixel the edge passes through: its distance to it.
            let d = 0.5 - a;
            outer[i] = if d > 0.0 { d * d } else { 0.0 };
            inner[i] = if d < 0.0 { d * d } else { 0.0 };
        }
    }
    edt(&mut outer, w, h);
    edt(&mut inner, w, h);
    (0..n)
        .map(|i| {
            let d = outer[i].sqrt() - inner[i].sqrt();
            (255.0 - 255.0 * (d / radius as f64 + cutoff as f64)).round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// Felzenszwalb and Huttenlocher's squared Euclidean distance transform,
/// columns then rows.
fn edt(grid: &mut [f64], w: usize, h: usize) {
    let len = w.max(h);
    let mut f = vec![0.0; len];
    let mut v = vec![0usize; len];
    let mut z = vec![0.0; len + 1];
    for x in 0..w {
        edt1d(grid, x, w, h, &mut f, &mut v, &mut z);
    }
    for y in 0..h {
        edt1d(grid, y * w, 1, w, &mut f, &mut v, &mut z);
    }
}

fn edt1d(grid: &mut [f64], offset: usize, stride: usize, n: usize, f: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    // TinySDF's, line for line.
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    f[0] = grid[offset];
    let mut k: isize = 0;
    for q in 1..n {
        f[q] = grid[offset + q * stride];
        let q2 = (q * q) as f64;
        let mut s;
        loop {
            let r = v[k as usize];
            s = (f[q] - f[r] + q2 - (r * r) as f64) / (q as f64 - r as f64) / 2.0;
            if s <= z[k as usize] {
                k -= 1;
                if k > -1 {
                    continue;
                }
            }
            break;
        }
        k += 1;
        v[k as usize] = q;
        z[k as usize] = s;
        z[k as usize + 1] = INF;
    }
    let mut k = 0usize;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let r = v[k];
        let d = q as f64 - r as f64;
        grid[offset + q * stride] = f[r] + d * d;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_sdf_measures_distance_to_the_edge() {
        // A 4-pixel square in a 20-pixel field.
        let (w, h) = (20u32, 20u32);
        let mut a = vec![0u8; 400];
        for y in 8..12 {
            for x in 8..12 {
                a[y * 20 + x] = 255;
            }
        }
        let f = tiny_sdf(&a, w, h, 8.0, 0.25);
        let at = |x: usize, y: usize| f[y * 20 + x] as f32;
        // Outside, 3 pixels from the square: 255 - 255·(3/8 + 0.25).
        assert!((at(5, 9) - (255.0 - 255.0 * (3.0 / 8.0 + 0.25))).abs() <= 1.0, "{}", at(5, 9));
        // Inside is above the edge's 191, outside below.
        assert!(at(9, 9) > 191.0 && at(6, 9) < 191.0);
    }

    #[test]
    fn atlas_glyphs_resample_to_any_size() {
        // One 4×8 glyph of solid ink at (2, 1) in a 16×16 atlas.
        let mut rgba = vec![0u8; 16 * 16 * 4];
        for y in 1..9 {
            for x in 2..6 {
                rgba[(y * 16 + x) * 4 + 3] = 255;
            }
        }
        let frame = AtlasFrame { x: 2.0, y: 1.0, width: 4.0, height: 8.0, anchor_y: 6.0, advance: 5.0 };
        let set = BitmapGlyphs::from_atlas(&rgba, 16, 16, [('I', frame)], 8.0, 6.0, 2.0);
        let mut font = CaptionFont::from_bitmap_glyphs(std::sync::Arc::new(set));
        assert!(font.has_glyph('I') && !font.has_glyph('J'));
        assert_eq!(font.advance('I', 16.0), 10.0);
        let g = font.glyph('I', 16.0).clone();
        // At twice the size: ink from 12 above the baseline to 4 below.
        let inked: Vec<i32> = (0..g.height as i32)
            .filter(|&r| g.coverage[(r as u32 * g.width + g.width / 2) as usize] > 128)
            .map(|r| r + g.offset_y)
            .collect();
        assert_eq!((inked[0], *inked.last().unwrap()), (-12, 3), "{inked:?}");
    }
}
