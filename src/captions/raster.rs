//! Compositing captions into RGBA pixels.
//!
//! [`CaptionPainter`] owns a face and a style and draws cue text into a
//! tightly-packed RGBA8 buffer — the same layout
//! [`HeadlessRenderer::render_to_rgba`](crate::HeadlessRenderer::render_to_rgba)
//! produces and `video::export_video` consumes. That
//! makes burned-in ("open") captions a single call per frame:
//!
//! ```
//! use threers::captions::{CaptionPainter, CaptionTrack};
//! let track = CaptionTrack::new().cue(0.0, 2.0, "Burned in");
//! let (w, h) = (320u32, 180u32);
//! let mut frame = vec![0u8; (w * h * 4) as usize];
//! let mut painter = CaptionPainter::new().for_height(h);
//! assert!(painter.burn_in(&mut frame, w, h, &track, 1.0));
//! assert!(!painter.burn_in(&mut frame, w, h, &track, 5.0)); // nothing active
//! ```
//!
//! Drawing order per block is background box → drop shadow → outline → fill,
//! each composited source-over so the result works on opaque video frames and
//! on transparent overlay buffers alike.

use crate::utils::srgb::{linear_to_srgb, linear_to_srgb_f32, srgb_to_linear, srgb_to_linear_f32};

use super::font::SUBPIXEL_PHASES;
use super::layout::{layout, LaidOutText};
use super::{CaptionAlign, CaptionAnchor, CaptionFont, CaptionStyle, CaptionTrack, Cue};

/// Where a block sits, taken from a cue's overrides or the style's defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placement {
    align: Option<CaptionAlign>,
    line: Option<f32>,
    position: Option<f32>,
}

impl Placement {
    fn of(cue: &Cue) -> Self {
        Self {
            align: cue.align,
            line: cue.line,
            position: cue.position,
        }
    }

    /// The style's own placement — what `draw_text` uses.
    fn default() -> Self {
        Self {
            align: None,
            line: None,
            position: None,
        }
    }
}

/// Draws caption text into RGBA frames.
///
/// Clone-free by design: keep one painter alive across frames so its glyph
/// cache is reused.
#[derive(Debug)]
pub struct CaptionPainter {
    /// The face glyphs are rasterized from.
    pub font: CaptionFont,
    /// Type size, colors, and placement.
    pub style: CaptionStyle,
    /// Rows the last draw wrote to. See [`CaptionPainter::last_rows`].
    last_rows: Option<(u32, u32)>,
}

impl Default for CaptionPainter {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptionPainter {
    /// A painter with the UI face ([`CaptionFont::ui`]) and the default style.
    pub fn new() -> Self {
        Self::with_font(CaptionFont::ui())
    }

    /// A painter drawing with `font`.
    pub fn with_font(font: CaptionFont) -> Self {
        Self {
            font,
            style: CaptionStyle::default(),
            last_rows: None,
        }
    }

    /// Replace the style (builder form).
    pub fn style(mut self, style: CaptionStyle) -> Self {
        self.style = style;
        self
    }

    /// Rescale the current style for a frame `height` pixels tall, treating it
    /// as authored for 1080p. See [`CaptionStyle::for_height`].
    pub fn for_height(mut self, height: u32) -> Self {
        self.style = self.style.for_height(height);
        self
    }

    /// Replace the style in place.
    pub fn set_style(&mut self, style: CaptionStyle) {
        self.style = style;
    }

    /// Wrap and measure `text` as it would be drawn into a `width`-pixel frame.
    pub fn measure(&mut self, text: &str, width: u32) -> LaidOutText {
        let max_width = self.wrap_width(width);
        layout(&mut self.font, text, &self.style, max_width)
    }

    /// Draw `text` into `frame` using the style's own placement.
    ///
    /// `frame` must be `width * height * 4` bytes of tightly-packed RGBA8.
    /// Returns `false` if the buffer is the wrong size or the text is blank.
    pub fn draw_text(&mut self, frame: &mut [u8], width: u32, height: u32, text: &str) -> bool {
        self.draw_block(frame, width, height, text, Placement::default())
    }

    /// Draw a single line with its top-left corner at `(x, y)` in frame pixels,
    /// in the style's colour, outline and shadow but without its background box.
    pub fn draw_text_at(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        x: f32,
        y: f32,
        text: &str,
    ) -> bool {
        self.draw_text_at_aligned(frame, width, height, x, y, text, CaptionAlign::Left)
    }

    /// Draw a single line right-aligned so its right edge sits at `x`.
    pub fn draw_text_at_right(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        x: f32,
        y: f32,
        text: &str,
    ) -> bool {
        self.draw_text_at_aligned(frame, width, height, x, y, text, CaptionAlign::Right)
    }

    /// The rows the most recent draw wrote to, as a half-open range, or `None`
    /// if it drew nothing.
    ///
    /// Reported rather than predicted, because the placement is not something a
    /// caller can work out in advance: the block re-wraps against the space
    /// left of its anchor, and is nudged back up when it would overhang the
    /// bottom. For a caller uploading the frame to a texture this is the band
    /// that changed, which for one line of type is tens of rows out of a
    /// thousand.
    pub fn last_rows(&self) -> Option<(u32, u32)> {
        self.last_rows
    }

    /// Draw a single line centred horizontally on `x`.
    pub fn draw_text_at_center(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        x: f32,
        y: f32,
        text: &str,
    ) -> bool {
        self.draw_text_at_aligned(frame, width, height, x, y, text, CaptionAlign::Center)
    }

    /// Draw one line at absolute pixel coordinates (no WebVTT-style line clamping).
    fn draw_text_at_aligned(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        x: f32,
        y: f32,
        text: &str,
        align: CaptionAlign,
    ) -> bool {
        if text.trim().is_empty() || width == 0 || height == 0 {
            return false;
        }
        let fw = width as f32;
        let saved = self.style.clone();
        // No box behind a positioned line; the outline stays, since it is
        // what [`Overlay::text`](crate::overlay::Overlay::text) turns a stroke
        // into (zeroing it here left every stroked overlay label bare).
        self.style.background[3] = 0;
        self.style.align = align;
        // Fit to the remaining horizontal space from the anchor point.
        let max_width = match align {
            CaptionAlign::Left => (fw - x - self.style.padding).max(self.style.font_size),
            CaptionAlign::Right => (x - self.style.padding).max(self.style.font_size),
            CaptionAlign::Center => (fw - self.style.padding * 2.0).max(self.style.font_size),
        };
        self.style.max_width = (max_width / fw).clamp(
            self.style.font_size / fw,
            1.0,
        );
        let ok = self.draw_block_at(frame, width, height, text, x, y, align);
        self.style = saved;
        ok
    }

    /// Draw one cue, honoring its `align` / `line` / `position` overrides.
    pub fn draw_cue(&mut self, frame: &mut [u8], width: u32, height: u32, cue: &Cue) -> bool {
        self.draw_block(frame, width, height, &cue.plain_text(), Placement::of(cue))
    }

    /// Composite every cue of `track` active at `time` seconds into `frame`.
    ///
    /// Cues that share a placement are merged into one block so simultaneous
    /// cues stack instead of overprinting. Returns whether anything was drawn.
    pub fn burn_in(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        track: &CaptionTrack,
        time: f64,
    ) -> bool {
        let mut groups: Vec<(Placement, String)> = Vec::new();
        for cue in track.cues.iter().filter(|c| c.is_active_at(time)) {
            let placement = Placement::of(cue);
            let text = cue.plain_text();
            match groups.iter_mut().find(|(p, _)| *p == placement) {
                Some((_, existing)) => {
                    existing.push('\n');
                    existing.push_str(&text);
                }
                None => groups.push((placement, text)),
            }
        }
        let mut drew = false;
        for (placement, text) in groups {
            drew |= self.draw_block(frame, width, height, &text, placement);
        }
        drew
    }

    /// Render the cues active at `time` into a fresh transparent RGBA buffer of
    /// `width * height * 4` bytes — the upload source for the on-screen overlay.
    pub fn render_overlay(
        &mut self,
        width: u32,
        height: u32,
        track: &CaptionTrack,
        time: f64,
    ) -> Vec<u8> {
        let mut buffer = vec![0u8; (width as usize) * (height as usize) * 4];
        self.burn_in(&mut buffer, width, height, track, time);
        buffer
    }

    /// Usable text width in pixels before wrapping, inside the padded box.
    fn wrap_width(&self, frame_width: u32) -> f32 {
        let area = frame_width as f32 * self.style.max_width;
        (area - self.style.padding * 2.0).max(self.style.font_size)
    }

    /// Lay out `text`, place it, and composite box + shadow + outline + fill.
    fn draw_block(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        placement: Placement,
    ) -> bool {
        self.last_rows = None;
        if width == 0 || height == 0 || frame.len() < (width as usize) * (height as usize) * 4 {
            return false;
        }
        let laid = {
            let max_width = self.wrap_width(width);
            layout(&mut self.font, text, &self.style, max_width)
        };
        if laid.is_empty() || laid.width <= 0.0 {
            return false;
        }

        let style = self.style.clone();
        let align = placement.align.unwrap_or(style.align);
        let (block_left, block_top) = self.place(&laid, &style, align, placement, width, height);
        self.composite_block(frame, width, height, &laid, &style, align, block_left, block_top)
    }

    /// Lay out and draw one line at absolute pixel coordinates.
    fn draw_block_at(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        text: &str,
        x: f32,
        y: f32,
        align: CaptionAlign,
    ) -> bool {
        self.last_rows = None;
        if width == 0 || height == 0 || frame.len() < (width as usize) * (height as usize) * 4 {
            return false;
        }
        let laid = {
            let max_width = self.wrap_width(width);
            layout(&mut self.font, text, &self.style, max_width)
        };
        if laid.is_empty() || laid.width <= 0.0 {
            return false;
        }

        let style = self.style.clone();
        let fw = width as f32;
        let fh = height as f32;
        let block_left = match align {
            CaptionAlign::Left => x,
            CaptionAlign::Center => x - laid.width * 0.5,
            CaptionAlign::Right => x - laid.width,
        };
        let block_top = y;
        let pad = style.padding;
        let block_left = block_left.clamp(pad, (fw - laid.width - pad).max(pad));
        let block_top = block_top.clamp(pad, (fh - laid.height - pad).max(pad));
        self.composite_block(frame, width, height, &laid, &style, align, block_left, block_top)
    }

    fn composite_block(
        &mut self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        laid: &LaidOutText,
        style: &CaptionStyle,
        align: CaptionAlign,
        block_left: f32,
        block_top: f32,
    ) -> bool {

        // The mask needs room for the outline and the shadow to spread into.
        let outline = style.outline_width.max(0.0);
        let shadow_on = style.shadow_color[3] > 0;
        let pad_left = outline
            + if shadow_on {
                (-style.shadow_offset[0]).max(0.0)
            } else {
                0.0
            };
        let pad_top = outline
            + if shadow_on {
                (-style.shadow_offset[1]).max(0.0)
            } else {
                0.0
            };
        let pad_right = outline
            + if shadow_on {
                style.shadow_offset[0].max(0.0)
            } else {
                0.0
            };
        let pad_bottom = outline
            + if shadow_on {
                style.shadow_offset[1].max(0.0)
            } else {
                0.0
            };

        let mask_x = (block_left - pad_left).floor() as i32;
        let mask_y = (block_top - pad_top).floor() as i32;
        let mask_w = (laid.width + pad_left + pad_right).ceil() as u32 + 2;
        let mask_h = (laid.height + pad_top + pad_bottom).ceil() as u32 + 2;

        // Text coverage, in mask-local coordinates: the glyphs, or with
        // `grow`, the glyphs grown by it — `None` when a face has no
        // outline to grow from.
        let fill_mask = |this: &mut Self, grow: Option<f32>| -> Option<Vec<u8>> {
            let mut mask = vec![0u8; (mask_w as usize) * (mask_h as usize)];
            for (i, line) in laid.lines.iter().enumerate() {
                if line.text.is_empty() {
                    continue;
                }
                let line_left = match align {
                    CaptionAlign::Left => block_left,
                    CaptionAlign::Center => block_left + (laid.width - line.width) * 0.5,
                    CaptionAlign::Right => block_left + (laid.width - line.width),
                };
                let baseline = block_top + laid.first_baseline + laid.line_advance * i as f32;
                let drawn = this.draw_line_into_mask(
                    &mut mask,
                    mask_w,
                    mask_h,
                    line_left - mask_x as f32,
                    baseline - mask_y as f32,
                    &line.text,
                    grow,
                );
                if !drawn {
                    return None;
                }
            }
            Some(mask)
        };
        let mask = fill_mask(self, None).unwrap_or_default();

        // 1. Background box behind the whole block.
        if style.background[3] > 0 {
            fill_rect(
                frame,
                width,
                height,
                block_left - style.padding,
                block_top - style.padding,
                laid.width + style.padding * 2.0,
                laid.height + style.padding * 2.0,
                style.background,
            );
        }

        // 2. Drop shadow — the same mask, offset.
        if shadow_on {
            blend_mask(
                frame,
                width,
                height,
                &mask,
                mask_w,
                mask_h,
                mask_x + style.shadow_offset[0].round() as i32,
                mask_y + style.shadow_offset[1].round() as i32,
                style.shadow_color,
            );
        }

        // 3. Outline — the glyphs grown by `outline_width`, drawn under the
        // fill: from their outlines where the face has them, else the mask
        // grown.
        if outline > 0.0 && style.outline_color[3] > 0 {
            let grown = fill_mask(self, Some(outline)).unwrap_or_else(|| dilate(&mask, mask_w, mask_h, outline));
            blend_mask(
                frame,
                width,
                height,
                &grown,
                mask_w,
                mask_h,
                mask_x,
                mask_y,
                style.outline_color,
            );
        }

        // 4. Fill.
        blend_mask(
            frame,
            width,
            height,
            &mask,
            mask_w,
            mask_h,
            mask_x,
            mask_y,
            style.color,
        );

        // The band everything above was written into: the mask, the shadow at
        // its offset, and the background box, whichever reaches furthest.
        let shadow_dy = if shadow_on {
            style.shadow_offset[1].round() as i32
        } else {
            0
        };
        let top = mask_y
            .min(mask_y + shadow_dy)
            .min((block_top - style.padding).floor() as i32);
        let bottom = (mask_y + mask_h as i32)
            .max(mask_y + shadow_dy + mask_h as i32)
            .max((block_top + laid.height + style.padding).ceil() as i32);
        self.last_rows = Some((
            top.clamp(0, height as i32) as u32,
            bottom.clamp(0, height as i32) as u32,
        ));
        true
    }

    /// Top-left corner of the text block in frame pixels.
    fn place(
        &self,
        laid: &LaidOutText,
        style: &CaptionStyle,
        align: CaptionAlign,
        placement: Placement,
        width: u32,
        height: u32,
    ) -> (f32, f32) {
        let (fw, fh) = (width as f32, height as f32);

        // Horizontal: an explicit `position` anchors the block per its
        // alignment; otherwise it sits inside the centered safe area.
        let left = match placement.position {
            Some(p) => {
                let anchor = p.clamp(0.0, 1.0) * fw;
                match align {
                    CaptionAlign::Left => anchor,
                    CaptionAlign::Center => anchor - laid.width * 0.5,
                    CaptionAlign::Right => anchor - laid.width,
                }
            }
            None => {
                let area_w = fw * style.max_width;
                let area_left = (fw - area_w) * 0.5;
                match align {
                    CaptionAlign::Left => area_left,
                    CaptionAlign::Center => area_left + (area_w - laid.width) * 0.5,
                    CaptionAlign::Right => area_left + area_w - laid.width,
                }
            }
        };

        // Vertical: an explicit `line` is the block's top edge as a fraction of
        // frame height; otherwise anchor to an edge with the style margin.
        let top = match placement.line {
            Some(l) => l.clamp(0.0, 1.0) * fh,
            None => match style.anchor {
                CaptionAnchor::Top => style.margin,
                CaptionAnchor::Middle => (fh - laid.height) * 0.5,
                CaptionAnchor::Bottom => fh - style.margin - laid.height,
            },
        };

        // Keep the padded box on screen even for oversized text or margins.
        let pad = style.padding;
        let left = left.clamp(pad, (fw - laid.width - pad).max(pad));
        let top = top.clamp(pad, (fh - laid.height - pad).max(pad));
        (left, top)
    }

    /// Rasterize one line's glyphs into `mask` at `pen_x` / `baseline_y`,
    /// grown by `grow` pixels if given — and then `false`, with the mask
    /// part-drawn, if the face has no outlines to grow.
    #[allow(clippy::too_many_arguments)]
    fn draw_line_into_mask(
        &mut self,
        mask: &mut [u8],
        mask_w: u32,
        mask_h: u32,
        pen_x: f32,
        baseline_y: f32,
        text: &str,
        grow: Option<f32>,
    ) -> bool {
        let size = self.style.font_size;
        let spacing = self.style.letter_spacing;
        let mut pen = pen_x;
        let mut previous: Option<super::order::Placed> = None;
        // The same order `measure` walked, or the drawing and the width disagree.
        let run = super::order::visual(text, &self.font, size);
        // Whether this face's glyphs overlap by design, which decides how two of them
        // share a cell. A property of the file, read once at parse.
        let joins = self.font.composites_run();
        // Two passes, because a mark hangs off another glyph's pen and in a right-to-left
        // run its base comes **after** it on screen. The first walks the advances and
        // records where each glyph would sit; the second moves the attached ones onto
        // their base. A mark's own advance is zero in the font, so it does not disturb
        // the walk, and letter spacing skips it — spacing is between letters, and
        // charging a dot for it would push the word apart.
        let mut pens: Vec<f32> = Vec::with_capacity(run.len());
        for placed in &run {
            if let Some(p) = previous {
                // The same adjustment `measure` accounted for, applied before the
                // glyph is placed rather than after.
                let (l, r) = super::order::pair(p, *placed);
                pen += self.font.kerning_of(l, r, size);
            }
            previous = Some(*placed);
            pens.push(pen);
            let phase = phase_of(pen).1;
            let advance = self.font.glyph_of(placed.gid, size, phase).advance;
            pen += advance;
            if placed.attach.is_none() {
                pen += spacing;
            }
        }

        for (k, placed) in run.iter().enumerate() {
            // `dy` is above the baseline and the mask grows downward.
            let (x, baseline) = match placed.attach {
                Some((to, dx, dy)) => match pens.get(to) {
                    Some(base) => (base + dx, baseline_y - dy),
                    None => (pens[k], baseline_y),
                },
                None => (pens[k], baseline_y),
            };
            // The pen is fractional, so the glyph is rasterized for the quarter
            // pixel it actually falls on rather than being rounded onto a whole
            // one. Rounding per glyph is what makes letter spacing uneven: a
            // 13.7-pixel advance lands on 0, 14, 27, 41, 55 — gaps of 14, 13,
            // 14, 14 — and no amount of care in the compositing fixes it.
            let (origin, phase) = phase_of(x);
            let glyph = match grow {
                None => self.font.glyph_of(placed.gid, size, phase),
                Some(r) => match self.font.grown_glyph_of(placed.gid, size, phase, r) {
                    Some(g) => g,
                    None => return false,
                },
            };
            if glyph.is_blank() {
                continue;
            }
            let gx = origin + glyph.offset_x;
            let gy = baseline.round() as i32 + glyph.offset_y;
            for row in 0..glyph.height {
                let y = gy + row as i32;
                if y < 0 || y >= mask_h as i32 {
                    continue;
                }
                for col in 0..glyph.width {
                    let x = gx + col as i32;
                    if x < 0 || x >= mask_w as i32 {
                        continue;
                    }
                    let v = glyph.coverage[(row * glyph.width + col) as usize];
                    let dst = &mut mask[(y as u32 * mask_w + x as u32) as usize];
                    let a = *dst as u32;
                    let b = v as u32;
                    *dst = if joins {
                        // A face whose letters **join** draws the same stroke twice: the
                        // tail of one letter and the head of the next are the same ink,
                        // so a union darkens every joint. The maximum is what Core
                        // Graphics does for these faces, measured — see
                        // [`crate::loaders::overlap`].
                        a.max(b) as u8
                    } else {
                        // Neighbouring glyphs genuinely share edge pixels once
                        // they are placed at subpixel positions, each covering
                        // part of one. Taking the stronger loses the other's
                        // contribution and thins the gap between letters; the
                        // union `a + b - ab` is what two partial coverages of
                        // the same pixel actually come to.
                        (a + b - (a * b) / 255).min(255) as u8
                    };
                }
            }
        }
        true
    }
}

/// A fractional pen as `(whole pixel, subpixel phase)`.
///
/// A fraction that rounds up to a whole pixel is that pixel.
fn phase_of(pen: f32) -> (i32, usize) {
    let cell = pen.floor();
    let phase = ((pen - cell) * SUBPIXEL_PHASES as f32).round() as i32;
    if phase as usize >= SUBPIXEL_PHASES {
        (cell as i32 + 1, 0)
    } else {
        (cell as i32, phase as usize)
    }
}

/// Grow a coverage mask by `radius` pixels — an outline, or the spread of a
/// drop shadow.
///
/// The shape wanted is the union of discs of `radius` centred on every point of
/// the glyph, which is the same as "every pixel within `radius` of the glyph".
/// So the mask is turned into a **distance field** and thresholded: coverage is
/// full where the distance is under the radius, zero where it is more than half a
/// pixel beyond, and a ramp across the pixel between — which is what makes the
/// grown edge as smooth as the one it came from.
///
/// This replaced repeated max-filtering with 4- and 8-neighbour kernels. That
/// approach could only grow by whole pixels, so a 1.5-pixel outline became 2 and
/// a 0.4-pixel one vanished, and alternating the two kernels approximates a disc
/// with an octagon — plainly visible at the one to three pixels an outline
/// actually uses.
fn dilate(mask: &[u8], width: u32, height: u32, radius: f32) -> Vec<u8> {
    if radius <= 0.0 || width == 0 || height == 0 {
        return mask.to_vec();
    }
    let distance = distance_field(mask, width, height);
    distance
        .iter()
        .zip(mask)
        .map(|(d, &original)| {
            // A half-pixel ramp centred on the radius: full inside, empty
            // outside, antialiased across the boundary.
            let grown = (radius + 0.5 - d).clamp(0.0, 1.0);
            let grown = (grown * 255.0).round() as u8;
            // Never below what was already there. Growing a shape is a union
            // with it, and the ramp alone does not say so: a pixel well inside
            // the glyph has a *negative* distance to its boundary, which the
            // field does not carry, so for a radius under half a pixel the ramp
            // came out below full and the dilation dimmed the very pixels it was
            // supposed to be spreading.
            grown.max(original)
        })
        .collect()
}

/// Distance in pixels from each cell to the nearest covered one.
///
/// Seeded from the coverage itself rather than from a threshold, so the field is
/// accurate to a fraction of a pixel: a cell that is fully inside is half a pixel
/// past the edge, a cell the edge crosses is `0.5 - coverage` from it, and one
/// that is empty has no seed and takes its distance from its neighbours. Rounding
/// the mask to inside-or-out first would quantize every grown edge to the pixel
/// grid, which is the thing being avoided.
fn distance_field(mask: &[u8], width: u32, height: u32) -> Vec<f32> {
    const FAR: f32 = 1.0e10;
    let mut squared: Vec<f32> = mask
        .iter()
        .map(|&v| {
            let coverage = v as f32 / 255.0;
            if coverage <= 0.0 {
                FAR
            } else {
                // Inside is negative distance, but an outline only grows
                // outwards, so it clamps to zero here.
                let d = (0.5 - coverage).max(0.0);
                d * d
            }
        })
        .collect();

    // Felzenszwalb's transform: exact Euclidean distance in two passes of a
    // one-dimensional lower envelope, rows then columns. Doing it as two 1-D
    // passes is what makes it exact *and* linear; a single pass of chamfer
    // weights is neither.
    let mut row = vec![0f32; width.max(height) as usize];
    for y in 0..height as usize {
        let base = y * width as usize;
        row[..width as usize].copy_from_slice(&squared[base..base + width as usize]);
        let out = transform_1d(&row[..width as usize]);
        squared[base..base + width as usize].copy_from_slice(&out);
    }
    for x in 0..width as usize {
        for y in 0..height as usize {
            row[y] = squared[y * width as usize + x];
        }
        let out = transform_1d(&row[..height as usize]);
        for (y, v) in out.iter().enumerate() {
            squared[y * width as usize + x] = *v;
        }
    }
    squared.iter().map(|d| d.max(0.0).sqrt()).collect()
}

/// `D(x) = min_y ((x - y)² + f(y))`, by the lower envelope of parabolas.
fn transform_1d(f: &[f32]) -> Vec<f32> {
    let n = f.len();
    let mut out = vec![0f32; n];
    if n == 0 {
        return out;
    }
    // `v` holds the parabolas making up the envelope so far, `z` the boundaries
    // between them.
    let mut v = vec![0usize; n];
    let mut z = vec![0f32; n + 1];
    let mut k = 0usize;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        // Where the parabola at `q` overtakes the one currently on top. Walking
        // back while it overtakes before that parabola's own start is what keeps
        // the envelope correct.
        loop {
            let p = v[k];
            let intersect = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32))
                / (2.0 * q as f32 - 2.0 * p as f32);
            if intersect <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = intersect;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }
    // Walk the envelope once, left to right, taking the parabola that owns each
    // position. `k` only ever moves forward, which is what makes this linear.
    let mut k = 0usize;
    for (q, slot) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let p = v[k];
        let d = q as f32 - p as f32;
        *slot = d * d + f[p];
    }
    out
}


/// Glyph coverage corrected for the text's colour, as a table over the 256
/// coverage values.
///
/// Coverage is a fraction of the pixel's area, and blending it in linear
/// light is physically right — but it is not what text has ever looked like.
/// CoreText, and Skia on every platform, put coverage down in the display's
/// encoding, where half coverage of black on white is the encoded midpoint
/// (a fifth of the light) rather than the linear one (half of it). Measured
/// against CoreText on the same face, size and string, a linear blend set
/// black-on-white text 22% light and white-on-black 44% heavy: thin dark
/// text and fat light text, the commonest complaint about "gamma-correct"
/// text.
///
/// So, as Skia does, the coverage is bent by the text's luminance: the table
/// gives the coverage whose linear blend lands where an encoded blend would,
/// against the background that contrasts with the text most (white behind
/// dark text, black behind light). The blend stays linear, so this is the
/// same on the CPU and through the GPU caption pass; a mid-grey colour has
/// no contrasting background and is left alone.
fn text_coverage_table(color: [u8; 4]) -> [u8; 256] {
    let mut table = [0u8; 256];
    let lin = [0, 1, 2].map(|c| srgb_to_linear(color[c]));
    let src_lin = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    let src = linear_to_srgb_f32(src_lin);
    let dst = 1.0 - src;
    let dst_lin = srgb_to_linear_f32(dst);
    for (i, t) in table.iter_mut().enumerate() {
        let c = i as f32 / 255.0;
        let v = if (src - dst).abs() < 1.0 / 64.0 {
            c
        } else {
            (srgb_to_linear_f32(src * c + dst * (1.0 - c)) - dst_lin) / (src_lin - dst_lin)
        };
        *t = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    }
    table
}

/// Composite `color` through `mask` into an RGBA8 frame at `(ox, oy)`, the
/// mask's coverage corrected for the colour by [`text_coverage_table`].
#[allow(clippy::too_many_arguments)]
fn blend_mask(
    frame: &mut [u8],
    width: u32,
    height: u32,
    mask: &[u8],
    mask_w: u32,
    mask_h: u32,
    ox: i32,
    oy: i32,
    color: [u8; 4],
) {
    if color[3] == 0 {
        return;
    }
    let table = text_coverage_table(color);
    for my in 0..mask_h as i32 {
        let y = oy + my;
        if y < 0 || y >= height as i32 {
            continue;
        }
        for mx in 0..mask_w as i32 {
            let x = ox + mx;
            if x < 0 || x >= width as i32 {
                continue;
            }
            let coverage = table[mask[(my as u32 * mask_w + mx as u32) as usize] as usize];
            if coverage == 0 {
                continue;
            }
            let alpha = (color[3] as u32 * coverage as u32) / 255;
            blend_pixel(frame, width, x as u32, y as u32, color, alpha as u8);
        }
    }
}

/// Fill an axis-aligned rectangle, antialiasing its fractional edges.
#[allow(clippy::too_many_arguments)]
fn fill_rect(
    frame: &mut [u8],
    width: u32,
    height: u32,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    color: [u8; 4],
) {
    if color[3] == 0 || w <= 0.0 || h <= 0.0 {
        return;
    }
    let x0 = x.max(0.0);
    let y0 = y.max(0.0);
    let x1 = (x + w).min(width as f32);
    let y1 = (y + h).min(height as f32);
    if x1 <= x0 || y1 <= y0 {
        return;
    }
    for py in y0.floor() as u32..(y1.ceil() as u32).min(height) {
        // Vertical overlap of this pixel row with the rectangle.
        let cover_y = (y1.min(py as f32 + 1.0) - y0.max(py as f32)).clamp(0.0, 1.0);
        if cover_y <= 0.0 {
            continue;
        }
        for px in x0.floor() as u32..(x1.ceil() as u32).min(width) {
            let cover_x = (x1.min(px as f32 + 1.0) - x0.max(px as f32)).clamp(0.0, 1.0);
            if cover_x <= 0.0 {
                continue;
            }
            let alpha = (color[3] as f32 * cover_x * cover_y).round() as u8;
            blend_pixel(frame, width, px, py, color, alpha);
        }
    }
}

/// Source-over one straight-alpha pixel.
fn blend_pixel(frame: &mut [u8], width: u32, x: u32, y: u32, color: [u8; 4], alpha: u8) {
    if alpha == 0 {
        return;
    }
    let i = ((y * width + x) * 4) as usize;
    if i + 3 >= frame.len() {
        return;
    }
    // Alpha is a fraction and is not gamma-encoded, so it needs no conversion.
    let sa = alpha as f32 / 255.0;
    let da = frame[i + 3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        frame[i..i + 4].copy_from_slice(&[0, 0, 0, 0]);
        return;
    }
    // Colour is mixed as *light*, not as the bytes that encode it.
    //
    // These bytes are sRGB-encoded — the renderer's shader encodes on the way
    // out, and an sRGB target encodes on write — so 128 is about a fifth of the
    // light of 255, not half. Mixing the encodings directly puts half coverage of
    // white on black at 128 where the light says 188, a pixel less than half as
    // bright as it should be, which is why antialiased text rendered that way
    // reads thin on a dark background and heavy on a light one.
    //
    // The GPU caption pass composites through an sRGB view of the target so its
    // fixed-function blend is in linear too; `tests/captions_overlay_gpu.rs`
    // holds the two within a channel or two of each other.
    for c in 0..3 {
        let src = srgb_to_linear(color[c]);
        let dst = srgb_to_linear(frame[i + c]);
        let out = (src * sa + dst * da * (1.0 - sa)) / out_a;
        frame[i + c] = linear_to_srgb(out);
    }
    frame[i + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::captions::{CaptionAnchor, CaptionStyle};

    const W: u32 = 160;
    const H: u32 = 90;

    fn opaque_frame() -> Vec<u8> {
        let mut f = vec![0u8; (W * H * 4) as usize];
        for px in f.chunks_exact_mut(4) {
            px.copy_from_slice(&[0, 0, 0, 255]);
        }
        f
    }

    /// The horizontal centre of mass of each vertical run of ink.
    ///
    /// The centroid is what reveals a glyph's *subpixel* position: a stem whose
    /// edge falls two thirds of the way across a pixel puts two thirds of the
    /// coverage there, and averaging the coverage recovers that. Ink bounds
    /// alone are quantized to whole pixels and would hide exactly the effect
    /// being measured.
    fn stem_centroids(frame: &[u8]) -> Vec<f32> {
        // Total ink per column.
        let mut columns = vec![0f32; W as usize];
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                columns[x as usize] += frame[i] as f32;
            }
        }
        // Group non-empty columns into runs and take each run's centroid.
        let mut out = Vec::new();
        let mut run: Vec<(usize, f32)> = Vec::new();
        for (x, &v) in columns.iter().enumerate() {
            if v > 0.0 {
                run.push((x, v));
            } else if !run.is_empty() {
                let weight: f32 = run.iter().map(|(_, v)| *v).sum();
                let moment: f32 = run.iter().map(|(x, v)| *x as f32 * *v).sum();
                out.push(moment / weight);
                run.clear();
            }
        }
        if !run.is_empty() {
            let weight: f32 = run.iter().map(|(_, v)| *v).sum();
            let moment: f32 = run.iter().map(|(x, v)| *x as f32 * *v).sum();
            out.push(moment / weight);
        }
        out
    }

    /// Repeated glyphs are evenly spaced, to a fraction of a pixel.
    ///
    /// This is the thing subpixel positioning is for. Advances are fractional —
    /// at this size the advance is not a whole number — and rounding the pen per
    /// glyph made the gaps alternate between two integers. Here the gaps between
    /// successive stems must all agree to well under a pixel, which is only
    /// possible if each glyph was rasterized for the position it actually
    /// occupies.
    #[test]
    fn repeated_glyphs_are_evenly_spaced() {
        let mut painter = CaptionPainter::new();
        painter.style.font_size = 13.7;
        painter.style.outline_width = 0.0;
        painter.style.shadow_offset = [0.0, 0.0];
        painter.style.background = [0, 0, 0, 0];
        let track = CaptionTrack::new().cue(0.0, 2.0, "IIIIIIII");

        let mut frame = opaque_frame();
        assert!(painter.burn_in(&mut frame, W, H, &track, 1.0));

        let centroids = stem_centroids(&frame);
        assert!(
            centroids.len() >= 6,
            "expected a row of separate stems, found {}",
            centroids.len()
        );
        let gaps: Vec<f32> = centroids.windows(2).map(|w| w[1] - w[0]).collect();
        let smallest = gaps.iter().cloned().fold(f32::MAX, f32::min);
        let largest = gaps.iter().cloned().fold(f32::MIN, f32::max);
        // Quarter-pixel phases bound the placement error at an eighth of a pixel
        // each side, so successive gaps cannot differ by more than a quarter.
        // Rounding the pen instead allows a whole pixel of swing.
        assert!(
            largest - smallest < 0.5,
            "gaps ranged {smallest}..{largest} — spacing is uneven: {gaps:?}"
        );
    }

    /// Vertical centroid of each band of inked rows, top to bottom.
    ///
    /// The horizontal twin of this is `stem_centroids`; a band here is a run of
    /// rows that carry any ink, and its centroid is weighted by how much.
    fn line_centroids(frame: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut run: Vec<(u32, f32)> = Vec::new();
        for y in 0..H {
            let mut sum = 0.0f32;
            let mut peak = 0u8;
            for x in 0..W {
                let v = frame[((y * W + x) * 4) as usize];
                sum += v as f32;
                peak = peak.max(v);
            }
            if peak > 8 {
                run.push((y, sum));
            } else if !run.is_empty() {
                let weight: f32 = run.iter().map(|(_, v)| *v).sum();
                let moment: f32 = run.iter().map(|(y, v)| *y as f32 * *v).sum();
                out.push(moment / weight);
                run.clear();
            }
        }
        if !run.is_empty() {
            let weight: f32 = run.iter().map(|(_, v)| *v).sum();
            let moment: f32 = run.iter().map(|(y, v)| *y as f32 * *v).sum();
            out.push(moment / weight);
        }
        out
    }

    /// Lines are evenly spaced, and the spacing does not drift down the block.
    ///
    /// The baseline is deliberately rounded to the pixel grid, and there are no
    /// vertical subpixel phases to match the horizontal ones. That is not an
    /// oversight: a glyph's horizontal stems sit on the baseline and the x-height
    /// line, so rounding the baseline is what keeps them crisp, and phasing it
    /// would soften every horizontal edge to fix half a pixel nobody can see.
    /// The horizontal case was different — there the error showed up as visibly
    /// uneven gaps between glyphs inside a word.
    ///
    /// What rounding must not do is accumulate. Each baseline is computed from
    /// the line index rather than by adding an advance to the line above, so the
    /// error stays bounded at half a pixel however tall the block gets. This test
    /// is what stops that being rewritten into an accumulation.
    #[test]
    fn lines_are_evenly_spaced_and_do_not_drift() {
        let mut painter = CaptionPainter::new();
        painter.style.font_size = 8.0;
        // A fractional advance, so rounding has something to get wrong.
        painter.style.line_height = 1.31;
        painter.style.outline_width = 0.0;
        painter.style.shadow_offset = [0.0, 0.0];
        painter.style.background = [0, 0, 0, 0];
        let track = CaptionTrack::new().cue(0.0, 2.0, "I\nI\nI\nI\nI\nI");

        let mut frame = opaque_frame();
        assert!(painter.burn_in(&mut frame, W, H, &track, 1.0));

        let centroids = line_centroids(&frame);
        assert_eq!(
            centroids.len(),
            6,
            "expected six separated lines, found {centroids:?}"
        );

        let advance = 8.0 * 1.31;
        let gaps: Vec<f32> = centroids.windows(2).map(|w| w[1] - w[0]).collect();
        for gap in &gaps {
            assert!(
                (gap - advance).abs() <= 1.0,
                "gap {gap} is more than a pixel from the {advance} advance: {gaps:?}"
            );
        }

        // Five advances of drift-free rounding land within half a pixel of the
        // ideal; five accumulated roundings could be off by two or three.
        let span = centroids[5] - centroids[0];
        assert!(
            (span - advance * 5.0).abs() <= 1.0,
            "six lines spanned {span}, expected {}",
            advance * 5.0
        );
    }

    /// Bounding box of pixels that differ from pure opaque black.
    fn ink_bounds(frame: &[u8]) -> Option<(u32, u32, u32, u32)> {
        let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0u32, 0u32);
        for y in 0..H {
            for x in 0..W {
                let i = ((y * W + x) * 4) as usize;
                if frame[i] > 8 || frame[i + 1] > 8 || frame[i + 2] > 8 {
                    x0 = x0.min(x);
                    y0 = y0.min(y);
                    x1 = x1.max(x);
                    y1 = y1.max(y);
                }
            }
        }
        (x0 != u32::MAX).then_some((x0, y0, x1, y1))
    }

    fn painter(anchor: CaptionAnchor) -> CaptionPainter {
        CaptionPainter::new().style(
            CaptionStyle::default()
                .font_size(10.0)
                .margin(8.0)
                .padding(2.0)
                .outline([0, 0, 0, 255], 1.0)
                .background([0, 0, 0, 0])
                .anchor(anchor),
        )
    }

    #[test]
    fn drawing_puts_white_text_on_the_frame() {
        let mut frame = opaque_frame();
        let mut p = painter(CaptionAnchor::Bottom);
        assert!(p.draw_text(&mut frame, W, H, "Hi"));
        let (_, y0, _, y1) = ink_bounds(&frame).expect("expected ink");
        // Bottom-anchored text lands in the lower half.
        assert!(y0 > H / 2, "y0={y0}");
        assert!(y1 < H, "y1={y1}");
        // Fully opaque frames stay opaque.
        assert!(frame.chunks_exact(4).all(|px| px[3] == 255));
    }

    #[test]
    fn anchor_moves_the_block_vertically() {
        let mut top = opaque_frame();
        painter(CaptionAnchor::Top).draw_text(&mut top, W, H, "Hi");
        let mut middle = opaque_frame();
        painter(CaptionAnchor::Middle).draw_text(&mut middle, W, H, "Hi");
        let mut bottom = opaque_frame();
        painter(CaptionAnchor::Bottom).draw_text(&mut bottom, W, H, "Hi");

        let t = ink_bounds(&top).unwrap().1;
        let m = ink_bounds(&middle).unwrap().1;
        let b = ink_bounds(&bottom).unwrap().1;
        assert!(t < m && m < b, "top={t} middle={m} bottom={b}");
    }

    #[test]
    fn alignment_moves_the_block_horizontally() {
        let mut left = opaque_frame();
        let mut p = painter(CaptionAnchor::Bottom);
        p.style = p.style.clone().align(CaptionAlign::Left);
        p.draw_text(&mut left, W, H, "Hi");

        let mut right = opaque_frame();
        let mut p = painter(CaptionAnchor::Bottom);
        p.style = p.style.clone().align(CaptionAlign::Right);
        p.draw_text(&mut right, W, H, "Hi");

        assert!(ink_bounds(&left).unwrap().0 < ink_bounds(&right).unwrap().0);
    }

    #[test]
    fn overlay_starts_transparent_and_only_the_caption_is_opaque() {
        let track = CaptionTrack::new().cue(0.0, 1.0, "Hi");
        let mut p = painter(CaptionAnchor::Bottom);
        let overlay = p.render_overlay(W, H, &track, 0.5);
        assert_eq!(overlay.len(), (W * H * 4) as usize);
        let opaque = overlay.chunks_exact(4).filter(|px| px[3] > 200).count();
        let clear = overlay.chunks_exact(4).filter(|px| px[3] == 0).count();
        assert!(opaque > 0, "expected caption pixels");
        assert!(clear > opaque, "expected mostly-transparent overlay");

        // Nothing active → fully transparent.
        let empty = p.render_overlay(W, H, &track, 5.0);
        assert!(empty.iter().all(|&b| b == 0));
    }

    #[test]
    fn background_box_covers_more_than_the_glyphs() {
        let mut plain = opaque_frame();
        painter(CaptionAnchor::Bottom).draw_text(&mut plain, W, H, "Hi");
        let plain_ink = plain
            .chunks_exact(4)
            .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
            .count();

        let mut boxed = opaque_frame();
        let mut p = painter(CaptionAnchor::Bottom);
        p.style = p.style.clone().background([40, 40, 200, 255]);
        p.draw_text(&mut boxed, W, H, "Hi");
        let boxed_ink = boxed
            .chunks_exact(4)
            .filter(|px| px[0] > 8 || px[1] > 8 || px[2] > 8)
            .count();
        assert!(boxed_ink > plain_ink, "{boxed_ink} vs {plain_ink}");
    }

    #[test]
    fn burn_in_follows_the_timeline() {
        let track = CaptionTrack::new()
            .cue(0.0, 1.0, "first")
            .cue(1.0, 2.0, "second");
        let mut p = painter(CaptionAnchor::Bottom);
        let mut frame = opaque_frame();
        assert!(p.burn_in(&mut frame, W, H, &track, 0.5));
        let mut frame = opaque_frame();
        assert!(p.burn_in(&mut frame, W, H, &track, 1.5));
        let mut frame = opaque_frame();
        assert!(!p.burn_in(&mut frame, W, H, &track, 3.0));
        assert!(ink_bounds(&frame).is_none(), "nothing should be drawn");
    }

    #[test]
    fn simultaneous_cues_with_the_same_placement_stack_into_one_block() {
        let track = CaptionTrack::new()
            .cue(0.0, 2.0, "one")
            .cue(0.0, 2.0, "two");
        let mut p = painter(CaptionAnchor::Bottom);
        let mut merged = opaque_frame();
        p.burn_in(&mut merged, W, H, &track, 1.0);
        let (_, y0, _, y1) = ink_bounds(&merged).unwrap();

        let mut single = opaque_frame();
        p.draw_text(&mut single, W, H, "one");
        let (_, sy0, _, sy1) = ink_bounds(&single).unwrap();
        assert!(y1 - y0 > sy1 - sy0, "two lines must be taller than one");
    }

    #[test]
    fn blocks_stay_inside_the_frame_even_with_absurd_margins() {
        let mut frame = opaque_frame();
        let mut p = CaptionPainter::new().style(
            CaptionStyle::default()
                .font_size(10.0)
                .margin(10_000.0)
                .background([0, 0, 0, 0]),
        );
        p.draw_text(&mut frame, W, H, "edge");
        let (x0, y0, x1, y1) = ink_bounds(&frame).unwrap();
        assert!(x1 < W && y1 < H, "ink escaped the frame");
        let _ = (x0, y0);
    }

    #[test]
    fn undersized_buffers_are_rejected_rather_than_panicking() {
        let mut tiny = vec![0u8; 16];
        let mut p = painter(CaptionAnchor::Bottom);
        assert!(!p.draw_text(&mut tiny, W, H, "Hi"));
        assert!(!p.draw_text(&mut tiny, 0, 0, "Hi"));
        // Blank text draws nothing but reports cleanly.
        let mut frame = opaque_frame();
        assert!(!p.draw_text(&mut frame, W, H, "   "));
    }

    #[test]
    fn dilation_grows_the_mask_by_the_radius() {
        let (w, h) = (9u32, 9u32);
        let mut mask = vec![0u8; (w * h) as usize];
        mask[(4 * w + 4) as usize] = 255;

        // Zero radius is the identity.
        assert_eq!(dilate(&mask, w, h, 0.0), mask);

        // Radius 1 reaches the diagonal neighbours: they are √2 ≈ 1.41 away and
        // the ramp runs out at 1.5, so they get a little coverage. A disc does
        // that; the octagon this replaced did not, which is why it counted five
        // pixels here and this counts nine.
        let r1 = dilate(&mask, w, h, 1.0);
        assert_eq!(r1.iter().filter(|&&v| v > 0).count(), 9);
        let at = |x: u32, y: u32| r1[(y * w + x) as usize];
        assert_eq!(at(4, 4), 255, "the centre stays full");
        assert!(at(3, 4) > at(3, 3), "the diagonal is further away than the side");
    }

    /// The grown shape is round, not a polygon.
    ///
    /// Measured as the furthest lit pixel along an axis against the furthest
    /// along the diagonal, in Euclidean distance: for a disc they agree. The
    /// octagon this replaced reached its full radius along the axes and fell
    /// short across the diagonals, and at the one-to-three pixels an outline
    /// actually uses that is a visibly cornered edge.
    #[test]
    fn the_grown_shape_is_a_disc() {
        let (w, h) = (41u32, 41u32);
        let mut mask = vec![0u8; (w * h) as usize];
        mask[(20 * w + 20) as usize] = 255;
        let radius = 8.0f32;
        let grown = dilate(&mask, w, h, radius);

        let mut axis = 0.0f32;
        let mut diagonal = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                if grown[(y * w + x) as usize] == 0 {
                    continue;
                }
                let (dx, dy) = (x as f32 - 20.0, y as f32 - 20.0);
                let d = (dx * dx + dy * dy).sqrt();
                // Near an axis versus near the 45° diagonal.
                if dy.abs() < 0.5 || dx.abs() < 0.5 {
                    axis = axis.max(d);
                } else if (dx.abs() - dy.abs()).abs() < 0.5 {
                    diagonal = diagonal.max(d);
                }
            }
        }
        assert!(
            (axis - diagonal).abs() < 1.5,
            "reach {axis} along the axis but {diagonal} along the diagonal — not round"
        );
        assert!(
            (axis - (radius + 0.5)).abs() < 1.5,
            "reach {axis} for a radius of {radius}"
        );
    }

    /// A fractional radius is a fractional radius.
    ///
    /// The previous implementation rounded to whole steps, so 1.5 became 2 and
    /// anything under 0.5 disappeared. An outline width is a style choice and
    /// half-pixel values are ordinary.
    #[test]
    fn a_fractional_radius_grows_by_a_fraction() {
        let (w, h) = (21u32, 21u32);
        let mut mask = vec![0u8; (w * h) as usize];
        mask[(10 * w + 10) as usize] = 255;
        let ink = |r: f32| {
            dilate(&mask, w, h, r)
                .iter()
                .map(|v| *v as u32)
                .sum::<u32>()
        };
        let (a, b, c) = (ink(1.0), ink(1.5), ink(2.0));
        assert!(a < b && b < c, "1.0 -> {a}, 1.5 -> {b}, 2.0 -> {c}");

        // A sub-pixel radius shows at an *edge*, which is where an outline meets
        // the glyph. Around a lone fully-lit pixel it correctly does nothing —
        // nothing else is within a third of a pixel — but a half-covered pixel is
        // already half a pixel from the boundary, so growing the boundary raises
        // it. Rounding the radius to whole steps lost both cases.
        let mut edge = vec![0u8; (w * h) as usize];
        edge[(10 * w + 10) as usize] = 128;
        let edge_ink = |r: f32| {
            dilate(&edge, w, h, r)
                .iter()
                .map(|v| *v as u32)
                .sum::<u32>()
        };
        assert!(
            edge_ink(0.3) > edge_ink(0.0),
            "a third of a pixel did not thicken an edge: {} vs {}",
            edge_ink(0.3),
            edge_ink(0.0)
        );
        assert!(edge_ink(0.3) < edge_ink(0.6), "and 0.6 should thicken it more");
    }

    #[test]
    fn source_over_blending_matches_the_straight_alpha_formula() {
        // Half coverage of white over opaque black: 188, the byte that encodes
        // half the light. This asserted 128 before — the result of averaging the
        // encodings rather than the light, half as bright as it should be.
        let mut frame = vec![0, 0, 0, 255u8];
        blend_pixel(&mut frame, 1, 0, 0, [255, 255, 255, 255], 128);
        assert!((frame[0] as i32 - 188).abs() <= 1, "{:?}", frame);
        assert_eq!(frame[3], 255);

        // The same over a transparent buffer keeps the source colour: there is
        // no destination light to mix with, so the encoding passes through
        // untouched and the conversions must not perturb it.
        let mut frame = vec![0, 0, 0, 0u8];
        blend_pixel(&mut frame, 1, 0, 0, [255, 0, 0, 255], 128);
        assert_eq!(&frame[..3], &[255, 0, 0]);
        assert_eq!(frame[3], 128);
    }

    /// A coverage of *c* puts *c* of the light on the pixel.
    ///
    /// Which is only true of a blend in linear. Mixing the encodings instead
    /// compresses the range towards the dark end — a quarter coverage of white
    /// on black came out at 64 where the light says 137, so thin strokes all but
    /// vanished.
    #[test]
    fn coverage_lands_the_light_it_should() {
        for (coverage, expected) in [(64u8, 137i32), (128, 188), (192, 225), (255, 255)] {
            let mut frame = vec![0, 0, 0, 255u8];
            blend_pixel(&mut frame, 1, 0, 0, [255, 255, 255, 255], coverage);
            assert!(
                (frame[0] as i32 - expected).abs() <= 2,
                "coverage {coverage} gave {}, expected about {expected}",
                frame[0]
            );
        }
    }

    /// The coverage table bends coverage by the text's polarity: up for dark
    /// text, down for light, as an encoded blend against the contrasting
    /// background would put it — and leaves the ends and mid-grey alone.
    #[test]
    fn text_coverage_follows_the_encoded_blend_for_each_polarity() {
        let white = text_coverage_table([255, 255, 255, 255]);
        let black = text_coverage_table([0, 0, 0, 255]);
        let grey = text_coverage_table([128, 128, 128, 255]);
        // Half coverage, encoded 128 of 255: a fifth of the light for white
        // on black, four fifths of the dark for black on white.
        assert!((white[128] as i32 - 55).abs() <= 1, "{}", white[128]);
        assert!((black[128] as i32 - 200).abs() <= 1, "{}", black[128]);
        for t in [&white, &black, &grey] {
            assert_eq!((t[0], t[255]), (0, 255));
            assert!(t.windows(2).all(|w| w[0] <= w[1]), "not monotonic");
        }
        assert!((0..256).all(|i| (grey[i] as i32 - i as i32).abs() <= 1));
    }

    /// Text weighs the same on either polarity, measured in the encoding.
    ///
    /// That is what CoreText draws: with the same face, size and string, its
    /// encoded ink is identical black-on-white and white-on-black. A plain
    /// linear blend made the white text 1.5× the black, and against
    /// CoreText 22% light dark-on-light and 44% heavy light-on-dark.
    #[test]
    fn text_weighs_the_same_on_either_polarity() {
        let ink = |fg: u8, bg: u8| -> f32 {
            let (w, h) = (200u32, 40u32);
            let mut frame = vec![0u8; (w * h * 4) as usize];
            for p in frame.chunks_mut(4) {
                p.copy_from_slice(&[bg, bg, bg, 255]);
            }
            // 13 px: the 8-row bitmap face lands off the pixel grid, so its
            // edges have partial coverage.
            let style = CaptionStyle {
                font_size: 13.0,
                color: [fg, fg, fg, 255],
                background: [0; 4],
                outline_width: 0.0,
                padding: 0.0,
                margin: 0.0,
                max_width: 1.0,
                ..Default::default()
            };
            let mut p = CaptionPainter::with_font(CaptionFont::builtin()).style(style);
            assert!(p.draw_text_at(&mut frame, w, h, 4.3, 4.0, "Weigh me"));
            frame.chunks(4).map(|p| (p[0] as f32 - bg as f32).abs() / 255.0).sum()
        };
        let (dark, light) = (ink(0, 255), ink(255, 0));
        assert!((dark / light - 1.0).abs() < 0.01, "black on white {dark}, white on black {light}");
    }

    /// The coverage an outline of `radius` should have, from the glyph's own
    /// geometry: per pixel, the share of an 8×8 grid of points inside the
    /// outline or within `radius` of it.
    fn true_outline(contours: &[Vec<[f32; 2]>], x0: i32, y0: i32, w: u32, h: u32, radius: f32) -> Vec<f32> {
        let segs: Vec<([f32; 2], [f32; 2])> = contours
            .iter()
            .flat_map(|c| (0..c.len()).map(move |i| (c[i], c[(i + 1) % c.len()])))
            .collect();
        let inside = |x: f32, y: f32| {
            let mut wn = 0i32;
            for &(a, b) in &segs {
                if (a[1] <= y) != (b[1] <= y) {
                    let t = (y - a[1]) / (b[1] - a[1]);
                    if a[0] + t * (b[0] - a[0]) > x {
                        wn += if b[1] > a[1] { 1 } else { -1 };
                    }
                }
            }
            wn != 0
        };
        let near = |x: f32, y: f32| {
            segs.iter().any(|&(a, b)| {
                let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
                let l = dx * dx + dy * dy;
                let t = if l > 0.0 { (((x - a[0]) * dx + (y - a[1]) * dy) / l).clamp(0.0, 1.0) } else { 0.0 };
                let (px, py) = (a[0] + t * dx - x, a[1] + t * dy - y);
                px * px + py * py <= radius * radius
            })
        };
        let mut out = vec![0.0; (w * h) as usize];
        for py in 0..h {
            for px in 0..w {
                let mut n = 0;
                for sy in 0..8 {
                    for sx in 0..8 {
                        let x = (x0 + px as i32) as f32 + (sx as f32 + 0.5) / 8.0;
                        let y = (y0 + py as i32) as f32 + (sy as f32 + 0.5) / 8.0;
                        if inside(x, y) || near(x, y) {
                            n += 1;
                        }
                    }
                }
                out[(py * w + px) as usize] = n as f32 / 64.0;
            }
        }
        out
    }

    /// An outline is the glyph's true offset outline: against the coverage
    /// of every point inside the glyph or within the radius of it (64 a
    /// pixel), its edge pixels are 1–2% off on average — against 19–22%
    /// when it was grown from the antialiased mask, which can only guess
    /// where in an edge pixel the edge is. The reference itself agrees with
    /// the glyph rasterizer to within its own 1/64 sampling at radius 0.
    /// Skipped without SF Compact.
    #[test]
    fn outlines_follow_the_glyph_geometry() {
        let Some(mut font) = CaptionFont::try_sf_compact() else { return };
        for (size, radius) in [(34.0f32, 2.0f32), (16.0, 1.0), (24.0, 3.0)] {
            let (mut grown_sum, mut grown_n, mut grown_max) = (0.0f32, 0, 0.0f32);
            let (mut mask_sum, mut mask_n) = (0.0f32, 0);
            for c in "aeoRSg&".chars() {
                let contours: Vec<Vec<[f32; 2]>> = font
                    .outline(c, size)
                    .unwrap()
                    .into_iter()
                    .map(|curves| curves.into_iter().flatten().collect())
                    .collect();
                let edge = |truth: f32, v: u8| truth > 0.0 && truth < 1.0 || v > 0 && v < 255;

                let g = font.grown_glyph_at(c, size, 0, radius).unwrap().clone();
                let truth = true_outline(&contours, g.offset_x, g.offset_y, g.width, g.height, radius);
                for (t, &v) in truth.iter().zip(&g.coverage) {
                    let e = (v as f32 / 255.0 - t).abs();
                    grown_max = grown_max.max(e);
                    if edge(*t, v) {
                        grown_sum += e;
                        grown_n += 1;
                    }
                }

                // The mask grown, as a face without outlines still is.
                let g = font.glyph(c, size).clone();
                let pad = radius.ceil() as u32 + 2;
                let (w, h) = (g.width + 2 * pad, g.height + 2 * pad);
                let mut mask = vec![0u8; (w * h) as usize];
                for y in 0..g.height {
                    for x in 0..g.width {
                        mask[((y + pad) * w + x + pad) as usize] = g.coverage[(y * g.width + x) as usize];
                    }
                }
                let dilated = dilate(&mask, w, h, radius);
                let truth = true_outline(&contours, g.offset_x - pad as i32, g.offset_y - pad as i32, w, h, radius);
                for (t, &v) in truth.iter().zip(&dilated) {
                    if edge(*t, v) {
                        mask_sum += (v as f32 / 255.0 - t).abs();
                        mask_n += 1;
                    }
                }
            }
            let grown = grown_sum / grown_n as f32;
            let masked = mask_sum / mask_n as f32;
            assert!(grown < 0.025, "size {size} radius {radius}: mean edge error {grown}");
            assert!(grown_max < 0.25, "size {size} radius {radius}: worst pixel {grown_max}");
            assert!(grown * 5.0 < masked, "from the outline {grown}, from the mask {masked}");
        }
    }

    /// Full coverage is exactly the source colour, at every channel value.
    ///
    /// A round trip through linear must not shift a fully opaque pixel; if it
    /// does, every solid caption background drifts in colour.
    #[test]
    fn full_coverage_is_the_source_colour_exactly() {
        for value in [0u8, 1, 17, 64, 128, 200, 254, 255] {
            let mut frame = vec![9, 9, 9, 255u8];
            blend_pixel(&mut frame, 1, 0, 0, [value, value, value, 255], 255);
            assert_eq!(frame[0], value, "full coverage of {value} became {}", frame[0]);
        }
    }
}
