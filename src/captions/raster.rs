//! Compositing captions into RGBA pixels.
//!
//! [`CaptionPainter`] owns a face and a style and draws cue text into a
//! tightly-packed RGBA8 buffer — the same layout
//! [`HeadlessRenderer::render_to_rgba`](crate::HeadlessRenderer::render_to_rgba)
//! produces and [`export_video`](crate::video::export_video) consumes. That
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

    /// Draw a single line with its top-left corner at `(x, y)` in frame pixels.
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
        self.style.background[3] = 0;
        self.style.outline_width = 0.0;
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

        // Text coverage, in mask-local coordinates.
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
            self.draw_line_into_mask(
                &mut mask,
                mask_w,
                mask_h,
                line_left - mask_x as f32,
                baseline - mask_y as f32,
                &line.text,
            );
        }

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

        // 3. Outline — the mask grown by `outline_width`, drawn under the fill.
        if outline > 0.0 && style.outline_color[3] > 0 {
            let grown = dilate(&mask, mask_w, mask_h, outline);
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

    /// Rasterize one line's glyphs into `mask` at `pen_x` / `baseline_y`.
    fn draw_line_into_mask(
        &mut self,
        mask: &mut [u8],
        mask_w: u32,
        mask_h: u32,
        pen_x: f32,
        baseline_y: f32,
        text: &str,
    ) {
        let size = self.style.font_size;
        let spacing = self.style.letter_spacing;
        let mut pen = pen_x;
        for c in text.chars() {
            let glyph = self.font.glyph(c, size);
            let advance = glyph.advance;
            if !glyph.is_blank() {
                let gx = pen.round() as i32 + glyph.offset_x;
                let gy = baseline_y.round() as i32 + glyph.offset_y;
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
                        // Glyphs never overlap within a line, but rounding can
                        // put two masks in one pixel — keep the stronger.
                        *dst = (*dst).max(v);
                    }
                }
            }
            pen += advance + spacing;
        }
    }
}

/// Grow a coverage mask by `radius` pixels, approximating a disc by alternating
/// diamond (4-neighbour) and square (8-neighbour) max passes.
fn dilate(mask: &[u8], width: u32, height: u32, radius: f32) -> Vec<u8> {
    let steps = radius.round().max(0.0) as u32;
    let mut current = mask.to_vec();
    if steps == 0 || width == 0 || height == 0 {
        return current;
    }
    let mut next = vec![0u8; current.len()];
    for step in 0..steps.min(16) {
        let diagonal = step % 2 == 1;
        for y in 0..height as i32 {
            for x in 0..width as i32 {
                let mut best = current[(y as u32 * width + x as u32) as usize];
                for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    best = best.max(sample(&current, width, height, x + dx, y + dy));
                }
                if diagonal {
                    for (dx, dy) in [(-1, -1), (1, -1), (-1, 1), (1, 1)] {
                        best = best.max(sample(&current, width, height, x + dx, y + dy));
                    }
                }
                next[(y as u32 * width + x as u32) as usize] = best;
            }
        }
        std::mem::swap(&mut current, &mut next);
    }
    current
}

fn sample(mask: &[u8], width: u32, height: u32, x: i32, y: i32) -> u8 {
    if x < 0 || y < 0 || x >= width as i32 || y >= height as i32 {
        0
    } else {
        mask[(y as u32 * width + x as u32) as usize]
    }
}

/// Composite `color` through `mask` into an RGBA8 frame at `(ox, oy)`.
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
            let coverage = mask[(my as u32 * mask_w + mx as u32) as usize];
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
    let sa = alpha as f32 / 255.0;
    let da = frame[i + 3] as f32 / 255.0;
    let out_a = sa + da * (1.0 - sa);
    if out_a <= 0.0 {
        frame[i..i + 4].copy_from_slice(&[0, 0, 0, 0]);
        return;
    }
    for c in 0..3 {
        let src = color[c] as f32 / 255.0;
        let dst = frame[i + c] as f32 / 255.0;
        let out = (src * sa + dst * da * (1.0 - sa)) / out_a;
        frame[i + c] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
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
        // A single lit pixel in a 9x9 mask.
        let (w, h) = (9u32, 9u32);
        let mut mask = vec![0u8; (w * h) as usize];
        mask[(4 * w + 4) as usize] = 255;
        assert_eq!(
            dilate(&mask, w, h, 0.0).iter().filter(|&&v| v > 0).count(),
            1
        );
        // Radius 1 → a plus shape (5 px).
        assert_eq!(
            dilate(&mask, w, h, 1.0).iter().filter(|&&v| v > 0).count(),
            5
        );
        // Radius 2 adds the diagonal pass → a 5x5 octagon minus nothing = 21 px.
        let r2 = dilate(&mask, w, h, 2.0).iter().filter(|&&v| v > 0).count();
        assert!((13..=25).contains(&r2), "unexpected radius-2 spread: {r2}");
    }

    #[test]
    fn source_over_blending_matches_the_straight_alpha_formula() {
        // Half-transparent white over opaque black → mid gray, still opaque.
        let mut frame = vec![0, 0, 0, 255u8];
        blend_pixel(&mut frame, 1, 0, 0, [255, 255, 255, 255], 128);
        assert!((frame[0] as i32 - 128).abs() <= 1, "{:?}", frame);
        assert_eq!(frame[3], 255);

        // The same over a transparent buffer keeps the source color.
        let mut frame = vec![0, 0, 0, 0u8];
        blend_pixel(&mut frame, 1, 0, 0, [255, 0, 0, 255], 128);
        assert_eq!(&frame[..3], &[255, 0, 0]);
        assert_eq!(frame[3], 128);
    }
}
