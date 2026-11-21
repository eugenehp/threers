//! Box layout: CSS block flow and flexbox, for the properties in
//! [`Style`](super::Style).
//!
//! Follows the CSS Flexible Box algorithm in its common path: flex base sizes
//! from `flex-basis`, the main-size property or the content; free space
//! shared by `flex-grow`, or taken back by `flex-shrink` weighted by base
//! size; `min-*`/`max-*` clamps (an `auto` minimum is the content size, as for
//! flex items); `align-items: stretch` for auto cross sizes; `justify-content`
//! and `gap` on the main axis. Block children stack vertically at the
//! container's width (vertical margins do not collapse). Absolutely
//! positioned boxes are placed in their nearest positioned ancestor's padding
//! box. Text is one line (`white-space: nowrap`), or wraps (`normal`), and
//! breaks where its text does under `pre`, `pre-wrap` and `pre-line`.

use super::style::{
    AlignContent, AlignItems, BoxSizing, Display, FlexDirection, FlexWrap, GridPlacement,
    JustifyContent, Length, Position, Style, Track, TrackSize,
};
use super::tree::{LayoutCache, NodeId, NodeKind, Rect, Ui};

/// The same box to the bit: a cached layout is put back only for exactly the
/// box it was computed in, so what comes back is what computing it again
/// would give.
fn same_rect(a: Rect, b: Rect) -> bool {
    [a.x, a.y, a.width, a.height]
        .iter()
        .zip([b.x, b.y, b.width, b.height])
        .all(|(p, q)| p.to_bits() == q.to_bits())
}

/// Measures a line of text: its advance width and its line box height
/// (`line-height: normal` when `line_height` is `None`).
pub trait TextMeasure {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32);
}

/// Every character `w` pixels wide, lines `1.2 ×` the font size: for tests
/// and layouts that do not depend on text.
pub struct FixedWidthText(pub f32);

impl TextMeasure for FixedWidthText {
    fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
        (
            text.chars().count() as f32 * self.0,
            style.line_height.unwrap_or(style.font_size * 1.2),
        )
    }
}

/// The non-content part of a box along one axis: padding plus border.
fn frame(style: &Style, horizontal: bool) -> f32 {
    if horizontal {
        style.padding.horizontal() + style.border.horizontal()
    } else {
        style.padding.vertical() + style.border.vertical()
    }
}

/// A `width`/`height`-like length as a border-box size.
fn border_box(style: &Style, len: Length, base: Option<f32>, horizontal: bool) -> Option<f32> {
    let v = len.resolve(base)?;
    Some(match style.box_sizing {
        BoxSizing::ContentBox => v + frame(style, horizontal),
        BoxSizing::BorderBox => v.max(frame(style, horizontal)),
    })
}

/// Wrapped text kept from layout to layout: two generations of up to
/// `KEEP` (when the newer fills, the older goes, so what is still in use
/// stays — a cache cleared whole at its limit starts over when busiest).
#[derive(Debug, Default, Clone)]
pub(crate) struct WrapMemo {
    new: std::collections::HashMap<u64, Vec<String>>,
    old: std::collections::HashMap<u64, Vec<String>>,
}

impl WrapMemo {
    const KEEP: usize = 4096;

    fn get(&mut self, key: u64) -> Option<Vec<String>> {
        if let Some(l) = self.new.get(&key) {
            return Some(l.clone());
        }
        let l = self.old.remove(&key)?;
        self.new.insert(key, l.clone());
        Some(l)
    }

    fn put(&mut self, key: u64, lines: Vec<String>) {
        if self.new.len() >= Self::KEEP {
            self.old = std::mem::take(&mut self.new);
        }
        self.new.insert(key, lines);
    }
}

/// What a wrap depends on: the text, what its measure reads of the style,
/// the width, its natural width and the layout units. FNV-1a.
fn wrap_key(text: &str, style: &Style, width: f32, natural: f32, units: bool) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    eat(text.as_bytes());
    eat(&[0xff]);
    eat(style.font_family.as_bytes());
    eat(&style.font_size.to_bits().to_le_bytes());
    eat(&crate::paint::TextStyle::of(style).weight().to_le_bytes());
    eat(&[style.italic as u8, style.vertical as u8, units as u8]);
    eat(&style.line_height.unwrap_or(-1.0).to_bits().to_le_bytes());
    eat(&width.to_bits().to_le_bytes());
    eat(&natural.to_bits().to_le_bytes());
    h
}

struct Ctx<'a> {
    ui: &'a mut Ui,
    text: &'a mut dyn TextMeasure,
    /// How many [`Ui::cache_layout`] boundaries the work in hand is inside:
    /// there, a measurement outlives its pass.
    stable: u32,
}

/// A length as a Blink `LayoutUnit` (`LayoutUnit(float)`): on the 1/64 px
/// grid, truncated toward zero.
pub fn layout_unit(v: f32) -> f32 {
    (v * 64.0).trunc() / 64.0
}

/// A length within float error (1/512 px) of the 1/64 px grid, put on it;
/// any other left as it is. Blink's lengths are exact `LayoutUnit`s, so one
/// a hair off the grid here is arithmetic noise (976.62463 for 976.625);
/// one well off it comes of layout this does not do Blink's way, and
/// moving it would not make it right.
fn onto_grid(v: f32) -> f32 {
    let g = (v * 64.0).round() / 64.0;
    if (v - g).abs() < 1.0 / 512.0 {
        g
    } else {
        v
    }
}

pub(crate) fn layout(ui: &mut Ui, width: f32, height: f32, text: &mut dyn TextMeasure) {
    let root = ui.root();
    ui.pass += 1;
    let mut ctx = Ctx {
        ui,
        text,
        stable: 0,
    };

    ctx.place(
        root,
        Rect {
            x: 0.0,
            y: 0.0,
            width,
            height,
        },
    );
    ctx.measure_drawn_lines();
}

/// `id`'s min-content width (see `Ctx::min_content_width`), measured
/// outside a layout.
pub(crate) fn min_content_width_of(ui: &mut Ui, id: NodeId, text: &mut dyn TextMeasure) -> f32 {
    let mut ctx = Ctx {
        ui,
        text,
        stable: 0,
    };
    ctx.min_content_width(id)
}

/// A flex item with its resolved main size.
struct FlexItem {
    id: NodeId,
    base: f32,
    min: f32,
    max: f32,
    main: f32,
    grow: f32,
    shrink: f32,
    margin_main: (f32, f32),
    margin_cross: (f32, f32),
}

impl Ctx<'_> {
    /// From the top of `id`'s border box to its first line of text's
    /// baseline: the first text in it, down through its first in-flow
    /// children. `None` without text (aligned by its bottom edge then).
    fn first_baseline(&self, id: NodeId) -> Option<f32> {
        let st = self.style(id);
        if let NodeKind::Text(_) = &self.ui.nodes[id.0].kind {
            let lh = st.line_height.unwrap_or(st.font_size * 1.2);
            // Half the leading, then the ascent (about 0.8 em in the usual faces).
            return Some((lh - st.font_size) / 2.0 + st.font_size * 0.8);
        }
        let inset = st.border.top + st.padding.top;
        self.ui.nodes[id.0]
            .children
            .iter()
            .copied()
            .filter(|&c| {
                let cs = self.style(c);
                cs.display != Display::None && cs.position != Position::Absolute
            })
            .find_map(|c| {
                self.first_baseline(c)
                    .map(|b| inset + self.style(c).margin.top + b)
            })
    }

    fn style(&self, id: NodeId) -> &Style {
        &self.ui.nodes[id.0].style
    }

    /// Whether `id` is a replaced element (an image): sized by its content
    /// rather than stretched in block flow.
    fn replaced(&self, id: NodeId) -> bool {
        matches!(
            self.ui.nodes[id.0].kind,
            NodeKind::Image { .. } | NodeKind::Surface { .. }
        )
    }

    fn in_flow(&self, id: NodeId) -> Vec<NodeId> {
        self.ui.nodes[id.0]
            .children
            .iter()
            .copied()
            .filter(|&c| {
                let s = &self.ui.nodes[c.0].style;
                s.display != Display::None && s.position != Position::Absolute
            })
            .collect()
    }

    /// The border-box size `id` takes by its content (max-content), given
    /// a definite border-box width if there is one.
    ///
    /// Remembered per node and arguments: flex sizes an item, then sizes its
    /// items, so without it a nested row is measured once per level above
    /// it. The answer depends only on the subtree, which marks itself when
    /// it changes, so it stands for the rest of the pass — and across passes
    /// inside a [`Ui::cache_layout`] boundary.
    fn content_size(
        &mut self,
        id: NodeId,
        width: Option<f32>,
        height_base: Option<f32>,
    ) -> (f32, f32) {
        let boundary = self.ui.nodes[id.0].cache_boundary;
        self.stable += boundary as u32;
        let size = self.content_size_remembered(id, width, height_base);
        self.stable -= boundary as u32;
        size
    }

    fn content_size_remembered(
        &mut self,
        id: NodeId,
        width: Option<f32>,
        height_base: Option<f32>,
    ) -> (f32, f32) {
        let key = (width.map(f32::to_bits), height_base.map(f32::to_bits));
        let pass = self.ui.pass;
        let node = &mut self.ui.nodes[id.0];
        if node.measured_pass != pass && self.stable == 0 {
            node.measured.clear();
        }
        node.measured_pass = pass;
        if let Some(&(_, _, size)) = node.measured.iter().find(|m| (m.0, m.1) == key) {
            return size;
        }
        let size = self.measure_content(id, width, height_base);
        self.ui.nodes[id.0].measured.push((key.0, key.1, size));
        size
    }

    fn measure_content(
        &mut self,
        id: NodeId,
        width: Option<f32>,
        height_base: Option<f32>,
    ) -> (f32, f32) {
        let style = self.style(id).clone();
        let (fx, fy) = (frame(&style, true), frame(&style, false));
        if let NodeKind::Text(t) = &self.ui.nodes[id.0].kind {
            let t = t.clone();
            // Blink keeps lengths in 1/64 px (`LayoutUnit`) and rounds a
            // run's width up to one.
            let h = self.text.measure(&t, &style).1;
            // Its own line breaks: as tall as its lines, as wide as the
            // widest (each wrapped too, under `pre-wrap`).
            if let Some(lines) = self.kept_lines(id.0, &t, &style, width.map(|w| (w - fx).max(0.0))) {
                let shown = style.line_clamp.map_or(lines.len(), |n| lines.len().min(n as usize));
                let widest = lines.iter().map(|l| self.text.measure(l.trim_end(), &style).0).fold(0.0, f32::max);
                let widest = (widest * 64.0).ceil() / 64.0;
                return (width.unwrap_or(widest + fx), shown.max(1) as f32 * h + fy);
            }
            let w = self.run_width(id.0, 0, &t, &style);
            let w = (w * 64.0).ceil() / 64.0;
            if let (true, Some(width)) = (style.wrap, width) {
                let lines = self.wrap(id.0, &t, &style, (width - fx).max(0.0));
                let shown = style
                    .line_clamp
                    .map_or(lines.len(), |n| lines.len().min(n as usize));
                return (width, shown.max(1) as f32 * h + fy);
            }
            if style.vertical {
                return (h + fx, w + fy);
            }
            return (width.unwrap_or(w + fx), h + fy);
        }
        let natural = match &self.ui.nodes[id.0].kind {
            NodeKind::Image { image } => Some((image.width as f32, image.height as f32)),
            NodeKind::Surface { width, height, .. } => Some((*width as f32, *height as f32)),
            _ => None,
        };
        if let Some((iw, ih)) = natural {
            // A replaced element: its own size, or the other side by its
            // aspect ratio when one side is set, as `<img>` sizes.
            let inner_w = width.map(|w| (w - fx).max(0.0));
            let inner_h =
                border_box(&style, style.height, height_base, false).map(|h| (h - fy).max(0.0));
            let (cw, ch) = match (inner_w, inner_h) {
                (Some(w), Some(h)) => (w, h),
                (Some(w), None) => (w, w * ih / iw),
                (None, Some(h)) => (h * iw / ih, h),
                (None, None) => (iw, ih),
            };
            return (width.unwrap_or(cw + fx), ch + fy);
        }
        let inner_w = width.map(|w| (w - fx).max(0.0));
        let children = self.in_flow(id);
        let (cw, ch) = match style.display {
            // A row with a definite width: its items' heights at the widths
            // flexing gives them (wrapping text grows its line), as CSS
            // sizes the container's cross axis after its main sizes.
            // Wrapping: its lines' heights, one under another.
            Display::Flex
                if style.flex_direction == FlexDirection::Row
                    && inner_w.is_some()
                    && style.flex_wrap != FlexWrap::NoWrap =>
            {
                let inner = inner_w.unwrap();
                let cross_gap = Self::flex_gaps(&style).1;
                let lines = self.flex_lines(id, &style, inner, height_base.unwrap_or(0.0));
                let mut total = 0.0f32;
                for (k, l) in lines.iter().enumerate() {
                    total += self.flex_line_cross(&style, l, height_base.unwrap_or(0.0))
                        + if k > 0 { cross_gap } else { 0.0 };
                }
                (inner, total)
            }
            Display::Flex if style.flex_direction == FlexDirection::Row && inner_w.is_some() => {
                let inner = inner_w.unwrap();
                let items = self.flex_items(id, &style, inner, height_base.unwrap_or(0.0));
                let mut cross = 0.0f32;
                for item in &items {
                    let (_, h) = self.outer_size(item.id, Some(item.main), height_base);
                    cross = cross.max(h + item.margin_cross.0 + item.margin_cross.1);
                }
                (inner, cross)
            }
            Display::Grid => {
                let plan = self.grid_plan(id, &style, inner_w, None);
                (plan.width(), plan.height())
            }
            Display::Flex => {
                let row = style.flex_direction == FlexDirection::Row;
                let mut main = 0.0f32;
                let mut cross = 0.0f32;
                for (i, &c) in children.iter().enumerate() {
                    let cs = self.style(c).clone();
                    // In a column, each at its own width (a card of 520 px
                    // in a wide column is as tall as it is at 520).
                    let (w, h) = self.outer_size(
                        c,
                        if row {
                            None
                        } else {
                            inner_w.map(|iw| own_width(&cs, iw))
                        },
                        height_base,
                    );
                    let (m_main, m_cross) = if row {
                        (cs.margin.horizontal(), cs.margin.vertical())
                    } else {
                        (cs.margin.vertical(), cs.margin.horizontal())
                    };
                    let (a, b) = if row { (w, h) } else { (h, w) };
                    // `column-gap` in a row, `row-gap` in a column, as
                    // flexing spaces them (the shorthand alone left a
                    // row's words narrower than their gaps).
                    main += a
                        + m_main
                        + if i > 0 {
                            Self::flex_gaps(&style).0
                        } else {
                            0.0
                        };
                    cross = cross.max(b + m_cross);
                }
                if row {
                    (main, cross)
                } else {
                    (cross, main)
                }
            }
            _ => {
                let mut w = 0.0f32;
                let mut h = 0.0f32;
                for &c in &children {
                    let cs = self.style(c).clone();
                    let cwidth = match (cs.width.resolve(inner_w), inner_w) {
                        (Some(_), _) => border_box(&cs, cs.width, inner_w, true),
                        (None, _) if self.replaced(c) => None,
                        (None, Some(iw)) => Some((iw - cs.margin.horizontal()).max(0.0)),
                        (None, None) => None,
                    };
                    let (ow, oh) = self.outer_size(c, cwidth, height_base);
                    w = w.max(ow + cs.margin.horizontal());
                    h += oh + cs.margin.vertical();
                }
                (w, h)
            }
        };
        (width.unwrap_or(cw + fx), ch + fy)
    }

    /// The narrowest border-box width `id` can take without its content
    /// spilling out (CSS's min-content width): text that wraps as wide as
    /// its longest word, other text whole; a row as its items side by side
    /// at theirs; a column or block as its widest child.
    fn min_content_width(&mut self, id: NodeId) -> f32 {
        let style = self.style(id).clone();
        let fx = frame(&style, true);
        let own = if let Some(w) = border_box(&style, style.width, None, true) {
            w
        } else if let NodeKind::Text(t) = &self.ui.nodes[id.0].kind {
            let t = t.clone();
            if style.vertical {
                self.text.measure(&t, &style).1 + fx
            } else if let Some(lines) = self.kept_lines(id.0, &t, &style, style.wrap.then_some(0.0)) {
                // Its widest line, or under `pre-wrap` its widest word.
                let widest = lines.iter().map(|l| self.text.measure(l.trim_end(), &style).0).fold(0.0, f32::max);
                (widest * 64.0).ceil() / 64.0 + fx
            } else if style.wrap {
                let mut widest = 0.0f32;
                let mut at = 0;
                for line in self.wrap(id.0, &t, &style, 0.0) {
                    let offset = t
                        .get(at..)
                        .and_then(|rest| rest.find(line.as_str()))
                        .map_or(at, |i| at + i);
                    widest = widest.max(self.run_width(id.0, offset, line.trim_end(), &style));
                    at = offset + line.len();
                }
                (widest * 64.0).ceil() / 64.0 + fx
            } else {
                self.content_size(id, None, None).0
            }
        } else if self.replaced(id) {
            self.content_size(id, None, None).0
        } else {
            let children = self.in_flow(id);
            // A row that wraps can be as narrow as its widest item.
            let row = style.display == Display::Flex
                && style.flex_direction == FlexDirection::Row
                && style.flex_wrap == FlexWrap::NoWrap;
            let mut w = 0.0f32;
            for (i, &c) in children.iter().enumerate() {
                let m = self.min_content_width(c) + self.style(c).margin.horizontal();
                if row {
                    w += m + if i > 0 {
                        Self::flex_gaps(&style).0
                    } else {
                        0.0
                    };
                } else {
                    w = w.max(m);
                }
            }
            if style.display == Display::Grid {
                // Tracks are not sized down here: as wide as its content.
                w = self.content_size(id, None, None).0 - fx;
            }
            w + fx
        };
        self.clamp(&style, own, true, None)
    }

    /// A box's border-box size from its own `width`/`height` where given,
    /// else its content, clamped by `min-*`/`max-*`.
    fn outer_size(
        &mut self,
        id: NodeId,
        width: Option<f32>,
        height_base: Option<f32>,
    ) -> (f32, f32) {
        let style = self.style(id).clone();
        let w = width.or_else(|| border_box(&style, style.width, None, true));
        let h = border_box(&style, style.height, height_base, false);
        let (cw, ch) = if w.is_none() || h.is_none() {
            self.content_size(id, w, height_base)
        } else {
            (0.0, 0.0)
        };
        let w = self.clamp(&style, w.unwrap_or(cw), true, None);
        let h = self.clamp(&style, h.unwrap_or(ch), false, height_base);
        (w, h)
    }

    fn clamp(&self, style: &Style, v: f32, horizontal: bool, base: Option<f32>) -> f32 {
        let (min, max) = if horizontal {
            (style.min_width, style.max_width)
        } else {
            (style.min_height, style.max_height)
        };
        let mut v = v;
        if let Some(m) = border_box(style, max, base, horizontal) {
            v = v.min(m);
        }
        if let Some(m) = border_box(style, min, base, horizontal) {
            v = v.max(m);
        }
        v.max(frame(style, horizontal))
    }

    /// The lines of a text that keeps its line breaks
    /// ([`keep_newlines`](Style::keep_newlines)), each wrapped to `width`
    /// as well when it wraps and a width is given. `None` for a text that
    /// does not keep them, or has none to keep: laid out as before.
    fn kept_lines(&mut self, node: usize, text: &str, style: &Style, width: Option<f32>) -> Option<Vec<String>> {
        if !style.keep_newlines || !text.contains('\n') {
            return None;
        }
        let mut lines = vec![];
        for hard in text.split('\n') {
            match (style.wrap, width) {
                (true, Some(w)) => lines.extend(self.wrap(node, hard, style, w)),
                _ => lines.push(hard.to_string()),
            }
        }
        Some(lines)
    }

    /// `text` broken into lines no wider than `width`: at spaces and
    /// zero-width spaces where they fit, else within a word. Kept across
    /// layouts ([`WrapMemo`]) for text in one style: a paragraph wraps by
    /// measuring each line with all the text before it, a string of its own
    /// each time, so a sidebar of them measured tens of thousands of strings
    /// a layout — every key typed into a search box, every frame it scrolled.
    fn wrap(&mut self, node: usize, text: &str, style: &Style, width: f32) -> Vec<String> {
        if self.ui.nodes[node].runs.is_some() {
            return self.wrap_now(node, text, style, width);
        }
        // its own width keys it too: another measurer (another scale, a
        // fixed-width one) breaks it elsewhere
        let natural = self.text.measure(text, style).0;
        let key = wrap_key(text, style, width, natural, self.ui.layout_units);
        if let Some(lines) = self.ui.wrap_memo.get(key) {
            return lines;
        }
        let lines = self.wrap_now(node, text, style, width);
        self.ui.wrap_memo.put(key, lines.clone());
        lines
    }

    /// [`wrap`](Self::wrap), done now.
    fn wrap_now(&mut self, node: usize, text: &str, style: &Style, width: f32) -> Vec<String> {
        // Pieces: each ends just after a break opportunity.
        let mut pieces: Vec<String> = vec![];
        let mut cur = String::new();
        for c in text.chars() {
            cur.push(c);
            if c == ' ' || c == '\u{200b}' {
                pieces.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            pieces.push(cur);
        }
        // A line's width: with layout units, as Blink takes it, between its
        // ends' offsets in the whole text (`before` is the text on earlier
        // lines), each rounded up to 1/64 px; "segments" after "Select " is
        // 57.78125 so, alone 57.796875.
        let units = self.ui.layout_units;
        let styled = self.ui.nodes[node].runs.is_some();
        let fits = |this: &mut Self, before: &str, s: &str| {
            let s = s.trim_end_matches([' ', '\u{200b}']);
            let w = if styled {
                // Each run in its own style, from where the line starts.
                this.run_width(node, before.len(), s, style)
            } else if units && !before.is_empty() {
                let up = |v: f32| (v * 64.0).ceil() / 64.0;
                up(this.text.measure(&format!("{before}{s}"), style).0)
                    - up(this.text.measure(before, style).0)
            } else {
                this.text.measure(s, style).0
            };
            w <= width + 1e-3
        };
        let mut lines: Vec<String> = vec![];
        let mut before = String::new();
        let mut line = String::new();
        // A line's width estimated from its pieces', each measured alone
        // (the same words recur, and a cache keeps them): measuring every
        // longer prefix of a paragraph is quadratic in its length. Only an
        // estimate within `margin` of the width (where kerning across a
        // space could tip it, a tenth of an em per space) is measured whole,
        // so lines break where they would if every one were.
        let mut line_est = 0.0;
        let mut line_pieces = 0;
        for piece in pieces {
            let candidate = format!("{line}{piece}");
            let margin = 1.0 + 0.1 * style.font_size.max(1.0) * (line_pieces + 1) as f32;
            let est = (!styled).then(|| {
                line_est
                    + self
                        .text
                        .measure(piece.trim_end_matches([' ', '\u{200b}']), style)
                        .0
            });
            let fit = match est {
                Some(e) if e <= width - margin => true,
                Some(e) if e > width + margin => false,
                _ => fits(self, &before, &candidate),
            };
            if fit {
                line = candidate;
                if !styled {
                    line_est += self.text.measure(&piece, style).0;
                }
                line_pieces += 1;
                continue;
            }
            if !line.is_empty() {
                before.push_str(&line);
                lines.push(std::mem::take(&mut line));
            }
            // `break-word`: a piece too long for a line of its own breaks
            // where it must.
            let mut rest = piece;
            while !fits(self, &before, &rest) {
                let chars: Vec<char> = rest.chars().collect();
                let mut n = chars.len();
                while n > 1 && !fits(self, &before, &chars[..n].iter().collect::<String>()) {
                    n -= 1;
                }
                let head: String = chars[..n].iter().collect();
                before.push_str(&head);
                lines.push(head);
                rest = chars[n..].iter().collect();
            }
            line_est = if styled || rest.is_empty() {
                0.0
            } else {
                self.text.measure(&rest, style).0
            };
            line_pieces = usize::from(!rest.is_empty());
            line = rest;
        }
        if !line.is_empty() || lines.is_empty() {
            lines.push(line);
        }
        lines
    }

    /// `line-clamp`: at most that many of `lines`, the last ending in `…`
    /// (fitted to `width`) when lines were cut.
    fn clamp_lines(&mut self, style: &Style, mut lines: Vec<String>, width: f32) -> Vec<String> {
        let Some(n) = style.line_clamp.map(|n| n as usize) else {
            return lines;
        };
        if lines.len() <= n {
            return lines;
        }
        lines.truncate(n);
        let last = lines.pop().unwrap_or_default();
        let mut chars: Vec<char> = last.trim_end().chars().collect();
        loop {
            let candidate: String = chars.iter().chain(std::iter::once(&'\u{2026}')).collect();
            if chars.is_empty() || self.text.measure(&candidate, style).0 <= width {
                lines.push(candidate);
                break;
            }
            chars.pop();
        }
        lines
    }

    /// `text-overflow: ellipsis`: `id`'s text children wider than `width`
    /// keep what fits before `…` (or the `…` alone), as one line.
    fn ellipsize(&mut self, id: NodeId, width: f32) {
        for c in self.ui.nodes[id.0].children.clone() {
            let NodeKind::Text(t) = &self.ui.nodes[c.0].kind else {
                continue;
            };
            let t = t.clone();
            let style = self.style(c).clone();
            let measure = |ctx: &mut Self, s: &str| ctx.text.measure(s, &style).0;
            if measure(self, &t) <= width {
                continue;
            }
            let chars: Vec<char> = t.chars().collect();
            let mut n = chars.len();
            let mut line = String::from("\u{2026}");
            while n > 0 {
                let candidate: String = chars[..n]
                    .iter()
                    .chain(std::iter::once(&'\u{2026}'))
                    .collect();
                if measure(self, &candidate) <= width {
                    line = candidate;
                    break;
                }
                n -= 1;
            }
            self.ui.nodes[c.0].lines = vec![line];
            let r = &mut self.ui.nodes[c.0].rect;
            r.width = r.width.min(width.max(0.0));
        }
    }

    /// The width of `s`, which is the node's text from byte `offset` on, with
    /// each of the node's runs in its own style (all in `style` when it has
    /// none). What of `s` does not match the text (an ellipsis) is measured
    /// in `style`.
    fn run_width(&mut self, node: usize, offset: usize, s: &str, style: &Style) -> f32 {
        let n = &self.ui.nodes[node];
        let (Some(runs), NodeKind::Text(text)) = (&n.runs, &n.kind) else {
            return self.text.measure(s, style).0;
        };
        let rest = text.get(offset..).unwrap_or("");
        let same = rest
            .bytes()
            .zip(s.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        let mut same = same;
        while !s.is_char_boundary(same) {
            same -= 1;
        }
        let (text, runs) = (text.clone(), runs.clone());
        let mut w = 0.0;
        for (a, b, k) in crate::rich_text::split(&text, &runs, offset, offset + same) {
            let st = runs
                .get(k as usize)
                .map_or_else(|| style.clone(), |r| r.style(style));
            w += self.text.measure(&text[a..b], &st).0;
        }
        if same < s.len() {
            w += self.text.measure(&s[same..], style).0;
        }
        w
    }

    /// The width of each drawn line of text that is underlined, has styled
    /// runs or is aligned other than at the start, now that the lines are
    /// final (after ellipsis).
    fn measure_drawn_lines(&mut self) {
        for i in 0..self.ui.nodes.len() {
            let n = &self.ui.nodes[i];
            let NodeKind::Text(t) = &n.kind else { continue };
            let aligned = n.style.text_align != crate::style::TextAlign::Start;
            if n.removed || !(n.style.underline || n.runs.is_some() || aligned) {
                if !n.line_widths.is_empty() || !n.pieces.is_empty() {
                    self.ui.nodes[i].line_widths.clear();
                    self.ui.nodes[i].pieces.clear();
                }
                continue;
            }
            let style = n.style.clone();
            let lines: Vec<String> = if n.lines.is_empty() {
                vec![t.clone()]
            } else {
                n.lines
                    .iter()
                    .map(|l| l.trim_end_matches([' ', '\u{200b}']).to_string())
                    .collect()
            };
            let mut widths = Vec::with_capacity(lines.len());
            let mut offset = 0;
            for line in &lines {
                widths.push(self.run_width(i, offset, line, &style));
                offset += line.len();
                // Wrapped lines lose the break they ended at.
                if let NodeKind::Text(t) = &self.ui.nodes[i].kind {
                    while t[offset.min(t.len())..].starts_with([' ', '\u{200b}', '\n']) {
                        offset += t[offset..].chars().next().map_or(1, char::len_utf8);
                    }
                }
            }
            self.ui.nodes[i].line_widths = widths;
            // Where each run's piece of each line goes.
            let mut pieces = vec![];
            if let (Some(runs), NodeKind::Text(text)) =
                (self.ui.nodes[i].runs.clone(), self.ui.nodes[i].kind.clone())
            {
                let mut offset = 0;
                for (li, line) in lines.iter().enumerate() {
                    if text
                        .get(offset..)
                        .is_some_and(|r| r.starts_with(line.as_str()))
                    {
                        let mut x = 0.0;
                        for (a, b, k) in
                            crate::rich_text::split(&text, &runs, offset, offset + line.len())
                        {
                            let st = runs
                                .get(k as usize)
                                .map_or_else(|| style.clone(), |r| r.style(&style));
                            let w = self.text.measure(&text[a..b], &st).0;
                            pieces.push(crate::rich_text::Piece {
                                line: li as u32,
                                x,
                                width: w,
                                start: a as u32,
                                end: b as u32,
                                run: k,
                            });
                            x += w;
                        }
                    }
                    offset += line.len();
                    while text[offset.min(text.len())..].starts_with([' ', '\u{200b}']) {
                        offset += text[offset..].chars().next().map_or(1, char::len_utf8);
                    }
                }
            }
            self.ui.nodes[i].pieces = pieces;
        }
    }

    /// Gives `id` its border box and lays out its children inside it — or,
    /// for a [`Ui::cache_layout`] boundary given the same box with nothing
    /// inside changed, puts back what that produced last time.
    fn place(&mut self, id: NodeId, rect: Rect) {
        let node = &self.ui.nodes[id.0];
        if !node.cache_boundary {
            self.place_now(id, rect);
            return;
        }
        let hit = !node.dirty
            && node
                .cache
                .as_ref()
                .is_some_and(|c| c.layout_units == self.ui.layout_units && same_rect(c.rect, rect));
        if hit {
            let cache = self.ui.nodes[id.0].cache.take().expect("checked");
            for (i, r, lines, scroll_height, scrolled) in &cache.nodes {
                let n = &mut self.ui.nodes[*i];
                n.rect = *r;
                if n.lines != *lines {
                    n.lines.clone_from(lines);
                }
                n.scroll_height = *scroll_height;
                n.scrolled = *scrolled;
            }
            self.ui.nodes[id.0].cache = Some(cache);
            return;
        }
        self.stable += 1;
        self.place_now(id, rect);
        self.stable -= 1;
        let mut nodes = vec![];
        let mut stack = vec![id.0];
        while let Some(i) = stack.pop() {
            let n = &self.ui.nodes[i];
            nodes.push((i, n.rect, n.lines.clone(), n.scroll_height, n.scrolled));
            stack.extend(n.children.iter().map(|c| c.0));
        }
        self.ui.nodes[id.0].cache = Some(Box::new(LayoutCache {
            rect,
            layout_units: self.ui.layout_units,
            nodes,
        }));
    }

    fn place_now(&mut self, id: NodeId, rect: Rect) {
        self.ui.nodes[id.0].dirty = false;
        self.ui.nodes[id.0].placed_pass = self.ui.pass;
        let rect = if self.ui.layout_units {
            Rect {
                x: onto_grid(rect.x),
                y: onto_grid(rect.y),
                width: onto_grid(rect.width),
                height: onto_grid(rect.height),
            }
        } else {
            rect
        };
        self.ui.nodes[id.0].rect = rect;
        let style = self.style(id).clone();
        // Its own line breaks, each line wrapped too under `pre-wrap`.
        let kept = match &self.ui.nodes[id.0].kind {
            NodeKind::Text(t) => {
                let t = t.clone();
                let width = (rect.width - frame(&style, true)).max(0.0);
                self.kept_lines(id.0, &t, &style, Some(width)).map(|lines| self.clamp_lines(&style, lines, width))
            }
            _ => None,
        };
        if let Some(lines) = kept {
            self.ui.nodes[id.0].lines = lines;
        } else {
            if let (NodeKind::Text(t), true) = (&self.ui.nodes[id.0].kind, style.wrap) {
                let t = t.clone();
                let width = (rect.width - frame(&style, true)).max(0.0);
                let lines = self.wrap(id.0, &t, &style, width);
                let lines = self.clamp_lines(&style, lines, width);
                self.ui.nodes[id.0].lines = lines;
            }
            // A run that does not wrap: one line, unless its box ellipsizes
            // it (after this, from the box).
            if let (NodeKind::Text(_), false) = (&self.ui.nodes[id.0].kind, style.wrap) {
                self.ui.nodes[id.0].lines.clear();
            }
        }
        let content = Rect {
            x: rect.x + style.border.left + style.padding.left,
            y: rect.y + style.border.top + style.padding.top,
            width: (rect.width - frame(&style, true)).max(0.0),
            height: (rect.height - frame(&style, false)).max(0.0),
        };
        match style.display {
            Display::Flex => self.flex(id, &style, content),
            Display::Grid => self.grid(id, &style, content),
            _ => {
                self.block(id, content);
                self.centre_button(id, content);
            }
        }
        self.absolute(id, rect, &style);
        // `display: none` generates no box: an empty one at the origin, as
        // a page reports for it, rather than wherever it last was.
        for i in 0..self.ui.nodes[id.0].children.len() {
            let c = self.ui.nodes[id.0].children[i];
            if self.style(c).display == Display::None {
                self.hide(c);
            }
        }
        if style.ellipsis && !style.wrap {
            self.ellipsize(id, content.width);
        }
        if style.scroll_x {
            // Sideways as below: the content's extent right of the padding
            // box's left, the offset clamped to it, everything moved left.
            let left = rect.x + style.border.left;
            let children = self.ui.nodes[id.0].children.clone();
            let right = children
                .iter()
                .filter(|c| self.style(**c).display != Display::None)
                .map(|c| {
                    let r = self.ui.nodes[c.0].rect;
                    r.x + r.width + self.style(*c).margin.right
                })
                .fold(left, f32::max)
                + style.padding.right;
            let width = right - left;
            let visible = (rect.width - style.border.horizontal()).max(0.0);
            let ratio = self.ui.pixel_ratio;
            let end = (width - visible).max(0.0);
            let scrolled = style.scroll_left.clamp(0.0, end);
            let scrolled = if scrolled < end {
                ((scrolled * ratio).round() / ratio).min(end)
            } else {
                end
            };
            self.ui.nodes[id.0].scroll_width = width;
            self.ui.nodes[id.0].scrolled_x = scrolled;
            if scrolled != 0.0 {
                for c in children {
                    self.translate_x(c, -scrolled);
                }
            }
        }
        if style.scroll {
            // The content's extent below the padding box's top, and the
            // scroll offset clamped to it; then everything inside moves up.
            let top = rect.y + style.border.top;
            let children = self.ui.nodes[id.0].children.clone();
            let bottom = children
                .iter()
                .filter(|c| self.style(**c).display != Display::None)
                .map(|c| {
                    let r = self.ui.nodes[c.0].rect;
                    r.y + r.height + self.style(*c).margin.bottom
                })
                .fold(top, f32::max)
                + style.padding.bottom;
            let height = bottom - top;
            let visible = (rect.height - style.border.vertical()).max(0.0);
            let end = (height - visible).max(0.0);
            let scrolled = style.scroll_top.clamp(0.0, end);
            // On whole device pixels: content between them rounds its text
            // and its boxes differently, and shimmers as it moves.
            // Not at the end, though: there the end is the place, even
            // between pixels, or a page scrolled to its end stops short.
            let ratio = self.ui.pixel_ratio;
            let scrolled = if scrolled < end {
                ((scrolled * ratio).round() / ratio).min(end)
            } else {
                end
            };
            self.ui.nodes[id.0].scroll_height = height;
            self.ui.nodes[id.0].scrolled = scrolled;
            if scrolled != 0.0 {
                for c in children {
                    self.translate(c, -scrolled);
                }
            }
        }
    }

    fn hide(&mut self, id: NodeId) {
        let mut stack = vec![id.0];
        while let Some(i) = stack.pop() {
            let n = &mut self.ui.nodes[i];
            n.rect = Rect::default();
            n.lines.clear();
            stack.extend(n.children.iter().map(|c| c.0));
        }
    }

    /// Moves `id` and everything inside it down by `dy`.
    fn translate_x(&mut self, id: NodeId, dx: f32) {
        if self.style(id).display == Display::None {
            return;
        }
        self.ui.nodes[id.0].rect.x += dx;
        for c in self.ui.nodes[id.0].children.clone() {
            self.translate_x(c, dx);
        }
    }

    /// Whether `id` is a button (its role): its content centred up and
    /// down, as a page's `<button>` centres it, when it is taller than its
    /// content — stretched by its row, or given a height.
    fn is_button(&self, id: NodeId) -> bool {
        use crate::Role;
        self.ui.semantics(id).is_some_and(|s| {
            matches!(
                s.role,
                Role::Button | Role::Tab | Role::Checkbox | Role::Radio | Role::Option
            )
        })
    }

    /// A block button's content moved down by half the room under it.
    fn centre_button(&mut self, id: NodeId, content: Rect) {
        if !self.is_button(id) || self.style(id).scroll {
            return;
        }
        let kids = self.in_flow(id);
        let Some(bottom) = kids
            .iter()
            .map(|c| {
                let r = self.ui.nodes[c.0].rect;
                r.y + r.height + self.style(*c).margin.bottom
            })
            .reduce(f32::max)
        else {
            return;
        };
        let slack = content.y + content.height - bottom;
        if slack > 0.5 {
            for c in kids {
                self.translate(c, slack / 2.0);
            }
        }
    }

    fn translate(&mut self, id: NodeId, dy: f32) {
        if self.style(id).display == Display::None {
            return;
        }
        self.ui.nodes[id.0].rect.y += dy;
        for c in self.ui.nodes[id.0].children.clone() {
            self.translate(c, dy);
        }
    }

    fn block(&mut self, id: NodeId, content: Rect) {
        let mut y = content.y;
        for c in self.in_flow(id) {
            let cs = self.style(c).clone();
            let width = match border_box(&cs, cs.width, Some(content.width), true) {
                Some(w) => w,
                // A replaced element keeps its own width; a box stretches.
                None if self.replaced(c) => self.outer_size(c, None, Some(content.height)).0,
                None => (content.width - cs.margin.horizontal()).max(0.0),
            };
            let width = self.clamp(&cs, width, true, None);
            let (_, height) = self.outer_size(c, Some(width), Some(content.height));
            let mut r = Rect {
                x: content.x + cs.margin.left,
                y: y + cs.margin.top,
                width,
                height,
            };
            if cs.position == Position::Relative {
                r.x += cs.left.resolve(Some(content.width)).unwrap_or(0.0);
                r.y += cs.top.resolve(Some(content.height)).unwrap_or(0.0);
            }
            self.place(c, r);
            y += height + cs.margin.vertical();
        }
    }

    /// The in-flow items of flex container `id` with their main sizes
    /// resolved (`flex-basis`, then growing or shrinking into `main_size`
    /// with min/max clamping).
    /// The gap along a flex container's main axis, and between its lines.
    fn flex_gaps(style: &Style) -> (f32, f32) {
        let (column, row) = (
            style.column_gap.unwrap_or(style.gap),
            style.row_gap.unwrap_or(style.gap),
        );
        if style.flex_direction == FlexDirection::Row {
            (column, row)
        } else {
            (row, column)
        }
    }

    /// `id`'s flex items, in `order`, at their resolved main sizes on one
    /// line.
    fn flex_items(
        &mut self,
        id: NodeId,
        style: &Style,
        main_size: f32,
        cross_size: f32,
    ) -> Vec<FlexItem> {
        let mut list = self.flex_collect(id, style, main_size, cross_size);
        Self::flex_resolve(&mut list, Self::flex_gaps(style).0, main_size);
        list
    }

    /// `id`'s flex items in `order` (then tree order), at their hypothetical
    /// main sizes: before free space is shared.
    fn flex_collect(
        &mut self,
        id: NodeId,
        style: &Style,
        main_size: f32,
        cross_size: f32,
    ) -> Vec<FlexItem> {
        let row = style.flex_direction == FlexDirection::Row;
        let mut items = self.in_flow(id);
        // `order`: stable, so equal orders keep the tree's.
        items.sort_by_key(|&c| self.style(c).order);
        let mut list: Vec<FlexItem> = vec![];
        for &c in &items {
            let cs = self.style(c).clone();
            let main_len = if row { cs.width } else { cs.height };
            let base = match cs.flex_basis {
                Length::Auto => border_box(&cs, main_len, Some(main_size), row),
                l => border_box(&cs, l, Some(main_size), row),
            };
            let content_main = {
                // In a column, measured at the width it will have: its own
                // (`width`, within `min-width` and `max-width`), else the
                // column's. At the column's, a row that wraps at its own
                // narrower width measured one line tall, and what followed
                // it was drawn over its second line.
                let width = (!row).then(|| own_width(&cs, cross_size));
                let (w, h) = self.content_size(c, width, Some(cross_size));
                if row {
                    w
                } else {
                    h
                }
            };
            let base = base.unwrap_or(content_main);
            let (min_len, max_len) = if row {
                (cs.min_width, cs.max_width)
            } else {
                (cs.min_height, cs.max_height)
            };
            let max = border_box(&cs, max_len, Some(main_size), row).unwrap_or(f32::INFINITY);
            // `min-width: auto`: the content's min-content size (in a row,
            // its widest unbreakable run: text that wraps can shrink to its
            // longest word), no more than the max size, or the specified
            // size if smaller; nothing for a box that clips (a scroll
            // container).
            let specified = border_box(&cs, main_len, Some(main_size), row);
            let automatic = if cs.clip {
                0.0
            } else {
                let min_content = if row {
                    self.min_content_width(c)
                } else {
                    content_main
                };
                let suggestion = min_content.min(content_main).min(max);
                specified.map_or(suggestion, |v| suggestion.min(v))
            };
            let min = border_box(&cs, min_len, Some(main_size), row)
                .unwrap_or(automatic)
                .max(frame(&cs, row));
            let margin_main = if row {
                (cs.margin.left, cs.margin.right)
            } else {
                (cs.margin.top, cs.margin.bottom)
            };
            let margin_cross = if row {
                (cs.margin.top, cs.margin.bottom)
            } else {
                (cs.margin.left, cs.margin.right)
            };
            list.push(FlexItem {
                id: c,
                base,
                min,
                max,
                main: base.clamp(min, max.max(min)),
                grow: cs.flex_grow,
                shrink: cs.flex_shrink,
                margin_main,
                margin_cross,
            });
        }
        list
    }

    /// Shares a line's free space among `list` (growing or shrinking each
    /// within its limits), `gap` between items, along `main_size`.
    fn flex_resolve(list: &mut [FlexItem], gap: f32, main_size: f32) {
        let gaps = gap * list.len().saturating_sub(1) as f32;
        let used: f32 = list
            .iter()
            .map(|i| i.main + i.margin_main.0 + i.margin_main.1)
            .sum::<f32>()
            + gaps;
        let free = main_size - used;
        // Resolve flexible lengths, freezing items that hit their limits.
        if free > 0.0 && list.iter().any(|i| i.grow > 0.0) {
            let mut frozen = vec![false; list.len()];
            for _ in 0..list.len() + 1 {
                let used: f32 = list
                    .iter()
                    .map(|i| i.main + i.margin_main.0 + i.margin_main.1)
                    .sum::<f32>()
                    + gaps;
                let remaining = main_size - used;
                let total: f32 = list
                    .iter()
                    .zip(&frozen)
                    .filter(|(_, f)| !**f)
                    .map(|(i, _)| i.grow)
                    .sum();
                if remaining <= 1e-4 || total <= 0.0 {
                    break;
                }
                let mut clamped = false;
                for (i, f) in list.iter_mut().zip(frozen.iter_mut()).filter(|(_, f)| !**f) {
                    let target = i.main + remaining * i.grow / total;
                    if target > i.max {
                        i.main = i.max;
                        *f = true;
                        clamped = true;
                    } else {
                        i.main = target;
                    }
                }
                if !clamped {
                    break;
                }
            }
        } else if free < 0.0 {
            let mut frozen = vec![false; list.len()];
            for _ in 0..list.len() + 1 {
                let used: f32 = list
                    .iter()
                    .map(|i| i.main + i.margin_main.0 + i.margin_main.1)
                    .sum::<f32>()
                    + gaps;
                let overflow = used - main_size;
                let total: f32 = list
                    .iter()
                    .zip(&frozen)
                    .filter(|(_, f)| !**f)
                    .map(|(i, _)| i.shrink * i.base)
                    .sum();
                if overflow <= 1e-4 || total <= 0.0 {
                    break;
                }
                let mut clamped = false;
                for (i, f) in list.iter_mut().zip(frozen.iter_mut()).filter(|(_, f)| !**f) {
                    let target = i.main - overflow * i.shrink * i.base / total;
                    if target < i.min {
                        i.main = i.min;
                        *f = true;
                        clamped = true;
                    } else {
                        i.main = target;
                    }
                }
                if !clamped {
                    break;
                }
            }
        }
    }

    fn flex(&mut self, id: NodeId, style: &Style, content: Rect) {
        let button = self.is_button(id);
        let row = style.flex_direction == FlexDirection::Row;
        let (main_size, cross_size) = if row {
            (content.width, content.height)
        } else {
            (content.height, content.width)
        };
        let cross_gap = Self::flex_gaps(style).1;
        if style.flex_wrap == FlexWrap::NoWrap {
            let list = self.flex_items(id, style, main_size, cross_size);
            self.flex_line(style, &list, content, 0.0, cross_size, button);
            return;
        }
        // Wrapping: lines by the items' hypothetical sizes, each sharing its
        // own free space, as tall (in a row) as its tallest item.
        let lines = self.flex_lines(id, style, main_size, cross_size);
        let mut crosses: Vec<f32> = lines
            .iter()
            .map(|l| self.flex_line_cross(style, l, cross_size))
            .collect();
        let n = lines.len();
        let total: f32 = crosses.iter().sum::<f32>() + cross_gap * n.saturating_sub(1) as f32;
        let free = cross_size - total;
        let (mut at, between) = match style.align_content {
            AlignContent::Stretch => {
                if free > 0.0 && n > 0 {
                    for c in &mut crosses {
                        *c += free / n as f32;
                    }
                }
                (0.0, 0.0)
            }
            AlignContent::FlexStart => (0.0, 0.0),
            AlignContent::Center => (free / 2.0, 0.0),
            AlignContent::FlexEnd => (free, 0.0),
            AlignContent::SpaceBetween if n > 1 => (0.0, free.max(0.0) / (n - 1) as f32),
            AlignContent::SpaceBetween => (0.0, 0.0),
            AlignContent::SpaceAround if free > 0.0 => (free / n as f32 / 2.0, free / n as f32),
            AlignContent::SpaceEvenly if free > 0.0 => {
                (free / (n + 1) as f32, free / (n + 1) as f32)
            }
            AlignContent::SpaceAround | AlignContent::SpaceEvenly => (0.0, 0.0),
        };
        for (k, line) in lines.iter().enumerate() {
            // `wrap-reverse`: the first line at the far side.
            let start = if style.flex_wrap == FlexWrap::WrapReverse {
                cross_size - at - crosses[k]
            } else {
                at
            };
            self.flex_line(style, line, content, start, crosses[k], button);
            at += crosses[k] + cross_gap + between;
        }
    }

    /// `id`'s flex items broken into lines along `main_size`, each line's
    /// main sizes resolved.
    fn flex_lines(
        &mut self,
        id: NodeId,
        style: &Style,
        main_size: f32,
        cross_size: f32,
    ) -> Vec<Vec<FlexItem>> {
        let gap = Self::flex_gaps(style).0;
        let all = self.flex_collect(id, style, main_size, cross_size);
        let mut lines: Vec<Vec<FlexItem>> = vec![];
        let mut line: Vec<FlexItem> = vec![];
        let mut used = 0.0;
        for item in all {
            let outer = item.main + item.margin_main.0 + item.margin_main.1;
            if !line.is_empty() && used + gap + outer > main_size + 0.01 {
                lines.push(std::mem::take(&mut line));
                used = 0.0;
            }
            used += if line.is_empty() { 0.0 } else { gap } + outer;
            line.push(item);
        }
        if !line.is_empty() {
            lines.push(line);
        }
        for l in &mut lines {
            Self::flex_resolve(l, gap, main_size);
        }
        lines
    }

    /// How much of the cross axis a line of items needs: its biggest item's
    /// cross size (not stretched) and margins.
    fn flex_line_cross(&mut self, style: &Style, line: &[FlexItem], cross_size: f32) -> f32 {
        let row = style.flex_direction == FlexDirection::Row;
        let mut most = 0.0f32;
        for item in line {
            let cs = self.style(item.id).clone();
            let cross_len = if row { cs.height } else { cs.width };
            let cross = match border_box(&cs, cross_len, Some(cross_size), !row) {
                Some(c) => c,
                None => {
                    let (w, h) = if row {
                        self.content_size(item.id, Some(item.main), Some(cross_size))
                    } else {
                        self.content_size(item.id, None, Some(item.main))
                    };
                    if row {
                        h
                    } else {
                        w
                    }
                }
            };
            let cross = self.clamp(&cs, cross, !row, Some(cross_size));
            most = most.max(cross + item.margin_cross.0 + item.margin_cross.1);
        }
        most
    }

    /// Places one line of items: `line_cross` of the cross axis from
    /// `cross_start`, justified along the main axis (from its end when the
    /// direction is reversed).
    fn flex_line(
        &mut self,
        style: &Style,
        list: &[FlexItem],
        content: Rect,
        cross_start: f32,
        line_cross: f32,
        button: bool,
    ) {
        let row = style.flex_direction == FlexDirection::Row;
        let (main_size, cross_size) = if row {
            (content.width, content.height)
        } else {
            (content.height, content.width)
        };
        let gap = Self::flex_gaps(style).0;
        let gaps = gap * list.len().saturating_sub(1) as f32;
        // Main-axis placement.
        let used: f32 = list
            .iter()
            .map(|i| i.main + i.margin_main.0 + i.margin_main.1)
            .sum::<f32>()
            + gaps;
        // Negative when the items overflow: centred and end-aligned items then
        // overflow both ends or the start, as in CSS (`unsafe` alignment).
        let leftover = main_size - used;
        let n = list.len();
        let (mut pos, between) = match style.justify_content {
            JustifyContent::FlexStart => (0.0, 0.0),
            JustifyContent::Center => (leftover / 2.0, 0.0),
            JustifyContent::FlexEnd => (leftover, 0.0),
            JustifyContent::SpaceBetween if n > 1 => (0.0, leftover.max(0.0) / (n - 1) as f32),
            JustifyContent::SpaceBetween => (0.0, 0.0),
            JustifyContent::SpaceAround if leftover > 0.0 && n > 0 => {
                (leftover / n as f32 / 2.0, leftover / n as f32)
            }
            JustifyContent::SpaceEvenly if leftover > 0.0 => {
                (leftover / (n + 1) as f32, leftover / (n + 1) as f32)
            }
            JustifyContent::SpaceAround | JustifyContent::SpaceEvenly => (0.0, 0.0),
        };
        // Baseline-aligned items of a row: their baselines on the lowest
        // one's (each item's own, from its top).
        let baseline_of = |ctx: &Self, item: &FlexItem, cross: f32| {
            ctx.first_baseline(item.id).unwrap_or(cross) + item.margin_cross.0
        };
        let mut line_baseline: Option<f32> = None;
        if row {
            for i in list {
                let cs = self.style(i.id).clone();
                if cs.align_self.unwrap_or(style.align_items) != AlignItems::Baseline {
                    continue;
                }
                let cross = match border_box(&cs, cs.height, Some(cross_size), false) {
                    Some(c) => c,
                    None => self.content_size(i.id, Some(i.main), Some(cross_size)).1,
                };
                let b = baseline_of(self, i, cross);
                line_baseline = Some(line_baseline.map_or(b, |m| m.max(b)));
            }
        }
        // In a button laid out as a row, its text centred up and down (as in
        // a page's `<button>`), not stretched to its height and drawn at
        // its top.
        let button_row = row && button;
        for item in list {
            let cs = self.style(item.id).clone();
            let mut align = cs.align_self.unwrap_or(style.align_items);
            if button_row
                && align == AlignItems::Stretch
                && matches!(self.ui.nodes[item.id.0].kind, NodeKind::Text(_))
            {
                align = AlignItems::Center;
            }
            let cross_len = if row { cs.height } else { cs.width };
            let avail_cross = line_cross - item.margin_cross.0 - item.margin_cross.1;
            let cross = match border_box(&cs, cross_len, Some(cross_size), !row) {
                Some(c) => c,
                None if align == AlignItems::Stretch => avail_cross.max(0.0),
                None => {
                    let (w, h) = if row {
                        self.content_size(item.id, Some(item.main), Some(cross_size))
                    } else {
                        self.content_size(item.id, None, Some(item.main))
                    };
                    if row {
                        h
                    } else {
                        w
                    }
                }
            };
            let cross = self.clamp(&cs, cross, !row, Some(cross_size));
            let cross_off = cross_start
                + match align {
                    AlignItems::Stretch | AlignItems::FlexStart => 0.0,
                    AlignItems::Center => (avail_cross - cross) / 2.0,
                    AlignItems::FlexEnd => avail_cross - cross,
                    AlignItems::Baseline => match line_baseline {
                        Some(line) => line - baseline_of(self, item, cross),
                        None => 0.0,
                    },
                }
                + item.margin_cross.0;
            pos += item.margin_main.0;
            // Reversed: the same places, counted from the end.
            let main_at = if style.flex_reverse {
                main_size - pos - item.main
            } else {
                pos
            };
            let mut r = if row {
                Rect {
                    x: content.x + main_at,
                    y: content.y + cross_off,
                    width: item.main,
                    height: cross,
                }
            } else {
                Rect {
                    x: content.x + cross_off,
                    y: content.y + main_at,
                    width: cross,
                    height: item.main,
                }
            };
            if cs.position == Position::Relative {
                r.x += cs.left.resolve(Some(content.width)).unwrap_or(0.0);
                r.y += cs.top.resolve(Some(content.height)).unwrap_or(0.0);
            }
            self.place(item.id, r);
            pos += item.main + item.margin_main.1 + gap + between;
        }
    }

    /// Places `id`'s absolutely positioned children. Their containing block
    /// is `id`'s padding box when `id` is positioned (and the root's
    /// otherwise, simplified to `id`'s here).
    fn absolute(&mut self, id: NodeId, rect: Rect, style: &Style) {
        let pad = Rect {
            x: rect.x + style.border.left,
            y: rect.y + style.border.top,
            width: (rect.width - style.border.horizontal()).max(0.0),
            height: (rect.height - style.border.vertical()).max(0.0),
        };
        let children: Vec<NodeId> = self.ui.nodes[id.0]
            .children
            .iter()
            .copied()
            .filter(|&c| {
                let s = &self.ui.nodes[c.0].style;
                s.display != Display::None && s.position == Position::Absolute
            })
            .collect();
        for c in children {
            let cs = self.style(c).clone();
            let left = cs.left.resolve(Some(pad.width));
            let right = cs.right.resolve(Some(pad.width));
            let top = cs.top.resolve(Some(pad.height));
            let bottom = cs.bottom.resolve(Some(pad.height));
            let width = border_box(&cs, cs.width, Some(pad.width), true).or(match (left, right) {
                (Some(l), Some(r)) => Some((pad.width - l - r - cs.margin.horizontal()).max(0.0)),
                _ => None,
            });
            let height =
                border_box(&cs, cs.height, Some(pad.height), false).or(match (top, bottom) {
                    (Some(t), Some(b)) => {
                        Some((pad.height - t - b - cs.margin.vertical()).max(0.0))
                    }
                    _ => None,
                });
            let (cw, ch) = if width.is_none() || height.is_none() {
                self.content_size(c, width, Some(pad.height))
            } else {
                (0.0, 0.0)
            };
            let w = self.clamp(&cs, width.unwrap_or(cw), true, Some(pad.width));
            let h = self.clamp(&cs, height.unwrap_or(ch), false, Some(pad.height));
            let x = match (left, right) {
                (Some(l), _) => pad.x + l + cs.margin.left,
                (None, Some(r)) => pad.x + pad.width - r - cs.margin.right - w,
                _ => pad.x + cs.margin.left,
            };
            let y = match (top, bottom) {
                (Some(t), _) => pad.y + t + cs.margin.top,
                (None, Some(b)) => pad.y + pad.height - b - cs.margin.bottom - h,
                _ => pad.y + cs.margin.top,
            };
            self.place(
                c,
                Rect {
                    x,
                    y,
                    width: w,
                    height: h,
                },
            );
        }
    }
}

/// A grid item, its columns and its rows (start, end exclusive).
type GridItem = (NodeId, (usize, usize), (usize, usize));

/// A grid's tracks, and which tracks each item covers.
struct GridPlan {
    cols: Vec<f32>,
    rows: Vec<f32>,
    col_gap: f32,
    row_gap: f32,
    /// Each item, its columns and its rows (start, end exclusive).
    items: Vec<GridItem>,
}

impl GridPlan {
    fn span(sizes: &[f32], gap: f32, (a, b): (usize, usize)) -> f32 {
        sizes[a..b].iter().sum::<f32>() + gap * (b - a).saturating_sub(1) as f32
    }
    fn width(&self) -> f32 {
        Self::span(&self.cols, self.col_gap, (0, self.cols.len()))
    }
    fn height(&self) -> f32 {
        Self::span(&self.rows, self.row_gap, (0, self.rows.len()))
    }
    fn offset(sizes: &[f32], gap: f32, i: usize) -> f32 {
        sizes[..i].iter().sum::<f32>() + gap * i as f32
    }
}

/// An item's tracks on one axis from its placement, against `explicit`
/// lines' worth of template: a start (`None` to place automatically) and a
/// span.
fn resolve_placement(p: GridPlacement, explicit: usize) -> (Option<usize>, usize) {
    let line = |n: i32| -> usize {
        if n > 0 {
            n as usize - 1
        } else {
            (explicit as i32 + 1 + n).max(0) as usize
        }
    };
    let start = p.start.filter(|&n| n != 0).map(line);
    let end = p.end.filter(|&n| n != 0).map(line);
    match (start, end) {
        (Some(a), Some(b)) => (Some(a.min(b)), a.abs_diff(b).max(1)),
        (None, Some(b)) => {
            let span = p.span.max(1) as usize;
            (Some(b.saturating_sub(span)), span)
        }
        (s, None) => (s, p.span.max(1) as usize),
    }
}

impl Ctx<'_> {
    /// The tracks of a grid with content box `inner_w × inner_h` (each
    /// `None` when not definite), and where its items go.
    fn grid_plan(
        &mut self,
        id: NodeId,
        style: &Style,
        inner_w: Option<f32>,
        inner_h: Option<f32>,
    ) -> GridPlan {
        let children = self.in_flow(id);
        let col_gap = style.column_gap.unwrap_or(style.gap);
        let row_gap = style.row_gap.unwrap_or(style.gap);
        let explicit_cols = style.grid_columns.len();
        let explicit_rows = style.grid_rows.len();
        // Placement: fixed items first, then the rest row by row.
        let wanted: Vec<_> = children
            .iter()
            .map(|&c| {
                let cs = self.style(c);
                (
                    c,
                    resolve_placement(cs.grid_column, explicit_cols),
                    resolve_placement(cs.grid_row, explicit_rows),
                )
            })
            .collect();
        let mut ncols = explicit_cols.max(1);
        for (_, (cs, cspan), _) in &wanted {
            ncols = ncols.max(cs.map_or(*cspan, |s| s + cspan));
        }
        let mut used: Vec<Vec<bool>> = vec![];
        let free = |used: &mut Vec<Vec<bool>>, r: usize, c: usize, rs: usize, cs: usize| -> bool {
            (r..r + rs).all(|rr| (c..c + cs).all(|cc| used.get(rr).is_none_or(|row| !row[cc])))
        };
        let take = |used: &mut Vec<Vec<bool>>, r: usize, c: usize, rs: usize, cs: usize| {
            while used.len() < r + rs {
                used.push(vec![false; ncols]);
            }
            for row in used.iter_mut().skip(r).take(rs) {
                for cell in row.iter_mut().skip(c).take(cs) {
                    *cell = true;
                }
            }
        };
        let mut items = vec![None; wanted.len()];
        for (k, &(c, (col, cspan), (row, rspan))) in wanted.iter().enumerate() {
            if let (Some(col), Some(row)) = (col, row) {
                take(&mut used, row, col, rspan, cspan);
                items[k] = Some((c, (col, col + cspan), (row, row + rspan)));
            }
        }
        let (mut cur_r, mut cur_c) = (0usize, 0usize);
        for (k, &(c, (col, cspan), (row, rspan))) in wanted.iter().enumerate() {
            if items[k].is_some() {
                continue;
            }
            let cspan = cspan.min(ncols);
            let (r, cc) = match (col, row) {
                (Some(col), None) => {
                    let r = (0..)
                        .find(|&r| free(&mut used, r, col, rspan, cspan))
                        .expect("rows are unbounded");
                    (r, col)
                }
                (None, Some(r)) => {
                    let cc = (0..=ncols - cspan)
                        .find(|&cc| free(&mut used, r, cc, rspan, cspan))
                        .unwrap_or(0);
                    (r, cc)
                }
                _ => loop {
                    if cur_c + cspan > ncols {
                        cur_c = 0;
                        cur_r += 1;
                    }
                    if free(&mut used, cur_r, cur_c, rspan, cspan) {
                        break (cur_r, cur_c);
                    }
                    cur_c += 1;
                },
            };
            take(&mut used, r, cc, rspan, cspan);
            if col.is_none() && row.is_none() {
                cur_c = cc + cspan;
            }
            items[k] = Some((c, (cc, cc + cspan), (r, r + rspan)));
        }
        let items: Vec<GridItem> = items.into_iter().flatten().collect();
        let nrows = used.len().max(explicit_rows).max(1);

        // Columns.
        let template = |list: &[Track], i: usize| list.get(i).copied().unwrap_or_else(Track::auto);
        let resolve = |s: TrackSize, base: Option<f32>| match s {
            TrackSize::Px(v) => Some(v),
            TrackSize::Percent(p) => base.map(|b| b * p),
            _ => None,
        };
        let mut cols = vec![0.0f32; ncols];
        let mut col_fr = vec![0.0f32; ncols];
        let mut need = vec![0.0f32; ncols];
        for &(c, (a, b), _) in &items {
            if b - a == 1 {
                let cs = self.style(c).clone();
                let (w, _) = self.outer_size(c, None, inner_h);
                need[a] = need[a].max(w + cs.margin.horizontal());
            }
        }
        for i in 0..ncols {
            let t = template(&style.grid_columns, i);
            let min = resolve(t.min, inner_w).unwrap_or(if t.min == TrackSize::Auto {
                need[i]
            } else {
                0.0
            });
            cols[i] = match t.max {
                TrackSize::Fr(f) if inner_w.is_some() => {
                    col_fr[i] = f;
                    min
                }
                TrackSize::Fr(_) | TrackSize::Auto => min.max(need[i]),
                m => resolve(m, inner_w).unwrap_or(need[i]).max(min),
            };
        }
        grow_spans(
            &mut cols,
            &col_fr,
            col_gap,
            items.iter().map(|&(c, cols, _)| (c, cols)),
            |ctx: &mut Self, c| {
                let cs = ctx.style(c).clone();
                ctx.outer_size(c, None, inner_h).0 + cs.margin.horizontal()
            },
            self,
        );
        if let Some(w) = inner_w {
            share_free(
                &mut cols,
                &col_fr,
                w - col_gap * (ncols - 1) as f32,
                &style.grid_columns,
            );
        }

        // Rows, with items at the widths their columns give them.
        let mut rows = vec![0.0f32; nrows];
        let mut row_fr = vec![0.0f32; nrows];
        let mut need = vec![0.0f32; nrows];
        // What the items must have, at least: their content, but nothing
        // (or their own minimum) for one that scrolls or clips, as CSS's
        // automatic minimum size of a grid item is zero for those.
        let mut need_min = vec![0.0f32; nrows];
        let mut heights = vec![];
        for &(c, ca, (a, b)) in &items {
            let cs = self.style(c).clone();
            let w = cs
                .width
                .resolve(Some(GridPlan::span(&cols, col_gap, ca)))
                .map_or(
                    (GridPlan::span(&cols, col_gap, ca) - cs.margin.horizontal()).max(0.0),
                    |_| {
                        border_box(
                            &cs,
                            cs.width,
                            Some(GridPlan::span(&cols, col_gap, ca)),
                            true,
                        )
                        .unwrap_or(0.0)
                    },
                );
            let (_, h) = self.outer_size(c, Some(w), inner_h);
            let h = h + cs.margin.vertical();
            if b - a == 1 {
                need[a] = need[a].max(h);
                let least = if cs.scroll || cs.scroll_x || cs.clip {
                    cs.min_height.resolve(inner_h).unwrap_or(0.0) + cs.margin.vertical()
                } else {
                    h
                };
                need_min[a] = need_min[a].max(least);
            }
            heights.push((c, (a, b), h));
        }
        for i in 0..nrows {
            let t = template(&style.grid_rows, i);
            let min = resolve(t.min, inner_h).unwrap_or(if t.min == TrackSize::Auto {
                need[i]
            } else {
                0.0
            });
            rows[i] = match t.max {
                TrackSize::Fr(f) if inner_h.is_some() => {
                    row_fr[i] = f;
                    // A flexible row in a box of known height starts from
                    // what its items must have, and shares the rest.
                    resolve(t.min, inner_h).unwrap_or(if t.min == TrackSize::Auto {
                        need_min[i]
                    } else {
                        0.0
                    })
                }
                TrackSize::Fr(_) | TrackSize::Auto => min.max(need[i]),
                m => resolve(m, inner_h).unwrap_or(need[i]).max(min),
            };
        }
        // An item over several rows: the last flexible-or-auto one grows.
        // (One in a single row is counted already, as that row's need.)
        for (_, (a, b), h) in heights {
            let have = GridPlan::span(&rows, row_gap, (a, b));
            if h > have && b - a > 1 {
                let i = (a..b)
                    .rev()
                    .find(|&i| template(&style.grid_rows, i).max == TrackSize::Auto)
                    .unwrap_or(b - 1);
                rows[i] += h - have;
            }
        }
        if let Some(hh) = inner_h {
            share_free(
                &mut rows,
                &row_fr,
                hh - row_gap * (nrows - 1) as f32,
                &style.grid_rows,
            );
        }
        GridPlan {
            cols,
            rows,
            col_gap,
            row_gap,
            items,
        }
    }

    /// Lays out a grid's items in its content box.
    fn grid(&mut self, id: NodeId, style: &Style, content: Rect) {
        // Placed, the grid's box is final, so its height is definite: `fr`
        // rows share what its rows leave of it (nothing, for a grid as tall
        // as its content).
        let plan = self.grid_plan(id, style, Some(content.width), Some(content.height));
        for &(c, ca, ra) in &plan.items {
            let cs = self.style(c).clone();
            let cell = Rect {
                x: content.x + GridPlan::offset(&plan.cols, plan.col_gap, ca.0),
                y: content.y + GridPlan::offset(&plan.rows, plan.row_gap, ra.0),
                width: GridPlan::span(&plan.cols, plan.col_gap, ca),
                height: GridPlan::span(&plan.rows, plan.row_gap, ra),
            };
            let room_w = (cell.width - cs.margin.horizontal()).max(0.0);
            let room_h = (cell.height - cs.margin.vertical()).max(0.0);
            let w = border_box(&cs, cs.width, Some(cell.width), true).unwrap_or(room_w);
            let w = self.clamp(&cs, w, true, Some(cell.width));
            let align = cs.align_self.unwrap_or(style.align_items);
            let h = match border_box(&cs, cs.height, Some(cell.height), false) {
                Some(h) => h,
                None if align == AlignItems::Stretch => room_h,
                None => self.outer_size(c, Some(w), Some(cell.height)).1,
            };
            let h = self.clamp(&cs, h, false, Some(cell.height));
            let dy = match align {
                AlignItems::Center => (room_h - h) / 2.0,
                AlignItems::FlexEnd => room_h - h,
                _ => 0.0,
            };
            self.place(
                c,
                Rect {
                    x: cell.x + cs.margin.left,
                    y: cell.y + cs.margin.top + dy,
                    width: w,
                    height: h,
                },
            );
        }
    }
}

/// Grows the inflexible tracks an item spans until it fits, evenly.
fn grow_spans<'a>(
    sizes: &mut [f32],
    fr: &[f32],
    gap: f32,
    items: impl Iterator<Item = (NodeId, (usize, usize))>,
    mut need: impl FnMut(&mut Ctx<'a>, NodeId) -> f32,
    ctx: &mut Ctx<'a>,
) {
    for (c, (a, b)) in items {
        if b - a < 2 {
            continue;
        }
        let want = need(ctx, c);
        let have = GridPlan::span(sizes, gap, (a, b));
        let grow: Vec<usize> = (a..b).filter(|&i| fr[i] == 0.0).collect();
        if want > have && !grow.is_empty() {
            let each = (want - have) / grow.len() as f32;
            for i in grow {
                sizes[i] += each;
            }
        }
    }
}

/// Shares `space` left over the tracks' sizes: by `fr` among flexible
/// tracks (none going below its minimum), or, with none, evenly among
/// `auto` ones, as `justify-content: normal` stretches them.
fn share_free(sizes: &mut [f32], fr: &[f32], space: f32, template: &[Track]) {
    let total_fr: f32 = fr.iter().sum();
    if total_fr > 0.0 {
        // Flexible tracks start from their minimums (already in `sizes`).
        let fixed: f32 = sizes
            .iter()
            .zip(fr)
            .filter(|(_, f)| **f == 0.0)
            .map(|(s, _)| *s)
            .sum();
        let mut flexible: Vec<usize> = (0..sizes.len()).filter(|&i| fr[i] > 0.0).collect();
        // A track whose minimum beats its share keeps its minimum, and the
        // rest share what is left: repeat until all fit.
        loop {
            let left = space
                - fixed
                - (0..sizes.len())
                    .filter(|i| fr[*i] > 0.0 && !flexible.contains(i))
                    .map(|i| sizes[i])
                    .sum::<f32>();
            let share: f32 = flexible.iter().map(|&i| fr[i]).sum();
            let unit = (left / share.max(1e-6)).max(0.0);
            let (stuck, ok): (Vec<usize>, Vec<usize>) =
                flexible.iter().partition(|&&i| sizes[i] > fr[i] * unit);
            if stuck.is_empty() {
                for i in ok {
                    sizes[i] = fr[i] * unit;
                }
                return;
            }
            flexible = ok;
            if flexible.is_empty() {
                return;
            }
        }
    }
    let used: f32 = sizes.iter().sum();
    let autos: Vec<usize> = (0..sizes.len())
        .filter(|&i| template.get(i).copied().unwrap_or_else(Track::auto).max == TrackSize::Auto)
        .collect();
    if space > used && !autos.is_empty() {
        let each = (space - used) / autos.len() as f32;
        for i in autos {
            sizes[i] += each;
        }
    }
}

/// The border-box width a column gives an item, in `avail` (the column's
/// content width): its own (`width`, within `min-width` and `max-width`),
/// else all there is but its margins.
fn own_width(cs: &Style, avail: f32) -> f32 {
    let w = border_box(cs, cs.width, Some(avail), true).unwrap_or(avail - cs.margin.horizontal());
    let w = border_box(cs, cs.max_width, Some(avail), true).map_or(w, |m| w.min(m));
    border_box(cs, cs.min_width, Some(avail), true).map_or(w, |m| w.max(m))
}

#[cfg(test)]
mod tests {
    use super::super::*;

    fn px(v: f32) -> Length {
        Length::Px(v)
    }

    /// Widths that do not add up: 7 px a character, less 1 for each "AV"
    /// pair and each space before a "T" (kerning, at a size where a face's
    /// is a fraction of that).
    struct Kerned;

    impl TextMeasure for Kerned {
        fn measure(&mut self, text: &str, _: &Style) -> (f32, f32) {
            let c: Vec<char> = text.chars().collect();
            let kerns = c
                .windows(2)
                .filter(|p| matches!(p, ['A', 'V'] | [' ', 'T']))
                .count();
            (c.len() as f32 * 7.0 - kerns as f32, 16.0)
        }
    }

    /// [`Kerned`], counting what it measures.
    struct Counted(usize);

    impl TextMeasure for Counted {
        fn measure(&mut self, text: &str, style: &Style) -> (f32, f32) {
            self.0 += 1;
            Kerned.measure(text, style)
        }
    }

    #[test]
    fn a_paragraph_laid_out_again_is_not_wrapped_again() {
        let text = "A paragraph long enough to wrap over several lines of a narrow box, as a sidebar's hint does, Tea AV VAT".repeat(3);
        for units in [false, true] {
            let mut ui = Ui::new();
            ui.set_layout_units(units);
            let root = ui.root();
            let b = ui.add_box(root, Style { width: px(160.0), wrap: true, ..Style::default() });
            ui.add_text(b, &text);
            let mut first = Counted(0);
            ui.layout(400.0, 800.0, &mut first);
            let lines = ui.rect(b).height;
            // the tree built again, as an app rebuilds it on every change
            let mut ui2 = ui.fresh();
            let root = ui2.root();
            let b2 = ui2.add_box(root, Style { width: px(160.0), wrap: true, ..Style::default() });
            ui2.add_text(b2, &text);
            let mut again = Counted(0);
            ui2.layout(400.0, 800.0, &mut again);
            assert_eq!(ui2.rect(b2).height, lines, "the same lines");
            assert!(again.0 * 10 < first.0, "units {units}: {} measures again, {} the first time", again.0, first.0);
            // narrower: wrapped anew
            ui2.update_style(b2, |s| s.width = px(120.0));
            ui2.layout(400.0, 800.0, &mut Kerned);
            assert!(ui2.rect(b2).height > lines);
        }
    }

    #[test]
    fn wrapped_lines_fit_and_could_take_no_more_words() {
        let words = [
            "AV", "Tea", "A", "VAT", "word", "T", "AVAV", "longer", "TT", "x",
        ];
        let mut seed = 7u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for case in 0..300 {
            let text: Vec<&str> = (0..5 + next() % 30)
                .map(|_| words[(next() % words.len() as u32) as usize])
                .collect();
            let text = text.join(" ");
            let width = 20.0 + (next() % 200) as f32;
            for units in [false, true] {
                let mut ui = Ui::new();
                ui.set_layout_units(units);
                let root = ui.root();
                let st = Style {
                    width: px(width),
                    wrap: true,
                    ..Style::default()
                };
                let b = ui.add_box(root, st);
                let t = ui.add_text(b, text.clone());
                ui.layout(800.0, 600.0, &mut Kerned);
                let lines = ui.text_lines_of(t);
                assert_eq!(lines.concat(), text, "case {case}: text lost or changed");
                // With layout units a line is measured between its ends'
                // offsets in the whole text, kerning across the break
                // included: only the plain measure is checked here.
                if units {
                    continue;
                }
                let w = |s: &str| Kerned.measure(s.trim_end(), &Style::default()).0;
                for (i, l) in lines.iter().enumerate() {
                    // A line holding one word may be too wide only when that
                    // word was broken to fit (it is then no wider).
                    assert!(
                        w(l) <= width + 1e-3
                            || !l.trim().contains(' ') && l.trim().chars().count() == 1,
                        "case {case}: {l:?} wider than {width} in {lines:?}"
                    );
                    if let Some(next_line) = lines.get(i + 1)
                        && let Some(word) = next_line.split(' ').next()
                    {
                        let more = format!("{}{word}", l);
                        assert!(
                            w(&more) > width + 1e-3,
                            "case {case}: {more:?} fits {width}; {lines:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn layout_units_put_boxes_on_the_64th_pixel_grid() {
        assert_eq!(
            (
                layout_unit(976.6199),
                layout_unit(976.63),
                layout_unit(-0.01)
            ),
            (976.0 + 39.0 / 64.0, 976.625, 0.0)
        );
        assert_eq!(
            (
                super::onto_grid(976.62463),
                super::onto_grid(976.6199),
                super::onto_grid(33.359375)
            ),
            (976.625, 976.6199, 33.359375)
        );
        let mut ui = Ui::new();
        ui.set_layout_units(true);
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let kids: Vec<_> = (0..3)
            .map(|_| {
                ui.add_box(
                    root,
                    Style {
                        flex_grow: 1.0,
                        ..Style::default()
                    },
                )
            })
            .collect();
        ui.layout(96.0, 10.0, &mut FixedWidthText(7.0));
        for k in kids {
            let r = ui.rect(k);
            assert_eq!((r.x * 64.0).fract(), 0.0);
            assert_eq!(r.width, 32.0);
        }
    }

    #[test]
    fn layout_units_fit_a_wrapped_line_by_its_offsets_in_the_text() {
        // Glyphs 7.01 wide: "bb" after "a " ends at 4 × 7.01 = 28.04
        // (28.046875 in 1/64 px) and starts at 14.02 (14.03125), so it is
        // 14.015625 wide there, and fits 14.02 where alone (14.02) it would not.
        let mut ui = Ui::new();
        ui.set_layout_units(true);
        let root = ui.root();
        let block = ui.add_box(
            root,
            crate::style! { width: 14.016px; white-space: normal; word-wrap: break-word; },
        );
        let t = ui.add_text(block, "a bb");
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.01));
        assert_eq!(ui.nodes[t.0].lines, vec!["a ", "bb"]);
        ui.set_layout_units(false);
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.01));
        assert_eq!(ui.nodes[t.0].lines, vec!["a ", "b", "b"]);
    }

    #[test]
    fn rows_share_free_space_by_grow_and_respect_margins() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let a = ui.add_box(
            root,
            Style {
                width: px(100.0),
                ..Style::default()
            },
        );
        let b = ui.add_box(
            root,
            Style {
                flex_grow: 1.0,
                margin: Edges::new(0.0, 5.0, 0.0, 5.0),
                ..Style::default()
            },
        );
        let c = ui.add_box(
            root,
            Style {
                flex_grow: 3.0,
                ..Style::default()
            },
        );
        ui.layout(500.0, 50.0, &mut FixedWidthText(7.0));
        assert_eq!(
            ui.rect(a),
            Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 50.0
            }
        );
        // 390 free: 97.5 and 292.5.
        assert_eq!(ui.rect(b).x, 105.0);
        assert_eq!(ui.rect(b).width, 97.5);
        assert_eq!(ui.rect(c).x, 207.5);
        assert_eq!(ui.rect(c).width, 292.5);
    }

    #[test]
    fn content_box_sizing_adds_padding_and_border() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let a = ui.add_box(
            root,
            Style {
                width: px(20.0),
                height: px(18.0),
                padding: Edges::symmetric(0.0, 2.0),
                border: Edges::all(1.0),
                align_self: Some(AlignItems::FlexStart),
                ..Style::default()
            },
        );
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(a).width, 26.0);
        assert_eq!(ui.rect(a).height, 20.0);
    }

    #[test]
    fn text_sizes_items_and_shrinks_stop_at_content() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            Style {
                align_items: AlignItems::Center,
                ..Style::flex(FlexDirection::Row)
            },
        );
        let label = ui.add_box(
            root,
            Style {
                padding: Edges::all(1.0),
                ..Style::default()
            },
        );
        let t = ui.add_text(label, "abcd");
        let filler = ui.add_box(
            root,
            Style {
                width: px(1000.0),
                ..Style::default()
            },
        );
        ui.layout(300.0, 40.0, &mut FixedWidthText(7.0));
        // The label keeps its 28 + 2 px; the filler takes the rest.
        assert_eq!(ui.rect(label).width, 30.0);
        assert_eq!(ui.rect(t).width, 28.0);
        assert_eq!(ui.rect(filler).width, 270.0);
        // Centered: 40 tall, the label 1.2·16 + 2.
        assert_eq!(ui.rect(label).y, (40.0 - 21.2) / 2.0);
    }

    #[test]
    fn columns_justify_and_absolute_children_place_by_offsets() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(
            root,
            Style {
                justify_content: JustifyContent::SpaceBetween,
                position: Position::Relative,
                ..Style::flex(FlexDirection::Column)
            },
        );
        let a = ui.add_box(
            root,
            Style {
                height: px(10.0),
                ..Style::default()
            },
        );
        let b = ui.add_box(
            root,
            Style {
                height: px(10.0),
                ..Style::default()
            },
        );
        let abs = ui.add_box(
            root,
            Style {
                position: Position::Absolute,
                right: px(4.0),
                bottom: px(6.0),
                width: px(16.0),
                height: px(16.0),
                ..Style::default()
            },
        );
        ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(a).y, 0.0);
        assert_eq!(ui.rect(b).y, 90.0);
        assert_eq!(ui.rect(a).width, 200.0, "stretched across");
        assert_eq!(
            ui.rect(abs),
            Rect {
                x: 180.0,
                y: 78.0,
                width: 16.0,
                height: 16.0
            }
        );
        assert_eq!(ui.hit_test(185.0, 80.0), Some(abs));
        assert_eq!(ui.hit_test(5.0, 95.0), Some(b));
    }

    #[test]
    fn overflowing_items_centre_and_keep_their_content_size() {
        // neuroglancer's layout buttons: an 18 px border-box button, 6 px
        // padding each side, centring a 15 px wide glyph box that may shrink
        // to its 13 px content and no further.
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let button = ui.add_box(
            root,
            Style {
                box_sizing: BoxSizing::BorderBox,
                width: px(18.0),
                height: px(18.0),
                padding: Edges::symmetric(1.0, 6.0),
                border: Edges::all(1.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                align_self: Some(AlignItems::FlexStart),
                ..Style::flex(FlexDirection::Row)
            },
        );
        let inner = ui.add_box(
            button,
            Style {
                width: px(15.0),
                height: px(15.0),
                ..Style::default()
            },
        );
        ui.add_text(inner, "ab");
        ui.layout(100.0, 100.0, &mut FixedWidthText(6.5));
        assert_eq!(
            ui.rect(inner),
            Rect {
                x: 7.0 + (4.0 - 13.0) / 2.0,
                y: 1.5,
                width: 13.0,
                height: 15.0
            }
        );
    }

    #[test]
    fn scrolling_boxes_move_their_content_within_bounds() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let list = ui.add_box(root, crate::style! { display: flex; flex-direction: column; height: 50px; overflow: auto; scroll-top: {30.0}; });
        let rows: Vec<_> = (0..4)
            .map(|_| {
                ui.add_box(
                    list,
                    Style {
                        height: px(20.0),
                        flex_shrink: 0.0,
                        ..Style::default()
                    },
                )
            })
            .collect();
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
        // Rows that do not shrink overflow.
        assert_eq!(ui.scroll_extent(list), (80.0, 50.0, 30.0));
        assert_eq!(ui.rect(rows[0]).y, -30.0);
        assert_eq!(ui.hit_test(5.0, 45.0), Some(rows[3]));
        assert_eq!(ui.scroll_container(rows[3]), Some(list));
        // Beyond the end: clamped.
        ui.style_mut(list).scroll_top = 500.0;
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.scroll_extent(list).2, 30.0);
    }

    #[test]
    fn text_wraps_at_break_opportunities_then_anywhere() {
        let mut ui = Ui::new();
        let root = ui.root();
        let block = ui.add_box(
            root,
            crate::style! { width: 70px; white-space: normal; word-wrap: break-word; },
        );
        // 10 px a character: "ab://" "cd/" fit in 7, "efghijkl" breaks.
        let t = ui.add_text(block, "ab://\u{200b}cd/\u{200b}efghijklmnop");
        ui.layout(200.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!(
            ui.nodes[t.0].lines,
            vec!["ab://\u{200b}", "cd/\u{200b}", "efghijk", "lmnop"]
        );
        assert_eq!(ui.rect(t).height, 4.0 * 19.2);
        // Vertical text is as tall as it is long.
        let v = ui.add_box(
            root,
            crate::style! { writing-mode: vertical-lr; position: absolute; },
        );
        let vt = ui.add_text(v, "abc");
        ui.layout(200.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!((ui.rect(vt).width, ui.rect(vt).height), (19.2, 30.0));
    }

    /// `text-overflow: ellipsis`: a run too wide for its clipping box keeps
    /// what fits before `…`; one that fits is left whole.
    #[test]
    fn text_overflow_ellipsizes() {
        let mut ui = Ui::new();
        let root = ui.root();
        let narrow = ui.add_box(
            root,
            crate::style! { width: 50px; overflow: hidden; text-overflow: ellipsis; },
        );
        let t = ui.add_text(narrow, "abcdefghij");
        let wide = ui.add_box(
            root,
            crate::style! { width: 200px; overflow: hidden; text-overflow: ellipsis; },
        );
        let u = ui.add_text(wide, "abcdefghij");
        ui.layout(300.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!(ui.nodes[t.0].lines, vec!["abcd\u{2026}"]);
        assert!(ui.nodes[u.0].lines.is_empty());
        assert_eq!(ui.rect(t).width, 50.0);
    }

    /// `white-space: pre` keeps the text's own line breaks: the box is as
    /// tall as its lines and as wide as the widest, and what follows it
    /// moves down; `pre-wrap` wraps each line as well; `normal` and
    /// `nowrap` lay it out as one run.
    #[test]
    fn line_breaks_kept_by_pre_make_the_box_as_tall_as_its_lines() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let pre = ui.add_box(root, crate::style! { white-space: pre; align-self: flex-start; });
        let t = ui.add_text(pre, "one\nthree\ntwo");
        let after = ui.add_box(root, crate::style! { height: 10px; });
        let wrapped = ui.add_box(root, crate::style! { white-space: pre-wrap; width: 50px; });
        let w = ui.add_text(wrapped, "short\nfour more words");
        let run = ui.add_box(root, crate::style! { white-space: nowrap; align-self: flex-start; });
        let r = ui.add_text(run, "a\nb");
        ui.layout(200.0, 400.0, &mut FixedWidthText(10.0));
        // Three lines of 19.2, as wide as "three".
        assert_eq!(ui.rect(pre).height, 3.0 * 19.2);
        assert_eq!(ui.rect(pre).width, 50.0);
        assert_eq!(ui.rect(after).y, 3.0 * 19.2);
        assert_eq!(ui.text_lines_of(t), vec!["one", "three", "two"]);
        // "short", then "four more words" wrapped at 50 px into three.
        assert_eq!(ui.rect(wrapped).height, 4.0 * 19.2);
        assert_eq!(ui.text_lines_of(w).len(), 4);
        // One run, one line.
        assert_eq!(ui.rect(run).height, 19.2);
        assert_eq!(ui.text_lines_of(r).len(), 1);
    }

    /// A row's height comes from its items at the widths flexing gives
    /// them: a narrow item's wrapped text makes the row two lines tall, and
    /// what follows the row moves down.
    #[test]
    fn rows_are_as_tall_as_their_items_wrap() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let row = ui.add_box(
            root,
            crate::style! { display: flex; width: 100px; align-items: flex-start; },
        );
        let a = ui.add_box(row, crate::style! { flex: 1; min-width: 0; });
        ui.add_text(a, "Name");
        let b = ui.add_box(row, crate::style! { flex: 1; min-width: 0; });
        let tb = ui.add_box(b, crate::style! { white-space: normal; });
        // 10 px a character: 50 px wide, "Default value" wraps in three
        // ("Defau", "lt", "value": the word is longer than the line). The
        // row is measured at its own 100 px, not the column's 200.
        ui.add_text(tb, "Default value");
        let after = ui.add_box(root, crate::style! { height: 10px; });
        ui.layout(200.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!(ui.rect(b).width, 50.0);
        assert_eq!(ui.rect(row).height, 3.0 * 19.2);
        assert_eq!(ui.rect(after).y, 3.0 * 19.2);
    }

    #[test]
    fn negative_margins_pull_boxes_outward() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let bar = ui.add_box(
            root,
            Style {
                height: px(20.0),
                margin: Edges::new(0.0, 0.0, 3.0, 0.0),
                ..Style::default()
            },
        );
        let panel = ui.add_box(
            root,
            Style {
                height: px(24.0),
                margin: Edges::new(-1.0, 0.0, 0.0, -5.0),
                ..Style::default()
            },
        );
        ui.layout(1200.0, 800.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(bar).height, 20.0);
        assert_eq!(
            ui.rect(panel),
            Rect {
                x: -5.0,
                y: 22.0,
                width: 1205.0,
                height: 24.0
            }
        );
    }
}
