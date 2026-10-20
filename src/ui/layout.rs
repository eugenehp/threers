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
//! box. Text is one line (`white-space: nowrap`).

use super::style::{AlignItems, BoxSizing, Display, FlexDirection, JustifyContent, Length, Position, Style};
use super::tree::{NodeId, NodeKind, Rect, Ui};

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
        (text.chars().count() as f32 * self.0, style.line_height.unwrap_or(style.font_size * 1.2))
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

struct Ctx<'a> {
    ui: &'a mut Ui,
    text: &'a mut dyn TextMeasure,
}

pub(crate) fn layout(ui: &mut Ui, width: f32, height: f32, text: &mut dyn TextMeasure) {
    let root = ui.root();
    let mut ctx = Ctx { ui, text };
    ctx.place(root, Rect { x: 0.0, y: 0.0, width, height });
}

impl Ctx<'_> {
    fn style(&self, id: NodeId) -> &Style {
        &self.ui.nodes[id.0].style
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
    fn content_size(&mut self, id: NodeId, width: Option<f32>, height_base: Option<f32>) -> (f32, f32) {
        let style = self.style(id).clone();
        let (fx, fy) = (frame(&style, true), frame(&style, false));
        if let NodeKind::Text(t) = &self.ui.nodes[id.0].kind {
            let t = t.clone();
            // Blink keeps lengths in 1/64 px (`LayoutUnit`) and rounds a
            // run's width up to one.
            let (w, h) = self.text.measure(&t, &style);
            let w = (w * 64.0).ceil() / 64.0;
            if let (true, Some(width)) = (style.wrap, width) {
                let lines = self.wrap(&t, &style, (width - fx).max(0.0));
                return (width, lines.len().max(1) as f32 * h + fy);
            }
            if style.vertical {
                return (h + fx, w + fy);
            }
            return (width.unwrap_or(w + fx), h + fy);
        }
        let inner_w = width.map(|w| (w - fx).max(0.0));
        let children = self.in_flow(id);
        let (cw, ch) = match style.display {
            Display::Flex => {
                let row = style.flex_direction == FlexDirection::Row;
                let mut main = 0.0f32;
                let mut cross = 0.0f32;
                for (i, &c) in children.iter().enumerate() {
                    let cs = self.style(c).clone();
                    let (w, h) = self.outer_size(c, if row { None } else { inner_w }, height_base);
                    let (m_main, m_cross) = if row {
                        (cs.margin.horizontal(), cs.margin.vertical())
                    } else {
                        (cs.margin.vertical(), cs.margin.horizontal())
                    };
                    let (a, b) = if row { (w, h) } else { (h, w) };
                    main += a + m_main + if i > 0 { style.gap } else { 0.0 };
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

    /// A box's border-box size from its own `width`/`height` where given,
    /// else its content, clamped by `min-*`/`max-*`.
    fn outer_size(&mut self, id: NodeId, width: Option<f32>, height_base: Option<f32>) -> (f32, f32) {
        let style = self.style(id).clone();
        let w = width.or_else(|| border_box(&style, style.width, None, true));
        let h = border_box(&style, style.height, height_base, false);
        let (cw, ch) = if w.is_none() || h.is_none() { self.content_size(id, w, height_base) } else { (0.0, 0.0) };
        let w = self.clamp(&style, w.unwrap_or(cw), true, None);
        let h = self.clamp(&style, h.unwrap_or(ch), false, height_base);
        (w, h)
    }

    fn clamp(&self, style: &Style, v: f32, horizontal: bool, base: Option<f32>) -> f32 {
        let (min, max) = if horizontal { (style.min_width, style.max_width) } else { (style.min_height, style.max_height) };
        let mut v = v;
        if let Some(m) = border_box(style, max, base, horizontal) {
            v = v.min(m);
        }
        if let Some(m) = border_box(style, min, base, horizontal) {
            v = v.max(m);
        }
        v.max(frame(style, horizontal))
    }

    /// `text` broken into lines no wider than `width`: at spaces and
    /// zero-width spaces where they fit, else within a word.
    fn wrap(&mut self, text: &str, style: &Style, width: f32) -> Vec<String> {
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
        let fits = |this: &mut Self, s: &str| this.text.measure(s.trim_end_matches([' ', '\u{200b}']), style).0 <= width + 1e-3;
        let mut lines: Vec<String> = vec![];
        let mut line = String::new();
        for piece in pieces {
            let candidate = format!("{line}{piece}");
            if fits(self, &candidate) {
                line = candidate;
                continue;
            }
            if !line.is_empty() {
                lines.push(std::mem::take(&mut line));
            }
            // `break-word`: a piece too long for a line of its own breaks
            // where it must.
            let mut rest = piece;
            while !fits(self, &rest) {
                let chars: Vec<char> = rest.chars().collect();
                let mut n = chars.len();
                while n > 1 && !fits(self, &chars[..n].iter().collect::<String>()) {
                    n -= 1;
                }
                lines.push(chars[..n].iter().collect());
                rest = chars[n..].iter().collect();
            }
            line = rest;
        }
        if !line.is_empty() || lines.is_empty() {
            lines.push(line);
        }
        lines
    }

    /// Gives `id` its border box and lays out its children inside it.
    fn place(&mut self, id: NodeId, rect: Rect) {
        self.ui.nodes[id.0].rect = rect;
        let style = self.style(id).clone();
        if let (NodeKind::Text(t), true) = (&self.ui.nodes[id.0].kind, style.wrap) {
            let t = t.clone();
            let lines = self.wrap(&t, &style, (rect.width - frame(&style, true)).max(0.0));
            self.ui.nodes[id.0].lines = lines;
        }
        let content = Rect {
            x: rect.x + style.border.left + style.padding.left,
            y: rect.y + style.border.top + style.padding.top,
            width: (rect.width - frame(&style, true)).max(0.0),
            height: (rect.height - frame(&style, false)).max(0.0),
        };
        match style.display {
            Display::Flex => self.flex(id, &style, content),
            _ => self.block(id, content),
        }
        self.absolute(id, rect, &style);
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
            let scrolled = style.scroll_top.clamp(0.0, (height - visible).max(0.0));
            self.ui.nodes[id.0].scroll_height = height;
            self.ui.nodes[id.0].scrolled = scrolled;
            if scrolled != 0.0 {
                for c in children {
                    self.translate(c, -scrolled);
                }
            }
        }
    }

    /// Moves `id` and everything inside it down by `dy`.
    fn translate(&mut self, id: NodeId, dy: f32) {
        self.ui.nodes[id.0].rect.y += dy;
        for c in self.ui.nodes[id.0].children.clone() {
            self.translate(c, dy);
        }
    }

    fn block(&mut self, id: NodeId, content: Rect) {
        let mut y = content.y;
        for c in self.in_flow(id) {
            let cs = self.style(c).clone();
            let width = border_box(&cs, cs.width, Some(content.width), true)
                .unwrap_or((content.width - cs.margin.horizontal()).max(0.0));
            let width = self.clamp(&cs, width, true, None);
            let (_, height) = self.outer_size(c, Some(width), Some(content.height));
            let mut r = Rect { x: content.x + cs.margin.left, y: y + cs.margin.top, width, height };
            if cs.position == Position::Relative {
                r.x += cs.left.resolve(Some(content.width)).unwrap_or(0.0);
                r.y += cs.top.resolve(Some(content.height)).unwrap_or(0.0);
            }
            self.place(c, r);
            y += height + cs.margin.vertical();
        }
    }

    fn flex(&mut self, id: NodeId, style: &Style, content: Rect) {
        let row = style.flex_direction == FlexDirection::Row;
        let (main_size, cross_size) = if row { (content.width, content.height) } else { (content.height, content.width) };
        let items = self.in_flow(id);
        struct Item {
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
        let mut list: Vec<Item> = vec![];
        for &c in &items {
            let cs = self.style(c).clone();
            let main_len = if row { cs.width } else { cs.height };
            let base = match cs.flex_basis {
                Length::Auto => border_box(&cs, main_len, Some(main_size), row),
                l => border_box(&cs, l, Some(main_size), row),
            };
            let content_main = {
                let (w, h) = self.content_size(c, if row { None } else { Some(cross_size - cs.margin.horizontal()) }, Some(cross_size));
                if row {
                    w
                } else {
                    h
                }
            };
            let base = base.unwrap_or(content_main);
            let (min_len, max_len) = if row { (cs.min_width, cs.max_width) } else { (cs.min_height, cs.max_height) };
            // `min-width: auto`: the content size, or the specified size if
            // smaller; nothing for a box that clips (a scroll container).
            let specified = border_box(&cs, main_len, Some(main_size), row);
            let automatic = if cs.clip { 0.0 } else { specified.map_or(content_main, |v| content_main.min(v)) };
            let min = border_box(&cs, min_len, Some(main_size), row).unwrap_or(automatic).max(frame(&cs, row));
            let max = border_box(&cs, max_len, Some(main_size), row).unwrap_or(f32::INFINITY);
            let margin_main = if row { (cs.margin.left, cs.margin.right) } else { (cs.margin.top, cs.margin.bottom) };
            let margin_cross = if row { (cs.margin.top, cs.margin.bottom) } else { (cs.margin.left, cs.margin.right) };
            list.push(Item {
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
        let gaps = style.gap * list.len().saturating_sub(1) as f32;
        let used: f32 = list.iter().map(|i| i.main + i.margin_main.0 + i.margin_main.1).sum::<f32>() + gaps;
        let free = main_size - used;
        // Resolve flexible lengths, freezing items that hit their limits.
        if free > 0.0 && list.iter().any(|i| i.grow > 0.0) {
            let mut frozen = vec![false; list.len()];
            for _ in 0..list.len() + 1 {
                let used: f32 = list.iter().map(|i| i.main + i.margin_main.0 + i.margin_main.1).sum::<f32>() + gaps;
                let remaining = main_size - used;
                let total: f32 = list.iter().zip(&frozen).filter(|(_, f)| !**f).map(|(i, _)| i.grow).sum();
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
                let used: f32 = list.iter().map(|i| i.main + i.margin_main.0 + i.margin_main.1).sum::<f32>() + gaps;
                let overflow = used - main_size;
                let total: f32 = list.iter().zip(&frozen).filter(|(_, f)| !**f).map(|(i, _)| i.shrink * i.base).sum();
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
        // Main-axis placement.
        let used: f32 = list.iter().map(|i| i.main + i.margin_main.0 + i.margin_main.1).sum::<f32>() + gaps;
        // Negative when the items overflow: centred and end-aligned items then
        // overflow both ends or the start, as in CSS (`unsafe` alignment).
        let leftover = main_size - used;
        let (mut pos, between) = match style.justify_content {
            JustifyContent::FlexStart => (0.0, 0.0),
            JustifyContent::Center => (leftover / 2.0, 0.0),
            JustifyContent::FlexEnd => (leftover, 0.0),
            JustifyContent::SpaceBetween if list.len() > 1 => (0.0, leftover.max(0.0) / (list.len() - 1) as f32),
            JustifyContent::SpaceBetween => (0.0, 0.0),
        };
        for item in &list {
            let cs = self.style(item.id).clone();
            let align = cs.align_self.unwrap_or(style.align_items);
            let cross_len = if row { cs.height } else { cs.width };
            let avail_cross = cross_size - item.margin_cross.0 - item.margin_cross.1;
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
            let cross_off = match align {
                AlignItems::Stretch | AlignItems::FlexStart => 0.0,
                AlignItems::Center => (avail_cross - cross) / 2.0,
                AlignItems::FlexEnd => avail_cross - cross,
            } + item.margin_cross.0;
            pos += item.margin_main.0;
            let mut r = if row {
                Rect { x: content.x + pos, y: content.y + cross_off, width: item.main, height: cross }
            } else {
                Rect { x: content.x + cross_off, y: content.y + pos, width: cross, height: item.main }
            };
            if cs.position == Position::Relative {
                r.x += cs.left.resolve(Some(content.width)).unwrap_or(0.0);
                r.y += cs.top.resolve(Some(content.height)).unwrap_or(0.0);
            }
            self.place(item.id, r);
            pos += item.main + item.margin_main.1 + style.gap + between;
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
            let height = border_box(&cs, cs.height, Some(pad.height), false).or(match (top, bottom) {
                (Some(t), Some(b)) => Some((pad.height - t - b - cs.margin.vertical()).max(0.0)),
                _ => None,
            });
            let (cw, ch) = if width.is_none() || height.is_none() { self.content_size(c, width, Some(pad.height)) } else { (0.0, 0.0) };
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
            self.place(c, Rect { x, y, width: w, height: h });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;

    fn px(v: f32) -> Length {
        Length::Px(v)
    }

    #[test]
    fn rows_share_free_space_by_grow_and_respect_margins() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Row));
        let a = ui.add_box(root, Style { width: px(100.0), ..Style::default() });
        let b = ui.add_box(root, Style { flex_grow: 1.0, margin: Edges::new(0.0, 5.0, 0.0, 5.0), ..Style::default() });
        let c = ui.add_box(root, Style { flex_grow: 3.0, ..Style::default() });
        ui.layout(500.0, 50.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(a), Rect { x: 0.0, y: 0.0, width: 100.0, height: 50.0 });
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
            Style { width: px(20.0), height: px(18.0), padding: Edges::symmetric(0.0, 2.0), border: Edges::all(1.0), align_self: Some(AlignItems::FlexStart), ..Style::default() },
        );
        ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(a).width, 26.0);
        assert_eq!(ui.rect(a).height, 20.0);
    }

    #[test]
    fn text_sizes_items_and_shrinks_stop_at_content() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style { align_items: AlignItems::Center, ..Style::flex(FlexDirection::Row) });
        let label = ui.add_box(root, Style { padding: Edges::all(1.0), ..Style::default() });
        let t = ui.add_text(label, "abcd");
        let filler = ui.add_box(root, Style { width: px(1000.0), ..Style::default() });
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
        ui.set_style(root, Style { justify_content: JustifyContent::SpaceBetween, position: Position::Relative, ..Style::flex(FlexDirection::Column) });
        let a = ui.add_box(root, Style { height: px(10.0), ..Style::default() });
        let b = ui.add_box(root, Style { height: px(10.0), ..Style::default() });
        let abs = ui.add_box(root, Style { position: Position::Absolute, right: px(4.0), bottom: px(6.0), width: px(16.0), height: px(16.0), ..Style::default() });
        ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(a).y, 0.0);
        assert_eq!(ui.rect(b).y, 90.0);
        assert_eq!(ui.rect(a).width, 200.0, "stretched across");
        assert_eq!(ui.rect(abs), Rect { x: 180.0, y: 78.0, width: 16.0, height: 16.0 });
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
        let inner = ui.add_box(button, Style { width: px(15.0), height: px(15.0), ..Style::default() });
        ui.add_text(inner, "ab");
        ui.layout(100.0, 100.0, &mut FixedWidthText(6.5));
        assert_eq!(ui.rect(inner), Rect { x: 7.0 + (4.0 - 13.0) / 2.0, y: 1.5, width: 13.0, height: 15.0 });
    }

    #[test]
    fn scrolling_boxes_move_their_content_within_bounds() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let list = ui.add_box(root, crate::style! { display: flex; flex-direction: column; height: 50px; overflow: auto; scroll-top: {30.0}; });
        let rows: Vec<_> = (0..4).map(|_| ui.add_box(list, Style { height: px(20.0), flex_shrink: 0.0, ..Style::default() })).collect();
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
        let block = ui.add_box(root, crate::style! { width: 70px; white-space: normal; word-wrap: break-word; });
        // 10 px a character: "ab://" "cd/" fit in 7, "efghijkl" breaks.
        let t = ui.add_text(block, "ab://\u{200b}cd/\u{200b}efghijklmnop");
        ui.layout(200.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!(ui.nodes[t.0].lines, vec!["ab://\u{200b}", "cd/\u{200b}", "efghijk", "lmnop"]);
        assert_eq!(ui.rect(t).height, 4.0 * 19.2);
        // Vertical text is as tall as it is long.
        let v = ui.add_box(root, crate::style! { writing-mode: vertical-lr; position: absolute; });
        let vt = ui.add_text(v, "abc");
        ui.layout(200.0, 200.0, &mut FixedWidthText(10.0));
        assert_eq!((ui.rect(vt).width, ui.rect(vt).height), (19.2, 30.0));
    }

    #[test]
    fn negative_margins_pull_boxes_outward() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, Style::flex(FlexDirection::Column));
        let bar = ui.add_box(root, Style { height: px(20.0), margin: Edges::new(0.0, 0.0, 3.0, 0.0), ..Style::default() });
        let panel = ui.add_box(root, Style { height: px(24.0), margin: Edges::new(-1.0, 0.0, 0.0, -5.0), ..Style::default() });
        ui.layout(1200.0, 800.0, &mut FixedWidthText(7.0));
        assert_eq!(ui.rect(bar).height, 20.0);
        assert_eq!(ui.rect(panel), Rect { x: -5.0, y: 22.0, width: 1205.0, height: 24.0 });
    }
}
