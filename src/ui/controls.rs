//! Form controls as Chrome draws them by default (its "refreshed" controls,
//! light scheme unless asked): checkbox, button, `<select>` and its option
//! list, `<input type=range>`, a text input's contents (selection, caret)
//! and focus ring, and a code view. Each builds an ordinary subtree, so
//! callers restyle it like any other node — the sizes here are Chrome's
//! defaults, measured against its rendering.
//!
//! ```
//! use threers::ui::{controls, FixedWidthText, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let font = threers::style! { font-family: Arial; font-size: 13px; color: black; };
//! let mut text = FixedWidthText(7.0);
//! let select = controls::select(&mut ui, root, "off", &["off", "on"], &font, &mut text);
//! ui.layout(200.0, 50.0, &mut text);
//! assert_eq!(ui.rect(select).width, 21.0 + 22.0);
//! ```

use super::a11y::{Role, Semantics};
use super::{Edges, Length, NodeId, Position, Rect, Rgba, Style, TextField, TextMeasure, Ui};
use crate::style;

/// The accent: a checked box, a range's fill and thumb (`#0075ff`).
pub const ACCENT: Rgba = [0.0, 0x75 as f32 / 255.0, 1.0, 1.0];
/// The accent under the pointer (`#005cc8`).
pub const ACCENT_HOVER: Rgba = [0.0, 0x5c as f32 / 255.0, 0xc8 as f32 / 255.0, 1.0];
/// The focus ring (`#005fcc`).
pub const FOCUS_RING: Rgba = [0.0, 0x5f as f32 / 255.0, 0xcc as f32 / 255.0, 1.0];
/// Selected text's background in a dark field (`#4d7198`).
pub const SELECTION: Rgba = [77.0 / 255.0, 113.0 / 255.0, 152.0 / 255.0, 1.0];
/// A control's border (`#767676`).
pub const BORDER: Rgba = [118.0 / 255.0, 118.0 / 255.0, 118.0 / 255.0, 1.0];

/// Chromium's tick (`NativeThemeBase::PaintCheckbox`) in a 13px box: from
/// (0.2, 0.5) down by 0.2 and up to (0.8, 0.2) of the box, 0.16 of its
/// height wide, butt-ended, as its outline, to fill.
const TICK: &str = "M1.865 7.235L5.286 10.657L11.212 3.250L9.588 1.950L5.114 7.543L3.335 5.765Z";

/// A checkbox `size` px square (13 by default): checked, the accent with a
/// white tick; else white inside a 1px `border` (Chrome's `#767676`, or
/// darker over a dark page).
pub fn checkbox(ui: &mut Ui, parent: NodeId, checked: bool, size: f32, border: Rgba, margin: Edges) -> NodeId {
    let node = ui.add_box(
        parent,
        style! {
            box-sizing: border-box;
            width: {size};
            height: {size};
            margin: {margin};
            border-radius: {if checked { 1.5 } else { 2.0 }};
            flex-shrink: 0;
        },
    );
    if checked && size == 13.0 {
        // As Chrome rasterizes it, which a rounded fill on a canvas does not
        // (its coverage comes in quarters): opaque but for the corner
        // pixels, each the accent's border blend at 0.443 (16, 60, 113
        // premultiplied); the pixels beside them, 0.99 there, opaque here.
        ui.style_mut(node).position = Position::Relative;
        let corner = [36.0 / 255.0, 135.0 / 255.0, 1.0, 0.443];
        for (x, y, w, h, color) in [(1.0, 0.0, 11.0, 13.0, ACCENT), (0.0, 1.0, 13.0, 11.0, ACCENT), (0.0, 0.0, 1.0, 1.0, corner), (12.0, 0.0, 1.0, 1.0, corner), (0.0, 12.0, 1.0, 1.0, corner), (12.0, 12.0, 1.0, 1.0, corner)] {
            ui.add_box(node, style! { position: absolute; left: {x}; top: {y}; width: {w}; height: {h}; background: {color}; });
        }
        ui.add_path(node, style! { position: absolute; left: 0; top: 0; width: {size}; height: {size}; color: white; }, vec![TICK.into()], [0.0, 0.0, 13.0, 13.0], None, true);
    } else if checked {
        ui.style_mut(node).background = Some(ACCENT);
        ui.add_path(node, style! { width: {size}; height: {size}; color: white; }, vec![TICK.into()], [0.0, 0.0, 13.0, 13.0], None, true);
    } else {
        let s = ui.style_mut(node);
        s.border = Edges::all(1.0);
        s.border_color = border;
        s.background = Some([1.0; 4]);
    }
    // Named by the app (`ui.semantics_mut(node).label`), or by the text
    // beside it once the two share a labelled parent.
    ui.set_semantics(node, Semantics::new(Role::Checkbox).checked(checked));
    node
}

/// A radio button, 13px: checked, an accent ring around white around a 7px
/// accent dot; else white in a `#767676` ring.
pub fn radio(ui: &mut Ui, parent: NodeId, checked: bool, margin: Edges) -> NodeId {
    let node = ui.add_box(parent, style! { position: relative; box-sizing: border-box; width: 13px; height: 13px; margin: {margin}; border-radius: 6.5px; flex-shrink: 0; });
    if checked {
        ui.style_mut(node).background = Some(ACCENT);
        ui.add_box(node, style! { position: absolute; left: 1px; top: 1px; width: 11px; height: 11px; border-radius: 5.5px; background: white; });
        let dot = ui.add_box(node, style! { position: absolute; left: 3px; top: 3px; width: 7px; height: 7px; border-radius: 3.5px; });
        ui.style_mut(dot).background = Some(ACCENT);
    } else {
        let s = ui.style_mut(node);
        s.border = Edges::all(1.0);
        s.border_color = BORDER;
        s.background = Some([1.0; 4]);
    }
    ui.set_semantics(node, Semantics::new(Role::Radio).checked(checked));
    node
}

/// An `<input type=color>` showing `color`, `width` wide and 27px tall: a
/// grey well, the colour in a bordered swatch.
pub fn color_well(ui: &mut Ui, parent: NodeId, color: [f32; 3], width: f32) -> NodeId {
    let node = ui.add_box(
        parent,
        style! { display: flex; box-sizing: border-box; width: {width}; height: 27px; padding: 5px 4px; border: 1px solid; border-color: {BORDER}; background: #efefef; flex-shrink: 0; },
    );
    ui.add_box(node, style! { flex: 1; border: 1px solid #777; background: {[color[0], color[1], color[2], 1.0]}; });
    // As a page's `<input type=color>` reads: a button whose value is the
    // colour.
    let hex = color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    ui.set_semantics(
        node,
        Semantics::new(Role::Button).label("Colour").value(format!("#{:02x}{:02x}{:02x}", hex[0], hex[1], hex[2])),
    );
    node
}

/// A `<button>` with `label` in `font` (Chrome's 13.33px Arial by default;
/// neuroglancer's 11pt): light grey, rounded, 6px padding, as tall as Arial's
/// line and 6px.
pub fn button(ui: &mut Ui, parent: NodeId, label: &str, font: &Style, text: &mut dyn TextMeasure) -> NodeId {
    let (w, _) = text.measure(label, font);
    let node = ui.add_box(
        parent,
        style! {
            display: flex;
            align-items: center;
            justify-content: center;
            box-sizing: border-box;
            width: {w + 16.0};
            // Arial's line (ascent and descent rounded), 1px padding and
            // 2px borders each way.
            height: {(font.font_size * 0.905).round() + (font.font_size * 0.212).round() + 6.0};
            border: 1px solid;
            border-color: {BORDER};
            border-radius: 2px;
            background: #efefef;
            flex-shrink: 0;
        },
    );
    let t = ui.add_box(node, font.clone());
    ui.add_text(t, label);
    // Named by its label, as a button is.
    ui.set_semantics(node, Semantics::new(Role::Button));
    node
}

/// A closed `<select>` showing `value`: white, rounded, a chevron; as wide
/// as its widest option plus 22px, 19px tall.
pub fn select(ui: &mut Ui, parent: NodeId, value: &str, options: &[&str], font: &Style, text: &mut dyn TextMeasure) -> NodeId {
    let widest = options.iter().map(|o| text.measure(o, font).0).fold(0.0, f32::max);
    let node = ui.add_box(
        parent,
        style! {
            display: flex;
            position: relative;
            align-items: center;
            box-sizing: border-box;
            width: {widest.ceil() + 22.0};
            height: 19px;
            padding-left: 4px;
            border: 1px solid;
            border-color: {BORDER};
            border-radius: 2px;
            background: white;
            flex-shrink: 0;
        },
    );
    let t = ui.add_box(node, font.clone());
    ui.add_text(t, value);
    ui.add_path(
        node,
        style! { position: absolute; right: 4px; top: 5.5px; width: 9px; height: 7px; color: {font.color}; },
        vec!["M1 1.5L4.5 5L8 1.5".into()],
        [0.0, 0.0, 9.0, 7.0],
        Some(2.0),
        false,
    );
    ui.set_semantics(node, Semantics::new(Role::ComboBox).value(value).expanded(false));
    // The choice is its value; the text showing it is not read again.
    hide_text(ui, node);
    node
}

/// An open `<select>`'s options as a list under `anchor` (the closed
/// select's box), added to `parent` (absolutely placed: give it the root).
/// The `hovered` option, else the `current` one, is highlighted. Returns
/// the list and each option's node.
pub fn select_popup(ui: &mut Ui, parent: NodeId, anchor: Rect, options: &[&str], current: Option<usize>, hovered: Option<usize>, font: &Style) -> (NodeId, Vec<NodeId>) {
    let list = ui.add_box(
        parent,
        style! {
            position: absolute;
            left: {anchor.x};
            top: {anchor.y + anchor.height};
            min-width: {anchor.width};
            display: flex;
            flex-direction: column;
            box-sizing: border-box;
            padding: 4px 0;
            border: 1px solid #a0a0a0;
            border-radius: 4px;
            background: white;
        },
    );
    let lit = hovered.or(current);
    let items = options
        .iter()
        .enumerate()
        .map(|(k, option)| {
            let on = lit == Some(k);
            let item = ui.add_box(list, style! { display: flex; align-items: center; height: 19px; padding: 0 8px; flex-shrink: 0; });
            ui.style_mut(item).background = on.then_some(ACCENT);
            let t = ui.add_box(item, style! { ..font.clone(); color: {if on { [1.0; 4] } else { font.color }}; });
            ui.add_text(t, *option);
            ui.set_semantics(item, Semantics::new(Role::Option).selected(current == Some(k)));
            item
        })
        .collect();
    ui.set_semantics(list, Semantics::new(Role::ListBox));
    (list, items)
}

/// An `<input type=range>` at `fraction` of its way (Chrome's 129 × 16): an
/// 8px track inset 1.5px, filled with the accent up to the thumb's centre,
/// and a 16px thumb whose centre runs from 8px in to 8px short of the end.
/// Darker while `hovered`. Set the returned box's width to resize it.
/// A range input's thumb run: its 129px less the thumb.
const RANGE_RUN: f32 = 129.0 - 16.0;

pub fn range(ui: &mut Ui, parent: NodeId, fraction: f32, hovered: bool) -> NodeId {
    let (accent, track, edge) = if hovered {
        (ACCENT_HOVER, [229.0 / 255.0, 229.0 / 255.0, 229.0 / 255.0, 1.0], [0x28 as f32 / 255.0, 0x55 as f32 / 255.0, 0x8b as f32 / 255.0, 1.0])
    } else {
        (ACCENT, [239.0 / 255.0, 239.0 / 255.0, 239.0 / 255.0, 1.0], [0x3b as f32 / 255.0, 0x76 as f32 / 255.0, 0xba as f32 / 255.0, 1.0])
    };
    let f = fraction.clamp(0.0, 1.0);
    let node = ui.add_box(parent, style! { position: relative; width: 129px; height: 16px; flex-shrink: 0; });
    ui.add_box(
        node,
        style! { position: absolute; left: 1px; right: 1px; top: 4px; height: 8px; box-sizing: border-box; border: 1px solid #b2b2b2; border-radius: 4px; background: {track}; },
    );
    // The thumb's run: 8px in from each end; the thumb at whole pixels
    // along it.
    let offset = (f * RANGE_RUN).round();
    let run = ui.add_box(node, style! { position: absolute; left: 8px; right: 8px; top: 0; bottom: 0; });
    // The fill: from the track's start (7px before the run) to the thumb.
    ui.add_box(
        run,
        style! { position: absolute; left: -7px; right: {RANGE_RUN - offset}; top: 4px; height: 8px; box-sizing: border-box; border: 1px solid; border-color: {edge}; border-radius: 4px; background: {accent}; },
    );
    ui.add_box(run, style! { position: absolute; left: {offset}; margin-left: -8px; top: 0; width: 16px; height: 16px; border-radius: 8px; background: {accent}; });
    // As a page's range input reads by default: 0 to 100.
    ui.set_semantics(node, Semantics::new(Role::Slider).range(0.0, 100.0, (f * 100.0).round()));
    node
}

/// The fraction of a range input's way at pointer `x`, for a box `r` (the
/// thumb's centre under the pointer), clamped to [0, 1].
pub fn range_fraction(r: Rect, x: f32) -> f32 {
    ((x - r.x - 8.0) / (r.width - 16.0)).clamp(0.0, 1.0)
}

/// What a text input shows.
#[derive(Debug, Clone, Copy)]
pub struct TextView<'a> {
    /// The value, shown while not editing.
    pub value: &'a str,
    /// Shown, in `placeholder_color`, while the text is empty.
    pub placeholder: &'a str,
    pub placeholder_color: Rgba,
    /// The text being edited, if focused.
    pub editing: Option<&'a TextField>,
    /// Whether the caret is in its visible half of the blink.
    pub caret_on: bool,
    pub font: &'a Style,
    /// Where the text starts in the box, past its left edge.
    pub x_offset: f32,
    /// The selection's background.
    pub selection: Rgba,
}

/// Fills a text input box `node` (give it `position: relative`): the
/// selection or the caret, then the text (or placeholder) over them.
pub fn text_contents(ui: &mut Ui, node: NodeId, view: TextView, text: &mut dyn TextMeasure) {
    let shown = view.editing.map_or(view.value, |t| t.text.as_str());
    if let Some(t) = view.editing {
        let (before, selected, _) = t.spans();
        let x = view.x_offset + text.measure(&before, view.font).0;
        let w = text.measure(&selected, view.font).0;
        if !selected.is_empty() {
            ui.add_box(node, style! { position: absolute; left: {x}; top: 1px; width: {w}; height: 15px; background: {view.selection}; });
        } else if view.caret_on {
            ui.add_box(node, style! { position: absolute; left: {x}; top: 2px; width: 1px; height: 15px; background: {view.font.color}; });
        }
    }
    let (s, color) = if shown.is_empty() { (view.placeholder, view.placeholder_color) } else { (shown, view.font.color) };
    if !s.is_empty() {
        let t = ui.add_box(node, style! { ..view.font.clone(); position: relative; color: {color}; white-space: pre; });
        ui.add_text(t, s);
    }
    // A text box whose value is the text, not the placeholder: that is a
    // hint, read as the description. An app's own label stays.
    let sem = ui.semantics_mut(node);
    if sem.role == Role::Generic {
        sem.role = Role::TextBox;
    }
    sem.value = Some(shown.to_owned());
    if !view.placeholder.is_empty() && sem.description.is_none() {
        sem.description = Some(view.placeholder.to_owned());
    }
    hide_text(ui, node);
}

/// Chrome's focus ring on `node` (`:focus-visible`, `outline: auto`): 2px of
/// [`FOCUS_RING`] across its edge, a 1px white line outside that. `node`
/// must be positioned.
pub fn focus_ring(ui: &mut Ui, node: NodeId) -> NodeId {
    ui.add_box(node, style! { position: absolute; left: -2px; right: -2px; top: -2px; bottom: -2px; border: 1px solid white; });
    let ring = ui.add_box(node, style! { position: absolute; left: -1px; right: -1px; top: -1px; bottom: -1px; border: 2px solid; });
    ui.style_mut(ring).border_color = FOCUS_RING;
    ui.set_semantics(ring, Semantics::new(Role::None).hidden());
    ring
}

/// A code view in a CodeMirror-like frame: white, a 16px #f7f7f7 gutter
/// with a #ddd edge, lines of coloured runs in `font`, 4px padding, `height`
/// tall (which may be fractional: the last pixel row is blended, as Chrome
/// does for an editor 8em tall).
pub fn code_view(ui: &mut Ui, parent: NodeId, lines: &[Vec<(String, Rgba)>], font: &Style, line_height: f32, height: f32) -> NodeId {
    let editor = ui.add_box(parent, style! { display: flex; position: relative; box-sizing: border-box; height: {height}; border: 1px solid transparent; flex-shrink: 0; });
    let whole = height.floor();
    ui.add_box(editor, style! { position: absolute; left: -1px; top: -1px; right: -1px; height: {whole}; background: white; });
    if height > whole {
        ui.add_box(editor, style! { position: absolute; left: -1px; top: {whole - 1.0}; right: -1px; height: {height - whole}; background: {[1.0, 1.0, 1.0, height - whole]}; });
    }
    // Over those (positioned, later): the gutter and the lines, clipped.
    let inner = ui.add_box(editor, style! { display: flex; position: relative; flex: 1; min-width: 0; overflow: hidden; });
    ui.add_box(inner, style! { width: 16px; border-width: 0 1px 0 0; border-color: #ddd; background: #f7f7f7; flex-shrink: 0; });
    let body = ui.add_box(inner, style! { display: flex; flex-direction: column; padding: 4px; flex: 1; min-width: 0; });
    for runs in lines {
        let line = ui.add_box(body, style! { display: flex; height: {line_height}; flex-shrink: 0; });
        for (s, color) in runs {
            let t = ui.add_box(line, style! { ..font.clone(); color: {*color}; white-space: pre; flex-shrink: 0; });
            ui.add_text(t, s);
        }
    }
    // Read as a block of text, not run by run; not editable, so not a Tab
    // stop.
    let value: Vec<String> = lines.iter().map(|l| l.iter().map(|(s, _)| s.as_str()).collect()).collect();
    ui.set_semantics(editor, Semantics::new(Role::TextArea).value(value.join("\n")).focusable(false));
    hide_text(ui, editor);
    editor
}

/// Hides `node`'s text descendants from assistive technology: a text box
/// reads its value once, rather than every coloured run of it again.
fn hide_text(ui: &mut Ui, node: NodeId) {
    for &c in ui.children(node).to_vec().iter() {
        if matches!(ui.kind(c), super::NodeKind::Text(_)) {
            ui.set_semantics(c, Semantics::new(Role::None).hidden());
        } else {
            hide_text(ui, c);
        }
    }
}


/// What [`code_editor`] shows.
pub struct CodeEditorView<'a> {
    pub area: &'a super::TextArea,
    pub folds: &'a super::code::Folds,
    /// A line's coloured runs (the whole line, in order).
    pub tokens: &'a dyn Fn(usize) -> Vec<(String, Rgba)>,
    pub font: &'a Style,
    pub line_height: f32,
    /// The editor's height (fractional heights blend the last row).
    pub height: f32,
    /// Whether it has the keyboard: the caret shows, the selection is
    /// `#d7d4f0` (else `#d9d9d9`).
    pub focused: bool,
    pub caret_on: bool,
    pub scroll_top: f32,
    /// A fold gutter (`CodeMirror-foldgutter`) after the lint one.
    pub fold_gutter: bool,
}

/// The nodes of a [`code_editor`]: the frame, the scrolling body, each shown
/// line's text box (with what it shows) and each fold marker (with its
/// line).
pub struct CodeEditorNodes {
    pub editor: NodeId,
    pub body: NodeId,
    pub lines: Vec<(NodeId, super::code::ShownLine)>,
    pub markers: Vec<(NodeId, usize)>,
}

/// A CodeMirror editor (default theme): white, its gutters, the lines with
/// folded ranges as `↔`, the selection and caret. Text starts 4px into each
/// line; [`code_column`] maps a pointer to a column.
pub fn code_editor(ui: &mut Ui, parent: NodeId, view: &CodeEditorView, text: &mut dyn TextMeasure) -> CodeEditorNodes {
    let editor = ui.add_box(parent, style! { display: flex; position: relative; box-sizing: border-box; height: {view.height}; flex-shrink: 0; });
    let whole = view.height.floor();
    ui.add_box(editor, style! { position: absolute; left: 0; top: 0; right: 0; height: {whole}; background: white; });
    if view.height > whole {
        ui.add_box(editor, style! { position: absolute; left: 0; top: {whole}; right: 0; height: {view.height - whole}; background: {[1.0, 1.0, 1.0, view.height - whole]}; });
    }
    let inner = ui.add_box(editor, style! { display: flex; position: relative; flex: 1; min-width: 0; overflow: hidden; });
    let fold_width = if view.fold_gutter { 0.7 * view.font.font_size } else { 0.0 };
    // `.CodeMirror-gutters`: #f7f7f7 with a #ddd edge.
    // The lines start past the gutters' `offsetWidth`: whole pixels.
    let gutters = ui.add_box(inner, style! { display: flex; flex-direction: column; width: {(16.0 + fold_width).round()}; border-width: 0 1px 0 0; border-color: #ddd; background: #f7f7f7; flex-shrink: 0; });
    let body = ui.add_box(inner, style! { display: flex; flex-direction: column; flex: 1; min-width: 0; overflow: auto; scroll-top: {view.scroll_top}; });
    let pad = ui.add_box(body, style! { height: 4px; flex-shrink: 0; });
    let _ = pad;
    let shown = super::code::shown_lines(&view.area.text, view.folds);
    let foldable: Vec<usize> = if view.fold_gutter { super::code::brace_folds(&view.area.text).iter().map(|f| f.start_line).collect() } else { vec![] };
    let lines_text: Vec<&str> = view.area.lines();
    let (caret_line, caret_col) = view.area.line_col(view.area.caret);
    let selection: Rgba = if view.focused { [0xd7 as f32 / 255.0, 0xd4 as f32 / 255.0, 0xf0 as f32 / 255.0, 1.0] } else { [0xd9 as f32 / 255.0; 4] };
    let gutter_pad = ui.add_box(gutters, style! { height: {4.0 - view.scroll_top}; flex-shrink: 0; });
    let _ = gutter_pad;
    let mut out_lines = vec![];
    let mut markers = vec![];
    let width_of = |text: &mut dyn TextMeasure, s: &str| text.measure(s, view.font).0;
    for s in shown {
        let row = ui.add_box(body, style! { display: flex; position: relative; height: {view.line_height}; padding: 0 4px; flex-shrink: 0; });
        let src: Vec<char> = lines_text[s.line].chars().collect();
        // The selection behind the text.
        if let Some((a, z, past)) = view.area.selection_in_line(s.line) {
            let x0 = 4.0 + width_of(text, &src[..a.min(src.len())].iter().collect::<String>());
            let x1 = 4.0 + width_of(text, &src[..z.min(src.len())].iter().collect::<String>());
            let sel = ui.add_box(row, style! { position: absolute; left: {x0}; top: 0; height: {view.line_height}; width: {(x1 - x0).max(0.0)}; background: {selection}; });
            if past {
                let s = ui.style_mut(sel);
                s.right = Length::Px(0.0);
                s.width = Length::Auto;
            }
        }
        let runs = (view.tokens)(s.line);
        let push_runs = |ui: &mut Ui, runs: &[(String, Rgba)], from: usize, to: usize| {
            let mut col = 0;
            for (t, color) in runs {
                let chars: Vec<char> = t.chars().collect();
                let (a, z) = (from.max(col).min(col + chars.len()), to.min(col + chars.len()).max(from.max(col).min(col + chars.len())));
                if z > a {
                    let n = ui.add_box(row, style! { ..view.font.clone(); color: {*color}; white-space: pre; position: relative; flex-shrink: 0; });
                    ui.add_text(n, chars[a - col..z - col].iter().collect::<String>());
                }
                col += chars.len();
            }
        };
        match s.fold {
            None => push_runs(ui, &runs, 0, usize::MAX),
            Some(f) => {
                push_runs(ui, &runs, 0, f.start_col + 1);
                // `.CodeMirror-foldmarker`: a blue `↔` in Arial with a #b9f
                // glow (`text-shadow` at ±1px both ways, blurred 2px). Its
                // size is `medium` again outside `monospace`: 16px.
                let size = if view.font.font_family == "monospace" { 16.0 } else { view.font.font_size };
                // (`line-height: .3` puts it 2px above the line.)
                let m = ui.add_box(row, style! { display: flex; align-items: center; height: {view.line_height}; top: -1px; font-family: Arial; font-size: {size}; color: blue; position: relative; flex-shrink: 0; });
                for (dx, dy) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
                    let glow = ui.add_box(m, style! { position: absolute; left: {dx}; top: {dy}; font-family: Arial; font-size: {size}; color: {[0xbb as f32 / 255.0, 0x99 as f32 / 255.0, 1.0, 0.5]}; });
                    ui.add_text(glow, "↔");
                }
                let t = ui.add_box(m, style! { position: relative; font-family: Arial; font-size: {size}; color: blue; });
                ui.add_text(t, "↔");
                markers.push((m, s.line));
                let end_runs = (view.tokens)(f.end_line);
                push_runs(ui, &end_runs, f.end_col, usize::MAX);
            }
        }
        if view.focused && view.caret_on && caret_line == s.line && view.area.selection().0 == view.area.selection().1 {
            let x = 4.0 + width_of(text, &src[..caret_col.min(src.len())].iter().collect::<String>());
            ui.add_box(row, style! { position: absolute; left: {x}; top: 0; width: 1px; height: {view.line_height}; background: black; });
        }
        // The gutter's row: `▾` open, `▸` folded.
        let g = ui.add_box(gutters, style! { display: flex; justify-content: flex-end; height: {view.line_height}; flex-shrink: 0; });
        if view.fold_gutter && foldable.contains(&s.line) {
            // `.CodeMirror-guttermarker-subtle`: #999, the editor's font.
            let cell = ui.add_box(g, style! { ..view.font.clone(); display: flex; width: {fold_width}; color: #999; overflow: hidden; });
            ui.add_text(cell, if s.fold.is_some() { "▸" } else { "▾" });
            markers.push((cell, s.line));
        }
        out_lines.push((row, s));
    }
    ui.set_semantics(editor, Semantics::new(Role::TextArea).value(view.area.text.clone()));
    hide_text(ui, editor);
    CodeEditorNodes { editor, body, lines: out_lines, markers }
}

/// The column of `line` under pointer `x`, for a line box `row` of a
/// [`code_editor`]: the nearest character boundary.
pub fn code_column(row: Rect, x: f32, line: &str, font: &Style, text: &mut dyn TextMeasure) -> usize {
    let chars: Vec<char> = line.chars().collect();
    let mut best = (0, f32::INFINITY);
    for k in 0..=chars.len() {
        let w = text.measure(&chars[..k].iter().collect::<String>(), font).0;
        let d = (row.x + 4.0 + w - x).abs();
        if d < best.1 {
            best = (k, d);
        }
    }
    best.0
}

/// A dimmed page and a white box in its middle (`Overlay`): returns the
/// box, `width` wide (padding 16px, black text), painted where centring puts
/// it (fractional edges blended, as a transformed box is).
pub fn overlay(ui: &mut Ui, root: NodeId, width: Length) -> NodeId {
    let shade = ui.add_box(
        root,
        style! { position: absolute; left: 0; top: 0; right: 0; bottom: 0; display: flex; align-items: center; justify-content: center; background: {[0.0, 0.0, 0.0, 0.8]}; },
    );
    let content = ui.add_box(shade, style! { position: relative; display: flex; flex-direction: column; width: {width}; padding: 16px; background: white; color: black; flex-shrink: 0; });
    // Centred by `transform: translate(-50%, -50%)`: at half pixels, not
    // snapped.
    ui.style_mut(content).subpixel = true;
    content
}

/// A dimmed page and a box `top` px from its top, centred across
/// (`margin: top auto`, not transformed): returns the box.
pub fn overlay_top(ui: &mut Ui, root: NodeId, width: Length, top: f32) -> NodeId {
    let shade = ui.add_box(
        root,
        style! { position: absolute; left: 0; top: 0; right: 0; bottom: 0; display: flex; justify-content: center; align-items: flex-start; background: {[0.0, 0.0, 0.0, 0.8]}; },
    );
    ui.add_box(shade, style! { position: relative; display: flex; flex-direction: column; width: {width}; margin-top: {top}; background: white; color: black; flex-shrink: 0; })
}

/// `hsv` (each 0–1) as RGB.
pub fn hsv_to_rgb([h, s, v]: [f32; 3]) -> [f32; 3] {
    let h = (h.rem_euclid(1.0)) * 6.0;
    let i = h.floor();
    let f = h - i;
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i as u32 % 6 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// RGB as HSV (each 0–1; the hue 0 for greys).
pub fn rgb_to_hsv([r, g, b]: [f32; 3]) -> [f32; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let h = if d == 0.0 {
        0.0
    } else if max == r {
        ((g - b) / d).rem_euclid(6.0) / 6.0
    } else if max == g {
        ((b - r) / d + 2.0) / 6.0
    } else {
        ((r - g) / d + 4.0) / 6.0
    };
    [h, if max == 0.0 { 0.0 } else { d / max }, max]
}

/// A colour picker's parts, for input: the saturation/value field, the hue
/// strip.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorPickerNodes {
    pub popup: NodeId,
    pub field: NodeId,
    pub hue: NodeId,
}

/// The field's size, the hue strip's height.
const PICKER_W: f32 = 200.0;
const PICKER_H: f32 = 130.0;
const HUE_H: f32 = 12.0;

/// A colour picker popup (as browsers draw `<input type=color>`'s where
/// the system has none): saturation across and value down a field of the
/// hue, a hue strip, then a swatch and the hex value. Absolutely placed at
/// `(x, y)` in `parent`. `hsv` is the colour shown (each 0–1).
pub fn color_picker(ui: &mut Ui, parent: NodeId, x: f32, y: f32, hsv: [f32; 3], font: &Style) -> ColorPickerNodes {
    let popup = ui.add_box(
        parent,
        style! { position: absolute; left: {x}; top: {y}; display: flex; flex-direction: column; padding: 8px; background: white; border: 1px solid #767676; border-radius: 2px; },
    );
    // The field: the hue, white fading out across, black fading in down
    // (in strips: gradients run across).
    let field = ui.add_box(popup, style! { position: relative; width: {PICKER_W}; height: {PICKER_H}; flex-shrink: 0; overflow: hidden; });
    let pure = hsv_to_rgb([hsv[0], 1.0, 1.0]);
    ui.style_mut(field).background = Some([pure[0], pure[1], pure[2], 1.0]);
    let white = ui.add_box(field, style! { position: absolute; left: 0; top: 0; width: {PICKER_W}; height: {PICKER_H}; });
    ui.style_mut(white).background_gradient = vec![(0.0, [1.0, 1.0, 1.0, 1.0]), (1.0, [1.0, 1.0, 1.0, 0.0])];
    const STRIPS: usize = 65;
    let strip = PICKER_H / STRIPS as f32;
    for k in 0..STRIPS {
        let alpha = k as f32 / (STRIPS - 1) as f32;
        ui.add_box(field, style! { position: absolute; left: 0; top: {k as f32 * strip}; width: {PICKER_W}; height: {strip + 0.5}; background: {[0.0, 0.0, 0.0, alpha]}; });
    }
    // The colour's place: a ring.
    let (cx, cy) = (hsv[1] * PICKER_W, (1.0 - hsv[2]) * PICKER_H);
    ui.add_box(field, style! { position: absolute; left: {cx - 6.0}; top: {cy - 6.0}; width: 12px; height: 12px; border: 2px solid white; border-radius: 6px; box-sizing: border-box; });
    // The hue strip, with its place.
    let hue = ui.add_box(popup, style! { position: relative; width: {PICKER_W}; height: {HUE_H}; margin-top: 8px; flex-shrink: 0; border-radius: 6px; });
    ui.style_mut(hue).background_gradient = (0..=6).map(|k| {
        let c = hsv_to_rgb([k as f32 / 6.0, 1.0, 1.0]);
        (k as f32 / 6.0, [c[0], c[1], c[2], 1.0])
    }).collect();
    let hx = hsv[0] * PICKER_W;
    ui.add_box(hue, style! { position: absolute; left: {hx - 6.0}; top: 0; width: 12px; height: 12px; border: 2px solid white; border-radius: 6px; box-sizing: border-box; });
    // The swatch and the hex value.
    let row = ui.add_box(popup, style! { display: flex; align-items: center; margin-top: 8px; flex-shrink: 0; });
    let rgb = hsv_to_rgb(hsv);
    ui.add_box(row, style! { width: 24px; height: 16px; border: 1px solid #767676; background: {[rgb[0], rgb[1], rgb[2], 1.0]}; flex-shrink: 0; });
    let hex = format!("#{:02x}{:02x}{:02x}", (rgb[0] * 255.0).round() as u8, (rgb[1] * 255.0).round() as u8, (rgb[2] * 255.0).round() as u8);
    let label = ui.add_box(row, style! { ..font.clone(); margin-left: 8px; });
    ui.add_text(label, hex);
    ColorPickerNodes { popup, field, hue }
}

/// Saturation and value under `(x, y)` on a picker's field `r`, clamped.
pub fn color_picker_field(r: Rect, x: f32, y: f32) -> (f32, f32) {
    (((x - r.x) / r.width).clamp(0.0, 1.0), (1.0 - (y - r.y) / r.height).clamp(0.0, 1.0))
}

/// The hue under `x` on a picker's hue strip `r`, clamped.
pub fn color_picker_hue(r: Rect, x: f32) -> f32 {
    ((x - r.x) / r.width).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::super::FixedWidthText;
    use super::*;

    /// The controls say what they are: a button named by its label, a
    /// checkbox's state, a select's value, a slider's position as 0–100, a
    /// text box's value read once (not again as its text), and the focus
    /// ring not at all.
    #[test]
    fn controls_describe_themselves() {
        use super::super::a11y::Role;
        let mut ui = Ui::new();
        let root = ui.root();
        let font = crate::style! { font-family: Arial; font-size: 13px; color: black; };
        let mut text = FixedWidthText(7.0);
        let b = button(&mut ui, root, "Apply", &font, &mut text);
        focus_ring(&mut ui, b);
        let c = checkbox(&mut ui, root, true, 13.0, BORDER, Edges::ZERO);
        ui.semantics_mut(c).label = Some("Wireframe".into());
        select(&mut ui, root, "Linear", &["Linear", "Nearest"], &font, &mut text);
        range(&mut ui, root, 0.25, false);
        let field = ui.add_box(root, crate::style! { position: relative; width: 100px; height: 19px; });
        ui.semantics_mut(field).label = Some("Name".into());
        let view = TextView {
            value: "cube",
            placeholder: "untitled",
            placeholder_color: [0.5; 4],
            editing: None,
            caret_on: false,
            font: &font,
            x_offset: 2.0,
            selection: [0.8; 4],
        };
        text_contents(&mut ui, field, view, &mut text);
        ui.layout(600.0, 200.0, &mut text);
        let tree = ui.accessibility_tree();
        let got: Vec<(Role, String, Option<String>, Option<bool>)> =
            tree[1..].iter().map(|n| (n.role, n.name.clone(), n.value.clone(), n.checked)).collect();
        assert_eq!(
            got,
            vec![
                (Role::Button, "Apply".into(), None, None),
                (Role::Checkbox, "Wireframe".into(), None, Some(true)),
                (Role::ComboBox, String::new(), Some("Linear".into()), None),
                (Role::Slider, String::new(), None, None),
                (Role::TextBox, "Name".into(), Some("cube".into()), None),
            ]
        );
        assert_eq!(tree[4].range.map(|r| r.now), Some(25.0));
        assert_eq!(tree[5].description.as_deref(), Some("untitled"));
        assert_eq!(ui.focus_order().len(), 5);
    }

    #[test]
    fn controls_take_chromes_sizes() {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.set_style(root, crate::style! { display: flex; align-items: flex-start; });
        let font = crate::style! { font-family: Arial; font-size: 13px; color: black; };
        let mut text = FixedWidthText(7.0);
        let b = button(&mut ui, root, "Min-Max", &font, &mut text);
        let c = checkbox(&mut ui, root, false, 13.0, BORDER, Edges::ZERO);
        let r = range(&mut ui, root, 0.25, false);
        ui.layout(400.0, 50.0, &mut text);
        assert_eq!((ui.rect(b).width, ui.rect(c).width, ui.rect(r).width), (49.0 + 16.0, 13.0, 129.0));
        // The thumb's centre: 8 + 0.25 × 113, at a whole pixel.
        let run = ui.children(r)[1];
        let thumb = ui.children(run)[1];
        let t = ui.rect(thumb);
        assert_eq!(t.x + 8.0 - ui.rect(r).x, 8.0 + (0.25f32 * 113.0).round());
        assert_eq!(range_fraction(ui.rect(r), ui.rect(r).x + 8.0 + 0.25 * 113.0), 0.25);
    }

    #[test]
    fn colours_convert_and_pick() {
        for c in [[1.0, 0.0, 0.0], [0.2, 0.4, 0.6], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]] {
            let back = hsv_to_rgb(rgb_to_hsv(c));
            for k in 0..3 {
                assert!((back[k] - c[k]).abs() < 1e-5, "{c:?} {back:?}");
            }
        }
        assert_eq!(rgb_to_hsv([0.0, 1.0, 0.0]), [1.0 / 3.0, 1.0, 1.0]);
        let mut ui = Ui::new();
        let root = ui.root();
        let font = crate::style! { font-family: monospace; font-size: 13px; color: black; };
        let p = color_picker(&mut ui, root, 10.0, 20.0, [0.5, 0.25, 0.75], &font);
        let mut text = FixedWidthText(7.0);
        ui.layout(400.0, 400.0, &mut text);
        let f = ui.rect(p.field);
        assert_eq!((f.x, f.y, f.width, f.height), (19.0, 29.0, 200.0, 130.0));
        assert_eq!(color_picker_field(f, f.x + 50.0, f.y + 130.0), (0.25, 0.0));
        assert_eq!(color_picker_hue(ui.rect(p.hue), f.x + 300.0), 1.0);
    }

    #[test]
    fn text_contents_put_the_selection_under_the_text() {
        let mut ui = Ui::new();
        let root = ui.root();
        let node = ui.add_box(root, crate::style! { display: flex; position: relative; width: 100px; height: 17px; });
        let font = crate::style! { font-family: monospace; font-size: 13px; color: white; };
        let mut field = TextField::selected("50");
        field.anchor = 1;
        let view = TextView { value: "50", placeholder: "", placeholder_color: [0.5; 4], editing: Some(&field), caret_on: true, font: &font, x_offset: 2.0, selection: SELECTION };
        let mut text = FixedWidthText(7.0);
        text_contents(&mut ui, node, view, &mut text);
        ui.layout(100.0, 20.0, &mut text);
        let kids = ui.children(node).to_vec();
        assert_eq!(kids.len(), 2);
        assert_eq!((ui.rect(kids[0]).x, ui.rect(kids[0]).width), (9.0, 7.0));
    }
}
