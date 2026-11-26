//! More of the kit: the small controls and notices most screens are made of.
//! See [`super::kit`].

use super::kit::{button, hstack, icon, text, vstack, Edge, Font, HorizontalAlignment};
use super::El;
use crate::components::ButtonKind;
use crate::environment::Token;
use crate::{Role, Semantics};

/// What a pill or a notice is about, by colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tint {
    #[default]
    Neutral,
    Accent,
    Success,
    Warning,
    Destructive,
}

impl Tint {
    /// The theme colour it draws in.
    pub fn token(self) -> Token {
        match self {
            Tint::Neutral => Token::SecondaryText,
            Tint::Accent => Token::Accent,
            Tint::Success => Token::Success,
            Tint::Warning => Token::Warning,
            Tint::Destructive => Token::Destructive,
        }
    }

    /// `percent` of its colour over a surface, as CSS.
    fn wash(self, percent: u32) -> String {
        format!(
            "color-mix(in srgb, var(--{}) {percent}%, var(--surface))",
            self.token().css_name()
        )
    }
}

/// A word or two in a tinted pill: a state ("Installed"), a kind ("beta").
pub fn pill<S: 'static>(words: impl Into<String>, tint: Tint) -> El<S> {
    text(words)
        .css(&format!(
            "padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; \
             flex-shrink: 0; white-space: nowrap; background: {}",
            tint.wash(if tint == Tint::Neutral { 14 } else { 18 })
        ))
        // In the text's colour when tinted: every tint reads on its wash.
        .foreground(if tint == Tint::Neutral {
            Token::SecondaryText
        } else {
            Token::Text
        })
}

/// A notice in a tinted card, an icon before it: what went wrong, what to
/// mind, what to know. Said as it appears (an alert when `Destructive`).
pub fn callout<S: 'static>(words: impl Into<String>, tint: Tint) -> El<S> {
    let words = words.into();
    let mark = match tint {
        Tint::Success => r#"<circle cx="12" cy="12" r="10"/><path d="M8 12l3 3 5-6"/>"#,
        Tint::Warning | Tint::Destructive => {
            r#"<path d="M12 3 22 20H2L12 3z"/><path d="M12 10v4M12 17v.5"/>"#
        }
        Tint::Neutral | Tint::Accent => {
            r#"<circle cx="12" cy="12" r="10"/><path d="M12 11v5M12 7.5v.5"/>"#
        }
    };
    let role = if tint == Tint::Destructive {
        Role::Alert
    } else {
        Role::Status
    };
    hstack(
        10.0,
        [
            icon(mark, 16.0, 2.0)
                .foreground(match tint {
                    Tint::Accent => Token::AccentText,
                    t => t.token(),
                })
                .css("flex-shrink: 0; align-self: flex-start; margin: 1px 0 0 0"),
            text(words.clone())
                .css("font-size: 13px; flex-grow: 1")
                .foreground(Token::Text)
                .wraps(),
        ],
    )
    .css(&format!(
        "align-self: stretch; padding: 9px 12px; border-radius: 10px; border: 1px solid; \
         border-color: {}; background: {}",
        tint.wash(40),
        tint.wash(10)
    ))
    .semantics(Semantics::new(role).label(words))
}

/// A button drawn as an icon (SVG markup, as [`icon`]'s), named `label` to
/// assistive technology and under the pointer.
pub fn icon_button<S: 'static>(
    markup: &str,
    label: impl Into<String>,
    action: impl Fn(&mut S) + 'static,
) -> El<S> {
    let label = label.into();
    icon(markup, 16.0, 2.0)
        .css("padding: 5px; border-radius: 7px; cursor: pointer; flex-shrink: 0")
        .foreground(Token::SecondaryText)
        .hover("background: var(--hover)")
        .help(label.clone())
        .semantics(Semantics::new(Role::Button).label(label))
        .on_click(move |s: &mut S, _| action(s))
}

/// `words` in an outlined pill with a ✕ that removes it, named `remove`
/// ("Remove focus").
pub fn chip<S: 'static>(
    words: impl Into<String>,
    remove: impl Into<String>,
    action: impl Fn(&mut S) + 'static,
) -> El<S> {
    let remove = remove.into();
    hstack(
        2.0,
        [
            text(words).css("font-size: 12px; white-space: nowrap"),
            icon(r#"<path d="M18 6 6 18M6 6l12 12"/>"#, 12.0, 2.0)
                .css("padding: 6px; border-radius: 12px; cursor: pointer; flex-shrink: 0")
                .foreground(Token::SecondaryText)
                .hover("background: var(--hover)")
                .semantics(Semantics::new(Role::Button).label(remove))
                .on_click(move |s: &mut S, _| action(s)),
        ],
    )
    .css("padding: 0 2px 0 10px; border-radius: 13px; flex-shrink: 0")
    .border(Token::Border, 1.0)
}

/// `value` between − and + in one outlined control (SwiftUI's `Stepper`
/// with its value), the buttons named `names` (decrease, increase); a side
/// with no action is greyed out, as at an end of the range.
pub fn stepper<S: 'static>(
    value: impl Into<String>,
    names: (impl Into<String>, impl Into<String>),
    down: Option<Box<dyn Fn(&mut S)>>,
    up: Option<Box<dyn Fn(&mut S)>>,
) -> El<S> {
    let side = |markup: &str, name: String, action: Option<Box<dyn Fn(&mut S)>>| {
        let can = action.is_some();
        icon(markup, 14.0, 2.0)
            .css("padding: 6px 9px; flex-shrink: 0")
            .foreground(Token::Text)
            .semantics(Semantics::new(Role::Button).label(name))
            .when_some(action, |e, a| {
                e.css("cursor: pointer")
                    .hover("background: var(--hover)")
                    .on_click(move |s: &mut S, _| a(s))
            })
            .disabled(!can)
    };
    hstack(
        0.0,
        [
            side(r#"<path d="M5 12h14"/>"#, names.0.into(), down),
            text(value).css(
                "min-width: 56px; padding: 0 6px; text-align: center; font-size: 13px; \
                 font-weight: 600; white-space: nowrap",
            ),
            side(r#"<path d="M12 5v14M5 12h14"/>"#, names.1.into(), up),
        ],
    )
    .corner_radius(8.0)
    .border(Token::Border, 1.0)
    .css("overflow: hidden; flex-shrink: 0")
}

/// A box to tick beside its `label` (SwiftUI's checkbox toggle style).
pub fn checkbox<S: 'static>(
    label: impl Into<String>,
    on: bool,
    set: impl Fn(&mut S, bool) + 'static,
) -> El<S> {
    let label = label.into();
    let mark = El::div(
        "width: 18px; height: 18px; border-radius: 5px; flex-shrink: 0; display: flex; \
         align-items: center; justify-content: center",
    )
    // (an empty box is all border: one that can be seen)
    .border(
        if on {
            Token::Accent
        } else {
            Token::SecondaryText
        },
        1.0,
    )
    .when(on, |e| {
        e.background(Token::Accent)
            .child(icon(r#"<path d="M5 12l5 5 9-10"/>"#, 12.0, 3.0).foreground(Token::OnAccent))
    });
    hstack(10.0, [mark, text(label.clone()).wraps()])
        .css("cursor: pointer; min-height: 24px")
        .semantics(Semantics::new(Role::Checkbox).label(label).checked(on))
        .on_click(move |s: &mut S, _| set(s, !on))
}

/// The chosen one of `options` on a button that opens them all in a menu
/// (SwiftUI's `Picker` with the menu style), for more than a row of choices
/// holds. `label` names it; `set` gets the index chosen.
pub fn picker_menu<S: 'static>(
    label: impl Into<String>,
    options: Vec<String>,
    selected: Option<usize>,
    set: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let label = label.into();
    let now = selected
        .and_then(|i| options.get(i).cloned())
        .unwrap_or_else(|| "—".into());
    let id = format!("picker:{label}:{selected:?}:{}", options.join("\u{1f}"));
    let shown = now.clone();
    let set = std::rc::Rc::new(set);
    El::host(id, move |ui, input, _, node| {
        let t = ui.add_box(
            node,
            super::style_of(
                "font-size: 13px; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis",
            ),
        );
        ui.add_text(t, shown.clone());
        let mark = crate::svg::add_icon(
            ui,
            node,
            super::style_of("width: 14px; height: 14px; flex-shrink: 0"),
            r#"<path d="M6 9l6 6 6-6"/>"#,
            2.0,
        );
        ui.token_style(mark, &[("color", Token::SecondaryText)]);
        // As they are: the host is made anew when they change.
        let items: Vec<crate::menu::MenuItem> = options
            .iter()
            .enumerate()
            .map(|(i, o)| {
                let item = crate::menu::MenuItem::action(i.to_string(), o.clone());
                if selected == Some(i) {
                    item.shortcut("✓")
                } else {
                    item
                }
            })
            .collect();
        let set = set.clone();
        crate::menu::popover_menu(
            input,
            node,
            move |_| items.clone(),
            move |s: &mut S, _, id: &str| {
                if let Ok(i) = id.parse() {
                    set(s, i);
                }
            },
        );
    })
    .css(
        "display: flex; flex-direction: row; align-items: center; gap: 6px; padding: 5px 8px 5px 12px; \
         border-radius: 8px; cursor: pointer; min-height: 18px; max-width: 100%",
    )
    .border(Token::Border, 1.0)
    .hover("background: var(--hover)")
    .semantics(Semantics::new(Role::ComboBox).label(label).value(now))
}

/// A settings row: its `title` over what it does, its `control` at the end
/// of the line, or under them where the line is too narrow for both.
pub fn setting_row<S: 'static>(
    title: impl Into<String>,
    description: impl Into<String>,
    control: El<S>,
) -> El<S> {
    let description = description.into();
    let mut words = vec![text(title).css("font-weight: 500").wraps()];
    if !description.is_empty() {
        words.push(
            text(description)
                .css("font-size: 12px")
                .foreground(Token::SecondaryText)
                .wraps(),
        );
    }
    hstack(
        16.0,
        [
            vstack(2.0, words)
                .alignment(HorizontalAlignment::Leading)
                .fill_width()
                .css("flex-basis: 240px"),
            control,
        ],
    )
    .css("flex-wrap: wrap; row-gap: 8px; align-self: stretch")
    .padding_edges(Edge::Vertical, 2.0)
    .as_row()
}

/// A figure over what it counts.
pub fn stat<S: 'static>(value: impl Into<String>, caption: impl Into<String>) -> El<S> {
    vstack(
        2.0,
        [
            text(value).css("font-size: 22px; font-weight: 600; white-space: nowrap"),
            text(caption.into())
                .css("font-size: 11px")
                .foreground(Token::SecondaryText),
        ],
    )
}

/// A person or a speaker as a disc of `initials`.
pub fn avatar<S: 'static>(initials: impl Into<String>, size: f32) -> El<S> {
    text(initials)
        .css(&format!(
            "min-width: {size}px; padding: 0 {pad}px; aspect-ratio: 1; box-sizing: border-box; \
             border-radius: 999px; font-size: {f}px; font-weight: 700; display: flex; \
             align-items: center; justify-content: center; flex-shrink: 0; white-space: nowrap",
            f = (size * 0.4).round(),
            pad = (size * 0.12).round()
        ))
        .background(Token::Accent)
        .foreground(Token::OnAccent)
        .accessibility_label("")
}

/// Where a list has nothing yet: an icon (SVG markup), a `title`, a line
/// on how to fill it, and the button that does (`action`: its words, what
/// it does). [`content_unavailable`](super::content_unavailable) with a
/// drawn icon and a way on.
pub fn empty_state<S: 'static>(
    markup: &str,
    title: impl Into<String>,
    hint: impl Into<String>,
    action: Option<(String, Box<dyn Fn(&mut S)>)>,
) -> El<S> {
    let (title, hint) = (title.into(), hint.into());
    let mut parts = vec![
        icon(markup, 28.0, 1.6)
            .css("padding: 14px; border-radius: 999px; background: color-mix(in srgb, var(--secondary-text) 12%, var(--surface))")
            .foreground(Token::SecondaryText),
        text(title.clone()).font(Font::Headline),
    ];
    if !hint.is_empty() {
        parts.push(
            text(hint.clone())
                .css(
                    "font-size: 13px; text-align: center; white-space: normal; align-self: stretch",
                )
                .foreground(Token::SecondaryText),
        );
    }
    if let Some((words, act)) = action {
        parts.push(button(words, move |s: &mut S| act(s)).button_style(ButtonKind::Primary));
    }
    vstack(8.0, parts)
        .css("align-self: stretch; align-items: center; min-width: 0")
        .padding_edges(Edge::Vertical, 18.0)
}

/// A grey block where something is still loading, `width` (`100%`, `120px`)
/// by `height` pixels; hidden from assistive technology.
pub fn skeleton<S: 'static>(width: &str, height: f32) -> El<S> {
    El::div(&format!(
        "width: {width}; height: {height}px; border-radius: {}px; max-width: 100%",
        (height / 2.0).min(8.0)
    ))
    .background(Token::Separator)
    .semantics(Semantics::new(Role::Image).hidden())
}

/// The empty ring of a gauge `size` pixels across: an outline.
fn ring_rail<S: 'static>(size: f32) -> Vec<El<S>> {
    // Its stroke is 1.6 of the icon's 24.
    let px = 24.0 / size.max(1.0);
    let (edge, middle) = crate::pickers::ring_strokes(1.6 / px);
    [(Some(edge), Token::Track), (middle, Token::Surface)]
        .into_iter()
        .filter_map(|(width, token)| {
            let ring = r#"<circle cx="12" cy="12" r="10"/>"#;
            Some(icon(ring, size, width? * px).foreground(token))
        })
        .collect()
}

/// How far along `value` (0 to 1) is as a ring in the accent, `shown` in its
/// middle (SwiftUI's `Gauge`, the circular style): a countdown, a goal.
/// `label` names it.
pub fn gauge<S: 'static>(
    value: f32,
    label: impl Into<String>,
    shown: impl Into<String>,
    size: f32,
) -> El<S> {
    let value = value.clamp(0.0, 1.0);
    let shown = shown.into();
    // A 100 x 100 box scaled to the icon's 24: the ring's radius is 10.
    let (c, r) = (12.0f32, 10.0f32);
    let arc = if value >= 0.999 {
        format!(r#"<circle cx="{c}" cy="{c}" r="{r}"/>"#)
    } else {
        let a = value * std::f32::consts::TAU;
        let (x, y) = (c + r * a.sin(), c - r * a.cos());
        let large = u8::from(value > 0.5);
        format!(
            r#"<path d="M{c} {top} A{r} {r} 0 {large} 1 {x:.3} {y:.3}"/>"#,
            top = c - r
        )
    };
    let mut layers = ring_rail(size);
    if value > 0.001 {
        layers.push(icon(&arc, size, 1.6).foreground(Token::AccentText));
    }
    layers.push(text(shown.clone()).css(&format!(
        "font-size: {}px; font-weight: 700; white-space: nowrap",
        (size * 0.2).round()
    )));
    super::kit::zstack(layers)
        .css(&format!(
            "width: {size}px; height: {size}px; flex-shrink: 0"
        ))
        .semantics(
            Semantics::new(Role::ProgressBar)
                .label(label)
                .value(shown)
                .range(0.0, 1.0, value),
        )
}

/// A [`gauge`] whose ring runs through `g` from its start.
pub fn gauge_gradient<S: 'static>(
    value: f32,
    label: impl Into<String>,
    shown: impl Into<String>,
    size: f32,
    g: &crate::fill::Gradient,
) -> El<S> {
    let value = value.clamp(0.0, 1.0);
    let shown = shown.into();
    let (c, r) = (12.0f32, 10.0f32);
    let mut layers = ring_rail(size);
    // Short arcs, each in its colour, lapping a little.
    let parts = ((value * 48.0).ceil() as usize).min(48);
    let point = |t: f32| {
        let a = t * std::f32::consts::TAU;
        (c + r * a.sin(), c - r * a.cos())
    };
    for i in 0..parts {
        let t0 = i as f32 / 48.0;
        let t1 = ((i as f32 + 1.15) / 48.0).min(value);
        let ((x0, y0), (x1, y1)) = (point(t0), point(t1));
        layers.push(
            icon(
                &format!(r#"<path d="M{x0:.3} {y0:.3} A{r} {r} 0 0 1 {x1:.3} {y1:.3}"/>"#),
                size,
                1.6,
            )
            .css(&format!("color: {}", g.css_at(t0 / value.max(1e-3)))),
        );
    }
    layers.push(text(shown.clone()).css(&format!(
        "font-size: {}px; font-weight: 700; white-space: nowrap",
        (size * 0.2).round()
    )));
    super::kit::zstack(layers)
        .css(&format!(
            "width: {size}px; height: {size}px; flex-shrink: 0"
        ))
        .semantics(
            Semantics::new(Role::ProgressBar)
                .label(label)
                .value(shown)
                .range(0.0, 1.0, value),
        )
}

/// `rows` of cells in `columns` lined up with each other, each as wide as
/// its widest cell (SwiftUI's `Grid`); the first takes what is over.
pub fn grid<S: 'static>(
    columns: usize,
    spacing: (f32, f32),
    cells: impl IntoIterator<Item = El<S>>,
) -> El<S> {
    let rest = " auto".repeat(columns.max(1) - 1);
    El::div(&format!(
        "display: grid; grid-template-columns: 1fr{rest}; column-gap: {}px; row-gap: {}px; \
         align-items: center; align-self: stretch",
        spacing.0, spacing.1
    ))
    .children(cells)
}

/// `buttons` joined in one outlined control (SwiftUI's `ControlGroup`).
pub fn control_group<S: 'static>(buttons: impl IntoIterator<Item = El<S>>) -> El<S> {
    hstack(0.0, buttons)
        .corner_radius(8.0)
        .border(Token::Border, 1.0)
        .css("overflow: hidden; flex-shrink: 0")
        .semantics(Semantics::new(Role::Group))
}

/// A button of a toolbar: an icon (SVG markup), raised while it is `on`
/// (a panel shown, a mode), named `label` to assistive technology and
/// under the pointer (SwiftUI's toolbar `Toggle`).
pub fn toolbar_button<S: 'static>(
    markup: &str,
    label: impl Into<String>,
    on: bool,
    action: impl Fn(&mut S) + 'static,
) -> El<S> {
    icon_button(markup, label, action)
        .css("padding: 7px")
        .when(on, |e| e.background(Token::Hover).foreground(Token::Text))
}

/// A shortcut as the keys it is, each on a cap (`["⌘", "⇧", "L"]`); read
/// out as one.
pub fn key_caps<S: 'static>(keys: impl IntoIterator<Item = impl Into<String>>) -> El<S> {
    let keys: Vec<String> = keys.into_iter().map(Into::into).collect();
    let name = keys.join(" ");
    hstack(
        3.0,
        keys.into_iter().map(|k| {
            // The Mac's modifiers drawn: few faces have them, none alike.
            let drawn = match k.as_str() {
                "⌘" => Some(r#"<path d="M15 6v12a3 3 0 1 0 3-3H6a3 3 0 1 0 3 3V6a3 3 0 1 0-3 3h12a3 3 0 1 0-3-3"/>"#),
                "⇧" => Some(r#"<path d="M9 18v-6H5l7-7 7 7h-4v6z"/>"#),
                "⌥" => Some(r#"<path d="M3 4h6l6 16h6M14 4h7"/>"#),
                "⌃" => Some(r#"<path d="M5 13l7-7 7 7"/>"#),
                _ => None,
            };
            let key = match drawn {
                Some(d) => icon(d, 12.0, 2.2),
                None => text(k).css("font-size: 12px"),
            };
            key.css("min-width: 12px; text-align: center")
                .foreground(Token::Text)
                .modify(|key| {
                    El::div("display: flex; align-items: center; justify-content: center; min-height: 16px; padding: 2px 5px; border-radius: 5px")
                        .child(key)
                })
                .background(Token::Separator)
                .border(Token::Border, 1.0)
        }),
    )
    // Keys are in the order they are pressed, whichever way words read.
    .css("flex-shrink: 0; direction: ltr")
    .semantics(Semantics::new(Role::Image).label(name))
}

/// A [`stat`] on a tile of its own, for a grid of figures.
pub fn stat_tile<S: 'static>(value: impl Into<String>, caption: impl Into<String>) -> El<S> {
    let value = value.into();
    // Words are smaller than figures, to stay whole on a narrow tile.
    let size = if value.chars().count() > 9 { 14 } else { 18 };
    vstack(
        2.0,
        [
            text(value)
                .css(&format!("font-size: {size}px; font-weight: 700; min-width: 0"))
                .wraps(),
            text(caption)
                .css("font-size: 12px")
                .foreground(Token::SecondaryText)
                .wraps(),
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    .css("padding: 10px 12px; border-radius: 10px; min-width: 0")
    .border(Token::Border, 1.0)
}

/// What `lines` of text will take, while they load (SwiftUI's
/// `.redacted(reason: .placeholder)`): `said` is what assistive technology
/// is told ("Loading…").
pub fn redacted<S: 'static>(lines: usize, said: impl Into<String>) -> El<S> {
    let widths = ["92%", "78%", "86%", "64%"];
    vstack(
        10.0,
        (0..lines).map(|i| skeleton(widths[i % widths.len()], 12.0)),
    )
    .alignment(HorizontalAlignment::Leading)
    .css("align-self: stretch")
    .padding_edges(Edge::Vertical, 6.0)
    .semantics(Semantics::new(Role::Status).label(said).live())
}

/// An [`inset_group`](super::inset_group) with a note under its card (a SwiftUI section's
/// footer): what the settings in it come to.
pub fn inset_group_with_footer<S: 'static>(
    title: impl Into<String>,
    children: impl IntoIterator<Item = El<S>>,
    footer: impl Into<String>,
) -> El<S> {
    vstack(
        6.0,
        [
            super::kit::inset_group(title, children),
            text(footer)
                .css("font-size: 12px; padding: 0 4px")
                .foreground(Token::SecondaryText)
                .wraps(),
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    .css("align-self: stretch")
}

/// An icon (SVG markup) before its `title` (SwiftUI's `Label`).
pub fn icon_label<S: 'static>(markup: &str, title: impl Into<String>) -> El<S> {
    hstack(
        8.0,
        [icon(markup, 16.0, 1.8).css("flex-shrink: 0"), text(title)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::{Event, Input, PointerButton};
    use crate::view::View;
    use crate::{FixedWidthText, Ui};

    #[derive(Default)]
    struct S {
        n: i32,
        on: bool,
        picked: Option<usize>,
    }

    struct H {
        ui: Ui,
        input: Input<S>,
        s: S,
        width: f32,
    }

    impl H {
        fn new(width: f32, els: Vec<El<S>>) -> H {
            let mut h = H {
                ui: Ui::new(),
                input: Input::new(),
                s: S::default(),
                width,
            };
            let root = h.ui.root();
            let mut v = View::new(&mut h.ui, root, "display: flex; flex-direction: column");
            v.update_with(&mut h.ui, &mut h.input, &h.s, els);
            h.layout();
            h
        }
        fn layout(&mut self) {
            self.ui.layout(self.width, 400.0, &mut FixedWidthText(7.0));
        }
        fn find(&self, role: Role, name: &str) -> crate::AccessNode {
            self.ui
                .accessibility_tree()
                .into_iter()
                .find(|n| n.role == role && n.name == name)
                .unwrap_or_else(|| panic!("no {role:?} {name:?}"))
        }
        fn press(&mut self, role: Role, name: &str) {
            let r = self.ui.rect(self.find(role, name).id);
            let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
            let button = PointerButton::Primary;
            for e in [
                Event::PointerDown { x, y, button },
                Event::PointerUp { x, y, button },
            ] {
                self.input.dispatch(&mut self.ui, &mut self.s, e);
            }
            self.layout();
        }
    }

    #[test]
    fn a_stepper_steps_and_stops_at_an_end() {
        let mut h = H::new(
            300.0,
            vec![stepper(
                "5 min",
                ("Less", "More"),
                None,
                Some(Box::new(|s: &mut S| s.n += 1)),
            )],
        );
        assert!(h.find(Role::Button, "Less").disabled);
        h.press(Role::Button, "Less");
        assert_eq!(h.s.n, 0);
        h.press(Role::Button, "More");
        assert_eq!(h.s.n, 1);
    }

    #[test]
    fn a_checkbox_says_and_flips_its_state() {
        let mut h = H::new(
            300.0,
            vec![checkbox("Keep", true, |s: &mut S, on| s.on = !on)],
        );
        assert_eq!(h.find(Role::Checkbox, "Keep").checked, Some(true));
        h.press(Role::Checkbox, "Keep");
        assert!(h.s.on, "asked to be off");
    }

    #[test]
    fn a_chip_is_removed_by_its_cross_alone() {
        let mut h = H::new(
            300.0,
            vec![chip("focus", "Remove focus", |s: &mut S| s.n += 1)],
        );
        h.press(Role::Button, "Remove focus");
        assert_eq!(h.s.n, 1);
    }

    #[test]
    fn a_picker_menu_shows_the_choice_and_opens_the_rest() {
        let options: Vec<String> = ["One", "Two", "Three"].map(String::from).into();
        let mut h = H::new(
            300.0,
            vec![picker_menu("Count", options, Some(1), |s: &mut S, i| {
                s.picked = Some(i)
            })],
        );
        assert_eq!(
            h.find(Role::ComboBox, "Count").value.as_deref(),
            Some("Two")
        );
        h.press(Role::ComboBox, "Count");
        h.press(Role::MenuItem, "Three");
        assert_eq!(h.s.picked, Some(2));
    }

    #[test]
    fn a_setting_row_wraps_its_control_where_it_is_narrow() {
        let row = || {
            vec![setting_row(
                "Interval",
                "How often",
                stepper("5", ("Less", "More"), None, Some(Box::new(|_: &mut S| {}))),
            )]
        };
        let wide = H::new(500.0, row());
        let r = wide.ui.rect(wide.find(Role::Button, "More").id);
        assert!(r.x > 400.0 && r.y < 30.0, "{r:?}");
        let narrow = H::new(240.0, row());
        let r = narrow.ui.rect(narrow.find(Role::Button, "More").id);
        assert!(r.x < 200.0 && r.y > 40.0, "{r:?}");
    }

    #[test]
    fn notices_are_said() {
        let h = H::new(
            300.0,
            vec![
                callout("It broke", Tint::Destructive),
                callout("Mind this", Tint::Warning),
            ],
        );
        h.find(Role::Alert, "It broke");
        h.find(Role::Status, "Mind this");
    }

    #[test]
    fn an_empty_state_offers_the_way_on() {
        let mut h = H::new(
            400.0,
            vec![empty_state(
                r#"<path d="M4 4h16"/>"#,
                "Nothing yet",
                "Add one",
                Some(("Add".into(), Box::new(|s: &mut S| s.n = 9))),
            )],
        );
        h.press(Role::Button, "Add");
        assert_eq!(h.s.n, 9);
    }

    #[test]
    fn a_gauge_says_how_far() {
        for v in [0.0, 0.3, 0.8, 1.0, 7.0] {
            let h = H::new(300.0, vec![gauge(v, "Goal", format!("{v}"), 120.0)]);
            let n = h.find(Role::ProgressBar, "Goal");
            assert_eq!(n.range.map(|r| r.now), Some(v.min(1.0)));
            assert_eq!(h.ui.rect(n.id).width, 120.0);
        }
    }

    #[test]
    fn a_grids_columns_line_up() {
        let h = H::new(
            400.0,
            vec![grid(
                2,
                (10.0, 4.0),
                ["a", "1", "bb", "22222"].map(|t| text::<S>(t)),
            )],
        );
        let x = |t: &str| h.ui.rect(h.find(Role::StaticText, t).id).x;
        assert_eq!(x("1"), x("22222"));
        assert_eq!(x("a"), x("bb"));
        assert!(x("1") > 300.0);
    }

    #[test]
    fn what_loads_is_said_once() {
        let h = H::new(300.0, vec![redacted(3, "Loading…")]);
        h.find(Role::Status, "Loading…");
        assert_eq!(
            h.ui.accessibility_tree()
                .iter()
                .filter(|n| n.role == Role::Image)
                .count(),
            0
        );
    }

    #[test]
    fn what_is_at_a_width_shows_from_it_on_and_survives_updates() {
        let els = |colour| {
            vec![text::<S>("Wide")
                .css("display: none")
                .foreground(colour)
                .at_width(300.0, "display: block")]
        };
        let mut ui = Ui::new();
        let (mut input, s) = (Input::new(), S::default());
        let root = ui.root();
        let mut v = View::new(
            &mut ui,
            root,
            "display: flex; flex-direction: column; align-self: stretch; width: 100%",
        );
        let shown = |ui: &Ui| {
            ui.accessibility_tree()
                .iter()
                .any(|n| n.name == "Wide" && ui.rect(n.id).width > 0.0)
        };
        for colour in [Token::Text, Token::Accent] {
            v.update_with(&mut ui, &mut input, &s, els(colour));
            ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
            assert!(!shown(&ui), "narrow");
            ui.layout(400.0, 100.0, &mut FixedWidthText(7.0));
            assert!(shown(&ui), "wide");
        }
    }

    #[test]
    fn a_menu_closes_when_its_button_goes() {
        let options: Vec<String> = ["One", "Two"].map(String::from).into();
        let mut ui = Ui::new();
        let (mut input, mut s) = (Input::new(), S::default());
        let root = ui.root();
        let mut v = View::new(&mut ui, root, "display: flex");
        v.update_with(
            &mut ui,
            &mut input,
            &s,
            vec![picker_menu("Count", options, Some(0), |_: &mut S, _| {})],
        );
        ui.layout(300.0, 300.0, &mut FixedWidthText(7.0));
        let open = |ui: &Ui| ui.accessibility_tree().iter().any(|n| n.role == Role::Menu);
        let id = ui
            .accessibility_tree()
            .iter()
            .find(|n| n.role == Role::ComboBox)
            .unwrap()
            .id;
        let r = ui.rect(id);
        let (x, y) = (r.x + 5.0, r.y + 5.0);
        let button = PointerButton::Primary;
        input.dispatch(&mut ui, &mut s, Event::PointerDown { x, y, button });
        input.dispatch(&mut ui, &mut s, Event::PointerUp { x, y, button });
        assert!(open(&ui));
        v.update_with(&mut ui, &mut input, &s, vec![]);
        input.tick(&mut ui, &mut s, 1.0);
        assert!(!open(&ui));
    }
}
