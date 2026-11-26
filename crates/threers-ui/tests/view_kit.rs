//! The SwiftUI-style views and modifiers: laid out as their names say,
//! worked by pointer and keyboard, and read as assistive technology reads
//! them.

use threers_ui::a11y::AccessNode;
use threers_ui::binding::Binding;
use threers_ui::components::ButtonKind;
use threers_ui::environment::Token;
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::view::{
    adaptive_grid, button, confirmation, content_unavailable, disclosure_group, divider, for_each,
    form, group_box, hero, hstack, inset_group, labeled_content, link, list, live_text, menu,
    picker, progress_view, section, segmented, slider, spacer, text, text_field, title_bar, toggle,
    toggle_with, vstack, zstack, El, Font, HorizontalAlignment, View,
};
use threers_ui::{FixedWidthText, NodeId, Role, Ui};

#[derive(Default)]
struct App {
    on: bool,
    saved: u32,
    followed: bool,
    name: String,
    level: f32,
}

struct Harness {
    ui: Ui,
    input: Input<App>,
    view: View<App>,
    app: App,
}

impl Harness {
    fn new() -> Self {
        let mut ui = Ui::new();
        let root = ui.root();
        let view = View::new(
            &mut ui,
            root,
            "display: flex; flex-direction: column; width: 400px",
        );
        Self {
            ui,
            input: Input::new(),
            view,
            app: App::default(),
        }
    }

    fn show(&mut self, body: impl Fn(&App) -> Vec<El<App>>) {
        let els = body(&self.app);
        // As an app does: hosts (a menu, a field) are built too.
        self.view
            .update_with(&mut self.ui, &mut self.input, &self.app, els);
        self.ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
    }

    fn named(&self, name: &str) -> AccessNode {
        self.ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == name)
            .unwrap_or_else(|| panic!("no {name:?}"))
    }

    fn with_role(&self, role: Role) -> Vec<AccessNode> {
        self.ui
            .accessibility_tree()
            .into_iter()
            .filter(|a| a.role == role)
            .collect()
    }

    fn tap(&mut self, id: NodeId) {
        let r = self.ui.rect(id);
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        let b = PointerButton::Primary;
        for e in [
            Event::PointerMove { x, y },
            Event::PointerDown { x, y, button: b },
            Event::PointerUp { x, y, button: b },
        ] {
            self.input.dispatch(&mut self.ui, &mut self.app, e);
        }
    }
}

/// A spacer pushes what follows to the end of its row; a column puts its
/// children `spacing` apart.
#[test]
fn stacks_and_spacers_lay_out_as_named() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![vstack(
            10.0,
            [
                hstack(
                    8.0,
                    [text("Left"), spacer(), button("Go", |_: &mut App| {})],
                )
                .css("align-self: stretch"),
                text("Below"),
            ],
        )
        .alignment(HorizontalAlignment::Leading)]
    });
    let go = h.ui.rect(h.named("Go").id);
    assert!(
        (go.x + go.width - 400.0).abs() < 0.5,
        "at the end of the row: {go:?}"
    );
    let row = h.ui.parent(h.named("Go").id).unwrap();
    let column = h.ui.parent(row).unwrap();
    let (row, below) = (h.ui.rect(row), h.ui.rect(h.ui.children(column)[1]));
    assert_eq!(below.y - (row.y + row.height), 10.0, "spacing apart");
    assert_eq!(below.x, 0.0, "leading");
}

/// A z-stack puts its children over one another, each centred.
#[test]
fn a_zstack_overlays_and_centres() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![zstack([
            El::div("width: 100px; height: 60px").accessibility_label("back"),
            El::div("width: 20px; height: 10px").accessibility_label("front"),
        ])]
    });
    let (back, front) = (
        h.ui.rect(h.named("back").id),
        h.ui.rect(h.named("front").id),
    );
    let centre = |r: threers_ui::Rect| (r.x + r.width / 2.0, r.y + r.height / 2.0);
    assert_eq!(centre(back), centre(front));
}

/// A toggle is one switch: pressed anywhere, it gives the other value, and
/// says the new one when drawn again.
#[test]
fn a_toggle_switches_and_says_so() {
    let mut h = Harness::new();
    let body = |a: &App| vec![toggle("Sounds", a.on, |a: &mut App, on| a.on = on)];
    h.show(body);
    let s = h.named("Sounds");
    assert_eq!(
        (s.role, s.checked, s.focusable),
        (Role::Switch, Some(false), true)
    );
    h.tap(s.id);
    assert!(h.app.on);
    h.show(body);
    assert_eq!(h.named("Sounds").checked, Some(true));
    let sounds = h.named("Sounds").id;
    h.input.set_focus(&mut h.ui, &mut h.app, Some(sounds));
    h.input
        .simulate_keystrokes(&mut h.ui, &mut h.app, "space")
        .unwrap();
    assert!(!h.app.on, "Space too");
}

/// A button runs its action by pointer and keys; disabled, it does not,
/// and Tab skips it. A link follows.
#[test]
fn buttons_press_and_disabled_ones_do_not() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![
            button("Save", |a: &mut App| a.saved += 1).button_style(ButtonKind::Primary),
            button("Delete", |a: &mut App| a.saved += 100).disabled(true),
            link("Docs", |a: &mut App| a.followed = true),
        ]
    });
    let save = h.named("Save");
    assert_eq!(
        h.ui.style(save.id).background,
        Some(h.ui.token(Token::Accent)),
        "the style"
    );
    h.tap(save.id);
    h.input.set_focus(&mut h.ui, &mut h.app, Some(save.id));
    h.input
        .simulate_keystrokes(&mut h.ui, &mut h.app, "enter")
        .unwrap();
    assert_eq!(h.app.saved, 2);
    let delete = h.named("Delete");
    assert!(!delete.focusable);
    h.tap(delete.id);
    assert_eq!(h.app.saved, 2, "disabled");
    let docs = h.named("Docs");
    assert_eq!(docs.role, Role::Link);
    h.tap(docs.id);
    assert!(h.app.followed);
}

/// The modifiers style the box: font, padding, colours by token.
#[test]
fn modifiers_style_what_they_modify() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![text("Title")
            .font(Font::Title)
            .padding(8.0)
            .foreground(Token::SecondaryText)
            .accessibility_label("t")]
    });
    let t = h.named("t").id;
    let s = h.ui.style(t);
    assert_eq!(s.font_size, Font::Title.size_and_weight().0);
    assert_eq!(s.padding.left, 8.0);
    assert_eq!(s.color, h.ui.token(Token::SecondaryText));
}

/// A form of sections reads as headed groups; a list as a list of items;
/// dividers as separators; progress and "nothing here" as such.
#[test]
fn forms_lists_and_states_read_as_they_are() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![
            form([section(
                "Account",
                [
                    labeled_content("Name", "Ada"),
                    labeled_content("Plan", "Free"),
                ],
            )]),
            list([text("One"), text("Two"), text("Three")]),
            divider(),
            progress_view(Some(0.25)),
            content_unavailable("🔍", "No results", "Try another search."),
        ]
    });
    assert_eq!(h.with_role(Role::Heading(2))[0].name, "Account");
    assert_eq!(h.with_role(Role::List).len(), 1);
    assert_eq!(h.with_role(Role::ListItem).len(), 3);
    // Between two values, two between three items, and one alone.
    assert_eq!(h.with_role(Role::Separator).len(), 1 + 2 + 1);
    assert_eq!(
        h.with_role(Role::ProgressBar)[0].value.as_deref(),
        Some("25%")
    );
    assert_eq!(
        h.with_role(Role::Status)[0].name,
        "No results. Try another search."
    );
    let tree = h.ui.text_content(h.view.root());
    assert!(tree.contains("Ada") && tree.contains("Free"), "{tree}");
}

/// A toggle's label can be any view: named as given, pressed anywhere. A
/// picker is a radio group, the chosen one checked, a press choosing.
#[test]
fn a_described_toggle_and_a_picker() {
    let mut h = Harness::new();
    let body = |a: &App| {
        vec![
            toggle_with(
                "Sounds",
                vstack(2.0, [text("Sounds"), text("Play a sound on events.")]),
                a.on,
                |a: &mut App, on| a.on = on,
            ),
            picker(
                ["Daily", "Weekly", "Never"],
                Some(a.saved as usize),
                |a: &mut App, i| a.saved = i as u32,
            ),
        ]
    };
    h.show(body);
    let s = h.named("Sounds");
    assert_eq!((s.role, s.checked), (Role::Switch, Some(false)));
    h.tap(s.id);
    assert!(h.app.on);
    assert_eq!(h.with_role(Role::RadioGroup).len(), 1);
    let checked: Vec<_> = h.with_role(Role::Radio).iter().map(|r| r.checked).collect();
    assert_eq!(checked, [Some(true), Some(false), Some(false)]);
    h.tap(h.named("Never").id);
    assert_eq!(h.app.saved, 2);
    h.show(body);
    assert_eq!(h.named("Never").checked, Some(true));
}

/// Wrapping text keeps to its stack's width, even in a leading-aligned
/// column and beside a switch: it goes onto more lines, not past the edge.
#[test]
fn wrapping_text_keeps_to_the_width_it_is_given() {
    let mut h = Harness::new();
    let long = "word ".repeat(40);
    let l2 = long.clone();
    h.show(move |a: &App| {
        vec![
            vstack(4.0, [text(l2.clone()).wraps().accessibility_label("para")])
                .alignment(HorizontalAlignment::Leading),
            toggle_with(
                "Sounds",
                vstack(2.0, [text("Sounds"), text(l2.clone()).wraps()])
                    .alignment(HorizontalAlignment::Leading),
                a.on,
                |a: &mut App, on| a.on = on,
            ),
        ]
    });
    let _ = long;
    let para = h.ui.rect(h.named("para").id);
    assert!(para.x + para.width <= 400.0 + 0.5, "inside: {para:?}");
    assert!(para.height > 20.0, "on more lines: {para:?}");
    // The switch's track stays inside, at the end.
    let row = h.named("Sounds").id;
    let track = *h.ui.children(row).last().unwrap();
    let r = h.ui.rect(track);
    assert!(r.x + r.width <= 400.0 + 0.5 && r.width > 30.0, "{r:?}");
}

/// A fixed colour takes the place of a token on the same property.
#[test]
fn a_fixed_colour_replaces_a_token() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![group_box("Warning", [text("Stale")])
            .fixed_color("border-color", "#f59e0b")
            .accessibility_label("box")]
    });
    let b = h.ui.style(h.named("box").id).border_color;
    assert!(
        (b[0] - 0xf5 as f32 / 255.0).abs() < 0.01 && (b[2] - 0x0b as f32 / 255.0).abs() < 0.01,
        "{b:?}"
    );
}

/// A text field writes what is typed to its binding, keeps its node, text
/// and focus when the view updates round it, and submits on Enter.
#[test]
fn a_text_field_keeps_what_is_typed_across_updates() {
    let mut h = Harness::new();
    let body = |a: &App| {
        vec![
            text(format!("Hello, {}", a.name)),
            text_field(
                "Name",
                Binding::new(|a: &App| a.name.clone(), |a: &mut App, v| a.name = v),
            )
            .on_submit(|a: &mut App| a.saved += 1),
        ]
    };
    let show = |h: &mut Harness| {
        let els = body(&h.app);
        h.view.update_with(&mut h.ui, &mut h.input, &h.app, els);
        h.ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
    };
    show(&mut h);
    let field = h.named("Name").id;
    h.tap(field);
    h.input.type_text(&mut h.ui, &mut h.app, "Ada");
    assert_eq!(h.app.name, "Ada");
    show(&mut h);
    assert_eq!(h.named("Name").id, field, "the same field, not built again");
    assert!(
        h.ui.text_content(h.view.root()).contains("Hello, Ada"),
        "the rest updated"
    );
    h.input.type_text(&mut h.ui, &mut h.app, " L");
    assert_eq!(h.app.name, "Ada L", "still focused, the caret at the end");
    h.input
        .simulate_keystrokes(&mut h.ui, &mut h.app, "enter")
        .unwrap();
    assert_eq!(h.app.saved, 1, "submitted");
}

/// A slider sets its binding by keys; text beside it follows the value
/// without the view updating.
#[test]
fn a_slider_and_live_text_follow_the_value() {
    let mut h = Harness::new();
    let els = vec![
        slider(
            "Level",
            Binding::new(|a: &App| a.level, |a: &mut App, v| a.level = v),
            0.0..=10.0,
            1.0,
        ),
        live_text("level", |a: &App| format!("{:.0}", a.level)).accessibility_label("shown"),
    ];
    h.view.update_with(&mut h.ui, &mut h.input, &h.app, els);
    h.ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
    let level = h.with_role(Role::Slider)[0].id;
    h.input.set_focus(&mut h.ui, &mut h.app, Some(level));
    h.input
        .simulate_keystrokes(&mut h.ui, &mut h.app, "right right right")
        .unwrap();
    assert_eq!(h.app.level, 3.0);
    assert_eq!(
        h.ui.text_content(h.named("shown").id),
        "3",
        "followed, the view not updated"
    );
}

/// A disclosure group shows its content only while expanded, says so, and
/// a press asks for the other state.
#[test]
fn a_disclosure_group_opens_and_closes() {
    let mut h = Harness::new();
    let body = |a: &App| {
        vec![disclosure_group(
            "Advanced",
            a.on,
            |a: &mut App, open| a.on = open,
            [text("Inside")],
        )]
    };
    h.show(body);
    assert_eq!(h.named("Advanced").expanded, Some(false));
    assert!(!h.ui.text_content(h.view.root()).contains("Inside"));
    h.tap(h.named("Advanced").id);
    h.show(body);
    assert_eq!(h.named("Advanced").expanded, Some(true));
    assert!(h.ui.text_content(h.view.root()).contains("Inside"));
}

/// Keyed rows keep their nodes when the list is reordered.
#[test]
fn for_each_keeps_each_rows_nodes_by_identity() {
    let mut h = Harness::new();
    let rows = |names: &[&str]| {
        for_each(
            names.to_vec(),
            |n| n.to_string(),
            |n: &str| text(n).accessibility_label(n),
        )
    };
    let els = rows(&["a", "b"]);
    h.view.update(&mut h.ui, &mut h.input, els);
    let a = h.named("a").id;
    let els = rows(&["b", "a"]);
    h.view.update(&mut h.ui, &mut h.input, els);
    assert_eq!(h.named("a").id, a, "moved, not rebuilt");
    assert_eq!(h.ui.children(h.view.root())[1], a);
}

/// An adaptive grid puts as many columns as fit, each at least its width;
/// `.when` applies a modifier only when asked; a badge follows its view; a
/// title bar heads the page with its actions at the end.
#[test]
fn grids_badges_conditions_and_title_bars() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![
            title_bar(
                "Devices",
                "What is paired",
                [button("Scan", |_: &mut App| {})],
            ),
            adaptive_grid(
                120.0,
                8.0,
                (0..5).map(|i| text(format!("cell{i}")).accessibility_label(format!("cell{i}"))),
            ),
            text("Inbox").badge("3").accessibility_label("inbox"),
            text("Lit")
                .when(true, |e| e.bold())
                .accessibility_label("lit"),
            text("Plain")
                .when(false, |e| e.bold())
                .accessibility_label("plain"),
        ]
    });
    assert_eq!(h.with_role(Role::Heading(1))[0].name, "Devices");
    let scan = h.ui.rect(h.named("Scan").id);
    assert!(
        (scan.x + scan.width - 400.0).abs() < 0.5,
        "at the end: {scan:?}"
    );
    // 400 px: three columns of 120 and the gaps fit, so the fourth wraps.
    let (c0, c3) = (
        h.ui.rect(h.named("cell0").id),
        h.ui.rect(h.named("cell3").id),
    );
    assert!(c3.y > c0.y && (c3.x - c0.x).abs() < 0.5, "{c0:?} {c3:?}");
    assert!(c0.width >= 120.0);
    // The last row's one cell keeps the column's width, not the row's.
    let c4 = h.ui.rect(h.named("cell4").id);
    assert!((c4.width - c0.width).abs() < 0.5, "{c0:?} {c4:?}");
    assert!(h.ui.text_content(h.view.root()).contains('3'), "the badge");
    let weight = |n: &str| h.ui.style(h.named(n).id).font_weight;
    assert_ne!(weight("lit"), weight("plain"), "bold only when asked");
}

/// A confirmation says its question at once and puts Cancel first.
#[test]
fn a_confirmation_asks_with_the_safe_choice_first() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![confirmation(
            "Delete it?",
            "Cancel",
            "Delete",
            |_: &mut App| {},
            |a: &mut App| a.saved = 9,
        )]
    });
    assert_eq!(h.with_role(Role::Alert)[0].name, "Delete it?");
    let (cancel, yes) = (h.named("Cancel").id, h.named("Delete").id);
    assert!(h.ui.rect(cancel).x < h.ui.rect(yes).x);
    h.tap(yes);
    assert_eq!(h.app.saved, 9);
}

/// `.help` shows its text as a tooltip when the pointer rests on it.
#[test]
fn help_shows_a_tooltip() {
    let mut h = Harness::new();
    h.show(|_| vec![button("Scan", |_: &mut App| {}).help("Look for devices nearby")]);
    let scan = h.named("Scan").id;
    let r = h.ui.rect(scan);
    h.input.dispatch(
        &mut h.ui,
        &mut h.app,
        Event::PointerMove {
            x: r.x + 4.0,
            y: r.y + 4.0,
        },
    );
    for i in 1..=40 {
        h.input.tick(&mut h.ui, &mut h.app, i as f64 * 0.05);
    }
    let tip = h.input.tooltip_shown().expect("a tooltip");
    assert_eq!(h.ui.text_content(tip), "Look for devices nearby");
}

/// A text-only button in a column lines its label up with the text above
/// it; in a row it keeps its place.
#[test]
fn a_text_only_button_lines_up_in_a_column() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![
            vstack(
                4.0,
                [
                    text("Above").accessibility_label("above"),
                    button("Open", |_: &mut App| {}).button_style(ButtonKind::Plain),
                ],
            )
            .alignment(HorizontalAlignment::Leading),
            hstack(
                8.0,
                [
                    text("Left").accessibility_label("left"),
                    button("Go", |_: &mut App| {}).button_style(ButtonKind::Plain),
                ],
            ),
        ]
    });
    let (above, open) = (h.ui.rect(h.named("above").id), h.named("Open").id);
    assert_eq!(
        h.ui.rect(open).x + h.ui.style(open).padding.left,
        above.x,
        "lined up"
    );
    let (left, go) = (h.ui.rect(h.named("left").id), h.ui.rect(h.named("Go").id));
    assert_eq!(go.x - (left.x + left.width), 8.0, "in a row, the gap kept");
}

/// An inset group's title sits above its card, not in it; without a title
/// it is the card alone.
#[test]
fn an_inset_groups_title_is_above_its_card() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![
            inset_group("Server", [text("row").accessibility_label("row")]),
            inset_group("", [text("alone").accessibility_label("alone")]),
        ]
    });
    let (title, row) = (h.named("Server").id, h.named("row").id);
    let (title, row) = (h.ui.rect(title), h.ui.rect(row));
    assert!(
        title.y + title.height + 8.0 + 15.0 <= row.y + 0.5,
        "above the card's padding: {title:?} {row:?}"
    );
    let alone = h.ui.rect(h.named("alone").id);
    assert!(
        alone.y - (row.y + row.height) >= 15.0 + 15.0 - 0.5,
        "a card of its own: {row:?} {alone:?}"
    );
}

/// A segmented control: one option checked; a tap on another chooses it.
#[test]
fn a_segmented_control_chooses_on_a_tap() {
    let mut h = Harness::new();
    let body = |a: &App| {
        vec![segmented(
            ["Live", "Find", "Ask"],
            Some(a.saved as usize),
            |a: &mut App, i| a.saved = i as u32,
        )]
    };
    h.show(body);
    assert_eq!(h.named("Live").checked, Some(true));
    assert_eq!(h.named("Ask").checked, Some(false));
    h.tap(h.named("Ask").id);
    assert_eq!(h.app.saved, 2);
    h.show(body);
    assert_eq!(h.named("Ask").checked, Some(true));
}

/// A menu opens its options under it on a tap; picking one says which.
#[test]
fn a_menu_opens_and_picks() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![menu(
            "More",
            None,
            vec!["Help".into(), "About".into()],
            |a: &mut App, i| a.saved = 10 + i as u32,
        )]
    });
    assert!(h.with_role(Role::MenuItem).is_empty(), "closed at first");
    h.tap(h.named("More").id);
    h.input.tick(&mut h.ui, &mut h.app, 0.0);
    h.ui.layout(400.0, 600.0, &mut FixedWidthText(7.0));
    let items = h.with_role(Role::MenuItem);
    assert_eq!(items.len(), 2, "open: {items:?}");
    let about = items.iter().find(|a| a.name == "About").unwrap().id;
    h.tap(about);
    assert_eq!(h.app.saved, 11);
}

/// A hero is a heading in a card washed with the theme's accent: its
/// gradient runs from a mix of the surface and the accent to the surface.
#[test]
fn a_hero_is_a_heading_in_the_accent() {
    let mut h = Harness::new();
    h.show(|_| {
        vec![hero(
            r#"<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>"#,
            "Devices",
            "Manage your headsets.",
            Some(("2".into(), "paired".into())),
        )]
    });
    assert_eq!(h.named("Devices").role, Role::Heading(1));
    // Its tile in the accent, the card in the surface's colour.
    let (accent, surface) = (h.ui.token(Token::Accent), h.ui.token(Token::Surface));
    let mut stack = vec![h.ui.root()];
    let mut seen = vec![];
    while let Some(n) = stack.pop() {
        seen.extend(h.ui.style(n).background);
        stack.extend(h.ui.children(n).iter().copied());
    }
    assert!(
        seen.contains(&accent) && seen.contains(&surface),
        "{seen:?}"
    );
}

/// A line between one row of a card and the next, none round other things.
#[test]
fn rows_of_a_card_are_parted_by_lines() {
    let lines = |els: fn() -> Vec<El<App>>| {
        let mut h = Harness::new();
        h.show(move |_| vec![inset_group("", els())]);
        let mut stack = vec![h.ui.root()];
        let mut n = 0;
        while let Some(id) = stack.pop() {
            n += usize::from((h.ui.rect(id).height - 1.0).abs() < 0.01);
            stack.extend(h.ui.children(id).iter().copied());
        }
        n
    };
    assert_eq!(lines(|| vec![threers_ui::view::setting_row("A", "", text("x")), threers_ui::view::setting_row("B", "", text("y"))]), 1);
    assert_eq!(lines(|| vec![text("note"), threers_ui::view::setting_row("A", "", text("x")), text("note")]), 0);
}
