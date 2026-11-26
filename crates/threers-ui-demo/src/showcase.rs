//! The components section: tabs over people (avatars, badges, a strip that
//! scrolls sideways), a split pane, and actions (a switch, notifications, a
//! prompt and a popover). What needs the input (opening a prompt or a
//! popover) is asked for through `State::command`, run by the app.

use threers_ui::binding::Binding;
use threers_ui::components::{avatar, badge, disclosure, switch};
use threers_ui::input::Input;
use threers_ui::split::{split, Axis};
use threers_ui::{style, NodeId, Role, Semantics, Ui};

use crate::app::State;

/// The nodes the app needs back.
pub struct Showcase {
    /// The button a popover opens under.
    pub details: NodeId,
    /// The panel the tasks are described into (see `App::frame`).
    pub tasks: NodeId,
    /// The people panel: drawn as placeholders while "loading".
    pub people: NodeId,
    /// The upload bar, following the volume slider.
    pub progress: threers_ui::pickers::Progress,
    /// The empty state, shown when the search finds nobody.
    pub nobody: NodeId,
    /// The people the search finds, by name.
    pub found: Vec<(String, NodeId)>,
    /// The Screens tab's stack (its transition is set on it).
    pub stack: NodeId,
    /// The 3D page's departures board, showing `State::departure`.
    pub board: threers_ui::depth::SplitFlap,
    /// The Depth tab's layers, under their depth of field.
    pub layers: NodeId,
    /// The 3D page's candles, flickered while it shows.
    pub candles: Vec<threers_ui::candle::Candle>,
}

const PEOPLE: [&str; 14] = [
    "Ada Lovelace",
    "Alan Turing",
    "Grace Hopper",
    "Edsger Dijkstra",
    "Barbara Liskov",
    "Donald Knuth",
    "Frances Allen",
    "Ken Thompson",
    "Margaret Hamilton",
    "John McCarthy",
    "Radia Perlman",
    "Niklaus Wirth",
    "Shafi Goldwasser",
    "Leslie Lamport",
];

fn small_button(
    ui: &mut Ui,
    input: &mut Input<State>,
    parent: NodeId,
    label: &str,
    command: &'static str,
) -> NodeId {
    let b = ui.add_box(parent, style! { padding: 4px 10px; border-radius: 6px; border: 1px solid #3a4150; font-size: 12px; cursor: pointer; });
    ui.appearance_style(
        b,
        threers_ui::states::Appearance::Light,
        "border: 1px solid #d0d4da",
    );
    ui.hover_style(b, "background: rgba(128,138,160,0.28)");
    ui.focus_visible_style(b, "border: 1px solid #93c5fd");
    ui.add_text(b, label);
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    input.on_click(b, move |s: &mut State, _, _| {
        s.command = Some(command.into())
    });
    b
}

/// Where the components go: the pages they belong on.
pub struct Pages {
    pub basics: NodeId,
    pub components: NodeId,
    pub screens: NodeId,
    pub depth: NodeId,
}

/// A titled group on a page.
pub fn group(ui: &mut Ui, page: NodeId, title: &str, about: &str) -> NodeId {
    use threers_ui::environment::Token;
    let g = ui.add_box(page, style! { display: flex; flex-direction: column; gap: 10px; padding: 14px; border-radius: 10px; flex-shrink: 0; });
    ui.token_style(
        g,
        &[
            ("background", Token::Surface),
            ("border-color", Token::Separator),
        ],
    );
    let h = ui.add_box(g, style! { font-size: 14px; font-weight: bold; });
    ui.add_text(h, title);
    ui.set_semantics(h, Semantics::new(Role::Heading(3)));
    if !about.is_empty() {
        let a = ui.add_box(g, style! { font-size: 12px; white-space: normal; });
        ui.token_style(a, &[("color", Token::SecondaryText)]);
        ui.add_text(a, about);
    }
    g
}

pub fn build(ui: &mut Ui, input: &mut Input<State>, state: &State, pages: &Pages) -> Showcase {
    let people_g = group(
        ui,
        pages.components,
        "People",
        "Avatars and badges, and a strip that scrolls sideways and settles on a person.",
    );
    let split_g = group(
        ui,
        pages.components,
        "Split pane",
        "Drag the handle, or focus it and use the arrow keys; a double-click resets it.",
    );
    let tasks_g = group(
        ui,
        pages.components,
        "Tasks",
        "Described from the state each frame: the view changes only what differs.",
    );
    let controls_g = group(ui, pages.components, "Controls", "Each bound to the state: a picker whose highlight slides, a stepper, a slider, progress, a search.");
    let actions_g = group(
        ui,
        pages.basics,
        "Actions",
        "Notifications, a prompt with roles, a popover, and switches bound to the state.",
    );
    let screens_g = group(
        ui,
        pages.screens,
        "A navigation stack",
        "Screens pushed and popped by a path in the state; pick how they turn.",
    );
    let d = pages.depth;
    // People: a few with badges, then a strip of everyone that scrolls
    // sideways (a wheel's dx, a trackpad, or a finger).
    let row = ui.add_box(
        people_g,
        style! { display: flex; align-items: center; gap: 8px; },
    );
    for (name, count) in [
        ("Ada Lovelace", "3"),
        ("Alan Turing", "12"),
        ("Grace Hopper", ""),
    ] {
        let who = ui.add_box(
            row,
            style! { display: flex; align-items: center; gap: 4px; },
        );
        avatar(ui, who, name, 28.0);
        if !count.is_empty() {
            badge(ui, who, count, [0.86, 0.15, 0.15, 1.0]);
        }
    }
    let strip = ui.add_box(
        people_g,
        style! { display: flex; gap: 6px; padding: 4px 0 8px 0; overflow-x: auto; scrollbar-width: thin; scrollbar-visibility: auto; },
    );
    ui.set_semantics(strip, Semantics::new(Role::List).label("Everyone"));
    // It comes to rest with a person at its start, as a carousel does.
    ui.set_scroll_snap(strip, Some(threers_ui::scroll_snap::Snap::Items));
    for name in PEOPLE {
        let p = ui.add_box(strip, style! { display: flex; flex-direction: column; align-items: center; gap: 2px; width: 64px; flex-shrink: 0; font-size: 10px; });
        avatar(ui, p, name, 36.0);
        let first = name.split_whitespace().next().unwrap_or(name);
        ui.add_text(p, first);
        ui.set_semantics(p, Semantics::new(Role::ListItem).label(name));
    }

    // Split: two panes and a handle to drag, or to focus and move with the
    // arrows; a double-click puts it back.
    let host = ui.add_box(split_g, style! { display: flex; height: 120px; border-radius: 6px; border: 1px solid #2a303c; overflow: hidden; });
    let sp = split(ui, input, host, Axis::Horizontal, 110.0, 60.0);
    for (pane, text) in [
        (sp.first, "Drag the handle"),
        (
            sp.second,
            "Or focus it and use the arrow keys; a double-click resets it.",
        ),
    ] {
        ui.update_style(pane, |s| {
            threers_ui::css::apply(s, "padding", "8px");
            threers_ui::css::apply(s, "font-size", "12px");
            threers_ui::css::apply(s, "white-space", "normal");
        });
        ui.add_text(pane, text);
    }

    // Actions.
    let actions = actions_g;
    // Bound to the state: the Settings dialog's switch is bound to the
    // same shadows, and the two never disagree.
    switch(
        ui,
        input,
        state,
        actions,
        "Notifications",
        Binding::new(|s: &State| !s.quiet, |s: &mut State, on| s.quiet = !on),
    );
    switch(
        ui,
        input,
        state,
        actions,
        "Card shadows",
        threers_ui::binding!(|s: State| s.shadows),
    );
    let buttons = ui.add_box(
        actions,
        style! { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; },
    );
    small_button(ui, input, buttons, "Notify", "notify");
    small_button(ui, input, buttons, "Error", "error");
    small_button(ui, input, buttons, "Clear name…", "clear");
    let details = small_button(ui, input, buttons, "Details", "details");
    // Controls, each bound to the state.
    use threers_ui::pickers::{empty_state, progress, search_field, segmented, slider, stepper};
    let c = controls_g;
    segmented(
        ui,
        input,
        state,
        c,
        "Size",
        &["Small", "Medium", "Large"],
        threers_ui::binding!(|s: State| s.size),
    );
    stepper(
        ui,
        input,
        state,
        c,
        "Guests",
        threers_ui::binding!(|s: State| s.guests),
        1..=12,
        1,
    );
    slider(
        ui,
        input,
        state,
        c,
        "Volume",
        threers_ui::binding!(|s: State| s.volume),
        0.0..=1.0,
        0.05,
    );
    let bar = progress(ui, c, "Upload", Some(state.volume));
    switch(
        ui,
        input,
        state,
        c,
        "Loading (placeholders)",
        threers_ui::binding!(|s: State| s.loading),
    );
    small_button(ui, input, c, "Open sheet…", "sheet");
    search_field(
        ui,
        input,
        state,
        c,
        "Search people",
        threers_ui::binding!(|s: State| s.query),
    );
    let results = ui.add_box(
        c,
        style! { display: flex; flex-direction: column; gap: 2px; },
    );
    let found: Vec<(String, NodeId)> = PEOPLE
        .iter()
        .map(|&name| {
            let row = ui.add_box(
                results,
                style! { display: flex; align-items: center; gap: 6px; },
            );
            avatar(ui, row, name, 20.0);
            ui.add_text(row, name);
            ui.set_semantics(row, Semantics::new(Role::ListItem));
            ui.combine_accessibility(row);
            (name.to_owned(), row)
        })
        .collect();
    let nobody = empty_state::<State>(ui, input, c, "⌕", "Nobody found", "Try another name.", None);

    // Screens: a stack whose path is in the state.
    let host = ui.add_box(screens_g, style! { display: flex; flex-direction: column; height: 220px; border-radius: 6px; border: 1px solid #2a303c; overflow: hidden; });
    segmented(
        ui,
        input,
        state,
        screens_g,
        "Transition",
        &["Slide", "Cube", "Page"],
        threers_ui::binding!(|s: State| s.nav_transition),
    );
    let stack = threers_ui::nav_stack::nav_stack(
        ui,
        input,
        state,
        host,
        "Mailboxes",
        threers_ui::binding!(|s: State| s.path),
        |ui, input, _, page, screen| {
            ui.add_text(screen, format!("Messages in {page}"));
            let more = ui.add_box(screen, style! { padding: 6px 0; cursor: pointer; });
            ui.token_style(more, &[("color", threers_ui::environment::Token::AccentText)]);
            ui.add_text(more, "Open the first one ›");
            ui.set_semantics(
                more,
                Semantics::new(Role::Button).label("Open the first one"),
            );
            let page = page.to_owned();
            input.on_click(more, move |s: &mut State, _, _| {
                s.path.push(format!("{page} · 1"))
            });
        },
        |ui, input, _, screen| {
            for name in ["Inbox", "Archive", "Sent"] {
                let row = ui.add_box(
                    screen,
                    style! { padding: 8px 4px; cursor: pointer; border-bottom: 1px solid #2a303c; },
                );
                ui.token_style(
                    row,
                    &[("border-color", threers_ui::environment::Token::Separator)],
                );
                ui.add_text(row, format!("{name} ›"));
                ui.set_semantics(row, Semantics::new(Role::Button).label(name));
                input.on_click(row, move |s: &mut State, _, _| s.path.push(name.to_owned()));
            }
        },
    );
    // Depth: cards that lean toward the pointer (their avatar and title
    // nearer than the card, so they slide over it), a card that flips,
    // and a strip whose items turn away from its middle.
    use threers_ui::environment::Token;
    let row = ui.add_box(d, style! { display: flex; gap: 10px; padding: 6px 2px; });
    for (name, role) in [("Ada Lovelace", "Analyst"), ("Grace Hopper", "Compiler")] {
        let card = ui.add_box(row, style! { flex-grow: 1; flex-basis: 0px; display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 12px 6px; border-radius: 12px; });
        ui.token_style(card, &[("background", Token::Raised)]);
        let a = avatar(ui, card, name, 40.0);
        ui.set_parallax(a, 22.0);
        let n = ui.add_box(card, style! { font-weight: bold; font-size: 12px; });
        ui.add_text(n, name);
        ui.set_parallax(n, 12.0);
        let r = ui.add_box(card, style! { font-size: 11px; });
        ui.token_style(r, &[("color", Token::SecondaryText)]);
        ui.add_text(r, role);
        ui.set_semantics(card, Semantics::new(Role::Group).label(name));
        input.tilt_on_hover(ui, card, 14.0);
    }
    let flip = threers_ui::depth::flip_card(
        ui,
        input,
        state,
        d,
        "Word card",
        threers_ui::binding!(|s: State| s.flipped),
        |ui, f| {
            ui.update_style(f, |s| {
                (s.align_items, s.justify_content) = (
                    threers_ui::AlignItems::Center,
                    threers_ui::JustifyContent::Center,
                )
            });
            let w = ui.add_box(f, style! { font-size: 22px; font-weight: bold; });
            ui.add_text(w, "der Hund");
        },
        |ui, b| {
            ui.update_style(b, |s| {
                (s.align_items, s.justify_content) = (
                    threers_ui::AlignItems::Center,
                    threers_ui::JustifyContent::Center,
                )
            });
            let w = ui.add_box(
                b,
                style! { font-size: 22px; font-weight: bold; color: rgb(37, 99, 235); },
            );
            ui.add_text(w, "the dog");
        },
    );
    ui.update_style(flip.card, |s| {
        s.height = threers_ui::Length::Px(90.0);
        s.margin.top = 6.0;
    });
    let hint = ui.add_box(d, style! { font-size: 11px; });
    ui.token_style(hint, &[("color", Token::SecondaryText)]);
    ui.add_text(hint, "Tap the card to turn it over.");
    let carousel = ui.add_box(d, style! { display: flex; gap: 0px; padding: 10px 100px; overflow-x: auto; scrollbar-width: none; });
    ui.set_semantics(carousel, Semantics::new(Role::List).label("Carousel"));
    for name in PEOPLE {
        let p = ui.add_box(carousel, style! { display: flex; flex-direction: column; align-items: center; gap: 4px; width: 80px; flex-shrink: 0; padding: 8px 0; border-radius: 10px; font-size: 10px; });
        ui.token_style(p, &[("background", Token::Raised)]);
        avatar(ui, p, name, 40.0);
        ui.add_text(p, name.split_whitespace().next().unwrap_or(name));
        ui.set_semantics(p, Semantics::new(Role::ListItem).label(name));
    }
    ui.carousel_3d(carousel);
    ui.set_scroll_snap(carousel, Some(threers_ui::scroll_snap::Snap::Items));

    // A model, turned by dragging it (or the arrow keys), lit by the light
    // the pointer holds.
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "A model to turn");
    threers_ui::model3d::model_view(
        ui,
        input,
        state,
        d,
        "A torus",
        threers_ui::model3d::Mesh3::torus(1.0, 0.38, 36, 18, [0.15, 0.39, 0.92]),
        threers_ui::binding!(|s: State| s.orbit),
        (300, 180),
    );

    // Layers at three depths, blurred by how far each is from the focus.
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "Depth of field");
    // Cards of one size at three depths, with room between them for the
    // near one to come forward into.
    let layers = ui.add_box(
        d,
        style! { display: flex; justify-content: center; gap: 36px; padding: 14px 0 18px; },
    );
    for (label, depth) in [("Far", -140.0), ("Middle", 0.0), ("Near", 70.0)] {
        let l = ui.add_box(layers, style! { width: 132px; padding: 14px 6px; border-radius: 10px; display: flex; justify-content: center; font-weight: bold; });
        ui.token_style(l, &[("background", Token::Raised)]);
        ui.add_text(l, label);
        ui.set_transform3d(
            l,
            threers_ui::depth::Transform3d {
                depth,
                shadow: true,
                ..Default::default()
            },
        );
    }
    slider(
        ui,
        input,
        state,
        d,
        "Focus",
        threers_ui::binding!(|s: State| s.focus),
        0.0..=1.0,
        0.05,
    );

    // A card that leans as the phone is tilted.
    let phone = ui.add_box(d, style! { padding: 14px; border-radius: 12px; display: flex; justify-content: center; font-size: 12px; });
    ui.token_style(phone, &[("background", Token::Raised)]);
    ui.add_text(phone, "Tilt your phone: this card leans with it");
    ui.set_semantics(
        phone,
        Semantics::new(Role::Group).label("Leans with the phone"),
    );
    input.tilt_with_device(ui, phone, 15.0);

    // A deck to swipe through.
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "Swipe the cards");
    let deck = threers_ui::depth::card_deck(
        ui,
        input,
        d,
        "People to meet",
        threers_ui::binding!(|s: State| s.deck),
        |ui, card, name| {
            ui.update_style(card, |s| {
                s.display = threers_ui::Display::Flex;
                s.flex_direction = threers_ui::FlexDirection::Column;
                s.align_items = threers_ui::AlignItems::Center;
                s.justify_content = threers_ui::JustifyContent::Center;
                s.gap = 8.0;
            });
            avatar(ui, card, name, 56.0);
            let n = ui.add_box(card, style! { font-weight: bold; });
            ui.add_text(n, name);
        },
        |s: &mut State, name, dir| {
            s.toast = Some(format!(
                "{} {name}",
                if dir == threers_ui::depth::SwipeDirection::Right {
                    "Liked"
                } else {
                    "Passed on"
                }
            ));
        },
    );
    ui.update_style(deck.deck, |s| {
        s.height = threers_ui::Length::Px(170.0);
        s.margin.bottom = 10.0;
    });

    // Its spacing grows with the text, as its text does.
    // Basics: a disclosure that folds open and shut.
    let (_, more) = disclosure(
        ui,
        input,
        state,
        actions_g,
        "How this works",
        Binding::local(false),
    );
    let m = ui.add_box(more, style! { font-size: 12px; white-space: normal; });
    ui.add_text(m, "Every control here keeps no state of its own: it reads and writes the app's state through a binding, so two controls bound to one value always agree.");

    // 3D: a cube of seasons, a drum of numbers, and reflections.
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "A cube (arrow keys, or pick a face)");
    segmented(
        ui,
        input,
        state,
        d,
        "Face",
        &["Spring", "Summer", "Autumn", "Winter"],
        threers_ui::binding!(|s: State| s.cube_face),
    );
    let c = threers_ui::depth::cube(
        ui,
        input,
        state,
        d,
        "Seasons",
        4,
        threers_ui::binding!(|s: State| s.cube_face),
        |ui, f, k| {
            let (name, colour) = [
                ("Spring", [0.18, 0.49, 0.2]),
                ("Summer", [0.71, 0.33, 0.04]),
                ("Autumn", [0.71, 0.14, 0.09]),
                ("Winter", [0.11, 0.31, 0.85]),
            ][k];
            ui.update_style(f, |s| {
                s.display = threers_ui::Display::Flex;
                s.align_items = threers_ui::AlignItems::Center;
                s.justify_content = threers_ui::JustifyContent::Center;
                s.font_size = 22.0;
                s.font_weight = threers_ui::FontWeight::Bold;
            });
            ui.token_style(f, &[("color", Token::OnAccent)]);
            // Its own colour, not the theme's.
            ui.clear_token_style(f, "background");
            ui.update_style(f, |s| {
                s.background = Some([colour[0], colour[1], colour[2], 1.0])
            });
            ui.add_text(f, name);
        },
    );
    ui.update_style(c.cube, |s| {
        s.height = threers_ui::Length::Px(110.0);
        s.margin.top = 4.0;
        s.margin.bottom = 8.0;
    });
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "A drum: scroll it");
    let drum = ui.add_box(d, style! { display: flex; flex-direction: column; height: 180px; overflow: auto; scrollbar-width: none; border-radius: 10px; padding: 70px 0; box-sizing: border-box; });
    ui.token_style(drum, &[("background", Token::Separator)]);
    ui.set_semantics(drum, Semantics::new(Role::List).label("Minutes"));
    for m in 0..60 {
        let row = ui.add_box(drum, style! { height: 36px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 20px; border-radius: 6px; });
        ui.token_style(row, &[("background", Token::Raised)]);
        ui.add_text(row, format!("{m:02}"));
        ui.set_semantics(row, Semantics::new(Role::ListItem));
    }
    ui.cylinder_3d(drum);
    ui.set_scroll_snap(drum, None);
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "Reflections");
    let shelf = ui.add_box(
        d,
        style! { display: flex; gap: 14px; justify-content: center; padding: 8px 0 56px 0; },
    );
    for name in ["Ada Lovelace", "Grace Hopper", "Alan Turing"] {
        let a = avatar(ui, shelf, name, 64.0);
        ui.set_reflection(a, Some((0.35, 0.6)));
    }
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "A split-flap board: click it, or the button");
    let row = ui.add_box(
        d,
        style! { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; },
    );
    let mut board =
        threers_ui::depth::split_flap(ui, row, "Next departure", DEPARTURE_CELLS, 30.0, 46.0);
    board.set(ui, DEPARTURES[state.departure % DEPARTURES.len()]);
    input.on_click(board.node, |s: &mut State, _, _| s.departure += 1);
    let next = ui.add_box(
        row,
        style! { padding: 6px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; },
    );
    ui.token_style(next, &[("background", Token::Raised)]);
    ui.hover_style(next, "background: rgba(128,138,160,0.28)");
    ui.focus_visible_style(next, "box-shadow: 0 0 0 2px #93c5fd");
    ui.add_text(next, "Next departure");
    ui.set_semantics(next, Semantics::new(Role::Button).label("Next departure"));
    input.on_click(next, |s: &mut State, _, _| s.departure += 1);

    // Candles in the dark: the simple flame (paths) and the realistic one
    // (a volume through a shader; the simple flame where none runs).
    let head = ui.add_box(d, style! { font-size: 12px; font-weight: bold; });
    ui.add_text(head, "Candles: a simple flame, and a realistic one");
    let stage = ui.add_box(d, style! { display: flex; gap: 36px; justify-content: center; align-items: flex-end; padding: 18px 24px 0 24px; border-radius: 12px; background: linear-gradient(180deg, #0b0a0d, #17130f 70%, #2a1e14); });
    ui.set_semantics(stage, Semantics::new(Role::Group).label("Candles"));
    let candles = [
        (threers_ui::candle::FlameStyle::Simple, 1.0),
        (threers_ui::candle::FlameStyle::Realistic, 2.0),
    ]
    .into_iter()
    .map(|(style, seed)| threers_ui::candle::candle(ui, stage, style, 120.0, seed))
    .collect::<Vec<_>>();
    for c in &candles {
        c.update(ui, 0.0);
    }

    for g in [people_g, split_g, tasks_g, controls_g, actions_g, screens_g] {
        ui.scale_tree_with_text(g);
    }
    Showcase {
        details,
        tasks: tasks_g,
        people: people_g,
        progress: bar,
        nobody,
        found,
        stack: stack.container,
        layers,
        board,
        candles,
    }
}

/// What the departures board cycles through.
pub const DEPARTURES: [&str; 6] = [
    "LISBON  ", "OSLO    ", "NAIROBI ", "KYOTO   ", "SANTIAGO", "REYKJAVK",
];
pub const DEPARTURE_CELLS: usize = 8;

/// A task in the Tasks tab.
#[derive(Debug, Clone, PartialEq)]
pub struct Task {
    pub id: u32,
    pub title: String,
    pub done: bool,
}

/// The Tasks tab, described from the state each frame: a row a task
/// (click to tick it, ✕ to delete it), a count, and a button to add one.
/// The view changes only what differs from last time.
pub fn tasks(s: &State) -> Vec<threers_ui::view::El<State>> {
    use threers_ui::environment::Token;
    use threers_ui::view::El;
    let mut out: Vec<El<State>> = s
        .tasks
        .iter()
        .map(|t| {
            let id = t.id;
            let mark = if t.done { "☑" } else { "☐" };
            let title = if t.done { "text-decoration: line-through; flex-grow: 1" } else { "flex-grow: 1" };
            El::div("display: flex; align-items: center; gap: 8px; padding: 4px 6px; border-radius: 6px; cursor: pointer")
                .key(id.to_string())
                .hover("background: rgba(128,138,160,0.2)")
                .semantics(Semantics::new(Role::Checkbox).label(t.title.clone()).checked(t.done))
                .on_click(move |s: &mut State, _| {
                    if let Some(t) = s.tasks.iter_mut().find(|t| t.id == id) {
                        t.done = !t.done;
                    }
                })
                .child(El::label(mark))
                .child(El::div(title).text(t.title.clone()))
                .child(
                    El::div("padding: 4px 8px; min-width: 24px; min-height: 24px; box-sizing: border-box")
                        .token("color", Token::SecondaryText)
                        .role(Role::Button, format!("Delete {}", t.title))
                        .text("✕")
                        .on_click(move |s: &mut State, _| s.tasks.retain(|t| t.id != id)),
                )
        })
        .collect();
    let left = s.tasks.iter().filter(|t| !t.done).count();
    out.push(
        El::div("font-size: 11px; padding: 2px 6px")
            .token("color", Token::SecondaryText)
            .text(format!("{left} of {} to do", s.tasks.len())),
    );
    out.push(
        El::div("align-self: flex-start; padding: 5px 12px; border-radius: 6px; font-size: 12px; cursor: pointer")
            .token("background", Token::Accent)
            .token("color", Token::OnAccent)
            .role(Role::Button, "Add task")
            .text("Add task")
            .on_click(|s: &mut State, _| {
                let id = s.tasks.iter().map(|t| t.id).max().unwrap_or(0) + 1;
                s.tasks.push(Task { id, title: format!("Task {id}"), done: false });
            }),
    );
    out
}
