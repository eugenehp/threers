//! The Toolkit page: materials over moving colour, numbers that roll and
//! words that fade, keyframes, a list whose rows tilt at its edges,
//! gradients of three shapes, clip shapes, a mask and blend modes, an
//! aspect ratio, gauges and charts, a searchable list, undo and redo, and
//! the shortcuts that work where focus is.

use threers_math::easing::Easing;
use threers_ui::animation::{Property, Repeat};
use threers_ui::charts::{
    bar_chart, gauge_styled, line_chart, Gauge, GaugeKind, GaugeStyle, GaugeTint,
};
use threers_ui::content_transition::{crossfade_text, rolling_text, FadingText, RollingText};
use threers_ui::environment::Token;
use threers_ui::input::Input;
use threers_ui::keyframes::Keyframes;
use threers_ui::material::Material;
use threers_ui::pickers::slider;
use threers_ui::scroll_transition::ScrollTransition;
use threers_ui::undo::{Redo, Undo};
use threers_ui::{binding, css, style, NodeId, Role, Semantics, Ui};

use crate::app::State;
use crate::showcase::group;

/// Words the fading label goes through.
const WORDS: [&str; 5] = ["Hello", "Bonjour", "Hola", "Ciao", "Hallo"];

/// What the page keeps up from the state each frame.
pub struct Toolkit {
    count: RollingText,
    word: FadingText,
    gauges: Vec<Gauge>,
    chips: NodeId,
    shown_chips: Vec<[f32; 4]>,
    undo_label: NodeId,
    shortcuts: NodeId,
    shown_shortcuts: Vec<(String, &'static str)>,
}

/// A button that runs `f` on the state when clicked.
fn button(
    ui: &mut Ui,
    input: &mut Input<State>,
    parent: NodeId,
    label: &str,
    f: impl Fn(&mut State, &mut Ui) + 'static,
) -> NodeId {
    let b = ui.add_box(parent, style! { padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; flex-shrink: 0; });
    ui.token_style(
        b,
        &[
            ("background", Token::Raised),
            ("color", Token::Text),
            ("border-color", Token::Border),
        ],
    );
    ui.update_style(b, |s| s.border = threers_ui::Edges::all(1.0));
    ui.add_text(b, label);
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    ui.focus_visible_style(b, "box-shadow: 0 0 0 2px #93c5fd");
    input.on_click(b, move |s, ui, _| f(s, ui));
    b
}

/// A box styled by CSS declarations given as text (`name: value; …`).
fn boxed(ui: &mut Ui, parent: NodeId, decls: &str) -> NodeId {
    let b = ui.add_box(parent, style! {});
    ui.update_style(b, |s| {
        for d in decls.split(';').filter(|d| !d.trim().is_empty()) {
            if let Some((n, v)) = d.split_once(':') {
                css::apply(s, n.trim(), v.trim());
            }
        }
    });
    b
}

fn caption(ui: &mut Ui, parent: NodeId, text: &str) {
    let c = ui.add_box(parent, style! { font-size: 11px; });
    ui.token_style(c, &[("color", Token::SecondaryText)]);
    ui.add_text(c, text);
}

pub fn build(ui: &mut Ui, input: &mut Input<State>, state: &State, page: NodeId) -> Toolkit {
    // Materials over colour that moves.
    let g = group(ui, page, "Materials", "Frosted glass of four thicknesses over moving shapes: what is behind shows blurred and brighter, through a tint for the light or dark appearance. Press Move to set the shapes going.");
    let stage = boxed(ui, g, "position: relative; height: 170px; border-radius: 12px; overflow: hidden; background: conic-gradient(from 30deg, #9a3412, #9d174d, #5b21b6, #155e75, #166534, #9a3412)");
    let mut blobs = vec![];
    for (i, color) in ["#a16207", "#1e3a8a", "#991b1b"].iter().enumerate() {
        // Wholly inside the stage: a shape cut by two rounded clips at once
        // keeps only one of them on the GPU.
        let b = boxed(ui, stage, &format!("position: absolute; top: {}px; left: {}px; width: 70px; height: 70px; background: {color}; clip-path: circle()", 12 + i * 38, 30 + i * 110));
        ui.semantics_mut(b).hidden = true;
        blobs.push(b);
    }
    let big = boxed(ui, stage, "position: absolute; left: 16px; bottom: 8px; font-size: 34px; font-weight: bold; color: #fcfcfc");
    ui.add_text(big, "threers");
    let row = boxed(
        ui,
        stage,
        "position: absolute; left: 12px; right: 12px; top: 40px; display: flex; gap: 10px",
    );
    for (name, m) in [
        ("Ultra thin", Material::UltraThin),
        ("Thin", Material::Thin),
        ("Regular", Material::Regular),
        ("Thick", Material::Thick),
    ] {
        // Large and bold: legible over glass as thin as this.
        let chip = boxed(ui, row, "flex-grow: 1; flex-basis: 0px; height: 80px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 19px; font-weight: bold");
        ui.set_material(chip, Some(m));
        ui.token_style(chip, &[("color", Token::Text)]);
        ui.add_text(chip, name);
    }
    let controls = boxed(ui, g, "display: flex; gap: 8px");
    let bl = blobs.clone();
    button(ui, input, controls, "Move", move |_, ui| {
        for (i, &b) in bl.iter().enumerate() {
            let k = Keyframes::new()
                .track(
                    Property::TranslateX,
                    &[(0.0, 0.0), (1.2, 160.0 - 80.0 * i as f32), (2.4, 0.0)],
                )
                .repeat(Repeat::Times(2));
            ui.play_keyframes(b, k);
        }
    });

    // Content transitions.
    let g = group(ui, page, "Changing text", "A number whose digits roll up as it grows and down as it shrinks, lined up on its units; a word that fades into the next.");
    let r = boxed(ui, g, "display: flex; align-items: center; gap: 10px");
    button(ui, input, r, "−", |s, _| s.toolkit_count -= 7);
    let num = boxed(ui, r, "font-size: 30px; font-weight: bold; min-width: 90px; display: flex; justify-content: center");
    let count = rolling_text(ui, num, "Count", &state.toolkit_count.to_string());
    button(ui, input, r, "+", |s, _| s.toolkit_count += 7);
    let w = boxed(
        ui,
        r,
        "font-size: 22px; margin-left: 24px; min-width: 110px",
    );
    let word = crossfade_text(ui, w, WORDS[0]);
    button(ui, input, r, "Next word", |s, _| s.toolkit_word += 1);

    // Keyframes.
    let g = group(ui, page, "Keyframes", "Several properties on tracks of their own, each eased from key to key: a star that wobbles, a heart that beats a few times.");
    let r = boxed(ui, g, "display: flex; align-items: center; gap: 24px");
    let bell = boxed(ui, r, "font-size: 34px; cursor: pointer; min-width: 48px; min-height: 48px; display: flex; align-items: center; justify-content: center");
    ui.add_text(bell, "★");
    // Amber 600: 3:1 or more on both appearances' surfaces.
    ui.update_style(bell, |s| s.color = [0.851, 0.467, 0.024, 1.0]);
    ui.set_semantics(bell, Semantics::new(Role::Button).label("Wobble the star"));
    input.on_click(bell, move |_, ui, _| {
        let ring = Keyframes::new()
            .track(
                Property::RotateZ,
                &[
                    (0.0, 0.0),
                    (0.08, 0.35),
                    (0.2, -0.35),
                    (0.32, 0.25),
                    (0.44, -0.2),
                    (0.56, 0.0),
                ],
            )
            .track_eased(
                Property::Scale,
                &[
                    (0.0, 1.0, Easing::Linear),
                    (0.15, 1.18, Easing::QuadOut),
                    (0.56, 1.0, Easing::BackOut),
                ],
            );
        ui.play_keyframes(bell, ring);
    });
    let heart = boxed(ui, r, "font-size: 34px; cursor: pointer; min-width: 48px; min-height: 48px; display: flex; align-items: center; justify-content: center; color: #e11d48");
    ui.add_text(heart, "♥");
    ui.set_semantics(
        heart,
        Semantics::new(Role::Button).label("Make the heart beat"),
    );
    input.on_click(heart, move |_, ui, _| {
        let beat = Keyframes::phases(
            &[
                &[(Property::Scale, 1.0)],
                &[(Property::Scale, 1.3)],
                &[(Property::Scale, 1.05)],
                &[(Property::Scale, 1.25)],
            ],
            0.12,
            Easing::QuadOut,
        )
        .repeat(Repeat::Times(3));
        ui.play_keyframes(heart, beat);
    });
    caption(ui, r, "Click either.");

    // Scroll transitions.
    let g = group(
        ui,
        page,
        "Scroll transitions",
        "Rows tilt back and shrink by how far out of view they are.",
    );
    let list = boxed(ui, g, "height: 180px; overflow: auto; display: flex; flex-direction: column; gap: 6px; padding: 4px; border-radius: 8px");
    ui.token_style(list, &[("background", Token::Background)]);
    for i in 0..30 {
        let row = boxed(ui, list, "height: 38px; flex-shrink: 0; border-radius: 8px; display: flex; align-items: center; padding: 0 12px");
        ui.token_style(
            row,
            &[("background", Token::Raised), ("color", Token::Text)],
        );
        ui.add_text(row, format!("Row {}", i + 1));
    }
    ui.set_semantics(list, Semantics::new(Role::List).label("Tilting rows"));
    // Tilted and shrunk, not faded: a row partly out of view stays legible.
    ui.scroll_transition(
        list,
        ScrollTransition {
            fade: 0.0,
            ..ScrollTransition::tilted()
        },
    );

    // Shapes and colour.
    let g = group(ui, page, "Gradients, shapes, masks and blending", "Radial, conic and turned linear gradients; a circle, a capsule and a rounded inset cut from a box; a box faded out by a mask; multiply and screen; a 16∶9 box.");
    let r = boxed(
        ui,
        g,
        "display: flex; flex-wrap: wrap; gap: 14px; align-items: flex-end",
    );
    // A swatch drawn as `decls`, named under it.
    let swatch = |ui: &mut Ui, decls: &str, label: &str| {
        let col = boxed(
            ui,
            r,
            "display: flex; flex-direction: column; align-items: center; gap: 4px",
        );
        let b = boxed(ui, col, decls);
        ui.semantics_mut(b).hidden = true;
        caption(ui, col, label);
        b
    };
    swatch(ui, "width: 90px; height: 64px; border-radius: 10px; background: radial-gradient(circle at 30% 30%, #fef08a, #f97316 45%, #7c2d12)", "Radial");
    swatch(ui, "width: 90px; height: 64px; border-radius: 10px; background: conic-gradient(from 0deg, #ef4444, #eab308, #22c55e, #3b82f6, #a855f7, #ef4444)", "Conic");
    swatch(ui, "width: 90px; height: 64px; border-radius: 10px; background: linear-gradient(135deg, #0ea5e9, #6366f1)", "Linear, 135°");
    for (shape, label) in [
        ("circle()", "Circle"),
        ("capsule", "Capsule"),
        ("inset(8px round 14px)", "Rounded inset"),
    ] {
        swatch(ui, &format!("width: 90px; height: 64px; background: linear-gradient(to bottom, #14b8a6, #0f766e); clip-path: {shape}"), label);
    }
    swatch(ui, "width: 90px; height: 64px; background: #2563eb; mask-image: linear-gradient(to right, #000000, transparent)", "Mask");
    let under = swatch(ui, "display: flex; padding: 8px; border-radius: 10px; background: linear-gradient(to right, #fde68a, #93c5fd)", "Multiply, screen");
    for mode in ["multiply", "screen"] {
        let b = boxed(
            ui,
            under,
            &format!("width: 60px; height: 48px; background: #db2777; mix-blend-mode: {mode}"),
        );
        ui.semantics_mut(b).hidden = true;
    }
    swatch(ui, "width: 120px; aspect-ratio: 16 / 9; border-radius: 8px; background: linear-gradient(to bottom, #334155, #0f172a)", "Aspect ratio 16∶9");
    caption(ui, g, "Shaders, masks, blending and materials need the GPU renderer; the 2D canvas draws gradients and shapes too, and the rest plainly.");

    // Gauges and charts.
    let g = group(ui, page, "Gauges and charts", "Four gauges following the slider: a bar that turns amber then red as it fills, a marker on a scale, a dial with a needle and a ring, each gliding to its value; bars on gridlines, growing in, and a smooth line. Each is read out as its value or values to assistive technology, the bars one by one.");
    let s = boxed(ui, g, "width: 260px");
    slider(
        ui,
        input,
        state,
        s,
        "Level",
        binding!(|s: State| s.toolkit_level),
        0.0..=100.0,
        1.0,
    );
    let r = boxed(
        ui,
        g,
        "display: flex; align-items: center; gap: 28px; flex-wrap: wrap",
    );
    let bars = boxed(
        ui,
        r,
        "display: flex; flex-direction: column; gap: 18px; width: 200px",
    );
    let pct = |tint: GaugeTint| GaugeStyle {
        tint,
        unit: "%".into(),
        ..Default::default()
    };
    let gauges = vec![
        gauge_styled(
            ui,
            bars,
            "Storage used",
            0.0..=100.0,
            GaugeKind::Linear,
            pct(GaugeTint::levels()),
        ),
        gauge_styled(
            ui,
            bars,
            "Air quality",
            0.0..=100.0,
            GaugeKind::Marker,
            GaugeStyle {
                tint: GaugeTint::heat(),
                ..Default::default()
            },
        ),
        gauge_styled(
            ui,
            r,
            "Speed",
            0.0..=100.0,
            GaugeKind::Circular,
            GaugeStyle {
                tint: GaugeTint::heat(),
                unit: "km/h".into(),
                size: 112.0,
                ticks: 10,
                needle: true,
                ..Default::default()
            },
        ),
        gauge_styled(
            ui,
            r,
            "Uploaded",
            0.0..=100.0,
            GaugeKind::Ring,
            GaugeStyle {
                size: 104.0,
                ..Default::default()
            },
        ),
    ];
    let r = boxed(
        ui,
        g,
        "display: flex; align-items: flex-end; gap: 24px; flex-wrap: wrap",
    );
    let chart_box = boxed(ui, r, "width: 240px");
    bar_chart(
        ui,
        chart_box,
        "Steps this week",
        &[
            ("Mon", 4.2),
            ("Tue", 7.9),
            ("Wed", 6.1),
            ("Thu", 9.4),
            ("Fri", 5.3),
            ("Sat", 11.0),
            ("Sun", 3.2),
        ],
        90.0,
    );
    line_chart(
        ui,
        r,
        "Temperature today",
        &[
            11.0, 10.5, 10.0, 11.5, 14.0, 17.5, 19.0, 18.0, 16.5, 14.0, 12.5, 12.0,
        ],
        (220.0, 60.0),
    );

    // Search.
    let g = group(ui, page, "Search", "Type to narrow the list: letters in order, word starts first, matches in bold; arrows move, Enter picks.");
    let fruit: Vec<String> = [
        "Apple",
        "Apricot",
        "Avocado",
        "Banana",
        "Blackberry",
        "Blueberry",
        "Cherry",
        "Clementine",
        "Coconut",
        "Date",
        "Dragon fruit",
        "Fig",
        "Grape",
        "Grapefruit",
        "Kiwi",
        "Lemon",
        "Lime",
        "Mango",
        "Melon",
        "Nectarine",
        "Orange",
        "Papaya",
        "Peach",
        "Pear",
        "Pineapple",
        "Plum",
        "Pomegranate",
        "Raspberry",
        "Strawberry",
        "Watermelon",
    ]
    .map(String::from)
    .to_vec();
    let names = fruit.clone();
    let box_ = boxed(ui, g, "max-width: 320px");
    threers_ui::search::searchable(
        ui,
        input,
        state,
        box_,
        "Fruit",
        binding!(|s: State| s.toolkit_query),
        fruit,
        6,
        move |s, i| s.toast = Some(format!("Picked {}", names[i])),
    );

    // Undo.
    let g = group(ui, page, "Undo and redo", "Each Add is one step; Undo takes it back and Redo makes it again (⌘Z, ⇧⌘Z or Ctrl+Z, Ctrl+Y too). A new step after an undo drops what could have been redone.");
    let r = boxed(
        ui,
        g,
        "display: flex; gap: 8px; align-items: center; flex-wrap: wrap",
    );
    button(ui, input, r, "Add", |s, _| {
        let palette = [
            [0.93, 0.27, 0.27, 1.0],
            [0.96, 0.62, 0.04, 1.0],
            [0.13, 0.77, 0.37, 1.0],
            [0.23, 0.51, 0.96, 1.0],
            [0.66, 0.33, 0.97, 1.0],
        ];
        let c = palette[s.toolkit_chips.len() % palette.len()];
        s.toolkit_chips.push(c);
        s.undo.record(
            "Add",
            |s: &mut State| {
                s.toolkit_chips.pop();
            },
            move |s: &mut State| s.toolkit_chips.push(c),
        );
    });
    button(ui, input, r, "Undo", |s, _| undo(s));
    button(ui, input, r, "Redo", |s, _| redo(s));
    let undo_label = boxed(ui, r, "font-size: 12px");
    ui.token_style(undo_label, &[("color", Token::SecondaryText)]);
    ui.add_text(undo_label, "Nothing to undo");
    let chips = boxed(
        ui,
        g,
        "display: flex; gap: 6px; flex-wrap: wrap; min-height: 26px",
    );
    let root = ui.root();
    input.on_action(root, |s: &mut State, _, _: &Undo, _| undo(s));
    input.on_action(root, |s: &mut State, _, _: &Redo, _| redo(s));
    threers_ui::undo::bind_undo_keys(&mut input.keymap);

    // Shortcuts.
    let g = group(ui, page, "Shortcuts here", "The keys that do something where focus is now, read from the keymap and the handlers on the way out from the focus. Move focus (Tab) and they change.");
    let shortcuts = boxed(ui, g, "display: grid; grid-template-columns: 140px 1fr; row-gap: 4px; column-gap: 12px; font-size: 12px");

    Toolkit {
        count,
        word,
        gauges,
        chips,
        shown_chips: vec![],
        undo_label,
        shortcuts,
        shown_shortcuts: vec![],
    }
}

fn undo(s: &mut State) {
    let mut u = std::mem::take(&mut s.undo);
    u.undo(s);
    s.undo = u;
}

fn redo(s: &mut State) {
    let mut u = std::mem::take(&mut s.undo);
    u.redo(s);
    s.undo = u;
}

impl Toolkit {
    /// The page from the state: the number, the word, the gauges, the
    /// chips and what undo would take back, and the shortcuts here.
    pub fn update(&mut self, ui: &mut Ui, input: &Input<State>, s: &State) {
        self.count.set(ui, &s.toolkit_count.to_string());
        self.word.set(ui, WORDS[s.toolkit_word % WORDS.len()]);
        for g in &mut self.gauges {
            g.set(ui, s.toolkit_level);
        }
        if self.shown_chips != s.toolkit_chips {
            for c in ui.children(self.chips).to_vec() {
                ui.remove(c);
            }
            for &c in &s.toolkit_chips {
                let chip = ui.add_box(
                    self.chips,
                    style! { width: 26px; height: 26px; border-radius: 13px; },
                );
                ui.update_style(chip, |st| st.background = Some(c));
            }
            self.shown_chips = s.toolkit_chips.clone();
        }
        let label = match (s.undo.undo_name(), s.undo.redo_name()) {
            (Some(u), _) => format!("Undo {u}"),
            (None, Some(r)) => format!("Redo {r}"),
            _ => "Nothing to undo".into(),
        };
        let t = ui.children(self.undo_label)[0];
        ui.set_text(t, label);
        let now = input.shortcuts_here(ui, s.mac);
        if now != self.shown_shortcuts {
            for c in ui.children(self.shortcuts).to_vec() {
                ui.remove(c);
            }
            for (keys, action) in &now {
                let k = ui.add_box(self.shortcuts, style! { font-family: monospace; });
                ui.add_text(k, keys.as_str());
                let a = ui.add_box(self.shortcuts, style! {});
                ui.add_text(a, *action);
            }
            if now.is_empty() {
                let none = ui.add_box(self.shortcuts, style! {});
                ui.add_text(none, "None here");
            }
            self.shown_shortcuts = now;
        }
    }
}
