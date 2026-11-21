//! The on-screen keyboard's presets and placements, drawn on the CPU to
//! PNGs in `out/keyboards/`: an iPhone's (light, dark, one-handed, held on
//! a letter for its accents, for an email address and a web address, a
//! number pad), an iPad's (docked, split, floating), an iPad Pro's, and a
//! kiosk's — centred at a width, in the UI's own theme, a number row and a
//! row of arrows added.
//!
//! ```text
//! cargo run --features ui --example virtual_keyboards
//! ```

use threers::exporters::DrawingFonts;
use threers::style;
use threers::ui::input::{Event, Input, PointerButton};
use threers::ui::states::Appearance;
use threers::ui::vector::ui_to_png;
use threers::ui::virtual_keyboard::{
    Align, Key, KeyboardConfig, KeyboardLayout, KeyboardTheme, Placement, ReturnKey,
    VirtualKeyboard,
};
use threers::ui::{FieldKey, Role, Semantics, TextField, Ui};

struct Shot {
    name: &'static str,
    config: KeyboardConfig,
    size: (f32, f32),
    dark: bool,
    /// A key held down this long (seconds) when drawn.
    hold: Option<(&'static str, f64)>,
    languages: usize,
    /// Reads the field, offering words for the one typed.
    suggest: bool,
}

fn shot(name: &'static str, config: KeyboardConfig, size: (f32, f32)) -> Shot {
    Shot {
        name,
        config,
        size,
        dark: false,
        hold: None,
        languages: 1,
        suggest: false,
    }
}

fn render(s: &Shot, fonts: &DrawingFonts) {
    let (w, h) = s.size;
    let mut ui = Ui::new();
    if s.dark {
        ui.set_system_appearance(Appearance::Dark);
    }
    let mut input: Input<TextField> = Input::new();
    let root = ui.root();
    ui.set_style(
        root,
        style! { display: flex; flex-direction: column; padding: 16px; gap: 8px; font-family: Arial; font-size: 15px; },
    );
    ui.update_style(root, |st| {
        st.background = Some(if s.dark {
            [0.07, 0.07, 0.08, 1.0]
        } else {
            [0.95, 0.95, 0.97, 1.0]
        });
        st.color = if s.dark {
            [0.92, 0.92, 0.94, 1.0]
        } else {
            [0.1, 0.1, 0.12, 1.0]
        };
    });
    let title = ui.add_box(root, style! { font-weight: bold; });
    ui.add_text(title, s.name);
    let field = ui.add_box(
        root,
        style! { height: 36px; border-radius: 8px; padding: 8px 10px; box-sizing: border-box; background: #ffffff; color: #111111; border: 1px solid #c8c8cc; },
    );
    let text = ui.add_text(field, "");
    ui.set_semantics(field, Semantics::new(Role::TextBox).label("Message"));
    input.on_key(field, |f: &mut TextField, _, key, m, _| {
        !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
    });
    let kb = VirtualKeyboard::with_config(&mut ui, &mut input, s.config.clone());
    kb.set_enabled(true);
    if s.suggest {
        kb.set_context(|f: &TextField, _| Some(f.before_caret()));
        kb.set_suggestions(|w| {
            ["Hello", "Help", "Helmet"]
                .iter()
                .filter(|s| s.to_lowercase().starts_with(&w.to_lowercase()))
                .map(|s| s.to_string())
                .collect()
        });
    }
    let mut layouts = vec![KeyboardLayout::english(), KeyboardLayout::french()];
    layouts.truncate(s.languages);
    kb.set_layouts(&mut ui, layouts);
    let mut state = TextField::new(if s.suggest { "Hel" } else { "Hello" });
    input.set_focus(&mut ui, &mut state, Some(field));
    let mut t = 0.0;
    let mut settle = |ui: &mut Ui, input: &mut Input<TextField>, state: &mut TextField, dt: f64| {
        t += dt;
        input.tick(ui, state, t);
        ui.step(0.3);
        let mut measure = threers::ui::FixedWidthText(7.0);
        ui.layout(w, h, &mut measure);
    };
    for _ in 0..3 {
        settle(&mut ui, &mut input, &mut state, 0.0);
    }
    if let Some((key, seconds)) = s.hold {
        let n = ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == key)
            .unwrap_or_else(|| panic!("no key {key}"))
            .id;
        let r = ui.rect(n);
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        input.dispatch(&mut ui, &mut state, Event::PointerMove { x, y });
        let button = PointerButton::Primary;
        input.dispatch(&mut ui, &mut state, Event::PointerDown { x, y, button });
        let steps = (seconds / 0.05).ceil() as usize;
        for _ in 0..steps {
            settle(&mut ui, &mut input, &mut state, 0.05);
        }
    }
    ui.set_text(text, state.text.clone());
    let png = ui_to_png(&mut ui, w, h, 2.0, fonts);
    std::fs::write(format!("out/keyboards/{}.png", s.name), png).unwrap();
    println!("out/keyboards/{}.png", s.name);
}

fn main() {
    std::fs::create_dir_all("out/keyboards").unwrap();
    let fonts = DrawingFonts::new();
    let phone = (402.0, 400.0);
    let ipad = (820.0, 440.0);
    let shots = vec![
        shot("iphone", KeyboardConfig::iphone(), phone),
        Shot {
            dark: true,
            languages: 2,
            ..shot("iphone-dark-two-languages", KeyboardConfig::iphone(), phone)
        },
        Shot {
            hold: Some(("e", 0.6)),
            ..shot("iphone-holding-e", KeyboardConfig::iphone(), phone)
        },
        Shot {
            hold: Some(("g", 0.1)),
            ..shot("iphone-pressing-g", KeyboardConfig::iphone(), phone)
        },
        Shot {
            suggest: true,
            ..shot("iphone-suggestions", KeyboardConfig::iphone(), phone)
        },
        shot(
            "iphone-one-handed-left",
            KeyboardConfig::iphone().placement(Placement::one_handed(Align::Left)),
            phone,
        ),
        shot(
            "iphone-email-send",
            KeyboardConfig::email().return_key(ReturnKey::Send),
            phone,
        ),
        shot("iphone-url", KeyboardConfig::url(), phone),
        shot("iphone-number-pad", KeyboardConfig::number_pad(), phone),
        shot("ipad", KeyboardConfig::ipad(), ipad),
        shot("ipad-pro", KeyboardConfig::ipad_pro(), (1024.0, 500.0)),
        shot(
            "ipad-split",
            KeyboardConfig::ipad().placement(Placement::docked().split(260.0)),
            (1024.0, 440.0),
        ),
        shot("ipad-floating", KeyboardConfig::ipad_floating(), ipad),
        shot(
            "kiosk-centred",
            KeyboardConfig::iphone()
                .theme(KeyboardTheme::system())
                .placement(Placement::centered(640.0))
                .with_number_row()
                .with_row_at_bottom(vec![
                    Key::named("ArrowLeft", "←"),
                    Key::named("ArrowRight", "→"),
                    Key::text(".com").units(2.0),
                ]),
            (1000.0, 480.0),
        ),
    ];
    for s in &shots {
        render(s, &fonts);
    }
}
