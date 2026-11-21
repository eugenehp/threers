//! Design an on-screen keyboard in a text file and see it: the file is
//! checked (each problem with its line), sketched as text, and each of its
//! pages drawn to `out/keyboards/<file>-<page>.png` on the CPU — again
//! every time the file is saved, with `--watch`. The format is
//! `KeyboardConfig::parse`'s; `examples/keyboards/kiosk.kbd` is one.
//!
//! ```text
//! cargo run --features ui --example keyboard_designer -- examples/keyboards/kiosk.kbd --watch
//! cargo run --features ui --example keyboard_designer -- my.kbd --size 820x1180 --dark
//! ```

use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

use threers::exporters::DrawingFonts;
use threers::style;
use threers::ui::input::Input;
use threers::ui::states::Appearance;
use threers::ui::vector::ui_to_png;
use threers::ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout, Page, VirtualKeyboard};
use threers::ui::{FieldKey, FixedWidthText, Role, Semantics, TextField, Ui};

struct Options {
    file: String,
    watch: bool,
    size: (f32, f32),
    dark: bool,
}

fn options() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mut o = Options {
        file: String::new(),
        watch: false,
        size: (402.0, 480.0),
        dark: false,
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--watch" => o.watch = true,
            "--dark" => o.dark = true,
            "--size" => {
                let s = args.next().ok_or("--size WxH")?;
                let (w, h) = s.split_once('x').ok_or("--size WxH, as 820x1180")?;
                o.size = (
                    w.parse().map_err(|_| "--size: width")?,
                    h.parse().map_err(|_| "--size: height")?,
                );
            }
            f if !f.starts_with("--") && o.file.is_empty() => o.file = f.to_owned(),
            other => {
                return Err(format!(
                    "{other}? keyboard_designer FILE [--watch] [--size WxH] [--dark]"
                ))
            }
        }
    }
    if o.file.is_empty() {
        return Err("keyboard_designer FILE [--watch] [--size WxH] [--dark]".into());
    }
    Ok(o)
}

/// `config`'s `page` drawn as it shows, over a field, to PNG bytes.
fn draw(config: &KeyboardConfig, page: Page, o: &Options, fonts: &DrawingFonts) -> Vec<u8> {
    let (w, h) = o.size;
    let mut ui = Ui::new();
    if o.dark {
        ui.set_system_appearance(Appearance::Dark);
    }
    let mut input: Input<TextField> = Input::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; padding: 16px; font-family: Arial; font-size: 15px; });
    ui.update_style(root, |s| {
        s.background = Some(if o.dark {
            [0.07, 0.07, 0.08, 1.0]
        } else {
            [0.95, 0.95, 0.97, 1.0]
        });
    });
    let field = ui.add_box(
        root,
        style! { height: 36px; border-radius: 8px; padding: 8px 10px; box-sizing: border-box; background: #ffffff; color: #111111; border: 1px solid #c8c8cc; },
    );
    ui.set_semantics(field, Semantics::new(Role::TextBox).label("Field"));
    input.on_key(field, |f: &mut TextField, _, key, m, _| {
        !matches!(f.key(key, m, |_| true), FieldKey::Ignored)
    });
    let kb = VirtualKeyboard::with_config(&mut ui, &mut input, config.clone());
    kb.set_enabled(true);
    kb.set_layouts(&mut ui, vec![KeyboardLayout::english()]);
    kb.set_page(page);
    let mut state = TextField::new("");
    input.set_focus(&mut ui, &mut state, Some(field));
    for _ in 0..3 {
        input.tick(&mut ui, &mut state, 0.0);
        ui.step(0.3);
        ui.layout(w, h, &mut FixedWidthText(7.0));
    }
    ui_to_png(&mut ui, w, h, 2.0, fonts)
}

/// Checks, sketches and draws the file once; false if it would not read.
fn render(o: &Options, fonts: &DrawingFonts) -> bool {
    let started = Instant::now();
    let text = match std::fs::read_to_string(&o.file) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{}: {e}", o.file);
            return false;
        }
    };
    let config = match KeyboardConfig::parse(&text) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {e}", o.file);
            return false;
        }
    };
    for issue in config.validate() {
        eprintln!("{}: warning: {issue}", o.file);
    }
    let stem = Path::new(&o.file)
        .file_stem()
        .map_or("keyboard".into(), |s| s.to_string_lossy().into_owned());
    std::fs::create_dir_all("out/keyboards").unwrap();
    let english = KeyboardLayout::english();
    for (page, name, rows) in [
        (Page::Letters, "letters", &config.letters),
        (Page::Symbols, "symbols", &config.symbols),
        (Page::More, "more", &config.more),
    ] {
        if rows.is_empty() {
            continue;
        }
        println!("── {name}\n{}", config.sketch(page, &english, 1));
        let path = format!("out/keyboards/{stem}-{name}.png");
        std::fs::write(&path, draw(&config, page, o, fonts)).unwrap();
        println!("   {path}");
    }
    println!("drawn in {:.0} ms", started.elapsed().as_secs_f64() * 1e3);
    true
}

fn modified(path: &str) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

fn main() {
    let o = match options() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let fonts = DrawingFonts::new();
    let ok = render(&o, &fonts);
    if !o.watch {
        std::process::exit(if ok { 0 } else { 1 });
    }
    println!("watching {} (ctrl-c to stop)", o.file);
    let mut seen = modified(&o.file);
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let now = modified(&o.file);
        if now != seen {
            seen = now;
            println!();
            render(&o, &fonts);
        }
    }
}
