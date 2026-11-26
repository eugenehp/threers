//! A gallery of the SwiftUI-style views (`threers::ui::view`), painted to
//! PNGs in the light and dark appearances, without a window:
//!
//! ```text
//! cargo run --example swiftui_kit --features ui -- /tmp
//! ```

use threers::ui::components::ButtonKind;
use threers::ui::environment::Token;
use threers::ui::input::Input;
use threers::ui::states::Appearance;
use threers::ui::view::{
    button, content_unavailable, form, group_box, hstack, label, labeled_content, link, list,
    progress_view, section, spacer, text, toggle, vstack, zstack, El, Font, HorizontalAlignment,
    View,
};
use threers::ui::{paint_to_rgba, OverlayPainter, Ui};

#[derive(Default)]
struct Gallery {
    sounds: bool,
}

fn body(g: &Gallery) -> Vec<El<Gallery>> {
    vec![vstack(
        16.0,
        [
            text("Settings").font(Font::LargeTitle),
            form([
                section(
                    "Account",
                    [
                        labeled_content("Name", "Ada Lovelace"),
                        labeled_content("Plan", "Free"),
                        toggle("Sounds", g.sounds, |g: &mut Gallery, on| g.sounds = on),
                        toggle("Notifications", true, |_: &mut Gallery, _| {}),
                    ],
                ),
                section(
                    "Recent",
                    [list([
                        label("Focus session", "🧠"),
                        label("Sleep", "🌙"),
                        label("Calibration", "🎯"),
                    ])],
                ),
            ]),
            group_box(
                "Download",
                [
                    progress_view(Some(0.6)),
                    hstack(
                        8.0,
                        [
                            text("60% of 1.2 GB")
                                .font(Font::Footnote)
                                .foreground(Token::SecondaryText),
                            spacer(),
                            button("Cancel", |_: &mut Gallery| {}).button_style(ButtonKind::Plain),
                            button("Pause", |_: &mut Gallery| {}),
                            button("Open", |_: &mut Gallery| {}).button_style(ButtonKind::Primary),
                        ],
                    )
                    .fill_width(),
                ],
            )
            .fill_width(),
            hstack(
                12.0,
                [
                    zstack([
                        El::div("width: 56px; height: 56px; border-radius: 28px")
                            .background(Token::Accent),
                        text("AL").bold().foreground(Token::OnAccent),
                    ]),
                    vstack(
                        2.0,
                        [
                            text("Ada Lovelace").font(Font::Headline),
                            link("View profile", |_: &mut Gallery| {}),
                        ],
                    )
                    .alignment(HorizontalAlignment::Leading),
                    spacer(),
                    button("Delete", |_: &mut Gallery| {}).button_style(ButtonKind::Destructive),
                    button("Disabled", |_: &mut Gallery| {}).disabled(true),
                ],
            )
            .fill_width(),
            content_unavailable("🔍", "No results", "Try another search, or fewer words."),
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    .padding(24.0)]
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    // Emoji in colour: a UI opts in, before its first text.
    threers::CaptionFont::use_color_emoji();
    let (w, h) = (560u32, 900u32);
    for (appearance, name) in [(Appearance::Light, "light"), (Appearance::Dark, "dark")] {
        let mut ui = Ui::new();
        ui.set_system_appearance(appearance);
        let mut input: Input<Gallery> = Input::new();
        let root = ui.root();
        let back = ui.add_box(root, threers::ui::style! { width: 100%; height: 100%; });
        ui.token_style(
            back,
            &[("background", Token::Background), ("color", Token::Text)],
        );
        let mut view = View::new(&mut ui, back, "display: flex; flex-direction: column");
        view.update(&mut ui, &mut input, body(&Gallery::default()));
        let mut overlay = threers::overlay::Overlay::new(w, h);
        ui.layout(w as f32, h as f32, &mut OverlayPainter::new(&mut overlay));
        let path = format!("{dir}/swiftui-kit-{name}.png");
        std::fs::write(
            &path,
            threers::encode_png(w, h, &paint_to_rgba(&ui, 1.0, w, h)),
        )
        .unwrap();
        println!("wrote {path}");
    }
}
