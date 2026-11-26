//! Gradients, glass and centring on the kit's components, light and dark,
//! drawn on the GPU: `cargo run --example ui_fills --features ui -- OUT_DIR`.

use threers::renderer::UiPass;
use threers::ui::components::ButtonKind;
use threers::ui::environment::Token;
use threers::ui::fill::{Fill, Gradient};
use threers::ui::input::Input;
use threers::ui::material::Material;
use threers::ui::states::Theme;
use threers::ui::view::*;
use threers::ui::Ui;

fn violet() -> Gradient {
    Gradient::linear(90.0, ["#7c3aed", "#ec4899"])
}

fn components() -> El<()> {
    vstack(
        10.0,
        [
            toggle("A switch", true, |_, _| {}),
            progress_view(Some(0.6)),
            hstack(
                8.0,
                [
                    button("Primary", |_| {}).button_style(ButtonKind::Primary),
                    button("Secondary", |_| {}),
                    pill("pill", Tint::Accent),
                    checkbox("Ticked", true, |_, _| {}),
                    avatar("AB", 28.0),
                ],
            ),
            segmented(["One", "Two"], Some(0), |_, _| {}),
            stepper("5", ("less", "more"), None, None),
        ],
    )
    .alignment(HorizontalAlignment::Leading)
    .padding(16.0)
    .corner_radius(14.0)
    .background(Token::Surface)
    .border(Token::Border, 1.0)
}

fn page() -> Vec<El<()>> {
    let blobs = El::div("position: absolute; left: 0; top: 0; width: 900px; height: 640px")
        .children([
            El::div("position: absolute; left: 480px; top: 330px; width: 150px; height: 150px; border-radius: 75px")
                .background_gradient(Gradient::radial(["#f59e0b", "#ef4444"])),
            El::div("position: absolute; left: 690px; top: 420px; width: 170px; height: 170px; border-radius: 85px")
                .background_gradient(Gradient::angular(0.0, ["#22d3ee", "#7c3aed", "#22d3ee"])),
        ]);
    let glass = |m: Material, name: &str| {
        text(name)
            .css("padding: 14px 18px; border-radius: 16px; font-size: 14px")
            .glass(m)
    };
    vec![
        blobs,
        hstack(
            20.0,
            [
                components().css("width: 400px"),
                components().css("width: 400px").tint(violet()),
            ],
        ),
        text("Gradient text, across the words")
            .css("font-size: 30px; font-weight: 700")
            .foreground_gradient(violet()),
        hstack(
            16.0,
            [
                gauge(0.7, "plain", "70%", 90.0),
                gauge_gradient(0.7, "gradient", "70%", 90.0, &violet()),
                text("Border")
                    .css("padding: 10px 18px")
                    .border_gradient(violet(), 2.0, 12.0, Token::Background),
                text("Of a token")
                    .css("padding: 10px 18px; border-radius: 12px; color: #ffffff")
                    .background_gradient(Gradient::of(Token::Accent)),
            ],
        ),
        hstack(
            12.0,
            [
                glass(Material::UltraThin, "Ultra thin"),
                glass(Material::Thin, "Thin"),
                glass(Material::Regular, "Regular"),
                glass(Material::Thick, "Thick"),
                text("Tinted")
                    .css("padding: 14px 18px; border-radius: 16px; font-size: 14px")
                    .glass_tinted(Material::Thin, Token::Accent),
            ],
        )
        .css("position: absolute; left: 440px; top: 440px"),
        El::div("position: absolute; left: 24px; top: 470px; width: 380px; height: 130px; display: flex; flex-direction: column; border-radius: 14px")
            .border(Token::Border, 1.0)
            .child(
                hstack(8.0, [text("Centred"), pill("both ways", Tint::Success)]).center(),
            ),
    ]
}

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let gpu = threers::gpu::GpuContext::headless_blocking().expect("a GPU");
    for (theme, name, themed) in [
        (Theme::Dark, "dark", false),
        (Theme::Light, "light", false),
        (Theme::Dark, "themed", true),
    ] {
        let mut ui = Ui::new();
        let mut input: Input<()> = Input::new();
        ui.set_theme(theme);
        if themed {
            ui.set_token_fill(
                Token::Accent,
                Some(Fill::Gradient(Gradient::linear(
                    90.0,
                    ["#22d3ee", "#34d399"],
                ))),
            );
            ui.set_token_fill(Token::Surface, Some(Fill::Glass(Material::Regular)));
        }
        let root = ui.root();
        ui.token_style(
            root,
            &[("background", Token::Background), ("color", Token::Text)],
        );
        let mut v = View::new(
            &mut ui,
            root,
            "display: flex; flex-direction: column; gap: 18px; padding: 24px; align-items: flex-start; width: 900px; height: 640px; box-sizing: border-box; position: relative",
        );
        v.update_with(&mut ui, &mut input, &(), page());
        let mut pass = UiPass::new(&gpu.device);
        ui.layout(900.0, 640.0, &mut pass.text_measure(1.0));
        let rgba = pass.render_to_rgba(
            &gpu.device,
            &gpu.queue,
            &ui.scene(),
            1.0,
            (900, 640),
            if theme == Theme::Light {
                [0.96, 0.96, 0.97, 1.0]
            } else {
                [0.07, 0.08, 0.1, 1.0]
            },
        );
        let path = format!("{dir}/ui_fills_{name}.png");
        std::fs::write(&path, threers::encode_png(900, 640, &rgba)).expect("written");
        println!("wrote {path}");
    }
}
