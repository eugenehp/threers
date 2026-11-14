//! Border beams (`threers::ui::border_beam`) as the GPU draws them: every
//! kind compiles, lights part of the border and not the rest, stays inside
//! its box, moves on, and fades out when turned off; and the shader
//! translates for WebGL2. `BEAM_PNG=path` writes the picture out.
#![cfg(not(target_arch = "wasm32"))]

use threers::renderer::UiPass;
use threers::style;
use threers::ui::border_beam::{
    border_beam, source, BeamDirection, BeamEdge, BeamKind, BeamPalette, BeamShape, BeamStyle,
    BorderBeam,
};
use threers::ui::shader::Shader;
use threers::ui::{Rect, Ui};

const W: u32 = 420;
const H: u32 = 360;

struct Picture {
    rgb: Vec<u8>,
}

impl Picture {
    fn at(&self, x: f32, y: f32) -> [u8; 3] {
        let i = (y as usize * W as usize + x as usize) * 3;
        [self.rgb[i], self.rgb[i + 1], self.rgb[i + 2]]
    }

    fn bright(&self, x: f32, y: f32) -> u8 {
        *self.at(x, y).iter().max().unwrap()
    }

    /// Brightness at points round the box `r`, `inset` pixels in from its
    /// edge (corners left out, so any radius will do), with where they are.
    fn round(&self, r: Rect, inset: f32) -> Vec<((f32, f32), u8)> {
        let mut out = vec![];
        let (x0, y0) = (r.x + inset, r.y + inset);
        let (x1, y1) = (r.x + r.width - 1.0 - inset, r.y + r.height - 1.0 - inset);
        let corner = 30.0;
        let mut x = r.x + corner;
        while x < r.x + r.width - corner {
            out.push(((x, y0), self.bright(x, y0)));
            out.push(((x, y1), self.bright(x, y1)));
            x += 2.0;
        }
        let mut y = r.y + corner;
        while y < r.y + r.height - corner {
            out.push(((x0, y), self.bright(x0, y)));
            out.push(((x1, y), self.bright(x1, y)));
            y += 2.0;
        }
        out
    }
}

fn styles() -> [BeamStyle; 4] {
    [
        BeamStyle::new(BeamKind::Large).shape(BeamShape::Rounded(20.0)),
        BeamStyle::new(BeamKind::Large)
            .shape(BeamShape::Rounded(12.0))
            .palette(BeamPalette::NEON)
            .direction(BeamDirection::CounterClockwise),
        BeamStyle::new(BeamKind::Small)
            .shape(BeamShape::Capsule)
            .palette(BeamPalette::SUNSET),
        BeamStyle::new(BeamKind::Line)
            .shape(BeamShape::Capsule)
            .edge(BeamEdge::Bottom)
            .palette(BeamPalette::SPECTRUM),
    ]
}

/// The beams at `time` seconds, turned off at `off` if given; the picture,
/// where each is, and the pass.
fn draw(time: f32, off: Option<f32>) -> Option<(Picture, Vec<Rect>, UiPass)> {
    let gpu = threers::gpu::testing::context("the UI pass")?;
    // On a Mac the pass must run on Metal, not on a fallback.
    #[cfg(target_os = "macos")]
    assert_eq!(gpu.adapter.get_info().backend, wgpu::Backend::Metal);
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 24px; padding: 30px; background: #0b0d12; });
    let beams: Vec<BorderBeam> = styles()
        .into_iter()
        .map(|st| {
            let b = border_beam(&mut ui, root, st);
            let (w, h) = match st.kind {
                BeamKind::Small => (160.0, 40.0),
                BeamKind::Line => (300.0, 44.0),
                BeamKind::Large => (300.0, 70.0),
            };
            ui.update_style(b.node, |s| {
                s.width = threers::ui::Length::Px(w);
                s.height = threers::ui::Length::Px(h);
            });
            b
        })
        .collect();
    // Laid out first: beams come on as the styles are laid out, then run.
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    match off {
        Some(at) => {
            ui.step(at);
            for b in &beams {
                b.set_active(&mut ui, false);
            }
            ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
            ui.step(time - at);
        }
        None => {
            ui.step(time);
        }
    }
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (W, H),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("BEAM_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(W, H, &image)).unwrap();
    }
    let rgb = image.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let rects = beams.iter().map(|b| ui.rect(b.node)).collect();
    Some((Picture { rgb }, rects, pass))
}

#[test]
fn every_kind_lights_part_of_its_border() {
    let Some((pic, rects, pass)) = draw(0.7, None) else {
        return;
    };
    let stats = pass.stats();
    assert_eq!((stats.effects, stats.failed_effects), (4, 0), "{stats:?}");
    // The surface is #1c1c1c: 28.
    for (r, st) in rects.iter().zip(styles()) {
        let what = format!("{:?} {:?}", st.kind, st.shape);
        let round = pic.round(*r, 1.5);
        let lit = round.iter().filter(|(_, b)| *b > 120).count();
        let dark = round.iter().filter(|(_, b)| *b < 50).count();
        let most = round.iter().map(|(_, b)| *b).max().unwrap();
        assert!(
            most > 170,
            "{what}: the beam is bright on the edge ({most})"
        );
        assert!(lit >= 3, "{what}: a stretch of edge is lit ({lit})");
        assert!(
            dark * 3 > round.len(),
            "{what}: much of the edge is not ({dark} of {})",
            round.len()
        );
        // The middle shows the surface, and nothing lights the page
        // outside the box.
        let mid = pic.bright(r.x + r.width * 0.5, r.y + r.height * 0.5);
        assert!(mid < 60, "{what}: the middle is the surface ({mid})");
        let mut outside = 0u8;
        let mut x = r.x;
        while x < r.x + r.width {
            outside = outside.max(pic.bright(x, r.y - 4.0));
            outside = outside.max(pic.bright(x, r.y + r.height + 3.0));
            x += 1.0;
        }
        assert!(outside < 30, "{what}: nothing past the box ({outside})");
    }
}

/// The brightest place on each beam's border.
fn heads(pic: &Picture, rects: &[Rect]) -> Vec<(f32, f32)> {
    rects
        .iter()
        .map(|r| {
            pic.round(*r, 1.5)
                .into_iter()
                .max_by_key(|(_, b)| *b)
                .unwrap()
                .0
        })
        .collect()
}

#[test]
fn the_beams_move_on() {
    let (Some((a, rects, _)), Some((b, _, _))) = (draw(0.3, None), draw(0.9, None)) else {
        return;
    };
    for (i, (p, q)) in heads(&a, &rects)
        .into_iter()
        .zip(heads(&b, &rects))
        .enumerate()
    {
        let moved = (p.0 - q.0).abs() + (p.1 - q.1).abs();
        assert!(moved > 20.0, "{:?}: {p:?} then {q:?}", styles()[i].kind);
    }
}

#[test]
fn turned_off_they_fade_away() {
    let Some((pic, rects, _)) = draw(2.0, Some(0.5)) else {
        return;
    };
    for (r, st) in rects.iter().zip(styles()) {
        let most = pic
            .round(*r, 1.5)
            .into_iter()
            .map(|(_, b)| b)
            .max()
            .unwrap();
        assert!(
            most < 60,
            "{:?}: only the hairline is left ({most})",
            st.kind
        );
    }
}

#[test]
fn the_beam_translates_for_webgl2() {
    translates(&Shader::new(source()), "the border beam");
}

fn translates(shader: &Shader, what: &str) {
    use wgpu::naga;
    assert_eq!(shader.error(), None);
    let module_source = UiPass::effect_wgsl(shader);
    let module = naga::front::wgsl::parse_str(&module_source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&module_source)));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "threers_vs"),
        (naga::ShaderStage::Fragment, "threers_fs"),
    ] {
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Embedded {
                version: 300,
                is_webgl: true,
            },
            ..Default::default()
        };
        let pipeline = naga::back::glsl::PipelineOptions {
            shader_stage: stage,
            entry_point: entry.to_string(),
            multiview: None,
        };
        let mut out = String::new();
        naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline,
            Default::default(),
        )
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
        .write()
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let bytes = out.as_bytes();
        let bad = out
            .match_indices("gl_")
            .filter(|(i, _)| {
                *i == 0 || !(bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_')
            })
            .filter(|(i, _)| !bytes.get(i + 3).is_some_and(|c| c.is_ascii_uppercase()))
            .count();
        assert_eq!(bad, 0, "{what} {entry}: GLSL names ANGLE rejects");
    }
}

/// What `build` puts on a dark page, drawn at `time`: the picture and where
/// the boxes it returns are.
fn render(
    time: f32,
    build: impl FnOnce(&mut Ui) -> Vec<threers::ui::NodeId>,
) -> Option<(Picture, Vec<Rect>)> {
    let gpu = threers::gpu::testing::context("the UI pass")?;
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 24px; padding: 30px; background: #0b0d12; });
    let nodes = build(&mut ui);
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    ui.step(time);
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (W, H),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("BEAM_STYLES_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(W, H, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0, "{:?}", pass.stats());
    let rgb = image.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let rects = nodes.iter().map(|&n| ui.rect(n)).collect();
    Some((Picture { rgb }, rects))
}

/// Beams set as styles: from CSS, a Tailwind class, and `set_border_beam`,
/// each following its box's own corners.
#[test]
fn beams_from_css_and_tailwind_draw() {
    let Some((pic, rects)) = render(0.5, |ui| {
        let root = ui.root();
        let css = ui.add_box(root, style! { width: 300px; height: 70px; border-radius: 18px; background: #1c1c1c; border-beam: large neon; });
        let tw = ui.add_tw(
            root,
            "w-[300px] h-[70px] rounded-full bg-[#1c1c1c] beam-line beam-sunset",
        );
        let rust = ui.add_box(
            root,
            style! { width: 300px; height: 70px; border-radius: 8px; background: #1c1c1c; },
        );
        ui.set_border_beam(
            rust,
            BeamStyle::default().palette(BeamPalette::AURORA).count(2),
        );
        vec![css, tw, rust]
    }) else {
        return;
    };
    for (i, r) in rects.iter().enumerate() {
        let most = pic
            .round(*r, 1.5)
            .into_iter()
            .map(|(_, b)| b)
            .max()
            .unwrap();
        assert!(most > 170, "beam {i} lights its edge ({most})");
    }
}

/// Held still at 90 degrees: the head sits on the right side, and nothing
/// lights the left.
#[test]
fn a_still_beam_sits_at_its_angle() {
    let Some((pic, rects)) = render(1.3, |ui| {
        let root = ui.root();
        let n = ui.add_box(
            root,
            style! { width: 300px; height: 100px; border-radius: 12px; background: #1c1c1c; },
        );
        ui.set_border_beam(n, BeamStyle::default().angle(90.0).speed(0.0).length(0.15));
        vec![n]
    }) else {
        return;
    };
    let r = rects[0];
    let x_right = r.x + r.width - 2.5;
    let mut right = 0u8;
    let mut left = 0u8;
    let mut y = r.y + 20.0;
    while y < r.y + r.height - 20.0 {
        right = right.max(pic.bright(x_right, y));
        left = left.max(pic.bright(r.x + 2.5, y));
        y += 1.0;
    }
    assert!(right > 150 && left < 50, "right {right}, left {left}");
}

/// Two comets: lit on opposite sides at once; and a wider line lights
/// deeper in.
#[test]
fn two_comets_and_a_wide_line() {
    let Some((pic, rects)) = render(0.0, |ui| {
        let root = ui.root();
        let two = ui.add_box(
            root,
            style! { width: 300px; height: 100px; background: #1c1c1c; },
        );
        ui.set_border_beam(
            two,
            BeamStyle::default()
                .angle(0.0)
                .speed(0.0)
                .count(2)
                .length(0.2)
                .glow(0.0),
        );
        let thin = ui.add_box(
            root,
            style! { width: 300px; height: 60px; background: #1c1c1c; },
        );
        ui.set_border_beam(
            thin,
            BeamStyle::new(BeamKind::Line)
                .edge(BeamEdge::Top)
                .speed(0.0)
                .angle(0.0)
                .width(1.0)
                .glow(0.0),
        );
        let wide = ui.add_box(
            root,
            style! { width: 300px; height: 60px; background: #1c1c1c; },
        );
        ui.set_border_beam(
            wide,
            BeamStyle::new(BeamKind::Line)
                .edge(BeamEdge::Top)
                .speed(0.0)
                .angle(0.0)
                .width(5.0)
                .glow(0.0),
        );
        vec![two, thin, wide]
    }) else {
        return;
    };
    // Heads at the top middle and the bottom middle, tails behind them.
    let r = rects[0];
    let top = pic.bright(r.x + r.width * 0.5 - 20.0, r.y + 1.0);
    let bottom = pic.bright(r.x + r.width * 0.5 + 20.0, r.y + r.height - 2.0);
    assert!(top > 120 && bottom > 120, "two comets: {top} and {bottom}");
    // A line beam held at its start sits at the left end of its edge.
    let deep = |r: Rect| pic.bright(r.x + r.width * 0.14, r.y + 4.0);
    let (thin, wide) = (deep(rects[1]), deep(rects[2]));
    assert!(wide > thin + 60, "4px in: wide {wide}, thin {thin}");
}
