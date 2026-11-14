//! The glowing SDF progress indicators (`threers::ui::indicators`) as the
//! GPU draws them: every kind compiles, the fill is lit up to the value and
//! dark past it, the glow reaches out of the box, and each module translates
//! for WebGL2. `INDICATORS_PNG=path` writes the picture out to look at.
#![cfg(not(target_arch = "wasm32"))]

use threers::renderer::UiPass;
use threers::style;
use threers::ui::indicators::{
    indicator, source, source_filled, Depth, Fill, IndicatorStyle, Label, LabelAlign, Motion,
    Palette, Preset, Shape,
};
use threers::ui::{Rect, Ui};

/// A custom fill: stripes sliding along, over the library's noise.
const STRIPES: &str = "
fn stripes(along: f32, up: f32, half: f32, env: f32, cover: f32, u: f32, now: f32, c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>) -> vec4<f32> {
    let band = step(0.5, fract((along + up) / (half * 2.0) - now));
    let grain = ui_fbm(vec2<f32>(along, up) * 0.2);
    return vec4<f32>(ui_ramp3(c0, c1, c2, u) * (0.6 + 0.4 * band) * (0.9 + 0.2 * grain), 1.0) * cover;
}";

const W: u32 = 460;
const H: u32 = 520;

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
}

fn kinds() -> [(Shape, Depth); 4] {
    [
        (Shape::Bar, Depth::Flat),
        (Shape::Bar, Depth::Solid),
        (Shape::Ring, Depth::Flat),
        (Shape::Ring, Depth::Solid),
    ]
}

/// Each kind at 60%, and busy; the picture, and where the indicators are
/// (at 60% first, then busy, in [`kinds`]' order).
fn draw(time: f64) -> Option<(Picture, Vec<Rect>, UiPass)> {
    let gpu = threers::gpu::testing::context("the UI pass")?;
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    // A dark page: the indicators glow as on one.
    ui.set_system_appearance(threers::ui::states::Appearance::Dark);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 28px; padding: 30px; background: #0b0d12; });
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; gap: 28px; width: 300px; },
    );
    let rings = ui.add_box(
        root,
        style! { display: flex; flex-wrap: wrap; gap: 28px 20px; width: 400px; },
    );
    let palettes = [
        Palette::AURORA,
        Palette::EMBER,
        Palette::NEON,
        Palette::OCEAN,
    ];
    let mut out = vec![];
    for busy in [false, true] {
        for (i, (shape, depth)) in kinds().into_iter().enumerate() {
            let parent = if shape == Shape::Bar { bars } else { rings };
            let mut st = IndicatorStyle::new(shape).depth(depth).palette(palettes[i]);
            if shape == Shape::Bar && depth == Depth::Solid {
                st = st.size(18.0);
            }
            if shape == Shape::Ring {
                st = st.size(80.0).thickness(0.3);
            }
            out.push(indicator(
                &mut ui,
                parent,
                "Loading",
                st,
                if busy { None } else { Some(0.6) },
            ));
        }
    }
    ui.step(time as f32);
    ui.layout(W as f32, H as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (W, H),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("INDICATORS_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(W, H, &image)).unwrap();
    }
    let rgb = image.chunks(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
    let rects = out.iter().map(|ind| ui.rect(ind.node)).collect();
    Some((Picture { rgb }, rects, pass))
}

#[test]
fn every_kind_compiles_and_draws() {
    let Some((pic, rects, pass)) = draw(0.7) else {
        return;
    };
    let stats = pass.stats();
    assert_eq!((stats.effects, stats.failed_effects), (8, 0), "{:?}", stats);
    for (i, (shape, depth)) in kinds().into_iter().enumerate() {
        let r = rects[i];
        let what = format!("{shape:?} {depth:?}");
        match shape {
            Shape::Bar | Shape::Battery => {
                let y = r.y + r.height * 0.5;
                let lit = pic.bright(r.x + r.width * 0.3, y);
                let dark = pic.bright(r.x + r.width * 0.9, y);
                assert!(
                    lit > 150,
                    "{what}: the fill is lit, {:?}",
                    pic.at(r.x + r.width * 0.3, y)
                );
                assert!(
                    dark < 110,
                    "{what}: past the value is dim, {:?}",
                    pic.at(r.x + r.width * 0.9, y)
                );
                // Light spills above the box, over the fill and not past it.
                let above = pic.bright(r.x + r.width * 0.3, r.y - 3.0);
                let above_past = pic.bright(r.x + r.width * 0.9, r.y - 3.0);
                assert!(
                    above > 30 && above > above_past + 15,
                    "{what}: a glow round the fill, {above} vs {above_past}"
                );
            }
            Shape::Ring => {
                // Round the middle of the stroke: 60% lit from the top,
                // clockwise. A solid ring is tipped back, so its points sit
                // nearer the middle vertically.
                let (cx, cy) = (r.x + r.width * 0.5, r.y + r.height * 0.5);
                let tube = 80.0 * 0.5 * 0.3;
                let rr = 40.0 - tube * 0.5 - 1.0;
                let squash = if depth == Depth::Solid {
                    0.5f32.cos()
                } else {
                    1.0
                };
                let point = |turn: f32| {
                    let a = turn * std::f32::consts::TAU;
                    (cx + rr * a.sin(), cy - rr * a.cos() * squash)
                };
                let (lx, ly) = point(0.25);
                let (dx, dy) = point(0.8);
                assert!(
                    pic.bright(lx, ly) > 150,
                    "{what}: the arc is lit, {:?}",
                    pic.at(lx, ly)
                );
                assert!(
                    pic.bright(dx, dy) < 110,
                    "{what}: past the value is dim, {:?}",
                    pic.at(dx, dy)
                );
                assert!(
                    pic.bright(cx, cy) < 90,
                    "{what}: the middle is clear, {:?}",
                    pic.at(cx, cy)
                );
            }
        }
    }
    // Busy ones draw something somewhere.
    for (i, r) in rects[4..].iter().enumerate() {
        let mut most = 0u8;
        let mut y = r.y;
        while y < r.y + r.height {
            let mut x = r.x;
            while x < r.x + r.width {
                most = most.max(pic.bright(x, y));
                x += 1.0;
            }
            y += 1.0;
        }
        assert!(most > 150, "busy {:?}: lit somewhere ({most})", kinds()[i]);
    }
}

/// A busy indicator moves: two moments, two pictures.
#[test]
fn a_busy_indicator_moves_on() {
    let Some((a, rects, _)) = draw(0.3) else {
        return;
    };
    let Some((b, _, _)) = draw(0.9) else { return };
    for (i, r) in rects[4..].iter().enumerate() {
        let mut changed = 0;
        for y in (r.y as usize)..((r.y + r.height) as usize) {
            for x in (r.x as usize)..((r.x + r.width) as usize) {
                if a.at(x as f32, y as f32) != b.at(x as f32, y as f32) {
                    changed += 1;
                }
            }
        }
        assert!(
            changed > 50,
            "busy {:?}: {changed} pixels changed",
            kinds()[i]
        );
    }
}

#[test]
fn every_kind_translates_for_webgl2() {
    for (shape, depth) in kinds() {
        for animated in [false, true] {
            translates(
                &threers::ui::shader::Shader::new(source(shape, depth, animated)),
                &format!("{shape:?} {depth:?}"),
            );
        }
    }
    for depth in [Depth::Flat, Depth::Solid] {
        for animated in [false, true] {
            translates(
                &threers::ui::shader::Shader::new(source(Shape::Battery, depth, animated)),
                &format!("battery {depth:?}"),
            );
        }
    }
    for shape in [Shape::Bar, Shape::Ring] {
        translates(
            &threers::ui::shader::Shader::new(source_filled(
                shape,
                Depth::Flat,
                Fill::Electric,
                true,
            )),
            &format!("electric {shape:?}"),
        );
    }
    for fill in [Fill::Fire, Fill::Water, Fill::Plasma, Fill::Frost] {
        for shape in [Shape::Bar, Shape::Ring] {
            translates(
                &threers::ui::shader::Shader::new(source_filled(shape, Depth::Flat, fill, true)),
                &format!("{fill:?} {shape:?}"),
            );
        }
    }
    translates(&threers::ui::solid::shader(), "a solid box");
    // The whole shader library, in a user's effect.
    let lib = threers::ui::ui_shader! {
        use Noise, Color, Sdf, Fx;
        fn everything(p: vec2<f32>, size: vec2<f32>, time: f32) -> vec4<f32> {
            let v = ui_voronoi(p * 0.1);
            let c = ui_hsl(v.x * 360.0, 0.8, 0.5) + ui_blackbody(3000.0) * ui_fbm(p * 0.05) + ui_to_srgb(ui_to_linear(vec3<f32>(0.5)));
            let d = min(ui_rbox(p, size * 0.5, size * 0.3, 6.0), ui_ring(p, size * 0.5, 20.0, 4.0));
            let k = vec4<f32>(1.0);
            let f = ui_fire(p.x, p.y, 6.0, 1.0, 0.0, 0.5, time, k, k, k) + ui_water(p.x, p.y, 6.0, 1.0, 1.0, 0.5, time, k, k, k)
                + ui_plasma(p.x, p.y, 6.0, 1.0, 1.0, 0.5, time, k, k, k) + ui_frost(p.x, p.y, 6.0, 1.0, 1.0, 0.5, time, k, k, k);
            return vec4<f32>(c * ui_cover(d, 1.0) + vec3<f32>(ui_segment(p, vec2<f32>(0.0), size) * 0.0), 1.0) + f * 0.0;
        }
    };
    translates(&lib, "the shader library");
    let stripes = Fill::custom(STRIPES);
    assert_eq!(stripes.custom_error(), None);
    for shape in [Shape::Bar, Shape::Ring] {
        translates(
            &threers::ui::shader::Shader::new(source_filled(shape, Depth::Flat, stripes, true)),
            &format!("a custom fill, {shape:?}"),
        );
    }
}

/// Boxes drawn solid: lit at the rim, darker under the light's far side,
/// shadowed below. `SOLID_PNG=path` writes them out.
#[test]
fn solid_boxes_are_lit_and_shadowed() {
    use threers::ui::solid::Depth;
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-wrap: wrap; align-items: center; gap: 30px; padding: 30px; background: #e5e7eb; });
    let mut boxes = vec![];
    for (i, css) in ["#2563eb", "#16a34a", "#e11d48", "#f59e0b"]
        .iter()
        .enumerate()
    {
        let b = ui.add_box(root, style! { width: 140px; height: 60px; border-radius: 14px; color: #fff; font-size: 16px; display: flex; align-items: center; justify-content: center; });
        let c = threers::ui::css::color(css).unwrap();
        ui.update_style(b, |s| s.background = Some(c));
        ui.add_text(b, "Button");
        if i % 2 == 0 {
            ui.set_depth(b, Depth::Solid);
        }
        boxes.push(b);
    }
    // A plain ring, solid: shaped by its stroke, not its square box.
    let ring = threers::ui::indicators::progress_view(
        &mut ui,
        root,
        "Sync",
        threers::ui::indicators::PlainStyle::new(threers::ui::indicators::PlainKind::Circular)
            .size(56.0)
            .stroke(8.0),
        Some(0.7),
    )
    .labelled(
        &mut ui,
        threers::ui::indicators::Label::percent().size(14.0),
    );
    ui.set_depth(ring.node, Depth::Solid);
    let glowing = indicator(
        &mut ui,
        root,
        "Up",
        IndicatorStyle::new(Shape::Ring)
            .depth(Depth::Solid)
            .size(90.0)
            .thickness(0.3)
            .palette(Palette::EMBER),
        Some(0.42),
    )
    .labelled(
        &mut ui,
        threers::ui::indicators::Label::percent().font("monospace"),
    );
    let _ = glowing;
    ui.layout(520.0, 300.0, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (520, 300),
        [0.9, 0.9, 0.92, 1.0],
    );
    if let Some(path) = std::env::var_os("SOLID_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(520, 300, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
    let px = |x: f32, y: f32| {
        let i = (y as usize * 520 + x as usize) * 4;
        [image[i] as i32, image[i + 1] as i32, image[i + 2] as i32]
    };
    let solid = ui.rect(boxes[0]);
    let flat = ui.rect(boxes[1]);
    // The top rim, lit, against the flat one's.
    let top_solid = px(solid.x + solid.width * 0.5, solid.y + 2.0);
    let mid_solid = px(solid.x + 20.0, solid.y + solid.height * 0.5);
    assert!(
        top_solid.iter().sum::<i32>() > mid_solid.iter().sum::<i32>() + 30,
        "a lit rim: {top_solid:?} over {mid_solid:?}"
    );
    // A shadow under the solid one, none under the flat one.
    let under_solid = px(solid.x + solid.width * 0.5, solid.y + solid.height + 3.0);
    let under_flat = px(flat.x + flat.width * 0.5, flat.y + flat.height + 3.0);
    assert!(
        under_solid.iter().sum::<i32>() + 20 < under_flat.iter().sum::<i32>(),
        "{under_solid:?} vs {under_flat:?}"
    );
}

fn translates(shader: &threers::ui::shader::Shader, what: &str) {
    use wgpu::naga;
    {
        {
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
    }
}

/// The glass ones large, to look at: `GLASS_PNG=path`.
#[test]
fn glass_close_up() {
    let Some(path) = std::env::var_os("GLASS_PNG") else {
        return;
    };
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    // A dark page: the indicators glow as on one.
    ui.set_system_appearance(threers::ui::states::Appearance::Dark);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 40px; padding: 40px; background: #0b0d12; });
    let value: f32 = std::env::var("GLASS_VALUE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.6);
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; gap: 40px; width: 520px; },
    );
    let motion = match std::env::var("GLASS_MOTION").as_deref() {
        Ok("spin") => Motion::Spin,
        Ok("rock") => Motion::Rock,
        Ok("roll") => Motion::Roll,
        Ok("bounce") => Motion::Bounce,
        Ok("buzz") => Motion::Buzz,
        Ok("pulse") => Motion::Pulse,
        Ok("float") => Motion::Float,
        _ => Motion::None,
    };
    let time: f32 = std::env::var("GLASS_TIME")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    indicator(
        &mut ui,
        bars,
        "a",
        IndicatorStyle::new(Shape::Bar)
            .size(20.0)
            .palette(Palette::AURORA)
            .motion(motion),
        Some(value),
    );
    indicator(
        &mut ui,
        bars,
        "b",
        IndicatorStyle::new(Shape::Bar)
            .depth(Depth::Solid)
            .size(40.0)
            .palette(Palette::EMBER)
            .motion(motion),
        Some(value),
    );
    let rings = ui.add_box(root, style! { display: flex; gap: 40px; });
    indicator(
        &mut ui,
        rings,
        "c",
        IndicatorStyle::new(Shape::Ring)
            .size(240.0)
            .thickness(0.25)
            .palette(Palette::AURORA)
            .motion(motion),
        Some(value),
    );
    indicator(
        &mut ui,
        rings,
        "d",
        IndicatorStyle::new(Shape::Ring)
            .depth(Depth::Solid)
            .size(240.0)
            .thickness(0.3)
            .palette(Palette::EMBER)
            .motion(motion),
        Some(value),
    );
    ui.step(time);
    ui.layout(600.0, 520.0, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (600, 520),
        [0.0, 0.0, 0.0, 1.0],
    );
    std::fs::write(path, threers::utils::png::encode_png(600, 520, &image)).unwrap();
}

/// Batteries and electricity, drawn: `EXTRAS_PNG=path`.
#[test]
fn batteries_and_electricity() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    // A dark page: the indicators glow as on one.
    ui.set_system_appearance(threers::ui::states::Appearance::Dark);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 34px; padding: 34px; background: #0b0d12; });
    let row = ui.add_box(
        root,
        style! { display: flex; gap: 60px; align-items: center; },
    );
    let white = [1.0, 1.0, 1.0, 1.0];
    for v in [0.12, 0.38, 0.86] {
        indicator(
            &mut ui,
            row,
            "Battery",
            Preset::Battery.style().size(36.0),
            Some(v),
        )
        .labelled(
            &mut ui,
            Label::percent().align(LabelAlign::Below).color(white),
        );
    }
    let row2 = ui.add_box(
        root,
        style! { display: flex; gap: 60px; align-items: center; },
    );
    indicator(
        &mut ui,
        row2,
        "Charging",
        Preset::Charging.style().size(40.0),
        Some(0.64),
    );
    indicator(
        &mut ui,
        row2,
        "Battery",
        IndicatorStyle::new(Shape::Battery)
            .depth(Depth::Solid)
            .palette(Palette::BATTERY)
            .size(40.0),
        Some(0.3),
    );
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; width: 420px; },
    );
    let lightning = indicator(
        &mut ui,
        bars,
        "Lightning",
        Preset::Lightning.style().size(14.0),
        Some(0.7),
    );
    let row3 = ui.add_box(root, style! { display: flex; gap: 40px; });
    indicator(
        &mut ui,
        row3,
        "Tesla",
        Preset::Tesla.style().size(130.0),
        Some(0.8),
    );
    indicator(
        &mut ui,
        row3,
        "Tesla",
        Preset::Tesla.style().size(130.0).palette(Palette::EMBER),
        None,
    );
    ui.step(0.4);
    let (w, h) = (520u32, 560u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("EXTRAS_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
    // The bolts light the bar's filled part, not past it.
    let r = ui.rect(lightning.node);
    let bright = |x0: f32, x1: f32| {
        let mut most = 0u8;
        for y in (r.y as usize)..((r.y + r.height) as usize) {
            for x in (x0 as usize)..(x1 as usize) {
                let i = (y * w as usize + x) * 4;
                most = most.max(image[i].max(image[i + 1]).max(image[i + 2]));
            }
        }
        most
    };
    let (lit, past) = (
        bright(r.x + r.width * 0.2, r.x + r.width * 0.5),
        bright(r.x + r.width * 0.85, r.x + r.width * 0.98),
    );
    assert!(
        lit > 200 && (past as f32) < lit as f32 * 0.7,
        "{lit} along the fill, {past} past it"
    );
}

/// Solid boxes turned: walls showing, the front foreshortened, the back
/// past a half turn. `TURNED_PNG=path`.
#[test]
fn solid_boxes_turn_in_3d() {
    use threers::ui::depth::Transform3d;
    use threers::ui::solid::Depth;
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-wrap: wrap; align-items: center; gap: 50px; padding: 40px; background: #e5e7eb; });
    let mut boxes = vec![];
    for (i, (yaw, pitch)) in [
        (0.0, 0.0),
        (0.7, 0.0),
        (-0.9, 0.3),
        (0.0, 0.8),
        (2.6, 0.0),
        (1.2, -0.4),
    ]
    .iter()
    .enumerate()
    {
        let b = ui.add_box(root, style! { width: 150px; height: 70px; border-radius: 16px; color: #fff; font-size: 18px; font-weight: bold; display: flex; align-items: center; justify-content: center; });
        let c = [
            [0.15, 0.39, 0.92, 1.0],
            [0.09, 0.64, 0.29, 1.0],
            [0.88, 0.11, 0.28, 1.0],
            [0.96, 0.62, 0.04, 1.0],
            [0.49, 0.23, 0.93, 1.0],
            [0.05, 0.6, 0.7, 1.0],
        ][i];
        ui.update_style(b, |s| s.background = Some(c));
        ui.add_text(b, "Solid");
        ui.set_transform3d(
            b,
            Transform3d {
                rotate_y: *yaw,
                rotate_x: *pitch,
                perspective: 500.0,
                ..Transform3d::default()
            },
        );
        ui.set_depth(b, Depth::Solid);
        boxes.push(b);
    }
    let (w, h) = (760u32, 380u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    ui.step(0.0);
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.9, 0.9, 0.92, 1.0],
    );
    if let Some(path) = std::env::var_os("TURNED_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
}

/// The glowing ones on a light page: tinted haze, pale tracks, dark
/// battery outline, coloured bolts. `LIGHT_PNG=path`.
#[test]
fn on_a_light_page() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    ui.set_system_appearance(threers::ui::states::Appearance::Light);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-wrap: wrap; align-items: center; gap: 36px; padding: 36px; background: #f5f6f8; });
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; gap: 30px; width: 300px; },
    );
    indicator(
        &mut ui,
        bars,
        "a",
        IndicatorStyle::new(Shape::Bar)
            .palette(Palette::AURORA)
            .size(10.0),
        Some(0.6),
    );
    indicator(
        &mut ui,
        bars,
        "b",
        IndicatorStyle::new(Shape::Bar)
            .depth(Depth::Solid)
            .palette(Palette::EMBER)
            .size(22.0),
        Some(0.6),
    );
    indicator(&mut ui, bars, "c", Preset::Lightning.style(), Some(0.6));
    indicator(
        &mut ui,
        root,
        "d",
        IndicatorStyle::new(Shape::Ring)
            .palette(Palette::NEON)
            .size(100.0),
        Some(0.6),
    )
    .labelled(&mut ui, Label::percent());
    indicator(
        &mut ui,
        root,
        "e",
        IndicatorStyle::new(Shape::Ring)
            .depth(Depth::Solid)
            .palette(Palette::OCEAN)
            .size(100.0)
            .thickness(0.3),
        Some(0.6),
    );
    indicator(
        &mut ui,
        root,
        "f",
        Preset::Battery.style().size(34.0),
        Some(0.45),
    );
    ui.step(0.3);
    let (w, h) = (560u32, 330u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.96, 0.965, 0.97, 1.0],
    );
    if let Some(path) = std::env::var_os("LIGHT_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
}

/// The same tilted slab with its front, middle or back on the page, and
/// lifted. `ALIGN_PNG=path`.
#[test]
fn solid_depth_alignment() {
    use threers::ui::depth::Transform3d;
    use threers::ui::solid::{DepthAlign, SolidStyle};
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    let root = ui.root();
    ui.set_style(root, style! { display: flex; align-items: center; gap: 50px; padding: 50px; background: #e5e7eb; });
    for (align, lift) in [
        (DepthAlign::Front, 0.0),
        (DepthAlign::Center, 0.0),
        (DepthAlign::Back, 0.0),
        (DepthAlign::Back, 24.0),
    ] {
        let b = ui.add_box(root, style! { width: 120px; height: 80px; border-radius: 14px; background: #1d4ed8; color: #fff; font-size: 15px; font-weight: bold; display: flex; align-items: center; justify-content: center; });
        ui.add_text(b, format!("{align:?}"));
        ui.set_transform3d(
            b,
            Transform3d {
                rotate_y: 0.55,
                rotate_x: 0.25,
                perspective: 400.0,
                ..Transform3d::default()
            },
        );
        ui.set_solid(
            b,
            SolidStyle::default()
                .thickness(22.0)
                .align(align)
                .lift(lift),
        );
    }
    let (w, h) = (780u32, 220u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    ui.step(0.0);
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.9, 0.9, 0.92, 1.0],
    );
    if let Some(path) = std::env::var_os("ALIGN_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
}

/// Fire, water, plasma and frost, bars and rings. `EFFECTS_PNG=path`.
#[test]
fn effects_are_drawn() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    ui.set_system_appearance(threers::ui::states::Appearance::Dark);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 40px; padding: 40px; background: #0b0d12; });
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; gap: 46px; width: 440px; },
    );
    let mut fire_bar = None;
    for p in [Preset::Inferno, Preset::Aquarium, Preset::Glacier] {
        let i = indicator(&mut ui, bars, p.name(), p.style(), Some(0.7));
        fire_bar.get_or_insert(i.node);
    }
    indicator(
        &mut ui,
        bars,
        "Plasma",
        IndicatorStyle::new(Shape::Bar)
            .palette(Palette::NEON)
            .size(18.0)
            .fill(Fill::Plasma),
        Some(0.7),
    );
    let rings = ui.add_box(root, style! { display: flex; gap: 40px; });
    for p in [Preset::FireRing, Preset::Nebula] {
        indicator(&mut ui, rings, p.name(), p.style().size(120.0), Some(0.75));
    }
    indicator(
        &mut ui,
        rings,
        "Frost",
        IndicatorStyle::new(Shape::Ring)
            .palette(Palette::ICE)
            .size(120.0)
            .thickness(0.3)
            .fill(Fill::Frost),
        Some(0.75),
    );
    ui.step(1.3);
    let (w, h) = (520u32, 560u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("EFFECTS_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(pass.stats().failed_effects, 0);
    // Flames above the burning part of the bar, none above the rest.
    let r = ui.rect(fire_bar.unwrap());
    let lit_above = |x0: f32, x1: f32| {
        let mut most = 0u8;
        for y in ((r.y - r.height * 1.5) as usize)..(r.y as usize) {
            for x in (x0 as usize)..(x1 as usize) {
                let i = (y * w as usize + x) * 4;
                most = most.max(image[i].max(image[i + 1]));
            }
        }
        most
    };
    let (burning, cold) = (
        lit_above(r.x + r.width * 0.2, r.x + r.width * 0.5),
        lit_above(r.x + r.width * 0.85, r.x + r.width * 0.98),
    );
    assert!(
        burning > 120 && cold < burning / 2,
        "flames over the fill: {burning}, past it: {cold}"
    );
}

/// A custom fill is drawn along a bar and round a ring, and a broken one
/// draws nothing rather than failing. `CUSTOM_PNG=path`.
#[test]
fn custom_fills_are_drawn() {
    let Some(gpu) = threers::gpu::testing::context("the UI pass") else {
        return;
    };
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut pass = UiPass::new(device);
    let mut ui = Ui::new();
    ui.set_system_appearance(threers::ui::states::Appearance::Dark);
    let root = ui.root();
    ui.set_style(root, style! { display: flex; flex-direction: column; gap: 30px; padding: 30px; background: #0b0d12; });
    let bars = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; width: 300px; },
    );
    let stripes = Fill::custom(STRIPES);
    let bar = indicator(
        &mut ui,
        bars,
        "Stripes",
        IndicatorStyle::new(Shape::Bar)
            .palette(Palette::CITRUS)
            .size(20.0)
            .fill(stripes),
        Some(0.6),
    );
    indicator(
        &mut ui,
        root,
        "Stripes",
        IndicatorStyle::new(Shape::Ring)
            .palette(Palette::NEON)
            .size(100.0)
            .thickness(0.3)
            .fill(stripes),
        Some(0.6),
    );
    let broken = Fill::custom("fn broken(along: f32) -> vec4<f32> { return vec4<f32>(1.0); }");
    assert!(
        broken.custom_error().unwrap().contains("ten")
            || broken.custom_error().unwrap().contains("takes 1")
    );
    indicator(
        &mut ui,
        root,
        "Broken",
        IndicatorStyle::new(Shape::Bar).size(10.0).fill(broken),
        Some(0.6),
    );
    ui.step(0.5);
    let (w, h) = (360u32, 260u32);
    ui.layout(w as f32, h as f32, &mut pass.text_measure(1.0));
    let image = pass.render_to_rgba(
        device,
        queue,
        &ui.scene(),
        1.0,
        (w, h),
        [0.0, 0.0, 0.0, 1.0],
    );
    if let Some(path) = std::env::var_os("CUSTOM_PNG") {
        std::fs::write(path, threers::utils::png::encode_png(w, h, &image)).unwrap();
    }
    assert_eq!(
        pass.stats().failed_effects,
        0,
        "a broken source is caught before the GPU"
    );
    let r = ui.rect(bar.node);
    let i = (((r.y + r.height * 0.5) as usize) * w as usize + (r.x + r.width * 0.3) as usize) * 4;
    assert!(
        image[i].max(image[i + 1]) > 120,
        "the stripes are drawn: {:?}",
        &image[i..i + 3]
    );
}
