//! Charts from `threers::marks`, no map in sight: a 3D bar chart coloured by
//! a sequential scale, a scatter plot of six thousand points in five
//! categories with one picked on the GPU and called out, and the named
//! colour ramps and a categorical palette as strips. Every bar, point, axis,
//! label and swatch is a mark row. Written to `out/data_charts.png`.
//!
//! ```bash
//! cargo run --release --features captions --example data_charts
//! ```

use threers::captions::CaptionFont;
use threers::marks::{
    LabelAnchor, Mark, MarkRenderer, MarkView, Origin, PointMarks, SegmentMarks, ShapeMarks,
    TextLabel, TextLabels,
};
use threers::math::{ColorRamp, ColorScale, Palette};
use threers::prelude::*;

const W: u32 = 600;
const H: u32 = 460;
const STRIP_H: u32 = 240;

fn font() -> CaptionFont {
    [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ]
    .iter()
    .find_map(|p| std::fs::read(p).ok())
    .and_then(|b| CaptionFont::from_ttf_bytes(&b).ok())
    .unwrap_or_else(CaptionFont::builtin)
}

fn bytes(c: Color) -> [u8; 4] {
    let h = c.to_hex();
    [(h >> 16) as u8, (h >> 8) as u8, h as u8, 255]
}

fn labels(size: f32, rows: Vec<TextLabel>) -> TextLabels {
    let mut l = TextLabels::new(font());
    l.font_size = size;
    l.halo_width = 2.0;
    l.halo_color = [12, 14, 20, 255];
    l.set_labels(rows);
    l
}

fn label(at: [f64; 3], text: impl Into<String>, anchor: LabelAnchor) -> TextLabel {
    let mut l = TextLabel::new(at, text);
    l.anchor = anchor;
    l.color = [214, 220, 232, 255];
    l
}

/// A deterministic hash in `[0, 1)`, for data that looks measured.
fn noise(i: u32) -> f32 {
    let mut x = i.wrapping_mul(0x9e37_79b9) ^ 0x85eb_ca6b;
    x ^= x >> 15;
    x = x.wrapping_mul(0x2c1b_3c6d);
    x ^= x >> 12;
    (x & 0xff_ffff) as f32 / 16_777_216.0
}

/// The view a panel's marks are drawn and labels placed through.
fn view_of(camera: &dyn Camera, size: (u32, u32)) -> MarkView {
    MarkView::new(camera, Origin::new(0.0, 0.0, 0.0), size)
}

/// Draw `layers` over `scene` from `camera`, headless, into RGBA; `then`
/// runs between the scene and the marks (a pick, for one).
fn draw(
    scene: &mut Scene,
    camera: &dyn Camera,
    view: &MarkView,
    layers: &mut [&mut dyn Mark],
    then: impl FnOnce(&mut HeadlessRenderer, &mut MarkRenderer, &mut [&mut dyn Mark]),
) -> Vec<u8> {
    let mut hr = HeadlessRenderer::builder()
        .size(view.viewport.0, view.viewport.1)
        .build()
        .expect("a GPU adapter");
    hr.render(scene, camera);
    let mut marks = MarkRenderer::new(hr.device());
    then(&mut hr, &mut marks, layers);
    let (device, queue) = (hr.device().clone(), hr.queue().clone());
    marks.draw(&device, &queue, &hr.mark_target(), view, layers);
    hr.read_rgba()
}

/// Monthly values over six years: a seasonal swing on a rising trend.
fn bar_chart() -> Vec<u8> {
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let years = 6;
    let mut centres = Vec::new();
    let mut scales = Vec::new();
    let mut values = Vec::new();
    for y in 0..years {
        for m in 0..12 {
            let season = (std::f32::consts::TAU * (m as f32 - 3.0) / 12.0).sin();
            let v = 2.0 + 0.35 * y as f32 + 1.2 * season + 0.5 * noise(y * 12 + m);
            let (x, z) = (m as f64 - 5.5, y as f64 * 1.3 - 3.0);
            centres.push([x, v as f64 / 2.0, z]);
            scales.push([0.72, v, 0.72]);
            values.push(v);
        }
    }
    let max = values.iter().cloned().fold(0.0, f32::max);
    let scale = ColorScale::sequential(ColorRamp::named("viridis").unwrap(), 0.0, max);
    let mut bars = ShapeMarks::new(&BoxGeometry::new(1.0, 1.0, 1.0));
    bars.set_positions(&centres);
    bars.set_scales(&scales);
    bars.set_colors(&scale.map_rgba8(&values));

    let mut rows: Vec<TextLabel> = MONTHS
        .iter()
        .enumerate()
        .map(|(m, name)| label([m as f64 - 5.5, 0.0, 4.2], *name, LabelAnchor::Below))
        .collect();
    for y in 0..years {
        rows.push(label(
            [-6.4, 0.0, y as f64 * 1.3 - 3.0],
            format!("{}", 2020 + y),
            LabelAnchor::Left,
        ));
    }
    let mut text = labels(13.0, rows);

    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10131a);
    scene.add_light(AmbientLight::new(Color::WHITE, 0.6));
    let mut sun = DirectionalLight::new(Color::WHITE, 1.6);
    sun.direction = Vector3::new(-0.4, -1.0, -0.6).normalize();
    scene.add_light(sun);
    let mut floor = StandardMaterial::new(Color::from_hex(0x1d2230));
    floor.roughness = 1.0;
    let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(14.0, 9.0), floor.into()));
    f.rotate_x(-std::f32::consts::FRAC_PI_2);
    f.position = Vector3::new(0.0, -0.01, 0.25);
    scene.add(f);
    let mut camera = PerspectiveCamera::new(38.0, W as f32 / H as f32, 0.1, 100.0);
    camera.position = Vector3::new(-6.5, 9.5, 15.5);
    camera.look_at(Vector3::new(0.3, 1.2, 0.0));
    let view = view_of(&camera, (W, H));
    text.place(&view);
    draw(
        &mut scene,
        &camera,
        &view,
        &mut [&mut bars, &mut text],
        |_, _, _| {},
    )
}

/// Five clusters of points, radius by value, coloured by cluster; the point
/// under one pixel is picked and called out.
fn scatter() -> Vec<u8> {
    let palette = Palette::named("tableau10").unwrap();
    let centres = [
        (2.2f32, 4.4f32),
        (5.0, 2.0),
        (7.6, 4.0),
        (3.6, 1.4),
        (6.4, 5.2),
    ];
    let (mut pos, mut radii, mut colors, mut info) = (vec![], vec![], vec![], vec![]);
    for i in 0..6_000u32 {
        let c = (i % 5) as usize;
        // Box–Muller from the hash, for clusters with soft edges.
        let (u, v) = (noise(i * 3 + 1).max(1e-6), noise(i * 3 + 2));
        let r = (-2.0 * u.ln()).sqrt() * 0.55;
        let a = std::f32::consts::TAU * v;
        let (x, y) = (centres[c].0 + r * a.cos(), centres[c].1 + r * a.sin() * 0.7);
        let value = noise(i * 7 + 5);
        pos.push([x as f64, y as f64, 0.0]);
        radii.push(1.5 + 4.5 * value * value);
        colors.push({
            let mut b = bytes(palette.at(c));
            b[3] = 200;
            b
        });
        info.push((c, value));
    }
    let mut points = PointMarks::new();
    points.set_positions(&pos);
    points.set_radii(&radii);
    points.set_colors(&colors);
    points.stroke_width = 0.6;
    points.stroke_color = [10, 12, 18, 160];

    // Axes, with a tick every unit.
    let (mut from, mut to) = (
        vec![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        vec![[10.0, 0.0, 0.0], [0.0, 6.5, 0.0]],
    );
    let mut ticks = Vec::new();
    for k in 0..=10 {
        from.push([k as f64, 0.0, 0.0]);
        to.push([k as f64, -0.12, 0.0]);
        ticks.push(label(
            [k as f64, -0.2, 0.0],
            k.to_string(),
            LabelAnchor::Below,
        ));
    }
    for k in 0..=6 {
        from.push([0.0, k as f64, 0.0]);
        to.push([-0.12, k as f64, 0.0]);
        ticks.push(label(
            [-0.2, k as f64, 0.0],
            k.to_string(),
            LabelAnchor::Left,
        ));
    }
    let mut axes = SegmentMarks::new();
    axes.set_segments(&from, &to);
    axes.set_widths(&vec![1.5; from.len()]);
    axes.set_colors(&vec![[150, 158, 175, 255]; from.len()]);

    let mut camera = OrthographicCamera::new(-0.9, 10.5, 7.0, -1.0, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10131a);

    // Pick under a pixel inside the third cluster, the way a hover would,
    // then ring and label what was there.
    let view = view_of(&camera, (W, H));
    let mut ticks = labels(12.0, ticks);
    ticks.place(&view);
    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.render(&mut scene, &camera);
    let (device, queue) = (hr.device().clone(), hr.queue().clone());
    let mut marks = MarkRenderer::new(&device);
    let [sx, sy, _] = view.project([7.6, 4.0, 0.0]).expect("on screen");
    let hits = marks
        .pick(&device, &queue, &view, &mut [&mut points], None, sx, sy, 6)
        .wait(&device);
    let mut callout = PointMarks::new();
    let mut callout_text = labels(13.0, Vec::new());
    if let Some(row) = hits.map(|h| h.index as usize) {
        let (cluster, value) = info[row];
        println!(
            "  picked row {row}: cluster {}, value {value:.2}",
            cluster + 1
        );
        callout.set_positions(&[pos[row]]);
        callout.set_radii(&[radii[row] + 7.0]);
        callout.set_colors(&[[255, 255, 255, 255]]);
        callout.hollow = true;
        callout.stroke_width = 2.0;
        callout.stroke_color = [255, 255, 255, 255];
        let mut l = label(
            pos[row],
            format!("row {row} · cluster {} · {value:.2}", cluster + 1),
            LabelAnchor::Left,
        );
        l.offset = [-14.0, 0.0];
        l.color = [255, 255, 255, 255];
        callout_text.set_labels(vec![l]);
        callout_text.place(&view);
    } else {
        println!("  nothing under the cursor");
    }
    marks.draw(
        &device,
        &queue,
        &hr.mark_target(),
        &view,
        &mut [
            &mut points,
            &mut axes,
            &mut callout,
            &mut callout_text,
            &mut ticks,
        ],
    );
    hr.read_rgba()
}

/// Named ramps and a palette as strips of swatches.
fn strips() -> Vec<u8> {
    let w = W * 2;
    let ramps = [
        "viridis", "magma", "turbo", "cividis", "yl_or_rd", "blues", "rd_bu", "spectral",
    ];
    let cells = 96;
    let (mut pos, mut scales, mut colors, mut names) = (vec![], vec![], vec![], vec![]);
    let x0 = 1.6;
    let width = 10.0;
    for (r, name) in ramps.iter().enumerate() {
        let ramp = ColorRamp::named(name).unwrap();
        let y = -(r as f64) * 0.45;
        for k in 0..cells {
            let t = (k as f32 + 0.5) / cells as f32;
            pos.push([x0 + width * t as f64, y, 0.0]);
            scales.push([width as f32 / cells as f32 + 0.002, 0.34, 0.01]);
            colors.push(bytes(ramp.at(t)));
        }
        names.push(label([x0 - 0.15, y, 0.0], *name, LabelAnchor::Left));
    }
    // And a categorical palette: ten distinct colours.
    let palette = Palette::named("tableau10").unwrap();
    let y = -(ramps.len() as f64) * 0.45 - 0.1;
    for k in 0..10 {
        pos.push([x0 + width * (k as f64 + 0.5) / 10.0, y, 0.0]);
        scales.push([width as f32 / 10.0 - 0.08, 0.34, 0.01]);
        colors.push(bytes(palette.at(k)));
    }
    names.push(label([x0 - 0.15, y, 0.0], "tableau10", LabelAnchor::Left));

    let mut swatches = ShapeMarks::new(&BoxGeometry::new(1.0, 1.0, 1.0));
    swatches.set_positions(&pos);
    swatches.set_scales(&scales);
    swatches.set_colors(&colors);
    swatches.ambient = 1.0;
    let mut text = labels(13.0, names);
    let mut camera = OrthographicCamera::new(-0.2, 12.0, 0.35, -4.4, 0.1, 10.0);
    camera.position = Vector3::new(0.0, 0.0, 5.0);
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10131a);
    let view = view_of(&camera, (w, STRIP_H));
    text.place(&view);
    draw(
        &mut scene,
        &camera,
        &view,
        &mut [&mut swatches, &mut text],
        |_, _, _| {},
    )
}

fn main() {
    let bars = bar_chart();
    let points = scatter();
    let ramps = strips();
    let (w, h) = (W * 2, H + STRIP_H);
    let mut sheet = vec![0u8; (w * h * 4) as usize];
    for y in 0..H {
        let at = (y * w * 4) as usize;
        let row = (y * W * 4) as usize..((y + 1) * W * 4) as usize;
        sheet[at..at + (W * 4) as usize].copy_from_slice(&bars[row.clone()]);
        sheet[at + (W * 4) as usize..at + (w * 4) as usize].copy_from_slice(&points[row]);
    }
    let top = (H * w * 4) as usize;
    sheet[top..].copy_from_slice(&ramps);
    std::fs::create_dir_all("out").unwrap();
    std::fs::write("out/data_charts.png", encode_png(w, h, &sheet)).unwrap();
    println!("wrote out/data_charts.png");
}
