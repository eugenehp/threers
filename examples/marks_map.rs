//! A kepler.gl-style map built from `threers::marks`.
//!
//! ```text
//! cargo run --release --example marks_map
//! ```
//!
//! Twenty thousand synthetic pickups around a made-up city, placed in Web
//! Mercator metres thirteen million metres from the origin, drawn as a
//! heatmap, as points filtered by hour on the GPU, as extruded hexbins, as
//! animated trips and arcs between neighbourhoods, and with labels over the
//! top.
//! Seven hundred buildings with courtyards are one `PolygonMarks` draw; the
//! ground is an ordinary mesh, and the marks depth-test against both.
//!
//! Left-drag: pan · Right-drag: turn and tilt · Scroll: zoom toward the cursor
//! · Hover: pick a point · 1–7: toggle buildings, heatmap, points, hexbins,
//! trips, arcs, labels · Space: pause the clock
//!
//! `--png out.png` renders one frame headless instead of opening a window;
//! `--layers 1110111` picks which of the seven layers it shows, in key order,
//! and `--distance 3000` how far from the city centre the camera is.
//!
//! See `docs/marks.md`.

use std::time::Instant;

use threers::cameras::Camera;
use threers::captions::CaptionFont;
use threers::marks::{
    DensityMarks, Mark, MarkRenderer, MarkView, Origin, PickHit, PickRequest, PointMarks,
    PolygonFeature, PolygonMarks, SegmentMarks, ShaderHooks, ShapeMarks, SizeUnits, TextLabel,
    TextLabels,
};
use threers::math::ColorRamp;
use threers::renderer::WindowRenderer;
use threers::{
    AmbientLight, Color, CylinderGeometry, DirectionalLight, MapControls, Mesh, Object3D,
    PerspectiveCamera, PointerEvent, PolygonGeometry, Scene, StandardMaterial, Vector3,
};

use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

/// The city centre, in Web Mercator metres with Y up and north toward −Z. An
/// `f32` here is a whole metre apart; the marks keep the `f64`.
const CENTRE: [f64; 3] = [-13_627_000.0, 0.0, 4_548_000.0];
const POINTS: usize = 20_000;
const HEX_RADIUS: f64 = 350.0;
/// Hours the time filter shows at once.
const WINDOW_HOURS: f32 = 4.0;

const HUBS: [(&str, f64, f64, f32); 8] = [
    // name, east and north of the centre in metres, the hour it peaks
    ("Harbor", -3_200.0, 2_600.0, 8.0),
    ("Old Town", 0.0, 0.0, 12.0),
    ("Mission", 900.0, -3_100.0, 20.0),
    ("Hillcrest", -4_800.0, -1_600.0, 18.0),
    ("Northgate", 2_400.0, 4_300.0, 7.0),
    ("Riverside", 5_200.0, 900.0, 22.0),
    ("Uptown", -1_500.0, 5_400.0, 16.0),
    ("Airport", 7_400.0, -5_200.0, 5.0),
];

/// xorshift: the demo needs repeatable noise, not a dependency.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let u = self.next().max(1e-12);
        let v = self.next();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// East/north offset from the centre as an absolute position.
fn at(east: f64, north: f64, up: f64) -> [f64; 3] {
    [CENTRE[0] + east, CENTRE[1] + up, CENTRE[2] - north]
}

fn srgb_bytes(c: Color, alpha: u8) -> [u8; 4] {
    let h = c.to_hex();
    [(h >> 16) as u8, (h >> 8) as u8, h as u8, alpha]
}

struct Data {
    positions: Vec<[f64; 3]>,
    hours: Vec<f32>,
    colors: Vec<[u8; 4]>,
    /// Which hub each pickup came from.
    hubs: Vec<usize>,
}

fn pickups() -> Data {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let ramp = ColorRamp::plasma();
    let mut d = Data {
        positions: Vec::with_capacity(POINTS),
        hours: Vec::with_capacity(POINTS),
        colors: Vec::with_capacity(POINTS),
        hubs: Vec::with_capacity(POINTS),
    };
    for _ in 0..POINTS {
        // Busier hubs first: hub k is picked with weight 1 / (k + 1).
        let total: f64 = (1..=HUBS.len()).map(|k| 1.0 / k as f64).sum();
        let mut pick = rng.next() * total;
        let mut hub = 0;
        while hub + 1 < HUBS.len() && pick > 1.0 / (hub + 1) as f64 {
            pick -= 1.0 / (hub + 1) as f64;
            hub += 1;
        }
        let (_, e, n, peak) = HUBS[hub];
        let spread = 600.0 + 500.0 * rng.next();
        let east = e + rng.gauss() * spread;
        let north = n + rng.gauss() * spread;
        let hour = (peak as f64 + rng.gauss() * 2.5).rem_euclid(24.0) as f32;
        d.positions.push(at(east, north, 0.5));
        d.hours.push(hour);
        d.colors.push(srgb_bytes(ramp.at(hour / 24.0), 230));
        d.hubs.push(hub);
    }
    d
}

/// Pointy-top hexagonal bins: (centre, count) for every non-empty cell.
fn hexbin(positions: &[[f64; 3]]) -> Vec<([f64; 3], u32)> {
    let mut cells = std::collections::HashMap::<(i64, i64), u32>::new();
    let s3 = 3f64.sqrt();
    for p in positions {
        let east = p[0] - CENTRE[0];
        let north = CENTRE[2] - p[2];
        // Axial coordinates, then cube rounding to the nearest cell.
        let q = (s3 / 3.0 * east - north / 3.0) / HEX_RADIUS;
        let r = (2.0 / 3.0 * north) / HEX_RADIUS;
        let (x, z) = (q, r);
        let y = -x - z;
        let (mut rx, ry, mut rz) = (x.round(), y.round(), z.round());
        let (dx, dy, dz) = ((rx - x).abs(), (ry - y).abs(), (rz - z).abs());
        if dx > dy && dx > dz {
            rx = -ry - rz;
        } else if dy <= dz {
            rz = -rx - ry;
        }
        *cells.entry((rx as i64, rz as i64)).or_default() += 1;
    }
    let mut out: Vec<_> = cells
        .into_iter()
        .map(|((q, r), n)| {
            let east = HEX_RADIUS * s3 * (q as f64 + r as f64 / 2.0);
            let north = HEX_RADIUS * 1.5 * r as f64;
            (at(east, north, 0.0), n)
        })
        .collect();
    // Stable order, so the same cell keeps the same row between runs.
    out.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]).then(a.0[2].total_cmp(&b.0[2])));
    out
}

/// Every mark layer in the demo, in draw order.
struct Layers {
    buildings: PolygonMarks,
    density: DensityMarks,
    points: PointMarks,
    hexes: ShapeMarks,
    routes: SegmentMarks,
    arcs: SegmentMarks,
    labels: TextLabels,
}

impl Layers {
    fn all(&mut self) -> [&mut dyn Mark; 7] {
        [
            &mut self.buildings,
            &mut self.density,
            &mut self.points,
            &mut self.hexes,
            &mut self.routes,
            &mut self.arcs,
            &mut self.labels,
        ]
    }
}

fn layers(data: &Data) -> Layers {
    let mut density = DensityMarks::new();
    density.set_positions(&data.positions);
    density.set_ramp(&ColorRamp::inferno());
    // Kernels in metres rather than pixels: the same streets are summed at
    // every zoom, so the colours mean the same thing zoomed in or out.
    density.common.style.units = SizeUnits::World;
    density.common.style.size_scale = 450.0;
    density.max_weight = 120.0;
    density.common.style.opacity = 0.75;

    let mut points = PointMarks::new();
    points.set_positions(&data.positions);
    points.set_colors(&data.colors);
    points.common.style.size_scale = 2.5;
    points.common.style.min_pixels = 1.5;
    // Enough to stay whole on the ground down to 14° above it, which is as
    // flat as the controls tilt.
    points.common.style.depth_lift = 4.0;
    // The hour rides along as custom column 0 and the hook drops rows outside
    // `params[0].xy`: sliding the window writes sixteen bytes a frame.
    points.common.set_custom(0, 1, &data.hours);
    points.common.hooks = ShaderHooks::range_filter(0, 0);

    let mut hexes = ShapeMarks::new(&CylinderGeometry::new(
        1.0,
        1.0,
        1.0,
        6,
        1,
        false,
        0.0,
        std::f32::consts::TAU,
    ));
    let bins = hexbin(&data.positions);
    let max = bins.iter().map(|b| b.1).max().unwrap_or(1) as f32;
    let ramp = ColorRamp::viridis();
    // The cylinder is centred on its position; lift each by half its height
    // so it stands on the ground.
    let heights: Vec<f32> = bins
        .iter()
        .map(|b| 20.0 + 1_800.0 * b.1 as f32 / max)
        .collect();
    let centres: Vec<[f64; 3]> = bins
        .iter()
        .zip(&heights)
        .map(|(b, h)| [b.0[0], b.0[1] + *h as f64 * 0.5, b.0[2]])
        .collect();
    let r = (HEX_RADIUS * 0.9) as f32;
    hexes.set_positions(&centres);
    hexes.set_scales(&heights.iter().map(|h| [r, *h, r]).collect::<Vec<_>>());
    hexes.set_colors(
        &bins
            .iter()
            .map(|b| srgb_bytes(ramp.at((b.1 as f32 / max).sqrt()), 255))
            .collect::<Vec<_>>(),
    );
    hexes.common.style.opacity = 0.9;
    hexes.common.style.visible = false;

    // An arc from every hub to every other, weighted by how near they are.
    let (mut src, mut dst, mut widths, mut c0, mut c1) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (i, a) in HUBS.iter().enumerate() {
        for b in HUBS.iter().skip(i + 1) {
            let d = ((a.1 - b.1).powi(2) + (a.2 - b.2).powi(2)).sqrt();
            src.push(at(a.1, a.2, 0.0));
            dst.push(at(b.1, b.2, 0.0));
            widths.push((14_000.0 / d).clamp(1.0, 4.0) as f32);
            c0.push([0x29, 0xb6, 0xf6, 220]);
            c1.push([0xff, 0x40, 0x81, 220]);
        }
    }
    let mut arcs = SegmentMarks::new();
    arcs.set_segments(&src, &dst);
    arcs.set_widths(&widths);
    arcs.set_colors(&c0);
    arcs.set_target_colors(&c1);
    arcs.arc_height = 0.25;

    let mut labels = TextLabels::new(label_font());
    labels.font_size = 16.0;
    labels.halo_width = 2.0;
    let mut counts = [0usize; HUBS.len()];
    for &h in &data.hubs {
        counts[h] += 1;
    }
    labels.set_labels(
        HUBS.iter()
            .zip(counts)
            .map(|(&(name, e, n, _), count)| {
                let mut l = TextLabel::new(at(e, n, 0.0), name);
                l.priority = count as f32;
                l
            })
            .collect(),
    );

    Layers {
        buildings: buildings(),
        density,
        points,
        hexes,
        routes: routes(),
        arcs,
        labels,
    }
}

/// Drives from every hub to Old Town along a 400 m street grid, four a route
/// at staggered start times, animated by the trail hook: each segment carries
/// the hours its two ends are reached, and the clock is a uniform.
fn routes() -> SegmentMarks {
    let mut rng = Rng(42);
    let block = 400.0;
    let (_, ex, ey, _) = HUBS[1];
    let mut paths = Vec::new();
    for &(_, sx, sy, _) in HUBS.iter().filter(|h| h.0 != "Old Town") {
        let snap = |v: f64| (v / block).round() * block;
        let (mut x, mut y) = (snap(sx), snap(sy));
        let mut path = vec![at(x, y, 1.0)];
        let mut east = rng.next() < 0.5;
        while (x - ex).abs() > 1.0 || (y - ey).abs() > 1.0 {
            let blocks = 1.0 + (rng.next() * 4.0).floor();
            if east && (x - ex).abs() > 1.0 {
                x += (ex - x).signum() * (blocks * block).min((ex - x).abs());
            } else if (y - ey).abs() > 1.0 {
                y += (ey - y).signum() * (blocks * block).min((ey - y).abs());
            } else {
                x += (ex - x).signum() * (blocks * block).min((ex - x).abs());
            }
            east = !east;
            path.push(at(x, y, 1.0));
        }
        paths.push(path);
    }
    // Every route four times over, at 2.5 km an hour of the demo's clock.
    let speed = 2_500.0;
    let mut trips = Vec::new();
    let mut times = Vec::new();
    for path in &paths {
        for _ in 0..4 {
            let mut t = rng.next() * 18.0;
            for w in path.windows(2) {
                let d = ((w[1][0] - w[0][0]).powi(2) + (w[1][2] - w[0][2]).powi(2)).sqrt();
                times.extend_from_slice(&[t as f32, (t + d / speed) as f32]);
                t += d / speed;
            }
            trips.push(path.clone());
        }
    }
    let mut routes = SegmentMarks::new();
    routes.set_paths(&trips);
    routes.set_path_widths(&[5.0]);
    routes.set_path_colors(&[[0x00, 0xe6, 0xc3, 255]]);
    routes.common.set_custom(0, 2, &times);
    routes.common.hooks = ShaderHooks::trail(0, 0);
    routes
}

/// An hour and a half of trail behind each trip's head.
const TRAIL_HOURS: f32 = 1.5;

/// A system TrueType face if one is around; the built-in bitmap face if not.
fn label_font() -> CaptionFont {
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

/// Ground and a few buildings, as ordinary scene-graph meshes in
/// origin-relative coordinates.
fn city(scene: &mut Scene, origin: &Origin) {
    let rel = |east: f64, north: f64| {
        let l = origin.to_local(at(east, north, 0.0));
        [l.x as f64, -l.z as f64]
    };
    let half = 12_000.0;
    let ground = [
        rel(-half, -half),
        rel(half, -half),
        rel(half, half),
        rel(-half, half),
    ];
    scene.add(Object3D::mesh(Mesh::new(
        PolygonGeometry::new(&ground, &[], 0.0, 0.0),
        StandardMaterial::new(Color::from_hex(0x15181f))
            .with_roughness(1.0)
            .into(),
    )));
}

/// Footprints around every hub, a third of them with a courtyard, extruded
/// to a height that falls off from the hub and coloured by it: one draw.
fn buildings() -> PolygonMarks {
    let mut rng = Rng(7);
    let ramp = ColorRamp::from_hex(&[0x2b3a55, 0x6b7fa8, 0xd8e2f5]);
    let (mut features, mut heights, mut colors) = (Vec::new(), Vec::new(), Vec::new());
    for &(_, e, n, _) in &HUBS {
        for _ in 0..88 {
            let (dx, dy) = (rng.gauss() * 700.0, rng.gauss() * 700.0);
            let (x, y) = (CENTRE[0] + e + dx, -CENTRE[2] + n + dy);
            let (w, d) = (40.0 + 90.0 * rng.next(), 40.0 + 90.0 * rng.next());
            let rect =
                |x0: f64, y0: f64, x1: f64, y1: f64| vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
            let mut f = PolygonFeature::new(rect(x, y, x + w, y + d));
            if rng.next() < 0.33 {
                f.holes
                    .push(rect(x + w * 0.3, y + d * 0.3, x + w * 0.7, y + d * 0.7));
            }
            let falloff = (-(dx * dx + dy * dy) / (2.0 * 600.0f64.powi(2))).exp();
            let h = (20.0 + 280.0 * falloff * (0.4 + 0.6 * rng.next())) as f32;
            features.push(f);
            heights.push(h);
            colors.push(srgb_bytes(ramp.at(h / 300.0), 255));
        }
    }
    let mut b = PolygonMarks::new();
    b.set_polygons(&features, 0.0);
    b.set_elevations(&heights);
    b.set_colors(&colors);
    // Points are what the cursor is for.
    b.common.style.pickable = false;
    b
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--png") {
        Some(i) => {
            let flag = |name: &str| {
                args.iter()
                    .position(|a| a == name)
                    .and_then(|j| args.get(j + 1))
                    .map(|s| s.as_str())
            };
            snapshot(
                args.get(i + 1).map_or("marks_map.png", |s| s.as_str()),
                flag("--layers").unwrap_or("1110111"),
                flag("--distance").and_then(|d| d.parse().ok()),
            )
        }
        None => pollster::block_on(run()),
    }
}

/// The opening view rendered headless to `path`, with the layers whose digit
/// in `shown` is 1.
fn snapshot(path: &str, shown: &str, distance: Option<f32>) {
    let (w, h) = (1200, 800);
    let mut hr = threers::HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("headless renderer");
    let origin = Origin::new(CENTRE[0], CENTRE[1], CENTRE[2]);
    let mut scene = scene(&origin);
    let data = pickups();
    let mut layers = layers(&data);
    for (m, on) in layers.all().into_iter().zip(shown.chars()) {
        m.style_mut().visible = on == '1';
    }
    layers.points.common.style.params[0] = [6.0, 6.0 + WINDOW_HOURS, 0.0, 0.0];
    layers.routes.common.style.params[0] = [9.0, TRAIL_HOURS, 0.0, 0.0];
    let (mut camera, mut controls) = opening_camera(w as f32 / h as f32);
    if let Some(d) = distance {
        controls.distance = d;
        controls.apply(&mut camera);
    }

    hr.render(&mut scene, &camera);
    let mut marks = hr.mark_renderer();
    let view = MarkView::new(&camera, origin, (w, h));
    layers.labels.place(&view);
    hr.draw_marks(&mut marks, &view, &mut layers.all());
    let rgba = hr.read_rgba();
    std::fs::write(path, threers::encode_png(w, h, &rgba)).expect("write png");
    println!("wrote {path}");
}

fn scene(origin: &Origin) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0b0d12);
    scene.add_light(AmbientLight::new(Color::from_hex(0xffffff), 0.4));
    let mut sun = Object3D::light(DirectionalLight::new(Color::from_hex(0xffffff), 1.2));
    sun.position = Vector3::new(4_000.0, 8_000.0, 4_500.0);
    scene.add(sun);
    city(&mut scene, origin);
    scene
}

fn opening_camera(aspect: f32) -> (PerspectiveCamera, MapControls) {
    let mut camera = PerspectiveCamera::new(45.0, aspect, 10.0, 200_000.0);
    let mut controls = MapControls::new(Vector3::ZERO, 22_000.0);
    controls.pitch = 0.75;
    controls.bearing = -0.3;
    controls.max_pitch = 76f32.to_radians();
    controls.min_distance = 300.0;
    controls.max_distance = 120_000.0;
    controls.apply(&mut camera);
    (camera, controls)
}

async fn run() {
    let attributes = winit::window::Window::default_attributes()
        .with_title("threers — marks map")
        .with_inner_size(winit::dpi::LogicalSize::new(1200, 800));
    threers::sketch::run_window(attributes, move |window| {
        // wgpu, or the native Metal backend under `THREERS_BACKEND=metal`.
        let mut wr = WindowRenderer::new(window.clone()).expect("a GPU for the window");
        // Single-sampled, so a pick can test against the frame's depth.
        wr.set_msaa(1);

        let origin = Origin::new(CENTRE[0], CENTRE[1], CENTRE[2]);
        let mut scene = scene(&origin);

        let data = pickups();
        let mut layers = layers(&data);
        let mut marks = MarkRenderer::new_on(&wr.marks_gpu());

        let (mut camera, mut controls) = opening_camera(wr.aspect());

        let mut buttons = (false, false); // (left: pan, right: rotate)
        let mut cursor = (0.0f32, 0.0f32);
        let mut cursor_moved = false;
        let mut pick: Option<PickRequest> = None;
        let mut hovered: Option<u32> = None;
        let mut paused = false;
        let mut hour = 0.0f32;
        let mut last = Instant::now();
        let window_for_loop = window.clone();

        move |event, target| {
            let Event::WindowEvent { event, window_id } = event else {
                return;
            };
            if window_id != window_for_loop.id() {
                return;
            }
            let viewport = (wr.size().0 as f32, wr.size().1 as f32);
            match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    wr.resize(s.width, s.height);
                    camera.set_aspect(wr.aspect());
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(code),
                            state: ElementState::Pressed,
                            repeat: false,
                            ..
                        },
                    ..
                } => {
                    let style = |m: &mut dyn Mark| {
                        let s = m.style_mut();
                        s.visible = !s.visible;
                    };
                    match code {
                        KeyCode::Digit1 => style(&mut layers.buildings),
                        KeyCode::Digit2 => style(&mut layers.density),
                        KeyCode::Digit3 => style(&mut layers.points),
                        KeyCode::Digit4 => style(&mut layers.hexes),
                        KeyCode::Digit5 => style(&mut layers.routes),
                        KeyCode::Digit6 => style(&mut layers.arcs),
                        KeyCode::Digit7 => style(&mut layers.labels),
                        KeyCode::Space => paused = !paused,
                        _ => {}
                    }
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    let down = state == ElementState::Pressed;
                    match button {
                        MouseButton::Left => buttons.0 = down,
                        MouseButton::Right => buttons.1 = down,
                        _ => {}
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let now = (position.x as f32, position.y as f32);
                    let ev = PointerEvent {
                        dx: now.0 - cursor.0,
                        dy: now.1 - cursor.1,
                        wheel: 0.0,
                        panning: buttons.0,
                        rotating: buttons.1,
                        ..Default::default()
                    };
                    cursor = now;
                    cursor_moved = true;
                    controls.update(ev, cursor, &mut camera, viewport);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let wheel = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    let ev = PointerEvent {
                        wheel,
                        ..Default::default()
                    };
                    controls.update(ev, cursor, &mut camera, viewport);
                }
                WindowEvent::RedrawRequested => {
                    let dt = last.elapsed().as_secs_f32();
                    last = Instant::now();
                    if !paused {
                        // A day every 24 seconds.
                        hour = (hour + dt) % 24.0;
                    }
                    layers.points.common.style.params[0] = [hour, hour + WINDOW_HOURS, 0.0, 0.0];
                    layers.routes.common.style.params[0] = [hour, TRAIL_HOURS, 0.0, 0.0];

                    wr.render_with(&mut scene, &camera, |f| {
                        let view = MarkView::for_surface(&camera, origin, &f.surface);
                        layers.labels.place(&view);
                        marks.draw_on(&f.gpu, &f.target, &view, &mut layers.all());

                        // One pick in flight at a time, started when the cursor
                        // has moved and collected on a later frame. The frame's
                        // depth hides marks behind buildings.
                        if pick.is_none() && cursor_moved && !buttons.0 && !buttons.1 {
                            cursor_moved = false;
                            let only_points: &mut [&mut dyn Mark] = &mut [&mut layers.points];
                            pick = Some(marks.pick_on(
                                &f.gpu,
                                &view,
                                only_points,
                                f.occluder.as_ref(),
                                cursor.0,
                                cursor.1,
                                4,
                            ));
                        }
                    });
                    wr.poll();
                    if let Some(answer) = pick.as_mut().and_then(PickRequest::try_take) {
                        pick = None;
                        let row = answer.map(|h: PickHit| h.index);
                        if row != hovered {
                            hover(&mut layers.points, &data, hovered, row);
                            hovered = row;
                        }
                    }
                    let what = match hovered {
                        Some(i) => format!(
                            "pickup {i} near {} at {:05.2} h",
                            HUBS[data.hubs[i as usize]].0, data.hours[i as usize]
                        ),
                        None => "hover a point".into(),
                    };
                    window_for_loop.set_title(&format!(
                        "threers — marks map · {:02}:00–{:02}:00{} · {what}",
                        hour as u32,
                        (hour + WINDOW_HOURS) as u32 % 24,
                        if paused { " (paused)" } else { "" },
                    ));
                    window_for_loop.request_redraw();
                }
                _ => {}
            }
        }
    });
}

/// Restore the previously hovered row and enlarge the new one. Only those two
/// rows are uploaded.
fn hover(points: &mut PointMarks, data: &Data, old: Option<u32>, new: Option<u32>) {
    if let Some(i) = old.map(|i| i as usize) {
        points.update_colors(i, &[data.colors[i]]);
        points.update_radii(i, &[1.0]);
    }
    if let Some(i) = new.map(|i| i as usize) {
        points.update_colors(i, &[[255, 255, 255, 255]]);
        points.update_radii(i, &[3.0]);
    }
}
