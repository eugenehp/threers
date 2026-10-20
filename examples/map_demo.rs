//! A small map: European cities over raster map tiles.
//!
//! ```text
//! cargo run --release --example map_demo --features captions
//! cargo run --release --example map_demo --features captions -- --png map.png
//! ```
//!
//! What a map built on `threers::marks` needs, and little else:
//!
//! - `MapView` keeps the camera, its controls, the precision origin and the
//!   label collision index in step, frame by frame.
//! - Raster tiles for the view (`MapView::covering_tiles`) — finer near the
//!   camera when tilted, coarser towards the horizon — are fetched in the
//!   background, cached on disk and drawn as ground quads (`ImageMarks`). A
//!   tile still loading shows the matching part of a loaded ancestor.
//!   OpenStreetMap's standard tiles by default; `--tiles URL` takes any
//!   `{z}/{x}/{y}` template (mind the provider's usage policy), and
//!   `--tiles none` draws procedural tiles offline. Fetching shells out to
//!   `curl`.
//! - Each city is one symbol: a pin icon sized by its population and its
//!   name beside it, in multi-channel distance-field text, placed and culled
//!   together.
//! - A heatmap of activity around the cities scales itself to the densest
//!   place in view (`DensityMarks::auto_max`).
//!
//! Left-drag: pan · Right-drag: turn and tilt · Scroll: zoom toward the
//! cursor · 1–4: toggle map tiles, tile grid, heatmap, cities
//!
//! `--png out.png` renders the opening view headless instead of opening a
//! window; `--distance 900000` sets how far away the camera starts.

use std::collections::{HashMap, HashSet};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use threers::captions::CaptionFont;
use threers::marks::{
    geo, DensityMarks, GlyphField, ImageMarks, ImageQuad, LabelAnchor, LabelIcon, MapView, Mark,
    MarkRenderer, MarkTarget, SegmentMarks, SharedAtlas, TextLabel, TextLabels, TextureAtlas,
};
use threers::math::ColorRamp;
use threers::{Color, PointerEvent, Renderer, Scene};

use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

/// Name, longitude, latitude, population in millions.
const CITIES: [(&str, f64, f64, f32); 16] = [
    ("London", -0.13, 51.51, 9.0),
    ("Paris", 2.35, 48.86, 11.0),
    ("Berlin", 13.40, 52.52, 3.7),
    ("Madrid", -3.70, 40.42, 6.7),
    ("Rome", 12.50, 41.90, 4.3),
    ("Vienna", 16.37, 48.21, 2.0),
    ("Warsaw", 21.01, 52.23, 1.9),
    ("Amsterdam", 4.90, 52.37, 1.2),
    ("Brussels", 4.35, 50.85, 2.1),
    ("Prague", 14.42, 50.08, 1.3),
    ("Budapest", 19.04, 47.50, 1.8),
    ("Stockholm", 18.07, 59.33, 1.6),
    ("Lisbon", -9.14, 38.72, 2.9),
    ("Athens", 23.73, 37.98, 3.2),
    ("Dublin", -6.26, 53.35, 1.4),
    ("Zürich", 8.54, 47.38, 1.4),
];

/// Where the map opens: over central Europe.
const START: [f64; 3] = [9.0, 49.0, 0.0];
const START_DISTANCE: f32 = 4_200_000.0;
/// The tile grid: no finer than this zoom, 256-pixel tiles, at most this many.
const MAX_ZOOM: u8 = 14;
const TILE_PX: f64 = 256.0;
const MAX_TILES: usize = 300;

fn world(lon: f64, lat: f64) -> [f64; 3] {
    geo::lon_lat_to_world([lon, lat, 0.0])
}

/// OpenStreetMap's standard tiles. Their policy asks for an identifying
/// User-Agent, visible attribution, and no heavy use: the disk cache keeps a
/// demo to a few hundred requests.
const OSM: &str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const OSM_ATTRIBUTION: &str = "© OpenStreetMap contributors";
const USER_AGENT: &str = concat!(
    "threers-map-demo/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/eugenehp/threers)"
);
/// Raster tiles: 256 pixels, no finer than zoom 19, at most this many drawn.
const IMAGERY_MAX_ZOOM: u8 = 19;
const IMAGERY_MAX_TILES: usize = 96;
/// Decoded tiles kept in memory, a quarter of a megabyte each.
const IMAGERY_CACHE: usize = 256;

/// Where tile images come from.
#[derive(Clone)]
enum Source {
    /// A `{z}/{x}/{y}` template, fetched with `curl`.
    Url(String),
    /// Drawn on the spot: no network.
    Procedural,
}

/// One tile's image: 256 × 256 RGBA, or `None` when it could not be had.
fn load_tile(source: &Source, cache: &std::path::Path, t: geo::TileId) -> Option<Vec<u8>> {
    let template = match source {
        Source::Procedural => return Some(procedural_tile(t)),
        Source::Url(u) => u,
    };
    let path = cache.join(format!("{}/{}/{}.png", t.z, t.x, t.y));
    if !path.exists() {
        std::fs::create_dir_all(path.parent()?).ok()?;
        let url = template
            .replace("{z}", &t.z.to_string())
            .replace("{x}", &t.x.to_string())
            .replace("{y}", &t.y.to_string());
        let partial = path.with_extension("part");
        let ok = std::process::Command::new("curl")
            .args(["-sfL", "-m", "20", "-A", USER_AGENT, "-o"])
            .arg(&partial)
            .arg(&url)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            let _ = std::fs::remove_file(&partial);
            return None;
        }
        std::fs::rename(&partial, &path).ok()?;
    }
    let png = threers::decode_png(&std::fs::read(&path).ok()?).ok()?;
    (png.width == 256 && png.height == 256).then_some(png.rgba)
}

/// An offline stand-in: a tint per zoom level, a checker per tile, and the
/// tile's edge.
fn procedural_tile(t: geo::TileId) -> Vec<u8> {
    let hue = t.z as f32 * 0.61;
    let base = [
        (0.5 + 0.5 * hue.cos()) * 60.0 + 30.0,
        (0.5 + 0.5 * (hue + 2.1).cos()) * 60.0 + 40.0,
        (0.5 + 0.5 * (hue + 4.2).cos()) * 60.0 + 60.0,
    ];
    let mut out = Vec::with_capacity(256 * 256 * 4);
    for y in 0..256u32 {
        for x in 0..256u32 {
            let edge = x < 2 || y < 2;
            let check = ((x / 32) + (y / 32) + t.x + t.y).is_multiple_of(2);
            let k = if edge {
                1.8
            } else if check {
                1.1
            } else {
                0.9
            };
            out.extend(base.iter().map(|c| (c * k).min(255.0) as u8));
            out.push(255);
        }
    }
    out
}

/// Raster map tiles under the marks: fetched on worker threads, kept in
/// memory and on disk, packed into an atlas and drawn as ground quads.
struct Imagery {
    requests: mpsc::Sender<geo::TileId>,
    results: mpsc::Receiver<(geo::TileId, Option<Vec<u8>>)>,
    loaded: HashMap<geo::TileId, Arc<Vec<u8>>>,
    pending: HashSet<geo::TileId>,
    failed: HashSet<geo::TileId>,
    atlas: SharedAtlas,
    layer: ImageMarks,
    /// What is drawn: each tile, and the loaded tile whose image covers it.
    drawn: Vec<(geo::TileId, geo::TileId)>,
    /// The atlas holds a fixed pool of 256² slots, packed once. A tile that
    /// comes into view takes a free slot, or the one drawn longest ago, and
    /// costs one tile's upload — the atlas is never repacked, which uploaded
    /// all of it and stalled the frame each time the view crossed a tile.
    slots: Vec<Option<geo::TileId>>,
    /// When each slot was last drawn, in updates.
    slot_used: Vec<u64>,
    slot_of: HashMap<geo::TileId, usize>,
    clock: u64,
    attribution: Option<&'static str>,
}

/// Atlas slot `k`'s name.
fn slot_name(k: usize) -> String {
    format!("slot {k}")
}

impl Imagery {
    fn new(source: Source) -> Self {
        let cache = std::env::temp_dir().join("threers-map-tiles");
        let (requests, jobs) = mpsc::channel::<geo::TileId>();
        let (done, results) = mpsc::channel();
        let jobs = Arc::new(Mutex::new(jobs));
        for _ in 0..4 {
            let (jobs, done, source, cache) =
                (jobs.clone(), done.clone(), source.clone(), cache.clone());
            std::thread::spawn(move || loop {
                let Ok(t) = jobs.lock().unwrap().recv() else {
                    return;
                };
                if done.send((t, load_tile(&source, &cache, t))).is_err() {
                    return;
                }
            });
        }
        // Eight tiles a row with their gutters; the atlas grows downwards.
        let mut atlas = TextureAtlas::new(8 * 260);
        let blank = vec![0u8; 256 * 256 * 4];
        for k in 0..IMAGERY_MAX_TILES {
            atlas
                .insert(slot_name(k), 256, 256, &blank)
                .expect("the slots fit");
        }
        let atlas = atlas.shared();
        let mut layer = ImageMarks::new(atlas.clone());
        // The ground: everything else draws over it.
        layer.common.style.depth_write = false;
        layer.common.style.depth_test = false;
        layer.common.style.pickable = false;
        Self {
            requests,
            results,
            loaded: HashMap::new(),
            pending: HashSet::new(),
            failed: HashSet::new(),
            atlas,
            layer,
            drawn: Vec::new(),
            slots: vec![None; IMAGERY_MAX_TILES],
            slot_used: vec![0; IMAGERY_MAX_TILES],
            slot_of: HashMap::new(),
            clock: 0,
            attribution: matches!(&source, Source::Url(u) if u.as_str() == OSM)
                .then_some(OSM_ATTRIBUTION),
        }
    }

    /// Requests in flight.
    fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Take in finished loads, ask for what `view` needs, and redraw when
    /// what can be shown changed.
    fn update(&mut self, map: &MapView, view: &threers::marks::MarkView) {
        while let Ok((t, image)) = self.results.try_recv() {
            self.pending.remove(&t);
            match image {
                Some(rgba) => {
                    self.loaded.insert(t, Arc::new(rgba));
                }
                None => {
                    self.failed.insert(t);
                }
            }
        }
        let wanted = map.covering_tiles(view, IMAGERY_MAX_ZOOM, 256.0, IMAGERY_MAX_TILES);
        let mut drawn = Vec::with_capacity(wanted.len());
        for &t in &wanted {
            if !self.loaded.contains_key(&t) && !self.failed.contains(&t) && self.pending.insert(t)
            {
                let _ = self.requests.send(t);
            }
            // The tile itself, or the nearest ancestor already here.
            let mut a = Some(t);
            while let Some(c) = a {
                if self.loaded.contains_key(&c) {
                    drawn.push((t, c));
                    break;
                }
                a = c.parent();
            }
        }
        if self.loaded.len() > IMAGERY_CACHE {
            let keep: HashSet<geo::TileId> = drawn.iter().map(|d| d.1).collect();
            let spare: Vec<geo::TileId> = self
                .loaded
                .keys()
                .filter(|t| !keep.contains(t))
                .take(self.loaded.len() - IMAGERY_CACHE)
                .copied()
                .collect();
            for t in spare {
                self.loaded.remove(&t);
            }
        }
        if drawn == self.drawn {
            return;
        }
        self.clock += 1;
        let now = self.clock;
        let mut atlas = self.atlas.write().unwrap();
        // The images this frame shows hold their slots first, so none of them
        // is evicted for another.
        for &(_, a) in &drawn {
            if let Some(&k) = self.slot_of.get(&a) {
                self.slot_used[k] = now;
            }
        }
        let mut quads = Vec::with_capacity(drawn.len());
        for &(t, a) in &drawn {
            let k = match self.slot_of.get(&a) {
                Some(&k) => k,
                None => {
                    // A free slot, else the one drawn longest ago — never one
                    // this frame already uses.
                    let Some(k) = (0..self.slots.len())
                        .filter(|&k| self.slot_used[k] < now)
                        .min_by_key(|&k| (self.slots[k].is_some(), self.slot_used[k]))
                    else {
                        continue;
                    };
                    if let Some(old) = self.slots[k].replace(a) {
                        self.slot_of.remove(&old);
                    }
                    self.slot_of.insert(a, k);
                    self.slot_used[k] = now;
                    if atlas.write(&slot_name(k), &self.loaded[&a]).is_err() {
                        continue;
                    }
                    k
                }
            };
            let Some(rect) = atlas.get(&slot_name(k)) else {
                continue;
            };
            // The part of the ancestor's image over this tile.
            let d = (t.z - a.z) as u32;
            let size = (256 >> d).max(1);
            let rect = threers::marks::AtlasRect {
                x: rect.x + (t.x - (a.x << d)) * size,
                y: rect.y + (t.y - (a.y << d)) * size,
                width: size,
                height: size,
            };
            let (lo, hi) = t.bounds();
            quads.push(ImageQuad::ground(lo, hi, 0.0, rect));
        }
        drop(atlas);
        self.layer.set_quads(&quads);
        self.drawn = drawn;
    }
}

/// A tiny xorshift, so the heatmap is the same every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal, mean 0 and spread 1.
    fn normal(&mut self) -> f64 {
        (0..4).map(|_| self.next()).sum::<f64>() - 2.0
    }
}

struct Layers {
    imagery: Imagery,
    /// The tile provider's credit, kept in the bottom-right corner.
    attribution: TextLabels,
    tiles: SegmentMarks,
    tile_names: TextLabels,
    heat: DensityMarks,
    /// City symbols: pin and name.
    cities: TextLabels,
    /// The tiles the outlines were built for.
    shown_tiles: Vec<geo::TileId>,
}

impl Layers {
    fn new(map: &MapView, source: Source) -> Self {
        let imagery = Imagery::new(source);
        let mut tiles = SegmentMarks::new();
        tiles.set_default_color([90, 120, 160, 140]);
        tiles.set_widths(&[]);
        tiles.common.style.size_scale = 1.0;
        tiles.common.style.depth_test = false;

        let mut tile_names = TextLabels::new(CaptionFont::builtin());
        tile_names.font_size = 11.0;
        tile_names.collision_index = Some(map.collision.clone());
        // Over a map the grid is clutter: shown on request (key 2).
        tiles.common.style.visible = false;
        tile_names.common.style.visible = false;

        // Activity: a cloud around each city, as many as its population.
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let mut heat_points = Vec::new();
        for &(_, lon, lat, pop) in &CITIES {
            for _ in 0..(pop * 400.0) as usize {
                // About 40 km either way.
                let spread = 0.4 + 0.15 * pop.sqrt() as f64;
                heat_points.push(world(
                    lon + rng.normal() * spread / lat.to_radians().cos(),
                    lat + rng.normal() * spread,
                ));
            }
        }
        let mut heat = DensityMarks::new();
        heat.set_positions(&heat_points);
        heat.style_mut().size_scale = 28.0;
        heat.auto_max = Some(0.9);
        // Fade in over the bottom quarter of the ramp: inferno starts black,
        // and a black rim reads as a hole in a light basemap.
        heat.threshold = 0.25;
        heat.common.style.opacity = 0.8;
        heat.set_ramp(&ColorRamp::inferno());

        let font = || {
            CaptionFont::from_ttf_bytes(include_bytes!("../src/openscad/DejaVuSans.ttf"))
                .expect("bundled font")
        };
        let mut attribution = TextLabels::new(font());
        attribution.font_size = 12.0;
        attribution.halo_width = 2.0;
        attribution.halo_color = [255, 255, 255, 200];
        attribution.collision = false;
        let mut cities = TextLabels::new(font());
        cities.glyph_field = GlyphField::Msdf;
        cities.raster_size = 32.0;
        cities.font_size = 15.0;
        cities.halo_width = 2.0;
        cities.halo_color = [10, 12, 20, 220];
        cities.collision_index = Some(map.collision.clone());
        // A pin for the icon: a filled circle, as an alpha mask.
        let pin: Vec<u8> = (0..24 * 24)
            .map(|i| {
                let (x, y) = ((i % 24) as f32 - 11.5, (i / 24) as f32 - 11.5);
                if x * x + y * y <= 11.0 * 11.0 {
                    255
                } else {
                    0
                }
            })
            .collect();
        cities.add_icon("pin", 24, 24, &pin);
        cities.set_labels(
            CITIES
                .iter()
                .map(|&(name, lon, lat, pop)| {
                    let size = 6.0 + 2.5 * pop.sqrt();
                    let mut icon = LabelIcon::new("pin", size);
                    icon.color = [255, 170, 60, 255];
                    let mut l = TextLabel::new(world(lon, lat), name).with_icon(icon);
                    // Text to the right, the pin centred on the city.
                    l.anchor = LabelAnchor::Right;
                    l.offset = [-size * 0.5, 0.0];
                    l.priority = pop;
                    l
                })
                .collect(),
        );

        Self {
            imagery,
            attribution,
            tiles,
            tile_names,
            heat,
            cities,
            shown_tiles: Vec::new(),
        }
    }

    /// Everything that follows the view: map tiles, the grid, the credit.
    fn update(&mut self, map: &MapView, view: &threers::marks::MarkView) {
        self.imagery.update(map, view);
        self.update_tiles(map, view);
        let (w, h) = map.viewport();
        let corner = self
            .imagery
            .attribution
            .zip(map.ground_point(w as f32 - 6.0, h as f32 - 6.0));
        match corner {
            Some((text, at)) => {
                let mut l = TextLabel::new(at, text);
                l.anchor = LabelAnchor::Left;
                l.offset = [0.0, -8.0];
                l.color = [40, 40, 40, 255];
                self.attribution.set_labels(vec![l]);
            }
            None => self.attribution.set_labels(Vec::new()),
        }
    }

    /// Outline and name the tiles `view` covers, when they change.
    fn update_tiles(&mut self, map: &MapView, view: &threers::marks::MarkView) {
        let tiles = map.covering_tiles(view, MAX_ZOOM, TILE_PX, MAX_TILES);
        if tiles == self.shown_tiles {
            return;
        }
        let (mut from, mut to, mut names) = (Vec::new(), Vec::new(), Vec::new());
        for t in &tiles {
            let (lo, hi) = t.bounds();
            let corners = [
                [lo[0], lo[1]],
                [hi[0], lo[1]],
                [hi[0], hi[1]],
                [lo[0], hi[1]],
            ]
            .map(|g| geo::ground_to_world(g, 0.0));
            for k in 0..4 {
                from.push(corners[k]);
                to.push(corners[(k + 1) % 4]);
            }
            let centre = geo::ground_to_world([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5], 0.0);
            let mut name = TextLabel::new(centre, format!("{}/{}/{}", t.z, t.x, t.y));
            name.color = [110, 140, 180, 255];
            // Below every city name.
            name.priority = -1.0;
            names.push(name);
        }
        self.tiles.set_segments(&from, &to);
        self.tile_names.set_labels(names);
        self.shown_tiles = tiles;
    }

    /// In draw order: earlier labels win collisions, so cities before tile
    /// names.
    fn all(&mut self) -> [&mut dyn Mark; 6] {
        [
            &mut self.imagery.layer,
            &mut self.tiles,
            &mut self.heat,
            &mut self.cities,
            &mut self.tile_names,
            &mut self.attribution,
        ]
    }
}

fn scene() -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x0b1220);
    scene
}

fn opening_map(viewport: (u32, u32), distance: f32) -> MapView {
    let mut map = MapView::new(world(START[0], START[1]), distance, viewport);
    map.controls.max_pitch = 70f32.to_radians();
    map.controls.min_distance = 200.0;
    map.controls.max_distance = 30_000_000.0;
    map
}

fn main() {
    env_logger::init();
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|j| args.get(j + 1))
            .map(|s| s.as_str())
    };
    let distance = flag("--distance")
        .and_then(|d| d.parse().ok())
        .unwrap_or(START_DISTANCE);
    let source = match flag("--tiles") {
        Some("none") => Source::Procedural,
        Some(url) => Source::Url(url.to_owned()),
        None => Source::Url(OSM.to_owned()),
    };
    if args.iter().any(|a| a == "--png") {
        snapshot(flag("--png").unwrap_or("map_demo.png"), distance, source);
    } else {
        pollster::block_on(run(distance, source));
    }
}

/// The opening view rendered headless to `path`, once its tiles are in (or
/// a minute has passed).
fn snapshot(path: &str, distance: f32, source: Source) {
    let (w, h) = (1200, 800);
    let mut hr = threers::HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("headless renderer");
    // 4× MSAA: off by default, as three.js's `antialias` is.
    hr.set_msaa(4);
    let mut map = opening_map((w, h), distance);
    map.controls.pitch = 0.5;
    let mut layers = Layers::new(&map, source);
    let mut marks = MarkRenderer::new(hr.device());
    let frame = map.begin_frame();
    let start = Instant::now();
    loop {
        layers.update(&map, &frame.view);
        if layers.imagery.pending() == 0 || start.elapsed() > Duration::from_secs(60) {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    hr.render(&mut scene(), &map.camera);
    let (device, queue) = (hr.device().clone(), hr.queue().clone());
    marks.draw(
        &device,
        &queue,
        &hr.mark_target(),
        &frame.view,
        &mut layers.all(),
    );
    let rgba = hr.read_rgba_resolved();
    std::fs::write(path, threers::encode_png(w, h, &rgba)).expect("write png");
    println!(
        "wrote {path}: zoom {:.1}, {} map tiles ({} missing), {} of {} city names shown",
        map.zoom(TILE_PX),
        layers.imagery.drawn.len(),
        layers.imagery.failed.len() + layers.imagery.pending(),
        layers.cities.visible().iter().filter(|v| **v).count(),
        CITIES.len()
    );
}

async fn run(distance: f32, source: Source) {
    let event_loop = EventLoop::new().expect("event loop");
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("threers — map demo")
            .with_inner_size(winit::dpi::LogicalSize::new(1200, 800))
            .build(&event_loop)
            .expect("window"),
    );
    let instance = {
        let mut d = wgpu::InstanceDescriptor::new_without_display_handle();
        d.backends = wgpu::Backends::PRIMARY;
        wgpu::Instance::new(d)
    };
    let surface = instance.create_surface(window.clone()).expect("surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("adapter");
    // The same limits `controls_orbit` raises for `Renderer::new`.
    let adapter_limits = adapter.limits();
    let mut limits = wgpu::Limits::downlevel_defaults().using_resolution(adapter_limits.clone());
    limits.max_sampled_textures_per_shader_stage =
        adapter_limits.max_sampled_textures_per_shader_stage;
    limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("threers device"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await
        .expect("device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);

    let size = window.inner_size();
    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
        color_space: wgpu::SurfaceColorSpace::Srgb,
    };
    surface.configure(&device, &config);
    let mut renderer = Renderer::new(
        device.clone(),
        queue.clone(),
        format,
        config.width,
        config.height,
    );
    renderer.set_msaa(4);

    let mut scene = scene();
    let mut map = opening_map((config.width, config.height), distance);
    let mut layers = Layers::new(&map, source);
    let mut marks = MarkRenderer::new(&device);
    let mut buttons = (false, false); // (left: pan, right: rotate)
    let mut cursor = (0.0f32, 0.0f32);
    let window_for_loop = window.clone();

    event_loop
        .run(move |event, target| {
            let Event::WindowEvent { event, window_id } = event else {
                return;
            };
            if window_id != window_for_loop.id() {
                return;
            }
            match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(s) => {
                    config.width = s.width.max(1);
                    config.height = s.height.max(1);
                    surface.configure(&device, &config);
                    renderer.resize(config.width, config.height);
                    map.set_viewport(config.width, config.height);
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
                    let toggle = |m: &mut dyn Mark| {
                        let s = m.style_mut();
                        s.visible = !s.visible;
                    };
                    match code {
                        KeyCode::Digit1 => toggle(&mut layers.imagery.layer),
                        KeyCode::Digit2 => {
                            toggle(&mut layers.tiles);
                            toggle(&mut layers.tile_names);
                        }
                        KeyCode::Digit3 => toggle(&mut layers.heat),
                        KeyCode::Digit4 => toggle(&mut layers.cities),
                        _ => return,
                    }
                    // Hidden labels leave boxes behind in the shared index.
                    map.collision.lock().unwrap().invalidate();
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
                    };
                    cursor = now;
                    map.handle(ev, cursor);
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let wheel = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y * 40.0,
                        MouseScrollDelta::PixelDelta(p) => p.y as f32,
                    };
                    map.handle(
                        PointerEvent {
                            wheel,
                            ..Default::default()
                        },
                        cursor,
                    );
                }
                WindowEvent::RedrawRequested => {
                    let surface_frame = match surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(f)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
                        wgpu::CurrentSurfaceTexture::Lost
                        | wgpu::CurrentSurfaceTexture::Outdated => {
                            surface.configure(&device, &config);
                            window_for_loop.request_redraw();
                            return;
                        }
                        wgpu::CurrentSurfaceTexture::Occluded
                        | wgpu::CurrentSurfaceTexture::Timeout => {
                            window_for_loop.request_redraw();
                            return;
                        }
                        e => {
                            log::error!("surface error: {e:?}");
                            return;
                        }
                    };
                    let frame_view = surface_frame.texture.create_view(&Default::default());
                    // Origin, camera, clip planes, the marks' view and the
                    // collision frame, in that order.
                    let frame = map.begin_frame_for(&renderer);
                    layers.update(&map, &frame.view);
                    renderer.render(&mut scene, &map.camera, &frame_view, false);
                    let target = MarkTarget::over_frame(&renderer, &frame_view);
                    marks.draw(&device, &queue, &target, &frame.view, &mut layers.all());
                    queue.present(surface_frame);
                    let ll = geo::world_to_ground(map.target()).0;
                    let ll = geo::inverse_mercator(ll);
                    window_for_loop.set_title(&format!(
                        "threers — map demo · {:.3}°, {:.3}° · zoom {:.1} · {} map tiles, {} loading",
                        ll[1],
                        ll[0],
                        map.zoom(TILE_PX),
                        layers.imagery.drawn.len(),
                        layers.imagery.pending()
                    ));
                    window_for_loop.request_redraw();
                }
                _ => {}
            }
        })
        .expect("event loop");
}
