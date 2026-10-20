//! How the marks hold up at kepler.gl sizes.
//!
//! ```text
//! cargo run --release --example marks_bench
//! cargo run --release --example marks_bench -- --size 3840x2160 --frames 60
//! ```
//!
//! Each layer is timed alone, headless, far from the origin the way map data
//! is: building it on the CPU, its first frame (upload and pipeline), its
//! steady frame, a frame after rewriting 1,000 of its rows, and a frame after
//! moving a GPU filter. A frame is the draw plus waiting for the GPU to finish
//! it, so it is an upper bound on what the layer adds to a real frame, which
//! would overlap the wait with other work.

use std::time::Instant;

use threers::marks::{
    DensityMarks, Mark, MarkRenderer, MarkView, Origin, PointMarks, PolygonFeature, PolygonMarks,
    SegmentMarks, ShaderHooks,
};
use threers::{HeadlessRenderer, MapControls, PerspectiveCamera, Scene, Vector3};

fn arg(flag: &str) -> Option<String> {
    std::env::args()
        .position(|a| a == flag)
        .and_then(|i| std::env::args().nth(i + 1))
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

const CENTRE: [f64; 3] = [-13_627_000.0, 0.0, 4_548_000.0];
/// Data spread over a 20 km square.
const SPAN: f64 = 20_000.0;

fn somewhere(rng: &mut Rng) -> [f64; 3] {
    [
        CENTRE[0] + (rng.next() - 0.5) * SPAN,
        0.0,
        CENTRE[2] + (rng.next() - 0.5) * SPAN,
    ]
}

struct Timing {
    name: String,
    rows: usize,
    build_ms: f64,
    first_ms: f64,
    frame_ms: f64,
    update_ms: f64,
    filter_ms: Option<f64>,
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

struct Bench {
    hr: HeadlessRenderer,
    marks: MarkRenderer,
    view: MarkView,
    frames: usize,
}

impl Bench {
    /// Draw `mark` once and wait for the GPU.
    fn frame(&mut self, mark: &mut dyn Mark) -> f64 {
        let (device, queue) = (self.hr.device().clone(), self.hr.queue().clone());
        let t = Instant::now();
        self.marks.draw(
            &device,
            &queue,
            &self.hr.mark_target(),
            &self.view,
            &mut [mark],
        );
        let cpu = ms(t);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        if std::env::var_os("CPU_ONLY").is_some() {
            cpu
        } else {
            ms(t)
        }
    }

    fn steady(&mut self, mark: &mut dyn Mark) -> f64 {
        let mut total = 0.0;
        for _ in 0..self.frames {
            total += self.frame(mark);
        }
        total / self.frames as f64
    }

    fn run<M: Mark>(
        &mut self,
        name: &str,
        rows: usize,
        build_ms: f64,
        mark: &mut M,
        mut update: impl FnMut(&mut M),
        filtered: bool,
    ) -> Timing {
        let first_ms = self.frame(mark);
        let frame_ms = self.steady(mark);
        let t = Instant::now();
        update(mark);
        let _ = self.frame(mark);
        let update_ms = ms(t);
        let filter_ms = filtered.then(|| {
            mark.style_mut().params[0] = [0.25, 0.75, 0.0, 0.0];
            let t = Instant::now();
            let _ = self.frame(mark);
            ms(t)
        });
        Timing {
            name: name.into(),
            rows,
            build_ms,
            first_ms,
            frame_ms,
            update_ms,
            filter_ms,
        }
    }
}

fn main() {
    let (w, h) = arg("--size")
        .and_then(|s| {
            let (a, b) = s.split_once('x')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .unwrap_or((1920u32, 1080u32));
    let frames = arg("--frames").and_then(|v| v.parse().ok()).unwrap_or(30);
    let scale: f64 = arg("--scale").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let n = |base: usize| ((base as f64) * scale).max(1.0) as usize;

    let mut hr = HeadlessRenderer::builder()
        .size(w, h)
        .build()
        .expect("no GPU adapter");
    let origin = Origin::new(CENTRE[0], CENTRE[1], CENTRE[2]);
    let mut camera = PerspectiveCamera::new(45.0, w as f32 / h as f32, 10.0, 200_000.0);
    let mut controls = MapControls::new(Vector3::ZERO, 26_000.0);
    controls.pitch = 0.7;
    controls.apply(&mut camera);
    hr.render(&mut Scene::new(), &camera);
    let marks = MarkRenderer::new(hr.device());
    let view = MarkView::new(&camera, origin, (w, h));
    let mut bench = Bench {
        hr,
        marks,
        view,
        frames,
    };
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut results = Vec::new();

    // Points, with a custom value per row and a range filter on it.
    {
        let rows = n(1_000_000);
        let t = Instant::now();
        let positions: Vec<[f64; 3]> = (0..rows).map(|_| somewhere(&mut rng)).collect();
        let colors: Vec<[u8; 4]> = (0..rows)
            .map(|i| [(i % 251) as u8, 120, 255 - (i % 251) as u8, 200])
            .collect();
        let values: Vec<f32> = (0..rows).map(|_| rng.next() as f32).collect();
        let mut p = PointMarks::new();
        p.set_positions(&positions);
        p.set_colors(&colors);
        p.common.set_custom(0, 1, &values);
        p.common.hooks = ShaderHooks::range_filter(0, 0);
        p.common.style.params[0] = [0.0, 1.0, 0.0, 0.0];
        p.common.style.size_scale = 2.0;
        let build = ms(t);
        let mut k = 0usize;
        results.push(bench.run(
            "points",
            rows,
            build,
            &mut p,
            |p| {
                k = (k + 7919) % rows.saturating_sub(1000).max(1);
                p.update_colors(k, &[[255, 255, 255, 255]; 1000]);
            },
            true,
        ));
    }

    // Paths: 10,000 trips of 11 points, mitred.
    {
        let paths_n = n(10_000);
        let t = Instant::now();
        let paths: Vec<Vec<[f64; 3]>> = (0..paths_n)
            .map(|_| {
                let mut p = somewhere(&mut rng);
                (0..11)
                    .map(|_| {
                        p[0] += (rng.next() - 0.5) * 600.0;
                        p[2] += (rng.next() - 0.5) * 600.0;
                        p
                    })
                    .collect()
            })
            .collect();
        let mut s = SegmentMarks::new();
        s.set_paths(&paths);
        s.set_default_color([0, 230, 195, 160]);
        s.set_widths(&vec![2.0; paths_n * 10]);
        let rows = s.instance_count();
        let values: Vec<f32> = (0..rows).map(|_| rng.next() as f32).collect();
        s.common.set_custom(0, 1, &values);
        s.common.hooks = ShaderHooks::range_filter(0, 0);
        s.common.style.params[0] = [0.0, 1.0, 0.0, 0.0];
        let build = ms(t);
        results.push(bench.run(
            "path segments",
            rows,
            build,
            &mut s,
            |s| s.update_widths(500, &[4.0; 1000]),
            true,
        ));
    }

    // Polygons: 50,000 footprints, a quarter with a courtyard, extruded.
    {
        let rows = n(50_000);
        let t = Instant::now();
        let features: Vec<PolygonFeature> = (0..rows)
            .map(|i| {
                let c = somewhere(&mut rng);
                let (x, y) = (c[0], -c[2]);
                let s = 20.0 + 60.0 * rng.next();
                let rect = |x0: f64, y0: f64, x1: f64, y1: f64| {
                    vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                };
                let mut f = PolygonFeature::new(rect(x, y, x + s, y + s));
                if i % 4 == 0 {
                    f.holes
                        .push(rect(x + s * 0.3, y + s * 0.3, x + s * 0.7, y + s * 0.7));
                }
                f
            })
            .collect();
        let gen = ms(t);
        let t = Instant::now();
        let mut p = PolygonMarks::new();
        p.set_polygons(&features, 0.0);
        let triangulate = ms(t);
        let heights: Vec<f32> = (0..rows)
            .map(|_| 10.0 + 200.0 * rng.next() as f32)
            .collect();
        p.set_elevations(&heights);
        p.set_default_color([120, 140, 190, 255]);
        let build = gen + ms(t);
        println!(
            "polygons: {rows} features, {} vertices; triangulating took {triangulate:.0} ms",
            p.vertex_count()
        );
        results.push(bench.run(
            "polygons",
            rows,
            build,
            &mut p,
            |p| p.update_elevations(100, &[300.0; 1000]),
            false,
        ));
    }

    // Density: 200,000 kernels 30 px across.
    {
        let rows = n(200_000);
        let t = Instant::now();
        let positions: Vec<[f64; 3]> = (0..rows).map(|_| somewhere(&mut rng)).collect();
        let mut d = DensityMarks::new();
        d.set_positions(&positions);
        d.max_weight = 50.0;
        let build = ms(t);
        results.push(bench.run(
            "density kernels",
            rows,
            build,
            &mut d,
            |d| d.update_weights(0, &[2.0; 1000]),
            false,
        ));
    }

    println!("\n{w}×{h}, {frames} frames averaged; times in ms\n");
    println!(
        "{:<16} {:>10} {:>9} {:>9} {:>9} {:>13} {:>11}",
        "layer", "rows", "build", "first", "frame", "1k-row edit", "filter move"
    );
    for r in &results {
        println!(
            "{:<16} {:>10} {:>9.1} {:>9.1} {:>9.2} {:>13.2} {:>11}",
            r.name,
            r.rows,
            r.build_ms,
            r.first_ms,
            r.frame_ms,
            r.update_ms,
            r.filter_ms.map_or("—".into(), |f| format!("{f:.2}")),
        );
    }
}
