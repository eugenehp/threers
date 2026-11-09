//! Marks on the native Metal backend draw what they draw on wgpu.
//!
//! The same `HeadlessRenderer` program — a scene, then layers of marks over
//! it through `mark_renderer` / `draw_marks` — under `THREERS_BACKEND=metal`
//! and without it. One test, run in sequence: the backend is read from the
//! environment when a renderer is built.

#![cfg(metal_backend)]

use threers::captions::CaptionFont;
use threers::lights::{AmbientLight, DirectionalLight};
use threers::marks::{
    DensityMarks, Mark, MarkView, Origin, PointMarks, PolygonFeature, PolygonMarks, SegmentMarks,
    ShapeMarks, TextLabel, TextLabels,
};
use threers::RenderBackend;
use threers::{
    BoxGeometry, Color, HeadlessRenderer, Mesh, Object3D, PerspectiveCamera, PlaneGeometry, Scene,
    StandardMaterial, Vector3,
};

const W: u32 = 160;
const H: u32 = 120;

fn scene() -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x10204a);
    s.add_light(AmbientLight::new(Color::WHITE, 0.3));
    s.add_light(
        DirectionalLight::new(Color::WHITE, 2.0).with_direction(Vector3::new(-0.4, -1.0, -0.6)),
    );
    let mut floor = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(6.0, 6.0),
        StandardMaterial::new(Color::from_hex(0x336644)).into(),
    ));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    s.add(floor);
    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        StandardMaterial::new(Color::from_hex(0xff8844)).into(),
    ));
    cube.position = Vector3::new(0.0, 0.5, 0.0);
    s.add(cube);
    s
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(50.0, W as f32 / H as f32, 0.1, 100.0);
    c.position = Vector3::new(2.5, 2.0, 3.5);
    c.look_at(Vector3::new(0.0, 0.4, 0.0));
    c
}

fn build(metal: bool, msaa: u32) -> HeadlessRenderer {
    let mut hr = HeadlessRenderer::builder()
        .backend(if metal {
            RenderBackend::Metal
        } else {
            RenderBackend::Wgpu
        })
        .size(W, H)
        .build()
        .expect("headless renderer");
    hr.set_msaa(msaa);
    assert_eq!(hr.is_native_metal(), metal);
    hr
}

fn diff(a: &[u8], b: &[u8]) -> (f64, u8) {
    assert_eq!(a.len(), b.len());
    let mut d: Vec<u8> = a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .flat_map(|(p, q)| (0..3).map(move |c| p[c].abs_diff(q[c])))
        .collect();
    let mean = d.iter().map(|&v| v as f64).sum::<f64>() / d.len() as f64;
    d.sort_unstable();
    (mean, d[d.len() * 99 / 100])
}

/// Points through the cube (half hidden by it) and a path over the floor.
fn layers() -> (PointMarks, SegmentMarks) {
    let mut points = PointMarks::new();
    let pos: Vec<[f64; 3]> = (0..9).map(|i| [-1.6 + i as f64 * 0.4, 0.5, 0.0]).collect();
    points.set_positions(&pos);
    points.set_colors(
        &(0..9)
            .map(|i| [255, (i * 28) as u8, 40, 255])
            .collect::<Vec<_>>(),
    );
    points.style_mut().size_scale = 7.0;
    let mut path = SegmentMarks::new();
    path.set_paths(&[vec![[-2.0, 0.01, 1.5], [0.0, 0.01, 2.0], [2.0, 0.01, 1.0]]]);
    path.style_mut().size_scale = 4.0;
    (points, path)
}

/// The rest of the kinds: instanced boxes, an extruded polygon with a hole,
/// a density field, and text labels.
struct More {
    shapes: ShapeMarks,
    polygons: PolygonMarks,
    density: DensityMarks,
    labels: TextLabels,
}

fn more(view: &MarkView) -> More {
    let mut shapes = ShapeMarks::new(&BoxGeometry::new(0.3, 0.3, 0.3));
    shapes.set_positions(&[[-1.5, 0.15, -1.5], [1.5, 0.15, -1.5], [1.5, 0.15, 1.5]]);
    let square = |x: f64, z: f64, r: f64| {
        vec![
            [x - r, z - r],
            [x + r, z - r],
            [x + r, z + r],
            [x - r, z + r],
        ]
    };
    let mut tower = PolygonFeature::new(square(-1.6, 1.0, 0.5));
    tower.holes.push(square(-1.6, 1.0, 0.2));
    let mut polygons = PolygonMarks::new();
    polygons.set_polygons(&[tower], 0.0);
    polygons.set_colors(&[[80, 160, 255, 220]]);
    polygons.set_elevations(&[0.6]);
    let mut density = DensityMarks::new();
    density.set_positions(
        &(0..40)
            .map(|i| [0.8 + (i % 7) as f64 * 0.1, 0.0, -0.5 - (i / 7) as f64 * 0.1])
            .collect::<Vec<_>>(),
    );
    let mut labels = TextLabels::new(CaptionFont::builtin());
    labels.font_size = 14.0;
    labels.set_labels(vec![
        TextLabel::new([0.0, 1.3, 0.0], "cube"),
        TextLabel::new([-1.6, 0.7, -1.0], "tower"),
    ]);
    labels.place(view);
    More {
        shapes,
        polygons,
        density,
        labels,
    }
}

#[test]
fn marks_on_metal_draw_what_they_draw_on_wgpu() {
    if threers::metal::MetalDevice::new().is_err() {
        eprintln!("skipping: no Metal device");
        return;
    }
    for msaa in [1, 4] {
        let mut picks = Vec::new();
        let frames: Vec<Vec<u8>> = [false, true]
            .into_iter()
            .map(|metal| {
                let mut hr = build(metal, msaa);
                let cam = camera();
                hr.render(&mut scene(), &cam);
                let mut marks = hr.mark_renderer();
                let view = MarkView::new(&cam, Origin::ZERO, (W, H));
                let (mut points, mut path) = layers();
                let mut m = more(&view);
                // A pick, before the marks are drawn: the same answer.
                let [sx, sy, _] = view.project([-0.4, 0.5, 0.0]).expect("on screen");
                let hit = hr.pick_marks(
                    &mut marks,
                    &view,
                    &mut [&mut points as &mut dyn Mark],
                    (sx, sy),
                    4,
                );
                picks.push(hit.map(|h| (h.mark, h.index)));
                let bare = hr.read_rgba();
                hr.draw_marks(
                    &mut marks,
                    &view,
                    &mut [
                        &mut points as &mut dyn Mark,
                        &mut path,
                        &mut m.shapes,
                        &mut m.polygons,
                        &mut m.density,
                        &mut m.labels,
                    ],
                );
                assert_eq!(hr.is_native_metal(), metal, "marks stay on the backend");
                let drawn = hr.read_rgba();
                // The marks are there: a few hundred pixels of them.
                let changed = bare
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .zip(drawn.as_chunks::<4>().0)
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    changed > 300,
                    "metal={metal} msaa{msaa}: marks changed only {changed} pixels"
                );
                drawn
            })
            .collect();
        assert!(picks[0].is_some(), "msaa{msaa}: wgpu picks a point");
        assert_eq!(
            picks[0], picks[1],
            "msaa{msaa}: the pick on Metal is wgpu's"
        );
        let (mean, p99) = diff(&frames[0], &frames[1]);
        assert!(
            mean < 2.0 && p99 < 24,
            "msaa{msaa}: marks on wgpu and Metal differ by {mean:.2} mean, {p99} at the 99th percentile"
        );
    }

    // `hr.mark_target()` with a renderer on `marks_gpu`: the same frame as
    // `draw_marks`, and still on Metal.
    let cam = camera();
    let view = MarkView::new(&cam, Origin::ZERO, (W, H));
    let via: Vec<Vec<u8>> = [false, true]
        .into_iter()
        .map(|by_target| {
            let mut hr = build(true, 4);
            hr.render(&mut scene(), &cam);
            let mut marks = hr.mark_renderer();
            let (mut points, mut path) = layers();
            let layers: &mut [&mut dyn Mark] = &mut [&mut points, &mut path];
            if by_target {
                let gpu = hr.marks_gpu();
                let target = hr.mark_target();
                marks.draw_on(&gpu, &target, &view, layers);
            } else {
                hr.draw_marks(&mut marks, &view, layers);
            }
            assert!(
                hr.is_native_metal(),
                "mark_target keeps the renderer on Metal"
            );
            hr.read_rgba()
        })
        .collect();
    assert_eq!(via[0], via[1], "mark_target draws what draw_marks draws");

    // Points binned into screen cells by a compute pass: the same counts.
    use threers::marks::ScreenBinner;
    let points: Vec<[f32; 3]> = (0..500)
        .map(|i| {
            [
                ((i * 37) % 100) as f32 * 0.04 - 2.0,
                0.5,
                ((i * 11) % 50) as f32 * 0.04 - 1.0,
            ]
        })
        .collect();
    let weights: Vec<f32> = (0..500).map(|i| (i % 7) as f32 * 0.5).collect();
    let bins: Vec<_> = [false, true]
        .into_iter()
        .map(|metal| {
            let mut hr = build(metal, 1);
            let gpu = hr.marks_gpu();
            assert!(ScreenBinner::supported_on(&gpu));
            let binner = ScreenBinner::new_on(&gpu, &points, Some(&weights));
            binner
                .bin_on(&gpu, view.view_proj, (W, H), 16.0)
                .wait_on(&gpu)
                .expect("bins")
        })
        .collect();
    assert!(
        bins[0].counts.iter().sum::<u32>() > 100,
        "points land in cells"
    );
    assert_eq!(bins[0], bins[1], "the binning on Metal is wgpu's");
}
