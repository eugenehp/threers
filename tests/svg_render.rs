//! End-to-end checks on [`SvgRenderer`].
//!
//! The unit tests beside the renderer check its pieces — winding, clipping,
//! colour encoding. This file checks the thing itself: for a grid of pixels,
//! does the SVG paint the surface that is actually there?
//!
//! Ground truth comes from [`Raycaster`], not from the wgpu renderer. Comparing
//! two renderers only tells you they disagree, and it needs a GPU and an SVG
//! rasteriser to run at all. Casting a ray per sample and asking the geometry
//! directly needs neither, runs in CI, and is right by construction — it is the
//! definition of what the picture should be, so a disagreement is a renderer
//! bug and never a fixture drifting.
//!
//! The SVG is "rasterised" by walking the document the way a viewer paints it:
//! the last `<path>` covering a point is the colour you see. That is the whole
//! painter's algorithm, so it tests exactly what a browser would show.

use std::collections::HashMap;

use threers::cameras::Camera;
use threers::core::ObjectId;
use threers::prelude::*;
use threers::{OrthographicCamera, Raycaster, SphereGeometry, TorusKnotGeometry};

// ======================================================================
//                         SVG → WHAT YOU SEE
// ======================================================================

/// One filled polygon parsed back out of the document, in paint order.
struct Painted {
    poly: Vec<[f32; 2]>,
    fill: String,
}

fn parse_paths(svg: &str) -> Vec<Painted> {
    let mut out = Vec::new();
    for chunk in svg.split("<path ").skip(1) {
        let Some(d) = attr(chunk, "d") else { continue };
        let Some(fill) = attr(chunk, "fill") else {
            continue;
        };
        if fill == "none" {
            continue;
        }
        let poly = flatten_path(d);
        if poly.len() >= 3 {
            out.push(Painted {
                poly,
                fill: fill.to_string(),
            });
        }
    }
    out
}

/// Walk a `d` string into the polygon a viewer would fill.
///
/// Faces come out with curved edges when the geometry has vertex normals, so
/// this has to understand `C` as well as `L` — and flattening it here is right:
/// the question a containment test asks is which pixels the filled path covers,
/// and that is the flattened curve.
fn flatten_path(d: &str) -> Vec<[f32; 2]> {
    const STEPS: usize = 8;
    let mut poly: Vec<[f32; 2]> = Vec::new();
    let mut cursor = [0.0f32, 0.0];
    let mut i = 0;
    let bytes = d.as_bytes();
    while i < d.len() {
        let command = bytes[i] as char;
        i += 1;
        if command == 'Z' {
            continue;
        }
        let start = i;
        while i < d.len() && !bytes[i].is_ascii_alphabetic() {
            i += 1;
        }
        let n: Vec<f32> = d[start..i]
            .split(' ')
            .filter(|t| !t.is_empty())
            .map(|t| t.parse().expect("coordinate"))
            .collect();
        match command {
            'M' | 'L' => {
                assert_eq!(n.len(), 2, "{command} takes one point");
                cursor = [n[0], n[1]];
                poly.push(cursor);
            }
            'C' => {
                assert_eq!(n.len(), 6, "C takes three points");
                let p0 = cursor;
                let (c1, c2, to) = ([n[0], n[1]], [n[2], n[3]], [n[4], n[5]]);
                for k in 1..=STEPS {
                    let t = k as f32 / STEPS as f32;
                    let u = 1.0 - t;
                    let mut p = [0.0f32; 2];
                    for a in 0..2 {
                        p[a] = u * u * u * p0[a]
                            + 3.0 * u * u * t * c1[a]
                            + 3.0 * u * t * t * c2[a]
                            + t * t * t * to[a];
                    }
                    poly.push(p);
                }
                cursor = to;
            }
            other => panic!("unexpected path command {other:?} in {d:?}"),
        }
    }
    poly
}

fn attr<'a>(chunk: &'a str, name: &str) -> Option<&'a str> {
    chunk
        .split(&format!("{name}=\""))
        .nth(1)
        .and_then(|t| t.split('"').next())
}

/// The colour a viewer ends up seeing at `(x, y)`: the last polygon covering it.
fn painted_at(paths: &[Painted], x: f32, y: f32) -> Option<&str> {
    paths
        .iter()
        .rev()
        .find(|p| contains(&p.poly, x, y))
        .map(|p| p.fill.as_str())
}

/// Ray casting along +x. Polygons here are convex triangles or the quads the
/// near clip leaves behind, so the parity rule is exact.
fn contains(poly: &[[f32; 2]], x: f32, y: f32) -> bool {
    let mut inside = false;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        if (a[1] > y) != (b[1] > y) {
            let t = (y - a[1]) / (b[1] - a[1]);
            if x < a[0] + (b[0] - a[0]) * t {
                inside = !inside;
            }
        }
    }
    inside
}

// ======================================================================
//                              GROUND TRUTH
// ======================================================================

/// What is actually in front of the camera at this pixel, as a colour key.
/// `None` means the ray escaped and the background should show.
fn truth_at(
    scene: &Scene,
    camera: &dyn Camera,
    keys: &HashMap<ObjectId, String>,
    perspective: bool,
    ndc: Vector2,
) -> Option<String> {
    let mut rc = Raycaster::new(Vector3::ZERO, Vector3::new(0.0, 0.0, -1.0), 0.0, 1e6);
    let (near, far) = camera.near_far();
    rc.near = near;
    rc.far = far;
    if perspective {
        rc.set_from_camera_perspective(ndc, camera);
    } else {
        rc.set_from_camera_ortho(ndc, camera);
    }
    let hits = rc.intersect_objects(&scene.arena, scene.root, true);
    hits.first().and_then(|h| keys.get(&h.object).cloned())
}

struct Report {
    tested: usize,
    skipped: usize,
    mismatches: Vec<(f32, f32, String, String)>,
}

impl Report {
    fn assert_clean(&self, label: &str) {
        assert!(
            self.tested > 200,
            "{label}: only {} samples landed in testable interior — the scene is \
             too small or too busy to say anything",
            self.tested
        );
        if !self.mismatches.is_empty() {
            let shown: Vec<String> = self
                .mismatches
                .iter()
                .take(8)
                .map(|(x, y, want, got)| format!("  ({x:.0},{y:.0}) want {want} got {got}"))
                .collect();
            panic!(
                "{label}: {} of {} sampled pixels show the wrong surface \
                 ({} skipped as edges)\n{}",
                self.mismatches.len(),
                self.tested,
                self.skipped,
                shown.join("\n"),
            );
        }
    }
}

/// Sample a grid and compare the SVG against the geometry.
///
/// Pixels near a silhouette are skipped: there the answer genuinely depends on
/// sub-pixel coverage, and a renderer is not wrong for landing either side of
/// an edge. A sample counts only if the four points a few pixels away agree
/// with it, which keeps the comparison to the interiors of regions where there
/// is exactly one right answer.
fn compare(
    scene: &mut Scene,
    camera: &dyn Camera,
    perspective: bool,
    w: u32,
    h: u32,
    options: SvgOptions,
) -> Report {
    let svg = SvgRenderer::new(w, h)
        .with_options(options)
        .render_to_string(scene, camera);
    let paths = parse_paths(&svg);

    let background = hex_of(scene.background);
    let mut keys: HashMap<ObjectId, String> = HashMap::new();
    scene.arena.traverse_visible(scene.root, &mut |id, obj| {
        if let ObjectKind::Mesh(m) = &obj.kind {
            keys.insert(id, hex_of(m.material.color()));
        }
    });

    let to_ndc =
        |x: f32, y: f32| Vector2::new((x / w as f32) * 2.0 - 1.0, 1.0 - (y / h as f32) * 2.0);

    let step = 6.0;
    let probe = 3.0;
    let mut report = Report {
        tested: 0,
        skipped: 0,
        mismatches: Vec::new(),
    };
    let mut y = step;
    while y < h as f32 - step {
        let mut x = step;
        while x < w as f32 - step {
            let want = truth_at(scene, camera, &keys, perspective, to_ndc(x, y));
            let interior = [
                (-probe, -probe),
                (probe, -probe),
                (-probe, probe),
                (probe, probe),
            ]
            .iter()
            .all(|(dx, dy)| {
                truth_at(scene, camera, &keys, perspective, to_ndc(x + dx, y + dy)) == want
            });
            if !interior {
                report.skipped += 1;
                x += step;
                continue;
            }
            report.tested += 1;
            let want = want.unwrap_or_else(|| background.clone());
            let got = painted_at(&paths, x, y).unwrap_or(&background);
            if got != want {
                report.mismatches.push((x, y, want, got.to_string()));
            }
            x += step;
        }
        y += step;
    }
    report
}

fn hex_of(c: Color) -> String {
    // `Color` is linear; SVG attributes are sRGB. Materials in these scenes are
    // built from `from_hex`, so this is the inverse of that decode and the
    // renderer should hand back exactly the literal that went in.
    let enc = |v: f32| {
        let s = if v <= 0.0031308 {
            v * 12.92
        } else {
            v.max(0.0).powf(1.0 / 2.4) * 1.055 - 0.055
        };
        (s.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    };
    format!("#{:02x}{:02x}{:02x}", enc(c.r), enc(c.g), enc(c.b))
}

// ======================================================================
//                               SCENES
// ======================================================================

/// Unlit, so the fill in the document is the material's own colour and a
/// mismatch names the object that should have been there.
fn flat(hex: u32) -> Material {
    Material::Basic(BasicMaterial::new(Color::from_hex(hex)))
}

fn mesh_at(geometry: BufferGeometry, hex: u32, pos: Vector3) -> Object3D {
    let mut o = Object3D::mesh(Mesh::new(geometry, flat(hex)));
    o.position = pos;
    o
}

fn look(pos: Vector3, at: Vector3, aspect: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, aspect, 0.1, 200.0);
    c.position = pos;
    c.target = at;
    c
}

fn ground(size: f32, hex: u32, y: f32) -> Object3D {
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(size, size), flat(hex)));
    floor.position = Vector3::new(0.0, y, 0.0);
    floor.quaternion =
        Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), -std::f32::consts::FRAC_PI_2);
    floor
}

const BG: u32 = 0x101820;

fn opts() -> SvgOptions {
    SvgOptions {
        shading: SvgShading::Flat,
        ..SvgOptions::default()
    }
}

// ----------------------------------------------------------------------

/// The reported bug, as an end-to-end check: a two-triangle ground plane that
/// runs off behind the camera, with things standing on it.
#[test]
fn ground_plane_with_objects_on_it() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(ground(80.0, 0x336699, 0.0));
    scene.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(-2.0, 1.0, 0.0),
    ));
    scene.add(mesh_at(
        SphereGeometry::new(1.0, 24, 16),
        0x22aa55,
        Vector3::new(1.6, 1.0, 0.5),
    ));

    let camera = look(
        Vector3::new(0.0, 3.0, 9.0),
        Vector3::new(0.0, 1.0, 0.0),
        4.0 / 3.0,
    );
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("ground plane");
}

/// Heavy self-occlusion: a knot is thousands of small faces that hide each
/// other, which is the case a painter's algorithm is supposed to be good at and
/// the one where a wrong depth key shows up as bites taken out of the surface.
#[test]
fn self_occluding_knot() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(mesh_at(
        TorusKnotGeometry::new(1.2, 0.42, 90, 14, 2, 3),
        0xd4553c,
        Vector3::ZERO,
    ));

    let camera = look(Vector3::new(4.0, 3.0, 6.0), Vector3::ZERO, 4.0 / 3.0);
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("knot");
}

/// Objects at a range of depths, all overlapping on screen.
#[test]
fn overlapping_depths_sort_correctly() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    for (i, hex) in [0xcc3311u32, 0x22aa55, 0x3366cc, 0xddaa22]
        .iter()
        .enumerate()
    {
        scene.add(mesh_at(
            SphereGeometry::new(1.1, 24, 16),
            *hex,
            Vector3::new(i as f32 * 0.7 - 1.0, 0.0, i as f32 * -1.6),
        ));
    }

    let camera = look(Vector3::new(0.0, 1.0, 7.0), Vector3::ZERO, 4.0 / 3.0);
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("overlapping depths");
}

/// Geometry crossing the camera plane, which is where an unclipped renderer
/// smears vertices across the viewport.
#[test]
fn camera_inside_the_scene() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(ground(60.0, 0x336699, -1.0));
    scene.add(mesh_at(
        BoxGeometry::new(3.0, 3.0, 3.0),
        0xcc3311,
        Vector3::new(0.0, 0.5, -6.0),
    ));

    // Low and close, so the floor runs from under the camera to the horizon.
    let camera = look(
        Vector3::new(0.0, 0.4, 2.0),
        Vector3::new(0.0, 0.3, -6.0),
        4.0 / 3.0,
    );
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("camera inside");
}

/// Objects running off every edge of the frame. Nothing inside the viewport may
/// go missing because part of its geometry is outside.
#[test]
fn geometry_crossing_the_frame_edges() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(mesh_at(
        BoxGeometry::new(14.0, 0.6, 0.6),
        0xcc3311,
        Vector3::new(0.0, 1.2, 0.0),
    ));
    scene.add(mesh_at(
        BoxGeometry::new(0.6, 14.0, 0.6),
        0x22aa55,
        Vector3::new(0.0, 0.0, 0.6),
    ));
    scene.add(mesh_at(
        PlaneGeometry::new(30.0, 30.0),
        0x3366cc,
        Vector3::new(0.0, 0.0, -3.0),
    ));

    let camera = look(Vector3::new(0.0, 0.0, 6.0), Vector3::ZERO, 4.0 / 3.0);
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("frame edges");
}

/// Orthographic projection: `w` is 1 everywhere, so every depth and clipping
/// path that divides by it has to behave differently and still be right.
#[test]
fn orthographic_camera() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(ground(40.0, 0x336699, 0.0));
    scene.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(-1.6, 1.0, 0.0),
    ));
    scene.add(mesh_at(
        SphereGeometry::new(1.0, 24, 16),
        0x22aa55,
        Vector3::new(1.6, 1.0, 0.0),
    ));

    let mut camera = OrthographicCamera::new(-5.0, 5.0, 3.75, -3.75, 0.1, 100.0);
    camera.position = Vector3::new(4.0, 3.0, 6.0);
    camera.target = Vector3::new(0.0, 1.0, 0.0);
    compare(&mut scene, &camera, false, 320, 240, opts()).assert_clean("orthographic");
}

/// Nested transforms and per-instance transforms have to reach the projection
/// intact — a scale or rotation lost on the way shows up as geometry in the
/// wrong place, which the ray cast will not agree with.
#[test]
fn nested_transforms_and_instances() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);

    let mut group = Object3D::group();
    group.position = Vector3::new(0.0, 0.5, 0.0);
    group.quaternion = Quaternion::from_axis_angle(Vector3::UP, 0.6);
    group.scale = Vector3::new(1.4, 0.8, 1.4);
    let group_id = scene.add(group);

    for (i, hex) in [0xcc3311u32, 0x22aa55, 0x3366cc].iter().enumerate() {
        let mut child = mesh_at(
            BoxGeometry::new(1.2, 1.2, 1.2),
            *hex,
            Vector3::new(i as f32 * 1.8 - 1.8, 0.0, 0.0),
        );
        child.quaternion = Quaternion::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), 0.4);
        let id = scene.arena.insert(child);
        scene.arena.add_child(group_id, id);
    }

    let camera = look(
        Vector3::new(1.0, 2.5, 6.5),
        Vector3::new(0.0, 0.4, 0.0),
        4.0 / 3.0,
    );
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("nested transforms");
}

/// A large polygon that occludes something while reaching past it.
///
/// The ball is entirely behind the wall, so not one pixel of it should show.
/// The depth key sorts a face by how far back it reaches, and the wall recedes
/// far past the ball, so unsplit it sorts *behind* the thing it is hiding and
/// the ball shows straight through. Splitting is the only thing that resolves
/// it, and this is the scene that justifies the option existing.
#[test]
fn a_receding_wall_hides_what_is_behind_it() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(20.0, 8.0), flat(0x996633)));
    wall.quaternion = Quaternion::from_axis_angle(Vector3::UP, 60f32.to_radians());
    scene.add(wall);
    scene.add(mesh_at(
        SphereGeometry::new(1.0, 24, 16),
        0x22aa55,
        Vector3::new(-3.0, 0.0, 2.0),
    ));

    let camera = look(Vector3::new(0.0, 2.0, 14.0), Vector3::ZERO, 4.0 / 3.0);
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("receding wall");

    // And it is genuinely splitting that does it, not the depth key.
    let unsplit = compare(
        &mut scene,
        &camera,
        true,
        320,
        240,
        SvgOptions {
            depth_split: None,
            ..opts()
        },
    );
    assert!(
        unsplit.mismatches.len() > 20,
        "this scene is supposed to need splitting; without it only {} pixels \
         were wrong, so it no longer demonstrates anything",
        unsplit.mismatches.len()
    );
}

/// Two solids pushed through each other. No per-face sort can resolve the
/// intersection, and splitting is what keeps the error down to the seam rather
/// than half the picture. Recorded because it is the renderer's known limit,
/// and a regression here would otherwise look like a mystery.
#[test]
fn interpenetrating_solids_resolve_at_the_default() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(mesh_at(
        BoxGeometry::new(2.0, 4.0, 2.0),
        0xcc3311,
        Vector3::ZERO,
    ));
    scene.add(mesh_at(
        BoxGeometry::new(8.0, 0.8, 0.8),
        0x22aa55,
        Vector3::new(0.0, 0.5, 0.0),
    ));

    let camera = look(Vector3::new(4.0, 2.0, 7.0), Vector3::ZERO, 4.0 / 3.0);
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("interpenetrating");
}

/// The far end of the range: a flat plane of two triangles reaching most of the
/// way to the far clip, with the camera close enough that the near clip cuts it.
/// Both the depth key and the clipper have to hold up at once.
#[test]
fn floor_running_off_behind_the_camera() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(ground(60.0, 0x6699cc, 0.0));
    scene.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(0.0, 1.0, 0.0),
    ));

    let camera = look(
        Vector3::new(0.0, 3.0, 8.0),
        Vector3::new(0.0, 1.0, 0.0),
        4.0 / 3.0,
    );
    compare(&mut scene, &camera, true, 320, 240, opts()).assert_clean("floor behind camera");
}

/// A tessellated mesh must not be split at all: splitting it would multiply the
/// document for nothing, since its faces are already too shallow to span
/// anything. This is the property that makes the default affordable.
#[test]
fn tessellated_geometry_is_not_split() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(mesh_at(
        TorusKnotGeometry::new(1.2, 0.42, 90, 14, 2, 3),
        0xd4553c,
        Vector3::ZERO,
    ));
    let camera = look(Vector3::new(4.0, 3.0, 6.0), Vector3::ZERO, 4.0 / 3.0);

    let mut count = |o: SvgOptions| {
        SvgRenderer::new(320, 240)
            .with_options(o)
            .render_to_string(&mut scene, &camera)
            .matches("<path")
            .count()
    };
    let split = count(opts());
    let unsplit = count(SvgOptions {
        depth_split: None,
        ..opts()
    });
    assert_eq!(
        split, unsplit,
        "the default tolerance split a knot's faces; it should leave them alone"
    );
}

/// Does curving actually recover the silhouette?
///
/// Ground truth here is the *analytic* sphere, not a ray cast. A ray cast hits
/// the same tessellated polyhedron the renderer draws, so a straight-edged
/// render agrees with it perfectly and the test would measure nothing. The
/// whole point of curving is to depart from the tessellation and back towards
/// the surface it stands for, so the surface is what it has to be scored
/// against.
///
/// For a sphere of radius `r` seen from distance `d`, the silhouette is a
/// circle of angular radius `asin(r / d)` — a known number of pixels, with no
/// fixture to drift.
#[test]
fn curving_recovers_a_polygonal_silhouette() {
    const R: f32 = 1.0;
    const DIST: f32 = 4.0;
    const FOV: f32 = 45.0;
    const SIZE: u32 = 400;

    // Where the tangent point lands, through the same projection the renderer
    // uses: half the viewport times tan(angular radius) over tan(half fov).
    let half = SIZE as f32 * 0.5;
    let radius_px = half * (R / DIST).asin().tan() / (FOV.to_radians() * 0.5).tan();

    // Samples inside the true silhouette that the document leaves unpainted.
    // Only undershoot is counted; a polygonal outline is inscribed in the
    // circle, so that is the whole of the error.
    let shortfall = |segments: usize, options: SvgOptions| -> usize {
        let mut scene = Scene::new();
        scene.background = Color::from_hex(BG);
        scene.add(mesh_at(
            SphereGeometry::new(R, segments, segments * 2 / 3),
            0x22aa55,
            Vector3::ZERO,
        ));
        let mut camera = PerspectiveCamera::new(FOV, 1.0, 0.1, 100.0);
        camera.position = Vector3::new(0.0, 0.0, DIST);
        camera.target = Vector3::ZERO;

        let svg = SvgRenderer::new(SIZE, SIZE)
            .with_options(options)
            .render_to_string(&mut scene, &camera);
        let paths = parse_paths(&svg);

        let mut missed = 0;
        for py in (0..SIZE).step_by(2) {
            for px in (0..SIZE).step_by(2) {
                let (x, y) = (px as f32 + 0.5, py as f32 + 0.5);
                let d = ((x - half).powi(2) + (y - half).powi(2)).sqrt();
                // Anti-aliasing makes the boundary pixel genuinely ambiguous,
                // so stay clear of the true edge.
                if d > radius_px - 1.5 {
                    continue;
                }
                if painted_at(&paths, x, y).is_none() {
                    missed += 1;
                }
            }
        }
        missed
    };

    let straight = SvgOptions {
        curve_tolerance: None,
        ..opts()
    };

    // The strong claim: from a middling tessellation upwards the outline is the
    // real silhouette, not an approximation of it. Straight edges are not.
    assert!(
        shortfall(16, straight) > 30,
        "a 16-segment sphere with straight edges should visibly under-fill its \
         own silhouette, or this test is measuring nothing"
    );
    assert_eq!(
        shortfall(16, opts()),
        0,
        "curving should put a 16-segment sphere exactly on its silhouette"
    );
    assert_eq!(shortfall(32, opts()), 0);

    // Below that it closes most of the gap but not all of it, and that is a
    // property of where the outline is allowed to be rather than a tolerance to
    // tighten: the corners are mesh vertices, and on a mesh this coarse those
    // sit inside the true silhouette to begin with. Curving bends the edges
    // between them onto the surface; it cannot move the corners.
    let coarse_straight = shortfall(12, straight);
    let coarse_curved = shortfall(12, opts());
    assert!(
        coarse_curved * 2 <= coarse_straight,
        "curving should at least halve the gap even on a coarse sphere: \
         {coarse_straight} straight, {coarse_curved} curved"
    );
}

/// The document has to stay inside its own frame.
///
/// A viewer clips to the `viewBox`, so unclipped geometry looks fine in a
/// browser and is still wrong: a ground plane running to the horizon projects
/// thousands of units past the edge, and any tool that fits the *content*
/// bounding box instead — an editor, a thumbnailer — then shows the artwork as
/// a small offset speck inside a mostly empty canvas. Which reads, reasonably
/// enough, as the picture being cut off.
#[test]
fn coordinates_stay_within_the_canvas() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    // 80 units of floor seen almost edge-on: the far corners project a very
    // long way outside a 320-pixel frame.
    scene.add(ground(80.0, 0x336699, 0.0));
    scene.add(mesh_at(
        BoxGeometry::new(2.0, 2.0, 2.0),
        0xcc3311,
        Vector3::new(0.0, 1.0, 0.0),
    ));
    let camera = look(
        Vector3::new(0.0, 1.2, 9.0),
        Vector3::new(0.0, 1.0, 0.0),
        4.0 / 3.0,
    );

    let (w, h) = (320.0f32, 240.0f32);
    let svg = SvgRenderer::new(w as u32, h as u32)
        .with_options(opts())
        .render_to_string(&mut scene, &camera);

    // A margin is expected and deliberate: cut edges carry the seam stroke, so
    // they are placed just off-frame rather than exactly on it.
    let limit = 0.1;
    let (mut min_x, mut max_x) = (f32::MAX, f32::MIN);
    let (mut min_y, mut max_y) = (f32::MAX, f32::MIN);
    for p in parse_paths(&svg) {
        for v in p.poly {
            min_x = min_x.min(v[0]);
            max_x = max_x.max(v[0]);
            min_y = min_y.min(v[1]);
            max_y = max_y.max(v[1]);
        }
    }
    assert!(
        min_x > -limit * w && max_x < w * (1.0 + limit),
        "x ran to [{min_x:.0}, {max_x:.0}] on a {w:.0}-wide canvas"
    );
    assert!(
        min_y > -limit * h && max_y < h * (1.0 + limit),
        "y ran to [{min_y:.0}, {max_y:.0}] on a {h:.0}-tall canvas"
    );
}

/// Wireframe draws no fills, so there is nothing to sort and nothing to split.
/// If splitting ran anyway, its cuts would be stroked and the tessellation
/// would be drawn onto the model as if it were part of the mesh.
#[test]
fn wireframe_draws_the_mesh_and_not_the_split() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BG);
    scene.add(ground(60.0, 0x6699cc, 0.0));
    let camera = look(
        Vector3::new(0.0, 3.0, 8.0),
        Vector3::new(0.0, 1.0, 0.0),
        4.0 / 3.0,
    );

    let svg = SvgRenderer::new(320, 240)
        .with_options(SvgOptions {
            shading: SvgShading::Wireframe,
            ..opts()
        })
        .render_to_string(&mut scene, &camera);
    // Two triangles in, at most two out — the near clip may drop one.
    let drawn = svg.matches("<path").count();
    assert!(
        drawn <= 2,
        "a two-triangle floor drew {drawn} wireframe paths"
    );
}
