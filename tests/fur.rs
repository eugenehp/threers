//! Shell fur (`threers::fur`): the shader compiles for every backend it runs
//! on, and what it draws is measured against the geometry it was grown from.
//!
//! The rendering tests skip without a GPU adapter.

use threers::fur::{fur_object, FurGroom, FurMaterial};
use threers::prelude::*;

const N: u32 = 160;

/// Native wgpu, and WebGL2 (no storage buffers) — then through naga's GLSL
/// ES 3.0 writer, as wgpu's WebGL2 backend translates it, looking for the
/// reserved `gl_` names ANGLE rejects and naga's own validation accepts.
#[test]
fn the_fur_shader_compiles_for_wgpu_and_webgl2() {
    let fur = FurMaterial::default();
    for m in [fur.shader_material(), fur.hair_shader_material(0.01)] {
        compiles_for_wgpu_and_webgl2(m);
    }
}

fn compiles_for_wgpu_and_webgl2(m: ShaderMaterial) {
    m.validate(true).expect("native wgpu");
    m.validate(false).expect("WebGL2 (no storage buffers)");

    let path = std::env::temp_dir().join(format!("threers-fur-{}.wgsl", std::process::id()));
    std::env::set_var("THREERS_DUMP_SHADER", &path);
    m.validate(false).unwrap();
    std::env::remove_var("THREERS_DUMP_SHADER");
    let source = std::fs::read_to_string(&path).expect("dumped module");
    let _ = std::fs::remove_file(&path);
    let module = naga::front::wgsl::parse_str(&source).expect("parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect("validates");
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "vs_custom"),
        (naga::ShaderStage::Fragment, "fs_main"),
    ] {
        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &module,
            &info,
            Some((stage, entry)),
            &Default::default(),
        )
        .unwrap();
        let mut out = String::new();
        naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &naga::back::glsl::Options {
                version: naga::back::glsl::Version::Embedded {
                    version: 300,
                    is_webgl: true,
                },
                ..Default::default()
            },
            &naga::back::glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.into(),
                multiview: None,
            },
            naga::proc::BoundsCheckPolicies::default(),
        )
        .unwrap_or_else(|e| panic!("{entry}: {e}"))
        .write()
        .unwrap_or_else(|e| panic!("{entry}: {e}"));
        for (i, _) in out.match_indices("gl_") {
            let prev = out
                .as_bytes()
                .get(i.wrapping_sub(1))
                .copied()
                .unwrap_or(b' ');
            let next = out.as_bytes()[i + 3];
            assert!(
                prev.is_ascii_alphanumeric() || prev == b'_' || next.is_ascii_uppercase(),
                "{entry}: a `gl_` name ANGLE rejects at byte {i}"
            );
        }
    }
}

fn scene_with(object: Object3D) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 1.0));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.3, -0.5, -1.0).normalize();
    scene.add(Object3D::light(threers::lights::Light::Directional(sun)));
    scene.add(object);
    scene
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    c.position = Vector3::new(0.0, 0.0, 5.0);
    c.look_at(Vector3::ZERO);
    c
}

fn render(hr: &mut HeadlessRenderer, object: Object3D) -> Vec<u8> {
    hr.render_to_rgba(&mut scene_with(object), &camera())
}

/// Whether a ray from the camera through pixel (x, y) passes within `r` of
/// the origin — the silhouette of a sphere of radius `r`, from geometry
/// rather than from another renderer.
fn ray_meets_sphere(x: u32, y: u32, r: f32) -> bool {
    let cam = camera();
    let ndc = Vector3::new(
        (x as f32 + 0.5) / N as f32 * 2.0 - 1.0,
        1.0 - (y as f32 + 0.5) / N as f32 * 2.0,
        0.5,
    );
    let dir =
        (ndc.unproject(&cam.view_matrix(), &cam.projection_matrix()) - cam.position).normalize();
    let to_centre = Vector3::ZERO - cam.position;
    let along = to_centre.dot(dir);
    (to_centre - dir * along).length() < r
}

fn lit(p: &[u8]) -> bool {
    p[0] as u32 + p[1] as u32 + p[2] as u32 > 30
}

#[test]
fn fur_grows_past_the_surface_in_strands() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.35,
            density: 60.0,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        root_color: Color::new(0.2, 0.1, 0.05),
        tip_color: Color::new(0.9, 0.75, 0.5),
        ..Default::default()
    };
    let px = render(&mut hr, fur_object(SphereGeometry::new(1.0, 64, 32), &fur));
    let at = |x: u32, y: u32| &px[((y * N + x) * 4) as usize..][..4];

    // Inside the skin's silhouette: covered. In a band past it that the fur
    // reaches: partly covered — strands with gaps between them, not a solid
    // shell and not nothing. Well beyond the tips: empty.
    let (mut skin, mut skin_lit, mut band, mut band_lit, mut beyond_lit) = (0, 0, 0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            let lit = lit(at(x, y));
            if ray_meets_sphere(x, y, 0.95) {
                skin += 1;
                skin_lit += lit as u32;
            } else if ray_meets_sphere(x, y, 1.2) && !ray_meets_sphere(x, y, 1.05) {
                band += 1;
                band_lit += lit as u32;
            } else if !ray_meets_sphere(x, y, 1.45) {
                beyond_lit += lit as u32;
            }
        }
    }
    let band_cover = band_lit as f32 / band as f32;
    eprintln!(
        "skin {skin_lit}/{skin}, band {band_lit}/{band} ({band_cover:.2}), beyond {beyond_lit}"
    );
    assert!(
        skin_lit as f32 > 0.97 * skin as f32,
        "the skin is not covered"
    );
    assert!(
        // Not nothing, and not every pixel. Seen edge-on near the roots a
        // dense coat does cover nearly all (each strand drawn swept across
        // its shell's slab, as the line of sight crosses it).
        band_cover > 0.08 && band_cover < 0.995,
        "fur past the surface covers {band_cover:.2} of the band"
    );
    assert_eq!(beyond_lit, 0, "something drawn beyond the tips");
}

#[test]
fn tips_are_lighter_than_roots_and_alpha_zero_is_bald() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.35,
            density: 60.0,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        root_color: Color::new(0.05, 0.05, 0.05),
        tip_color: Color::new(1.0, 1.0, 1.0),
        occlusion: 0.0,
        specular: 0.0,
        ..Default::default()
    };
    let px = render(&mut hr, fur_object(SphereGeometry::new(1.0, 64, 32), &fur));
    let mean = |r_in: f32, r_out: f32| {
        let (mut sum, mut n) = (0u64, 0u64);
        for y in 0..N {
            for x in 0..N {
                let p = &px[((y * N + x) * 4) as usize..][..4];
                if lit(p) && ray_meets_sphere(x, y, r_out) && !ray_meets_sphere(x, y, r_in) {
                    sum += p[0] as u64;
                    n += 1;
                }
            }
        }
        sum as f64 / n.max(1) as f64
    };
    // Seen edge-on at the silhouette, the outer band is tips, the inner roots.
    let (roots, tips) = (mean(0.9, 1.0), mean(1.15, 1.3));
    eprintln!("root band {roots:.1}, tip band {tips:.1}");
    assert!(
        tips > roots + 30.0,
        "tips {tips:.1} not lighter than roots {roots:.1}"
    );

    // Vertex alpha 0 everywhere: no fur, just the skin.
    let mut bald = SphereGeometry::new(1.0, 64, 32);
    let count = bald.get_attribute("position").unwrap().count();
    bald.set_attribute(
        "color",
        threers::BufferAttribute::new(vec![1.0; count * 4], 4),
    );
    if let Some(c) = bald.attributes.get_mut("color") {
        for k in 0..count {
            c.array[k * 4 + 3] = 0.0;
        }
    }
    let px = render(&mut hr, fur_object(bald, &fur));
    let past = (0..N * N)
        .filter(|&i| {
            let (x, y) = (i % N, i / N);
            lit(&px[(i * 4) as usize..][..4]) && !ray_meets_sphere(x, y, 1.03)
        })
        .count();
    assert_eq!(
        past, 0,
        "alpha-zero fur still grew {past} pixels past the skin"
    );
}

/// Columns of the picture with anything drawn in them: (leftmost, rightmost).
fn extent(px: &[u8]) -> (u32, u32) {
    let (mut lo, mut hi) = (N, 0);
    for y in 0..N {
        for x in 0..N {
            if lit(&px[((y * N + x) * 4) as usize..][..4]) {
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
    }
    (lo, hi)
}

/// No UVs at all: strands scattered over the surface in object space still
/// grow out past the skin in strands, and stop at their tips.
#[test]
fn surface_placement_needs_no_uvs() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut g = SphereGeometry::new(1.0, 64, 32);
    g.attributes.remove("uv");
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.35,
            placement: threers::fur::FurPlacement::Surface { cell: 0.03 },
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        tip_color: Color::new(0.9, 0.8, 0.6),
        ..Default::default()
    };
    fur.shader_material().validate(false).expect("WebGL2");
    let px = render(&mut hr, fur_object(g, &fur));
    let (mut band, mut band_lit, mut beyond_lit) = (0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            let l = lit(&px[((y * N + x) * 4) as usize..][..4]);
            if ray_meets_sphere(x, y, 1.2) && !ray_meets_sphere(x, y, 1.05) {
                band += 1;
                band_lit += l as u32;
            } else if !ray_meets_sphere(x, y, 1.45) {
                beyond_lit += l as u32;
            }
        }
    }
    let cover = band_lit as f32 / band as f32;
    eprintln!("surface placement: band {cover:.2}, beyond {beyond_lit}");
    assert!(cover > 0.08 && cover < 0.995, "band cover {cover:.2}");
    assert_eq!(beyond_lit, 0);
}

/// Inertia: a ball that accelerates to the right leaves its fur trailing to
/// the left — further out on the left than the right — and once it stops the
/// fur swings back to even.
/// The coat's pull, as the animator drives it (`data[9]`, flagged by `w`).
fn coat_pull(scene: &Scene, id: ObjectId) -> Vector3 {
    let m = match &scene.get(id).unwrap().kind {
        threers::core::ObjectKind::InstancedMesh(im) => {
            im.materials.first().unwrap_or(&im.material).clone()
        }
        _ => panic!("not fur"),
    };
    let Material::Shader(m) = &*m else {
        panic!("not a shader")
    };
    assert_eq!(m.data[9][3], 1.0, "the animator is not driving the pull");
    Vector3::new(m.data[9][0], m.data[9][1], m.data[9][2])
}

/// Inertia and drag: accelerating right, the coat's pull swings left (the
/// strands trail) by about what the acceleration and the air going by ask
/// for; stopped, it rings down to rest.
#[test]
fn fur_trails_an_accelerating_object_and_settles() {
    use threers::fur::{FurAnimator, FurDynamics};
    let dynamics = FurDynamics {
        stiffness: 40.0,
        damping: 0.7,
        response: 1.0,
        max_lag: 2.0,
    };
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.4,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        dynamics,
        ..Default::default()
    };
    let mut scene = scene_with(Object3D::group());
    let ball = scene.add(fur_object(SphereGeometry::new(0.8, 16, 8), &fur));
    let mut animator = FurAnimator::new();
    animator.track(ball, &fur);
    let dt = 1.0 / 60.0;
    let (mut x, mut v) = (0.0f32, 0.0f32);
    // Accelerate right at 8 m/s² for a second: the pull heads for the
    // inertial 8 / (0.4 × 40) = 0.5 plus the drag of the air at the speed
    // reached, 0.15 × 8² = 9.6 — capped at `max_lag` 2 off rest.
    let mut first = None;
    for f in 0..60 {
        v += 8.0 * dt;
        x += v * dt;
        scene.get_mut(ball).unwrap().position.x = x;
        animator.update(&mut scene, f as f32 * dt, dt);
        if f == 20 {
            first = Some(coat_pull(&scene, ball));
        }
    }
    let early = first.unwrap();
    let late = coat_pull(&scene, ball);
    eprintln!("pull after 1/3 s {early:?}, after 1 s {late:?}");
    assert!(
        early.x < -0.3 && early.y.abs() < 1e-4,
        "no trail after 1/3 s: {early:?}"
    );
    assert!(
        late.x < early.x,
        "the trail did not grow with the speed: {late:?}"
    );
    // Stop dead: it rings back through rest and settles.
    let mut crossed = false;
    for f in 0..240 {
        animator.update(&mut scene, (60 + f) as f32 * dt, dt);
        crossed |= coat_pull(&scene, ball).x > 0.0;
    }
    let rest = coat_pull(&scene, ball);
    eprintln!("swung past rest {crossed}; settled {rest:?}");
    assert!(crossed, "the coat did not swing forward when it stopped");
    assert!(rest.length() < 1e-3, "not settled: {rest:?}");
}

/// A sideways pull leans the fur that way on screen.
#[test]
fn a_sideways_pull_leans_the_fur() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut top = |gx: f32| {
        let fur = FurMaterial {
            groom: FurGroom {
                length: 0.4,
                density: 50.0,
                gravity: Vector3::new(gx, 0.0, 0.0),
                ..Default::default()
            },
            tip_color: Color::new(0.9, 0.8, 0.6),
            ..Default::default()
        };
        let px = render(&mut hr, fur_object(SphereGeometry::new(0.8, 64, 32), &fur));
        let (mut sx, mut n) = (0.0, 0);
        for y in 0..N / 4 {
            for x in 0..N {
                if lit(&px[((y * N + x) * 4) as usize..][..4]) {
                    sx += x as f32;
                    n += 1;
                }
            }
        }
        sx / n.max(1) as f32
    };
    let (left, none, right) = (top(-1.5), top(0.0), top(1.5));
    eprintln!("top fur mean x: pulled left {left:.1}, none {none:.1}, right {right:.1}");
    // (The sun is off to one side, so the lit pixels counted lean less one
    // way than the other.)
    assert!(
        right > none + 2.0 && left < none - 1.0,
        "no lean: {left:.1} / {none:.1} / {right:.1}"
    );
}

/// Fur grown on a skinned mesh goes where the bone takes the mesh.
#[test]
fn fur_follows_a_skinned_mesh() {
    use threers::core::{Bone, BufferAttribute, Skeleton, SkinnedMesh};
    use threers::fur::FurAnimator;
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut scene = scene_with(Object3D::group());
    let bone = scene.add(Object3D::group());
    let mut g = SphereGeometry::new(0.6, 48, 24);
    let count = g.get_attribute("position").unwrap().count();
    g.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
    g.set_attribute(
        "weight",
        BufferAttribute::new((0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
    );
    let skeleton = Skeleton::new(vec![Bone {
        node: bone,
        inverse_bind: Matrix4::identity(),
    }]);
    let body = scene.add(Object3D::skinned_mesh(SkinnedMesh::new(
        g,
        Material::Basic(BasicMaterial::new(Color::new(0.05, 0.05, 0.05))),
        skeleton,
    )));
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.3,
            density: 40.0,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        tip_color: Color::new(0.9, 0.8, 0.6),
        ..Default::default()
    };
    let mut animator = FurAnimator::new();
    animator
        .grow_on_skinned(&mut scene, body, &fur)
        .expect("a skinned mesh");
    let mut centres = Vec::new();
    for bx in [-1.0f32, 1.0] {
        scene.get_mut(bone).unwrap().position.x = bx;
        for f in 0..90 {
            animator.update(&mut scene, f as f32 / 60.0, 1.0 / 60.0);
        }
        let px = hr.render_to_rgba(&mut scene, &camera());
        let (lo, hi) = extent(&px);
        centres.push((lo as f32 + hi as f32) / 2.0);
    }
    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let moved = (centres[1] - centres[0]) / per_unit;
    eprintln!("fur centres {centres:?}: moved {moved:.2} units for a 2-unit bone move");
    assert!(
        (moved - 2.0).abs() < 0.2,
        "the fur moved {moved:.2}, the bone 2"
    );
}

/// Wind blows the fur about over time: the same scene at two moments of
/// the gust cycle draws differently, and without wind it does not.
#[test]
fn wind_moves_the_fur_over_time() {
    use threers::fur::set_fur_time;
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let frames = |wind: Vector3, hr: &mut HeadlessRenderer| {
        let fur = FurMaterial {
            groom: FurGroom {
                length: 0.35,
                density: 50.0,
                gravity: Vector3::ZERO,
                ..Default::default()
            },
            wind,
            flutter: 0.15,
            ..Default::default()
        };
        let mut obj = fur_object(SphereGeometry::new(1.0, 48, 24), &fur);
        set_fur_time(&mut obj, 0.0);
        let a = render(hr, obj.clone());
        set_fur_time(&mut obj, 0.7);
        let b = render(hr, obj);
        a.chunks(4)
            .zip(b.chunks(4))
            .filter(|(p, q)| lit(p) != lit(q))
            .count()
    };
    let windy = frames(Vector3::new(0.6, 0.0, 0.0), &mut hr);
    eprintln!("pixels that changed with wind: {windy}");
    assert!(windy > 200, "wind moved only {windy} pixels");
}

// ---------------------------------------------------------------------------
// Strand hair
// ---------------------------------------------------------------------------

use threers::fur::strands::{grow_strands, hair_object, StrandOptions};

fn straight(length: f32) -> FurGroom {
    FurGroom {
        length,
        curl: 0.0,
        clumping: 0.0,
        length_variation: 0.0,
        gravity: Vector3::ZERO,
        ..Default::default()
    }
}

#[test]
fn strands_root_on_the_surface_and_grow_out_along_the_normal() {
    let sphere = SphereGeometry::new(1.0, 48, 24);
    let hair = grow_strands(&sphere, &straight(0.4), 500, 6);
    assert_eq!(hair.count(), 500);
    assert_eq!(hair.points.len(), 500 * 7);
    for i in 0..hair.count() {
        let s = hair.strand(i);
        // A tessellated sphere: roots sit on its flat faces, just inside 1.
        let r = s[0].length();
        assert!(r > 0.995 && r <= 1.0001, "root {i} at radius {r}");
        let out = s[6] - s[0];
        assert!(
            (out.length() - 0.4).abs() < 1e-4,
            "strand {i} is {} long",
            out.length()
        );
        assert!(
            out.normalize().dot(s[0].normalize()) > 0.99,
            "strand {i} not along the normal"
        );
    }
    // Spread by area: as many roots in the upper half as the lower.
    let upper = (0..hair.count())
        .filter(|&i| hair.strand(i)[0].y > 0.0)
        .count();
    assert!(
        (200..300).contains(&upper),
        "{upper} of 500 roots in the upper half"
    );
    // The same seed regrows the same hair; another does not.
    assert_eq!(grow_strands(&sphere, &straight(0.4), 500, 6), hair);
    let other = FurGroom {
        seed: 7,
        ..straight(0.4)
    };
    assert_ne!(grow_strands(&sphere, &other, 500, 6).points, hair.points);
}

#[test]
fn strands_take_the_groom_curl_clumps_variation_and_paint() {
    let sphere = SphereGeometry::new(1.0, 32, 16);
    let tip_spread = |g: &FurGroom| {
        let h = grow_strands(&sphere, g, 400, 8);
        let lens: Vec<f32> = (0..h.count())
            .map(|i| (h.strand(i)[8] - h.strand(i)[0]).length())
            .collect();
        (h, lens)
    };
    // Length variation shortens some strands, none past the groom length.
    let (_, lens) = tip_spread(&FurGroom {
        length_variation: 0.5,
        ..straight(0.4)
    });
    let min = lens.iter().copied().fold(f32::MAX, f32::min);
    let max = lens.iter().copied().fold(0.0, f32::max);
    assert!(min < 0.25 && max <= 0.4001, "lengths {min}..{max}");

    // Curl bends a strand away from the straight line through its ends.
    let (h, _) = tip_spread(&FurGroom {
        curl: 1.0,
        ..straight(0.4)
    });
    let s = h.strand(0);
    let mid = (s[4] - s[0]).cross((s[8] - s[0]).normalize()).length();
    assert!(mid > 0.01, "a curled strand is straight ({mid})");

    // Clumping draws neighbouring tips together: tips are closer to their
    // nearest neighbour's than roots are.
    let gaps = |h: &threers::fur::strands::HairStrands, k: usize| {
        let pts: Vec<Vector3> = (0..h.count()).map(|i| h.strand(i)[k]).collect();
        pts.iter()
            .map(|p| {
                pts.iter()
                    .filter(|q| *q != p)
                    .map(|q| (*q - *p).length())
                    .fold(f32::MAX, f32::min)
            })
            .sum::<f32>()
            / pts.len() as f32
    };
    let (loose, _) = tip_spread(&straight(0.4));
    let (clumped, _) = tip_spread(&FurGroom {
        clumping: 1.0,
        clump_size: 4.0,
        ..straight(0.4)
    });
    let (lt, ct) = (gaps(&loose, 8), gaps(&clumped, 8));
    eprintln!("mean tip gap: loose {lt:.4}, clumped {ct:.4}");
    assert!(
        ct < 0.6 * lt,
        "clumping did not gather the tips ({lt:.4} -> {ct:.4})"
    );

    // Vertex colour: alpha 0 grows nothing, the RGB tints.
    let mut painted = SphereGeometry::new(1.0, 32, 16);
    let n = painted.get_attribute("position").unwrap().count();
    let pos = painted.get_attribute("position").unwrap().array.clone();
    let colors: Vec<f32> = (0..n)
        .flat_map(|v| [1.0, 0.2, 0.2, if pos[v * 3 + 1] > 0.0 { 1.0 } else { 0.0 }])
        .collect();
    painted.set_attribute("color", BufferAttribute::new(colors, 4));
    let h = grow_strands(&painted, &straight(0.3), 400, 4);
    assert!(
        h.count() < 260,
        "{} strands grew where half is bald",
        h.count()
    );
    assert!(
        (0..h.count()).all(|i| h.strand(i)[0].y > -0.2),
        "a strand on the bald half (past the ring of faces that straddle the edge)"
    );
    assert!(h
        .tints
        .iter()
        .all(|t| (t[0] - 1.0).abs() < 1e-4 && (t[1] - 0.2).abs() < 1e-4));
}

#[test]
fn tubes_wrap_each_strand() {
    let hair = grow_strands(&SphereGeometry::new(1.0, 16, 8), &straight(0.5), 10, 4);
    let g = hair.tube_geometry(6, 0.01, 0.5);
    let pos = &g.get_attribute("position").unwrap().array;
    assert_eq!(pos.len() / 3, 10 * 5 * 6);
    assert_eq!(g.index.as_ref().unwrap().len(), 10 * 4 * 6 * 6);
    // Each ring sits round its strand point at the tapered radius.
    for i in 0..10 {
        for k in 0..5 {
            let c = hair.strand(i)[k];
            let want = 0.01 * (1.0 - 0.5 * k as f32 / 4.0);
            for j in 0..6 {
                let v = (i * 5 + k) * 6 + j;
                let p = Vector3::new(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
                assert!(((p - c).length() - want).abs() < 1e-5);
            }
        }
    }
}

fn hair(gravity: Vector3) -> FurMaterial {
    FurMaterial {
        groom: FurGroom {
            gravity,
            ..straight(0.35)
        },
        root_color: Color::new(0.2, 0.1, 0.05),
        tip_color: Color::new(0.9, 0.75, 0.5),
        ..Default::default()
    }
}

#[test]
fn hair_draws_past_the_surface_and_hangs_by_gravity() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let options = StrandOptions {
        count: 6000,
        segments: 8,
        width: 0.012,
        ..Default::default()
    };
    let sphere = SphereGeometry::new(1.0, 48, 24);
    let px = render(
        &mut hr,
        hair_object(&sphere, &hair(Vector3::ZERO), &options),
    );
    let at = |x: u32, y: u32| &px[((y * N + x) * 4) as usize..][..4];
    let (mut band, mut band_lit, mut beyond_lit) = (0, 0, 0);
    for y in 0..N {
        for x in 0..N {
            if ray_meets_sphere(x, y, 1.25) && !ray_meets_sphere(x, y, 1.05) {
                band += 1;
                band_lit += lit(at(x, y)) as u32;
            } else if !ray_meets_sphere(x, y, 1.45) {
                beyond_lit += lit(at(x, y)) as u32;
            }
        }
    }
    let cover = band_lit as f32 / band as f32;
    eprintln!("hair: band cover {cover:.2}, beyond {beyond_lit}");
    assert!(cover > 0.15 && cover < 0.95, "band cover {cover:.2}");
    assert_eq!(beyond_lit, 0, "hair drawn beyond its tips");

    // Gravity pulls it down: more below the sphere than above it.
    let px = render(
        &mut hr,
        hair_object(&sphere, &hair(Vector3::new(0.0, -2.0, 0.0)), &options),
    );
    let rows = |range: std::ops::Range<u32>| -> u32 {
        range
            .flat_map(|y| (0..N).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                !ray_meets_sphere(x, y, 1.0) && lit(&px[((y * N + x) * 4) as usize..][..4])
            })
            .count() as u32
    };
    let (top, bottom) = (rows(0..N / 2), rows(N / 2..N));
    eprintln!("hanging hair: {top} lit above, {bottom} below");
    assert!(
        bottom > top * 3 / 2,
        "gravity did not pull the hair down ({top} above, {bottom} below)"
    );
}

#[test]
fn hair_trails_with_the_animator_and_moves_in_wind() {
    use threers::fur::FurAnimator;
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let options = StrandOptions {
        count: 3000,
        segments: 6,
        width: 0.012,
        ..Default::default()
    };
    let mut mat = hair(Vector3::ZERO);
    mat.wind = Vector3::new(0.6, 0.0, 0.0);
    mat.flutter = 0.15;
    let mut scene = scene_with(Object3D::group());
    let id = scene.add(hair_object(
        &SphereGeometry::new(0.8, 32, 16),
        &mat,
        &options,
    ));
    let mut animator = FurAnimator::new();
    animator.track(id, &mat);
    animator.update(&mut scene, 0.0, 1.0 / 60.0);
    let a = hr.render_to_rgba(&mut scene, &camera());
    animator.update(&mut scene, 0.7, 1.0 / 60.0);
    let b = hr.render_to_rgba(&mut scene, &camera());
    let changed = a
        .chunks(4)
        .zip(b.chunks(4))
        .filter(|(p, q)| lit(p) != lit(q))
        .count();
    eprintln!("hair pixels changed by wind: {changed}");
    assert!(changed > 200, "wind moved only {changed} pixels");
}

/// A skinned ball of radius 0.6 on one bone, its hair painted onto the top
/// cap only (vertex alpha), grown by `grow_hair_on_skinned`.
fn hairy_skinned_ball(gravity: Vector3) -> (Scene, threers::fur::FurAnimator, ObjectId, ObjectId) {
    use threers::core::{Bone, Skeleton, SkinnedMesh};
    let mut scene = scene_with(Object3D::group());
    let bone = scene.add(Object3D::group());
    let mut g = SphereGeometry::new(0.6, 48, 24);
    let pos = g.get_attribute("position").unwrap().array.clone();
    let count = pos.len() / 3;
    g.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
    g.set_attribute(
        "weight",
        BufferAttribute::new((0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
    );
    let cap: Vec<f32> = (0..count)
        .flat_map(|v| [1.0, 1.0, 1.0, if pos[v * 3 + 1] > 0.3 { 1.0 } else { 0.0 }])
        .collect();
    g.set_attribute("color", BufferAttribute::new(cap, 4));
    let skeleton = Skeleton::new(vec![Bone {
        node: bone,
        inverse_bind: Matrix4::identity(),
    }]);
    let body = scene.add(Object3D::skinned_mesh(SkinnedMesh::new(
        g,
        Material::Basic(BasicMaterial::new(Color::BLACK)),
        skeleton,
    )));
    let mut hair = hair(gravity);
    hair.dynamics.stiffness = 30.0;
    let mut animator = threers::fur::FurAnimator::new();
    let options = StrandOptions {
        count: 3000,
        segments: 6,
        width: 0.015,
        ..Default::default()
    };
    animator
        .grow_hair_on_skinned(&mut scene, body, &hair, &options)
        .expect("a skinned mesh");
    (scene, animator, bone, body)
}

/// The lit pixels' mean position, in pixels.
fn centroid(px: &[u8]) -> (f32, f32, u32) {
    let (mut sx, mut sy, mut n) = (0.0, 0.0, 0);
    for y in 0..N {
        for x in 0..N {
            if lit(&px[((y * N + x) * 4) as usize..][..4]) {
                sx += x as f32;
                sy += y as f32;
                n += 1;
            }
        }
    }
    (sx / n.max(1) as f32, sy / n.max(1) as f32, n)
}

#[test]
fn hair_follows_a_skinned_mesh_as_it_moves_and_turns() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, mut animator, bone, _) = hairy_skinned_ball(Vector3::ZERO);
    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let settle = |scene: &mut Scene, animator: &mut threers::fur::FurAnimator| {
        for f in 0..90 {
            animator.update(scene, f as f32 / 60.0, 1.0 / 60.0);
        }
    };

    // Moved: the hair goes with the bone.
    let mut at = Vec::new();
    for bx in [-1.0f32, 1.0] {
        scene.get_mut(bone).unwrap().position.x = bx;
        settle(&mut scene, &mut animator);
        at.push(centroid(&hr.render_to_rgba(&mut scene, &camera())));
    }
    let moved = (at[1].0 - at[0].0) / per_unit;
    eprintln!("hair centroids {at:?}: moved {moved:.2} for a 2-unit bone move");
    assert!(at[0].2 > 300, "only {} hair pixels", at[0].2);
    assert!(
        (moved - 2.0).abs() < 0.1,
        "the hair moved {moved:.2}, the bone 2"
    );

    // Turned upside down: the cap's hair is now below the centre, pointing
    // down — carried by the surface, not left where it grew.
    let b = scene.get_mut(bone).unwrap();
    b.position.x = 0.0;
    let up = {
        settle(&mut scene, &mut animator);
        centroid(&hr.render_to_rgba(&mut scene, &camera()))
    };
    scene.get_mut(bone).unwrap().rotate_z(std::f32::consts::PI);
    settle(&mut scene, &mut animator);
    let down = centroid(&hr.render_to_rgba(&mut scene, &camera()));
    let centre = N as f32 / 2.0;
    eprintln!(
        "cap hair centroid y: upright {:.1}, turned over {:.1} (centre {centre})",
        up.1, down.1
    );
    assert!(
        up.1 < centre - 0.5 * per_unit,
        "upright, the hair is not on top ({:.1})",
        up.1
    );
    assert!(
        down.1 > centre + 0.5 * per_unit,
        "turned over, the hair did not go under ({:.1})",
        down.1
    );
}

#[test]
fn skinned_hair_trails_its_roots_and_settles() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (mut scene, mut animator, bone, _) = hairy_skinned_ball(Vector3::ZERO);
    let dt = 1.0 / 60.0;
    let (mut x, mut v, mut t) = (-0.5f32, 0.0f32, 0.0f32);
    let mut run =
        |frames: u32, accel: f32, scene: &mut Scene, animator: &mut threers::fur::FurAnimator| {
            for _ in 0..frames {
                v = if accel == 0.0 { 0.0 } else { v + accel * dt };
                x += v * dt;
                t += dt;
                scene.get_mut(bone).unwrap().position.x = x;
                animator.update(scene, t, dt);
            }
            x
        };
    run(60, 0.0, &mut scene, &mut animator);
    // Accelerating right: the strands swing back, left of where they hang
    // at rest at the same spot.
    run(12, 12.0, &mut scene, &mut animator);
    let moving = centroid(&hr.render_to_rgba(&mut scene, &camera()));
    run(180, 0.0, &mut scene, &mut animator);
    let settled = centroid(&hr.render_to_rgba(&mut scene, &camera()));
    let trail = moving.0 - settled.0;
    eprintln!(
        "skinned hair: accelerating centroid x {:.1}, settled {:.1}: trail {trail:.1} px",
        moving.0, settled.0
    );
    assert!(
        trail < -1.5,
        "the hair did not trail its roots ({trail:.1} px)"
    );

    // Settled is the same as never having moved: the springs come to rest.
    let (mut still, mut still_anim, still_bone, _) = hairy_skinned_ball(Vector3::ZERO);
    still.get_mut(still_bone).unwrap().position.x = x_of(&scene, bone);
    for f in 0..3 {
        still_anim.update(&mut still, f as f32 * dt, dt);
    }
    let rest = centroid(&hr.render_to_rgba(&mut still, &camera()));
    eprintln!("  settled {:.2}, never moved {:.2}", settled.0, rest.0);
    assert!((settled.0 - rest.0).abs() < 0.3, "the hair did not settle");
}

fn x_of(scene: &Scene, id: ObjectId) -> f32 {
    scene.get(id).unwrap().position.x
}

// ---------------------------------------------------------------------------
// Hair shadows
// ---------------------------------------------------------------------------

const SHADOW_N: u32 = 200;
const HAIR_AT: Vector3 = Vector3::new(0.0, 2.0, 0.0);

fn shadow_camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(40.0, 1.0, 0.1, 50.0);
    c.position = Vector3::new(0.0, 8.0, 6.0);
    c.look_at(Vector3::ZERO);
    c
}

/// A white floor at y = 0 under a sun straight overhead, and `hair` (an
/// object) at `HAIR_AT`.
fn shadow_scene(hair: Object3D) -> (Scene, ObjectId) {
    let mut s = Scene::new();
    s.background = Color::BLACK;
    s.add_light(AmbientLight::new(Color::WHITE, 0.3));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(0.0, -1.0, 0.0);
    sun.cast_shadow = true;
    sun.shadow.map_size = 1024;
    sun.shadow.camera_size = 4.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 30.0;
    let mut l = Object3D::light(threers::lights::Light::Directional(sun));
    l.position = Vector3::new(0.0, 10.0, 0.0);
    s.add(l);
    let mut m = StandardMaterial::new(Color::WHITE);
    m.metalness = 0.0;
    m.roughness = 1.0;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), m.into()));
    floor.rotate_x(-std::f32::consts::FRAC_PI_2);
    floor.receive_shadow = true;
    s.add(floor);
    let mut hair = hair;
    hair.position = HAIR_AT;
    let id = s.add(hair);
    (s, id)
}

/// Where each floor pixel's ray meets the floor — `None` for a pixel that
/// misses it, or whose ray passes the hair on the way.
fn floor_points() -> Vec<Option<(f32, f32)>> {
    let cam = shadow_camera();
    (0..SHADOW_N * SHADOW_N)
        .map(|i| {
            let (x, y) = (i % SHADOW_N, i / SHADOW_N);
            let ndc = Vector3::new(
                (x as f32 + 0.5) / SHADOW_N as f32 * 2.0 - 1.0,
                1.0 - (y as f32 + 0.5) / SHADOW_N as f32 * 2.0,
                0.5,
            );
            let dir = (ndc.unproject(&cam.view_matrix(), &cam.projection_matrix()) - cam.position)
                .normalize();
            let to_hair = HAIR_AT - cam.position;
            if (to_hair - dir * to_hair.dot(dir)).length() < 1.2 {
                return None;
            }
            let t = -cam.position.y / dir.y;
            (t > 0.0).then(|| {
                let p = cam.position + dir * t;
                (p.x, p.z)
            })
        })
        .collect()
}

/// (darkness under the hair: 1 − its light over the open floor's, the
/// shade's centroid in x)
fn shade(px: &[u8]) -> (f32, f32) {
    let floor = floor_points();
    let (mut far, mut nf) = (0.0f64, 0u32);
    for (i, p) in floor.iter().enumerate() {
        if let Some((x, z)) = p {
            if (2.5..3.5).contains(&(x * x + z * z).sqrt()) {
                far += px[i * 4] as f64;
                nf += 1;
            }
        }
    }
    let far = far / nf.max(1) as f64;
    let (mut under, mut nu, mut wsum, mut wx) = (0.0f64, 0u32, 0.0f64, 0.0f64);
    for (i, p) in floor.iter().enumerate() {
        let Some((x, z)) = p else { continue };
        let r = (x * x + z * z).sqrt();
        let v = px[i * 4] as f64;
        if r < 0.4 {
            under += v;
            nu += 1;
        }
        if r < 1.8 {
            let w = (1.0 - v / far).max(0.0);
            wsum += w;
            wx += w * *x as f64;
        }
    }
    (
        (1.0 - under / nu.max(1) as f64 / far) as f32,
        (wx / wsum.max(1e-9)) as f32,
    )
}

fn shadow_hair(pull: Vector3, shadow_strands: f32) -> Object3D {
    let mut m = hair(pull);
    m.groom.length = 0.5;
    let options = StrandOptions {
        count: 4000,
        segments: 6,
        width: 0.012,
        shadow_strands,
    };
    hair_object(&SphereGeometry::new(0.4, 32, 16), &m, &options)
}

#[test]
fn hair_casts_a_shadow_that_bends_with_it() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(SHADOW_N, SHADOW_N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut render = |hair: Object3D| {
        let (mut s, _) = shadow_scene(hair);
        shade(&hr.render_to_rgba(&mut s, &shadow_camera()))
    };
    let (dark, cx) = render(shadow_hair(Vector3::ZERO, 0.25));
    let (none, _) = render(shadow_hair(Vector3::ZERO, 0.0));
    eprintln!("under the hair: {dark:.2} darker (centroid x {cx:.2}); without a caster {none:.2}");
    assert!(dark > 0.25, "the hair's shadow is too faint ({dark:.2})");
    assert!(none.abs() < 0.03, "shadow_strands 0 still cast ({none:.2})");
    assert!(
        cx.abs() < 0.08,
        "a straight hair's shadow is off centre ({cx:.2})"
    );

    // Pulled sideways, the hair's shadow goes with it — the caster is bent
    // as the ribbons are.
    let (_, pulled) = render(shadow_hair(Vector3::new(1.5, 0.0, 0.0), 0.25));
    eprintln!("pulled +x: shadow centroid x {pulled:.2}");
    assert!(
        pulled > 0.12,
        "the shadow did not follow the bent hair ({pulled:.2})"
    );
}

#[test]
fn the_animator_bends_the_shadow_in_world_space() {
    use threers::fur::FurAnimator;
    let Ok(mut hr) = HeadlessRenderer::builder().size(SHADOW_N, SHADOW_N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Turned a quarter round z, so the pull (+x in world) is −y in the
    // hair's own frame: built without knowing that, its caster leans the
    // wrong way until the animator gives it the world.
    let pull = Vector3::new(1.5, 0.0, 0.0);
    let mut hair = shadow_hair(pull, 0.25);
    hair.rotate_z(std::f32::consts::FRAC_PI_2);
    let (mut s, id) = shadow_scene(hair);
    let (_, before) = shade(&hr.render_to_rgba(&mut s, &shadow_camera()));
    let mut animator = FurAnimator::new();
    let mut mat = threers::fur::FurMaterial::default();
    mat.groom.gravity = pull;
    animator.track(id, &mat);
    animator.update(&mut s, 0.0, 1.0 / 60.0);
    let (_, after) = shade(&hr.render_to_rgba(&mut s, &shadow_camera()));
    eprintln!("turned hair, shadow centroid x: built {before:.2}, after the animator {after:.2}");
    assert!(
        after > 0.12,
        "the animated shadow does not lean with world gravity ({after:.2})"
    );
    assert!(
        after > before + 0.08,
        "the animator changed nothing ({before:.2} -> {after:.2})"
    );
}

/// The floor's shadow, as the furthest a darkened floor point lies from
/// under the centre along ±x (half the darkened pixels' run on the x axis).
fn shadow_half_width(px: &[u8]) -> f32 {
    let floor = floor_points();
    let (mut far, mut nf) = (0.0f64, 0u32);
    for (i, p) in floor.iter().enumerate() {
        if let Some((x, z)) = p {
            if (2.5..3.5).contains(&(x * x + z * z).sqrt()) {
                far += px[i * 4] as f64;
                nf += 1;
            }
        }
    }
    let far = far / nf.max(1) as f64;
    floor
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            let (x, z) = (*p)?;
            (z.abs() < 0.05 && (px[i * 4] as f64) < 0.6 * far).then_some(x.abs())
        })
        .fold(0.0, f32::max)
}

fn shadow_fur(pull: Vector3, shadow_height: f32) -> Object3D {
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.4,
            density: 50.0,
            gravity: pull,
            ..Default::default()
        },
        shadow_height,
        ..Default::default()
    };
    fur_object(SphereGeometry::new(0.5, 48, 24), &fur)
}

#[test]
fn fur_casts_a_shadow_as_wide_as_its_coat() {
    let Ok(mut hr) = HeadlessRenderer::builder().size(SHADOW_N, SHADOW_N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut render = |fur: Object3D| {
        let (mut s, _) = shadow_scene(fur);
        hr.render_to_rgba(&mut s, &shadow_camera())
    };
    let px = render(shadow_fur(Vector3::ZERO, 0.6));
    let (dark, cx) = shade(&px);
    let half = shadow_half_width(&px);
    let (none, _) = shade(&render(shadow_fur(Vector3::ZERO, 0.0)));
    eprintln!(
        "fur shadow: {dark:.2} darker under it, half-width {half:.2} (skin 0.5, caster 0.74), \
         centroid x {cx:.2}; shadow_height 0: {none:.2}"
    );
    assert!(dark > 0.5, "the fur's shadow is too faint ({dark:.2})");
    assert!(none.abs() < 0.03, "shadow_height 0 still cast ({none:.2})");
    // Wider than the bare skin's, as wide as the caster's 0.5 + 0.6 × 0.4.
    assert!(
        (half - 0.74).abs() < 0.08,
        "shadow half-width {half:.2}, caster 0.74"
    );
    assert!(
        cx.abs() < 0.05,
        "a straight coat's shadow is off centre ({cx:.2})"
    );

    let (_, pulled) = shade(&render(shadow_fur(Vector3::new(1.5, 0.0, 0.0), 0.6)));
    eprintln!("pulled +x: fur shadow centroid x {pulled:.2}");
    assert!(
        pulled > 0.05,
        "the fur's shadow did not follow its droop ({pulled:.2})"
    );
}

// ---------------------------------------------------------------------------
// Anti-aliasing
// ---------------------------------------------------------------------------

/// Mean absolute difference, per channel, over two RGBA images.
fn mean_diff(a: &[u8], b: &[u8]) -> f64 {
    a.iter()
        .zip(b)
        .enumerate()
        .filter(|(i, _)| i % 4 != 3)
        .map(|(_, (x, y))| (*x as f64 - *y as f64).abs())
        .sum::<f64>()
        / (a.len() as f64 * 0.75)
}

/// Fur and hair under MSAA come out as a supersampled render of the same
/// scene does — each sample tested against the strands — and much closer to
/// it than a single-sampled frame, whose strands alias.
#[test]
fn msaa_resolves_strands_as_supersampling_does() {
    const S: u32 = 128;
    let Ok(mut single) = HeadlessRenderer::builder().size(S, S).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut msaa = HeadlessRenderer::builder().size(S, S).build().unwrap();
    msaa.set_msaa(4);
    let mut truth = HeadlessRenderer::builder()
        .size(S, S)
        .supersample(4)
        .build()
        .unwrap();
    let coat = FurMaterial {
        groom: FurGroom {
            length: 0.35,
            density: 90.0,
            gravity: Vector3::new(0.0, -0.3, 0.0),
            ..Default::default()
        },
        root_color: Color::new(0.2, 0.1, 0.05),
        tip_color: Color::new(0.95, 0.8, 0.55),
        lod: false,
        ..Default::default()
    };
    type Make<'a> = Box<dyn Fn() -> Object3D + 'a>;
    let objects: [(&str, Make); 2] = [
        (
            "fur",
            Box::new(|| fur_object(SphereGeometry::new(1.0, 64, 32), &coat)),
        ),
        (
            "hair",
            Box::new(|| {
                hair_object(
                    &SphereGeometry::new(0.9, 48, 24),
                    &hair(Vector3::new(0.0, -1.0, 0.0)),
                    &StrandOptions {
                        count: 6000,
                        segments: 8,
                        width: 0.006,
                        ..Default::default()
                    },
                )
            }),
        ),
    ];
    for (what, make) in &objects {
        let r = truth.render_to_rgba_resolved(&mut scene_with(make()), &camera());
        let a = single.render_to_rgba(&mut scene_with(make()), &camera());
        let m = msaa.render_to_rgba(&mut scene_with(make()), &camera());
        assert!(
            msaa.renderer().last_frame_multisampled(),
            "{what}: the frame fell back to one sample"
        );
        let (ea, em) = (mean_diff(&a, &r), mean_diff(&m, &r));
        eprintln!(
            "{what}: off a 16× supersampled render by {ea:.2} single-sampled, {em:.2} under MSAA"
        );
        assert!(
            em < 0.7 * ea,
            "{what}: MSAA ({em:.2}) no closer than one sample ({ea:.2})"
        );
    }
}

/// A hand in the coat (`FurAnimator::stroke`) moves the fur where it is and
/// nowhere else; let go, the coat springs back.
#[test]
fn a_hand_strokes_the_fur_and_it_springs_back() {
    use threers::fur::FurAnimator;
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fur = FurMaterial {
        groom: FurGroom {
            length: 0.4,
            density: 50.0,
            gravity: Vector3::ZERO,
            ..Default::default()
        },
        tip_color: Color::new(0.9, 0.8, 0.6),
        ..Default::default()
    };
    let mut scene = scene_with(Object3D::group());
    let ball = scene.add(fur_object(SphereGeometry::new(0.8, 64, 32), &fur));
    let mut animator = FurAnimator::new();
    animator.track(ball, &fur);
    let dt = 1.0 / 60.0;
    for f in 0..5 {
        animator.update(&mut scene, f as f32 * dt, dt);
    }
    let before = hr.render_to_rgba(&mut scene, &camera());
    // A hand at the ball's upper right, sweeping left and parting the fur.
    let at = Vector3::new(0.55, 0.55, 0.3);
    animator.stroke(ball, at, 0.35, Vector3::new(-1.5, 0.0, 0.0), 1.0);
    animator.update(&mut scene, 5.0 * dt, dt);
    let stroked = hr.render_to_rgba(&mut scene, &camera());
    // Where it changed: near the hand's place on screen, not across the ball.
    let per_unit = N as f32 / 2.0 / (5.0 * 20f32.to_radians().tan());
    let (hx, hy) = (
        N as f32 / 2.0 + at.x * per_unit,
        N as f32 / 2.0 - at.y * per_unit,
    );
    let (mut near, mut far) = (0, 0);
    for (i, (p, q)) in before.chunks(4).zip(stroked.chunks(4)).enumerate() {
        if (0..3).any(|c| (p[c] as i32 - q[c] as i32).abs() > 24) {
            let (x, y) = ((i as u32 % N) as f32, (i as u32 / N) as f32);
            if ((x - hx).powi(2) + (y - hy).powi(2)).sqrt() < 0.5 * per_unit {
                near += 1;
            } else {
                far += 1;
            }
        }
    }
    eprintln!("stroked: {near} pixels changed near the hand, {far} elsewhere");
    assert!(near > 60, "the hand moved almost nothing ({near})");
    assert!(
        far < near / 2,
        "the hand moved fur far from it ({far} vs {near})"
    );
    // Let go: a second later, back as it was.
    for f in 0..60 {
        animator.update(&mut scene, (6 + f) as f32 * dt, dt);
    }
    let after = hr.render_to_rgba(&mut scene, &camera());
    let left = before
        .chunks(4)
        .zip(after.chunks(4))
        .filter(|(p, q)| (0..3).any(|c| (p[c] as i32 - q[c] as i32).abs() > 24))
        .count();
    eprintln!("a second after letting go: {left} pixels still differ");
    assert!(left < 3, "the coat did not spring back ({left} pixels)");
}
