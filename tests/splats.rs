//! Gaussian splats (`threers::splats`): the file formats against layouts
//! written by hand, the covariance against the crate's own matrices, and the
//! sort against a plain comparison sort. The rendering tests skip without a
//! GPU adapter.

use threers::math::{Matrix4, Quaternion};
use threers::prelude::*;
use threers::splats::*;

fn normalize_quat(q: [f32; 4]) -> [f32; 4] {
    let l = q.iter().map(|v| v * v).sum::<f32>().sqrt();
    q.map(|v| v / l)
}

/// A few splats with every field distinct, at SH degree `degree`.
fn sample(degree: u32) -> GaussianSplats {
    let mut s = GaussianSplats {
        sh_degree: degree,
        ..Default::default()
    };
    for i in 0..7 {
        let f = i as f32;
        s.push(
            Vector3::new(f * 0.3 - 1.0, (f * 1.7).sin(), f * -0.2),
            [0.01 + f * 0.02, 0.05, 0.1 + f * 0.01],
            [0.1 * f, -0.2, 0.3, 1.0],
            0.2 + f * 0.1,
            [0.1 * f, 0.5, 1.0 - 0.1 * f],
        );
    }
    let n = s.sh.len();
    s.sh = (0..n).map(|k| ((k as f32) * 0.37).sin() * 0.2).collect();
    s
}

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn covariance_is_r_s_s_r_transposed() {
    // Against the crate's own quaternion-to-matrix, not the formula above.
    let (s, q) = ([0.3, 0.05, 1.2], [0.2, -0.4, 0.1, 0.9]);
    let [x, y, z, w] = normalize_quat(q);
    let m = Matrix4::compose(
        Vector3::ZERO,
        Quaternion::new(x, y, z, w),
        Vector3::new(s[0], s[1], s[2]),
    );
    // Column-major: M[i][j] is elements[j * 4 + i].
    let at = |i: usize, j: usize| m.elements[j * 4 + i];
    let e = |a: usize, b: usize| (0..3).map(|k| at(a, k) * at(b, k)).sum::<f32>();
    let want = [e(0, 0), e(0, 1), e(0, 2), e(1, 1), e(1, 2), e(2, 2)];
    let got = covariance(s, q);
    for k in 0..6 {
        assert!(close(got[k], want[k], 1e-5), "{k}: {} vs {}", got[k], want[k]);
    }
}

#[test]
fn ply_round_trips_at_every_degree() {
    for degree in 0..=3 {
        let s = sample(degree);
        let back = GaussianSplats::from_ply(&s.to_ply().unwrap()).unwrap();
        assert_eq!(back.sh_degree, degree);
        assert_eq!(back.len(), s.len());
        for i in 0..s.len() {
            assert_eq!(back.positions[i], s.positions[i]);
            for k in 0..3 {
                assert!(close(back.scales[i][k], s.scales[i][k], 1e-5));
                assert!(close(back.colors[i][k], s.colors[i][k], 1e-5));
            }
            for k in 0..4 {
                assert!(close(back.rotations[i][k], s.rotations[i][k], 1e-5));
            }
            assert!(close(back.opacities[i], s.opacities[i], 1e-5));
        }
        assert_eq!(back.sh.len(), s.sh.len());
        for (a, b) in back.sh.iter().zip(&s.sh) {
            assert!(close(*a, *b, 1e-6));
        }
    }
}

/// A file laid out by hand as the reference trainer lays it out, rather
/// than by `to_ply`: big-endian, a leading element to skip, and `f_rest`
/// channel-major.
#[test]
fn ply_reads_the_trainers_layout() {
    let mut names = vec!["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
    names.extend((0..9).map(|k| format!("f_rest_{k}")));
    names.extend(
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
            .map(String::from),
    );
    let mut text = String::from(
        "ply\nformat binary_big_endian 1.0\ncomment made by hand\nelement camera 1\n\
         property uchar id\nproperty list uchar int ids\nelement vertex 1\n",
    );
    for n in &names {
        text.push_str(&format!("property float {n}\n"));
    }
    text.push_str("end_header\n");
    let mut bytes = text.into_bytes();
    bytes.extend([7u8, 2]);
    bytes.extend(5i32.to_be_bytes());
    bytes.extend(6i32.to_be_bytes());
    // f_rest_k = 10 * channel + coefficient
    let rest: Vec<f32> = (0..3).flat_map(|ch| (0..3).map(move |k| (10 * ch + k) as f32)).collect();
    let values: Vec<f32> = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
        .into_iter()
        .chain(rest)
        .chain([0.0, (0.5f32).ln(), 0.0, (2.0f32).ln(), 0.0, 0.0, 0.0, 2.0])
        .collect();
    for v in values {
        bytes.extend(v.to_be_bytes());
    }
    let s = GaussianSplats::from_ply(&bytes).unwrap();
    assert_eq!(s.positions, vec![[1.0, 2.0, 3.0]]);
    assert!(close(s.colors[0][0], 0.5, 1e-6));
    assert!(close(s.colors[0][1], 0.5 + SH_C0, 1e-6));
    assert!(close(s.opacities[0], 0.5, 1e-6));
    assert!(close(s.scales[0][0], 0.5, 1e-6) && close(s.scales[0][2], 2.0, 1e-6));
    // rot_0 is w: (w, x, y, z) = (0, 0, 0, 2) → z axis.
    assert_eq!(s.rotations[0], [0.0, 0.0, 1.0, 0.0]);
    assert_eq!(s.sh_degree, 1);
    // Coefficient-major in memory: coefficient 1 is (1, 11, 21).
    assert_eq!(&s.sh[3..6], &[1.0, 11.0, 21.0]);
}

#[test]
fn ascii_ply_with_byte_colours() {
    let text = "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n\
                property float z\nproperty float scale_0\nproperty float scale_1\n\
                property float scale_2\nproperty uchar red\nproperty uchar green\n\
                property uchar blue\nend_header\n\
                0 0 0 0 0 0 255 0 51\n1 2 3 -1 -1 -1 0 255 0\n";
    let s = GaussianSplats::from_ply(text.as_bytes()).unwrap();
    assert_eq!(s.len(), 2);
    assert_eq!(s.colors[0], [1.0, 0.0, 0.2]);
    assert_eq!(s.opacities, vec![1.0, 1.0]);
    assert_eq!(s.rotations[1], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn malformed_plys_are_errors_not_panics() {
    let good = sample(1).to_ply().unwrap();
    let header_end = good.windows(10).position(|w| w == b"end_header").unwrap();
    for cut in [0, 3, header_end, header_end + 11, good.len() - 1] {
        assert!(GaussianSplats::from_ply(&good[..cut]).is_err(), "cut at {cut}");
    }
    let huge = String::from_utf8_lossy(&good[..header_end + 11])
        .replace("element vertex 7", "element vertex 99999999999")
        .into_bytes();
    assert!(GaussianSplats::from_ply(&huge).is_err());
    let no_scale = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n\
                     property float y\nproperty float z\nend_header\n0 0 0\n";
    let e = GaussianSplats::from_ply(no_scale).unwrap_err();
    assert!(e.to_string().contains("scale_0"), "{e}");
}

#[test]
fn splat_format_round_trips_within_quantisation() {
    let mut s = sample(0);
    s.colors[0] = [1.5, -0.2, 0.5];
    let bytes = s.to_splat().unwrap();
    assert_eq!(bytes.len(), s.len() * 32);
    let back = GaussianSplats::from_splat(&bytes).unwrap();
    assert_eq!(back.positions, s.positions);
    assert_eq!(back.scales, s.scales);
    assert_eq!(back.colors[0], [1.0, 0.0, 128.0 / 255.0]);
    for i in 0..s.len() {
        assert!((back.opacities[i] - s.opacities[i]).abs() <= 0.5 / 255.0 + 1e-6);
        // A quaternion and its negation are one rotation.
        let d: f32 = (0..4).map(|k| back.rotations[i][k] * s.rotations[i][k]).sum();
        assert!(d.abs() > 0.9995, "splat {i}: {d}");
    }
    assert!(GaussianSplats::from_splat(&bytes[..31]).is_err());
}

#[test]
fn degree_zero_colour_is_band_zero() {
    let s = sample(0);
    assert_eq!(s.color_towards(2, Vector3::new(0.0, 0.0, 1.0)), s.colors[2]);
}

#[test]
fn truncating_sh_keeps_the_low_bands() {
    let s3 = sample(3);
    let mut s1 = s3.clone();
    s1.truncate_sh(1);
    s1.validate().unwrap();
    for i in 0..s3.len() {
        assert_eq!(&s1.sh[i * 9..][..9], &s3.sh[i * 45..][..9]);
    }
}

#[test]
fn validate_names_the_short_vector() {
    let mut s = sample(1);
    s.opacities.pop();
    assert!(s.validate().unwrap_err().to_string().contains("opacities"));
    let mut s = sample(1);
    s.sh.pop();
    assert!(s.validate().is_err());
}

#[test]
fn the_sorter_orders_back_to_front() {
    let mut s = GaussianSplats::default();
    let mut seed = 12345u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32 * 20.0 - 10.0
    };
    for _ in 0..5000 {
        s.push(Vector3::new(rnd(), rnd(), rnd()), [0.1; 3], [0.0, 0.0, 0.0, 1.0], 1.0, [1.0; 3]);
    }
    let material = splat_material(&s, &SplatOptions::default()).unwrap();
    let mut sorter = SplatSorter::new(&s, &material);
    let mut cam = threers::cameras::PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    for (eye, model) in [
        (Vector3::new(0.0, 0.0, 30.0), Matrix4::identity()),
        (Vector3::new(25.0, 10.0, -5.0), Matrix4::scale(Vector3::new(2.0, 0.5, 1.0))),
    ] {
        cam.position = eye;
        threers::cameras::Camera::look_at(&mut cam, Vector3::ZERO);
        let mv = threers::cameras::Camera::view_matrix(&cam).multiply(&model);
        assert!(sorter.sort(&mv));
        assert!(!sorter.sort(&mv), "an unchanged view re-sorted");
        let z = |i: u32| {
            let p = s.positions[i as usize];
            Vector3::new(p[0], p[1], p[2]).apply_matrix4(&mv).z
        };
        let order = sorter.order();
        let mut seen = order.to_vec();
        seen.sort();
        assert!(seen.iter().enumerate().all(|(k, &i)| k as u32 == i), "not a permutation");
        let range = order.iter().map(|&i| z(i)).fold(0.0f32, |m, v| m.max(v.abs()));
        for w in order.windows(2) {
            // Quantised to 24 bits of the depth range.
            assert!(z(w[0]) <= z(w[1]) + range * 1e-6, "{} before {}", z(w[0]), z(w[1]));
        }
    }
}

// --- On the GPU. ---

const N: u32 = 128;

fn camera_at(eye: Vector3) -> PerspectiveCamera {
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    cam.position = eye;
    threers::cameras::Camera::look_at(&mut cam, Vector3::ZERO);
    cam
}

/// Pixels per world unit at distance `d` on this camera's axis.
fn focal(d: f32) -> f32 {
    (N as f32 / 2.0) / (25.0f32.to_radians().tan() * d)
}

fn render_splats(
    hr: &mut HeadlessRenderer,
    splats: &GaussianSplats,
    extra: Option<Object3D>,
    eye: Vector3,
) -> Vec<u8> {
    let mut scene = Scene::new();
    scene.background = Color::new(0.0, 0.0, 0.0);
    let (object, mut sorter) = splat_object(splats, &SplatOptions::default()).unwrap();
    let id = scene.add(object);
    if let Some(e) = extra {
        scene.add(e);
    }
    let cam = camera_at(eye);
    sorter.update(&mut scene, id, &cam);
    hr.render_to_rgba(&mut scene, &cam)
}

fn renderer() -> Option<HeadlessRenderer> {
    match HeadlessRenderer::builder().size(N, N).build() {
        Ok(hr) => Some(hr),
        Err(_) => {
            eprintln!("skipping: no GPU adapter");
            None
        }
    }
}

fn px(img: &[u8], x: u32, y: u32) -> [u8; 4] {
    img[((y * N + x) * 4) as usize..][..4].try_into().unwrap()
}

fn one(position: Vector3, scale: [f32; 3], rotation: [f32; 4], opacity: f32, color: [f32; 3]) -> GaussianSplats {
    let mut s = GaussianSplats::default();
    s.push(position, scale, rotation, opacity, color);
    s
}

/// Native wgpu, WebGL2 (no storage buffers), and naga's GLSL ES 3.0 output
/// for both stages, which wgpu's WebGL2 backend compiles.
#[test]
fn the_splat_shader_compiles_for_wgpu_and_webgl2() {
    let m = splat_material(&sample(3), &SplatOptions::default()).unwrap();
    m.validate(true).expect("native wgpu");
    m.validate(false).expect("WebGL2 (no storage buffers)");
    let source = m.module_source(false);
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
    }
}

/// One round splat against the Gaussian it is: across the row through its
/// centre, each pixel is `colour × opacity × exp(−r²/2σ²)` over black, with
/// σ² the projected variance plus the 0.3 px² low-pass, in sRGB bytes.
#[test]
fn a_splat_draws_its_gaussian() {
    let Some(mut hr) = renderer() else { return };
    let sigma = 0.15;
    let (opacity, color) = (0.9, [1.0, 0.6, 0.2]);
    let img = render_splats(
        &mut hr,
        &one(Vector3::ZERO, [sigma; 3], [0.0, 0.0, 0.0, 1.0], opacity, color),
        None,
        Vector3::new(0.0, 0.0, 5.0),
    );
    let s2 = (sigma * focal(5.0)).powi(2) + 0.3;
    let encode = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        255.0 * if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
    };
    let decode = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    let (mut worst, mut lit) = (0.0f32, 0);
    for x in 0..N {
        // Pixel centres sit half a pixel off the splat's centre at N/2.
        let r = x as f32 + 0.5 - N as f32 / 2.0;
        let a = (opacity * (-0.5 * r * r / s2).exp()).min(0.99);
        let a = if a < 1.0 / 255.0 { 0.0 } else { a };
        let p = px(&img, x, N / 2 - 1);
        for ch in 0..3 {
            // Blending happens on the stored (encoded) values.
            let want = encode(decode(color[ch])) * a;
            worst = worst.max((p[ch] as f32 - want).abs());
        }
        lit += (p[0] > 8) as u32;
    }
    assert!(lit > 4, "nothing drawn");
    assert!(worst <= 3.0, "off the Gaussian by {worst} levels");
}

/// A splat stretched along the x = y diagonal reaches further along it than
/// across it.
#[test]
fn an_anisotropic_splat_turns_with_its_rotation() {
    let Some(mut hr) = renderer() else { return };
    let h = std::f32::consts::FRAC_PI_8; // 45° about z
    let img = render_splats(
        &mut hr,
        &one(Vector3::ZERO, [0.6, 0.05, 0.05], [0.0, 0.0, h.sin(), h.cos()], 1.0, [1.0; 3]),
        None,
        Vector3::new(0.0, 0.0, 5.0),
    );
    let c = N / 2;
    let along = px(&img, c + 15, c - 16)[0]; // up and right on screen: +x +y
    let across = px(&img, c - 16, c - 16)[0];
    assert!(along > 100 && across < 10, "along {along}, across {across}");
}

/// An opaque box in front of the splat hides it; the splat in front of a box
/// shows over it.
#[test]
fn splats_are_depth_tested_against_the_scene() {
    let Some(mut hr) = renderer() else { return };
    let splat = one(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 1.0, [1.0, 0.0, 0.0]);
    let mut wall = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(1.0, 4.0),
        Material::Basic(BasicMaterial {
            color: Color::new(0.0, 0.0, 1.0),
            ..Default::default()
        }),
    ));
    // Covering the left half of the screen, in front of the splat.
    wall.position = Vector3::new(-0.5, 0.0, 1.0);
    let img = render_splats(&mut hr, &splat, Some(wall.clone()), Vector3::new(0.0, 0.0, 5.0));
    let (c, d) = (N / 2, 6);
    assert!(px(&img, c - d, c)[0] < 10 && px(&img, c - d, c)[2] > 200, "{:?}", px(&img, c - d, c));
    assert!(px(&img, c + d, c)[0] > 150, "{:?}", px(&img, c + d, c));
    // Behind the splat instead: red over blue on the left.
    wall.position.z = -1.0;
    let img = render_splats(&mut hr, &splat, Some(wall), Vector3::new(0.0, 0.0, 5.0));
    assert!(px(&img, c - d, c)[0] > 150, "{:?}", px(&img, c - d, c));
}

/// Two opaque-ish splats one behind the other: whichever is nearer the
/// camera covers the other, seen from either side — the sort, not the order
/// they were given in.
#[test]
fn the_nearer_splat_is_drawn_over_the_farther() {
    let Some(mut hr) = renderer() else { return };
    let mut s = GaussianSplats::default();
    s.push(Vector3::new(0.0, 0.0, 0.5), [0.4, 0.4, 0.02], [0.0, 0.0, 0.0, 1.0], 0.99, [1.0, 0.0, 0.0]);
    s.push(Vector3::new(0.0, 0.0, -0.5), [0.4, 0.4, 0.02], [0.0, 0.0, 0.0, 1.0], 0.99, [0.0, 0.0, 1.0]);
    let c = N / 2;
    let front = render_splats(&mut hr, &s, None, Vector3::new(0.0, 0.0, 5.0));
    let p = px(&front, c, c);
    assert!(p[0] > 200 && p[2] < 30, "from +z, red is nearer: {p:?}");
    let back = render_splats(&mut hr, &s, None, Vector3::new(0.0, 0.0, -5.0));
    let p = px(&back, c, c);
    assert!(p[2] > 200 && p[0] < 30, "from -z, blue is nearer: {p:?}");
}

/// Degree-1 colour seen from opposite sides: the band's sign flips with the
/// direction, as `color_towards` says it should.
#[test]
fn view_dependent_colour_follows_the_spherical_harmonics() {
    let Some(mut hr) = renderer() else { return };
    let mut s = GaussianSplats { sh_degree: 1, ..Default::default() };
    s.push(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 0.99, [0.5, 0.5, 0.5]);
    // Coefficient 1 multiplies +z (C1·z): red rises looking along +z.
    s.sh[3] = 0.6;
    let c = N / 2;
    for (eye, look) in [(5.0, -1.0), (-5.0, 1.0)] {
        let img = render_splats(&mut hr, &s, None, Vector3::new(0.0, 0.0, eye));
        let want = s.color_towards(0, Vector3::new(0.0, 0.0, look))[0].clamp(0.0, 1.0);
        let got = px(&img, c, c)[0] as f32 / 255.0 / 0.99;
        assert!((got - want).abs() < 0.03, "eye z {eye}: red {got}, expected {want}");
    }
}

#[test]
fn the_storage_limit_is_slot_zero() {
    let s = sample(3);
    let opts = |bytes: u64| SplatOptions { max_storage_bytes: bytes, ..Default::default() };
    let n = s.len() as u64;
    // 16 floats a splat in slot 0; in slot 2, 16, 12 and 4 at degrees 3, 2
    // and 1 — never more, so whatever fits slot 0 keeps every band.
    let m = splat_material(&s, &opts(16 * 4 * n)).unwrap();
    assert_eq!(splat_sh_degree(&m), 3);
    assert_eq!(m.storage0.len(), 16 * n as usize);
    assert_eq!(m.storage2.len(), 16 * n as usize);
    for (degree, stride) in [(1, 4), (2, 12)] {
        let capped = SplatOptions { max_sh_degree: degree, ..Default::default() };
        let m = splat_material(&s, &capped).unwrap();
        assert_eq!(splat_sh_degree(&m), degree);
        assert_eq!(m.storage2.len(), stride * n as usize);
    }
    assert!(splat_material(&s, &opts(16 * 4 * n - 1)).is_err());
}

/// Degree 3 with every coefficient set, seen from six directions, against
/// `color_towards` on the unquantised coefficients: the packed bands are
/// within a byte of each splat's largest.
#[test]
fn degree_three_colour_survives_packing() {
    let Some(mut hr) = renderer() else { return };
    let mut s = GaussianSplats { sh_degree: 3, ..Default::default() };
    s.push(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 0.99, [0.5, 0.45, 0.55]);
    let n = s.sh.len();
    s.sh = (0..n).map(|k| ((k as f32) * 1.3).sin() * 0.12).collect();
    let c = N / 2;
    for eye in [
        Vector3::new(0.0, 0.0, 5.0),
        Vector3::new(0.0, 0.0, -5.0),
        Vector3::new(5.0, 0.0, 0.0),
        Vector3::new(-3.0, 3.0, 2.0),
        Vector3::new(0.5, 4.9, 0.5),
        Vector3::new(2.0, -3.0, -3.0),
    ] {
        let img = render_splats(&mut hr, &s, None, eye);
        let want = s.color_towards(0, -eye);
        let p = px(&img, c, c);
        for ch in 0..3 {
            let got = p[ch] as f32 / 255.0 / 0.99;
            let w = want[ch].clamp(0.0, 1.0);
            assert!((got - w).abs() < 0.03, "eye {eye:?} channel {ch}: {got} vs {w}");
        }
    }
}

/// The sort's cost for a million splats, the least of twenty turns (the
/// machine this runs on is shared, and the least is what the code costs).
/// `cargo test --release --features splats --test splats -- --ignored sort_timing --nocapture`
/// Bands at the ends of the packing's range: every coefficient ±m, so every
/// byte 255 or 0. The shader once decoded the packed integer as `u32(x +
/// 0.5)`, which past 2^23 rounds to even: at blue 255 it carried into green,
/// and green and blue came out as their opposites (real captures of shiny
/// things, horned lizard statue among them, are full of these).
#[test]
fn bands_at_the_ends_of_the_packing_decode_exactly() {
    let Some(mut hr) = renderer() else { return };
    for (degree, sign) in [(1, 1.0), (1, -1.0), (3, 1.0), (3, -1.0)] {
        let mut s = GaussianSplats { sh_degree: degree, ..Default::default() };
        s.push(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 0.99, [0.5, 0.45, 0.55]);
        s.sh.fill(0.15 * sign);
        for eye in [Vector3::new(0.0, 0.0, 5.0), Vector3::new(3.0, 2.0, 3.0), Vector3::new(-4.0, -1.0, 2.0)] {
            let img = render_splats(&mut hr, &s, None, eye);
            let want = s.color_towards(0, -eye);
            let p = px(&img, N / 2, N / 2);
            for ch in 0..3 {
                let got = p[ch] as f32 / 255.0 / 0.99;
                let w = want[ch].clamp(0.0, 1.0);
                assert!(
                    (got - w).abs() < 0.02,
                    "degree {degree}, bands {sign:+}, eye {eye:?}, channel {ch}: {got} vs {w}"
                );
            }
        }
    }
}

#[test]
#[ignore]
fn sort_timing() {
    let mut s = GaussianSplats::default();
    let mut seed = 99u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32 * 20.0 - 10.0
    };
    for _ in 0..1_000_000 {
        s.push(Vector3::new(rnd(), rnd(), rnd()), [0.1; 3], [0.0, 0.0, 0.0, 1.0], 1.0, [1.0; 3]);
    }
    let material = splat_material(&s, &SplatOptions::default()).unwrap();
    let mut sorter = SplatSorter::new(&s, &material);
    let mut best = f64::INFINITY;
    for k in 0..20 {
        let a = k as f32 * 0.3;
        let mut m = Matrix4::identity();
        // A fresh view axis each time, so every call sorts.
        m.elements[2] = a.sin();
        m.elements[6] = 0.3;
        m.elements[10] = a.cos();
        let t = std::time::Instant::now();
        assert!(sorter.sort(&m));
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("sort of 1M splats: {best:.2} ms (least of 20)");
}

// --- `.spz` and the compressed PLY ---------------------------------------
//
// The fixtures come from splat-transform, versions 3 and 4 written by the
// reference library; the round trips go through ports of the reference
// writers (Niantic's `packGaussians`, PlayCanvas's `CompressedChunk`), so
// the reader is checked against layouts it did not write itself.

fn fixture(name: &str) -> Vec<u8> {
    let path = format!("{}/tests/splat_fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// A deterministic pseudo-random stream in 0..1.
fn rng(seed: u32) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    move || {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }
}

/// `n` splats with every field random, at `degree`, plus `extra` more
/// coefficients a channel (degree 4's nine) returned on the side.
fn random_splats(n: usize, degree: u32, seed: u32) -> GaussianSplats {
    let mut r = rng(seed);
    let mut s = GaussianSplats { sh_degree: degree, ..Default::default() };
    for _ in 0..n {
        let mut q = [r() - 0.5, r() - 0.5, r() - 0.5, r() - 0.5];
        if q.iter().all(|v| v.abs() < 0.05) {
            q[3] = 1.0;
        }
        s.push(
            Vector3::new(r() * 20.0 - 10.0, r() * 6.0 - 3.0, r() * 2.0 - 1.0),
            [0.01 + r(), 0.01 + r() * 0.2, 0.002 + r() * 0.05],
            q,
            0.05 + r() * 0.9,
            [r(), r(), r() * 1.2 - 0.1],
        );
    }
    for v in &mut s.sh {
        *v = r() * 1.8 - 0.9;
    }
    s
}

fn to_u8(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

/// Niantic's `packQuaternionSmallestThree`, ported.
fn pack_smallest_three(q: [f32; 4]) -> [u8; 4] {
    let q = normalize_quat(q);
    let mut largest = 0;
    for i in 1..4 {
        if q[i].abs() > q[largest].abs() {
            largest = i;
        }
    }
    let negate = (q[largest] < 0.0) as u32;
    let mut comp = largest as u32;
    for (i, &v) in q.iter().enumerate() {
        if i != largest {
            let negbit = (v < 0.0) as u32 ^ negate;
            let mag = (511.0 * (v.abs() / std::f32::consts::FRAC_1_SQRT_2) + 0.5) as u32;
            comp = (comp << 10) | (negbit << 9) | mag;
        }
    }
    comp.to_le_bytes()
}

/// Niantic's `packGaussians`: the six attribute streams, in file order, for
/// `version` (2: three-byte rotations; 3 and 4: smallest three) with
/// `file_degree` coefficients a channel (degree 4's beyond the splats' 15
/// filled from `extra`).
fn pack_spz(s: &GaussianSplats, version: u32, file_degree: u32, extra: &[f32]) -> [Vec<u8>; 6] {
    const FRACTIONAL_BITS: u32 = 12;
    let mut out: [Vec<u8>; 6] = Default::default();
    let n_rest = sh_rest_coefficients(s.sh_degree);
    let file_dim = ((file_degree + 1) * (file_degree + 1) - 1) as usize;
    for i in 0..s.len() {
        for v in s.positions[i] {
            let fixed = (v * (1 << FRACTIONAL_BITS) as f32).round() as i32;
            out[0].extend_from_slice(&fixed.to_le_bytes()[..3]);
        }
        out[1].push(to_u8(s.opacities[i] * 255.0));
        for c in s.colors[i] {
            let dc = (c - 0.5) / SH_C0;
            out[2].push(to_u8(dc * (0.15 * 255.0) + 0.5 * 255.0));
        }
        for v in s.scales[i] {
            out[3].push(to_u8((v.ln() + 10.0) * 16.0));
        }
        let [x, y, z, w] = normalize_quat(s.rotations[i]);
        if version >= 3 {
            out[4].extend_from_slice(&pack_smallest_three([x, y, z, w]));
        } else {
            let k = if w < 0.0 { -127.5 } else { 127.5 };
            out[4].extend([x, y, z].map(|v| to_u8(v * k + 127.5)));
        }
        let quantise = |v: f32| (((v * 128.0).round() + 128.0) as i32).clamp(0, 255) as u8;
        out[5].extend(s.sh[i * n_rest * 3..][..n_rest * 3].iter().map(|&v| quantise(v)));
        let more = (file_dim - n_rest) * 3;
        out[5].extend(extra[i * more..][..more].iter().map(|&v| quantise(v)));
    }
    out
}

/// A gzip member around stored (uncompressed) deflate blocks: all the
/// reader needs to see, and nothing borrowed from it.
fn gzip_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
    let blocks: Vec<&[u8]> = data.chunks(65_535).collect();
    for (k, b) in blocks.iter().enumerate() {
        out.push((k + 1 == blocks.len()) as u8);
        out.extend_from_slice(&(b.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
        out.extend_from_slice(b);
    }
    out.extend_from_slice(&[0; 8]); // CRC-32 and size: not checked
    out
}

const NGSP: [u8; 4] = *b"NGSP";

fn spz_file(s: &GaussianSplats, version: u32, file_degree: u32, flags: u8, extra: &[f32]) -> Vec<u8> {
    let streams = pack_spz(s, version, file_degree, extra);
    let n = (s.len() as u32).to_le_bytes();
    if version < 4 {
        let mut raw = NGSP.to_vec();
        raw.extend_from_slice(&version.to_le_bytes());
        raw.extend_from_slice(&n);
        raw.extend_from_slice(&[file_degree as u8, 12, flags, 0]);
        for st in &streams {
            raw.extend_from_slice(st);
        }
        return gzip_stored(&raw);
    }
    let streams: Vec<&Vec<u8>> = streams.iter().filter(|s| !s.is_empty()).collect();
    let mut out = NGSP.to_vec();
    out.extend_from_slice(&4u32.to_le_bytes());
    out.extend_from_slice(&n);
    out.extend_from_slice(&[file_degree as u8, 12, flags, streams.len() as u8]);
    out.extend_from_slice(&32u32.to_le_bytes());
    out.resize(32, 0);
    let mut frames: Vec<Vec<u8>> = streams
        .iter()
        .map(|s| ruzstd::encoding::compress_to_vec(&s[..], ruzstd::encoding::CompressionLevel::Fastest))
        .collect();
    // Extension frames follow the last stream's, inside its compressed
    // size: a skippable frame here, which a reader must leave alone.
    let last = frames.last_mut().unwrap();
    last.extend_from_slice(&0x184d_2a50u32.to_le_bytes());
    last.extend_from_slice(&5u32.to_le_bytes());
    last.extend_from_slice(b"later");
    for (f, s) in frames.iter().zip(&streams) {
        out.extend_from_slice(&(f.len() as u64).to_le_bytes());
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    }
    for f in &frames {
        out.extend_from_slice(f);
    }
    out
}

fn quat_angle(a: [f32; 4], b: [f32; 4]) -> f32 {
    let d: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    2.0 * d.abs().min(1.0).acos()
}

#[test]
fn spz_fixtures_hold_what_splat_transform_wrote() {
    // Version 2, laid out by hand: a 2×2 grid at fixed point 1/256, alpha
    // byte 230, colour bytes (gx + 1) / 3 · 255 and 128, scale byte 123,
    // rotation bytes 128 (a hair off identity).
    let s = GaussianSplats::from_spz(&fixture("minimal-v2.spz")).unwrap();
    assert_eq!((s.len(), s.sh_degree, s.antialiased), (4, 0, false));
    let colour = |b: f32| 0.5 + SH_C0 * ((b / 255.0 - 0.5) / 0.15);
    for i in 0..4 {
        let (gx, gz) = ((i % 2) as f32, (i / 2) as f32);
        assert_eq!(s.positions[i], [gx - 1.0, 0.0, gz - 1.0]);
        assert_eq!(s.opacities[i], 230.0 / 255.0);
        let cr = colour(((gx + 1.0) / 3.0 * 255.0).round());
        let cg = colour(((gz + 1.0) / 3.0 * 255.0).round());
        for (c, want) in s.colors[i].iter().zip([cr, cg, colour(128.0)]) {
            assert!((c - want).abs() < 1e-6, "{i}: {:?}", s.colors[i]);
        }
        let scale = (123.0f32 / 16.0 - 10.0).exp();
        assert!(s.scales[i].iter().all(|v| (v - scale).abs() < 1e-6));
        assert!(quat_angle(s.rotations[i], [0.0, 0.0, 0.0, 1.0]) < 0.02);
    }
    // Versions 3 and 4, written by @adobe/spz from y-down inputs (grid
    // positions, scale 0.1, opacity 0.9, colours (gx + 1) / 3, (gz + 1) / 3,
    // 0.5, identity) converted to y-up: y and z negated.
    for name in ["minimal-v3.spz", "minimal-v4.spz"] {
        let s = GaussianSplats::from_spz(&fixture(name)).unwrap();
        assert_eq!((s.len(), s.sh_degree), (4, 0), "{name}");
        for i in 0..4 {
            let (gx, gz) = ((i % 2) as f32, (i / 2) as f32);
            let want = [gx - 1.0, 0.0, -(gz - 1.0)];
            for (p, w) in s.positions[i].iter().zip(want) {
                assert!((p - w).abs() < 1e-3, "{name} {i}: {:?}", s.positions[i]);
            }
            assert!((s.opacities[i] - 0.9).abs() < 0.003, "{name}: {}", s.opacities[i]);
            for (c, w) in s.colors[i].iter().zip([(gx + 1.0) / 3.0, (gz + 1.0) / 3.0, 0.5]) {
                assert!((c - w).abs() < 0.005, "{name} {i}: {:?}", s.colors[i]);
            }
            assert!(s.scales[i].iter().all(|v| (v / 0.1).ln().abs() < 1.0 / 32.0 + 1e-4));
            assert!(quat_angle(s.rotations[i], [0.0, 0.0, 0.0, 1.0]) < 1e-3);
        }
    }
}

#[test]
fn spz_reads_every_version_the_reference_writes() {
    let s = random_splats(700, 3, 7);
    for version in [2, 3, 4] {
        for flags in [0, 1] {
            let file = spz_file(&s, version, 3, flags, &[]);
            let d = GaussianSplats::from_bytes(&file)
                .unwrap_or_else(|e| panic!("version {version}: {e}"));
            assert_eq!((d.len(), d.sh_degree, d.antialiased), (700, 3, flags == 1));
            let mut worst_angle = 0.0f32;
            for i in 0..s.len() {
                for (a, b) in d.positions[i].iter().zip(s.positions[i]) {
                    assert!((a - b).abs() <= 0.5 / 4096.0 + 1e-5, "v{version} {i}");
                }
                for (a, b) in d.scales[i].iter().zip(s.scales[i]) {
                    assert!((a / b).ln().abs() <= 1.0 / 32.0 + 1e-4, "v{version} {i}: {a} {b}");
                }
                assert!((d.opacities[i] - s.opacities[i]).abs() <= 0.5 / 255.0 + 1e-6);
                for (a, b) in d.colors[i].iter().zip(s.colors[i]) {
                    assert!((a - b).abs() <= SH_C0 / 0.15 * 0.5 / 255.0 + 1e-5, "v{version} {i}");
                }
                for (a, b) in d.sh[i * 45..][..45].iter().zip(&s.sh[i * 45..][..45]) {
                    assert!((a - b).abs() <= 0.5 / 128.0 + 1e-6, "v{version} {i}: {a} {b}");
                }
                // Three bytes with `w` rebuilt from them lose precision as
                // `w` nears zero (√(1 − |xyz|²) is steep there): judge
                // version 2 where `w` is not.
                if version >= 3 || s.rotations[i][3].abs() > 0.3 {
                    worst_angle = worst_angle.max(quat_angle(d.rotations[i], s.rotations[i]));
                }
                assert!(quat_angle(d.rotations[i], s.rotations[i]) < 0.25);
            }
            let limit = if version >= 3 { 0.004 } else { 0.03 };
            assert!(worst_angle < limit, "v{version}: rotations off by {worst_angle} rad");
        }
    }
}

#[test]
fn spz_degree_four_keeps_degree_three() {
    let s = random_splats(50, 3, 11);
    let mut r = rng(5);
    let extra: Vec<f32> = (0..50 * 27).map(|_| r() - 0.5).collect();
    for version in [3, 4] {
        let d = GaussianSplats::from_spz(&spz_file(&s, version, 4, 0, &extra)).unwrap();
        assert_eq!(d.sh_degree, 3);
        for (a, b) in d.sh.iter().zip(&s.sh) {
            assert!((a - b).abs() <= 0.5 / 128.0 + 1e-6);
        }
    }
}

#[test]
fn spz_without_harmonics_has_five_streams() {
    let s = random_splats(30, 0, 3);
    let file = spz_file(&s, 4, 0, 0, &[]);
    assert_eq!(file[15], 5);
    let d = GaussianSplats::from_spz(&file).unwrap();
    assert_eq!((d.len(), d.sh_degree, d.sh.len()), (30, 0, 0));
}

/// PlayCanvas's `CompressedChunk.pack` and the PLY `write-compressed-ply`
/// writes around it, ported: 256 splats a chunk, the last padded with
/// copies of its final splat; `chunk_colours` picks the 18-property chunk.
fn compressed_ply(s: &GaussianSplats, chunk_colours: bool) -> Vec<u8> {
    const CHUNK: usize = 256;
    let n = s.len();
    let chunks = n.div_ceil(CHUNK);
    let n_rest = sh_rest_coefficients(s.sh_degree);
    let mut h = format!("ply\nformat binary_little_endian 1.0\nelement chunk {chunks}\n");
    let mut props = vec!["min_x", "min_y", "min_z", "max_x", "max_y", "max_z"];
    props.extend(["min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z"]);
    if chunk_colours {
        props.extend(["min_r", "min_g", "min_b", "max_r", "max_g", "max_b"]);
    }
    for p in props {
        h += &format!("property float {p}\n");
    }
    h += &format!("element vertex {n}\n");
    for p in ["packed_position", "packed_rotation", "packed_scale", "packed_color"] {
        h += &format!("property uint {p}\n");
    }
    if n_rest > 0 {
        h += &format!("element sh {n}\n");
        for k in 0..n_rest * 3 {
            h += &format!("property uchar f_rest_{k}\n");
        }
    }
    h += "end_header\n";
    let pack_unorm = |v: f32, bits: u32| {
        let t = ((1u32 << bits) - 1) as f32;
        (v * t + 0.5).floor().clamp(0.0, t) as u32
    };
    let normalise = |x: f32, min: f32, max: f32| {
        if x <= min {
            0.0
        } else if x >= max {
            1.0
        } else if max - min < 0.00001 {
            0.0
        } else {
            (x - min) / (max - min)
        }
    };
    let (mut chunk_data, mut vertex) = (Vec::new(), Vec::new());
    for c in 0..chunks {
        let idx: Vec<usize> = (0..CHUNK).map(|j| (c * CHUNK + j).min(n - 1)).collect();
        let range = |f: &dyn Fn(usize) -> f32| {
            idx.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &i| (lo.min(f(i)), hi.max(f(i))))
        };
        let pos: Vec<_> = (0..3).map(|k| range(&|i| s.positions[i][k])).collect();
        let scale: Vec<_> = (0..3)
            .map(|k| {
                let (lo, hi) = range(&|i| s.scales[i][k].ln());
                (lo.clamp(-20.0, 20.0), hi.clamp(-20.0, 20.0))
            })
            .collect();
        // Colours as `f_dc · SH_C0 + 0.5`, which is what they are here.
        let colour: Vec<_> = (0..3).map(|k| range(&|i| s.colors[i][k])).collect();
        let mut floats: Vec<f32> = pos.iter().map(|r| r.0).chain(pos.iter().map(|r| r.1)).collect();
        floats.extend(scale.iter().map(|r| r.0).chain(scale.iter().map(|r| r.1)));
        if chunk_colours {
            floats.extend(colour.iter().map(|r| r.0).chain(colour.iter().map(|r| r.1)));
        }
        for v in floats {
            chunk_data.extend_from_slice(&v.to_le_bytes());
        }
        for &i in idx.iter().take(n - c * CHUNK) {
            let p = (0..3).map(|k| normalise(s.positions[i][k], pos[k].0, pos[k].1)).collect::<Vec<_>>();
            let position = (pack_unorm(p[0], 11) << 21) | (pack_unorm(p[1], 10) << 11) | pack_unorm(p[2], 11);
            let t = (0..3).map(|k| normalise(s.scales[i][k].ln(), scale[k].0, scale[k].1)).collect::<Vec<_>>();
            let sc = (pack_unorm(t[0], 11) << 21) | (pack_unorm(t[1], 10) << 11) | pack_unorm(t[2], 11);
            let rgb: Vec<f32> = (0..3)
                .map(|k| {
                    if chunk_colours {
                        normalise(s.colors[i][k], colour[k].0, colour[k].1)
                    } else {
                        s.colors[i][k]
                    }
                })
                .collect();
            let color = (pack_unorm(rgb[0], 8) << 24)
                | (pack_unorm(rgb[1], 8) << 16)
                | (pack_unorm(rgb[2], 8) << 8)
                | pack_unorm(s.opacities[i], 8);
            // rot_0..3 is w x y z; the largest made positive, the other
            // three as unorms of `v · √½ + 0.5`.
            let [x, y, z, w] = normalize_quat(s.rotations[i]);
            let mut a = [w, x, y, z];
            let largest = (1..4).fold(0, |m, k| if a[k].abs() > a[m].abs() { k } else { m });
            if a[largest] < 0.0 {
                a = a.map(|v| -v);
            }
            let mut rotation = largest as u32;
            for (k, v) in a.iter().enumerate() {
                if k != largest {
                    rotation = (rotation << 10) | pack_unorm(v * std::f32::consts::FRAC_1_SQRT_2 + 0.5, 10);
                }
            }
            for v in [position, rotation, sc, color] {
                vertex.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let mut out = h.into_bytes();
    out.extend(chunk_data);
    out.extend(vertex);
    for i in 0..n {
        // Channel-major on disk, as the plain PLY's `f_rest_*`.
        for ch in 0..3 {
            for k in 0..n_rest {
                let v = s.sh[(i * n_rest + k) * 3 + ch];
                out.push(((v / 8.0 + 0.5) * 256.0).trunc().clamp(0.0, 255.0) as u8);
            }
        }
    }
    out
}

#[test]
fn compressed_ply_reads_what_supersplat_writes() {
    for (degree, chunk_colours) in [(3, true), (1, false), (0, true)] {
        // Three chunks, the last part full.
        let s = random_splats(600, degree, 21 + degree);
        let d = GaussianSplats::from_bytes(&compressed_ply(&s, chunk_colours))
            .unwrap_or_else(|e| panic!("degree {degree}: {e}"));
        assert_eq!((d.len(), d.sh_degree), (600, degree));
        let mut worst_angle = 0.0f32;
        for i in 0..600 {
            // Positions over at most 20 units in 2047 or 1023 steps.
            let steps = [2047.0, 1023.0, 2047.0];
            let spans = [20.0, 6.0, 2.0];
            for k in 0..3 {
                assert!((d.positions[i][k] - s.positions[i][k]).abs() <= spans[k] / steps[k] * 0.5 + 1e-4);
                let log = (d.scales[i][k] / s.scales[i][k]).ln().abs();
                assert!(log <= 6.0 / steps[k] * 0.5 + 1e-3, "{i} scale {k}: {log}");
                let step = if chunk_colours { 1.3 / 255.0 } else { 1.0 / 255.0 };
                let want = if chunk_colours { s.colors[i][k] } else { s.colors[i][k].clamp(0.0, 1.0) };
                assert!((d.colors[i][k] - want).abs() <= step * 0.5 + 1e-4, "{i}: {:?} {:?}", d.colors[i], s.colors[i]);
            }
            assert!((d.opacities[i] - s.opacities[i]).abs() <= 0.5 / 255.0 + 1e-6);
            let n = sh_rest_coefficients(degree) * 3;
            for (a, b) in d.sh[i * n..][..n].iter().zip(&s.sh[i * n..][..n]) {
                assert!((a - b).abs() <= 8.0 / 256.0 + 1e-5, "{i}: {a} {b}");
            }
            worst_angle = worst_angle.max(quat_angle(d.rotations[i], s.rotations[i]));
        }
        assert!(worst_angle < 0.004, "rotations off by {worst_angle} rad");
    }
}

#[test]
fn damaged_spz_and_compressed_plys_are_errors_not_panics() {
    let s = random_splats(40, 2, 9);
    let files = [
        fixture("minimal-v2.spz"),
        fixture("minimal-v3.spz"),
        fixture("minimal-v4.spz"),
        spz_file(&s, 3, 2, 0, &[]),
        spz_file(&s, 4, 2, 0, &[]),
        compressed_ply(&s, true),
    ];
    for file in &files {
        // Every truncation.
        for len in 0..file.len() {
            let _ = GaussianSplats::from_bytes(&file[..len]);
        }
        // Single bytes changed, at random.
        let mut r = rng(file.len() as u32);
        for _ in 0..3000 {
            let mut f = file.clone();
            let at = (r() * f.len() as f32) as usize % f.len();
            f[at] = (r() * 256.0) as u8;
            let _ = GaussianSplats::from_bytes(&f);
        }
    }
    // Some damage that must be refused, not misread.
    let mut f = spz_file(&s, 4, 2, 0, &[]);
    f[15] = 4;
    assert!(GaussianSplats::from_spz(&f).is_err(), "a stream short");
    let mut f = spz_file(&s, 4, 2, 0, &[]);
    f[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(GaussianSplats::from_spz(&f).is_err(), "a huge count");
    let mut f = spz_file(&s, 4, 2, 0, &[]);
    f.push(0);
    assert!(GaussianSplats::from_spz(&f).is_err(), "bytes past the streams");
}

/// The same capture as PlayCanvas ships it twice, as a v4 `.spz` and a
/// compressed PLY (`examples/assets/splats/biker.*` in playcanvas/engine):
/// two readers, two quantisations, one scene. Set `THREERS_SPLAT_DATA` to
/// the directory holding both.
#[test]
#[ignore]
fn real_spz_and_compressed_ply_agree() {
    let dir = std::env::var("THREERS_SPLAT_DATA").expect("THREERS_SPLAT_DATA");
    let spz = GaussianSplats::from_spz(&std::fs::read(format!("{dir}/biker.spz")).unwrap()).unwrap();
    let ply = GaussianSplats::from_ply(&std::fs::read(format!("{dir}/biker.compressed.ply")).unwrap()).unwrap();
    assert_eq!(spz.len(), ply.len());
    // The `.spz` is y-up, the PLY y-down: match each spz splat to the
    // nearest PLY splat after negating y and z.
    let cell = 0.05;
    let key = |p: [f32; 3]| p.map(|v| (v / cell).floor() as i32);
    let mut grid = std::collections::HashMap::<[i32; 3], Vec<usize>>::new();
    for (i, p) in ply.positions.iter().enumerate() {
        grid.entry(key(*p)).or_default().push(i);
    }
    let (mut dist, mut colour, mut opacity, mut scale, mut angle) =
        (vec![], vec![], vec![], vec![], vec![]);
    for i in 0..spz.len() {
        let [x, y, z] = spz.positions[i];
        let p = [x, -y, -z];
        let k = key(p);
        let mut best = (f32::MAX, usize::MAX);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid.get(&[k[0] + dx, k[1] + dy, k[2] + dz]).into_iter().flatten() {
                        let q = ply.positions[j];
                        let d = (0..3).map(|c| (p[c] - q[c]).powi(2)).sum::<f32>().sqrt();
                        if d < best.0 {
                            best = (d, j);
                        }
                    }
                }
            }
        }
        if best.1 == usize::MAX {
            dist.push(f32::MAX);
            continue;
        }
        let j = best.1;
        dist.push(best.0);
        colour.push((0..3).map(|c| (spz.colors[i][c] - ply.colors[j][c]).abs()).fold(0.0, f32::max));
        opacity.push((spz.opacities[i] - ply.opacities[j]).abs());
        scale.push((0..3).map(|c| (spz.scales[i][c] / ply.scales[j][c]).ln().abs()).fold(0.0, f32::max));
        // y-up to y-down turns the rotation: x stays, y and z negate.
        let [qx, qy, qz, qw] = spz.rotations[i];
        angle.push(quat_angle([qx, -qy, -qz, qw], ply.rotations[j]));
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        (v[v.len() / 2], v[v.len() * 99 / 100])
    };
    let (d, c, o, s, a) =
        (median(&mut dist), median(&mut colour), median(&mut opacity), median(&mut scale), median(&mut angle));
    println!("median, 99th percentile: distance {d:?}, colour {c:?}, opacity {o:?}, log scale {s:?}, angle {a:?}");
    // Measured: 0.00018, 0.0037, 0, 0.031 and 0.0023 at the 99th percentile.
    assert!(d.1 < 0.001 && c.1 < 0.006 && o.1 < 0.004 && s.1 < 0.04 && a.1 < 0.004);
}

/// The reference rasteriser's forward pass (`diff-gaussian-rasterization`),
/// on the CPU: the same projection and low-pass, splats front to back by
/// view depth, `alpha = min(0.99, o·exp(power))`, nothing below 1/255, a
/// pixel finished below 1e-4 transmittance, colours accumulated in float as
/// they are stored (display space) over black. Returns display values.
fn reference_render(s: &GaussianSplats, camera: &PerspectiveCamera, w: usize, h: usize) -> Vec<[f32; 3]> {
    let view = camera.view_matrix();
    let e = view.elements;
    let fy = h as f32 / 2.0 / (camera.fov / 2.0).tan();
    let fx = fy;
    let (tanx, tany) = (w as f32 / 2.0 / fx, h as f32 / 2.0 / fy);
    let eye = camera.position;
    struct P { depth: f32, px: f32, py: f32, conic: [f32; 3], cov2: [f32; 3], radius: i32, color: [f32; 3], opacity: f32 }
    let mut ps = Vec::new();
    for i in 0..s.len() {
        let [x, y, z] = s.positions[i];
        let t = [
            e[0] * x + e[4] * y + e[8] * z + e[12],
            e[1] * x + e[5] * y + e[9] * z + e[13],
            e[2] * x + e[6] * y + e[10] * z + e[14],
        ];
        let d = -t[2];
        if d <= 0.2 {
            continue;
        }
        // `REF_CULL13`: drop centres past 1.3 times the frustum, as this
        // crate's shader did.
        if std::env::var("REF_CULL13").is_ok() && ((t[0] / d).abs() > 1.3 * tanx || (t[1] / d).abs() > 1.3 * tany) {
            continue;
        }
        let (lx, ly) = (1.3 * tanx, 1.3 * tany);
        let tx = (t[0] / d).clamp(-lx, lx) * d;
        let ty = (t[1] / d).clamp(-ly, ly) * d;
        // J in view space (y up, looking down -z), rows for screen x and y.
        let j0 = [fx / d, 0.0, fx * tx / (d * d)];
        let j1 = [0.0, fy / d, fy * ty / (d * d)];
        let c = s.covariance(i);
        let sig = [[c[0], c[1], c[2]], [c[1], c[3], c[4]], [c[2], c[4], c[5]]];
        let wm = [[e[0], e[4], e[8]], [e[1], e[5], e[9]], [e[2], e[6], e[10]]];
        let mut cv = [[0.0f32; 3]; 3];
        for a in 0..3 {
            for b in 0..3 {
                for k in 0..3 {
                    for l in 0..3 {
                        cv[a][b] += wm[a][k] * sig[k][l] * wm[b][l];
                    }
                }
            }
        }
        let q = |u: [f32; 3], v: [f32; 3]| (0..3).map(|a| (0..3).map(|b| u[a] * cv[a][b] * v[b]).sum::<f32>()).sum::<f32>();
        let (a, b, cc) = (q(j0, j0) + 0.3, q(j0, j1), q(j1, j1) + 0.3);
        let det = a * cc - b * b;
        if det <= 0.0 {
            continue;
        }
        // Variants, to find what a renderer differs in: `REF_AA` applies
        // Mip-Splatting's opacity compensation, `REF_CLAMP1` clamps colours
        // to 1 before blending, `REF_LINEAR` blends in linear light.
        let env = |k: &str| std::env::var(k).is_ok();
        let mut opacity = s.opacities[i];
        if env("REF_AA") {
            let det0 = (a - 0.3) * (cc - 0.3) - b * b;
            opacity *= (det0.max(0.0) / det).sqrt();
        }
        let mid = 0.5 * (a + cc);
        let l1 = mid + (mid * mid - det).max(0.1).sqrt();
        let radius = (3.0 * l1.sqrt()).ceil() as i32;
        // Pixel coordinates with y down, pixel centres at integers.
        let px = fx * (t[0] / d) + w as f32 / 2.0 - 0.5;
        let py = -fy * (t[1] / d) + h as f32 / 2.0 - 0.5;
        // y down flips the sign of the off-diagonal.
        let conic = [cc / det, b / det, a / det];
        let dir = Vector3::new(x, y, z) - eye;
        let mut color = s.color_towards(i, dir);
        if env("REF_CLAMP1") {
            color = color.map(|v| v.min(1.0));
        }
        if env("REF_LINEAR") {
            color = color.map(|v| {
                let v = v.min(1.0);
                if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
            });
        }
        ps.push(P { depth: d, px, py, conic, cov2: [a, b, cc], radius, color, opacity });
    }
    ps.sort_by(|a, b| a.depth.total_cmp(&b.depth));
    let mut col = vec![[0.0f32; 3]; w * h];
    let mut tr = vec![1.0f32; w * h];
    for p in &ps {
        let (x0, x1) = (((p.px as i32) - p.radius).max(0), ((p.px as i32) + p.radius + 1).min(w as i32));
        let (y0, y1) = (((p.py as i32) - p.radius).max(0), ((p.py as i32) + p.radius + 1).min(h as i32));
        for yy in y0..y1 {
            for xx in x0..x1 {
                let k = yy as usize * w + xx as usize;
                if tr[k] < 1e-4 {
                    continue;
                }
                let (dx, dy) = (p.px - xx as f32, p.py - yy as f32);
                let power = -0.5 * (p.conic[0] * dx * dx + p.conic[2] * dy * dy) - p.conic[1] * dx * dy;
                if power > 0.0 {
                    continue;
                }
                // `REF_ELLIPSE3`: nothing past three deviations on the
                // footprint's axes — a box of 3σ by 3σ, as the shader's quad.
                if std::env::var("REF_ELLIPSE3").is_ok() {
                    let (a, b, c) = (p.cov2[0], p.cov2[1], p.cov2[2]);
                    let mid = 0.5 * (a + c);
                    let rad = (0.25 * (a - c) * (a - c) + b * b).sqrt();
                    let (l1, l2) = (mid + rad, (mid - rad).max(1e-4));
                    let e1 = if a >= c { [l1 - c, b] } else { [b, l1 - a] };
                    let n = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt();
                    let e1 = if n > 1e-12 { [e1[0] / n, e1[1] / n] } else { [1.0, 0.0] };
                    // Screen y is down here, up in the shader: flip dy.
                    let (u, v) = (dx * e1[0] - dy * e1[1], dx * -e1[1] - dy * e1[0]);
                    if u.abs() > 3.0 * l1.sqrt() || v.abs() > 3.0 * l2.sqrt() {
                        continue;
                    }
                }
                let alpha = (p.opacity * power.exp()).min(0.99);
                if alpha < 1.0 / 255.0 {
                    continue;
                }
                let next = tr[k] * (1.0 - alpha);
                if next < 1e-4 {
                    tr[k] = 0.0;
                    continue;
                }
                for c in 0..3 {
                    col[k][c] += p.color[c] * alpha * tr[k];
                }
                tr[k] = next;
            }
        }
    }
    if std::env::var("REF_LINEAR").is_ok() {
        for c in &mut col {
            *c = c.map(|v| {
                let v = v.clamp(0.0, 1.0);
                if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
            });
        }
    }
    col
}

/// How far the GPU's picture of a real capture is from the reference
/// rasteriser's, on an sRGB target (blending linear light) and a plain one
/// (blending stored values, as the reference does). Set `THREERS_SPLAT_FILE`.
#[test]
#[ignore]
fn compare_with_the_reference_rasteriser() {
    let path = std::env::var("THREERS_SPLAT_FILE").expect("THREERS_SPLAT_FILE");
    let mut s = GaussianSplats::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    // `REF_SH=<degree>` cuts the view-dependent colour, for both renderers.
    if let Some(d) = std::env::var("REF_SH").ok().and_then(|v| v.parse().ok()) {
        s.truncate_sh(d);
    }
    // `THREERS_SPLAT_SIZE=<w>x<h>`, 480x270 by default.
    let (w, h) = std::env::var("THREERS_SPLAT_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((480u32, 270u32));
    let mut d: Vec<f32> = s.positions.iter().map(|p| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()).collect();
    let mid = d.len() / 2;
    let r = *d.select_nth_unstable_by(mid, f32::total_cmp).1;
    let mut camera = PerspectiveCamera::new(50.0, w as f32 / h as f32, 0.01, 1000.0);
    camera.position = Vector3::new(0.3 * r, 0.2 * r, 1.2 * r);
    camera.look_at(Vector3::ZERO);
    let p = camera.position;
    println!("camera at {},{},{} looking at 0,0,0, fov 50", p.x, p.y, p.z);
    // `REF_PACK`: the reference with the higher bands quantised as the
    // material packs them (bytes over ±the splat's largest coefficient).
    let mut packed = s.clone();
    if std::env::var("REF_PACK").is_ok() {
        let n = sh_rest_coefficients(s.sh_degree) * 3;
        for splat in packed.sh.chunks_mut(n.max(1)) {
            let m = splat.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            for v in splat.iter_mut() {
                if m > 0.0 {
                    let q = ((*v / m) * 127.5 + 127.5).round().clamp(0.0, 255.0);
                    *v = (q - 127.5) * (m / 127.5);
                }
            }
        }
    }
    let reference = reference_render(&packed, &camera, w as usize, h as usize);
    let to_byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round();
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        let Ok(mut hr) = HeadlessRenderer::builder().size(w, h).color_format(format).build() else {
            return;
        };
        let mut scene = Scene::new();
        scene.background = Color::new(0.0, 0.0, 0.0);
        let (object, mut sorter) = splat_object(&s, &SplatOptions::default()).unwrap();
        let id = scene.add(object);
        sorter.update(&mut scene, id, &camera);
        let px = hr.render_to_rgba(&mut scene, &camera);
        let (mut se, mut sad) = (0.0f64, 0.0f64);
        for k in 0..(w * h) as usize {
            for c in 0..3 {
                let e = px[k * 4 + c] as f64 - to_byte(reference[k][c]) as f64;
                se += e * e;
                sad += e.abs();
            }
        }
        let n = (w * h * 3) as f64;
        let psnr = 10.0 * (255.0f64 * 255.0 / (se / n)).log10();
        println!("{format:?}: PSNR {psnr:.2} dB against the reference, mean |diff| {:.2}", sad / n);
        let mut img = Vec::with_capacity((w * h * 4) as usize);
        for v in &reference {
            img.extend(v.map(to_byte).map(|b| b as u8));
            img.push(255);
        }
        std::fs::create_dir_all("out").unwrap();
        std::fs::write("out/splats_reference.png", threers::utils::png::encode_png(w, h, &img)).ok();
        std::fs::write(format!("out/splats_gpu_{format:?}.png"), threers::utils::png::encode_png(w, h, &px)).ok();
    }
}

/// One real splat with strong view-dependent colour, drawn alone: its centre
/// pixel against `color_towards`. Set `THREERS_SPLAT_FILE`.
#[test]
#[ignore]
fn one_real_splat_against_the_cpu() {
    let path = std::env::var("THREERS_SPLAT_FILE").expect("THREERS_SPLAT_FILE");
    let all = GaussianSplats::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
    let n = sh_rest_coefficients(all.sh_degree) * 3;
    // Splats with large degree-1 bands, opaque and of moderate colour.
    let mut picks: Vec<usize> = (0..all.len())
        .filter(|&i| all.opacities[i] > 0.9 && all.colors[i].iter().all(|c| (0.3..0.7).contains(c)))
        .collect();
    picks.sort_by(|&a, &b| {
        let m = |i: usize| all.sh[i * n..i * n + 9].iter().map(|v| v.abs()).sum::<f32>();
        m(b).total_cmp(&m(a))
    });
    let Some(mut hr) = renderer() else { return };
    for &i in picks.iter().take(6) {
        let mut s = GaussianSplats { sh_degree: 1, ..Default::default() };
        // At the origin, big enough to read, keeping its colour data.
        s.push(Vector3::ZERO, [0.3; 3], [0.0, 0.0, 0.0, 1.0], 0.99, all.colors[i]);
        s.sh.copy_from_slice(&all.sh[i * n..i * n + 9]);
        for eye in [Vector3::new(0.0, 0.0, 5.0), Vector3::new(3.0, 2.0, 3.0), Vector3::new(-4.0, -1.0, 2.0)] {
            let img = render_splats(&mut hr, &s, None, eye);
            let want = s.color_towards(0, -eye).map(|v| v.clamp(0.0, 1.0));
            let p = px(&img, N / 2, N / 2);
            let got = [0, 1, 2].map(|c| p[c] as f32 / 255.0 / 0.99);
            println!("splat {i} eye {eye:?}: gpu {got:.3?} cpu {want:.3?} sh1 {:.2?}", &s.sh[..9]);
        }
    }
}
