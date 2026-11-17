//! Gaussian splats on the native Metal renderer, which translates the splat
//! shader to MSL itself: drawn against the wgpu backend, splat for splat. A
//! shader Metal could not translate would come out as the material's flat
//! fallback colour, nothing like the Gaussians.
//!
//! Skips without a Metal device or a wgpu adapter.

#![cfg(metal_backend)]

use threers::metal::MetalHeadlessRenderer;
use threers::prelude::*;
use threers::splats::{splat_object, GaussianSplats, SplatOptions};

const N: u32 = 128;

/// A small cloud of overlapping splats of every shape, at SH degree 2.
fn splats() -> GaussianSplats {
    let mut s = GaussianSplats { sh_degree: 2, ..Default::default() };
    let mut seed = 7u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32
    };
    for _ in 0..300 {
        let p = Vector3::new(rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0);
        let scale = [0.02 + rnd() * 0.15, 0.02 + rnd() * 0.15, 0.02 + rnd() * 0.05];
        let q = [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5];
        s.push(p, scale, q, 0.3 + rnd() * 0.7, [rnd(), rnd(), rnd()]);
    }
    let n = s.sh.len();
    s.sh = (0..n).map(|k| ((k as f32) * 0.71).sin() * 0.15).collect();
    s
}

#[test]
fn metal_draws_the_splats_wgpu_draws() {
    let Ok(mut metal) = MetalHeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let Ok(mut hr) = HeadlessRenderer::builder().size(N, N).build() else {
        eprintln!("skipping: no wgpu adapter");
        return;
    };
    let mut cam = PerspectiveCamera::new(50.0, 1.0, 0.1, 50.0);
    cam.position = Vector3::new(1.5, 1.0, 3.5);
    cam.look_at(Vector3::ZERO);
    // A turned, non-uniformly scaled object: the transform and the SH frame.
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let (mut object, mut sorter) = splat_object(&splats(), &SplatOptions::default()).unwrap();
    object.set_rotation_from_axis_angle(Vector3::new(0.0, 1.0, 0.0), 0.6);
    object.scale = Vector3::new(1.2, 0.8, 1.0);
    let id = scene.add(object);
    sorter.update(&mut scene, id, &cam);

    let m = metal.render_to_rgba(&mut scene, &cam).unwrap();
    let w = hr.render_to_rgba(&mut scene, &cam);
    let (mut lit, mut total_diff, mut worst) = (0u32, 0u64, 0u8);
    for (a, b) in m.as_chunks::<4>().0.iter().zip(w.as_chunks::<4>().0) {
        lit += (b[0] as u32 + b[1] as u32 + b[2] as u32 > 30) as u32;
        for ch in 0..3 {
            let d = a[ch].abs_diff(b[ch]);
            total_diff += d as u64;
            worst = worst.max(d);
        }
    }
    let mean = total_diff as f64 / (N * N * 3) as f64;
    eprintln!("lit {lit}, mean diff {mean:.3}, worst {worst}");
    assert!(lit > N * N / 20, "wgpu drew little: {lit} lit pixels");
    assert!(mean < 1.0 && worst <= 12, "Metal differs: mean {mean:.3}, worst {worst}");
}
