//! Gaussian splats: a scene captured as splats — a `.ply` from the reference
//! trainer (plain or compressed), an `.spz` or a `.splat` — rendered from a few points of an orbit, with the
//! sort and frame timed. Without a file it makes a million splats of its own:
//! a shell of small splats round a dense, colourful core.
//!
//! ```bash
//! cargo run --release --features splats --example splats [scene.ply|scene.spz|scene.splat]
//! ```
//!
//! With `SPLATS_BENCH=<frames>` it then times that many frames of a slow
//! orbit (a sort each frame) and as many with the camera still (the GPU
//! alone), each rendered and waited for, and prints the medians.
//!
//! Writes `out/splats_<k>.png`. Trained scenes are usually upside down in
//! three.js's y-up frame (the trainer's cameras are y-down); the object is
//! turned over when a file is given, unless it is an `.spz`, which is y-up.

use std::time::Instant;
use threers::prelude::*;
use threers::splats::{splat_object, GaussianSplats, SplatOptions};

const W: u32 = 1280;
const H: u32 = 720;

fn generated(n: usize) -> GaussianSplats {
    let mut s = GaussianSplats::default();
    let mut seed = 1u32;
    let mut rnd = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / (1 << 24) as f32
    };
    for i in 0..n {
        // A random direction, then a radius: most on a thin shell, the rest
        // filling the core.
        let (u, v) = (rnd() * 2.0 - 1.0, rnd() * std::f32::consts::TAU);
        let d = Vector3::new((1.0 - u * u).sqrt() * v.cos(), u, (1.0 - u * u).sqrt() * v.sin());
        let shell = i % 3 != 0;
        let r = if shell { 1.0 + rnd() * 0.02 } else { rnd().cbrt() * 0.6 };
        let size = if shell { 0.004 + rnd() * 0.01 } else { 0.01 + rnd() * 0.03 };
        let color = if shell {
            let band = (d.y * 8.0).sin() * 0.5 + 0.5;
            [0.2 + 0.6 * band, 0.3, 0.9 - 0.5 * band]
        } else {
            [1.0, 0.5 + 0.5 * (r * 9.0).sin(), 0.2]
        };
        s.push(
            d * r,
            [size, size, size * 0.3],
            [rnd() - 0.5, rnd() - 0.5, rnd() - 0.5, rnd() - 0.5],
            if shell { 0.04 } else { 0.9 },
            color,
        );
    }
    s
}

fn main() {
    let path = std::env::args().nth(1);
    let t = Instant::now();
    let splats = match &path {
        Some(p) => {
            let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            GaussianSplats::from_bytes(&bytes).unwrap_or_else(|e| panic!("{p}: {e}"))
        }
        // `SPLATS_N=<count>` sizes the generated set.
        None => generated(std::env::var("SPLATS_N").ok().and_then(|v| v.parse().ok()).unwrap_or(1_000_000)),
    };
    println!(
        "{} splats, SH degree {}, in {:.0} ms",
        splats.len(),
        splats.sh_degree,
        t.elapsed().as_secs_f64() * 1e3
    );
    // Where to look from: close for a captured scene, all in view for an
    // object (see `GaussianSplats::framing`).
    let (centre, distance) = if path.is_some() {
        splats.framing()
    } else {
        (Vector3::ZERO, 2.6)
    };

    // Trainers write `y` down; `.spz` is converted to `y` up.
    let flip = path.as_ref().is_some_and(|p| !p.ends_with(".spz"));
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let (mut object, mut sorter) = splat_object(&splats, &SplatOptions::default()).unwrap();
    if flip {
        object.set_rotation_from_axis_angle(Vector3::new(1.0, 0.0, 0.0), std::f32::consts::PI);
    }
    let id = scene.add(object);
    // `SPLATS_SIZE=<w>x<h>` overrides the size, to tell vertex from fragment cost.
    let (w, h) = std::env::var("SPLATS_SIZE")
        .ok()
        .and_then(|v| {
            let (w, h) = v.split_once('x')?;
            Some((w.parse().ok()?, h.parse().ok()?))
        })
        .unwrap_or((W, H));
    let mut hr = HeadlessRenderer::builder().size(w, h).build().expect("a GPU adapter");
    let mut camera = PerspectiveCamera::new(50.0, w as f32 / h as f32, distance * 0.003, distance * 500.0);
    std::fs::create_dir_all("out").unwrap();
    for k in 0..4 {
        // From the +z side, where a capture's cameras look from (a trainer's
        // y-down PLY turned over, or an .spz), swinging a little either way.
        let a = (k as f32 - 1.5) * 0.25;
        let target = if flip {
            Vector3::new(centre.x, -centre.y, -centre.z)
        } else {
            centre
        };
        camera.position = target + Vector3::new(a.sin(), 0.15, a.cos()).normalize() * distance;
        camera.look_at(target);
        let t = Instant::now();
        sorter.update(&mut scene, id, &camera);
        let sort = t.elapsed();
        let t = Instant::now();
        let px = hr.render_to_rgba(&mut scene, &camera);
        let frame = t.elapsed();
        let name = format!("out/splats_{k}.png");
        std::fs::write(&name, encode_png(w, h, &px)).unwrap();
        println!(
            "{name}: sort {:.1} ms, frame {:.1} ms (with readback{})",
            sort.as_secs_f64() * 1e3,
            frame.as_secs_f64() * 1e3,
            if k == 0 { ", and the first upload" } else { "" }
        );
    }
    if let Some(frames) = std::env::var("SPLATS_BENCH").ok().and_then(|v| v.parse::<usize>().ok()) {
        let target = if flip { Vector3::new(centre.x, -centre.y, -centre.z) } else { centre };
        let stats = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            (v[v.len() / 2], v[v.len() * 9 / 10])
        };
        let (mut sorts, mut orbit, mut still) = (Vec::new(), Vec::new(), Vec::new());
        let mut submits = Vec::new();
        for f in 0..frames * 2 {
            // A slow orbit, a hundredth of a radian a frame, then standing still.
            let a = 0.3 + 0.01 * f.min(frames) as f32;
            camera.position = target + Vector3::new(a.sin(), 0.15, a.cos()).normalize() * distance;
            camera.look_at(target);
            let t = Instant::now();
            sorter.update(&mut scene, id, &camera);
            let sort = t.elapsed().as_secs_f64() * 1e3;
            let t = Instant::now();
            hr.render(&mut scene, &camera);
            let submit = t.elapsed().as_secs_f64() * 1e3;
            hr.wait_idle();
            let frame = t.elapsed().as_secs_f64() * 1e3;
            submits.push(submit);
            if f < frames {
                sorts.push(sort);
                orbit.push(frame);
            } else {
                still.push(frame);
            }
        }
        // The first orbit frame uploads everything; leave it out.
        let (s, o, st) = (stats(sorts[1..].to_vec()), stats(orbit[1..].to_vec()), stats(still));
        let sub = stats(submits);
        println!("bench: render() itself {:.2} ms (p90 {:.2}) of each frame", sub.0, sub.1);
        println!(
            "bench: sort {:.2} ms (p90 {:.2}); frame while orbiting {:.2} ms (p90 {:.2}); still {:.2} ms (p90 {:.2})",
            s.0, s.1, o.0, o.1, st.0, st.1
        );
    }
}
