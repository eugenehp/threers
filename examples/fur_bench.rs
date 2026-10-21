//! Fur and hair frame costs: the animator's CPU time and the GPU frame, for
//! shell fur and strand hair, still and skinned, in one 1280×720 view with a
//! shadow-casting sun.
//!
//! ```bash
//! cargo run --release --example fur_bench
//! ```

use std::time::Instant;
use threers::core::{Bone, Skeleton, SkinnedMesh};
use threers::fur::strands::{hair_object, StrandOptions};
use threers::fur::{fur_object, FurAnimator, FurGroom, FurMaterial};
use threers::lights::Light;
use threers::prelude::*;

fn base_scene() -> Scene {
    let mut scene = Scene::new();
    scene.add_light(AmbientLight::new(Color::WHITE, 0.5));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.0);
    sun.direction = Vector3::new(-0.4, -0.8, -0.4).normalize();
    sun.cast_shadow = std::env::var("SUN_SHADOW").map_or(true, |v| v != "0");
    sun.shadow.camera_size = 4.0;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 30.0;
    let mut l = Object3D::light(Light::Directional(sun));
    l.position = Vector3::new(4.0, 8.0, 4.0);
    scene.add(l);
    scene
}

fn coat() -> FurMaterial {
    FurMaterial {
        groom: FurGroom {
            length: 0.3,
            density: 120.0,
            gravity: Vector3::new(0.0, -0.4, 0.0),
            placement: if std::env::var("SURFACE").is_ok() {
                threers::fur::FurPlacement::Surface { cell: 0.01 }
            } else {
                threers::fur::FurPlacement::Uv
            },
            ..Default::default()
        },
        shells: std::env::var("SHELLS").ok().and_then(|v| v.parse().ok()).unwrap_or(32),
        shadow_height: std::env::var("FUR_SHADOW").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6),
        wind: if std::env::var("NOWIND").is_ok() {
            Vector3::ZERO
        } else {
            Vector3::new(3.0, 0.0, 1.0)
        },
        flutter: if std::env::var("NOWIND").is_ok() { 0.0 } else { 0.1 },
        ..Default::default()
    }
}

fn skinned_ball(scene: &mut Scene) -> (ObjectId, ObjectId) {
    let bone = scene.add(Object3D::group());
    let mut g = SphereGeometry::new(1.0, 96, 48);
    let count = g.get_attribute("position").unwrap().count();
    g.set_attribute("joint", BufferAttribute::new(vec![0.0; count * 4], 4));
    g.set_attribute(
        "weight",
        BufferAttribute::new((0..count).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect(), 4),
    );
    let body = scene.add(Object3D::skinned_mesh(SkinnedMesh::new(
        g,
        Material::Standard(StandardMaterial::new(Color::from_hex(0x553322))),
        Skeleton::new(vec![Bone {
            node: bone,
            inverse_bind: Matrix4::identity(),
        }]),
    )));
    (bone, body)
}

fn main() {
    let (w, h) = (1280u32, 720u32);
    let mut hr = HeadlessRenderer::builder().size(w, h).build().expect("a GPU adapter");
    if std::env::var("MSAA").is_ok() {
        hr.set_msaa(4);
    }
    let mut camera = PerspectiveCamera::new(40.0, w as f32 / h as f32, 0.1, 50.0);
    let dist: f32 = std::env::var("DIST").ok().and_then(|v| v.parse().ok()).unwrap_or(4.0);
    camera.position = Vector3::new(0.0, 0.5, dist);
    camera.look_at(Vector3::ZERO);
    let hair_options = StrandOptions {
        count: 30_000,
        segments: 12,
        width: 0.004,
        ..Default::default()
    };

    type Build = Box<dyn Fn(&mut Scene, &mut FurAnimator) -> Option<ObjectId>>;
    let cases: Vec<(&str, Build)> = vec![
        ("empty", Box::new(|_: &mut Scene, _: &mut FurAnimator| None)),
        ("sphere", Box::new(|s: &mut Scene, _: &mut FurAnimator| {
            s.add(Object3D::mesh(Mesh::new(
                SphereGeometry::new(1.0, 96, 48),
                Material::Standard(StandardMaterial::new(Color::WHITE)),
            )));
            None
        })),
        ("fur, still", Box::new(|s: &mut Scene, a: &mut FurAnimator| {
            let seg: usize = std::env::var("SEG").ok().and_then(|v| v.parse().ok()).unwrap_or(96);
            let id = s.add(fur_object(SphereGeometry::new(1.0, seg, seg / 2), &coat()));
            a.track(id, &coat());
            None
        })),
        ("fur, skinned", Box::new(|s: &mut Scene, a: &mut FurAnimator| {
            let (bone, body) = skinned_ball(s);
            a.grow_on_skinned(s, body, &coat());
            Some(bone)
        })),
        ("hair 30k, still", Box::new(move |s: &mut Scene, a: &mut FurAnimator| {
            let id = s.add(hair_object(&SphereGeometry::new(1.0, 64, 32), &coat(), &hair_options));
            a.track(id, &coat());
            None
        })),
        ("hair 30k, skinned", Box::new(move |s: &mut Scene, a: &mut FurAnimator| {
            let (bone, body) = skinned_ball(s);
            a.grow_hair_on_skinned(s, body, &coat(), &hair_options);
            Some(bone)
        })),
    ];
    let frames = 120;
    println!("{:<20} {:>12} {:>12} {:>10}", "case", "update ms", "median ms", "fps");
    let only = std::env::var("CASE").ok();
    for (name, build) in &cases {
        if only.as_deref().is_some_and(|o| !name.starts_with(o)) {
            continue;
        }
        let mut scene = base_scene();
        let mut animator = FurAnimator::new();
        let bone = build(&mut scene, &mut animator);
        // Warm up: pipelines, uploads.
        for f in 0..5 {
            animator.update(&mut scene, f as f32 / 60.0, 1.0 / 60.0);
            hr.render(&mut scene, &camera);
        }
        let _ = hr.device().poll(wgpu::PollType::wait_indefinitely());
        let (mut update, mut frame, mut submit) = (0.0f64, Vec::new(), 0.0f64);
        for f in 0..frames {
            let t = (f + 5) as f32 / 60.0;
            if let Some(b) = bone {
                scene.get_mut(b).unwrap().position.x = (t * 3.0).sin() * 0.3;
            } else {
                // Move the object itself so inertia has work to do.
            }
            let t0 = Instant::now();
            if std::env::var("NOANIM").is_err() {
                animator.update(&mut scene, t, 1.0 / 60.0);
            }
            let t1 = Instant::now();
            hr.render(&mut scene, &camera);
            let ts = Instant::now();
            let _ = hr.device().poll(wgpu::PollType::wait_indefinitely());
            let t2 = Instant::now();
            submit += (ts - t1).as_secs_f64() * 1e3;
            update += (t1 - t0).as_secs_f64() * 1e3;
            frame.push((t2 - t0).as_secs_f64() * 1e3);
        }
        // The median frame: a GPU shared with other work has outliers.
        frame.sort_by(f64::total_cmp);
        let (u, fr) = (update / frames as f64, frame[frame.len() / 2]);
        println!(
            "{name:<20} {u:>12.2} {fr:>12.2} {:>10.1}   (render call {:.2} ms)",
            1e3 / fr,
            submit / frames as f64
        );
    }
}
