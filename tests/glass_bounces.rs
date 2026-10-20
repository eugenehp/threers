//! Rays through stacked glass: what a ray cast reports, and how far the path
//! tracer follows light through it.
//!
//! Three solid glass slabs stand one behind another in front of a bright,
//! unlit panel. A ray through all three crosses six surfaces — in and out of
//! each — and the raycaster has to report all six, nearest first. The path
//! tracer counts each crossing as a transmission bounce
//! (`RaytraceSettings::transmission_max_bounces`): five let a path through
//! two slabs and into the third, where it stops and the pixel goes dark; six
//! let it out the back to the background. On the CPU backend and the GPU
//! one.

use threers::core::Raycaster;
use threers::prelude::*;
use threers::raytrace::{BackgroundMode, RaytraceRenderer, RaytraceSettings};

const N: u32 = 24;

fn slabs() -> (Scene, Vec<threers::core::ObjectId>) {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    // What the light comes from: an unlit white panel behind the glass (a
    // path that has been through glass no longer sees `background`).
    let mut panel = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(20.0, 20.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    panel.position = Vector3::new(0.0, 0.0, -4.0);
    scene.add(panel);
    let mut ids = Vec::new();
    for k in 0..3 {
        let mut m = PhysicalMaterial::new(Color::WHITE)
            .with_roughness(0.0)
            .with_transmission(1.0, 1.0, 0.2);
        // Both faces: a ray cast must see the exits as well as the entries.
        m.side = 2;
        let mut slab = Object3D::mesh(Mesh::new(
            BoxGeometry::new(4.0, 4.0, 0.2),
            Material::Physical(m),
        ));
        slab.position = Vector3::new(0.0, 0.0, -(k as f32) * 0.6);
        ids.push(scene.add(slab));
    }
    (scene, ids)
}

fn camera() -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(10.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 6.0);
    c.look_at(Vector3::new(0.0, 0.0, -1.0));
    c
}

#[test]
fn a_ray_cast_reports_every_glass_surface_nearest_first() {
    let (mut scene, ids) = slabs();
    scene.update_world();
    // Off the diagonal each face's two triangles share, which a ray down
    // the middle would hit twice (as three.js's does).
    let ray = Raycaster::new(
        Vector3::new(0.13, 0.07, 6.0),
        Vector3::new(0.0, 0.0, -1.0),
        0.0,
        100.0,
    );
    let hits = ray.intersect_object_list(&scene.arena, &ids, false);
    let z: Vec<f32> = hits.iter().map(|h| h.point.z).collect();
    eprintln!("hits at z = {z:?}");
    let want = [0.1, -0.1, -0.5, -0.7, -1.1, -1.3];
    assert_eq!(
        hits.len(),
        6,
        "a ray through three slabs crosses six surfaces: {z:?}"
    );
    for (got, want) in z.iter().zip(want) {
        assert!(
            (got - want).abs() < 1e-4,
            "surfaces out of order or out of place: {z:?}"
        );
    }
    assert!(
        hits.windows(2).all(|w| w[0].distance <= w[1].distance),
        "not nearest first"
    );
}

/// The centre pixel's brightness through the three slabs, at a limit.
fn centre(r: &mut RaytraceRenderer, limit: u32) -> f32 {
    let (mut scene, _) = slabs();
    r.set_settings(RaytraceSettings {
        samples_per_pixel: 16,
        max_bounces: 32,
        min_bounces: 32,
        background: BackgroundMode::Color,
        clamp_indirect: 0.0,
        adaptive_threshold: 0.0,
        denoise: false,
        transmission_max_bounces: limit,
        ..Default::default()
    });
    let px = r.render_to_rgba(&mut scene, &camera());
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    (px[i] as f32 + px[i + 1] as f32 + px[i + 2] as f32) / 3.0
}

fn check(r: &mut RaytraceRenderer, backend: &str) {
    let five = centre(r, 5);
    let six = centre(r, 6);
    let free = centre(r, u32::MAX);
    eprintln!(
        "{backend}: through three slabs at limit 5: {five:.1}, 6: {six:.1}, unlimited: {free:.1}"
    );
    assert!(
        five < 30.0,
        "{backend}: five transmissions should stop inside the third slab ({five:.1})"
    );
    assert!(
        six > 200.0,
        "{backend}: six transmissions should reach the background ({six:.1})"
    );
    assert!(
        (six - free).abs() < 20.0,
        "{backend}: six should be enough ({six:.1} against {free:.1})"
    );
}

#[test]
fn transmission_bounces_limit_how_deep_light_goes_through_glass() {
    let mut cpu = RaytraceRenderer::new(N, N);
    check(&mut cpu, "cpu");
    match threers::raytrace::gpu::GpuBackend::headless() {
        Ok(backend) => {
            let mut gpu = RaytraceRenderer::with_backend(N, N, Box::new(backend));
            check(&mut gpu, "gpu");
        }
        Err(e) => eprintln!("skipping the GPU backend: {e:?}"),
    }
}
