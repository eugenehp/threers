//! Particles as drawn: the instanced quads face the camera, streaks stretch
//! along the motion, sphere grains are lit from the sun's side, alpha
//! particles sort from the eye, and additive ones add light.
//!
//! Each scene is one or two large particles placed by hand, so what is
//! measured is the drawing, not the simulation. Skipped when no GPU adapter
//! is available.

use threers::lights::Light;
use threers::particles::render::write_instances;
use threers::particles::{
    particles_object, Gradient, ParticleBlend, ParticleKind, ParticleLook, ParticleSystem,
};
use threers::prelude::*;
use threers::ObjectKind;

const N: u32 = 96;

fn renderer() -> Option<HeadlessRenderer> {
    match HeadlessRenderer::builder().size(N, N).build() {
        Ok(r) => Some(r),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

/// A system holding exactly these particles: (position, velocity, size,
/// colour, alpha).
fn placed(
    look: ParticleLook,
    particles: &[(Vector3, Vector3, f32, [f32; 3], f32)],
) -> ParticleSystem {
    let mut s = ParticleSystem::new(look);
    for &(p, v, size, _, _) in particles {
        let ps = &mut s.particles;
        ps.position.push(p);
        ps.velocity.push(v);
        ps.age.push(0.0);
        ps.lifetime.push(f32::INFINITY);
        ps.size.push(size);
        ps.drag.push(0.0);
        ps.rotation.push(0.0);
        ps.spin.push(0.0);
        ps.volume.push(0.0);
        ps.resting.push(false);
        ps.seed.push(0.5);
        ps.generation.push(0);
    }
    // One colour for all: these tests use one each, or the same.
    let [r, g, b] = particles[0].3;
    s.color_over_life = Gradient(vec![(0.0, [r, g, b, particles[0].4])]);
    s
}

/// `system` drawn alone over black from `eye` toward `at`; returns RGB as
/// floats in 0..1.
fn draw(
    hr: &mut HeadlessRenderer,
    system: &ParticleSystem,
    eye: Vector3,
    at: Vector3,
    sun: Option<Vector3>,
) -> Vec<[f32; 3]> {
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    if let Some(d) = sun {
        let mut l = DirectionalLight::new(Color::WHITE, 3.0);
        l.direction = d.normalize();
        scene.add(Object3D::light(Light::Directional(l)));
    }
    let id = scene.add(particles_object(system));
    if let Some(o) = scene.get_mut(id) {
        if let ObjectKind::InstancedMesh(m) = &mut o.kind {
            write_instances(system, m, Some(eye));
        }
    }
    let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.05, 100.0);
    camera.position = eye;
    camera.look_at(at);
    let px = hr.render_to_rgba(&mut scene, &camera);
    px.chunks(4)
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect()
}

/// The lit pixels' extent: (width, height) in pixels.
fn extent(px: &[[f32; 3]], threshold: f32) -> (u32, u32) {
    let (mut x0, mut x1, mut y0, mut y1) = (N, 0, N, 0);
    for y in 0..N {
        for x in 0..N {
            let p = px[(y * N + x) as usize];
            if p[0].max(p[1]).max(p[2]) > threshold {
                x0 = x0.min(x);
                x1 = x1.max(x);
                y0 = y0.min(y);
                y1 = y1.max(y);
            }
        }
    }
    if x0 > x1 {
        (0, 0)
    } else {
        (x1 - x0 + 1, y1 - y0 + 1)
    }
}

/// A billboard faces the camera: round from any direction.
#[test]
fn a_billboard_is_round_from_every_side() {
    let Some(mut hr) = renderer() else { return };
    let look = ParticleLook::new(
        ParticleKind::Billboard { softness: 0.0 },
        ParticleBlend::Opaque,
    );
    let s = placed(
        look,
        &[(Vector3::ZERO, Vector3::ZERO, 1.0, [1.0, 1.0, 1.0], 1.0)],
    );
    for eye in [
        Vector3::new(0.0, 0.0, 3.0),
        Vector3::new(3.0, 0.0, 0.0),
        Vector3::new(1.5, 2.4, 1.0),
        Vector3::new(-0.3, -2.9, 0.2),
    ] {
        let (w, h) = extent(&draw(&mut hr, &s, eye, Vector3::ZERO, None), 0.5);
        assert!(w > 20, "nothing drawn from {eye:?}");
        assert!(
            (w as i32 - h as i32).abs() <= 2,
            "from {eye:?} the disc is {w}×{h}"
        );
    }
}

/// A streak lies along the motion across the view, as long as the
/// distance covered in its shutter time.
#[test]
fn a_streak_stretches_along_the_motion() {
    let Some(mut hr) = renderer() else { return };
    let look = ParticleLook::new(
        ParticleKind::Streak {
            softness: 0.0,
            shutter: 0.1,
        },
        ParticleBlend::Opaque,
    );
    let eye = Vector3::new(0.0, 0.0, 4.0);
    let across = placed(
        look,
        &[(
            Vector3::ZERO,
            Vector3::new(12.0, 0.0, 0.0),
            0.2,
            [1.0, 1.0, 1.0],
            1.0,
        )],
    );
    let (w, h) = extent(&draw(&mut hr, &across, eye, Vector3::ZERO, None), 0.5);
    assert!(w > 4 * h, "a horizontal streak is {w}×{h}");
    let down = placed(
        look,
        &[(
            Vector3::ZERO,
            Vector3::new(0.0, -12.0, 0.0),
            0.2,
            [1.0, 1.0, 1.0],
            1.0,
        )],
    );
    let (w, h) = extent(&draw(&mut hr, &down, eye, Vector3::ZERO, None), 0.5);
    assert!(h > 4 * w, "a falling streak is {w}×{h}");
    // Straight at the camera, there is nothing across the view to stretch.
    let toward = placed(
        look,
        &[(
            Vector3::ZERO,
            Vector3::new(0.0, 0.0, 12.0),
            0.4,
            [1.0, 1.0, 1.0],
            1.0,
        )],
    );
    let (w, h) = extent(&draw(&mut hr, &toward, eye, Vector3::ZERO, None), 0.5);
    assert!(
        (w as i32 - h as i32).abs() <= 2,
        "head-on a streak is {w}×{h}"
    );
}

/// A grain is a lit sphere: its sun side bright, its far side dark.
#[test]
fn a_sphere_grain_is_lit_from_the_sun_side() {
    let Some(mut hr) = renderer() else { return };
    let look = ParticleLook {
        lit: true,
        ..ParticleLook::new(ParticleKind::Sphere, ParticleBlend::Opaque)
    };
    let s = placed(
        look,
        &[(Vector3::ZERO, Vector3::ZERO, 1.2, [0.8, 0.8, 0.8], 1.0)],
    );
    // Light from the right (travelling toward −x).
    let px = draw(
        &mut hr,
        &s,
        Vector3::new(0.0, 0.0, 3.0),
        Vector3::ZERO,
        Some(Vector3::new(-1.0, 0.0, 0.0)),
    );
    let row = N / 2;
    let at = |x: u32| px[(row * N + x) as usize][0];
    let (left, right) = (at(N / 2 - 14), at(N / 2 + 14));
    assert!(right > 0.3, "the sun side is dark: {right}");
    assert!(
        right > left + 0.25,
        "lit side {right} against shadow side {left}"
    );
}

/// Alpha particles draw back to front from the eye: the nearer one's
/// colour wins where they overlap, whichever was made first.
#[test]
fn alpha_particles_sort_back_to_front() {
    let Some(mut hr) = renderer() else { return };
    let look = ParticleLook::new(
        ParticleKind::Billboard { softness: 0.0 },
        ParticleBlend::Alpha,
    );
    let eye = Vector3::new(0.0, 0.0, 4.0);
    for near_first in [true, false] {
        let mut s = ParticleSystem::new(look);
        let order = if near_first {
            [1.0f32, -1.0]
        } else {
            [-1.0, 1.0]
        };
        for z in order {
            let ps = &mut s.particles;
            ps.position.push(Vector3::new(0.0, 0.0, z));
            ps.velocity.push(Vector3::ZERO);
            ps.age.push(0.0);
            ps.lifetime.push(f32::INFINITY);
            ps.size.push(1.0);
            ps.drag.push(0.0);
            ps.rotation.push(0.0);
            ps.spin.push(0.0);
            ps.volume.push(0.0);
            ps.resting.push(false);
            // The seed marks which is which: red near, blue far.
            ps.seed.push(if z > 0.0 { 1.0 } else { 0.0 });
            ps.generation.push(0);
        }
        s.color_over_life = Gradient(vec![(0.0, [1.0, 1.0, 1.0, 0.8])]);
        // Colour per particle by drawing: tint the near one red, the far one
        // blue, through the instance colours after writing.
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let id = scene.add(particles_object(&s));
        if let Some(o) = scene.get_mut(id) {
            if let ObjectKind::InstancedMesh(m) = &mut o.kind {
                write_instances(&s, m, Some(eye));
                for (k, t) in m.transforms.iter().enumerate() {
                    let near = t.elements[14] > 0.0;
                    m.colors[k] = if near {
                        Color::new(1.0, 0.0, 0.0)
                    } else {
                        Color::new(0.0, 0.0, 1.0)
                    };
                }
            }
        }
        let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.05, 100.0);
        camera.position = eye;
        camera.look_at(Vector3::ZERO);
        let px = hr.render_to_rgba(&mut scene, &camera);
        let c = ((N / 2 * N + N / 2) * 4) as usize;
        let (r, b) = (px[c], px[c + 2]);
        assert!(
            r > b * 2,
            "near particle made {}: centre red {r}, blue {b}",
            if near_first { "first" } else { "last" }
        );
    }
}

/// Additive particles add their light: two overlapping are brighter than
/// one.
#[test]
fn additive_particles_add_light() {
    let Some(mut hr) = renderer() else { return };
    let look = ParticleLook::new(
        ParticleKind::Billboard { softness: 0.0 },
        ParticleBlend::Additive,
    );
    let one = placed(
        look,
        &[(Vector3::ZERO, Vector3::ZERO, 1.0, [0.3, 0.15, 0.05], 1.0)],
    );
    let two = placed(
        look,
        &[
            (Vector3::ZERO, Vector3::ZERO, 1.0, [0.3, 0.15, 0.05], 1.0),
            (
                Vector3::new(0.0, 0.0, -0.5),
                Vector3::ZERO,
                1.0,
                [0.3, 0.15, 0.05],
                1.0,
            ),
        ],
    );
    let eye = Vector3::new(0.0, 0.0, 4.0);
    let centre = |px: &[[f32; 3]]| px[(N / 2 * N + N / 2) as usize][0];
    let (a, b) = (
        centre(&draw(&mut hr, &one, eye, Vector3::ZERO, None)),
        centre(&draw(&mut hr, &two, eye, Vector3::ZERO, None)),
    );
    assert!(a > 0.1, "one additive particle drew {a}");
    assert!(b > a * 1.2, "two drew {b}, one {a}");
}

/// Particles that cast shadows darken the ground where their shadow falls;
/// with casting off, the same ground is lit.
#[test]
fn particles_cast_shadows() {
    use threers::particles::ParticleEffect;
    let Some(mut hr) = renderer() else { return };
    let brightness = |casts: bool, hr: &mut HeadlessRenderer| {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
        sun.cast_shadow = true;
        sun.shadow.camera_size = 6.0;
        let mut l = Object3D::light(Light::Directional(sun));
        // From +x at 45°: a cloud at height 2 shadows the ground at x = −2.
        l.position = Vector3::new(5.0, 5.0, 0.0);
        scene.add(l);
        let mut floor = Object3D::mesh(Mesh::new(
            PlaneGeometry::new(20.0, 20.0),
            Material::Standard(StandardMaterial::new(Color::WHITE)),
        ));
        floor.rotate_x(-std::f32::consts::FRAC_PI_2);
        floor.receive_shadow = true;
        scene.add(floor);
        let look = ParticleLook {
            casts_shadow: casts,
            ..ParticleLook::new(
                ParticleKind::Billboard { softness: 0.0 },
                ParticleBlend::Alpha,
            )
        };
        let cloud: Vec<_> = (0..64)
            .map(|k| {
                let (a, b) = ((k % 8) as f32 * 0.12 - 0.42, (k / 8) as f32 * 0.12 - 0.42);
                (
                    Vector3::new(a, 2.0, b),
                    Vector3::ZERO,
                    0.4,
                    [1.0, 1.0, 1.0],
                    1.0,
                )
            })
            .collect();
        let fx = ParticleEffect::add_to(&mut scene, placed(look, &cloud));
        scene.update_world();
        fx.sync(&mut scene, None);
        // Looking straight down at x = −2, the cloud itself out of view.
        let mut camera = PerspectiveCamera::new(20.0, 1.0, 0.05, 100.0);
        camera.position = Vector3::new(-2.0, 1.0, 0.0);
        camera.look_at(Vector3::new(-2.0, 0.0, 0.001));
        let px = hr.render_to_rgba(&mut scene, &camera);
        let c = ((N / 2 * N + N / 2) * 4) as usize;
        px[c] as f32
    };
    let (lit, shaded) = (brightness(false, &mut hr), brightness(true, &mut hr));
    assert!(lit > 100.0, "the ground is dark without a caster: {lit}");
    assert!(
        shaded < lit * 0.6,
        "the cloud's shadow: {shaded} against {lit} lit"
    );
}

/// GPU weather: flakes placed by the vertex shader appear in their box, fall
/// as time passes, and none are drawn outside the box.
#[test]
fn weather_falls_inside_its_box() {
    use threers::particles::Weather;
    let Some(mut hr) = renderer() else { return };
    let mut w = Weather::snow(400);
    w.half_width = 1.0;
    w.half_depth = 0.2;
    w.height = 2.0;
    w.floor = -1.0;
    w.size = [0.06, 0.06];
    w.sway = 0.0;
    w.wind = Vector3::ZERO;
    w.look.lit = false;
    w.look.flake = false;
    w.look.glint = 0.0;
    w.look.scattering = 0.0;
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let id = scene.add(w.object());
    let mut camera = PerspectiveCamera::new(60.0, 1.0, 0.05, 100.0);
    camera.position = Vector3::new(0.0, 0.0, 4.0);
    camera.look_at(Vector3::ZERO);
    let mut frame = |t: f32, scene: &mut Scene| {
        w.update(scene, id, t, Vector3::ZERO);
        let px = hr.render_to_rgba(scene, &camera);
        px.chunks(4)
            .map(|p| p[0] as f32 / 255.0)
            .collect::<Vec<f32>>()
    };
    let a = frame(1.0, &mut scene);
    let b = frame(1.4, &mut scene);
    let lit = |px: &[f32]| px.iter().filter(|&&v| v > 0.3).count();
    assert!(lit(&a) > 100, "only {} pixels of snow", lit(&a));
    let moved = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (*x - *y).abs() > 0.3)
        .count();
    assert!(moved > lit(&a) / 2, "{moved} pixels changed between frames");
    // The box spans x ±1, y −1..1 at z ≈ 0: about ±0.43 of the view at
    // 4 m with a 60° field, so columns and rows past ±0.5 stay black.
    let tan = (30f32).to_radians().tan();
    let lim = (1.25 / (4.2 * tan) * 0.5 + 0.5) * N as f32;
    for y in 0..N {
        for x in 0..N {
            let v = a[(y * N + x) as usize];
            let (fx, fy) = (x as f32, y as f32);
            let outside = fx > lim || fx < N as f32 - lim || fy > lim || fy < N as f32 - lim;
            assert!(
                !(outside && v > 0.05),
                "snow drawn outside the box at ({x}, {y})"
            );
        }
    }
}

/// Particles receive shadows as little volumes: one under a slab is dark,
/// one in the open lit, and one with the slab's edge across it in between —
/// shaded by how much of it the light reaches, not cut by a hard line.
#[test]
fn particles_receive_shadows_by_how_much_is_covered() {
    let Some(mut hr) = renderer() else { return };
    let mut brightness = |slab_x: Option<f32>| {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
        sun.cast_shadow = true;
        sun.shadow.camera_size = 6.0;
        let mut l = Object3D::light(Light::Directional(sun));
        l.position = Vector3::new(0.0, 6.0, 0.001);
        scene.add(l);
        if let Some(x) = slab_x {
            // A slab overhead, its +x edge at `x`.
            let mut slab = Object3D::mesh(Mesh::new(
                BoxGeometry::new(4.0, 0.1, 4.0),
                Material::Standard(StandardMaterial::new(Color::WHITE)),
            ));
            slab.position = Vector3::new(x - 2.0, 2.5, 0.0);
            slab.cast_shadow = true;
            scene.add(slab);
        }
        let look = ParticleLook {
            lit: true,
            ..ParticleLook::new(
                ParticleKind::Billboard { softness: 0.0 },
                ParticleBlend::Alpha,
            )
        };
        let system = placed(
            look,
            &[(Vector3::ZERO, Vector3::ZERO, 1.0, [1.0, 1.0, 1.0], 1.0)],
        );
        let id = scene.add(particles_object(&system));
        if let Some(o) = scene.get_mut(id) {
            o.receive_shadow = true;
            if let ObjectKind::InstancedMesh(m) = &mut o.kind {
                write_instances(&system, m, None);
            }
        }
        let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.05, 100.0);
        camera.position = Vector3::new(0.0, 0.0, 4.0);
        camera.look_at(Vector3::ZERO);
        let px = hr.render_to_rgba(&mut scene, &camera);
        px[((N / 2 * N + N / 2) * 4) as usize] as f32
    };
    let open = brightness(None);
    let under = brightness(Some(3.0));
    let half = brightness(Some(0.0));
    assert!(open > 60.0, "a particle in the sun is dark: {open}");
    assert!(
        under < open * 0.3,
        "under the slab: {under} against {open} in the open"
    );
    assert!(
        half > under + (open - under) * 0.2 && half < open - (open - under) * 0.2,
        "half covered: {half}, between {under} and {open}"
    );
}

/// Smoke as a medium: a column lit from straight above is darker at the
/// bottom, under the rest of it, than at the top; drawn as plain discs
/// (no extinction) the same column is no darker at the bottom.
#[test]
fn a_column_of_smoke_shades_itself_from_the_sun() {
    use threers::particles::ParticleEffect;
    let Some(mut hr) = renderer() else { return };
    let ends = |extinction: f32, hr: &mut HeadlessRenderer| {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
        sun.direction = Vector3::new(0.0, -1.0, 0.0);
        scene.add(Object3D::light(Light::Directional(sun)));
        let look = ParticleLook {
            lit: true,
            extinction,
            ..ParticleLook::new(
                ParticleKind::Billboard { softness: 1.0 },
                ParticleBlend::Alpha,
            )
        };
        // A column 3 m tall, 0.6 m across: a particle every 5 cm.
        let column: Vec<_> = (0..60)
            .flat_map(|k| {
                let y = -1.5 + k as f32 * 0.05;
                [(-0.1f32, -0.1f32), (0.1, 0.1), (0.1, -0.1), (-0.1, 0.1)].map(|(x, z)| {
                    (
                        Vector3::new(x, y, z),
                        Vector3::ZERO,
                        0.5,
                        [1.0, 1.0, 1.0],
                        0.6,
                    )
                })
            })
            .collect();
        let fx = ParticleEffect::add_to(&mut scene, placed(look, &column));
        scene.update_world();
        fx.sync(&mut scene, Some(Vector3::new(0.0, 0.0, 6.0)));
        let mut camera = PerspectiveCamera::new(40.0, 1.0, 0.05, 100.0);
        camera.position = Vector3::new(0.0, 0.0, 6.0);
        camera.look_at(Vector3::ZERO);
        let px = hr.render_to_rgba(&mut scene, &camera);
        // Brightness near the top and near the bottom of the column, in
        // the middle of it.
        let at = |y: u32| {
            (N / 2 - 2..N / 2 + 2)
                .map(|x| px[((y * N + x) * 4) as usize] as f32)
                .sum::<f32>()
                / 4.0
        };
        (at(N / 2 - 25), at(N / 2 + 25))
    };
    let (top, bottom) = ends(0.6, &mut hr);
    let (plain_top, plain_bottom) = ends(0.0, &mut hr);
    assert!(top > 40.0, "the top of the column is dark: {top}");
    assert!(bottom < top * 0.6, "bottom {bottom} against top {top}");
    // The darkening is the medium's: as discs, the bottom is no darker.
    assert!(
        plain_bottom > plain_top * 0.85,
        "as discs: bottom {plain_bottom} against top {plain_top}"
    );
}

/// A flat sand surface drawn alone, lit by a sun from `light` (azimuth,
/// elevation in degrees) and seen from `eye` (the same), its wind along +x:
/// the brightness at its centre.
fn sand_seen(
    hr: &mut HeadlessRenderer,
    look: threers::particles::surface::SurfaceLook,
    light: (f32, f32),
    eye: (f32, f32),
) -> f32 {
    let dir = |(az, el): (f32, f32)| {
        let (az, el) = (az.to_radians(), el.to_radians());
        Vector3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos())
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.direction = dir(light) * -1.0;
    scene.add(Object3D::light(Light::Directional(sun)));
    let mut plane = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), look.material()));
    plane.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(plane);
    let mut camera = PerspectiveCamera::new(10.0, 1.0, 0.05, 200.0);
    camera.position = dir(eye) * 30.0;
    camera.look_at(Vector3::ZERO);
    let px = hr.render_to_rgba(&mut scene, &camera);
    let c = ((N / 2 * N + N / 2) * 4) as usize;
    (px[c] as f32 + px[c + 1] as f32 + px[c + 2] as f32) / 3.0
}

/// Wind-rippled sand's sheen is anisotropic: off the mirror direction it
/// falls off slowly where the plane of the light and the eye runs along the
/// wind (rough that way, across the ripples) and fast where it runs across
/// the wind (smooth along the crests).
#[test]
fn sand_sheen_is_stretched_along_the_wind() {
    use threers::particles::surface::SurfaceLook;
    let Some(mut hr) = renderer() else { return };
    let mut look = SurfaceLook::sand(Color::new(0.05, 0.04, 0.03));
    look.glint = 0.0;
    look.grain = 0.0;
    look.ripple_strength = 0.0;
    // Lit at 20°, seen from 60° on the far side: the half vector 20° off
    // the normal, along the wind or across it.
    let along = sand_seen(&mut hr, look, (180.0, 20.0), (0.0, 60.0));
    let across = sand_seen(&mut hr, look, (270.0, 20.0), (90.0, 60.0));
    assert!(along > 20.0, "no sheen along the wind: {along}");
    assert!(
        along > across * 1.5,
        "along the wind {along}, across {across}"
    );
}

/// Sand is brighter seen down-sun, with the light behind the eye, than from
/// the side at the same angles — the opposition surge of a granular
/// surface, which Lambert's law has no way to give.
#[test]
fn sand_is_brighter_seen_down_sun() {
    use threers::particles::surface::SurfaceLook;
    let Some(mut hr) = renderer() else { return };
    let mut look = SurfaceLook::sand(Color::new(0.5, 0.4, 0.3));
    look.glint = 0.0;
    look.grain = 0.0;
    look.ripple_strength = 0.0;
    look.roughness = [0.03, 0.03];
    let down_sun = sand_seen(&mut hr, look, (0.0, 40.0), (2.0, 41.0));
    let aside = sand_seen(&mut hr, look, (0.0, 40.0), (90.0, 41.0));
    assert!(aside > 20.0, "dark from the side: {aside}");
    assert!(
        down_sun > aside * 1.2,
        "down-sun {down_sun}, from the side {aside}"
    );
}

/// Rain seen with the sun behind the eye makes a rainbow where a sphere of
/// water puts it: the primary's red outside its blue, peaking near 42° from
/// the point opposite the sun; a dark band beyond; the secondary near 51°
/// with its colours the other way round.
#[test]
fn rain_makes_a_rainbow_at_42_degrees() {
    const S: u32 = 240;
    let Ok(mut hr) = HeadlessRenderer::builder().size(S, S).build() else {
        return;
    };
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    // Shining past the eye toward −z: the antisolar point straight ahead.
    sun.direction = Vector3::new(0.0, 0.0, -1.0);
    scene.add(Object3D::light(Light::Directional(sun)));
    let look = ParticleLook {
        lit: true,
        rainbow: 1.0,
        ..ParticleLook::new(
            ParticleKind::Streak {
                softness: 0.8,
                shutter: 0.02,
            },
            ParticleBlend::Alpha,
        )
    };
    // A wall of drops 10 m ahead, black but for what they send back.
    let mut drops = Vec::new();
    for j in -150..=150 {
        for i in -150..=150 {
            let p = Vector3::new(i as f32 * 0.12, j as f32 * 0.12, -10.0);
            drops.push((p, Vector3::ZERO, 0.16, [0.0, 0.0, 0.0], 1.0));
        }
    }
    let system = placed(look, &drops);
    let id = scene.add(particles_object(&system));
    if let Some(o) = scene.get_mut(id) {
        o.receive_shadow = false;
        if let ObjectKind::InstancedMesh(m) = &mut o.kind {
            write_instances(&system, m, Some(Vector3::ZERO));
        }
    }
    let mut camera = PerspectiveCamera::new(120.0, 1.0, 0.05, 100.0);
    camera.position = Vector3::ZERO;
    camera.look_at(Vector3::new(0.0, 0.0, -1.0));
    let px = hr.render_to_rgba(&mut scene, &camera);
    // Each colour by angle from the centre, a degree a bin.
    let f = (S as f32 / 2.0) / 60f32.to_radians().tan();
    let mut bins = vec![[0.0f64; 4]; 70];
    for y in 0..S {
        for x in 0..S {
            let (dx, dy) = (
                x as f32 + 0.5 - S as f32 / 2.0,
                y as f32 + 0.5 - S as f32 / 2.0,
            );
            let a = (dx.hypot(dy) / f).atan().to_degrees() as usize;
            if a < bins.len() {
                let p = &px[((y * S + x) * 4) as usize..][..3];
                for c in 0..3 {
                    bins[a][c] += p[c] as f64;
                }
                bins[a][3] += 1.0;
            }
        }
    }
    let mean = |a: usize, c: usize| bins[a][c] / bins[a][3].max(1.0);
    let peak = |c: usize, lo: usize, hi: usize| {
        (lo..hi)
            .max_by(|&a, &b| mean(a, c).total_cmp(&mean(b, c)))
            .unwrap()
    };
    let (red, blue) = (peak(0, 35, 47), peak(2, 35, 47));
    assert!(
        (41..=43).contains(&red),
        "the primary's red peaks at {red}°"
    );
    assert!(blue < red, "the primary's blue at {blue}°, red at {red}°");
    let (red2, blue2) = (peak(0, 48, 58), peak(2, 48, 58));
    assert!(
        red2 < blue2,
        "the secondary's red at {red2}°, blue at {blue2}°"
    );
    let band = mean(46, 0) + mean(47, 1);
    assert!(
        band < (mean(red, 0) + mean(red, 1)) * 0.3,
        "Alexander's band is bright: {band}"
    );
}

/// Plate crystals falling level glint in a column under a low sun (the
/// sub-sun: the sun mirrored in their faces, as far below the horizon as it
/// is above); flakes with facets every way at random make none.
#[test]
fn level_plates_glint_in_a_column() {
    const S: u32 = 160;
    let Ok(mut hr) = HeadlessRenderer::builder().size(S, S).build() else {
        return;
    };
    let mut share = |plates: f32| {
        let mut scene = Scene::new();
        scene.background = Color::BLACK;
        let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
        let el = 8f32.to_radians();
        sun.direction = Vector3::new(0.0, -el.sin(), el.cos());
        scene.add(Object3D::light(Light::Directional(sun)));
        let look = ParticleLook {
            lit: true,
            glint: 1.0,
            plates,
            ..ParticleLook::new(
                ParticleKind::Billboard { softness: 0.5 },
                ParticleBlend::Alpha,
            )
        };
        let mut flakes = Vec::new();
        for j in -60..=60 {
            for i in -60..=60 {
                let p = Vector3::new(i as f32 * 0.09, j as f32 * 0.09, -10.0);
                flakes.push((p, Vector3::ZERO, 0.08, [0.0, 0.0, 0.0], 1.0));
            }
        }
        let mut system = placed(look, &flakes);
        // Each its own seed and turn.
        for (k, s) in system.particles.seed.iter_mut().enumerate() {
            *s = ((k as f32 * 0.618_034) % 1.0 + 1.0) % 1.0;
        }
        for (k, r) in system.particles.rotation.iter_mut().enumerate() {
            *r = (k as f32 * 2.399_963) % std::f32::consts::TAU;
        }
        let id = scene.add(particles_object(&system));
        if let Some(o) = scene.get_mut(id) {
            o.receive_shadow = false;
            if let ObjectKind::InstancedMesh(m) = &mut o.kind {
                write_instances(&system, m, Some(Vector3::ZERO));
            }
        }
        let mut camera = PerspectiveCamera::new(60.0, 1.0, 0.05, 100.0);
        camera.position = Vector3::ZERO;
        camera.look_at(Vector3::new(0.0, 0.0, -1.0));
        let px = hr.render_to_rgba(&mut scene, &camera);
        // The share of the glints' light within a tenth of the view of the
        // sun's azimuth (the middle column).
        let (mut middle, mut all) = (0.0f64, 0.0f64);
        for y in 0..S {
            for x in 0..S {
                let v = px[((y * S + x) * 4) as usize] as f64;
                all += v;
                if (x as i32 - S as i32 / 2).abs() < (S / 20) as i32 {
                    middle += v;
                }
            }
        }
        (middle / all.max(1.0), all)
    };
    let (level, level_all) = share(0.12);
    let (_, random_all) = share(0.0);
    assert!(level_all > 1000.0, "level plates: no glints");
    assert!(
        level > 0.5,
        "level plates: {level:.2} of the glints in the column"
    );
    // Facets every way at random make no sub-sun.
    assert!(
        random_all < level_all * 0.2,
        "random facets {random_all}, level plates {level_all}"
    );
}
