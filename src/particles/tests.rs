use super::*;

/// One particle dropped from rest at `height`, with this terminal speed.
fn dropped(height: f32, terminal: f32) -> ParticleSystem {
    let mut e = Emitter::new(EmitterShape::Point, Vector3::new(0.0, height, 0.0));
    e.bursts = vec![(0.0, 1)];
    e.lifetime = Range::fixed(f32::INFINITY);
    e.terminal_speed = Range::fixed(terminal);
    ParticleSystem::new(ParticleLook::new(
        ParticleKind::Sphere,
        ParticleBlend::Opaque,
    ))
    .with_emitter(e)
}

fn run(s: &mut ParticleSystem, seconds: f32, dt: f32) {
    let steps = (seconds / dt).round() as usize;
    for _ in 0..steps {
        s.step(dt, None);
    }
}

/// With no air, a dropped particle falls as `g t` and `g t² / 2`.
#[test]
fn without_drag_it_falls_ballistically() {
    let mut s = dropped(100.0, 0.0);
    run(&mut s, 2.0, 1.0 / 120.0);
    let v = s.particles.velocity[0].y;
    let y = s.particles.position[0].y;
    let t = s.particles.age[0];
    assert!((v + 9.81 * t).abs() < 1e-3 * 9.81 * t, "v {v} at {t} s");
    let expect = 100.0 - 0.5 * 9.81 * t * t;
    assert!(
        (y - expect).abs() < 0.01 * (100.0 - expect),
        "y {y}, expected {expect}"
    );
}

/// Drag settles a falling particle at exactly its terminal speed, and stays
/// stable at a step far longer than its response time.
#[test]
fn drag_settles_at_the_terminal_speed_at_any_step() {
    for (vt, dt) in [(1.0f32, 1.0 / 60.0), (9.0, 1.0 / 60.0), (0.2, 0.5)] {
        let mut s = dropped(10_000.0, vt);
        // Ten kilometres up is outside the default bounds.
        s.bounds = f32::INFINITY;
        run(&mut s, 30.0, dt);
        let v = s.particles.velocity[0].length();
        assert!(
            (v - vt).abs() < 0.01 * vt,
            "terminal {vt} at step {dt}: settled at {v}"
        );
    }
}

/// The turbulence is the curl of a potential: no divergence anywhere, so it
/// neither gathers particles into clumps nor empties holes.
#[test]
fn turbulence_is_divergence_free() {
    let mut s = dropped(0.0, 1.0);
    s.forces.turbulence = Turbulence {
        strength: 1.0,
        scale: 0.7,
        churn: 0.5,
        anisotropy: 0.0,
    };
    s.forces.wind = Vector3::new(2.0, 0.0, 1.0);
    let mut rng = Rng::new(3);
    let h = 1e-3;
    let mut worst = 0.0f32;
    let mut typical = 0.0f32;
    for _ in 0..200 {
        let p = rng.unit_vector() * 3.0;
        let t = rng.next() * 10.0;
        let u = |q: Vector3| s.turbulence_at(q, t);
        let ex = Vector3::new(h, 0.0, 0.0);
        let ey = Vector3::new(0.0, h, 0.0);
        let ez = Vector3::new(0.0, 0.0, h);
        let dudx = (u(p + ex).x - u(p - ex).x) / (2.0 * h);
        let dvdy = (u(p + ey).y - u(p - ey).y) / (2.0 * h);
        let dwdz = (u(p + ez).z - u(p - ez).z) / (2.0 * h);
        worst = worst.max((dudx + dvdy + dwdz).abs());
        typical = typical.max(dudx.abs().max(dvdy.abs()).max(dwdz.abs()));
    }
    assert!(typical > 0.5, "the field barely varies ({typical})");
    assert!(
        worst < 0.01 * typical,
        "divergence {worst} against gradients of {typical}"
    );
}

/// In a side wind, a light flake drifts far further on its way down than a
/// heavy grain does: the terminal speed is what sets how much the air
/// carries a particle.
#[test]
fn light_particles_drift_with_the_wind_heavy_ones_do_not() {
    let drift = |vt: f32| {
        let mut s = dropped(5.0, vt);
        s.forces.wind = Vector3::new(3.0, 0.0, 0.0);
        s.colliders.push(Collider::Ground { height: 0.0 });
        s.response = Response::Stick;
        run(&mut s, 30.0, 1.0 / 60.0);
        s.particles.position[0].x
    };
    let (flake, grain) = (drift(1.0), drift(5.0));
    // Time to fall 5 m at terminal speed, times the wind it takes on.
    assert!((flake - 15.0).abs() < 1.5, "flake drifted {flake} m");
    assert!(
        grain < flake * 0.35,
        "grain drifted {grain} m against the flake's {flake}"
    );
}

/// A rate emits that many a second, in fractions carried across steps, and
/// a burst its count at its time.
#[test]
fn emission_follows_the_rate_and_the_bursts() {
    let mut e = Emitter::new(EmitterShape::Point, Vector3::ZERO);
    e.rate = 37.0;
    e.bursts = vec![(0.5, 100), (1.5, 25)];
    e.lifetime = Range::fixed(f32::INFINITY);
    let mut s = ParticleSystem::new(ParticleLook::new(
        ParticleKind::Sphere,
        ParticleBlend::Opaque,
    ))
    .with_emitter(e);
    s.forces.gravity = Vector3::ZERO;
    run(&mut s, 2.0, 1.0 / 60.0);
    let n = s.len() as i64;
    assert!((n - (74 + 125)).abs() <= 1, "{n} particles after 2 s");
}

/// A bounce returns `e²` of the drop height.
#[test]
fn a_bounce_returns_restitution_squared_of_the_height() {
    let mut s = dropped(2.0, 0.0);
    s.colliders.push(Collider::Ground { height: 0.0 });
    s.response = Response::Bounce {
        restitution: 0.6,
        friction: 0.0,
    };
    let mut peak = 0.0f32;
    let mut was_down = false;
    for _ in 0..(3.0 * 1000.0) as usize {
        s.step(1.0 / 1000.0, None);
        let y = s.particles.position[0].y;
        if y < 0.01 {
            was_down = true;
        }
        if was_down {
            peak = peak.max(y);
        }
        if was_down && s.particles.velocity[0].y < 0.0 && peak > 0.1 {
            break;
        }
    }
    let expect = 2.0 * 0.36;
    assert!(
        (peak - expect).abs() < 0.03 * expect,
        "rebounded to {peak}, expected {expect}"
    );
}

/// Everything deposited is in the heightfield, before and after it slumps.
#[test]
fn deposits_and_slumping_conserve_volume() {
    let mut field = Heightfield::new([0.0, 0.0], [4.0, 4.0], 64, 0.0);
    let mut s = super::presets::sand_stream(Vector3::new(0.1, 2.0, -0.2), 600.0, 0.002);
    let ground = field.clone();
    let mut sent = 0.0;
    for _ in 0..240 {
        s.step(1.0 / 60.0, Some(&ground));
        for l in s.take_landed() {
            sent += l.volume;
            field.deposit(l.position.x, l.position.z, l.volume);
        }
    }
    assert!(sent > 0.005, "only {sent} m³ landed");
    let before = field.volume();
    assert!(
        (before - sent).abs() < 1e-4 * sent,
        "{before} m³ held of {sent} m³ sent"
    );
    field.relax(34f32.to_radians(), 400);
    let after = field.volume();
    assert!(
        (after - sent).abs() < 1e-3 * sent,
        "{after} m³ after slumping, of {sent}"
    );
}

/// Sand poured on one spot ends as a cone at its angle of repose: no slope
/// steeper, and the flanks right at it.
#[test]
fn a_poured_pile_stands_at_the_angle_of_repose() {
    let mut field = Heightfield::new([0.0, 0.0], [3.0, 3.0], 96, 0.0);
    let angle = 34f32.to_radians();
    for _ in 0..400 {
        field.deposit(0.0, 0.0, 2.0e-4);
        field.relax(angle, 6);
    }
    field.relax(angle, 3000);
    let steepest = field.steepest(1e-4);
    assert!(
        steepest <= angle + 2f32.to_radians(),
        "a slope of {}°",
        steepest.to_degrees()
    );
    // The flank: height against distance from the peak along a row.
    let mid = field.nz / 2;
    let peak_i = (0..field.nx)
        .max_by(|&a, &b| field.surface(a, mid).total_cmp(&field.surface(b, mid)))
        .unwrap();
    let peak = field.surface(peak_i, mid);
    let mut i = peak_i;
    while i + 1 < field.nx && field.surface(i + 1, mid) > peak * 0.25 {
        i += 1;
    }
    let run = (i - peak_i) as f32 * field.cell;
    let rise = peak - field.surface(i, mid);
    let flank = (rise / run).atan();
    assert!(
        (flank - angle).abs() < 3f32.to_radians(),
        "the flank stands at {}°, not 34°",
        flank.to_degrees()
    );
}

/// An isolated mound in Werner's model keeps its sand and moves downwind.
#[test]
fn a_dune_migrates_downwind_and_keeps_its_sand() {
    let field = Heightfield::new([0.0, 0.0], [12.8, 6.4], 128, 0.0);
    let mut bed = sand::SandBed::new(field);
    for j in 0..bed.field.nz {
        for i in 0..bed.field.nx {
            let [x, z] = bed.field.cell_center(i, j);
            let r = ((x + 3.0).powi(2) + z * z).sqrt();
            let k = i + bed.field.nx * j;
            bed.field.depth[k] = (0.5 - r * 0.35).max(0.0);
        }
    }
    bed.field.touch_all();
    let centroid = |b: &sand::SandBed| {
        let f = &b.field;
        let (mut m, mut mx) = (0.0, 0.0);
        for j in 0..f.nz {
            for i in 0..f.nx {
                let d = f.depth[i + f.nx * j];
                m += d;
                mx += d * f.cell_center(i, j)[0];
            }
        }
        (m * f.cell * f.cell, mx / m.max(1e-9))
    };
    let (v0, x0) = centroid(&bed);
    for _ in 0..40 {
        bed.blow(sand::Downwind::PlusX, 2000);
    }
    let (v1, x1) = centroid(&bed);
    assert!((v1 - v0).abs() < 1e-3 * v0, "volume {v0} → {v1}");
    assert!(x1 > x0 + 0.2, "the mound moved from x {x0} to {x1}");
}

/// Same seed, same particles.
#[test]
fn the_same_seed_gives_the_same_particles() {
    let make = || {
        let mut s = super::presets::smoke(Vector3::ZERO, 0.3).with_seed(9);
        run(&mut s, 1.5, 1.0 / 60.0);
        s.particles.position.clone()
    };
    assert_eq!(make(), make());
}

/// Move a system's object and its emitters go with it.
#[test]
fn emitters_follow_their_object() {
    let mut scene = crate::Scene::new();
    let mut anim = ParticleAnimator::new();
    let id = anim.add(&mut scene, super::presets::fire(Vector3::ZERO, 0.2));
    scene.get_mut(id).unwrap().position = Vector3::new(3.0, 1.0, -2.0);
    for _ in 0..30 {
        anim.update(&mut scene, 1.0 / 60.0, None);
    }
    let s = anim.system(id).unwrap();
    assert!(s.len() > 20, "{} particles", s.len());
    let n = s.len() as f32;
    let mean = s
        .particles
        .position
        .iter()
        .fold(Vector3::ZERO, |a, p| a + *p)
        * (1.0 / n);
    assert!(
        (mean.x - 3.0).abs() < 0.3 && (mean.z + 2.0).abs() < 0.3,
        "the fire is at {mean:?}"
    );
    assert!(mean.y > 1.0, "the fire is below its object: {mean:?}");
}

/// Snow over a scene lands on what is there: a crate's top as well as the
/// ground, each at its own height.
#[test]
fn snow_settles_on_the_objects_under_it() {
    use crate::core::{Mesh, Object3D};
    use crate::geometries::{BoxGeometry, PlaneGeometry};
    use crate::materials::{BasicMaterial, Material};
    let mut scene = crate::Scene::new();
    let mut ground = Object3D::mesh(Mesh::new(
        PlaneGeometry::new(20.0, 20.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    ground.rotate_x(-std::f32::consts::FRAC_PI_2);
    scene.add(ground);
    let mut crate_ = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Basic(BasicMaterial::new(Color::WHITE)),
    ));
    crate_.position = Vector3::new(0.0, 0.5, 0.0);
    scene.add(crate_);

    let mut anim = ParticleAnimator::new();
    let cover = snow::SnowCover::new(Heightfield::new([0.0, 0.0], [4.0, 4.0], 40, 0.0));
    let layer = anim.add_snow(&mut scene, cover, Some(5.0));
    let flakes = anim.add(
        &mut scene,
        super::presets::snowfall(Vector3::ZERO, [4.0, 4.0], 3.0, 3000.0, 0.01, Vector3::ZERO),
    );
    anim.settle(flakes, layer);
    for _ in 0..(10 * 30) {
        anim.update(&mut scene, 1.0 / 30.0, None);
    }
    let f = anim.layer(layer).unwrap().field();
    let at = |x: f32, z: f32| {
        let i = ((x - f.origin[0]) / f.cell) as usize;
        let j = ((z - f.origin[1]) / f.cell) as usize;
        (f.base[i + f.nx * j], f.depth[i + f.nx * j])
    };
    let (roof_base, roof_snow) = at(0.05, 0.05);
    let (ground_base, ground_snow) = at(1.5, 1.5);
    assert!(
        (roof_base - 1.0).abs() < 1e-3,
        "the crate's top is at {roof_base}"
    );
    assert!(ground_base.abs() < 1e-3, "the ground is at {ground_base}");
    // About 5 cm fell (10 mm/s for the ~5 s after the first flakes land).
    assert!(
        roof_snow > 0.02 && ground_snow > 0.02,
        "snow: roof {roof_snow}, ground {ground_snow}"
    );
}

/// The log-law wind: nothing at the ground, as given at the reference
/// height, and a slower growth the higher it goes.
#[test]
fn the_wind_slows_toward_the_ground_by_the_log_law() {
    let mut s = dropped(0.0, 1.0);
    s.forces.wind = Vector3::new(5.0, 0.0, 0.0);
    s.forces.boundary_layer = Some(BoundaryLayer {
        ground: 0.0,
        roughness: 0.01,
        reference: 10.0,
    });
    let at = |y: f32| s.air_at(Vector3::new(0.0, y, 0.0)).x;
    assert!(at(0.0).abs() < 1e-4, "{} m/s at the ground", at(0.0));
    assert!((at(10.0) - 5.0).abs() < 1e-3, "{} m/s at 10 m", at(10.0));
    // Each tenfold rise in height adds the same speed.
    let (a, b, c) = (at(0.1), at(1.0), at(10.0));
    assert!(((b - a) - (c - b)).abs() < 1e-3, "{a} {b} {c}");
}

/// A cooling plume climbs fast, then slows as its buoyancy fades.
#[test]
fn a_cooling_plume_rises_and_slows() {
    let mut s = dropped(0.0, 2.0);
    s.forces.gravity = Vector3::new(0.0, 8.0, 0.0);
    s.forces.cooling = Some(Cooling {
        time: 0.5,
        cold_gravity: Vector3::new(0.0, 0.2, 0.0),
    });
    let mut speeds = Vec::new();
    for _ in 0..240 {
        s.step(1.0 / 60.0, None);
        speeds.push(s.particles.velocity[0].y);
    }
    let peak = speeds.iter().cloned().fold(0.0, f32::max);
    let end = *speeds.last().unwrap();
    // Hot, it would rise at its 2 m/s terminal speed; it cools before it
    // gets there.
    assert!(peak > 1.0, "it never climbed: peak {peak}");
    assert!(peak > 3.0 * end, "it did not slow: peak {peak}, end {end}");
    // Cold, its buoyancy 0.2 against drag set for 2 m/s at 8 m/s²: about
    // 2·√(0.2/8) = 0.32 m/s.
    assert!(
        (end - 0.32).abs() < 0.05,
        "settled at {end} m/s, peak {peak}"
    );
}

/// A splash throws droplets off, and droplets never splash.
#[test]
fn a_splash_throws_droplets_that_do_not_splash() {
    let mut s = dropped(1.0, 0.0);
    s.colliders.push(Collider::Ground { height: 0.0 });
    s.response = Response::Die;
    s.spray = Some(Spray {
        count: Range::fixed(5.0),
        speed: Range::fixed(1.0),
        angle: 0.5,
        carry: 0.0,
        lifetime: Range::fixed(10.0),
        size: Range::fixed(0.5),
        terminal_speed: Range::fixed(0.0),
    });
    let mut most = 0;
    for _ in 0..240 {
        s.step(1.0 / 120.0, None);
        most = most.max(s.len());
        if !s.is_empty() {
            assert!(s.particles.generation.iter().all(|&g| g <= 1));
        }
    }
    assert_eq!(most, 5, "the drop's splash");
    assert_eq!(
        s.len(),
        0,
        "droplets died where they landed, without splashing"
    );
}

/// Flutter spreads falling flakes sideways; without it, in still air, they
/// fall straight.
#[test]
fn flutter_spreads_falling_flakes() {
    let spread = |flutter: f32| {
        let mut e = Emitter::new(EmitterShape::Point, Vector3::new(0.0, 10.0, 0.0));
        e.bursts = vec![(0.0, 200)];
        e.lifetime = Range::fixed(f32::INFINITY);
        e.terminal_speed = Range::fixed(1.0);
        let mut s = ParticleSystem::new(ParticleLook::new(
            ParticleKind::Sphere,
            ParticleBlend::Opaque,
        ))
        .with_emitter(e);
        s.forces.flutter = flutter;
        run(&mut s, 4.0, 1.0 / 60.0);
        s.particles
            .position
            .iter()
            .map(|p| (p.x * p.x + p.z * p.z).sqrt())
            .sum::<f32>()
            / s.len() as f32
    };
    // Snowfall's flutter: about half of gravity, swinging a 1 m/s flake a
    // few centimetres either way.
    let (still, fluttering) = (spread(0.0), spread(5.0));
    assert!(still < 1e-3, "without flutter they drift {still} m");
    assert!(
        fluttering > 0.02,
        "with flutter they spread only {fluttering} m"
    );
}

/// The fast cosine is within a thousandth everywhere.
#[test]
fn fast_cos_is_close() {
    let mut worst = 0.0f32;
    for k in -100_000..100_000 {
        let x = k as f32 * 0.001;
        worst = worst.max((fast_cos(x) - x.cos()).abs());
    }
    assert!(worst < 1.2e-3, "off by {worst}");
}

/// A shadow-casting system with nothing alive still has a drawable caster:
/// a geometry with positions (if none), hidden.
#[test]
fn an_empty_caster_is_hidden_and_drawable() {
    let mut scene = crate::Scene::new();
    let look = ParticleLook {
        casts_shadow: true,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Alpha,
        )
    };
    let fx = ParticleEffect::add_to(&mut scene, ParticleSystem::new(look));
    fx.sync(&mut scene, None);
    let caster = scene.get(fx.caster.unwrap()).unwrap();
    assert!(!caster.visible, "an empty caster is drawn");
    if let crate::core::ObjectKind::Mesh(m) = &caster.kind {
        assert!(
            m.geometry.get_attribute("position").is_some(),
            "no position attribute"
        );
    }
}

/// Light through a block of smoke goes as Beer–Lambert's `τ = σ̄ z`: a
/// particle a layer deeper reads the block's mean extinction times the extra
/// path to the light, from straight above and from a slant; a particle in
/// the open reads nothing.
#[test]
fn optical_depth_through_a_block_is_beer_lambert() {
    // A 2 m cube of particles 0.1 m apart, each a sphere of radius 0.08 m
    // with extinction 5/m inside it.
    let (r, s, h) = (0.08f32, 5.0f32, 0.1f32);
    let index = |i: usize, j: usize, k: usize| i + 20 * (j + 20 * k);
    let mut centre = Vec::new();
    for k in 0..20 {
        for j in 0..20 {
            for i in 0..20 {
                centre.push(Vector3::new(i as f32 * h, j as f32 * h, k as f32 * h));
            }
        }
    }
    // A lone particle well outside, above the block.
    centre.push(Vector3::new(1.0, 6.0, 1.0));
    let n = centre.len();
    let radius = vec![r; n];
    let sigma = vec![s; n];
    let mean = s * (4.0 / 3.0) * std::f32::consts::PI * r.powi(3) / h.powi(3);
    for toward in [
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.4, 1.0, 0.0).normalize(),
    ] {
        let tau = volume::optical_depths(&centre, &radius, &sigma, toward);
        assert!(
            tau[n - 1] < 1e-3,
            "the particle in the open reads {}",
            tau[n - 1]
        );
        // Down the middle of the block, layer by layer: each adds the mean
        // extinction over the extra path, h / cos θ.
        let (mut rate, mut count) = (0.0, 0);
        for j in 4..16 {
            for (i, k) in [(9, 9), (10, 10), (8, 11)] {
                let (deeper, shallower) = (tau[index(i, j, k)], tau[index(i, j + 1, k)]);
                rate += (deeper - shallower) / (h / toward.y);
                count += 1;
            }
        }
        let rate = rate / count as f32;
        assert!(
            (rate / mean - 1.0).abs() < 0.1,
            "depth grows {rate:.2}/m toward {toward:?}, the block's extinction is {mean:.2}/m"
        );
    }
}

/// Anisotropic turbulence is the surface layer's: gusts spread most along
/// the wind, less across it, least up and down — σ_v/σ_u ≈ 0.76 and
/// σ_w/σ_u ≈ 0.5 (Panofsky & Dutton) — and still divergence-free.
#[test]
fn surface_layer_turbulence_is_strongest_along_the_wind() {
    let (mut su, mut sv, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    let mut worst_div = 0.0f32;
    for seed in 0..8u64 {
        let mut s = ParticleSystem::new(ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Alpha,
        ))
        .with_seed(seed);
        s.forces.wind = Vector3::new(3.0, 0.0, 0.0);
        s.forces.turbulence = Turbulence {
            strength: 1.0,
            scale: 2.0,
            churn: 0.0,
            anisotropy: 1.0,
        };
        let mut rng = Rng::new(seed + 99);
        for _ in 0..4000 {
            let p = Vector3::new(rng.next() * 60.0, rng.next() * 60.0, rng.next() * 60.0);
            let u = s.turbulence_at(p, 0.0);
            su += (u.x * u.x) as f64;
            sv += (u.z * u.z) as f64;
            sw += (u.y * u.y) as f64;
        }
        // Divergence by central differences, against the gradients' size.
        for _ in 0..50 {
            let p = Vector3::new(rng.next() * 20.0, rng.next() * 20.0, rng.next() * 20.0);
            let e = 1e-2;
            let d = |a: Vector3| s.turbulence_at(p + a * e, 0.0) - s.turbulence_at(p - a * e, 0.0);
            let (dx, dy, dz) = (
                d(Vector3::new(1.0, 0.0, 0.0)),
                d(Vector3::new(0.0, 1.0, 0.0)),
                d(Vector3::new(0.0, 0.0, 1.0)),
            );
            let div = (dx.x + dy.y + dz.z).abs();
            let size = dx.length() + dy.length() + dz.length();
            worst_div = worst_div.max(div / size.max(1e-3));
        }
    }
    let (v, w) = ((sv / su).sqrt(), (sw / su).sqrt());
    assert!((v - 0.76).abs() < 0.12, "σ_v/σ_u = {v:.2}");
    assert!((w - 0.5).abs() < 0.1, "σ_w/σ_u = {w:.2}");
    assert!(worst_div < 0.03, "divergence {worst_div}");
}

/// Integrate one grain's temperature and diameter to `seconds`.
fn ember_after(
    th: &Thermal,
    t0: f32,
    d0: f32,
    speed: f32,
    resting: bool,
    seconds: f32,
) -> (f32, f32) {
    let (mut t, mut d) = (t0, d0);
    let h = 1e-3;
    for _ in 0..(seconds / h) as usize {
        let (dt, dd) = th.rates(t, d, speed, th.burning, resting);
        t += dt * h;
        d = (d + dd * h).max(1e-5);
    }
    (t, d)
}

/// An ember's heat balance: at rest in still air a 2 mm grain of burning
/// char settles bright orange, near 1290 K; blown at 3 m/s it burns hotter
/// and faster; one that is merely warm (600 K) does not catch, but cools;
/// and one come to rest on snow is quenched in a fraction of a second.
#[test]
fn embers_glow_by_their_heat_balance() {
    let th = Thermal::ember(285.0);
    let (rest, d_rest) = ember_after(&th, 1200.0, 0.002, 0.0, false, 2.0);
    let (blown, d_blown) = ember_after(&th, 1200.0, 0.002, 3.0, false, 2.0);
    assert!((1220.0..1330.0).contains(&rest), "at rest: {rest} K");
    assert!(
        blown > rest + 200.0,
        "blown: {blown} K against {rest} K at rest"
    );
    assert!(
        (0.002 - d_blown) > 2.5 * (0.002 - d_rest),
        "burns faster blown"
    );
    let (warm, _) = ember_after(&th, 600.0, 0.002, 0.0, false, 2.0);
    assert!(warm < 500.0, "a warm grain caught: {warm} K");
    let snow = Thermal {
        contact: 3000.0,
        surface_temperature: 273.0,
        ..th
    };
    let (quenched, _) = ember_after(&snow, 1250.0, 0.002, 0.0, true, 0.5);
    assert!(quenched < 600.0, "on snow after 0.5 s: {quenched} K");
}

/// No shader here names anything `gl…`: GLSL reserves the prefix, naga
/// renames a WGSL `gl` to `gl_1` on WebGL2, and ANGLE then rejects the whole
/// module — every pipeline using it fails, silently, in the browser alone.
#[test]
fn no_shader_names_anything_gl() {
    let sources = [
        ("particle vertex", render::PARTICLE_VERTEX_WGSL),
        ("particle", render::PARTICLE_WGSL),
        ("distortion", render::PARTICLE_DISTORT_WGSL),
        ("surface", surface::SURFACE_WGSL),
        ("sky", dunes::SKY_WGSL),
        ("weather", weather::WEATHER_VERTEX_WGSL),
    ];
    for (name, src) in sources {
        let words = src.split(|c: char| !(c.is_alphanumeric() || c == '_'));
        for w in words {
            assert!(
                !(w == "gl" || w.starts_with("gl_")),
                "the {name} shader names `{w}`"
            );
        }
    }
}
