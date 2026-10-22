//! Ready-made systems. Each is an ordinary [`ParticleSystem`]: change any
//! field after making it.
//!
//! The numbers are physical where there is physics to go on: rain falls at
//! about 9 m/s and is seen as a streak a few centimetres long at a camera's
//! shutter; a snowflake settles at about 1 m/s and goes where the air takes
//! it; a sand grain drops at a few m/s; flame and smoke rise because they are
//! lighter than the air (an upward "gravity"), smoke slower and longer.

use crate::math::{Color, Vector3};

use super::{
    BoundaryLayer, Cooling, Curve, Direction, Emitter, EmitterShape, Forces, Gradient,
    ParticleBlend, ParticleKind, ParticleLook, ParticleSystem, Range, Response, Spray, Turbulence,
};

fn up() -> Vector3 {
    Vector3::new(0.0, 1.0, 0.0)
}

/// A fire on a disc `radius` metres across at `at`: hot, rising, flickering,
/// added light that cools from white-yellow to red and fades.
pub fn fire(at: Vector3, radius: f32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius,
            normal: up(),
        },
        at,
    );
    e.rate = 150.0 * (radius / 0.3).powi(2).max(0.2);
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.25,
    };
    e.speed = Range::new(0.3, 0.9);
    e.lifetime = Range::new(0.5, 1.1);
    e.size = Range::new(radius * 0.7, radius * 1.2);
    e.terminal_speed = Range::new(1.6, 2.4);
    e.spin = Range::new(-1.5, 1.5);
    let mut s = ParticleSystem::new(ParticleLook {
        intensity: 0.35,
        // Flame breaks into tongues and wisps, drawn out as it licks up.
        detail: 1.0,
        stretch: 0.25,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Additive,
        )
    })
    .with_emitter(e);
    s.forces = Forces {
        // Hot gas: strongly buoyant at first, cooling as it mixes.
        gravity: Vector3::new(0.0, 7.0, 0.0),
        cooling: Some(Cooling {
            time: 0.45,
            cold_gravity: Vector3::new(0.0, 1.0, 0.0),
        }),
        turbulence: Turbulence {
            strength: 0.6,
            scale: radius * 1.5,
            churn: 1.2,
            anisotropy: 0.0,
        },
        ..Forces::default()
    };
    s.size_over_life = Curve(vec![(0.0, 0.55), (0.25, 1.0), (1.0, 0.35)]);
    // Glowing soot: a black body cooling from about 1900 K at the base —
    // yellow-white — through orange to a dull red as it rises and mixes,
    // its brightness falling as T⁴.
    s.temperature_over_life = Some(Curve(vec![
        (0.0, 1900.0),
        (0.25, 1650.0),
        (0.6, 1300.0),
        (1.0, 1000.0),
    ]));
    s.color_over_life = Gradient(vec![
        (0.0, [1.0, 1.0, 1.0, 0.0]),
        (0.06, [1.0, 1.0, 1.0, 0.7]),
        (0.5, [1.0, 1.0, 1.0, 0.5]),
        (1.0, [1.0, 1.0, 1.0, 0.0]),
    ]);
    s
}

/// Embers from a fire at `at`: glowing motes carried up by its plume,
/// wandering in the eddies, cooling from orange to nothing.
pub fn embers(at: Vector3, radius: f32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius,
            normal: up(),
        },
        at,
    );
    e.rate = 14.0;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.4,
    };
    e.speed = Range::new(0.8, 2.0);
    e.lifetime = Range::new(3.0, 6.0);
    e.size = Range::new(0.025, 0.04);
    e.terminal_speed = Range::new(1.0, 1.6);
    let mut s = ParticleSystem::new(ParticleLook {
        intensity: 1.0,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 0.5 },
            ParticleBlend::Additive,
        )
    })
    .with_emitter(e);
    s.forces = Forces {
        gravity: Vector3::new(0.0, 3.5, 0.0),
        cooling: Some(Cooling {
            time: 0.8,
            cold_gravity: Vector3::new(0.0, -2.0, 0.0),
        }),
        turbulence: Turbulence {
            strength: 0.8,
            scale: 0.5,
            churn: 0.8,
            anisotropy: 0.0,
        },
        ..Forces::default()
    };
    // Burning char: glowing by its own heat balance, cooling as it flies,
    // gone when burnt to ash.
    s.thermal = Some(super::Thermal::ember(285.0));
    s.color_over_life = Gradient(vec![
        (0.0, [1.0, 1.0, 1.0, 1.0]),
        (0.9, [1.0, 1.0, 1.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 0.0]),
    ]);
    s
}

/// Heat haze over a fire at `at`: invisible billows rising with the hot
/// air, bending the view of what is behind them. Drawn in the screen-space
/// pass.
pub fn heat_haze(at: Vector3, radius: f32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius,
            normal: up(),
        },
        at,
    );
    e.rate = 24.0;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.2,
    };
    e.speed = Range::new(0.6, 1.0);
    e.lifetime = Range::new(1.0, 1.6);
    e.size = Range::new(radius * 2.0, radius * 3.0);
    e.terminal_speed = Range::new(1.0, 1.4);
    let mut s = ParticleSystem::new(ParticleLook::new(
        ParticleKind::Distortion { strength: 0.006 },
        ParticleBlend::Alpha,
    ))
    .with_emitter(e);
    s.forces.gravity = Vector3::new(0.0, 1.5, 0.0);
    s.color_over_life = Gradient(vec![
        (0.0, [1.0, 1.0, 1.0, 0.0]),
        (0.2, [1.0, 1.0, 1.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 0.0]),
    ]);
    s
}

/// Smoke rising from `at`: slow, spreading, turning in the eddies, grey,
/// lit by the scene.
pub fn smoke(at: Vector3, radius: f32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius,
            normal: up(),
        },
        at,
    );
    e.rate = 45.0;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.3,
    };
    e.speed = Range::new(0.3, 0.6);
    e.lifetime = Range::new(4.0, 6.0);
    e.size = Range::new(radius * 1.6, radius * 2.4);
    e.terminal_speed = Range::new(0.7, 1.0);
    e.spin = Range::new(-0.4, 0.4);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        detail: 0.75,
        scattering: 0.45,
        self_shadow: 0.5,
        casts_shadow: true,
        extinction: 2.0,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    s.forces = Forces {
        // Warm at first: climbs, then slows and spreads as it cools.
        gravity: Vector3::new(0.0, 2.5, 0.0),
        cooling: Some(Cooling {
            time: 1.5,
            cold_gravity: Vector3::new(0.0, 0.25, 0.0),
        }),
        wind: Vector3::new(0.4, 0.0, 0.0),
        turbulence: Turbulence {
            strength: 0.35,
            scale: 0.9,
            churn: 0.4,
            anisotropy: 0.0,
        },
        ..Forces::default()
    };
    s.size_over_life = Curve(vec![(0.0, 0.6), (1.0, 3.2)]);
    s.color_over_life = Gradient(vec![
        (0.0, [0.5, 0.5, 0.52, 0.0]),
        (0.12, [0.5, 0.5, 0.52, 0.35]),
        // Thinning as it spreads: the same smoke over a bigger puff.
        (0.5, [0.58, 0.58, 0.6, 0.14]),
        (1.0, [0.65, 0.65, 0.68, 0.0]),
    ]);
    s
}

/// Sparks from `at`: a shower thrown up and out, falling in arcs, bouncing
/// off the ground at `ground`, streaked by their speed and cooling from
/// white-hot.
pub fn sparks(at: Vector3) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Sphere {
            radius: 0.03,
            surface: false,
        },
        at,
    );
    e.rate = 300.0;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.9,
    };
    e.speed = Range::new(1.5, 4.0);
    e.lifetime = Range::new(0.6, 1.6);
    e.size = Range::new(0.016, 0.03);
    // Tiny and dense: barely slowed.
    e.terminal_speed = Range::new(14.0, 22.0);
    let mut s = ParticleSystem::new(ParticleLook {
        intensity: 2.5,
        ..ParticleLook::new(
            ParticleKind::Streak {
                softness: 0.6,
                shutter: 1.0 / 40.0,
            },
            ParticleBlend::Additive,
        )
    })
    .with_emitter(e);
    s.response = Response::Bounce {
        restitution: 0.35,
        friction: 0.4,
    };
    // A spark that hits breaks up into a few smaller ones.
    s.spray = Some(Spray {
        count: Range::new(0.0, 3.0),
        speed: Range::new(0.6, 1.8),
        angle: 1.0,
        carry: 0.5,
        lifetime: Range::new(0.15, 0.4),
        size: Range::new(0.5, 0.8),
        terminal_speed: Range::new(10.0, 15.0),
    });
    s.colliders.push(super::Collider::Ground { height: 0.0 });
    // White-hot iron cooling: 2000 K down to a dull 1000 K.
    s.temperature_over_life = Some(Curve(vec![(0.0, 2000.0), (0.5, 1500.0), (1.0, 1000.0)]));
    s.color_over_life = Gradient(vec![
        (0.0, [1.0, 1.0, 1.0, 1.0]),
        (0.8, [1.0, 1.0, 1.0, 1.0]),
        (1.0, [1.0, 1.0, 1.0, 0.0]),
    ]);
    s
}

/// Rain over a `size` (x, z) area centred on `center`, falling from
/// `height`: drops at terminal speed (~9 m/s), streaked as a camera sees
/// them, slanted by `wind`. `rate` drops a second.
pub fn rain(
    center: Vector3,
    size: [f32; 2],
    height: f32,
    rate: f32,
    wind: Vector3,
) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Box {
            half: Vector3::new(size[0] * 0.5, 0.5, size[1] * 0.5),
        },
        Vector3::new(center.x, center.y + height, center.z),
    );
    e.rate = rate;
    e.speed = Range::fixed(0.0);
    // Born at terminal speed, already falling with the wind.
    e.velocity = Vector3::new(wind.x, -9.0, wind.z);
    e.lifetime = Range::fixed(f32::INFINITY);
    e.size = Range::new(0.006, 0.01);
    e.terminal_speed = Range::new(8.0, 9.5);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        // Sun behind you, a bow in front.
        rainbow: 1.0,
        ..ParticleLook::new(
            ParticleKind::Streak {
                softness: 0.8,
                shutter: 1.0 / 50.0,
            },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    s.forces.wind = wind;
    s.colliders
        .push(super::Collider::Ground { height: center.y });
    s.response = Response::Die;
    // Each drop splashes: a few droplets thrown up and out, falling back.
    s.spray = Some(Spray {
        count: Range::new(1.0, 4.0),
        speed: Range::new(0.6, 1.6),
        angle: 1.0,
        carry: 0.2,
        lifetime: Range::new(0.25, 0.5),
        size: Range::new(0.5, 0.8),
        terminal_speed: Range::new(3.0, 5.0),
    });
    s.color_over_life = Gradient::constant(Color::new(0.75, 0.8, 0.88), 0.55);
    s.max_particles = 40_000;
    s
}

/// Snow over a `size` (x, z) area centred on `center`, falling from
/// `height`: flakes settling at about 1 m/s, carried by `wind` and its
/// eddies, turning as they fall, and depositing where they land — `rate`
/// flakes a second, building `depth_per_second` metres of snow a second
/// over the area (real snowfall is a few centimetres an hour; a scene
/// usually wants a time-lapse). Lit and shadowed by the scene.
pub fn snowfall(
    center: Vector3,
    size: [f32; 2],
    height: f32,
    rate: f32,
    depth_per_second: f32,
    wind: Vector3,
) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Box {
            half: Vector3::new(size[0] * 0.5, 0.3, size[1] * 0.5),
        },
        Vector3::new(center.x, center.y + height, center.z),
    );
    e.rate = rate;
    e.velocity = Vector3::new(wind.x, -1.0, wind.z);
    e.lifetime = Range::fixed(f32::INFINITY);
    e.size = Range::new(0.012, 0.028);
    e.terminal_speed = Range::new(0.7, 1.3);
    e.spin = Range::new(-3.0, 3.0);
    e.volume = depth_per_second * size[0] * size[1] / rate.max(1e-3);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        // Six-armed, and each with a facet that catches the sun.
        flake: true,
        glint: 1.0,
        scattering: 0.3,
        // Falling broadside, rocking a few degrees: glints in pillars.
        plates: 0.12,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 0.7 },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    s.forces.wind = wind;
    s.forces.turbulence = Turbulence {
        strength: 0.45,
        scale: 1.2,
        churn: 0.5,
        anisotropy: 1.0,
    };
    // Flakes zig-zag as they fall, and the wind given is the wind at the
    // top of the fall, slowing toward the snow.
    s.forces.flutter = 5.0;
    s.forces.boundary_layer = Some(BoundaryLayer {
        ground: center.y,
        roughness: 0.002,
        reference: height.max(1.0),
    });
    s.response = Response::Deposit;
    s.colliders
        .push(super::Collider::Ground { height: center.y });
    s.color_over_life = Gradient::constant(Color::new(1.0, 1.0, 1.0), 0.95);
    s.max_particles = 60_000;
    s
}

/// Sand blown off dunes' crests (see [`dunes::crests`](super::dunes::crests)):
/// lifted from the brinks into a `wind` (m/s) and streaming over the slip
/// faces in thin veils that spread and settle — the plumes that smoke off
/// dunes in a strong wind. Lit as a medium: bright against a low sun.
pub fn spindrift(
    crests: std::sync::Arc<[Vector3]>,
    wind: Vector3,
    rate: f32,
    color: Color,
) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Points {
            points: crests,
            jitter: 0.6,
        },
        Vector3::ZERO,
    );
    e.rate = rate;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.6,
    };
    e.speed = Range::new(0.3, 1.2);
    // Leaving the crest at most of the wind's speed.
    e.velocity = wind * 0.7;
    e.lifetime = Range::new(1.5, 3.5);
    e.size = Range::new(0.2, 0.5);
    e.terminal_speed = Range::new(0.4, 0.8);
    e.spin = Range::new(-0.3, 0.3);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        detail: 0.9,
        stretch: 0.3,
        scattering: 0.6,
        self_shadow: 0.3,
        // A thin veil: a metre of it dims the light behind by a fifth.
        extinction: 0.25,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    s.forces = Forces {
        // Fine sand: settling slowly out of the air that carries it.
        gravity: Vector3::new(0.0, -1.2, 0.0),
        wind,
        turbulence: Turbulence {
            strength: wind.length() * 0.25,
            scale: 3.0,
            churn: 0.5,
            anisotropy: 1.0,
        },
        ..Forces::default()
    };
    s.max_particles = 12_000;
    s.size_over_life = Curve(vec![(0.0, 0.5), (1.0, 3.0)]);
    s.color_over_life = Gradient(vec![
        (0.0, [color.r, color.g, color.b, 0.0]),
        (0.15, [color.r, color.g, color.b, 0.3]),
        (1.0, [color.r, color.g, color.b, 0.0]),
    ]);
    s
}

/// Sand poured from a nozzle at `at`: grains falling at a few m/s in a
/// narrow stream, depositing where they land — `rate` grains a second,
/// carrying `flow` m³ of sand a second between them. Drawn as lit grains,
/// each a slightly different shade.
pub fn sand_stream(at: Vector3, rate: f32, flow: f32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius: 0.025,
            normal: up(),
        },
        at,
    );
    e.rate = rate;
    e.direction = Direction::Cone {
        axis: Vector3::new(0.0, -1.0, 0.0),
        angle: 0.05,
    };
    e.speed = Range::new(0.3, 0.5);
    e.lifetime = Range::fixed(f32::INFINITY);
    e.size = Range::new(0.009, 0.016);
    e.terminal_speed = Range::new(3.0, 5.0);
    e.volume = flow / rate.max(1e-3);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        // Quartz catches the light. (Set `casts_shadow` for the stream's
        // shadow: thousands of casters, redrawn every frame.)
        glint: 0.25,
        ..ParticleLook::new(ParticleKind::Sphere, ParticleBlend::Opaque)
    })
    .with_emitter(e);
    s.response = Response::Deposit;
    // A few grains glance off the pile and skitter down it.
    s.spray = Some(Spray {
        count: Range::new(0.0, 1.3),
        speed: Range::new(0.3, 0.9),
        angle: 1.1,
        carry: 0.6,
        lifetime: Range::new(0.2, 0.5),
        size: Range::new(0.8, 1.0),
        terminal_speed: Range::new(3.0, 5.0),
    });
    s.colliders.push(super::Collider::Ground { height: 0.0 });
    s.color_over_life = Gradient::constant(Color::from_hex(0xd8b98a), 1.0);
    s
}

/// A puff of dust kicked up at `at`: a burst of `count` motes, lifted and
/// spreading in the eddies, settling slowly, lit by the scene.
pub fn dust(at: Vector3, count: u32) -> ParticleSystem {
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius: 0.3,
            normal: up(),
        },
        at,
    );
    e.bursts = vec![(0.0, count)];
    e.direction = Direction::Cone {
        axis: up(),
        angle: 1.1,
    };
    e.speed = Range::new(2.0, 4.0);
    e.lifetime = Range::new(2.5, 4.5);
    e.size = Range::new(0.15, 0.35);
    e.terminal_speed = Range::new(0.35, 0.6);
    e.spin = Range::new(-0.5, 0.5);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        detail: 0.7,
        scattering: 0.5,
        self_shadow: 0.6,
        extinction: 2.0,
        ..ParticleLook::new(
            ParticleKind::Billboard { softness: 1.0 },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    // Fine motes are all but carried by the air: what settles them is the
    // little their weight exceeds its buoyancy.
    s.forces.gravity = Vector3::new(0.0, -0.6, 0.0);
    s.forces.turbulence = Turbulence {
        strength: 0.3,
        scale: 0.8,
        churn: 0.3,
        anisotropy: 0.6,
    };
    s.colliders.push(super::Collider::Ground { height: at.y });
    s.response = Response::Stick;
    s.size_over_life = Curve(vec![(0.0, 0.5), (1.0, 2.5)]);
    s.color_over_life = Gradient(vec![
        (0.0, [0.55, 0.45, 0.33, 0.0]),
        (0.1, [0.55, 0.45, 0.33, 0.35]),
        (1.0, [0.6, 0.52, 0.42, 0.0]),
    ]);
    s
}

/// A fountain jet from `at`, `height` metres high: water thrown up in a
/// narrow cone, falling back in a sheet of streaks onto the basin at
/// `at.y`.
pub fn fountain(at: Vector3, height: f32) -> ParticleSystem {
    // v² = 2 g h for the top of the jet.
    let v = (2.0 * 9.81 * height).sqrt();
    let mut e = Emitter::new(
        EmitterShape::Disc {
            radius: 0.04,
            normal: up(),
        },
        at,
    );
    e.rate = 1800.0;
    e.direction = Direction::Cone {
        axis: up(),
        angle: 0.22,
    };
    e.speed = Range::new(v * 0.95, v * 1.02);
    e.lifetime = Range::fixed(f32::INFINITY);
    e.size = Range::new(0.012, 0.022);
    e.terminal_speed = Range::new(7.0, 9.0);
    let mut s = ParticleSystem::new(ParticleLook {
        lit: true,
        ..ParticleLook::new(
            ParticleKind::Streak {
                softness: 0.7,
                shutter: 1.0 / 60.0,
            },
            ParticleBlend::Alpha,
        )
    })
    .with_emitter(e);
    s.colliders.push(super::Collider::Ground {
        height: at.y - 0.01,
    });
    s.response = Response::Die;
    // Falling back into the basin, it splashes.
    s.spray = Some(Spray {
        count: Range::new(0.0, 2.0),
        speed: Range::new(0.4, 1.2),
        angle: 0.9,
        carry: 0.3,
        lifetime: Range::new(0.2, 0.45),
        size: Range::new(0.5, 0.8),
        terminal_speed: Range::new(3.0, 5.0),
    });
    s.color_over_life = Gradient::constant(Color::new(0.75, 0.86, 0.95), 0.6);
    s
}
