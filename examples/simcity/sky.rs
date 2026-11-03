//! Part of the `simcity` example; see `mod.rs`.
#![allow(dead_code)]

use super::*;

// ---------------------------------------------------------------------------
// Time of day.
// ---------------------------------------------------------------------------

/// Everything the clock decides: where the sun is, what colour the air is, and
/// how many windows are lit.
pub(crate) struct Sky {
    pub(crate) sun_dir: Vector3,
    pub(crate) sun_color: Color,
    pub(crate) sun_intensity: f32,
    pub(crate) moon_intensity: f32,
    pub(crate) hemi_sky: Color,
    pub(crate) hemi_ground: Color,
    pub(crate) hemi_intensity: f32,
    pub(crate) background: Color,
    pub(crate) fog_color: Color,
    /// 0 in daylight, 1 once the street lights are on.
    pub(crate) lights: f32,
    /// 0 at night, 1 in full daylight.
    pub(crate) day: f32,
    /// 0 except around sunrise and sunset, peaking just after the sun has set.
    pub(crate) twilight: f32,
    /// Sine of the sun's elevation.
    pub(crate) elevation: f32,
}

/// Which half of the signal cycle is showing green, from the wall clock.
/// Shared by the lights and by the traffic that has to obey them.
pub(crate) fn signal_phase(time: f32) -> usize {
    ((time / 8.0).floor() as i64).rem_euclid(2) as usize
}

/// `t` is the fraction of a day: 0 = sunrise, 0.25 = noon, 0.5 = sunset.
pub(crate) fn sky_at(t: f32) -> Sky {
    let theta = t * TAU;
    // Two things fix the axis tilt. A sun straight overhead degenerates the
    // shadow camera's look-at; and a sun on the same side as the camera hides
    // every shadow behind the thing casting it, which reads as "shadows are
    // broken". -Z throws them back toward the default three-quarter view.
    let sun_dir = Vector3::new(theta.cos() * 0.74, theta.sin(), -0.42).normalize();
    let h = sun_dir.y;
    // `day` used to reach 1 at eleven degrees of elevation, which this sun
    // clears about three percent of the way into the day: the golden hour was
    // over almost before it began, and `--time 0.05` already looked like noon.
    let day = smoothstep(-0.12, 0.46, h);
    // Peaks while the sun is just above the horizon, and now stays there long
    // enough to be worth pointing a camera at.
    let golden = (-(((h - 0.11) / 0.32).powi(2))).exp() * smoothstep(-0.18, 0.02, h);
    // How far past sunset (or before sunrise) we are, 0 to 1. The sky keeps a
    // lit band above the horizon for a good while after the sun has gone; with
    // nothing standing in for that, sunset cut straight to night.
    let twilight = smoothstep(0.10, -0.16, h) * smoothstep(-0.34, -0.14, h);

    let night_sky = Color::from_hex(0x080c18);
    let day_sky = Color::from_hex(0x89b3dd);
    let dusk = Color::from_hex(0xd97b3f);
    let lights = 1.0 - smoothstep(0.0, 0.24, h);
    let background = mix_color(
        mix_color(mix_color(night_sky, day_sky, day), dusk, golden * 0.66),
        // Light pollution. A city of this size does not sit under a black
        // sky; it sits under its own glow, and the fog inherits it.
        Color::from_hex(0x241b13),
        0.30 * lights,
    );

    Sky {
        sun_dir,
        sun_color: mix_color(
            Color::from_hex(0xff7a2e),
            Color::from_hex(0xfff2e2),
            smoothstep(0.05, 0.58, h),
        ),
        // Airmass, roughly. Sunlight at ten degrees crosses about five times
        // the atmosphere it does overhead, and arrives about half as strong
        // and much redder for it. `h^0.5` gave the dimming but not the shape:
        // it went to zero exactly at the horizon, so the most interesting
        // minute of the day had no sun in it at all.
        sun_intensity: {
            let airmass = 1.0 / (h.max(0.03) + 0.10);
            let extinction = (-0.17 * (airmass - 0.91)).exp();
            4.5 * smoothstep(-0.035, 0.13, h) * extinction
        },
        moon_intensity: 0.34 * (1.0 - day),
        // The sky fill takes the sky's own colour, so a golden hour is warm on
        // the shaded sides too rather than staying midday blue.
        hemi_sky: mix_color(
            mix_color(Color::from_hex(0x141c2e), Color::from_hex(0x9cc3ea), day),
            background,
            golden * 0.58,
        ),
        hemi_ground: mix_color(Color::from_hex(0x0b0e14), Color::from_hex(0x4b4438), day),
        // Twilight is *sky* light and nothing else, so the fill has to carry
        // it: with only `day` in here the city went black the moment the sun
        // dropped below the horizon.
        // A city does not sit under a black sky; it sits under its own glow
        // reflected off the underside of the air, which is why you can read a
        // newspaper at midnight in one and not in a field. The `lights` term
        // is that.
        hemi_intensity: 0.06 + 0.18 * day + 0.10 * twilight + 0.055 * lights,
        background,
        // The haze has to land on the sky dome's colour near the horizon,
        // which is far brighter than the analytic background used for the
        // clear colour. Mismatch here shows as a dark band along the skyline.
        // Measured off the dome rather than guessed: the Preetham horizon is
        // a desaturated grey-teal, much darker and far less blue than the
        // analytic background used for the clear colour. Fading the terrain to
        // the wrong one puts a bright band along the skyline.
        // Measured off the rendered dome rather than guessed, at six times of
        // day: the Preetham horizon stays a bright, desaturated grey-cyan even
        // at sunrise, where the analytic background is a dark warm dusk. Fading
        // the terrain to the background put a hundred-and-sixty-level step
        // along the skyline; fading it to this puts the ground and the sky
        // within a few levels of each other all day.
        // The blend was tuned against a dome that stayed lit all night. Now
        // the dome retires after dusk and the sky goes properly dark, so a fog
        // still half way to daytime grey leaves a bright band along the
        // skyline with a black sky above it.
        // At night the fog has to *be* the background, and "nearly" is not
        // close enough. The previous form mixed in 2% of a daylight grey, but
        // that grey is some thirty times brighter than a night sky, so 2% of
        // it still quadrupled the result: measured, the fully-fogged terrain
        // came out at 34 against a sky of 8, which is exactly the pale band
        // along the skyline this was supposed to remove.
        //
        // Both terms are now scaled by `day`, so at day = 0 this is the
        // background and nothing else. The daylight end is unchanged.
        fog_color: mix_color(
            scale_color(background, 1.0 + 0.30 * day),
            Color::from_hex(0x9fadb1),
            0.86 * day,
        ),
        lights,
        day,
        twilight,
        elevation: h,
    }
}

/// Frame the city. `extent` is half the built area's width in metres, so every
/// view scales with `--blocks` instead of needing a hand-tuned distance.
#[allow(dead_code)]
/// How close the near plane can safely sit, given what the camera is looking
/// at.
///
/// This is the single most damaging number in the whole scene and it was 1.0
/// everywhere. The depth buffer is `Depth32Float` with the *standard* z range,
/// which puts the near plane at 0 and the far plane at 1 — so all of the
/// float's precision bunches up against the near plane and there is almost
/// none left at the far one. The resolvable depth difference at distance `z`
/// goes as `z² / near`, so with `near = 1` and a far plane out at nine
/// kilometres, two surfaces a couple of centimetres apart stop being
/// distinguishable somewhere around a kilometre out — which is exactly where
/// the countryside, the lanes, the ring road and the mountains all are. They
/// were not badly modelled. They were fighting each other for the same depth,
/// and which one won changed with every pixel and every frame.
///
/// Measured on a 0.7 m camera nudge: at `near = 1` a tenth of the frame
/// changed drastically; at 4 it was 1.4%, and raising it further changed
/// nothing, so 1.4% is the honest floor from sub-pixel aliasing.
///
/// One percent of the distance to the subject is safe by construction —
/// nothing between the camera and what it is aimed at can be clipped — and it
/// scales, so a street-level camera three metres from somebody's back still
/// gets a near plane of half a metre.
pub(crate) fn near_for(eye: Vector3, target: Vector3) -> f32 {
    ((eye - target).length() * 0.01).clamp(0.4, 14.0)
}

pub(crate) fn frame_camera(
    view: &str,
    extent: f32,
    aspect: f32,
    vista: (Vector3, Vector3),
) -> threers::PerspectiveCamera {
    let (fov, eye, target) = match view {
        // Standing in an avenue looking downtown. The position comes from the
        // generator, so it is on tarmac rather than inside a building.
        "street" => (55.0, vista.0, vista.1),
        // Low and far: the skyline against the sky, which is what the height
        // falloff and the roof plant are for.
        "skyline" => (
            32.0,
            Vector3::new(extent * 0.26, extent * 0.20, extent * 2.05),
            Vector3::new(0.0, extent * 0.26, 0.0),
        ),
        // Down among the blocks: crossings, street trees, traffic.
        "close" => (
            46.0,
            Vector3::new(extent * 0.44, extent * 0.26, extent * 0.62),
            Vector3::new(-extent * 0.06, extent * 0.06, -extent * 0.10),
        ),
        // Three-quarter aerial — the SimCity camera. Brought in and down: at
        // the old distance a person was about two pixels and a car fifteen, so
        // every hour spent on crowds, wildlife and street furniture was spent
        // below the resolution of the shot the example ships with.
        // Pulled in by about a quarter, not by half: closer than this and the
        // frame holds four blocks, which trades the shape of the city for the
        // detail in it. The oversized movers do most of the work; the camera
        // only has to stop fighting them.
        _ => (
            34.0,
            Vector3::new(extent * 0.70, extent * 0.56, extent * 0.88),
            Vector3::new(-extent * 0.03, extent * 0.06, -extent * 0.04),
        ),
    };
    // The far plane is what decides where the world stops, and this is the
    // number that was cutting the background off early. It is worth being
    // precise about how early: for a camera `h` above the ground the plane
    // crops the ground `(half_height / tan(fov/2)) * h / far` pixels below the
    // horizon, so at 18 half-widths a 900x560 frame lost some forty rows of
    // landscape — the whole of the far distance, replaced by sky.
    //
    // Measured against a minimal one-plane scene (`examples/ground_cut.rs`),
    // which is also how I confirmed the renderer does no culling of its own:
    // the ground reaches exactly as far as this and no further.
    let mut camera =
        threers::PerspectiveCamera::new(fov, aspect, near_for(eye, target), extent * 45.0);
    camera.position = eye;
    camera.look_at(target);
    camera
}

/// A night sky: stars, and a moon to hang them off.
///
/// There was none. After dark the Preetham dome goes to a dark grey wash and
/// that was the whole of it — no stars, no moon, nothing to look at above the
/// skyline. The dome cannot draw them (it is a fixed analytic shader), so they
/// are geometry: a shell of small quads at a radius well inside the camera's
/// far plane, unlit, and only visible once the street lights are on.
///
/// The distribution is deliberately uneven. Stars scattered uniformly read as
/// noise; real ones clump, and a band of them across the sky reads as the
/// galaxy without anyone having to be told.
pub(crate) fn star_field(radius: f32, seed: u64, sky: Color) -> MeshBuilder {
    let mut m = MeshBuilder::default();
    let mut rng = Rng::new(seed ^ 0x5741_5253);
    // The plane the band lies in, tilted off the horizon.
    let band = rng.range(0.0, TAU);
    let (bs, bc) = (band.sin(), band.cos());
    for _ in 0..2400 {
        // Uniform on the sphere, then rejected below the horizon.
        let u = rng.range(-1.0, 1.0);
        let phi = rng.range(0.0, TAU);
        let r = (1.0 - u * u).max(0.0).sqrt();
        let mut d = Vector3::new(r * phi.cos(), u, r * phi.sin());
        if d.y < 0.04 {
            continue;
        }
        // Pull a third of them toward the band: a distance from the plane, and
        // a coin flip on whether this star respects it.
        if rng.chance(0.38) {
            let plane = d.x * bs + d.z * bc;
            d = Vector3::new(d.x - bs * plane * 0.85, d.y, d.z - bc * plane * 0.85).normalize();
            if d.y < 0.04 {
                continue;
            }
        }
        // Magnitude: mostly faint, a few bright. A uniform brightness is half
        // of what makes a scattered star field read as noise.
        let mag = rng.f().powf(2.6);
        let size = radius * (0.0011 + 0.0034 * mag);
        let tint = mix_color(
            Color::from_hex(0x9fb4ff),
            Color::from_hex(0xffe6c4),
            rng.f(),
        );
        let core = scale_color(tint, 0.55 + 0.45 * mag);

        // A fan, not a quad. Square stars are the other half of what makes a
        // star field look wrong, and at a couple of pixels across a square is
        // exactly what the eye reads as "cube".
        //
        // The falloff is per-vertex colour rather than alpha: this material is
        // opaque, so the rim is set to the sky's own colour instead of to
        // black. It fades into the background rather than painting a dark ring
        // round every star, and costs nothing that a blend would have.
        let up = if d.y.abs() > 0.95 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let rt = d.cross(up).normalize();
        let bt = rt.cross(d).normalize();
        let p = Vector3::new(d.x * radius, d.y * radius, d.z * radius);
        const N: usize = 7;
        let at = |a: f32, s: f32| {
            let (sn, cs) = (a.sin(), a.cos());
            [
                p.x + rt.x * cs * s + bt.x * sn * s,
                p.y + rt.y * cs * s + bt.y * sn * s,
                p.z + rt.z * cs * s + bt.z * sn * s,
            ]
        };
        let n = [-d.x, -d.y, -d.z];
        // Two rings, not one. A single fan from a bright point to the sky is a
        // gradient with no star in the middle of it — at two pixels across
        // that averages out to the background and vanishes, which is what the
        // first attempt did. The inner ring is the star; the outer one is the
        // edge falling off.
        let inner = size * 0.45;
        for i in 0..N {
            let (a0, a1) = (
                i as f32 / N as f32 * TAU,
                (i + 1) as f32 / N as f32 * TAU,
            );
            m.tri(
                [[p.x, p.y, p.z], at(a1, inner), at(a0, inner)],
                n,
                [[0.5, 0.5]; 3],
                [core, core, core],
            );
            m.quad_shaded(
                [at(a0, inner), at(a0, size), at(a1, size), at(a1, inner)],
                n,
                [core, sky, sky, core],
            );
        }
    }
    m
}

/// The moon: a disc at the anti-solar point, with a soft halo behind it.
pub(crate) fn moon_disc(radius: f32, dir: Vector3) -> MeshBuilder {
    let mut m = MeshBuilder::default();
    let d = dir.normalize();
    let up = if d.y.abs() > 0.95 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let rt = d.cross(up).normalize();
    let bt = rt.cross(d).normalize();
    let p = Vector3::new(d.x * radius, d.y * radius, d.z * radius);
    // Halo first, then the disc over it.
    for (size, c) in [
        (radius * 0.075, Color::from_hex(0x2a3550)),
        (radius * 0.048, Color::from_hex(0x5a6a8a)),
        (radius * 0.026, Color::from_hex(0xf2efe4)),
    ] {
        const N: usize = 18;
        for i in 0..N {
            let (a0, a1) = (
                i as f32 / N as f32 * TAU,
                (i + 1) as f32 / N as f32 * TAU,
            );
            let at = |a: f32| {
                let (s, co) = (a.sin(), a.cos());
                [
                    p.x + rt.x * co * size + bt.x * s * size,
                    p.y + rt.y * co * size + bt.y * s * size,
                    p.z + rt.z * co * size + bt.z * s * size,
                ]
            };
            m.tri(
                [[p.x, p.y, p.z], at(a1), at(a0)],
                [-d.x, -d.y, -d.z],
                [[0.5, 0.5]; 3],
                [c; 3],
            );
        }
    }
    m
}

/// A deck of cumulus, well below the stars and well above the towers.
///
/// The dome is an analytic gradient with nothing in it, so an empty sky is
/// exactly what it renders — which is fine at noon looking down and obvious in
/// any wide shot. Clouds are geometry: flattened blobs heaped into cells, on a
/// deck a few hundred metres up.
///
/// Two things do most of the work, and neither is the shape of an individual
/// lump. The first is a **flat base**: every lump in a heap has its underside
/// on the same plane, because a cumulus condenses at the altitude where rising
/// air hits its dew point and that altitude is the same across the sky. A heap
/// of blobs at scattered heights reads as cotton wool; the same blobs sharing
/// a base read as weather. The second is the dark underside `add_blob` already
/// gives foliage, which here is doing its actual physical job — the base of a
/// cumulus is the one part of it that is not white.
///
/// They are lit rather than unlit, so they take the sun's colour and go warm
/// at dawn along with everything else. That is most of what makes a sky read
/// as being at a time of day.
pub(crate) fn cloud_deck(reach: f32, base: f32, cover: f32, seed: u64) -> MeshBuilder {
    let mut m = MeshBuilder::default();
    let mut rng = Rng::new(seed ^ 0xc10d_5eed);
    // Scattered on a jittered grid rather than at random. Uniform random
    // points clump and leave holes — which is right for stars, and wrong here,
    // where the eye reads a bald patch as a missing cloud rather than as
    // weather.
    let cells = 13usize;
    let step = reach * 2.0 / cells as f32;
    for gx in 0..cells {
        for gz in 0..cells {
            if rng.f() > cover {
                continue;
            }
            let cx = -reach + (gx as f32 + rng.range(0.15, 0.85)) * step;
            let cz = -reach + (gz as f32 + rng.range(0.15, 0.85)) * step;
            let d = (cx * cx + cz * cz).sqrt();
            if d > reach {
                continue;
            }
            // Size grows with distance so the far ones do not vanish into the
            // haze: at a couple of kilometres a hundred-metre cloud is a
            // smudge, and a deck made only of smudges reads as dirt on the
            // lens.
            let s = (step * 0.34 + d * 0.045) * rng.range(0.75, 1.45);
            // The base wanders a little between heaps but not within one.
            let y = base * rng.range(0.92, 1.16);
            // Heaps run with the wind, so give the whole deck one axis and let
            // each heap stretch along it.
            let (sa, ca) = (0.55f32).sin_cos();
            let lumps = 6 + rng.below(7);
            for k in 0..lumps {
                let t = k as f32 / (lumps - 1).max(1) as f32;
                // Tall in the middle, tapering to the ends: a cumulus is a
                // cauliflower with a shoulder, not a row of equal balls.
                let bulk = (1.0 - (t - 0.5).abs() * 1.55).max(0.30);
                let along = (t - 0.5) * s * 1.32 + rng.range(-s * 0.12, s * 0.12);
                let across = rng.range(-s * 0.30, s * 0.30);
                let lx = cx + along * ca - across * sa;
                let lz = cz + along * sa + across * ca;
                let (rx, rz) = (
                    s * 0.60 * bulk * rng.range(0.85, 1.2),
                    s * 0.54 * bulk * rng.range(0.85, 1.2),
                );
                let ry = s * 0.30 * bulk * rng.range(0.85, 1.25);
                m.add_blob(
                    // Underside on the deck, not the centre: this is the flat
                    // base, and it is the whole difference between weather and
                    // cotton wool.
                    Vector3::new(lx, y + ry * 0.82, lz),
                    rx,
                    ry,
                    rz,
                    4,
                    9,
                    0.16,
                    rng.next_u32() as i32 & 0xffff,
                    0.62,
                    Color::WHITE,
                );
            }
        }
    }
    m
}
