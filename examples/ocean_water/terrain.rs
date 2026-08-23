//! The sea floor.
//!
//! Written once here in Rust and once in WGSL (in [`crate::waves_gpu`]'s shared
//! cascade code, which every GPU consumer includes), and the two must stay
//! identical. This one builds the visible mesh; the other drives shoaling, the
//! depth-graded colour and the surf. Any disagreement shows up immediately as a
//! waterline in the wrong place — and now also as waves breaking over the wrong
//! part of the bed. `SHORE_RADIUS` and `BASE_DEPTH` are substituted into the
//! shader from here so at least the constants cannot drift apart.

use threers::{BufferAttribute, BufferGeometry};

/// Depth in metres at which the graded shelf bottoms out.
pub const BASE_DEPTH: f32 = 40.0;

/// Radius of the island's waterline, metres.
pub const SHORE_RADIUS: f32 = 60.0;

/// Radius of the seabed mesh, metres.
///
/// Past this there is no floor, and the water shader can tell — which draws a
/// straight seam across the sea wherever the edge crosses the frame. Putting it
/// out here rather than at the 900 m the shelf needs costs nothing (the rings
/// are geometric, so the extra distance is a handful of very coarse ones) and
/// puts the seam under 70 m of water, where the column is opaque and there is
/// nothing left to give it away.
pub const SEABED_RADIUS: f32 = 1500.0;

/// Height of the sea floor at `(x, z)`, in metres relative to still water.
pub fn seabed_height(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let land = 18.0 * (1.0 - smoothstep(0.0, SHORE_RADIUS, r));
    // Two grades, because one cannot do both jobs. The gentle term spreads the
    // surf zone over ~140 m so it is more than two pixels tall; the steep term
    // gets the open water genuinely deep, without which sand light floods back
    // up through it and the whole ocean washes out to the colour of the beach.
    let shelf = -(BASE_DEPTH * smoothstep(SHORE_RADIUS, 900.0, r)
        + 6.0 * smoothstep(SHORE_RADIUS, SHORE_RADIUS + 140.0, r));
    // A beach has a *slope*. Both terms above reach sea level with zero
    // gradient — `smoothstep` is flat at both ends — so the bed used to hover
    // within a metre of the waterline across a 40 m annulus. A metre of water is
    // all shoreline foam, so the island came with a white apron several times
    // its own size, and the only reason it was not obvious before is that an 8 m
    // mesh was too coarse to draw the apron and lifted it clear of the water
    // instead.
    let face = -1.9 * smoothstep(SHORE_RADIUS - 16.0, SHORE_RADIUS + 6.0, r);
    let relief = (land / 18.0)
        * (4.0 * (x * 0.031).sin() * (z * 0.026).cos() + 2.0 * (x * 0.09 + z * 0.07).sin());
    let deep = -30.0 * smoothstep(1200.0, 2500.0, r);
    let ripple = 1.6 * (x * 0.021).sin() * (z * 0.017).sin();
    land + shelf + face + relief + deep + ripple + land_detail(x, z, land)
}

/// Dunes and ridges, on the dry part of the island only.
///
/// The island used to be a smooth analytic dome: the finest term in the height
/// field had a wavelength of about 57 m, so from any distance where the island
/// filled the frame there was, quite literally, nothing to see. These are the
/// two scales that read as land — dunes across it, and ridges down its flanks.
///
/// Masked by height above water rather than applied everywhere, because the
/// submerged bed feeds shoaling: bumps down there are read as bathymetry, and
/// the surf starts breaking on noise. The mask is off by the time the bed is
/// under water and the surf zone keeps the smooth profile it was designed with.
fn land_detail(x: f32, z: f32, land: f32) -> f32 {
    let mask = smoothstep(0.5, 6.0, land);
    if mask <= 0.0 {
        return 0.0;
    }
    // ~26 m dunes and ~10 m ridges. Both are several quads across on the mesh
    // that carries them, which is the point — detail finer than the mesh is not
    // detail, it is aliasing.
    let dunes = 1.35 * (x * 0.24 + z * 0.11).sin() * (z * 0.19 - x * 0.07).cos();
    // Phase-warped, so the ridges meander instead of running dead straight.
    // Parallel ridges of a single wavelength read as corduroy under a low sun.
    let warp = 2.0 * (x * 0.06 + z * 0.05).sin();
    let ridges = 0.34 * (x * 0.52 - z * 0.38 + warp).sin();
    mask * (dunes + ridges)
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The seabed as a static mesh, so the island above the waterline is something
/// you can actually see. Below the waterline the water draws over it — the
/// depth-based colour in the shader is doing that job instead.
///
/// # Why rings rather than a grid
///
/// This was a uniform grid over the whole 1.8 km square, which spent its
/// vertices where nothing looks at them and starved the one place that is
/// actually on screen: at 220 segments a quad is 8 m, and the island's shore is
/// 60 m across, so the waterline was a fifteen-sided polygon with a visible
/// staircase around it.
///
/// Rings spaced geometrically from the island outwards put the resolution where
/// the detail is — about 1.3 m along the shore, tens of metres out on the deep
/// shelf where the water is opaque anyway — for the same vertex count.
pub fn build_seabed(extent: f32, rings: usize, sectors: usize) -> BufferGeometry {
    let mut positions = Vec::with_capacity((rings * sectors + 1) * 3);
    let mut normals = Vec::with_capacity((rings * sectors + 1) * 3);
    let mut uvs = Vec::with_capacity((rings * sectors + 1) * 2);

    // Innermost ring. Small enough that the island's peak is not a facet.
    const INNER: f32 = 0.7;

    let push = |x: f32,
                z: f32,
                e: f32,
                positions: &mut Vec<f32>,
                normals: &mut Vec<f32>,
                uvs: &mut Vec<f32>| {
        positions.extend_from_slice(&[x, seabed_height(x, z), z]);
        let hx = seabed_height(x + e, z) - seabed_height(x - e, z);
        let hz = seabed_height(x, z + e) - seabed_height(x, z - e);
        let inv = 1.0 / (hx * hx + 4.0 * e * e + hz * hz).sqrt();
        normals.extend_from_slice(&[-hx * inv, 2.0 * e * inv, -hz * inv]);
        uvs.extend_from_slice(&[x / extent * 0.5 + 0.5, z / extent * 0.5 + 0.5]);
    };

    push(
        0.0,
        0.0,
        INNER * 0.5,
        &mut positions,
        &mut normals,
        &mut uvs,
    );

    let growth = (extent / INNER).powf(1.0 / (rings - 1) as f32);
    for ring in 0..rings {
        let r = INNER * growth.powi(ring as i32);
        // Difference over roughly one ring's spacing, so the normals are
        // band-limited to what the mesh can carry.
        let e = (r * (growth - 1.0)).clamp(0.3, 40.0);
        for s in 0..sectors {
            let a = std::f32::consts::TAU * s as f32 / sectors as f32;
            push(
                r * a.cos(),
                r * a.sin(),
                e,
                &mut positions,
                &mut normals,
                &mut uvs,
            );
        }
    }

    // Winding, carefully.
    //
    // The grid this replaced walked (x, z); this walks (angle, radius), and that
    // is the *opposite* handedness — the Jacobian of (theta, r) -> (x, z) has a
    // negative determinant, because the two coordinates are swapped relative to
    // it. Carrying the old index order across therefore turns every triangle
    // inside out, and an inside-out heightfield is not subtly wrong: it is
    // backface-culled, so the island is invisible from above and shows only its
    // far flank from low down.
    let mut index = Vec::with_capacity(rings * sectors * 6);
    // Fan from the centre to the first ring.
    for s in 0..sectors {
        let a = 1 + s;
        let b = 1 + (s + 1) % sectors;
        index.extend_from_slice(&[0, b as u32, a as u32]);
    }
    for ring in 0..rings - 1 {
        let inner = 1 + ring * sectors;
        let outer = inner + sectors;
        for s in 0..sectors {
            let s1 = (s + 1) % sectors;
            let (a, b) = ((inner + s) as u32, (inner + s1) as u32);
            let (c, d) = ((outer + s) as u32, (outer + s1) as u32);
            index.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }

    let mut geom = BufferGeometry::new();
    geom.set_attribute("position", BufferAttribute::new(positions, 3));
    geom.set_attribute("normal", BufferAttribute::new(normals, 3));
    geom.set_attribute("uv", BufferAttribute::new(uvs, 2));
    geom.set_index(index);
    geom
}

/// Local wavenumber for a deep-water `k0` over depth `d`.
///
/// The Rust twin of `local_wavenumber` in the shader — Fenton & McKee's explicit
/// inverse of `w^2 = g k tanh(kd)`. It exists so the formula the surf zone is
/// built on can be checked against its own limits without a GPU; the shader's
/// copy has to match it, and only the tests run this one.
#[cfg(test)]
pub fn local_wavenumber(k0: f32, d: f32) -> f32 {
    let x = k0 * d;
    if x > 3.0 {
        return k0;
    }
    let t = (x.max(1e-4).powf(0.75)).tanh();
    k0 * t.powf(-2.0 / 3.0)
}

/// Green's law shoaling coefficient: how much a wave grows over depth `d`.
/// Test-only, for the same reason as [`local_wavenumber`].
#[cfg(test)]
pub fn shoaling_gain(k0: f32, d: f32) -> f32 {
    if k0 * d > 3.0 {
        return 1.0;
    }
    let k = local_wavenumber(k0, d);
    let kd = k * d;
    let sh = if kd > 5.0 {
        0.0
    } else {
        2.0 * kd / (2.0 * kd).sinh()
    };
    (k / (k0 * (1.0 + sh))).max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn island_breaks_the_surface_and_deep_water_is_deep() {
        assert!(seabed_height(0.0, 0.0) > 5.0, "island should be dry land");
        // The waterline sits near SHORE_RADIUS, and the shelf grades away from it.
        assert!(seabed_height(SHORE_RADIUS + 400.0, 0.0) < -8.0);
        assert!(
            seabed_height(3000.0, 0.0) < -60.0,
            "open ocean should be deep"
        );
    }

    #[test]
    fn the_shoreline_is_finely_resolved() {
        // The defect this guards against is a staircase: a mesh whose quads are
        // 8 m across cannot draw a 60 m island's waterline as anything but a
        // polygon. Measure the spacing of the ring nearest the shore.
        let (rings, sectors) = (180usize, 320usize);
        const INNER: f32 = 0.7;
        let growth = (SEABED_RADIUS / INNER).powf(1.0 / (rings - 1) as f32);
        let ring = (0..rings)
            .min_by(|&a, &b| {
                let ra = INNER * growth.powi(a as i32);
                let rb = INNER * growth.powi(b as i32);
                (ra - SHORE_RADIUS)
                    .abs()
                    .partial_cmp(&(rb - SHORE_RADIUS).abs())
                    .unwrap()
            })
            .unwrap();
        let r = INNER * growth.powi(ring as i32);
        let arc = std::f32::consts::TAU * r / sectors as f32;
        let radial = r * (growth - 1.0);
        assert!(arc < 2.0, "shoreline arc {arc:.2} m per quad");
        assert!(radial < 4.0, "shoreline radial step {radial:.2} m per quad");
    }

    #[test]
    fn detail_stays_out_of_the_water() {
        // Bumps on the submerged bed are read as bathymetry by the shoaling
        // code, so the surf would start breaking on decoration.
        for r in [SHORE_RADIUS, SHORE_RADIUS + 20.0, SHORE_RADIUS + 200.0] {
            let land = 18.0 * (1.0 - smoothstep(0.0, SHORE_RADIUS, r));
            assert_eq!(land_detail(r, 0.0, land), 0.0, "detail at r={r}");
        }
        // And it is present on the dry island.
        let peak_land = 18.0f32;
        let any = (0..40)
            .map(|i| land_detail(i as f32, i as f32 * 0.7, peak_land).abs())
            .fold(0.0f32, f32::max);
        assert!(any > 0.4, "island has no relief: {any}");
    }

    #[test]
    fn dispersion_hits_both_limits() {
        let k0 = 2.0 * std::f32::consts::PI / 130.0; // storm's peak
                                                     // Deep water: the local wavenumber is the deep-water one.
        assert!((local_wavenumber(k0, 400.0) - k0).abs() < 1e-6);
        // Shallow water: w^2 = g k^2 d, so k -> sqrt(k0/d).
        for &d in &[0.5f32, 1.0, 2.0] {
            let expect = (k0 / d).sqrt();
            let got = local_wavenumber(k0, d);
            assert!(
                (got - expect).abs() / expect < 0.05,
                "d={d} got {got} want {expect}"
            );
        }
        // And it only ever shortens the wave.
        for i in 1..200 {
            let d = i as f32;
            assert!(local_wavenumber(k0, d) >= k0 - 1e-6);
        }
    }

    #[test]
    fn shoaling_dips_then_grows() {
        let k0 = 2.0 * std::f32::consts::PI / 130.0;
        assert!(
            (shoaling_gain(k0, 500.0) - 1.0).abs() < 1e-3,
            "deep water is unchanged"
        );
        // Green's law: a wave first *shrinks* by a few percent before it grows.
        // Getting this backwards is the classic way to notice the formula is wrong.
        let dip = (5..40)
            .map(|i| shoaling_gain(k0, i as f32))
            .fold(f32::MAX, f32::min);
        assert!(dip < 0.99 && dip > 0.85, "dip {dip}");
        // Then it runs away as d^(-1/4): ~1.29x at 2 m, ~1.52x at 1 m for this
        // swell, and monotonically increasing as the water runs out.
        assert!(
            shoaling_gain(k0, 2.0) > 1.25,
            "shallow water should amplify"
        );
        assert!(shoaling_gain(k0, 1.0) > 1.45);
        assert!(shoaling_gain(k0, 1.0) > shoaling_gain(k0, 2.0));
    }

    #[test]
    fn waves_break_before_the_shore() {
        // A storm swell must reach its depth limit somewhere on this bed, or the
        // surf zone is decoration rather than a consequence.
        let k0 = 2.0 * std::f32::consts::PI / 130.0;
        let h_s = 8.1f32;
        let breaks = (0..900).map(|i| SHORE_RADIUS + i as f32).any(|r| {
            let d = (-seabed_height(r, 0.0)).max(0.0);
            d > 0.1 && shoaling_gain(k0, d) * h_s * 0.5 > 0.78 * d
        });
        assert!(breaks, "no depth on this bed limits the wave");
    }

    #[test]
    fn surf_zone_is_wide_enough_to_see() {
        // Shoreline foam keys off water shallower than a few metres. If that band
        // is only metres wide it lands on two pixels and reads as nothing, which
        // is what a steeper profile did.
        let shallow = (0..2000)
            .map(|i| SHORE_RADIUS + i as f32)
            .take_while(|&r| seabed_height(r, 0.0) > -4.5)
            .count();
        // The failure this guards against is a surf zone metres wide, which
        // lands on two pixels and reads as nothing at all.
        assert!(shallow > 50, "surf zone only {shallow} m wide");
    }
}
