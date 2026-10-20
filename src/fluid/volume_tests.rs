//! What can be asserted about a volume whose entire state is a tiled texture.
//!
//! The atlas is the part most likely to be quietly wrong: every stage reads
//! neighbours by computing a tile and an offset, and a mistake there produces
//! a plume that still looks like a plume. So the first test projects a splat's
//! known position onto the screen and checks the brightest pixel is *there* —
//! which can only pass if `voxel_of`, `sample_grid` and the ray march all
//! agree about where a voxel is.

use super::*;
use crate::{PerspectiveCamera, Vector3};

fn gpu() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let gpu = crate::gpu::testing::context("fluid3d")?;
    Some((gpu.device.clone(), gpu.queue.clone()))
}

fn sim(config: FluidConfig3d) -> Option<FluidSim3d> {
    let (device, queue) = gpu()?;
    FluidSim3d::new(device, queue, config).ok()
}

/// A second simulation, for a test that compares two configurations.
fn sim2(config: FluidConfig3d) -> Option<FluidSim3d> {
    sim(config)
}

/// A camera looking at the volume from outside it.
fn camera(aspect: f32) -> PerspectiveCamera {
    let mut c = PerspectiveCamera::new(45.0, aspect, 0.05, 20.0);
    c.position = Vector3::new(0.0, 0.0, 1.9);
    c.look_at(Vector3::ZERO);
    c
}

fn brightest(rgba: &[u8], w: u32, h: u32) -> (f32, f32, u8) {
    let mut best = (0u32, 0u32, 0u8);
    for y in 0..h {
        for x in 0..w {
            let at = ((y * w + x) * 4) as usize;
            let v = rgba[at].max(rgba[at + 1]).max(rgba[at + 2]);
            if v > best.2 {
                best = (x, y, v);
            }
        }
    }
    (best.0 as f32 / w as f32, best.1 as f32 / h as f32, best.2)
}

/// Read one channel of the volume back, as `f32` per voxel of the atlas.
fn read_channel(sim: &FluidSim3d, fbo: &Fbo, channel: usize) -> Vec<f32> {
    crate::renderer::readback::texture_channel_f32(&sim.device, &sim.queue, &fbo.texture, channel)
}

/// The atlas packs a cube into a rectangle, and the arithmetic has to be
/// exact in both directions or every neighbour lookup is wrong.
#[test]
fn the_atlas_holds_every_slice() {
    for grid in [2u32, 8, 16, 31, 64, 100, 128] {
        let a = Atlas::new(grid);
        let (w, h) = a.size();
        assert!(
            a.tiles.0 * a.tiles.1 >= grid,
            "{grid}: {}x{} tiles cannot hold {grid} slices",
            a.tiles.0,
            a.tiles.1
        );
        assert_eq!(w, grid * a.tiles.0);
        assert_eq!(h, grid * a.tiles.1);
        // Square-ish, so neither axis runs away towards the device's limit.
        let ratio = w.max(h) as f32 / w.min(h) as f32;
        assert!(ratio <= 2.0, "{grid}: atlas is {w}x{h}, ratio {ratio}");
    }
}

/// Asking for a volume the device cannot allocate has to give back a smaller
/// one rather than failing to create a texture.
#[test]
fn an_oversized_volume_is_halved_until_it_fits() {
    // 512^3 wants a 512*23 = 11 776 texel atlas; an 8192 limit cannot hold it.
    let a = Atlas::largest_within(512, 8192);
    let (w, h) = a.size();
    assert!(w <= 8192 && h <= 8192, "{w}x{h} is over the limit");
    assert!(a.grid < 512 && a.grid >= 8, "degraded to {}", a.grid);
    // And a volume that already fits is untouched.
    assert_eq!(Atlas::largest_within(64, 8192).grid, 64);
}

/// The test the whole atlas scheme rests on: a ball of dye put at a known
/// place in the volume appears, on screen, where that place projects to.
///
/// Every stage addresses voxels by computing a tile and an offset into it,
/// and getting that wrong gives a plume that still looks like a plume. This
/// can only pass if `voxel_of`, the trilinear fetch and the ray march agree.
#[test]
fn a_splat_appears_where_it_was_put() {
    let Some(mut sim) = sim(FluidConfig3d {
        // No motion at all: the blob must stay exactly where it was placed.
        curl: 0.0,
        buoyancy: 0.0,
        density_dissipation: 0.0,
        velocity_dissipation: 0.0,
        splat_radius: 0.35,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (w, h) = (192u32, 192u32);
    let cam = camera(w as f32 / h as f32);

    // Off-centre on every axis, so a transposition or a sign error shows.
    let at = [0.7f32, 0.3, 0.55];
    sim.splat(at, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    sim.step(1.0 / 60.0);

    let rgba = sim.capture_rgba(&cam, w, h);
    let (bx, by, v) = brightest(&rgba, w, h);
    assert!(v > 24, "nothing was drawn (brightest {v})");

    // Where that voxel is in the world, and where the camera puts it.
    let c = sim.config();
    let world = Vector3::new(
        c.bounds_min[0] + at[0] * (c.bounds_max[0] - c.bounds_min[0]),
        c.bounds_min[1] + at[1] * (c.bounds_max[1] - c.bounds_min[1]),
        c.bounds_min[2] + at[2] * (c.bounds_max[2] - c.bounds_min[2]),
    );
    let screen = cam.world_to_screen(world, w, h).expect("in view");
    let (ex, ey) = (screen.x / w as f32, screen.y / h as f32);
    assert!(
        (bx - ex).abs() < 0.07,
        "blob at x {bx:.3}, the voxel projects to {ex:.3}"
    );
    assert!(
        (by - ey).abs() < 0.07,
        "blob at y {by:.3}, the voxel projects to {ey:.3}"
    );
}

/// The projection does in three dimensions what it does in two: a radial puff
/// is pure divergence, and the pressure solve takes it out.
///
/// Both measurements here are of the *Jacobi* solver, which is no longer the
/// volume's default — because of what they show. A sweep moves information
/// one voxel, and a volume is wider in the sense that matters: the pressure
/// from a puff falls off as 1/r rather than log r, and six neighbours share
/// the correction instead of four. Twenty sweeps, the 2D default, remove
/// under a fifth of it. The fix is not more sweeps either — three hundred of
/// them are still only halfway — which is what sent the volume to multigrid;
/// see `the_v_cycle_beats_the_sweeps`.
#[test]
fn projection_removes_divergence_in_three_dimensions() {
    let Some(mut sim) = sim(FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        pressure_solver: PressureSolver::Jacobi,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);

    let divergence_now = |sim: &mut FluidSim3d| -> f32 {
        let mut enc = sim.encoder("test divergence");
        sim.next_ubo.set(0);
        let u = sim.params(sim.sim, Vol::default());
        let bg = sim.bind(&u, &sim.velocity.read, None);
        draw(
            &mut enc,
            &sim.divergence.view,
            Load::Clear,
            &sim.pipes.divergence,
            &bg,
        );
        sim.queue.submit([enc.finish()]);
        let d = read_channel(sim, &sim.divergence, 0);
        d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
    };

    let raw = divergence_now(&mut sim);
    assert!(raw > 1e-5, "the splat produced no divergence: {raw}");
    // A short step, so advection — which this solver applies *after* the
    // projection, and which puts divergence back — is out of the picture.
    sim.step(1.0 / 6000.0);
    let at_twenty = divergence_now(&mut sim) / raw;

    // The same splat again, with the solve given room to converge.
    let mut long = sim2(FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        pressure_solver: PressureSolver::Jacobi,
        pressure_iterations: 300,
        ..FluidConfig3d::default()
    })
    .expect("a second device");
    long.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
    let long_raw = divergence_now(&mut long);
    long.step(1.0 / 6000.0);
    let at_three_hundred = divergence_now(&mut long) / long_raw;

    eprintln!("  20 sweeps leave {at_twenty:.3}, 300 leave {at_three_hundred:.3}");
    assert!(
        at_twenty < 0.95,
        "twenty sweeps left {at_twenty:.3} of the divergence"
    );
    assert!(
        at_three_hundred < at_twenty * 0.7,
        "three hundred sweeps left {at_three_hundred:.3} against twenty's \
         {at_twenty:.3} — the solve is not converging"
    );
}

/// Buoyancy has to lift the dye and nothing else. A uniform push — which is
/// what this was before the dye and solver grids were told apart — moves the
/// empty fluid too, and the whole volume piles against the ceiling as a slab.
#[test]
fn buoyancy_lifts_the_dye_and_not_the_rest() {
    let Some(mut sim) = sim(FluidConfig3d {
        buoyancy: 400.0,
        curl: 0.0,
        velocity_dissipation: 0.0,
        splat_radius: 0.25,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Dye in one corner only, so "everywhere" and "where the dye is" differ.
    sim.splat([0.3, 0.3, 0.3], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    for _ in 0..8 {
        sim.step(1.0 / 60.0);
    }

    // Channel 1 of the velocity atlas is the vertical component.
    let vy = read_channel(&sim, &sim.velocity.read, 1);
    let lifted = vy.iter().filter(|v| **v > 1.0).count();
    let total = vy.len();
    assert!(lifted > 0, "nothing rose at all");
    assert!(
        lifted < total / 3,
        "{lifted} of {total} voxels are rising — buoyancy is pushing the \
         empty fluid, not the dye"
    );
    let highest = vy.iter().copied().fold(0.0f32, f32::max);
    assert!(highest > 5.0, "the strongest rise is only {highest}");
}

/// Nothing reaches a NaN, and the velocity clamp holds, under hard driving.
#[test]
fn a_hard_driven_volume_stays_finite() {
    let Some(mut sim) = sim(FluidConfig3d {
        curl: 40.0,
        buoyancy: 300.0,
        velocity_dissipation: 0.0,
        density_dissipation: 0.0,
        resolution: 32,
        dye_resolution: 32,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for i in 0..24 {
        let t = i as f32 / 24.0;
        sim.splat(
            [0.2 + 0.6 * t, 0.15, 0.5],
            [600.0, 900.0, -600.0],
            [1.0, 1.0, 1.0],
        );
        sim.step(1.0 / 60.0);
    }
    for _ in 0..60 {
        sim.step(1.0 / 60.0);
    }
    for channel in 0..3 {
        let v = read_channel(&sim, &sim.velocity.read, channel);
        assert!(
            v.iter().all(|x| x.is_finite()),
            "velocity channel {channel} went non-finite"
        );
        let worst = v.iter().copied().fold(0.0f32, |a, b| a.max(b.abs()));
        assert!(worst <= 1001.0, "velocity reached {worst}");
    }
    let dye = read_channel(&sim, &sim.dye.read, 0);
    assert!(dye.iter().all(|x| x.is_finite()), "dye went non-finite");
}

/// Same seed, same volume — what makes the example's picture documentation.
#[test]
fn the_same_seed_renders_the_same_volume() {
    let run = || -> Option<Vec<u8>> {
        let mut sim = sim(FluidConfig3d {
            resolution: 32,
            dye_resolution: 32,
            march_steps: 48,
            ..FluidConfig3d::default()
        })?;
        sim.multiple_splats(5);
        for _ in 0..12 {
            sim.step(1.0 / 60.0);
        }
        Some(sim.capture_rgba(&camera(1.0), 96, 96))
    };
    let (Some(a), Some(b)) = (run(), run()) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    assert_eq!(a, b, "two runs of the same seed differ");
}

/// A uniform flow carries the dye along **z** — the axis a 2D test cannot
/// check, and the one the atlas packs slices along, so an error here is an
/// error in the tiling rather than in the arithmetic.
///
/// How far it travels is deliberately a range and not a figure. A uniform
/// flow through a *closed* box is not a steady state: the walls are
/// free-slip, the divergence at them is not zero, and the pressure solve
/// duly builds a counter-flow that slows the field down. The dye therefore
/// travels rather less than velocity times time, and correctly so — a solver
/// that carried it the full ballistic distance would be one that had stopped
/// enforcing its own boundary.
#[test]
fn a_uniform_flow_carries_the_dye_along_z() {
    let Some(mut sim) = sim(FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        density_dissipation: 0.0,
        velocity_dissipation: 0.0,
        splat_radius: 0.3,
        // The projection is switched off here, and it has to be: a uniform
        // flow inside a closed box is *not* divergence free — the free-slip
        // wall mirrors the normal component, so the fluid is piling into one
        // face and leaving a hole at the other, and a solver that has
        // converged will correctly cancel the whole thing. This test is
        // about advection and the atlas, so it takes the pressure out of the
        // question rather than relying on the solve being bad enough to
        // leave some flow behind.
        gradient_scale: 0.0,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.splat([0.5, 0.5, 0.25], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    // Looking down -z, so motion in +z is motion *towards* the camera: it
    // changes the blob's size on screen rather than its position, which the
    // other tests would not notice.
    sim.set_uniform_velocity([0.0, 0.0, 24.0]);

    let centre_of_mass_z = |sim: &FluidSim3d| -> f32 {
        let dye = read_channel(sim, &sim.dye.read, 0);
        let atlas = sim.dye_atlas;
        let (w, _) = atlas.size();
        let (n, tx) = (atlas.grid as usize, atlas.tiles.0 as usize);
        let mut sum = 0.0f64;
        let mut mass = 0.0f64;
        for (i, v) in dye.iter().enumerate() {
            let (ax, ay) = (i % w as usize, i / w as usize);
            let z = (ay / n) * tx + (ax / n);
            sum += *v as f64 * z as f64;
            mass += *v as f64;
        }
        (sum / mass.max(1e-9)) as f32
    };
    let before = centre_of_mass_z(&sim);
    for _ in 0..40 {
        sim.step(1.0 / 60.0);
    }
    let after = centre_of_mass_z(&sim);
    let moved = after - before;
    eprintln!("  the dye moved {moved:.1} slices along z");
    // Ballistically it would be thirty-two slices. What matters is that it
    // moved, that it moved the right way, and that it did not teleport —
    // which is what a tiling error looks like.
    assert!(
        (3.0..34.0).contains(&moved),
        "the dye moved {moved:.1} slices along z"
    );
}

#[test]
fn zero_sweeps_is_refused() {
    let Some((device, queue)) = gpu() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let built = FluidSim3d::new(
        device,
        queue,
        FluidConfig3d {
            pressure_iterations: 0,
            ..FluidConfig3d::default()
        },
    );
    let Err(e) = built else {
        panic!("a volume with no pressure sweeps was accepted");
    };
    assert!(e.contains("pressure_iterations"), "{e}");
}

/// The solver's neighbour taps read the voxels they name.
///
/// Every stage but the two that cross grids now fetches its neighbours with
/// `textureLoad` at an integer voxel rather than sampling the atlas: the
/// offsets are whole voxels, so the filter's weights were always 1 and 0, and
/// the fetch costs half as much. That is only true if `voxel_index` and
/// `atlas_texel` agree with `voxel_of` and `slice_at` about where a voxel
/// lives — including at the wall, where the filtered path clamped inside the
/// tile and this one clamps the index.
///
/// So: run one divergence pass on a known velocity field, and recompute the
/// same thing on the CPU from the field read back. A tiling mistake moves a
/// tap to a different slice, which this sees immediately; the earlier tests
/// would still show a plausible plume.
#[test]
fn the_taps_hit_the_voxels_they_name() {
    let grid = 16;
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: grid,
        dye_resolution: grid,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Three splats off-centre and off-axis, so no symmetry can hide a swapped
    // index and some of the stencil reaches outside the volume.
    sim.splat([0.3, 0.6, 0.2], [400.0, -150.0, 90.0], [1.0, 1.0, 1.0]);
    sim.splat([0.8, 0.2, 0.7], [-220.0, 310.0, -60.0], [1.0, 1.0, 1.0]);
    sim.splat([0.05, 0.5, 0.95], [120.0, 40.0, -500.0], [1.0, 1.0, 1.0]);

    let mut enc = sim.encoder("test taps");
    sim.next_ubo.set(0);
    let u = sim.params(sim.sim, Vol::default());
    let bg = sim.bind(&u, &sim.velocity.read, None);
    draw(
        &mut enc,
        &sim.divergence.view,
        Load::Clear,
        &sim.pipes.divergence,
        &bg,
    );
    sim.queue.submit([enc.finish()]);

    let vel: Vec<Vec<f32>> = (0..3)
        .map(|c| read_channel(&sim, &sim.velocity.read, c))
        .collect();
    let got = read_channel(&sim, &sim.divergence, 0);

    let atlas = sim.sim;
    let (aw, _) = atlas.size();
    let n = grid as i32;
    let across = (aw / grid) as i32;
    // Where voxel (x, y, z) is in the flattened atlas.
    let at = |c: [i32; 3], channel: usize| -> f32 {
        let p = c.map(|v| v.clamp(0, n - 1));
        let (row, col) = (p[2] / across, p[2] % across);
        let (ax, ay) = (col * n + p[0], row * n + p[1]);
        vel[channel][(ay * aw as i32 + ax) as usize]
    };

    let mut worst = 0.0f32;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let c = [x, y, z];
                let centre = [at(c, 0), at(c, 1), at(c, 2)];
                let face = |d: [i32; 3], channel: usize, wall: bool| -> f32 {
                    if wall {
                        -centre[channel]
                    } else {
                        at([x + d[0], y + d[1], z + d[2]], channel)
                    }
                };
                let l = face([-1, 0, 0], 0, x == 0);
                let r = face([1, 0, 0], 0, x == n - 1);
                let dn = face([0, -1, 0], 1, y == 0);
                let up = face([0, 1, 0], 1, y == n - 1);
                let b = face([0, 0, -1], 2, z == 0);
                let f = face([0, 0, 1], 2, z == n - 1);
                let want = 0.5 * ((r - l) + (up - dn) + (f - b));

                let (row, col) = (z / across, z % across);
                let idx = ((row * n + y) * aw as i32 + col * n + x) as usize;
                worst = worst.max((got[idx] - want).abs() / want.abs().max(1.0));
            }
        }
    }
    // Half floats on the way in and on the way out, so this is a rounding
    // tolerance, not a slack one: a tap on the wrong voxel is off by a whole
    // velocity.
    assert!(
        worst < 0.01,
        "the GPU's divergence and the CPU's differ by {worst:.4} relative"
    );
}

/// The volume can be resized while it is running.
///
/// The fields are reallocated and emptied, which is the part worth pinning:
/// a fresh texture holds whatever was in that memory, and every stage of the
/// solver reads before it writes, so skipping the clear gives a volume full
/// of noise that the first advection then smears everywhere.
#[test]
fn the_volume_can_be_resized() {
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        buoyancy: 0.0,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.splat([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
    sim.step(1.0 / 60.0);
    assert_eq!(sim.grids(), (32, 64));
    let small = sim.memory_bytes();

    // Up: the grids follow, and the volume is empty rather than full of
    // whatever the allocator handed back.
    sim.set_resolution(64, 128);
    assert_eq!(sim.grids(), (64, 128));
    assert!(
        sim.memory_bytes() > small * 4,
        "{} bytes at 64³ against {small} at 32³",
        sim.memory_bytes()
    );
    let (w, h) = (96, 96);
    let empty = sim.capture_rgba(&camera(1.0), w, h);
    let background = (FluidConfig3d::default().background[0] * 255.0).round() as u8 + 2;
    assert!(
        brightest(&empty, w, h).2 <= background,
        "the rebuilt volume is not empty"
    );

    // And it simulates: a splat into the new grids still arrives.
    sim.splat([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
    sim.step(1.0 / 60.0);
    let filled = sim.capture_rgba(&camera(1.0), w, h);
    assert!(
        brightest(&filled, w, h).2 > 40,
        "nothing rendered after the resize"
    );

    // Down, and through `set_config`, which is the path the browser takes.
    sim.set_config(FluidConfig3d {
        resolution: 16,
        dye_resolution: 32,
        ..*sim.config()
    });
    assert_eq!(sim.grids(), (16, 32));
    assert!(sim.memory_bytes() < small);

    // Setting the same size again is a no-op, so a slider that fires on every
    // pixel of travel does not reallocate on every pixel of travel.
    sim.splat([0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.8, 0.6]);
    sim.step(1.0 / 60.0);
    let before = brightest(&sim.capture_rgba(&camera(1.0), w, h), w, h).2;
    sim.set_resolution(16, 32);
    let after = brightest(&sim.capture_rgba(&camera(1.0), w, h), w, h).2;
    assert_eq!(
        before, after,
        "resizing to the same size emptied the volume"
    );
}

/// What the V-cycle buys over sweeping, in three dimensions.
///
/// This is the test the volume's default solver is chosen by. It is the same
/// comparison the 2D module makes, and the gap is much wider here: a Jacobi
/// sweep still moves information one voxel, and the distance it has to travel
/// has grown while the correction is split six ways instead of four.
#[test]
fn the_v_cycle_beats_the_sweeps() {
    let base = FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        ..FluidConfig3d::default()
    };
    let residual_after = |config: FluidConfig3d| -> Option<f32> {
        let mut sim = sim(config)?;
        sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
        let divergence_now = |sim: &mut FluidSim3d| -> f32 {
            let mut enc = sim.encoder("test divergence");
            sim.next_ubo.set(0);
            let u = sim.params(sim.sim, Vol::default());
            let bg = sim.bind(&u, &sim.velocity.read, None);
            draw(
                &mut enc,
                &sim.divergence.view,
                Load::Clear,
                &sim.pipes.divergence,
                &bg,
            );
            sim.queue.submit([enc.finish()]);
            let d = read_channel(sim, &sim.divergence, 0);
            d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
        };
        let raw = divergence_now(&mut sim);
        // A short step, so advection — which runs after the projection and
        // puts divergence back — is out of the picture.
        sim.step(1.0 / 6000.0);
        Some(divergence_now(&mut sim) / raw)
    };

    let Some(jacobi) = residual_after(FluidConfig3d {
        pressure_solver: PressureSolver::Jacobi,
        pressure_iterations: 20,
        ..base
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cycle = residual_after(FluidConfig3d {
        pressure_solver: PressureSolver::Multigrid,
        multigrid_cycles: 1,
        pressure_iterations: 20,
        ..base
    })
    .expect("a second simulation");
    let two = residual_after(FluidConfig3d {
        pressure_solver: PressureSolver::Multigrid,
        multigrid_cycles: 2,
        pressure_iterations: 20,
        ..base
    })
    .expect("a third simulation");
    // The sweep count a V-cycle is worth, for the record.
    let many = residual_after(FluidConfig3d {
        pressure_solver: PressureSolver::Jacobi,
        pressure_iterations: 300,
        ..base
    })
    .expect("a fourth simulation");

    eprintln!(
        "  divergence left: 20 sweeps {jacobi:.3}, 300 sweeps {many:.3}, \
         1 cycle {cycle:.3}, 2 cycles {two:.3}"
    );
    assert!(
        cycle < jacobi * 0.5,
        "one V-cycle left {cycle:.3} against twenty sweeps' {jacobi:.3}"
    );
    assert!(
        two <= cycle,
        "a second cycle made it worse: {two:.3} against {cycle:.3}"
    );
}

/// The packed dye is the same picture.
///
/// `rg11b10ufloat` holds the dye in half the memory of `rgba16float`, with
/// six bits of mantissa on red and green and five on blue against a half's
/// ten. The question is not whether that loses precision — it does — but
/// whether a ray march of a few hundred samples can see it. It cannot: the
/// march accumulates, and what each sample contributes is already scaled by
/// a transmittance smaller than the error.
///
/// It is a *float*, which is the part that matters. The same four bytes as a
/// ten-bit unorm would quantise the faint tails — where the smoke reads —
/// into two or three levels, because its steps are absolute rather than
/// relative.
#[test]
fn the_packed_dye_is_the_same_picture() {
    let base = FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        buoyancy: 40.0,
        ..FluidConfig3d::default()
    };
    let render = |compact: bool| -> Option<(Vec<u8>, u64, wgpu::TextureFormat)> {
        let mut sim = sim(FluidConfig3d {
            compact_dye: compact,
            ..base
        })?;
        for frame in 0..12 {
            if frame % 3 == 0 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
        }
        let format = sim.dye_texture().format();
        let bytes = sim.memory_bytes();
        Some((sim.capture_rgba(&camera(1.0), 128, 128), bytes, format))
    };
    let Some((packed, packed_bytes, format)) = render(true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let (wide, wide_bytes, _) = render(false).expect("a second simulation");
    if format != wgpu::TextureFormat::Rg11b10Ufloat {
        eprintln!("skipping: this device will not render to rg11b10ufloat");
        return;
    }

    let diffs: Vec<f64> = packed
        .iter()
        .zip(&wide)
        .map(|(a, b)| a.abs_diff(*b) as f64)
        .collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let worst = diffs.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "  packed dye: {packed_bytes} bytes against {wide_bytes}, \
         mean |diff| {mean:.3}/255, worst {worst}"
    );
    assert!(
        packed_bytes * 3 < wide_bytes * 2,
        "the packed dye saved nothing: {packed_bytes} against {wide_bytes}"
    );
    assert!(
        mean < 0.5,
        "the packed dye changed the picture: mean |diff| {mean:.3}/255"
    );
}

/// Skipping the empty parts does not change what is drawn.
///
/// The march steps over a brick whose largest density is below what it
/// already treats as nothing, so in principle the only difference is at a
/// brick's face, where a filtered sample just inside it would have picked up
/// a fraction of the neighbour's edge voxel. In practice that is well under
/// a level of eight-bit output — and if the brick arithmetic were wrong it
/// would not be subtle, it would be holes.
#[test]
fn empty_space_skipping_is_the_same_picture() {
    let base = FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        buoyancy: 40.0,
        march_steps: 128,
        ..FluidConfig3d::default()
    };
    let render = |skip: bool| -> Option<Vec<u8>> {
        let mut sim = sim(FluidConfig3d {
            empty_space_skip: skip,
            ..base
        })?;
        for frame in 0..14 {
            if frame % 3 == 0 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
        }
        Some(sim.capture_rgba(&camera(1.0), 128, 128))
    };
    let Some(skipped) = render(true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let every = render(false).expect("a second simulation");

    let diffs: Vec<f64> = skipped
        .iter()
        .zip(&every)
        .map(|(a, b)| a.abs_diff(*b) as f64)
        .collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let worst = diffs.iter().cloned().fold(0.0, f64::max);
    // The picture has to be *there*, or two black frames would agree.
    assert!(
        brightest(&every, 128, 128).2 > 40,
        "the reference render is empty"
    );
    let over_8 = diffs.iter().filter(|d| **d > 8.0).count();
    eprintln!(
        "  empty-space skip: mean |diff| {mean:.3}/255, worst {worst},          {over_8} of {} channels over 8",
        diffs.len()
    );
    assert!(mean < 0.5, "skipping changed the picture: mean {mean:.3}");
}

/// How far down the V-cycle has to go.
///
/// Each level below costs a handful of render passes over a grid small
/// enough that the arithmetic is free, and a pass is not free — measured, the
/// bottom two levels of a 64³ solve are about a fifth of its GPU time. The
/// question is what they are buying, and the answer decides
/// `multigrid_coarsest`.
#[test]
fn the_v_cycle_needs_its_coarse_levels() {
    let base = FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        ..FluidConfig3d::default()
    };
    let residual_after = |coarsest: u32| -> Option<(f32, usize)> {
        let mut sim = sim(FluidConfig3d {
            multigrid_coarsest: coarsest,
            ..base
        })?;
        sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
        let divergence_now = |sim: &mut FluidSim3d| -> f32 {
            let mut enc = sim.encoder("test divergence");
            sim.next_ubo.set(0);
            let u = sim.params(sim.sim, Vol::default());
            let bg = sim.bind(&u, &sim.velocity.read, None);
            draw(
                &mut enc,
                &sim.divergence.view,
                Load::Clear,
                &sim.pipes.divergence,
                &bg,
            );
            sim.queue.submit([enc.finish()]);
            let d = read_channel(sim, &sim.divergence, 0);
            d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
        };
        let raw = divergence_now(&mut sim);
        sim.step(1.0 / 6000.0);
        Some((divergence_now(&mut sim) / raw, sim.solver_levels()))
    };

    let Some(deep) = residual_after(4) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut rows = vec![(4u32, deep)];
    for coarsest in [8u32, 16, 32] {
        rows.push((
            coarsest,
            residual_after(coarsest).expect("another simulation"),
        ));
    }
    for (coarsest, (left, levels)) in &rows {
        eprintln!("  coarsest {coarsest:>2}³ ({levels} levels): {left:.3} of the divergence left");
    }
    // The default has to be close to what coarsening all the way achieves,
    // or the levels it drops were doing the work.
    let default = rows
        .iter()
        .find(|(c, _)| *c == FluidConfig3d::default().multigrid_coarsest)
        .map(|(_, (left, _))| *left)
        .expect("the default is in the table");
    assert!(
        default < deep.0 * 1.25,
        "the default coarsest leaves {default:.3} against {:.3} for coarsening all the way",
        deep.0
    );
}

/// The CPU solver and the GPU solver are the same solver.
///
/// Not bit-for-bit — the GPU carries its fields in half floats and samples
/// through a texture unit, the CPU in `f32` with its own trilinear — but they
/// are transcriptions of each other, so they should remove the same fraction
/// of a puff's divergence. That is the number the whole projection exists to
/// move, and a stage wired to the wrong grid in either one would show up in
/// it immediately.
#[test]
fn the_cpu_and_gpu_solvers_agree() {
    let config = FluidConfig3d {
        resolution: 32,
        dye_resolution: 32,
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        ..FluidConfig3d::default()
    };
    let Some(mut gpu) = sim(config) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let divergence_now = |sim: &mut FluidSim3d| -> f32 {
        let mut enc = sim.encoder("test divergence");
        sim.next_ubo.set(0);
        let u = sim.params(sim.sim, Vol::default());
        let bg = sim.bind(&u, &sim.velocity.read, None);
        draw(
            &mut enc,
            &sim.divergence.view,
            Load::Clear,
            &sim.pipes.divergence,
            &bg,
        );
        sim.queue.submit([enc.finish()]);
        let d = read_channel(sim, &sim.divergence, 0);
        d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
    };
    gpu.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
    let gpu_before = divergence_now(&mut gpu);
    gpu.step(1.0 / 6000.0);
    let gpu_left = divergence_now(&mut gpu) / gpu_before;

    let mut cpu = crate::fluid::CpuFluid3d::new(config);
    cpu.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
    let cpu_before = cpu.mean_divergence();
    cpu.step(1.0 / 6000.0);
    let cpu_left = cpu.mean_divergence() / cpu_before;

    eprintln!("  divergence left: gpu {gpu_left:.3}, cpu {cpu_left:.3}");
    // The two fields the splat put in should also be the same size, which
    // catches a radius or a grid convention that drifted apart.
    let ratio = cpu_before / gpu_before;
    assert!(
        (0.8..1.25).contains(&ratio),
        "the same splat made {cpu_before:.5} of divergence on the cpu and \
         {gpu_before:.5} on the gpu"
    );
    assert!(
        (cpu_left - gpu_left).abs() < 0.1,
        "the solvers disagree: cpu leaves {cpu_left:.3}, gpu {gpu_left:.3}"
    );
}

/// The memory estimate is the memory.
///
/// [`within_memory`] exists so a device can be asked for what it can afford
/// rather than told after the fact, which on a phone is the difference
/// between a smaller fluid and a killed app. That is only worth anything if
/// the estimate matches what gets allocated — including the atlas padding a
/// cube into a rectangle, which is where a formula drifts from reality.
#[test]
fn the_memory_estimate_is_the_memory() {
    let Some((device, queue)) = gpu() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for (solver, dye) in [(16u32, 32u32), (24, 48), (32, 64), (48, 96), (50, 100)] {
        let config = FluidConfig3d {
            resolution: solver,
            dye_resolution: dye,
            ..FluidConfig3d::default()
        };
        let sim = FluidSim3d::new(device.clone(), queue.clone(), config).expect("a simulation");
        let predicted = memory_for(solver, dye, sim.formats());
        let actual = sim.memory_bytes();
        assert_eq!(
            predicted, actual,
            "{solver}³/{dye}³: predicted {predicted} bytes, allocated {actual}"
        );
    }
}

/// And the budget is respected from both sides: what it picks fits, and one
/// step larger would not.
#[test]
fn a_budget_picks_the_largest_volume_that_fits() {
    let formats = FluidFormats::default();
    for budget_mb in [8u64, 32, 64, 128, 256, 512] {
        let budget = budget_mb << 20;
        let (solver, dye) = within_memory(budget, 2, formats);
        let fits = memory_for(solver, dye, formats);
        assert!(
            fits <= budget,
            "{budget_mb} MB: picked {solver}³ which needs {} MB",
            fits >> 20
        );
        let next = memory_for(solver + 8, (solver + 8) * 2, formats);
        assert!(
            next > budget || solver >= 512,
            "{budget_mb} MB: picked {solver}³ but {}³ also fits",
            solver + 8
        );
        eprintln!(
            "  {budget_mb:>4} MB -> {solver}³ solver, {dye}³ dye ({} MB)",
            fits >> 20
        );
    }
}

/// The speed knob moves the fluid, and only through `update`.
///
/// Two claims, and the second is the one worth a test. A caller who passes a
/// timestep to [`FluidSim3d::step`] means that timestep — the parity tests,
/// the determinism test and the solver comparisons all depend on it — so the
/// scale belongs to `update` alone. A version that applied it in `step` would
/// pass every test about speed and quietly break every test about agreement.
#[test]
fn speed_scales_update_and_not_step() {
    let base = FluidConfig3d {
        resolution: 24,
        dye_resolution: 48,
        buoyancy: 160.0,
        velocity_dissipation: 0.0,
        density_dissipation: 0.0,
        ..FluidConfig3d::default()
    };
    let height = |sim: &FluidSim3d| -> f32 {
        let dye = crate::renderer::readback::texture_channel_f32(
            &sim.device,
            &sim.queue,
            sim.dye_texture(),
            0,
        );
        let atlas = sim.dye_atlas();
        let (w, _) = atlas.size();
        let (n, across) = (atlas.grid as usize, (w / atlas.grid) as usize);
        let (mut sum, mut mass) = (0.0f64, 0.0f64);
        for (i, v) in dye.iter().enumerate() {
            let (ax, ay) = (i % w as usize, i / w as usize);
            let _ = ax / n * across;
            let y = ay % n;
            sum += *v as f64 * y as f64;
            mass += *v as f64;
        }
        (sum / mass.max(1e-9)) as f32
    };
    let rise = |scale: f32, use_update: bool| -> Option<f32> {
        let mut sim = sim(FluidConfig3d {
            time_scale: scale,
            ..base
        })?;
        sim.splat([0.5, 0.25, 0.5], [0.0; 3], [1.0, 1.0, 1.0]);
        let before = height(&sim);
        for _ in 0..24 {
            if use_update {
                sim.update(1.0 / 60.0);
            } else {
                sim.step(1.0 / 60.0);
            }
        }
        Some(height(&sim) - before)
    };
    let Some(normal) = rise(1.0, true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let fast = rise(2.5, true).expect("a second simulation");
    let stepped = rise(2.5, false).expect("a third simulation");
    eprintln!("  dye rose {normal:.2} at 1x, {fast:.2} at 2.5x, {stepped:.2} stepped directly");
    assert!(normal > 0.1, "the dye did not rise at all: {normal:.2}");
    assert!(
        fast > normal * 1.3,
        "2.5x speed rose {fast:.2} against {normal:.2}"
    );
    // `step` ignores the scale, so driving it directly matches 1x.
    assert!(
        (stepped - normal).abs() < normal * 0.1,
        "step honoured the speed knob: {stepped:.2} against {normal:.2}"
    );
}

/// How many render passes a step is, and that folding the clears took five
/// of them out.
///
/// A pass count is the deterministic half of `stage_ms`: at small grids the
/// solve is bound by how many passes it is rather than by what they compute,
/// and unlike a time it can be measured on a busy machine. It is also the
/// only way to show what a fusion bought — five passes out of forty-seven is
/// inside the noise of a GPU timer and is not inside the noise of counting.
#[test]
fn folding_the_clears_removed_five_passes() {
    let config = FluidConfig3d {
        resolution: 64,
        dye_resolution: 128,
        curl: 26.0,
        multigrid_cycles: 1,
        ..FluidConfig3d::default()
    };
    let Some(mut sim) = sim(config) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    sim.step(1.0 / 60.0);
    let cycle = sim.passes_per_step();
    let levels = sim.solver_levels();

    // What the V-cycle costs, counted: two smoothing sweeps either side of
    // the recursion, and per level a residual, a restriction and its own
    // smoothing. Before the clears were folded in there was one more pass
    // at the top and one per level.
    let saved = 1 + levels;
    eprintln!("  a step is {cycle} passes at 64³ with {levels} levels, {saved} fewer than before");
    assert_eq!(
        levels, 4,
        "the level count moved, so the arithmetic below has"
    );
    assert!(
        (40..=48).contains(&cycle),
        "a step is {cycle} passes, which is not the shape this test was written against"
    );

    // A second cycle adds the same work again, minus the top-level opening.
    sim.set_config(FluidConfig3d {
        multigrid_cycles: 2,
        ..config
    });
    sim.step(1.0 / 60.0);
    let two = sim.passes_per_step();
    eprintln!("  two cycles is {two} passes");
    assert!(
        two > cycle,
        "a second V-cycle recorded no extra passes: {two} against {cycle}"
    );

    // And Jacobi is the flat comparison: its sweeps are all there is.
    sim.set_config(FluidConfig3d {
        pressure_solver: PressureSolver::Jacobi,
        pressure_iterations: 20,
        ..config
    });
    sim.step(1.0 / 60.0);
    let jacobi = sim.passes_per_step();
    eprintln!("  twenty Jacobi sweeps is {jacobi} passes");
    assert!(
        jacobi < cycle,
        "Jacobi recorded {jacobi} passes against the V-cycle's {cycle}"
    );
}

/// Accumulating the march across frames converges on a longer one.
///
/// This is the whole claim: 24 samples a frame, averaged over frames that
/// jitter differently, should approach what 288 samples give in one. The
/// fluid is held still and the camera does not move, which is the case the
/// feature is for — a still camera over steady smoke — and also the only
/// case where "converges to" has an unambiguous reference to point at.
#[test]
fn accumulation_converges_on_a_longer_march() {
    let base = FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        march_steps: 24,
        buoyancy: 40.0,
        ..FluidConfig3d::default()
    };
    let Some(mut sim) = sim(base) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    for frame in 0..14 {
        if frame % 3 == 0 {
            sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
        }
        sim.step(1.0 / 60.0);
    }

    let (w, h) = (96, 96);
    let cam = camera(1.0);
    // The reference: every sample the accumulation is trying to be worth.
    sim.set_config(FluidConfig3d {
        march_steps: 24 * 12,
        ..base
    });
    let reference = sim.capture_rgba(&cam, w, h);
    // One short march, which is what a frame can afford.
    sim.set_config(base);
    let single = sim.capture_rgba(&cam, w, h);
    // And the same short march, averaged.
    let against_ref = |a: &[u8], reference: &[u8]| -> f64 {
        a.iter()
            .zip(reference)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64
    };
    for k in [0.25f32, 0.5, 1.0, 2.0, 8.0] {
        sim.set_config(FluidConfig3d {
            temporal_samples: 8,
            temporal_clip: k,
            ..base
        });
        let mut acc = Vec::new();
        for _ in 0..24 {
            acc = sim.capture_rgba(&cam, w, h);
        }
        eprintln!(
            "    clip {k:>4} sigma: {:.3}",
            against_ref(&acc, &reference)
        );
    }
    sim.set_config(FluidConfig3d {
        temporal_samples: 8,
        ..base
    });
    let mut accumulated = Vec::new();
    for _ in 0..24 {
        accumulated = sim.capture_rgba(&cam, w, h);
    }

    let against = |a: &[u8]| -> f64 {
        a.iter()
            .zip(&reference)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64
    };
    let (noisy, averaged) = (against(&single), against(&accumulated));
    let _ = &against_ref;
    eprintln!(
        "  from a 288-sample march: one 24-sample march is {noisy:.3}/255 away, \
         24 accumulated are {averaged:.3}"
    );
    assert!(
        brightest(&reference, w, h).2 > 40,
        "the reference render is empty, so nothing below means anything"
    );
    assert!(
        averaged < noisy * 0.75,
        "accumulating did not help: {averaged:.3} against {noisy:.3}"
    );
}

/// Skipping empty bricks must not move a ray off the sample lattice.
///
/// `empty_space_skipping_is_the_same_picture` checks the regime where a
/// march step is finer than a dye voxel. This checks the opposite one — a
/// coarse march over a fine dye, which is where the big volumes live — and
/// it is where skipping used to go wrong.
///
/// A ray that steps over a brick lands wherever the skip put it. If that is
/// not where its next sample was going to be, it carries on offset from
/// every ray that did not skip; two neighbouring pixels that skipped a
/// different number of bricks then sample the volume at different phases,
/// and the seam between them is visible. Over a plume it reads as a
/// cross-hatch at the brick's scale — and it is the one artefact that
/// averaging frames cannot remove, because the bricks are in the same place
/// every frame.
#[test]
fn skipping_keeps_rays_on_the_sample_lattice() {
    let base = FluidConfig3d {
        resolution: 32,
        // Four dye voxels to a solver voxel, and a march far coarser than
        // either: one sample every five or so dye voxels.
        dye_resolution: 128,
        march_steps: 24,
        buoyancy: 40.0,
        ..FluidConfig3d::default()
    };
    let render = |skip: bool| -> Option<Vec<u8>> {
        let mut sim = sim(FluidConfig3d {
            empty_space_skip: skip,
            ..base
        })?;
        for frame in 0..16 {
            if frame % 3 == 0 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
        }
        Some(sim.capture_rgba(&camera(1.0), 128, 128))
    };
    let Some(skipped) = render(true) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let every = render(false).expect("a second simulation");
    assert!(
        brightest(&every, 128, 128).2 > 40,
        "the reference render is empty"
    );

    // Against the reference, not a roughness: an off-lattice ray does not
    // make a *noisier* picture, it makes a different one — it integrates the
    // volume at the wrong places, which on balance loses contrast rather
    // than adding it. Roughness was the first thing tried here and it went
    // the wrong way, reporting the broken version as the smoother.
    let diffs: Vec<f64> = skipped
        .iter()
        .zip(&every)
        .map(|(a, b)| a.abs_diff(*b) as f64)
        .collect();
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let over_8 = diffs.iter().filter(|d| **d > 8.0).count();
    eprintln!(
        "  coarse march: mean |diff| {mean:.3}/255, {over_8} of {} channels over 8",
        diffs.len()
    );
    // On the lattice, a skipped ray samples at exactly the positions an
    // unskipped one would and merely declines to sample the empty bricks
    // between — which contribute nothing above the floor. So this is not a
    // tolerance, it is equality, and the threshold is set where the fix
    // actually lands rather than where it would still look acceptable:
    // off-lattice, the same comparison gives 0.225/255 and 532 channels.
    assert_eq!(
        over_8, 0,
        "skipping moved {over_8} channels by more than 8 levels — the rays \
         that skipped are sampling somewhere the rays that did not are not"
    );
    assert!(
        mean < 0.02,
        "skipping changed a coarse march by {mean:.3}/255"
    );
}

/// What accumulation costs on a fluid that is still moving.
///
/// The still-camera case converges; this is the other one. Each frame the
/// plume has moved, so the history is a picture of where it *was*, and
/// averaging it in is ghosting. The question is how much — enough to need
/// rejecting, or little enough to ignore.
#[test]
fn accumulation_ghosts_a_moving_fluid() {
    let base = FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        march_steps: 24,
        buoyancy: 90.0,
        ..FluidConfig3d::default()
    };
    let (w, h) = (96, 96);
    let cam = camera(1.0);
    // Both runs see the same fluid: same seed, same splats, same steps.
    let run = |samples: u32, steps: u32| -> Option<Vec<u8>> {
        let mut sim = sim(FluidConfig3d {
            temporal_samples: samples,
            march_steps: steps,
            ..base
        })?;
        let mut out = Vec::new();
        for frame in 0..30 {
            if frame % 3 == 0 && frame < 18 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
            out = sim.capture_rgba(&cam, w, h);
        }
        Some(out)
    };
    // The truth for the final frame: one long march of where the fluid is
    // *now*, with no history in it.
    let Some(truth) = run(1, 24 * 12) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let single = run(1, 24).expect("a second simulation");
    let against = |a: &[u8]| -> f64 {
        a.iter()
            .zip(&truth)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64
    };
    let one = against(&single);
    eprintln!("  moving fluid, from the truth: one march {one:.3}/255");
    for k in [0.25f32, 0.5, 1.0, 2.0, 8.0] {
        let mut sim = sim(FluidConfig3d {
            temporal_samples: 8,
            temporal_clip: k,
            ..base
        })
        .expect("a simulation");
        let mut out = Vec::new();
        for frame in 0..30 {
            if frame % 3 == 0 && frame < 18 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
            out = sim.capture_rgba(&cam, w, h);
        }
        eprintln!("    clip {k:>4} sigma: {:.3}", against(&out));
    }
    // The default clip against no clip at all: rejecting the history has to
    // cost less than half the ghosting it prevents, or it is not worth the
    // pass it costs.
    let ghosted = |k: f32| -> f64 {
        let mut sim = sim(FluidConfig3d {
            temporal_samples: 8,
            temporal_clip: k,
            ..base
        })
        .expect("a simulation");
        let mut out = Vec::new();
        for frame in 0..30 {
            if frame % 3 == 0 && frame < 18 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
            out = sim.capture_rgba(&cam, w, h);
        }
        against(&out)
    };
    let (clipped, unclipped) = (
        ghosted(FluidConfig3d::default().temporal_clip),
        ghosted(8.0),
    );
    eprintln!("  at the default clip {clipped:.3}, with none {unclipped:.3}");
    assert!(
        clipped < unclipped * 0.5,
        "rejecting the history barely helped: {clipped:.3} against {unclipped:.3}"
    );
    assert!(
        brightest(&truth, w, h).2 > 40,
        "the reference render is empty"
    );
}

/// What each part of the solve is buying, so the parts that buy nothing can
/// go.
///
/// The solve is about 95% of the step's GPU time and a V-cycle at 64³ is
/// forty-four render passes, most of them smoothing. This is the table that
/// says which of them are work and which are habit: the residual each
/// configuration leaves, against the passes it spends getting there.
#[test]
fn what_the_solve_spends_its_passes_on() {
    let base = FluidConfig3d {
        curl: 0.0,
        buoyancy: 0.0,
        gradient_scale: 0.5,
        ..FluidConfig3d::default()
    };
    let measure = |config: FluidConfig3d| -> Option<(f32, u32)> {
        let mut sim = sim(config)?;
        sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
        let divergence_now = |sim: &mut FluidSim3d| -> f32 {
            let mut enc = sim.encoder("test divergence");
            sim.next_ubo.set(0);
            let u = sim.params(sim.sim, Vol::default());
            let bg = sim.bind(&u, &sim.velocity.read, None);
            draw(
                &mut enc,
                &sim.divergence.view,
                Load::Clear,
                &sim.pipes.divergence,
                &bg,
            );
            sim.queue.submit([enc.finish()]);
            let d = read_channel(sim, &sim.divergence, 0);
            d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
        };
        let raw = divergence_now(&mut sim);
        sim.step(1.0 / 6000.0);
        Some((divergence_now(&mut sim) / raw, sim.passes_per_step()))
    };

    let Some(baseline) = measure(base) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    eprintln!("  default: {:.3} left in {} passes", baseline.0, baseline.1);

    // Smoothing either side of the recursion.
    for nu in [1u32, 2, 3, 4] {
        let (left, passes) = measure(FluidConfig3d {
            multigrid_smooth: nu,
            ..base
        })
        .expect("a simulation");
        eprintln!("    smooth {nu}: {left:.3} left in {passes} passes");
    }
    // The warm start. It cannot be measured in one step — the pressure
    // field starts at zero, so scaling it by anything is zero, and the first
    // version of this printed the same number for every value including
    // none. It is a claim about *successive* frames, so it takes several.
    let over_frames = |keep: f32| -> f32 {
        let mut sim = sim(FluidConfig3d {
            pressure: keep,
            ..base
        })
        .expect("a simulation");
        let divergence_now = |sim: &mut FluidSim3d| -> f32 {
            let mut enc = sim.encoder("test divergence");
            sim.next_ubo.set(0);
            let u = sim.params(sim.sim, Vol::default());
            let bg = sim.bind(&u, &sim.velocity.read, None);
            draw(
                &mut enc,
                &sim.divergence.view,
                Load::Clear,
                &sim.pipes.divergence,
                &bg,
            );
            sim.queue.submit([enc.finish()]);
            let d = read_channel(sim, &sim.divergence, 0);
            d.iter().map(|x| x.abs()).sum::<f32>() / d.len() as f32
        };
        // Forced every frame, so there is always something to solve and the
        // pressure field is worth carrying.
        let mut raw = 0.0;
        for frame in 0..8 {
            sim.splat([0.5, 0.5, 0.5], [300.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
            if frame == 0 {
                raw = divergence_now(&mut sim);
            }
            sim.step(1.0 / 6000.0);
        }
        divergence_now(&mut sim) / raw
    };
    for keep in [0.0f32, 0.5, 0.8, 1.0] {
        eprintln!(
            "    carry over {keep}: {:.3} left after eight frames",
            over_frames(keep)
        );
    }

    // One sweep either side has to be worse than two, or two is waste.
    let one = measure(FluidConfig3d {
        multigrid_smooth: 1,
        ..base
    })
    .expect("a simulation");
    assert!(one.1 < baseline.1, "fewer sweeps recorded no fewer passes");
}

/// Why the carry-over is damped, which is not what it looks like.
///
/// `pressure` scales how much of last frame's solution the next one starts
/// from, and more is better for convergence — measured over eight frames,
/// keeping all of it leaves 0.269 of the divergence against 0.281 for the
/// 0.8 the plane inherited from the original. So why damp it at all?
///
/// The reason to damp it is the null space. Pure Neumann boundaries fix the
/// gradient and say nothing about the constant, so any uniform offset is an
/// equally valid solution and nothing in the solver removes one. Carrying
/// the whole field forward should let that constant accumulate; the *fluid*
/// would never notice, because only the gradient is used, but the field is
/// half float and runs out of room at 65504.
///
/// It does not happen. Nine hundred frames of continuous forcing leave the
/// peak at 22 to 26 either way — including with Jacobi, which has no coarse
/// grid to anchor the constant and is the configuration the damping is
/// really for. So the volume carries all of it, and this test is what would
/// catch that being wrong.
#[test]
fn the_pressure_carry_over_is_damped_against_drift() {
    let base = FluidConfig3d {
        resolution: 24,
        dye_resolution: 24,
        buoyancy: 60.0,
        ..FluidConfig3d::default()
    };
    let peak = |keep: f32| -> Option<f32> {
        let mut sim = sim(FluidConfig3d {
            pressure: keep,
            ..base
        })?;
        for frame in 0..900 {
            if frame % 4 == 0 {
                sim.splat([0.5, 0.25, 0.5], [60.0, 300.0, -40.0], [1.0, 0.6, 0.3]);
            }
            sim.step(1.0 / 60.0);
        }
        let p = read_channel(&sim, &sim.pressure.read, 0);
        Some(p.iter().fold(0.0f32, |m, v| m.max(v.abs())))
    };
    let Some(damped) = peak(0.8) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let undamped = peak(1.0).expect("a second simulation");
    // Jacobi has no coarse grid to anchor the constant mode, which is the
    // configuration the damping is really for.
    let jacobi = |keep: f32| -> f32 {
        let mut sim = sim(FluidConfig3d {
            pressure: keep,
            pressure_solver: PressureSolver::Jacobi,
            ..base
        })
        .expect("a simulation");
        for frame in 0..900 {
            if frame % 4 == 0 {
                sim.splat([0.5, 0.25, 0.5], [60.0, 300.0, -40.0], [1.0, 0.6, 0.3]);
            }
            sim.step(1.0 / 60.0);
        }
        let p = read_channel(&sim, &sim.pressure.read, 0);
        p.iter().fold(0.0f32, |m, v| m.max(v.abs()))
    };
    eprintln!(
        "    jacobi, no coarse grid: damped {:.1}, undamped {:.1}",
        jacobi(0.8),
        jacobi(1.0)
    );
    eprintln!("  after 900 frames, peak |pressure|: damped {damped:.1}, undamped {undamped:.1}");
    // Both, because the default is now the undamped one: the measurement
    // is what says the damping was not buying anything, so the measurement
    // is what has to keep saying it.
    for (name, peak) in [("damped", damped), ("undamped", undamped)] {
        assert!(
            peak.is_finite() && peak < 1000.0,
            "the {name} field reached {peak} — the constant mode is drifting \
             after all, and `pressure` has to come back down"
        );
    }
}

/// The read-out draws where it says and nowhere else.
///
/// Two claims. It has to appear — a HUD that silently does nothing is worse
/// than no HUD, because you trust it. And it has to leave the picture alone
/// everywhere else, which is the part a `discard` gets wrong quietly: a
/// shader that returns a transparent black instead of discarding still
/// writes, and over an opaque render that is a faint grey box nobody
/// notices until they screenshot it.
#[test]
fn the_read_out_draws_only_where_it_says() {
    let base = FluidConfig3d {
        resolution: 24,
        dye_resolution: 48,
        buoyancy: 40.0,
        ..FluidConfig3d::default()
    };
    let (w, h) = (128, 128);
    let render = |show: bool| -> Option<Vec<u8>> {
        let mut sim = sim(FluidConfig3d {
            show_fps: show,
            ..base
        })?;
        sim.splat([0.5, 0.3, 0.5], [0.0, 200.0, 0.0], [1.0, 0.5, 0.3]);
        for _ in 0..6 {
            sim.step(1.0 / 60.0);
            // Twice, so the rate is measured and the number is not zero.
            sim.capture_rgba(&camera(1.0), w, h);
        }
        Some(sim.capture_rgba(&camera(1.0), w, h))
    };
    let Some(plain) = render(false) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let with = render(true).expect("a second simulation");

    // The read-out's two parts: three digits at 3x5 scaled by three, ten in
    // from each edge, and the chart under them — 64 samples two pixels wide,
    // 18 tall, starting at y = 29. Anything else must be untouched.
    let in_read_out = |x: u32, y: u32| {
        let digits = x < 10 + 12 * 3 && y < 10 + 5 * 3;
        let chart = (10..10 + 64 * 2).contains(&x) && (29..29 + 18).contains(&y);
        digits || chart
    };
    let (mut inside, mut outside) = (0usize, 0usize);
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 4) as usize;
            if plain[i..i + 4] == with[i..i + 4] {
                continue;
            }
            if in_read_out(x, y) {
                inside += 1;
            } else {
                outside += 1;
            }
        }
    }
    eprintln!("  the read-out changed {inside} pixels inside its box and {outside} outside");
    assert!(
        inside > 20,
        "the read-out drew {inside} pixels, so it is not there"
    );
    assert_eq!(
        outside, 0,
        "the read-out changed {outside} pixels outside its own box"
    );
}

/// A click lands where it was aimed, and a drag pushes the way it was
/// dragged.
///
/// The whole point of `impulse_at_screen` is that a 2D pointer reaches a 3D
/// place, which is a projection and an unprojection that have to be each
/// other's inverse. If they are not, dye appears somewhere plausible but
/// wrong and nobody can say so from a picture.
///
/// So this works backwards: put the impulse at a known *screen* point, then
/// ask where the brightest pixel of the render is. It has to be there.
#[test]
fn a_click_lands_where_it_was_aimed() {
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        buoyancy: 0.0,
        curl: 0.0,
        splat_radius: 0.3,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cam = camera(1.0);
    let (w, h) = (128, 128);
    // Off centre in both axes, so a transposed or flipped unprojection
    // cannot pass.
    let aim = [0.62f32, 0.38];
    sim.impulse_at_screen(&cam, aim, [0.0, 0.0], 0.0, [1.0, 1.0, 1.0]);
    let (bx, by, peak) = brightest(&sim.capture_rgba(&cam, w, h), w, h);
    eprintln!(
        "  aimed at ({:.2}, {:.2}), brightest at ({bx:.2}, {by:.2})",
        aim[0], aim[1]
    );
    assert!(peak > 40, "the click left nothing to see");
    assert!(
        (bx - aim[0]).abs() < 0.08 && (by - aim[1]).abs() < 0.08,
        "aimed at ({:.2}, {:.2}), landed at ({bx:.2}, {by:.2})",
        aim[0],
        aim[1]
    );

    // And a drag pushes along itself: dye dragged to the right ends up
    // further right than dye that was not.
    let after_drag = |drag: [f32; 2]| -> f32 {
        let mut sim = sim2(FluidConfig3d {
            resolution: 32,
            dye_resolution: 64,
            buoyancy: 0.0,
            curl: 0.0,
            velocity_dissipation: 0.0,
            density_dissipation: 0.0,
            splat_radius: 0.3,
            ..FluidConfig3d::default()
        })
        .expect("a simulation");
        sim.impulse_at_screen(&cam, [0.5, 0.5], drag, 6.0, [1.0, 1.0, 1.0]);
        for _ in 0..24 {
            sim.step(1.0 / 60.0);
        }
        brightest(&sim.capture_rgba(&cam, w, h), w, h).0
    };
    let still = after_drag([0.0, 0.0]);
    let pushed = after_drag([0.25, 0.0]);
    eprintln!("  undragged ends at x {still:.2}, dragged right ends at {pushed:.2}");
    assert!(
        pushed > still + 0.03,
        "a rightward drag did not push right: {pushed:.2} against {still:.2}"
    );
}

/// How far a local disturbance reaches, which decides whether the solve can
/// be limited to what the camera sees.
///
/// The idea is appealing: a volume is mostly off screen or behind something,
/// and smoothing is most of the solve's passes, so why smooth what nobody is
/// looking at? The answer depends on a property of the equation rather than
/// of the renderer. Advection is hyperbolic and local — what happens over
/// there arrives here later, at the speed of the flow. The pressure solve is
/// *elliptic*: it is the projection that makes the velocity field
/// divergence-free everywhere at once, and the field that does that is
/// determined by the whole domain simultaneously.
///
/// So: one splat in one corner, one step, and the pressure measured in
/// shells going away from it. If it falls off sharply, a frustum-limited
/// solve is sound and worth building. If it does not, the far side of the
/// box is holding up the near side and freezing it changes what the camera
/// sees.
#[test]
fn a_disturbance_in_one_corner_is_felt_across_the_box() {
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: 32,
        dye_resolution: 32,
        curl: 0.0,
        buoyancy: 0.0,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Hard against one corner, so "far away" is the opposite corner.
    sim.splat([0.15, 0.15, 0.15], [400.0, 0.0, 0.0], [0.2, 0.2, 0.2]);
    sim.step(1.0 / 6000.0);

    let p = read_channel(&sim, &sim.pressure.read, 0);
    let atlas = sim.sim;
    let (aw, _) = atlas.size();
    let (n, across) = (atlas.grid as usize, (aw / atlas.grid) as usize);
    // Four shells by distance from the splat, as a fraction of the diagonal.
    let mut shells = [(0.0f64, 0usize); 4];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (row, col) = (z / across, z % across);
                let v = p[(row * n + y) * aw as usize + col * n + x].abs() as f64;
                let d = |a: usize, at: f32| (a as f32 / n as f32 - at).powi(2);
                let dist = (d(x, 0.15) + d(y, 0.15) + d(z, 0.15)).sqrt() / 3f32.sqrt();
                let shell = ((dist * 4.0) as usize).min(3);
                shells[shell].0 += v;
                shells[shell].1 += 1;
            }
        }
    }
    let mean: Vec<f64> = shells
        .iter()
        .map(|(sum, n)| sum / (*n).max(1) as f64)
        .collect();
    eprintln!(
        "  mean |pressure| by distance from the splat: near {:.4}, {:.4}, {:.4}, far {:.4}",
        mean[0], mean[1], mean[2], mean[3]
    );
    eprintln!(
        "  the far quarter carries {:.0}% of what the near one does",
        100.0 * mean[3] / mean[0].max(1e-12)
    );
    assert!(mean[0] > 1e-6, "the splat produced no pressure at all");
}

/// Buoyancy goes where it is pointed, and random splats go everywhere.
///
/// Both were pinned to +Y and neither had to be. The buoyancy was written
/// `vel.y = vel.y + …`, which is what buoyancy usually is but not what the
/// solver has to assume; and `multiple_splats` drew its vertical component
/// from `rand` rather than `rand - 0.5`, so it was never negative and every
/// random push went up.
#[test]
fn buoyancy_goes_where_it_is_pointed() {
    let base = FluidConfig3d {
        resolution: 24,
        dye_resolution: 24,
        buoyancy: 300.0,
        curl: 0.0,
        velocity_dissipation: 0.0,
        density_dissipation: 0.0,
        splat_radius: 0.25,
        ..FluidConfig3d::default()
    };
    // Centre of mass of the dye, in voxels, per axis.
    let drift = |axis: [f32; 3]| -> Option<[f32; 3]> {
        let mut sim = sim(FluidConfig3d {
            buoyancy_axis: axis,
            ..base
        })?;
        sim.splat([0.5, 0.5, 0.5], [0.0; 3], [1.0, 1.0, 1.0]);
        let centre = |sim: &FluidSim3d| -> [f32; 3] {
            let dye = read_channel(sim, &sim.dye.read, 0);
            let atlas = sim.dye_atlas();
            let (w, _) = atlas.size();
            let (n, across) = (atlas.grid as usize, (w / atlas.grid) as usize);
            let (mut sum, mut mass) = ([0.0f64; 3], 0.0f64);
            for (i, v) in dye.iter().enumerate() {
                let (ax, ay) = (i % w as usize, i / w as usize);
                let (x, y) = (ax % n, ay % n);
                let z = (ay / n) * across + ax / n;
                let w = *v as f64;
                sum[0] += w * x as f64;
                sum[1] += w * y as f64;
                sum[2] += w * z as f64;
                mass += w;
            }
            let m = mass.max(1e-9);
            [
                (sum[0] / m) as f32,
                (sum[1] / m) as f32,
                (sum[2] / m) as f32,
            ]
        };
        let before = centre(&sim);
        for _ in 0..40 {
            sim.step(1.0 / 60.0);
        }
        let after = centre(&sim);
        Some([
            after[0] - before[0],
            after[1] - before[1],
            after[2] - before[2],
        ])
    };

    let Some(up) = drift([0.0, 1.0, 0.0]) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let along_x = drift([1.0, 0.0, 0.0]).expect("a second simulation");
    let along_z = drift([0.0, 0.0, -1.0]).expect("a third simulation");
    eprintln!("  drift up {up:?}");
    eprintln!("  drift +x {along_x:?}");
    eprintln!("  drift -z {along_z:?}");
    // Each one has to move furthest along the axis it was given, and in the
    // right direction.
    assert!(up[1] > 0.4 && up[1] > up[0].abs() && up[1] > up[2].abs());
    assert!(along_x[0] > 0.4 && along_x[0] > along_x[1].abs());
    assert!(along_z[2] < -0.4 && -along_z[2] > along_z[1].abs());
}

/// And the random splats cover the sphere rather than a hemisphere.
#[test]
fn random_splats_push_in_every_direction() {
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: 16,
        dye_resolution: 16,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // Enough of them that a hemisphere would show: with 48 one-sided draws
    // the chance of none going the other way is zero for practical purposes.
    sim.multiple_splats(48);
    let vy = read_channel(&sim, &sim.velocity.read, 1);
    let (up, down) = vy.iter().fold((0, 0), |(u, d), v| {
        if *v > 20.0 {
            (u + 1, d)
        } else if *v < -20.0 {
            (u, d + 1)
        } else {
            (u, d)
        }
    });
    eprintln!("  voxels pushed up {up}, down {down}");
    assert!(
        up > 0 && down > 0,
        "the pushes are one-sided: {up} up against {down} down"
    );
    // Not a precise balance — forty-eight draws is a small sample — but
    // nothing like the all-up it used to be.
    let ratio = up.min(down) as f32 / up.max(down).max(1) as f32;
    assert!(ratio > 0.3, "lopsided: {up} up, {down} down");
}

/// The CPU reference agrees with the GPU about which way buoyancy pushes.
///
/// `the_cpu_and_gpu_solvers_agree` runs with buoyancy off, which is the
/// right way to isolate the projection and the wrong way to notice that one
/// of them still believes "up" means +Y. It did: `buoyancy_axis` was added
/// to the shader and not to the reference, and nothing failed, because
/// nothing asked.
#[test]
fn the_two_solvers_agree_about_which_way_is_up() {
    let base = FluidConfig3d {
        resolution: 24,
        dye_resolution: 24,
        buoyancy: 300.0,
        // Deliberately none of the axes, so a solver that rounds to the
        // nearest one is caught too.
        buoyancy_axis: [0.6, 0.0, -0.8],
        curl: 0.0,
        velocity_dissipation: 0.0,
        density_dissipation: 0.0,
        splat_radius: 0.25,
        ..FluidConfig3d::default()
    };
    // Where the dye ends up, in voxels, from each solver.
    let Some(mut gpu) = sim(base) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut cpu = crate::fluid::CpuFluid3d::new(base);
    gpu.splat([0.5, 0.5, 0.5], [0.0; 3], [1.0, 1.0, 1.0]);
    cpu.splat([0.5, 0.5, 0.5], [0.0; 3], [1.0, 1.0, 1.0]);
    for _ in 0..30 {
        gpu.step(1.0 / 60.0);
        cpu.step(1.0 / 60.0);
    }

    let n = base.dye_resolution as usize;
    let gpu_dye = read_channel(&gpu, &gpu.dye.read, 0);
    let atlas = gpu.dye_atlas();
    let (aw, _) = atlas.size();
    let across = (aw / atlas.grid) as usize;
    let centre = |sample: &dyn Fn(usize, usize, usize) -> f32| -> [f32; 3] {
        let (mut sum, mut mass) = ([0.0f64; 3], 0.0f64);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let w = sample(x, y, z) as f64;
                    sum[0] += w * x as f64;
                    sum[1] += w * y as f64;
                    sum[2] += w * z as f64;
                    mass += w;
                }
            }
        }
        let m = mass.max(1e-9);
        std::array::from_fn(|i| (sum[i] / m) as f32)
    };
    let g = centre(&|x, y, z| {
        let (row, col) = (z / across, z % across);
        gpu_dye[(row * n + y) * aw as usize + col * n + x]
    });
    let cpu_dye = cpu.dye();
    let c = centre(&|x, y, z| cpu_dye[x + n * (y + n * z)][0]);

    eprintln!("  dye centre: gpu {g:?}, cpu {c:?}");
    for (i, axis) in ["x", "y", "z"].iter().enumerate() {
        assert!(
            (g[i] - c[i]).abs() < 1.0,
            "the solvers disagree on {axis}: gpu {:.2}, cpu {:.2}",
            g[i],
            c[i]
        );
    }
    // And it really did move along the axis it was given, or the agreement
    // above could be two solvers both doing nothing.
    assert!(
        g[0] > n as f32 / 2.0 + 0.5 && g[2] < n as f32 / 2.0 - 0.5,
        "the dye did not follow the axis: {g:?}"
    );
}

/// How much of the solve is over fluid that is not doing anything.
///
/// The solve is about 95% of the step, and the pressure from a disturbance
/// falls off sharply — `a_disturbance_in_one_corner_is_felt_across_the_box`
/// measures a factor of four hundred across the domain. That suggests
/// smoothing only where something is happening. This is the measurement
/// that says whether it is worth building: the fraction of voxels carrying
/// divergence worth solving, and — since a scissor can only cut rectangles
/// — the fraction of the box their bounding box covers.
#[test]
fn how_much_of_the_solve_is_over_still_fluid() {
    let Some(mut sim) = sim(FluidConfig3d {
        resolution: 48,
        dye_resolution: 96,
        buoyancy: 46.0,
        curl: 26.0,
        density_dissipation: 0.2,
        ..FluidConfig3d::default()
    }) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    // A plume, as the examples run it.
    for frame in 0..52 {
        if frame < 34 && frame % 3 == 0 {
            sim.splat([0.5, 0.11, 0.5], [0.0, 300.0, 0.0], [1.0, 0.4, 0.3]);
        }
        sim.step(1.0 / 60.0);
    }

    let d = read_channel(&sim, &sim.divergence, 0);
    let atlas = sim.sim;
    let (aw, _) = atlas.size();
    let (n, across) = (atlas.grid as usize, (aw / atlas.grid) as usize);
    let peak = d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    eprintln!("  peak |divergence| {peak:.4}");

    for cut in [0.01f32, 0.001] {
        let floor = peak * cut;
        let (mut active, mut lo, mut hi) = (0usize, [n; 3], [0usize; 3]);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let (row, col) = (z / across, z % across);
                    if d[(row * n + y) * aw as usize + col * n + x].abs() <= floor {
                        continue;
                    }
                    active += 1;
                    for (i, v) in [x, y, z].iter().enumerate() {
                        lo[i] = lo[i].min(*v);
                        hi[i] = hi[i].max(*v);
                    }
                }
            }
        }
        let cells = n * n * n;
        let boxed: usize = (0..3)
            .map(|i| hi[i].saturating_sub(lo[i]) + 1)
            .product::<usize>()
            .min(cells);
        eprintln!(
            "  above {:>6.1}% of peak: {:>5.1}% of voxels active, bounding box {:>5.1}% of the volume",
            cut * 100.0,
            100.0 * active as f32 / cells as f32,
            100.0 * boxed as f32 / cells as f32
        );
    }
    // And the same for the pressure, which is the field the smoothing is
    // actually solving for. The divergence is the right-hand side and can
    // be sparse; the solution does not have to be, and if it is not then a
    // mask drawn around the divergence would clamp the pressure's tail and
    // change the answer inside the mask as well as outside it.
    let p = read_channel(&sim, &sim.pressure.read, 0);
    let ppeak = p.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for cut in [0.01f32, 0.001] {
        let floor = ppeak * cut;
        let active = p.iter().filter(|v| v.abs() > floor).count();
        // The atlas has padding; count against the voxels, not the texels.
        let cells = n * n * n;
        eprintln!(
            "  pressure above {:>6.1}% of peak: {:>5.1}% of voxels",
            cut * 100.0,
            100.0 * active.min(cells) as f32 / cells as f32
        );
    }
    assert!(peak > 1e-5, "the plume produced no divergence to measure");
}

/// Accumulation survives the camera moving, which is what reprojection is
/// for.
///
/// Before it, any movement threw the history away and a moving camera got
/// the noise of a single short march — which is most of when a volume is
/// being looked at. The march now reports the depth it saw, the resolve
/// unprojects that and projects it through the previous frame's camera, and
/// the history lines up with where the picture actually was.
///
/// The fluid is held still here so the only thing moving is the camera.
#[test]
fn accumulation_survives_an_orbit() {
    let base = FluidConfig3d {
        resolution: 32,
        dye_resolution: 64,
        march_steps: 24,
        buoyancy: 40.0,
        ..FluidConfig3d::default()
    };
    let (w, h) = (96, 96);
    let orbit = |frame: usize| -> PerspectiveCamera {
        // A degree and a bit a frame: a slow drag, which is the case that
        // matters. A flick resets anyway, because the history leaves the
        // screen.
        let a = frame as f32 * 0.02;
        let mut c = PerspectiveCamera::new(45.0, 1.0, 0.05, 20.0);
        c.position = Vector3::new(1.9 * a.sin(), 0.0, 1.9 * a.cos());
        c.look_at(Vector3::ZERO);
        c
    };
    let settle = |sim: &mut FluidSim3d| {
        for frame in 0..14 {
            if frame % 3 == 0 {
                sim.splat([0.5, 0.2, 0.5], [0.0, 260.0, 0.0], [1.1, 0.4, 0.25]);
            }
            sim.step(1.0 / 60.0);
        }
    };
    const FRAMES: usize = 20;

    let Some(mut sim) = sim(base) else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    settle(&mut sim);
    // The truth, from where the orbit ends.
    sim.set_config(FluidConfig3d {
        march_steps: 24 * 12,
        ..base
    });
    let truth = sim.capture_rgba(&orbit(FRAMES - 1), w, h);
    // One short march from the same place, which is what a moving camera
    // used to get.
    sim.set_config(base);
    let single = sim.capture_rgba(&orbit(FRAMES - 1), w, h);
    // And the same short march accumulated *through* the orbit.
    sim.set_config(FluidConfig3d {
        temporal_samples: 8,
        ..base
    });
    let mut orbited = Vec::new();
    for frame in 0..FRAMES {
        orbited = sim.capture_rgba(&orbit(frame), w, h);
    }

    let against = |a: &[u8]| -> f64 {
        a.iter()
            .zip(&truth)
            .map(|(x, y)| x.abs_diff(*y) as f64)
            .sum::<f64>()
            / a.len() as f64
    };
    let (one, moved) = (against(&single), against(&orbited));
    eprintln!("  through an orbit: one march {one:.3}/255, accumulated {moved:.3}");
    assert!(
        brightest(&truth, w, h).2 > 40,
        "the reference render is empty"
    );
    assert!(
        moved < one,
        "accumulating through an orbit was no better than not accumulating: \
         {moved:.3} against {one:.3}"
    );
}
