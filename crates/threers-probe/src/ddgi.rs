//! DDGI-lite: bin probe radiance into a coarse world-space grid, then broadcast
//! each cell back to every pixel that falls in it — cheap distant context
//! without extra rays.
//!
//! Each cell also stores a mean hit distance from its centre (RTXGI-style). That
//! field gates Jacobi propagation and per-pixel gather so wall tint does not
//! leak across empty space onto the floor.

use crate::{compress, OUT_CHANNELS};

/// Plane index in the packed input: DDGI visibility in `[0, 1]`.
pub const DDGI_VIS_PLANE: usize = 22;

/// Horizontal / depth grid resolution.
pub const DDGI_GRID: usize = 8;
/// Vertical slices — enough to separate floor, mid, and ceiling in a room.
pub const DDGI_GRID_Y: usize = 4;

/// World positions and scene bounds for DDGI binning.
pub struct DdgiContext<'a> {
    pub world: &'a [[f32; 3]],
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

fn cells() -> usize {
    DDGI_GRID * DDGI_GRID_Y * DDGI_GRID
}

fn bin_axis(v: f32, lo: f32, span: f32, n: usize) -> usize {
    ((v - lo) / span * n as f32)
        .floor()
        .clamp(0.0, (n - 1) as f32) as usize
}

fn cell_span(ctx: &DdgiContext<'_>) -> [f32; 3] {
    [
        (ctx.bounds_max[0] - ctx.bounds_min[0]).max(1e-4),
        (ctx.bounds_max[1] - ctx.bounds_min[1]).max(1e-4),
        (ctx.bounds_max[2] - ctx.bounds_min[2]).max(1e-4),
    ]
}

fn cell_of(w: [f32; 3], ctx: &DdgiContext<'_>) -> usize {
    let span = cell_span(ctx);
    let gx = bin_axis(w[0], ctx.bounds_min[0], span[0], DDGI_GRID);
    let gy = bin_axis(w[1], ctx.bounds_min[1], span[1], DDGI_GRID_Y);
    let gz = bin_axis(w[2], ctx.bounds_min[2], span[2], DDGI_GRID);
    (gy * DDGI_GRID + gz) * DDGI_GRID + gx
}

fn cell_center(cell: usize, ctx: &DdgiContext<'_>) -> [f32; 3] {
    let span = cell_span(ctx);
    let gx = DDGI_GRID;
    let gz = DDGI_GRID;
    let x = cell % gx;
    let rest = cell / gx;
    let z = rest % gz;
    let y = rest / gz;
    [
        ctx.bounds_min[0] + (x as f32 + 0.5) / DDGI_GRID as f32 * span[0],
        ctx.bounds_min[1] + (y as f32 + 0.5) / DDGI_GRID_Y as f32 * span[1],
        ctx.bounds_min[2] + (z as f32 + 0.5) / DDGI_GRID as f32 * span[2],
    ]
}

fn dist3(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// RTXGI-style weight: full trust near the cell centre, fade when the hit is
/// farther than the stored mean surface distance.
fn ddgi_visibility(hit_dist: f32, cell_dist: f32) -> f32 {
    let d = cell_dist.max(1e-3);
    ((d - hit_dist) / d).clamp(0.0, 1.0)
}

/// Three compressed planes: DDGI cell average probe RGB, nearest-upsampled per pixel.
pub fn ddgi_planes(
    width: usize,
    height: usize,
    probe_rgba: &[f32],
    ctx: &DdgiContext<'_>,
) -> Vec<f32> {
    let (colour, _vis) = ddgi_planes_with_visibility(width, height, probe_rgba, ctx);
    colour
}

/// Colour planes plus per-pixel RTXGI-style visibility (for experiments / offline analysis).
pub fn ddgi_planes_with_visibility(
    width: usize,
    height: usize,
    probe_rgba: &[f32],
    ctx: &DdgiContext<'_>,
) -> (Vec<f32>, Vec<f32>) {
    let n = width * height;
    let ncell = cells();
    let mut sums = vec![0.0f32; OUT_CHANNELS * ncell];
    let mut counts = vec![0u32; ncell];
    let mut dist_sum = vec![0.0f32; ncell];
    for i in 0..n {
        let w = ctx.world[i];
        if !w[0].is_finite() {
            continue;
        }
        let cell = cell_of(w, ctx);
        counts[cell] += 1;
        dist_sum[cell] += dist3(w, cell_center(cell, ctx));
        for c in 0..3 {
            sums[c * ncell + cell] += probe_rgba[i * 4 + c].max(0.0);
        }
    }
    let mut cell_colour = vec![0.0f32; OUT_CHANNELS * ncell];
    let mut cell_dist = vec![0.0f32; ncell];
    for cell in 0..ncell {
        if counts[cell] == 0 {
            continue;
        }
        let inv = 1.0 / counts[cell] as f32;
        cell_dist[cell] = dist_sum[cell] * inv;
        for c in 0..3 {
            cell_colour[c * ncell + cell] = compress(sums[c * ncell + cell] * inv);
        }
    }
    let wall_cells = cell_colour.clone();
    propagate_grid(&mut cell_colour, 6);
    let mut colour = vec![0.0f32; OUT_CHANNELS * n];
    let mut vis = vec![0.0f32; n];
    for i in 0..n {
        let w = ctx.world[i];
        if !w[0].is_finite() {
            continue;
        }
        let cell = cell_of(w, ctx);
        let hit_dist = dist3(w, cell_center(cell, ctx));
        vis[i] = ddgi_visibility(hit_dist, cell_dist[cell]);
        for c in 0..3 {
            colour[c * n + i] = cell_colour[c * ncell + cell];
        }
    }
    if let Some(mix) = room_chroma_mix() {
        paint_lateral_wall_chroma(&mut colour, ctx, &wall_cells, mix);
    }
    (colour, vis)
}

fn room_chroma_mix() -> Option<f32> {
    let raw = std::env::var("DDGI_ROOM_CHROMA").ok()?;
    if raw == "1" || raw.eq_ignore_ascii_case("true") {
        return Some(0.55);
    }
    let v: f32 = raw.parse().ok()?;
    (v > 0.0).then_some(v.clamp(0.0, 1.0))
}

fn rgb_chroma(r: f32, g: f32, b: f32) -> f32 {
    let mean = (r + g + b) / 3.0;
    (r - mean).abs().max((g - mean).abs()).max((b - mean).abs())
}

/// Floor/ceiling cells see local (white) radiance. Pull undiluted left/right
/// wall cell colour across X so the in-graph room tint has chroma to copy.
fn paint_lateral_wall_chroma(
    colour: &mut [f32],
    ctx: &DdgiContext<'_>,
    wall_cells: &[f32],
    mix: f32,
) {
    let gx = DDGI_GRID;
    let gy = DDGI_GRID_Y;
    let gz = DDGI_GRID;
    let ncell = gx * gy * gz;
    let idx = |x: usize, y: usize, z: usize| (y * gz + z) * gx + x;
    let mut left = [0.0f32; 3];
    let mut right = [0.0f32; 3];
    let mut left_w = 0.0f32;
    let mut right_w = 0.0f32;
    for y in 0..gy {
        for z in 0..gz {
            for (x, acc, wacc) in [(0usize, &mut left, &mut left_w), (gx - 1, &mut right, &mut right_w)]
            {
                let cell = idx(x, y, z);
                let rgb = [
                    crate::expand(wall_cells[cell]),
                    crate::expand(wall_cells[ncell + cell]),
                    crate::expand(wall_cells[2 * ncell + cell]),
                ];
                let w = rgb_chroma(rgb[0], rgb[1], rgb[2]).max(1e-4);
                for c in 0..3 {
                    acc[c] += rgb[c] * w;
                }
                *wacc += w;
            }
        }
    }
    if left_w < 1e-3 || right_w < 1e-3 {
        return;
    }
    for c in 0..3 {
        left[c] /= left_w;
        right[c] /= right_w;
    }
    if rgb_chroma(left[0], left[1], left[2]).max(rgb_chroma(right[0], right[1], right[2])) < 0.04 {
        return;
    }
    let n = ctx.world.len();
    let span_x = (ctx.bounds_max[0] - ctx.bounds_min[0]).max(1e-4);
    for i in 0..n {
        let w = ctx.world[i];
        if !w[0].is_finite() {
            continue;
        }
        let cell = cell_of(w, ctx);
        let rest = cell / gx;
        let y = rest / gz;
        if y != 0 && y + 1 != gy {
            continue;
        }
        let t = ((w[0] - ctx.bounds_min[0]) / span_x).clamp(0.0, 1.0);
        let mut wall = [0.0f32; 3];
        let mut local = [0.0f32; 3];
        for c in 0..3 {
            wall[c] = left[c] * (1.0 - t) + right[c] * t;
            local[c] = crate::expand(colour[c * n + i]);
        }
        if rgb_chroma(wall[0], wall[1], wall[2]) + 0.004
            <= rgb_chroma(local[0], local[1], local[2])
        {
            continue;
        }
        for c in 0..3 {
            let lin = local[c] * (1.0 - mix) + wall[c] * mix;
            colour[c * n + i] = crate::compress(lin.max(0.0));
        }
    }
}

/// Diffuse light into neighbouring cells — a few Jacobi hops, LPV-style.
fn propagate_grid(colour: &mut [f32], iters: usize) {
    let gx = DDGI_GRID;
    let gy = DDGI_GRID_Y;
    let gz = DDGI_GRID;
    let ncell = gx * gy * gz;
    let idx = |x: usize, y: usize, z: usize| (y * gz + z) * gx + x;
    let decode = |cell: usize| {
        let x = cell % gx;
        let rest = cell / gx;
        let z = rest % gz;
        let y = rest / gz;
        (x, y, z)
    };
    for _ in 0..iters {
        let old = colour.to_vec();
        for cell in 0..ncell {
            let (x, y, z) = decode(cell);
            for c in 0..3 {
                let mut acc = old[c * ncell + cell];
                let mut n = 1.0f32;
                let bump = |xx: usize, yy: usize, zz: usize, acc: &mut f32, n: &mut f32| {
                    *acc += old[c * ncell + idx(xx, yy, zz)];
                    *n += 1.0;
                };
                if x > 0 {
                    bump(x - 1, y, z, &mut acc, &mut n);
                }
                if x + 1 < gx {
                    bump(x + 1, y, z, &mut acc, &mut n);
                }
                if y > 0 {
                    bump(x, y - 1, z, &mut acc, &mut n);
                }
                if y + 1 < gy {
                    bump(x, y + 1, z, &mut acc, &mut n);
                }
                if z > 0 {
                    bump(x, y, z - 1, &mut acc, &mut n);
                }
                if z + 1 < gz {
                    bump(x, y, z + 1, &mut acc, &mut n);
                }
                colour[c * ncell + cell] = acc / n;
            }
        }
    }
}

/// Axis-aligned bounds over finite hit points, padded 10%.
pub fn bounds_from_world(world: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    let mut any = false;
    for w in world {
        if !w[0].is_finite() {
            continue;
        }
        any = true;
        for c in 0..3 {
            min[c] = min[c].min(w[c]);
            max[c] = max[c].max(w[c]);
        }
    }
    if !any {
        return ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    }
    let pad = |a: f32, b: f32| ((b - a).max(1e-3)) * 0.1;
    for c in 0..3 {
        let p = pad(min[c], max[c]);
        min[c] -= p;
        max[c] += p;
    }
    (min, max)
}

/// Scene centre and extent for encoding world positions.
pub fn encode_frame(world: &[[f32; 3]]) -> ([f32; 3], f32) {
    let (min, max) = bounds_from_world(world);
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let scale = (max[0] - min[0])
        .max(max[1] - min[1])
        .max(max[2] - min[2])
        .max(1.0);
    (center, scale)
}

/// Scene-centred world position in `[0, 1)` — signed, so left/right of centre
/// are distinguishable (the old `compress(max(0, ·))` folded half the room).
pub fn encode_world_planes(world: &[[f32; 3]], center: [f32; 3], scale: f32) -> Vec<f32> {
    let n = world.len();
    let inv = 1.0 / scale.max(1e-4);
    let mut out = vec![0.0f32; OUT_CHANNELS * n];
    for (i, w) in world.iter().enumerate() {
        if !w[0].is_finite() {
            continue;
        }
        for c in 0..3 {
            let t = (w[c] - center[c]) * inv;
            out[c * n + i] = (t * 0.5 + 0.5).clamp(0.0, 0.999);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visibility_is_highest_at_cell_centres() {
        let ctx = DdgiContext {
            world: &[
                [0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0],
            ],
            bounds_min: [-1.0, -1.0, -1.0],
            bounds_max: [1.0, 1.0, 1.0],
        };
        let probe = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let (_, vis) = ddgi_planes_with_visibility(1, 2, &probe, &ctx);
        assert!(vis[0] >= vis[1], "centre hit should be at least as visible as edge hit");
    }

    #[test]
    fn ddgi_visibility_formula() {
        assert!((ddgi_visibility(0.0, 1.0) - 1.0).abs() < 1e-6);
        assert!((ddgi_visibility(1.0, 1.0) - 0.0).abs() < 1e-6);
        assert!((ddgi_visibility(0.5, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn floor_lookup_picks_up_lateral_wall_chroma() {
        std::env::set_var("DDGI_ROOM_CHROMA", "1");
        let mut world = vec![[f32::NAN; 3]; 8];
        // left wall (x bin 0), mid height, chromatic
        world[0] = [-0.9, 0.0, 0.0];
        world[1] = [-0.9, 0.1, 0.1];
        // right wall
        world[2] = [0.9, 0.0, 0.0];
        world[3] = [0.9, 0.1, 0.1];
        // floor centre (y bin 0)
        world[4] = [0.0, -0.9, 0.0];
        world[5] = [0.1, -0.9, 0.1];
        world[6] = [-0.1, -0.9, -0.1];
        world[7] = [0.2, -0.9, 0.0];
        let mut probe = vec![0.0f32; 8 * 4];
        // red left, green right, white floor
        for i in 0..2 {
            probe[i * 4] = 0.8;
            probe[i * 4 + 3] = 1.0;
        }
        for i in 2..4 {
            probe[i * 4 + 1] = 0.8;
            probe[i * 4 + 3] = 1.0;
        }
        for i in 4..8 {
            probe[i * 4] = 0.5;
            probe[i * 4 + 1] = 0.5;
            probe[i * 4 + 2] = 0.5;
            probe[i * 4 + 3] = 1.0;
        }
        let ctx = DdgiContext {
            world: &world,
            bounds_min: [-1.0, -1.0, -1.0],
            bounds_max: [1.0, 1.0, 1.0],
        };
        let colour = ddgi_planes(2, 4, &probe, &ctx);
        let n = 8;
        let floor_r = crate::expand(colour[4]);
        let floor_g = crate::expand(colour[n + 4]);
        let floor_b = crate::expand(colour[2 * n + 4]);
        std::env::remove_var("DDGI_ROOM_CHROMA");
        assert!(
            rgb_chroma(floor_r, floor_g, floor_b) > 0.02,
            "floor DDGI should carry wall chroma, got {floor_r:.3} {floor_g:.3} {floor_b:.3}"
        );
    }
}
