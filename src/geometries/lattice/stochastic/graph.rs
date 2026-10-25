//! Every seed a lattice can reach, with its exact Voronoi (or power)
//! neighbours, worked out once.
//!
//! [`voronoi_distance`](super::voronoi_distance) has to find, for every
//! sample, which seeds could shape the cells near it — a search bounded by the
//! cap, which is most of its cost. But the diagram itself does not change from
//! sample to sample: seed `i`'s cell is bounded by its bisectors with its
//! neighbours and no others. So the neighbours are found once, by clipping a
//! box around each seed with its candidates' planes and keeping the planes that
//! leave a face, and a sample then needs no search at all:
//!
//! - its own seed is found by walking: from any seed, step to whichever
//!   neighbour is nearer the sample (in power, for weighted seeds) until none
//!   is. The cells are the facets of a convex lower envelope, so the walk
//!   cannot stop anywhere but the nearest — the reason the simplex method
//!   works;
//! - its cell's walls are the neighbours' bisectors, exactly;
//! - the cells whose edges or bubbles could come within the cap are reached
//!   from its own through the graph. Each lies beyond its bisector with the
//!   sample's cell, so only those whose bisector is nearer than the best so far
//!   are opened, and the cells that meet a ball are connected, so the walk
//!   finds them all.
//!
//! A seed whose cell the box could not close — at the edge of the block — is
//! marked incomplete, and a sample that lands there falls back to the search.

use super::{dot, nearest_edge, outside_distance, Plane, Skeleton, MAX_PLANES};

/// Cells of margin round the grid's own, so every seed a sample can need has
/// all of its neighbours inside the block.
const MARGIN: i32 = 3;

/// The most seeds a graph is built for; a bigger lattice uses the search.
const MAX_GRAPH_SEEDS: usize = 1 << 21;

/// A tile seed's neighbours, as (tile seed, cell offset), whether its cell
/// closed, and the cell's edges relative to the seed.
type TileCell = (Vec<(usize, [i32; 3])>, bool, Vec<[[f64; 3]; 2]>);

pub(crate) struct SeedGraph {
    /// The block of cells, `lo` and size, and how many seeds each holds.
    lo: [i32; 3],
    dims: [i32; 3],
    per_cell: usize,
    /// World size of a cell.
    size: [f64; 3],
    /// Seed `id`'s position in world units relative to the lattice origin,
    /// and its weight.
    pos: Vec<[f64; 3]>,
    weight: Vec<f64>,
    /// `neighbours[start[id]..start[id + 1]]`.
    start: Vec<u32>,
    neighbours: Vec<u32>,
    /// Whether the seed's cell closed inside the box it was clipped from.
    complete: Vec<bool>,
    /// Every edge of every complete cell, once, in world units relative to
    /// the lattice origin: the open foam's skeleton, as plain segments.
    edges: Vec<[[f64; 3]; 2]>,
}

/// The graphs built most recently, by everything that decides them. Fitting a
/// density builds a sampler per guess, and only the thickness changes between
/// them — which the graph does not depend on.
pub(crate) fn shared(
    lo: [i32; 3],
    hi: [i32; 3],
    cell: [f64; 3],
    tile: &[[f64; 3]],
    weights: &[f64],
    jitter: f32,
    seed: u32,
) -> Option<std::sync::Arc<SeedGraph>> {
    use std::sync::{Arc, Mutex, OnceLock};
    type Key = (Vec<u64>, u32, u32);
    type Held = Mutex<Vec<(Key, Arc<SeedGraph>)>>;
    static CACHE: OnceLock<Held> = OnceLock::new();
    let mut bits: Vec<u64> = Vec::with_capacity(9 + tile.len() * 3 + weights.len());
    bits.extend(lo.iter().chain(&hi).map(|&v| v as i64 as u64));
    bits.extend(cell.iter().map(|v| v.to_bits()));
    bits.extend(tile.iter().flatten().map(|v| v.to_bits()));
    bits.push(u64::MAX);
    bits.extend(weights.iter().map(|v| v.to_bits()));
    let key = (bits, jitter.to_bits(), seed);
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    if let Some((_, g)) = cache.lock().unwrap().iter().find(|(k, _)| *k == key) {
        return Some(g.clone());
    }
    let graph = Arc::new(SeedGraph::new(lo, hi, cell, tile, weights, jitter, seed)?);
    let mut held = cache.lock().unwrap();
    if held.len() >= 4 {
        held.remove(0);
    }
    held.push((key, graph.clone()));
    Some(graph)
}

impl SeedGraph {
    /// The graph for cells `lo..=hi` (before the margin), with `per_cell`
    /// seeds a cell: the `tile` (cell units, with `weights`) in every cell, or
    /// with an empty tile one seed a cell hashed from `seed` and moved by
    /// `jitter`, as [`voronoi_distance`](super::voronoi_distance) places them.
    pub(crate) fn new(
        lo: [i32; 3],
        hi: [i32; 3],
        cell: [f64; 3],
        tile: &[[f64; 3]],
        weights: &[f64],
        jitter: f32,
        seed: u32,
    ) -> Option<SeedGraph> {
        let lo = lo.map(|v| v - MARGIN);
        let hi = hi.map(|v| v + MARGIN);
        let dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
        if dims.iter().any(|&d| d <= 0) {
            return None;
        }
        let per_cell = tile.len().max(1);
        let cells = dims.iter().map(|&d| d as usize).product::<usize>();
        if cells.checked_mul(per_cell)? > MAX_GRAPH_SEEDS {
            return None;
        }
        let jitter = jitter.clamp(0.0, 1.0) as f64;
        // Where seed k of cell c sits, in cell units.
        let site = |c: [i32; 3], k: usize| -> [f64; 3] {
            if tile.is_empty() {
                super::seed_position(c, jitter, seed)
            } else {
                [0, 1, 2].map(|a| c[a] as f64 + tile[k][a])
            }
        };
        let weight_of = |k: usize| if weights.is_empty() { 0.0 } else { weights[k] };
        let mut pos = Vec::with_capacity(cells * per_cell);
        let mut weight = Vec::with_capacity(cells * per_cell);
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let c = [lo[0] + x, lo[1] + y, lo[2] + z];
                    for k in 0..per_cell {
                        let s = site(c, k);
                        pos.push([s[0] * cell[0], s[1] * cell[1], s[2] * cell[2]]);
                        weight.push(weight_of(k));
                    }
                }
            }
        }
        let mut graph = SeedGraph {
            lo,
            dims,
            per_cell,
            size: cell,
            pos,
            weight,
            start: Vec::with_capacity(cells * per_cell + 1),
            neighbours: Vec::new(),
            complete: Vec::with_capacity(cells * per_cell),
            edges: Vec::new(),
        };

        // A tile repeats, so its seeds' neighbours are worked out once, as
        // (seed, cell offset), and copied into every cell.
        let tiled: Option<Vec<TileCell>> = (!tile.is_empty()).then(|| {
            crate::utils::parallel::par_map_range_coarse(per_cell, |k| {
                    let me = [0, 1, 2].map(|a| tile[k][a] * cell[a]);
                    let mut candidates = Vec::new();
                    for oz in -2..=2 {
                        for oy in -2..=2 {
                            for ox in -2..=2 {
                                let o = [ox, oy, oz];
                                for (m, t) in tile.iter().enumerate() {
                                    if o == [0, 0, 0] && m == k {
                                        continue;
                                    }
                                    let q = [0, 1, 2].map(|a| (o[a] as f64 + t[a]) * cell[a]);
                                    candidates.push((q, weight_of(m), (m, o)));
                                }
                            }
                        }
                    }
                    clip_cell(me, weight_of(k), &candidates, cell)
            })
        });

        graph.start.push(0);
        let quantum = 1e-7 * cell[0].max(cell[1]).max(cell[2]);
        let mut edge_keys = std::collections::HashSet::new();
        let key = |a: [f64; 3], b: [f64; 3]| {
            let q = |v: [f64; 3]| v.map(|x| (x / quantum).round() as i64);
            let (qa, qb) = (q(a), q(b));
            (qa != qb).then(|| if qa < qb { (qa, qb) } else { (qb, qa) })
        };
        // A tile's edges, once: every edge of the foam is a copy of one of
        // these, so they are kept per periodic class — shifted until the
        // midpoint is in the unit cell — and laid into the block by offset.
        let mut tile_edges: Vec<[[f64; 3]; 2]> = Vec::new();
        if let Some(t) = &tiled {
            let mut classes = std::collections::HashSet::new();
            for (k, (_, closed, edges)) in t.iter().enumerate() {
                if !closed {
                    continue;
                }
                let me = [0, 1, 2].map(|a| tile[k][a] * cell[a]);
                for [a, b] in edges {
                    let (a, b) = (
                        [0, 1, 2].map(|i| a[i] + me[i]),
                        [0, 1, 2].map(|i| b[i] + me[i]),
                    );
                    let shift =
                        [0, 1, 2].map(|i| ((a[i] + b[i]) * 0.5 / cell[i]).floor() * cell[i]);
                    let (a, b) = (
                        [0, 1, 2].map(|i| a[i] - shift[i]),
                        [0, 1, 2].map(|i| b[i] - shift[i]),
                    );
                    if let Some(k) = key(a, b) {
                        if classes.insert(k) {
                            tile_edges.push([a, b]);
                        }
                    }
                }
            }
        }
        // A random block's cells, clipped layer by layer — on every core
        // with `parallel` — and laid out in seed order, so the graph is the
        // same whichever way it was built.
        type Clipped = (Vec<u32>, bool, Vec<[[f64; 3]; 2]>);
        let clip_random = |c: [i32; 3]| -> Clipped {
            'cell: {
                                // A seed near the block's edge has candidates
                                // missing, so its cell cannot be trusted — and
                                // is not worth clipping at all.
                                let edge = (0..3)
                                    .any(|a| c[a] - lo[a] < 2 || lo[a] + dims[a] - 1 - c[a] < 2);
                                if edge {
                                    break 'cell (Vec::new(), false, Vec::new());
                                }
                                let me = graph.pos[graph.id(c, 0).unwrap() as usize];
                                let mut candidates = Vec::new();
                                for oz in -2..=2 {
                                    for oy in -2..=2 {
                                        for ox in -2..=2 {
                                            if [ox, oy, oz] == [0, 0, 0] {
                                                continue;
                                            }
                                            let d = [c[0] + ox, c[1] + oy, c[2] + oz];
                                            if let Some(id) = graph.id(d, 0) {
                                                let q = graph.pos[id as usize];
                                                let rel =
                                                    [q[0] - me[0], q[1] - me[1], q[2] - me[2]];
                                                candidates.push((rel, 0.0, (id as usize, [0; 3])));
                                            }
                                        }
                                    }
                                }
                                let (faces, closed, edges) =
                                    clip_cell([0.0; 3], 0.0, &candidates, cell);
                                (
                                    faces.into_iter().map(|(id, _)| id as u32).collect(),
                                    closed,
                                    edges,
                                )
            }
        };
        let mut clipped: Vec<Clipped> = if tiled.is_none() {
            let layers = crate::utils::parallel::par_map_range_coarse(dims[2] as usize, |z| {
                let mut out = Vec::with_capacity((dims[0] * dims[1]) as usize);
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        out.push(clip_random([lo[0] + x, lo[1] + y, lo[2] + z as i32]));
                    }
                }
                out
            });
            layers.into_iter().flatten().collect()
        } else {
            Vec::new()
        };
        for z in 0..dims[2] {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let c = [lo[0] + x, lo[1] + y, lo[2] + z];
                    for k in 0..per_cell {
                        let id = graph.complete.len();
                        let (list, closed, edges) = match &tiled {
                            Some(t) => {
                                let (faces, closed, edges) = &t[k];
                                let mut list = Vec::with_capacity(faces.len());
                                let mut inside = *closed;
                                for &(m, o) in faces {
                                    match graph.id([c[0] + o[0], c[1] + o[1], c[2] + o[2]], m) {
                                        Some(id) => list.push(id),
                                        None => inside = false,
                                    }
                                }
                                (list, inside, edges.clone())
                            }
                            None => {
                                let slot = &mut clipped[id];
                                std::mem::take(slot)
                            }
                        };
                        graph.neighbours.extend(list);
                        graph.start.push(graph.neighbours.len() as u32);
                        graph.complete.push(closed);
                        // The cell's edges in world terms, each shared edge
                        // kept once: the three or more cells round an edge
                        // each clip their own copy, equal to rounding.
                        if closed && tiled.is_none() {
                            let me = graph.pos[id];
                            for [a, b] in edges {
                                let a = [a[0] + me[0], a[1] + me[1], a[2] + me[2]];
                                let b = [b[0] + me[0], b[1] + me[1], b[2] + me[2]];
                                if key(a, b).is_some_and(|k| edge_keys.insert(k)) {
                                    graph.edges.push([a, b]);
                                }
                            }
                        }
                    }
                    // A tiled block's edges: the tile's, in every cell clear
                    // of the block's edge.
                    let interior = (0..3).all(|a| {
                        let at = [x, y, z][a];
                        at >= 2 && at <= dims[a] - 3
                    });
                    if interior {
                        let origin = [0, 1, 2].map(|a| c[a] as f64 * cell[a]);
                        for [a, b] in &tile_edges {
                            graph.edges.push([
                                [a[0] + origin[0], a[1] + origin[1], a[2] + origin[2]],
                                [b[0] + origin[0], b[1] + origin[1], b[2] + origin[2]],
                            ]);
                        }
                    }
                }
            }
        }
        Some(graph)
    }

    /// The id of seed `k` of cell `c`, if the cell is in the block.
    fn id(&self, c: [i32; 3], k: usize) -> Option<u32> {
        let r = [0, 1, 2].map(|a| c[a] - self.lo[a]);
        if (0..3).any(|a| r[a] < 0 || r[a] >= self.dims[a]) {
            return None;
        }
        let cell = ((r[2] * self.dims[1] + r[1]) * self.dims[0] + r[0]) as usize;
        Some((cell * self.per_cell + k) as u32)
    }

    fn neighbours_of(&self, id: u32) -> &[u32] {
        &self.neighbours[self.start[id as usize] as usize..self.start[id as usize + 1] as usize]
    }

    /// The distance [`voronoi_distance`](super::voronoi_distance) would give
    /// at `p` (world units, relative to the lattice origin), or `None` when
    /// the sample falls outside what the graph can answer.
    pub(crate) fn distance(
        &self,
        p: [f64; 3],
        skeleton: Skeleton,
        cap: f64,
        floor: f64,
    ) -> Option<f32> {
        let c = [0, 1, 2].map(|a| (p[a] / self.size[a]).floor() as i32);
        let mut own = self.id(c, 0)?;
        // Vector to a seed from p and its power there.
        let at = |id: u32| {
            let s = self.pos[id as usize];
            let v = [s[0] - p[0], s[1] - p[1], s[2] - p[2]];
            (v, dot(v, v) - self.weight[id as usize])
        };
        // Walk to the seed whose cell holds p.
        let mut here = at(own).1;
        loop {
            if !self.complete[own as usize] {
                return None;
            }
            let mut step = None;
            for &n in self.neighbours_of(own) {
                let power = at(n).1;
                if power < here {
                    here = power;
                    step = Some(n);
                }
            }
            match step {
                Some(n) => own = n,
                None => break,
            }
        }
        let plane = |j: u32, k: u32| -> Option<Plane> {
            let ((vj, pj), (vk, pk)) = (at(j), at(k));
            let w = [vk[0] - vj[0], vk[1] - vj[1], vk[2] - vj[2]];
            let l = dot(w, w).sqrt();
            (l > 1e-12).then(|| Plane {
                n: [w[0] / l, w[1] / l, w[2] / l],
                c: (pk - pj) / (2.0 * l),
            })
        };
        let eps = 1e-9 * self.size[0].max(self.size[1]).max(self.size[2]);

        if skeleton == Skeleton::Walls {
            let best = self
                .neighbours_of(own)
                .iter()
                .filter_map(|&n| plane(own, n))
                .fold(cap, |b, pl| b.min(pl.c));
            return Some(best.max(0.0) as f32);
        }
        let inset = match skeleton {
            Skeleton::Wet { border } => (border as f64).max(0.0),
            _ => 0.0,
        };

        // Out in the void, for an unweighted foam, the old reading settles it:
        // a point within δ of an edge has three seeds within 2δ of each other,
        // so if the third nearest is more than twice the cap further than the
        // nearest, no edge is in reach. The second nearest is always a
        // neighbour of the nearest, and the third a neighbour of one of them.
        let unweighted = self.weight.iter().all(|&w| w == 0.0);
        if unweighted && inset == 0.0 {
            let d0 = here.sqrt();
            let mut near = [(f64::INFINITY, u32::MAX); 2];
            for &n in self.neighbours_of(own) {
                let d = at(n).1.sqrt();
                if d < near[0].0 {
                    near = [(d, n), near[0]];
                } else if d < near[1].0 {
                    near[1] = (d, n);
                }
            }
            let mut third = near[1].0;
            if near[0].1 != u32::MAX {
                for &n in self.neighbours_of(near[0].1) {
                    if n != own && n != near[0].1 {
                        third = third.min(at(n).1.sqrt());
                    }
                }
            }
            if (third - d0) * 0.5 >= cap {
                return Some(cap as f32);
            }
        }

        // The cells near enough to matter, breadth first from p's own. Each
        // lies beyond its bisector with p's cell, so one whose bisector is no
        // nearer than the best so far is not opened, nor walked through. Every
        // cell is weighed once, when it comes off the queue.
        let mut best = cap;
        let mut queue: Vec<u32> = Vec::with_capacity(64);
        queue.push(own);
        let mut planes = [Plane {
            n: [0.0; 3],
            c: 0.0,
        }; MAX_PLANES];
        let mut head = 0;
        while head < queue.len() {
            let j = queue[head];
            head += 1;
            if j != own {
                match plane(own, j) {
                    Some(pl) if pl.c < best => {}
                    _ => continue,
                }
                if !self.complete[j as usize] {
                    return None;
                }
            }
            let mut count = 0usize;
            let mut beyond = false;
            let mut inside = true;
            let reach = best + inset;
            for &k in self.neighbours_of(j) {
                let Some(mut pl) = plane(j, k) else { continue };
                pl.c -= inset;
                if pl.c <= -reach {
                    beyond = true;
                    break;
                }
                if pl.c < 0.0 {
                    inside = false;
                }
                if pl.c.abs() < reach && count < MAX_PLANES {
                    planes[count] = pl;
                    count += 1;
                }
            }
            if !beyond {
                let planes = &mut planes[..count];
                planes.sort_by(|a, b| a.c.abs().total_cmp(&b.c.abs()));
                if inset > 0.0 {
                    // A bubble: the cell moved in by the border, then out.
                    let d = if inside {
                        0.0
                    } else {
                        outside_distance(planes, reach, eps)
                    };
                    best = best.min(d - inset);
                } else {
                    best = nearest_edge(planes, best, eps);
                    if best <= floor {
                        return Some(best as f32);
                    }
                }
            }
            for &n in self.neighbours_of(j) {
                if !queue.contains(&n) {
                    queue.push(n);
                }
            }
        }
        Some(best as f32)
    }
}

/// An open foam's edges, binned so a sample measures only the few that can
/// reach it.
///
/// The skeleton of an open-cell foam is nothing but straight segments — the
/// edges of the graph's cells — so the distance to it is the distance to the
/// nearest segment, which is what a strut lattice measures. A bin lists every
/// edge whose box, grown by the cull, touches it; a sample reads its own bin.
/// Exact up to the cull it was drawn up for, as the search is.
pub(crate) struct EdgeBins {
    lo: [f64; 3],
    bin: f64,
    dims: [usize; 3],
    cull: f64,
    start: Vec<u32>,
    list: Vec<u32>,
    edges: Vec<[[f64; 3]; 2]>,
}

impl EdgeBins {
    /// Bins over the box `lo..hi` (world units relative to the lattice
    /// origin) for samples culled at `cull`, or `None` if there are no edges
    /// or too many entries to be worth holding.
    pub(crate) fn new(
        graph: &SeedGraph,
        lo: [f64; 3],
        hi: [f64; 3],
        cull: f64,
    ) -> Option<EdgeBins> {
        // Negated so a NaN cull is refused too.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let unusable = !(cull > 0.0);
        if graph.edges.is_empty() || unusable {
            return None;
        }
        let extent = [0, 1, 2].map(|a| (hi[a] - lo[a]).max(1e-9));
        // About a cull across, but no more than 96 a side.
        let bin = cull.max(extent.iter().fold(0.0f64, |m, &e| m.max(e)) / 96.0);
        let dims = extent.map(|e| ((e / bin).ceil() as usize).max(1));
        let reach = cull * 1.001 + 1e-9 * bin;
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); dims[0] * dims[1] * dims[2]];
        let mut entries = 0usize;
        for (n, [a, b]) in graph.edges.iter().enumerate() {
            let range = |axis: usize| {
                let (lo_e, hi_e) = (a[axis].min(b[axis]) - reach, a[axis].max(b[axis]) + reach);
                let first = ((lo_e - lo[axis]) / bin).floor().max(0.0) as usize;
                let last = (((hi_e - lo[axis]) / bin).floor().max(-1.0) as isize)
                    .min(dims[axis] as isize - 1);
                (first, last)
            };
            let (rx, ry, rz) = (range(0), range(1), range(2));
            if rx.1 < rx.0 as isize || ry.1 < ry.0 as isize || rz.1 < rz.0 as isize {
                continue;
            }
            for z in rz.0..=rz.1 as usize {
                for y in ry.0..=ry.1 as usize {
                    for x in rx.0..=rx.1 as usize {
                        lists[(z * dims[1] + y) * dims[0] + x].push(n as u32);
                        entries += 1;
                    }
                }
            }
            if entries > 1 << 26 {
                return None;
            }
        }
        let mut start = Vec::with_capacity(lists.len() + 1);
        let mut list = Vec::with_capacity(entries);
        start.push(0);
        for l in lists {
            list.extend(l);
            start.push(list.len() as u32);
        }
        Some(EdgeBins {
            lo,
            bin,
            dims,
            cull,
            start,
            list,
            edges: graph.edges.clone(),
        })
    }

    /// Distance from `p` to the nearest edge, up to `cull`; `None` outside
    /// the bins or past the cull they were drawn up for.
    pub(crate) fn distance(&self, p: [f64; 3], cull: f64) -> Option<f32> {
        // Negated so a NaN cull falls back, as one past the bins' does.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(cull <= self.cull) {
            return None;
        }
        let mut cell = [0usize; 3];
        for a in 0..3 {
            let i = ((p[a] - self.lo[a]) / self.bin).floor();
            if !(i >= 0.0 && (i as usize) < self.dims[a]) {
                return None;
            }
            cell[a] = i as usize;
        }
        let b = (cell[2] * self.dims[1] + cell[1]) * self.dims[0] + cell[0];
        let mut best = cull * cull;
        for &n in &self.list[self.start[b] as usize..self.start[b + 1] as usize] {
            let [a, e] = self.edges[n as usize];
            let ab = [e[0] - a[0], e[1] - a[1], e[2] - a[2]];
            let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2]];
            let len = dot(ab, ab);
            let t = if len > 0.0 {
                (dot(ap, ab) / len).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d = [ap[0] - ab[0] * t, ap[1] - ab[1] * t, ap[2] - ab[2] * t];
            best = best.min(dot(d, d));
        }
        Some(best.sqrt() as f32)
    }
}

/// A convex polytope being cut down to a cell: its vertices, and its faces as
/// loops of vertex indices with the candidate each belongs to (`None` for the
/// starting box).
struct Polytope {
    vertices: Vec<[f64; 3]>,
    faces: Vec<(Option<usize>, Vec<usize>)>,
}

impl Polytope {
    /// An axis-aligned box of half-sides `h` about the origin.
    fn cube(h: [f64; 3]) -> Polytope {
        let v = |x: f64, y: f64, z: f64| [x * h[0], y * h[1], z * h[2]];
        let vertices = vec![
            v(-1.0, -1.0, -1.0),
            v(1.0, -1.0, -1.0),
            v(1.0, 1.0, -1.0),
            v(-1.0, 1.0, -1.0),
            v(-1.0, -1.0, 1.0),
            v(1.0, -1.0, 1.0),
            v(1.0, 1.0, 1.0),
            v(-1.0, 1.0, 1.0),
        ];
        let faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [0, 4, 7, 3],
        ]
        .iter()
        .map(|f| (None, f.to_vec()))
        .collect();
        Polytope { vertices, faces }
    }

    fn radius(&self) -> f64 {
        self.vertices
            .iter()
            .map(|v| dot(*v, *v))
            .fold(0.0, f64::max)
            .sqrt()
    }

    /// Keep the part where `n · x ≤ c`; the cut face belongs to `owner`.
    fn clip(&mut self, n: [f64; 3], c: f64, owner: usize, eps: f64) {
        let side: Vec<f64> = self.vertices.iter().map(|v| dot(n, *v) - c).collect();
        if side.iter().all(|&s| s <= eps) {
            return;
        }
        let mut cut_points: Vec<usize> = Vec::new();
        // New vertices on cut edges, each shared by the two faces on its edge:
        // a handful a cut, so a list beats a map.
        let mut edge_point: Vec<((usize, usize), usize)> = Vec::new();
        let mut faces = Vec::with_capacity(self.faces.len() + 1);
        for (who, face) in std::mem::take(&mut self.faces) {
            let mut kept = Vec::with_capacity(face.len() + 2);
            for i in 0..face.len() {
                let (a, b) = (face[i], face[(i + 1) % face.len()]);
                let (sa, sb) = (side[a], side[b]);
                if sa <= eps {
                    kept.push(a);
                    if sa >= -eps {
                        cut_points.push(a);
                    }
                }
                // A strict crossing gets a new vertex, shared by both faces
                // on the edge.
                if (sa < -eps && sb > eps) || (sa > eps && sb < -eps) {
                    let key = (a.min(b), a.max(b));
                    let id = match edge_point.iter().find(|(k, _)| *k == key) {
                        Some(&(_, id)) => id,
                        None => {
                            let t = sa / (sa - sb);
                            let (va, vb) = (self.vertices[a], self.vertices[b]);
                            self.vertices
                                .push([0, 1, 2].map(|k| va[k] + t * (vb[k] - va[k])));
                            edge_point.push((key, self.vertices.len() - 1));
                            self.vertices.len() - 1
                        }
                    };
                    kept.push(id);
                    cut_points.push(id);
                }
            }
            if kept.len() >= 3 {
                faces.push((who, kept));
            }
        }
        // The new face: the cut points, in order round their centre.
        cut_points.sort_unstable();
        cut_points.dedup();
        if cut_points.len() >= 3 {
            let centre = cut_points.iter().fold([0.0; 3], |acc, &i| {
                let v = self.vertices[i];
                [acc[0] + v[0], acc[1] + v[1], acc[2] + v[2]]
            });
            let centre = centre.map(|x| x / cut_points.len() as f64);
            let seed = if n[0].abs() < 0.9 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let u = cross(n, seed);
            let w = cross(n, u);
            let angle = |i: usize| {
                let v = self.vertices[i];
                let d = [v[0] - centre[0], v[1] - centre[1], v[2] - centre[2]];
                dot(d, w).atan2(dot(d, u))
            };
            cut_points.sort_by(|&a, &b| angle(a).total_cmp(&angle(b)));
            faces.push((Some(owner), cut_points));
        }
        self.faces = faces;
    }
}

/// Clip a box about `me` (weight `w_me`) by its bisectors with `candidates`
/// (positions relative to the same origin as `me`, weights, and a label), and
/// return the labels of the candidates left owning a face of real area, and
/// whether the box's own faces are all gone — the cell closed inside it.
fn clip_cell<L: Copy>(
    me: [f64; 3],
    w_me: f64,
    candidates: &[([f64; 3], f64, L)],
    cell: [f64; 3],
) -> (Vec<L>, bool, Vec<[[f64; 3]; 2]>) {
    let scale = cell[0].max(cell[1]).max(cell[2]);
    let eps = 1e-10 * scale;
    let mut poly = Polytope::cube(cell.map(|c| 2.5 * c));
    // Nearest candidates first, so the cell shrinks fast and the rest can be
    // ruled out by distance alone.
    let mut order: Vec<(f64, usize)> = candidates
        .iter()
        .enumerate()
        .map(|(i, (q, _, _))| {
            let d = [q[0] - me[0], q[1] - me[1], q[2] - me[2]];
            (dot(d, d).sqrt(), i)
        })
        .collect();
    order.sort_by(|a, b| a.0.total_cmp(&b.0));
    let w_max = candidates.iter().map(|c| c.1).fold(w_me, f64::max);
    for &(d, i) in &order {
        if d < 1e-12 {
            continue;
        }
        // The plane sits at (d² + w_me − w)/2d from me, at least
        // (d² + w_me − w_max)/2d, which only grows with d: once that clears
        // the cell, so does every candidate after.
        if (d * d + w_me - w_max) / (2.0 * d) > poly.radius() {
            break;
        }
        let (q, w, _) = candidates[i];
        let delta = [q[0] - me[0], q[1] - me[1], q[2] - me[2]];
        let n = delta.map(|x| x / d);
        let c = (d * d + w_me - w) / (2.0 * d);
        poly.clip(n, c, i, eps);
    }
    let closed = poly.faces.iter().all(|(who, _)| who.is_some());
    // The cell's edges, relative to `me`: each face's sides, once each.
    let mut edges = Vec::new();
    if closed {
        let mut sides = std::collections::HashSet::new();
        for (_, loop_) in &poly.faces {
            for i in 0..loop_.len() {
                let (a, b) = (loop_[i], loop_[(i + 1) % loop_.len()]);
                if sides.insert((a.min(b), a.max(b))) {
                    edges.push([poly.vertices[a], poly.vertices[b]]);
                }
            }
        }
    }
    // A face that is only an edge or a corner touching does not make a
    // neighbour; the cell's edges all lie on its real faces.
    let mut labels = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (who, loop_) in &poly.faces {
        let Some(i) = who else { continue };
        let a = poly.vertices[loop_[0]];
        let mut area = [0.0; 3];
        for k in 1..loop_.len() - 1 {
            let (b, c) = (poly.vertices[loop_[k]], poly.vertices[loop_[k + 1]]);
            let cr = cross(
                [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
            );
            area = [area[0] + cr[0], area[1] + cr[1], area[2] + cr[2]];
        }
        if dot(area, area).sqrt() > 1e-9 * scale * scale && seen.insert(*i) {
            labels.push(candidates[*i].2);
        }
    }
    (labels, closed, edges)
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
