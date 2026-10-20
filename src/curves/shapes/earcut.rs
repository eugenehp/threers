//! three.js r165 `Earcut` — its port of mapbox/earcut 2.2.4 — translated line
//! for line, so `ShapeUtils.triangulateShape` here returns the same triangles
//! in the same order as in the browser.
//!
//! The linked list lives in an arena: a node is an index, and `None` stands in
//! for JavaScript's `null` in the z-order links.

#[derive(Debug, Clone)]
struct Node {
    i: usize,
    x: f64,
    y: f64,
    prev: usize,
    next: usize,
    z: i32,
    prev_z: Option<usize>,
    next_z: Option<usize>,
    steiner: bool,
}

struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn node(&mut self, i: usize, x: f64, y: f64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            i,
            x,
            y,
            prev: id,
            next: id,
            z: 0,
            prev_z: None,
            next_z: None,
            steiner: false,
        });
        id
    }

    #[inline]
    fn x(&self, n: usize) -> f64 {
        self.nodes[n].x
    }
    #[inline]
    fn y(&self, n: usize) -> f64 {
        self.nodes[n].y
    }
    #[inline]
    fn next(&self, n: usize) -> usize {
        self.nodes[n].next
    }
    #[inline]
    fn prev(&self, n: usize) -> usize {
        self.nodes[n].prev
    }

    fn insert_node(&mut self, i: usize, x: f64, y: f64, last: Option<usize>) -> usize {
        let p = self.node(i, x, y);
        if let Some(last) = last {
            let ln = self.next(last);
            self.nodes[p].next = ln;
            self.nodes[p].prev = last;
            self.nodes[ln].prev = p;
            self.nodes[last].next = p;
        }
        p
    }

    fn remove_node(&mut self, p: usize) {
        let (pn, pp) = (self.next(p), self.prev(p));
        self.nodes[pn].prev = pp;
        self.nodes[pp].next = pn;
        let (pz, nz) = (self.nodes[p].prev_z, self.nodes[p].next_z);
        if let Some(pz) = pz {
            self.nodes[pz].next_z = nz;
        }
        if let Some(nz) = nz {
            self.nodes[nz].prev_z = pz;
        }
    }

    fn area(&self, p: usize, q: usize, r: usize) -> f64 {
        (self.y(q) - self.y(p)) * (self.x(r) - self.x(q))
            - (self.x(q) - self.x(p)) * (self.y(r) - self.y(q))
    }

    fn equals(&self, a: usize, b: usize) -> bool {
        self.x(a) == self.x(b) && self.y(a) == self.y(b)
    }
}

/// `Earcut.triangulate(data, holeIndices, dim)`: indices of vertices (not of
/// coordinates), three per triangle.
pub fn triangulate(data: &[f64], hole_indices: &[usize], dim: usize) -> Vec<u32> {
    let dim = dim.max(2);
    let has_holes = !hole_indices.is_empty();
    let outer_len = if has_holes {
        hole_indices[0] * dim
    } else {
        data.len()
    };
    let mut arena = Arena {
        nodes: Vec::with_capacity(data.len() / dim * 3 / 2 + 8),
    };
    let mut triangles = Vec::new();
    let Some(mut outer) = linked_list(&mut arena, data, 0, outer_len, dim, true) else {
        return triangles;
    };
    if arena.next(outer) == arena.prev(outer) {
        return triangles;
    }
    if has_holes {
        outer = eliminate_holes(&mut arena, data, hole_indices, outer, dim);
    }
    let (mut min_x, mut min_y, mut inv_size) = (0.0, 0.0, 0.0);
    if data.len() > 80 * dim {
        min_x = data[0];
        let mut max_x = data[0];
        min_y = data[1];
        let mut max_y = data[1];
        let mut i = dim;
        while i < outer_len {
            let (x, y) = (data[i], data[i + 1]);
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
            i += dim;
        }
        inv_size = f64::max(max_x - min_x, max_y - min_y);
        inv_size = if inv_size != 0.0 {
            32767.0 / inv_size
        } else {
            0.0
        };
    }
    earcut_linked(
        &mut arena,
        Some(outer),
        &mut triangles,
        dim,
        min_x,
        min_y,
        inv_size,
        0,
    );
    triangles
}

fn signed_area(data: &[f64], start: usize, end: usize, dim: usize) -> f64 {
    let mut sum = 0.0;
    if end < dim {
        return sum;
    }
    let mut j = end - dim;
    let mut i = start;
    while i < end {
        sum += (data[j] - data[i]) * (data[i + 1] + data[j + 1]);
        j = i;
        i += dim;
    }
    sum
}

fn linked_list(
    a: &mut Arena,
    data: &[f64],
    start: usize,
    end: usize,
    dim: usize,
    clockwise: bool,
) -> Option<usize> {
    let mut last: Option<usize> = None;
    if end <= start {
        return None;
    }
    if clockwise == (signed_area(data, start, end, dim) > 0.0) {
        let mut i = start;
        while i < end {
            last = Some(a.insert_node(i, data[i], data[i + 1], last));
            i += dim;
        }
    } else {
        let mut i = end as isize - dim as isize;
        while i >= start as isize {
            let iu = i as usize;
            last = Some(a.insert_node(iu, data[iu], data[iu + 1], last));
            i -= dim as isize;
        }
    }
    if let Some(l) = last {
        if a.equals(l, a.next(l)) {
            a.remove_node(l);
            last = Some(a.next(l));
        }
    }
    last
}

fn filter_points(a: &mut Arena, start: Option<usize>, end: Option<usize>) -> Option<usize> {
    let start = start?;
    let mut end = end.unwrap_or(start);
    let mut p = start;
    loop {
        let mut again = false;
        let pn = a.next(p);
        if !a.nodes[p].steiner && (a.equals(p, pn) || a.area(a.prev(p), p, pn) == 0.0) {
            a.remove_node(p);
            p = a.prev(p);
            end = p;
            if p == a.next(p) {
                break;
            }
            again = true;
        } else {
            p = a.next(p);
        }
        if !(again || p != end) {
            break;
        }
    }
    Some(end)
}

#[allow(clippy::too_many_arguments)]
fn earcut_linked(
    a: &mut Arena,
    ear: Option<usize>,
    triangles: &mut Vec<u32>,
    dim: usize,
    min_x: f64,
    min_y: f64,
    inv_size: f64,
    pass: u8,
) {
    let Some(mut ear) = ear else { return };
    if pass == 0 && inv_size != 0.0 {
        index_curve(a, ear, min_x, min_y, inv_size);
    }
    let mut stop = ear;
    while a.prev(ear) != a.next(ear) {
        let prev = a.prev(ear);
        let next = a.next(ear);
        let is_ear = if inv_size != 0.0 {
            is_ear_hashed(a, ear, min_x, min_y, inv_size)
        } else {
            is_ear(a, ear)
        };
        if is_ear {
            triangles.push((a.nodes[prev].i / dim) as u32);
            triangles.push((a.nodes[ear].i / dim) as u32);
            triangles.push((a.nodes[next].i / dim) as u32);
            a.remove_node(ear);
            ear = a.next(next);
            stop = a.next(next);
            continue;
        }
        ear = next;
        if ear == stop {
            if pass == 0 {
                let f = filter_points(a, Some(ear), None);
                earcut_linked(a, f, triangles, dim, min_x, min_y, inv_size, 1);
            } else if pass == 1 {
                let f = filter_points(a, Some(ear), None);
                let e = cure_local_intersections(a, f, triangles, dim);
                earcut_linked(a, e, triangles, dim, min_x, min_y, inv_size, 2);
            } else if pass == 2 {
                split_earcut(a, ear, triangles, dim, min_x, min_y, inv_size);
            }
            break;
        }
    }
}

fn bbox(a: &Arena, pa: usize, pb: usize, pc: usize) -> (f64, f64, f64, f64) {
    let (ax, bx, cx) = (a.x(pa), a.x(pb), a.x(pc));
    let (ay, by, cy) = (a.y(pa), a.y(pb), a.y(pc));
    let x0 = if ax < bx {
        if ax < cx {
            ax
        } else {
            cx
        }
    } else if bx < cx {
        bx
    } else {
        cx
    };
    let y0 = if ay < by {
        if ay < cy {
            ay
        } else {
            cy
        }
    } else if by < cy {
        by
    } else {
        cy
    };
    let x1 = if ax > bx {
        if ax > cx {
            ax
        } else {
            cx
        }
    } else if bx > cx {
        bx
    } else {
        cx
    };
    let y1 = if ay > by {
        if ay > cy {
            ay
        } else {
            cy
        }
    } else if by > cy {
        by
    } else {
        cy
    };
    (x0, y0, x1, y1)
}

fn is_ear(a: &Arena, ear: usize) -> bool {
    let (pa, pb, pc) = (a.prev(ear), ear, a.next(ear));
    if a.area(pa, pb, pc) >= 0.0 {
        return false;
    }
    let (ax, bx, cx, ay, by, cy) = (a.x(pa), a.x(pb), a.x(pc), a.y(pa), a.y(pb), a.y(pc));
    let (x0, y0, x1, y1) = bbox(a, pa, pb, pc);
    let mut p = a.next(pc);
    while p != pa {
        let (px, py) = (a.x(p), a.y(p));
        if px >= x0
            && px <= x1
            && py >= y0
            && py <= y1
            && point_in_triangle(ax, ay, bx, by, cx, cy, px, py)
            && a.area(a.prev(p), p, a.next(p)) >= 0.0
        {
            return false;
        }
        p = a.next(p);
    }
    true
}

fn is_ear_hashed(a: &Arena, ear: usize, min_x: f64, min_y: f64, inv_size: f64) -> bool {
    let (pa, pb, pc) = (a.prev(ear), ear, a.next(ear));
    if a.area(pa, pb, pc) >= 0.0 {
        return false;
    }
    let (ax, bx, cx, ay, by, cy) = (a.x(pa), a.x(pb), a.x(pc), a.y(pa), a.y(pb), a.y(pc));
    let (x0, y0, x1, y1) = bbox(a, pa, pb, pc);
    let min_z = z_order(x0, y0, min_x, min_y, inv_size);
    let max_z = z_order(x1, y1, min_x, min_y, inv_size);
    let blocks = |n: usize| -> bool {
        let (nx, ny) = (a.x(n), a.y(n));
        nx >= x0
            && nx <= x1
            && ny >= y0
            && ny <= y1
            && n != pa
            && n != pc
            && point_in_triangle(ax, ay, bx, by, cx, cy, nx, ny)
            && a.area(a.prev(n), n, a.next(n)) >= 0.0
    };
    let mut p = a.nodes[ear].prev_z;
    let mut n = a.nodes[ear].next_z;
    while let (Some(pp), Some(nn)) = (p, n) {
        if !(a.nodes[pp].z >= min_z && a.nodes[nn].z <= max_z) {
            break;
        }
        if blocks(pp) {
            return false;
        }
        p = a.nodes[pp].prev_z;
        if blocks(nn) {
            return false;
        }
        n = a.nodes[nn].next_z;
    }
    while let Some(pp) = p {
        if a.nodes[pp].z < min_z {
            break;
        }
        if blocks(pp) {
            return false;
        }
        p = a.nodes[pp].prev_z;
    }
    while let Some(nn) = n {
        if a.nodes[nn].z > max_z {
            break;
        }
        if blocks(nn) {
            return false;
        }
        n = a.nodes[nn].next_z;
    }
    true
}

fn cure_local_intersections(
    a: &mut Arena,
    start: Option<usize>,
    triangles: &mut Vec<u32>,
    dim: usize,
) -> Option<usize> {
    let mut start = start?;
    let mut p = start;
    loop {
        let pa = a.prev(p);
        let b = a.next(a.next(p));
        if !a.equals(pa, b)
            && intersects(a, pa, p, a.next(p), b)
            && locally_inside(a, pa, b)
            && locally_inside(a, b, pa)
        {
            triangles.push((a.nodes[pa].i / dim) as u32);
            triangles.push((a.nodes[p].i / dim) as u32);
            triangles.push((a.nodes[b].i / dim) as u32);
            a.remove_node(p);
            let pn = a.next(p);
            a.remove_node(pn);
            p = b;
            start = b;
        }
        p = a.next(p);
        if p == start {
            break;
        }
    }
    filter_points(a, Some(p), None)
}

fn split_earcut(
    a: &mut Arena,
    start: usize,
    triangles: &mut Vec<u32>,
    dim: usize,
    min_x: f64,
    min_y: f64,
    inv_size: f64,
) {
    let mut pa = start;
    loop {
        let mut b = a.next(a.next(pa));
        while b != a.prev(pa) {
            if a.nodes[pa].i != a.nodes[b].i && is_valid_diagonal(a, pa, b) {
                let c = split_polygon(a, pa, b);
                let an = a.next(pa);
                let fa = filter_points(a, Some(pa), Some(an));
                let cn = a.next(c);
                let fc = filter_points(a, Some(c), Some(cn));
                earcut_linked(a, fa, triangles, dim, min_x, min_y, inv_size, 0);
                earcut_linked(a, fc, triangles, dim, min_x, min_y, inv_size, 0);
                return;
            }
            b = a.next(b);
        }
        pa = a.next(pa);
        if pa == start {
            break;
        }
    }
}

fn eliminate_holes(
    a: &mut Arena,
    data: &[f64],
    hole_indices: &[usize],
    mut outer: usize,
    dim: usize,
) -> usize {
    let mut queue = Vec::with_capacity(hole_indices.len());
    let len = hole_indices.len();
    for i in 0..len {
        let start = hole_indices[i] * dim;
        let end = if i < len - 1 {
            hole_indices[i + 1] * dim
        } else {
            data.len()
        };
        if let Some(list) = linked_list(a, data, start, end, dim, false) {
            if list == a.next(list) {
                a.nodes[list].steiner = true;
            }
            queue.push(get_leftmost(a, list));
        }
    }
    // `Array.prototype.sort` is stable, as `sort_by` is.
    queue.sort_by(|&p, &q| {
        let d = a.x(p) - a.x(q);
        d.partial_cmp(&0.0).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &h in &queue {
        outer = eliminate_hole(a, h, outer);
    }
    outer
}

fn eliminate_hole(a: &mut Arena, hole: usize, outer: usize) -> usize {
    let Some(bridge) = find_hole_bridge(a, hole, outer) else {
        return outer;
    };
    let bridge_reverse = split_polygon(a, bridge, hole);
    let brn = a.next(bridge_reverse);
    filter_points(a, Some(bridge_reverse), Some(brn));
    let bn = a.next(bridge);
    filter_points(a, Some(bridge), Some(bn)).unwrap_or(bridge)
}

fn find_hole_bridge(a: &Arena, hole: usize, outer: usize) -> Option<usize> {
    let mut p = outer;
    let mut qx = f64::NEG_INFINITY;
    let mut m: Option<usize> = None;
    let (hx, hy) = (a.x(hole), a.y(hole));
    loop {
        let pn = a.next(p);
        if hy <= a.y(p) && hy >= a.y(pn) && a.y(pn) != a.y(p) {
            let x = a.x(p) + (hy - a.y(p)) * (a.x(pn) - a.x(p)) / (a.y(pn) - a.y(p));
            if x <= hx && x > qx {
                qx = x;
                let mm = if a.x(p) < a.x(pn) { p } else { pn };
                m = Some(mm);
                if x == hx {
                    return Some(mm);
                }
            }
        }
        p = pn;
        if p == outer {
            break;
        }
    }
    let mut m = m?;
    let stop = m;
    let (mx, my) = (a.x(m), a.y(m));
    let mut tan_min = f64::INFINITY;
    p = m;
    loop {
        let (px, py) = (a.x(p), a.y(p));
        if hx >= px
            && px >= mx
            && hx != px
            && point_in_triangle(
                if hy < my { hx } else { qx },
                hy,
                mx,
                my,
                if hy < my { qx } else { hx },
                hy,
                px,
                py,
            )
        {
            let tan = (hy - py).abs() / (hx - px);
            if locally_inside(a, p, hole)
                && (tan < tan_min
                    || (tan == tan_min
                        && (px > a.x(m) || (px == a.x(m) && sector_contains_sector(a, m, p)))))
            {
                m = p;
                tan_min = tan;
            }
        }
        p = a.next(p);
        if p == stop {
            break;
        }
    }
    Some(m)
}

fn sector_contains_sector(a: &Arena, m: usize, p: usize) -> bool {
    a.area(a.prev(m), m, a.prev(p)) < 0.0 && a.area(a.next(p), m, a.next(m)) < 0.0
}

fn index_curve(a: &mut Arena, start: usize, min_x: f64, min_y: f64, inv_size: f64) {
    let mut p = start;
    loop {
        if a.nodes[p].z == 0 {
            a.nodes[p].z = z_order(a.x(p), a.y(p), min_x, min_y, inv_size);
        }
        a.nodes[p].prev_z = Some(a.prev(p));
        a.nodes[p].next_z = Some(a.next(p));
        p = a.next(p);
        if p == start {
            break;
        }
    }
    if let Some(pz) = a.nodes[p].prev_z {
        a.nodes[pz].next_z = None;
    }
    a.nodes[p].prev_z = None;
    sort_linked(a, p);
}

fn sort_linked(a: &mut Arena, list: usize) -> Option<usize> {
    let mut list = Some(list);
    let mut in_size = 1;
    loop {
        let mut p = list;
        list = None;
        let mut tail: Option<usize> = None;
        let mut num_merges = 0;
        while let Some(pp) = p {
            let _ = pp;
            num_merges += 1;
            let mut q = p;
            let mut p_size = 0;
            for _ in 0..in_size {
                p_size += 1;
                q = q.and_then(|qq| a.nodes[qq].next_z);
                if q.is_none() {
                    break;
                }
            }
            let mut q_size = in_size;
            while p_size > 0 || (q_size > 0 && q.is_some()) {
                let e;
                if p_size != 0
                    && (q_size == 0
                        || q.is_none()
                        || a.nodes[p.unwrap()].z <= a.nodes[q.unwrap()].z)
                {
                    e = p.unwrap();
                    p = a.nodes[e].next_z;
                    p_size -= 1;
                } else {
                    e = q.unwrap();
                    q = a.nodes[e].next_z;
                    q_size -= 1;
                }
                if let Some(t) = tail {
                    a.nodes[t].next_z = Some(e);
                } else {
                    list = Some(e);
                }
                a.nodes[e].prev_z = tail;
                tail = Some(e);
            }
            p = q;
        }
        if let Some(t) = tail {
            a.nodes[t].next_z = None;
        }
        in_size *= 2;
        if num_merges <= 1 {
            break;
        }
    }
    list
}

/// JavaScript `x | 0`: ToInt32.
fn to_i32(v: f64) -> i32 {
    if !v.is_finite() {
        return 0;
    }
    let t = v.trunc();
    let m = t.rem_euclid(4294967296.0);
    (m as u32) as i32
}

fn z_order(x: f64, y: f64, min_x: f64, min_y: f64, inv_size: f64) -> i32 {
    let mut x = to_i32((x - min_x) * inv_size);
    let mut y = to_i32((y - min_y) * inv_size);
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    x | (y << 1)
}

fn get_leftmost(a: &Arena, start: usize) -> usize {
    let mut p = start;
    let mut leftmost = start;
    loop {
        if a.x(p) < a.x(leftmost) || (a.x(p) == a.x(leftmost) && a.y(p) < a.y(leftmost)) {
            leftmost = p;
        }
        p = a.next(p);
        if p == start {
            break;
        }
    }
    leftmost
}

#[allow(clippy::too_many_arguments)]
fn point_in_triangle(
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
    px: f64,
    py: f64,
) -> bool {
    (cx - px) * (ay - py) >= (ax - px) * (cy - py)
        && (ax - px) * (by - py) >= (bx - px) * (ay - py)
        && (bx - px) * (cy - py) >= (cx - px) * (by - py)
}

fn is_valid_diagonal(a: &Arena, pa: usize, b: usize) -> bool {
    let (an, ap) = (a.next(pa), a.prev(pa));
    let (bn, bp) = (a.next(b), a.prev(b));
    a.nodes[an].i != a.nodes[b].i
        && a.nodes[ap].i != a.nodes[b].i
        && !intersects_polygon(a, pa, b)
        && ((locally_inside(a, pa, b)
            && locally_inside(a, b, pa)
            && middle_inside(a, pa, b)
            && (a.area(ap, pa, bp) != 0.0 || a.area(pa, bp, b) != 0.0))
            || (a.equals(pa, b) && a.area(ap, pa, an) > 0.0 && a.area(bp, b, bn) > 0.0))
}

fn sign(n: f64) -> i32 {
    if n > 0.0 {
        1
    } else if n < 0.0 {
        -1
    } else {
        0
    }
}

fn on_segment(a: &Arena, p: usize, q: usize, r: usize) -> bool {
    a.x(q) <= a.x(p).max(a.x(r))
        && a.x(q) >= a.x(p).min(a.x(r))
        && a.y(q) <= a.y(p).max(a.y(r))
        && a.y(q) >= a.y(p).min(a.y(r))
}

fn intersects(a: &Arena, p1: usize, q1: usize, p2: usize, q2: usize) -> bool {
    let o1 = sign(a.area(p1, q1, p2));
    let o2 = sign(a.area(p1, q1, q2));
    let o3 = sign(a.area(p2, q2, p1));
    let o4 = sign(a.area(p2, q2, q1));
    if o1 != o2 && o3 != o4 {
        return true;
    }
    if o1 == 0 && on_segment(a, p1, p2, q1) {
        return true;
    }
    if o2 == 0 && on_segment(a, p1, q2, q1) {
        return true;
    }
    if o3 == 0 && on_segment(a, p2, p1, q2) {
        return true;
    }
    if o4 == 0 && on_segment(a, p2, q1, q2) {
        return true;
    }
    false
}

fn intersects_polygon(a: &Arena, pa: usize, b: usize) -> bool {
    let mut p = pa;
    let (ai, bi) = (a.nodes[pa].i, a.nodes[b].i);
    loop {
        let pn = a.next(p);
        if a.nodes[p].i != ai
            && a.nodes[pn].i != ai
            && a.nodes[p].i != bi
            && a.nodes[pn].i != bi
            && intersects(a, p, pn, pa, b)
        {
            return true;
        }
        p = pn;
        if p == pa {
            break;
        }
    }
    false
}

fn locally_inside(a: &Arena, pa: usize, b: usize) -> bool {
    let (ap, an) = (a.prev(pa), a.next(pa));
    if a.area(ap, pa, an) < 0.0 {
        a.area(pa, b, an) >= 0.0 && a.area(pa, ap, b) >= 0.0
    } else {
        a.area(pa, b, ap) < 0.0 || a.area(pa, an, b) < 0.0
    }
}

fn middle_inside(a: &Arena, pa: usize, b: usize) -> bool {
    let mut p = pa;
    let mut inside = false;
    let px = (a.x(pa) + a.x(b)) / 2.0;
    let py = (a.y(pa) + a.y(b)) / 2.0;
    loop {
        let pn = a.next(p);
        if ((a.y(p) > py) != (a.y(pn) > py))
            && a.y(pn) != a.y(p)
            && (px < (a.x(pn) - a.x(p)) * (py - a.y(p)) / (a.y(pn) - a.y(p)) + a.x(p))
        {
            inside = !inside;
        }
        p = pn;
        if p == pa {
            break;
        }
    }
    inside
}

fn split_polygon(a: &mut Arena, pa: usize, b: usize) -> usize {
    let a2 = a.node(a.nodes[pa].i, a.x(pa), a.y(pa));
    let b2 = a.node(a.nodes[b].i, a.x(b), a.y(b));
    let an = a.next(pa);
    let bp = a.prev(b);
    a.nodes[pa].next = b;
    a.nodes[b].prev = pa;
    a.nodes[a2].next = an;
    a.nodes[an].prev = a2;
    a.nodes[b2].next = a2;
    a.nodes[a2].prev = b2;
    a.nodes[bp].next = b2;
    a.nodes[b2].prev = bp;
    b2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square() {
        // Earcut's own README example.
        let tris = triangulate(&[10.0, 0.0, 0.0, 50.0, 60.0, 60.0, 70.0, 10.0], &[], 2);
        assert_eq!(tris, vec![1, 0, 3, 3, 2, 1]);
    }

    #[test]
    fn square_with_hole() {
        let tris = triangulate(
            &[
                0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0, 20.0, 20.0, 80.0, 20.0, 80.0, 80.0,
                20.0, 80.0,
            ],
            &[4],
            2,
        );
        assert_eq!(
            tris,
            vec![0, 4, 7, 5, 4, 0, 3, 0, 7, 5, 0, 1, 2, 3, 7, 6, 5, 1, 2, 7, 6, 6, 1, 2]
        );
    }
}
