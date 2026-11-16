//! Boolean operations on polygons (union, intersection, difference,
//! exclusive or) by Vatti's sweep, with results as paths or as a tree of
//! outers and holes.
//!
//! Records link by index into arenas, `NONE` for no link; records no longer
//! needed stay in their arena, unreachable. Where records are sorted with
//! ties, [`crate::std_sort`] puts them in libc++ `std::sort`'s order.

use std::collections::BinaryHeap;

use crate::std_sort;

use super::core::*;

const NONE: usize = usize::MAX;

/// `ClipType`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipType {
    NoClip,
    Intersection,
    Union,
    Difference,
    Xor,
}

/// `PathType`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathType {
    Subject,
    Clip,
}

/// `JoinWith`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JoinWith {
    NoJoin,
    Left,
    Right,
}

/// `VertexFlags`
const VF_OPEN_START: u32 = 1;
const VF_OPEN_END: u32 = 2;
const VF_LOCAL_MAX: u32 = 4;
const VF_LOCAL_MIN: u32 = 8;

#[derive(Clone, Copy, Debug)]
struct Vertex {
    pt: Point64,
    next: usize,
    prev: usize,
    flags: u32,
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            pt: Point64::default(),
            next: NONE,
            prev: NONE,
            flags: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct OutPt {
    pt: Point64,
    next: usize,
    prev: usize,
    outrec: usize,
    /// Index into the horizontal segment list.
    horz: usize,
}

/// `StripeIndex`: point-in-polygon tests by horizontal stripes.
#[derive(Clone, Debug, Default)]
struct StripeIndex {
    edges: Vec<(Point64, Point64)>,
    stripes: Vec<Vec<i32>>,
    y_min: i64,
    stripe_height: i64,
    stripe_count: i32,
}

impl StripeIndex {
    fn y_to_stripe(&self, y: i64) -> i32 {
        let s = (y - self.y_min) / self.stripe_height;

        if s < 0 {
            return 0;
        }

        if s >= i64::from(self.stripe_count) {
            return self.stripe_count - 1;
        }

        s as i32
    }

    fn build(&mut self, edges: Vec<(Point64, Point64)>) {
        self.edges = edges;
        self.stripes.clear();

        if self.edges.is_empty() {
            self.y_min = 0;
            self.stripe_height = 1;
            self.stripe_count = 1;
            return;
        }

        self.y_min = self.edges[0].0.y;
        let mut y_max = self.edges[0].0.y;

        for (a, b) in &self.edges {
            self.y_min = self.y_min.min(a.y.min(b.y));
            y_max = y_max.max(a.y.max(b.y));
        }

        self.stripe_count = ((self.edges.len() as f64).sqrt() as i32).clamp(1, 65536);

        let y_range = y_max - self.y_min;
        self.stripe_height = if y_range <= 0 {
            1
        } else {
            (y_range + i64::from(self.stripe_count) - 1) / i64::from(self.stripe_count)
        };

        let mut total_refs: i64 = 0;

        for (a, b) in &self.edges {
            total_refs +=
                i64::from(self.y_to_stripe(a.y.max(b.y)) - self.y_to_stripe(a.y.min(b.y)) + 1);
        }

        if total_refs > self.edges.len() as i64 * 8 {
            self.edges.clear();
            return;
        }

        self.stripes = vec![Vec::new(); self.stripe_count as usize];

        for i in 0..self.edges.len() {
            let (a, b) = self.edges[i];
            let s_min = self.y_to_stripe(a.y.min(b.y));
            let s_max = self.y_to_stripe(a.y.max(b.y));

            for s in s_min..=s_max {
                self.stripes[s as usize].push(i as i32);
            }
        }
    }

    fn point_in_polygon(&self, pt: Point64) -> PointInPolygonResult {
        if self.edges.is_empty() {
            return PointInPolygonResult::IsOutside;
        }

        let stripe = &self.stripes[self.y_to_stripe(pt.y) as usize];
        let mut inside = false;

        for &idx in stripe {
            let (a, b) = self.edges[idx as usize];
            let cross = cross_product_sign(a, b, pt);

            if cross == 0
                && pt.x >= a.x.min(b.x)
                && pt.x <= a.x.max(b.x)
                && pt.y >= a.y.min(b.y)
                && pt.y <= a.y.max(b.y)
            {
                return PointInPolygonResult::IsOn;
            }

            if (a.y > pt.y) != (b.y > pt.y) && (if b.y > pt.y { cross > 0 } else { cross < 0 }) {
                inside = !inside;
            }
        }

        if inside {
            PointInPolygonResult::IsInside
        } else {
            PointInPolygonResult::IsOutside
        }
    }
}

#[derive(Clone, Debug)]
struct OutRec {
    idx: usize,
    owner: usize,
    front_edge: usize,
    back_edge: usize,
    pts: usize,
    polypath: usize,
    splits: Option<Vec<usize>>,
    recursive_split: usize,
    bounds: Rect64,
    path: Path64,
    is_open: bool,
    pip_index: Option<StripeIndex>,
}

#[derive(Clone, Copy, Debug)]
struct Active {
    bot: Point64,
    top: Point64,
    curr_x: i64,
    dx: f64,
    wind_dx: i32,
    wind_cnt: i32,
    wind_cnt2: i32,
    outrec: usize,
    prev_in_ael: usize,
    next_in_ael: usize,
    prev_in_sel: usize,
    next_in_sel: usize,
    jump: usize,
    vertex_top: usize,
    local_min: usize,
    is_left_bound: bool,
    join_with: JoinWith,
}

impl Default for Active {
    fn default() -> Self {
        Active {
            bot: Point64::default(),
            top: Point64::default(),
            curr_x: 0,
            dx: 0.0,
            wind_dx: 1,
            wind_cnt: 0,
            wind_cnt2: 0,
            outrec: NONE,
            prev_in_ael: NONE,
            next_in_ael: NONE,
            prev_in_sel: NONE,
            next_in_sel: NONE,
            jump: NONE,
            vertex_top: NONE,
            local_min: NONE,
            is_left_bound: false,
            join_with: JoinWith::NoJoin,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LocalMinima {
    vertex: usize,
    polytype: PathType,
    is_open: bool,
}

#[derive(Clone, Copy, Debug)]
struct IntersectNode {
    pt: Point64,
    edge1: usize,
    edge2: usize,
}

#[derive(Clone, Copy, Debug)]
struct HorzSegment {
    left_op: usize,
    right_op: usize,
    left_to_right: bool,
}

#[derive(Clone, Copy, Debug)]
struct HorzJoin {
    op1: usize,
    op2: usize,
}

/// `ZCallback64`: given the two edges that cross and the point where, sets
/// its `z`.
pub type ZCallback64 = Box<dyn FnMut(&Point64, &Point64, &Point64, &Point64, &mut Point64)>;

/// `PolyPath64`: a closed path and the paths inside it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PolyPath64 {
    pub polygon: Path64,
    pub childs: Vec<PolyPath64>,
}

impl PolyPath64 {
    /// `Area`: of the path and all below it.
    pub fn area(&self) -> f64 {
        self.childs
            .iter()
            .fold(area(&self.polygon), |a, child| a + child.area())
    }
}

/// A node of the tree as it is built: by index, the root at 0.
#[derive(Clone, Debug, Default)]
struct PolyNode {
    polygon: Path64,
    childs: Vec<usize>,
}

/// `Boolean64`
pub struct Boolean64 {
    cliptype: ClipType,
    fillrule: FillRule,
    fillpos: FillRule,
    bot_y: i64,
    minima_list_sorted: bool,
    using_polytree: bool,
    actives: usize,
    sel: usize,
    minima: Vec<LocalMinima>,
    /// The order of `minima`, as `minima_list_` holds them.
    minima_list: Vec<usize>,
    current_locmin: usize,
    vertices: Vec<Vertex>,
    scanline_list: BinaryHeap<i64>,
    intersect_nodes: Vec<IntersectNode>,
    horz_seg_list: Vec<HorzSegment>,
    horz_join_list: Vec<HorzJoin>,
    actives_arena: Vec<Active>,
    outpts: Vec<OutPt>,
    outrec_list: Vec<OutRec>,
    tree: Vec<PolyNode>,
    pub preserve_collinear: bool,
    pub reverse_solution: bool,
    error_code: i32,
    has_open_paths: bool,
    succeeded: bool,
    pub pip_index_threshold: usize,
    z_callback: Option<ZCallback64>,
    pub default_z: i64,
}

impl Default for Boolean64 {
    fn default() -> Self {
        Self::new()
    }
}

/// `GetDx`
fn get_dx(pt1: Point64, pt2: Point64) -> f64 {
    let dy = (pt2.y - pt1.y) as f64;

    if dy != 0.0 {
        (pt2.x - pt1.x) as f64 / dy
    } else if pt2.x > pt1.x {
        -f64::MAX
    } else {
        f64::MAX
    }
}

/// `PtsReallyClose`
fn pts_really_close(pt1: Point64, pt2: Point64) -> bool {
    (pt1.x - pt2.x).abs() < 2 && (pt1.y - pt2.y).abs() < 2
}

/// `AreaTriangle`
fn area_triangle(pt1: Point64, pt2: Point64, pt3: Point64) -> f64 {
    // ((a + b) + c), each sum fused with a product
    let ab = ((pt3.y + pt1.y) as f64).mul_add(
        (pt3.x - pt1.x) as f64,
        (pt1.y + pt2.y) as f64 * (pt1.x - pt2.x) as f64,
    );
    ((pt2.y + pt3.y) as f64).mul_add((pt2.x - pt3.x) as f64, ab)
}

impl Boolean64 {
    pub fn new() -> Self {
        Boolean64 {
            cliptype: ClipType::NoClip,
            fillrule: FillRule::EvenOdd,
            fillpos: FillRule::Positive,
            bot_y: 0,
            minima_list_sorted: false,
            using_polytree: false,
            actives: NONE,
            sel: NONE,
            minima: Vec::new(),
            minima_list: Vec::new(),
            current_locmin: 0,
            vertices: Vec::new(),
            scanline_list: BinaryHeap::new(),
            intersect_nodes: Vec::new(),
            horz_seg_list: Vec::new(),
            horz_join_list: Vec::new(),
            actives_arena: Vec::new(),
            outpts: Vec::new(),
            outrec_list: Vec::new(),
            tree: Vec::new(),
            preserve_collinear: true,
            reverse_solution: false,
            error_code: 0,
            has_open_paths: false,
            succeeded: true,
            pip_index_threshold: 64,
            z_callback: None,
            default_z: 0,
        }
    }

    pub fn error_code(&self) -> i32 {
        self.error_code
    }

    /// `SetZCallback`
    pub fn set_z_callback(&mut self, cb: ZCallback64) {
        self.z_callback = Some(cb);
    }

    pub fn add_subject(&mut self, subjects: &[Path64]) {
        self.add_paths(subjects, PathType::Subject, false);
    }

    pub fn add_open_subject(&mut self, open_subjects: &[Path64]) {
        self.add_paths(open_subjects, PathType::Subject, true);
    }

    pub fn add_clip(&mut self, clips: &[Path64]) {
        self.add_paths(clips, PathType::Clip, false);
    }

    /// `Execute` to closed and open paths.
    pub fn execute_paths(
        &mut self,
        clip_type: ClipType,
        fill_rule: FillRule,
    ) -> (bool, Paths64, Paths64) {
        let mut closed = Vec::new();
        let mut open = Vec::new();

        if self.execute_internal(clip_type, fill_rule, false) {
            self.build_paths64(&mut closed, &mut open);
        }

        self.clean_up();
        (self.succeeded, closed, open)
    }

    /// `Execute` to a tree, and the open paths.
    pub fn execute_tree(
        &mut self,
        clip_type: ClipType,
        fill_rule: FillRule,
    ) -> (bool, PolyPath64, Paths64) {
        let mut open = Vec::new();
        let mut tree = PolyPath64::default();

        if self.execute_internal(clip_type, fill_rule, true) {
            self.tree = vec![PolyNode::default()];
            self.build_tree64(&mut open);
            tree = self.export_node(0);
        }

        self.clean_up();
        (self.succeeded, tree, open)
    }

    fn export_node(&self, index: usize) -> PolyPath64 {
        let node = &self.tree[index];

        PolyPath64 {
            polygon: node.polygon.clone(),
            childs: node.childs.iter().map(|c| self.export_node(*c)).collect(),
        }
    }

    // ---- small helpers on records

    fn is_hot_edge(&self, e: usize) -> bool {
        self.actives_arena[e].outrec != NONE
    }

    fn is_open(&self, e: usize) -> bool {
        self.minima[self.actives_arena[e].local_min].is_open
    }

    fn is_open_end_v(&self, v: usize) -> bool {
        self.vertices[v].flags & (VF_OPEN_START | VF_OPEN_END) != 0
    }

    fn is_open_end(&self, e: usize) -> bool {
        self.is_open_end_v(self.actives_arena[e].vertex_top)
    }

    fn get_prev_hot_edge(&self, e: usize) -> usize {
        let mut prev = self.actives_arena[e].prev_in_ael;

        while prev != NONE && (self.is_open(prev) || !self.is_hot_edge(prev)) {
            prev = self.actives_arena[prev].prev_in_ael;
        }

        prev
    }

    fn is_front(&self, e: usize) -> bool {
        e == self.outrec_list[self.actives_arena[e].outrec].front_edge
    }

    fn top_x(&self, e: usize, current_y: i64) -> i64 {
        let ae = &self.actives_arena[e];

        if current_y == ae.top.y || ae.top.x == ae.bot.x {
            ae.top.x
        } else if current_y == ae.bot.y {
            ae.bot.x
        } else {
            ae.bot.x + nearbyint(ae.dx * (current_y - ae.bot.y) as f64) as i64
        }
    }

    fn is_horizontal(&self, e: usize) -> bool {
        let ae = &self.actives_arena[e];
        ae.top.y == ae.bot.y
    }

    fn is_heading_right_horz(&self, e: usize) -> bool {
        self.actives_arena[e].dx == -f64::MAX
    }

    fn is_heading_left_horz(&self, e: usize) -> bool {
        self.actives_arena[e].dx == f64::MAX
    }

    fn get_poly_type(&self, e: usize) -> PathType {
        self.minima[self.actives_arena[e].local_min].polytype
    }

    fn is_same_poly_type(&self, e1: usize, e2: usize) -> bool {
        self.get_poly_type(e1) == self.get_poly_type(e2)
    }

    fn set_dx(&mut self, e: usize) {
        let ae = &mut self.actives_arena[e];
        ae.dx = get_dx(ae.bot, ae.top);
    }

    fn next_vertex(&self, e: usize) -> usize {
        let ae = &self.actives_arena[e];

        if ae.wind_dx > 0 {
            self.vertices[ae.vertex_top].next
        } else {
            self.vertices[ae.vertex_top].prev
        }
    }

    fn prev_prev_vertex(&self, e: usize) -> usize {
        let ae = &self.actives_arena[e];
        let v = &self.vertices;

        if ae.wind_dx > 0 {
            v[v[ae.vertex_top].prev].prev
        } else {
            v[v[ae.vertex_top].next].next
        }
    }

    fn extract_from_sel(&mut self, ae: usize) -> usize {
        let res = self.actives_arena[ae].next_in_sel;
        let prev = self.actives_arena[ae].prev_in_sel;

        if res != NONE {
            self.actives_arena[res].prev_in_sel = prev;
        }

        self.actives_arena[prev].next_in_sel = res;
        res
    }

    fn insert1_before2_in_sel(&mut self, ae1: usize, ae2: usize) {
        let prev = self.actives_arena[ae2].prev_in_sel;
        self.actives_arena[ae1].prev_in_sel = prev;

        if prev != NONE {
            self.actives_arena[prev].next_in_sel = ae1;
        }

        self.actives_arena[ae1].next_in_sel = ae2;
        self.actives_arena[ae2].prev_in_sel = ae1;
    }

    fn is_maxima_v(&self, v: usize) -> bool {
        self.vertices[v].flags & VF_LOCAL_MAX != 0
    }

    fn is_maxima(&self, e: usize) -> bool {
        self.is_maxima_v(self.actives_arena[e].vertex_top)
    }

    fn get_curr_y_maxima_vertex_open(&self, e: usize) -> usize {
        let ae = &self.actives_arena[e];
        let v = &self.vertices;
        let mut result = ae.vertex_top;

        if ae.wind_dx > 0 {
            while v[v[result].next].pt.y == v[result].pt.y
                && v[result].flags & (VF_OPEN_END | VF_LOCAL_MAX) == 0
            {
                result = v[result].next;
            }
        } else {
            while v[v[result].prev].pt.y == v[result].pt.y
                && v[result].flags & (VF_OPEN_END | VF_LOCAL_MAX) == 0
            {
                result = v[result].prev;
            }
        }

        if !self.is_maxima_v(result) {
            return NONE;
        }

        result
    }

    fn get_curr_y_maxima_vertex(&self, e: usize) -> usize {
        let ae = &self.actives_arena[e];
        let v = &self.vertices;
        let mut result = ae.vertex_top;

        if ae.wind_dx > 0 {
            while v[v[result].next].pt.y == v[result].pt.y {
                result = v[result].next;
            }
        } else {
            while v[v[result].prev].pt.y == v[result].pt.y {
                result = v[result].prev;
            }
        }

        if !self.is_maxima_v(result) {
            return NONE;
        }

        result
    }

    fn get_maxima_pair(&self, e: usize) -> usize {
        let mut e2 = self.actives_arena[e].next_in_ael;

        while e2 != NONE {
            if self.actives_arena[e2].vertex_top == self.actives_arena[e].vertex_top {
                return e2;
            }

            e2 = self.actives_arena[e2].next_in_ael;
        }

        NONE
    }

    fn new_out_pt(&mut self, pt: Point64, outrec: usize) -> usize {
        let index = self.outpts.len();
        self.outpts.push(OutPt {
            pt,
            next: index,
            prev: index,
            outrec,
            horz: NONE,
        });
        index
    }

    fn duplicate_op(&mut self, op: usize, insert_after: bool) -> usize {
        let result = self.new_out_pt(self.outpts[op].pt, self.outpts[op].outrec);

        if insert_after {
            let next = self.outpts[op].next;
            self.outpts[result].next = next;
            self.outpts[next].prev = result;
            self.outpts[result].prev = op;
            self.outpts[op].next = result;
        } else {
            let prev = self.outpts[op].prev;
            self.outpts[result].prev = prev;
            self.outpts[prev].next = result;
            self.outpts[result].next = op;
            self.outpts[op].prev = result;
        }

        result
    }

    fn dispose_out_pt(&mut self, op: usize) -> usize {
        let OutPt { next, prev, .. } = self.outpts[op];
        self.outpts[prev].next = next;
        self.outpts[next].prev = prev;
        next
    }

    fn dispose_out_pts(&mut self, outrec: usize) {
        self.outrec_list[outrec].pts = NONE;
    }

    fn set_sides(&mut self, outrec: usize, start_edge: usize, end_edge: usize) {
        self.outrec_list[outrec].front_edge = start_edge;
        self.outrec_list[outrec].back_edge = end_edge;
    }

    fn swap_outrecs(&mut self, e1: usize, e2: usize) {
        let or1 = self.actives_arena[e1].outrec;
        let or2 = self.actives_arena[e2].outrec;

        if or1 == or2 {
            let o = &mut self.outrec_list[or1];
            std::mem::swap(&mut o.front_edge, &mut o.back_edge);
            return;
        }

        if or1 != NONE {
            if e1 == self.outrec_list[or1].front_edge {
                self.outrec_list[or1].front_edge = e2;
            } else {
                self.outrec_list[or1].back_edge = e2;
            }
        }

        if or2 != NONE {
            if e2 == self.outrec_list[or2].front_edge {
                self.outrec_list[or2].front_edge = e1;
            } else {
                self.outrec_list[or2].back_edge = e1;
            }
        }

        self.actives_arena[e1].outrec = or2;
        self.actives_arena[e2].outrec = or1;
    }

    /// `Area( OutPt* )`
    fn area_op(&self, op: usize) -> f64 {
        let mut result = 0.0;
        let mut op2 = op;

        loop {
            let prev = self.outpts[op2].prev;
            result = ((self.outpts[prev].pt.y + self.outpts[op2].pt.y) as f64).mul_add(
                (self.outpts[prev].pt.x - self.outpts[op2].pt.x) as f64,
                result,
            );
            op2 = self.outpts[op2].next;

            if op2 == op {
                break;
            }
        }

        result * 0.5
    }

    fn get_real_outrec(&self, mut outrec: usize) -> usize {
        while outrec != NONE && self.outrec_list[outrec].pts == NONE {
            outrec = self.outrec_list[outrec].owner;
        }

        outrec
    }

    fn is_valid_owner(&self, outrec: usize, mut test_owner: usize) -> bool {
        while test_owner != NONE && test_owner != outrec {
            test_owner = self.outrec_list[test_owner].owner;
        }

        test_owner == NONE
    }

    fn uncouple_outrec(&mut self, ae: usize) {
        let outrec = self.actives_arena[ae].outrec;

        if outrec == NONE {
            return;
        }

        let front = self.outrec_list[outrec].front_edge;
        let back = self.outrec_list[outrec].back_edge;
        self.actives_arena[front].outrec = NONE;
        self.actives_arena[back].outrec = NONE;
        self.outrec_list[outrec].front_edge = NONE;
        self.outrec_list[outrec].back_edge = NONE;
    }

    fn is_very_small_triangle(&self, op: usize) -> bool {
        let o = &self.outpts;
        let OutPt { next, prev, pt, .. } = o[op];

        o[next].next == prev
            && (pts_really_close(o[prev].pt, o[next].pt)
                || pts_really_close(pt, o[next].pt)
                || pts_really_close(pt, o[prev].pt))
    }

    fn is_valid_closed_path(&self, op: usize) -> bool {
        op != NONE
            && self.outpts[op].next != op
            && self.outpts[op].next != self.outpts[op].prev
            && !self.is_very_small_triangle(op)
    }

    fn outrec_is_ascending(&self, hot_edge: usize) -> bool {
        hot_edge == self.outrec_list[self.actives_arena[hot_edge].outrec].front_edge
    }

    fn swap_front_back_sides(&mut self, outrec: usize) {
        let o = &mut self.outrec_list[outrec];
        std::mem::swap(&mut o.front_edge, &mut o.back_edge);
        o.pts = self.outpts[o.pts].next;
    }

    fn edges_adjacent_in_ael(&self, inode: &IntersectNode) -> bool {
        self.actives_arena[inode.edge1].next_in_ael == inode.edge2
            || self.actives_arena[inode.edge1].prev_in_ael == inode.edge2
    }

    fn is_joined(&self, e: usize) -> bool {
        self.actives_arena[e].join_with != JoinWith::NoJoin
    }

    fn set_owner(&mut self, outrec: usize, new_owner: usize) {
        let real = self.get_real_outrec(self.outrec_list[new_owner].owner);
        self.outrec_list[new_owner].owner = real;
        let mut tmp = new_owner;

        while tmp != NONE && tmp != outrec {
            tmp = self.outrec_list[tmp].owner;
        }

        if tmp != NONE {
            self.outrec_list[new_owner].owner = self.outrec_list[outrec].owner;
        }

        self.outrec_list[outrec].owner = new_owner;
    }

    /// `PointInOpPolygon`
    fn point_in_op_polygon(&self, pt: Point64, mut op: usize) -> PointInPolygonResult {
        let o = &self.outpts;

        if op == o[op].next || o[op].prev == o[op].next {
            return PointInPolygonResult::IsOutside;
        }

        let mut op2 = op;

        loop {
            if o[op].pt.y != pt.y {
                break;
            }

            op = o[op].next;

            if op == op2 {
                break;
            }
        }

        // not a proper polygon
        if o[op].pt.y == pt.y {
            return PointInPolygonResult::IsOutside;
        }

        let mut is_above = o[op].pt.y < pt.y;
        let starting_above = is_above;
        let mut val = 0;
        op2 = o[op].next;

        while op2 != op {
            if is_above {
                while op2 != op && o[op2].pt.y < pt.y {
                    op2 = o[op2].next;
                }
            } else {
                while op2 != op && o[op2].pt.y > pt.y {
                    op2 = o[op2].next;
                }
            }

            if op2 == op {
                break;
            }

            let prev = o[op2].prev;

            // touching the horizontal
            if o[op2].pt.y == pt.y {
                if o[op2].pt.x == pt.x
                    || (o[op2].pt.y == o[prev].pt.y
                        && (pt.x < o[prev].pt.x) != (pt.x < o[op2].pt.x))
                {
                    return PointInPolygonResult::IsOn;
                }

                op2 = o[op2].next;

                if op2 == op {
                    break;
                }

                continue;
            }

            if pt.x < o[op2].pt.x && pt.x < o[prev].pt.x {
                // to the right of the point
            } else if pt.x > o[prev].pt.x && pt.x > o[op2].pt.x {
                val = 1 - val;
            } else {
                let i = cross_product_sign(o[prev].pt, o[op2].pt, pt);

                if i == 0 {
                    return PointInPolygonResult::IsOn;
                }

                if (i < 0) == is_above {
                    val = 1 - val;
                }
            }

            is_above = !is_above;
            op2 = o[op2].next;
        }

        if is_above != starting_above {
            let prev = o[op2].prev;
            let i = cross_product_sign(o[prev].pt, o[op2].pt, pt);

            if i == 0 {
                return PointInPolygonResult::IsOn;
            }

            if (i < 0) == is_above {
                val = 1 - val;
            }
        }

        if val == 0 {
            PointInPolygonResult::IsOutside
        } else {
            PointInPolygonResult::IsInside
        }
    }

    /// `GetCleanPath`
    fn get_clean_path(&self, op: usize) -> Path64 {
        let o = &self.outpts;
        let mut result = Vec::new();
        let mut op2 = op;

        while o[op2].next != op
            && ((o[op2].pt.x == o[o[op2].next].pt.x && o[op2].pt.x == o[o[op2].prev].pt.x)
                || (o[op2].pt.y == o[o[op2].next].pt.y && o[op2].pt.y == o[o[op2].prev].pt.y))
        {
            op2 = o[op2].next;
        }

        result.push(o[op2].pt);
        let mut prev_op = op2;
        op2 = o[op2].next;

        while op2 != op {
            let next = o[op2].next;

            if (o[op2].pt.x != o[next].pt.x || o[op2].pt.x != o[prev_op].pt.x)
                && (o[op2].pt.y != o[next].pt.y || o[op2].pt.y != o[prev_op].pt.y)
            {
                result.push(o[op2].pt);
                prev_op = op2;
            }

            op2 = next;
        }

        result
    }

    /// `Path2ContainsPath1( OutPt*, OutPt* )`
    fn path2_contains_path1_op(&self, op1: usize, op2: usize) -> bool {
        let mut pip = PointInPolygonResult::IsOn;
        let mut op = op1;

        loop {
            match self.point_in_op_polygon(self.outpts[op].pt, op2) {
                PointInPolygonResult::IsOutside => {
                    if pip == PointInPolygonResult::IsOutside {
                        return false;
                    }

                    pip = PointInPolygonResult::IsOutside;
                }
                PointInPolygonResult::IsInside => {
                    if pip == PointInPolygonResult::IsInside {
                        return true;
                    }

                    pip = PointInPolygonResult::IsInside;
                }
                PointInPolygonResult::IsOn => {}
            }

            op = self.outpts[op].next;

            if op == op1 {
                break;
            }
        }

        path2_contains_path1(&self.get_clean_path(op1), &self.get_clean_path(op2))
    }

    /// `Path2ContainsPath1Indexed`
    fn path2_contains_path1_indexed(&mut self, op1: usize, owner: usize) -> bool {
        if self.outrec_list[owner].path.len() <= self.pip_index_threshold {
            return self.path2_contains_path1_op(op1, self.outrec_list[owner].pts);
        }

        if self.outrec_list[owner].pip_index.is_none() {
            let mut index = StripeIndex::default();
            index.build(self.ring_edges(self.outrec_list[owner].pts));
            self.outrec_list[owner].pip_index = Some(index);
        }

        if self.outrec_list[owner]
            .pip_index
            .as_ref()
            .is_some_and(|i| i.edges.is_empty())
        {
            return self.path2_contains_path1_op(op1, self.outrec_list[owner].pts);
        }

        let mut pip = PointInPolygonResult::IsOn;
        let mut op = op1;

        loop {
            let index = self.outrec_list[owner]
                .pip_index
                .as_ref()
                .expect("built above");

            match index.point_in_polygon(self.outpts[op].pt) {
                PointInPolygonResult::IsOutside => {
                    if pip == PointInPolygonResult::IsOutside {
                        return false;
                    }

                    pip = PointInPolygonResult::IsOutside;
                }
                PointInPolygonResult::IsInside => {
                    if pip == PointInPolygonResult::IsInside {
                        return true;
                    }

                    pip = PointInPolygonResult::IsInside;
                }
                PointInPolygonResult::IsOn => {}
            }

            op = self.outpts[op].next;

            if op == op1 {
                break;
            }
        }

        path2_contains_path1(
            &self.get_clean_path(op1),
            &self.get_clean_path(self.outrec_list[owner].pts),
        )
    }

    /// The edges of `StripeIndex::BuildFromRing`.
    fn ring_edges(&self, op: usize) -> Vec<(Point64, Point64)> {
        let o = &self.outpts;
        let mut edges = Vec::new();

        if op != NONE && o[op].next != op && o[op].prev != o[op].next {
            let mut cur = op;

            loop {
                let nxt = o[cur].next;

                if o[cur].pt.x != o[nxt].pt.x || o[cur].pt.y != o[nxt].pt.y {
                    edges.push((o[cur].pt, o[nxt].pt));
                }

                cur = nxt;

                if cur == op {
                    break;
                }
            }
        }

        edges
    }

    // ---- adding paths

    fn add_loc_min(&mut self, vert: usize, polytype: PathType, is_open: bool) {
        if self.vertices[vert].flags & VF_LOCAL_MIN != 0 {
            return;
        }

        self.vertices[vert].flags |= VF_LOCAL_MIN;
        self.minima_list.push(self.minima.len());
        self.minima.push(LocalMinima {
            vertex: vert,
            polytype,
            is_open,
        });
    }

    /// `AddPaths_`
    fn add_paths_(&mut self, paths: &[Path64], polytype: PathType, is_open: bool) {
        let total_vertex_count: usize = paths.iter().map(Vec::len).sum();

        if total_vertex_count == 0 {
            return;
        }

        let base = self.vertices.len();
        self.vertices
            .resize(base + total_vertex_count, Vertex::default());
        let mut v = base;

        for path in paths {
            let v0 = v;
            let mut curr_v = v;
            let mut prev_v = NONE;

            if path.is_empty() {
                continue;
            }

            self.vertices[v].prev = NONE;
            let mut cnt = 0;

            for pt in path {
                if prev_v != NONE {
                    // skip duplicates
                    if self.vertices[prev_v].pt == *pt {
                        continue;
                    }

                    self.vertices[prev_v].next = curr_v;
                }

                self.vertices[curr_v].prev = prev_v;
                self.vertices[curr_v].pt = *pt;
                self.vertices[curr_v].flags = 0;
                prev_v = curr_v;
                curr_v += 1;
                cnt += 1;
            }

            if prev_v == NONE || self.vertices[prev_v].prev == NONE {
                continue;
            }

            if !is_open && self.vertices[prev_v].pt == self.vertices[v0].pt {
                prev_v = self.vertices[prev_v].prev;
            }

            self.vertices[prev_v].next = v0;
            self.vertices[v0].prev = prev_v;
            // ready for the next path
            v = curr_v;

            if cnt < 2 || (cnt == 2 && !is_open) {
                continue;
            }

            let mut going_up;

            if is_open {
                curr_v = self.vertices[v0].next;

                while curr_v != v0 && self.vertices[curr_v].pt.y == self.vertices[v0].pt.y {
                    curr_v = self.vertices[curr_v].next;
                }

                going_up = self.vertices[curr_v].pt.y <= self.vertices[v0].pt.y;

                if going_up {
                    self.vertices[v0].flags = VF_OPEN_START;
                    self.add_loc_min(v0, polytype, true);
                } else {
                    self.vertices[v0].flags = VF_OPEN_START | VF_LOCAL_MAX;
                }
            } else {
                prev_v = self.vertices[v0].prev;

                while prev_v != v0 && self.vertices[prev_v].pt.y == self.vertices[v0].pt.y {
                    prev_v = self.vertices[prev_v].prev;
                }

                // only open paths can be completely flat
                if prev_v == v0 {
                    continue;
                }

                going_up = self.vertices[prev_v].pt.y > self.vertices[v0].pt.y;
            }

            let going_up0 = going_up;
            prev_v = v0;
            curr_v = self.vertices[v0].next;

            while curr_v != v0 {
                if self.vertices[curr_v].pt.y > self.vertices[prev_v].pt.y && going_up {
                    self.vertices[prev_v].flags |= VF_LOCAL_MAX;
                    going_up = false;
                } else if self.vertices[curr_v].pt.y < self.vertices[prev_v].pt.y && !going_up {
                    going_up = true;
                    self.add_loc_min(prev_v, polytype, is_open);
                }

                prev_v = curr_v;
                curr_v = self.vertices[curr_v].next;
            }

            if is_open {
                self.vertices[prev_v].flags |= VF_OPEN_END;

                if going_up {
                    self.vertices[prev_v].flags |= VF_LOCAL_MAX;
                } else {
                    self.add_loc_min(prev_v, polytype, is_open);
                }
            } else if going_up != going_up0 {
                if going_up0 {
                    self.add_loc_min(prev_v, polytype, false);
                } else {
                    self.vertices[prev_v].flags |= VF_LOCAL_MAX;
                }
            }
        }
    }

    /// `AddPaths`
    pub fn add_paths(&mut self, paths: &[Path64], polytype: PathType, is_open: bool) {
        if is_open {
            self.has_open_paths = true;
        }

        self.minima_list_sorted = false;
        self.add_paths_(paths, polytype, is_open);
    }

    // ---- the sweep

    fn clean_up(&mut self) {
        self.actives = NONE;
        self.actives_arena.clear();
        self.scanline_list.clear();
        self.intersect_nodes.clear();
        self.outrec_list.clear();
        self.outpts.clear();
        self.horz_seg_list.clear();
        self.horz_join_list.clear();
    }

    /// `Clear`
    pub fn clear(&mut self) {
        self.clean_up();
        self.minima.clear();
        self.minima_list.clear();
        self.vertices.clear();
        self.current_locmin = 0;
        self.minima_list_sorted = false;
        self.has_open_paths = false;
    }

    fn reset(&mut self) {
        if !self.minima_list_sorted {
            // LocMinSorter, stable (#594): by y descending, then x ascending
            let minima = &self.minima;
            let vertices = &self.vertices;

            self.minima_list.sort_by(|&a, &b| {
                let pa = vertices[minima[a].vertex].pt;
                let pb = vertices[minima[b].vertex].pt;
                pb.y.cmp(&pa.y).then(pa.x.cmp(&pb.x))
            });

            self.minima_list_sorted = true;
        }

        for i in (0..self.minima_list.len()).rev() {
            let y = self.vertices[self.minima[self.minima_list[i]].vertex].pt.y;
            self.scanline_list.push(y);
        }

        self.current_locmin = 0;
        self.actives = NONE;
        self.sel = NONE;
        self.succeeded = true;
    }

    fn set_z(&mut self, e1: usize, e2: usize, ip: &mut Point64) {
        let Some(cb) = self.z_callback.as_mut() else {
            return;
        };
        let a1 = self.actives_arena[e1];
        let a2 = self.actives_arena[e2];
        let polytype = self.minima[a1.local_min].polytype;

        if polytype == PathType::Subject {
            if *ip == a1.bot {
                ip.z = a1.bot.z;
            } else if *ip == a1.top {
                ip.z = a1.top.z;
            } else if *ip == a2.bot {
                ip.z = a2.bot.z;
            } else if *ip == a2.top {
                ip.z = a2.top.z;
            } else {
                ip.z = self.default_z;
            }

            cb(&a1.bot, &a1.top, &a2.bot, &a2.top, ip);
        } else {
            if *ip == a2.bot {
                ip.z = a2.bot.z;
            } else if *ip == a2.top {
                ip.z = a2.top.z;
            } else if *ip == a1.bot {
                ip.z = a1.bot.z;
            } else if *ip == a1.top {
                ip.z = a1.top.z;
            } else {
                ip.z = self.default_z;
            }

            cb(&a2.bot, &a2.top, &a1.bot, &a1.top, ip);
        }
    }

    /// `SetZ` on the point of an output record.
    fn set_z_op(&mut self, e1: usize, e2: usize, op: usize) {
        let mut pt = self.outpts[op].pt;
        self.set_z(e1, e2, &mut pt);
        self.outpts[op].pt = pt;
    }

    fn insert_scanline(&mut self, y: i64) {
        self.scanline_list.push(y);
    }

    fn pop_scanline(&mut self) -> Option<i64> {
        let y = self.scanline_list.pop()?;

        // pop duplicates
        while self.scanline_list.peek() == Some(&y) {
            self.scanline_list.pop();
        }

        Some(y)
    }

    fn pop_local_minima(&mut self, y: i64) -> Option<usize> {
        if self.current_locmin == self.minima_list.len()
            || self.vertices[self.minima[self.minima_list[self.current_locmin]].vertex]
                .pt
                .y
                != y
        {
            return None;
        }

        let result = self.minima_list[self.current_locmin];
        self.current_locmin += 1;
        Some(result)
    }

    fn is_contributing_closed(&self, e: usize) -> bool {
        let ae = &self.actives_arena[e];

        match self.fillrule {
            FillRule::EvenOdd => {}
            FillRule::NonZero => {
                if ae.wind_cnt.abs() != 1 {
                    return false;
                }
            }
            FillRule::Positive => {
                if ae.wind_cnt != 1 {
                    return false;
                }
            }
            FillRule::Negative => {
                if ae.wind_cnt != -1 {
                    return false;
                }
            }
        }

        match self.cliptype {
            ClipType::NoClip => false,
            ClipType::Intersection => match self.fillrule {
                FillRule::Positive => ae.wind_cnt2 > 0,
                FillRule::Negative => ae.wind_cnt2 < 0,
                _ => ae.wind_cnt2 != 0,
            },
            ClipType::Union => match self.fillrule {
                FillRule::Positive => ae.wind_cnt2 <= 0,
                FillRule::Negative => ae.wind_cnt2 >= 0,
                _ => ae.wind_cnt2 == 0,
            },
            ClipType::Difference => {
                let result = match self.fillrule {
                    FillRule::Positive => ae.wind_cnt2 <= 0,
                    FillRule::Negative => ae.wind_cnt2 >= 0,
                    _ => ae.wind_cnt2 == 0,
                };

                if self.get_poly_type(e) == PathType::Subject {
                    result
                } else {
                    !result
                }
            }
            ClipType::Xor => true,
        }
    }

    fn is_contributing_open(&self, e: usize) -> bool {
        let ae = &self.actives_arena[e];

        let (is_in_clip, is_in_subj) = match self.fillrule {
            FillRule::Positive => (ae.wind_cnt2 > 0, ae.wind_cnt > 0),
            FillRule::Negative => (ae.wind_cnt2 < 0, ae.wind_cnt < 0),
            _ => (ae.wind_cnt2 != 0, ae.wind_cnt != 0),
        };

        match self.cliptype {
            ClipType::Intersection => is_in_clip,
            ClipType::Union => !is_in_subj && !is_in_clip,
            _ => !is_in_clip,
        }
    }

    fn set_wind_count_for_closed_path_edge(&mut self, e: usize) {
        let mut e2 = self.actives_arena[e].prev_in_ael;
        let pt = self.get_poly_type(e);

        while e2 != NONE && (self.get_poly_type(e2) != pt || self.is_open(e2)) {
            e2 = self.actives_arena[e2].prev_in_ael;
        }

        if e2 == NONE {
            self.actives_arena[e].wind_cnt = self.actives_arena[e].wind_dx;
            e2 = self.actives;
        } else if self.fillrule == FillRule::EvenOdd {
            self.actives_arena[e].wind_cnt = self.actives_arena[e].wind_dx;
            self.actives_arena[e].wind_cnt2 = self.actives_arena[e2].wind_cnt2;
            e2 = self.actives_arena[e2].next_in_ael;
        } else {
            let a2 = self.actives_arena[e2];
            let wind_dx = self.actives_arena[e].wind_dx;

            let wind_cnt = if a2.wind_cnt * a2.wind_dx < 0 {
                if a2.wind_cnt.abs() > 1 {
                    if a2.wind_dx * wind_dx < 0 {
                        a2.wind_cnt
                    } else {
                        a2.wind_cnt + wind_dx
                    }
                } else if self.is_open(e) {
                    1
                } else {
                    wind_dx
                }
            } else if a2.wind_dx * wind_dx < 0 {
                a2.wind_cnt
            } else {
                a2.wind_cnt + wind_dx
            };

            self.actives_arena[e].wind_cnt = wind_cnt;
            self.actives_arena[e].wind_cnt2 = a2.wind_cnt2;
            // ready for wind_cnt2
            e2 = a2.next_in_ael;
        }

        if self.fillrule == FillRule::EvenOdd {
            while e2 != e {
                if self.get_poly_type(e2) != pt && !self.is_open(e2) {
                    let w = &mut self.actives_arena[e].wind_cnt2;
                    *w = if *w == 0 { 1 } else { 0 };
                }

                e2 = self.actives_arena[e2].next_in_ael;
            }
        } else {
            while e2 != e {
                if self.get_poly_type(e2) != pt && !self.is_open(e2) {
                    self.actives_arena[e].wind_cnt2 += self.actives_arena[e2].wind_dx;
                }

                e2 = self.actives_arena[e2].next_in_ael;
            }
        }
    }

    fn set_wind_count_for_open_path_edge(&mut self, e: usize) {
        let mut e2 = self.actives;

        if self.fillrule == FillRule::EvenOdd {
            let (mut cnt1, mut cnt2) = (0, 0);

            while e2 != e {
                if self.get_poly_type(e2) == PathType::Clip {
                    cnt2 += 1;
                } else if !self.is_open(e2) {
                    cnt1 += 1;
                }

                e2 = self.actives_arena[e2].next_in_ael;
            }

            self.actives_arena[e].wind_cnt = cnt1 & 1;
            self.actives_arena[e].wind_cnt2 = cnt2 & 1;
        } else {
            while e2 != e {
                if self.get_poly_type(e2) == PathType::Clip {
                    self.actives_arena[e].wind_cnt2 += self.actives_arena[e2].wind_dx;
                } else if !self.is_open(e2) {
                    self.actives_arena[e].wind_cnt += self.actives_arena[e2].wind_dx;
                }

                e2 = self.actives_arena[e2].next_in_ael;
            }
        }
    }

    /// `IsValidAelOrder`
    fn is_valid_ael_order(&self, resident: usize, newcomer: usize) -> bool {
        let r = &self.actives_arena[resident];
        let n = &self.actives_arena[newcomer];

        if n.curr_x != r.curr_x {
            return n.curr_x > r.curr_x;
        }

        // get the turning direction  a1.top, a2.bot, a2.top
        let i = cross_product_sign(r.top, n.bot, n.top);

        if i != 0 {
            return i < 0;
        }

        // edges must be collinear to get here
        if !self.is_maxima(resident) && r.top.y > n.top.y {
            return cross_product_sign(n.bot, r.top, self.vertices[self.next_vertex(resident)].pt)
                <= 0;
        } else if !self.is_maxima(newcomer) && n.top.y > r.top.y {
            return cross_product_sign(n.bot, n.top, self.vertices[self.next_vertex(newcomer)].pt)
                >= 0;
        }

        let y = n.bot.y;
        let newcomer_is_left = n.is_left_bound;

        if r.bot.y != y || self.vertices[self.minima[r.local_min].vertex].pt.y != y {
            n.is_left_bound
        } else if r.is_left_bound != newcomer_is_left {
            newcomer_is_left
        } else if is_collinear(
            self.vertices[self.prev_prev_vertex(resident)].pt,
            r.bot,
            r.top,
        ) {
            true
        } else {
            (cross_product_sign(
                self.vertices[self.prev_prev_vertex(resident)].pt,
                n.bot,
                self.vertices[self.prev_prev_vertex(newcomer)].pt,
            ) > 0)
                == newcomer_is_left
        }
    }

    fn insert_left_edge(&mut self, e: usize) {
        if self.actives == NONE {
            self.actives_arena[e].prev_in_ael = NONE;
            self.actives_arena[e].next_in_ael = NONE;
            self.actives = e;
        } else if !self.is_valid_ael_order(self.actives, e) {
            self.actives_arena[e].prev_in_ael = NONE;
            self.actives_arena[e].next_in_ael = self.actives;
            let a = self.actives;
            self.actives_arena[a].prev_in_ael = e;
            self.actives = e;
        } else {
            let mut e2 = self.actives;

            while self.actives_arena[e2].next_in_ael != NONE
                && self.is_valid_ael_order(self.actives_arena[e2].next_in_ael, e)
            {
                e2 = self.actives_arena[e2].next_in_ael;
            }

            // don't separate joined edges
            if self.actives_arena[e2].join_with == JoinWith::Right {
                e2 = self.actives_arena[e2].next_in_ael;
            }

            if e2 == NONE {
                return;
            }

            let next = self.actives_arena[e2].next_in_ael;
            self.actives_arena[e].next_in_ael = next;

            if next != NONE {
                self.actives_arena[next].prev_in_ael = e;
            }

            self.actives_arena[e].prev_in_ael = e2;
            self.actives_arena[e2].next_in_ael = e;
        }
    }

    fn insert_right_edge(&mut self, e: usize, e2: usize) {
        let next = self.actives_arena[e].next_in_ael;
        self.actives_arena[e2].next_in_ael = next;

        if next != NONE {
            self.actives_arena[next].prev_in_ael = e2;
        }

        self.actives_arena[e2].prev_in_ael = e;
        self.actives_arena[e].next_in_ael = e2;
    }

    fn new_active(&mut self, active: Active) -> usize {
        self.actives_arena.push(active);
        self.actives_arena.len() - 1
    }

    fn insert_local_minima_into_ael(&mut self, bot_y: i64) {
        while let Some(local_minima) = self.pop_local_minima(bot_y) {
            let vertex = self.minima[local_minima].vertex;
            let flags = self.vertices[vertex].flags;

            let mut left_bound = if flags & VF_OPEN_START != 0 {
                NONE
            } else {
                let vertex_top = self.vertices[vertex].prev;
                let bot = self.vertices[vertex].pt;
                let e = self.new_active(Active {
                    bot,
                    curr_x: bot.x,
                    wind_dx: -1,
                    // descending
                    vertex_top,
                    top: self.vertices[vertex_top].pt,
                    local_min: local_minima,
                    ..Active::default()
                });
                self.set_dx(e);
                e
            };

            let mut right_bound = if flags & VF_OPEN_END != 0 {
                NONE
            } else {
                let vertex_top = self.vertices[vertex].next;
                let bot = self.vertices[vertex].pt;
                let e = self.new_active(Active {
                    bot,
                    curr_x: bot.x,
                    wind_dx: 1,
                    // ascending
                    vertex_top,
                    top: self.vertices[vertex_top].pt,
                    local_min: local_minima,
                    ..Active::default()
                });
                self.set_dx(e);
                e
            };

            // currently left_bound is descending and right_bound ascending,
            // so they may need swapping
            if left_bound != NONE && right_bound != NONE {
                if self.is_horizontal(left_bound) {
                    if self.is_heading_right_horz(left_bound) {
                        std::mem::swap(&mut left_bound, &mut right_bound);
                    }
                } else if self.is_horizontal(right_bound) {
                    if self.is_heading_left_horz(right_bound) {
                        std::mem::swap(&mut left_bound, &mut right_bound);
                    }
                } else if self.actives_arena[left_bound].dx < self.actives_arena[right_bound].dx {
                    std::mem::swap(&mut left_bound, &mut right_bound);
                }
            } else if left_bound == NONE {
                left_bound = right_bound;
                right_bound = NONE;
            }

            self.actives_arena[left_bound].is_left_bound = true;
            self.insert_left_edge(left_bound);

            let contributing = if self.is_open(left_bound) {
                self.set_wind_count_for_open_path_edge(left_bound);
                self.is_contributing_open(left_bound)
            } else {
                self.set_wind_count_for_closed_path_edge(left_bound);
                self.is_contributing_closed(left_bound)
            };

            if right_bound != NONE {
                self.actives_arena[right_bound].is_left_bound = false;
                self.actives_arena[right_bound].wind_cnt = self.actives_arena[left_bound].wind_cnt;
                self.actives_arena[right_bound].wind_cnt2 =
                    self.actives_arena[left_bound].wind_cnt2;
                self.insert_right_edge(left_bound, right_bound);

                if contributing {
                    let bot = self.actives_arena[left_bound].bot;
                    self.add_local_min_poly(left_bound, right_bound, bot, true);

                    if !self.is_horizontal(left_bound) {
                        self.check_join_left(left_bound, bot, false);
                    }
                }

                while self.actives_arena[right_bound].next_in_ael != NONE
                    && self.is_valid_ael_order(
                        self.actives_arena[right_bound].next_in_ael,
                        right_bound,
                    )
                {
                    let next = self.actives_arena[right_bound].next_in_ael;
                    let bot = self.actives_arena[right_bound].bot;
                    self.intersect_edges(right_bound, next, bot);
                    self.swap_positions_in_ael(right_bound, next);
                }

                if self.is_horizontal(right_bound) {
                    self.push_horz(right_bound);
                } else {
                    let bot = self.actives_arena[right_bound].bot;
                    self.check_join_right(right_bound, bot, false);
                    self.insert_scanline(self.actives_arena[right_bound].top.y);
                }
            } else if contributing {
                let bot = self.actives_arena[left_bound].bot;
                self.start_open_path(left_bound, bot);
            }

            if self.is_horizontal(left_bound) {
                self.push_horz(left_bound);
            } else {
                self.insert_scanline(self.actives_arena[left_bound].top.y);
            }
        }
    }

    fn push_horz(&mut self, e: usize) {
        self.actives_arena[e].next_in_sel = self.sel;
        self.sel = e;
    }

    fn pop_horz(&mut self) -> Option<usize> {
        let e = self.sel;

        if e == NONE {
            return None;
        }

        self.sel = self.actives_arena[e].next_in_sel;
        Some(e)
    }

    fn add_local_min_poly(&mut self, e1: usize, e2: usize, pt: Point64, is_new: bool) -> usize {
        let outrec = self.new_out_rec();
        self.actives_arena[e1].outrec = outrec;
        self.actives_arena[e2].outrec = outrec;

        if self.is_open(e1) {
            self.outrec_list[outrec].owner = NONE;
            self.outrec_list[outrec].is_open = true;

            if self.actives_arena[e1].wind_dx > 0 {
                self.set_sides(outrec, e1, e2);
            } else {
                self.set_sides(outrec, e2, e1);
            }
        } else {
            let prev_hot_edge = self.get_prev_hot_edge(e1);

            if prev_hot_edge != NONE {
                if self.using_polytree {
                    self.set_owner(outrec, self.actives_arena[prev_hot_edge].outrec);
                }

                if self.outrec_is_ascending(prev_hot_edge) == is_new {
                    self.set_sides(outrec, e2, e1);
                } else {
                    self.set_sides(outrec, e1, e2);
                }
            } else {
                self.outrec_list[outrec].owner = NONE;

                if is_new {
                    self.set_sides(outrec, e1, e2);
                } else {
                    self.set_sides(outrec, e2, e1);
                }
            }
        }

        let op = self.new_out_pt(pt, outrec);
        self.outrec_list[outrec].pts = op;
        op
    }

    /// `AddLocalMaxPoly`: `NONE` when it fails.
    fn add_local_max_poly(&mut self, e1: usize, e2: usize, pt: Point64) -> usize {
        if self.is_joined(e1) {
            self.split(e1, pt);
        }

        if self.is_joined(e2) {
            self.split(e2, pt);
        }

        if self.is_front(e1) == self.is_front(e2) {
            if self.is_open_end(e1) {
                self.swap_front_back_sides(self.actives_arena[e1].outrec);
            } else if self.is_open_end(e2) {
                self.swap_front_back_sides(self.actives_arena[e2].outrec);
            } else {
                self.succeeded = false;
                return NONE;
            }
        }

        let mut result = self.add_out_pt(e1, pt);

        if self.actives_arena[e1].outrec == self.actives_arena[e2].outrec {
            let outrec = self.actives_arena[e1].outrec;
            self.outrec_list[outrec].pts = result;

            if self.using_polytree {
                let e = self.get_prev_hot_edge(e1);

                if e == NONE {
                    self.outrec_list[outrec].owner = NONE;
                } else {
                    self.set_owner(outrec, self.actives_arena[e].outrec);
                }
                // nb: outRec.owner here is likely NOT the real owner but
                // this will be fixed in DeepCheckOwner()
            }

            self.uncouple_outrec(e1);
            result = self.outrec_list[outrec].pts;
            let owner = self.outrec_list[outrec].owner;

            if owner != NONE && self.outrec_list[owner].front_edge == NONE {
                self.outrec_list[outrec].owner = self.get_real_outrec(owner);
            }
        } else if self.is_open(e1) {
            // and to preserve the winding orientation of outrec
            if self.actives_arena[e1].wind_dx < 0 {
                self.join_outrec_paths(e1, e2);
            } else {
                self.join_outrec_paths(e2, e1);
            }
        } else if self.outrec_list[self.actives_arena[e1].outrec].idx
            < self.outrec_list[self.actives_arena[e2].outrec].idx
        {
            self.join_outrec_paths(e1, e2);
        } else {
            self.join_outrec_paths(e2, e1);
        }

        result
    }

    fn join_outrec_paths(&mut self, e1: usize, e2: usize) {
        // join e2 outrec path onto e1 outrec path and then delete e2 outrec
        // path pointers. (nb: Only very rarely do the joining ends share
        // the same coords.)
        let or1 = self.actives_arena[e1].outrec;
        let or2 = self.actives_arena[e2].outrec;
        let p1_st = self.outrec_list[or1].pts;
        let p2_st = self.outrec_list[or2].pts;
        let p1_end = self.outpts[p1_st].next;
        let p2_end = self.outpts[p2_st].next;

        if self.is_front(e1) {
            self.outpts[p2_end].prev = p1_st;
            self.outpts[p1_st].next = p2_end;
            self.outpts[p2_st].next = p1_end;
            self.outpts[p1_end].prev = p2_st;
            self.outrec_list[or1].pts = p2_st;
            self.outrec_list[or1].front_edge = self.outrec_list[or2].front_edge;
            let front = self.outrec_list[or1].front_edge;

            if front != NONE {
                self.actives_arena[front].outrec = or1;
            }
        } else {
            self.outpts[p1_end].prev = p2_st;
            self.outpts[p2_st].next = p1_end;
            self.outpts[p1_st].next = p2_end;
            self.outpts[p2_end].prev = p1_st;
            self.outrec_list[or1].back_edge = self.outrec_list[or2].back_edge;
            let back = self.outrec_list[or1].back_edge;

            if back != NONE {
                self.actives_arena[back].outrec = or1;
            }
        }

        // after joining, the e2.OutRec must contains no vertices ...
        self.outrec_list[or2].front_edge = NONE;
        self.outrec_list[or2].back_edge = NONE;
        self.outrec_list[or2].pts = NONE;

        if self.is_open_end(e1) {
            self.outrec_list[or2].pts = self.outrec_list[or1].pts;
            self.outrec_list[or1].pts = NONE;
        } else {
            self.set_owner(or2, or1);
        }

        // and e1 and e2 are maxima and are about to be dropped from the
        // Actives list.
        self.actives_arena[e1].outrec = NONE;
        self.actives_arena[e2].outrec = NONE;
    }

    fn new_out_rec(&mut self) -> usize {
        let idx = self.outrec_list.len();

        self.outrec_list.push(OutRec {
            idx,
            owner: NONE,
            front_edge: NONE,
            back_edge: NONE,
            pts: NONE,
            polypath: NONE,
            splits: None,
            recursive_split: NONE,
            bounds: Rect64::default(),
            path: Vec::new(),
            is_open: false,
            pip_index: None,
        });

        idx
    }

    fn add_out_pt(&mut self, e: usize, pt: Point64) -> usize {
        // Outrec.OutPts: a circular doubly-linked-list of POutPt where ...
        // op_front[.Prev]* ~~~> op_back & op_back == op_front.Next
        let outrec = self.actives_arena[e].outrec;
        let to_front = self.is_front(e);
        let op_front = self.outrec_list[outrec].pts;
        let op_back = self.outpts[op_front].next;

        if to_front {
            if pt == self.outpts[op_front].pt {
                return op_front;
            }
        } else if pt == self.outpts[op_back].pt {
            return op_back;
        }

        let new_op = self.new_out_pt(pt, outrec);
        self.outpts[op_back].prev = new_op;
        self.outpts[new_op].prev = op_front;
        self.outpts[new_op].next = op_back;
        self.outpts[op_front].next = new_op;

        if to_front {
            self.outrec_list[outrec].pts = new_op;
        }

        new_op
    }

    fn clean_collinear(&mut self, outrec: usize) {
        let outrec = self.get_real_outrec(outrec);

        if outrec == NONE || self.outrec_list[outrec].is_open {
            return;
        }

        if !self.is_valid_closed_path(self.outrec_list[outrec].pts) {
            self.dispose_out_pts(outrec);
            return;
        }

        let mut start_op = self.outrec_list[outrec].pts;
        let mut op2 = start_op;

        loop {
            let OutPt { prev, next, pt, .. } = self.outpts[op2];
            let prev_pt = self.outpts[prev].pt;
            let next_pt = self.outpts[next].pt;

            // NB if preserveCollinear == true, then only remove 180 deg.
            // spikes
            if is_collinear(prev_pt, pt, next_pt)
                && (pt == prev_pt
                    || pt == next_pt
                    || !self.preserve_collinear
                    || dot_product(prev_pt, pt, next_pt) < 0.0)
            {
                if op2 == self.outrec_list[outrec].pts {
                    self.outrec_list[outrec].pts = prev;
                }

                op2 = self.dispose_out_pt(op2);

                if !self.is_valid_closed_path(op2) {
                    self.dispose_out_pts(outrec);
                    return;
                }

                start_op = op2;
                continue;
            }

            op2 = next;

            if op2 == start_op {
                break;
            }
        }

        self.fix_self_intersects(outrec);
    }

    fn do_split_op(&mut self, outrec: usize, split_op: usize) {
        // split_op.prev <=> split_op &&
        // split_op.next <=> split_op.next.next are intersecting
        let prev_op = self.outpts[split_op].prev;
        let split_next = self.outpts[split_op].next;
        let next_next_op = self.outpts[split_next].next;
        self.outrec_list[outrec].pts = prev_op;

        let mut ip = Point64::default();
        get_line_intersect_pt(
            self.outpts[prev_op].pt,
            self.outpts[split_op].pt,
            self.outpts[split_next].pt,
            self.outpts[next_next_op].pt,
            &mut ip,
        );

        if let Some(cb) = self.z_callback.as_mut() {
            cb(
                &self.outpts[prev_op].pt,
                &self.outpts[split_op].pt,
                &self.outpts[split_next].pt,
                &self.outpts[next_next_op].pt,
                &mut ip,
            );
        }

        let area1 = self.area_op(self.outrec_list[outrec].pts);
        let abs_area1 = area1.abs();

        if abs_area1 < 2.0 {
            self.dispose_out_pts(outrec);
            return;
        }

        let area2 = area_triangle(ip, self.outpts[split_op].pt, self.outpts[split_next].pt);
        let abs_area2 = area2.abs();

        // de-link split_op and split_op.next from the path while inserting
        // the intersection point
        if ip == self.outpts[prev_op].pt || ip == self.outpts[next_next_op].pt {
            self.outpts[next_next_op].prev = prev_op;
            self.outpts[prev_op].next = next_next_op;
        } else {
            let new_op2 = self.new_out_pt(ip, self.outpts[prev_op].outrec);
            self.outpts[new_op2].prev = prev_op;
            self.outpts[new_op2].next = next_next_op;
            self.outpts[next_next_op].prev = new_op2;
            self.outpts[prev_op].next = new_op2;
        }

        // nb: area1 is the path's area *before* splitting, whereas area2 is
        // the area of the triangle containing splitOp & splitOp.next. So the
        // only way for these areas to have the same sign is if the split
        // triangle is larger than the path containing prevOp or if there's
        // more than one self-intersection.
        if abs_area2 >= 1.0 && (abs_area2 > abs_area1 || (area2 > 0.0) == (area1 > 0.0)) {
            let new_or = self.new_out_rec();
            self.outrec_list[new_or].owner = self.outrec_list[outrec].owner;
            self.outpts[split_op].outrec = new_or;
            self.outpts[split_next].outrec = new_or;

            let new_op = self.new_out_pt(ip, new_or);
            self.outpts[new_op].prev = split_next;
            self.outpts[new_op].next = split_op;
            self.outrec_list[new_or].pts = new_op;
            self.outpts[split_op].prev = new_op;
            self.outpts[split_next].next = new_op;

            if self.using_polytree {
                if self.path2_contains_path1_op(prev_op, new_op) {
                    self.outrec_list[new_or].splits = Some(vec![outrec]);
                } else {
                    self.outrec_list[outrec]
                        .splits
                        .get_or_insert_with(Vec::new)
                        .push(new_or);
                }
            }
        }
        // else: split_op and split_op.next are dropped
    }

    fn fix_self_intersects(&mut self, outrec: usize) {
        let mut op2 = self.outrec_list[outrec].pts;

        // triangles can't self-intersect
        if self.outpts[op2].prev == self.outpts[self.outpts[op2].next].next {
            return;
        }

        loop {
            let prev = self.outpts[op2].prev;
            let next = self.outpts[op2].next;
            let next_next = self.outpts[next].next;

            if segments_intersect(
                self.outpts[prev].pt,
                self.outpts[op2].pt,
                self.outpts[next].pt,
                self.outpts[next_next].pt,
            ) {
                let pts = self.outrec_list[outrec].pts;

                if op2 == pts || next == pts {
                    self.outrec_list[outrec].pts = self.outpts[pts].prev;
                }

                self.do_split_op(outrec, op2);

                if self.outrec_list[outrec].pts == NONE {
                    break;
                }

                op2 = self.outrec_list[outrec].pts;

                // again, because triangles can't self-intersect
                if self.outpts[op2].prev == self.outpts[self.outpts[op2].next].next {
                    break;
                }

                continue;
            } else {
                op2 = next;
            }

            if op2 == self.outrec_list[outrec].pts {
                break;
            }
        }
    }

    fn start_open_path(&mut self, e: usize, pt: Point64) -> usize {
        let outrec = self.new_out_rec();
        self.outrec_list[outrec].is_open = true;

        if self.actives_arena[e].wind_dx > 0 {
            self.outrec_list[outrec].front_edge = e;
            self.outrec_list[outrec].back_edge = NONE;
        } else {
            self.outrec_list[outrec].front_edge = NONE;
            self.outrec_list[outrec].back_edge = e;
        }

        self.actives_arena[e].outrec = outrec;
        let op = self.new_out_pt(pt, outrec);
        self.outrec_list[outrec].pts = op;
        op
    }

    /// `TrimHorz`
    fn trim_horz(&mut self, horz_edge: usize, preserve_collinear: bool) {
        let mut was_trimmed = false;
        let mut pt = self.vertices[self.next_vertex(horz_edge)].pt;

        while pt.y == self.actives_arena[horz_edge].top.y {
            // always trim 180 deg. spikes (in closed paths) but otherwise
            // break if preserveCollinear = true
            let h = self.actives_arena[horz_edge];

            if preserve_collinear && ((pt.x < h.top.x) != (h.bot.x < h.top.x)) {
                break;
            }

            let next = self.next_vertex(horz_edge);
            self.actives_arena[horz_edge].vertex_top = next;
            self.actives_arena[horz_edge].top = pt;
            was_trimmed = true;

            if self.is_maxima(horz_edge) {
                break;
            }

            pt = self.vertices[self.next_vertex(horz_edge)].pt;
        }

        // +/-infinity
        if was_trimmed {
            self.set_dx(horz_edge);
        }
    }

    fn update_edge_into_ael(&mut self, e: usize) {
        let next = self.next_vertex(e);
        let a = &mut self.actives_arena[e];
        a.bot = a.top;
        a.vertex_top = next;
        a.top = self.vertices[next].pt;
        a.curr_x = a.bot.x;
        self.set_dx(e);

        if self.is_joined(e) {
            self.split(e, self.actives_arena[e].bot);
        }

        if self.is_horizontal(e) {
            if !self.is_open(e) {
                self.trim_horz(e, self.preserve_collinear);
            }

            return;
        }

        self.insert_scanline(self.actives_arena[e].top.y);
        let bot = self.actives_arena[e].bot;
        self.check_join_left(e, bot, false);
        // (#500)
        self.check_join_right(e, bot, true);
    }

    fn find_edge_with_matching_loc_min(&self, e: usize) -> usize {
        let a = &self.actives_arena;
        let mut result = a[e].next_in_ael;

        while result != NONE {
            if a[result].local_min == a[e].local_min {
                return result;
            } else if !self.is_horizontal(result) && a[e].bot != a[result].bot {
                result = NONE;
            } else {
                result = a[result].next_in_ael;
            }
        }

        result = a[e].prev_in_ael;

        while result != NONE {
            if a[result].local_min == a[e].local_min {
                return result;
            } else if !self.is_horizontal(result) && a[e].bot != a[result].bot {
                return NONE;
            } else {
                result = a[result].prev_in_ael;
            }
        }

        result
    }

    fn intersect_edges(&mut self, e1: usize, e2: usize, pt: Point64) {
        // MANAGE OPEN PATH INTERSECTIONS SEPARATELY ...
        if self.has_open_paths && (self.is_open(e1) || self.is_open(e2)) {
            if self.is_open(e1) && self.is_open(e2) {
                return;
            }

            let (edge_o, edge_c) = if self.is_open(e1) { (e1, e2) } else { (e2, e1) };

            // needed for safety
            if self.is_joined(edge_c) {
                self.split(edge_c, pt);
            }

            if self.actives_arena[edge_c].wind_cnt.abs() != 1 {
                return;
            }

            match self.cliptype {
                ClipType::Union => {
                    if !self.is_hot_edge(edge_c) {
                        return;
                    }
                }
                _ => {
                    if self.minima[self.actives_arena[edge_c].local_min].polytype
                        == PathType::Subject
                    {
                        return;
                    }
                }
            }

            match self.fillrule {
                FillRule::Positive => {
                    if self.actives_arena[edge_c].wind_cnt != 1 {
                        return;
                    }
                }
                FillRule::Negative => {
                    if self.actives_arena[edge_c].wind_cnt != -1 {
                        return;
                    }
                }
                _ => {
                    if self.actives_arena[edge_c].wind_cnt.abs() != 1 {
                        return;
                    }
                }
            }

            let result_op;

            // toggle contribution ...
            if self.is_hot_edge(edge_o) {
                result_op = self.add_out_pt(edge_o, pt);
                let outrec = self.actives_arena[edge_o].outrec;

                if self.is_front(edge_o) {
                    self.outrec_list[outrec].front_edge = NONE;
                } else {
                    self.outrec_list[outrec].back_edge = NONE;
                }

                self.actives_arena[edge_o].outrec = NONE;
            } else {
                // horizontal edges can pass under open paths at a LocMins
                let local_min = self.actives_arena[edge_o].local_min;
                let lm_vertex = self.minima[local_min].vertex;

                if pt == self.vertices[lm_vertex].pt && !self.is_open_end_v(lm_vertex) {
                    // find the other side of the LocMin and if it's 'hot'
                    // join up with it ...
                    let e3 = self.find_edge_with_matching_loc_min(edge_o);

                    if e3 != NONE && self.is_hot_edge(e3) {
                        let e3_outrec = self.actives_arena[e3].outrec;
                        self.actives_arena[edge_o].outrec = e3_outrec;

                        if self.actives_arena[edge_o].wind_dx > 0 {
                            self.set_sides(e3_outrec, edge_o, e3);
                        } else {
                            self.set_sides(e3_outrec, e3, edge_o);
                        }

                        return;
                    } else {
                        result_op = self.start_open_path(edge_o, pt);
                    }
                } else {
                    result_op = self.start_open_path(edge_o, pt);
                }
            }

            if self.z_callback.is_some() {
                self.set_z_op(edge_o, edge_c, result_op);
            }

            return;
        }

        // MANAGING CLOSED PATHS FROM HERE ON
        if self.is_joined(e1) {
            self.split(e1, pt);
        }

        if self.is_joined(e2) {
            self.split(e2, pt);
        }

        // UPDATE WINDING COUNTS...
        let (old_e1_windcnt, old_e2_windcnt);
        let same_type = self.minima[self.actives_arena[e1].local_min].polytype
            == self.minima[self.actives_arena[e2].local_min].polytype;

        {
            let (a1, a2) = (self.actives_arena[e1], self.actives_arena[e2]);
            let (mut n1, mut n2) = (a1, a2);

            if same_type {
                if self.fillrule == FillRule::EvenOdd {
                    n1.wind_cnt = a2.wind_cnt;
                    n2.wind_cnt = a1.wind_cnt;
                } else {
                    if a1.wind_cnt + a2.wind_dx == 0 {
                        n1.wind_cnt = -a1.wind_cnt;
                    } else {
                        n1.wind_cnt = a1.wind_cnt + a2.wind_dx;
                    }

                    if a2.wind_cnt - n1.wind_dx == 0 {
                        n2.wind_cnt = -a2.wind_cnt;
                    } else {
                        n2.wind_cnt = a2.wind_cnt - n1.wind_dx;
                    }
                }
            } else if self.fillrule != FillRule::EvenOdd {
                n1.wind_cnt2 = a1.wind_cnt2 + a2.wind_dx;
                n2.wind_cnt2 = a2.wind_cnt2 - a1.wind_dx;
            } else {
                n1.wind_cnt2 = if a1.wind_cnt2 == 0 { 1 } else { 0 };
                n2.wind_cnt2 = if a2.wind_cnt2 == 0 { 1 } else { 0 };
            }

            self.actives_arena[e1].wind_cnt = n1.wind_cnt;
            self.actives_arena[e1].wind_cnt2 = n1.wind_cnt2;
            self.actives_arena[e2].wind_cnt = n2.wind_cnt;
            self.actives_arena[e2].wind_cnt2 = n2.wind_cnt2;
        }

        let (w1, w2) = (
            self.actives_arena[e1].wind_cnt,
            self.actives_arena[e2].wind_cnt,
        );

        match self.fillrule {
            FillRule::EvenOdd | FillRule::NonZero => {
                old_e1_windcnt = w1.abs();
                old_e2_windcnt = w2.abs();
            }
            _ => {
                if self.fillrule == self.fillpos {
                    old_e1_windcnt = w1;
                    old_e2_windcnt = w2;
                } else {
                    old_e1_windcnt = -w1;
                    old_e2_windcnt = -w2;
                }
            }
        }

        let e1_windcnt_in_01 = old_e1_windcnt == 0 || old_e1_windcnt == 1;
        let e2_windcnt_in_01 = old_e2_windcnt == 0 || old_e2_windcnt == 1;

        if (!self.is_hot_edge(e1) && !e1_windcnt_in_01)
            || (!self.is_hot_edge(e2) && !e2_windcnt_in_01)
        {
            return;
        }

        // NOW PROCESS THE INTERSECTION ...
        let has_z = self.z_callback.is_some();

        // if both edges are 'hot' ...
        if self.is_hot_edge(e1) && self.is_hot_edge(e2) {
            if (old_e1_windcnt != 0 && old_e1_windcnt != 1)
                || (old_e2_windcnt != 0 && old_e2_windcnt != 1)
                || (self.minima[self.actives_arena[e1].local_min].polytype
                    != self.minima[self.actives_arena[e2].local_min].polytype
                    && self.cliptype != ClipType::Xor)
            {
                let result_op = self.add_local_max_poly(e1, e2, pt);

                if has_z && result_op != NONE {
                    self.set_z_op(e1, e2, result_op);
                }
            } else if self.is_front(e1)
                || self.actives_arena[e1].outrec == self.actives_arena[e2].outrec
            {
                // this 'else if' condition isn't strictly needed but it's
                // sensible to split polygons that only touch at a common
                // vertex (not at common edges).
                let result_op = self.add_local_max_poly(e1, e2, pt);
                let op2 = self.add_local_min_poly(e1, e2, pt, false);

                if has_z && result_op != NONE {
                    self.set_z_op(e1, e2, result_op);
                }

                if has_z {
                    self.set_z_op(e1, e2, op2);
                }
            } else {
                // can't treat as maxima & minima
                let result_op = self.add_out_pt(e1, pt);
                let op2 = self.add_out_pt(e2, pt);

                if has_z {
                    self.set_z_op(e1, e2, result_op);
                    self.set_z_op(e1, e2, op2);
                }

                self.swap_outrecs(e1, e2);
            }
        }
        // if one or other edge is 'hot' ...
        else if self.is_hot_edge(e1) {
            let result_op = self.add_out_pt(e1, pt);

            if has_z {
                self.set_z_op(e1, e2, result_op);
            }

            self.swap_outrecs(e1, e2);
        } else if self.is_hot_edge(e2) {
            let result_op = self.add_out_pt(e2, pt);

            if has_z {
                self.set_z_op(e1, e2, result_op);
            }

            self.swap_outrecs(e1, e2);
        }
        // neither edge is 'hot'
        else {
            let (c1, c2) = (
                self.actives_arena[e1].wind_cnt2 as i64,
                self.actives_arena[e2].wind_cnt2 as i64,
            );

            let (e1_wc2, e2_wc2) = match self.fillrule {
                FillRule::EvenOdd | FillRule::NonZero => (c1.abs(), c2.abs()),
                _ => {
                    if self.fillrule == self.fillpos {
                        (c1, c2)
                    } else {
                        (-c1, -c2)
                    }
                }
            };

            if !self.is_same_poly_type(e1, e2) {
                let result_op = self.add_local_min_poly(e1, e2, pt, false);

                if has_z {
                    self.set_z_op(e1, e2, result_op);
                }
            } else if old_e1_windcnt == 1 && old_e2_windcnt == 1 {
                let mut result_op = NONE;

                match self.cliptype {
                    ClipType::Union => {
                        if e1_wc2 <= 0 && e2_wc2 <= 0 {
                            result_op = self.add_local_min_poly(e1, e2, pt, false);
                        }
                    }
                    ClipType::Difference => {
                        if (self.get_poly_type(e1) == PathType::Clip && e1_wc2 > 0 && e2_wc2 > 0)
                            || (self.get_poly_type(e1) == PathType::Subject
                                && e1_wc2 <= 0
                                && e2_wc2 <= 0)
                        {
                            result_op = self.add_local_min_poly(e1, e2, pt, false);
                        }
                    }
                    ClipType::Xor => {
                        result_op = self.add_local_min_poly(e1, e2, pt, false);
                    }
                    _ => {
                        if e1_wc2 > 0 && e2_wc2 > 0 {
                            result_op = self.add_local_min_poly(e1, e2, pt, false);
                        }
                    }
                }

                if result_op != NONE && has_z {
                    self.set_z_op(e1, e2, result_op);
                }
            }
        }
    }

    fn delete_from_ael(&mut self, e: usize) {
        let prev = self.actives_arena[e].prev_in_ael;
        let next = self.actives_arena[e].next_in_ael;

        // already deleted
        if prev == NONE && next == NONE && e != self.actives {
            return;
        }

        if prev != NONE {
            self.actives_arena[prev].next_in_ael = next;
        } else {
            self.actives = next;
        }

        if next != NONE {
            self.actives_arena[next].prev_in_ael = prev;
        }

        // the C++ deletes it: nothing may reach it from here on
        self.actives_arena[e].prev_in_ael = NONE;
        self.actives_arena[e].next_in_ael = NONE;
    }

    fn adjust_curr_x_and_copy_to_sel(&mut self, top_y: i64) {
        let mut e = self.actives;
        self.sel = e;

        while e != NONE {
            let a = self.actives_arena[e];
            self.actives_arena[e].prev_in_sel = a.prev_in_ael;
            self.actives_arena[e].next_in_sel = a.next_in_ael;
            self.actives_arena[e].jump = a.next_in_ael;
            self.actives_arena[e].curr_x = self.top_x(e, top_y);
            e = a.next_in_ael;
        }
    }

    fn execute_internal(&mut self, ct: ClipType, fillrule: FillRule, use_polytrees: bool) -> bool {
        self.cliptype = ct;
        self.fillrule = fillrule;
        self.using_polytree = use_polytrees;
        self.reset();

        if ct == ClipType::NoClip {
            return true;
        }

        let Some(mut y) = self.pop_scanline() else {
            return true;
        };

        while self.succeeded {
            self.insert_local_minima_into_ael(y);

            while let Some(e) = self.pop_horz() {
                self.do_horizontal(e);
            }

            if !self.horz_seg_list.is_empty() {
                self.convert_horz_segs_to_joins();
                self.horz_seg_list.clear();
            }

            // bottom of scanbeam
            self.bot_y = y;

            // new top of scanbeam
            match self.pop_scanline() {
                Some(next) => y = next,
                None => break,
            }

            self.do_intersections(y);
            self.do_top_of_scanbeam(y);

            while let Some(e) = self.pop_horz() {
                self.do_horizontal(e);
            }
        }

        if self.succeeded {
            self.process_horz_joins();
        }

        self.succeeded
    }

    fn fix_out_rec_pts(&mut self, outrec: usize) {
        let start = self.outrec_list[outrec].pts;
        let mut op = start;

        loop {
            self.outpts[op].outrec = outrec;
            op = self.outpts[op].next;

            if op == start {
                break;
            }
        }
    }

    fn set_horz_seg_heading_forward(&mut self, hs: usize, op_p: usize, op_n: usize) -> bool {
        if self.outpts[op_p].pt.x == self.outpts[op_n].pt.x {
            return false;
        }

        let h = &mut self.horz_seg_list[hs];

        if self.outpts[op_p].pt.x < self.outpts[op_n].pt.x {
            h.left_op = op_p;
            h.right_op = op_n;
            h.left_to_right = true;
        } else {
            h.left_op = op_n;
            h.right_op = op_p;
            h.left_to_right = false;
        }

        true
    }

    fn update_horz_segment(&mut self, hs: usize) -> bool {
        let op = self.horz_seg_list[hs].left_op;
        let outrec = self.get_real_outrec(self.outpts[op].outrec);
        let outrec_has_edges = self.outrec_list[outrec].front_edge != NONE;
        let curr_y = self.outpts[op].pt.y;
        let (mut op_p, mut op_n) = (op, op);
        let o = &self.outpts;

        if outrec_has_edges {
            let op_a = self.outrec_list[outrec].pts;
            let op_z = o[op_a].next;

            while op_p != op_z && o[o[op_p].prev].pt.y == curr_y {
                op_p = o[op_p].prev;
            }

            while op_n != op_a && o[o[op_n].next].pt.y == curr_y {
                op_n = o[op_n].next;
            }
        } else {
            while o[op_p].prev != op_n && o[o[op_p].prev].pt.y == curr_y {
                op_p = o[op_p].prev;
            }

            while o[op_n].next != op_p && o[o[op_n].next].pt.y == curr_y {
                op_n = o[op_n].next;
            }
        }

        let result = self.set_horz_seg_heading_forward(hs, op_p, op_n)
            && self.outpts[self.horz_seg_list[hs].left_op].horz == NONE;

        if result {
            let left = self.horz_seg_list[hs].left_op;
            self.outpts[left].horz = hs;
        } else {
            // for sorting
            self.horz_seg_list[hs].right_op = NONE;
        }

        result
    }

    fn convert_horz_segs_to_joins(&mut self) {
        let mut j = 0;

        for hs in 0..self.horz_seg_list.len() {
            if self.update_horz_segment(hs) {
                j += 1;
            }
        }

        if j < 2 {
            return;
        }

        // HorzSegSorter, stable
        {
            let outpts = &self.outpts;

            self.horz_seg_list.sort_by(|hs1, hs2| {
                let less = |a: &HorzSegment, b: &HorzSegment| {
                    if a.right_op == NONE || b.right_op == NONE {
                        a.right_op != NONE
                    } else {
                        outpts[b.left_op].pt.x > outpts[a.left_op].pt.x
                    }
                };

                if less(hs1, hs2) {
                    std::cmp::Ordering::Less
                } else if less(hs2, hs1) {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            });
        }

        // nb: the ops keep pointing at the slots their segments had before
        // the sort; only whether they point at one is ever asked

        let hs_end = j;

        for hs1 in 0..hs_end - 1 {
            for hs2 in hs1 + 1..hs_end {
                let (s1, s2) = (self.horz_seg_list[hs1], self.horz_seg_list[hs2]);

                if self.outpts[s2.left_op].pt.x >= self.outpts[s1.right_op].pt.x
                    || s2.left_to_right == s1.left_to_right
                    || self.outpts[s2.right_op].pt.x <= self.outpts[s1.left_op].pt.x
                {
                    continue;
                }

                let curr_y = self.outpts[s1.left_op].pt.y;

                if s1.left_to_right {
                    let mut l1 = s1.left_op;
                    let mut l2 = s2.left_op;

                    while self.outpts[self.outpts[l1].next].pt.y == curr_y
                        && self.outpts[self.outpts[l1].next].pt.x <= self.outpts[l2].pt.x
                    {
                        l1 = self.outpts[l1].next;
                    }

                    while self.outpts[self.outpts[l2].prev].pt.y == curr_y
                        && self.outpts[self.outpts[l2].prev].pt.x <= self.outpts[l1].pt.x
                    {
                        l2 = self.outpts[l2].prev;
                    }

                    self.horz_seg_list[hs1].left_op = l1;
                    self.horz_seg_list[hs2].left_op = l2;
                    let op1 = self.duplicate_op(l1, true);
                    let op2 = self.duplicate_op(l2, false);
                    self.horz_join_list.push(HorzJoin { op1, op2 });
                } else {
                    let mut l1 = s1.left_op;
                    let mut l2 = s2.left_op;

                    while self.outpts[self.outpts[l1].prev].pt.y == curr_y
                        && self.outpts[self.outpts[l1].prev].pt.x <= self.outpts[l2].pt.x
                    {
                        l1 = self.outpts[l1].prev;
                    }

                    while self.outpts[self.outpts[l2].next].pt.y == curr_y
                        && self.outpts[self.outpts[l2].next].pt.x <= self.outpts[l1].pt.x
                    {
                        l2 = self.outpts[l2].next;
                    }

                    self.horz_seg_list[hs1].left_op = l1;
                    self.horz_seg_list[hs2].left_op = l2;
                    let op1 = self.duplicate_op(l2, true);
                    let op2 = self.duplicate_op(l1, false);
                    self.horz_join_list.push(HorzJoin { op1, op2 });
                }
            }
        }
    }

    fn move_splits(&mut self, from_or: usize, to_or: usize) {
        let from = self.outrec_list[from_or].splits.take().unwrap_or_default();
        let to = self.outrec_list[to_or].splits.get_or_insert_with(Vec::new);

        for or in &from {
            // #987
            if to_or != *or {
                to.push(*or);
            }
        }

        self.outrec_list[from_or].splits = Some(Vec::new());
    }

    fn process_horz_joins(&mut self) {
        for k in 0..self.horz_join_list.len() {
            let j = self.horz_join_list[k];
            let or1 = self.get_real_outrec(self.outpts[j.op1].outrec);
            let mut or2 = self.get_real_outrec(self.outpts[j.op2].outrec);

            let op1b = self.outpts[j.op1].next;
            let op2b = self.outpts[j.op2].prev;
            self.outpts[j.op1].next = j.op2;
            self.outpts[j.op2].prev = j.op1;
            self.outpts[op1b].prev = op2b;
            self.outpts[op2b].next = op1b;

            // 'join' is really a split
            if or1 == or2 {
                or2 = self.new_out_rec();
                self.outrec_list[or2].pts = op1b;
                self.fix_out_rec_pts(or2);

                // if or1->pts has moved to or2 then update or1->pts !!
                let or1_pts = self.outrec_list[or1].pts;

                if self.outpts[or1_pts].outrec == or2 {
                    self.outrec_list[or1].pts = j.op1;
                    self.outpts[j.op1].outrec = or1;
                }

                // #498, #520, #584, D#576, #618
                if self.using_polytree {
                    if self.path2_contains_path1_op(
                        self.outrec_list[or1].pts,
                        self.outrec_list[or2].pts,
                    ) {
                        // swap or1's & or2's pts
                        let tmp = self.outrec_list[or1].pts;
                        self.outrec_list[or1].pts = self.outrec_list[or2].pts;
                        self.outrec_list[or2].pts = tmp;
                        self.fix_out_rec_pts(or1);
                        self.fix_out_rec_pts(or2);
                        // or2 is now inside or1
                        self.outrec_list[or2].owner = or1;
                    } else if self.path2_contains_path1_op(
                        self.outrec_list[or2].pts,
                        self.outrec_list[or1].pts,
                    ) {
                        self.outrec_list[or2].owner = or1;
                    } else {
                        self.outrec_list[or2].owner = self.outrec_list[or1].owner;
                    }

                    self.outrec_list[or1]
                        .splits
                        .get_or_insert_with(Vec::new)
                        .push(or2);
                } else {
                    self.outrec_list[or2].owner = or1;
                }
            } else {
                self.outrec_list[or2].pts = NONE;

                if self.using_polytree {
                    self.set_owner(or2, or1);

                    // #618
                    if self.outrec_list[or2].splits.is_some() {
                        self.move_splits(or2, or1);
                    }
                } else {
                    self.outrec_list[or2].owner = or1;
                }
            }
        }
    }

    fn do_intersections(&mut self, top_y: i64) {
        if self.build_intersect_list(top_y) {
            self.process_intersect_list();
            self.intersect_nodes.clear();
        }
    }

    fn add_new_intersect_node(&mut self, e1: usize, e2: usize, top_y: i64) {
        let (a1, a2) = (self.actives_arena[e1], self.actives_arena[e2]);
        let mut ip = Point64::default();

        // parallel edges
        if !get_line_intersect_pt(a1.bot, a1.top, a2.bot, a2.top, &mut ip) {
            ip = Point64::new(a1.curr_x, top_y);
        }

        if ip.y > self.bot_y || ip.y < top_y {
            let abs_dx1 = a1.dx.abs();
            let abs_dx2 = a2.dx.abs();

            if abs_dx1 > 100.0 && abs_dx2 > 100.0 {
                if abs_dx1 > abs_dx2 {
                    ip = get_closest_point_on_segment(ip, a1.bot, a1.top);
                } else {
                    ip = get_closest_point_on_segment(ip, a2.bot, a2.top);
                }
            } else if abs_dx1 > 100.0 {
                ip = get_closest_point_on_segment(ip, a1.bot, a1.top);
            } else if abs_dx2 > 100.0 {
                ip = get_closest_point_on_segment(ip, a2.bot, a2.top);
            } else {
                if ip.y < top_y {
                    ip.y = top_y;
                } else {
                    ip.y = self.bot_y;
                }

                if abs_dx1 < abs_dx2 {
                    ip.x = self.top_x(e1, ip.y);
                } else {
                    ip.x = self.top_x(e2, ip.y);
                }
            }
        }

        self.intersect_nodes.push(IntersectNode {
            pt: ip,
            edge1: e1,
            edge2: e2,
        });
    }

    fn build_intersect_list(&mut self, top_y: i64) -> bool {
        if self.actives == NONE || self.actives_arena[self.actives].next_in_ael == NONE {
            return false;
        }

        // calculate edge positions at the top of the current scanbeam, and
        // from this we will determine the intersections required to
        // reach these new positions.
        self.adjust_curr_x_and_copy_to_sel(top_y);

        // Find all edge intersections in the current scanbeam using a
        // stable merge sort that ensures only adjacent edges are
        // intersecting. Intersect info is stored in FIntersectList ready
        // to be processed in ProcessIntersectList.
        let mut left = self.sel;

        while left != NONE && self.actives_arena[left].jump != NONE {
            let mut prev_base = NONE;

            while left != NONE && self.actives_arena[left].jump != NONE {
                let mut curr_base = left;
                let mut right = self.actives_arena[left].jump;
                let mut l_end = right;
                let r_end = self.actives_arena[right].jump;
                self.actives_arena[left].jump = r_end;

                while left != l_end && right != r_end {
                    if self.actives_arena[right].curr_x < self.actives_arena[left].curr_x {
                        let mut tmp = self.actives_arena[right].prev_in_sel;

                        loop {
                            self.add_new_intersect_node(tmp, right, top_y);

                            if tmp == left {
                                break;
                            }

                            tmp = self.actives_arena[tmp].prev_in_sel;
                        }

                        tmp = right;
                        right = self.extract_from_sel(tmp);
                        l_end = right;
                        self.insert1_before2_in_sel(tmp, left);

                        if left == curr_base {
                            curr_base = tmp;
                            self.actives_arena[curr_base].jump = r_end;

                            if prev_base == NONE {
                                self.sel = curr_base;
                            } else {
                                self.actives_arena[prev_base].jump = curr_base;
                            }
                        }
                    } else {
                        left = self.actives_arena[left].next_in_sel;
                    }
                }

                prev_base = curr_base;
                left = r_end;
            }

            left = self.sel;
        }

        !self.intersect_nodes.is_empty()
    }

    fn process_intersect_list(&mut self) {
        // We now have a list of intersections required so that edges will be
        // correctly positioned at the top of the scanbeam. However, it's
        // important that edge intersections are processed from the bottom
        // up, but it's also crucial that intersections only occur between
        // adjacent edges.

        // first we do a quicksort so intersections proceed in a bottom up
        // order (IntersectListSort, with std::sort)
        std_sort::sort(&mut self.intersect_nodes, |a, b| {
            if a.pt.y == b.pt.y {
                a.pt.x < b.pt.x
            } else {
                a.pt.y > b.pt.y
            }
        });

        // Now as we process these intersections, we must sometimes adjust
        // the order to ensure that intersecting edges are always adjacent
        for i in 0..self.intersect_nodes.len() {
            if !self.edges_adjacent_in_ael(&self.intersect_nodes[i]) {
                let mut i2 = i + 1;

                while !self.edges_adjacent_in_ael(&self.intersect_nodes[i2]) {
                    i2 += 1;
                }

                self.intersect_nodes.swap(i, i2);
            }

            let node = self.intersect_nodes[i];
            self.intersect_edges(node.edge1, node.edge2, node.pt);
            self.swap_positions_in_ael(node.edge1, node.edge2);

            self.actives_arena[node.edge1].curr_x = node.pt.x;
            self.actives_arena[node.edge2].curr_x = node.pt.x;
            self.check_join_left(node.edge2, node.pt, true);
            self.check_join_right(node.edge1, node.pt, true);
        }
    }

    fn swap_positions_in_ael(&mut self, e1: usize, e2: usize) {
        // preconditon: e1 must be immediately to the left of e2
        let next = self.actives_arena[e2].next_in_ael;

        if next != NONE {
            self.actives_arena[next].prev_in_ael = e1;
        }

        let prev = self.actives_arena[e1].prev_in_ael;

        if prev != NONE {
            self.actives_arena[prev].next_in_ael = e2;
        }

        self.actives_arena[e2].prev_in_ael = prev;
        self.actives_arena[e2].next_in_ael = e1;
        self.actives_arena[e1].prev_in_ael = e2;
        self.actives_arena[e1].next_in_ael = next;

        if self.actives_arena[e2].prev_in_ael == NONE {
            self.actives = e2;
        }
    }

    fn get_last_op(&self, hot_edge: usize) -> usize {
        let outrec = self.actives_arena[hot_edge].outrec;
        let mut result = self.outrec_list[outrec].pts;

        if hot_edge != self.outrec_list[outrec].front_edge {
            result = self.outpts[result].next;
        }

        result
    }

    fn add_trial_horz_join(&mut self, op: usize) {
        if self.outrec_list[self.outpts[op].outrec].is_open {
            return;
        }

        self.horz_seg_list.push(HorzSegment {
            left_op: op,
            right_op: NONE,
            left_to_right: true,
        });
    }

    /// `ResetHorzDirection`: whether it runs left to right, and its ends.
    fn reset_horz_direction(&self, horz: usize, max_vertex: usize) -> (bool, i64, i64) {
        let h = &self.actives_arena[horz];

        if h.bot.x == h.top.x {
            // the horizontal edge is going nowhere ...
            let mut e = h.next_in_ael;

            while e != NONE && self.actives_arena[e].vertex_top != max_vertex {
                e = self.actives_arena[e].next_in_ael;
            }

            (e != NONE, h.curr_x, h.curr_x)
        } else if h.curr_x < h.top.x {
            (true, h.curr_x, h.top.x)
        } else {
            // right to left
            (false, h.top.x, h.curr_x)
        }
    }

    fn do_horizontal(&mut self, horz: usize) {
        let horz_is_open = self.is_open(horz);
        let y = self.actives_arena[horz].bot.y;

        let vertex_max = if horz_is_open {
            self.get_curr_y_maxima_vertex_open(horz)
        } else {
            self.get_curr_y_maxima_vertex(horz)
        };

        let (mut is_left_to_right, mut horz_left, mut horz_right) =
            self.reset_horz_direction(horz, vertex_max);

        if self.is_hot_edge(horz) {
            let h = self.actives_arena[horz];
            let op = self.add_out_pt(horz, Point64::with_z(h.curr_x, y, h.bot.z));
            self.add_trial_horz_join(op);
        }

        // loop through consec. horizontal edges
        loop {
            let mut e = if is_left_to_right {
                self.actives_arena[horz].next_in_ael
            } else {
                self.actives_arena[horz].prev_in_ael
            };

            while e != NONE {
                if self.actives_arena[e].vertex_top == vertex_max {
                    // do this first!!
                    if self.is_hot_edge(horz) && self.is_joined(e) {
                        let top = self.actives_arena[e].top;
                        self.split(e, top);
                    }

                    if self.is_hot_edge(horz) {
                        while self.actives_arena[horz].vertex_top != vertex_max {
                            let top = self.actives_arena[horz].top;
                            self.add_out_pt(horz, top);
                            self.update_edge_into_ael(horz);
                        }

                        let top = self.actives_arena[horz].top;

                        if is_left_to_right {
                            self.add_local_max_poly(horz, e, top);
                        } else {
                            self.add_local_max_poly(e, horz, top);
                        }
                    }

                    self.delete_from_ael(e);
                    self.delete_from_ael(horz);
                    return;
                }

                // if horzEdge is a maxima, keep going until we reach its
                // maxima pair, otherwise check for break conditions
                if vertex_max != self.actives_arena[horz].vertex_top || self.is_open_end(horz) {
                    // otherwise stop when 'ae' is beyond the end of the
                    // horizontal line
                    let ex = self.actives_arena[e].curr_x;

                    if (is_left_to_right && ex > horz_right)
                        || (!is_left_to_right && ex < horz_left)
                    {
                        break;
                    }

                    if ex == self.actives_arena[horz].top.x && !self.is_horizontal(e) {
                        let pt = self.vertices[self.next_vertex(horz)].pt;

                        // to maximize the possibility of putting open
                        // edges into solutions, we'll only break if it's
                        // past HorzEdge's end
                        let special = self.is_open(e)
                            && !self.is_same_poly_type(e, horz)
                            && !self.is_hot_edge(e);

                        if is_left_to_right {
                            if special {
                                if self.top_x(e, pt.y) > pt.x {
                                    break;
                                }
                            } else if self.top_x(e, pt.y) >= pt.x {
                                break;
                            }
                        } else if special {
                            if self.top_x(e, pt.y) < pt.x {
                                break;
                            }
                        } else if self.top_x(e, pt.y) <= pt.x {
                            break;
                        }
                    }
                }

                let pt = Point64::new(self.actives_arena[e].curr_x, self.actives_arena[horz].bot.y);

                if is_left_to_right {
                    self.intersect_edges(horz, e, pt);
                    self.swap_positions_in_ael(horz, e);
                    self.check_join_left(e, pt, false);
                    self.actives_arena[horz].curr_x = self.actives_arena[e].curr_x;
                    e = self.actives_arena[horz].next_in_ael;
                } else {
                    self.intersect_edges(e, horz, pt);
                    self.swap_positions_in_ael(e, horz);
                    self.check_join_right(e, pt, false);
                    self.actives_arena[horz].curr_x = self.actives_arena[e].curr_x;
                    e = self.actives_arena[horz].prev_in_ael;
                }

                if self.actives_arena[horz].outrec != NONE {
                    let op = self.get_last_op(horz);
                    self.add_trial_horz_join(op);
                }
            }

            // check if we've finished looping through consecutive
            // horizontals: ie open at top
            if horz_is_open && self.is_open_end(horz) {
                if self.is_hot_edge(horz) {
                    let top = self.actives_arena[horz].top;
                    self.add_out_pt(horz, top);
                    let outrec = self.actives_arena[horz].outrec;

                    if self.is_front(horz) {
                        self.outrec_list[outrec].front_edge = NONE;
                    } else {
                        self.outrec_list[outrec].back_edge = NONE;
                    }

                    self.actives_arena[horz].outrec = NONE;
                }

                self.delete_from_ael(horz);
                return;
            } else if self.vertices[self.next_vertex(horz)].pt.y != self.actives_arena[horz].top.y {
                break;
            }

            // still more horizontals in bound to process ...
            if self.is_hot_edge(horz) {
                let top = self.actives_arena[horz].top;
                self.add_out_pt(horz, top);
            }

            self.update_edge_into_ael(horz);

            (is_left_to_right, horz_left, horz_right) = self.reset_horz_direction(horz, vertex_max);
        }

        if self.is_hot_edge(horz) {
            let top = self.actives_arena[horz].top;
            let op = self.add_out_pt(horz, top);
            self.add_trial_horz_join(op);
        }

        // end of an intermediate horiz.
        self.update_edge_into_ael(horz);
    }

    fn do_top_of_scanbeam(&mut self, y: i64) {
        // sel_ is reused to flag horizontals (see PushHorz below)
        self.sel = NONE;
        let mut e = self.actives;

        while e != NONE {
            // nb: 'e' will never be horizontal here
            if self.actives_arena[e].top.y == y {
                self.actives_arena[e].curr_x = self.actives_arena[e].top.x;

                if self.is_maxima(e) {
                    // TOP OF BOUND (MAXIMA)
                    e = self.do_maxima(e);
                    continue;
                } else {
                    // INTERMEDIATE VERTEX ...
                    if self.is_hot_edge(e) {
                        let top = self.actives_arena[e].top;
                        self.add_out_pt(e, top);
                    }

                    self.update_edge_into_ael(e);

                    // horizontals are processed later
                    if self.is_horizontal(e) {
                        self.push_horz(e);
                    }
                }
            } else {
                // i.e. not the top of the edge
                self.actives_arena[e].curr_x = self.top_x(e, y);
            }

            e = self.actives_arena[e].next_in_ael;
        }
    }

    fn do_maxima(&mut self, e: usize) -> usize {
        let prev_e = self.actives_arena[e].prev_in_ael;
        let mut next_e = self.actives_arena[e].next_in_ael;

        if self.is_open_end(e) {
            if self.is_hot_edge(e) {
                let top = self.actives_arena[e].top;
                self.add_out_pt(e, top);
            }

            if !self.is_horizontal(e) {
                if self.is_hot_edge(e) {
                    let outrec = self.actives_arena[e].outrec;

                    if self.is_front(e) {
                        self.outrec_list[outrec].front_edge = NONE;
                    } else {
                        self.outrec_list[outrec].back_edge = NONE;
                    }

                    self.actives_arena[e].outrec = NONE;
                }

                self.delete_from_ael(e);
            }

            return next_e;
        }

        let max_pair = self.get_maxima_pair(e);

        // max_pair is horizontal
        if max_pair == NONE {
            return next_e;
        }

        if self.is_joined(e) {
            let top = self.actives_arena[e].top;
            self.split(e, top);
        }

        if self.is_joined(max_pair) {
            let top = self.actives_arena[max_pair].top;
            self.split(max_pair, top);
        }

        // only non-horizontal maxima here.
        // process any edges between maxima pair ...
        while next_e != max_pair {
            let top = self.actives_arena[e].top;
            self.intersect_edges(e, next_e, top);
            self.swap_positions_in_ael(e, next_e);
            next_e = self.actives_arena[e].next_in_ael;
        }

        if self.is_open(e) {
            if self.is_hot_edge(e) {
                let top = self.actives_arena[e].top;
                self.add_local_max_poly(e, max_pair, top);
            }

            self.delete_from_ael(max_pair);
            self.delete_from_ael(e);
            return if prev_e != NONE {
                self.actives_arena[prev_e].next_in_ael
            } else {
                self.actives
            };
        }

        // e.next_in_ael == max_pair ...
        if self.is_hot_edge(e) {
            let top = self.actives_arena[e].top;
            self.add_local_max_poly(e, max_pair, top);
        }

        self.delete_from_ael(e);
        self.delete_from_ael(max_pair);

        if prev_e != NONE {
            self.actives_arena[prev_e].next_in_ael
        } else {
            self.actives
        }
    }

    fn split(&mut self, e: usize, pt: Point64) {
        if self.actives_arena[e].join_with == JoinWith::Right {
            self.actives_arena[e].join_with = JoinWith::NoJoin;
            let next = self.actives_arena[e].next_in_ael;
            self.actives_arena[next].join_with = JoinWith::NoJoin;
            self.add_local_min_poly(e, next, pt, true);
        } else {
            self.actives_arena[e].join_with = JoinWith::NoJoin;
            let prev = self.actives_arena[e].prev_in_ael;
            self.actives_arena[prev].join_with = JoinWith::NoJoin;
            self.add_local_min_poly(prev, e, pt, true);
        }
    }

    fn check_join_left(&mut self, e: usize, pt: Point64, check_curr_x: bool) {
        let prev = self.actives_arena[e].prev_in_ael;

        if prev == NONE
            || !self.is_hot_edge(e)
            || !self.is_hot_edge(prev)
            || self.is_horizontal(e)
            || self.is_horizontal(prev)
            || self.is_open(e)
            || self.is_open(prev)
        {
            return;
        }

        let (a, p) = (self.actives_arena[e], self.actives_arena[prev]);

        // avoid trivial joins
        if (pt.y < a.top.y + 2 || pt.y < p.top.y + 2) && (a.bot.y > pt.y || p.bot.y > pt.y) {
            return;
        }

        if check_curr_x {
            if perpendic_dist_from_line_sqrd(pt, p.bot, p.top) > 0.25 {
                return;
            }
        } else if a.curr_x != p.curr_x {
            return;
        }

        if !is_collinear(a.top, pt, p.top) {
            return;
        }

        let (idx_e, idx_p) = (
            self.outrec_list[a.outrec].idx,
            self.outrec_list[p.outrec].idx,
        );

        if idx_e == idx_p {
            self.add_local_max_poly(prev, e, pt);
        } else if idx_e < idx_p {
            self.join_outrec_paths(e, prev);
        } else {
            self.join_outrec_paths(prev, e);
        }

        self.actives_arena[prev].join_with = JoinWith::Right;
        self.actives_arena[e].join_with = JoinWith::Left;
    }

    fn check_join_right(&mut self, e: usize, pt: Point64, check_curr_x: bool) {
        let next = self.actives_arena[e].next_in_ael;

        if next == NONE
            || !self.is_hot_edge(e)
            || !self.is_hot_edge(next)
            || self.is_horizontal(e)
            || self.is_horizontal(next)
            || self.is_open(e)
            || self.is_open(next)
        {
            return;
        }

        let (a, n) = (self.actives_arena[e], self.actives_arena[next]);

        // avoid trivial joins
        if (pt.y < a.top.y + 2 || pt.y < n.top.y + 2) && (a.bot.y > pt.y || n.bot.y > pt.y) {
            return;
        }

        if check_curr_x {
            if perpendic_dist_from_line_sqrd(pt, n.bot, n.top) > 0.35 {
                return;
            }
        } else if a.curr_x != n.curr_x {
            return;
        }

        if !is_collinear(a.top, pt, n.top) {
            return;
        }

        let (idx_e, idx_n) = (
            self.outrec_list[a.outrec].idx,
            self.outrec_list[n.outrec].idx,
        );

        if idx_e == idx_n {
            self.add_local_max_poly(e, next, pt);
        } else if idx_e < idx_n {
            self.join_outrec_paths(e, next);
        } else {
            self.join_outrec_paths(next, e);
        }

        self.actives_arena[e].join_with = JoinWith::Right;
        self.actives_arena[next].join_with = JoinWith::Left;
    }

    /// `BuildPath64`
    fn build_path64(&self, mut op: usize, reverse: bool, is_open: bool, path: &mut Path64) -> bool {
        let o = &self.outpts;

        if op == NONE || o[op].next == op || (!is_open && o[op].next == o[op].prev) {
            return false;
        }

        path.clear();
        let mut last_pt;
        let mut op2;

        if reverse {
            last_pt = o[op].pt;
            op2 = o[op].prev;
        } else {
            op = o[op].next;
            last_pt = o[op].pt;
            op2 = o[op].next;
        }

        path.push(last_pt);

        while op2 != op {
            if o[op2].pt != last_pt {
                last_pt = o[op2].pt;
                path.push(last_pt);
            }

            op2 = if reverse { o[op2].prev } else { o[op2].next };
        }

        !(!is_open && path.len() == 3 && self.is_very_small_triangle(op2))
    }

    fn check_bounds(&mut self, outrec: usize) -> bool {
        if self.outrec_list[outrec].pts == NONE {
            return false;
        }

        if !self.outrec_list[outrec].bounds.is_empty() {
            return true;
        }

        self.clean_collinear(outrec);

        if self.outrec_list[outrec].pts == NONE {
            return false;
        }

        // into the record's own path, which a failure may leave filled
        let mut path = std::mem::take(&mut self.outrec_list[outrec].path);
        let built = self.build_path64(
            self.outrec_list[outrec].pts,
            self.reverse_solution,
            false,
            &mut path,
        );
        self.outrec_list[outrec].path = path;

        if !built {
            return false;
        }

        self.outrec_list[outrec].bounds = get_bounds(&self.outrec_list[outrec].path);
        true
    }

    /// `CheckSplitOwner`, over the splits of `splits_of`, which may grow
    /// while they are checked.
    fn check_split_owner(&mut self, outrec: usize, splits_of: usize) -> bool {
        let mut idx = 0;

        while idx
            < self.outrec_list[splits_of]
                .splits
                .as_ref()
                .map_or(0, Vec::len)
        {
            let mut split = self.outrec_list[splits_of]
                .splits
                .as_ref()
                .expect("counted above")[idx];
            idx += 1;

            // #942
            if self.outrec_list[split].pts == NONE
                && self.outrec_list[split].splits.is_some()
                && self.check_split_owner(outrec, split)
            {
                return true;
            }

            split = self.get_real_outrec(split);

            if split == NONE || split == outrec || self.outrec_list[split].recursive_split == outrec
            {
                continue;
            }

            // prevent infinite loops
            self.outrec_list[split].recursive_split = outrec;

            if self.outrec_list[split].splits.is_some() && self.check_split_owner(outrec, split) {
                return true;
            }

            if !self.check_bounds(split)
                || !self.outrec_list[split]
                    .bounds
                    .contains_rect(&self.outrec_list[outrec].bounds)
            {
                continue;
            }

            let pts = self.outrec_list[outrec].pts;

            if !self.path2_contains_path1_indexed(pts, split) {
                continue;
            }

            // split is owned by outrec! (#957)
            if !self.is_valid_owner(outrec, split) {
                self.outrec_list[split].owner = self.outrec_list[outrec].owner;
            }

            self.outrec_list[outrec].owner = split;
            return true;
        }

        false
    }

    fn add_tree_child(&mut self, parent: usize, path: Path64) -> usize {
        let index = self.tree.len();
        self.tree.push(PolyNode {
            polygon: path,
            childs: Vec::new(),
        });
        self.tree[parent].childs.push(index);
        index
    }

    fn recursive_check_owners(&mut self, outrec: usize, polypath: usize) {
        if self.outrec_list[outrec].polypath != NONE || self.outrec_list[outrec].bounds.is_empty() {
            return;
        }

        while self.outrec_list[outrec].owner != NONE {
            let owner = self.outrec_list[outrec].owner;

            if self.outrec_list[owner].splits.is_some() && self.check_split_owner(outrec, owner) {
                break;
            }

            let owner = self.outrec_list[outrec].owner;

            if self.outrec_list[owner].pts != NONE
                && self.check_bounds(owner)
                && self.outrec_list[owner]
                    .bounds
                    .contains_rect(&self.outrec_list[outrec].bounds)
                && self.path2_contains_path1_indexed(self.outrec_list[outrec].pts, owner)
            {
                break;
            }

            self.outrec_list[outrec].owner = self.outrec_list[owner].owner;
        }

        let owner = self.outrec_list[outrec].owner;
        let path = self.outrec_list[outrec].path.clone();

        if owner != NONE {
            if self.outrec_list[owner].polypath == NONE {
                self.recursive_check_owners(owner, polypath);
            }

            let parent = self.outrec_list[owner].polypath;
            self.outrec_list[outrec].polypath = self.add_tree_child(parent, path);
        } else {
            self.outrec_list[outrec].polypath = self.add_tree_child(polypath, path);
        }
    }

    fn build_paths64(&mut self, solution_closed: &mut Paths64, solution_open: &mut Paths64) {
        solution_closed.clear();
        solution_open.clear();

        // nb: cleaning a record can add records
        let mut i = 0;

        while i < self.outrec_list.len() {
            i += 1;
            let i = i - 1;

            if self.outrec_list[i].pts == NONE {
                continue;
            }

            let mut path = Vec::new();

            if self.outrec_list[i].is_open {
                if self.build_path64(
                    self.outrec_list[i].pts,
                    self.reverse_solution,
                    true,
                    &mut path,
                ) {
                    solution_open.push(path);
                }
            } else {
                // nb: CleanCollinear can add to outrec_list_
                self.clean_collinear(i);

                if self.build_path64(
                    self.outrec_list[i].pts,
                    self.reverse_solution,
                    false,
                    &mut path,
                ) {
                    solution_closed.push(path);
                }
            }
        }
    }

    fn build_tree64(&mut self, open_paths: &mut Paths64) {
        open_paths.clear();

        let mut i = 0;

        while i < self.outrec_list.len() {
            i += 1;
            let i = i - 1;

            if self.outrec_list[i].pts == NONE {
                continue;
            }

            if self.outrec_list[i].is_open {
                let mut path = Vec::new();

                if self.build_path64(
                    self.outrec_list[i].pts,
                    self.reverse_solution,
                    true,
                    &mut path,
                ) {
                    open_paths.push(path);
                }

                continue;
            }

            if self.check_bounds(i) {
                self.recursive_check_owners(i, 0);
            }
        }
    }
}
