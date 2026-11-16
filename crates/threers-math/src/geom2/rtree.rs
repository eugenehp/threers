//! A static R-tree over boxes, packed once by sort-tile-recursive (STR)
//! order, for finding what overlaps a box or lies nearest a point.

use super::P2;

/// An axis-aligned box, `[min, max]`.
pub type Bounds = [P2; 2];

fn union(a: Bounds, b: Bounds) -> Bounds {
    [
        [a[0][0].min(b[0][0]), a[0][1].min(b[0][1])],
        [a[1][0].max(b[1][0]), a[1][1].max(b[1][1])],
    ]
}

fn overlaps(a: &Bounds, b: &Bounds) -> bool {
    a[0][0] <= b[1][0] && b[0][0] <= a[1][0] && a[0][1] <= b[1][1] && b[0][1] <= a[1][1]
}

/// The squared distance from `p` to the box (0 inside it).
fn dist2(b: &Bounds, p: P2) -> f64 {
    let dx = (b[0][0] - p[0]).max(0.0).max(p[0] - b[1][0]);
    let dy = (b[0][1] - p[1]).max(0.0).max(p[1] - b[1][1]);
    dx * dx + dy * dy
}

/// A packed R-tree: each node's children are a run of the level below.
#[derive(Clone, Debug)]
pub struct RTree {
    /// the levels from the leaves up, each node's box and its first child
    /// (an item index at the leaf level)
    levels: Vec<Vec<(Bounds, usize)>>,
    fanout: usize,
}

impl RTree {
    /// The tree of `boxes`, item `i` the box at `i`, `fanout` children to
    /// a node (at least 2).
    pub fn new(boxes: &[Bounds], fanout: usize) -> Self {
        let fanout = fanout.max(2);
        let mut order: Vec<usize> = (0..boxes.len()).collect();
        let center = |i: usize, k: usize| boxes[i][0][k] + boxes[i][1][k];

        // STR: sort by x, cut into vertical slices, sort each by y
        let leaves = boxes.len().div_ceil(fanout).max(1);
        let slices = (leaves as f64).sqrt().ceil() as usize;
        let per_slice = slices * fanout;
        order.sort_by(|&a, &b| center(a, 0).total_cmp(&center(b, 0)));

        for slice in order.chunks_mut(per_slice.max(1)) {
            slice.sort_by(|&a, &b| center(a, 1).total_cmp(&center(b, 1)));
        }

        let mut levels = vec![order.iter().map(|&i| (boxes[i], i)).collect::<Vec<_>>()];

        while levels.last().is_some_and(|l| l.len() > 1) {
            let below = levels.last().unwrap();
            let up = below
                .chunks(fanout)
                .enumerate()
                .map(|(k, run)| {
                    (
                        run.iter().skip(1).fold(run[0].0, |acc, n| union(acc, n.0)),
                        k * fanout,
                    )
                })
                .collect();
            levels.push(up);
        }

        RTree { levels, fanout }
    }

    pub fn len(&self) -> usize {
        self.levels[0].len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The items whose boxes overlap `query` (touching counts).
    pub fn search(&self, query: &Bounds) -> Vec<usize> {
        let mut out = Vec::new();

        if self.is_empty() {
            return out;
        }

        let top = self.levels.len() - 1;
        let mut stack: Vec<(usize, usize)> =
            (0..self.levels[top].len()).map(|i| (top, i)).collect();

        while let Some((level, i)) = stack.pop() {
            let (b, first) = self.levels[level][i];

            if !overlaps(&b, query) {
                continue;
            }

            if level == 0 {
                out.push(first);
            } else {
                let end = (first + self.fanout).min(self.levels[level - 1].len());
                stack.extend((first..end).map(|c| (level - 1, c)));
            }
        }

        out.sort_unstable();
        out
    }

    /// The item whose box is nearest `p`, and the distance to it.
    pub fn nearest(&self, p: P2) -> Option<(usize, f64)> {
        if self.is_empty() {
            return None;
        }

        let top = self.levels.len() - 1;
        let mut best: Option<(usize, f64)> = None;
        let mut stack: Vec<(usize, usize)> =
            (0..self.levels[top].len()).map(|i| (top, i)).collect();

        while let Some((level, i)) = stack.pop() {
            let (b, first) = self.levels[level][i];
            let d = dist2(&b, p);

            if best.is_some_and(|(_, bd)| d >= bd) {
                continue;
            }

            if level == 0 {
                best = Some((first, d));
            } else {
                let end = (first + self.fanout).min(self.levels[level - 1].len());
                stack.extend((first..end).map(|c| (level - 1, c)));
            }
        }

        best.map(|(i, d)| (i, d.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn against_brute_force() {
        let mut boxes = Vec::new();

        for i in 0..400 {
            let (x, y) = ((i * 37 % 101) as f64, (i * 53 % 97) as f64);
            boxes.push([[x, y], [x + (i % 5) as f64, y + (i % 3) as f64]]);
        }

        let tree = RTree::new(&boxes, 8);
        let query = [[20.0, 30.0], [45.0, 50.0]];
        let brute: Vec<usize> = (0..boxes.len())
            .filter(|&i| overlaps(&boxes[i], &query))
            .collect();
        assert_eq!(tree.search(&query), brute);

        let p = [50.5, 60.5];
        let best = (0..boxes.len())
            .map(|i| dist2(&boxes[i], p))
            .fold(f64::INFINITY, f64::min)
            .sqrt();
        assert!((tree.nearest(p).unwrap().1 - best).abs() < 1e-12);
    }
}
