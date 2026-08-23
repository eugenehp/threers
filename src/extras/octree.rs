use crate::math::{Box3, Vector3};

/// Spatial acceleration structure for point queries. Mirrors three.js's
/// `Octree` for the API; the internal node layout is the simple 8-child
/// recursive subdivision.
pub struct Octree {
    pub bounds: Box3,
    pub points: Vec<Vector3>,
    pub children: Option<Box<[Octree; 8]>>,
    pub max_depth: u32,
    pub max_points: usize,
}

impl Octree {
    pub fn new(bounds: Box3, max_depth: u32, max_points: usize) -> Self {
        Self {
            bounds,
            points: Vec::new(),
            children: None,
            max_depth,
            max_points,
        }
    }

    pub fn insert(&mut self, p: Vector3) {
        if !self.bounds.contains_point(p) {
            return;
        }
        if let Some(children) = self.children.as_mut() {
            for child in children.iter_mut() {
                child.insert(p);
            }
            return;
        }
        self.points.push(p);
        if self.points.len() > self.max_points && self.max_depth > 0 {
            self.subdivide();
        }
    }

    fn subdivide(&mut self) {
        let center = self.bounds.center();
        let mins = [
            Vector3::new(self.bounds.min.x, self.bounds.min.y, self.bounds.min.z),
            Vector3::new(center.x, self.bounds.min.y, self.bounds.min.z),
            Vector3::new(self.bounds.min.x, center.y, self.bounds.min.z),
            Vector3::new(center.x, center.y, self.bounds.min.z),
            Vector3::new(self.bounds.min.x, self.bounds.min.y, center.z),
            Vector3::new(center.x, self.bounds.min.y, center.z),
            Vector3::new(self.bounds.min.x, center.y, center.z),
            Vector3::new(center.x, center.y, center.z),
        ];
        let maxs = [
            Vector3::new(center.x, center.y, center.z),
            Vector3::new(self.bounds.max.x, center.y, center.z),
            Vector3::new(center.x, self.bounds.max.y, center.z),
            Vector3::new(self.bounds.max.x, self.bounds.max.y, center.z),
            Vector3::new(center.x, center.y, self.bounds.max.z),
            Vector3::new(self.bounds.max.x, center.y, self.bounds.max.z),
            Vector3::new(center.x, self.bounds.max.y, self.bounds.max.z),
            Vector3::new(self.bounds.max.x, self.bounds.max.y, self.bounds.max.z),
        ];
        let mut children: Vec<Octree> = Vec::with_capacity(8);
        for i in 0..8 {
            children.push(Octree::new(
                Box3::new(mins[i], maxs[i]),
                self.max_depth - 1,
                self.max_points,
            ));
        }
        let pts = std::mem::take(&mut self.points);
        let mut arr: [Octree; 8] = children.try_into().unwrap_or_else(|_| unreachable!());
        for p in pts {
            for c in arr.iter_mut() {
                c.insert(p);
            }
        }
        self.children = Some(Box::new(arr));
    }

    /// Collect all points within `radius` of `target`.
    pub fn nearest(&self, target: Vector3, radius: f32) -> Vec<Vector3> {
        let mut out = Vec::new();
        if self.bounds.distance_to_point(target) > radius {
            return out;
        }
        if let Some(children) = &self.children {
            for c in children.iter() {
                out.extend(c.nearest(target, radius));
            }
        } else {
            for p in &self.points {
                if (target - *p).length() <= radius {
                    out.push(*p);
                }
            }
        }
        out
    }
}
