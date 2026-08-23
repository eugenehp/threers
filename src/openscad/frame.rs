//! One evaluated instant of a model.
//!
//! Separate from [`animate`](crate::openscad::animate) because it is only data:
//! the parts and the viewport, with no renderer, no worker threads and no
//! filesystem behind it. That is what lets a simulated mechanism hand a frame
//! to a caller on wasm32, where the rest of `animate` cannot go.

use std::sync::Arc;

use crate::openscad::{scad, ScadPart};

/// A frame of an animation: the model as it stands at one instant.
#[derive(Debug, Clone)]
pub struct ScadFrame {
    /// Frame index, `0..frames`.
    pub index: usize,
    /// The animation variable for this frame.
    pub t: f64,
    /// Colored pieces of the evaluated model.
    pub parts: Arc<Vec<ScadPart>>,
    /// The viewport the model asked for at this `t`, if it set one.
    pub viewport: scad::Viewport,
}

impl ScadFrame {
    /// World-space bounds of every part, or `None` when the frame is empty.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        let mut any = false;
        for part in self.parts.iter() {
            let (min, max) = crate::openscad::geometry_bounds(&part.geometry);
            if min[0] > max[0] {
                continue;
            }
            any = true;
            for i in 0..3 {
                lo[i] = lo[i].min(min[i]);
                hi[i] = hi[i].max(max[i]);
            }
        }
        any.then_some((lo, hi))
    }

    /// Total triangle count across the parts.
    pub fn triangle_count(&self) -> usize {
        self.parts
            .iter()
            .map(|p| match &p.geometry.index {
                Some(idx) => idx.len() / 3,
                None => p
                    .geometry
                    .attributes
                    .get("position")
                    .map(|a| a.array.len() / 9)
                    .unwrap_or(0),
            })
            .sum()
    }
}
