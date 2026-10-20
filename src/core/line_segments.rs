use super::{BufferAttribute, BufferGeometry};
use crate::materials::Material;
use std::sync::Arc;

/// Line-list primitive. Each two consecutive vertices form a segment.
/// Mirrors three.js's `LineSegments` — and, with a [`strip`](Self::strip),
/// its `Line` and `LineLoop`.
#[derive(Debug, Clone)]
pub struct LineSegments {
    /// The segments as drawn: a line list, two vertices per segment.
    pub geometry: Arc<BufferGeometry>,
    pub material: Arc<Material>,
    /// For a three.js `Line` / `LineLoop`: the vertex strip as authored,
    /// which [`geometry`](Self::geometry) is the line-list expansion of. The
    /// raycaster walks the strip, so a hit's `index` is the strip vertex
    /// three.js reports. `None` for a plain `LineSegments`.
    pub strip: Option<LineStrip>,
}

/// A three.js `Line` (open) or `LineLoop` (closed) vertex strip.
#[derive(Debug, Clone)]
pub struct LineStrip {
    pub geometry: Arc<BufferGeometry>,
    /// `LineLoop`: a last segment joins the end back to the start.
    pub closed: bool,
}

impl LineSegments {
    pub fn new(geometry: BufferGeometry, material: Material) -> Self {
        Self {
            geometry: Arc::new(geometry),
            material: Arc::new(material),
            strip: None,
        }
    }

    pub fn from_arc(geometry: Arc<BufferGeometry>, material: Arc<Material>) -> Self {
        Self {
            geometry,
            material,
            strip: None,
        }
    }

    /// three.js `Line` (`closed` false) or `LineLoop` (`closed` true) over a
    /// vertex strip: consecutive vertices — in index order when there is an
    /// index, within the strip's draw range — are joined, and a loop joins
    /// the last back to the first, as `gl.LINE_STRIP` / `LINE_LOOP` draw it.
    /// [`geometry`](Self::geometry) becomes that line list (every per-vertex
    /// attribute carried along), the strip is kept for raycasting.
    pub fn strip(strip: Arc<BufferGeometry>, material: Arc<Material>, closed: bool) -> Self {
        Self {
            geometry: Arc::new(strip_to_list(&strip, closed)),
            material,
            strip: Some(LineStrip {
                geometry: strip,
                closed,
            }),
        }
    }
}

/// The segments of a strip as a line list (see [`LineSegments::strip`]).
fn strip_to_list(strip: &BufferGeometry, closed: bool) -> BufferGeometry {
    let vertices = strip
        .get_attribute("position")
        .map(|a| a.count())
        .unwrap_or(0);
    let count = match &strip.index {
        Some(idx) => idx.len(),
        None => vertices,
    };
    let start = (strip.draw_range.start as usize).min(count);
    let end = (strip.draw_range.end() as usize).min(count);
    let at = |i: usize| match &strip.index {
        Some(idx) => idx[i] as usize,
        None => i,
    };
    let mut pairs = Vec::new();
    if end > start + 1 {
        for i in start..end - 1 {
            pairs.push((at(i), at(i + 1)));
        }
    }
    if closed && end > start {
        pairs.push((at(end - 1), at(start)));
    }
    let mut out = BufferGeometry::new();
    for (name, attr) in &strip.attributes {
        let n = attr.item_size.max(1);
        if attr.count() != vertices {
            continue;
        }
        let mut array = Vec::with_capacity(pairs.len() * 2 * n);
        for &(a, b) in &pairs {
            for v in [a, b] {
                match attr.array.get(v * n..v * n + n) {
                    Some(s) => array.extend_from_slice(s),
                    None => array.extend(std::iter::repeat_n(0.0, n)),
                }
            }
        }
        out.set_attribute(name.clone(), BufferAttribute::new(array, n));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn material() -> Arc<Material> {
        Arc::new(Material::Basic(crate::materials::BasicMaterial::new(
            crate::math::Color::WHITE,
        )))
    }

    fn strip(n: usize) -> Arc<BufferGeometry> {
        let mut g = BufferGeometry::new();
        let p: Vec<f32> = (0..n).flat_map(|i| [i as f32, 0.0, 0.0]).collect();
        g.set_attribute("position", BufferAttribute::new(p, 3));
        Arc::new(g)
    }

    fn xs(ls: &LineSegments) -> Vec<f32> {
        let a = ls.geometry.get_attribute("position").unwrap();
        a.array.chunks(3).map(|c| c[0]).collect()
    }

    #[test]
    fn a_strip_becomes_its_segments() {
        let m = material();
        assert_eq!(
            xs(&LineSegments::strip(strip(4), m.clone(), false)),
            [0., 1., 1., 2., 2., 3.]
        );
        assert_eq!(
            xs(&LineSegments::strip(strip(3), m, true)),
            [0., 1., 1., 2., 2., 0.],
            "a loop closes"
        );
    }

    #[test]
    fn the_index_and_draw_range_order_the_strip() {
        let mut g = (*strip(5)).clone();
        g.set_index(vec![4, 2, 0, 1]);
        g.set_draw_range(1, 3);
        let ls = LineSegments::strip(Arc::new(g), material(), true);
        assert_eq!(xs(&ls), [2., 0., 0., 1., 1., 2.]);
    }
}
