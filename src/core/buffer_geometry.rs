use super::BufferAttribute;
use crate::math::{Box3, Sphere, Vector3};
use std::collections::HashMap;

#[cfg(any(feature = "mesh-bvh", feature = "brep"))]
use std::sync::Arc;

/// A collection of named vertex attributes plus an optional index buffer.
/// Mirrors three.js's `BufferGeometry`: `geometry.setAttribute("position", ...)`,
/// `geometry.setIndex(...)`.
#[derive(Debug, Clone, Default)]
pub struct BufferGeometry {
    pub attributes: HashMap<String, BufferAttribute>,
    pub index: Option<Vec<u32>>,
    pub bounding_box: Option<Box3>,
    pub bounding_sphere: Option<Sphere>,
    /// Bumped whenever attributes/index change so the renderer re-uploads GPU buffers.
    pub geometry_version: u32,
    /// Give this geometry's GPU vertex buffer `STORAGE` usage, so a compute pass
    /// can write it directly.
    ///
    /// The point is vertex animation that never round-trips through the CPU:
    /// upload the rest pose once, then let a compute shader rewrite positions and
    /// normals in place each frame. Nothing bumps
    /// [`geometry_version`](Self::geometry_version), so the renderer never
    /// re-uploads and the CPU-side attribute arrays simply stop being the truth.
    ///
    /// Pair it with [`Renderer::vertex_buffer`](crate::Renderer::vertex_buffer).
    /// Off by default: `STORAGE` is not free on every backend, and only a
    /// geometry that is actually driven this way should ask for it.
    pub gpu_writable: bool,
    /// Optional BVH acceleration structure (three-mesh-bvh `boundsTree`).
    #[cfg(feature = "mesh-bvh")]
    pub bounds_tree: Option<Arc<crate::mesh_bvh::MeshBvh>>,
    /// Optional analytic surface provenance: which surface each triangle was
    /// sampled from. See [`crate::brep`].
    ///
    /// Always optional and always droppable. It is invalidated by any write to
    /// the positions or the index, for the same reason `bounds_tree` is: a
    /// sidecar describing geometry that has since changed is worse than no
    /// sidecar, because consumers trust it.
    #[cfg(feature = "brep")]
    pub surfaces: Option<Arc<crate::brep::SurfaceTable>>,
}

/// A process-wide monotonic version stamp. The renderer's GPU-buffer cache is
/// keyed by a geometry's heap *pointer* and re-uploads only when `geometry_version`
/// changes. When scenes are rebuilt every frame (e.g. an animation) a freed
/// geometry's address is often *recycled* for a new one; if versions could
/// coincide, the cache would false-hit and draw the freed geometry's stale
/// buffers ("ghost" meshes). Drawing every version from one global counter makes
/// every stamp unique, so a recycled address can never false-match.
fn next_geometry_version() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static VERSION: AtomicU32 = AtomicU32::new(1);
    VERSION.fetch_add(1, Ordering::Relaxed)
}

impl BufferGeometry {
    pub fn new() -> Self {
        Self {
            geometry_version: next_geometry_version(),
            ..Self::default()
        }
    }

    pub fn set_attribute(&mut self, name: impl Into<String>, attr: BufferAttribute) -> &mut Self {
        self.attributes.insert(name.into(), attr);
        // Invalidate cached bounds — they depend on positions.
        self.bounding_box = None;
        self.bounding_sphere = None;
        self.geometry_version = next_geometry_version();
        #[cfg(feature = "mesh-bvh")]
        {
            self.bounds_tree = None;
        }
        #[cfg(feature = "brep")]
        {
            self.surfaces = None;
        }
        self
    }

    pub fn get_attribute(&self, name: &str) -> Option<&BufferAttribute> {
        self.attributes.get(name)
    }

    pub fn set_index(&mut self, indices: Vec<u32>) -> &mut Self {
        self.index = Some(indices);
        self.geometry_version = next_geometry_version();
        #[cfg(feature = "mesh-bvh")]
        {
            self.bounds_tree = None;
        }
        #[cfg(feature = "brep")]
        {
            self.surfaces = None;
        }
        self
    }

    /// Attach analytic surface provenance.
    ///
    /// Rejected — leaving the geometry untagged — unless the table describes
    /// exactly this triangle count. Silently accepting a mismatched table would
    /// hand every consumer an out-of-range index.
    ///
    /// Call this *after* the last `set_attribute` / `set_index`, both of which
    /// clear it.
    #[cfg(feature = "brep")]
    pub fn set_surfaces(&mut self, table: crate::brep::SurfaceTable) -> &mut Self {
        if table.matches(self) {
            self.surfaces = Some(Arc::new(table));
        }
        self
    }

    /// The provenance table, if one is attached and still describes this
    /// geometry.
    #[cfg(feature = "brep")]
    pub fn surface_table(&self) -> Option<&crate::brep::SurfaceTable> {
        self.surfaces.as_deref().filter(|t| t.matches(self))
    }

    /// Total draw count: index count if indexed, otherwise position count.
    pub fn draw_count(&self) -> usize {
        if let Some(idx) = &self.index {
            idx.len()
        } else {
            self.attributes
                .get("position")
                .map(|a| a.count())
                .unwrap_or(0)
        }
    }

    /// Iterate position vertices as `Vector3` (item_size must be 3).
    pub fn positions(&self) -> Option<impl Iterator<Item = Vector3> + '_> {
        let pos = self.attributes.get("position")?;
        if pos.item_size != 3 {
            return None;
        }
        Some(
            pos.array
                .chunks_exact(3)
                .map(|c| Vector3::new(c[0], c[1], c[2])),
        )
    }

    /// Compute (or refresh) the bounding box from the "position" attribute.
    /// Matches three.js's `computeBoundingBox`.
    pub fn compute_bounding_box(&mut self) -> Box3 {
        let bb = match self.positions() {
            Some(iter) => {
                let mut b = Box3::empty();
                for p in iter {
                    b.expand_by_point(p);
                }
                if b.is_empty() {
                    Box3::new(Vector3::ZERO, Vector3::ZERO)
                } else {
                    b
                }
            }
            None => Box3::new(Vector3::ZERO, Vector3::ZERO),
        };
        self.bounding_box = Some(bb);
        bb
    }

    /// Compute (or refresh) the bounding sphere from positions.
    /// Matches three.js's `computeBoundingSphere`: center on the bounding box
    /// center, then radius = max distance to any vertex.
    pub fn compute_bounding_sphere(&mut self) -> Sphere {
        let bb = self
            .bounding_box
            .unwrap_or_else(|| self.compute_bounding_box());
        let center = bb.center();
        let mut max_r2 = 0.0f32;
        if let Some(iter) = self.positions() {
            for p in iter {
                let d2 = (p - center).length_sq();
                if d2 > max_r2 {
                    max_r2 = d2;
                }
            }
        }
        let s = Sphere::new(center, max_r2.sqrt());
        self.bounding_sphere = Some(s);
        s
    }

    /// Build and store a BVH bounds tree on this geometry.
    #[cfg(feature = "mesh-bvh")]
    pub fn compute_bounds_tree(
        &mut self,
        options: crate::mesh_bvh::BuildOptions,
    ) -> Option<Arc<crate::mesh_bvh::MeshBvh>> {
        let bvh = Arc::new(crate::mesh_bvh::MeshBvh::build(self, options)?);
        self.bounds_tree = Some(bvh.clone());
        Some(bvh)
    }

    /// Drop the stored BVH bounds tree.
    #[cfg(feature = "mesh-bvh")]
    pub fn dispose_bounds_tree(&mut self) {
        self.bounds_tree = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_positions() -> BufferAttribute {
        BufferAttribute::new(
            vec![
                -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
                1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
            3,
        )
    }

    #[test]
    fn bounding_box_of_unit_cube() {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", unit_cube_positions());
        let b = g.compute_bounding_box();
        assert_eq!(b.min, Vector3::new(-1.0, -1.0, -1.0));
        assert_eq!(b.max, Vector3::new(1.0, 1.0, 1.0));
    }

    #[test]
    fn bounding_sphere_of_unit_cube_radius() {
        let mut g = BufferGeometry::new();
        g.set_attribute("position", unit_cube_positions());
        let s = g.compute_bounding_sphere();
        assert_eq!(s.center, Vector3::ZERO);
        assert!((s.radius - 3.0f32.sqrt()).abs() < 1e-5);
    }
}
