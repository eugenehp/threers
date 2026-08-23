//! `BufferGeometry` from a NURBS surface.
//!
//! The thin adapter between [`crate::nurbs`] and the renderer. Everything
//! interesting lives in [`crate::nurbs::tessellate`]; this exists so NURBS
//! surfaces reach a `Mesh` the same way every other geometry generator does.

use crate::core::BufferGeometry;
use crate::nurbs::{tessellate, NurbsSurface, TessellationOptions};

pub struct NurbsGeometry;

impl NurbsGeometry {
    /// Tessellate to the default chord tolerance (1e-3 model units).
    ///
    /// Unlike the fixed-segment generators next door, the vertex count is not a
    /// parameter — it falls out of the surface's curvature. A plane costs four
    /// vertices no matter how large it is.
    pub fn new(surface: &NurbsSurface) -> BufferGeometry {
        tessellate::tessellate_surface(surface, &TessellationOptions::default())
    }

    /// Tessellate to a specific chord tolerance, in model units.
    pub fn with_tolerance(surface: &NurbsSurface, tolerance: f64) -> BufferGeometry {
        tessellate::tessellate_surface(surface, &TessellationOptions::with_tolerance(tolerance))
    }

    /// Tessellate on a uniform `u_segments × v_segments` grid.
    ///
    /// For callers that need a predictable vertex layout — parity fixtures,
    /// exports with a fixed budget — rather than a tolerance guarantee.
    pub fn with_segments(
        surface: &NurbsSurface,
        u_segments: usize,
        v_segments: usize,
    ) -> BufferGeometry {
        let (u0, u1) = surface.domain_u();
        let (v0, v1) = surface.domain_v();
        let nu = u_segments.max(1);
        let nv = v_segments.max(1);
        let params_u: Vec<f64> = (0..=nu)
            .map(|i| u0 + (u1 - u0) * i as f64 / nu as f64)
            .collect();
        let params_v: Vec<f64> = (0..=nv)
            .map(|j| v0 + (v1 - v0) * j as f64 / nv as f64)
            .collect();
        tessellate::tessellate_surface_grid(
            surface,
            &params_u,
            &params_v,
            &TessellationOptions::default(),
        )
    }

    /// Full control over sampling.
    pub fn with_options(surface: &NurbsSurface, opts: &TessellationOptions) -> BufferGeometry {
        tessellate::tessellate_surface(surface, opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nurbs::construct;

    #[test]
    fn fixed_grid_has_the_expected_vertex_count() {
        let s = construct::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        let g = NurbsGeometry::with_segments(&s, 16, 4);
        assert_eq!(g.get_attribute("position").unwrap().count(), 17 * 5);
    }

    #[test]
    fn tolerance_drives_the_vertex_count() {
        let s = construct::sphere([0.0; 3], 1.0);
        let coarse = NurbsGeometry::with_tolerance(&s, 1e-2);
        let fine = NurbsGeometry::with_tolerance(&s, 1e-5);
        assert!(
            fine.get_attribute("position").unwrap().count()
                > coarse.get_attribute("position").unwrap().count()
        );
    }

    #[test]
    fn a_plane_costs_four_vertices() {
        let s = construct::plane([0.0; 3], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]);
        let g = NurbsGeometry::new(&s);
        assert_eq!(g.get_attribute("position").unwrap().count(), 4);
        assert_eq!(g.index.as_ref().unwrap().len(), 6);
    }
}
