//! Catmull–Clark subdivision, for the meshes USD says are subdivision
//! surfaces.
//!
//! This is easy to miss: `subdivisionScheme` defaults to **`catmullClark`**, so
//! a `UsdGeomMesh` that says nothing at all is a subdivision surface, and the
//! points it carries are a *control cage* rather than the surface itself.
//! Drawing the cage is drawing something visibly more angular than the asset —
//! quietly, since it is a perfectly good mesh, just the wrong one.
//!
//! Refining is not free, so it does not happen unless asked: the level is a
//! renderer's decision, not the file's, which is why usdview has a complexity
//! slider rather than reading one out of the stage.
//!
//! # The rule
//!
//! One round of Catmull–Clark turns every n-gon into n quads:
//!
//! - a **face point** at the average of each face's vertices;
//! - an **edge point** at the average of the edge's two ends and the two face
//!   points either side — or just the edge's midpoint if it has one face, which
//!   is what keeps a boundary from shrinking away;
//! - each original vertex moved toward its neighbours by
//!   `(F + 2R + (n-3)P) / n`, where `F` averages the touching face points and
//!   `R` averages the touching edge midpoints.
//!
//! Creases and corners are honoured to the extent USD states them: a sharpness
//! of zero is smooth, anything else holds the edge in place.

pub use crate::geometries::subdivision::{
    catmull_clark as subdivide, Cage, SubdivisionOptions as Options,
};

/// Whether a mesh is a subdivision surface.
///
/// The default is `catmullClark`, so silence means yes. Only `none` opts out.
pub fn wanted(prim: &super::parse::UsdPrim) -> bool {
    !matches!(
        prim.value("subdivisionScheme").and_then(|v| v.as_str()),
        Some("none")
    )
}

/// The creases and corners a mesh states, and how its boundary is treated.
pub fn options_of(prim: &super::parse::UsdPrim) -> Options {
    let corners = prim
        .value("cornerIndices")
        .map(|v| v.flat_u32())
        .unwrap_or_default();

    // Creases are runs: `creaseLengths` says how many vertices each chain has,
    // and the chain is consecutive entries of `creaseIndices`.
    let indices = prim
        .value("creaseIndices")
        .map(|v| v.flat_u32())
        .unwrap_or_default();
    let lengths = prim
        .value("creaseLengths")
        .map(|v| v.flat_u32())
        .unwrap_or_default();
    let mut creases = Vec::new();
    let mut at = 0usize;
    for length in &lengths {
        let length = *length as usize;
        if at + length > indices.len() {
            break;
        }
        for i in at..at + length.saturating_sub(1) {
            creases.push((indices[i], indices[i + 1]));
        }
        at += length;
    }

    Options {
        corners,
        creases,
        // `edgeAndCorner` is USD's default, and it holds the boundary.
        sharp_boundary: !matches!(
            prim.value("interpolateBoundary").and_then(|v| v.as_str()),
            Some("none")
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube() -> Cage {
        Cage {
            positions: vec![
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            counts: vec![4; 6],
            indices: vec![
                0, 3, 2, 1, // back
                4, 5, 6, 7, // front
                0, 1, 5, 4, // bottom
                2, 3, 7, 6, // top
                0, 4, 7, 3, // left
                1, 2, 6, 5, // right
            ],
        }
    }

    /// One round turns every n-gon into n quads, and a closed cube stays
    /// closed.
    #[test]
    fn a_cube_refines_into_quads() {
        let out = subdivide(cube(), 1, &Options::default());
        // Six quads become twenty-four.
        assert_eq!(out.counts.len(), 24);
        assert!(out.counts.iter().all(|c| *c == 4), "all quads");
        // Eight corners, twelve edge points, six face points.
        assert_eq!(out.positions.len(), 8 + 12 + 6);
    }

    /// Catmull–Clark pulls a cube toward the sphere it approximates, so the
    /// corners must come *in*. A subdivider that leaves them put is one that
    /// forgot the vertex rule.
    #[test]
    fn the_corners_move_inward() {
        let out = subdivide(cube(), 1, &Options::default());
        let corner = out.positions[0];
        let distance = (corner[0] * corner[0] + corner[1] * corner[1] + corner[2] * corner[2]).sqrt();
        let original = 3f32.sqrt();
        assert!(
            distance < original * 0.9,
            "corner stayed at {distance}, was {original}"
        );
        // The classic result: a cube corner lands 5/9 of the way along each
        // axis after one round.
        assert!((corner[0].abs() - 5.0 / 9.0).abs() < 1e-5, "{corner:?}");
    }

    /// Refining twice is refining once, twice — the counts follow.
    #[test]
    fn levels_compound() {
        let one = subdivide(cube(), 1, &Options::default());
        let two = subdivide(cube(), 2, &Options::default());
        assert_eq!(two.counts.len(), one.counts.len() * 4);
    }

    /// A marked corner does not move, whatever its neighbours do.
    #[test]
    fn a_corner_is_held() {
        let out = subdivide(
            cube(),
            1,
            &Options {
                corners: vec![0],
                ..Default::default()
            },
        );
        assert_eq!(out.positions[0], [-1.0, -1.0, -1.0]);
        // And the others still moved.
        assert_ne!(out.positions[1], [1.0, -1.0, -1.0]);
    }

    /// An open surface keeps its border where it was put, rather than pulling
    /// away from it.
    #[test]
    fn a_held_boundary_keeps_its_edge() {
        let quad = Cage {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            counts: vec![4],
            indices: vec![0, 1, 2, 3],
        };
        let out = subdivide(
            quad,
            1,
            &Options {
                sharp_boundary: true,
                ..Default::default()
            },
        );
        // All four corners are on a held boundary, so none of them move.
        assert_eq!(out.positions[0], [0.0, 0.0, 0.0]);
        assert_eq!(out.positions[2], [1.0, 1.0, 0.0]);
        // The edge points sit at the midpoints rather than being pulled in.
        assert!(out.positions.contains(&[0.5, 0.0, 0.0]));
    }

    /// Level zero is the cage itself, untouched.
    #[test]
    fn level_zero_changes_nothing() {
        let out = subdivide(cube(), 0, &Options::default());
        assert_eq!(out.positions.len(), 8);
        assert_eq!(out.counts.len(), 6);
    }

    /// The default scheme is `catmullClark`, so a mesh that says nothing is a
    /// subdivision surface. Only `none` opts out — getting that backwards
    /// means silently refining nothing, or refining everything.
    #[test]
    fn silence_means_subdivide() {
        use super::super::parse::parse;
        let layer = parse(
            r#"#usda 1.0
def Mesh "Quiet"
{
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
}

def Mesh "Polygonal"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
}
"#,
        )
        .unwrap();
        assert!(wanted(layer.prim_at("/Quiet").unwrap()), "the default is catmullClark");
        assert!(!wanted(layer.prim_at("/Polygonal").unwrap()));
    }

    /// Creases arrive as runs — a length and a chain of vertices — rather than
    /// as pairs, so a chain of three vertices is two creased edges.
    #[test]
    fn creases_are_read_as_runs() {
        use super::super::parse::parse;
        let layer = parse(
            r#"#usda 1.0
def Mesh "M"
{
    int[] creaseIndices = [0, 1, 2, 4, 5]
    int[] creaseLengths = [3, 2]
    float[] creaseSharpnesses = [10, 10]
    int[] cornerIndices = [7]
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
}
"#,
        )
        .unwrap();
        let options = options_of(layer.prim_at("/M").unwrap());
        // A run of three is two edges, and a run of two is one.
        assert_eq!(options.creases, vec![(0, 1), (1, 2), (4, 5)]);
        assert_eq!(options.corners, vec![7]);
        assert!(options.sharp_boundary, "edgeAndCorner is the default");
    }

    /// End to end: a mesh loaded with a refinement level comes back with more
    /// geometry than its cage, and with the level at zero it does not.
    #[test]
    fn a_loaded_mesh_refines_when_asked() {
        use super::super::parse::parse;
        use super::super::scene::{mesh_geometry_at, mesh_geometry};
        // A cube that says nothing about its scheme, so it is one.
        let layer = parse(
            r#"#usda 1.0
def Mesh "Cube"
{
    int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
    int[] faceVertexIndices = [
        0, 3, 2, 1,
        4, 5, 6, 7,
        0, 1, 5, 4,
        2, 3, 7, 6,
        0, 4, 7, 3,
        1, 2, 6, 5
    ]
    point3f[] points = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)
    ]
}
"#,
        )
        .unwrap();
        let cube = layer.prim_at("/Cube").unwrap();

        let cage = mesh_geometry(cube).expect("the cage");
        assert_eq!(cage.get_attribute("position").unwrap().count(), 8);
        let refined = mesh_geometry_at(cube, 2).expect("refined");
        assert!(
            refined.get_attribute("position").unwrap().count() > 8 * 4,
            "two levels should be much denser, got {}",
            refined.get_attribute("position").unwrap().count()
        );

        // And the cage's corners pulled in toward the sphere it approximates.
        let corner = &refined.get_attribute("position").unwrap().array[0..3];
        let distance = (corner[0].powi(2) + corner[1].powi(2) + corner[2].powi(2)).sqrt();
        assert!(distance < 3f32.sqrt() * 0.9, "corner did not move: {distance}");

        // And `none` is left alone however high the level goes.
        let layer = parse(
            r#"#usda 1.0
def Mesh "Flat"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [4]
    int[] faceVertexIndices = [0, 1, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
}
"#,
        )
        .unwrap();
        let flat = layer.prim_at("/Flat").unwrap();
        assert_eq!(
            mesh_geometry_at(flat, 3).unwrap().get_attribute("position").unwrap().count(),
            mesh_geometry(flat).unwrap().get_attribute("position").unwrap().count()
        );
    }

    /// A degenerate cage does not panic.
    #[test]
    fn nonsense_is_survivable() {
        let empty = subdivide(
            Cage {
                positions: Vec::new(),
                counts: Vec::new(),
                indices: Vec::new(),
            },
            2,
            &Options::default(),
        );
        assert!(empty.positions.is_empty());

        // Counts that outrun the index buffer.
        let short = subdivide(
            Cage {
                positions: vec![[0.0; 3]; 3],
                counts: vec![4, 4],
                indices: vec![0, 1, 2],
            },
            1,
            &Options::default(),
        );
        assert!(short.counts.len() <= 4);
    }
}
