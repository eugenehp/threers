//! Exact 2D Delaunay triangulation and discrete Voronoi, behind the `rlx-geo`
//! feature.
//!
//! "Exact" is the whole point: `rlx-geo` evaluates its orientation and
//! in-circle predicates in integer arithmetic, so the answer is *the* Delaunay
//! triangulation rather than one that a floating-point predicate rounded into
//! a near-miss. Near-degenerate inputs — four points on a circle, three nearly
//! collinear — are where floating-point triangulators produce flipped or
//! missing triangles, and they are exactly the inputs a scattered height
//! sample or a jittered grid produces.
//!
//! The price is that the points must first land on an integer lattice. That is
//! [`crate::rlx::geo::Grid`], and it is the one lossy step in this module: two points closer
//! together than one lattice step become one point, and the duplicate is
//! dropped rather than triangulated twice.
//!
//! ```
//! use threers::math::{Vector2, Vector3};
//! use threers::rlx::geo;
//!
//! let square = [
//!     Vector2::new(0.0, 0.0),
//!     Vector2::new(1.0, 0.0),
//!     Vector2::new(0.0, 1.0),
//!     Vector2::new(1.0, 1.0),
//! ];
//! assert_eq!(geo::delaunay(&square).unwrap().len(), 2);
//!
//! // …or straight to a mesh, with y as the height over the xz plane.
//! let samples = [
//!     Vector3::new(0.0, 0.2, 0.0),
//!     Vector3::new(1.0, 0.0, 0.0),
//!     Vector3::new(0.0, 0.1, 1.0),
//!     Vector3::new(1.0, 0.4, 1.0),
//! ];
//! let terrain = geo::heightfield_geometry(&samples).unwrap();
//! assert_eq!(terrain.index.as_ref().unwrap().len(), 6);
//! ```

use ::rlx_geo::{voronoi_grid_exact, FAST_COORDINATE_SPAN};

use crate::core::{BufferAttribute, BufferGeometry};
use crate::math::{Color, Vector2, Vector3};
use crate::textures::{Texture, TextureFormat};
use crate::utils::compute_vertex_normals;

use super::linear_to_srgb;

/// rlx-geo's own error — a coordinate span past what its predicates certify.
/// Reachable only through [`TriangulationError::Exact`], and only if a future
/// [`crate::rlx::geo::Grid`] outgrows it.
pub use ::rlx_geo::GeoError;

/// How finely the float coordinates are laid onto the integer lattice.
///
/// The triangulation is exact on the lattice either way; this decides how much
/// of the input's own precision survives the trip onto it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Grid {
    /// 16 777 216 steps across the wider of the two extents — `2^24`, so every
    /// value an f32 mantissa can tell apart stays distinguishable. rlx-geo
    /// evaluates this span with its 128-bit predicates: still exact, and
    /// slower than [`Grid::Fast`] by roughly the cost of the wider arithmetic.
    #[default]
    Fine,
    /// 29 609 steps — the span rlx-geo's 64-bit predicates cover, and its fast
    /// path. Enough for a few thousand well-spread points; points nearer each
    /// other than 1/29 609 of the extent merge.
    Fast,
}

impl Grid {
    fn span(self) -> f32 {
        match self {
            Self::Fine => 16_777_216.0,
            Self::Fast => FAST_COORDINATE_SPAN as f32,
        }
    }
}

/// What a triangulation can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriangulationError {
    /// A coordinate was NaN or infinite. There is no lattice point for it, and
    /// quantising it would put it somewhere arbitrary and plausible-looking.
    NonFinite { index: usize },
    /// rlx-geo declined the input.
    Exact(GeoError),
}

impl std::fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFinite { index } => write!(f, "point {index} is not finite"),
            Self::Exact(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for TriangulationError {}

impl From<GeoError> for TriangulationError {
    fn from(e: GeoError) -> Self {
        Self::Exact(e)
    }
}

/// The Delaunay triangulation of `points`, as triangles of indices into
/// `points` itself.
///
/// Points that quantise onto the same lattice site are represented once, by
/// the lowest of their indices; the others appear in no triangle. Fewer than
/// three distinct sites, or all sites collinear, is not an error — it is an
/// empty triangulation.
pub fn delaunay(points: &[Vector2]) -> Result<Vec<[u32; 3]>, TriangulationError> {
    delaunay_with(points, Grid::default())
}

/// [`delaunay`] on a chosen lattice.
pub fn delaunay_with(points: &[Vector2], grid: Grid) -> Result<Vec<[u32; 3]>, TriangulationError> {
    let lattice = quantize(points, grid)?;
    Ok(triangulate(&lattice)?)
}

/// rlx-geo's triangulator, by the door that is open on this target.
///
/// `rlx_geo::triangulate` — and the Dwyer builder under it — calls
/// `std::time::Instant::now()` unconditionally, reading the result only when
/// `GEO_PROF` is set. `wasm32-unknown-unknown` has no clock, and that call
/// does not fail politely: it **traps**, taking the whole module instance with
/// it. Every triangulation in a browser aborts the page's wasm, and nothing in
/// the type system or the compiler says so — `cargo check` is perfectly happy.
///
/// So wasm goes through `delaunay32::Triangulator`, rlx-geo's Guibas-Stolfi
/// divide-and-conquer path, which touches neither the clock nor the
/// environment. Same exact integer predicates, same index convention, same
/// answer — measured against the Dwyer build on 5 000, 50 000 and 200 000
/// points, identical triangle counts at 2.3×, 1.9× and 2.4× the time. That is
/// the price of triangulating at all in a browser. Forced to one thread,
/// because wasm has none.
///
/// Delete both arms once rlx-geo gates its profiling behind
/// `cfg(not(target_arch = "wasm32"))` — nothing else here needs to change.
#[cfg(not(target_arch = "wasm32"))]
fn triangulate(points: &[[i32; 2]]) -> Result<Vec<[u32; 3]>, GeoError> {
    ::rlx_geo::triangulate(points)
}

#[cfg(target_arch = "wasm32")]
fn triangulate(points: &[[i32; 2]]) -> Result<Vec<[u32; 3]>, GeoError> {
    use ::rlx_geo::delaunay32::{Point, Triangulator};
    let sites: Vec<Point> = points.iter().map(|p| Point::new(p[0], p[1])).collect();
    Ok(Triangulator::with_threads(1)
        .triangulate(&sites)
        .into_iter()
        .map(|t| [t.i0, t.i1, t.i2])
        .collect())
}

/// [`delaunay`] flattened into the index buffer
/// [`BufferGeometry::set_index`] takes.
pub fn delaunay_indices(points: &[Vector2]) -> Result<Vec<u32>, TriangulationError> {
    Ok(delaunay(points)?.into_iter().flatten().collect())
}

/// The convex hull of `points`, counter-clockwise, as indices into `points`.
pub fn convex_hull(points: &[Vector2]) -> Result<Vec<u32>, TriangulationError> {
    Ok(::rlx_geo::convex_hull(&quantize(points, Grid::default())?))
}

/// Scattered samples as a mesh: triangulate in xz, keep y as the height.
///
/// The usual reason to want an exact triangulator in a renderer. The result
/// carries `position`, `normal` (computed from the triangles) and `uv`
/// (normalised over the xz bounding box), and its triangles face +y — the
/// winding is reversed on the way out of rlx-geo's plane, whose y is this
/// mesh's z.
pub fn heightfield_geometry(points: &[Vector3]) -> Result<BufferGeometry, TriangulationError> {
    heightfield_geometry_with(points, Grid::default())
}

/// [`heightfield_geometry`] on a chosen lattice.
pub fn heightfield_geometry_with(
    points: &[Vector3],
    grid: Grid,
) -> Result<BufferGeometry, TriangulationError> {
    let plane: Vec<Vector2> = points.iter().map(|p| Vector2::new(p.x, p.z)).collect();
    let triangles = delaunay_with(&plane, grid)?;

    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_z, mut max_z) = (f32::INFINITY, f32::NEG_INFINITY);
    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_z = min_z.min(p.z);
        max_z = max_z.max(p.z);
    }
    let span_x = (max_x - min_x).max(f32::MIN_POSITIVE);
    let span_z = (max_z - min_z).max(f32::MIN_POSITIVE);

    let mut positions = Vec::with_capacity(points.len() * 3);
    let mut uvs = Vec::with_capacity(points.len() * 2);
    for p in points {
        positions.extend_from_slice(&[p.x, p.y, p.z]);
        uvs.extend_from_slice(&[(p.x - min_x) / span_x, (p.z - min_z) / span_z]);
    }

    let mut geometry = BufferGeometry::new();
    geometry.set_attribute("position", BufferAttribute::new(positions, 3));
    geometry.set_attribute("uv", BufferAttribute::new(uvs, 2));
    geometry.set_index(
        triangles
            .into_iter()
            .flat_map(|[a, b, c]| [a, c, b])
            .collect(),
    );
    compute_vertex_normals(&mut geometry);
    Ok(geometry)
}

/// Which site owns each pixel of a `width × height` grid, row-major.
///
/// Sites are in pixel coordinates and are rounded to the nearest pixel; the
/// label is an index into `sites`. With no sites every label is `u32::MAX`.
pub fn voronoi_labels(sites: &[Vector2], width: u32, height: u32) -> Vec<u32> {
    let lattice: Vec<[i32; 2]> = sites
        .iter()
        .map(|s| [s.x.round() as i32, s.y.round() as i32])
        .collect();
    voronoi_grid_exact(&lattice, width, height)
}

/// A Voronoi diagram as a texture, one colour per site.
///
/// `colors` is indexed by site and wraps if it is shorter, so a handful of
/// colours tiles a thousand cells. Pixels no site claims — only possible when
/// `sites` is empty — come out transparent.
pub fn voronoi_texture(sites: &[Vector2], colors: &[Color], width: u32, height: u32) -> Texture {
    let labels = voronoi_labels(sites, width, height);
    let mut data = Vec::with_capacity(labels.len() * 4);
    for label in labels {
        match colors.is_empty() || label == u32::MAX {
            true => data.extend_from_slice(&[0, 0, 0, 0]),
            false => {
                let c = colors[label as usize % colors.len()];
                data.extend_from_slice(&[
                    quantize_srgb(c.r),
                    quantize_srgb(c.g),
                    quantize_srgb(c.b),
                    255,
                ]);
            }
        }
    }
    Texture::new(width, height, TextureFormat::Rgba8UnormSrgb, data)
}

/// Distance from every pixel to the site that owns it, in pixels.
///
/// The F1 distance field, and the one texture-generation primitive that a
/// label map cannot stand in for: cell *noise* is a function of how far into
/// a cell you are, not of which cell you are in.
pub fn voronoi_distance_field(sites: &[Vector2], width: u32, height: u32) -> Vec<f32> {
    let labels = voronoi_labels(sites, width, height);
    let mut out = vec![0.0f32; labels.len()];
    for (i, label) in labels.iter().enumerate() {
        let Some(site) = sites.get(*label as usize) else {
            continue;
        };
        let x = (i % width.max(1) as usize) as f32;
        let y = (i / width.max(1) as usize) as f32;
        out[i] = ((x - site.x).powi(2) + (y - site.y).powi(2)).sqrt();
    }
    out
}

/// Distance from every pixel to the nearest cell *wall*, in pixels.
///
/// The quantity most cell textures actually want.
/// [`voronoi_distance_field`] measures towards the site, so it is smallest in
/// the middle of a cell and largest at its edge — which builds pits where mud,
/// leather and paint have domes. This is the other way round, and it is what
/// makes a cell read as a raised plate with a sunken seam around it.
///
/// Exact, and not a blur of the edge mask: a Voronoi cell is an intersection
/// of half-planes, so the distance from a point inside it to its boundary is
/// the smallest distance to any of the bisectors that bound it — which is
/// `(|p−sⱼ|² − |p−sᵢ|²) / 2|sⱼ−sᵢ|`, minimised over the other sites.
///
/// That minimisation is over *every* site, so the cost is pixels × sites. Fine
/// for the few hundred cells a texture wants; not the way to do ten thousand.
pub fn voronoi_wall_distance(sites: &[Vector2], width: u32, height: u32) -> Vec<f32> {
    let labels = voronoi_labels(sites, width, height);
    let mut out = vec![0.0f32; labels.len()];
    if sites.len() < 2 {
        return out;
    }
    for (i, label) in labels.iter().enumerate() {
        let Some(mine) = sites.get(*label as usize) else {
            continue;
        };
        let x = (i % width.max(1) as usize) as f32;
        let y = (i / width.max(1) as usize) as f32;
        let d_mine = (x - mine.x).powi(2) + (y - mine.y).powi(2);
        let mut nearest = f32::INFINITY;
        for (j, other) in sites.iter().enumerate() {
            if j == *label as usize {
                continue;
            }
            let separation = (other.x - mine.x).hypot(other.y - mine.y);
            if separation <= f32::EPSILON {
                continue;
            }
            let d_other = (x - other.x).powi(2) + (y - other.y).powi(2);
            nearest = nearest.min((d_other - d_mine) / (2.0 * separation));
        }
        out[i] = nearest.max(0.0);
    }
    out
}

/// 1 on a pixel whose right or lower neighbour belongs to another cell, 0
/// elsewhere — the cell walls.
///
/// Two neighbours rather than four: every wall is found exactly once this way,
/// which keeps a wall one pixel wide instead of two.
pub fn voronoi_edges(sites: &[Vector2], width: u32, height: u32) -> Vec<f32> {
    let labels = voronoi_labels(sites, width, height);
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0.0f32; labels.len()];
    for y in 0..h {
        for x in 0..w {
            let here = labels[y * w + x];
            let right = (x + 1 < w).then(|| labels[y * w + x + 1]);
            let below = (y + 1 < h).then(|| labels[(y + 1) * w + x]);
            if right.is_some_and(|l| l != here) || below.is_some_and(|l| l != here) {
                out[y * w + x] = 1.0;
            }
        }
    }
    out
}

/// A height grid as a tangent-space normal map, `Rgba8Unorm`.
///
/// Central differences, so a sample's slope is measured from its neighbours on
/// both sides rather than from itself and one neighbour — the one-sided form
/// shifts every feature half a pixel, which shows up as a normal map that does
/// not line up with the height it came from.
///
/// The format is linear, not sRGB: these are vectors, and encoding them with a
/// transfer function meant for light would bend them.
pub fn normal_map_from_height(field: &[f32], width: u32, height: u32, strength: f32) -> Texture {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 || field.len() < w * h {
        return Texture::new(0, 0, TextureFormat::Rgba8Unorm, Vec::new());
    }
    let at = |x: usize, y: usize| field[y.min(h - 1) * w + x.min(w - 1)];
    let mut data = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let dx = (at(x + 1, y) - at(x.saturating_sub(1), y)) * 0.5 * strength;
            let dy = (at(x, y + 1) - at(x, y.saturating_sub(1))) * 0.5 * strength;
            // The surface normal of a height field: (-∂h/∂x, -∂h/∂y, 1).
            let inv = 1.0 / (dx * dx + dy * dy + 1.0).sqrt();
            let n = [-dx * inv, -dy * inv, inv];
            for c in n {
                data.push(((c * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            data.push(255);
        }
    }
    Texture::new(width, height, TextureFormat::Rgba8Unorm, data)
}

/// What [`refine_heightfield`] converged to.
#[derive(Debug, Clone)]
pub struct Refinement {
    /// The samples it ended up taking, `y` holding the field's value.
    pub points: Vec<Vector3>,
    /// Those samples triangulated, ready to render.
    pub geometry: BufferGeometry,
    /// The largest remaining gap between the mesh and the field, at the
    /// centroid of any triangle.
    pub error: f32,
    /// How many insert-and-retriangulate rounds it took.
    pub rounds: usize,
}

/// Sample a height field where it needs sampling.
///
/// Starts from a coarse grid and repeatedly inserts points at the centroids of
/// the triangles that misrepresent the field worst — the classic Delaunay
/// refinement loop. A flat region keeps its four corners; a ridge collects
/// points until it is a ridge and not a chord.
///
/// This is where exactness earns its keep. Every inserted point is the
/// centroid of an existing triangle, which is precisely the construction that
/// produces near-cocircular configurations, and a floating-point in-circle
/// test that flips on one of them leaves a sliver or a fold in the middle of
/// the mesh. There is no tolerance to tune here because there is no tolerance.
///
/// Stops at `target_error`, at `max_points`, or when a round adds nothing.
pub fn refine_heightfield(
    field: impl Fn(f32, f32) -> f32,
    min: Vector2,
    max: Vector2,
    target_error: f32,
    max_points: usize,
) -> Result<Refinement, TriangulationError> {
    // A 3×3 seed: the corners alone are two triangles, and every insertion
    // after that would come from the same two chords.
    let mut points: Vec<Vector3> = Vec::new();
    for j in 0..3 {
        for i in 0..3 {
            let x = min.x + (max.x - min.x) * i as f32 / 2.0;
            let z = min.y + (max.y - min.y) * j as f32 / 2.0;
            points.push(Vector3::new(x, field(x, z), z));
        }
    }

    let mut error = f32::INFINITY;
    let mut rounds = 0;
    while points.len() < max_points {
        let plane: Vec<Vector2> = points.iter().map(|p| Vector2::new(p.x, p.z)).collect();
        let triangles = delaunay(&plane)?;
        if triangles.is_empty() {
            break;
        }

        // How wrong is the mesh at each triangle's centroid? The centroid's
        // interpolated height is the mean of the three corners.
        let mut worst: Vec<(f32, Vector3)> = triangles
            .iter()
            .map(|[a, b, c]| {
                let (a, b, c) = (
                    points[*a as usize],
                    points[*b as usize],
                    points[*c as usize],
                );
                let x = (a.x + b.x + c.x) / 3.0;
                let z = (a.z + b.z + c.z) / 3.0;
                let interpolated = (a.y + b.y + c.y) / 3.0;
                let actual = field(x, z);
                ((actual - interpolated).abs(), Vector3::new(x, actual, z))
            })
            .collect();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        error = worst[0].0;
        rounds += 1;
        if error <= target_error {
            break;
        }

        // Insert a batch rather than one point per round: one at a time is the
        // textbook loop and would retriangulate thousands of times to place
        // thousands of points.
        let budget = (max_points - points.len()).min((worst.len() / 8).max(1));
        let before = points.len();
        for (_, p) in worst.into_iter().take(budget) {
            points.push(p);
        }
        if points.len() == before {
            break;
        }
    }

    let geometry = heightfield_geometry(&points)?;
    Ok(Refinement {
        points,
        geometry,
        error,
        rounds,
    })
}

/// Lay the points on the integer lattice, scaled so the wider extent spans the
/// grid and the lower-left corner sits at the origin.
///
/// Both axes take the *same* scale. Scaling them independently would stretch
/// the plane, and a Delaunay triangulation is not invariant under that — the
/// answer would be exact and would be the answer to a different question.
fn quantize(points: &[Vector2], grid: Grid) -> Result<Vec<[i32; 2]>, TriangulationError> {
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for (index, p) in points.iter().enumerate() {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(TriangulationError::NonFinite { index });
        }
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }
    let extent = (max_x - min_x).max(max_y - min_y);
    // Every point coincident: any scale gives the same single lattice site.
    let scale = if extent > 0.0 {
        grid.span() / extent
    } else {
        0.0
    };
    Ok(points
        .iter()
        .map(|p| {
            [
                ((p.x - min_x) * scale).round() as i32,
                ((p.y - min_y) * scale).round() as i32,
            ]
        })
        .collect())
}

fn quantize_srgb(v: f32) -> u8 {
    (linear_to_srgb(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_square_triangulates_into_two_triangles_covering_it() {
        let square = [
            Vector2::new(-1.0, -1.0),
            Vector2::new(1.0, -1.0),
            Vector2::new(1.0, 1.0),
            Vector2::new(-1.0, 1.0),
        ];
        let tris = delaunay(&square).unwrap();
        assert_eq!(tris.len(), 2);
        let mut used: Vec<u32> = tris.iter().flatten().copied().collect();
        used.sort_unstable();
        used.dedup();
        assert_eq!(used, vec![0, 1, 2, 3]);
    }

    #[test]
    fn degenerate_inputs_are_empty_rather_than_wrong() {
        assert!(delaunay(&[]).unwrap().is_empty());
        assert!(delaunay(&[Vector2::ZERO; 4]).unwrap().is_empty());
        let collinear: Vec<Vector2> = (0..5).map(|i| Vector2::new(i as f32, 0.0)).collect();
        assert!(delaunay(&collinear).unwrap().is_empty());
    }

    #[test]
    fn a_non_finite_coordinate_is_refused() {
        let points = [Vector2::ZERO, Vector2::new(f32::NAN, 1.0)];
        assert_eq!(
            delaunay(&points),
            Err(TriangulationError::NonFinite { index: 1 })
        );
    }

    #[test]
    fn a_heightfield_faces_up() {
        let samples = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 1.0),
        ];
        let geometry = heightfield_geometry(&samples).unwrap();
        let normals = geometry.get_attribute("normal").unwrap();
        for n in normals.array.chunks_exact(3) {
            assert!(n[1] > 0.99, "normal points {n:?}, not up");
        }
        let uv = geometry.get_attribute("uv").unwrap();
        assert_eq!(uv.item_size, 2);
        assert_eq!(uv.count(), 4);
    }

    #[test]
    fn every_pixel_belongs_to_its_nearest_site() {
        let sites = [Vector2::new(1.0, 1.0), Vector2::new(6.0, 6.0)];
        let labels = voronoi_labels(&sites, 8, 8);
        assert_eq!(labels.len(), 64);
        assert_eq!(labels[0], 0);
        assert_eq!(labels[63], 1);
    }

    #[test]
    fn a_voronoi_texture_colours_by_site() {
        let sites = [Vector2::new(0.0, 0.0), Vector2::new(3.0, 3.0)];
        let texture = voronoi_texture(&sites, &[Color::WHITE, Color::BLACK], 4, 4);
        assert_eq!(texture.format, TextureFormat::Rgba8UnormSrgb);
        assert_eq!(&texture.data[0..4], &[255, 255, 255, 255]);
        assert_eq!(&texture.data[60..64], &[0, 0, 0, 255]);
    }

    #[test]
    fn the_distance_field_is_zero_at_a_site_and_grows_away_from_it() {
        let sites = [Vector2::new(4.0, 4.0)];
        let field = voronoi_distance_field(&sites, 9, 9);
        assert_eq!(field[4 * 9 + 4], 0.0);
        assert!((field[4 * 9 + 7] - 3.0).abs() < 1e-5);
        assert!(field[0] > field[4 * 9 + 7], "corner should be furthest");
    }

    #[test]
    fn wall_distance_is_zero_at_the_seam_and_peaks_in_the_cell() {
        let sites = [Vector2::new(1.0, 4.0), Vector2::new(7.0, 4.0)];
        let wall = voronoi_wall_distance(&sites, 9, 9);
        // The bisector is at x = 4: on it, the wall is underfoot.
        assert!(wall[4 * 9 + 4] < 0.51, "at the seam: {}", wall[4 * 9 + 4]);
        // At each site, the wall is half the separation away.
        assert!((wall[4 * 9 + 1] - 3.0).abs() < 0.01, "{}", wall[4 * 9 + 1]);
        assert!((wall[4 * 9 + 7] - 3.0).abs() < 0.01, "{}", wall[4 * 9 + 7]);
        // …and it is the inverse of the distance to the site, which is the
        // whole reason this function exists alongside that one.
        let to_site = voronoi_distance_field(&sites, 9, 9);
        assert!(to_site[4 * 9 + 1] < to_site[4 * 9 + 4]);
        assert!(wall[4 * 9 + 1] > wall[4 * 9 + 4]);
    }

    #[test]
    fn a_single_cell_has_no_walls() {
        let wall = voronoi_wall_distance(&[Vector2::new(2.0, 2.0)], 5, 5);
        assert!(wall.iter().all(|d| *d == 0.0));
    }

    #[test]
    fn edges_land_between_cells_and_nowhere_else() {
        let sites = [Vector2::new(1.0, 4.0), Vector2::new(7.0, 4.0)];
        let edges = voronoi_edges(&sites, 9, 9);
        // Sites 6 apart put the bisector at x = 4, where the tie goes to the
        // lower index — so x = 4 is the last pixel of cell 0 and the wall.
        assert_eq!(edges[4 * 9 + 4], 1.0, "no wall at the bisector");
        assert_eq!(edges[4 * 9 + 3], 0.0, "wall inside a cell");
        assert_eq!(edges[4 * 9], 0.0, "wall inside a cell");
        assert_eq!(edges[4 * 9 + 8], 0.0, "wall inside a cell");
    }

    #[test]
    fn a_flat_height_field_has_normals_pointing_straight_up() {
        let flat = vec![0.5f32; 16];
        let map = normal_map_from_height(&flat, 4, 4, 4.0);
        assert_eq!(map.format, TextureFormat::Rgba8Unorm);
        for px in map.data.chunks_exact(4) {
            assert_eq!(px[0], 128, "x deflected on a flat field");
            assert_eq!(px[1], 128, "y deflected on a flat field");
            assert_eq!(px[2], 255, "z should be full");
        }
    }

    #[test]
    // `1 * w + x` is row-and-column arithmetic written out: the row index
    // stays visible next to the column instead of being folded away.
    #[allow(clippy::identity_op)]
    fn a_slope_tilts_the_normal_against_it() {
        // Height rising with x: the normal should lean towards −x.
        let ramp: Vec<f32> = (0..16).map(|i| (i % 4) as f32 * 0.25).collect();
        let map = normal_map_from_height(&ramp, 4, 4, 4.0);
        let centre = &map.data[(1 * 4 + 1) * 4..(1 * 4 + 1) * 4 + 3];
        assert!(centre[0] < 128, "normal did not lean against the slope");
        assert_eq!(centre[1], 128, "leaned in y on an x-only slope");
    }

    #[test]
    fn refinement_spends_its_points_where_the_field_bends() {
        // Flat on the left, a sharp bump on the right. Curved, deliberately:
        // a piecewise-*linear* field is represented exactly by any mesh whose
        // vertices sit on its creases, so refinement would have nothing to do
        // and the test would pass a broken implementation.
        let field =
            |x: f32, z: f32| 1.5 * (-(((x - 0.75).powi(2) + (z - 0.5).powi(2)) * 40.0)).exp();
        let refined = refine_heightfield(
            field,
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 1.0),
            0.02,
            600,
        )
        .unwrap();

        assert!(refined.rounds > 1, "converged without refining");
        assert!(
            refined.error <= 0.02 || refined.points.len() >= 600,
            "stopped early at error {}",
            refined.error
        );
        let right = refined.points.iter().filter(|p| p.x > 0.5).count();
        let left = refined.points.len() - right;
        assert!(
            right > left,
            "{right} points on the bent half, {left} on the flat one"
        );
    }

    #[test]
    fn refining_a_plane_stops_immediately() {
        let refined = refine_heightfield(
            |_, _| 2.0,
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 1.0),
            0.01,
            500,
        )
        .unwrap();
        assert_eq!(refined.rounds, 1);
        assert_eq!(refined.points.len(), 9, "a plane needs no more samples");
        assert!(refined.error < 1e-6);
    }

    #[test]
    fn no_sites_leaves_the_texture_empty() {
        let texture = voronoi_texture(&[], &[Color::WHITE], 2, 2);
        assert!(texture.data.iter().all(|b| *b == 0));
    }
}
