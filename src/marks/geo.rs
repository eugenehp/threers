//! Longitude and latitude to the flat map the marks draw on.
//!
//! Web Mercator (EPSG:3857), in metres at the equator, is what web maps and
//! their tiles use: `x` east from Greenwich, `y` north from the equator, both
//! up to about 20 million. Those magnitudes are why the marks keep `f64`
//! positions and draw relative to an [`Origin`](super::Origin).
//!
//! The ground plane `[x, y]` becomes world `(x, height, -y)`: Y is up, and
//! north runs away from a camera looking down −Z — the convention
//! [`PolygonFeature`](super::PolygonFeature) and
//! [`PolygonGeometry`](crate::PolygonGeometry) share.

use std::f64::consts::PI;

/// The sphere Web Mercator projects, in metres.
pub const EARTH_RADIUS: f64 = 6_378_137.0;
/// The latitude at which the projected map is square, and past which Web
/// Mercator stops: `y` runs to infinity at the poles.
pub const MAX_LATITUDE: f64 = 85.051_128_779_806_59;
/// Half the map's width (and height) in metres: `x` and `y` lie in
/// `-EXTENT..=EXTENT`.
pub const EXTENT: f64 = PI * EARTH_RADIUS;

/// `[longitude, latitude]` in degrees to Web Mercator `[x, y]` in metres.
/// Latitude is clamped to ±[`MAX_LATITUDE`].
///
/// ```
/// use threers::marks::geo::{mercator, EXTENT};
/// let [x, y] = mercator([180.0, 0.0]);
/// assert!((x - EXTENT).abs() < 1e-6 && y.abs() < 1e-9);
/// ```
pub fn mercator(lon_lat: [f64; 2]) -> [f64; 2] {
    let lon = lon_lat[0].to_radians();
    let lat = lon_lat[1].clamp(-MAX_LATITUDE, MAX_LATITUDE).to_radians();
    [
        EARTH_RADIUS * lon,
        // ln(tan(π/4 + φ/2)), in the form that is exact at the equator.
        EARTH_RADIUS * lat.sin().atanh(),
    ]
}

/// Web Mercator `[x, y]` back to `[longitude, latitude]` in degrees.
pub fn inverse_mercator(xy: [f64; 2]) -> [f64; 2] {
    [
        (xy[0] / EARTH_RADIUS).to_degrees(),
        (xy[1] / EARTH_RADIUS).sinh().atan().to_degrees(),
    ]
}

/// Map units to a true metre at `latitude`: Mercator stretches everything by
/// `1 / cos(latitude)`, so a 100 m building in Oslo (60° N) is 200 map units
/// tall if it is to keep its proportions.
pub fn units_per_meter(latitude: f64) -> f64 {
    1.0 / latitude
        .clamp(-MAX_LATITUDE, MAX_LATITUDE)
        .to_radians()
        .cos()
}

/// A point on the ground plane, `height` up, in world coordinates.
pub fn ground_to_world(xy: [f64; 2], height: f64) -> [f64; 3] {
    [xy[0], height, -xy[1]]
}

/// [`ground_to_world`] backwards: the ground point and the height.
pub fn world_to_ground(p: [f64; 3]) -> ([f64; 2], f64) {
    ([p[0], -p[2]], p[1])
}

/// `[longitude, latitude, altitude in metres]` to world coordinates through
/// Web Mercator, with the altitude scaled to the map's units at that
/// latitude.
pub fn lon_lat_to_world(p: [f64; 3]) -> [f64; 3] {
    ground_to_world(mercator([p[0], p[1]]), p[2] * units_per_meter(p[1]))
}

/// [`lon_lat_to_world`] for data rather than maps: latitudes past
/// [`MAX_LATITUDE`] are projected where Mercator puts them (far off the
/// map), as deck.gl projects each vertex, only kept short of the poles'
/// infinity. A shape reaching a pole runs off the top of the map instead of
/// being cut flat at the map's edge.
pub fn lon_lat_to_world_extended(p: [f64; 3]) -> [f64; 3] {
    let lat = p[1].clamp(-89.9999, 89.9999);
    let xy = [EARTH_RADIUS * p[0].to_radians(), EARTH_RADIUS * lat.to_radians().sin().atanh()];
    ground_to_world(xy, p[2] / lat.to_radians().cos())
}

/// A Web Mercator tile: zoom `z`, column `x` from the west, row `y` from
/// the north — the XYZ scheme slippy maps and vector tiles use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl TileId {
    pub fn new(z: u8, x: u32, y: u32) -> Self {
        Self { z, x, y }
    }

    /// The tile's side in map units (metres at the equator).
    pub fn size(&self) -> f64 {
        2.0 * EXTENT / (1u64 << self.z) as f64
    }

    /// `(min, max)` corners in Web Mercator `[x, y]`.
    pub fn bounds(&self) -> ([f64; 2], [f64; 2]) {
        let s = self.size();
        let x0 = -EXTENT + self.x as f64 * s;
        let y1 = EXTENT - self.y as f64 * s;
        ([x0, y1 - s], [x0 + s, y1])
    }

    /// The tile at zoom `z` holding Web Mercator point `xy`.
    pub fn containing(xy: [f64; 2], z: u8) -> Self {
        let n = 1u64 << z;
        let s = 2.0 * EXTENT / n as f64;
        let clamp = |v: f64| (v.floor().max(0.0) as u64).min(n - 1) as u32;
        Self {
            z,
            x: clamp((xy[0] + EXTENT) / s),
            y: clamp((EXTENT - xy[1]) / s),
        }
    }

    pub fn parent(&self) -> Option<Self> {
        (self.z > 0).then(|| Self::new(self.z - 1, self.x / 2, self.y / 2))
    }

    /// The four tiles one zoom in: north-west, north-east, south-west,
    /// south-east.
    pub fn children(&self) -> [Self; 4] {
        let (z, x, y) = (self.z + 1, self.x * 2, self.y * 2);
        [
            Self::new(z, x, y),
            Self::new(z, x + 1, y),
            Self::new(z, x, y + 1),
            Self::new(z, x + 1, y + 1),
        ]
    }

    /// Bing's quadkey: one digit per zoom level.
    pub fn quadkey(&self) -> String {
        (1..=self.z)
            .rev()
            .map(|i| {
                let m = 1 << (i - 1);
                let d = u8::from(self.x & m != 0) + 2 * u8::from(self.y & m != 0);
                (b'0' + d) as char
            })
            .collect()
    }

    pub fn from_quadkey(key: &str) -> Option<Self> {
        let mut t = Self::new(0, 0, 0);
        for c in key.chars() {
            let d = c.to_digit(10).filter(|&d| d < 4)?;
            t = t.children()[[0, 1, 2, 3][d as usize]];
        }
        Some(t)
    }
}

/// Map units per pixel of a `tile_px`-pixel tile drawn at its own size at
/// zoom `zoom` — at the equator; divide by
/// [`units_per_meter`] for true metres elsewhere.
pub fn units_per_pixel_at_zoom(zoom: f64, tile_px: f64) -> f64 {
    2.0 * EXTENT / (tile_px * zoom.exp2())
}

/// The zoom at which `tile_px`-pixel tiles show `units_per_pixel`.
pub fn zoom_for_units_per_pixel(units_per_pixel: f64, tile_px: f64) -> f64 {
    (2.0 * EXTENT / (tile_px * units_per_pixel.max(1e-12))).log2()
}

/// The tiles to draw for `view`: the tile pyramid walked from zoom 0, each
/// tile dropped when the view frustum misses it and split while it covers
/// more than `tile_px` pixels, up to `max_zoom`.
///
/// So a tilted view gets fine tiles near the camera and coarse ones towards
/// the horizon, as map renderers do; ground is the plane at `height` (world
/// Y) in Web Mercator. Sorted nearest first. At most `max_tiles` are
/// returned, the farthest dropped.
pub fn covering_tiles(
    view: &super::MarkView,
    max_zoom: u8,
    tile_px: f64,
    height: f64,
    max_tiles: usize,
) -> Vec<TileId> {
    let o = view.origin.position;
    let m = view.view_proj.map(|v| v as f64);
    let clip = |p: [f64; 3]| -> [f64; 4] {
        let l = [p[0] - o[0], p[1] - o[1], p[2] - o[2]];
        let r = |k: usize| m[k] * l[0] + m[4 + k] * l[1] + m[8 + k] * l[2] + m[12 + k];
        [r(0), r(1), r(2), r(3)]
    };
    let (vw, vh) = (view.viewport.0 as f64, view.viewport.1 as f64);
    let mut out: Vec<(f64, TileId)> = Vec::new();
    let mut stack = vec![TileId::new(0, 0, 0)];
    while let Some(t) = stack.pop() {
        let (lo, hi) = t.bounds();
        let corners = [
            [lo[0], lo[1]],
            [hi[0], lo[1]],
            [hi[0], hi[1]],
            [lo[0], hi[1]],
        ]
        .map(|g| clip(ground_to_world(g, height)));
        // Outside one frustum plane with all four corners: not visible.
        let outside = |f: &dyn Fn(&[f64; 4]) -> bool| corners.iter().all(f);
        if outside(&|c| c[0] < -c[3])
            || outside(&|c| c[0] > c[3])
            || outside(&|c| c[1] < -c[3])
            || outside(&|c| c[1] > c[3])
            || outside(&|c| c[3] <= 0.0)
        {
            continue;
        }
        // Its size on screen: unbounded when part of it is behind the eye.
        let px = if corners.iter().any(|c| c[3] <= 1e-9) {
            f64::INFINITY
        } else {
            let s: Vec<[f64; 2]> = corners
                .iter()
                .map(|c| [c[0] / c[3] * vw * 0.5, c[1] / c[3] * vh * 0.5])
                .collect();
            (0..4)
                .map(|i| {
                    let (a, b) = (s[i], s[(i + 1) % 4]);
                    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        };
        if px > tile_px && t.z < max_zoom {
            stack.extend(t.children());
        } else {
            let w = corners.iter().map(|c| c[3]).fold(f64::INFINITY, f64::min);
            out.push((w, t));
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    out.truncate(max_tiles);
    out.into_iter().map(|(_, t)| t).collect()
}

/// Where ground coordinates go in the world, for meshes built from them
/// ([`PolygonMesh::build_with`](super::PolygonMesh::build_with)).
pub trait GroundProjection: Sync {
    /// The world position of ground point `xy`, `height` up.
    fn world(&self, xy: [f64; 2], height: f64) -> [f64; 3];
    /// The point extrusion rises away from, for a curved ground; `None` for
    /// a flat one, which rises along +Y.
    fn centre(&self) -> Option<[f64; 3]> {
        None
    }
    /// The longest edge, in ground units, a mesh may keep before it is split
    /// to follow the curvature. `None` never splits.
    fn max_edge(&self) -> Option<f64> {
        None
    }
    /// The normal of a wall along an edge from ground point `at`, given the
    /// outward direction across the edge in ground coordinates; `None` (the
    /// default) for the true one, level with the surface. For renderers that
    /// light walls their own way.
    fn wall_normal(&self, _at: [f64; 2], _outward: [f64; 2]) -> Option<[f64; 3]> {
        None
    }
}

/// The flat ground: `[x, y]` to `(x, height, -y)`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlatGround;

impl GroundProjection for FlatGround {
    fn world(&self, xy: [f64; 2], height: f64) -> [f64; 3] {
        ground_to_world(xy, height)
    }
}

/// A sphere, with ground coordinates `[longitude, latitude]` in degrees —
/// see [`lon_lat_to_globe`].
#[derive(Debug, Clone, Copy)]
pub struct GlobeGround {
    pub radius: f64,
    pub centre: [f64; 3],
    /// Edges longer than this many degrees are split, so a large polygon
    /// bends with the surface instead of cutting through it.
    pub max_edge_degrees: f64,
}

impl GlobeGround {
    /// The Earth's mean radius in metres, centred on the origin, split every
    /// half a degree.
    pub fn earth() -> Self {
        Self {
            radius: EARTH_MEAN_RADIUS,
            centre: [0.0; 3],
            max_edge_degrees: 0.5,
        }
    }
}

impl GroundProjection for GlobeGround {
    fn world(&self, xy: [f64; 2], height: f64) -> [f64; 3] {
        let p = lon_lat_to_globe([xy[0], xy[1], height], self.radius);
        [
            p[0] + self.centre[0],
            p[1] + self.centre[1],
            p[2] + self.centre[2],
        ]
    }
    fn centre(&self) -> Option<[f64; 3]> {
        Some(self.centre)
    }
    fn max_edge(&self) -> Option<f64> {
        Some(self.max_edge_degrees.max(1e-6))
    }
}

/// The Earth's mean radius, metres — what a globe view uses as its sphere.
pub const EARTH_MEAN_RADIUS: f64 = 6_371_008.8;

/// `[longitude, latitude, altitude]` (degrees, degrees, metres) on a sphere
/// of `radius` around the origin, Y up: the north pole is `+Y`, and longitude
/// 0 on the equator is `+Z`, facing a camera on the +Z axis. East is `+X`.
pub fn lon_lat_to_globe(p: [f64; 3], radius: f64) -> [f64; 3] {
    let (lon, lat) = (p[0].to_radians(), p[1].to_radians());
    let r = radius + p[2];
    [
        r * lat.cos() * lon.sin(),
        r * lat.sin(),
        r * lat.cos() * lon.cos(),
    ]
}

/// [`lon_lat_to_globe`] backwards: `[longitude, latitude, altitude]`.
pub fn globe_to_lon_lat(p: [f64; 3], radius: f64) -> [f64; 3] {
    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if r == 0.0 {
        return [0.0, 0.0, -radius];
    }
    [
        p[0].atan2(p[2]).to_degrees(),
        (p[1] / r).clamp(-1.0, 1.0).asin().to_degrees(),
        r - radius,
    ]
}

/// The great-circle distance between two `[longitude, latitude]` points in
/// degrees, on a sphere of `radius` (haversine).
pub fn great_circle_distance(a: [f64; 2], b: [f64; 2], radius: f64) -> f64 {
    let (la1, la2) = (a[1].to_radians(), b[1].to_radians());
    let dla = la2 - la1;
    let dlo = (b[0] - a[0]).to_radians();
    let h = (dla * 0.5).sin().powi(2) + la1.cos() * la2.cos() * (dlo * 0.5).sin().powi(2);
    2.0 * radius * h.sqrt().min(1.0).asin()
}

/// The point a fraction `t` of the way from `a` to `b` along the great
/// circle, `[longitude, latitude]` in degrees.
pub fn great_circle_point(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let pa = lon_lat_to_globe([a[0], a[1], 0.0], 1.0);
    let pb = lon_lat_to_globe([b[0], b[1], 0.0], 1.0);
    let dot = (pa[0] * pb[0] + pa[1] * pb[1] + pa[2] * pb[2]).clamp(-1.0, 1.0);
    let w = dot.acos();
    if w < 1e-12 {
        return a;
    }
    let (sa, sb) = (((1.0 - t) * w).sin() / w.sin(), (t * w).sin() / w.sin());
    let p = [
        sa * pa[0] + sb * pb[0],
        sa * pa[1] + sb * pb[1],
        sa * pa[2] + sb * pb[2],
    ];
    let ll = globe_to_lon_lat(p, 1.0);
    [ll[0], ll[1]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_points_and_round_trip() {
        assert_eq!(mercator([0.0, 0.0]), [0.0, 0.0]);
        let [_, top] = mercator([0.0, MAX_LATITUDE]);
        assert!((top - EXTENT).abs() < 1e-3, "square at the limit: {top}");
        // Beyond the limit is clamped rather than infinite.
        assert_eq!(mercator([0.0, 90.0]), mercator([0.0, MAX_LATITUDE]));
        for ll in [[-122.4194, 37.7749], [151.2093, -33.8688], [10.75, 59.91]] {
            let back = inverse_mercator(mercator(ll));
            assert!((back[0] - ll[0]).abs() < 1e-9 && (back[1] - ll[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn scale_doubles_at_sixty_degrees() {
        assert!((units_per_meter(60.0) - 2.0).abs() < 1e-12);
        // A metre of latitude near 60° N is two map units of y.
        let a = mercator([0.0, 60.0])[1];
        let b = mercator([0.0, 60.0 + 1.0 / 111_320.0])[1];
        assert!(((b - a) - 2.0).abs() < 0.01, "{}", b - a);
    }

    #[test]
    fn globe_round_trips_and_great_circles() {
        for ll in [
            [0.0, 0.0, 0.0],
            [-122.4, 37.8, 120.0],
            [151.2, -33.9, -5.0],
            [179.0, 89.0, 0.0],
        ] {
            let p = lon_lat_to_globe(ll, EARTH_MEAN_RADIUS);
            let back = globe_to_lon_lat(p, EARTH_MEAN_RADIUS);
            for k in 0..3 {
                assert!((back[k] - ll[k]).abs() < 1e-6, "{ll:?} -> {back:?}");
            }
        }
        assert_eq!(lon_lat_to_globe([0.0, 0.0, 0.0], 1.0), [0.0, 0.0, 1.0]);
        // A quarter of the equator is a quarter of the circumference.
        let d = great_circle_distance([0.0, 0.0], [90.0, 0.0], 1.0);
        assert!((d - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        // London to New York's midpoint lies well north of both: the great
        // circle bows toward the pole.
        let mid = great_circle_point([-0.13, 51.5], [-74.0, 40.7], 0.5);
        assert!(mid[1] > 52.0, "{mid:?}");
        let start = great_circle_point([10.0, 20.0], [30.0, 40.0], 0.0);
        assert!((start[0] - 10.0).abs() < 1e-9 && (start[1] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn tiles_nest_and_quadkeys_round_trip() {
        let t = TileId::new(3, 5, 2);
        assert_eq!(t.quadkey(), "121");
        assert_eq!(TileId::from_quadkey("121"), Some(t));
        assert_eq!(TileId::from_quadkey("4"), None);
        for c in t.children() {
            assert_eq!(c.parent(), Some(t));
            let (lo, hi) = c.bounds();
            let (plo, phi) = t.bounds();
            // Edges computed independently agree to rounding.
            let e = 1e-6;
            assert!(
                lo[0] >= plo[0] - e
                    && hi[0] <= phi[0] + e
                    && lo[1] >= plo[1] - e
                    && hi[1] <= phi[1] + e
            );
        }
        let (lo, hi) = TileId::new(0, 0, 0).bounds();
        assert_eq!((lo, hi), ([-EXTENT, -EXTENT], [EXTENT, EXTENT]));
        // San Francisco at zoom 12.
        let sf = mercator([-122.4194, 37.7749]);
        assert_eq!(TileId::containing(sf, 12), TileId::new(12, 655, 1583));
        let z = zoom_for_units_per_pixel(units_per_pixel_at_zoom(11.5, 512.0), 512.0);
        assert!((z - 11.5).abs() < 1e-9);
    }

    fn view_over(origin_xy: [f64; 2], height: f32, pitch: f32) -> super::super::MarkView {
        let origin = super::super::Origin::new(origin_xy[0], 0.0, -origin_xy[1]);
        let mut cam = crate::PerspectiveCamera::new(60.0, 1024.0 / 768.0, 1.0, 1e7);
        let mut controls = crate::MapControls::new(crate::Vector3::ZERO, height);
        controls.pitch = pitch;
        controls.apply(&mut cam);
        super::super::MarkView::new(&cam, origin, (1024, 768))
    }

    #[test]
    fn straight_down_tiles_are_one_zoom_and_cover_the_screen() {
        let sf = mercator([-122.4194, 37.7749]);
        // 512 px tiles at zoom 12 are 19.1 units a pixel; 768 px of a 60°
        // view show that from about 12,700 units up.
        let view = view_over(sf, 12_700.0, 0.0);
        let tiles = covering_tiles(&view, 18, 512.0, 0.0, 200);
        assert!(!tiles.is_empty());
        let zooms: std::collections::BTreeSet<u8> = tiles.iter().map(|t| t.z).collect();
        assert!(zooms.iter().all(|&z| (11..=13).contains(&z)), "{zooms:?}");
        // The ground under the centre and the corners is covered.
        let covered = |p: [f64; 2]| {
            tiles.iter().any(|t| {
                let (lo, hi) = t.bounds();
                p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1]
            })
        };
        let upp = view.units_per_pixel(ground_to_world(sf, 0.0)).unwrap();
        for (dx, dy) in [
            (0.0, 0.0),
            (-500.0, -370.0),
            (500.0, 370.0),
            (-500.0, 370.0),
        ] {
            let p = [sf[0] + dx * upp, sf[1] + dy * upp];
            assert!(
                covered(p),
                "screen point ({dx}, {dy}) px from the centre is uncovered"
            );
        }
    }

    #[test]
    fn a_tilted_view_gets_finer_tiles_near_and_coarser_far() {
        let sf = mercator([-122.4194, 37.7749]);
        let view = view_over(sf, 5_000.0, 1.2);
        let tiles = covering_tiles(&view, 18, 512.0, 0.0, 1000);
        let (zmin, zmax) = tiles
            .iter()
            .fold((99, 0), |(a, b), t| (a.min(t.z), b.max(t.z)));
        assert!(zmax >= zmin + 3, "a range of zooms: {zmin}..={zmax}");
        // Nearest first: the first tiles are the finest.
        assert_eq!(tiles[0].z, zmax);
        // Nothing behind the eye: the camera looks north, so no tile lies
        // wholly south of it by more than its own size.
        let eye_y = sf[1] - 5_000.0 * 1.2f64.sin();
        for t in &tiles {
            let (_, hi) = t.bounds();
            assert!(hi[1] > eye_y - t.size(), "{t:?} is behind the camera");
        }
        // The cap keeps the nearest.
        let few = covering_tiles(&view, 18, 512.0, 0.0, 5);
        assert_eq!(few, tiles[..5]);
    }

    #[test]
    fn world_mapping_is_y_up_north_away() {
        let p = lon_lat_to_world([0.0, 10.0, 0.0]);
        assert!(p[2] < 0.0, "north of the equator is towards -Z");
        assert_eq!(
            world_to_ground(ground_to_world([3.0, 4.0], 5.0)),
            ([3.0, 4.0], 5.0)
        );
    }
}
