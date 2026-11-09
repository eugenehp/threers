//! Longitude and latitude to the flat map the marks draw on: everything in
//! [`threers_math::geo`], and [`covering_tiles`], which works from a
//! [`MarkView`](super::MarkView).

pub use threers_math::geo::*;

/// The tiles to draw for `view`: the tile pyramid walked from zoom 0, each
/// tile dropped when the view frustum misses it and split while it covers
/// more than `tile_px` pixels, up to `max_zoom`.
///
/// So a tilted view gets fine tiles near the camera and coarse ones towards
/// the horizon, as map renderers do; ground is the plane at `height` (world
/// Y) in Web Mercator. Sorted nearest first. At most `max_tiles` are
/// returned, the farthest dropped. Works from a
/// [`MarkView`](super::MarkView).
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
