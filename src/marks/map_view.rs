//! One object for the parts of a map view that have to agree every frame.

use super::{geo, CollisionIndex, MarkView, Origin, SharedCollision};
use crate::cameras::PerspectiveCamera;
use crate::controls::{MapControls, PointerEvent};
use crate::math::Vector3;

/// A map's camera, controls, precision origin and collision index, kept in
/// step.
///
/// A map view has several things that must agree each frame: the controls
/// move the camera; the camera's pose is in coordinates relative to an
/// [`Origin`] that has to follow the view as it travels, or `f32` runs out;
/// the clip planes have to fit the tilt; the [`MarkView`] has to be built from
/// all of that; and a [`CollisionIndex`] shared by the label layers has to
/// start the frame. `MapView` does them in order:
///
/// ```no_run
/// # use threers::marks::{MapView, geo};
/// # use threers::controls::PointerEvent;
/// # let (ev, cursor) = (PointerEvent::default(), (0.0, 0.0));
/// let mut map = MapView::new(geo::lon_lat_to_world([-122.42, 37.77, 0.0]), 5_000.0, (1280, 720));
/// // Per frame:
/// map.handle(ev, cursor);
/// let frame = map.begin_frame();
/// // `frame.shift`: add it to meshes placed relative to the origin.
/// // Render the scene with `map.camera`, then draw marks with `frame.view`.
/// ```
///
/// Coordinates: Y up, the ground the plane `y = ground` in absolute world
/// units — Web Mercator metres with [`geo`]'s helpers.
#[derive(Debug, Clone)]
pub struct MapView {
    /// The controls, in coordinates relative to [`origin`](Self::origin).
    pub controls: MapControls,
    /// The camera, relative to [`origin`](Self::origin) like the controls.
    pub camera: PerspectiveCamera,
    pub origin: Origin,
    /// Rebase once the look-at point strays more than this many view
    /// distances from the origin. `f32` keeps about seven digits, so at 100
    /// the camera's position is still good to a thousandth of a pixel.
    pub rebase_distances: f32,
    /// Started by [`begin_frame`](Self::begin_frame); hand clones to label and
    /// sprite layers' `collision_index`.
    pub collision: SharedCollision,
    viewport: (u32, u32),
}

/// What [`MapView::begin_frame`] hands back.
#[derive(Debug, Clone)]
pub struct MapFrame {
    pub view: MarkView,
    /// How far the origin moved this frame, to add to anything placed
    /// relative to it (meshes beside the marks). Zero most frames.
    pub shift: Vector3,
}

impl MapView {
    /// Looking straight down at absolute point `target` from `distance` away,
    /// with a 45° vertical field of view.
    pub fn new(target: [f64; 3], distance: f32, viewport: (u32, u32)) -> Self {
        let origin = Origin::new(target[0], target[1], target[2]);
        let mut camera = PerspectiveCamera::new(
            45.0,
            viewport.0.max(1) as f32 / viewport.1.max(1) as f32,
            1.0,
            10.0,
        );
        let controls = MapControls::new(Vector3::new(0.0, 0.0, 0.0), distance);
        controls.apply(&mut camera);
        controls.fit_clip_planes(&mut camera);
        Self {
            controls,
            camera,
            origin,
            rebase_distances: 100.0,
            collision: CollisionIndex::new().shared(),
            viewport: (viewport.0.max(1), viewport.1.max(1)),
        }
    }

    pub fn viewport(&self) -> (u32, u32) {
        self.viewport
    }

    /// The drawing size changed.
    pub fn set_viewport(&mut self, width: u32, height: u32) {
        self.viewport = (width.max(1), height.max(1));
        self.camera.aspect = self.viewport.0 as f32 / self.viewport.1 as f32;
    }

    /// One frame of pointer input; `cursor` in pixels from the top left.
    pub fn handle(&mut self, ev: PointerEvent, cursor: (f32, f32)) {
        let vp = (self.viewport.0 as f32, self.viewport.1 as f32);
        self.controls.update(ev, cursor, &mut self.camera, vp);
    }

    /// The look-at point, absolute.
    pub fn target(&self) -> [f64; 3] {
        self.origin.to_world(self.controls.target)
    }

    /// Look at absolute point `target` from `distance` away, keeping bearing
    /// and pitch. Moves the origin there too.
    pub fn look_at(&mut self, target: [f64; 3], distance: f32) -> Vector3 {
        let shift = self.origin.rebase(target);
        self.controls.target = Vector3::new(0.0, 0.0, 0.0);
        self.controls.ground += shift.y;
        self.controls.distance =
            distance.clamp(self.controls.min_distance, self.controls.max_distance);
        self.sync_camera();
        shift
    }

    /// Where the ray through pixel `(x, y)` meets the ground, absolute.
    pub fn ground_point(&self, x: f32, y: f32) -> Option<[f64; 3]> {
        let vp = (self.viewport.0 as f32, self.viewport.1 as f32);
        self.controls
            .ground_point(&self.camera, x, y, vp)
            .map(|p| self.origin.to_world(p))
    }

    /// Move the origin under the look-at point if it has strayed past
    /// [`rebase_distances`](Self::rebase_distances); the shift, or zero.
    pub fn rebase_if_needed(&mut self) -> Vector3 {
        let t = self.controls.target;
        let reach = (t.x * t.x + t.y * t.y + t.z * t.z).sqrt();
        if reach <= self.controls.distance * self.rebase_distances {
            return Vector3::new(0.0, 0.0, 0.0);
        }
        // Along the ground only: the origin's height stays where the ground
        // is measured from.
        let w = self.target();
        let shift = self.origin.rebase([w[0], self.origin.position[1], w[2]]);
        self.controls.target = self.controls.target + shift;
        self.controls.ground += shift.y;
        self.sync_camera();
        shift
    }

    fn sync_camera(&mut self) {
        self.controls.apply(&mut self.camera);
        self.controls.fit_clip_planes(&mut self.camera);
    }

    /// The marks' view of this frame, without touching the collision index.
    pub fn mark_view(&self) -> MarkView {
        MarkView::new(&self.camera, self.origin, self.viewport)
    }

    /// Rebase if needed, place the camera and fit its planes, build the
    /// frame's [`MarkView`] and begin the collision frame.
    pub fn begin_frame(&mut self) -> MapFrame {
        let shift = self.rebase_if_needed();
        self.sync_camera();
        let view = self.mark_view();
        self.collision.lock().unwrap().begin_frame(&view);
        MapFrame { view, shift }
    }

    /// [`begin_frame`](Self::begin_frame) for marks drawn over `renderer`'s
    /// frames: its drawing size, tone mapping and depth direction.
    pub fn begin_frame_for(&mut self, renderer: &crate::Renderer) -> MapFrame {
        let (w, h) = renderer.drawing_buffer_size();
        if (w, h) != self.viewport {
            self.set_viewport(w, h);
        }
        let shift = self.rebase_if_needed();
        self.sync_camera();
        let view = MarkView::from_renderer(&self.camera, self.origin, renderer);
        self.collision.lock().unwrap().begin_frame(&view);
        MapFrame { view, shift }
    }

    /// The Web Mercator tiles this frame shows; see [`geo::covering_tiles`].
    pub fn covering_tiles(
        &self,
        view: &MarkView,
        max_zoom: u8,
        tile_px: f64,
        max_tiles: usize,
    ) -> Vec<geo::TileId> {
        let ground = self.origin.position[1] + self.controls.ground as f64;
        geo::covering_tiles(view, max_zoom, tile_px, ground, max_tiles)
    }

    /// The zoom level whose `tile_px` tiles match the scale at the look-at
    /// point.
    pub fn zoom(&self, tile_px: f64) -> f64 {
        let upp = self
            .mark_view()
            .units_per_pixel(self.target())
            .unwrap_or(1.0);
        geo::zoom_for_units_per_pixel(upp, tile_px)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pan of a hundred kilometres in small steps, far from zero: the
    /// origin follows, and the absolute look-at point is where the steps
    /// add up to.
    #[test]
    fn the_origin_follows_a_long_pan() {
        let start = [-13_627_000.0, 0.0, 4_548_000.0];
        let mut map = MapView::new(start, 500.0, (800, 600));
        let first = map.origin.position;
        let mut shifts = 0;
        for _ in 0..1000 {
            map.controls.target.x += 100.0;
            let f = map.begin_frame();
            if f.shift.x != 0.0 {
                shifts += 1;
            }
        }
        assert!(shifts > 0, "rebased along the way");
        assert_ne!(map.origin.position, first);
        let t = map.target();
        assert!((t[0] - (start[0] + 100_000.0)).abs() < 1e-3, "{t:?}");
        assert!(map.controls.target.x.abs() <= 500.0 * map.rebase_distances);
    }

    /// The look-at point is in the middle of the frame, at the ground.
    #[test]
    fn the_target_projects_to_the_centre() {
        let target = [2_000_000.0, 0.0, -1_000_000.0];
        let mut map = MapView::new(target, 1000.0, (640, 480));
        map.controls.pitch = 0.9;
        map.controls.bearing = 0.4;
        let f = map.begin_frame();
        let p = f.view.project(target).unwrap();
        assert!(
            (p[0] - 320.0).abs() < 0.01 && (p[1] - 240.0).abs() < 0.01,
            "{p:?}"
        );
        let g = map.ground_point(320.0, 240.0).unwrap();
        assert!((g[0] - target[0]).abs() < 0.01 && (g[2] - target[2]).abs() < 0.01);
    }

    /// Begun each frame with a still view, the collision index keeps its
    /// boxes; moved, it starts over.
    #[test]
    fn a_still_frame_keeps_the_collision_boxes() {
        let mut map = MapView::new([0.0; 3], 1000.0, (640, 480));
        map.begin_frame();
        let generation = map.collision.lock().unwrap().generation();
        map.begin_frame();
        assert_eq!(map.collision.lock().unwrap().generation(), generation);
        map.controls.distance = 900.0;
        map.begin_frame();
        assert_ne!(map.collision.lock().unwrap().generation(), generation);
    }
}
