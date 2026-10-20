//! A CAD-style quad viewport: one target split into top, front and side
//! orthographic panes plus a perspective pane.
//!
//! # Conventions
//!
//! The world is **Y-up** (three.js). The three orthographic views follow the
//! engineering-drawing (third-angle) convention, which is also what Blender's
//! numpad views give for a Y-up scene:
//!
//! | pane    | camera sits at | looks along | screen right | screen up |
//! |---------|----------------|-------------|--------------|-----------|
//! | `Top`   | +Y             | −Y          | +X           | −Z        |
//! | `Front` | +Z             | −Z          | +X           | +Y        |
//! | `Side`  | +X (right)     | −X          | −Z           | +Y        |
//!
//! The default [`layout`](QuadView::layout) is the third-angle sheet — top
//! view above the front view, right-side view beside it, perspective in the
//! spare corner:
//!
//! ```text
//! +-----------+-------------+
//! |   Top     | Perspective |
//! +-----------+-------------+
//! |   Front   |    Side     |
//! +-----------+-------------+
//! ```
//!
//! so the top and front views share their X axis column-wise, and front and
//! side share Y row-wise. All three orthographic panes use **one** scale,
//! which is what makes a feature line up across them.
//!
//! Rectangles are bottom-left origin in logical pixels — the renderer's
//! [`set_viewport`](crate::Renderer::set_viewport) convention, three.js's
//! too. Flip `y` (`height - y`) before hit-testing DOM/window mouse
//! coordinates with [`pane_at`](QuadView::pane_at).
//!
//! # Rendering
//!
//! ```no_run
//! use threers::{HeadlessRenderer, PerspectiveCamera, Scene, Vector3};
//! use threers::helpers::QuadView;
//! let mut hr = HeadlessRenderer::builder().size(800, 600).build().unwrap();
//! let mut scene = Scene::new();
//! let mut persp = PerspectiveCamera::new(45.0, 1.0, 0.1, 100.0);
//! persp.position = Vector3::new(4.0, 3.0, 5.0);
//! let mut quad = QuadView::new();
//! quad.frame_scene(&mut scene);
//! quad.render(&mut hr, &mut scene, &persp);
//! let rgba = hr.read_rgba();
//! # let _ = rgba;
//! ```
//!
//! For a window, acquire the frame once and hand all four panes the same view
//! through [`ViewSurface`] — each render submits on its own, and each pane
//! loads what the previous ones drew. Acquiring a surface texture per pane
//! does not work natively: each acquire can be a different swapchain image,
//! so the panes are split across frames and each frame shows some of them.
//!
//! In the browser `WebRenderer` deals with this itself, so a page can call
//! `render` once per pane as it would with three.js. On WebGPU the canvas
//! texture is one object for the whole animation frame. On WebGL2 wgpu hands
//! out the canvas as a fresh, uninitialised texture on every acquire, so once
//! a page uses a viewport, the scissor test, auto-clear off or `clear()`,
//! `WebRenderer` draws into a persistent canvas-sized buffer instead and
//! copies it to the canvas after each draw.
//!
//! The Metal backend takes the same `render`, through
//! `metal::MetalHeadlessRenderer` or, for a
//! window, `MetalViewSurface` (which presents the drawable once, after the
//! last pane), with the same viewport, scissor and clear semantics — see
//! `crate::metal::viewport`. `QuadView::metal_views` is the cheaper
//! single-pass alternative there, without gap colour or per-pane backgrounds.
//!
//! The renderer's per-pane rules apply — see [`crate::renderer::viewport`]:
//! MSAA works in a pane (each pane is drawn into a pane-sized multisampled
//! target and composited into its rectangle), TAA and screen-space refraction
//! do not, and post-processing applies to the finished composite only.
//!
//! # Shadows
//!
//! Shadow maps depend on lights and casters, not on the camera, so the four
//! panes share one set: [`render`](QuadView::render) has the first pane draw
//! them and the other three reuse them, through the renderer's three.js-style
//! `shadow_map.auto_update` / `needs_update` (see
//! [`ShadowMap`]). The pictures are identical
//! either way; [`share_shadows`](QuadView::share_shadows) turns it off.
//!
//! # Interaction
//!
//! [`QuadViewControls`](super::QuadViewControls) turns pointer events into
//! orbit/pan/zoom of the perspective pane, a shared pan and zoom of the
//! orthographic panes, maximise on double-click, and framing.

use crate::cameras::{Camera, OrthographicCamera, PerspectiveCamera};
use crate::core::ObjectKind;
use crate::math::{Box3, Color, Vector3};
use crate::renderer::{Renderer, ShadowMap, ViewportRect};
use crate::scene::Scene;

/// One pane of a [`QuadView`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pane {
    Top,
    Front,
    Side,
    Perspective,
}

impl Pane {
    pub const ALL: [Pane; 4] = [Pane::Top, Pane::Front, Pane::Side, Pane::Perspective];

    /// Index into per-pane arrays such as [`QuadView::backgrounds`].
    pub const fn index(self) -> usize {
        match self {
            Pane::Top => 0,
            Pane::Front => 1,
            Pane::Side => 2,
            Pane::Perspective => 3,
        }
    }

    /// The pane's name in the JavaScript, Node and Python bindings:
    /// `"top"`, `"front"`, `"side"` or `"perspective"`.
    pub const fn name(self) -> &'static str {
        match self {
            Pane::Top => "top",
            Pane::Front => "front",
            Pane::Side => "side",
            Pane::Perspective => "perspective",
        }
    }

    /// The pane called `name`, ignoring case: a [`name`](Self::name), or the
    /// aliases `"persp"` and `"right"` (the side view is the right-side view).
    pub fn from_name(name: &str) -> Option<Pane> {
        match name.to_ascii_lowercase().as_str() {
            "top" => Some(Pane::Top),
            "front" => Some(Pane::Front),
            "side" | "right" => Some(Pane::Side),
            "perspective" | "persp" => Some(Pane::Perspective),
            _ => None,
        }
    }
}

/// The three.js viewport state both renderers carry — three.js's
/// `setViewport`, `setScissor`, `setScissorTest`, `autoClear` and `getSize`,
/// with the same coordinates (bottom-left origin, logical pixels). Implemented
/// by [`Renderer`] and, with the `metal` feature,
/// `metal::MetalRenderer`; every method forwards to
/// the renderer's own method of the same name.
pub trait ViewportControl {
    /// The drawing buffer in logical pixels.
    fn size(&self) -> (f32, f32);
    fn viewport(&self) -> Option<ViewportRect>;
    fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32);
    fn reset_viewport(&mut self);
    fn scissor(&self) -> Option<ViewportRect>;
    fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32);
    fn reset_scissor(&mut self);
    fn scissor_test(&self) -> bool;
    fn set_scissor_test(&mut self, enabled: bool);
    fn auto_clear(&self) -> bool;
    fn set_auto_clear(&mut self, enabled: bool);
    /// three.js `renderer.shadowMap` — `auto_update` / `needs_update` — or
    /// `None` for a renderer that redraws its shadow maps every render.
    fn shadow_map(&mut self) -> Option<&mut ShadowMap> {
        None
    }
}

macro_rules! forward_viewport_control {
    ($ty:ty) => {
        impl ViewportControl for $ty {
            fn size(&self) -> (f32, f32) {
                <$ty>::size(self)
            }
            fn viewport(&self) -> Option<ViewportRect> {
                <$ty>::viewport(self)
            }
            fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32) {
                <$ty>::set_viewport(self, x, y, width, height)
            }
            fn reset_viewport(&mut self) {
                <$ty>::reset_viewport(self)
            }
            fn scissor(&self) -> Option<ViewportRect> {
                <$ty>::scissor(self)
            }
            fn set_scissor(&mut self, x: f32, y: f32, width: f32, height: f32) {
                <$ty>::set_scissor(self, x, y, width, height)
            }
            fn reset_scissor(&mut self) {
                <$ty>::reset_scissor(self)
            }
            fn scissor_test(&self) -> bool {
                <$ty>::scissor_test(self)
            }
            fn set_scissor_test(&mut self, enabled: bool) {
                <$ty>::set_scissor_test(self, enabled)
            }
            fn auto_clear(&self) -> bool {
                <$ty>::auto_clear(self)
            }
            fn set_auto_clear(&mut self, enabled: bool) {
                <$ty>::set_auto_clear(self, enabled)
            }
            fn shadow_map(&mut self) -> Option<&mut ShadowMap> {
                Some(<$ty>::shadow_map_mut(self))
            }
        }
    };
}

forward_viewport_control!(Renderer);
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
forward_viewport_control!(crate::metal::MetalRenderer);

/// Something the panes can be drawn into: a renderer plus the target it
/// writes. Implemented for [`HeadlessRenderer`](crate::HeadlessRenderer) and
/// for [`ViewSurface`] (a renderer and a `wgpu::TextureView`, e.g. a window
/// frame), and with the `metal` feature for
/// `metal::MetalHeadlessRenderer` and
/// `MetalViewSurface` (a Metal renderer and a pass, e.g. a window's
/// drawable).
pub trait PaneSurface {
    fn renderer(&mut self) -> &mut dyn ViewportControl;
    /// Render `scene` from `camera` with the renderer's current viewport.
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera);
    /// Colour clear honouring the scissor test ([`Renderer::clear`]).
    fn clear(&mut self, color: Color, alpha: f32);
}

/// A [`Renderer`] and the view it draws into this frame.
pub struct ViewSurface<'a> {
    pub renderer: &'a mut Renderer,
    pub view: &'a wgpu::TextureView,
    /// As for [`Renderer::render`]: true for a float (linear) target.
    pub linear_framebuffer: bool,
}

impl PaneSurface for ViewSurface<'_> {
    fn renderer(&mut self) -> &mut dyn ViewportControl {
        self.renderer
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        self.renderer
            .render(scene, camera, self.view, self.linear_framebuffer);
    }
    fn clear(&mut self, color: Color, alpha: f32) {
        self.renderer
            .clear(self.view, self.linear_framebuffer, color, alpha);
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PaneSurface for crate::HeadlessRenderer {
    fn renderer(&mut self) -> &mut dyn ViewportControl {
        crate::HeadlessRenderer::renderer(self)
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        self.render(scene, camera);
    }
    fn clear(&mut self, color: Color, alpha: f32) {
        crate::HeadlessRenderer::clear(self, color, alpha);
    }
}

/// The Metal headless renderer as a pane target. A render or clear that fails
/// is logged and the pane skipped — [`PaneSurface`] has no error channel.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl PaneSurface for crate::metal::MetalHeadlessRenderer {
    fn renderer(&mut self) -> &mut dyn ViewportControl {
        crate::metal::MetalHeadlessRenderer::renderer(self)
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        if let Err(e) = self.render(scene, camera) {
            log::warn!("quad view: metal pane render failed: {e}");
        }
    }
    fn clear(&mut self, color: Color, alpha: f32) {
        if let Err(e) = crate::metal::MetalHeadlessRenderer::clear(self, color, alpha) {
            log::warn!("quad view: metal clear failed: {e}");
        }
    }
}

/// A [`MetalRenderer`](crate::metal::MetalRenderer) and the attachments it
/// draws into this frame — the Metal [`ViewSurface`], e.g. for a window's
/// [`SurfaceFrame`](crate::metal::SurfaceFrame).
///
/// Every pane is its own render on its own command buffer, so the drawable
/// must not be presented by each of them: the surface draws with it stripped,
/// and [`present`](Self::present) presents it once at the end.
///
/// ```no_run
/// # #[cfg(all(feature = "metal", target_os = "macos"))]
/// # fn frame(surface: &threers::metal::MetalSurface, renderer: &mut threers::metal::MetalRenderer,
/// #          quad: &mut threers::helpers::QuadView, scene: &mut threers::Scene,
/// #          persp: &threers::PerspectiveCamera) {
/// use threers::helpers::MetalViewSurface;
/// if let Some(frame) = surface.next_frame() {
///     let mut target = MetalViewSurface::new(renderer, frame.attachments());
///     quad.render(&mut target, scene, persp);
///     target.present().unwrap();
/// }
/// # }
/// ```
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub struct MetalViewSurface<'a> {
    renderer: &'a mut crate::metal::MetalRenderer,
    pass: crate::metal::PassAttachments,
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl<'a> MetalViewSurface<'a> {
    /// Also tells the renderer the target's size, which
    /// [`QuadView::render`] lays the panes out by.
    pub fn new(
        renderer: &'a mut crate::metal::MetalRenderer,
        pass: crate::metal::PassAttachments,
    ) -> Self {
        let (w, h) = pass.size();
        renderer.set_drawing_buffer_size(w, h);
        Self { renderer, pass }
    }

    /// Present the pass's drawable, if it has one — after the last pane.
    pub fn present(self) -> Result<(), crate::metal::MetalError> {
        self.renderer.present(&self.pass)
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
impl PaneSurface for MetalViewSurface<'_> {
    fn renderer(&mut self) -> &mut dyn ViewportControl {
        self.renderer
    }
    fn draw(&mut self, scene: &mut Scene, camera: &dyn Camera) {
        let pass = self.pass.without_drawable();
        if let Err(e) = self.renderer.render(scene, camera, &pass) {
            log::warn!("quad view: metal pane render failed: {e}");
        }
    }
    fn clear(&mut self, color: Color, alpha: f32) {
        if let Err(e) = self.renderer.clear(&self.pass, color, alpha) {
            log::warn!("quad view: metal clear failed: {e}");
        }
    }
}

/// Top / front / side orthographic cameras and a caller's perspective camera,
/// laid out as four panes of one target. See the [module docs](self).
#[derive(Clone, Debug)]
pub struct QuadView {
    pub top: OrthographicCamera,
    pub front: OrthographicCamera,
    pub side: OrthographicCamera,
    /// Panes in grid order: `[top-left, top-right, bottom-left, bottom-right]`.
    pub layout: [Pane; 4],
    /// Space between panes, logical pixels.
    pub gap: f32,
    /// Space between the panes and the edge of the target, logical pixels.
    pub border: f32,
    /// What the gap and border are cleared to; `None` leaves them untouched.
    pub gap_color: Option<Color>,
    /// Per-pane background overriding `scene.background`, indexed by
    /// [`Pane::index`]. `None` uses the scene's.
    pub backgrounds: [Option<Color>; 4],
    /// The one pane shown full-size, CAD "maximise" — `None` for all four.
    pub maximized: Option<Pane>,
    /// Padding round the framed box in the orthographic panes, as a fraction
    /// of its extent.
    pub margin: f32,
    /// Magnification of the orthographic panes over the fitted scale — one
    /// value for all three, so they keep one scale. `1` shows the framed box
    /// with [`margin`](Self::margin); [`frame_box`](Self::frame_box) resets
    /// it.
    pub zoom: f32,
    /// Draw the shadow maps once per [`render`](Self::render) rather than
    /// once per pane. On by default; the panes are identical either way. See
    /// the [module docs](self#shadows).
    pub share_shadows: bool,
    /// What the orthographic cameras look at: the framed box's centre, moved
    /// by [`pan_by`](Self::pan_by).
    center: Vector3,
    /// The framed box's centre, which sets each camera's depth range — so a
    /// pan across one pane never pushes the part out of another's near..far.
    box_center: Vector3,
    half: Vector3,
}

impl Default for QuadView {
    fn default() -> Self {
        Self::new()
    }
}

impl QuadView {
    /// Third-angle layout, 2 px gap, framed on the unit cube round the origin.
    pub fn new() -> Self {
        let cam = || OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
        let mut q = Self {
            top: cam(),
            front: cam(),
            side: cam(),
            layout: [Pane::Top, Pane::Perspective, Pane::Front, Pane::Side],
            gap: 2.0,
            border: 0.0,
            gap_color: Some(Color::from_hex(0x303030)),
            backgrounds: [None; 4],
            maximized: None,
            margin: 0.1,
            zoom: 1.0,
            share_shadows: true,
            center: Vector3::ZERO,
            box_center: Vector3::ZERO,
            half: Vector3::new(1.0, 1.0, 1.0),
        };
        q.place_cameras();
        q
    }

    /// Show `pane` alone, or — if it already is — go back to all four.
    pub fn toggle_maximize(&mut self, pane: Pane) {
        self.maximized = if self.maximized == Some(pane) {
            None
        } else {
            Some(pane)
        };
    }

    /// The orthographic camera for `pane`; `None` for the perspective pane,
    /// which is the caller's.
    pub fn ortho_camera(&self, pane: Pane) -> Option<&OrthographicCamera> {
        match pane {
            Pane::Top => Some(&self.top),
            Pane::Front => Some(&self.front),
            Pane::Side => Some(&self.side),
            Pane::Perspective => None,
        }
    }

    pub fn ortho_camera_mut(&mut self, pane: Pane) -> Option<&mut OrthographicCamera> {
        match pane {
            Pane::Top => Some(&mut self.top),
            Pane::Front => Some(&mut self.front),
            Pane::Side => Some(&mut self.side),
            Pane::Perspective => None,
        }
    }

    /// Aim the orthographic cameras at `bounds`. Their frustum is sized to it
    /// at render time, when the panes' aspect is known. Undoes any
    /// [`zoom`](Self::zoom) and [`pan_by`](Self::pan_by).
    pub fn frame_box(&mut self, bounds: &Box3) {
        if bounds.is_empty() {
            return;
        }
        self.center = bounds.center();
        self.box_center = self.center;
        self.zoom = 1.0;
        let s = bounds.size();
        // A flat part still needs some depth to sit inside near..far.
        let eps = (s.x.max(s.y).max(s.z) * 1e-3).max(1e-4);
        self.half = Vector3::new(
            (s.x * 0.5).max(eps),
            (s.y * 0.5).max(eps),
            (s.z * 0.5).max(eps),
        );
        self.place_cameras();
    }

    /// Frame every visible mesh in `scene` (world-space bounds of their
    /// geometry). Returns the box, or `None` when there is nothing to frame.
    pub fn frame_scene(&mut self, scene: &mut Scene) -> Option<Box3> {
        let b = scene_bounds(scene)?;
        self.frame_box(&b);
        Some(b)
    }

    /// What the orthographic panes are centred on: the framed box's centre,
    /// plus any [`pan_by`](Self::pan_by).
    pub fn target(&self) -> Vector3 {
        self.center
    }

    /// Move what the orthographic panes are centred on by `delta` (world
    /// units). One point for all three, so a pan along X in the top pane is
    /// the same pan along X in the front pane. Each camera follows the part
    /// of the move across its own view; the part along its line of sight
    /// changes nothing it shows, and is left out so its depth range stays on
    /// the framed box.
    pub fn pan_by(&mut self, delta: Vector3) {
        self.center = self.center + delta;
        self.place_cameras();
    }

    /// The world directions of `pane`'s screen right and screen up — the
    /// table in the [module docs](self). `None` for the perspective pane.
    pub fn pane_axes(pane: Pane) -> Option<(Vector3, Vector3)> {
        match pane {
            Pane::Top => Some((Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.0, -1.0))),
            Pane::Front => Some((Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 1.0, 0.0))),
            Pane::Side => Some((Vector3::new(0.0, 0.0, -1.0), Vector3::new(0.0, 1.0, 0.0))),
            Pane::Perspective => None,
        }
    }

    fn place_cameras(&mut self) {
        let r = self.half.length().max(1e-4);
        // Far enough back that the whole box is in front of the camera; near
        // and far bracket it with room to spare.
        let d = 2.0 * r;
        let (pan, depth) = (self.center, self.box_center);
        let place = |cam: &mut OrthographicCamera, dir: Vector3, up: Vector3, _: Vector3| {
            // Across the view: the pan target. Along it: the framed box.
            let c = pan - dir * (pan - depth).dot(dir);
            cam.position = c + dir * d;
            cam.target = c;
            cam.up = up;
            cam.near = 0.5 * r;
            cam.far = 3.5 * r;
        };
        let c = self.center;
        place(
            &mut self.top,
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, -1.0),
            c,
        );
        place(
            &mut self.front,
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 1.0, 0.0),
            c,
        );
        place(
            &mut self.side,
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            c,
        );
    }

    /// Half extents of the framed box as seen by `pane` — (screen x, screen y).
    fn view_half(&self, pane: Pane) -> (f32, f32) {
        let h = self.half;
        match pane {
            Pane::Top => (h.x, h.z),
            Pane::Front => (h.x, h.y),
            Pane::Side => (h.z, h.y),
            Pane::Perspective => (0.0, 0.0),
        }
    }

    /// The panes shown at this size, with their rectangles.
    pub fn visible_panes(&self, width: f32, height: f32) -> Vec<(Pane, ViewportRect)> {
        self.panes(width, height).collect()
    }

    /// [`visible_panes`](Self::visible_panes) without the `Vec`, for the
    /// per-frame paths.
    fn panes(&self, width: f32, height: f32) -> impl Iterator<Item = (Pane, ViewportRect)> + '_ {
        Pane::ALL
            .into_iter()
            .filter_map(move |p| self.pane_rect(p, width, height).map(|r| (p, r)))
    }

    /// Where `pane` goes on a `width`×`height` target (logical pixels,
    /// bottom-left origin), or `None` when it is hidden by a maximised pane
    /// or not in the [`layout`](Self::layout). Pane rectangles are whole
    /// pixels for whole-pixel inputs; odd remainders go to the right column
    /// and the top row so the gap is always exactly `gap`.
    pub fn pane_rect(&self, pane: Pane, width: f32, height: f32) -> Option<ViewportRect> {
        let b = self.border.max(0.0);
        let g = self.gap.max(0.0);
        let inner_w = (width - 2.0 * b).max(0.0);
        let inner_h = (height - 2.0 * b).max(0.0);
        if let Some(m) = self.maximized {
            return (m == pane).then(|| ViewportRect::new(b, b, inner_w, inner_h));
        }
        let slot = self.layout.iter().position(|&p| p == pane)?;
        let left_w = ((inner_w - g) * 0.5).floor().max(0.0);
        let bottom_h = ((inner_h - g) * 0.5).floor().max(0.0);
        let right_x = b + left_w + g;
        let right_w = (width - b - right_x).max(0.0);
        let top_y = b + bottom_h + g;
        let top_h = (height - b - top_y).max(0.0);
        let (col_right, row_top) = (slot % 2 == 1, slot < 2);
        let (x, w) = if col_right {
            (right_x, right_w)
        } else {
            (b, left_w)
        };
        let (y, h) = if row_top {
            (top_y, top_h)
        } else {
            (b, bottom_h)
        };
        Some(ViewportRect::new(x, y, w, h))
    }

    /// The pane under `(x, y)` — bottom-left origin, logical pixels. `None`
    /// over a gap or border.
    pub fn pane_at(&self, x: f32, y: f32, width: f32, height: f32) -> Option<Pane> {
        Pane::ALL.into_iter().find(|&p| {
            self.pane_rect(p, width, height)
                .is_some_and(|r| r.contains(x, y))
        })
    }

    /// World units per logical pixel in the orthographic panes on a
    /// `width`×`height` target: the [`fit`](Self::fit) scale over
    /// [`zoom`](Self::zoom). `None` when no orthographic pane is visible.
    pub fn units_per_pixel(&self, width: f32, height: f32) -> Option<f32> {
        let pad = 1.0 + self.margin.max(0.0);
        // World units per logical pixel needed by each visible ortho pane.
        let mut upp: f32 = 0.0;
        for (p, r) in self.panes(width, height) {
            if p == Pane::Perspective || r.width <= 0.0 || r.height <= 0.0 {
                continue;
            }
            let (hx, hy) = self.view_half(p);
            upp = upp
                .max(2.0 * hx * pad / r.width)
                .max(2.0 * hy * pad / r.height);
        }
        let zoom = if self.zoom.is_finite() && self.zoom > 0.0 {
            self.zoom
        } else {
            1.0
        };
        (upp > 0.0).then_some(upp / zoom)
    }

    /// Size the orthographic frusta for this target: one world-units-per-pixel
    /// scale for all three — the largest that fits the framed box (plus
    /// [`margin`](Self::margin)) in every visible orthographic pane, over
    /// [`zoom`](Self::zoom).
    pub fn fit(&mut self, width: f32, height: f32) {
        let Some(upp) = self.units_per_pixel(width, height) else {
            return;
        };
        for p in [Pane::Top, Pane::Front, Pane::Side] {
            let Some(r) = self.pane_rect(p, width, height) else {
                continue;
            };
            if let Some(cam) = self.ortho_camera_mut(p) {
                let (hw, hh) = (0.5 * r.width * upp, 0.5 * r.height * upp);
                cam.left = -hw;
                cam.right = hw;
                cam.top = hh;
                cam.bottom = -hh;
            }
        }
    }

    /// Draw every visible pane into `surface`'s target, fitting the
    /// orthographic cameras first. The perspective pane uses a copy of
    /// `perspective` with its aspect set to the pane's. Leaves the renderer's
    /// viewport, scissor and auto-clear as it found them.
    pub fn render<S: PaneSurface + ?Sized>(
        &mut self,
        surface: &mut S,
        scene: &mut Scene,
        perspective: &PerspectiveCamera,
    ) {
        let (w, h) = surface.renderer().size();
        self.fit(w, h);
        let r = surface.renderer();
        let saved = (r.viewport(), r.scissor(), r.scissor_test(), r.auto_clear());
        r.set_auto_clear(true);
        r.set_scissor_test(true);
        // One set of shadow maps for every pane: the first pane draws them,
        // the rest reuse them. Only when the caller left auto-update on — a
        // caller managing `needs_update` itself keeps doing so.
        let shared_shadows = if self.share_shadows {
            r.shadow_map().filter(|m| m.auto_update).map(|m| {
                let saved = *m;
                m.auto_update = false;
                m.needs_update = true;
                saved
            })
        } else {
            None
        };

        if let Some(gc) = self.gap_color {
            if self.gap > 0.0 || self.border > 0.0 {
                surface.renderer().reset_scissor();
                surface.clear(gc, 1.0);
            }
        }
        for (pane, rect) in self.panes(w, h) {
            if rect.width <= 0.0 || rect.height <= 0.0 {
                continue;
            }
            let r = surface.renderer();
            r.set_viewport(rect.x, rect.y, rect.width, rect.height);
            r.set_scissor(rect.x, rect.y, rect.width, rect.height);
            let saved_bg = scene.background;
            if let Some(bg) = self.backgrounds[pane.index()] {
                scene.background = bg;
            }
            match pane {
                Pane::Perspective => {
                    let mut cam = perspective.clone();
                    cam.set_aspect(rect.aspect());
                    surface.draw(scene, &cam);
                }
                _ => {
                    let cam = self.ortho_camera(pane).cloned().expect("ortho pane");
                    surface.draw(scene, &cam);
                }
            }
            scene.background = saved_bg;
        }

        let r = surface.renderer();
        match saved.0 {
            Some(v) => r.set_viewport(v.x, v.y, v.width, v.height),
            None => r.reset_viewport(),
        }
        match saved.1 {
            Some(s) => r.set_scissor(s.x, s.y, s.width, s.height),
            None => r.reset_scissor(),
        }
        r.set_scissor_test(saved.2);
        r.set_auto_clear(saved.3);
        if let (Some(saved), Some(m)) = (shared_shadows, r.shadow_map()) {
            *m = saved;
        }
    }

    /// The panes as Metal [`RenderView`](crate::metal::RenderView)s for one
    /// [`MetalRenderer::render_views`](crate::metal::MetalRenderer::render_views)
    /// call, after [`fit`](Self::fit)ting to the target. `width`/`height` are
    /// logical pixels, `pixel_ratio` maps them to the attachment's pixels.
    ///
    /// # This or [`render`](Self::render)?
    ///
    /// On Metal, [`render`](Self::render) through
    /// [`MetalHeadlessRenderer`](crate::metal::MetalHeadlessRenderer) or
    /// [`MetalViewSurface`] behaves exactly as on wgpu: gap colour, per-pane
    /// backgrounds, one command buffer and one scene traversal per pane (plus
    /// one for the gap clear, and one more per pane for the copy out of its
    /// multisampled pane target when MSAA is on).
    ///
    /// These views draw all four panes in **one** `render_views` call — one
    /// traversal, one sort, one command buffer — multisampled straight into
    /// the target, because it is cleared once, as a whole, to
    /// `scene.background`. The
    /// price is that [`gap_color`](Self::gap_color) and
    /// [`backgrounds`](Self::backgrounds) do not apply: gaps and panes share
    /// the scene's background. Use these when the scene is heavy and plain
    /// backgrounds will do; use `render` otherwise.
    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    pub fn metal_views(
        &mut self,
        perspective: &PerspectiveCamera,
        width: f32,
        height: f32,
        pixel_ratio: f32,
    ) -> Vec<crate::metal::RenderView> {
        use crate::metal::{enums::MTLViewport, RenderView};
        self.fit(width, height);
        let pr = if pixel_ratio > 0.0 { pixel_ratio } else { 1.0 };
        let target_h = (height * pr).round() as i64;
        self.panes(width, height)
            .filter(|(_, r)| r.width > 0.0 && r.height > 0.0)
            .map(|(pane, r)| {
                let view = match pane {
                    Pane::Perspective => {
                        let mut cam = perspective.clone();
                        cam.set_aspect(r.aspect());
                        RenderView::from_camera(&cam)
                    }
                    _ => RenderView::from_camera(self.ortho_camera(pane).expect("ortho pane")),
                };
                // Rounded as the renderers round viewports; Metal's origin is
                // top-left.
                let d = r.to_device(pr);
                view.with_viewport(MTLViewport {
                    origin_x: d.x as f64,
                    origin_y: (target_h - d.y - d.h) as f64,
                    width: d.w as f64,
                    height: d.h as f64,
                    znear: 0.0,
                    zfar: 1.0,
                })
            })
            .collect()
    }
}

/// World-space bounds of every visible [`Mesh`](crate::Mesh) in `scene`
/// (other object kinds are ignored).
pub fn scene_bounds(scene: &mut Scene) -> Option<Box3> {
    scene.update_world();
    let mut total = Box3::empty();
    for (_, obj) in scene.arena.nodes.iter() {
        if !obj.visible {
            continue;
        }
        let geom = match &obj.kind {
            ObjectKind::Mesh(m) => &m.geometry,
            _ => continue,
        };
        let local = match geom.bounding_box {
            Some(b) => b,
            None => {
                let Some(points) = geom.positions() else {
                    continue;
                };
                let mut b = Box3::empty();
                for p in points {
                    b.expand_by_point(p);
                }
                b
            }
        };
        if local.is_empty() {
            continue;
        }
        total = total.union(&local.apply_matrix4(&obj.matrix_world));
    }
    (!total.is_empty()).then_some(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> QuadView {
        let mut q = QuadView::new();
        q.gap = 4.0;
        q.border = 2.0;
        q
    }

    #[test]
    fn panes_tile_the_target_with_exact_gaps() {
        let q = quad();
        let (w, h) = (801.0, 601.0);
        let tl = q.pane_rect(Pane::Top, w, h).unwrap();
        let tr = q.pane_rect(Pane::Perspective, w, h).unwrap();
        let bl = q.pane_rect(Pane::Front, w, h).unwrap();
        let br = q.pane_rect(Pane::Side, w, h).unwrap();
        // Border on every side.
        assert_eq!((bl.x, bl.y), (2.0, 2.0));
        assert_eq!(tr.x + tr.width, w - 2.0);
        assert_eq!(tr.y + tr.height, h - 2.0);
        // Gaps exactly 4 between columns and rows.
        assert_eq!(br.x - (bl.x + bl.width), 4.0);
        assert_eq!(tl.y - (bl.y + bl.height), 4.0);
        // Columns and rows line up.
        assert_eq!((tl.x, tl.width), (bl.x, bl.width));
        assert_eq!((tr.x, tr.width), (br.x, br.width));
        assert_eq!((tl.y, tl.height), (tr.y, tr.height));
        assert_eq!((bl.y, bl.height), (br.y, br.height));
        // Whole pixels; the odd one goes right / up.
        for r in [tl, tr, bl, br] {
            for v in [r.x, r.y, r.width, r.height] {
                assert_eq!(v, v.floor());
            }
        }
        assert_eq!(bl.width, 396.0);
        assert_eq!(br.width, 397.0);
        // Third-angle: top view above front view.
        assert!(tl.y > bl.y);
    }

    #[test]
    fn pane_names_round_trip() {
        for p in Pane::ALL {
            assert_eq!(Pane::from_name(p.name()), Some(p));
        }
        assert_eq!(Pane::from_name("Persp"), Some(Pane::Perspective));
        assert_eq!(Pane::from_name("right"), Some(Pane::Side));
        assert_eq!(Pane::from_name("back"), None);
    }

    #[test]
    fn maximise_shows_one_pane_and_toggles_back() {
        let mut q = quad();
        q.toggle_maximize(Pane::Front);
        let panes = q.visible_panes(400.0, 300.0);
        assert_eq!(panes.len(), 1);
        assert_eq!(
            panes[0],
            (Pane::Front, ViewportRect::new(2.0, 2.0, 396.0, 296.0))
        );
        assert!(q.pane_rect(Pane::Top, 400.0, 300.0).is_none());
        q.toggle_maximize(Pane::Front);
        assert_eq!(q.visible_panes(400.0, 300.0).len(), 4);
        // Maximising a different pane switches rather than toggles off.
        q.toggle_maximize(Pane::Top);
        q.toggle_maximize(Pane::Side);
        assert_eq!(q.maximized, Some(Pane::Side));
    }

    #[test]
    fn hit_testing_finds_panes_and_misses_gaps() {
        let q = quad();
        let (w, h) = (400.0, 300.0);
        assert_eq!(q.pane_at(10.0, 10.0, w, h), Some(Pane::Front));
        assert_eq!(q.pane_at(390.0, 290.0, w, h), Some(Pane::Perspective));
        assert_eq!(q.pane_at(10.0, 290.0, w, h), Some(Pane::Top));
        assert_eq!(q.pane_at(390.0, 10.0, w, h), Some(Pane::Side));
        let bl = q.pane_rect(Pane::Front, w, h).unwrap();
        assert_eq!(q.pane_at(bl.x + bl.width + 1.0, 10.0, w, h), None);
        assert_eq!(q.pane_at(0.5, 0.5, w, h), None, "border");
    }

    #[test]
    fn cameras_follow_the_documented_axes() {
        let mut q = QuadView::new();
        q.frame_box(&Box3::new(
            Vector3::new(-1.0, -2.0, -3.0),
            Vector3::new(1.0, 2.0, 3.0),
        ));
        q.fit(400.0, 400.0);
        // Project a point through each camera; screen right/up must be the
        // table's axes.
        let ndc = |cam: &OrthographicCamera, p: Vector3| cam.world_to_ndc(p).unwrap();
        let o = Vector3::ZERO;
        let check = |cam: &OrthographicCamera, right: Vector3, up: Vector3| {
            let c = ndc(cam, o);
            let r = ndc(cam, right);
            let u = ndc(cam, up);
            assert!(r.x > c.x + 1e-3 && (r.y - c.y).abs() < 1e-4, "right {r:?}");
            assert!(u.y > c.y + 1e-3 && (u.x - c.x).abs() < 1e-4, "up {u:?}");
        };
        check(
            &q.top,
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.0, -0.5),
        );
        check(
            &q.front,
            Vector3::new(0.5, 0.0, 0.0),
            Vector3::new(0.0, 0.5, 0.0),
        );
        check(
            &q.side,
            Vector3::new(0.0, 0.0, -0.5),
            Vector3::new(0.0, 0.5, 0.0),
        );
    }

    #[test]
    fn ortho_fit_contains_the_box_with_one_shared_scale() {
        let mut q = QuadView::new();
        q.gap = 0.0;
        q.margin = 0.0;
        let b = Box3::new(Vector3::new(-1.0, -2.0, -3.0), Vector3::new(1.0, 2.0, 3.0));
        q.frame_box(&b);
        // Panes 400×300, aspect 4:3.
        q.fit(800.0, 600.0);
        // Every corner of the box is inside every ortho frustum, in x, y and z.
        for cam in [&q.top, &q.front, &q.side] {
            for i in 0..8 {
                let p = Vector3::new(
                    if i & 1 == 0 { b.min.x } else { b.max.x },
                    if i & 2 == 0 { b.min.y } else { b.max.y },
                    if i & 4 == 0 { b.min.z } else { b.max.z },
                );
                let n = cam.world_to_ndc(p).unwrap();
                assert!(n.x.abs() <= 1.0 + 1e-4 && n.y.abs() <= 1.0 + 1e-4, "{n:?}");
                assert!((0.0..=1.0).contains(&n.z), "depth {n:?}");
            }
        }
        // One scale: same world height per pane (all panes are 300 px tall).
        let height = |c: &OrthographicCamera| c.top - c.bottom;
        assert!((height(&q.top) - height(&q.front)).abs() < 1e-5);
        assert!((height(&q.front) - height(&q.side)).abs() < 1e-5);
        // Aspect matches the pane, so nothing is stretched.
        for cam in [&q.top, &q.front, &q.side] {
            let a = (cam.right - cam.left) / (cam.top - cam.bottom);
            assert!((a - 4.0 / 3.0).abs() < 1e-5);
        }
        // Tight: the binding constraint is the top view's Z extent (6 units)
        // down 300 px — 0.02 units a pixel, against 0.015 for the side view's
        // Z across 400 — so the box touches the edge there and nowhere is it
        // fitted looser than needed.
        let top_h = q.top.top - q.top.bottom;
        assert!((top_h - 6.0).abs() < 1e-4, "{top_h}");
        let side_w = q.side.right - q.side.left;
        assert!((side_w - 8.0).abs() < 1e-4, "{side_w}");
    }

    #[test]
    fn frame_scene_finds_mesh_bounds() {
        use crate::{BasicMaterial, BoxGeometry, Material, Mesh, Object3D};
        let mut scene = Scene::new();
        let mut o = Object3D::mesh(Mesh::new(
            BoxGeometry::new(2.0, 2.0, 2.0),
            Material::Basic(BasicMaterial::new(Color::WHITE)),
        ));
        o.position = Vector3::new(10.0, 0.0, 0.0);
        scene.add(o);
        let mut q = QuadView::new();
        let b = q.frame_scene(&mut scene).unwrap();
        assert!((b.min.x - 9.0).abs() < 1e-5 && (b.max.x - 11.0).abs() < 1e-5);
        assert_eq!(q.target(), Vector3::new(10.0, 0.0, 0.0));
        assert!(QuadView::new().frame_scene(&mut Scene::new()).is_none());
    }
}
