//! A UI box showing a 3D view drawn on another thread: a surface node fed by
//! [`ViewFrames`], so the UI composites the view's newest frame at its own
//! rate and never waits for the scene to draw.
//!
//! [`SurfaceView::update`] does the UI side of a frame. Call it after layout
//! and before drawing. It sizes the view to the box (times the device pixel
//! ratio), tells the producer whether the box can be seen at all (a box
//! scrolled away, clipped off or under `display: none` is not drawn), and
//! hands the newest frame to the [`UiPass`]. The producer side is a
//! [`ViewProducer`](crate::renderer::ViewProducer) on the same device, usually
//! run by a [`RenderThread`](crate::renderer::RenderThread).
//!
//! ```ignore
//! let gpu = window.shared_gpu()?;
//! let (producer, frames) = ViewProducer::on(&gpu, ViewOptions::default());
//! let mut view = SurfaceView::new(&mut ui, root, style! { flex-grow: 1; }, frames);
//! let render = RenderThread::spawn(RenderLoop::new(producer, world, draw));
//! // Each UI frame:
//! ui.layout(width, height, &mut pass.text_measure(scale));
//! view.update(&ui, &mut pass, scale);
//! window.present_with(|device, queue, target, size, format| {
//!     pass.draw_retained(device, queue, &ui.scene(), scale, target, size, format, clear);
//! });
//! ```

use crate::renderer::{UiPass, ViewFrame, ViewFrames, ViewStats};
use crate::ui::{Display, NodeId, Rect, Style, Ui};

/// Where a point in the UI falls in a [`SurfaceView`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewPoint {
    /// 0..1 across the box, from its top-left corner.
    pub uv: [f32; 2],
    /// -1..1, y up: three.js's normalised device coordinates, for
    /// [`Raycaster::set_from_camera`](crate::core::Raycaster) on the render
    /// thread.
    pub ndc: [f32; 2],
}

/// A surface node showing a [`ViewFrames`]' newest frame.
pub struct SurfaceView {
    pub node: NodeId,
    /// The surface id the node and the [`UiPass`] share.
    pub id: u64,
    frames: ViewFrames,
    shown: Option<u64>,
}

/// Surface ids for views, apart from any an app picks itself.
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1 << 62);

impl SurfaceView {
    /// A surface box under `parent`, styled by `style` (give it a size or let
    /// it flex), showing what `frames`' producer draws. Until the first frame
    /// the box shows its own background.
    #[track_caller]
    pub fn new(ui: &mut Ui, parent: NodeId, style: Style, frames: ViewFrames) -> Self {
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let node = ui.add_surface(parent, style, id, (1, 1));
        Self {
            node,
            id,
            frames,
            shown: None,
        }
    }

    /// The consumer half, for [`request_redraw`](ViewFrames::request_redraw),
    /// [`set_priority`](ViewFrames::set_priority) and the rest.
    pub fn frames(&self) -> &ViewFrames {
        &self.frames
    }

    /// Whether the first frame has arrived. Until it has, show something in
    /// its place if the wait is long (a scene still uploading).
    pub fn is_ready(&self) -> bool {
        self.frames.has_frame()
    }

    /// How the view is doing; see [`ViewStats`].
    pub fn stats(&self) -> ViewStats {
        self.frames.stats()
    }

    /// Why the view stopped drawing, when its producer failed (a panic on
    /// the render thread). Worth showing: otherwise the box keeps the last
    /// frame and looks merely frozen.
    pub fn failure(&self) -> Option<String> {
        self.frames.failure()
    }

    /// The box's size in device pixels at `scale`, as laid out now.
    pub fn pixel_size(&self, ui: &Ui, scale: f32) -> (u32, u32) {
        let r = ui.rect(self.node);
        (
            ((r.width * scale).round() as u32).max(1),
            ((r.height * scale).round() as u32).max(1),
        )
    }

    /// The part of the box that can be seen, in UI pixels: its rectangle cut
    /// by the window and every ancestor that clips or scrolls. `None` when
    /// nothing of it can be, or it or an ancestor has `display: none` or no
    /// opacity.
    pub fn visible_rect(&self, ui: &Ui) -> Option<Rect> {
        let mut r = ui.rect(self.node);
        let hidden = |s: &Style| s.display == Display::None || s.opacity <= 0.0;
        // The box itself, then its ancestors.
        for a in ui.ancestors(self.node) {
            let s = ui.style(a);
            if hidden(s) {
                return None;
            }
            if s.clip || s.scroll || s.scroll_x || ui.parent(a).is_none() {
                r = intersect(r, ui.rect(a))?;
            }
        }
        Some(r)
    }

    /// Where the UI point `(x, y)` (in UI pixels, as pointer events give it)
    /// falls in the view, or `None` when it is outside the box or something
    /// drawn over the box is there. Send it to the render thread for
    /// orbiting or picking.
    pub fn point(&self, ui: &Ui, x: f32, y: f32) -> Option<ViewPoint> {
        if ui.hit_test(x, y) != Some(self.node) {
            return None;
        }
        let r = ui.rect(self.node);
        if r.width <= 0.0 || r.height <= 0.0 {
            return None;
        }
        let uv = [(x - r.x) / r.width, (y - r.y) / r.height];
        Some(ViewPoint {
            uv,
            ndc: [uv[0] * 2.0 - 1.0, 1.0 - uv[1] * 2.0],
        })
    }

    /// The UI side of a frame. Call it after layout and before drawing: it asks
    /// the producer for the box's size at `scale`, says whether the box can be
    /// seen (see [`visible_rect`](Self::visible_rect)), and gives `pass` the
    /// newest frame. Returns that frame, or `None` before the first.
    pub fn update(&mut self, ui: &Ui, pass: &mut UiPass, scale: f32) -> Option<ViewFrame> {
        let (w, h) = self.pixel_size(ui, scale);
        self.frames.set_size(w, h);
        self.frames.set_visible(self.visible_rect(ui).is_some());
        let frame = self.frames.latest()?;
        if frame.new || self.shown.is_none() {
            pass.set_surface(self.id, frame.view.clone());
            self.shown = Some(frame.frame);
        }
        Some(frame)
    }
}

fn intersect(a: Rect, b: Rect) -> Option<Rect> {
    let x = a.x.max(b.x);
    let y = a.y.max(b.y);
    let right = (a.x + a.width).min(b.x + b.width);
    let bottom = (a.y + a.height).min(b.y + b.height);
    (right > x && bottom > y).then_some(Rect {
        x,
        y,
        width: right - x,
        height: bottom - y,
    })
}
