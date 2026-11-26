//! A window with no title bar of its own to see: on a Mac its contents
//! reach the top, the close, minimise and zoom buttons over them, and the
//! app's own bar moves the window. Elsewhere the system's bar stays. Either
//! way the buttons are the system's own, on its side and in its style.
//!
//! With the UI, [`TitleBar::on_window_event`], [`TitleBar::fit`] and
//! [`TitleBar::press_at`] do all of it: one call for every window event,
//! the bar padded round the buttons, a press moving the window.
//!
//! ```no_run
//! # use threers::renderer::titlebar::{self, TitleBar};
//! # fn demo(w: winit::window::Window, ui: &threers::ui::Ui, bar: threers::ui::NodeId) {
//! let attributes = titlebar::invisible(winit::window::Window::default_attributes());
//! # let _ = attributes;
//! let mut title_bar = TitleBar::new(44.0);
//! // Once there is a window, and on `Resized`, `ThemeChanged` and `Focused`:
//! title_bar.place_buttons(&w);
//! // On a press nothing of the app's took:
//! if threers::ui::window_drag::grabs(ui, bar, 120.0, 10.0) {
//!     title_bar.press(&w);
//! }
//! # }
//! ```

use std::time::{Duration, Instant};

use winit::window::{Window, WindowAttributes};

/// What the window's buttons cover of the app's bar, in logical pixels, for
/// the bar to leave free there.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Room {
    pub left: f32,
    pub right: f32,
}

/// The room the buttons of `window` take: none where the system draws them
/// in a bar of its own, nor in full screen, where they are away. They are
/// on the left, on a Mac that reads right to left on the right.
pub fn room(window: &Window) -> Room {
    #[cfg(target_os = "macos")]
    if window.fullscreen().is_none() {
        return if mac::right_to_left(window) {
            Room {
                left: 0.0,
                right: mac::WIDTH,
            }
        } else {
            Room {
                left: mac::WIDTH,
                right: 0.0,
            }
        };
    }
    let _ = window;
    Room::default()
}

/// `attributes` of a window whose contents go under its title bar.
pub fn invisible(attributes: WindowAttributes) -> WindowAttributes {
    #[cfg(target_os = "macos")]
    {
        use winit::platform::macos::WindowAttributesExtMacOS;
        attributes
            .with_titlebar_transparent(true)
            .with_title_hidden(true)
            .with_fullsize_content_view(true)
    }
    #[cfg(not(target_os = "macos"))]
    attributes
}

/// The app's bar in the title bar's place.
#[derive(Debug)]
pub struct TitleBar {
    height: f32,
    pressed: Option<Instant>,
}

impl TitleBar {
    /// One `height` logical pixels tall.
    pub fn new(height: f32) -> Self {
        Self {
            height,
            pressed: None,
        }
    }

    /// Centres the window's buttons in the bar (AppKit puts them back where
    /// they were when it lays the window out: call it again then). In full
    /// screen they are the system's.
    pub fn place_buttons(&self, window: &Window) {
        #[cfg(target_os = "macos")]
        if window.fullscreen().is_none() {
            mac::place_buttons(window, f64::from(self.height));
        }
        #[cfg(not(target_os = "macos"))]
        let _ = (window, self.height);
    }

    /// The bar pressed where nothing of the app's is: moves the window with
    /// the pointer, or on a second press soon after zooms it, and back.
    pub fn press(&mut self, window: &Window) {
        let now = Instant::now();
        let again = self
            .pressed
            .replace(now)
            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(400));
        if again {
            self.pressed = None;
            window.set_maximized(!window.is_maximized());
        } else if let Err(e) = window.drag_window() {
            log::debug!("the window is not moved: {e}");
        }
    }
}

impl TitleBar {
    /// What a window event means for the bar: AppKit puts the buttons back
    /// where it keeps them when it lays the window out (a resize, a change
    /// of theme, of focus, of scale, full screen), so they are centred in
    /// the bar again. Whether the room they take may have changed: fit the
    /// bar again then ([`fit`](Self::fit)).
    ///
    /// One call for every event, in place of a call in each of five arms.
    pub fn on_window_event(&self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        use winit::event::WindowEvent as E;
        match event {
            E::Resized(_) | E::ScaleFactorChanged { .. } => {
                self.place_buttons(window);
                true
            }
            E::ThemeChanged(_) | E::Focused(true) | E::Occluded(false) => {
                self.place_buttons(window);
                false
            }
            _ => false,
        }
    }
}

/// With the UI: the bar laid out round the buttons, and pressed.
#[cfg(feature = "ui")]
impl TitleBar {
    /// Pads `bar` (a row in `ui`) so that nothing of it is under the
    /// window's buttons: `room` of them, on their side, or `least` where
    /// there are none, mirrored in a layout read right to left. `zoom`: the
    /// UI's own scale over the window's (1 as most have it).
    pub fn fit(&self, window: &Window, ui: &mut crate::ui::Ui, bar: crate::ui::NodeId, zoom: f32, least: f32) {
        self.place_buttons(window);
        let r = room(window);
        let (left, right) = ((r.left / zoom).max(least), (r.right / zoom).max(least));
        let mirrored = ui.layout_direction() == crate::ui::environment::LayoutDirection::RightToLeft;
        let (start, end) = if mirrored { (right, left) } else { (left, right) };
        ui.update_style(bar, |s| {
            s.padding.left = start;
            s.padding.right = end;
        });
    }

    /// A press at (`x`, `y`) (logical pixels, in `ui`): on `bar` and on
    /// nothing in it that takes presses (see
    /// [`window_drag::grabs`](crate::ui::window_drag::grabs)) it moves the
    /// window, or zooms it on a second press. Whether it took the press.
    pub fn press_at(&mut self, window: &Window, ui: &crate::ui::Ui, bar: crate::ui::NodeId, x: f32, y: f32) -> bool {
        let grabs = crate::ui::window_drag::grabs(ui, bar, x, y);
        if grabs {
            self.press(window);
        }
        grabs
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use objc2::encode::{Encode, Encoding};
    use objc2::msg_send;
    use objc2::runtime::AnyObject;
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::Window;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Point {
        x: f64,
        y: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Size {
        width: f64,
        height: f64,
    }
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Rect {
        origin: Point,
        size: Size,
    }
    // SAFETY: CGPoint, CGSize and CGRect, field for field.
    unsafe impl Encode for Point {
        const ENCODING: Encoding = Encoding::Struct("CGPoint", &[f64::ENCODING, f64::ENCODING]);
    }
    // SAFETY: as above.
    unsafe impl Encode for Size {
        const ENCODING: Encoding = Encoding::Struct("CGSize", &[f64::ENCODING, f64::ENCODING]);
    }
    // SAFETY: as above.
    unsafe impl Encode for Rect {
        const ENCODING: Encoding = Encoding::Struct("CGRect", &[Point::ENCODING, Size::ENCODING]);
    }

    /// From the window's edge to the first button, as in a window with a
    /// toolbar.
    const INSET: f64 = 14.0;
    /// What the three of them and the space round them take.
    pub(super) const WIDTH: f32 = 88.0;

    /// The window and its close, minimise and zoom buttons.
    fn parts(window: &Window) -> Option<(*mut AnyObject, [*mut AnyObject; 3])> {
        let handle = window.window_handle().ok()?;
        let RawWindowHandle::AppKit(h) = handle.as_raw() else {
            return None;
        };
        let view = h.ns_view.as_ptr().cast::<AnyObject>();
        // SAFETY: the view is the live window's, and winit hands its handle
        // out on the main thread only; each object is checked for nil.
        unsafe {
            let win: *mut AnyObject = msg_send![view, window];
            if win.is_null() {
                return None;
            }
            let buttons: [*mut AnyObject; 3] =
                [0usize, 1, 2].map(|i| msg_send![win, standardWindowButton: i]);
            buttons
                .iter()
                .all(|b| !b.is_null())
                .then_some((win, buttons))
        }
    }

    /// Whether AppKit has them from the right, as on a Mac that reads so.
    pub(super) fn right_to_left(window: &Window) -> bool {
        parts(window).is_some_and(|(_, b)| {
            // SAFETY: two live buttons, on the main thread (see `parts`).
            let [first, second] = [b[0], b[1]].map(|b| unsafe {
                let r: Rect = msg_send![b, frame];
                r
            });
            first.origin.x > second.origin.x
        })
    }

    pub(super) fn place_buttons(window: &Window, height: f64) {
        let Some((win, buttons)) = parts(window) else {
            return;
        };
        // SAFETY: live objects, on the main thread (see `parts`); each
        // further one is checked for nil.
        unsafe {
            let bar: *mut AnyObject = msg_send![buttons[0], superview];
            if bar.is_null() {
                return;
            }
            let holder: *mut AnyObject = msg_send![bar, superview];
            if holder.is_null() {
                return;
            }
            // The title bar as tall as the app's, at the window's top; then
            // its buttons in the middle of that.
            let frame: Rect = msg_send![win, frame];
            let mut r: Rect = msg_send![holder, frame];
            r.size.height = height;
            r.origin.y = frame.size.height - height;
            let _: () = msg_send![holder, setFrame: r];
            let first: Rect = msg_send![buttons[0], frame];
            let second: Rect = msg_send![buttons[1], frame];
            let step = second.origin.x - first.origin.x;
            // From the side AppKit has them on.
            let start = if step < 0.0 {
                frame.size.width - INSET - first.size.width
            } else {
                INSET
            };
            for (i, b) in buttons.into_iter().enumerate() {
                let at: Rect = msg_send![b, frame];
                let to = Point {
                    x: start + step * i as f64,
                    y: ((height - at.size.height) / 2.0).round(),
                };
                let _: () = msg_send![b, setFrameOrigin: to];
                log::debug!(
                    "window button {i}: {} × {} at {}, {} of {} × {height}",
                    at.size.width,
                    at.size.height,
                    to.x,
                    to.y,
                    frame.size.width
                );
            }
        }
    }
}
