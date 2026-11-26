//! A window resized without its picture trailing its frame.

use winit::window::Window;

/// Has what is drawn in `window` shown together with the window's own
/// changes (on a Mac: presented in the Core Animation transaction that
/// resizes it), so a resize under way neither stretches the last picture
/// nor shows the next a frame late. Call it once the surface is made; frames
/// have to be presented on the main thread, where the window is. Nothing
/// elsewhere.
pub fn in_step(window: &Window) {
    #[cfg(target_os = "macos")]
    mac::in_step(window);
    #[cfg(not(target_os = "macos"))]
    let _ = window;
}

#[cfg(target_os = "macos")]
mod mac {
    use objc2::msg_send;
    use objc2::runtime::{AnyClass, AnyObject};
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use winit::window::Window;

    pub(super) fn in_step(window: &Window) {
        let Ok(handle) = window.window_handle() else {
            return;
        };
        let RawWindowHandle::AppKit(h) = handle.as_raw() else {
            return;
        };
        let Some(metal) = AnyClass::get(c"CAMetalLayer") else {
            return;
        };
        let view = h.ns_view.as_ptr().cast::<AnyObject>();
        // SAFETY: the view is the live window's, its handle given out on the
        // main thread; its layer and theirs are checked for nil.
        unsafe {
            let layer: *mut AnyObject = msg_send![view, layer];
            if !layer.is_null() {
                set(layer, metal, 3);
            }
        }
    }

    /// The Metal layers at `layer` and under it.
    unsafe fn set(layer: *mut AnyObject, metal: &AnyClass, depth: usize) {
        // SAFETY: `layer` is a live CALayer, as each of its sublayers is.
        unsafe {
            let is: bool = msg_send![layer, isKindOfClass: metal];
            if is {
                let _: () = msg_send![layer, setPresentsWithTransaction: true];
            }
            if depth == 0 {
                return;
            }
            let under: *mut AnyObject = msg_send![layer, sublayers];
            if under.is_null() {
                return;
            }
            let n: usize = msg_send![under, count];
            for i in 0..n {
                let l: *mut AnyObject = msg_send![under, objectAtIndex: i];
                set(l, metal, depth - 1);
            }
        }
    }
}
