//! The system clipboard for text inputs: what ⌘/Ctrl+C and ⌘/Ctrl+X put
//! there other apps can paste, and ⌘/Ctrl+V pastes what they copied. macOS,
//! Windows and Linux (X11, and Wayland through XWayland), with the
//! `clipboard` feature.
//!
//! ```no_run
//! // Once, on the thread that runs the UI.
//! threers::ui::system_clipboard::install().expect("a clipboard");
//! ```
//!
//! Without it, [`text_shortcuts`](crate::ui::text_shortcuts) copies and
//! pastes through a clipboard in memory, between the app's own fields.

use crate::ui::text_shortcuts::{set_clipboard, Clipboard};

/// The system clipboard, as a [`Clipboard`].
pub struct SystemClipboard(arboard::Clipboard);

impl SystemClipboard {
    pub fn new() -> Result<Self, String> {
        arboard::Clipboard::new()
            .map(Self)
            .map_err(|e| format!("the system clipboard: {e}"))
    }
}

impl Clipboard for SystemClipboard {
    fn read(&mut self) -> Option<String> {
        self.0.get_text().ok()
    }

    fn write(&mut self, text: &str) {
        if let Err(e) = self.0.set_text(text) {
            log::warn!("copying to the system clipboard: {e}");
        }
    }
}

/// Text inputs on this thread copy to and paste from the system clipboard
/// from now on.
pub fn install() -> Result<(), String> {
    set_clipboard(SystemClipboard::new()?);
    Ok(())
}
