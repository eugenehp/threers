//! A field's ⌘/Ctrl+C reaches the system clipboard and ⌘/Ctrl+V reads it.
//!
//! Ignored by default: it uses the real clipboard (saving and restoring what
//! was on it). `cargo test --features clipboard --test system_clipboard -- --ignored`.

use threers::ui::system_clipboard::{install, SystemClipboard};
use threers::ui::text_shortcuts::{set_platform, Clipboard, Platform, TextCommand};
use threers::ui::TextField;

#[test]
#[ignore = "uses the system clipboard"]
fn copy_and_paste_go_through_the_system_clipboard() {
    let mut system = match SystemClipboard::new() {
        Ok(c) => c,
        Err(e) => return eprintln!("skipping: {e}"),
    };
    let saved = system.read();
    install().expect("a clipboard");
    set_platform(Platform::Mac);
    let mut a = TextField::selected("threers clipboard probe 7d1c");
    a.command(TextCommand::Copy, |_| true);
    // Another reader of the system clipboard sees it.
    assert_eq!(system.read().as_deref(), Some("threers clipboard probe 7d1c"));
    system.write("pasted from elsewhere 41b9");
    let mut b = TextField::new("");
    b.command(TextCommand::Paste, |_| true);
    assert_eq!(b.text, "pasted from elsewhere 41b9");
    if let Some(saved) = saved {
        system.write(&saved);
    }
}
