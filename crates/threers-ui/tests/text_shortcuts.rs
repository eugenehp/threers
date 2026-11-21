//! Text inputs answer to their platform's shortcuts — select all, copy, cut,
//! paste, undo, redo, by word and by line — ahead of an app's bindings of the
//! same keys, which still work outside fields and for whatever a field
//! leaves (formatting in a plain field, undo with nothing left to undo).

use threers_ui::input::Input;
use threers_ui::text_shortcuts::{set_clipboard, set_platform, MemoryClipboard, Platform};
use threers_ui::undo::{bind_undo_keys, Redo, Undo};
use threers_ui::{style, FieldKey, FixedWidthText, NodeId, Role, Semantics, TextField, Ui};

#[derive(Debug, Clone)]
struct SelectEverything;
#[derive(Debug, Clone)]
struct ToggleSidebar;

#[derive(Default)]
struct App {
    a: TextField,
    b: TextField,
    log: Vec<&'static str>,
}

struct Page {
    ui: Ui,
    input: Input<App>,
    app: App,
    a: NodeId,
    b: NodeId,
    button: NodeId,
}

fn page() -> Page {
    let mut ui = Ui::new();
    let root = ui.root();
    let field = |ui: &mut Ui| {
        let f = ui.add_box(root, style! { height: 20px; });
        ui.set_semantics(f, Semantics::new(Role::TextBox));
        f
    };
    let (a, b) = (field(&mut ui), field(&mut ui));
    let button = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(button, Semantics::new(Role::Button).label("Go"));
    ui.layout(200.0, 200.0, &mut FixedWidthText(7.0));
    let mut input: Input<App> = Input::new();
    input.on_key(a, |s: &mut App, _, key, m, _| {
        !matches!(s.a.key(key, m, |_| true), FieldKey::Ignored | FieldKey::Tab { .. })
    });
    input.on_key(b, |s: &mut App, _, key, m, _| {
        !matches!(s.b.key(key, m, |_| true), FieldKey::Ignored | FieldKey::Tab { .. })
    });
    // The app's own shortcuts on the same keys.
    input.keymap.bind("secondary-a", SelectEverything, None).unwrap();
    input.keymap.bind("secondary-b", ToggleSidebar, None).unwrap();
    bind_undo_keys(&mut input.keymap);
    input.on_action(root, |s: &mut App, _, _: &SelectEverything, _| s.log.push("select everything"));
    input.on_action(root, |s: &mut App, _, _: &ToggleSidebar, _| s.log.push("toggle sidebar"));
    input.on_action(root, |s: &mut App, _, _: &Undo, _| s.log.push("app undo"));
    input.on_action(root, |s: &mut App, _, _: &Redo, _| s.log.push("app redo"));
    set_clipboard(MemoryClipboard::default());
    Page { ui, input, app: App::default(), a, b, button }
}

impl Page {
    fn focus(&mut self, node: NodeId) {
        self.input.set_focus(&mut self.ui, &mut self.app, Some(node));
    }
    fn keys(&mut self, keys: &str) {
        self.input.simulate_keystrokes(&mut self.ui, &mut self.app, keys).unwrap();
    }
    fn sel(f: &TextField) -> (usize, usize) {
        f.selection()
    }
}

#[test]
fn a_field_takes_its_shortcuts_from_the_app_and_gives_back_what_it_cannot_use() {
    set_platform(Platform::Mac);
    let mut p = page();
    p.focus(p.a);
    p.keys("h e l l o space w o r l d");
    assert_eq!(p.app.a.text, "hello world");
    // ⌘A selects the field's text, not the app's everything.
    p.keys("cmd-a");
    assert_eq!(Page::sel(&p.app.a), (0, 11));
    assert!(p.app.log.is_empty());
    // ⌘B means nothing to a plain field: the app's binding runs.
    p.keys("cmd-b");
    assert_eq!(p.app.log, ["toggle sidebar"]);
    // Outside a field, ⌘A is the app's.
    p.focus(p.button);
    p.keys("cmd-a");
    assert_eq!(p.app.log, ["toggle sidebar", "select everything"]);
}

#[test]
fn undo_in_a_field_is_the_fields_until_it_has_nothing_left() {
    set_platform(Platform::Mac);
    let mut p = page();
    p.focus(p.a);
    p.keys("h i space t h e r e");
    assert_eq!(p.app.a.text, "hi there");
    // A word at a time.
    p.keys("cmd-z");
    assert_eq!(p.app.a.text, "hi");
    p.keys("cmd-z");
    assert_eq!(p.app.a.text, "");
    assert!(p.app.log.is_empty());
    p.keys("shift-cmd-z");
    assert_eq!(p.app.a.text, "hi");
    // Nothing left to undo in the field: the app's undo.
    p.keys("cmd-z cmd-z");
    assert_eq!(p.app.a.text, "");
    assert_eq!(p.app.log, ["app undo"]);
}

#[test]
fn copy_cut_and_paste_move_text_between_fields() {
    set_platform(Platform::Mac);
    let mut p = page();
    p.focus(p.a);
    p.keys("o n e space t w o");
    // ⇧⌥← selects the last word; ⌘X cuts it.
    p.keys("shift-alt-left cmd-x");
    assert_eq!(p.app.a.text, "one ");
    p.focus(p.b);
    p.keys("cmd-v cmd-v");
    assert_eq!(p.app.b.text, "twotwo");
    // ⌘C with nothing selected leaves the key to the app (it has no use for
    // it here either), and the clipboard alone.
    p.keys("cmd-c cmd-v");
    assert_eq!(p.app.b.text, "twotwotwo");
    // Copy keeps the text.
    p.keys("cmd-a cmd-c right cmd-v");
    assert_eq!(p.app.b.text, "twotwotwotwotwotwo");
    // ⌥⌫ deletes a word; ⌘⌫ to the line's start.
    p.focus(p.a);
    p.keys("t h r e e alt-backspace");
    assert_eq!(p.app.a.text, "one ");
    p.keys("cmd-backspace");
    assert_eq!(p.app.a.text, "");
}

#[test]
fn windows_and_linux_use_ctrl() {
    set_platform(Platform::Other);
    let mut p = page();
    p.focus(p.a);
    p.keys("a b c space d e f");
    p.keys("ctrl-left");
    assert_eq!(Page::sel(&p.app.a), (4, 4));
    p.keys("ctrl-shift-right");
    assert_eq!(Page::sel(&p.app.a), (4, 7));
    p.keys("ctrl-c end ctrl-v");
    assert_eq!(p.app.a.text, "abc defdef");
    p.keys("ctrl-z");
    assert_eq!(p.app.a.text, "abc def");
    p.keys("ctrl-y");
    assert_eq!(p.app.a.text, "abc defdef");
    p.keys("ctrl-a shift-delete");
    assert_eq!(p.app.a.text, "");
    p.keys("shift-insert");
    assert_eq!(p.app.a.text, "abc defdef");
    // ⌘ (the Windows key) is no command here, and Ctrl+B is the app's.
    p.keys("ctrl-b");
    assert_eq!(p.app.log, ["toggle sidebar"]);
}

#[test]
fn a_capture_listener_hears_keys_before_the_field() {
    set_platform(Platform::Mac);
    let mut p = page();
    let root = p.ui.root();
    p.input.on_key_capture(root, |s: &mut App, _, key, _, _| {
        if key == "ArrowDown" {
            s.log.push("list down");
            return true;
        }
        false
    });
    p.focus(p.a);
    p.keys("x down");
    assert_eq!(p.app.a.text, "x");
    assert_eq!(p.app.log, ["list down"]);
}
