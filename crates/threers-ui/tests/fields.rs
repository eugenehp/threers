//! The password field, the tag input and the date picker.

use threers_ui::binding::Binding;
use threers_ui::environment::Token;
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::states::Appearance;
use threers_ui::view::{date_picker, password_field, tag_input, CalendarWords, Day, El, View};
use threers_ui::{AccessNode, FixedWidthText, KeyModifiers, NodeId, NodeKind, Role, Ui};

#[derive(Default)]
struct App {
    secret: String,
    tags: Vec<String>,
    day: Option<Day>,
    sent: u32,
}

struct H {
    ui: Ui,
    input: Input<App>,
    view: View<App>,
    app: App,
    body: fn() -> Vec<El<App>>,
}

fn password() -> Vec<El<App>> {
    vec![password_field("Password", ("Show password", "Hide password"), Binding::new(|a: &App| a.secret.clone(), |a: &mut App, v| a.secret = v)).on_submit(|a: &mut App| a.sent += 1)]
}

fn tags() -> Vec<El<App>> {
    vec![tag_input("Tags", "Remove {tag}", Binding::new(|a: &App| a.tags.clone(), |a: &mut App, v| a.tags = v))]
}

fn date() -> Vec<El<App>> {
    vec![date_picker("Due", "No date", Binding::new(|a: &App| a.day, |a: &mut App, v| a.day = v), Day::new(2026, 11, 24).unwrap(), CalendarWords::for_language("en-GB"))]
}

impl H {
    fn new(body: fn() -> Vec<El<App>>) -> H {
        H::with(body, App::default())
    }
    fn with(body: fn() -> Vec<El<App>>, app: App) -> H {
        let mut ui = Ui::new();
        let root = ui.root();
        let view = View::new(&mut ui, root, "display: flex; flex-direction: column; width: 400px");
        let mut h = H { ui, input: Input::new(), view, app, body };
        h.show();
        h
    }
    fn show(&mut self) {
        self.view.update_with(&mut self.ui, &mut self.input, &self.app, (self.body)());
        self.ui.step(1.0);
        self.ui.layout(500.0, 600.0, &mut FixedWidthText(7.0));
    }
    fn all(&self, role: Role) -> Vec<AccessNode> {
        self.ui.accessibility_tree().into_iter().filter(|a| a.role == role).collect()
    }
    fn find(&self, role: Role, name: &str) -> AccessNode {
        self.all(role).into_iter().find(|a| a.name == name).unwrap_or_else(|| panic!("no {role:?} {name:?}"))
    }
    fn press_at(&mut self, x: f32, y: f32) {
        let button = PointerButton::Primary;
        for e in [Event::PointerMove { x, y }, Event::PointerDown { x, y, button }, Event::PointerUp { x, y, button }] {
            self.input.dispatch(&mut self.ui, &mut self.app, e);
        }
        self.show();
    }
    fn press(&mut self, id: NodeId) {
        let r = self.ui.rect(id);
        self.press_at(r.x + r.width / 2.0, r.y + r.height / 2.0);
    }
    fn focus(&mut self, id: NodeId) {
        self.input.set_focus(&mut self.ui, &mut self.app, Some(id));
    }
    fn key(&mut self, key: &str, modifiers: KeyModifiers) {
        self.input.dispatch(&mut self.ui, &mut self.app, Event::Key { key: key.into(), modifiers });
        self.show();
    }
    fn keys(&mut self, keys: &str) {
        for k in keys.split(' ') {
            self.key(k, KeyModifiers::default());
        }
    }
    /// Every text there is, shown or not.
    fn texts(&self) -> String {
        fn walk(ui: &Ui, n: NodeId, out: &mut String) {
            if let NodeKind::Text(t) = ui.kind(n) {
                out.push_str(t);
            }
            for &c in ui.children(n) {
                walk(ui, c, out);
            }
        }
        let mut out = String::new();
        walk(&self.ui, self.ui.root(), &mut out);
        out
    }
    fn controls_named(&self, name: &str) -> NodeId {
        let all = self.all(Role::Button);
        all.iter().find(|n| n.name == name).unwrap_or(&all[0]).id
    }
    fn audit(&mut self) {
        for a in [Appearance::Light, Appearance::Dark] {
            self.ui.set_system_appearance(a);
            let root = self.view.root();
            self.ui.token_style(root, &[("background", Token::Surface), ("color", Token::Text)]);
            self.show();
            let page = self.ui.token(Token::Background);
            assert_eq!(self.ui.accessibility_audit(), vec![], "{a:?}");
            assert_eq!(self.ui.contrast_issues(page, false), vec![], "{a:?}");
        }
    }
}

/// What the platform's own shortcuts are held with.
fn command() -> KeyModifiers {
    let mac = cfg!(any(target_os = "macos", target_os = "ios"));
    KeyModifiers { meta: mac, control: !mac, ..Default::default() }
}

fn word() -> KeyModifiers {
    let mac = cfg!(any(target_os = "macos", target_os = "ios"));
    KeyModifiers { alt: mac, control: !mac, ..Default::default() }
}

#[test]
fn a_password_is_nowhere_until_it_is_shown() {
    let mut h = H::new(password);
    let f = h.find(Role::TextBox, "Password").id;
    h.focus(f);
    h.keys("h u n t e r 2");
    assert_eq!(h.app.secret, "hunter2");
    let hidden = |h: &H| {
        let said = format!("{:?}", h.ui.accessibility_tree());
        !said.contains("hunter2") && !h.texts().contains("hunter2")
    };
    assert!(hidden(&h), "{}", h.texts());
    assert_eq!(h.find(Role::TextBox, "Password").value.as_deref(), Some("•••••••"));
    assert!(h.input.focused_text_is_secret(&h.ui), "a host is told");
    h.press(h.find(Role::Button, "Show password").id);
    assert!(h.texts().contains("hunter2"));
    assert_eq!(h.find(Role::TextBox, "Password").value.as_deref(), Some("hunter2"));
    h.press(h.find(Role::Button, "Hide password").id);
    assert!(hidden(&h));
    assert!(!h.input.focused_text_is_secret(&h.ui), "the eye holds none");
    h.audit();
}

#[test]
fn a_hidden_password_is_not_copied_nor_its_words_found() {
    let mut h = H::with(password, App { secret: "one two".into(), ..Default::default() });
    let f = h.find(Role::TextBox, "Password").id;
    h.focus(f);
    struct Kept(String);
    impl threers_ui::text_shortcuts::Clipboard for Kept {
        fn read(&mut self) -> Option<String> {
            Some(self.0.clone())
        }
        fn write(&mut self, text: &str) {
            self.0 = text.into();
        }
    }
    threers_ui::text_shortcuts::set_clipboard(Kept("kept".into()));
    for key in ["a", "c", "x"] {
        h.key(key, command());
    }
    assert_eq!(threers_ui::text_shortcuts::with_clipboard(|c| c.read()).as_deref(), Some("kept"));
    assert_eq!(h.app.secret, "one two", "cutting takes nothing");
    h.keys("End");
    h.key("Backspace", word());
    assert_eq!(h.app.secret, "", "a word back is all of it");
}

#[test]
fn a_password_is_edited_at_its_caret_and_sent() {
    let mut h = H::new(password);
    let f = h.find(Role::TextBox, "Password").id;
    h.focus(f);
    h.keys("a c ArrowLeft b");
    assert_eq!(h.app.secret, "abc");
    h.keys("Enter");
    assert_eq!((h.app.sent, h.app.secret.as_str()), (1, "abc"));
    // Space on the eye shows it; it is not typed.
    let eye = h.find(Role::Button, "Show password").id;
    h.focus(eye);
    h.keys("Space");
    h.key(" ", KeyModifiers::default());
    assert_eq!(h.app.secret, "abc");
    // The app empties it: it is empty.
    h.app.secret.clear();
    h.focus(f);
    h.keys("z");
    assert_eq!(h.app.secret, "z");
    // Typed on a host's keyboard.
    assert!(h.input.set_focused_text(&mut h.ui, &mut h.app, "pa\nss".into(), 5, 5));
    assert_eq!(h.app.secret, "pass");
}

#[test]
fn tags_are_kept_once_each() {
    let mut h = H::new(tags);
    let f = h.find(Role::TextBox, "Tags").id;
    h.focus(f);
    h.keys("r u s t Enter");
    h.keys("g o , r u s t Enter");
    assert_eq!(h.app.tags, ["rust", "go"]);
    h.keys("Enter");
    assert_eq!(h.app.tags.len(), 2, "nothing typed, nothing kept");
    // A list pasted, the last of it still being typed.
    h.input.set_focused_text(&mut h.ui, &mut h.app, "c, zig ,, go\nada".into(), 16, 16);
    h.show();
    assert_eq!(h.app.tags, ["rust", "go", "c", "zig"]);
    assert_eq!(h.find(Role::TextBox, "Tags").value.as_deref(), Some("ada"));
    h.keys("Enter");
    assert_eq!(h.app.tags.last().unwrap(), "ada");
    h.keys("Backspace");
    assert_eq!(h.app.tags.len(), 4);
    let names: Vec<_> = h.all(Role::ListItem).into_iter().map(|n| (n.name, n.position)).collect();
    assert_eq!(names[1], ("go".into(), Some((2, 4))));
    h.audit();
}

#[test]
fn a_tags_cross_removes_it_and_focus_stays_in_the_field() {
    let mut h = H::with(tags, App { tags: vec!["a".into(), "b".into(), "c".into()], ..Default::default() });
    h.press(h.find(Role::Button, "Remove b").id);
    assert_eq!(h.app.tags, ["a", "c"]);
    assert_eq!(h.input.focused(), Some(h.find(Role::TextBox, "Tags").id));
    // By the app, they follow at the next event.
    h.app.tags = vec!["z".into()];
    h.keys("ArrowLeft");
    assert_eq!(h.all(Role::ListItem).len(), 1);
    h.find(Role::Button, "Remove z");
}

#[test]
fn a_day_is_picked_from_the_calendar_that_opens() {
    let mut h = H::new(date);
    let b = h.find(Role::ComboBox, "Due");
    assert_eq!((b.value, b.expanded), (None, Some(false)));
    assert!(h.texts().contains("No date"));
    h.press(b.id);
    h.find(Role::Dialog, "Due");
    assert_eq!(h.find(Role::ComboBox, "Due").expanded, Some(true));
    assert_eq!(h.input.focused(), Some(h.find(Role::Group, "November 2026").id), "on the days");
    h.audit();
    h.press(h.find(Role::Button, "November 12, 2026").id);
    assert_eq!(h.app.day, Day::new(2026, 11, 12));
    assert!(h.all(Role::Dialog).is_empty(), "closed");
    let b = h.find(Role::ComboBox, "Due");
    assert_eq!((b.value.as_deref(), b.expanded), (Some("November 12, 2026"), Some(false)));
    h.audit();
}

#[test]
fn the_keys_move_the_day_and_escape_puts_it_back() {
    let mut h = H::with(date, App { day: Day::new(2026, 11, 29), ..Default::default() });
    let b = h.find(Role::ComboBox, "Due").id;
    h.focus(b);
    h.keys("Enter");
    h.find(Role::Dialog, "Due");
    h.keys("ArrowRight ArrowRight");
    assert_eq!(h.app.day, Day::new(2026, 12, 1));
    h.find(Role::Group, "December 2026");
    h.keys("Escape");
    assert!(h.all(Role::Dialog).is_empty());
    assert_eq!(h.app.day, Day::new(2026, 11, 29), "as it was");
    assert_eq!(h.input.focused(), Some(b), "focus is given back");
    h.keys("Enter ArrowRight Enter");
    assert!(h.all(Role::Dialog).is_empty());
    assert_eq!(h.app.day, Day::new(2026, 11, 30), "taken");
    // Open, looked through, and left by a press elsewhere.
    h.press(b);
    h.press(h.find(Role::Button, "Next month").id);
    h.find(Role::Group, "December 2026");
    h.press_at(490.0, 590.0);
    assert!(h.all(Role::Dialog).is_empty());
    assert_eq!(h.app.day, Day::new(2026, 11, 30));
    // Its button gone, so is its calendar.
    h.press(b);
    h.body = Vec::new;
    h.show();
    h.keys("a b");
    assert!(h.all(Role::Dialog).is_empty());
}

#[test]
fn a_combo_box_opens_from_the_keyboard() {
    use threers_ui::view::picker_menu;
    for key in ["Enter", " ", "ArrowDown"] {
        let mut h = H::new(|| vec![picker_menu("Count", vec!["One".into(), "Two".into()], Some(0), |a: &mut App, i| a.sent = i as u32)]);
        h.focus(h.find(Role::ComboBox, "Count").id);
        h.key(key, KeyModifiers::default());
        assert_eq!(h.all(Role::Menu).len(), 1, "{key:?}");
        h.keys("End Enter");
        assert_eq!(h.app.sent, 1, "{key:?}");
        assert!(h.all(Role::Menu).is_empty());
    }
}

#[test]
fn a_native_shell_is_not_given_a_password() {
    use threers_ui::native::{controls, to_json, version};
    let mut h = H::new(password);
    let field = |h: &H| controls(&h.ui).into_iter().find(|c| c.editable).unwrap();
    assert!(field(&h).secret);
    assert_eq!(field(&h).value.as_deref(), Some(""), "empty is said, to empty its own");
    h.focus(h.find(Role::TextBox, "Password").id);
    h.keys("h u n t e r");
    let before = version(&controls(&h.ui));
    for shown in [false, true] {
        assert_eq!(field(&h).value, None, "shown {shown}");
        let json = to_json(&controls(&h.ui));
        assert!(json.contains("\"secret\":true") && !json.contains("hunter") && !json.contains('•'), "{json}");
        h.press(h.controls_named("Show password"));
    }
    // What the shell's own field holds comes in whole, over what there was.
    h.focus(h.find(Role::TextBox, "Password").id);
    assert!(h.input.set_focused_text(&mut h.ui, &mut h.app, "new".into(), 3, 3));
    assert_eq!(h.app.secret, "new");
    // Emptied by the app: the shell hears of it.
    h.app.secret.clear();
    h.keys("Shift");
    assert_eq!(field(&h).value.as_deref(), Some(""));
    assert_ne!(version(&controls(&h.ui)), before);
    // Other fields are given as they were.
    let h = H::new(tags);
    assert_eq!((field(&h).secret, field(&h).value.as_deref()), (false, Some("")));
}

#[cfg(feature = "accesskit")]
#[test]
fn accesskit_is_told_it_is_a_password() {
    let mut h = H::new(password);
    h.focus(h.find(Role::TextBox, "Password").id);
    h.keys("h u n t e r");
    let u = threers_ui::a11y_accesskit::tree_update(&h.ui, None);
    assert!(u.nodes.iter().any(|(_, n)| n.role() == accesskit::Role::PasswordInput));
    assert!(!format!("{u:?}").contains("hunter"));
}

/// Hidden characters are drawn, not written: a font without a bullet would
/// show boxes.
#[test]
fn a_hidden_password_is_dots_no_font_can_lack() {
    let mut h = H::with(password, App { secret: "hunter2".into(), ..Default::default() });
    let dots = |h: &H| -> Vec<f32> {
        fn walk(ui: &Ui, n: NodeId, out: &mut Vec<f32>) {
            let r = ui.rect(n);
            if ui.children(n).is_empty() && (r.width, r.height) == (6.0, 6.0) {
                out.push(r.x);
            }
            for &c in ui.children(n) {
                walk(ui, c, out);
            }
        }
        let mut out = vec![];
        walk(&h.ui, h.ui.root(), &mut out);
        out
    };
    let evenly = |d: &[f32]| d.windows(2).all(|w| (w[1] - w[0] - 8.0).abs() < 0.6);
    h.show();
    assert!(dots(&h).len() == 7 && evenly(&dots(&h)), "{:?}", dots(&h));
    assert!(!h.texts().contains('•'));
    let f = h.find(Role::TextBox, "Password").id;
    h.focus(f);
    h.keys("End Backspace");
    assert_eq!(dots(&h).len(), 6, "{}", h.app.secret);
    // Selected, they are the selection's ink.
    h.key("ArrowLeft", KeyModifiers { shift: true, ..Default::default() });
    assert_eq!(dots(&h).len(), 6);
    h.audit();
    h.press(h.find(Role::Button, "Show password").id);
    assert!(dots(&h).is_empty() && h.texts().contains("hunter"));
    h.press(h.find(Role::Button, "Hide password").id);
    assert_eq!(dots(&h).len(), 6);
    assert!(!h.texts().contains("hunter"));
}
