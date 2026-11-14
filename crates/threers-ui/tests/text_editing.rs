//! `TextEditing` on its own: a field and a paragraph, edited and selected
//! with a finger and the mouse, as an app wires it.

use threers_ui::input::{Input, PointerButton, TouchPhase};
use threers_ui::text_editing::TextEditing;
use threers_ui::text_touch::Request;
use threers_ui::{style, FixedWidthText, KeyModifiers, NodeId, Role, Semantics, TextField, Ui};

#[derive(Default)]
struct State {
    field: TextField,
}

struct Setup {
    ui: Ui,
    input: Input<State>,
    state: State,
    editing: TextEditing<State>,
    field: NodeId,
    para: NodeId,
    t: f64,
}

impl Setup {
    fn new() -> Self {
        let mut ui = Ui::new();
        let root = ui.root();
        let para = ui.add_box(root, style! { width: 300px; height: 40px; });
        ui.add_text(para, "Some words to read and copy");
        let field = ui.add_box(
            root,
            style! { width: 200px; padding: 4px; overflow: hidden; },
        );
        ui.set_semantics(field, Semantics::new(Role::TextBox).label("Name"));
        let text = ui.add_text(field, "");
        let mut input = Input::new();
        let mut editing = TextEditing::new(&mut ui, &mut input);
        editing.add_field(&mut ui, field, text, |s: &mut State| &mut s.field);
        let state = State {
            field: TextField::new("one two three"),
        };
        let mut s = Setup {
            ui,
            input,
            state,
            editing,
            field,
            para,
            t: 0.0,
        };
        s.frame();
        s
    }

    fn frame(&mut self) {
        self.t += 1.0 / 60.0;
        let m = &mut FixedWidthText(10.0);
        self.input.tick(&mut self.ui, &mut self.state, self.t);
        self.editing
            .tick(&mut self.ui, &self.input, &mut self.state, self.t, m);
        for _ in 0..2 {
            self.ui.layout(400.0, 300.0, m);
            self.editing
                .place(&mut self.ui, &mut self.input, &mut self.state, m);
        }
        self.ui.layout(400.0, 300.0, m);
    }

    fn tap(&mut self, id: u64, (x, y): (f32, f32)) {
        let m = &mut FixedWidthText(10.0);
        let t = self.t;
        self.editing.touch(
            &mut self.ui,
            &mut self.input,
            &mut self.state,
            id,
            TouchPhase::Start,
            x,
            y,
            t,
            m,
        );
        self.editing.touch(
            &mut self.ui,
            &mut self.input,
            &mut self.state,
            id,
            TouchPhase::End,
            x,
            y,
            t + 0.05,
            m,
        );
        self.t += 0.1;
        self.frame();
    }

    fn names(&self) -> Vec<String> {
        self.ui
            .accessibility_tree()
            .into_iter()
            .filter(|a| a.role == Role::MenuItem)
            .map(|a| a.name)
            .collect()
    }

    fn centre(&self, name: &str) -> (f32, f32) {
        let b = self
            .ui
            .accessibility_tree()
            .into_iter()
            .find(|a| a.name == name)
            .unwrap()
            .bounds;
        (b.x + b.width / 2.0, b.y + b.height / 2.0)
    }

    /// Character `i` of the field's text.
    fn at(&self, i: f32) -> (f32, f32) {
        let r = self.ui.rect(self.field);
        (r.x + 5.0 + 10.0 * i + 4.0, r.y + r.height / 2.0)
    }
}

#[test]
fn a_finger_edits_a_field_with_handles_and_the_edit_menu() {
    let mut s = Setup::new();
    // The tap that focuses it puts the caret where it lands.
    let p = s.at(1.0);
    s.tap(1, p);
    assert_eq!(s.input.focused(), Some(s.field));
    assert_eq!((s.state.field.anchor, s.state.field.caret), (1, 1));
    // Two taps on "two": selected, with both handles and the menu.
    let p = s.at(5.0);
    s.tap(2, p);
    s.tap(3, p);
    assert_eq!(s.state.field.selection(), (4, 7));
    let (handles, menu) = s.editing.showing(&s.ui);
    assert_eq!((handles, menu), ([true, true], true));
    assert_eq!(&s.names()[..4], ["Cut", "Copy", "Paste", "Select All"]);
    // Cut: the text goes, and the clipboard is asked for.
    let cut = s.centre("Cut");
    s.tap(4, cut);
    assert_eq!(s.editing.take_requests(), [Request::Copy("two".into())]);
    assert_eq!(s.state.field.text, "one  three");
}

#[test]
fn page_text_selects_with_a_held_finger_and_the_mouse() {
    let mut s = Setup::new();
    let r = s.ui.rect(s.para);
    let (x, y) = (r.x + 10.0 * 6.5, r.y + 5.0);
    // Held: the word, and a read-only menu.
    let m = &mut FixedWidthText(10.0);
    let t = s.t;
    s.editing.touch(
        &mut s.ui,
        &mut s.input,
        &mut s.state,
        9,
        TouchPhase::Start,
        x,
        y,
        t,
        m,
    );
    for _ in 0..40 {
        s.frame();
    }
    let t = s.t;
    s.editing.touch(
        &mut s.ui,
        &mut s.input,
        &mut s.state,
        9,
        TouchPhase::End,
        x,
        y,
        t,
        m,
    );
    s.frame();
    assert_eq!(s.editing.page_selection().as_deref(), Some("words"));
    assert!(s.names().contains(&"Copy".to_owned()) && !s.names().contains(&"Cut".to_owned()));
    // The mouse drags out "Some words" and ⌘C copies it.
    s.editing.pointer_down(
        &mut s.ui,
        &mut s.input,
        &mut s.state,
        r.x + 1.0,
        y,
        PointerButton::Primary,
        false,
        m,
    );
    s.editing
        .pointer_move(&mut s.ui, &mut s.input, &mut s.state, r.x + 100.0, y, m);
    s.editing.pointer_up(
        &mut s.ui,
        &mut s.input,
        &mut s.state,
        r.x + 100.0,
        y,
        PointerButton::Primary,
    );
    assert!(s.editing.key(
        "c",
        KeyModifiers {
            meta: true,
            ..Default::default()
        }
    ));
    assert_eq!(
        s.editing.take_requests(),
        [Request::Copy("Some words".into())]
    );
}

#[test]
fn a_long_line_scrolls_to_its_caret() {
    let mut s = Setup::new();
    let p = s.at(1.0);
    s.tap(1, p);
    s.state
        .field
        .insert(" and many more words past the end of the field");
    s.frame();
    let text = s.ui.children(s.field)[0];
    assert!(s.ui.rect(text).x < s.ui.rect(s.field).x, "scrolled left");
}
