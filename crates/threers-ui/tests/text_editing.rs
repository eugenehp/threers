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

/// A list of thirty 20 px lines in a 100 px box, two labels beside each
/// other over it, a button among them.
struct Page {
    ui: Ui,
    input: Input<State>,
    state: State,
    editing: TextEditing<State>,
    list: NodeId,
    lines: Vec<NodeId>,
    t: f64,
}

impl Page {
    fn new() -> Self {
        let mut ui = Ui::new();
        let root = ui.root();
        ui.update_style(root, |s| s.flex_direction = threers_ui::FlexDirection::Column);
        let row = ui.add_box(root, style! { display: flex; flex-direction: row; gap: 20px; height: 20px; });
        ui.add_text(row, "Name");
        ui.add_text(row, "Value");
        let b = ui.add_box(root, style! { height: 20px; });
        ui.set_semantics(b, Semantics::new(Role::Button).label("Press"));
        ui.add_text(b, "Press");
        let end = ui.add_box(root, style! { height: 20px; });
        ui.add_text(end, "After");
        let list = ui.add_box(root, style! { display: flex; flex-direction: column; width: 300px; height: 100px; overflow-y: auto; });
        let lines = (0..30)
            .map(|i| {
                let l = ui.add_box(list, style! { height: 20px; flex-shrink: 0; });
                ui.add_text(l, format!("line {i:02}"))
            })
            .collect();
        let mut input = Input::new();
        let editing = TextEditing::new(&mut ui, &mut input);
        let mut p = Page { ui, input, state: State::default(), editing, list, lines, t: 0.0 };
        p.frame();
        p
    }

    fn frame(&mut self) {
        self.t += 1.0 / 60.0;
        let m = &mut FixedWidthText(10.0);
        self.input.tick(&mut self.ui, &mut self.state, self.t);
        self.editing.tick(&mut self.ui, &self.input, &mut self.state, self.t, m);
        for _ in 0..2 {
            self.ui.layout(400.0, 300.0, m);
            self.editing.place(&mut self.ui, &mut self.input, &mut self.state, m);
        }
        self.ui.layout(400.0, 300.0, m);
    }

    /// Left of character `i` of line `k`, at its middle.
    fn at(&self, k: usize, i: usize) -> (f32, f32) {
        let r = self.ui.rect(self.lines[k]);
        (r.x + i as f32 * 10.0 + 1.0, r.y + r.height / 2.0)
    }

    fn down(&mut self, (x, y): (f32, f32), shift: bool) {
        let m = &mut FixedWidthText(10.0);
        self.editing.pointer_down(&mut self.ui, &mut self.input, &mut self.state, x, y, PointerButton::Primary, shift, m);
    }

    fn drag(&mut self, (x, y): (f32, f32)) {
        let m = &mut FixedWidthText(10.0);
        self.editing.pointer_move(&mut self.ui, &mut self.input, &mut self.state, x, y, m);
    }

    fn up(&mut self, (x, y): (f32, f32)) {
        self.editing.pointer_up(&mut self.ui, &mut self.input, &mut self.state, x, y, PointerButton::Primary);
    }

    fn touch(&mut self, phase: TouchPhase, (x, y): (f32, f32)) {
        let m = &mut FixedWidthText(10.0);
        let t = self.t;
        self.editing.touch(&mut self.ui, &mut self.input, &mut self.state, 1, phase, x, y, t, m);
    }

    fn text(&self) -> String {
        self.editing.page_selection().unwrap_or_default()
    }

    fn scrolled(&self) -> f32 {
        self.ui.style(self.list).scroll_top
    }
}

#[test]
fn a_page_selection_runs_from_one_piece_of_text_to_the_next() {
    let mut p = Page::new();
    let (a, b) = (p.at(0, 2), p.at(2, 4));
    p.down(a, false);
    p.drag(b);
    p.up(b);
    assert_eq!(p.text(), "ne 00\nline 01\nline");
    // Shift moves its end, on or back
    let c = p.at(1, 4);
    p.down(c, true);
    p.up(c);
    assert_eq!(p.text(), "ne 00\nline");
    let d = p.at(3, 4);
    p.down(d, true);
    let e = p.at(1, 4);
    p.drag(e);
    p.up(e);
    assert_eq!(p.text(), "ne 00\nline", "dragged on after a Shift-click, it shrinks too");
    // outside the list: beside each other a space apart, a button's label left out
    p.down((1.0, 10.0), false);
    p.drag((390.0, 50.0));
    p.up((390.0, 50.0));
    assert_eq!(p.text(), "Name Value\nAfter");
    // Copy has it all
    assert!(p.editing.key("c", KeyModifiers { meta: true, ..Default::default() }));
    assert_eq!(p.editing.take_requests(), vec![Request::Copy("Name Value\nAfter".into())]);
}

#[test]
fn a_drag_held_at_the_edge_scrolls_and_selects_on() {
    let mut p = Page::new();
    let a = p.at(1, 0);
    p.down(a, false);
    // inside the box a mouse scrolls nothing
    let b = p.at(4, 6);
    p.drag(b);
    for _ in 0..30 {
        p.frame();
    }
    assert_eq!(p.scrolled(), 0.0);
    let below = p.ui.rect(p.list);
    let out = (40.0, below.y + below.height + 30.0);
    p.drag(out);
    for _ in 0..40 {
        p.frame();
    }
    assert!(p.scrolled() > 100.0, "{}", p.scrolled());
    let lines = p.text().lines().count();
    assert!(lines > 8, "{lines}");
    for _ in 0..200 {
        p.frame();
    }
    assert!(p.text().lines().count() == 29 && p.text().contains("line 28"), "to the end: {}", p.text());
    p.up(out);
    let at = p.scrolled();
    p.frame();
    assert_eq!(p.scrolled(), at);
}

#[test]
fn a_fingers_handle_crosses_pieces_and_scrolls_at_the_edge() {
    let mut p = Page::new();
    let at = p.at(1, 1);
    p.touch(TouchPhase::Start, at);
    for _ in 0..60 {
        p.frame();
    }
    p.touch(TouchPhase::End, at);
    p.frame();
    assert_eq!(p.text(), "line");
    assert_eq!(p.editing.showing(&p.ui).0, [true, true]);
    // the last handle, pulled two lines down
    let (x, y) = p.at(1, 4);
    p.touch(TouchPhase::Start, (x, y + 14.0));
    let (tx, ty) = p.at(3, 7);
    p.touch(TouchPhase::Move, (tx, ty + 14.0));
    assert_eq!(p.text(), "line 01\nline 02\nline 03");
    // above the first: the other end now
    let (ux, uy) = p.at(0, 5);
    p.touch(TouchPhase::Move, (ux, uy + 14.0));
    assert_eq!(p.text(), "00\n");
    // held at the bottom
    let r = p.ui.rect(p.list);
    p.touch(TouchPhase::Move, (tx, r.y + r.height - 2.0));
    for _ in 0..90 {
        p.frame();
    }
    assert!(p.scrolled() > 100.0, "{}", p.scrolled());
    assert!(p.text().lines().count() > 8, "{}", p.text());
    p.touch(TouchPhase::End, (tx, r.y + r.height - 2.0));
    p.frame();
    // the first end's scrolled out of view: one handle
    assert_eq!(p.editing.showing(&p.ui).0, [false, true]);
}

#[test]
fn a_selection_is_kept_through_text_added_and_put_away_with_its_own() {
    let mut p = Page::new();
    let (a, b) = (p.at(0, 0), p.at(1, 7));
    p.down(a, false);
    p.drag(b);
    p.up(b);
    let want = "line 00\nline 01";
    assert_eq!(p.text(), want);
    // another piece's text grows, and one's added: the same still
    let (l, list) = (p.lines[4], p.list);
    p.ui.set_text(l, "line 04, streamed on");
    let more = p.ui.add_box(list, style! { height: 20px; flex-shrink: 0; });
    p.ui.add_text(more, "one more");
    p.frame();
    assert_eq!(p.text(), want);
    // its own says something else: gone
    let l = p.lines[1];
    p.ui.set_text(l, "changed");
    p.frame();
    assert_eq!(p.text(), "");
}
