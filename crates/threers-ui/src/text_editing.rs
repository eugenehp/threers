//! Text editing and selecting, put together: a UI's text fields edited with
//! the pointer, the keyboard's caret keys and a finger, and any text on the
//! page selected and copied, with what shows for them — the caret, the
//! selection, scrolling a long line to its caret, a finger's handles,
//! highlights over selected page text, and the edit menu.
//!
//! The app keeps each field's [`TextField`] in its state and tells
//! [`TextEditing`] where to find it ([`add_field`](TextEditing::add_field));
//! it sends the pointer and fingers through [`pointer_down`](TextEditing::pointer_down),
//! [`pointer_move`](TextEditing::pointer_move), [`pointer_up`](TextEditing::pointer_up)
//! and [`touch`](TextEditing::touch), which pass to [`Input`] what they do
//! not take, and calls [`place`](TextEditing::place) after each layout (and
//! lays out again: the menu is centred by its own width). What only the
//! platform can do — the clipboard, looking up, translating, sharing — comes
//! back as [`Request`]s ([`take_requests`](TextEditing::take_requests)).
//!
//! On a finger, the gestures are [`TouchSelection`]'s; page text follows
//! [`Ui::selectable_text_at`] (`user-select`).

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::input::{Event, Gesture, Input, Phase, PointerButton, TouchPhase};
use crate::rich_text::TextRun;
use crate::text_field::{KeyModifiers, TextField};
use crate::text_touch::{EditAction, Request, TouchConfig, TouchSelection};
use crate::tree::{NodeId, NodeKind, Rect, Ui};
use crate::{style, Display, Length, Role, Semantics, TextMeasure};

/// A text field: its box, its text node, its caret, and where its
/// [`TextField`] is in the app's state.
struct Field<S> {
    boxed: NodeId,
    text: NodeId,
    caret: NodeId,
    model: fn(&mut S) -> &mut TextField,
    /// How far its text is scrolled left, to show the caret.
    scroll: f32,
}

/// Text selected on the page: which text node, its text and selection, and
/// the touch editing of it.
struct PageSelection {
    node: NodeId,
    model: TextField,
    touch: TouchSelection,
}

/// What a button of the edit menu does, as placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MenuEntry {
    Back,
    More,
    Action(EditAction),
}

/// Actions to a page of the edit menu.
const MENU_PER_PAGE: usize = 4;

/// Text fields and page text, edited and selected. See the
/// [module docs](self).
pub struct TextEditing<S> {
    fields: Vec<Field<S>>,
    /// Touch editing of the focused field.
    field_touch: TouchSelection,
    /// The finger a field took, and the tap that focused one.
    field_finger: Option<u64>,
    focus_tap: Option<(u64, usize)>,
    /// The pointer is selecting in a field.
    selecting: bool,
    page: Option<PageSelection>,
    page_template: TouchSelection,
    page_finger: Option<u64>,
    page_dragging: bool,
    /// A finger has been used: handles and the menu show.
    touch_mode: bool,
    handles: [NodeId; 2],
    menu: NodeId,
    menu_items: Vec<(NodeId, NodeId)>,
    menu_entries: Vec<MenuEntry>,
    highlights: Vec<NodeId>,
    /// Set by the menu's buttons and a held finger, taken after dispatch.
    picked: Rc<Cell<Option<usize>>>,
    long_press: Rc<Cell<Option<(f32, f32)>>>,
    requests: Vec<Request>,
    /// The page's input draws a field's text itself (see
    /// [`set_native_fields`](Self::set_native_fields)).
    native: bool,
    /// The field the edit menu shows for, as placed.
    menu_field: Option<usize>,
}

impl<S: 'static> TextEditing<S> {
    /// Text editing for `ui`, with its handles, highlights and edit menu
    /// (drawn over everything), and the gestures as iOS has them.
    pub fn new(ui: &mut Ui, input: &mut Input<S>) -> Self {
        Self::with_config(ui, input, TouchConfig::default())
    }

    /// Text editing with touch gestures as `config` says.
    pub fn with_config(ui: &mut Ui, input: &mut Input<S>, config: TouchConfig) -> Self {
        let root = ui.root();
        let handles = [0, 1].map(|end| {
            let h = ui.add_box(root, style! { display: none; position: absolute; width: 2px; pointer-events: none; background: rgb(37, 99, 235); });
            let knob = ui.add_box(h, style! { position: absolute; left: -5px; width: 10px; height: 10px; border-radius: 5px; background: rgb(37, 99, 235); });
            if end == 0 {
                ui.style_mut(knob).top = Length::Px(-9.0);
            } else {
                ui.style_mut(knob).bottom = Length::Px(-9.0);
            }
            ui.defer(h, 30);
            h
        });
        let menu = ui.add_box(
            root,
            style! { display: none; position: absolute; flex-direction: row; border-radius: 8px; background: #2b303b; box-shadow: 0 6px 18px rgba(0,0,0,0.5); overflow: hidden; color: #fcfcfc; user-select: none; },
        );
        ui.appearance_style(
            menu,
            crate::states::Appearance::Light,
            "background: #ffffff; color: #1f2329; border: 1px solid #d5d9e0",
        );
        ui.set_semantics(menu, Semantics::new(Role::Menu).label("Edit"));
        ui.defer(menu, 31);
        // A press on the menu keeps the field's focus (and the keyboard).
        input.on(menu, Phase::Bubble, |_, _, e, cx| {
            if let Event::PointerDown { .. } = e {
                cx.prevent_default();
            }
        });
        let picked: Rc<Cell<Option<usize>>> = Rc::default();
        let menu_items: Vec<(NodeId, NodeId)> = (0..MENU_PER_PAGE + 2)
            .map(|i| {
                let b = ui.add_box(
                    menu,
                    style! { padding: 9px 12px; font-size: 13px; cursor: pointer; },
                );
                if i > 0 {
                    crate::css::apply(
                        ui.style_mut(b),
                        "border-left",
                        "1px solid rgba(128,128,128,0.35)",
                    );
                }
                let t = ui.add_text(b, "");
                ui.set_semantics(b, Semantics::new(Role::MenuItem));
                ui.hover_style(b, "background: rgba(128,128,128,0.25)");
                let p = picked.clone();
                input.on_click(b, move |_, _, _| p.set(Some(i)));
                (b, t)
            })
            .collect();
        // A finger held still on the page's text selects its word.
        let long_press: Rc<Cell<Option<(f32, f32)>>> = Rc::default();
        let l = long_press.clone();
        input.on(root, Phase::Bubble, move |_, _, e, _| {
            if let Event::Gesture(Gesture::LongPress { x, y }) = e {
                l.set(Some((*x, *y)));
            }
        });
        TextEditing {
            fields: vec![],
            field_touch: TouchSelection::with_config(config.clone()),
            field_finger: None,
            focus_tap: None,
            selecting: false,
            page: None,
            page_template: TouchSelection::with_config(config).into_read_only(),
            page_finger: None,
            page_dragging: false,
            touch_mode: false,
            handles,
            menu,
            menu_items,
            menu_entries: vec![],
            highlights: vec![],
            picked,
            long_press,
            requests: vec![],
            native: false,
            menu_field: None,
        }
    }

    /// Makes `boxed` (with text node `text` inside) a text field whose
    /// [`TextField`] `model` finds in the app's state: it gets a caret, its
    /// selection shows, it scrolls to its caret, and the pointer and a
    /// finger edit it. Give `boxed` `overflow: hidden` and a text-box role.
    pub fn add_field(
        &mut self,
        ui: &mut Ui,
        boxed: NodeId,
        text: NodeId,
        model: fn(&mut S) -> &mut TextField,
    ) {
        let caret = ui.add_box(boxed, style! { position: absolute; width: 1px; height: 16px; display: none; pointer-events: none; });
        let c = ui.color_of(text);
        ui.style_mut(caret).background = Some(c);
        self.fields.push(Field {
            boxed,
            text,
            caret,
            model,
            scroll: 0.0,
        });
    }

    /// The touch editing of fields: its config, and the app's own menu
    /// actions ([`TouchSelection::add_action`]).
    pub fn field_touch(&mut self) -> &mut TouchSelection {
        &mut self.field_touch
    }

    /// The touch editing new page selections start from.
    pub fn page_touch(&mut self) -> &mut TouchSelection {
        &mut self.page_template
    }

    /// On a touch screen, the page's own text input draws a field's text,
    /// its caret and its selection: the field draws its box only.
    pub fn set_native_fields(&mut self, native: bool) {
        self.native = native;
    }

    /// What the platform has to do: copy, paste, look up, translate, share.
    pub fn take_requests(&mut self) -> Vec<Request> {
        std::mem::take(&mut self.requests)
    }

    /// The focused field, if one is.
    fn focused(&self, input: &Input<S>) -> Option<usize> {
        let f = input.focused()?;
        self.fields.iter().position(|x| x.boxed == f)
    }

    /// Which field `node` is in.
    fn field_at(&self, ui: &Ui, node: Option<NodeId>) -> Option<usize> {
        let n = node.filter(|&n| ui.contains(n))?;
        self.fields
            .iter()
            .position(|f| ui.ancestors(n).any(|a| a == f.boxed))
    }

    fn on_menu(&self, ui: &Ui, x: f32, y: f32) -> bool {
        ui.hit_test(x, y)
            .is_some_and(|t| ui.ancestors(t).any(|a| a == self.menu))
    }

    /// Whether a finger has been used (the handles and menu show).
    pub fn touch_mode(&self) -> bool {
        self.touch_mode
    }

    /// Whether the handles and the menu show now, for a host's debugging.
    pub fn showing(&self, ui: &Ui) -> ([bool; 2], bool) {
        let shown = |n: NodeId| ui.style(n).display != Display::None;
        (
            [shown(self.handles[0]), shown(self.handles[1])],
            shown(self.menu),
        )
    }

    /// The finger a field took, if one did.
    pub fn field_finger(&self) -> Option<u64> {
        self.field_finger
    }

    /// The page's selected text, if any.
    pub fn page_selection(&self) -> Option<String> {
        let p = self.page.as_ref()?;
        let (lo, hi) = p.model.selection();
        Some(p.model.text.chars().skip(lo).take(hi - lo).collect())
    }

    /// Puts the page's selection away.
    pub fn clear_page_selection(&mut self) {
        self.page = None;
    }

    /// The field was typed into (from the platform's text input): the menu
    /// goes, as on a phone.
    pub fn typed(&mut self) {
        self.field_touch.dismiss();
    }

    /// The tap that focused a field from outside the UI (the page's own
    /// input took it), at character `index`: the next tap counts as the
    /// second of a double tap.
    pub fn note_focus_tap(&mut self, time: f64, index: usize) {
        self.field_touch.note_tap(time, index);
    }

    /// Starts a selection of the page's text at `(x, y)`, if there is
    /// selectable text there (not a field's: that is edited, not read).
    fn select_page_at(&mut self, ui: &Ui, x: f32, y: f32, measure: &mut dyn TextMeasure) -> bool {
        let Some(node) = ui.selectable_text_at(x, y) else {
            return false;
        };
        if self.field_at(ui, Some(node)).is_some() {
            return false;
        }
        let NodeKind::Text(text) = ui.kind(node) else {
            return false;
        };
        let mut model = TextField::new(text.clone());
        let i = ui.text_index_at(node, x, y, measure);
        model.anchor = i;
        model.caret = i;
        self.page = Some(PageSelection {
            node,
            model,
            touch: self.page_template.clone(),
        });
        true
    }

    /// The mouse pressed: in a field it places the caret (two clicks a
    /// word, three all, Shift extends); on page text it starts a selection;
    /// anywhere else it puts a page selection away. Passes the press on.
    #[allow(clippy::too_many_arguments)]
    pub fn pointer_down(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        x: f32,
        y: f32,
        button: PointerButton,
        shift: bool,
        measure: &mut dyn TextMeasure,
    ) {
        let on_menu = self.on_menu(ui, x, y);
        if shift
            && button == PointerButton::Primary
            && let Some(p) = &mut self.page
            && ui.selectable_text_at(x, y) == Some(p.node)
        {
            p.model.caret = ui.text_index_at(p.node, x, y, measure);
            self.page_dragging = true;
            return;
        }
        if !on_menu {
            self.page = None;
        }
        let out = input.dispatch(ui, state, Event::PointerDown { x, y, button });
        let field = self.field_at(ui, out.target);
        if field.is_none()
            && !on_menu
            && button == PointerButton::Primary
            && self.select_page_at(ui, x, y, measure)
        {
            let clicks = input.click_count();
            if let Some(p) = &mut self.page {
                let i = p.model.caret;
                p.model.pointer_down(i, clicks, false);
            }
            self.page_dragging = true;
        }
        if let (Some(k), PointerButton::Primary) = (field, button) {
            let f = &self.fields[k];
            let i = ui.text_index_at(f.text, x, y, measure);
            (f.model)(state).pointer_down(i, input.click_count(), shift);
            self.selecting = true;
        }
    }

    /// The mouse moved: a selection being dragged out follows it.
    pub fn pointer_move(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        x: f32,
        y: f32,
        measure: &mut dyn TextMeasure,
    ) {
        input.dispatch(ui, state, Event::PointerMove { x, y });
        if self.page_dragging
            && let Some(p) = &mut self.page
        {
            p.model.caret = ui.text_index_at(p.node, x, y, measure);
        }
        if self.selecting
            && let Some(k) = self.focused(input)
        {
            let f = &self.fields[k];
            let i = ui.text_index_at(f.text, x, y, measure);
            (f.model)(state).pointer_drag(i);
        }
    }

    /// The mouse let go. Passes it on, then carries out a menu choice.
    pub fn pointer_up(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        x: f32,
        y: f32,
        button: PointerButton,
    ) {
        if std::mem::take(&mut self.selecting) {
            for f in &self.fields {
                (f.model)(state).pointer_up();
            }
        }
        if std::mem::take(&mut self.page_dragging)
            && let Some(p) = &mut self.page
        {
            p.model.pointer_up();
            if p.model.anchor == p.model.caret {
                // A click, not a selection.
                self.page = None;
            }
        }
        input.dispatch(ui, state, Event::PointerUp { x, y, button });
        self.take_pick(state);
    }

    /// ⌘/Ctrl-C with page text selected copies it. Returns whether it did.
    pub fn key(&mut self, key: &str, modifiers: KeyModifiers) -> bool {
        if (modifiers.meta || modifiers.control)
            && key.eq_ignore_ascii_case("c")
            && let Some(text) = self.page_selection()
        {
            self.requests.push(Request::Copy(text));
            return true;
        }
        false
    }

    /// A finger: on a focused field's text or handles, or a page
    /// selection's handles, it edits (and nothing else hears it); the tap
    /// that focuses a field puts its caret where it lands; anything else is
    /// passed to `input`.
    #[allow(clippy::too_many_arguments)]
    pub fn touch(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
        measure: &mut dyn TextMeasure,
    ) {
        self.touch_mode = true;
        let focused = self.focused(input);
        let ends = matches!(phase, TouchPhase::End | TouchPhase::Cancel);
        if self.field_finger == Some(id) {
            if ends {
                self.field_finger = None;
            }
            if let Some(k) = focused {
                self.field_gesture(ui, state, k, phase, x, y, time, measure);
            }
            return;
        }
        if self.page_finger == Some(id) {
            if ends {
                self.page_finger = None;
            }
            self.page_gesture(ui, phase, x, y, time, measure);
            return;
        }
        let on_menu = self.on_menu(ui, x, y);
        if phase == TouchPhase::Start
            && !on_menu
            && let Some(p) = &self.page
        {
            let node = p.node;
            let m = RefCell::new(&mut *measure);
            let mut caret_at = |i: usize| ui.text_caret(node, i, *m.borrow_mut());
            if p.touch.wants(&p.model, x, y, false, &mut caret_at) {
                self.page_finger = Some(id);
                self.page_gesture(ui, phase, x, y, time, measure);
                return;
            }
            self.page = None;
        }
        if phase == TouchPhase::Start
            && !input.touching()
            && !on_menu
            && let Some(k) = focused
        {
            let f = &self.fields[k];
            let inside = self.field_at(ui, ui.hit_test(x, y)) == Some(k);
            let text = f.text;
            let m = RefCell::new(&mut *measure);
            let mut caret_at = |i: usize| ui.text_caret(text, i, *m.borrow_mut());
            if self
                .field_touch
                .wants((f.model)(state), x, y, inside, &mut caret_at)
            {
                self.field_finger = Some(id);
                self.field_gesture(ui, state, k, phase, x, y, time, measure);
                return;
            }
        }
        // A field without focus: the tap that focuses it puts the caret
        // where it lands, and is the first of a double tap.
        let first = phase == TouchPhase::Start && !input.touching();
        let pressed = if first && focused.is_none() {
            self.field_at(ui, ui.hit_test(x, y))
        } else {
            None
        };
        input.dispatch(
            ui,
            state,
            Event::Touch {
                id,
                phase,
                x,
                y,
                time,
            },
        );
        if let Some(k) = pressed {
            let f = &self.fields[k];
            let i = ui.text_index_at(f.text, x, y, measure);
            let model = (f.model)(state);
            model.anchor = i;
            model.caret = i;
            self.focus_tap = Some((id, i));
        }
        if phase == TouchPhase::End
            && let Some((_, i)) = self.focus_tap.take_if(|f| f.0 == id)
        {
            self.field_touch.note_tap(time, i);
        }
        self.take_pick(state);
    }

    /// How firmly finger `id` presses (0 to 1, where the screen measures
    /// it): on a field's text, pressed firmly, it selects the word.
    pub fn touch_force(&mut self, input: &Input<S>, state: &mut S, id: u64, force: f32) {
        if self.field_finger == Some(id)
            && let Some(k) = self.focused(input)
        {
            self.field_touch.force((self.fields[k].model)(state), force);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn field_gesture(
        &mut self,
        ui: &Ui,
        state: &mut S,
        k: usize,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
        measure: &mut dyn TextMeasure,
    ) {
        let f = &self.fields[k];
        let text = f.text;
        let m = RefCell::new(measure);
        let mut index_at = |x: f32, y: f32| ui.text_index_at(text, x, y, *m.borrow_mut());
        let mut caret_at = |i: usize| ui.text_caret(text, i, *m.borrow_mut());
        self.field_touch.touch(
            (f.model)(state),
            phase,
            x,
            y,
            time,
            &mut index_at,
            &mut caret_at,
        );
    }

    fn page_gesture(
        &mut self,
        ui: &Ui,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
        measure: &mut dyn TextMeasure,
    ) {
        let Some(p) = &mut self.page else { return };
        let node = p.node;
        let m = RefCell::new(measure);
        let mut index_at = |x: f32, y: f32| ui.text_index_at(node, x, y, *m.borrow_mut());
        let mut caret_at = |i: usize| ui.text_caret(node, i, *m.borrow_mut());
        p.touch.touch(
            &mut p.model,
            phase,
            x,
            y,
            time,
            &mut index_at,
            &mut caret_at,
        );
    }

    /// A tap on the edit menu: carried out now, while the host still
    /// handles the tap (a browser lets the clipboard be written only then).
    fn take_pick(&mut self, state: &mut S) {
        let Some(i) = self.picked.take() else { return };
        let Some(&entry) = self.menu_entries.get(i) else {
            return;
        };
        let (touch, model): (&mut TouchSelection, &mut TextField) = match &mut self.page {
            Some(p) => (&mut p.touch, &mut p.model),
            None => {
                let Some(k) = self.menu_field else { return };
                (&mut self.field_touch, (self.fields[k].model)(state))
            }
        };
        let action = match entry {
            MenuEntry::Back => return touch.turn_page(false),
            MenuEntry::More => return touch.turn_page(true),
            MenuEntry::Action(a) => a,
        };
        let request = touch.apply(model, action);
        if request.is_some() && self.page.is_some() {
            // Copied, looked up or shared: the page's selection has done.
            self.page = None;
        }
        self.requests.extend(request);
    }

    /// Time passing (call each frame, before layout): a held finger in a
    /// field takes the caret, one held on page text selects its word, and
    /// a field that lost focus puts its menu away.
    pub fn tick(
        &mut self,
        ui: &mut Ui,
        input: &Input<S>,
        state: &mut S,
        time: f64,
        measure: &mut dyn TextMeasure,
    ) {
        match self.focused(input) {
            Some(k) => {
                self.field_touch.tick((self.fields[k].model)(state), time);
            }
            None => self.field_touch.dismiss(),
        }
        if let Some((x, y)) = self.long_press.take()
            && self.select_page_at(ui, x, y, measure)
            && let Some(p) = &mut self.page
        {
            let i = p.model.caret;
            p.touch.select_word(&mut p.model, i);
            self.touch_mode = true;
        }
    }

    /// Whether frames are wanted (a finger is held, waiting to be long).
    pub fn needs_frame(&self) -> bool {
        self.field_touch.needs_frame()
    }

    /// Shows what editing and selecting look like now (call after each
    /// layout, then lay out again: the menu is centred by its own width):
    /// each field's selection, composition, caret and scroll; the page's
    /// selected text; the handles; the edit menu.
    pub fn place(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        measure: &mut dyn TextMeasure,
    ) {
        let focused = self.focused(input);
        for k in 0..self.fields.len() {
            self.place_field(ui, input, state, k, focused == Some(k), measure);
        }
        if self.page.as_ref().is_some_and(|p| !ui.contains(p.node)) {
            // Its text went (a list row scrolled out of sight).
            self.page = None;
        }
        self.place_highlights(ui, measure);
        self.place_handles_and_menu(ui, state, focused, measure);
    }

    #[allow(clippy::too_many_arguments)]
    fn place_field(
        &mut self,
        ui: &mut Ui,
        input: &mut Input<S>,
        state: &mut S,
        k: usize,
        focused: bool,
        measure: &mut dyn TextMeasure,
    ) {
        let (boxed, text, caret_node, model, mut scrolled) = {
            let f = &self.fields[k];
            (f.boxed, f.text, f.caret, f.model, f.scroll)
        };
        let f = model(state).clone();
        if ui.text_content(text) != f.text {
            ui.set_text(text, f.text.clone());
        }
        // The selection as a highlighted run, a composition underlined.
        let byte = |i: usize| {
            f.text
                .char_indices()
                .nth(i)
                .map_or(f.text.len(), |(b, _)| b)
        };
        let (a, z) = f.selection();
        let (ba, bz) = (byte(a), byte(z));
        let mut runs = if focused {
            vec![
                TextRun::new(ba),
                TextRun::new(bz - ba).background([0.145, 0.388, 0.922, 0.45]),
                TextRun::new(f.text.len() - bz),
            ]
        } else {
            vec![TextRun::new(f.text.len())]
        };
        if let Some((m0, m1)) = f.marked {
            let (b0, b1) = (byte(m0), byte(m1));
            runs = vec![
                TextRun::new(b0),
                TextRun::new(b1 - b0).underline(false),
                TextRun::new(f.text.len() - b1),
            ];
        }
        if self.native {
            // The page's own input draws the text over the field.
            runs = vec![TextRun::new(f.text.len()).color([0.0; 4])];
        }
        ui.set_text_runs(text, runs);
        let shows_caret = focused && !self.native && f.anchor == f.caret;
        let mut caret = ui.text_caret(text, f.caret, measure);
        let r = ui.rect(boxed);
        // A line longer than the field scrolls sideways to keep the caret in
        // view, as a text input does.
        let t = ui.rect(text);
        let bs = ui.style(boxed);
        let width = (r.width - bs.padding.horizontal() - bs.border.horizontal() - 1.0).max(1.0);
        let at = caret.x - t.x;
        let mut scroll = scrolled;
        if self.native {
            scroll = 0.0;
        } else if at - scroll > width {
            scroll = at - width;
        } else if at < scroll {
            scroll = at;
        }
        let end = ui.text_caret(text, f.text.chars().count(), measure).x - t.x;
        let scroll = scroll.clamp(0.0, (end - width).max(0.0)).round();
        if scroll != scrolled {
            caret.x -= scroll - scrolled;
            scrolled = scroll;
            let st = ui.style_mut(text);
            st.position = crate::Position::Relative;
            st.left = Length::Px(-scroll);
        }
        self.fields[k].scroll = scrolled;
        let mut s = ui.style(caret_node).clone();
        s.display = if shows_caret {
            Display::Block
        } else {
            Display::None
        };
        s.left = Length::Px(caret.x - r.x - 1.0);
        s.top = Length::Px(caret.y - r.y - 1.0 + (caret.height - 16.0) / 2.0);
        s.background = Some(ui.color_of(text));
        if &s != ui.style(caret_node) {
            ui.set_style(caret_node, s);
        }
        if focused {
            input.set_ime_area(Some(caret));
        }
    }

    /// The box that clips `node`: the intersection of its clipping
    /// ancestors (a scrolling list, a panel), or the window.
    fn clip_of(ui: &Ui, node: NodeId) -> Rect {
        let mut r = ui.rect(ui.root());
        for a in ui.ancestors(node).skip(1) {
            if ui.style(a).clip {
                let c = ui.rect(a);
                let (x0, y0) = (r.x.max(c.x), r.y.max(c.y));
                let (x1, y1) = (
                    (r.x + r.width).min(c.x + c.width),
                    (r.y + r.height).min(c.y + c.height),
                );
                r = Rect {
                    x: x0,
                    y: y0,
                    width: (x1 - x0).max(0.0),
                    height: (y1 - y0).max(0.0),
                };
            }
        }
        r
    }

    /// Translucent boxes over the page's selected text, inside what clips
    /// it, so styled runs keep their colours.
    fn place_highlights(&mut self, ui: &mut Ui, measure: &mut dyn TextMeasure) {
        let rects: Vec<Rect> = match &self.page {
            Some(p) => {
                let (lo, hi) = p.model.selection();
                let clip = Self::clip_of(ui, p.node);
                ui.text_selection_rects(p.node, lo, hi, measure)
                    .into_iter()
                    .filter_map(|b| {
                        let (x0, y0) = (b.x.max(clip.x), b.y.max(clip.y));
                        let (x1, y1) = (
                            (b.x + b.width).min(clip.x + clip.width),
                            (b.y + b.height).min(clip.y + clip.height),
                        );
                        (x1 > x0 && y1 > y0).then_some(Rect {
                            x: x0,
                            y: y0,
                            width: x1 - x0,
                            height: y1 - y0,
                        })
                    })
                    .collect()
            }
            None => vec![],
        };
        while self.highlights.len() < rects.len() {
            let root = ui.root();
            let h = ui.add_box(root, style! { display: none; position: absolute; pointer-events: none; background: rgba(59,130,246,0.32); });
            ui.defer(h, 29);
            self.highlights.push(h);
        }
        for (k, &h) in self.highlights.iter().enumerate() {
            let mut s = ui.style(h).clone();
            match rects.get(k) {
                Some(r) => {
                    s.display = Display::Block;
                    s.left = Length::Px(r.x);
                    s.top = Length::Px(r.y);
                    s.width = Length::Px(r.width);
                    s.height = Length::Px(r.height);
                }
                None => s.display = Display::None,
            }
            if &s != ui.style(h) {
                ui.set_style(h, s);
            }
        }
    }

    /// The selection's handles and the edit menu, for the page's selected
    /// text or the focused field, while a finger edits.
    fn place_handles_and_menu(
        &mut self,
        ui: &mut Ui,
        state: &mut S,
        focused: Option<usize>,
        measure: &mut dyn TextMeasure,
    ) {
        let field_model;
        let (text, model, sel, area, editing): (NodeId, &TextField, &TouchSelection, Rect, bool) =
            match &self.page {
                Some(p) => (
                    p.node,
                    &p.model,
                    &p.touch,
                    Self::clip_of(ui, p.node),
                    self.touch_mode,
                ),
                None => match focused {
                    Some(k) if !self.native => {
                        let f = &self.fields[k];
                        field_model = (f.model)(state).clone();
                        (
                            f.text,
                            &field_model,
                            &self.field_touch,
                            ui.rect(f.boxed),
                            self.touch_mode,
                        )
                    }
                    _ => {
                        self.hide(ui);
                        return;
                    }
                },
            };
        let m = RefCell::new(&mut *measure);
        let mut caret_at = |i: usize| ui.text_caret(text, i, *m.borrow_mut());
        let handles = sel.handles(model, &mut caret_at).filter(|_| editing);
        let page = sel.menu_page(model, MENU_PER_PAGE).filter(|_| editing);
        let labels: Vec<String> = page
            .iter()
            .flat_map(|p| p.actions.iter().map(|&a| sel.label(a).to_owned()))
            .collect();
        // Where the menu points: the selection's middle, or the caret.
        let (lo, hi) = model.selection();
        let (a, b) = (caret_at(lo), caret_at(hi));
        let clamp = |x: f32| x.clamp(area.x, area.x + area.width);
        let handles_at = handles.map(|h| {
            [
                (clamp(h.start.x), h.start.y, h.start.height),
                (clamp(h.end.x), h.end.y, h.end.height),
            ]
        });
        let menu_at = page
            .as_ref()
            .map(|_| ((clamp(a.x) + clamp(b.x)) / 2.0, a.y, a.y + a.height));
        self.menu_field = if self.page.is_some() { None } else { focused };
        for (k, &h) in self.handles.iter().enumerate() {
            let mut s = ui.style(h).clone();
            match handles_at {
                Some(at) => {
                    let (x, y, height) = at[k];
                    s.display = Display::Block;
                    s.left = Length::Px(x - 1.0);
                    s.top = Length::Px(y);
                    s.height = Length::Px(height);
                }
                None => s.display = Display::None,
            }
            if &s != ui.style(h) {
                ui.set_style(h, s);
            }
        }
        let mut place: Option<(f32, f32)> = None;
        match (page, menu_at) {
            (Some(page), Some((x, top, bottom))) => {
                let mut entries = vec![];
                if page.back {
                    entries.push(MenuEntry::Back);
                }
                entries.extend(page.actions.iter().map(|&a| MenuEntry::Action(a)));
                if page.more {
                    entries.push(MenuEntry::More);
                }
                for (i, &(b, t)) in self.menu_items.iter().enumerate() {
                    let d = if i < entries.len() {
                        Display::Block
                    } else {
                        Display::None
                    };
                    if ui.style(b).display != d {
                        // Through its base style: it has a hover style.
                        ui.update_style(b, |s| s.display = d);
                    }
                    let (label, name) = match entries.get(i) {
                        Some(MenuEntry::Back) => ("‹".to_owned(), "Back".to_owned()),
                        Some(MenuEntry::More) => ("›".to_owned(), "More".to_owned()),
                        Some(MenuEntry::Action(_)) => {
                            let l = labels
                                .get(i - usize::from(page.back))
                                .cloned()
                                .unwrap_or_default();
                            (l.clone(), l)
                        }
                        None => (String::new(), String::new()),
                    };
                    if ui.text_content(b) != label {
                        ui.set_text(t, label);
                        ui.set_semantics(b, Semantics::new(Role::MenuItem).label(name));
                    }
                }
                self.menu_entries = entries;
                // Over the line, or under it when there is no room above;
                // centred on the selection, inside the window.
                let size = ui.rect(self.menu);
                let window = ui.rect(ui.root());
                let (w, h) = (size.width.max(1.0), size.height.max(36.0));
                let y = if top - h - 14.0 >= 4.0 {
                    top - h - 14.0
                } else {
                    bottom + 18.0
                };
                let x = (x - w / 2.0).clamp(4.0, (window.width - w - 4.0).max(4.0));
                place = Some((x.round(), y.round()));
            }
            _ => self.menu_entries.clear(),
        }
        // Through its base style: it has a light appearance's.
        let (display, left, top) = match place {
            Some((x, y)) => (Display::Flex, Length::Px(x), Length::Px(y)),
            None => (
                Display::None,
                ui.style(self.menu).left,
                ui.style(self.menu).top,
            ),
        };
        let s = ui.style(self.menu);
        if s.display != display || s.left != left || s.top != top {
            ui.update_style(self.menu, |s| {
                s.display = display;
                s.left = left;
                s.top = top;
            });
        }
    }

    fn hide(&mut self, ui: &mut Ui) {
        for n in self.handles.into_iter().chain([self.menu]) {
            if ui.style(n).display != Display::None {
                ui.update_style(n, |s| s.display = Display::None);
            }
        }
        self.menu_entries.clear();
    }
}
