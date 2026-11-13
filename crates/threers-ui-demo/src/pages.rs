//! The demo's pages and the navigation between them: a sidebar beside the
//! page showing on a wide window, a bar under it on a phone. Each page has
//! a title and a line about what it shows. The Layout page (flexbox and a
//! data table) and the Accessibility page (settings and a live audit) are
//! built here; the rest are filled by the app and the showcase.

use threers_ui::environment::Token;
use threers_ui::input::Input;
use threers_ui::view::El;
use threers_ui::{style, NodeId, Role, Semantics, Ui};

use crate::app::State;

/// The pages, in the order the navigation lists them.
pub const PAGES: [(&str, &str); 12] = [
    ("Basics", "Drag and drop, buttons, notifications, prompts and a popover."),
    ("Lists", "Virtual lists of 100,000 rows, built as they show: pull to refresh, swipe to delete, arrow keys."),
    ("Text", "Rich text with links, a text field with touch selection and an edit menu, an image, a container query."),
    ("Layout", "Flexbox with wrapping, reversing, ordering and alignment, and a sortable data table of 10,000 rows."),
    ("Components", "People, a split pane, tasks described from state, and controls bound to it."),
    ("Screens", "A navigation stack with slide, cube and page-turn transitions, and a sheet."),
    ("3D", "Cards that tilt, flip and fly, a model to turn, depth of field, a cube, a drum, reflections and a split-flap board."),
    ("Shaders", "Custom shaders on boxes: fifteen effects, from a ripple and a lens to neon edges, a flag and a glitch, some turned by sliders."),
    ("Progress", "Progress indicators: plain bars, rings and spinners; glowing ones drawn by signed distance fields, flat and in glass, with presets; motion for any box; any control flat or solid."),
    ("Toolkit", "Materials, rolling numbers, keyframes, scroll transitions, gradients, shapes, masks, blending, gauges, charts, search, undo and shortcuts."),
    ("Accessibility", "Text size, reduced motion, right to left, an on-screen keyboard, and a live audit."),
    ("XR", "The whole demo on a panel in a headset (WebXR): Meta Quest, Apple Vision Pro, or a phone in AR."),
];

pub const BASICS: usize = 0;
pub const LISTS: usize = 1;
pub const TEXT: usize = 2;
pub const LAYOUT: usize = 3;
pub const COMPONENTS: usize = 4;
pub const SCREENS: usize = 5;
pub const DEPTH: usize = 6;
pub const SHADERS: usize = 7;
pub const PROGRESS: usize = 8;
pub const TOOLKIT: usize = 9;
pub const ACCESSIBILITY: usize = 10;
pub const XR: usize = 11;

/// The shell's nodes.
pub struct Shell {
    /// What scrolls: the page showing.
    pub content: NodeId,
    pub nav: NodeId,
    pub links: Vec<NodeId>,
    pub pages: Vec<NodeId>,
    /// The Layout page's flex container, restyled from the state.
    pub flex: NodeId,
    pub flex_items: Vec<NodeId>,
    /// The Accessibility page's audit, described from it.
    pub audit: NodeId,
    /// The XR page: what it says about XR here, and its buttons.
    xr: XrPage,
    /// The page last shown, and the settings last applied.
    shown: Option<usize>,
    applied: Option<(usize, usize, usize, usize, bool)>,
}

impl Shell {
    /// The navigation and the pages under `parent`.
    pub fn new(ui: &mut Ui, input: &mut Input<State>, state: &State, parent: NodeId) -> Self {
        let body = ui.add_box(parent, style! { display: flex; gap: 12px; flex-grow: 1; min-height: 0; });
        let nav = ui.add_box(body, style! { display: flex; flex-direction: column; gap: 2px; width: 170px; flex-shrink: 0; padding: 8px; border-radius: 10px; box-sizing: border-box; });
        ui.token_style(nav, &[("background", Token::Surface)]);
        ui.set_semantics(nav, Semantics::new(Role::Navigation).label("Pages"));
        let mut links = vec![];
        for (i, (name, _)) in PAGES.iter().enumerate() {
            let l = ui.add_box(nav, style! { padding: 8px 10px; border-radius: 6px; cursor: pointer; font-size: 13px; white-space: nowrap; flex-shrink: 0; });
            ui.add_text(l, *name);
            ui.set_semantics(l, Semantics::new(Role::Link).label(*name));
            ui.hover_style(l, "background: rgba(128,138,160,0.2)");
            ui.focus_visible_style(l, "box-shadow: 0 0 0 2px #93c5fd");
            input.on_click(l, move |s: &mut State, _, _| s.page = i);
            links.push(l);
        }
        let content = ui.add_box(body, style! { display: flex; flex-direction: column; flex-grow: 1; min-width: 0; min-height: 0; overflow: auto; scrollbar-width: thin; scrollbar-visibility: auto; scrollbar-fade: 800ms 300ms; });
        let mut pages = vec![];
        for (name, about) in PAGES {
            let p = ui.add_box(content, style! { display: none; flex-direction: column; gap: 12px; flex-shrink: 0; padding-bottom: 12px; });
            let h = ui.add_box(p, style! { font-size: 20px; font-weight: bold; });
            ui.add_text(h, name);
            ui.set_semantics(h, Semantics::new(Role::Heading(2)));
            let a = ui.add_box(p, style! { font-size: 13px; white-space: normal; });
            ui.token_style(a, &[("color", Token::SecondaryText)]);
            ui.add_text(a, about);
            ui.set_semantics(p, Semantics::new(Role::Region).label(name));
            pages.push(p);
        }
        // Narrow: the links in a row that scrolls, under the page.
        ui.container_query(
            body,
            ui.root(),
            vec![(0.0, style! { display: flex; flex-direction: column; gap: 8px; flex-grow: 1; min-height: 0; }), (700.0, style! { display: flex; gap: 12px; flex-grow: 1; min-height: 0; })],
        );
        ui.container_query(
            nav,
            ui.root(),
            vec![
                (0.0, style! { display: flex; flex-direction: row; gap: 2px; order: 1; padding: 6px; border-radius: 10px; overflow-x: auto; scrollbar-width: none; flex-shrink: 0; }),
                (700.0, style! { display: flex; flex-direction: column; gap: 2px; width: 170px; flex-shrink: 0; padding: 8px; border-radius: 10px; box-sizing: border-box; }),
            ],
        );
        let (flex, flex_items) = layout_page(ui, input, state, pages[LAYOUT]);
        let audit = accessibility_page(ui, input, state, pages[ACCESSIBILITY]);
        let xr = xr_page(ui, input, pages[XR]);
        Shell { content, nav, links, pages, flex, flex_items, audit, xr, shown: None, applied: None }
    }

    /// The state into the shell: the page showing and its link marked, and
    /// the Layout page's flexbox.
    pub fn apply(&mut self, ui: &mut Ui, state: &State) {
        self.xr.apply(ui, state);
        let page = state.page.min(self.pages.len() - 1);
        if self.shown != Some(page) {
            for (i, (&p, &l)) in self.pages.iter().zip(&self.links).enumerate() {
                let on = i == page;
                ui.update_style(p, |s| s.display = if on { threers_ui::Display::Flex } else { threers_ui::Display::None });
                ui.token_style(l, &[("background", if on { Token::Accent } else { Token::Surface }), ("color", if on { Token::OnAccent } else { Token::Text })]);
                // The page shown keeps its accent under the pointer (grey
                // behind its white text would be hard to read).
                ui.set_hover_style(l, if on { "background: #1d4ed8" } else { "background: rgba(128,138,160,0.2)" });
                ui.semantics_mut(l).current = on;
            }
            ui.scroll_to(self.content, 0.0);
            ui.scroll_into_view(self.links[page]);
            self.shown = Some(page);
        }
        let want = (state.flex_direction, state.flex_wrap, state.flex_justify, state.flex_align, state.flex_order);
        if self.applied != Some(want) {
            let dir = ["row", "column", "row-reverse", "column-reverse"][want.0.min(3)];
            let wrap = ["nowrap", "wrap", "wrap-reverse"][want.1.min(2)];
            let justify = ["flex-start", "center", "flex-end", "space-between", "space-around", "space-evenly"][want.2.min(5)];
            let align = ["flex-start", "center", "space-between", "stretch"][want.3.min(3)];
            ui.update_style(self.flex, |s| {
                for (k, v) in [("flex-direction", dir), ("flex-wrap", wrap), ("justify-content", justify), ("align-content", align)] {
                    threers_ui::css::apply(s, k, v);
                }
            });
            if let Some(&last) = self.flex_items.last() {
                ui.update_style(last, |s| s.order = if want.4 { -1 } else { 0 });
            }
            self.applied = Some(want);
        }
    }
}

/// The Layout page: a flexbox to try every setting on, and a data table.
fn layout_page(ui: &mut Ui, input: &mut Input<State>, state: &State, page: NodeId) -> (NodeId, Vec<NodeId>) {
    use threers_ui::pickers::segmented;
    use threers_ui::binding;
    let g = crate::showcase::group(ui, page, "Flexbox", "Pick a direction, wrapping, how items are spread along a line and how lines share the box.");
    segmented(ui, input, state, g, "Direction", &["Row", "Column", "Row reverse", "Column reverse"], binding!(|s: State| s.flex_direction));
    segmented(ui, input, state, g, "Wrap", &["No wrap", "Wrap", "Wrap reverse"], binding!(|s: State| s.flex_wrap));
    segmented(ui, input, state, g, "Justify content", &["Start", "Center", "End", "Between", "Around", "Evenly"], binding!(|s: State| s.flex_justify));
    segmented(ui, input, state, g, "Align content", &["Start", "Center", "Between", "Stretch"], binding!(|s: State| s.flex_align));
    threers_ui::components::switch(ui, input, state, g, "Last item first (order: -1)", binding!(|s: State| s.flex_order));
    let flex = ui.add_box(g, style! { display: flex; gap: 8px; height: 220px; padding: 8px; border-radius: 8px; overflow: auto; });
    ui.token_style(flex, &[("background", Token::Separator)]);
    ui.set_semantics(flex, Semantics::new(Role::Group).label("Flex container"));
    let mut items = vec![];
    for (i, w) in [70.0f32, 110.0, 50.0, 90.0, 130.0, 60.0, 80.0, 100.0, 70.0].into_iter().enumerate() {
        let b = ui.add_box(flex, style! { height: 40px; flex-shrink: 0; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; });
        ui.update_style(b, |s| s.width = threers_ui::Length::Px(w));
        let hue = [[0.11, 0.31, 0.85], [0.18, 0.49, 0.2], [0.71, 0.33, 0.04], [0.43, 0.16, 0.85], [0.71, 0.14, 0.09], [0.06, 0.46, 0.43]][i % 6];
        ui.update_style(b, |s| {
            s.background = Some([hue[0], hue[1], hue[2], 1.0]);
            s.color = [0.99, 0.99, 0.99, 1.0];
        });
        ui.add_text(b, (i + 1).to_string());
        items.push(b);
    }
    let t = crate::showcase::group(ui, page, "A data table", "10,000 files: click a heading to sort by it (again to reverse), click a row or use the arrow keys, Home, End, Page Up and Down.");
    let table = threers_ui::table::data_table(
        ui,
        input,
        state,
        t,
        "Files",
        vec![
            threers_ui::table::TableColumn::new("Name", "2fr"),
            threers_ui::table::TableColumn::new("Kind", "1fr"),
            threers_ui::table::TableColumn::new("Size (KB)", "90px").numeric(),
            threers_ui::table::TableColumn::new("Modified", "110px"),
        ],
        binding!(|s: State| s.table_rows),
        binding!(|s: State| s.table_sort),
        binding!(|s: State| s.table_selected),
        28.0,
    );
    ui.update_style(table.table, |s| s.height = threers_ui::Length::Px(320.0));
    (flex, items)
}

/// The rows of the Layout page's table.
pub fn table_rows() -> Vec<Vec<String>> {
    let kinds = ["Document", "Image", "Video", "Archive", "Spreadsheet", "Code"];
    let words = ["report", "holiday", "budget", "draft", "notes", "design", "photo", "invoice", "plan", "backup"];
    (0..10_000u64)
        .map(|i| {
            let h = i.wrapping_mul(2654435761) ^ (i >> 3);
            let kind = kinds[(h % 6) as usize];
            let ext = ["docx", "png", "mp4", "zip", "xlsx", "rs"][(h % 6) as usize];
            vec![
                format!("{}-{:04}.{ext}", words[((h >> 8) % 10) as usize], i),
                kind.to_owned(),
                ((h >> 4) % 900_000 / 10).to_string(),
                format!("2026-{:02}-{:02}", 1 + (h >> 12) % 12, 1 + (h >> 16) % 28),
            ]
        })
        .collect()
}

/// The Accessibility page: settings that change how the whole demo reads
/// and moves, a button that announces, and the audit, kept up as it runs.
fn accessibility_page(ui: &mut Ui, input: &mut Input<State>, state: &State, page: NodeId) -> NodeId {
    use threers_ui::binding;
    use threers_ui::components::switch;
    let g = crate::showcase::group(ui, page, "Settings", "These change the whole demo: try them, then visit the other pages.");
    // Light, dark, or the system's (the default), for the whole demo.
    threers_ui::theme::theme_picker(ui, input, state, g, binding!(|s: State| s.theme));
    // A part of a UI can hold its own appearance, whatever the rest shows.
    let both = ui.add_box(g, style! { display: flex; gap: 10px; flex-wrap: wrap; });
    for (theme, name) in [(threers_ui::states::Theme::Light, "Always light"), (threers_ui::states::Theme::Dark, "Always dark")] {
        let card = ui.add_box(both, style! { display: flex; flex-direction: column; gap: 6px; padding: 10px 12px; border-radius: 10px; min-width: 150px; border: 1px; });
        ui.token_style(card, &[("background", Token::Surface), ("color", Token::Text), ("border-color", Token::Border)]);
        let title = ui.add_box(card, style! { font-weight: bold; font-size: 13px; });
        ui.add_text(title, name);
        let note = ui.add_box(card, style! { font-size: 12px; });
        ui.token_style(note, &[("color", Token::SecondaryText)]);
        ui.add_text(note, "Its own theme, whatever the demo's");
        let b = ui.add_box(card, style! { padding: 4px 10px; border-radius: 6px; align-self: flex-start; font-size: 12px; });
        ui.token_style(b, &[("background", Token::Accent), ("color", Token::OnAccent)]);
        ui.add_text(b, "A button");
        ui.set_semantics(card, Semantics::new(Role::Group).label(format!("{name}: a preview")));
        ui.set_subtree_theme(card, Some(theme));
    }
    threers_ui::pickers::stepper(ui, input, state, g, "Text size (steps of 25%)", binding!(|s: State| s.text_scale), 0..=4, 1);
    switch(ui, input, state, g, "Reduce motion", binding!(|s: State| s.reduce_motion));
    switch(ui, input, state, g, "Right to left", binding!(|s: State| s.right_to_left));
    let kb = ui.add_box(g, style! { padding: 6px 12px; border-radius: 6px; cursor: pointer; align-self: flex-start; });
    ui.token_style(kb, &[("background", Token::Accent), ("color", Token::OnAccent)]);
    ui.add_text(kb, "Turn the on-screen keyboard on or off");
    ui.set_semantics(kb, Semantics::new(Role::Button).label("Toggle the on-screen keyboard"));
    input.on_click(kb, |s: &mut State, _, _| s.command = Some("keyboard".into()));
    let say = ui.add_box(g, style! { padding: 6px 12px; border-radius: 6px; cursor: pointer; align-self: flex-start; });
    ui.token_style(say, &[("background", Token::Raised), ("color", Token::Text)]);
    ui.add_text(say, "Announce something");
    ui.set_semantics(say, Semantics::new(Role::Button).label("Announce something"));
    input.on_click(say, |_: &mut State, ui, _| ui.announce("Hello from threers-ui: this was read without a node to carry it."));
    let a = crate::showcase::group(ui, page, "Audit", "What would trouble someone using assistive technology or a keyboard in what shows now: unnamed controls, small targets, low contrast, skipped headings. Kept up as you change the settings.");
    let list = ui.add_box(a, style! { display: flex; flex-direction: column; gap: 4px; });
    ui.set_semantics(list, Semantics::new(Role::List).label("Audit results"));
    list
}

/// The XR page's nodes.
struct XrPage {
    /// The status line's text.
    status: NodeId,
    enter: [NodeId; 3],
    exit: NodeId,
    shown: Option<(String, bool, [bool; 3])>,
}

impl XrPage {
    /// The status line, and Enter or Exit, from the state.
    fn apply(&mut self, ui: &mut Ui, state: &State) {
        let want = (state.xr_status.clone(), state.xr_presenting, state.xr_can);
        if self.shown.as_ref() == Some(&want) {
            return;
        }
        let status = if want.0.is_empty() { "WebXR runs in a browser: open this demo in one." } else { want.0.as_str() };
        ui.set_text(self.status, status);
        let shown = |on: bool| move |s: &mut threers_ui::Style| s.display = if on { threers_ui::Display::Flex } else { threers_ui::Display::None };
        // Each way in shown while nothing is, and only where it can open:
        // a button that can only fail is not offered.
        for (b, can) in self.enter.into_iter().zip(want.2) {
            ui.update_style(b, shown(!want.1 && can));
        }
        ui.update_style(self.exit, shown(want.1));
        self.shown = Some(want);
    }
}

/// The XR page: how to use the demo in a headset, whether this browser
/// can, and the buttons that start and end a session (the page starts it:
/// see `State::xr_request`).
fn xr_page(ui: &mut Ui, input: &mut Input<State>, page: NodeId) -> XrPage {
    let g = crate::showcase::group(
        ui,
        page,
        "In a headset",
        "The demo shows on a panel in front of you, every page of it. With controllers, point and pull the trigger to click, push a thumbstick to scroll, and squeeze to bring the panel back in front of you. On Apple Vision Pro, look at something and pinch: a pinch is a tap, and pinching and dragging scrolls and swipes as a finger does.",
    );
    let status = ui.add_box(g, style! { font-size: 13px; white-space: normal; });
    ui.token_style(status, &[("color", Token::Text)]);
    let status_text = ui.add_text(status, "");
    ui.set_semantics(status, Semantics::new(Role::Status));
    let row = ui.add_box(g, style! { display: flex; gap: 8px; flex-wrap: wrap; });
    let button = |ui: &mut Ui, input: &mut Input<State>, label: &str, about: &str, request: &'static str, accent: bool| {
        let b = ui.add_box(row, style! { padding: 8px 14px; border-radius: 6px; cursor: pointer; });
        ui.token_style(b, &[("background", if accent { Token::Accent } else { Token::Raised }), ("color", if accent { Token::OnAccent } else { Token::Text })]);
        ui.add_text(b, label);
        ui.set_semantics(b, Semantics::new(Role::Button).label(about));
        ui.focus_visible_style(b, "box-shadow: 0 0 0 2px #93c5fd");
        input.on_click(b, move |s: &mut State, _, _| s.xr_request = Some(request.into()));
        b
    };
    let vr = button(ui, input, "Enter VR", "Show the demo in a VR headset", "immersive-vr", true);
    let ar = button(ui, input, "Enter AR", "Show the demo in AR, over the room", "immersive-ar", false);
    let phone = button(ui, input, "Look around", "Show the demo around you, the phone's camera behind it: turn the phone to look, tap to press", "phone", true);
    let exit = button(ui, input, "Exit", "Leave the headset view", "end", true);
    crate::showcase::group(
        ui,
        page,
        "What it needs",
        "A browser with WebXR (Meta Quest Browser; Safari on Apple Vision Pro, VR only; Chrome on Android for AR), the GPU renderer (WebGPU or WebGL2, not the 2D canvas), and a secure page: https, or localhost. Safari on iPhone and iPad has no immersive WebXR: there, Look around turns the view with the phone and shows its camera behind the panel (turning, not walking).",
    );
    XrPage { status: status_text, enter: [vr, ar, phone], exit, shown: None }
}

/// The audit's results, described for the view.
pub fn audit_rows(found: &[threers_ui::a11y_core::AuditIssue]) -> Vec<El<State>> {
    if found.is_empty() {
        return vec![El::div("display: flex; gap: 8px; align-items: center; padding: 6px 0")
            .token("color", Token::Success)
            .role(Role::ListItem, "No issues found")
            .text("✓ No issues found")];
    }
    found
        .iter()
        .take(50)
        .enumerate()
        .map(|(i, f)| {
            El::div("display: flex; gap: 8px; font-size: 12px; white-space: normal; padding: 4px 0")
                .key(i.to_string())
                .role(Role::ListItem, format!("{:?}: {}", f.kind, f.detail))
                .child(El::div("font-weight: bold; flex-shrink: 0").text(format!("⚠ {:?}", f.kind)))
                .child(El::div("min-width: 0").text(f.detail.clone()))
        })
        .collect()
}
