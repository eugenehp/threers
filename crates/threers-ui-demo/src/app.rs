//! The demo's UI and what it does, independent of where it is drawn: the
//! browser wires it to a canvas, and the tests drive it natively.

use threers_ui::debug_overlay::{FrameSample, FrameStats};
use threers_ui::image::Image;
use threers_ui::input::{Event, Input, Phase, PointerButton, TouchPhase};
use threers_ui::menu::{popover_menu, MenuItem};
use threers_ui::navigation::{ListNavigation, Nav};
use threers_ui::notifications::{Notice, Notifications, Tone};
use threers_ui::palette::{open_palette, Command};
use threers_ui::rich_text::TextRun;
use threers_ui::states::Theme;
use threers_ui::text_editing::TextEditing;
use threers_ui::text_touch::Request;
use threers_ui::{
    style, FieldKey, KeyModifiers, NodeId, Painter, Role, Semantics, TextField, TextMeasure, Ui,
};

/// What the UI's listeners change.
#[derive(Debug, Default)]
pub struct State {
    pub field: TextField,
    /// The drop slot, while it shows in a lane.
    pub slot: Option<NodeId>,
    pub dialog_open: bool,
    pub shadows: bool,
    /// A message for a notification, waiting to be shown.
    pub toast: Option<String>,
    /// A command asked for (by a menu, the palette or a button), for the
    /// app to run: what needs the input (a prompt, a popover) runs there.
    pub command: Option<String>,
    /// The components section's tab.
    pub showcase_tab: usize,
    /// The Tasks tab's tasks (its view is described from them).
    pub tasks: Vec<crate::showcase::Task>,
    /// Notifications turned off.
    pub quiet: bool,
    /// Asking whether to clear the name: the prompt shows while this is set.
    pub confirm_clear: bool,
    /// The Controls tab's values.
    pub size: usize,
    pub guests: i64,
    pub volume: f32,
    pub loading: bool,
    pub query: String,
    /// The sheet shows while this is set.
    pub sheet_open: bool,
    /// The Screens tab's stack.
    pub path: Vec<String>,
    /// The Depth tab's card, turned over.
    pub flipped: bool,
    /// The Depth tab's model, as it is turned.
    pub orbit: threers_ui::model3d::Orbit,
    /// Its depth of field's focus, 0 (far) to 1 (near).
    pub focus: f32,
    /// Its deck of people to swipe.
    pub deck: Vec<String>,
    /// The Screens tab's transition: slide, cube or page turn.
    pub nav_transition: usize,
    /// The page showing.
    pub page: usize,
    /// The 3D page's cube: the face in front.
    pub cube_face: usize,
    /// The Shaders page's sliders: the colour split, the pixel size and
    /// how far the card has burnt.
    pub shader_split: f32,
    pub shader_cell: f32,
    pub shader_burn: f32,
    /// The Progress page's slider.
    pub progress: f32,
    /// Whether the Progress page's controls are drawn solid.
    pub progress_solid: bool,
    /// The Tailwind page: its tab, and the buttons pressed there.
    pub tw_tab: usize,
    pub tw_presses: u32,
    pub tw_pressed: &'static str,
    /// The 3D page's departures board: which one it shows.
    pub departure: usize,
    /// The Toolkit page: the rolling number, the fading word, the gauges'
    /// level, the search, the chips added (undoably), and the undo.
    pub toolkit_count: i64,
    pub toolkit_word: usize,
    pub toolkit_level: f32,
    pub toolkit_query: String,
    pub toolkit_chips: Vec<[f32; 4]>,
    pub undo: threers_ui::undo::UndoManager<State>,
    /// Whether keys are labelled as on a Mac (⌘) or not (Ctrl): set by the
    /// page, which knows where it runs.
    pub mac: bool,
    /// The Layout page's flexbox: direction, wrap, justify, align-content,
    /// and whether the last item is moved first.
    pub flex_direction: usize,
    pub flex_wrap: usize,
    pub flex_justify: usize,
    pub flex_align: usize,
    pub flex_order: bool,
    /// The Layout page's table: its rows, sort and selection.
    pub table_rows: std::rc::Rc<Vec<Vec<String>>>,
    pub table_sort: Option<(usize, bool)>,
    pub table_selected: Option<usize>,
    /// The Accessibility page's settings.
    pub text_scale: i64,
    /// The system's text size (1, or 0 unset), times the page's own steps.
    pub system_text_scale: f32,
    pub reduce_motion: bool,
    pub right_to_left: bool,
    /// Lines for the page's console.
    pub log: Vec<String>,
    /// The lists panel's tab showing.
    pub list_tab: usize,
    /// Rows ticked in the `Select` list.
    pub selected: std::collections::HashSet<usize>,
    /// A node tapped in the `Select` list, for the frame to find its row.
    pub list_clicked: Option<NodeId>,
    /// Text cut or copied, for the page to put on the clipboard.
    pub copy_out: Option<String>,
    /// Paste asked for: the page reads the clipboard and calls `paste`.
    pub paste_wanted: bool,
    /// Text to share (the menu's Share…), for the page's share sheet.
    pub share_out: Option<String>,
    /// Where text with links was clicked or tapped.
    pub clicked_at: Option<(f32, f32)>,
    /// A link followed, for the page to open.
    pub open_out: Option<String>,
    /// A row swiped past deleting, for the frame to delete.
    pub delete_row: Option<NodeId>,
    /// The list being refreshed (pulled down and let go), by tab.
    pub refreshing: Option<usize>,
    /// Refreshes so far: the lists' content changes with each.
    pub refreshes: usize,
    /// The XR page: a session asked for (`immersive-vr`, `immersive-ar`,
    /// or `end`), for the page to start; what the page says about XR here;
    /// and whether the demo is showing in a headset.
    pub xr_request: Option<String>,
    pub xr_status: String,
    pub xr_presenting: bool,
    /// What this browser can open: a VR session, an AR session, and the
    /// phone's look-around view (motion sensors and the camera, where it
    /// has no immersive WebXR: Safari on iPhone).
    pub xr_can: [bool; 3],
    /// The appearance asked for (System by default), as the Accessibility
    /// page's picker shows it; the menu and the palette change it too.
    pub theme: Theme,
}

/// The app-wide settings: a global, changed from listeners that have only
/// the `Ui`, and read by the frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    pub theme: Theme,
}

/// A card being dragged.
#[derive(Debug, Clone, Copy)]
struct Card(NodeId);

#[derive(Debug, Clone)]
struct Save;
#[derive(Debug, Clone)]
struct CloseDialog;
#[derive(Debug, Clone)]
struct OpenPalette;

/// The nodes the app reaches back into.
struct Nodes {
    field_box: NodeId,
    field_text: NodeId,
    dialog: NodeId,
    backdrop: NodeId,
    cards: Vec<NodeId>,
    lanes: Vec<NodeId>,
    slot: NodeId,
}

pub struct App {
    pub ui: Ui,
    pub input: Input<State>,
    pub state: State,
    nodes: Nodes,
    pub lists: crate::lists::Lists,
    stats: FrameStats,
    last: Option<f64>,
    /// Text nodes with links, and their links' boxes as last laid out.
    link_nodes: Vec<NodeId>,
    links: Vec<(threers_ui::Rect, String)>,
    /// The field's editing and the page's text selection, with a finger and
    /// the pointer: threers-ui's `TextEditing`.
    pub editing: TextEditing<State>,
    /// When the refresh under way finishes (it pretends to load).
    refresh_until: Option<f64>,
    /// Inspector on: the box under the pointer is outlined, and a click
    /// logs what it is.
    pub inspecting: bool,
    pub show_stats: bool,
    viewport: (f32, f32),
    /// The dragged card's height, for its slot.
    drag_height: f32,
    /// The card shadow setting the cards were last given.
    shadows_drawn: Option<bool>,
    /// Layouts the last frame ran.
    pub layouts_last: u32,
    /// The notification stack.
    pub notices: Notifications<State>,
    pub showcase: crate::showcase::Showcase,
    /// The focused list's highlighted row, and the tab it is for.
    nav: ListNavigation,
    nav_tab: usize,
    /// The row marked selected, and its index then (rows are reused).
    marked: Option<(NodeId, usize)>,
    /// The Tasks tab, described from the state.
    tasks_view: threers_ui::view::View<State>,
    /// Where settings are kept between runs (the page's `localStorage`),
    /// and the theme last kept there.
    storage: Option<std::rc::Rc<dyn threers_ui::storage::Storage>>,
    stored_theme: Option<Theme>,
    /// The state's theme as last applied: the picker changed it if the
    /// state's differs, else the menu or the palette did.
    state_theme: Theme,
    /// The on-screen keyboard (off unless asked for).
    pub keyboard: threers_ui::virtual_keyboard::VirtualKeyboard,
    /// The navigation and the pages.
    pub shell: crate::pages::Shell,
    /// The Accessibility page's audit, and when it was last run.
    audit_view: threers_ui::view::View<State>,
    audit_at: f64,
    /// A link's section, to go to once its page is laid out, and how many
    /// frames it has waited.
    pending_link: Option<(String, u8)>,
    /// The node a link last went to, for the mirror to move focus to.
    followed: Option<NodeId>,
    /// The Shaders page's cards, turned by its sliders.
    shaders: crate::shaders::Shaders,
    progress: crate::progress::Progress,
    /// The Toolkit page, kept up from the state.
    toolkit: crate::toolkit::Toolkit,
    tailwind: crate::tailwind::TailwindPage,
}

const ACCENT: &str = "#2563eb";

fn button(ui: &mut Ui, parent: NodeId, label: &str, primary: bool) -> NodeId {
    let b = ui.add_box(
        parent,
        if primary {
            style! { padding: 6px 14px; border-radius: 6px; background: rgb(37, 99, 235); color: #fcfcfc; font-size: 13px; cursor: pointer; }
        } else {
            style! { padding: 5px 13px; border-radius: 6px; border: 1px solid #3a4150; background: #232833; color: #d6dae3; font-size: 13px; cursor: pointer; }
        },
    );
    ui.add_text(b, label);
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    if !primary {
        light(
            ui,
            b,
            "background: #ffffff; border: 1px solid #d0d4da; color: #1f2329",
        );
    }
    ui.hover_style(
        b,
        if primary {
            "background: #1d4ed8"
        } else {
            "background: rgba(128,138,160,0.28)"
        },
    );
    ui.active_style(
        b,
        if primary {
            "background: #1e40af"
        } else {
            "background: rgba(128,138,160,0.4)"
        },
    );
    ui.focus_visible_style(b, "border: 2px solid #93c5fd");
    ui.transition(b, 0.12);
    b
}

/// Styles `node` takes in the light appearance (the demo is drawn dark).
fn light(ui: &mut Ui, node: NodeId, css: &str) {
    ui.appearance_style(node, threers_ui::states::Appearance::Light, css);
}

fn panel(ui: &mut Ui, parent: NodeId, title: &str) -> NodeId {
    let p = ui.add_box(
        parent,
        style! { display: flex; flex-direction: column; gap: 10px; padding: 12px; border-radius: 10px; background: #1b1f27; border: 1px solid #2a303c; overflow: hidden; },
    );
    light(ui, p, "background: #ffffff; border: 1px solid #e2e5ea");
    let h = ui.add_box(
        p,
        style! { font-size: 12px; color: #8b93a5; font-weight: bold; },
    );
    ui.token_style(
        h,
        &[("color", threers_ui::environment::Token::SecondaryText)],
    );
    ui.add_text(h, title);
    ui.set_semantics(h, Semantics::new(Role::Heading(2)));
    p
}

/// A small generated picture, as decoded pixels.
fn picture() -> threers_ui::image::ImageRef {
    let (w, h) = (96u32, 64u32);
    let mut px = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
            let ring = ((u - 0.5).hypot(v - 0.6) * 18.0).sin() * 0.5 + 0.5;
            px.extend([
                (40.0 + 200.0 * u) as u8,
                (60.0 + 120.0 * ring) as u8,
                (180.0 + 60.0 * v) as u8,
                255,
            ]);
        }
    }
    Image::new(w, h, px).expect("sized")
}

impl App {
    pub fn new() -> Self {
        let mut ui = Ui::new();
        // Built dark, the demo's own look: styles read back while building
        // (for container queries) are the dark ones; the page then says
        // what the system shows.
        ui.set_system_appearance(threers_ui::states::Appearance::Dark);
        let mut input: Input<State> = Input::new();
        let state = State {
            shadows: true,
            field: TextField::new("Edit me: click, drag, double-click"),
            tasks: ["Read the docs", "Try the palette", "Resize the split"]
                .iter()
                .enumerate()
                .map(|(i, t)| crate::showcase::Task {
                    id: i as u32 + 1,
                    title: (*t).into(),
                    done: i == 0,
                })
                .collect(),
            guests: 2,
            volume: 0.4,
            focus: 0.5,
            shader_split: 14.0,
            shader_cell: 6.0,
            shader_burn: 0.35,
            progress: 0.62,
            progress_solid: true,
            toolkit_count: 41,
            mac: cfg!(target_os = "macos"),
            toolkit_level: 64.0,
            table_rows: std::rc::Rc::new(crate::pages::table_rows()),
            deck: [
                "Ada Lovelace",
                "Alan Turing",
                "Grace Hopper",
                "Edsger Dijkstra",
                "Barbara Liskov",
                "Donald Knuth",
            ]
            .map(String::from)
            .to_vec(),
            ..Default::default()
        };
        let root = ui.root();
        ui.set_style(root, style! { background: #12151b; color: #d6dae3; font-family: system-ui; font-size: 13px; });
        light(&mut ui, root, "background: #f2f4f7; color: #1f2329");
        // The page scrolls on a narrow screen; the dialog, its backdrop and
        // the toast stay on the root, over it, where scrolling does not
        // move them. Its columns follow the window's width.
        // A header over the navigation and the page showing; the dialog, its
        // backdrop and the notifications stay on the root, over it.
        let page = ui.add_box(root, style! { display: flex; flex-direction: column; gap: 12px; padding: 12px; height: 100%; box-sizing: border-box; });

        // Header: buttons with state styles and tooltips, and the keymap.
        let header = ui.add_box(
            page,
            style! { min-height: 46px; flex-shrink: 0; display: flex; align-items: center; gap: 10px; padding: 0 14px; border-radius: 10px; background: #1b1f27; border: 1px solid #2a303c; },
        );
        light(
            &mut ui,
            header,
            "background: #ffffff; border: 1px solid #e2e5ea",
        );
        // The longest title that fits.
        let title = ui.add_box(
            header,
            style! { font-size: 15px; font-weight: bold; flex-grow: 1; min-width: 0; },
        );
        for t in ["threers-ui — a retained UI in Rust", "threers-ui"] {
            let b = ui.add_box(title, style! {});
            ui.add_text(b, t);
        }
        ui.view_that_fits(title);
        // A menu of everything, opened from a button: its theme submenu
        // sets the app-wide setting.
        input
            .keymap
            .bind("secondary-k secondary-s", Save, None)
            .expect("binds");
        input
            .keymap
            .bind("secondary-shift-p", OpenPalette, None)
            .expect("binds");
        // In a browser `secondary` is Ctrl; on a Mac, ⌘ is expected too.
        input
            .keymap
            .bind("cmd-shift-p", OpenPalette, None)
            .expect("binds");
        let mac = cfg!(target_os = "macos");
        let save_keys = input.keymap.label_for("Save", mac).unwrap_or_default();
        let palette_keys = input
            .keymap
            .label_for("OpenPalette", mac)
            .unwrap_or_default();
        let more = button(&mut ui, header, "⋯", false);
        ui.semantics_mut(more).label = Some("Menu".into());
        popover_menu(
            &mut input,
            more,
            move |_| {
                vec![
                    MenuItem::action("palette", "Command Palette…").shortcut(palette_keys.clone()),
                    MenuItem::separator(),
                    MenuItem::action("stats", "Frame Stats").shortcut("F"),
                    MenuItem::action("inspect", "Inspector").shortcut("I"),
                    MenuItem::action("keyboard", "On-Screen Keyboard"),
                    MenuItem::submenu(
                        "Theme",
                        vec![
                            MenuItem::action("theme-system", "System"),
                            MenuItem::action("theme-light", "Light"),
                            MenuItem::action("theme-dark", "Dark"),
                        ],
                    ),
                    MenuItem::separator(),
                    MenuItem::action("settings", "Settings…"),
                    MenuItem::action("clear", "Clear the Name…").destructive(true),
                    MenuItem::action("save", "Save").shortcut(save_keys.clone()),
                ]
            },
            |s: &mut State, _, id| s.command = Some(id.to_owned()),
        );
        let settings = button(&mut ui, header, "Settings…", false);
        let save = button(&mut ui, header, "Save", true);
        for (b, tip) in [
            (settings, "Open the settings dialog"),
            (save, "Save  (⌘/Ctrl-K ⌘/Ctrl-S)"),
        ] {
            input.tooltip(b, move |_, ui, root| {
                let t = ui.add_box(
                    root,
                    style! { position: absolute; padding: 4px 8px; border-radius: 4px; background: #e5e7eb; color: #111827; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.4); },
                );
                ui.add_text(t, tip);
                t
            });
        }
        input.on_click(save, |s: &mut State, _, _| s.toast = Some("Saved".into()));
        input.on_click(settings, |s: &mut State, _, _| s.dialog_open = true);
        input
            .keymap
            .bind("escape", CloseDialog, Some("Dialog"))
            .expect("binds");
        input.on_action(root, |s: &mut State, _, _: &Save, _| {
            s.toast = Some("Saved from the keyboard".into())
        });
        input.on_action(root, |s: &mut State, _, _: &OpenPalette, _| {
            s.command = Some("palette".into())
        });
        ui.set_global(Settings {
            theme: Theme::System,
        });

        // The navigation and the pages.
        let mut shell = crate::pages::Shell::new(&mut ui, &mut input, &state, page);
        let pg = |k: usize| shell.pages[k];
        // Basics: a board whose cards drag between columns.
        let board = panel(&mut ui, pg(crate::pages::BASICS), "DRAG AND DROP");
        ui.update_style(board, |s| {
            s.min_height = threers_ui::Length::Px(260.0);
            s.flex_shrink = 0.0;
        });
        // Its cards are dragged, not their text selected.
        ui.update_style(board, |s| s.user_select = threers_ui::UserSelect::None);
        let lanes = ui.add_box(
            board,
            style! { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; flex-grow: 1; },
        );
        let mut cards = vec![];
        let mut lanes_list = vec![];
        for (k, (name, items)) in [
            ("To do", ["Grid layout", "Scrollbars", "IME"]),
            ("Done", ["Shadows", "Tooltips", "Springs"]),
        ]
        .into_iter()
        .enumerate()
        {
            let lane = ui.add_box(lanes, style! { display: flex; flex-direction: column; gap: 6px; padding: 6px; border-radius: 8px; background: #161a21; min-height: 120px; });
            let h = ui.add_box(
                lane,
                style! { font-size: 11px; color: #8b93a5; padding: 2px 4px; },
            );
            ui.token_style(
                h,
                &[("color", threers_ui::environment::Token::SecondaryText)],
            );
            ui.add_text(h, name);
            ui.set_semantics(lane, Semantics::new(Role::List).label(name));
            for item in items {
                let card = ui.add_box(
                    lane,
                    style! { padding: 8px; border-radius: 6px; background: #232833; box-shadow: 0 1px 3px rgba(0,0,0,0.6); cursor: grab; },
                );
                ui.add_text(card, item);
                ui.set_semantics(card, Semantics::new(Role::ListItem).label(item));
                light(&mut ui, card, "background: #ffffff; color: #1f2329");
                ui.hover_style(card, "background: rgba(128,138,160,0.3)");
                input.draggable(card, move |_, _| Some(Card(card)));
                cards.push(card);
            }
            ui.group(lane, "lane");
            lanes_list.push(lane);
            input.on_drop::<Card>(lane, move |s: &mut State, ui, card, _| {
                // Into the slot that showed where it would go.
                let at = ui.children(lane).iter().position(|&c| Some(c) == s.slot);
                ui.reparent(card.0, lane);
                let at = at.unwrap_or(ui.children(lane).len() - 1);
                ui.move_child(lane, card.0, at);
                s.log.push(format!(
                    "moved a card to {} at {}",
                    if k == 0 { "To do" } else { "Done" },
                    at
                ));
            });
        }
        // The slot a dragged card would drop into, moved from place to place
        // as it is dragged: the cards around make room for it.
        let slot = ui.add_box(
            root,
            style! { display: none; border-radius: 6px; border: 1px solid rgb(37, 99, 235); background: rgba(59,130,246,0.15); },
        );
        // The preview: the card itself, lifted, as wide as the card.
        input.drag_preview_with(|_, ui, _, source, root| {
            let width = ui.rect(source).width;
            let label = ui.children(source).first().map(|&t| match ui.kind(t) {
                threers_ui::NodeKind::Text(t) => t.clone(),
                _ => String::new(),
            });
            let mut s = style! { position: absolute; padding: 8px; border-radius: 6px; background: #2f3747; color: #fcfcfc; box-shadow: 0 10px 24px rgba(0,0,0,0.55); pointer-events: none; };
            s.width = threers_ui::Length::Px(width);
            s.box_sizing = threers_ui::BoxSizing::BorderBox;
            let copy = ui.add_box(root, s);
            ui.add_text(copy, label.unwrap_or_default());
            copy
        });

        // Lists: virtual lists of several kinds, built as they show.
        let middle = panel(&mut ui, pg(crate::pages::LISTS), "VIRTUAL LISTS");
        ui.update_style(middle, |s| {
            s.height = threers_ui::Length::Px(520.0);
            s.flex_shrink = 0.0;
        });
        let lists = crate::lists::Lists::new(&mut ui, &mut input, middle);

        // Text: rich text, an image, a responsive card, and a text field.
        let right = crate::showcase::group(&mut ui, pg(crate::pages::TEXT), "Rich text, an image and a field", "Click or tap the link; select any text; edit the field (double-click a word, or on a phone use the handles and the edit menu).");
        // Each panel's layout is kept while nothing in it changes: scrolling
        // the list lays out the list's panel alone.
        for p in [header, board, middle] {
            ui.cache_layout(p, true);
        }
        let showcase = crate::showcase::build(
            &mut ui,
            &mut input,
            &state,
            &crate::showcase::Pages {
                basics: pg(crate::pages::BASICS),
                components: pg(crate::pages::COMPONENTS),
                screens: pg(crate::pages::SCREENS),
                depth: pg(crate::pages::DEPTH),
            },
        );
        let shaders = crate::shaders::build(&mut ui, &mut input, &state, pg(crate::pages::SHADERS));
        let progress =
            crate::progress::build(&mut ui, &mut input, &state, pg(crate::pages::PROGRESS));
        let toolkit = crate::toolkit::build(&mut ui, &mut input, &state, pg(crate::pages::TOOLKIT));
        let tailwind =
            crate::tailwind::build(&mut ui, &mut input, &state, pg(crate::pages::TAILWIND));
        let tasks_view = threers_ui::view::View::new(
            &mut ui,
            showcase.tasks,
            "display: flex; flex-direction: column; gap: 2px",
        );
        let para = ui.add_box(
            right,
            style! { white-space: normal; line-height: 19px; color: #c3c8d2; },
        );
        light(&mut ui, para, "color: #374151");
        let words = "Rich text has runs: a keyword, a mispelled word, and a link, each in its own style, wrapping as one.";
        let t = ui.add_text(para, words);
        let link_text = t;
        // A click or tap on it: the frame finds the link under it.
        input.on(para, Phase::Bubble, |s: &mut State, _, e, _| {
            if let Event::Click { x, y, .. } = e {
                s.clicked_at = Some((*x, *y));
            }
        });
        let at = |w: &str| words.find(w).expect("in the text");
        let (kw, typo, link) = (at("keyword"), at("mispelled"), at("link"));
        ui.set_text_runs(
            t,
            vec![
                TextRun::new(kw),
                TextRun::new("keyword".len())
                    .color([0.78, 0.47, 0.87, 1.0])
                    .weight(threers_ui::FontWeight::Bold),
                TextRun::new(typo - kw - "keyword".len()),
                TextRun::new("mispelled".len())
                    .underline(true)
                    .decoration_color([0.94, 0.33, 0.31, 1.0]),
                TextRun::new(link - typo - "mispelled".len()),
                TextRun::new("link".len())
                    .color([0.38, 0.65, 0.98, 1.0])
                    .underline(false)
                    .link("https://example.com/"),
                TextRun::new(words.len() - link - "link".len()),
            ],
        );
        let media = ui.add_box(
            right,
            style! { display: flex; gap: 10px; align-items: center; },
        );
        ui.add_image(
            media,
            style! { width: 96px; height: 64px; border-radius: 6px; object-fit: cover; },
            picture(),
        );
        let card = ui.add_box(media, style! { display: flex; flex-direction: column; gap: 4px; flex-grow: 1; padding: 8px; border-radius: 6px; background: #232833; });
        ui.container_query(card, media, vec![(520.0, style! { display: flex; flex-direction: row; gap: 8px; flex-grow: 1; padding: 8px; border-radius: 6px; background: #2a3448; })]);
        light(&mut ui, card, "background: #eef1f6");
        for line in [
            "Container query:",
            "this card turns into a row",
            "when its panel is wide.",
        ] {
            let l = ui.add_box(card, style! { font-size: 12px; });
            ui.add_text(l, line);
        }
        let keys = ui.add_box(
            right,
            style! { white-space: normal; font-size: 12px; color: #8b93a5; },
        );
        ui.token_style(
            keys,
            &[("color", threers_ui::environment::Token::SecondaryText)],
        );
        ui.add_text(keys, "Keys: Tab moves focus, F shows frame stats, I inspects, ⌘/Ctrl-K ⌘/Ctrl-S saves, ⌘/Ctrl-⇧-P opens the command palette, arrows move through a focused list.");
        let field_box = ui.add_box(
            right,
            style! { position: relative; padding: 6px 8px; border-radius: 6px; border: 1px solid #3a4150; background: #0f1218; cursor: text; overflow: hidden; },
        );
        ui.set_semantics(field_box, Semantics::new(Role::TextBox).label("Name"));
        light(
            &mut ui,
            field_box,
            "background: #ffffff; border: 1px solid #c9ced6; color: #1f2329",
        );
        ui.focus_style(field_box, "border: 1px solid #2563eb");
        let field_text = ui.add_text(field_box, "");
        input.on_key(field_box, |s: &mut State, _, key, m, _| {
            !matches!(
                s.field.key(key, m, |_| true),
                FieldKey::Ignored | FieldKey::Tab { .. }
            )
        });
        input.on(field_box, Phase::Bubble, |s: &mut State, _, e, _| {
            if let Event::Ime(ime) = e {
                s.field.ime(ime);
            }
        });
        // The page scrolls: what is on it keeps its height rather than being
        // squeezed to fit.
        for c in ui.children(right).to_vec() {
            ui.update_style(c, |s| s.flex_shrink = 0.0);
        }
        let audit_view = threers_ui::view::View::new(
            &mut ui,
            shell.audit,
            "display: flex; flex-direction: column; gap: 2px",
        );
        shell.apply(&mut ui, &state);

        // The dialog: modal, over a backdrop, closed by a click outside.
        let backdrop = ui.add_box(root, style! { display: none; position: absolute; left: 0; top: 0; width: 4000px; height: 4000px; background: rgba(0,0,0,0.5); });
        ui.defer(backdrop, 10);
        let dialog = ui.add_box(
            root,
            style! { display: none; position: absolute; width: 300px; flex-direction: column; gap: 12px; padding: 18px; border-radius: 12px; background: #1f2430; border: 1px solid #3a4150; box-shadow: 0 12px 32px rgba(0,0,0,0.6); },
        );
        ui.defer(dialog, 11);
        light(&mut ui, dialog, "background: #ffffff; border: 1px solid #d5d9e0; box-shadow: 0 12px 32px rgba(0,0,0,0.2)");
        ui.set_semantics(dialog, Semantics::new(Role::Dialog).label("Settings"));
        input.set_key_context(dialog, "Dialog");
        let h = ui.add_box(dialog, style! { font-size: 15px; font-weight: bold; });
        ui.add_text(h, "Settings");
        // Bound to the same state as the components section's switch.
        threers_ui::components::switch(
            &mut ui,
            &mut input,
            &state,
            dialog,
            "Card shadows",
            threers_ui::binding!(|s: State| s.shadows),
        );
        // Selecting text, anywhere in the demo (`Ui::set_text_selection`).
        let check = ui.add_box(
            dialog,
            style! { display: flex; gap: 8px; align-items: center; cursor: pointer; },
        );
        let tick = ui.add_box(check, style! { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #5b6478; background: rgb(37, 99, 235); });
        ui.add_text(check, "Selectable text");
        ui.set_semantics(
            check,
            Semantics::new(Role::Checkbox)
                .label("Selectable text")
                .checked(true),
        );
        input.on_click(check, move |_: &mut State, ui, _| {
            let on = !ui.text_selection();
            ui.set_text_selection(on);
            ui.semantics_mut(check).checked = Some(on);
            ui.style_mut(tick).background = on.then_some([0.145, 0.388, 0.922, 1.0]);
        });
        let row = ui.add_box(
            dialog,
            style! { display: flex; gap: 8px; justify-content: flex-end; },
        );
        let cancel = button(&mut ui, row, "Cancel", false);
        let ok = button(&mut ui, row, "OK", true);
        input.on_click(cancel, |s: &mut State, _, _| s.dialog_open = false);
        input.on_click(ok, |s: &mut State, _, _| {
            s.dialog_open = false;
            s.toast = Some("Settings saved".into());
        });
        input.on_click_outside(dialog, |s: &mut State, _, _, _| s.dialog_open = false);
        input.on_action(dialog, |s: &mut State, _, _: &CloseDialog, _| {
            s.dialog_open = false
        });

        // Asked while the state says so, and the state told when answered.
        input.present(
            |s: &State| s.confirm_clear,
            |s: &mut State| s.confirm_clear = false,
            |ui, input, s| {
                use threers_ui::button::ButtonRole;
                let buttons = [
                    ("Cancel", ButtonRole::Cancel),
                    ("Clear", ButtonRole::Destructive),
                ];
                threers_ui::prompt::prompt_with(
                    ui,
                    input,
                    s,
                    "Clear the name?",
                    "Its text will be lost.",
                    &buttons,
                    |s: &mut State, _, answer| {
                        if answer == 1 {
                            s.field = TextField::new("");
                            s.toast = Some("Cleared".into());
                        }
                    },
                )
                .dialog
            },
        );
        // A sheet, while the state says so.
        threers_ui::sheet::sheet(
            &mut input,
            threers_ui::binding!(|s: State| s.sheet_open),
            vec![
                threers_ui::sheet::Detent::Medium,
                threers_ui::sheet::Detent::Large,
            ],
            "Share",
            |ui, input, _, c| {
                let h = ui.add_box(c, style! { font-size: 15px; font-weight: bold; });
                ui.add_text(h, "Share");
                for (i, name) in ["Ada Lovelace", "Alan Turing", "Grace Hopper"]
                    .into_iter()
                    .enumerate()
                {
                    let row =
                        ui.add_box(c, style! { display: flex; align-items: center; gap: 8px; });
                    threers_ui::components::avatar(ui, row, name, 28.0);
                    ui.add_text(row, name);
                    let _ = i;
                }
                let hint = ui.add_box(c, style! { font-size: 12px; white-space: normal; });
                ui.token_style(
                    hint,
                    &[("color", threers_ui::environment::Token::SecondaryText)],
                );
                ui.add_text(hint, "Drag the grabber up for more room, or down to close.");
                let done = threers_ui::button::button(
                    ui,
                    input,
                    c,
                    "Done",
                    threers_ui::button::ButtonRole::Primary,
                    |s: &mut State, _| s.sheet_open = false,
                );
                let _ = done;
            },
        );
        // The page keeps clear of a phone's notch and home indicator.
        ui.pad_safe_area(page, [true, true, true, true]);
        // The field is a text control a host mirrors (a phone's keyboard
        // types into it through the page's input), as the search box is.
        input.register_text(
            field_box,
            |s: &State| (s.field.text.clone(), s.field.anchor, s.field.caret),
            |s: &mut State, _, text, anchor, caret| {
                let n = text.chars().count();
                s.field.text = text;
                s.field.anchor = anchor.min(n);
                s.field.caret = caret.min(n);
                s.field.marked = None;
            },
        );
        // An on-screen keyboard, for touch screens without one of their own.
        let keyboard = threers_ui::virtual_keyboard::VirtualKeyboard::new(&mut ui, &mut input);
        // The field's own button brings it up for the field, on or not.
        keyboard.attach_button(&mut ui, &mut input, field_box);
        // Glare and shading follow the pointer, as if it held the light.
        input.light_follows_pointer();
        // Notifications, stacked in the corner.
        let notices = Notifications::new(&mut ui);

        // The field edited, and the page's text selected, with the pointer
        // and a finger.
        let mut editing = TextEditing::new(&mut ui, &mut input);
        editing.add_field(&mut ui, field_box, field_text, |s: &mut State| &mut s.field);

        let _ = ACCENT;
        App {
            ui,
            input,
            state,
            nodes: Nodes {
                field_box,
                field_text,
                dialog,
                backdrop,
                cards,
                lanes: lanes_list,
                slot,
            },
            lists,
            stats: FrameStats::default(),
            last: None,
            refresh_until: None,
            link_nodes: vec![link_text],
            links: vec![],
            editing,
            inspecting: false,
            show_stats: false,
            viewport: (1.0, 1.0),
            drag_height: 0.0,
            layouts_last: 0,
            shadows_drawn: None,
            notices,
            showcase,
            nav: ListNavigation::default(),
            nav_tab: 0,
            marked: None,
            tasks_view,
            shell,
            audit_view,
            audit_at: f64::NEG_INFINITY,
            pending_link: None,
            followed: None,
            shaders,
            progress,
            toolkit,
            tailwind,
            storage: None,
            stored_theme: None,
            state_theme: Theme::System,
            keyboard,
        }
    }

    /// Runs one frame: timers and animations, the app's state into the
    /// tree, layout (again while the list builds rows), at `width × height`
    /// CSS pixels, `time` seconds on.
    pub fn frame(&mut self, time: f64, width: f32, height: f32, measure: &mut dyn TextMeasure) {
        self.frame_timed(time, width, height, measure, &|| 0.0);
    }

    /// Follows a link into the demo, as a page's `#fragment`: `page`,
    /// `page/section` or `section` (on the page showing), each a name as
    /// [`slug`](threers_ui::anchors::slug) makes it — `toolkit`,
    /// `toolkit/gauges-and-charts`, `3d`. The page shows at once; the
    /// section is scrolled to and focused once laid out. Returns whether
    /// anything in it was found.
    pub fn follow_link(&mut self, link: &str) -> bool {
        use threers_ui::anchors::slug;
        let link = link.trim().trim_start_matches('#');
        if link.is_empty() {
            return false;
        }
        let (first, rest) = link.split_once('/').unwrap_or((link, ""));
        let page = crate::pages::PAGES
            .iter()
            .position(|(name, _)| slug(name) == first.to_lowercase());
        let section = match page {
            Some(p) => {
                self.state.page = p;
                rest
            }
            None => link,
        };
        if !section.is_empty() {
            self.pending_link = Some((section.to_owned(), 0));
        }
        page.is_some() || !section.is_empty()
    }

    /// Where the demo is, as a link [`follow_link`](Self::follow_link)
    /// takes back: the page, and the section focus is in (a heading's, or
    /// a named node's) if it is in one.
    pub fn link(&self) -> String {
        use threers_ui::anchors::slug;
        let page = slug(crate::pages::PAGES[self.state.page.min(crate::pages::PAGES.len() - 1)].0);
        // A link on its way stays as it was asked for.
        if let Some((section, _)) = &self.pending_link {
            return format!("{page}/{section}");
        }
        let section = self
            .input
            .focused()
            .filter(|&f| self.ui.contains(f))
            .and_then(|f| self.ui.ancestors(f).find_map(|n| self.ui.anchor_name(n)));
        match section {
            Some(s) => format!("{page}/{s}"),
            None => page,
        }
    }

    /// The node a link went to since the last call: for a mirror of the
    /// accessibility tree to move the page's focus there too.
    pub fn take_followed(&mut self) -> Option<NodeId> {
        self.followed.take()
    }

    /// [`frame`](Self::frame), timing layout with `clock` (milliseconds).
    pub fn frame_timed(
        &mut self,
        time: f64,
        width: f32,
        height: f32,
        measure: &mut dyn TextMeasure,
        clock: &dyn Fn() -> f64,
    ) {
        let started = clock();
        self.layouts_last = 0;
        let dt = self.last.map_or(0.0, |l| (time - l).max(0.0));
        self.last = Some(time);
        self.input.tick(&mut self.ui, &mut self.state, time);
        self.editing
            .tick(&mut self.ui, &self.input, &mut self.state, time, measure);
        self.run_commands();
        // On a phone-sized window, a budget for the shader effects (the
        // Progress page's dozens of moving indicators): the largest few at
        // up to 30 frames a second, the rest slower.
        let budget = if width < 700.0 {
            threers_ui::shader::AnimationBudget::PHONE
        } else {
            threers_ui::shader::AnimationBudget::UNLIMITED
        };
        if self.ui.animation_budget() != budget {
            self.ui.set_animation_budget(budget);
        }
        // The theme: the picker's (in the state) or the menu's and the
        // palette's (the global), whichever changed last.
        if let Some(global) = self.ui.global::<Settings>().map(|g| g.read().theme) {
            if self.state.theme != self.state_theme {
                let t = self.state.theme;
                self.ui.update_global(|g: &mut Settings| g.theme = t);
            } else if global != self.state.theme {
                self.state.theme = global;
            }
            self.state_theme = self.state.theme;
            if self.state.theme != self.ui.theme() {
                self.ui.set_theme(self.state.theme);
            }
        }
        // The theme kept for the next run.
        let theme = self.ui.theme();
        if let Some(store) = &self.storage
            && self.stored_theme != Some(theme)
        {
            store.set("theme", &format!("{theme:?}"));
            self.stored_theme = Some(theme);
        }
        // A list let go of past its pull: a second's pretend load, then new
        // content and the pull closing.
        if let Some(tab) = self.state.refreshing {
            let until = *self.refresh_until.get_or_insert(time + 1.0);
            if time >= until {
                self.state.refreshes += 1;
                self.state.refreshing = None;
                self.refresh_until = None;
                self.input
                    .finish_refresh(self.lists.lists[tab].1.container());
            }
        }
        self.ui.step(dt as f32);
        self.apply_state();
        self.apply_controls();
        self.apply_depth();
        self.shaders.update(&mut self.ui, &self.state);
        self.progress.update(&mut self.ui, &self.state);
        self.toolkit.update(&mut self.ui, &self.input, &self.state);
        self.tailwind.update(&mut self.ui, &self.state);
        self.shell.apply(&mut self.ui, &self.state);
        self.apply_access(time);
        let tasks = crate::showcase::tasks(&self.state);
        self.tasks_view.update(&mut self.ui, &mut self.input, tasks);
        self.lists.apply(&mut self.ui, &mut self.state);
        let active = self.lists.active(&self.state).container();
        let pull = self.input.pull(active);
        self.lists.show_pull(&mut self.ui, pull);
        let swipe = self.input.swipe(active);
        self.lists.show_swipe(&mut self.ui, swipe);
        self.ui.layout(width, height, measure);
        self.layouts_last += 1;
        for _ in 0..4 {
            let changed = self.lists.sync(&mut self.ui, &self.state);
            if !changed {
                break;
            }
            self.ui.layout(width, height, measure);
            self.layouts_last += 1;
        }
        // A link's section, once its page is laid out (a few frames for
        // what builds as it shows, then given up).
        if let Some((section, waited)) = self.pending_link.take() {
            match self
                .input
                .go_to_anchor(&mut self.ui, &mut self.state, &section)
            {
                Some(n) => {
                    self.followed = Some(n);
                    self.ui.layout(width, height, measure);
                    self.layouts_last += 1;
                }
                None if waited < 3 => self.pending_link = Some((section, waited + 1)),
                None => {}
            }
        }
        // The window changed size (on a phone, the keyboard came up and
        // took the bottom of it): what has focus stays in view, as a browser
        // keeps a focused input in view.
        if self.viewport != (width, height)
            && let Some(f) = self.input.focused().filter(|&f| self.ui.contains(f))
        {
            self.ui.scroll_into_view(f);
            self.ui.layout(width, height, measure);
            self.layouts_last += 1;
        }
        self.editing
            .place(&mut self.ui, &mut self.input, &mut self.state, measure);
        // The dialog in the middle of the window.
        let d = self.ui.rect(self.nodes.dialog);
        let (left, top) = (
            ((width - d.width) / 2.0).max(0.0),
            ((height - d.height) / 3.0).max(0.0),
        );
        if self.ui.style(self.nodes.dialog).left != threers_ui::Length::Px(left)
            || self.ui.style(self.nodes.dialog).top != threers_ui::Length::Px(top)
        {
            self.ui.update_style(self.nodes.dialog, |s| {
                s.left = threers_ui::Length::Px(left);
                s.top = threers_ui::Length::Px(top);
            });
        }
        self.ui.layout(width, height, measure);
        self.layouts_last += 1;
        // The edit menu is centred by its width, which is known only once it
        // is laid out: placed again now, in the same frame, it does not move
        // under the finger about to tap it. (A layout with nothing changed
        // does nothing.)
        self.editing
            .place(&mut self.ui, &mut self.input, &mut self.state, measure);
        self.ui.layout(width, height, measure);
        self.mark_highlighted();
        self.links = self
            .link_nodes
            .iter()
            .flat_map(|&n| self.ui.link_rects(n, measure))
            .collect();
        let mut sample =
            FrameSample::of(&self.ui, (dt * 1000.0) as f32, (clock() - started) as f32);
        if self.show_stats {
            sample.painted = self.ui.scene().items.len();
        }
        self.stats.record(sample);
        self.viewport = (width, height);
    }

    /// The app's state into the tree: the dialog, notifications, the cards.
    fn apply_state(&mut self) {
        let n = &self.nodes;
        let open = self.state.dialog_open;
        let shown = |d| if open { d } else { threers_ui::Display::None };
        if self.ui.style(n.dialog).display != shown(threers_ui::Display::Flex) {
            self.ui
                .update_style(n.dialog, |s| s.display = shown(threers_ui::Display::Flex));
            self.ui.update_style(n.backdrop, |s| {
                s.display = shown(threers_ui::Display::Block)
            });
            let modal = open.then_some(n.dialog);
            self.input.set_modal(&mut self.ui, &mut self.state, modal);
        }
        if let Some(msg) = self.state.toast.take()
            && !self.state.quiet
        {
            self.notices.push(
                &mut self.ui,
                &mut self.input,
                Notice::new(msg, "").timeout(Some(3.0)),
                None,
            );
        }
        // Through the cards' base styles (they have hover and light ones),
        // and only when the setting changes.
        if self.shadows_drawn != Some(self.state.shadows) {
            self.shadows_drawn = Some(self.state.shadows);
            let shadow = if self.state.shadows {
                "0 1px 3px rgba(0,0,0,0.6)"
            } else {
                "none"
            };
            for &c in &n.cards {
                self.ui
                    .update_style(c, |s| threers_ui::css::apply(s, "box-shadow", shadow));
            }
        }
        // While a card is dragged it leaves its place, and a slot opens
        // where it would land: in the lane under the pointer, before the
        // first card whose middle is below it. Set only when they change.
        let (from, over) = (self.input.drag_source(), self.input.drag_over());
        for &c in &n.cards {
            let d = if from == Some(c) {
                threers_ui::Display::None
            } else {
                threers_ui::Display::Block
            };
            if self.ui.style(c).display != d {
                self.ui.update_style(c, |s| s.display = d);
            }
        }
        match (from, over) {
            (Some(card), Some(lane)) => {
                let (_, py) = self.input.pointer_position();
                let height = self.ui.rect(card).height.max(self.drag_height);
                self.drag_height = height;
                let children: Vec<NodeId> = self.ui.children(lane).to_vec();
                // Lane children: its heading, then cards (and the slot).
                let mut at = children.len();
                for (i, &c) in children.iter().enumerate().skip(1) {
                    if c == n.slot || c == card || !n.cards.contains(&c) {
                        continue;
                    }
                    let r = self.ui.rect(c);
                    if py < r.y + r.height / 2.0 {
                        at = i;
                        break;
                    }
                }
                if self.ui.parent(n.slot) != Some(lane) {
                    self.ui.reparent(n.slot, lane);
                }
                let now = self
                    .ui
                    .children(lane)
                    .iter()
                    .position(|&c| c == n.slot)
                    .expect("in the lane");
                // Moving the slot out from before `at` shifts `at` down one.
                let to = if now < at { at - 1 } else { at };
                if now != to {
                    self.ui.move_child(lane, n.slot, to);
                }
                let mut s = self.ui.style(n.slot).clone();
                s.display = threers_ui::Display::Block;
                s.height = threers_ui::Length::Px(height);
                self.ui.set_style(n.slot, s);
                self.state.slot = Some(n.slot);
            }
            _ => {
                if self.ui.style(n.slot).display != threers_ui::Display::None {
                    self.ui.style_mut(n.slot).display = threers_ui::Display::None;
                }
                if from.is_none() {
                    self.drag_height = 0.0;
                }
                self.state.slot = None;
            }
        }
        for &l in &n.lanes {
            let rest = if self.ui.appearance() == threers_ui::states::Appearance::Light {
                [0.925, 0.933, 0.949, 1.0]
            } else {
                [0.086, 0.102, 0.129, 1.0]
            };
            let bg = Some(if over == Some(l) {
                [0.145, 0.388, 0.922, 0.16]
            } else {
                rest
            });
            if self.ui.style(l).background != bg {
                self.ui.style_mut(l).background = bg;
            }
        }
    }

    /// Paints the frame, the inspector's outline and the stats over it.
    pub fn paint(&self, painter: &mut dyn Painter) {
        self.ui.paint(painter);
        if self.inspecting
            && let Some(h) = self.input.hovered()
        {
            self.ui.paint_highlight(h, painter);
        }
        // The list row the keyboard is on, while its list has focus.
        if let Some(row) = self.highlighted_row() {
            self.ui.paint_highlight(row, painter);
        }
        if self.show_stats {
            self.stats.paint(painter, 16.0, self.viewport.1 - 120.0);
        }
    }

    pub fn pointer_move(&mut self, x: f32, y: f32, measure: &mut dyn TextMeasure) {
        self.editing.pointer_move(
            &mut self.ui,
            &mut self.input,
            &mut self.state,
            x,
            y,
            measure,
        );
    }

    pub fn pointer_down(
        &mut self,
        x: f32,
        y: f32,
        button: PointerButton,
        shift: bool,
        measure: &mut dyn TextMeasure,
    ) {
        if self.inspecting {
            if let Some(i) = self.ui.inspect(x, y) {
                self.state.log.push(i.to_string());
            }
            return;
        }
        self.editing.pointer_down(
            &mut self.ui,
            &mut self.input,
            &mut self.state,
            x,
            y,
            button,
            shift,
            measure,
        );
    }

    pub fn pointer_up(&mut self, x: f32, y: f32, button: PointerButton) {
        self.editing
            .pointer_up(&mut self.ui, &mut self.input, &mut self.state, x, y, button);
        self.take_requests();
        self.take_link_click();
        self.run_commands();
    }

    pub fn wheel(&mut self, x: f32, y: f32, dx: f32, dy: f32) {
        self.input
            .dispatch(&mut self.ui, &mut self.state, Event::Wheel { x, y, dx, dy });
    }

    /// A key; returns whether the app used it (and the page should not).
    pub fn key(&mut self, key: &str, modifiers: KeyModifiers) -> bool {
        // ⌘/Ctrl-C copies the page's selected text.
        if self.editing.key(key, modifiers) {
            self.take_requests();
            return true;
        }
        if self.list_key(key, modifiers) {
            return true;
        }
        let field_focused = self.input.focused() == Some(self.nodes.field_box);
        if !field_focused && !modifiers.control && !modifiers.meta {
            match key {
                "f" | "F" => {
                    self.show_stats = !self.show_stats;
                    return true;
                }
                "i" | "I" => {
                    self.inspecting = !self.inspecting;
                    return true;
                }
                _ => {}
            }
        }
        let out = self.input.dispatch(
            &mut self.ui,
            &mut self.state,
            Event::Key {
                key: key.to_owned(),
                modifiers,
            },
        );
        self.run_commands();
        out.handled || out.action.is_some() || out.focus_changed || key == "Tab"
    }

    pub fn event(&mut self, event: Event) {
        self.input.dispatch(&mut self.ui, &mut self.state, event);
        self.run_commands();
    }

    /// The commands asked for since the last call (by the menu, the
    /// palette, buttons), run.
    fn run_commands(&mut self) {
        while let Some(c) = self.state.command.take() {
            self.run(&c);
        }
    }

    fn run(&mut self, command: &str) {
        let (ui, input, state) = (&mut self.ui, &mut self.input, &mut self.state);
        let theme = |ui: &mut Ui, t: Theme| {
            ui.update_global(|g: &mut Settings| g.theme = t);
        };
        match command {
            "palette" => {
                let mac = cfg!(target_os = "macos");
                let label = |a: &str| input.keymap.label_for(a, mac);
                let with = |c: Command, a: &str| match label(a) {
                    Some(l) => c.shortcut(l),
                    None => c,
                };
                let commands = vec![
                    with(Command::new("save", "Save"), "Save"),
                    Command::new("settings", "Open Settings"),
                    Command::new("stats", "Toggle Frame Stats").shortcut("F"),
                    Command::new("inspect", "Toggle Inspector").shortcut("I"),
                    Command::new("theme-system", "Theme: System"),
                    Command::new("theme-light", "Theme: Light"),
                    Command::new("theme-dark", "Theme: Dark"),
                    Command::new("notify", "Show a Notification"),
                    Command::new("error", "Show an Error"),
                    Command::new("clear", "Clear the Name…"),
                    Command::new("text-larger", "Larger Text"),
                    Command::new("text-smaller", "Smaller Text"),
                    Command::new("reduce-motion", "Toggle Reduced Motion"),
                    Command::new("rtl", "Toggle Right to Left"),
                    Command::new("sheet", "Open the Share Sheet"),
                    Command::new("keyboard", "Toggle On-Screen Keyboard"),
                ];
                open_palette(ui, input, state, commands, |s: &mut State, _, id| {
                    s.command = Some(id.to_owned())
                });
            }
            "save" => state.toast = Some("Saved".into()),
            "settings" => state.dialog_open = true,
            "stats" => self.show_stats = !self.show_stats,
            "inspect" => self.inspecting = !self.inspecting,
            "theme-system" => theme(ui, Theme::System),
            "theme-light" => theme(ui, Theme::Light),
            "theme-dark" => theme(ui, Theme::Dark),
            "notify" => {
                let n = Notice::new("Build finished", "3 crates compiled in 4.2 s.")
                    .tone(Tone::Success)
                    .action("Show log");
                self.notices.push(
                    ui,
                    input,
                    n,
                    Some(Box::new(|s: &mut State, _: &mut Ui| {
                        s.log.push("the build log".into())
                    })),
                );
            }
            "error" => {
                let n = Notice::new(
                    "Couldn't sync",
                    "The server didn't answer. It stays until closed.",
                )
                .tone(Tone::Error);
                self.notices.push(ui, input, n, None);
            }
            // Shown by the state (see `Input::present` in `new`).
            "clear" => state.confirm_clear = true,
            "sheet" => state.sheet_open = true,
            "keyboard" => {
                let on = !self.keyboard.enabled();
                self.keyboard.set_enabled(on);
                if let Some(store) = &self.storage {
                    store.set("keyboard", if on { "on" } else { "off" });
                }
                state.toast = Some(
                    if on {
                        "On-screen keyboard on"
                    } else {
                        "On-screen keyboard off"
                    }
                    .into(),
                );
            }
            "rtl" => {
                use threers_ui::environment::LayoutDirection;
                let d = if ui.layout_direction() == LayoutDirection::RightToLeft {
                    LayoutDirection::LeftToRight
                } else {
                    LayoutDirection::RightToLeft
                };
                ui.set_layout_direction(d);
            }
            "text-larger" => ui.set_text_scale((ui.text_scale() + 0.25).min(2.0)),
            "text-smaller" => ui.set_text_scale((ui.text_scale() - 0.25).max(0.75)),
            "reduce-motion" => {
                let on = !ui.reduce_motion();
                ui.set_reduce_motion(on);
                state.toast = Some(if on { "Reduced motion" } else { "Full motion" }.into());
            }
            "details" => {
                let anchor = self.showcase.details;
                threers_ui::components::popover(
                    ui,
                    input,
                    state,
                    anchor,
                    "Details",
                    |ui, panel| {
                        let row = ui.add_box(
                            panel,
                            style! { display: flex; align-items: center; gap: 8px; },
                        );
                        threers_ui::components::avatar(ui, row, "Grace Hopper", 32.0);
                        let who =
                            ui.add_box(row, style! { display: flex; flex-direction: column; });
                        let name = ui.add_box(who, style! { font-weight: bold; });
                        ui.add_text(name, "Grace Hopper");
                        ui.add_text(who, "Rear admiral, compiler pioneer");
                        let note = ui.add_box(
                            panel,
                            style! { font-size: 12px; white-space: normal; color: #8b93a5; },
                        );
                        ui.token_style(
                            note,
                            &[("color", threers_ui::environment::Token::SecondaryText)],
                        );
                        ui.add_text(note, "A popover: anchored under its button, it closes on Escape or a click outside, and gives focus back.");
                    },
                );
            }
            other => state.log.push(format!("no command {other}")),
        }
    }

    /// The Depth tab's settings into the UI: the depth of field's focus,
    /// and the Screens tab's transition.
    fn apply_depth(&mut self) {
        // Focus from the far layer (−120) to the near one (+80).
        let focus = -120.0 + self.state.focus * 200.0;
        self.ui
            .set_depth_of_field(self.showcase.layers, Some((focus, 3.0)));
        let d = crate::showcase::DEPARTURES;
        self.showcase
            .board
            .set(&mut self.ui, d[self.state.departure % d.len()]);
        // The candles flicker while their page shows (and motion is wanted).
        if self.candles_lit() {
            let t = self.last.unwrap_or(0.0) as f32;
            for c in &self.showcase.candles {
                c.update(&mut self.ui, t);
            }
        }
        use threers_ui::nav_stack::NavTransition;
        let t = [
            NavTransition::Slide,
            NavTransition::Cube,
            NavTransition::PageTurn,
        ][self.state.nav_transition.min(2)];
        if self.ui.environment::<NavTransition>(self.showcase.stack) != Some(t) {
            self.ui.set_environment(self.showcase.stack, t);
        }
    }

    /// The Accessibility page's settings into the UI, and its audit (twice
    /// a second while it shows).
    fn apply_access(&mut self, time: f64) {
        let system = if self.state.system_text_scale > 0.0 {
            self.state.system_text_scale
        } else {
            1.0
        };
        let scale = (1.0 + 0.25 * self.state.text_scale as f32) * system;
        if self.ui.text_scale() != scale {
            self.ui.set_text_scale(scale);
        }
        if self.ui.reduce_motion() != self.state.reduce_motion {
            self.ui.set_reduce_motion(self.state.reduce_motion);
        }
        use threers_ui::environment::LayoutDirection;
        let dir = if self.state.right_to_left {
            LayoutDirection::RightToLeft
        } else {
            LayoutDirection::LeftToRight
        };
        if self.ui.layout_direction() != dir {
            self.ui.set_layout_direction(dir);
        }
        if self.state.page == crate::pages::ACCESSIBILITY && time - self.audit_at > 0.5 {
            self.audit_at = time;
            let found = self.ui.accessibility_audit();
            let rows = crate::pages::audit_rows(&found);
            self.audit_view.update(&mut self.ui, &mut self.input, rows);
        }
    }

    /// The page showing: its scrolling box.
    pub fn scroller(&self) -> NodeId {
        self.shell.content
    }

    /// How the device is held (the page's `deviceorientation`), for the
    /// card that leans with it.
    pub fn set_device_tilt(&mut self, beta: f32, gamma: f32) {
        self.ui.set_device_tilt(beta, gamma);
    }

    /// The Controls tab's state into what follows it: the upload bar, the
    /// placeholders, the people the search finds.
    fn apply_controls(&mut self) {
        let sc = &self.showcase;
        sc.progress.set(&mut self.ui, Some(self.state.volume));
        self.ui.set_redacted(sc.people, self.state.loading);
        let q = self.state.query.to_lowercase();
        let mut any = false;
        for (name, row) in &sc.found {
            let hit = !q.is_empty() && name.to_lowercase().contains(&q);
            any |= hit;
            let d = if hit {
                threers_ui::Display::Flex
            } else {
                threers_ui::Display::None
            };
            if self.ui.style(*row).display != d {
                self.ui.update_style(*row, |s| s.display = d);
            }
        }
        let d = if !q.is_empty() && !any {
            threers_ui::Display::Flex
        } else {
            threers_ui::Display::None
        };
        if self.ui.style(sc.nobody).display != d {
            self.ui.update_style(sc.nobody, |s| s.display = d);
        }
    }

    /// Up, down, Home, End, Page Up and Down on the focused list move its
    /// highlighted row (scrolled into view); Enter or Space activates it.
    fn list_key(&mut self, key: &str, modifiers: KeyModifiers) -> bool {
        if modifiers.control || modifiers.meta || modifiers.alt {
            return false;
        }
        let tab = self.state.list_tab.min(self.lists.lists.len() - 1);
        if self.input.focused() != Some(self.lists.lists[tab].1.container()) {
            return false;
        }
        if self.nav_tab != tab {
            self.nav = ListNavigation::default();
            self.nav_tab = tab;
        }
        match self
            .nav
            .key_list(&mut self.ui, &mut self.lists.lists[tab].1, key)
        {
            Some(Nav::Activated(i)) => {
                self.state.log.push(format!("activated row {i}"));
                true
            }
            Some(Nav::Moved(_)) => true,
            None => false,
        }
    }

    /// The highlighted list row's node, while its list has focus.
    fn highlighted_row(&self) -> Option<NodeId> {
        let tab = self.state.list_tab.min(self.lists.lists.len() - 1);
        let list = &self.lists.lists[tab].1;
        if self.nav_tab != tab || self.input.focused() != Some(list.container()) {
            return None;
        }
        list.row(self.nav.highlighted?)
    }

    /// The highlighted row said to be selected, for assistive technology
    /// (and the one it was before, no longer).
    fn mark_highlighted(&mut self) {
        let now = self.highlighted_row().zip(self.nav.highlighted);
        if self.marked == now {
            return;
        }
        if let Some((old, _)) = self.marked.filter(|&(r, _)| self.ui.contains(r)) {
            self.ui.semantics_mut(old).selected = None;
        }
        if let Some((row, _)) = now {
            let s = self.ui.semantics_mut(row);
            if s.role == Role::Generic {
                s.role = Role::ListItem;
            }
            s.selected = Some(true);
        }
        self.marked = now;
    }

    /// A finger, as [`pointer_down`](Self::pointer_down) and the rest are
    /// the mouse (see `TextEditing::touch`).
    pub fn touch(
        &mut self,
        id: u64,
        phase: TouchPhase,
        x: f32,
        y: f32,
        time: f64,
        measure: &mut dyn TextMeasure,
    ) {
        self.editing.touch(
            &mut self.ui,
            &mut self.input,
            &mut self.state,
            id,
            phase,
            x,
            y,
            time,
            measure,
        );
        self.take_requests();
        self.take_link_click();
    }

    /// How firmly finger `id` presses (0 to 1, where the screen measures
    /// it): on the field's text, pressed firmly, it selects the word.
    pub fn touch_force(&mut self, id: u64, force: f32) {
        self.editing
            .touch_force(&self.input, &mut self.state, id, force);
    }

    /// What the edit menu asked of the platform: the page, for the
    /// clipboard and the share sheet; a toast stands in for a dictionary
    /// and a translator.
    fn take_requests(&mut self) {
        for r in self.editing.take_requests() {
            match r {
                Request::Copy(text) => self.state.copy_out = Some(text),
                Request::Paste => self.state.paste_wanted = true,
                Request::LookUp(text) => self.state.toast = Some(format!("Look Up “{text}”")),
                Request::Translate(text) => self.state.toast = Some(format!("Translate “{text}”")),
                Request::Share(text) => self.state.share_out = Some(text),
            }
        }
    }

    /// A click or tap on text with links: the link under it, for the page
    /// to open while it still handles the click.
    fn take_link_click(&mut self) {
        if let Some((x, y)) = self.state.clicked_at.take()
            && let Some((_, target)) = self.links.iter().find(|(r, _)| r.contains(x, y))
        {
            self.state.open_out = Some(target.clone());
        }
    }

    /// What touch editing is doing, for the page's `?debug` reports.
    pub fn debug_state(&self) -> String {
        let (handles, menu) = self.editing.showing(&self.ui);
        format!(
            "{{\"touch_mode\":{},\"focused\":{},\"selection\":[{},{}],\"field_finger\":{},\"handles\":[{},{}],\"menu\":{}}}",
            self.editing.touch_mode(),
            self.input.focused() == Some(self.nodes.field_box),
            self.state.field.anchor,
            self.state.field.caret,
            self.editing.field_finger().map_or("null".to_owned(), |f| f.to_string()),
            handles[0],
            handles[1],
            menu,
        )
    }

    /// Keeps settings in `storage` between runs, starting from what it has
    /// (the theme).
    pub fn set_storage(&mut self, storage: std::rc::Rc<dyn threers_ui::storage::Storage>) {
        let theme = match storage.get("theme").as_deref() {
            Some("Light") => Some(Theme::Light),
            Some("Dark") => Some(Theme::Dark),
            Some("System") => Some(Theme::System),
            _ => None,
        };
        if let Some(t) = theme {
            self.ui.update_global(|g: &mut Settings| g.theme = t);
            self.stored_theme = Some(t);
        }
        self.keyboard
            .set_enabled(storage.get("keyboard").as_deref() == Some("on"));
        self.storage = Some(storage);
    }

    /// The edges of the window a phone draws over (the page's
    /// `env(safe-area-inset-*)`), which the page keeps clear of.
    pub fn set_safe_area(&mut self, top: f32, right: f32, bottom: f32, left: f32) {
        self.ui.set_safe_area(threers_ui::Edges {
            top,
            right,
            bottom,
            left,
        });
    }

    /// The system's appearance, as the page's `prefers-color-scheme` says:
    /// the System theme (the default) follows it.
    pub fn set_system_dark(&mut self, dark: bool) {
        use threers_ui::states::Appearance;
        self.ui.set_system_appearance(if dark {
            Appearance::Dark
        } else {
            Appearance::Light
        });
    }

    /// Text from the clipboard, for a paste the menu asked for.
    pub fn paste(&mut self, text: &str) {
        self.state.field.insert(text);
        self.editing.typed();
    }

    /// The field as the page's text input holds it: its text, and the
    /// selection from `anchor` to `caret` in characters. The page's input
    /// does the editing on a phone (its keyboard, autocorrect, input
    /// methods), and this keeps the field in step.
    pub fn set_field(&mut self, text: String, anchor: usize, caret: usize) {
        // Another text control has focus (the search box): it takes it.
        if self
            .input
            .focused_text(&self.ui, &self.state)
            .is_some_and(|(n, ..)| n != self.nodes.field_box)
        {
            self.input
                .set_focused_text(&mut self.ui, &mut self.state, text, anchor, caret);
            return;
        }
        let n = text.chars().count();
        if text != self.state.field.text {
            // Typed in: the menu goes, as it does on a phone.
            self.editing.typed();
        }
        let f = &mut self.state.field;
        f.text = text;
        f.anchor = anchor.min(n);
        f.caret = caret.min(n);
        f.marked = None;
    }

    /// The field's box on screen, while a tap at its middle would land on
    /// it (not scrolled out of sight, nor under the dialog). On a phone the
    /// page lays its text input over this, so the tap lands on the input and
    /// the browser opens the keyboard itself.
    pub fn field_rect_and_text(
        &self,
    ) -> Option<(threers_ui::Rect, threers_ui::Rect, String, threers_ui::Rgba)> {
        // Another text control with focus: the page's input goes over it.
        if let Some((n, ..)) = self
            .input
            .focused_text(&self.ui, &self.state)
            .filter(|(n, ..)| *n != self.nodes.field_box)
        {
            let mut stack = vec![n];
            let mut text = None;
            while let Some(c) = stack.pop() {
                if matches!(self.ui.kind(c), threers_ui::NodeKind::Text(_)) {
                    text = Some(c);
                    break;
                }
                stack.extend(self.ui.children(c).iter().rev().copied());
            }
            let t = text?;
            let style = self.ui.style(t);
            return Some((
                self.ui.rect(n),
                self.ui.rect(t),
                threers_ui::TextStyle::of(style).css_font(),
                style.color,
            ));
        }
        let r = self.field_rect()?;
        let t = self.ui.rect(self.nodes.field_text);
        let style = self.ui.style(self.nodes.field_text);
        Some((
            r,
            t,
            threers_ui::TextStyle::of(style).css_font(),
            style.color,
        ))
    }

    /// The text control with focus (the field when none has): its text,
    /// anchor and caret, for the page's input to hold.
    pub fn text_now(&self) -> (String, usize, usize) {
        match self.input.focused_text(&self.ui, &self.state) {
            Some((_, t, a, c)) => (t, a, c),
            None => (
                self.state.field.text.clone(),
                self.state.field.anchor,
                self.state.field.caret,
            ),
        }
    }

    /// Whether the platform's keyboard should be up: a text control has
    /// focus and the on-screen keyboard is not standing in for it.
    pub fn wants_text(&self) -> bool {
        self.input.wants_ime(&self.ui) && !self.keyboard.enabled()
    }

    /// On a touch screen, the page's text input draws the field's text and
    /// does all of its editing natively (a phone's selection handles, its
    /// copy and paste menu, its magnifier); the canvas draws only the
    /// field's box under it.
    pub fn set_native_field(&mut self, native: bool) {
        self.editing.set_native_fields(native);
    }

    /// Focus moved into or out of the field from outside the UI (the page's
    /// text input was tapped, or put its keyboard away).
    pub fn focus_field(&mut self, focus: bool) {
        let field = self.nodes.field_box;
        if focus {
            self.input
                .set_focus(&mut self.ui, &mut self.state, Some(field));
        } else if self.input.focused() == Some(field) {
            self.input.set_focus(&mut self.ui, &mut self.state, None);
        }
    }

    pub fn field_rect(&self) -> Option<threers_ui::Rect> {
        let field = self.nodes.field_box;
        let r = self.ui.rect(field);
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        self.ui
            .hit_test(x, y)
            .filter(|&t| self.ui.ancestors(t).any(|a| a == field))
            .map(|_| r)
    }

    /// The caret's box, for the page to put its text input by (where the
    /// keyboard's suggestions and an input method's candidates appear).
    pub fn ime_area(&self) -> Option<threers_ui::Rect> {
        self.input
            .ime_area()
            .filter(|_| self.input.wants_ime(&self.ui))
    }

    /// The pointer's CSS cursor now.
    pub fn cursor(&self) -> &'static str {
        if self.inspecting {
            return "crosshair";
        }
        let (x, y) = self.input.pointer_position();
        if self.links.iter().any(|(r, _)| r.contains(x, y)) {
            return "pointer";
        }
        // Over text that can be selected, an I-beam, as text views
        // in browsers show (unless the box asks for a cursor of its own).
        let own = self.input.cursor(&self.ui);
        if matches!(own, threers_ui::Cursor::Auto | threers_ui::Cursor::Default)
            && self.ui.selectable_text_at(x, y).is_some()
        {
            return "text";
        }
        self.input.cursor(&self.ui).css()
    }

    /// Whether to draw another frame soon.
    /// Whether the 3D page's candles are flickering: while one shows (not
    /// scrolled away) and motion is wanted. Otherwise they stay as they
    /// are, and the page draws no frames for them.
    fn candles_lit(&self) -> bool {
        self.state.page == crate::pages::DEPTH
            && !self.ui.reduce_motion()
            && self
                .showcase
                .candles
                .iter()
                .any(|c| self.ui.is_on_screen(c.root))
    }

    pub fn needs_frame(&self) -> bool {
        self.ui.needs_frame()
            || self.candles_lit()
            || self.input.needs_frame()
            || self.show_stats
            || self.refresh_until.is_some()
            || self.editing.needs_frame()
            // Notifications time out.
            || self.input.next_timer().is_some()
    }

    /// Lines for the page's console, taken.
    pub fn take_log(&mut self) -> Vec<String> {
        std::mem::take(&mut self.state.log)
    }

    pub fn dialog(&self) -> NodeId {
        self.nodes.dialog
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}
