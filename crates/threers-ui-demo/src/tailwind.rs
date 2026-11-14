//! The Tailwind page: boxes styled by Tailwind's utility classes (each
//! with its classes under it), shadcn/ui's components and theme, variants
//! that stack (states, groups, the appearance, the window's width), and
//! cards turned and lifted in 3D by classes. The UI reads classes with
//! `Tailwind::shadcn()`: Tailwind's defaults with shadcn's colours, which
//! switch with the appearance, and its components.

use threers_ui::environment::Token;
use threers_ui::input::Input;
use threers_ui::tailwind::Tailwind;
use threers_ui::{style, Length, NodeId, Role, Semantics, Ui};

use crate::app::State;
use crate::showcase::group;

const TABS: [(&str, &str); 2] = [
    (
        "Account",
        "Make changes to your account here. Click save when you are done.",
    ),
    (
        "Password",
        "Change your password here. After saving, you will be logged out.",
    ),
];

/// What the page keeps up from the state.
pub struct TailwindPage {
    /// Each tab: its trigger shown inactive, shown active, and its panel.
    tabs: Vec<(NodeId, NodeId, NodeId)>,
    shown_tab: Option<usize>,
    bar: NodeId,
    shown_progress: f32,
    status: NodeId,
    shown_presses: u32,
}

impl TailwindPage {
    pub fn update(&mut self, ui: &mut Ui, s: &State) {
        if self.shown_tab != Some(s.tw_tab) {
            self.shown_tab = Some(s.tw_tab);
            for (i, &(off, on, panel)) in self.tabs.iter().enumerate() {
                let shown = |yes: bool| {
                    move |st: &mut threers_ui::Style| {
                        st.display = if yes {
                            threers_ui::Display::Flex
                        } else {
                            threers_ui::Display::None
                        }
                    }
                };
                ui.update_style(off, shown(i != s.tw_tab));
                ui.update_style(on, shown(i == s.tw_tab));
                ui.update_style(panel, shown(i == s.tw_tab));
            }
        }
        if self.shown_progress != s.progress {
            self.shown_progress = s.progress;
            ui.update_style(self.bar, |st| {
                st.width = Length::Percent(s.progress * 100.0)
            });
        }
        if self.shown_presses != s.tw_presses {
            self.shown_presses = s.tw_presses;
            let text = if s.tw_presses == 1 {
                format!("Pressed “{}” once.", s.tw_pressed)
            } else {
                format!(
                    "Pressed “{}”; {} presses in all.",
                    s.tw_pressed, s.tw_presses
                )
            };
            ui.set_text(self.status, text);
        }
    }
}

/// The classes, as code, under an example.
fn code(ui: &mut Ui, parent: NodeId, classes: &str) {
    let c = ui.add_box(
        parent,
        style! { font-family: monospace; font-size: 11px; line-height: 15px; white-space: normal; },
    );
    ui.token_style(c, &[("color", Token::SecondaryText)]);
    ui.add_text(c, classes);
}

/// A box styled by `classes`, saying `text`, with its classes under it.
fn example(ui: &mut Ui, parent: NodeId, classes: &str, text: &str) -> NodeId {
    let column = ui.add_tw(parent, "flex flex-col gap-1.5 min-w-0");
    let n = ui.add_tw(column, classes);
    if !text.is_empty() {
        ui.add_text(n, text);
    }
    code(ui, column, classes);
    n
}

/// A button styled by `classes` that counts its presses.
fn button(
    ui: &mut Ui,
    input: &mut Input<State>,
    parent: NodeId,
    classes: &str,
    label: &'static str,
) -> NodeId {
    let b = ui.add_tw(parent, classes);
    ui.tw(b, "cursor-pointer");
    ui.add_text(b, label);
    ui.set_semantics(b, Semantics::new(Role::Button).label(label));
    input.on_click(b, move |s: &mut State, _, _| {
        s.tw_presses += 1;
        s.tw_pressed = label;
    });
    b
}

/// A line of text in the theme's muted colour.
fn muted(ui: &mut Ui, parent: NodeId, text: &str) -> NodeId {
    let n = ui.add_tw(parent, "text-sm text-muted-foreground");
    ui.add_text(n, text)
}

pub fn build(ui: &mut Ui, input: &mut Input<State>, _state: &State, page: NodeId) -> TailwindPage {
    ui.set_global(Tailwind::shadcn());

    // Tailwind's own palette, spacing, radii, shadows and gradients.
    let g = group(ui, page, "Utility classes", "Boxes styled by Tailwind's classes alone: its palette, spacing, radii, shadows, rings and gradients. Each shows its classes.");
    let banner = example(ui, g, "flex flex-col gap-1 rounded-xl bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600 p-6 text-white shadow-lg", "");
    let title = ui.add_tw(banner, "text-2xl font-bold");
    ui.add_text(title, "Ship faster with utility classes");
    let sub = ui.add_tw(banner, "text-sm");
    ui.add_text(sub, "The same class strings as the web, laid out natively.");
    let stats = ui.add_tw(g, "flex flex-col gap-1.5");
    let grid = ui.add_tw(stats, "grid grid-cols-1 sm:grid-cols-3 gap-3");
    for (label, value, change) in [
        ("Revenue", "$45,231", "+20.1% from last month"),
        ("Subscriptions", "+2,350", "+180.1% from last month"),
        ("Active now", "573", "+201 since the last hour"),
    ] {
        let tile = ui.add_tw(
            grid,
            "flex flex-col gap-1 rounded-lg border bg-card p-4 text-card-foreground shadow-sm",
        );
        muted(ui, tile, label);
        let v = ui.add_tw(tile, "text-2xl font-semibold");
        ui.add_text(v, value);
        let c = ui.add_tw(tile, "text-xs text-muted-foreground");
        ui.add_text(c, change);
    }
    code(
        ui,
        stats,
        "grid grid-cols-1 sm:grid-cols-3 gap-3 · rounded-lg border bg-card p-4 shadow-sm",
    );
    let chips = ui.add_tw(g, "flex flex-wrap items-center gap-2");
    for (classes, text) in [
        ("rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-800", "Shipped"),
        ("rounded-full bg-sky-100 px-3 py-1 text-xs font-medium text-sky-800", "In review"),
        ("rounded-full bg-amber-100 px-3 py-1 text-xs font-medium text-amber-900", "Blocked"),
        ("rounded-full px-3 py-1 text-xs font-medium text-rose-600 dark:text-rose-400 ring-2 ring-rose-500 ring-offset-2 ring-offset-background", "Ring with an offset"),
    ] {
        let c = ui.add_tw(chips, classes);
        ui.add_text(c, text);
    }
    code(ui, g, "rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-800 · ring-2 ring-rose-500 ring-offset-2");

    // shadcn/ui: its theme and components, as classes.
    let g = group(ui, page, "shadcn/ui", "shadcn/ui's components as classes (button button-outline button-sm, card, input, badge, alert, tabs, progress) in its theme, whose colours switch with light and dark.");
    let card = ui.add_tw(g, "card max-w-md");
    let header = ui.add_tw(card, "card-header");
    let t = ui.add_tw(header, "card-title");
    ui.add_text(t, "Create project");
    ui.set_semantics(t, Semantics::new(Role::Heading(4)));
    let d = ui.add_tw(header, "card-description");
    ui.add_text(d, "Deploy your new project in one click.");
    let content = ui.add_tw(card, "card-content flex flex-col gap-2");
    let l = ui.add_tw(content, "label");
    ui.add_text(l, "Name");
    let field = ui.add_tw(content, "input items-center");
    let placeholder = ui.add_tw(field, "text-muted-foreground");
    ui.add_text(placeholder, "Name of your project");
    let footer = ui.add_tw(card, "card-footer justify-between");
    button(ui, input, footer, "button button-outline", "Cancel");
    button(ui, input, footer, "button", "Deploy");
    code(ui, g, "card · card-header · card-title · card-description · card-content · label · input · card-footer · button button-outline");

    let row = ui.add_tw(g, "flex flex-wrap items-center gap-2");
    for (classes, label) in [
        ("button", "Default"),
        ("button button-secondary", "Secondary"),
        ("button button-destructive", "Destructive"),
        ("button button-outline", "Outline"),
        ("button button-ghost", "Ghost"),
        ("button button-link", "Link"),
    ] {
        button(ui, input, row, classes, label);
    }
    let row = ui.add_tw(g, "flex flex-wrap items-center gap-2");
    button(ui, input, row, "button button-sm", "Small");
    button(ui, input, row, "button button-lg", "Large");
    button(
        ui,
        input,
        row,
        "button button-outline button-icon text-lg",
        "+",
    );
    let status = muted(ui, row, "Press a button.");
    code(ui, g, "button · button-secondary · button-destructive · button-outline · button-ghost · button-link · button-sm · button-lg · button-icon");

    let row = ui.add_tw(g, "flex flex-wrap items-center gap-2");
    for (classes, text) in [
        ("badge", "Badge"),
        ("badge badge-secondary", "Secondary"),
        ("badge badge-destructive", "Destructive"),
        ("badge badge-outline", "Outline"),
    ] {
        let b = ui.add_tw(row, classes);
        ui.add_text(b, text);
    }
    ui.add_tw(g, "separator");
    let alerts = ui.add_tw(g, "grid grid-cols-1 md:grid-cols-2 gap-3");
    for (classes, title, text) in [
        (
            "alert",
            "Heads up!",
            "Classes read once, when a node is given them; states and the theme restyle it after.",
        ),
        (
            "alert alert-destructive",
            "Something went wrong",
            "Your session has expired. Please log in again.",
        ),
    ] {
        let a = ui.add_tw(alerts, classes);
        let t = ui.add_tw(a, "alert-title");
        ui.add_text(t, title);
        let d = ui.add_tw(a, "alert-description");
        ui.add_text(d, text);
    }

    // Tabs: each trigger in both looks, one shown.
    let tabs_box = ui.add_tw(g, "flex flex-col gap-2 max-w-md");
    let list = ui.add_tw(tabs_box, "tabs-list");
    ui.set_semantics(list, Semantics::new(Role::TabList).label("Settings"));
    let panels = ui.add_tw(tabs_box, "flex flex-col");
    let mut tabs = vec![];
    for (i, (name, text)) in TABS.iter().enumerate() {
        let off = ui.add_tw(list, "tabs-trigger cursor-pointer");
        let on = ui.add_tw(list, "tabs-trigger tabs-trigger-active cursor-pointer");
        for (n, selected) in [(off, false), (on, true)] {
            ui.add_text(n, *name);
            ui.set_semantics(n, Semantics::new(Role::Tab).label(*name).selected(selected));
            input.on_click(n, move |s: &mut State, _, _| s.tw_tab = i);
        }
        let panel = ui.add_tw(panels, "card px-6 py-4 gap-2");
        let t = ui.add_tw(panel, "card-title");
        ui.add_text(t, *name);
        muted(ui, panel, text);
        tabs.push((off, on, panel));
    }
    code(
        ui,
        tabs_box,
        "tabs-list · tabs-trigger · tabs-trigger-active",
    );
    let progress = ui.add_tw(g, "flex flex-col gap-1.5 max-w-md");
    let track = ui.add_tw(progress, "progress");
    let bar = ui.add_tw(track, "progress-indicator");
    code(
        ui,
        progress,
        "progress · progress-indicator (its width from the Progress page's slider)",
    );

    // Variants: states, groups, the appearance and the window's width,
    // stacked.
    let g = group(ui, page, "Variants", "hover:, active:, focus-visible:, group-hover:, dark: and the breakpoints, alone and stacked (dark:hover:, md:hover:). Tab to the button to see its ring.");
    let row = ui.add_tw(g, "flex flex-wrap items-start gap-4");
    let column = ui.add_tw(row, "flex flex-col gap-1.5 min-w-0 max-w-sm");
    button(
        ui,
        input,
        column,
        "rounded-md bg-sky-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-sky-800 active:bg-sky-900 focus-visible:ring-2 focus-visible:ring-sky-400 focus-visible:ring-offset-2 focus-visible:ring-offset-background",
        "Hover, press, or Tab to me",
    );
    code(ui, column, "bg-sky-700 hover:bg-sky-800 active:bg-sky-900 focus-visible:ring-2 focus-visible:ring-offset-2 transition-colors");
    let column = ui.add_tw(row, "flex flex-col gap-1.5 min-w-0 max-w-sm flex-1");
    let item = ui.add_tw(column, "group/row flex items-center justify-between gap-4 rounded-lg border px-4 py-3 transition-colors hover:bg-accent");
    let name = ui.add_tw(item, "text-sm font-medium");
    ui.add_text(name, "Hover the row");
    let edit = ui.add_tw(
        item,
        "text-sm text-muted-foreground opacity-0 transition-opacity group-hover/row:opacity-100",
    );
    ui.add_text(edit, "Edit →");
    code(
        ui,
        column,
        "group/row hover:bg-accent · opacity-0 group-hover/row:opacity-100",
    );
    example(ui, g, "rounded-lg bg-amber-100 p-3 text-sm text-amber-900 dark:bg-indigo-950 dark:text-indigo-200", "Amber in the light appearance, indigo in the dark.");
    example(
        ui,
        g,
        "rounded-lg p-3 text-sm bg-rose-100 text-rose-900 sm:bg-amber-100 sm:text-amber-900 md:bg-emerald-100 md:text-emerald-900 lg:bg-sky-100 lg:text-sky-900 md:hover:bg-emerald-200 lg:hover:bg-sky-200",
        "Resize the window: rose, amber from sm, emerald from md, sky from lg, each a shade darker hovered.",
    );
    example(ui, g, "rounded-lg border p-3 text-sm transition-colors hover:bg-slate-100 dark:hover:bg-indigo-900", "hover: in the light; dark:hover: in the dark.");

    // 3D.
    let g = group(ui, page, "3D", "Cards turned, lifted and scaled by classes (rotate-x, rotate-y, translate, scale, perspective, origin, glare), moving into their hovered state over a transition.");
    let grid = ui.add_tw(g, "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6");
    for (classes, text) in [
        ("flex h-32 items-center justify-center rounded-xl bg-indigo-700 text-lg font-semibold text-white shadow-md transition duration-300 hover:-translate-y-2 hover:scale-105 hover:shadow-xl", "Lift"),
        ("flex h-32 items-center justify-center rounded-xl bg-emerald-700 text-lg font-semibold text-white perspective-normal glare transition duration-500 hover:rotate-y-12 hover:-rotate-x-6", "Tilt"),
        ("flex h-32 items-center justify-center rounded-xl bg-rose-700 text-lg font-semibold text-white perspective-normal origin-bottom rotate-x-30 depth-shadow transition duration-500 hover:rotate-x-0", "Stand up"),
        ("flex h-32 items-center justify-center rounded-xl bg-amber-400 text-lg font-semibold text-amber-950 rotate-3 transition duration-200 hover:rotate-0 hover:scale-105", "Straighten"),
    ] {
        example(ui, grid, classes, text);
    }

    TailwindPage {
        tabs,
        shown_tab: None,
        bar,
        shown_progress: f32::NAN,
        status,
        shown_presses: 0,
    }
}
