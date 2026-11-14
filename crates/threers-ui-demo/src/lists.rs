//! The lists panel: one virtual list at a time, chosen by tabs. Each kind
//! shows something a long list is used for: plain rows, a chat of messages
//! of every length, contacts in lettered sections, a grid of cards, and
//! rows to select.

use std::collections::HashSet;

use threers_ui::input::{Input, Phase};
use threers_ui::virtual_list::{RowHeight, VirtualList};
use threers_ui::{style, NodeId, Role, Semantics, Ui};

use crate::app::State;

/// The kinds of list, in tab order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// 100,000 rows of one height.
    Rows,
    /// Chat messages, each as tall as its text wraps to.
    Messages,
    /// Contacts under letter headings: two heights of row.
    Contacts,
    /// Cards three to a row.
    Cards,
    /// Rows that a tap ticks.
    Select,
}

impl Kind {
    pub const ALL: [Kind; 5] = [
        Kind::Rows,
        Kind::Messages,
        Kind::Contacts,
        Kind::Cards,
        Kind::Select,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Kind::Rows => "Rows",
            Kind::Messages => "Messages",
            Kind::Contacts => "Contacts",
            Kind::Cards => "Cards",
            Kind::Select => "Select",
        }
    }
}

const MESSAGES: usize = 10_000;
const CONTACTS_PER_LETTER: usize = 300;
const CARDS: usize = 30_000;
const CARDS_PER_ROW: usize = 3;
const SELECTABLE: usize = 5_000;

/// The tabs, the lists and what is selected.
pub struct Lists {
    pub tabs: Vec<NodeId>,
    pub lists: Vec<(Kind, VirtualList)>,
    /// Under the tabs: what the list shows, or how many are selected.
    pub caption: NodeId,
    /// The tab showing, and which it was when the list was last shown.
    shown: Option<usize>,
    /// Selected rows of the `Select` list, as drawn.
    drawn_selection: HashSet<usize>,
    /// Over the list: the pull to refresh's indicator, and its label.
    pull: NodeId,
    pull_label: NodeId,
    /// Refreshes the lists' content was built for.
    drawn_refreshes: usize,
    /// The appearance the tabs were styled for.
    drawn_appearance: Option<threers_ui::states::Appearance>,
    /// Per list, the items deleted, ascending.
    removed: Vec<Vec<usize>>,
    /// Swipeable rows built: their Delete action and their content.
    parts: std::collections::HashMap<NodeId, (NodeId, NodeId)>,
    /// The row slid open, as drawn.
    swiped: Option<NodeId>,
}

/// A small deterministic hash, for varied but repeatable content.
fn mix(i: usize) -> u64 {
    let mut x = i as u64 ^ 0x9e37_79b9_7f4a_7c15;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

const WORDS: [&str; 24] = [
    "sure",
    "the",
    "layout",
    "is",
    "done",
    "can",
    "you",
    "check",
    "scrolling",
    "on",
    "your",
    "phone",
    "it",
    "glides",
    "now",
    "and",
    "the",
    "keyboard",
    "opens",
    "great",
    "thanks",
    "one",
    "more",
    "thing",
];

/// Message `i`: who sent it, and its text, from a few words to a paragraph.
fn message(i: usize) -> (bool, String) {
    let h = mix(i);
    let words = 2 + (h % 7) as usize * (1 + (h >> 8) as usize % 5);
    let text: Vec<&str> = (0..words)
        .map(|k| WORDS[(mix(i * 31 + k) % WORDS.len() as u64) as usize])
        .collect();
    let mut text = text.join(" ");
    if let Some(c) = text.get_mut(0..1) {
        c.make_ascii_uppercase();
    }
    (h & 1 == 0, text + ".")
}

const FIRST: [&str; 12] = [
    "Ada", "Ben", "Cleo", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea",
];

/// Contact row `i` of the `Contacts` list: a letter's heading, or a person.
fn contact(i: usize, seed: usize) -> Result<char, (String, String)> {
    let per = CONTACTS_PER_LETTER + 1;
    let letter = (b'A' + (i / per) as u8) as char;
    match i % per {
        0 => Ok(letter),
        k => {
            let first = FIRST[(mix(i + seed) % FIRST.len() as u64) as usize];
            let last = format!(
                "{letter}{}",
                ["ardy", "elton", "ines", "ovak", "urst", "ash"][(mix(i + seed + 7) % 6) as usize]
            );
            Err((
                format!("{first} {last}"),
                format!("+1 555 {:04}", (k * 37 + i) % 10_000),
            ))
        }
    }
}

fn tab_style() -> threers_ui::Style {
    style! { padding: 6px 10px; border-radius: 6px; font-size: 12px; color: #c3c8d2; background: #232833; cursor: pointer; }
}

impl Lists {
    /// The tabs and lists, under `panel`; the `Rows` list shows.
    pub fn new(ui: &mut Ui, input: &mut Input<State>, panel: NodeId) -> Lists {
        let bar = ui.add_box(panel, style! { display: flex; gap: 4px; });
        ui.set_semantics(bar, Semantics::new(Role::TabList).label("Lists"));
        let mut tabs = vec![];
        for (i, kind) in Kind::ALL.into_iter().enumerate() {
            let tab = ui.add_box(bar, tab_style());
            ui.add_text(tab, kind.label());
            ui.set_semantics(tab, Semantics::new(Role::Tab).label(kind.label()));
            ui.hover_style(tab, "background: rgba(128,138,160,0.3)");
            input.on_click(tab, move |s: &mut State, _, _| s.list_tab = i);
            tabs.push(tab);
        }
        let caption = ui.add_box(
            panel,
            style! { font-size: 11px; color: #8b93a5; white-space: normal; },
        );
        ui.token_style(
            caption,
            &[("color", threers_ui::environment::Token::SecondaryText)],
        );
        ui.add_text(caption, "");
        // As tall as the list is pulled down: the list below moves down
        // with the finger.
        let pull = ui.add_box(
            panel,
            style! { display: none; height: 0px; flex-shrink: 0; align-items: flex-end; justify-content: center; overflow: hidden; font-size: 12px; color: #8b93a5; },
        );
        ui.token_style(
            pull,
            &[("color", threers_ui::environment::Token::SecondaryText)],
        );
        let pull_label = ui.add_text(pull, "");
        ui.set_semantics(pull, Semantics::new(Role::Status).live());
        let list_style = style! {
            flex-grow: 1; min-height: 0; height: 100px; scrollbar-width: auto;
            scrollbar-visibility: auto; scrollbar-fade: 800ms 300ms; border-radius: 6px; background: #161a21;
        };
        let mut lists = vec![];
        for kind in Kind::ALL {
            let (count, height) = match kind {
                Kind::Rows => (100_000, RowHeight::Uniform(26.0)),
                Kind::Messages => (MESSAGES, RowHeight::Measured { estimate: 48.0 }),
                Kind::Contacts => (
                    26 * (CONTACTS_PER_LETTER + 1),
                    RowHeight::Measured { estimate: 46.0 },
                ),
                Kind::Cards => (CARDS.div_ceil(CARDS_PER_ROW), RowHeight::Uniform(92.0)),
                Kind::Select => (SELECTABLE, RowHeight::Uniform(36.0)),
            };
            let list = VirtualList::new(ui, panel, list_style.clone(), count, height);
            ui.appearance_style(
                list.container(),
                threers_ui::states::Appearance::Light,
                "background: #f7f8fa",
            );
            let c = list.container();
            // Focusable: the arrow keys then move through its rows (the app
            // handles them).
            ui.set_semantics(
                c,
                Semantics::new(Role::List).label(kind.label()).tab_index(0),
            );
            let tab = lists.len();
            input.on_refresh(c, move |s: &mut State, _| s.refreshing = Some(tab));
            // A tap on a row (to tick it, or on its Delete action): the row
            // is found from the node tapped, since rows come and go as the
            // list scrolls.
            input.on(c, Phase::Bubble, |s: &mut State, _, e, cx| {
                if let threers_ui::input::Event::Click { .. } = e {
                    s.list_clicked = Some(cx.target);
                }
            });
            if kind != Kind::Cards {
                input.on_swipe_delete(c, |s: &mut State, _, row| s.delete_row = Some(row));
            }
            if kind != Kind::Rows {
                ui.update_style(c, |s| s.display = threers_ui::Display::None);
            }
            lists.push((kind, list));
        }
        Lists {
            tabs,
            removed: vec![vec![]; lists.len()],
            lists,
            caption,
            shown: None,
            drawn_selection: HashSet::new(),
            pull,
            pull_label,
            drawn_refreshes: 0,
            drawn_appearance: None,
            parts: Default::default(),
            swiped: None,
        }
    }

    /// Shows the pull to refresh of the list showing: open as far as it is
    /// pulled, saying what letting go will do.
    pub fn show_pull(&mut self, ui: &mut Ui, pull: Option<threers_ui::input::Pull>) {
        use threers_ui::input::PullPhase;
        let (display, height, label) = match pull {
            None => (threers_ui::Display::None, 0.0, ""),
            Some(p) => (
                threers_ui::Display::Flex,
                p.distance.round(),
                match p.phase {
                    PullPhase::Pulling => "Pull to refresh",
                    PullPhase::Armed => "Release to refresh",
                    PullPhase::Refreshing => "Refreshing…",
                    PullPhase::Settling => "",
                },
            ),
        };
        let s = ui.style(self.pull);
        if s.display != display || s.height != threers_ui::Length::Px(height) {
            let s = ui.style_mut(self.pull);
            s.display = display;
            s.height = threers_ui::Length::Px(height);
        }
        if ui.text_content(self.pull) != label {
            ui.set_text(self.pull_label, label);
        }
    }

    /// The list showing.
    pub fn active(&self, state: &State) -> &VirtualList {
        &self.lists[state.list_tab.min(self.lists.len() - 1)].1
    }

    /// Applies what the user did since the last frame: a tab chosen, a row
    /// ticked. Call before layout.
    pub fn apply(&mut self, ui: &mut Ui, state: &mut State) {
        let tab = state.list_tab.min(self.lists.len() - 1);
        let row_of = |lists: &Self, n: NodeId| {
            let list = &lists.lists[tab].1;
            list.built()
                .iter()
                .copied()
                .find(|&(_, row)| ui.contains(n) && ui.ancestors(n).any(|a| a == row))
        };
        if let Some(target) = state.list_clicked.take()
            && let Some((i, row)) = row_of(self, target)
        {
            let on_action = self
                .parts
                .get(&row)
                .is_some_and(|&(action, _)| ui.ancestors(target).any(|a| a == action));
            if on_action {
                self.delete(ui, state, tab, i);
            } else if Kind::ALL[tab] == Kind::Select {
                let d = data_index(&self.removed[tab], i);
                if !state.selected.remove(&d) {
                    state.selected.insert(d);
                }
            }
        }
        if let Some(row) = state.delete_row.take()
            && let Some(&(i, _)) = self.lists[tab].1.built().iter().find(|&&(_, r)| r == row)
        {
            self.delete(ui, state, tab, i);
        }
        // Restyled when the tab or the appearance changes: the chosen tab's
        // blue is its own style, which an appearance style would cover.
        let appearance = ui.appearance();
        if self.shown != Some(tab) || self.drawn_appearance != Some(appearance) {
            self.drawn_appearance = Some(appearance);
            self.shown = Some(tab);
            for (i, (_, list)) in self.lists.iter().enumerate() {
                let display = if i == tab {
                    threers_ui::Display::Block
                } else {
                    threers_ui::Display::None
                };
                if ui.style(list.container()).display != display {
                    ui.update_style(list.container(), |s| s.display = display);
                }
            }
            for (i, &t) in self.tabs.iter().enumerate() {
                // Through `set_style`: a node with a hover style goes back to
                // the style it was set, not this, when the pointer leaves.
                let mut style = tab_style();
                if appearance == threers_ui::states::Appearance::Light {
                    style.background = Some([0.914, 0.925, 0.945, 1.0]);
                    style.color = [0.216, 0.255, 0.32, 1.0];
                }
                if i == tab {
                    style.background = Some([0.145, 0.388, 0.922, 1.0]);
                    style.color = [0.99, 0.99, 0.99, 1.0];
                }
                ui.set_style(t, style);
                // The chosen tab keeps its accent under the pointer.
                ui.set_hover_style(
                    t,
                    if i == tab {
                        "background: #1d4ed8"
                    } else {
                        "background: rgba(128,138,160,0.3)"
                    },
                );
                ui.set_semantics(
                    t,
                    Semantics::new(Role::Tab)
                        .label(Kind::ALL[i].label())
                        .selected(i == tab),
                );
            }
        }
        if self.drawn_refreshes != state.refreshes {
            self.drawn_refreshes = state.refreshes;
            for (_, list) in &mut self.lists {
                list.rebuild(ui);
            }
        }
        if self.drawn_selection != state.selected {
            self.drawn_selection = state.selected.clone();
            if let Some((_, list)) = self.lists.iter_mut().find(|(k, _)| *k == Kind::Select) {
                list.rebuild(ui);
            }
        }
        let mut caption = match Kind::ALL[tab] {
            Kind::Rows => "100,000 rows of one height, built as they show.".to_owned(),
            Kind::Messages => {
                format!("{MESSAGES} messages, each as tall as its text: measured as it shows.")
            }
            Kind::Contacts => format!(
                "{} contacts under letter headings.",
                26 * CONTACTS_PER_LETTER
            ),
            Kind::Cards => format!("{CARDS} cards, {CARDS_PER_ROW} to a row."),
            Kind::Select => format!(
                "{} of {SELECTABLE} selected. Tap a row to tick it.",
                state.selected.len()
            ),
        };
        if state.refreshes > 0 {
            caption += &format!(" Refreshed {}×.", state.refreshes);
        }
        if !self.removed[tab].is_empty() {
            caption += &format!(" {} deleted.", self.removed[tab].len());
        } else if Kind::ALL[tab] != Kind::Cards {
            caption += " Swipe a row left to delete it.";
        }
        if ui.text_content(self.caption) != caption {
            let t = ui.children(self.caption)[0];
            ui.set_text(t, caption);
        }
    }

    /// Builds the rows that came into view in the list showing; returns
    /// whether it built any (lay out again).
    pub fn sync(&mut self, ui: &mut Ui, state: &State) -> bool {
        let tab = state.list_tab.min(self.lists.len() - 1);
        let removed = &self.removed[tab];
        let (kind, list) = &mut self.lists[tab];
        let kind = *kind;
        let selected = &state.selected;
        // Content changes with each refresh.
        let seed = state.refreshes * 7919;
        let mut parts = vec![];
        let changed = list.sync(ui, |ui, row, i| {
            let d = data_index(removed, i);
            let header = kind == Kind::Contacts && contact(d, seed).is_ok();
            if kind == Kind::Cards || header {
                build(kind, ui, row, d, seed, selected);
                return;
            }
            // Swipeable: the content over a Delete action that shows as the
            // content slides left.
            ui.set_style(row, style! { position: relative; overflow: hidden; });
            let action = ui.add_box(
                row,
                style! { display: none; position: absolute; right: 0px; top: 0px; height: 100%; width: 0px; align-items: center; justify-content: center; overflow: hidden; background: #dc2626; color: #fcfcfc; font-size: 12px; font-weight: bold; },
            );
            ui.add_text(action, "Delete");
            ui.set_semantics(action, Semantics::new(Role::Button).label("Delete"));
            let content = ui.add_box(row, style! { position: relative; });
            build(kind, ui, content, d, seed, selected);
            parts.push((row, action, content));
        });
        for (row, action, content) in parts {
            self.parts.insert(row, (action, content));
        }
        self.parts.retain(|&row, _| ui.contains(row));
        changed
    }

    /// Slides the swiped row's content and shows its action in the space
    /// it leaves.
    pub fn show_swipe(&mut self, ui: &mut Ui, swipe: Option<threers_ui::input::Swipe>) {
        // A deleted row's parts are gone from the tree before the next sync
        // forgets them.
        self.parts.retain(|&row, _| ui.contains(row));
        let now = swipe.filter(|s| self.parts.contains_key(&s.row));
        if let Some(old) = self.swiped.filter(|&r| now.is_none_or(|s| s.row != r))
            && let Some(&(action, content)) = self.parts.get(&old)
        {
            ui.style_mut(content).left = threers_ui::Length::Px(0.0);
            ui.style_mut(action).display = threers_ui::Display::None;
        }
        self.swiped = now.map(|s| s.row);
        if let Some(s) = now {
            let (action, content) = self.parts[&s.row];
            let offset = s.offset.round();
            if ui.style(content).left != threers_ui::Length::Px(offset) {
                ui.style_mut(content).left = threers_ui::Length::Px(offset);
                let a = ui.style_mut(action);
                a.display = if offset < 0.0 {
                    threers_ui::Display::Flex
                } else {
                    threers_ui::Display::None
                };
                a.width = threers_ui::Length::Px(-offset);
            }
        }
    }

    /// Deletes row `i` of the list on `tab` (a contact or message: letter
    /// headings stay).
    fn delete(&mut self, ui: &mut Ui, state: &mut State, tab: usize, i: usize) {
        let (kind, list) = &mut self.lists[tab];
        let d = data_index(&self.removed[tab], i);
        if *kind == Kind::Cards
            || (*kind == Kind::Contacts && contact(d, state.refreshes * 7919).is_ok())
        {
            return;
        }
        let at = self.removed[tab].partition_point(|&r| r < d);
        self.removed[tab].insert(at, d);
        list.remove(ui, i);
        state.selected.remove(&d);
    }
}

/// Row `i` as shown: data index `i` past the rows removed before it.
fn data_index(removed: &[usize], i: usize) -> usize {
    let mut d = i;
    for &r in removed {
        if r <= d {
            d += 1;
        } else {
            break;
        }
    }
    d
}

/// Builds item `i` of a list of `kind` into `row`.
fn build(kind: Kind, ui: &mut Ui, row: NodeId, i: usize, seed: usize, selected: &HashSet<usize>) {
    match kind {
        Kind::Rows => {
            ui.set_style(row, style! { height: 26px; display: flex; align-items: center; padding: 0 10px; font-size: 12px; });
            ui.add_text(row, format!("Row {:>6}", i + 1));
            if i % 2 == 1 {
                ui.style_mut(row).background = Some([1.0, 1.0, 1.0, 0.03]);
            }
            ui.hover_style(row, "background: rgba(59,130,246,0.18)");
        }
        Kind::Messages => {
            let (mine, text) = message(i + seed);
            ui.set_style(row, style! { display: flex; padding: 3px 8px; });
            ui.style_mut(row).justify_content = if mine {
                threers_ui::JustifyContent::FlexEnd
            } else {
                threers_ui::JustifyContent::FlexStart
            };
            let bubble = ui.add_box(
                row,
                style! { max-width: 78%; padding: 6px 10px; border-radius: 12px; font-size: 12px; line-height: 16px; white-space: normal; },
            );
            ui.style_mut(bubble).background = Some(if mine {
                [0.145, 0.388, 0.922, 1.0]
            } else {
                [0.17, 0.19, 0.24, 1.0]
            });
            if mine {
                ui.style_mut(bubble).color = [0.99, 0.99, 0.99, 1.0];
            } else {
                ui.appearance_style(
                    bubble,
                    threers_ui::states::Appearance::Light,
                    "background: #e6e9ee; color: #1f2329",
                );
            }
            ui.add_text(bubble, text);
        }
        Kind::Contacts => {
            match contact(i, seed) {
                Ok(letter) => {
                    ui.set_style(
                    row,
                    style! { padding: 4px 10px; font-size: 11px; font-weight: bold; color: #8b93a5; background: #1d222b; },
                );
                    ui.appearance_style(
                        row,
                        threers_ui::states::Appearance::Light,
                        "background: #eef0f3",
                    );
                    ui.token_style(
                        row,
                        &[("color", threers_ui::environment::Token::SecondaryText)],
                    );
                    ui.add_text(row, letter.to_string());
                    ui.set_semantics(
                        row,
                        Semantics::new(Role::Heading(3)).label(letter.to_string()),
                    );
                }
                Err((name, phone)) => {
                    ui.set_style(row, style! { display: flex; align-items: center; gap: 10px; padding: 6px 10px; });
                    let avatar = ui.add_box(
                    row,
                    style! { width: 30px; height: 30px; border-radius: 15px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: bold; color: #fcfcfc; },
                );
                    let hue = (mix(i + seed) % 6) as usize;
                    // Dark enough for white initials (4.5:1 at least).
                    let colours = [
                        [0.706, 0.137, 0.094],
                        [0.180, 0.490, 0.196],
                        [0.114, 0.306, 0.847],
                        [0.706, 0.325, 0.035],
                        [0.427, 0.157, 0.851],
                        [0.059, 0.463, 0.431],
                    ];
                    let c = colours[hue];
                    ui.style_mut(avatar).background = Some([c[0], c[1], c[2], 1.0]);
                    let initials: String =
                        name.split(' ').filter_map(|w| w.chars().next()).collect();
                    ui.add_text(avatar, initials);
                    let col = ui.add_box(
                        row,
                        style! { display: flex; flex-direction: column; gap: 2px; },
                    );
                    let n = ui.add_box(col, style! { font-size: 12px; });
                    ui.add_text(n, name.clone());
                    let p = ui.add_box(col, style! { font-size: 11px; color: #8b93a5; });
                    ui.token_style(
                        p,
                        &[("color", threers_ui::environment::Token::SecondaryText)],
                    );
                    ui.add_text(p, phone);
                    ui.set_semantics(row, Semantics::new(Role::ListItem).label(name));
                    ui.hover_style(row, "background: rgba(59,130,246,0.12)");
                }
            }
        }
        Kind::Cards => {
            ui.set_style(
                row,
                style! { height: 92px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; padding: 4px 8px; box-sizing: border-box; },
            );
            for k in 0..CARDS_PER_ROW {
                let n = i * CARDS_PER_ROW + k;
                if n >= CARDS {
                    break;
                }
                let card = ui.add_box(
                    row,
                    style! { display: flex; flex-direction: column; justify-content: flex-end; gap: 2px; padding: 8px; border-radius: 8px; font-size: 11px; color: #fcfcfc; },
                );
                let t = (mix(n + seed) % 360) as f32 / 360.0;
                let rgb = |o: f32| 0.25 + 0.2 * ((t + o) * std::f32::consts::TAU).sin();
                ui.style_mut(card).background = Some([rgb(0.0), rgb(0.33), rgb(0.66), 1.0]);
                let title = ui.add_box(card, style! { font-weight: bold; font-size: 12px; });
                ui.add_text(title, format!("Card {}", n + 1));
                ui.add_text(card, format!("{} items", mix(n + seed) % 90 + 1));
                ui.set_semantics(
                    card,
                    Semantics::new(Role::ListItem).label(format!("Card {}", n + 1)),
                );
                ui.hover_style(card, "border: 1px solid #2563eb");
            }
        }
        Kind::Select => {
            let on = selected.contains(&i);
            ui.set_style(row, style! { height: 36px; display: flex; align-items: center; gap: 10px; padding: 0 10px; font-size: 12px; cursor: pointer; });
            if on {
                ui.style_mut(row).background = Some([0.145, 0.388, 0.922, 0.22]);
            }
            let tick = ui.add_box(row, style! { width: 16px; height: 16px; border-radius: 4px; border: 1px solid #5b6478; });
            if on {
                ui.style_mut(tick).background = Some([0.145, 0.388, 0.922, 1.0]);
            }
            ui.add_text(row, format!("Item {}", i + 1));
            ui.set_semantics(
                row,
                Semantics::new(Role::Checkbox)
                    .label(format!("Item {}", i + 1))
                    .checked(on),
            );
            ui.hover_style(row, "background: rgba(59,130,246,0.12)");
        }
    }
}
