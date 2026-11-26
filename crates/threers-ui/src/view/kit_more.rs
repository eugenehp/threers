//! More of the kit: choosing, moving about, showing where things stand.
//! See [`super::kit`]. Those with a set of items are one stop for Tab, the
//! arrow keys moving inside it.

use std::rc::Rc;

use super::kit::{
    chevron, disclosure_group, divider, hstack, icon, text, vstack, Chevron, HorizontalAlignment,
};
use super::calendar_words::CalendarWords;
use super::kit_controls::icon_button;
use super::El;
use crate::environment::Token;
use crate::{Role, Semantics};

/// `options` one under another, each a title over what it means (none when
/// empty), one `selected`: a choice that needs explaining. `label` names the
/// group; the arrow keys move the choice.
pub fn radio_group<S: 'static>(
    label: impl Into<String>,
    options: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
    selected: Option<usize>,
    set: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let set = Rc::new(set);
    let rows: Vec<El<S>> = options
        .into_iter()
        .enumerate()
        .map(|(i, (title, about))| {
            let (title, about) = (title.into(), about.into());
            let on = selected == Some(i);
            let set = set.clone();
            let dot = El::div(
                "width: 18px; height: 18px; border-radius: 9px; box-sizing: border-box; \
                 flex-shrink: 0; display: flex; align-items: center; justify-content: center",
            )
            .border(if on { Token::Accent } else { Token::SecondaryText }, 1.0)
            .when(on, |e| {
                e.background(Token::Accent).child(
                    El::div("width: 8px; height: 8px; border-radius: 4px")
                        .background(Token::OnAccent),
                )
            });
            let mut words = vec![text(title.clone()).wraps()];
            if !about.is_empty() {
                words.push(
                    text(about.clone())
                        .css("font-size: 13px")
                        .foreground(Token::SecondaryText)
                        .wraps(),
                );
            }
            let mut sem = Semantics::new(Role::Radio)
                .label(title)
                .checked(on)
                .tab_index(if on || (selected.is_none() && i == 0) { 0 } else { -1 });
            if !about.is_empty() {
                sem = sem.description(about);
            }
            hstack(
                10.0,
                [
                    dot,
                    vstack(2.0, words)
                        .alignment(HorizontalAlignment::Leading)
                        .fill_width(),
                ],
            )
            .css(
                "align-self: stretch; padding: 6px 8px; border-radius: 8px; min-height: 24px; \
                 cursor: pointer",
            )
            // Half the wash: the muted words read on it.
            .hover("background: color-mix(in srgb, var(--hover) 50%, transparent)")
            .semantics(sem)
            .on_click(move |s: &mut S, _| set(s, i))
        })
        .collect();
    let n = rows.len();
    vstack(2.0, rows)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
        .role(Role::RadioGroup, label)
        .on_key(move |s, ui, key, group, cx| {
            let at = selected.unwrap_or(0);
            let to = match ui.reading_key(group, key) {
                "ArrowDown" | "ArrowRight" if n > 0 => (at + 1) % n,
                "ArrowUp" | "ArrowLeft" if n > 0 => (at + n - 1) % n,
                _ => return false,
            };
            set(s, to);
            if let Some(&row) = ui.children(group).get(to) {
                cx.focus(row);
            }
            true
        })
}

/// `sections` (a title, what is under it) of which one is `open` at a time;
/// `set` gets the one to open, or `None` to close it.
pub fn accordion<S: 'static>(
    sections: impl IntoIterator<Item = (impl Into<String>, Vec<El<S>>)>,
    open: Option<usize>,
    set: impl Fn(&mut S, Option<usize>) + 'static,
) -> El<S> {
    let set = Rc::new(set);
    let mut parts = vec![];
    for (i, (title, content)) in sections.into_iter().enumerate() {
        if i > 0 {
            parts.push(divider().key(format!("line:{i}")));
        }
        let set = set.clone();
        parts.push(
            disclosure_group(
                title,
                open == Some(i),
                move |s: &mut S, on| set(s, on.then_some(i)),
                content,
            )
            .key(format!("section:{i}")),
        );
    }
    vstack(6.0, parts)
        .alignment(HorizontalAlignment::Leading)
        .css("align-self: stretch")
}

/// Where this is: `crumbs` from the top down, the last one here. Those
/// before it go back to them (`go` gets the index).
pub fn breadcrumbs<S: 'static>(
    label: impl Into<String>,
    crumbs: impl IntoIterator<Item = impl Into<String>>,
    go: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let go = Rc::new(go);
    let crumbs: Vec<String> = crumbs.into_iter().map(Into::into).collect();
    let last = crumbs.len().saturating_sub(1);
    let mut parts = vec![];
    for (i, c) in crumbs.into_iter().enumerate() {
        if i > 0 {
            parts.push(chevron(Chevron::Forward, 12.0).foreground(Token::SecondaryText));
        }
        let base = text(c.clone()).css("padding: 4px 6px; border-radius: 6px; min-height: 16px");
        parts.push(if i == last {
            let mut here = Semantics::new(Role::StaticText).label(c);
            here.current = true;
            base.bold().semantics(here).key("here")
        } else {
            let go = go.clone();
            base.css("cursor: pointer")
                .foreground(Token::SecondaryText)
                // (muted text does not read on the wash)
                .hover("background: var(--hover); color: var(--text)")
                .role(Role::Link, c)
                .on_click(move |s: &mut S, _| go(s, i))
                .key(format!("crumb:{i}"))
        });
    }
    hstack(2.0, parts)
        .css("flex-wrap: wrap; min-width: 0")
        .role(Role::Navigation, label)
}

/// The pages (from 0) a [`pagination`] shows of `pages`, at `page`: the
/// ends, `page` and those by it; `None` where some are left out.
pub fn pages_shown(page: usize, pages: usize) -> Vec<Option<usize>> {
    if pages <= 7 {
        return (0..pages).map(Some).collect();
    }
    let page = page.min(pages - 1);
    // Always seven, so that the buttons stay where they are.
    let (from, to) = if page < 4 {
        (1, 4)
    } else if page >= pages - 4 {
        (pages - 5, pages - 2)
    } else {
        (page - 1, page + 1)
    };
    let mut out = vec![Some(0)];
    if from > 1 {
        out.push(None);
    }
    out.extend((from..=to).map(Some));
    if to < pages - 2 {
        out.push(None);
    }
    out.push(Some(pages - 1));
    out
}

/// Pages to go to, `page` (from 0) of `pages` the one shown, between a way
/// back and a way on named `names`. `label` names it all.
pub fn pagination<S: 'static>(
    label: impl Into<String>,
    page: usize,
    pages: usize,
    names: (impl Into<String>, impl Into<String>),
    go: impl Fn(&mut S, usize) + 'static,
) -> El<S> {
    let go = Rc::new(go);
    let arrow = |d: &str, name: String, to: Option<usize>| {
        let go = go.clone();
        icon_button(d, name, move |s: &mut S| {
            if let Some(to) = to {
                go(s, to)
            }
        })
        .flips_for_right_to_left()
        .disabled(to.is_none())
    };
    let mut parts = vec![arrow(
        r#"<path d="M15 18l-6-6 6-6"/>"#,
        names.0.into(),
        page.checked_sub(1),
    )
    .key("back")];
    for (i, p) in pages_shown(page, pages).into_iter().enumerate() {
        let cell = "min-width: 28px; height: 28px; box-sizing: border-box; padding: 0 6px; \
                    border-radius: 7px; font-size: 13px; display: flex; align-items: center; \
                    justify-content: center";
        parts.push(match p {
            None => text("…")
                .css(cell)
                .foreground(Token::SecondaryText)
                .semantics(Semantics::new(Role::Generic).hidden())
                .key(format!("gap:{i}")),
            Some(p) => {
                let on = p == page;
                let go = go.clone();
                let name = (p + 1).to_string();
                let mut sem = Semantics::new(Role::Button).label(name.clone());
                sem.current = on;
                text(name)
                    .css(cell)
                    .css("cursor: pointer")
                    .when(on, |e| {
                        e.background(Token::Prominent)
                            .foreground(Token::OnProminent)
                    })
                    .when(!on, |e| {
                        e.foreground(Token::Text).hover("background: var(--hover)")
                    })
                    .semantics(sem)
                    .on_click(move |s: &mut S, _| go(s, p))
                    .key(format!("page:{p}"))
            }
        });
    }
    let next = (page + 1 < pages).then_some(page + 1);
    parts.push(arrow(r#"<path d="M9 18l6-6-6-6"/>"#, names.1.into(), next).key("on"));
    hstack(2.0, parts)
        .css("flex-wrap: wrap")
        .role(Role::Navigation, label)
}

/// `value` of `max` stars, named `label`. With `set` it is worked by a
/// click on a star and by the arrow keys, Home and End (a slider to
/// assistive technology); without, it is only shown.
pub fn rating<S: 'static>(
    label: impl Into<String>,
    value: u32,
    max: u32,
    set: Option<Box<dyn Fn(&mut S, u32)>>,
) -> El<S> {
    const STAR: &str = "M12 2.5l2.9 6.2 6.8.8-5 4.7 1.3 6.7-6-3.3-6 3.3 1.3-6.7-5-4.7 6.8-.8z";
    let value = value.min(max);
    let set = set.map(Rc::new);
    let stars = (1..=max).map(|k| {
        let on = k <= value;
        let markup = if on {
            format!(r#"<path fill="currentColor" d="{STAR}"/>"#)
        } else {
            format!(r#"<path d="{STAR}"/>"#)
        };
        // (a drawing of 20 px in a target of 24)
        hstack(0.0, [icon(&markup, 20.0, 1.6)])
            .css("width: 24px; height: 24px; justify-content: center; flex-shrink: 0")
            .foreground(if on { Token::Warning } else { Token::Track })
            .when_some(set.clone(), |e, set| {
                e.css("cursor: pointer")
                    .on_click(move |s: &mut S, _| set(s, k))
            })
    });
    let role = if set.is_some() {
        Role::Slider
    } else {
        Role::Image
    };
    let sem = Semantics::new(role)
        .label(label)
        .value(format!("{value}/{max}"))
        .range(0.0, max as f32, value as f32);
    hstack(0.0, stars)
        .corner_radius(6.0)
        .semantics(sem)
        .when_some(set, |e, set| {
            e.on_key(move |s, ui, key, node, _| {
                let to = match ui.reading_key(node, key) {
                    "ArrowRight" | "ArrowUp" => (value + 1).min(max),
                    "ArrowLeft" | "ArrowDown" => value.saturating_sub(1),
                    "Home" => 0,
                    "End" => max,
                    _ => return false,
                };
                set(s, to);
                true
            })
        })
}

/// The `steps` of something done in order (a checkout, a setup), `current`
/// (from 0) the one under way: those before it ticked, those after it to
/// come.
pub fn step_indicator<S: 'static>(
    label: impl Into<String>,
    steps: impl IntoIterator<Item = impl Into<String>>,
    current: usize,
) -> El<S> {
    let steps: Vec<String> = steps.into_iter().map(Into::into).collect();
    let n = steps.len();
    let mut parts = vec![];
    for (i, name) in steps.into_iter().enumerate() {
        if i > 0 {
            let line = if i <= current { Token::Accent } else { Token::Track };
            parts.push(
                El::div("height: 2px; border-radius: 1px; flex-grow: 1; min-width: 12px")
                    .background(line)
                    .key(format!("line:{i}")),
            );
        }
        let (done, here) = (i < current, i == current);
        let disc = El::div(
            "width: 24px; height: 24px; border-radius: 12px; box-sizing: border-box; \
             flex-shrink: 0; display: flex; align-items: center; justify-content: center; \
             font-size: 12px; font-weight: 700",
        )
        .border(if done || here { Token::Accent } else { Token::Track }, 2.0)
        .when(done, |e| {
            e.background(Token::Accent).child(
                icon(r#"<path d="M5 12l5 5 9-10"/>"#, 14.0, 3.0).foreground(Token::OnAccent),
            )
        })
        .when(!done, |e| {
            e.foreground(if here { Token::AccentText } else { Token::SecondaryText })
                .text((i + 1).to_string())
        });
        let mut sem = Semantics::new(Role::ListItem).label(name.clone()).checked(done);
        sem.current = here;
        sem.position = Some((i + 1, n));
        sem.combine = true;
        parts.push(
            hstack(
                8.0,
                [
                    disc,
                    text(name)
                        .css("font-size: 13px; white-space: nowrap")
                        .foreground(if here { Token::Text } else { Token::SecondaryText })
                        .when(here, El::bold),
                ],
            )
            .css("flex-shrink: 0")
            .semantics(sem)
            .key(format!("step:{i}")),
        );
    }
    hstack(8.0, parts)
        .css("align-self: stretch; flex-wrap: wrap; row-gap: 8px")
        .role(Role::List, label)
}

/// What happened, in order: each event's title, when, and what came of it
/// (either may be empty), along a line.
pub fn timeline<S: 'static>(
    label: impl Into<String>,
    events: impl IntoIterator<Item = (impl Into<String>, impl Into<String>, impl Into<String>)>,
) -> El<S> {
    let events: Vec<(String, String, String)> = events
        .into_iter()
        .map(|(a, b, c)| (a.into(), b.into(), c.into()))
        .collect();
    let n = events.len();
    let rows = events.into_iter().enumerate().map(|(i, (title, when, about))| {
        let mut rail = vec![El::div(
            "width: 10px; height: 10px; border-radius: 5px; margin: 5px 0 0 0; flex-shrink: 0",
        )
        .background(Token::Accent)];
        if i + 1 < n {
            rail.push(El::div("width: 2px; flex-grow: 1; min-height: 12px").background(Token::Track));
        }
        let name = [title.as_str(), when.as_str(), about.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(". ");
        let mut words = vec![text(title).bold().wraps()];
        for s in [when, about].into_iter().filter(|s| !s.is_empty()) {
            words.push(
                text(s)
                    .css("font-size: 13px")
                    .foreground(Token::SecondaryText)
                    .wraps(),
            );
        }
        let mut sem = Semantics::new(Role::ListItem).label(name);
        sem.position = Some((i + 1, n));
        sem.combine = true;
        El::div("display: flex; flex-direction: row; gap: 12px; align-items: stretch")
            .children([
                vstack(3.0, rail).css("width: 10px; flex-shrink: 0; align-items: center"),
                vstack(2.0, words)
                    .alignment(HorizontalAlignment::Leading)
                    .css("flex-grow: 1; min-width: 0; padding: 0 0 14px 0"),
            ])
            .semantics(sem)
    });
    vstack(0.0, rows)
        .css("align-self: stretch; align-items: stretch")
        .role(Role::List, label)
}

/// `names` as overlapping discs of initials `size` pixels across, `shown`
/// of them and how many more; read out as all the names.
pub fn avatar_group<S: 'static>(
    names: impl IntoIterator<Item = impl Into<String>>,
    shown: usize,
    size: f32,
) -> El<S> {
    let names: Vec<String> = names.into_iter().map(Into::into).collect();
    let more = names.len().saturating_sub(shown);
    // Half the overlap on either side of each: the same read either way.
    let lap = (size * 0.14).round();
    let disc = |words: String, fill: Token, ink: Token| {
        text(words)
            .css(&format!(
                "min-width: {size}px; padding: 0 {pad}px; aspect-ratio: 1; border-radius: 999px; \
                 box-sizing: border-box; font-size: {f}px; font-weight: 700; display: flex; \
                 align-items: center; justify-content: center; flex-shrink: 0; margin: 0 -{lap}px; \
                 white-space: nowrap",
                f = (size * 0.36).round(),
                // (`size` across, wider and as tall round larger text)
                pad = (size * 0.12).round(),
            ))
            .background(fill)
            .foreground(ink)
            // The surface's, so that each stands off the one under it.
            .border(Token::Surface, 2.0)
    };
    let mut parts: Vec<El<S>> = names
        .iter()
        .take(shown)
        .map(|name| {
            let initials: String = name
                .split_whitespace()
                .filter_map(|w| w.chars().next())
                .take(2)
                .flat_map(char::to_uppercase)
                .collect();
            disc(initials, Token::Accent, Token::OnAccent)
        })
        .collect();
    if more > 0 {
        parts.push(disc(
            format!("+{more}"),
            Token::Separator,
            Token::Text,
        ));
    }
    let mut sem = Semantics::new(Role::Image).label(names.join(", "));
    sem.combine = true;
    hstack(0.0, parts)
        .css(&format!("flex-shrink: 0; padding: 0 {lap}px"))
        .semantics(sem)
}

/// A row of a [`tree`], and what is in it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeItem {
    /// Who it is, whatever it is called.
    pub id: String,
    pub label: String,
    pub expanded: bool,
    pub children: Vec<TreeItem>,
}

impl TreeItem {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            ..Self::default()
        }
    }

    pub fn expanded(mut self, on: bool) -> Self {
        self.expanded = on;
        self
    }

    pub fn children(mut self, children: impl IntoIterator<Item = TreeItem>) -> Self {
        self.children = children.into_iter().collect();
        self
    }
}

/// A [`tree`]'s row as the keys see it.
struct TreeRow {
    id: String,
    depth: usize,
    branch: bool,
    expanded: bool,
}

/// `items` and what is in those expanded, `selected` (an id) highlighted;
/// `select` and `expand` get ids. A tree to assistive technology. One stop for Tab: up and down, Home and
/// End move the selection, forward and back open and close a branch or go
/// into it and out of it. A click selects; the arrow opens and closes.
pub fn tree<S: 'static>(
    label: impl Into<String>,
    items: &[TreeItem],
    selected: Option<&str>,
    select: impl Fn(&mut S, &str) + 'static,
    expand: impl Fn(&mut S, &str, bool) + 'static,
) -> El<S> {
    // Each with how deep it is and its place among those it is with.
    type Flat<'a> = (usize, (usize, usize), &'a TreeItem);
    fn flatten<'a>(items: &'a [TreeItem], depth: usize, out: &mut Vec<Flat<'a>>) {
        for (k, i) in items.iter().enumerate() {
            out.push((depth, (k + 1, items.len()), i));
            if i.expanded {
                flatten(&i.children, depth + 1, out);
            }
        }
    }
    let mut flat = vec![];
    flatten(items, 0, &mut flat);
    let (select, expand) = (Rc::new(select), Rc::new(expand));
    let chosen = flat
        .iter()
        .position(|(_, _, i)| selected == Some(i.id.as_str()));
    let rows = flat.iter().enumerate().map(|(i, &(depth, place, item))| {
        let on = selected == Some(item.id.as_str());
        let branch = !item.children.is_empty();
        // Indented by a box, so that it is from the side lines start at.
        let mut parts = vec![El::div(&format!(
            "width: {}px; flex-shrink: 0",
            depth as f32 * 16.0
        ))];
        if branch {
            let (expand, id, open) = (expand.clone(), item.id.clone(), item.expanded);
            let towards = if open { Chevron::Down } else { Chevron::Forward };
            parts.push(
                hstack(0.0, [chevron(towards, 12.0).foreground(Token::SecondaryText)])
                    .css("width: 24px; height: 24px; justify-content: center; flex-shrink: 0; cursor: pointer")
                    .semantics(
                        Semantics::new(Role::Button)
                            .label(item.label.clone())
                            .expanded(open)
                            .tab_index(-1),
                    )
                    .on_click(move |s: &mut S, _| expand(s, &id, !open)),
            );
        } else {
            parts.push(El::div("width: 24px; flex-shrink: 0"));
        }
        parts.push(text(item.label.clone()).css("white-space: nowrap; padding: 3px 8px 3px 0"));
        let mut sem = Semantics::new(Role::TreeItem((depth + 1).min(255) as u8))
            .label(item.label.clone())
            .selected(on)
            .tab_index(if chosen.unwrap_or(0) == i { 0 } else { -1 });
        sem.position = Some(place);
        if branch {
            sem = sem.expanded(item.expanded);
        }
        let (select, id) = (select.clone(), item.id.clone());
        hstack(0.0, parts)
            .css("align-self: stretch; min-height: 26px; border-radius: 6px; cursor: pointer")
            .when(on, |e| e.background(Token::Hover))
            .when(!on, |e| e.hover("background: var(--hover)"))
            .semantics(sem)
            .on_click(move |s: &mut S, _| select(s, &id))
            .key(item.id.clone())
    });
    let rows: Vec<El<S>> = rows.collect();
    let keys: Vec<TreeRow> = flat
        .iter()
        .map(|(depth, _, i)| TreeRow {
            id: i.id.clone(),
            depth: *depth,
            branch: !i.children.is_empty(),
            expanded: i.expanded,
        })
        .collect();
    let at = chosen;
    vstack(1.0, rows)
        .css("align-self: stretch; align-items: stretch; border-radius: 8px")
        .role(Role::Tree, label)
        .on_key(move |s, ui, key, node, cx| {
            let key = ui.reading_key(node, key);
            let last = keys.len().checked_sub(1);
            let to = match (key, at) {
                ("ArrowDown", None) | ("Home", _) => (last.is_some()).then_some(0),
                ("ArrowUp", None) | ("End", _) => last,
                ("ArrowDown", Some(i)) => Some((i + 1).min(keys.len() - 1)),
                ("ArrowUp", Some(i)) => Some(i.saturating_sub(1)),
                ("ArrowRight", Some(i)) if keys[i].branch => {
                    if !keys[i].expanded {
                        expand(s, &keys[i].id, true);
                        return true;
                    }
                    Some(i + 1)
                }
                ("ArrowLeft", Some(i)) => {
                    if keys[i].branch && keys[i].expanded {
                        expand(s, &keys[i].id, false);
                        return true;
                    }
                    // Out to what it is in.
                    (0..i).rev().find(|&j| keys[j].depth < keys[i].depth)
                }
                _ => None,
            };
            let Some(to) = to else { return false };
            select(s, &keys[to].id);
            // Focus goes with it: what is read out is the row.
            if let Some(&row) = ui.children(node).get(to) {
                cx.focus(row);
                ui.scroll_into_view(row);
            }
            true
        })
}

/// A day of the (proleptic Gregorian) calendar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Day {
    pub year: i32,
    /// 1 to 12.
    pub month: u32,
    /// 1 to 31.
    pub day: u32,
}

impl Day {
    /// `None` for a day there is not (February 30).
    pub fn new(year: i32, month: u32, day: u32) -> Option<Day> {
        ((1..=12).contains(&month) && day >= 1 && day <= days_in(year, month))
            .then_some(Day { year, month, day })
    }

    /// Days since 1970-01-01.
    fn number(self) -> i64 {
        let y = self.year as i64 - i64::from(self.month <= 2);
        let era = y.div_euclid(400);
        let of_era = y.rem_euclid(400);
        let of_year = (153 * ((self.month as i64 + 9) % 12) + 2) / 5 + self.day as i64 - 1;
        era * 146_097 + of_era * 365 + of_era / 4 - of_era / 100 + of_year - 719_468
    }

    fn from_number(n: i64) -> Day {
        let z = n + 719_468;
        let era = z.div_euclid(146_097);
        let e = z.rem_euclid(146_097);
        let of_era = (e - e / 1460 + e / 36_524 - e / 146_096) / 365;
        let of_year = e - (365 * of_era + of_era / 4 - of_era / 100);
        let m = (5 * of_year + 2) / 153;
        let month = if m < 10 { m + 3 } else { m - 9 };
        Day {
            year: (of_era + era * 400 + i64::from(month <= 2)) as i32,
            month: month as u32,
            day: (of_year - (153 * m + 2) / 5 + 1) as u32,
        }
    }

    /// 0 Monday to 6 Sunday.
    pub fn weekday(self) -> u32 {
        (self.number() + 3).rem_euclid(7) as u32
    }

    /// The day `days` after it (before it, when negative).
    pub fn plus_days(self, days: i64) -> Day {
        Day::from_number(self.number() + days)
    }

    /// The same day `months` later, or that month's last.
    pub fn plus_months(self, months: i32) -> Day {
        let m = self.year * 12 + self.month as i32 - 1 + months;
        let (year, month) = (m.div_euclid(12), m.rem_euclid(12) as u32 + 1);
        Day {
            year,
            month,
            day: self.day.min(days_in(year, month)),
        }
    }
}

fn days_in(year: i32, month: u32) -> u32 {
    match month {
        4 | 6 | 9 | 11 => 30,
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        _ => 31,
    }
}

/// The month of `shown`, a day of it `selected` and `today` marked. `set`
/// gets the day picked; `show` a day of the month to show. The days are one
/// stop for Tab: the arrow keys move by days and weeks, Page Up and Down by
/// months, Home and End to the week's ends.
pub fn calendar<S: 'static>(
    shown: Day,
    selected: Option<Day>,
    today: Option<Day>,
    words: &CalendarWords,
    set: impl Fn(&mut S, Day) + 'static,
    show: impl Fn(&mut S, Day) + 'static,
) -> El<S> {
    let set = Rc::new(set);
    let pick = set.clone();
    calendar_picking(
        shown,
        selected,
        today,
        words,
        move |s: &mut S, d| set(s, d),
        move |s: &mut S, d| pick(s, d),
        show,
    )
}

/// A [`calendar`] that tells a day moved to by the keys (`set`) from one
/// picked by a click (`pick`).
pub(crate) fn calendar_picking<S: 'static>(
    shown: Day,
    selected: Option<Day>,
    today: Option<Day>,
    words: &CalendarWords,
    set: impl Fn(&mut S, Day) + 'static,
    pick: impl Fn(&mut S, Day) + 'static,
    show: impl Fn(&mut S, Day) + 'static,
) -> El<S> {
    let (pick, show) = (Rc::new(pick), Rc::new(show));
    let first = Day { day: 1, ..shown };
    let title = words.month_title(first);
    let turn = |d: &str, label: &str, by: i32| {
        let show = show.clone();
        icon_button(d, label, move |s: &mut S| show(s, first.plus_months(by)))
            .flips_for_right_to_left()
    };
    let head = hstack(
        4.0,
        [
            text(title.clone())
                .bold()
                .css("flex-grow: 1; padding: 0 0 0 8px")
                .semantics(Semantics::new(Role::Status).label(title.clone()).live()),
            turn(r#"<path d="M15 18l-6-6 6-6"/>"#, &words.previous, -1),
            turn(r#"<path d="M9 18l6-6-6-6"/>"#, &words.next, 1),
        ],
    )
    .css("align-self: stretch");
    let start = words.week_starts % 7;
    let cell = "width: 34px; height: 32px; border-radius: 8px; box-sizing: border-box; \
                font-size: 13px; display: flex; align-items: center; justify-content: center";
    let week = (0..7).map(|k| {
        text(words.weekdays[((start + k) % 7) as usize].clone())
            .css(cell)
            .css("height: 24px; font-size: 12px")
            .foreground(Token::SecondaryText)
    });
    let before = (first.weekday() + 7 - start) % 7;
    let cells = (0..42).map(|k| {
        let d = first.plus_days(k as i64 - before as i64);
        let (on, here) = (selected == Some(d), today == Some(d));
        let other = d.month != first.month;
        let pick = pick.clone();
        let mut sem = Semantics::new(Role::Button)
            .label(words.day(d))
            .selected(on)
            .tab_index(-1);
        sem.current = here;
        text(d.day.to_string())
            .css(cell)
            .css("cursor: pointer")
            .foreground(match (on, other) {
                (true, _) => Token::OnAccent,
                (_, true) => Token::SecondaryText,
                _ => Token::Text,
            })
            .when(on, |e| e.background(Token::Accent))
            .when(!on, |e| e.hover("background: var(--hover)"))
            .when(here && !on, |e| e.border(Token::AccentText, 1.0))
            .semantics(sem)
            .on_click(move |s: &mut S, _| pick(s, d))
            .key(k.to_string())
    });
    let mut sem = Semantics::new(Role::Group).label(title).tab_index(0);
    sem.value = selected.map(|d| words.day(d));
    let days = El::div(
        "display: grid; grid-template-columns: repeat(7, 34px); row-gap: 2px; border-radius: 8px",
    )
    .children(cells)
    .semantics(sem)
    .on_key(move |s, ui, key, node, _| {
        // From the day picked, if this month shows it; else its first.
        let at = selected.filter(|d| (d.year, d.month) == (first.year, first.month));
        let d = at.unwrap_or(first);
        let in_week = ((d.weekday() + 7 - start) % 7) as i64;
        let to = match ui.reading_key(node, key) {
            "PageDown" => d.plus_months(1),
            "PageUp" => d.plus_months(-1),
            "ArrowRight" | "ArrowLeft" | "ArrowUp" | "ArrowDown" if at.is_none() => d,
            "ArrowRight" => d.plus_days(1),
            "ArrowLeft" => d.plus_days(-1),
            "ArrowDown" => d.plus_days(7),
            "ArrowUp" => d.plus_days(-7),
            "Home" => d.plus_days(-in_week),
            "End" => d.plus_days(6 - in_week),
            _ => return false,
        };
        set(s, to);
        if (to.year, to.month) != (first.year, first.month) {
            show(s, to);
        }
        true
    });
    vstack(
        6.0,
        [
            head,
            El::div("display: grid; grid-template-columns: repeat(7, 34px)").children(week),
            days,
        ],
    )
    .css("width: 238px; flex-shrink: 0")
    .alignment(HorizontalAlignment::Leading)
}

/// A [`pickers::range_slider`](crate::pickers::range_slider) for a low and a
/// high value.
pub fn range_slider<S: 'static>(
    label: impl Into<String>,
    names: (impl Into<String>, impl Into<String>),
    value: crate::binding::Binding<S, (f32, f32)>,
    range: std::ops::RangeInclusive<f32>,
    step: f32,
) -> El<S> {
    let (label, lo, hi) = (label.into(), names.0.into(), names.1.into());
    El::host(
        format!("range:{label}"),
        move |ui, input, state, node| {
            crate::pickers::range_slider(
                ui,
                input,
                state,
                node,
                &label,
                (&lo, &hi),
                value.clone(),
                range.clone(),
                step,
            );
        },
    )
    .css("display: flex; flex-direction: column; align-self: stretch")
}

/// A [`fields::password_field`](crate::fields::password_field): hidden until
/// its eye, named `names` (to show, to hide), is pressed (SwiftUI's
/// `SecureField`). Hosted; see [`El::on_submit`].
pub fn password_field<S: 'static>(
    label: impl Into<String>,
    names: (impl Into<String>, impl Into<String>),
    text: crate::binding::Binding<S, String>,
) -> El<S> {
    let (label, show, hide) = (label.into(), names.0.into(), names.1.into());
    El::host(
        format!("password:{label}"),
        move |ui, input, state, node| {
            let names = (show.as_str(), hide.as_str());
            crate::fields::password_field(ui, input, state, node, &label, names, text.clone());
        },
    )
    .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// A [`fields::tag_input`](crate::fields::tag_input): words kept as chips.
/// `remove` names a chip's ✕, `{tag}` filled in. Hosted.
pub fn tag_input<S: 'static>(
    label: impl Into<String>,
    remove: impl Into<String>,
    tags: crate::binding::Binding<S, Vec<String>>,
) -> El<S> {
    let (label, remove) = (label.into(), remove.into());
    El::host(
        format!("tags:{label}"),
        move |ui, input, state, node| {
            crate::fields::tag_input(ui, input, state, node, &label, &remove, tags.clone());
        },
    )
    .css("display: flex; flex-direction: column; align-self: stretch; min-width: 0")
}

/// A [`fields::date_picker`](crate::fields::date_picker): a day, picked from
/// a [`calendar`] that opens under it (SwiftUI's `DatePicker`). Hosted.
pub fn date_picker<S: 'static>(
    label: impl Into<String>,
    placeholder: impl Into<String>,
    day: crate::binding::Binding<S, Option<Day>>,
    today: Day,
    words: CalendarWords,
) -> El<S> {
    let (label, placeholder) = (label.into(), placeholder.into());
    El::host(
        format!("date:{label}"),
        move |ui, input, state, node| {
            crate::fields::date_picker(
                ui,
                input,
                state,
                node,
                &label,
                &placeholder,
                day.clone(),
                today,
                words.clone(),
            );
        },
    )
    .css("display: flex; flex-direction: column; align-items: flex-start")
}
