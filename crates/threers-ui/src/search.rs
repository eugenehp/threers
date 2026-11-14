//! Searching a list as it is typed: [`fuzzy_match`] finds a query's letters
//! in order in a text (scoring word starts and runs higher) and says which
//! they are; [`filter`] ranks a list by it; [`highlight`] makes text runs
//! that set the matched letters off; and [`searchable`] puts them together:
//! a search field over a list of results that narrows as you type, its
//! matches in bold, arrows to move through it and Enter to pick.
//!
//! ```
//! use threers_ui::search::{filter, fuzzy_match};
//!
//! let (score, at) = fuzzy_match("tsb", "Toggle Sidebar").unwrap();
//! assert_eq!(at, [0, 7, 11]);
//! assert!(score > 0);
//! let fruit = ["Apple", "Banana", "Apricot", "Grape"];
//! let found: Vec<&str> = filter("ap", &fruit, |f| f).into_iter().map(|(i, _)| fruit[i]).collect();
//! assert_eq!(found, ["Apple", "Apricot", "Grape"]);
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::binding::Binding;
use crate::environment::Token;
use crate::input::Input;
use crate::rich_text::TextRun;
use crate::tree::{NodeId, Ui};
use crate::{style, FontWeight, Role, Semantics};

/// How well `query` matches `text`, its letters in order (case aside),
/// and where (character indices); `None` if they are not all there. Spaces
/// in the query are skipped.
pub fn fuzzy_match(query: &str, text: &str) -> Option<(i32, Vec<usize>)> {
    let label: Vec<char> = text.chars().collect();
    let lower: Vec<char> = label
        .iter()
        .map(|c| c.to_lowercase().next().unwrap_or(*c))
        .collect();
    let mut score = 0;
    let mut at = 0;
    let mut found_at = vec![];
    let mut last: Option<usize> = None;
    for q in query.chars().filter(|c| !c.is_whitespace()) {
        let q = q.to_lowercase().next().unwrap_or(q);
        // A letter at a word's start, if there is one ahead, before any.
        let starts = (at..label.len())
            .find(|&i| lower[i] == q && (i == 0 || !label[i - 1].is_alphanumeric()));
        let next = (at..label.len()).find(|&i| lower[i] == q)?;
        // In a run with the last one, that; else a word's start; else the next.
        let found = if last.is_some_and(|l| l + 1 == next) {
            next
        } else {
            starts.unwrap_or(next)
        };
        score += 1;
        if found == 0 || !label[found - 1].is_alphanumeric() {
            score += 8;
        }
        if last.is_some_and(|l| l + 1 == found) {
            score += 5;
        }
        score -= (found - at) as i32 / 4;
        last = Some(found);
        found_at.push(found);
        at = found + 1;
    }
    Some((score, found_at))
}

/// `items` that match `query` (all of them for an empty one), best first
/// (ties in their order), each with where it matched.
pub fn filter<T>(query: &str, items: &[T], label: impl Fn(&T) -> &str) -> Vec<(usize, Vec<usize>)> {
    let mut found: Vec<(i32, usize, Vec<usize>)> = items
        .iter()
        .enumerate()
        .filter_map(|(i, t)| fuzzy_match(query, label(t)).map(|(s, at)| (s, i, at)))
        .collect();
    found.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    found.into_iter().map(|(_, i, at)| (i, at)).collect()
}

/// Text runs for a text of `len` characters with the characters at
/// `matched` in bold: for [`Ui::set_text_runs`].
pub fn highlight(len: usize, matched: &[usize]) -> Vec<TextRun> {
    let mut runs = vec![];
    let mut i = 0;
    while i < len {
        let on = matched.contains(&i);
        let n = (i..len).take_while(|j| matched.contains(j) == on).count();
        let run = TextRun::new(n);
        runs.push(if on {
            run.weight(FontWeight::Bold)
        } else {
            run
        });
        i += n;
    }
    runs
}

/// A search field over results: what [`searchable`] made.
pub struct Searchable {
    pub node: NodeId,
    pub field: NodeId,
    pub results: NodeId,
}

/// The list's rows, the query they were built for, and the one marked.
struct Shown {
    query: String,
    rows: Vec<(NodeId, usize)>,
    marked: usize,
}

/// A search field bound to `query` over the `items` that match it (all of
/// them while it is empty), best first, the matched letters in bold, at
/// most `limit` shown. Up and down move a mark through them and Enter (or a
/// click) calls `pick` with the item's index. Assistive technology hears a
/// search box and a list, and how many matched.
#[allow(clippy::too_many_arguments)]
pub fn searchable<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    placeholder: &str,
    query: Binding<S, String>,
    items: Vec<String>,
    limit: usize,
    pick: impl Fn(&mut S, usize) + 'static,
) -> Searchable {
    let node = ui.add_box(
        parent,
        style! { display: flex; flex-direction: column; gap: 6px; },
    );
    let field = crate::pickers::search_field(ui, input, state, node, placeholder, query.clone());
    let results = ui.add_box(
        node,
        style! { display: flex; flex-direction: column; gap: 2px; },
    );
    ui.set_semantics(
        results,
        Semantics::new(Role::List).label(format!("Results for {placeholder}")),
    );
    let items = Rc::new(items);
    let pick = Rc::new(pick);
    let shown = Rc::new(RefCell::new(Shown {
        query: "\u{0}".into(),
        rows: vec![],
        marked: 0,
    }));
    // The rows for the query now, and the mark on one.
    let build = {
        let (items, shown, pick) = (items.clone(), shown.clone(), pick.clone());
        move |ui: &mut Ui, input: &mut Input<S>, q: &str| {
            let mut sh = shown.borrow_mut();
            for (r, _) in sh.rows.drain(..) {
                ui.remove(r);
            }
            let found = filter(q, &items, |s| s.as_str());
            for (i, at) in found.iter().take(limit) {
                let row = ui.add_box(
                    results,
                    style! { padding: 5px 8px; border-radius: 6px; cursor: pointer; },
                );
                let t = ui.add_text(row, items[*i].as_str());
                if !q.trim().is_empty() {
                    ui.set_text_runs(t, highlight(items[*i].chars().count(), at));
                }
                ui.set_semantics(
                    row,
                    Semantics::new(Role::ListItem).label(items[*i].as_str()),
                );
                ui.hover_style(row, "background: rgba(128,138,160,0.2)");
                let (i, pick) = (*i, pick.clone());
                input.on_click(row, move |s, _, _| pick(s, i));
                sh.rows.push((row, i));
            }
            if found.is_empty() {
                let none = ui.add_box(results, style! { padding: 5px 8px; font-size: 12px; });
                ui.token_style(none, &[("color", Token::SecondaryText)]);
                ui.add_text(none, "No results");
                sh.rows.push((none, usize::MAX));
            }
            ui.semantics_mut(results).value = Some(format!("{} found", found.len()));
            sh.query = q.to_owned();
            sh.marked = 0;
        }
    };
    let build = Rc::new(build);
    let mark = {
        let shown = shown.clone();
        move |ui: &mut Ui| {
            let sh = shown.borrow();
            for (k, &(row, i)) in sh.rows.iter().enumerate() {
                if i == usize::MAX {
                    continue;
                }
                let on = k == sh.marked;
                ui.update_style(row, |s| {
                    s.background = on.then_some([0.145, 0.388, 0.922, 0.16])
                });
                ui.semantics_mut(row).selected = Some(on);
            }
        }
    };
    let mark = Rc::new(mark);
    build(ui, input, &query.get(state));
    mark(ui);
    // Rebuilt when the query changes.
    let (b, m, q, sh) = (build.clone(), mark.clone(), query.clone(), shown.clone());
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(results) {
            return false;
        }
        let now = q.get(s);
        if sh.borrow().query != now {
            b(ui, input, &now);
            m(ui);
        }
        true
    });
    // Up and down move the mark; Enter picks it.
    let (sh, m) = (shown.clone(), mark.clone());
    input.on_key(field, move |s, ui, key, _, _| {
        let n = sh
            .borrow()
            .rows
            .iter()
            .filter(|r| r.1 != usize::MAX)
            .count();
        match key {
            "ArrowDown" | "ArrowUp" if n > 0 => {
                {
                    let mut st = sh.borrow_mut();
                    st.marked = if key == "ArrowDown" {
                        (st.marked + 1) % n
                    } else {
                        (st.marked + n - 1) % n
                    };
                }
                m(ui);
                true
            }
            "Enter" if n > 0 => {
                let i = {
                    let st = sh.borrow();
                    st.rows[st.marked].1
                };
                pick(s, i);
                true
            }
            _ => false,
        }
    });
    Searchable {
        node,
        field,
        results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{binding, FixedWidthText};

    #[derive(Default)]
    struct S {
        query: String,
        picked: Option<usize>,
    }

    fn rows(ui: &Ui, s: &Searchable) -> Vec<String> {
        ui.children(s.results)
            .iter()
            .map(|&r| ui.text_lines_of(ui.children(r)[0]).join(""))
            .collect()
    }

    #[test]
    fn it_narrows_as_typed_marks_the_best_and_enter_picks_it() {
        let mut ui = Ui::new();
        let mut input: Input<S> = Input::new();
        let mut s = S::default();
        let root = ui.root();
        let names = [
            "Toggle Sidebar",
            "Open File",
            "Save File",
            "Toggle Terminal",
            "Settings",
        ]
        .map(String::from)
        .to_vec();
        let found = searchable(
            &mut ui,
            &mut input,
            &s,
            root,
            "Commands",
            binding!(|s: S| s.query),
            names,
            10,
            |s, i| s.picked = Some(i),
        );
        ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
        assert_eq!(rows(&ui, &found).len(), 5, "all while empty");
        input.set_focus(&mut ui, &mut s, Some(found.field));
        input.simulate_keystrokes(&mut ui, &mut s, "t t").unwrap();
        ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
        assert_eq!(s.query, "tt");
        // ("Toggle Sidebar" has one t.)
        assert_eq!(rows(&ui, &found), ["Toggle Terminal", "Settings"]);
        // Down to the second; Enter picks it.
        input
            .simulate_keystrokes(&mut ui, &mut s, "down enter")
            .unwrap();
        assert_eq!(s.picked, Some(4), "Settings is item 4");
        input.simulate_keystrokes(&mut ui, &mut s, "z z z").unwrap();
        ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
        assert_eq!(rows(&ui, &found), ["No results"]);
    }

    #[test]
    fn matched_letters_are_bold() {
        let runs = highlight(5, &[0, 1, 4]);
        assert_eq!(
            runs.iter()
                .map(|r| (r.len, r.weight.is_some()))
                .collect::<Vec<_>>(),
            [(2, true), (2, false), (1, true)]
        );
    }
}
