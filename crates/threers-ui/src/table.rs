//! A data table: columns with headings, rows of text, sorted by a column
//! (click or press Enter on its heading; again to reverse), one row
//! selected (a click, or the arrow keys, Home, End, Page Up and Down with
//! the table focused, scrolled into view). The rows are virtual: only those
//! that show are built, so a table of 100,000 rows costs what its view
//! does. The rows, the sort and the selection are the state's (bindings).
//! The rows are shared (`Rc`), so reading them costs nothing: to change
//! them, give the state a new `Rc` (the table sorts again when it sees
//! one).
//!
//! Assistive technology reads a table with column headings (each saying
//! how it is sorted), and rows that know their place among all the rows.
//!
//! ```
//! use threers_ui::input::Input;
//! use threers_ui::table::{data_table, TableColumn};
//! use threers_ui::{binding, style, FixedWidthText, Ui};
//!
//! #[derive(Default)]
//! struct S { rows: std::rc::Rc<Vec<Vec<String>>>, sort: Option<(usize, bool)>, selected: Option<usize> }
//!
//! let mut ui = Ui::new();
//! let mut input: Input<S> = Input::new();
//! let mut s = S { rows: std::rc::Rc::new(vec![vec!["b".into(), "2".into()], vec!["a".into(), "10".into()]]), ..S::default() };
//! let root = ui.root();
//! let t = data_table(&mut ui, &mut input, &s, root, "Files",
//!     vec![TableColumn::new("Name", "1fr"), TableColumn::new("Size", "60px").numeric()],
//!     binding!(|s: S| s.rows), binding!(|s: S| s.sort), binding!(|s: S| s.selected), 24.0);
//! ui.update_style(t.table, |st| st.height = threers_ui::Length::Px(200.0));
//! s.sort = Some((1, true));
//! for k in 0..3 {
//!     input.tick(&mut ui, &mut s, k as f64);
//!     ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
//! }
//! assert_eq!(t.visible_order(&ui), [0, 1], "2 before 10: sorted as numbers");
//! ```

use std::cell::RefCell;
use std::rc::Rc;

use crate::binding::Binding;
use crate::environment::Token;
use crate::input::Input;
use crate::tree::{NodeId, Ui};
use crate::virtual_list::{RowHeight, VirtualList};
use crate::{style, Display, Role, Semantics, Style};

/// A column of a [`data_table`].
#[derive(Debug, Clone, PartialEq)]
pub struct TableColumn {
    pub title: String,
    /// Its width as a grid track: `"120px"`, `"1fr"`, `"2fr"`.
    pub width: String,
    /// Holds numbers: aligned to the right, sorted by value.
    pub numeric: bool,
    /// Its heading sorts the table.
    pub sortable: bool,
}

impl TableColumn {
    pub fn new(title: impl Into<String>, width: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            width: width.into(),
            numeric: false,
            sortable: true,
        }
    }

    pub fn numeric(mut self) -> Self {
        self.numeric = true;
        self
    }

    pub fn unsortable(mut self) -> Self {
        self.sortable = false;
        self
    }
}

/// A table's nodes, and the order its rows show in.
pub struct DataTable {
    /// The whole table (give it a height: its body scrolls).
    pub table: NodeId,
    pub header: NodeId,
    order: Rc<RefCell<Vec<usize>>>,
}

impl DataTable {
    /// The rows' indices in the order they show (sorted).
    pub fn visible_order(&self, _ui: &Ui) -> Vec<usize> {
        self.order.borrow().clone()
    }
}

/// `a` against `b`, as numbers when both are.
fn compare(a: &str, b: &str, numeric: bool) -> std::cmp::Ordering {
    let num = |s: &str| {
        s.trim()
            .trim_end_matches('%')
            .replace([',', '_'], "")
            .parse::<f64>()
            .ok()
    };
    match (numeric, num(a), num(b)) {
        (true, Some(x), Some(y)) => x.total_cmp(&y),
        _ => a.to_lowercase().cmp(&b.to_lowercase()),
    }
}

fn row_style(template: &str) -> Style {
    let mut s = style! { display: grid; align-items: center; column-gap: 12px; padding: 0 10px; cursor: pointer; };
    crate::css::apply(&mut s, "grid-template-columns", template);
    s
}

/// A table under `parent` of `columns`, its `rows` (each a cell's text a
/// column) sorted by `sort` (a column and whether ascending), with
/// `selected` (an index into `rows`) highlighted, `row_height` pixels a
/// row. See the [module docs](self).
#[allow(clippy::too_many_arguments)]
pub fn data_table<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    columns: Vec<TableColumn>,
    rows: Binding<S, Rc<Vec<Vec<String>>>>,
    sort: Binding<S, Option<(usize, bool)>>,
    selected: Binding<S, Option<usize>>,
    row_height: f32,
) -> DataTable {
    let template: String = columns
        .iter()
        .map(|c| c.width.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let table = ui.add_box(parent, style! { display: flex; flex-direction: column; border-radius: 8px; overflow: hidden; min-height: 0; });
    ui.token_style(
        table,
        &[
            ("background", Token::Surface),
            ("border-color", Token::Border),
        ],
    );
    let mut sem = Semantics::new(Role::Table).label(label).tab_index(0);
    sem.row_index = Some((0, rows.get(state).len()));
    ui.set_semantics(table, sem);
    ui.focus_visible_style(table, "box-shadow: 0 0 0 2px var(--focus)");
    // The headings, over the scrolling body, on the same columns.
    let header = ui.add_box(table, row_style(&template));
    ui.update_style(header, |s| {
        s.min_height = crate::Length::Px(row_height + 4.0);
        s.flex_shrink = 0.0;
        s.cursor = crate::Cursor::Default;
    });
    ui.token_style(header, &[("background", Token::Raised)]);
    ui.set_semantics(header, Semantics::new(Role::Row));
    let mut heads = vec![];
    for (k, c) in columns.iter().enumerate() {
        let h = ui.add_box(header, style! { display: flex; gap: 4px; font-weight: bold; font-size: 12px; min-width: 0; overflow: hidden; white-space: nowrap; });
        if c.numeric {
            ui.update_style(h, |s| s.justify_content = crate::JustifyContent::FlexEnd);
        }
        if c.sortable {
            ui.update_style(h, |s| s.cursor = crate::Cursor::Pointer);
        }
        ui.add_text(h, c.title.clone());
        let arrow = ui.add_text(h, "");
        ui.set_semantics(
            h,
            Semantics::new(Role::ColumnHeader)
                .label(c.title.clone())
                .tab_index(if c.sortable { -1 } else { 0 }),
        );
        heads.push((h, arrow));
        if c.sortable {
            let sort = sort.clone();
            input.on_click(h, move |s, _, _| {
                let now = sort.get(s);
                sort.set(s, Some((k, !matches!(now, Some((col, true)) if col == k))));
            });
        }
    }
    let body_style = style! { flex-grow: 1; min-height: 0; height: 100px; scrollbar-width: thin; scrollbar-visibility: auto; };
    let count = rows.get(state).len();
    let list = Rc::new(RefCell::new(VirtualList::new(
        ui,
        table,
        body_style,
        count,
        RowHeight::Uniform(row_height),
    )));
    // A click on a row selects it (the row from the node clicked: rows are
    // made and dropped as the body scrolls).
    let built_rows: Rc<RefCell<Vec<(NodeId, usize)>>> = Rc::default();
    let container = list.borrow().container();
    {
        let (sel, b) = (selected.clone(), built_rows.clone());
        input.on(
            container,
            crate::input::Phase::Bubble,
            move |s, ui, e, _| {
                if let crate::input::Event::PointerDown { x, y, .. } = *e
                    && let Some(hit) = ui.hit_test(x, y)
                    && let Some(&(_, i)) = b
                        .borrow()
                        .iter()
                        .find(|(row, _)| ui.ancestors(hit).any(|a| a == *row))
                {
                    sel.set(s, Some(i));
                }
            },
        );
    }
    let order: Rc<RefCell<Vec<usize>>> = Rc::default();
    // The keys move the selection along the sorted order.
    {
        let (sel, ord, l) = (selected.clone(), order.clone(), list.clone());
        input.on_key(table, move |s, ui, key, _, _| {
            let o = ord.borrow();
            if o.is_empty() {
                return false;
            }
            let page = (ui.scroll_extent(l.borrow().container()).1 / row_height)
                .floor()
                .max(1.0) as usize;
            let at = sel.get(s).and_then(|i| o.iter().position(|&r| r == i));
            let to = match (key, at) {
                ("ArrowDown", None) | ("Home", _) => 0,
                ("ArrowUp", None) | ("End", _) => o.len() - 1,
                ("ArrowDown", Some(p)) => (p + 1).min(o.len() - 1),
                ("ArrowUp", Some(p)) => p.saturating_sub(1),
                ("PageDown", p) => (p.unwrap_or(0) + page).min(o.len() - 1),
                ("PageUp", p) => p.unwrap_or(0).saturating_sub(page),
                _ => return false,
            };
            sel.set(s, Some(o[to]));
            l.borrow_mut().scroll_to(ui, to);
            true
        });
    }
    // Kept in step with the state: sorted when the rows or the sort change,
    // rows built as they show, the selection drawn.
    let ord = order.clone();
    // The rows and the sort the order was last made for.
    type Sorted = (Rc<Vec<Vec<String>>>, Option<(usize, bool)>);
    let mut sorted_for: Option<Sorted> = None;
    let mut drawn_sel: Option<Option<usize>> = None;
    let cols = columns.clone();
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(table) {
            return false;
        }
        let data = rows.get(s);
        let how = sort.get(s);
        // New rows are a new `Rc`: compared by pointer, not by content.
        let resort = sorted_for.as_ref().is_none_or(|(d, h)| !Rc::ptr_eq(d, &data) || *h != how);
        if resort {
            let mut o: Vec<usize> = (0..data.len()).collect();
            if let Some((c, up)) = how {
                let numeric = cols.get(c).is_some_and(|c| c.numeric);
                o.sort_by(|&a, &b| {
                    let (x, y) = (data[a].get(c).map_or("", |v| v), data[b].get(c).map_or("", |v| v));
                    let ord = compare(x, y, numeric);
                    if up { ord } else { ord.reverse() }
                });
            }
            *ord.borrow_mut() = o;
            for (k, &(h, arrow)) in heads.iter().enumerate() {
                let mine = how.filter(|(c, _)| *c == k).map(|(_, up)| up);
                ui.set_text(arrow, match mine {
                    Some(true) => "▲",
                    Some(false) => "▼",
                    None => "",
                });
                ui.semantics_mut(h).sort = mine;
            }
            let mut l = list.borrow_mut();
            l.set_len(ui, data.len());
            l.rebuild(ui);
            ui.semantics_mut(table).row_index = Some((0, data.len()));
            sorted_for = Some((data.clone(), how));
            drawn_sel = None;
        }
        // Build the rows that show.
        let sel = selected.get(s);
        let o = ord.borrow().clone();
        let mut made: Vec<(NodeId, usize)> = vec![];
        let tpl = template.clone();
        let count = o.len();
        list.borrow_mut().sync(ui, |ui, row, at| {
            let Some(&i) = o.get(at) else { return };
            // As tall as the list counts it: rows are placed by that.
            let mut st = row_style(&tpl);
            st.height = crate::Length::Px(row_height);
            st.clip = true;
            ui.set_style(row, st);
            for (k, c) in cols.iter().enumerate() {
                let cell = ui.add_box(row, style! { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-size: 13px; });
                if c.numeric {
                    ui.update_style(cell, |s| s.display = Display::Flex);
                    ui.update_style(cell, |s| s.justify_content = crate::JustifyContent::FlexEnd);
                }
                ui.add_text(cell, data[i].get(k).cloned().unwrap_or_default());
                ui.set_semantics(cell, Semantics::new(Role::Cell));
            }
            let mut sem = Semantics::new(Role::Row).label(data[i].join(", "));
            sem.row_index = Some((at + 2, count + 1));
            ui.set_semantics(row, sem);
            made.push((row, i));
        });
        {
            let mut b = built_rows.borrow_mut();
            b.retain(|(r, _)| ui.contains(*r));
            b.extend(made);
        }
        // The selection: drawn on the rows built (again when they change).
        let built: Vec<(NodeId, usize)> = built_rows.borrow().clone();
        if drawn_sel != Some(sel) || resort || !built.is_empty() {
            for (row, i) in built {
                let on = Some(i) == sel;
                let want = on.then(|| ui.token(Token::Accent));
                if ui.style(row).background != want {
                    ui.style_mut(row).background = want;
                    ui.style_mut(row).color = if on { ui.token(Token::OnAccent) } else { [1.0; 4] };
                    ui.semantics_mut(row).selected = Some(on);
                }
            }
            drawn_sel = Some(sel);
        }
        true
    });
    DataTable {
        table,
        header,
        order,
    }
}
