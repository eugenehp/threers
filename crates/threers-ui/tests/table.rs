//! The data table: virtual rows, columns lined up under their headings,
//! sorting, selection by pointer and keys, and what assistive technology
//! reads.

use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::table::{data_table, TableColumn};
use threers_ui::{binding, FixedWidthText, Role, Ui};

#[derive(Default)]
struct S {
    rows: std::rc::Rc<Vec<Vec<String>>>,
    sort: Option<(usize, bool)>,
    selected: Option<usize>,
}

#[test]
fn a_table_sorts_selects_and_reads_well() {
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S {
        rows: std::rc::Rc::new(
            (0..100_000)
                .map(|i| vec![format!("file {i:05}"), format!("{}", (i * 7919) % 1000)])
                .collect(),
        ),
        ..S::default()
    };
    let root = ui.root();
    let t = data_table(
        &mut ui,
        &mut input,
        &s,
        root,
        "Files",
        vec![
            TableColumn::new("Name", "1fr"),
            TableColumn::new("Size", "80px").numeric(),
        ],
        binding!(|s: S| s.rows),
        binding!(|s: S| s.sort),
        binding!(|s: S| s.selected),
        24.0,
    );
    ui.update_style(t.table, |st| st.height = threers_ui::Length::Px(300.0));
    let mut time = 0.0;
    let mut frame = |input: &mut Input<S>, ui: &mut Ui, s: &mut S| {
        for _ in 0..2 {
            time += 0.016;
            input.tick(ui, s, time);
            ui.layout(500.0, 400.0, &mut FixedWidthText(7.0));
        }
    };
    frame(&mut input, &mut ui, &mut s);
    let tree = ui.accessibility_tree();
    let rows: Vec<_> = tree.iter().filter(|a| a.role == Role::Row).collect();
    assert!(
        rows.len() > 5 && rows.len() < 30,
        "only what shows: {}",
        rows.len()
    );
    assert_eq!(
        tree.iter()
            .find(|a| a.role == Role::Table)
            .unwrap()
            .row_index,
        Some((0, 100_000))
    );
    assert_eq!(
        rows[1].row_index,
        Some((2, 100_001)),
        "the first data row, after the headings"
    );
    // Cells under their headings.
    let heads: Vec<_> = tree
        .iter()
        .filter(|a| a.role == Role::ColumnHeader)
        .map(|a| a.bounds)
        .collect();
    let cells: Vec<_> = tree
        .iter()
        .filter(|a| a.role == Role::Cell)
        .take(2)
        .map(|a| a.bounds)
        .collect();
    assert_eq!((heads[1].x, heads[1].width), (cells[1].x, cells[1].width));
    // Sorted by size, then by size the other way.
    let size = tree.iter().find(|a| a.name == "Size").unwrap().id;
    let click = |input: &mut Input<S>, ui: &mut Ui, s: &mut S, n| {
        let r = ui.rect(n);
        let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
        input.dispatch(
            ui,
            s,
            Event::PointerDown {
                x,
                y,
                button: PointerButton::Primary,
            },
        );
        input.dispatch(
            ui,
            s,
            Event::PointerUp {
                x,
                y,
                button: PointerButton::Primary,
            },
        );
    };
    click(&mut input, &mut ui, &mut s, size);
    frame(&mut input, &mut ui, &mut s);
    assert_eq!(s.sort, Some((1, true)));
    let first = t.visible_order(&ui)[0];
    assert_eq!(s.rows[first][1], "0");
    assert_eq!(
        ui.accessibility_tree()
            .into_iter()
            .find(|a| a.name == "Size")
            .unwrap()
            .sort,
        Some(true)
    );
    click(&mut input, &mut ui, &mut s, size);
    frame(&mut input, &mut ui, &mut s);
    assert_eq!(s.sort, Some((1, false)));
    assert_eq!(s.rows[t.visible_order(&ui)[0]][1], "999");
    // A click selects a row; the keys move the selection down the sorted
    // order, scrolling to it.
    let row2 = ui
        .accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::Row)
        .nth(2)
        .unwrap()
        .id;
    click(&mut input, &mut ui, &mut s, row2);
    frame(&mut input, &mut ui, &mut s);
    assert_eq!(s.selected, Some(t.visible_order(&ui)[1]));
    assert!(ui
        .accessibility_tree()
        .into_iter()
        .any(|a| a.role == Role::Row && a.selected == Some(true)));
    input.set_focus(&mut ui, &mut s, Some(t.table));
    input.simulate_keystrokes(&mut ui, &mut s, "end").unwrap();
    frame(&mut input, &mut ui, &mut s);

    assert_eq!(s.selected, Some(*t.visible_order(&ui).last().unwrap()));
    let last = ui
        .accessibility_tree()
        .into_iter()
        .rfind(|a| a.role == Role::Row)
        .unwrap();
    assert_eq!(
        last.row_index,
        Some((100_001, 100_001)),
        "scrolled to the end"
    );
    assert_eq!(last.selected, Some(true));
}
