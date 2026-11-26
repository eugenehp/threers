//! The kit's choosers, ways about and progress through things: worked by
//! pointer and keyboard, and read as assistive technology reads them.

use threers_ui::a11y::AccessNode;
use threers_ui::binding::Binding;
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::states::Appearance;
use threers_ui::view::{
    accordion, avatar_group, breadcrumbs, calendar, pages_shown, pagination, radio_group,
    range_slider, rating, step_indicator, text, timeline, tree, CalendarWords, Day, El, TreeItem,
    View,
};
use threers_ui::{FixedWidthText, NodeId, Role, Ui};

#[derive(Default)]
struct App {
    at: Option<usize>,
    page: usize,
    stars: u32,
    range: (f32, f32),
    picked: Option<String>,
    open: Vec<String>,
    day: Option<Day>,
    month: Option<Day>,
}

struct H {
    ui: Ui,
    input: Input<App>,
    view: View<App>,
    app: App,
    body: fn(&App) -> Vec<El<App>>,
}

impl H {
    fn new(body: fn(&App) -> Vec<El<App>>) -> H {
        let mut ui = Ui::new();
        let root = ui.root();
        let view = View::new(&mut ui, root, "display: flex; flex-direction: column; width: 500px");
        let mut h = H { ui, input: Input::new(), view, app: App::default(), body };
        h.show();
        h
    }
    fn show(&mut self) {
        let els = (self.body)(&self.app);
        self.view.update_with(&mut self.ui, &mut self.input, &self.app, els);
        self.ui.layout(500.0, 600.0, &mut FixedWidthText(7.0));
    }
    fn all(&self, role: Role) -> Vec<AccessNode> {
        self.ui.accessibility_tree().into_iter().filter(|a| a.role == role).collect()
    }
    fn find(&self, role: Role, name: &str) -> AccessNode {
        self.all(role).into_iter().find(|a| a.name == name).unwrap_or_else(|| panic!("no {role:?} {name:?}"))
    }
    fn press_at(&mut self, x: f32, y: f32) {
        let button = PointerButton::Primary;
        for e in [Event::PointerMove { x, y }, Event::PointerDown { x, y, button }, Event::PointerUp { x, y, button }] {
            self.input.dispatch(&mut self.ui, &mut self.app, e);
        }
        self.show();
    }
    fn press(&mut self, id: NodeId) {
        let r = self.ui.rect(id);
        self.press_at(r.x + r.width / 2.0, r.y + r.height / 2.0);
    }
    fn focus(&mut self, id: NodeId) {
        self.input.set_focus(&mut self.ui, &mut self.app, Some(id));
    }
    /// Keys one at a time, the view following each as an app's does.
    fn keys(&mut self, keys: &str) {
        for k in keys.split(' ') {
            self.input.simulate_keystrokes(&mut self.ui, &mut self.app, k).unwrap();
            self.show();
        }
    }
    /// Nothing to trouble anyone, in either appearance.
    fn audit(&mut self) {
        for a in [Appearance::Light, Appearance::Dark] {
            self.ui.set_system_appearance(a);
            let css = "display: flex; flex-direction: column; width: 500px";
            let root = self.view.root();
            self.view.set_root_css(&mut self.ui, css);
            self.ui.token_style(root, &[("background", threers_ui::environment::Token::Surface), ("color", threers_ui::environment::Token::Text)]);
            self.show();
            let page = self.ui.token(threers_ui::environment::Token::Background);
            assert_eq!(self.ui.accessibility_audit(), vec![], "{a:?}");
            assert_eq!(self.ui.contrast_issues(page, false), vec![], "{a:?}");
        }
    }
}

#[test]
fn a_radio_group_is_one_stop_and_its_arrows_choose() {
    let mut h = H::new(|a| {
        vec![radio_group("Plan", [("Free", "For one"), ("Team", "For five"), ("All", "")], a.at, |a: &mut App, i| a.at = Some(i))]
    });
    let team = h.find(Role::Radio, "Team");
    assert_eq!(team.description.as_deref(), Some("For five"));
    h.press(team.id);
    assert_eq!(h.app.at, Some(1));
    assert_eq!(h.find(Role::Radio, "Team").checked, Some(true));
    h.keys("down");
    assert_eq!(h.app.at, Some(2));
    assert_eq!(h.input.focused(), Some(h.find(Role::Radio, "All").id), "focus goes with it");
    h.keys("down");
    assert_eq!(h.app.at, Some(0), "wraps");
    assert_eq!(h.all(Role::Radio).iter().filter(|r| h.ui.semantics(r.id).unwrap().tab_index == Some(0)).count(), 1);
    h.audit();
}

#[test]
fn an_accordion_has_one_section_open() {
    let mut h = H::new(|a| {
        vec![accordion([("One", vec![text("first")]), ("Two", vec![text("second")])], a.at, |a: &mut App, i| a.at = i)]
    });
    let shown = |h: &H| h.all(Role::StaticText).iter().map(|n| n.name.clone()).filter(|n| n == "first" || n == "second").collect::<Vec<_>>();
    assert!(shown(&h).is_empty());
    h.press(h.find(Role::Button, "One").id);
    assert_eq!(shown(&h), ["first"]);
    h.press(h.find(Role::Button, "Two").id);
    assert_eq!(shown(&h), ["second"]);
    assert_eq!(h.find(Role::Button, "One").expanded, Some(false));
    h.press(h.find(Role::Button, "Two").id);
    assert!(shown(&h).is_empty());
    h.audit();
}

#[test]
fn breadcrumbs_go_back_and_say_which_is_here() {
    let mut h = H::new(|_| vec![breadcrumbs("Path", ["Home", "Docs", "Intro"], |a: &mut App, i| a.at = Some(i))]);
    assert_eq!(h.all(Role::Link).len(), 2, "here is no link");
    assert!(h.find(Role::StaticText, "Intro").current);
    h.press(h.find(Role::Link, "Docs").id);
    assert_eq!(h.app.at, Some(1));
    h.audit();
}

#[test]
fn pages_keep_their_places() {
    for pages in 0..30 {
        for page in 0..pages.max(1) {
            let p = pages_shown(page, pages);
            assert_eq!(p.len(), pages.min(7), "{page} of {pages}: {p:?}");
            let got: Vec<usize> = p.iter().flatten().copied().collect();
            assert!(got.windows(2).all(|w| w[0] < w[1]), "{p:?}");
            if pages > 0 {
                assert!(got.contains(&page) && got.contains(&0) && got.contains(&(pages - 1)), "{page}: {p:?}");
                // A gap stands for pages, never for none.
                for (i, g) in p.iter().enumerate().filter(|(_, g)| g.is_none()) {
                    assert!(p[i + 1].unwrap() - p[i - 1].unwrap() > 1, "{g:?} {p:?}");
                }
            }
        }
    }
    assert_eq!(pages_shown(99, 3).len(), 3, "past the end");
}

#[test]
fn pagination_turns_pages_and_stops_at_the_ends() {
    let mut h = H::new(|a| vec![pagination("Results", a.page, 20, ("Back", "On"), |a: &mut App, p| a.page = p)]);
    assert!(h.find(Role::Button, "Back").disabled);
    h.press(h.find(Role::Button, "On").id);
    assert_eq!(h.app.page, 1);
    assert!(h.find(Role::Button, "2").current && !h.find(Role::Button, "1").current);
    h.press(h.find(Role::Button, "20").id);
    assert_eq!(h.app.page, 19);
    assert!(h.find(Role::Button, "On").disabled && !h.find(Role::Button, "Back").disabled);
    h.press(h.find(Role::Button, "Back").id);
    assert_eq!(h.app.page, 18, "usable again");
    h.audit();
}

#[test]
fn a_rating_is_given_by_star_or_key_or_only_shown() {
    let mut h = H::new(|a| vec![rating("Score", a.stars, 5, Some(Box::new(|a: &mut App, n| a.stars = n)))]);
    let s = h.find(Role::Slider, "Score");
    let r = s.bounds;
    h.press_at(r.x + 24.0 * 3.5, r.y + 12.0);
    assert_eq!(h.app.stars, 4);
    h.focus(s.id);
    h.keys("right right");
    assert_eq!(h.app.stars, 5, "no further");
    h.keys("home left");
    assert_eq!(h.app.stars, 0);
    assert_eq!(h.find(Role::Slider, "Score").value.as_deref(), Some("0/5"));
    h.audit();
    let mut h = H::new(|_| vec![rating("Score", 9, 5, None)]);
    assert_eq!(h.find(Role::Image, "Score").value.as_deref(), Some("5/5"));
    h.audit();
}

#[test]
fn steps_and_events_are_read_in_order() {
    let mut h = H::new(|_| {
        vec![
            step_indicator("Checkout", ["Cart", "Pay", "Done"], 1),
            timeline("History", [("Sent", "Mon", ""), ("Arrived", "Tue", "At the door")]),
            avatar_group(["Ada Lovelace", "Alan Turing", "Grace Hopper"], 2, 32.0),
        ]
    });
    let steps: Vec<_> = h.all(Role::ListItem).into_iter().map(|n| (n.name, n.current, n.position)).collect();
    assert_eq!(steps[0], ("Cart".into(), false, Some((1, 3))));
    assert_eq!(steps[1], ("Pay".into(), true, Some((2, 3))));
    assert_eq!(steps[4].0, "Arrived. Tue. At the door");
    h.find(Role::Image, "Ada Lovelace, Alan Turing, Grace Hopper");
    h.audit();
}

fn files(a: &App) -> Vec<El<App>> {
    let open = |id: &str| a.open.iter().any(|o| o == id);
    let items = [
        TreeItem::new("src", "src").expanded(open("src")).children([
            TreeItem::new("view", "view").expanded(open("view")).children([TreeItem::new("kit", "kit.rs")]),
            TreeItem::new("lib", "lib.rs"),
        ]),
        TreeItem::new("readme", "README"),
    ];
    vec![tree(
        "Files",
        &items,
        a.picked.as_deref(),
        |a: &mut App, id| a.picked = Some(id.into()),
        |a: &mut App, id, on| {
            a.open.retain(|o| o != id);
            if on {
                a.open.push(id.into());
            }
        },
    )]
}

#[test]
fn a_tree_is_walked_by_the_arrows() {
    let mut h = H::new(files);
    h.focus(h.find(Role::TreeItem(1), "src").id);
    h.keys("down");
    assert_eq!(h.app.picked.as_deref(), Some("src"));
    assert_eq!(h.input.focused(), Some(h.find(Role::TreeItem(1), "src").id));
    h.keys("right right");
    assert_eq!(h.app.picked.as_deref(), Some("view"), "opened, then into it");
    h.keys("right right");
    assert_eq!(h.app.picked.as_deref(), Some("kit"));
    assert_eq!(h.input.focused(), Some(h.find(Role::TreeItem(3), "kit.rs").id), "focus goes with it");
    assert_eq!(h.find(Role::TreeItem(2), "lib.rs").position, Some((2, 2)), "among its own");
    h.keys("left");
    assert_eq!(h.app.picked.as_deref(), Some("view"), "out of it");
    h.keys("left left left");
    assert_eq!(h.app.picked.as_deref(), Some("src"));
    assert_eq!(h.ui.accessibility_tree().iter().filter(|n| matches!(n.role, Role::TreeItem(_))).count(), 2, "closed");
    h.keys("end");
    assert_eq!(h.app.picked.as_deref(), Some("readme"));
    h.keys("right down");
    assert_eq!(h.app.picked.as_deref(), Some("readme"), "a leaf, the last");
    let stops = h.ui.focus_order();
    assert_eq!(stops, [h.find(Role::TreeItem(1), "README").id], "one stop");
}

#[test]
fn a_trees_arrow_opens_and_its_row_selects() {
    let mut h = H::new(files);
    h.press(h.find(Role::Button, "src").id);
    assert_eq!((h.app.open.as_slice(), &h.app.picked), (&["src".to_string()][..], &None));
    h.press(h.find(Role::TreeItem(2), "lib.rs").id);
    assert_eq!(h.app.picked.as_deref(), Some("lib"));
    h.press(h.find(Role::TreeItem(1), "README").id);
    let drawn = |h: &H, n: &str| h.ui.style(h.ui.accessibility_tree().into_iter().find(|a| a.name == n && matches!(a.role, Role::TreeItem(_))).unwrap().id).background.is_some_and(|c| c[3] > 0.0);
    h.input.dispatch(&mut h.ui, &mut h.app, Event::PointerMove { x: 499.0, y: 599.0 });
    assert!(drawn(&h, "README") && !drawn(&h, "lib.rs"), "one is selected");
    h.audit();
    // A tree of nothing has nowhere to go.
    let mut h = H::new(|_| vec![tree("None", &[], None, |_: &mut App, _| panic!(), |_: &mut App, _, _| panic!())]);
    h.focus(h.find(Role::Tree, "None").id);
    h.keys("down up home end left right");
}

#[test]
fn days_are_the_calendars() {
    let d = |y, m, day| Day::new(y, m, day).unwrap();
    assert_eq!(d(2026, 11, 24).weekday(), 1, "a Tuesday");
    assert_eq!(d(1970, 1, 1).weekday(), 3, "a Thursday");
    assert_eq!(d(2000, 1, 1).weekday(), 5);
    assert_eq!(d(2024, 2, 28).plus_days(1), d(2024, 2, 29));
    assert_eq!(d(1900, 2, 28).plus_days(1), d(1900, 3, 1));
    assert_eq!(d(2000, 2, 28).plus_days(1), d(2000, 2, 29));
    assert_eq!(d(2026, 1, 31).plus_months(1), d(2026, 2, 28));
    assert_eq!(d(2026, 1, 15).plus_months(-1), d(2025, 12, 15));
    assert_eq!(d(2026, 3, 31).plus_months(-13), d(2025, 2, 28));
    assert_eq!((Day::new(2026, 2, 29), Day::new(2026, 13, 1), Day::new(2026, 1, 0)), (None, None, None));
    // Every day for 900 years follows the one before, there and back.
    let (mut at, mut week) = (d(1600, 1, 1), d(1600, 1, 1).weekday());
    for _ in 0..328_000 {
        let next = at.plus_days(1);
        assert!(next > at && Day::new(next.year, next.month, next.day).is_some(), "{next:?}");
        assert_eq!(next.plus_days(-1), at);
        week = (week + 1) % 7;
        assert_eq!(next.weekday(), week, "{next:?}");
        at = next;
    }
}

fn month(a: &App) -> Vec<El<App>> {
    let shown = a.month.unwrap_or(Day::new(2026, 11, 1).unwrap());
    vec![calendar(
        shown,
        a.day,
        Day::new(2026, 11, 24),
        &CalendarWords { week_starts: 0, ..CalendarWords::for_language("en") },
        |a: &mut App, d| a.day = Some(d),
        |a: &mut App, d| a.month = Some(d),
    )]
}

#[test]
fn a_calendar_is_one_stop_and_turns_its_months() {
    let mut h = H::new(month);
    assert!(h.find(Role::Button, "November 24, 2026").current);
    // November 2026 begins on a Sunday: the seventh column of weeks from Monday.
    let first = h.find(Role::Button, "November 1, 2026").bounds;
    let grid = h.find(Role::Group, "November 2026").bounds;
    assert_eq!(first.x - grid.x, 6.0 * 34.0);
    h.press(h.find(Role::Button, "November 24, 2026").id);
    assert_eq!(h.app.day, Day::new(2026, 11, 24));
    h.focus(h.find(Role::Group, "November 2026").id);
    h.keys("right down");
    assert_eq!(h.app.day, Day::new(2026, 12, 2), "into the next month");
    h.find(Role::Group, "December 2026");
    h.keys("home");
    assert_eq!(h.app.day, Day::new(2026, 11, 30), "the week's Monday");
    h.keys("end");
    assert_eq!(h.app.day, Day::new(2026, 12, 6));
    h.keys("pagedown");
    assert_eq!(h.app.day, Day::new(2027, 1, 6));
    h.press(h.find(Role::Button, "Previous month").id);
    h.find(Role::Group, "December 2026");
    assert_eq!(h.app.day, Day::new(2027, 1, 6), "looking picks nothing");
    h.audit();
}

#[test]
fn a_range_sliders_ends_do_not_pass() {
    let mut h = H::new(|_| {
        vec![range_slider("Price", ("From", "To"), Binding::new(|a: &App| a.range, |a: &mut App, r| a.range = r), 0.0..=100.0, 10.0)]
    });
    h.app.range = (20.0, 80.0);
    h.press_at(250.0, 590.0);
    let (lo, hi) = (h.find(Role::Slider, "From").id, h.find(Role::Slider, "To").id);
    h.focus(lo);
    h.keys("right");
    assert_eq!(h.app.range, (30.0, 80.0));
    h.keys("end");
    assert_eq!(h.app.range, (80.0, 80.0), "stops at the other");
    h.focus(hi);
    h.keys("home");
    assert_eq!(h.app.range, (80.0, 80.0));
    h.keys("end");
    assert_eq!(h.app.range, (80.0, 100.0));
    assert_eq!(h.find(Role::Slider, "From").range.map(|r| (r.min, r.max, r.now)), Some((0.0, 100.0, 80.0)));
    // A press moves the nearer end; from together, the one that can go.
    let g = h.find(Role::Group, "Price").bounds;
    let x = |v: f32| g.x + 10.0 + (g.width - 20.0) * v / 100.0;
    h.press_at(x(20.0), g.y + 12.0);
    assert_eq!(h.app.range, (20.0, 100.0));
    h.app.range = (50.0, 50.0);
    h.press_at(x(90.0), g.y + 12.0);
    assert_eq!(h.app.range, (50.0, 90.0));
    h.app.range = (50.0, 50.0);
    h.press_at(x(10.0), g.y + 12.0);
    assert_eq!(h.app.range, (10.0, 50.0));
    h.audit();
}
