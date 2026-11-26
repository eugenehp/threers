//! What every component owes whoever uses it, in both appearances: colours
//! that can be told apart, the theme's and no others, and looks that answer
//! the pointer.

use threers_ui::binding::Binding;
use threers_ui::components::{badge, disclosure, switch, switch_row, tabs};
use threers_ui::environment::Token;
use threers_ui::input::{Event, Input};
use threers_ui::menu::{open_menu, MenuAt, MenuItem};
use threers_ui::notifications::{Notice, Notifications, Tone};
use threers_ui::pickers::{empty_state, slider};
use threers_ui::states::Appearance;
use threers_ui::{contrast_ratio, FixedWidthText, NodeId, Rgba, Role, Ui};

type S = ();

const BOTH: [Appearance; 2] = [Appearance::Light, Appearance::Dark];

fn rgb(c: Rgba) -> [f32; 3] {
    [c[0], c[1], c[2]]
}

fn ratio(a: Rgba, b: Rgba) -> f32 {
    contrast_ratio(rgb(a), rgb(b))
}

fn same(a: Option<Rgba>, b: Rgba) -> bool {
    a.is_some_and(|a| (0..4).all(|i| (a[i] - b[i]).abs() < 1.5 / 255.0))
}

fn ui(a: Appearance) -> (Ui, Input<S>) {
    let mut ui = Ui::new();
    ui.set_system_appearance(a);
    (ui, Input::new())
}

fn layout(ui: &mut Ui) {
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
}

fn hover(ui: &mut Ui, input: &mut Input<S>, n: NodeId) {
    layout(ui);
    let r = ui.rect(n);
    let (x, y) = (r.x + r.width / 2.0, r.y + r.height / 2.0);
    input.dispatch(ui, &mut (), Event::PointerMove { x, y });
}

#[test]
fn the_focus_ring_and_the_tones_stand_out_from_every_surface() {
    for a in BOTH {
        let (ui, _) = ui(a);
        for on in [Token::Background, Token::Surface, Token::Raised] {
            let back = ui.token(on);
            let focus = ratio(ui.token(Token::Focus), back);
            assert!(focus >= 3.0, "{a:?} focus on {on:?}: {focus:.2}");
            for t in [Token::Success, Token::Warning, Token::Destructive] {
                let r = ratio(ui.token(t), back);
                assert!(r >= 4.5, "{a:?} {t:?} on {on:?}: {r:.2}");
            }
        }
    }
}

#[test]
fn accent_text_reads_and_is_the_accent_where_that_does() {
    use threers_ui::view::{link, View};
    for a in BOTH {
        // The default, and accents too pale or too dark for a page.
        for accent in [None, Some([1.0, 0.9, 0.2, 1.0]), Some([0.1, 0.0, 0.3, 1.0])] {
            let (mut ui, mut input) = ui(a);
            let mut t = *ui.tokens();
            if let Some(c) = accent {
                t.light.accent = c;
                t.dark.accent = c;
                ui.set_tokens(t);
            }
            let root = ui.root();
            let mut v = View::new(&mut ui, root, "display: flex");
            v.update_with(&mut ui, &mut input, &(), vec![link("Profile", |_: &mut S| {})]);
            layout(&mut ui);
            let l = ui.accessibility_tree().into_iter().find(|n| n.role == Role::Link).unwrap().id;
            for on in [Token::Background, Token::Surface, Token::Raised] {
                let r = ratio(ui.style(l).color, ui.token(on));
                assert!(r >= 4.5, "{a:?} {accent:?} on {on:?}: {r:.2}");
            }
        }
    }
    let (ui, _) = ui(Appearance::Light);
    assert_eq!(ui.token(Token::AccentText), ui.token(Token::Accent));
    assert_eq!(Token::from_css_name("accent-text"), Some(Token::AccentText));
}

#[test]
fn a_switch_is_seen_on_or_off() {
    for a in BOTH {
        for on in [false, true] {
            let (mut ui, mut input) = ui(a);
            let root = ui.root();
            let first = switch(&mut ui, &mut input, &(), root, "Wifi", Binding::local(on));
            let second = switch_row(&mut ui, root, "Wifi", "", on);
            for track in [ui.children(first)[0], ui.children(second)[1]] {
                let knob = ui.children(track)[0];
                let (t, k) = (ui.style(track).background, ui.style(knob).background);
                let r = ratio(t.unwrap(), k.unwrap());
                assert!(r >= 3.0, "{a:?} on={on} knob on track: {r:.2}");
                if !on {
                    let r = ratio(t.unwrap(), ui.token(Token::Surface));
                    assert!(r >= 3.0, "{a:?} track on surface: {r:.2}");
                }
            }
        }
    }
}

#[test]
fn a_menu_is_in_the_themes_colours_highlighted_or_not() {
    for a in BOTH {
        let (mut ui, mut input) = ui(a);
        let mut t = *ui.tokens();
        for set in [&mut t.light, &mut t.dark] {
            set.accent = [0.4, 0.1, 0.5, 1.0];
            set.raised = [0.2, 0.2, 0.1, 1.0];
            set.text = [0.9, 0.9, 0.8, 1.0];
        }
        ui.set_tokens(t);
        let items = vec![
            MenuItem::action("copy", "Copy").shortcut("C"),
            MenuItem::action("delete", "Delete").destructive(true),
        ];
        open_menu(&mut ui, &mut input, &mut (), MenuAt::Point(10.0, 10.0), items, |_, _, _| {});
        layout(&mut ui);
        let menu = ui.accessibility_tree().into_iter().find(|n| n.role == Role::Menu).unwrap().id;
        assert!(same(ui.style(menu).background, t.light.raised), "{a:?} {:?}", ui.style(menu).background);
        let rows: Vec<NodeId> = ui.children(menu).to_vec();
        assert!(same(Some(ui.style(rows[1]).color), ui.token(Token::Destructive)));
        for &r in &rows {
            hover(&mut ui, &mut input, r);
            assert!(same(ui.style(r).background, t.light.accent));
            assert!(same(Some(ui.style(r).color), ui.token(Token::OnAccent)));
        }
        // The shortcut reads on the highlight, and is quiet again off it.
        let key = ui.children(rows[0])[1];
        assert!(same(Some(ui.style(key).color), ui.token(Token::SecondaryText)));
        hover(&mut ui, &mut input, rows[0]);
        assert!(same(Some(ui.style(key).color), ui.token(Token::OnAccent)));
        assert!(ui.style(rows[1]).background.is_none_or(|c| c[3] == 0.0));
        assert!(same(Some(ui.style(rows[1]).color), ui.token(Token::Destructive)));
    }
}

#[test]
fn a_notices_tone_is_the_themes_and_its_cross_a_target() {
    for (tone, token) in [
        (Tone::Info, Token::Accent),
        (Tone::Success, Token::Success),
        (Tone::Error, Token::Destructive),
    ] {
        let (mut ui, mut input) = ui(Appearance::Light);
        let mut all = Notifications::<S>::new(&mut ui);
        let n = Notice::new("Saved", "").tone(tone);
        let card = all.push(&mut ui, &mut input, n, None);
        layout(&mut ui);
        assert!(same(Some(ui.style(card).border_color), ui.token(token)), "{tone:?}");
        let r = ui.accessibility_tree().into_iter().find(|n| n.role == Role::Button).unwrap().bounds;
        assert!(r.width >= 24.0 && r.height >= 24.0, "{r:?}");
    }
}

#[test]
fn what_can_be_pressed_answers_the_pointer() {
    let (mut ui, mut input) = ui(Appearance::Dark);
    let root = ui.root();
    let t = tabs(&mut ui, &mut input, &(), root, &["One", "Two"], Binding::local(0));
    let (head, _) = disclosure(&mut ui, &mut input, &(), root, "More", Binding::local(false));
    for n in [t.tabs[1], head] {
        let before = ui.style(n).clone();
        hover(&mut ui, &mut input, n);
        assert_ne!(*ui.style(n), before);
    }
}

#[test]
fn tabs_of_nothing_are_no_tabs() {
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let t = tabs(&mut ui, &mut input, &(), root, &[], Binding::local(0));
    layout(&mut ui);
    assert!(t.tabs.is_empty() && t.panels.is_empty());
}

#[test]
fn an_empty_states_button_is_the_default_button() {
    let (mut ui, mut input) = ui(Appearance::Light);
    let mut t = *ui.tokens();
    t.light.prominent = [0.1, 0.1, 0.1, 1.0];
    ui.set_tokens(t);
    let root = ui.root();
    empty_state(&mut ui, &mut input, root, "x", "None", "Yet", Some(("Add", Box::new(|_: &mut S, _: &mut Ui| {}))));
    layout(&mut ui);
    let b = ui.accessibility_tree().into_iter().find(|n| n.role == Role::Button).unwrap();
    assert!(same(ui.style(b.id).background, t.light.prominent));
    assert!(b.bounds.height >= 24.0, "{:?}", b.bounds);
}

#[test]
fn a_badges_words_read_on_any_colour() {
    for c in [[1.0, 0.9, 0.2, 1.0], [0.06, 0.73, 0.51, 1.0], [0.1, 0.1, 0.4, 1.0]] {
        let (mut ui, _) = ui(Appearance::Light);
        let root = ui.root();
        let b = badge(&mut ui, root, "3", c);
        let r = ratio(ui.style(b).color, c);
        assert!(r >= 4.5, "{c:?}: {r:.2}");
    }
}

#[test]
fn a_progress_bar_shows_how_far_and_out_of_what() {
    use threers_ui::fill::{Fill, Gradient};
    use threers_ui::indicators::{progress_view as plain, PlainKind, PlainStyle};
    use threers_ui::pickers::progress;
    use threers_ui::view::{progress_view, View};
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let col = ui.add_box(root, threers_ui::style! { display: flex; flex-direction: column; gap: 8px; width: 300px; });
    let a = progress(&mut ui, col, "A", Some(0.5)).bar;
    let b = plain(&mut ui, col, "B", PlainStyle::new(PlainKind::Linear), Some(0.5)).node;
    let host = ui.add_box(col, threers_ui::style! {});
    let mut v = View::new(&mut ui, host, "display: flex; flex-direction: column");
    v.update_with(&mut ui, &mut input, &(), vec![progress_view::<S>(Some(0.5))]);
    layout(&mut ui);
    let c = ui.accessibility_tree().into_iter().find(|n| n.role == Role::ProgressBar && n.id != a && n.id != b).unwrap().id;
    // Built light, then the theme moves under them.
    for contrast in [false, true] {
        ui.set_increase_contrast(contrast);
        for look in [Appearance::Light, Appearance::Dark] {
            ui.set_system_appearance(look);
            layout(&mut ui);
            for (bar, edge_is_border) in [(a, false), (ui.children(b)[0], false), (c, true)] {
                let rail = ui.style(bar).background.unwrap();
                let fill = ui.style(ui.children(bar)[0]).background.unwrap();
                let edge = if edge_is_border {
                    ui.style(bar).border_color
                } else {
                    ui.style(bar).box_shadow[0].color
                };
                let r = ratio(fill, rail);
                assert!(r >= 4.5, "{look:?} fill on rail: {r:.2}");
                for on in [Token::Background, Token::Surface, Token::Raised] {
                    let r = ratio(edge, ui.token(on));
                    assert!(r >= 3.0, "{look:?} outline on {on:?}: {r:.2}");
                }
                // The outline takes none of the fill's room.
                assert_eq!(ui.rect(ui.children(a)[0]).height, 6.0);
                assert_eq!(ui.rect(ui.children(c)[0]).height, 6.0);
            }
        }
    }
    // The accent's gradient fills them still, and a part's own.
    let g = Fill::Gradient(Gradient::linear(90.0, [Token::Accent, Token::Success]));
    ui.set_token_fill(Token::Accent, Some(g.clone()));
    for bar in [a, ui.children(b)[0], c] {
        assert!(!ui.style(ui.children(bar)[0]).background_gradient.is_empty());
    }
    ui.set_token_fill(Token::Accent, None);
    ui.set_fill_in(a, Token::Accent, Some(g));
    assert!(!ui.style(ui.children(a)[0]).background_gradient.is_empty());
    assert!(ui.style(ui.children(c)[0]).background_gradient.is_empty());
}

#[test]
fn a_ring_or_gauge_shows_how_far_and_out_of_what() {
    use threers_ui::charts::{gauge_styled, GaugeKind, GaugeStyle, GaugeTint};
    use threers_ui::indicators::{progress_view as plain, PlainKind, PlainStyle};
    use threers_ui::view::{gauge, View};
    use threers_ui::NodeKind;
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let col = ui.add_box(root, threers_ui::style! { display: flex; flex-direction: column; gap: 8px; width: 300px; });
    let mut rings = vec![plain(&mut ui, col, "A", PlainStyle::new(PlainKind::Circular).size(48.0).stroke(5.0), Some(0.5)).node];
    for kind in [GaugeKind::Circular, GaugeKind::Ring] {
        let mut g = gauge_styled(&mut ui, col, "B", 0.0..=1.0, kind, GaugeStyle { animate: false, ..Default::default() });
        g.set(&mut ui, 0.5);
        rings.push(ui.children(g.node)[0]);
    }
    let host = ui.add_box(col, threers_ui::style! {});
    let mut v = View::new(&mut ui, host, "display: flex");
    v.update_with(&mut ui, &mut input, &(), vec![gauge::<S>(0.5, "C", "50", 96.0)]);
    rings.push(host);
    // Every path of a ring, as painted: the rail's edge, its middle, the arc.
    fn paths(ui: &Ui, n: NodeId, out: &mut Vec<(f32, Rgba)>) {
        if let NodeKind::Path { stroke: Some(w), .. } = ui.kind(n) {
            let mut at = n;
            let mut c = ui.style(at).color;
            while c == [1.0; 4] && ui.parent(at).is_some() {
                at = ui.parent(at).unwrap();
                c = ui.style(at).color;
            }
            out.push((*w, c));
        }
        for &k in ui.children(n) {
            paths(ui, k, out);
        }
    }
    for contrast in [false, true] {
        ui.set_increase_contrast(contrast);
        for look in BOTH {
            ui.set_system_appearance(look);
            layout(&mut ui);
            for (i, &ring) in rings.iter().enumerate() {
                let mut p = vec![];
                paths(&ui, ring, &mut p);
                assert_eq!(p.len(), 3, "ring {i}: {p:?}");
                let [(edge_w, edge), (mid_w, mid), (_, arc)] = [p[0], p[1], p[2]];
                assert!(mid_w < edge_w, "ring {i}");
                let r = ratio(arc, mid);
                assert!(r >= 4.5, "{look:?} ring {i} arc on rail: {r:.2}");
                for on in [Token::Background, Token::Surface, Token::Raised] {
                    let r = ratio(edge, ui.token(on));
                    assert!(r >= 3.0, "{look:?} ring {i} outline on {on:?}: {r:.2}");
                }
            }
            // The stock levels' colours are told from the rail too.
            for tint in [GaugeTint::levels(), GaugeTint::heat()] {
                let (GaugeTint::Bands(stops) | GaugeTint::Gradient(stops)) = tint else { unreachable!() };
                for (_, c) in stops {
                    let r = ratio(c, ui.token(Token::Surface));
                    assert!(r >= 3.0, "{look:?} {c:?}: {r:.2}");
                }
            }
        }
    }
    // The bar and the marker.
    for kind in [GaugeKind::Linear, GaugeKind::Marker] {
        let g = gauge_styled(&mut ui, col, "D", 0.0..=1.0, kind, GaugeStyle::default());
        let track = ui.children(g.node)[1];
        let rail = if kind == GaugeKind::Linear { track } else { ui.children(track)[0] };
        assert!(same(Some(ui.style(rail).box_shadow[0].color), ui.token(Token::Track)));
        let s = ui.style(g.fill);
        let c = if kind == GaugeKind::Linear { s.background.unwrap() } else { s.border_color };
        assert!(ratio(c, ui.style(rail).background.unwrap()) >= 4.5, "{kind:?}");
    }
    // Too thin to outline: a hairline, thinner than what fills it.
    let thin = plain(&mut ui, col, "E", PlainStyle::new(PlainKind::Circular).size(20.0).stroke(2.0), Some(0.5)).node;
    let mut p = vec![];
    paths(&ui, thin, &mut p);
    assert_eq!(p.len(), 2);
    assert!(p[0].0 <= 0.5 * p[1].0);
}

#[test]
fn a_sliders_rail_is_seen_on_every_surface() {
    for a in BOTH {
        for border in [None, Some([0.5, 0.5, 0.5, 0.1])] {
            let (mut ui, mut input) = ui(a);
            let mut t = *ui.tokens();
            if let Some(c) = border {
                t.light.border = c;
                t.dark.border = c;
                ui.set_tokens(t);
            }
            let root = ui.root();
            let n = slider(&mut ui, &mut input, &(), root, "Volume", Binding::local(0.5f32), 0.0..=1.0, 0.1);
            let rail = ui.style(ui.children(n)[0]).background.unwrap();
            assert_eq!(rail[3], 1.0, "{a:?}: a see-through rail is fainter than it measures");
            for on in [Token::Background, Token::Surface, Token::Raised] {
                let r = ratio(rail, ui.token(on));
                assert!(r >= 3.0, "{a:?} rail on {on:?}: {r:.2}");
            }
        }
    }
}

#[test]
fn a_sliders_thumb_has_an_edge_where_the_page_is_light() {
    let (mut ui, mut input) = ui(Appearance::Dark);
    let root = ui.root();
    let n = slider(&mut ui, &mut input, &(), root, "Volume", Binding::local(0.5f32), 0.0..=1.0, 0.1);
    let thumb = ui.children(n)[2];
    for a in [Appearance::Dark, Appearance::Light, Appearance::Dark] {
        ui.set_system_appearance(a);
        layout(&mut ui);
        let s = ui.style(thumb);
        let width = if a == Appearance::Light { 1.0 } else { 0.0 };
        assert_eq!(s.border.left, width, "{a:?}");
        if a == Appearance::Light {
            assert!(same(Some(s.border_color), ui.token(Token::Track)));
            let r = ratio(s.border_color, s.background.unwrap());
            assert!(r >= 3.0, "{r:.2}");
        }
        // The edge takes no more room.
        let r = ui.rect(thumb);
        assert_eq!((r.width, r.height), (20.0, 20.0), "{a:?}");
    }
}

#[test]
fn a_sliders_value_is_said_to_its_steps_precision() {
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let n = slider(&mut ui, &mut input, &(), root, "Opacity", Binding::local(7.0f32 * 0.1), 0.0..=1.0, 0.1);
    assert_eq!(ui.semantics(n).unwrap().value.as_deref(), Some("0.7"));
    let (mut ui, mut input) = self::ui(Appearance::Light);
    let root = ui.root();
    let n = slider(&mut ui, &mut input, &(), root, "Volume", Binding::local(50.0f32), 0.0..=100.0, 5.0);
    assert_eq!(ui.semantics(n).unwrap().value.as_deref(), Some("50"));
}

#[test]
fn a_choice_no_longer_made_is_drawn_so() {
    use threers_ui::view::{segmented, text, View};
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let mut v = View::new(&mut ui, root, "display: flex");
    for k in [0, 1] {
        v.update_with(&mut ui, &mut input, &(), vec![segmented(["One", "Two"], Some(k), |_: &mut S, _| {})]);
        layout(&mut ui);
    }
    let group = ui.children(v.root())[0];
    let items = ui.children(group).to_vec();
    assert!(ui.style(items[0]).background.is_none_or(|c| c[3] == 0.0), "still drawn chosen");
    assert!(ui.style(items[1]).background.is_some_and(|c| c[3] > 0.0), "the choice is drawn");
    // And hovers: what it has now, not what it had.
    for css in [Some("opacity: 0.5"), None] {
        let el = text::<S>("x").key("x");
        v.update_with(&mut ui, &mut input, &(), vec![match css {
            Some(c) => el.hover(c),
            None => el,
        }]);
    }
    let x = ui.children(v.root())[0];
    hover(&mut ui, &mut input, x);
    assert_eq!(ui.style(x).opacity, 1.0, "its old hover");
}

#[test]
fn a_switchs_knob_stays_where_it_is_as_things_are_hovered() {
    let (mut ui, mut input) = ui(Appearance::Light);
    let root = ui.root();
    let row = switch(&mut ui, &mut input, &(), root, "Wifi", Binding::local(true));
    let second = switch_row(&mut ui, root, "Sounds", "", true);
    let other = ui.add_box(root, threers_ui::style! { width: 50px; height: 50px; });
    ui.hover_style(other, "opacity: 0.5");
    layout(&mut ui);
    let knob = ui.children(ui.children(row)[0])[0];
    let at = ui.rect(knob).x;
    assert!(at > 10.0, "on: {at}");
    let knob2 = ui.children(ui.children(second)[1])[0];
    let at2 = ui.rect(knob2).x;
    hover(&mut ui, &mut input, other);
    layout(&mut ui);
    assert_eq!((ui.rect(knob).x, ui.rect(knob2).x), (at, at2));
    ui.set_system_appearance(Appearance::Dark);
    layout(&mut ui);
    assert_eq!((ui.rect(knob).x, ui.rect(knob2).x), (at, at2), "nor as the theme changes");
}
