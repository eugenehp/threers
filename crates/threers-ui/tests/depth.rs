//! 3D effects: projection, hidden backs, parallax, tilt, flipping, the
//! carousel, and what the painter is given.

use threers_ui::depth::{flip_card, Transform3d};
use threers_ui::input::{Event, Input, PointerButton};
use threers_ui::scene::Item;
use threers_ui::{binding, style, FixedWidthText, Role, Ui};

fn layout(ui: &mut Ui) {
    ui.layout(600.0, 400.0, &mut FixedWidthText(7.0));
}

#[test]
fn a_turn_narrows_a_lift_grows_and_the_middle_stays() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 200px; height: 100px; });
    layout(&mut ui);
    let at = |m: [f32; 6], x: f32, y: f32| (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]);
    ui.set_transform3d(card, Transform3d::default());
    let flat = ui.affine_of(card).unwrap();
    assert_eq!(at(flat, 30.0, 40.0), (30.0, 40.0), "untransformed");
    ui.set_transform3d(
        card,
        Transform3d {
            rotate_y: 1.0,
            ..Transform3d::default()
        },
    );
    let m = ui.affine_of(card).unwrap();
    assert!((m[0] - 1f32.cos()).abs() < 0.02, "{m:?}");
    let (cx, cy) = at(m, 100.0, 50.0);
    assert!(
        (cx - 100.0).abs() < 0.01 && (cy - 50.0).abs() < 0.01,
        "the middle stays put"
    );
    ui.set_transform3d(
        card,
        Transform3d {
            depth: 100.0,
            perspective: 400.0,
            ..Transform3d::default()
        },
    );
    let m = ui.affine_of(card).unwrap();
    assert!(
        (m[0] - 400.0 / 300.0).abs() < 1e-4,
        "nearer is larger: {m:?}"
    );
}

#[test]
fn the_painter_is_given_the_transform_and_hidden_backs_are_skipped() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(
        root,
        style! { width: 100px; height: 60px; background: #ffffff; },
    );
    let face = ui.add_box(
        card,
        style! { width: 50px; height: 20px; background: #ff0000; },
    );
    ui.set_transform3d(
        card,
        Transform3d {
            rotate_y: 0.5,
            shadow: true,
            glare: true,
            ..Transform3d::default()
        },
    );
    ui.set_transform3d(
        face,
        Transform3d {
            hide_backface: true,
            ..Transform3d::default()
        },
    );
    layout(&mut ui);
    let items = ui.scene().items.clone();
    let pushes = items
        .iter()
        .filter(|i| matches!(i, Item::PushProjective { .. }))
        .count();
    let pops = items
        .iter()
        .filter(|i| matches!(i, Item::PopTransform))
        .count();
    assert_eq!((pushes, pops), (2, 2), "the card and its face");
    assert!(
        items.iter().any(|i| matches!(i, Item::Shadow { .. })),
        "lifted: a shadow"
    );
    assert!(
        items.iter().any(|i| matches!(i, Item::Gradient { .. })),
        "turned: a glare"
    );
    let red = |items: &[Item]| {
        items
            .iter()
            .any(|i| matches!(i, Item::Rect { color, .. } if *color == [1.0, 0.0, 0.0, 1.0]))
    };
    assert!(red(&items));
    // Turned round: the face's back is to the viewer, and it goes.
    ui.set_transform3d(
        card,
        Transform3d {
            rotate_y: 3.0,
            ..Transform3d::default()
        },
    );
    assert!(!red(&ui.scene().items));
}

#[test]
fn parallax_layers_move_by_their_depth() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 100px; height: 60px; });
    let near = ui.add_box(card, style! { width: 10px; height: 10px; });
    ui.set_parallax(near, 20.0);
    layout(&mut ui);
    ui.set_transform3d(
        card,
        Transform3d {
            rotate_y: 0.3,
            ..Transform3d::default()
        },
    );
    let pushes: Vec<[f32; 6]> = ui
        .scene()
        .items
        .iter()
        .filter_map(|i| {
            if let Item::PushProjective { affine, .. } = i {
                Some(*affine)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(pushes.len(), 2);
    assert!(
        (pushes[1][4] - 0.3f32.sin() * 20.0).abs() < 1e-4,
        "{pushes:?}"
    );
}

#[test]
fn a_card_tilts_toward_the_pointer_and_settles_back() {
    let mut ui = Ui::new();
    let mut input: Input<()> = Input::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 200px; height: 100px; });
    let away = ui.add_box(root, style! { width: 200px; height: 100px; });
    layout(&mut ui);
    input.tilt_on_hover(&mut ui, card, 10.0);
    input.dispatch(&mut ui, &mut (), Event::PointerMove { x: 190.0, y: 50.0 });
    ui.step(0.3);
    let t = ui.transform3d(card).unwrap();
    assert!(
        t.rotate_y < -0.1 && t.depth > 10.0,
        "the right side comes toward it: {t:?}"
    );
    let a = ui.rect(away);
    input.dispatch(
        &mut ui,
        &mut (),
        Event::PointerMove {
            x: a.x + 10.0,
            y: a.y + 10.0,
        },
    );
    for _ in 0..120 {
        ui.step(1.0 / 60.0);
    }
    let t = ui.transform3d(card).unwrap();
    assert!(
        t.rotate_y.abs() < 1e-4 && t.depth.abs() < 1e-3,
        "settled: {t:?}"
    );
}

#[test]
fn a_flip_card_turns_over_and_reads_the_face_showing() {
    #[derive(Default)]
    struct S {
        flipped: bool,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let c = flip_card(
        &mut ui,
        &mut input,
        &s,
        root,
        "Word card",
        binding!(|s: S| s.flipped),
        |ui, f| {
            ui.add_text(f, "Hund");
        },
        |ui, b| {
            ui.add_text(b, "Dog");
        },
    );
    ui.update_style(c.card, |st| {
        st.width = threers_ui::Length::Px(200.0);
        st.height = threers_ui::Length::Px(120.0);
    });
    layout(&mut ui);
    let texts = |ui: &Ui| {
        ui.accessibility_tree()
            .into_iter()
            .filter(|a| a.role == Role::StaticText)
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    assert_eq!(texts(&ui), ["Hund"]);
    let r = ui.rect(c.card);
    let b = PointerButton::Primary;
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerDown {
            x: r.x + 10.0,
            y: r.y + 10.0,
            button: b,
        },
    );
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerUp {
            x: r.x + 10.0,
            y: r.y + 10.0,
            button: b,
        },
    );
    assert!(s.flipped);
    for _ in 0..120 {
        ui.step(1.0 / 60.0);
    }
    assert!((ui.transform3d(c.card).unwrap().rotate_y - std::f32::consts::PI).abs() < 1e-4);
    assert_eq!(texts(&ui), ["Dog"]);
    let card = ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.id == c.card)
        .unwrap();
    assert_eq!(card.value.as_deref(), Some("Back"));
    // The front is turned away: not drawn.
    assert!(!ui
        .scene()
        .items
        .iter()
        .any(|i| matches!(i, Item::Text { text, .. } if text == "Hund")));
    assert!(ui
        .scene()
        .items
        .iter()
        .any(|i| matches!(i, Item::Text { text, .. } if text == "Dog")));
}

#[test]
fn a_carousel_turns_its_items_away_from_the_middle() {
    let mut ui = Ui::new();
    let root = ui.root();
    let strip = ui.add_box(
        root,
        style! { display: flex; overflow-x: auto; width: 300px; },
    );
    let items: Vec<_> = (0..7)
        .map(|_| {
            ui.add_box(
                strip,
                style! { width: 100px; height: 80px; flex-shrink: 0; },
            )
        })
        .collect();
    ui.carousel_3d(strip);
    layout(&mut ui);
    let t = |ui: &Ui, i: usize| ui.transform3d(items[i]).unwrap();
    assert!(
        t(&ui, 1).rotate_y.abs() < 1e-4,
        "the middle one faces front"
    );
    assert!(
        t(&ui, 0).rotate_y > 0.5 && t(&ui, 2).rotate_y < -0.5,
        "the others turn away"
    );
    ui.scroll_to_x(strip, 100.0);
    layout(&mut ui);
    assert!(
        t(&ui, 2).rotate_y.abs() < 1e-4,
        "scrolled: the next is in the middle"
    );
}

#[test]
fn the_homography_is_exact_at_the_corners_and_shows_the_keystone() {
    use threers_ui::depth::apply_homography;
    let r = threers_ui::Rect {
        x: 100.0,
        y: 50.0,
        width: 200.0,
        height: 100.0,
    };
    let t = Transform3d {
        rotate_y: 0.8,
        perspective: 400.0,
        ..Transform3d::default()
    };
    let h = t.homography(r);
    for (x, y) in [
        (100.0, 50.0),
        (300.0, 50.0),
        (300.0, 150.0),
        (100.0, 150.0),
        (200.0, 100.0),
    ] {
        let (a, b) = apply_homography(h, x, y);
        let (p, q) = t.project(r, x, y);
        assert!(
            (a - p).abs() < 0.05 && (b - q).abs() < 0.05,
            "({x}, {y}): {a},{b} vs {p},{q}"
        );
    }
    // Turned right side away: the left edge, nearer, is taller.
    let left = apply_homography(h, 100.0, 150.0).1 - apply_homography(h, 100.0, 50.0).1;
    let right = apply_homography(h, 300.0, 150.0).1 - apply_homography(h, 300.0, 50.0).1;
    assert!(left > right + 10.0, "{left} {right}");
}

#[test]
fn a_box_turns_about_its_origin_and_in_its_plane() {
    let r = threers_ui::Rect {
        x: 0.0,
        y: 0.0,
        width: 100.0,
        height: 50.0,
    };
    // About its left edge, the left edge stays where it is.
    let door = Transform3d {
        rotate_y: 1.0,
        origin: (0.0, 0.5),
        ..Transform3d::default()
    };
    let (x, y) = door.project(r, 0.0, 25.0);
    assert!(x.abs() < 1e-3 && (y - 25.0).abs() < 1e-3);
    assert!(
        door.project(r, 100.0, 25.0).0 < 100.0,
        "the right edge swings in"
    );
    // Half a turn in its plane: a corner goes to the opposite one.
    let spun = Transform3d {
        rotate_z: std::f32::consts::PI,
        ..Transform3d::default()
    };
    let (x, y) = spun.project(r, 0.0, 0.0);
    assert!((x - 100.0).abs() < 1e-3 && (y - 50.0).abs() < 1e-3);
}

#[test]
fn shading_darkens_a_face_turned_from_the_light_and_depth_blurs() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(
        root,
        style! { width: 100px; height: 60px; background: #ffffff; },
    );
    layout(&mut ui);
    let dark = |ui: &Ui| {
        ui.scene()
            .items
            .iter()
            .filter_map(|i| {
                if let Item::Rect { color, .. } = i {
                    (color[0] == 0.0 && color[3] > 0.0).then_some(color[3])
                } else {
                    None
                }
            })
            .fold(0.0f32, f32::max)
    };
    ui.set_transform3d(
        card,
        Transform3d {
            shade: true,
            ..Transform3d::default()
        },
    );
    assert_eq!(dark(&ui), 0.0, "flat on: not darkened");
    ui.set_transform3d(
        card,
        Transform3d {
            shade: true,
            rotate_y: 1.2,
            ..Transform3d::default()
        },
    );
    let away = dark(&ui);
    assert!(away > 0.2, "turned away from the light: {away}");
    ui.set_depth_of_field(root, Some((0.0, 4.0)));
    ui.set_transform3d(
        card,
        Transform3d {
            depth: -200.0,
            ..Transform3d::default()
        },
    );
    assert!(ui
        .scene()
        .items
        .iter()
        .any(|i| matches!(i, Item::PushBlur(r) if (*r - 8.0).abs() < 1e-3)));
    ui.set_transform3d(card, Transform3d::default());
    assert!(
        !ui.scene()
            .items
            .iter()
            .any(|i| matches!(i, Item::PushBlur(_))),
        "in focus"
    );
}

#[test]
fn a_turned_box_is_hit_where_it_is_drawn() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 200px; height: 100px; });
    let right_half = ui.add_box(
        card,
        style! { position: absolute; left: 100px; top: 0px; width: 100px; height: 100px; },
    );
    layout(&mut ui);
    assert_eq!(ui.hit_test(190.0, 50.0), Some(right_half));
    // Turned about its left edge, half shown: its right part is drawn left
    // of where it was, and what was there is not it any more.
    ui.set_transform3d(
        card,
        Transform3d {
            rotate_y: 1.0,
            origin: (0.0, 0.5),
            perspective: 1e6,
            ..Transform3d::default()
        },
    );
    let drawn_right = Transform3d {
        rotate_y: 1.0,
        origin: (0.0, 0.5),
        perspective: 1e6,
        ..Transform3d::default()
    }
    .project(ui.rect(card), 190.0, 50.0);
    assert_eq!(ui.hit_test(drawn_right.0, 50.0), Some(right_half));
    assert_eq!(ui.hit_test(190.0, 50.0), Some(root), "nothing there now");
}

#[test]
fn a_spring_follows_a_moving_target() {
    use threers_ui::anim::Spring;
    use threers_ui::animation::Property;
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_box(root, style! { width: 100px; height: 60px; });
    for k in 0..30 {
        ui.spring_to(
            card,
            Property::RotateY,
            (k as f32 * 0.2).sin() * 0.4,
            Spring::snappy(),
        );
        ui.step(1.0 / 60.0);
    }
    let v = ui.spring_velocity(card, Property::RotateY);
    assert!(v.abs() > 0.0, "still moving with the target");
    ui.spring_to(card, Property::RotateY, 0.0, Spring::snappy());
    for _ in 0..120 {
        ui.step(1.0 / 60.0);
    }
    assert_eq!(ui.transform3d(card).unwrap().rotate_y, 0.0);
}

#[test]
fn a_card_deck_swipes_its_top_card_off_and_brings_the_next_up() {
    use threers_ui::depth::{card_deck, SwipeDirection};
    #[derive(Default)]
    struct S {
        cards: Vec<String>,
        swiped: Vec<(String, SwipeDirection)>,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S {
        cards: ["Ada", "Alan", "Grace", "Edsger"]
            .map(String::from)
            .to_vec(),
        ..Default::default()
    };
    let root = ui.root();
    let d = card_deck(
        &mut ui,
        &mut input,
        root,
        "People",
        binding!(|s: S| s.cards),
        |ui, card, name| {
            ui.add_text(card, name);
        },
        |s: &mut S, name, dir| s.swiped.push((name.to_owned(), dir)),
    );
    ui.update_style(d.deck, |st| {
        st.width = threers_ui::Length::Px(300.0);
        st.height = threers_ui::Length::Px(200.0);
    });
    let mut t = 0.0;
    let mut settle = |input: &mut Input<S>, ui: &mut Ui, s: &mut S| {
        for _ in 0..40 {
            t += 1.0 / 60.0;
            input.tick(ui, s, t);
            ui.step(1.0 / 60.0);
            layout(ui);
        }
    };
    settle(&mut input, &mut ui, &mut s);
    let items = |ui: &Ui| {
        ui.accessibility_tree()
            .into_iter()
            .filter(|a| a.role == Role::ListItem)
            .map(|a| (a.name, a.position))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        items(&ui),
        [("Ada".to_owned(), Some((1, 3)))],
        "the top card is read; those under it wait"
    );
    // A short drag springs back.
    let r = ui.rect(d.deck);
    let b = PointerButton::Primary;
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerDown {
            x: r.x + 150.0,
            y: r.y + 100.0,
            button: b,
        },
    );
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerMove {
            x: r.x + 200.0,
            y: r.y + 100.0,
        },
    );
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerUp {
            x: r.x + 200.0,
            y: r.y + 100.0,
            button: b,
        },
    );
    settle(&mut input, &mut ui, &mut s);
    assert!(s.swiped.is_empty());
    // A long one sends it off to that side.
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerDown {
            x: r.x + 150.0,
            y: r.y + 100.0,
            button: b,
        },
    );
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerMove {
            x: r.x + 20.0,
            y: r.y + 110.0,
        },
    );
    input.dispatch(
        &mut ui,
        &mut s,
        Event::PointerUp {
            x: r.x + 20.0,
            y: r.y + 110.0,
            button: b,
        },
    );
    settle(&mut input, &mut ui, &mut s);
    assert_eq!(s.swiped, [("Ada".to_owned(), SwipeDirection::Left)]);
    assert_eq!(s.cards, ["Alan", "Grace", "Edsger"]);
    assert_eq!(items(&ui)[0].0, "Alan");
    // The keys, and assistive technology's action.
    input.set_focus(&mut ui, &mut s, Some(d.deck));
    input.simulate_keystrokes(&mut ui, &mut s, "right").unwrap();
    settle(&mut input, &mut ui, &mut s);
    let top = ui
        .accessibility_tree()
        .into_iter()
        .find(|a| a.role == Role::ListItem)
        .unwrap();
    assert_eq!(top.actions, ["Swipe left", "Swipe right"]);
    assert!(input.accessibility_action(&mut ui, &mut s, top.id, "Swipe left"));
    settle(&mut input, &mut ui, &mut s);
    assert_eq!(
        s.swiped.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
        ["Ada", "Alan", "Grace"]
    );
    assert_eq!(s.cards, ["Edsger"]);
}

#[test]
fn a_nav_stack_turns_screens_like_a_cube() {
    use threers_ui::nav_stack::{nav_stack, NavTransition};
    #[derive(Default)]
    struct S {
        path: Vec<String>,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let host = ui.add_box(
        root,
        style! { display: flex; flex-direction: column; height: 400px; },
    );
    let stack = nav_stack(
        &mut ui,
        &mut input,
        &s,
        host,
        "Home",
        binding!(|s: S| s.path),
        |ui, _, _, p, sc| {
            ui.add_text(sc, p.to_owned());
        },
        |_, _, _, _| {},
    );
    ui.set_environment(stack.container, NavTransition::Cube);
    layout(&mut ui);
    s.path.push("a".into());
    input.tick(&mut ui, &mut s, 0.0);
    layout(&mut ui);
    let screens: Vec<_> = ui.children(stack.body).to_vec();
    let (home, a) = (screens[0], screens[1]);
    assert!(
        ui.transform3d(a).unwrap().rotate_y > 1.5,
        "the new face starts turned in from the right"
    );
    for _ in 0..90 {
        ui.step(1.0 / 60.0);
    }
    assert!(ui.transform3d(a).unwrap().rotate_y.abs() < 1e-3);
    assert!(
        (ui.transform3d(home).unwrap().rotate_y + std::f32::consts::FRAC_PI_2).abs() < 1e-3,
        "the old one turned away"
    );
}

#[test]
fn a_cube_turns_to_its_face_the_nearest_way() {
    use threers_ui::depth::cube;
    #[derive(Default)]
    struct S {
        face: usize,
    }
    let mut ui = Ui::new();
    let mut input: Input<S> = Input::new();
    let mut s = S::default();
    let root = ui.root();
    let c = cube(
        &mut ui,
        &mut input,
        &s,
        root,
        "Seasons",
        4,
        binding!(|s: S| s.face),
        |ui, f, k| {
            ui.add_text(f, ["Spring", "Summer", "Autumn", "Winter"][k]);
        },
    );
    ui.update_style(c.cube, |st| {
        st.width = threers_ui::Length::Px(200.0);
        st.height = threers_ui::Length::Px(120.0);
    });
    let mut t = 0.0;
    let mut settle = |input: &mut Input<S>, ui: &mut Ui, s: &mut S| {
        for _ in 0..120 {
            t += 1.0 / 60.0;
            input.tick(ui, s, t);
            ui.step(1.0 / 60.0);
            layout(ui);
        }
    };
    settle(&mut input, &mut ui, &mut s);
    let read = |ui: &Ui| {
        ui.accessibility_tree()
            .into_iter()
            .filter(|a| a.role == Role::StaticText)
            .map(|a| a.name)
            .collect::<Vec<_>>()
    };
    assert_eq!(read(&ui), ["Spring"], "only the face showing");
    let front = |ui: &Ui, k: usize| ui.transform3d(c.faces[k]).unwrap();
    assert!(front(&ui, 0).rotate_y.abs() < 1e-3 && front(&ui, 0).translate_x.abs() < 1e-2);
    // Left from the first face: round the short way, to the last.
    input.set_focus(&mut ui, &mut s, Some(c.cube));
    input.simulate_keystrokes(&mut ui, &mut s, "left").unwrap();
    assert_eq!(s.face, 3);
    settle(&mut input, &mut ui, &mut s);
    let w = front(&ui, 3).rotate_y.rem_euclid(std::f32::consts::TAU);
    assert!(
        w < 1e-3 || (std::f32::consts::TAU - w) < 1e-3,
        "winter faces front: {w}"
    );
    assert_eq!(read(&ui), ["Winter"]);
    // The face in front is painted last.
    assert_eq!(
        *ui.children(c.cube)
            .iter()
            .rfind(|&&n| c.faces.contains(&n))
            .unwrap(),
        c.faces[3]
    );
}

#[test]
fn a_cylinder_rolls_rows_and_a_reflection_is_painted_below() {
    let mut ui = Ui::new();
    let root = ui.root();
    let list = ui.add_box(
        root,
        style! { height: 200px; overflow: auto; display: flex; flex-direction: column; },
    );
    let rows: Vec<_> = (0..20)
        .map(|_| ui.add_box(list, style! { height: 40px; flex-shrink: 0; }))
        .collect();
    ui.cylinder_3d(list);
    layout(&mut ui);
    // The middle row faces front; those above and below roll away.
    let t = |ui: &Ui, i: usize| ui.transform3d(rows[i]).unwrap();
    assert!(t(&ui, 2).rotate_x.abs() < 0.01, "{:?}", t(&ui, 2));
    assert!(t(&ui, 0).rotate_x > 0.5 && t(&ui, 4).rotate_x < -0.5);
    assert!(t(&ui, 0).depth < -10.0);
    // A reflection: the box painted again, mirrored under it, faded.
    let card = ui.add_box(
        root,
        style! { width: 100px; height: 50px; background: #ff0000; },
    );
    ui.set_reflection(card, Some((0.4, 0.5)));
    layout(&mut ui);
    let items = ui.scene().items;
    let reds = items
        .iter()
        .filter(|i| matches!(i, Item::Rect { color, .. } if color[0] == 1.0 && color[1] == 0.0))
        .count();
    assert_eq!(reds, 2, "it and its reflection");
    assert!(
        items
            .iter()
            .any(|i| matches!(i, Item::PushTransform(m) if m[3] == -1.0)),
        "mirrored"
    );
}
