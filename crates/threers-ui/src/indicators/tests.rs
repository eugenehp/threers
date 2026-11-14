use super::*;
use crate::FixedWidthText;

#[test]
fn every_kind_is_an_effect_with_its_arguments() {
    for shape in [Shape::Bar, Shape::Ring, Shape::Battery] {
        for depth in [Depth::Flat, Depth::Solid] {
            for animated in [false, true] {
                let s = shader(shape, depth, Fill::Neon, animated);
                assert_eq!(s.error(), None, "{shape:?} {depth:?}");
                assert_eq!(s.function(), "progress_indicator");
                assert_eq!(s.animated(), animated);
                let e = Effect::new(&s);
                for name in ["value", "c0", "c1", "c2", "look", "reach", "motion"] {
                    assert!(e.clone().set(name, 0.0), "{name}");
                }
            }
        }
    }
}

#[test]
fn a_still_indicator_asks_for_no_frames_and_a_busy_one_does() {
    let mut ui = Ui::new();
    let root = ui.root();
    let bar = indicator(
        &mut ui,
        root,
        "Copying",
        IndicatorStyle::new(Shape::Bar),
        Some(0.5),
    );
    ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
    ui.paint(&mut crate::scene::Scene::default().recorder());
    assert!(!ui.needs_frame());
    assert_eq!(ui.rect(bar.node).width, 300.0, "a bar spans its parent");
    assert_eq!(ui.rect(bar.node).height, 10.0);
    bar.set(&mut ui, None);
    ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
    assert!(ui.needs_frame());
    assert_eq!(ui.semantics(bar.node).unwrap().value, None);
    assert!(ui.effect(bar.effect_node()).unwrap().shader.animated());
    // Back to a value: still again.
    bar.set(&mut ui, Some(1.0));
    assert!(!ui.effect(bar.effect_node()).unwrap().shader.animated());
    assert_eq!(
        ui.semantics(bar.node).unwrap().value.as_deref(),
        Some("100%")
    );
}

#[test]
fn the_glow_reaches_past_the_box() {
    let st = IndicatorStyle::new(Shape::Bar).size(12.0).glow(1.0);
    assert!(st.overflow() >= 24.0, "{}", st.overflow());
    assert_eq!(IndicatorStyle::new(Shape::Bar).glow(0.0).overflow(), 1.0);
    // And room to hop.
    let hop = IndicatorStyle::new(Shape::Ring).motion(Motion::Bounce);
    assert!(hop.overflow() > IndicatorStyle::new(Shape::Ring).overflow() + hop.size * 0.1);
}

#[test]
fn a_ring_falls_back_to_an_arc() {
    let mut ui = Ui::new();
    let root = ui.root();
    let ring = indicator(
        &mut ui,
        root,
        "Sync",
        IndicatorStyle::new(Shape::Ring).depth(Depth::Solid),
        Some(0.25),
    );
    let crate::NodeKind::Path { d, .. } = ui.kind(ring.fill).clone() else {
        panic!("a path")
    };
    assert_eq!(d.len(), 1);
    assert!(d[0].contains(" 0 0 1 "), "a quarter: the small arc, {d:?}");
    ring.set(&mut ui, Some(0.0));
    let crate::NodeKind::Path { d, .. } = ui.kind(ring.fill).clone() else {
        panic!("a path")
    };
    assert!(d.is_empty());
}

#[test]
fn palettes_mix_through_the_middle() {
    let p = Palette([
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ]);
    assert_eq!(p.at(0.5), [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(p.at(0.25), [0.5, 0.0, 0.0, 1.0]);
    assert_eq!(p.at(2.0), [1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn plain_views_follow_the_theme_and_turn_while_busy() {
    for kind in [PlainKind::Linear, PlainKind::Circular, PlainKind::Spokes] {
        let mut ui = Ui::new();
        let root = ui.root();
        let v = progress_view(&mut ui, root, "Load", PlainStyle::new(kind), Some(0.75));
        ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
        ui.paint(&mut crate::scene::Scene::default().recorder());
        assert!(!ui.needs_frame(), "{kind:?}: still with a value");
        assert_eq!(ui.semantics(v.node).unwrap().value.as_deref(), Some("75%"));
        assert!(ui.effect(v.node).is_none(), "no shader");
        v.set(&mut ui, None);
        ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
        assert!(ui.needs_frame(), "{kind:?}: busy moves");
        v.set(&mut ui, Some(0.25));
        ui.layout(300.0, 100.0, &mut FixedWidthText(7.0));
        ui.paint(&mut crate::scene::Scene::default().recorder());
        assert!(!ui.needs_frame(), "{kind:?}: and stops again");
    }
}

#[test]
fn spokes_light_up_to_the_value() {
    let mut ui = Ui::new();
    let root = ui.root();
    let v = progress_view(
        &mut ui,
        root,
        "Load",
        PlainStyle::new(PlainKind::Spokes),
        Some(0.5),
    );
    let lit = v
        .parts
        .iter()
        .filter(|&&n| ui.style(n).opacity == 1.0)
        .count();
    assert_eq!(lit, SPOKES / 2);
}

#[test]
fn a_moving_indicator_asks_for_frames_unless_motion_is_reduced() {
    let mut ui = Ui::new();
    let root = ui.root();
    let st = IndicatorStyle::new(Shape::Ring)
        .depth(Depth::Solid)
        .motion(Motion::Spin);
    let ring = indicator(&mut ui, root, "Sync", st, Some(0.5));
    ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
    assert!(ui.effect(ring.effect_node()).unwrap().shader.animated());
    assert!(ui.needs_frame());
    ui.set_reduce_motion(true);
    ring.set(&mut ui, Some(0.5));
    assert!(
        !ui.effect(ring.effect_node()).unwrap().shader.animated(),
        "still, as asked"
    );
}

#[test]
fn palettes_from_css_stops_and_hues() {
    assert_eq!(
        Palette::css(&["#ff0000", "#0000ff"]).unwrap().0[1],
        [0.5, 0.0, 0.5, 1.0]
    );
    assert_eq!(
        Palette::css(&["red"]).unwrap(),
        Palette::solid([1.0, 0.0, 0.0, 1.0])
    );
    assert!(Palette::css(&["not a colour"]).is_none());
    let p = Palette::stops(&[(0.0, [0.0, 0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 1.0, 1.0])]).unwrap();
    assert!((p.0[1][0] - 0.5).abs() < 0.01, "{p:?}");
    let h = Palette::hues(0.0, 240.0, 1.0, 0.5);
    assert_eq!(h.0[0], [1.0, 0.0, 0.0, 1.0]);
    assert_eq!(h.0[1], [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(h.0[2], [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(Palette::named("ember"), Some(Palette::EMBER));
    assert_eq!(Palette::EMBER.reversed().0[0], Palette::EMBER.0[2]);
    assert_eq!(
        Palette::solid([0.0, 0.0, 0.0, 1.0]).lighter(1.0).0[0],
        [1.0, 1.0, 1.0, 1.0]
    );
}

#[test]
fn every_preset_builds() {
    let mut ui = Ui::new();
    let root = ui.root();
    for p in Preset::ALL {
        let i = indicator(&mut ui, root, p.name(), p.into(), Some(0.5));
        assert!(ui.effect(i.effect_node()).is_some(), "{}", p.name());
    }
}

#[test]
fn every_effect_compiles_for_bar_and_ring() {
    for fill in [Fill::Fire, Fill::Water, Fill::Plasma, Fill::Frost] {
        for shape in [Shape::Bar, Shape::Ring] {
            let s = shader(shape, Depth::Flat, fill, true);
            assert_eq!(s.error(), None, "{fill:?} {shape:?}");
        }
    }
    let fire = IndicatorStyle::new(Shape::Bar).size(12.0).fire();
    assert!(
        fire.overflow() >= 24.0,
        "room for the flames: {}",
        fire.overflow()
    );
}

#[test]
fn electric_ones_compile_and_always_move() {
    for shape in [Shape::Bar, Shape::Ring, Shape::Battery] {
        let s = shader(shape, Depth::Flat, Fill::Electric, true);
        assert_eq!(s.error(), None);
    }
    let mut ui = Ui::new();
    let root = ui.root();
    let bar = indicator(
        &mut ui,
        root,
        "Charge",
        IndicatorStyle::new(Shape::Bar).electric(),
        Some(0.5),
    );
    assert!(ui.effect(bar.effect_node()).unwrap().shader.animated());
}

#[test]
fn phases_pick_one_colour() {
    let st = IndicatorStyle::new(Shape::Battery)
        .palette(Palette::BATTERY)
        .phases(0.2, 0.5);
    assert_eq!(st.colours_at(Some(0.1)), [Palette::BATTERY.0[0]; 3]);
    assert_eq!(st.colours_at(Some(0.3)), [Palette::BATTERY.0[1]; 3]);
    assert_eq!(st.colours_at(Some(0.9)), [Palette::BATTERY.0[2]; 3]);
    assert_eq!(
        st.colours_at(None),
        Palette::BATTERY.0,
        "busy: the whole palette"
    );
}
