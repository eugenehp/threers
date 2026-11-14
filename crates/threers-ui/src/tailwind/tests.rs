use super::theme::rgb;
use super::*;
use crate::input::{Event, Input};
use crate::states::Theme;
use crate::{
    AlignItems, Edges, FixedWidthText, FlexDirection, FontWeight, JustifyContent, Length, Rgba,
};

fn close(a: Rgba, b: Rgba) -> bool {
    (0..4).all(|i| (a[i] - b[i]).abs() < 1e-5)
}

fn hover(ui: &mut Ui, x: f32, y: f32) {
    Input::<()>::new().dispatch(ui, &mut (), Event::PointerMove { x, y });
}

#[test]
fn layout_and_spacing() {
    let s = tw("flex flex-col items-center justify-between gap-x-3 p-4 px-2 -mt-1 w-1/3 h-full min-w-0 max-w-md");
    assert_eq!(s.flex_direction, FlexDirection::Column);
    assert_eq!(s.align_items, AlignItems::Center);
    assert_eq!(s.justify_content, JustifyContent::SpaceBetween);
    assert_eq!(s.column_gap, Some(12.0));
    assert_eq!(
        s.padding,
        Edges::new(16.0, 8.0, 16.0, 8.0),
        "the longhand wins over the shorthand"
    );
    assert_eq!(s.margin.top, -4.0);
    assert!(matches!(s.width, Length::Percent(p) if (p - 100.0 / 3.0).abs() < 1e-4));
    assert_eq!(
        (s.height, s.min_width, s.max_width),
        (Length::Percent(100.0), Length::Px(0.0), Length::Px(448.0))
    );
    assert_eq!(
        tw("px-2 p-4").padding,
        tw("p-4 px-2").padding,
        "whatever the order"
    );
    assert_eq!(
        tw("border-x-2 border-t-4").border,
        Edges::new(4.0, 2.0, 0.0, 2.0)
    );
    assert_eq!(
        tw("p-4").box_sizing,
        crate::BoxSizing::BorderBox,
        "preflight"
    );
    assert_eq!(
        tw("p-4 box-content").box_sizing,
        crate::BoxSizing::ContentBox
    );
}

#[test]
fn preflight_measures_border_box() {
    let mut ui = Ui::new();
    let root = ui.root();
    let b = ui.add_tw(root, "h-9 py-2 border");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(b).height, 36.0);
    ui.set_global(Tailwind::default().preflight(false));
    let c = ui.add_tw(root, "h-9 py-2");
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(c).height, 52.0);
}

#[test]
fn colours_and_text() {
    let s = tw("bg-blue-500/50 text-white text-sm font-semibold border-2 rounded-full shadow-md opacity-75");
    assert!(close(
        s.background.unwrap(),
        [59.0 / 255.0, 130.0 / 255.0, 246.0 / 255.0, 0.5]
    ));
    assert_eq!(s.color, [1.0; 4]);
    assert_eq!((s.font_size, s.line_height), (14.0, Some(20.0)));
    assert_eq!(s.font_weight, FontWeight::Weight(600));
    assert_eq!(s.border, Edges::all(2.0));
    assert_eq!(s.border_color, rgb(0xe5e7eb), "preflight's border colour");
    assert_eq!(s.border_radius, Length::Px(9999.0));
    assert_eq!(s.box_shadow.len(), 2);
    assert_eq!(s.opacity, 0.75);
    assert_eq!(tw("border border-red-500").border_color, rgb(0xef4444));
    assert_eq!(
        tw("leading-tight text-2xl").line_height,
        Some(30.0),
        "leading wins over the size's"
    );
    assert_eq!(tw("text-sm/6").line_height, Some(24.0));
    assert_eq!(tw("text-[13px] leading-[18px]").line_height, Some(18.0));
    let byte = |c: Rgba| c.map(|v| (v * 255.0).round() as u8);
    assert_eq!(
        byte(tw("text-[oklch(0.205_0_0)]").color),
        byte(crate::css::color("#171717").unwrap())
    );
}

#[test]
fn modern_css_colours() {
    let c = |v: &str| crate::css::color(v).unwrap();
    // Tailwind 4's neutral-900 and red-500, given in both forms.
    let b = |c: Rgba| c.map(|v| (v * 255.0).round() as u8);
    assert_eq!(b(c("oklch(0.205 0 0)")), b(c("#171717")));
    let red = b(c("oklch(63.7% 0.237 25.331)"));
    let want = [0xfb, 0x2c, 0x36];
    assert!((0..3).all(|i| red[i].abs_diff(want[i]) <= 1), "{red:?}");
    assert_eq!(c("rgb(255 0 0 / 50%)"), [1.0, 0.0, 0.0, 0.5]);
    assert_eq!(c("hsl(120 100% 50%)"), [0.0, 1.0, 0.0, 1.0]);
    assert_eq!(c("hsla(0, 100%, 50%, 0.25)"), [1.0, 0.0, 0.0, 0.25]);
    assert_eq!(c("oklch(1 0 0 / 10%)")[3], 0.1);
}

#[test]
fn arbitrary_values_and_properties() {
    let s =
        tw("w-[137px] bg-[#1da1f2] grid grid-cols-[200px_1fr] col-span-2 [opacity:0.4] top-[-3px]");
    assert_eq!(s.width, Length::Px(137.0));
    assert_eq!(s.background, crate::css::color("#1da1f2"));
    assert_eq!(s.grid_columns.len(), 2);
    assert_eq!(s.grid_column.span, 2);
    assert_eq!(s.opacity, 0.4);
    assert_eq!(s.top, Length::Px(-3.0));
    assert_eq!(tw("grid-cols-3").grid_columns.len(), 3);
    let p = tw("col-start-2 col-span-3").grid_column;
    assert_eq!((p.start, p.span), (Some(2), 3));
}

#[test]
fn several_classes_make_one_property() {
    let s = tw("bg-gradient-to-r from-pink-500 via-red-500 to-yellow-500");
    assert_eq!(s.background_gradient.len(), 3);
    assert_eq!(s.background_gradient[0].1, rgb(0xec4899));
    assert_eq!(s.gradient_shape, crate::GradientShape::Linear(90.0));
    let s = tw("backdrop-blur-md backdrop-saturate-150");
    assert_eq!((s.backdrop_blur, s.backdrop_saturate), (12.0, 1.5));
    // A ring over its offset over the shadow.
    let s = tw("shadow-sm ring-2 ring-red-500 ring-offset-2 ring-offset-black");
    let spreads: Vec<f32> = s.box_shadow.iter().map(|b| b.spread).collect();
    assert_eq!(spreads, [2.0, 4.0, 0.0]);
    assert_eq!(s.box_shadow[0].color, [0.0, 0.0, 0.0, 1.0]);
    assert_eq!(s.box_shadow[1].color, rgb(0xef4444));
    assert!(
        Tailwind::default().style("from-red-500").is_err(),
        "stops without a direction"
    );
}

#[test]
fn unsupported_classes_are_named_with_a_reason() {
    let t = Tailwind::default();
    for bad in [
        "fixed",
        "z-10",
        "mx-auto",
        "space-x-2",
        "tracking-wide",
        "rounded-t-lg",
        "shadow-inner",
        "bg-nope-500",
        "-p-2",
        "translate-x-1/2",
        "scale-x-50",
        "ease-in",
        "first:p-2",
        "w-screen",
    ] {
        let e = t.parse(bad).err().unwrap_or_else(|| panic!("`{bad}` read"));
        assert_eq!(e.class, bad);
        assert!(!e.reason.is_empty());
    }
    assert_eq!(t.parse("p-4 flex sticky").unwrap_err().class, "sticky");
    assert!(t
        .parse("mx-auto")
        .unwrap_err()
        .reason
        .contains("justify-center"));
    assert!(
        t.parse("bg-blu-500")
            .unwrap_err()
            .reason
            .contains("`bg-blue-500`"),
        "a suggestion"
    );
    assert!(t.parse("hovr:p-2").unwrap_err().reason.contains("`hover:`"));
}

#[test]
fn states_and_groups() {
    let mut ui = Ui::new();
    let root = ui.root();
    let row = ui.add_tw(
        root,
        "group h-5 bg-slate-800 hover:bg-slate-700 transition duration-300",
    );
    let x = ui.add_tw(row, "w-5 h-5 opacity-0 group-hover:opacity-100");
    ui.layout(200.0, 100.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.style(row).background, Some(rgb(0x1e293b)));
    hover(&mut ui, 10.0, 10.0);
    assert_eq!(ui.style(x).opacity, 1.0);
    assert_eq!(ui.node(row).states.as_ref().unwrap().transition, 0.3);
    ui.set_theme(Theme::Dark);
    let d = ui.add_tw(root, "bg-white dark:bg-black");
    assert_eq!(ui.style(d).background, Some([0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn variants_stack() {
    let mut ui = Ui::new();
    let root = ui.root();
    let n = ui.add_tw(
        root,
        "h-10 bg-white hover:bg-red-500 dark:bg-black dark:hover:bg-blue-500",
    );
    ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
    hover(&mut ui, 5.0, 5.0);
    assert_eq!(ui.style(n).background, Some(rgb(0xef4444)));
    ui.set_theme(Theme::Dark);
    assert_eq!(
        ui.style(n).background,
        Some(rgb(0x3b82f6)),
        "dark and hovered"
    );
    hover(&mut ui, 5.0, 200.0);
    assert_eq!(ui.style(n).background, Some([0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn breakpoints_follow_the_window() {
    let mut ui = Ui::new();
    let root = ui.root();
    let n = ui.add_tw(
        root,
        "flex flex-col h-10 md:flex-row lg:gap-8 max-sm:p-1 md:hover:bg-red-500",
    );
    let lay = |ui: &mut Ui, w: f32| ui.layout(w, 400.0, &mut FixedWidthText(7.0));
    lay(&mut ui, 500.0);
    assert_eq!(
        (ui.style(n).flex_direction, ui.style(n).padding.top),
        (FlexDirection::Column, 4.0)
    );
    lay(&mut ui, 700.0);
    assert_eq!(
        (ui.style(n).flex_direction, ui.style(n).padding.top),
        (FlexDirection::Column, 0.0)
    );
    lay(&mut ui, 800.0);
    assert_eq!(
        (ui.style(n).flex_direction, ui.style(n).gap),
        (FlexDirection::Row, 0.0)
    );
    hover(&mut ui, 5.0, 5.0);
    assert_eq!(ui.style(n).background, Some(rgb(0xef4444)), "md:hover:");
    lay(&mut ui, 1100.0);
    assert_eq!(ui.style(n).gap, 32.0);
    assert_eq!(ui.style(n).background, Some(rgb(0xef4444)), "still hovered");
    lay(&mut ui, 700.0);
    assert_eq!(
        (ui.style(n).flex_direction, ui.style(n).background),
        (FlexDirection::Column, None)
    );
}

#[test]
fn transforms_in_3d() {
    let mut ui = Ui::new();
    let root = ui.root();
    let card = ui.add_tw(root, "w-40 h-40 perspective-near rotate-y-12 -rotate-x-6 translate-z-4 origin-left backface-hidden glare");
    let t = ui.transform3d(card).expect("a transform");
    assert!(
        (t.rotate_y - 12f32.to_radians()).abs() < 1e-6
            && (t.rotate_x + 6f32.to_radians()).abs() < 1e-6
    );
    assert_eq!(
        (t.depth, t.perspective, t.origin, t.hide_backface, t.glare),
        (16.0, 300.0, (0.0, 0.5), true, true)
    );
    // A state moves it, and leaving the state moves it back.
    let lift = ui.add_tw(
        root,
        "w-40 h-40 hover:-translate-y-1 hover:scale-105 active:rotate-3",
    );
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.transform3d(lift).unwrap().scale, 1.0);
    hover(&mut ui, 5.0, 200.0);
    let t = ui.transform3d(lift).unwrap();
    assert_eq!((t.scale, t.translate_y), (1.05, -4.0));
    hover(&mut ui, 300.0, 390.0);
    assert_eq!(ui.transform3d(lift).unwrap().scale, 1.0);
    assert!(tw("p-1").box_sizing == crate::BoxSizing::BorderBox);
    assert!(
        Tailwind::default().style("rotate-y-12").is_err(),
        "a transform needs a node"
    );
}

#[test]
fn transforms_move_over_a_transition() {
    let mut ui = Ui::new();
    let root = ui.root();
    let n = ui.add_tw(root, "w-20 h-20 transition duration-200 hover:scale-110");
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    hover(&mut ui, 5.0, 5.0);
    assert_eq!(ui.transform3d(n).unwrap().scale, 1.0, "not there at once");
    ui.step(0.1);
    let mid = ui.transform3d(n).unwrap().scale;
    assert!(mid > 1.0 && mid < 1.1, "{mid}");
    ui.step(0.2);
    assert!((ui.transform3d(n).unwrap().scale - 1.1).abs() < 1e-5);
}

#[test]
fn components_and_config() {
    let t = Tailwind::default()
        .color("brand", "#ff6600")
        .component("pad", "px-4 py-2 rounded")
        .component("btn", "pad hover:bg-brand")
        .screen("tablet", 900.0);
    let c = t.parse("btn md:pad tablet:p-1").unwrap();
    assert_eq!(c.base.len(), 6, "preflight and five from `pad`");
    assert_eq!(
        c.rules[0].declarations,
        vec![(
            "background-color".to_owned(),
            "rgba(255,102,0,1)".to_owned()
        )]
    );
    let widths: Vec<f32> = c.rules.iter().filter_map(|r| r.when.min_width).collect();
    assert_eq!(widths, [768.0, 900.0]);
    // A component's variant stacks with the variants inside it.
    let c = t.parse("md:btn").unwrap();
    assert!(c
        .rules
        .iter()
        .any(|r| r.when == When::default().hover().min_width(768.0)));
    let e = Tailwind::default()
        .component("a", "b")
        .component("b", "a")
        .parse("a")
        .unwrap_err();
    assert!(e.reason.contains("nest"), "{e}");
}

#[test]
fn adaptive_colours_switch_with_the_appearance() {
    let mut ui = Ui::new();
    ui.set_global(Tailwind::default().adaptive_color("surface", "#ffffff", "#000000"));
    let root = ui.root();
    let n = ui.add_tw(root, "h-10 bg-surface hover:bg-red-500");
    ui.layout(400.0, 400.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.style(n).background, Some([1.0; 4]));
    ui.set_theme(Theme::Dark);
    assert_eq!(ui.style(n).background, Some([0.0, 0.0, 0.0, 1.0]));
    // The theme's colour switches under the state's, not over it.
    hover(&mut ui, 5.0, 5.0);
    assert_eq!(ui.style(n).background, Some(rgb(0xef4444)));
    // A later class replaces it in the dark too (a variant over a
    // component's colour).
    let over = ui.add_tw(root, "h-10 bg-surface text-white bg-transparent");
    assert_eq!(ui.style(over).background, Some([0.0; 4]));
}

#[test]
fn shadcn_components() {
    let mut ui = Ui::new();
    ui.set_global(Tailwind::shadcn());
    let root = ui.root();
    let b = ui.add_tw(root, "button");
    let outline = ui.add_tw(root, "button button-outline button-sm");
    let c = |v: &str| crate::css::color(v).unwrap();
    ui.layout(800.0, 600.0, &mut FixedWidthText(7.0));
    assert_eq!(ui.rect(b).height, 36.0);
    assert_eq!(ui.rect(outline).height, 32.0);
    assert!(close(
        ui.style(b).background.unwrap(),
        c("oklch(0.205 0 0)")
    ));
    assert!(close(ui.style(b).color, c("oklch(0.985 0 0)")));
    assert_eq!(ui.style(b).border_radius, Length::Px(8.0));
    assert_eq!(ui.style(outline).border, Edges::all(1.0));
    assert!(close(ui.style(outline).border_color, c("oklch(0.922 0 0)")));
    ui.set_theme(Theme::Dark);
    // `transition-all`: the theme's colours move there.
    assert!(ui.is_animating(b));
    ui.step(1.0);
    assert!(close(
        ui.style(b).background.unwrap(),
        c("oklch(0.922 0 0)")
    ));
    assert!(
        close(ui.style(outline).border_color, c("oklch(1 0 0 / 15%)")),
        "dark:border-input"
    );
    // Every component reads, alone and with each of its variants.
    let t = Tailwind::shadcn();
    for name in t.components.keys() {
        t.parse(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn focus_rings_on_keyboard_focus() {
    let mut ui = Ui::new();
    ui.set_global(Tailwind::shadcn());
    let root = ui.root();
    let b = ui.add_tw(root, "button");
    assert_eq!(ui.style(b).box_shadow.len(), 1, "shadow-xs");
    ui.set_interaction(crate::states::Interaction {
        focused: Some(b),
        focus_visible: true,
        ..Default::default()
    });
    let s = ui.style(b);
    assert_eq!(s.box_shadow.len(), 2, "the ring over the shadow");
    assert_eq!(s.box_shadow[0].spread, 3.0);
}

#[cfg(feature = "json")]
#[test]
fn json_replaces_or_extends() {
    let t = Tailwind::from_json(r##"{"theme": {"colors": {"ink": "#111"}, "screens": {"wide": "1000px"}, "extend": {"borderRadius": {"DEFAULT": "0.5rem"}, "perspective": {"far": "2000px"}}}}"##).unwrap();
    assert_eq!(
        t.style("bg-ink rounded").unwrap().border_radius,
        Length::Px(8.0)
    );
    assert!(
        t.style("bg-red-500").is_err(),
        "theme.colors replaced the palette"
    );
    assert!(t.parse("md:p-1").is_err() && t.parse("wide:p-1").is_ok());
    assert!(t.parse("perspective-far").is_ok());
    assert!(Tailwind::from_json(r#"{"spacing": {"x": "1em"}}"#).is_err());
    assert!(Tailwind::from_json(r##"{"colors": {"x": ["#fff"]}}"##).is_err());
}
