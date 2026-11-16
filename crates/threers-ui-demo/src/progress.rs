//! The Progress page: plain indicators as a platform draws them (a bar, a
//! ring, an activity indicator's spokes) in the theme's colours, glowing
//! ones drawn by signed distance fields (flat neon and solid glass), their
//! presets, motion for indicators and for any box, and any control drawn
//! flat or solid. One slider moves every indicator that has a value; the
//! rest are busy.

use threers_ui::button::{button, ButtonRole};
use threers_ui::environment::Token;
use threers_ui::indicators::{
    indicator, progress_view, Depth, Fill, Indicator, IndicatorStyle, Label, LabelAlign, Motion,
    Palette, PlainKind, PlainStyle, Preset, ProgressView, Shape,
};
use threers_ui::input::Input;
use threers_ui::motion::MotionSpec;
use threers_ui::pickers::slider;
use threers_ui::solid::{DepthAlign, SolidStyle};
use threers_ui::states::Appearance;
use threers_ui::{style, NodeId, Ui};
use threers_ui::{Role, Semantics};

use crate::app::State;
use crate::showcase::group;

/// The indicators with a value, to move when the slider does.
pub struct Progress {
    plain: Vec<ProgressView>,
    glowing: Vec<Indicator>,
    /// The busy ones and those at a value of their own, redrawn when the
    /// appearance changes.
    busy: Vec<Indicator>,
    fixed: Vec<(Indicator, f32)>,
    page: NodeId,
    last: f32,
    appearance: Appearance,
}

impl Progress {
    /// The slider's value onto the indicators, when it moved; and the
    /// glowing ones redrawn for a light or a dark page when the appearance
    /// changes.
    pub fn update(&mut self, ui: &mut Ui, s: &State) {
        let appearance = ui.appearance_of(self.page);
        if s.progress == self.last && appearance == self.appearance {
            return;
        }
        self.last = s.progress;
        self.appearance = appearance;
        for v in &self.plain {
            v.set(ui, Some(s.progress));
        }
        for i in &self.glowing {
            i.set(ui, Some(s.progress));
        }
        for i in &self.busy {
            i.set(ui, None);
        }
        for (i, v) in &self.fixed {
            i.set(ui, Some(*v));
        }
    }
}

/// A labelled cell: what it is under the indicator.
fn cell(ui: &mut Ui, parent: NodeId, label: &str, width: Option<f32>) -> NodeId {
    let c = ui.add_box(parent, style! { display: flex; flex-direction: column; align-items: center; justify-content: flex-end; gap: 10px; flex-shrink: 0; });
    if let Some(w) = width {
        ui.update_style(c, |s| {
            s.width = threers_ui::Length::Px(w);
            s.align_items = threers_ui::AlignItems::Stretch;
        });
    }
    let slot = ui.add_box(c, style! { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 24px; });
    if width.is_some() {
        ui.update_style(slot, |s| s.align_items = threers_ui::AlignItems::Stretch);
    }
    let l = ui.add_box(c, style! { font-size: 11px; text-align: center; });
    ui.token_style(l, &[("color", Token::SecondaryText)]);
    ui.add_text(l, label);
    slot
}

/// A panel for the glowing ones: pale on a light page, near black on a
/// dark one (where light shows best).
fn stage(ui: &mut Ui, parent: NodeId) -> NodeId {
    let s = ui.add_box(parent, style! { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 30px 34px; padding: 26px; border-radius: 12px; background: #eef0f4; });
    ui.appearance_style(s, Appearance::Dark, "background: #0b0d12");
    s
}

/// A labelled cell on a panel.
fn dark_cell(ui: &mut Ui, parent: NodeId, label: &str, width: Option<f32>) -> NodeId {
    let c = ui.add_box(parent, style! { display: flex; flex-direction: column; align-items: center; gap: 12px; flex-shrink: 0; });
    let slot = ui.add_box(c, style! { display: flex; flex-direction: column; align-items: center; justify-content: center; });
    if let Some(w) = width {
        ui.update_style(c, |s| {
            s.width = threers_ui::Length::Px(w);
            s.align_items = threers_ui::AlignItems::Stretch;
        });
        ui.update_style(slot, |s| s.align_items = threers_ui::AlignItems::Stretch);
    }
    let l = ui.add_box(c, style! { font-size: 11px; text-align: center; });
    ui.token_style(l, &[("color", Token::SecondaryText)]);
    ui.add_text(l, label);
    slot
}

pub fn build(ui: &mut Ui, input: &mut Input<State>, state: &State, page: NodeId) -> Progress {
    let v = Some(state.progress);
    let mut plain = vec![];
    let mut glowing = vec![];
    let mut busy = vec![];
    let mut fixed = vec![];

    // The slider that moves them all.
    let g = group(
        ui,
        page,
        "Value",
        "Every indicator with a value follows this; the others are busy and keep moving.",
    );
    let row = ui.add_box(
        g,
        style! { display: flex; align-items: center; gap: 12px; max-width: 420px; },
    );
    let l = ui.add_box(
        row,
        style! { font-size: 12px; width: 60px; flex-shrink: 0; },
    );
    ui.add_text(l, "Progress");
    let s = slider(
        ui,
        input,
        state,
        row,
        "Progress",
        threers_ui::binding!(|s: State| s.progress),
        0.0..=1.0,
        0.01,
    );
    ui.update_style(s, |s| s.flex_grow = 1.0);

    // Plain, in the theme's colours.
    let g = group(ui, page, "Plain", "As a platform draws them, in the theme's accent: a bar, a ring and an activity indicator, each with a value or busy. No shader: every painter draws these.");
    let grid = ui.add_box(
        g,
        style! { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 24px 32px; },
    );
    let slot = cell(ui, grid, "Linear", Some(200.0));
    plain.push(progress_view(
        ui,
        slot,
        "Downloading",
        PlainStyle::new(PlainKind::Linear),
        v,
    ));
    let slot = cell(ui, grid, "Linear, busy", Some(200.0));
    progress_view(
        ui,
        slot,
        "Preparing",
        PlainStyle::new(PlainKind::Linear),
        None,
    );
    let slot = cell(ui, grid, "Circular", None);
    plain.push(progress_view(
        ui,
        slot,
        "Syncing",
        PlainStyle::new(PlainKind::Circular),
        v,
    ));
    let slot = cell(ui, grid, "Circular, busy", None);
    progress_view(
        ui,
        slot,
        "Connecting",
        PlainStyle::new(PlainKind::Circular),
        None,
    );
    let slot = cell(ui, grid, "Large", None);
    plain.push(progress_view(
        ui,
        slot,
        "Exporting",
        PlainStyle::new(PlainKind::Circular).size(36.0).stroke(4.0),
        v,
    ));
    let slot = cell(ui, grid, "Spinner", None);
    progress_view(
        ui,
        slot,
        "Loading",
        PlainStyle::new(PlainKind::Spokes),
        None,
    );
    let slot = cell(ui, grid, "Spinner, large", None);
    progress_view(
        ui,
        slot,
        "Loading",
        PlainStyle::new(PlainKind::Spokes).size(32.0),
        None,
    );
    let slot = cell(ui, grid, "Spokes, value", None);
    plain.push(progress_view(
        ui,
        slot,
        "Steps",
        PlainStyle::new(PlainKind::Spokes).size(32.0),
        v,
    ));
    let slot = cell(ui, grid, "Tinted", Some(160.0));
    plain.push(progress_view(
        ui,
        slot,
        "Battery",
        PlainStyle::new(PlainKind::Linear)
            .size(6.0)
            .tint([0.2, 0.78, 0.35, 1.0]),
        v,
    ));
    let slot = cell(ui, grid, "Tinted ring", None);
    plain.push(progress_view(
        ui,
        slot,
        "Storage",
        PlainStyle::new(PlainKind::Circular)
            .size(36.0)
            .stroke(5.0)
            .tint([1.0, 0.58, 0.0, 1.0]),
        v,
    ));

    // Glowing, flat.
    let palettes = [
        ("Aurora", Palette::AURORA),
        ("Ember", Palette::EMBER),
        ("Ocean", Palette::OCEAN),
        ("Neon", Palette::NEON),
        ("Citrus", Palette::CITRUS),
    ];
    let g = group(ui, page, "Neon", "Signed distance fields drawn in one shader per indicator: a white-hot core, colour through three stops along the length, and light spilling past the box. The 2D canvas draws a plain gradient bar or arc instead.");
    let panel = stage(ui, g);
    for (i, (name, p)) in palettes.iter().enumerate() {
        let slot = dark_cell(ui, panel, name, Some(220.0));
        let st = IndicatorStyle::new(Shape::Bar)
            .palette(*p)
            .size(8.0 + i as f32 * 2.0)
            .shimmer(i == 3);
        glowing.push(indicator(ui, slot, name, st, v));
    }
    for (name, p) in palettes {
        let slot = dark_cell(ui, panel, name, None);
        glowing.push(indicator(
            ui,
            slot,
            name,
            IndicatorStyle::new(Shape::Ring).palette(p).size(84.0),
            v,
        ));
    }

    // Glowing, solid.
    let g = group(ui, page, "Glass", "The same, solid: a glass tube with glowing liquid in it, shaded in 3D. The bar's tube is exact from its circular sections; the ring's torus is ray-marched, tipped back so its depth shows.");
    let panel = stage(ui, g);
    for (i, (name, p)) in palettes.iter().enumerate() {
        let slot = dark_cell(ui, panel, name, Some(220.0));
        let st = IndicatorStyle::new(Shape::Bar)
            .depth(Depth::Solid)
            .palette(*p)
            .size(18.0 + i as f32 * 2.0)
            .shimmer(i == 1);
        glowing.push(indicator(ui, slot, name, st, v));
    }
    for (name, p) in palettes {
        let slot = dark_cell(ui, panel, name, None);
        glowing.push(indicator(
            ui,
            slot,
            name,
            IndicatorStyle::new(Shape::Ring)
                .depth(Depth::Solid)
                .palette(p)
                .size(96.0)
                .thickness(0.3),
            v,
        ));
    }

    // Busy.
    let g = group(ui, page, "Busy", "With no value: a comet runs along the bar, an arc chases round the ring, growing and shrinking.");
    let panel = stage(ui, g);
    for (depth, (name, p)) in [(Depth::Flat, palettes[3]), (Depth::Solid, palettes[2])] {
        let slot = dark_cell(ui, panel, name, Some(220.0));
        let size = if depth == Depth::Flat { 10.0 } else { 20.0 };
        busy.push(indicator(
            ui,
            slot,
            name,
            IndicatorStyle::new(Shape::Bar)
                .depth(depth)
                .palette(p)
                .size(size),
            None,
        ));
    }
    for (depth, (name, p)) in [
        (Depth::Flat, palettes[0]),
        (Depth::Flat, palettes[1]),
        (Depth::Solid, palettes[3]),
        (Depth::Solid, palettes[4]),
    ] {
        let slot = dark_cell(ui, panel, name, None);
        busy.push(indicator(
            ui,
            slot,
            name,
            IndicatorStyle::new(Shape::Ring)
                .depth(depth)
                .palette(p)
                .size(84.0)
                .thickness(0.28),
            None,
        ));
    }

    // Presets.
    let g = group(ui, page, "Presets", "Ready-made looks, each a style to change further: IndicatorStyle::from(Preset::LavaTube).palette(Palette::OCEAN).");
    let panel = stage(ui, g);
    for p in Preset::ALL {
        let st = p.style();
        let width = (st.shape == Shape::Bar).then_some(200.0);
        let slot = dark_cell(ui, panel, p.name(), width);
        glowing.push(indicator(ui, slot, p.name(), st, v));
    }

    // Labels.
    let g = group(ui, page, "Labels", "Text inside or beside any indicator: the percentage, a template filled from the value, or words, with their own size, weight, font, colour and place; what it says while busy too.");
    let panel = stage(ui, g);
    let ring = |p: Palette, d: Depth| {
        IndicatorStyle::new(Shape::Ring)
            .depth(d)
            .palette(p)
            .size(96.0)
            .thickness(0.26)
    };
    let slot = dark_cell(ui, panel, "Percent", None);
    glowing.push(
        indicator(ui, slot, "Upload", ring(Palette::AURORA, Depth::Flat), v)
            .labelled(ui, Label::percent().size(22.0)),
    );
    let slot = dark_cell(ui, panel, "Monospace", None);
    glowing.push(
        indicator(ui, slot, "Render", ring(Palette::EMBER, Depth::Solid), v).labelled(
            ui,
            Label::percent()
                .font("monospace")
                .color([0.92, 0.35, 0.05, 1.0]),
        ),
    );
    let slot = dark_cell(ui, panel, "Decimals", None);
    glowing.push(
        indicator(ui, slot, "Index", ring(Palette::ICE, Depth::Solid), v)
            .labelled(ui, Label::percent().decimals(1).size(17.0).regular()),
    );
    let slot = dark_cell(ui, panel, "Busy text", None);
    busy.push(
        indicator(ui, slot, "Sync", ring(Palette::NEON, Depth::Flat), None)
            .labelled(ui, Label::text("Sync").busy("Syncing…").italic().size(15.0)),
    );
    let slot = dark_cell(ui, panel, "Inside", Some(220.0));
    glowing.push(
        indicator(
            ui,
            slot,
            "Copy",
            IndicatorStyle::new(Shape::Bar)
                .depth(Depth::Solid)
                .palette(Palette::OCEAN)
                .size(26.0),
            v,
        )
        .labelled(ui, Label::percent().size(13.0)),
    );
    let slot = dark_cell(ui, panel, "Above", Some(220.0));
    glowing.push(
        indicator(
            ui,
            slot,
            "Disk",
            IndicatorStyle::new(Shape::Bar)
                .palette(Palette::CITRUS)
                .size(8.0),
            v,
        )
        .labelled(
            ui,
            Label::template("{value} of {max} GB")
                .max(256.0)
                .align(LabelAlign::Above)
                .regular(),
        ),
    );
    let slot = dark_cell(ui, panel, "After", Some(170.0));
    glowing.push(
        indicator(
            ui,
            slot,
            "Steps",
            IndicatorStyle::new(Shape::Bar)
                .palette(Palette::MINT)
                .size(8.0),
            v,
        )
        .labelled(
            ui,
            Label::template("{value}/{max}")
                .max(12.0)
                .align(LabelAlign::After),
        ),
    );
    let slot = dark_cell(ui, panel, "Charging", Some(160.0));
    glowing.push(
        indicator(ui, slot, "Battery", Preset::Charging.style(), v)
            .labelled(ui, Label::percent().align(LabelAlign::After).size(13.0)),
    );
    let row = ui.add_box(g, style! { display: flex; flex-wrap: wrap; align-items: center; gap: 28px; padding: 18px 4px 4px; });
    let slot = cell(ui, row, "Plain ring", None);
    plain.push(
        progress_view(
            ui,
            slot,
            "Export",
            PlainStyle::new(PlainKind::Circular).size(48.0).stroke(5.0),
            v,
        )
        .labelled(ui, Label::percent().size(12.0)),
    );
    let slot = cell(ui, row, "Plain bar", Some(220.0));
    plain.push(
        progress_view(
            ui,
            slot,
            "Download",
            PlainStyle::new(PlainKind::Linear).size(6.0),
            v,
        )
        .labelled(
            ui,
            Label::template("{percent}% of 1.2 GB")
                .align(LabelAlign::Above)
                .regular(),
        ),
    );

    // Batteries.
    let g = group(ui, page, "Battery", "A battery's charge, coloured by phase: below the first share the palette's first colour, below the second its middle one, else its last. Flat or glass; any palette and any thresholds.");
    let panel = stage(ui, g);
    for v in [0.12, 0.38, 0.86] {
        let slot = dark_cell(ui, panel, &format!("{}%", (v * 100.0) as i32), None);
        fixed.push((
            indicator(
                ui,
                slot,
                "Battery",
                Preset::Battery.style().size(30.0),
                Some(v),
            ),
            v,
        ));
    }
    let slot = dark_cell(ui, panel, "Follows the slider", None);
    glowing.push(
        indicator(ui, slot, "Battery", Preset::Battery.style().size(30.0), v)
            .labelled(ui, Label::percent().align(LabelAlign::After).size(12.0)),
    );
    let slot = dark_cell(ui, panel, "Glass, charging", None);
    glowing.push(indicator(ui, slot, "Charging", Preset::Charging.style(), v));
    let slot = dark_cell(ui, panel, "Own colours, 30/70", None);
    glowing.push(indicator(
        ui,
        slot,
        "Battery",
        IndicatorStyle::new(Shape::Battery)
            .palette(Palette::hex(0x7c3aed, 0x2563eb, 0x06b6d4))
            .phases(0.3, 0.7)
            .size(30.0),
        v,
    ));

    // Electricity.
    let g = group(ui, page, "Electric", "Bolts of lightning crackling along the fill, struck anew many times a second, plasma in the tube and sparks at the head: Fill::Electric on any bar or ring.");
    let panel = stage(ui, g);
    let slot = dark_cell(ui, panel, "Lightning", Some(260.0));
    glowing.push(indicator(
        ui,
        slot,
        "Lightning",
        Preset::Lightning.style(),
        v,
    ));
    let slot = dark_cell(ui, panel, "Ember", Some(260.0));
    glowing.push(indicator(
        ui,
        slot,
        "Ember",
        IndicatorStyle::new(Shape::Bar)
            .palette(Palette::EMBER)
            .size(14.0)
            .electric(),
        v,
    ));
    let slot = dark_cell(ui, panel, "Tesla", None);
    glowing.push(
        indicator(ui, slot, "Tesla", Preset::Tesla.style(), v)
            .labelled(ui, Label::percent().size(18.0)),
    );
    let slot = dark_cell(ui, panel, "Busy", None);
    busy.push(indicator(
        ui,
        slot,
        "Arc",
        Preset::Tesla.style().palette(Palette::NEON),
        None,
    ));

    // Effects.
    let g = group(ui, page, "Effects", "Fills that move: fire licking up off the bar (out from a ring) with embers rising, water with caustics and bubbles, swirling plasma, cracked and sparkling frost. Fill::Fire, Water, Plasma, Frost on any bar or ring, in any palette.");
    let panel = stage(ui, g);
    for p in [Preset::Inferno, Preset::Aquarium, Preset::Glacier] {
        let slot = dark_cell(ui, panel, p.name(), Some(260.0));
        glowing.push(indicator(ui, slot, p.name(), p.style(), v));
    }
    let slot = dark_cell(ui, panel, "Plasma", Some(260.0));
    glowing.push(indicator(
        ui,
        slot,
        "Plasma",
        IndicatorStyle::new(Shape::Bar)
            .palette(Palette::NEON)
            .size(18.0)
            .fill(Fill::Plasma),
        v,
    ));
    for p in [Preset::FireRing, Preset::Nebula] {
        let slot = dark_cell(ui, panel, p.name(), None);
        glowing.push(
            indicator(ui, slot, p.name(), p.style(), v).labelled(ui, Label::percent().size(18.0)),
        );
    }
    let slot = dark_cell(ui, panel, "Frost", None);
    glowing.push(indicator(
        ui,
        slot,
        "Frost",
        IndicatorStyle::new(Shape::Ring)
            .palette(Palette::ICE)
            .size(96.0)
            .thickness(0.3)
            .fill(Fill::Frost),
        v,
    ));
    let slot = dark_cell(ui, panel, "Fire, busy", None);
    busy.push(indicator(
        ui,
        slot,
        "Burning",
        IndicatorStyle::new(Shape::Ring)
            .palette(Palette::FIRE)
            .size(96.0)
            .thickness(0.2)
            .fire(),
        None,
    ));

    // Motion.
    let g = group(ui, page, "Motion", "Spin, rock, roll, bounce, buzz, pulse and float. Glass rings turn in true 3D inside their shader; any other box moves through its 3D transform (Ui::set_motion), in every painter. All of it stops with reduced motion.");
    let panel = stage(ui, g);
    let palettes_cycle = [
        Palette::AURORA,
        Palette::EMBER,
        Palette::OCEAN,
        Palette::NEON,
        Palette::CITRUS,
        Palette::SUNSET,
        Palette::MINT,
    ];
    for (i, m) in Motion::ALL.into_iter().skip(1).enumerate() {
        let slot = dark_cell(ui, panel, m.name(), None);
        let st = IndicatorStyle::new(Shape::Ring)
            .depth(Depth::Solid)
            .palette(palettes_cycle[i])
            .size(84.0)
            .thickness(0.3)
            .motion(m);
        glowing.push(indicator(ui, slot, m.name(), st, v));
    }
    let row = ui.add_box(g, style! { display: flex; flex-wrap: wrap; gap: 26px; padding: 16px 4px; align-items: center; });
    for m in Motion::ALL.into_iter().skip(1) {
        let b = ui.add_box(row, style! { display: flex; align-items: center; justify-content: center; width: 86px; height: 44px; border-radius: 12px; font-size: 13px; font-weight: bold; });
        ui.token_style(
            b,
            &[("background", Token::Accent), ("color", Token::OnAccent)],
        );
        ui.add_text(b, m.name());
        ui.set_semantics(
            b,
            Semantics::new(Role::Image).label(format!("A box that moves: {}", m.name())),
        );
        ui.set_motion(b, MotionSpec::new(m).amount(0.8));
    }

    // Flat or solid, for any control.
    let g = group(ui, page, "Depth", "Any control drawn flat or solid: Ui::set_subtree_depth turns every button, field, slider and progress bar under a box into a slab of lit, bevelled glass, ray-traced in 3D with its walls and shadow, and back. Turned (a motion, a tilt, a 3D transform), its walls show; which face lies on the page is up to you. The GPU draws it; the 2D canvas stays flat.");
    let panel = ui.add_box(g, style! { display: flex; flex-wrap: wrap; align-items: center; gap: 18px; padding: 8px 2px; });
    let controls = ui.add_box(
        panel,
        style! { display: flex; flex-wrap: wrap; align-items: center; gap: 14px; },
    );
    button(ui, input, controls, "Save", ButtonRole::Primary, |_, _| {});
    button(ui, input, controls, "Cancel", ButtonRole::Normal, |_, _| {});
    button(
        ui,
        input,
        controls,
        "Delete",
        ButtonRole::Destructive,
        |_, _| {},
    );
    let ring_slot = ui.add_box(controls, style! { display: flex; });
    plain.push(progress_view(
        ui,
        ring_slot,
        "Uploading",
        PlainStyle::new(PlainKind::Circular).size(32.0).stroke(4.0),
        v,
    ));
    let bar_slot = ui.add_box(
        controls,
        style! { display: flex; flex-direction: column; width: 160px; },
    );
    plain.push(progress_view(
        ui,
        bar_slot,
        "Copying",
        PlainStyle::new(PlainKind::Linear).size(10.0),
        v,
    ));
    let toggle = ui.add_box(panel, style! { display: flex; });
    button(
        ui,
        input,
        toggle,
        "Flat / solid",
        ButtonRole::Normal,
        move |s: &mut State, ui: &mut Ui| {
            s.progress_solid = !s.progress_solid;
            ui.set_subtree_depth(
                controls,
                if s.progress_solid {
                    Depth::Solid
                } else {
                    Depth::Flat
                },
            );
        },
    );
    if state.progress_solid {
        ui.set_subtree_depth(controls, Depth::Solid);
    }
    // Real 3D: slabs with their front, middle or back on the page, rocking
    // so their walls show.
    let row = ui.add_box(g, style! { display: flex; flex-wrap: wrap; gap: 36px; padding: 22px 8px 8px; align-items: center; });
    for (i, (name, align)) in [
        ("Front on the page", DepthAlign::Front),
        ("Middle", DepthAlign::Center),
        ("Back: stands out", DepthAlign::Back),
    ]
    .into_iter()
    .enumerate()
    {
        let card = ui.add_box(row, style! { display: flex; align-items: center; justify-content: center; width: 150px; height: 80px; border-radius: 16px; font-size: 13px; font-weight: bold; padding: 8px; box-sizing: border-box; text-align: center; white-space: normal; });
        ui.token_style(
            card,
            &[("background", Token::Accent), ("color", Token::OnAccent)],
        );
        ui.add_text(card, name);
        ui.set_semantics(card, Semantics::new(Role::Image).label(name));
        ui.set_solid(card, SolidStyle::default().thickness(20.0).align(align));
        ui.set_motion(
            card,
            MotionSpec::new(Motion::Rock).speed(0.6 + 0.15 * i as f32),
        );
    }

    let appearance = ui.appearance_of(page);
    Progress {
        plain,
        glowing,
        busy,
        fixed,
        page,
        last: state.progress,
        appearance,
    }
}
