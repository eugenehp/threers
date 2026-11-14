//! Ready-made indicator styles.

use super::*;

/// Ready-made looks. Each is an [`IndicatorStyle`] to change further:
/// `IndicatorStyle::from(Preset::LavaTube).palette(Palette::OCEAN)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Preset {
    /// A thin flat neon bar, pink to blue.
    NeonBar,
    /// A flat bar with no glow, in one colour: quiet.
    Minimal,
    /// A glass tube of molten orange, a shimmer running through it.
    LavaTube,
    /// A battery, red when low, amber, green when full.
    Battery,
    /// A glass ring, aurora colours.
    GlassRing,
    /// A glass ring turning round and round.
    Hologram,
    /// A flat ember ring, breathing.
    Heartbeat,
    /// A neon ring rocking side to side.
    Cyber,
    /// A glass ring bobbing gently, ocean colours.
    Buoy,
    /// A glass ring hopping like a dribbled ball.
    Dribble,
    /// A flat sunset bar buzzing now and then.
    Alert,
    /// A glass battery charging: a band of light running along the charge,
    /// breathing gently.
    Charging,
    /// A bar of crackling electricity.
    Lightning,
    /// A ring of crackling electricity.
    Tesla,
    /// A bar on fire.
    Inferno,
    /// A ring of fire.
    FireRing,
    /// A tube of water, bubbling.
    Aquarium,
    /// A ring of swirling plasma.
    Nebula,
    /// A bar freezing over.
    Glacier,
}

impl Preset {
    /// Every preset, in order.
    pub const ALL: [Preset; 19] = [
        Preset::NeonBar,
        Preset::Minimal,
        Preset::LavaTube,
        Preset::Battery,
        Preset::GlassRing,
        Preset::Hologram,
        Preset::Heartbeat,
        Preset::Cyber,
        Preset::Buoy,
        Preset::Dribble,
        Preset::Alert,
        Preset::Charging,
        Preset::Lightning,
        Preset::Tesla,
        Preset::Inferno,
        Preset::FireRing,
        Preset::Aquarium,
        Preset::Nebula,
        Preset::Glacier,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Preset::NeonBar => "Neon bar",
            Preset::Minimal => "Minimal",
            Preset::LavaTube => "Lava tube",
            Preset::Battery => "Battery",
            Preset::GlassRing => "Glass ring",
            Preset::Hologram => "Hologram",
            Preset::Heartbeat => "Heartbeat",
            Preset::Cyber => "Cyber",
            Preset::Buoy => "Buoy",
            Preset::Dribble => "Dribble",
            Preset::Alert => "Alert",
            Preset::Charging => "Charging",
            Preset::Lightning => "Lightning",
            Preset::Tesla => "Tesla",
            Preset::Inferno => "Inferno",
            Preset::FireRing => "Ring of fire",
            Preset::Aquarium => "Aquarium",
            Preset::Nebula => "Nebula",
            Preset::Glacier => "Glacier",
        }
    }

    pub fn style(self) -> IndicatorStyle {
        let bar = IndicatorStyle::new(Shape::Bar);
        let ring = IndicatorStyle::new(Shape::Ring);
        match self {
            Preset::NeonBar => bar.palette(Palette::NEON).size(6.0).glow(1.0),
            Preset::Minimal => bar
                .palette(Palette::solid(hex(0x3b82f6)))
                .size(6.0)
                .glow(0.0),
            Preset::LavaTube => bar
                .depth(Depth::Solid)
                .palette(Palette::EMBER)
                .size(22.0)
                .shimmer(true),
            Preset::Battery => IndicatorStyle::new(Shape::Battery)
                .palette(Palette::BATTERY)
                .phases(0.2, 0.5)
                .glow(0.5),
            Preset::GlassRing => ring
                .depth(Depth::Solid)
                .palette(Palette::AURORA)
                .thickness(0.3),
            Preset::Hologram => ring
                .depth(Depth::Solid)
                .palette(Palette::ICE)
                .thickness(0.26)
                .motion(Motion::Spin)
                .motion_tuned(0.5, 1.0),
            Preset::Heartbeat => ring
                .palette(Palette::EMBER)
                .thickness(0.2)
                .motion(Motion::Pulse),
            Preset::Cyber => ring
                .palette(Palette::NEON)
                .thickness(0.16)
                .glow(1.2)
                .motion(Motion::Rock),
            Preset::Buoy => ring
                .depth(Depth::Solid)
                .palette(Palette::OCEAN)
                .thickness(0.3)
                .motion(Motion::Float),
            Preset::Dribble => ring
                .depth(Depth::Solid)
                .palette(Palette::SUNSET)
                .size(72.0)
                .thickness(0.34)
                .motion(Motion::Bounce),
            Preset::Alert => bar.palette(Palette::SUNSET).size(10.0).motion(Motion::Buzz),
            Preset::Lightning => bar
                .palette(Palette::hex(0x7c3aed, 0x38bdf8, 0xe0f2fe))
                .size(12.0)
                .glow(1.0)
                .electric(),
            Preset::Tesla => ring
                .palette(Palette::hex(0x6366f1, 0x22d3ee, 0xf0f9ff))
                .thickness(0.18)
                .glow(1.0)
                .electric(),
            Preset::Inferno => bar.palette(Palette::FIRE).size(14.0).glow(0.8).fire(),
            Preset::FireRing => ring.palette(Palette::FIRE).thickness(0.2).glow(0.8).fire(),
            Preset::Aquarium => bar
                .palette(Palette::hex(0x0b4f8a, 0x1e90d6, 0x7fdcff))
                .size(22.0)
                .glow(0.5)
                .fill(Fill::Water),
            Preset::Nebula => ring
                .palette(Palette::hex(0x7c3aed, 0xec4899, 0x22d3ee))
                .thickness(0.3)
                .glow(0.9)
                .fill(Fill::Plasma),
            Preset::Glacier => bar
                .palette(Palette::ICE)
                .size(18.0)
                .glow(0.6)
                .fill(Fill::Frost),
            Preset::Charging => IndicatorStyle::new(Shape::Battery)
                .depth(Depth::Solid)
                .palette(Palette::BATTERY)
                .phases(0.2, 0.5)
                .size(32.0)
                .glow(0.6)
                .shimmer(true)
                .motion(Motion::Pulse)
                .motion_tuned(0.6, 0.25),
        }
    }
}

impl From<Preset> for IndicatorStyle {
    fn from(p: Preset) -> Self {
        p.style()
    }
}
