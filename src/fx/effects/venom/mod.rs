//! A liquid that hunts for the pointer: [`Venom`].
//!
//! Bring the pointer near and the surface swells toward it while pseudopods
//! ooze out and feel about in its direction ([`VenomState::Seeking`]). Within
//! [`VenomStyle::reach`] they lunge, each grabbing a grip of the
//! [`Cursor`](crate::fx::Cursor) — a finger of the glove — and coiling round
//! it ([`VenomState::Grabbing`], then [`VenomState::Holding`]), a pulse running
//! along them while a coat of the surface creeps over the hand. Past
//! [`VenomStyle::release`] they stretch, snap and slump back
//! ([`VenomState::Snapping`]); the body, a damped spring, wobbles and throws
//! up spikes and droplets.
//!
//! The strands are verlet ropes, heavily damped so they move like something
//! viscous, rooted on the shape's [`Outline`](crate::fx::Outline) and drawn as
//! chains of tapered capsules in the [`Surface`].

mod sim;

use sim::{Mode, Sim, NP};

use crate::fx::draw;
use crate::fx::surface::{Blob, Mound, Spike, Surface};
use crate::fx::{CursorDecor, Effect, EffectInput, Look};
use crate::math::Vector2;
use crate::overlay::Overlay;

/// Everything that can be changed about [`Venom`]. `Default` is the
/// original: black lacquer, seven strands.
#[derive(Debug, Clone, PartialEq)]
pub struct VenomStyle {
    pub look: Look,
    /// Thick strands, 2 to 7; with [`hairs`](Self::hairs) a fine hair runs
    /// between each pair.
    pub strands: usize,
    pub hairs: bool,
    /// Fine threads strung between neighbouring strands while it holds.
    pub webs: bool,
    /// Distance from the outline, in CSS pixels, at which it notices the
    /// pointer and starts reaching.
    pub sense: f32,
    /// Distance at which the strands lunge and grab.
    pub reach: f32,
    /// Distance at which they snap. Kept at least `reach`.
    pub release: f32,
    /// Strand thickness, as a multiple of the original's.
    pub strand_width: f32,
    /// Fraction of a strand point's velocity kept each frame: lower is
    /// thicker. The original's 0.92 moves like syrup.
    pub damping: f32,
    /// Pull on the strands, CSS pixels per second squared.
    pub gravity: f32,
    /// How far the surface swells toward the pointer, as a multiple.
    pub mound: f32,
    /// Spikes thrown up and droplets flung off when the strands snap.
    pub spikes: bool,
    pub droplets: bool,
    /// Deepest the round profile goes, in CSS pixels: a large shape gets a
    /// pillowed edge and a flat top instead of one giant dome.
    pub bevel: f32,
    /// A coat of the surface creeping over the cursor while it holds.
    pub coat: bool,
}

impl Default for VenomStyle {
    fn default() -> Self {
        Self {
            look: Look::LACQUER,
            strands: 7,
            hairs: true,
            webs: true,
            sense: 290.0,
            reach: 140.0,
            release: 215.0,
            strand_width: 1.0,
            damping: 0.92,
            gravity: 260.0,
            mound: 1.0,
            spikes: true,
            droplets: true,
            bevel: 32.0,
            coat: true,
        }
    }
}

impl VenomStyle {
    pub fn with_look(mut self, look: Look) -> Self {
        self.look = look;
        self
    }
    pub fn with_strands(mut self, strands: usize) -> Self {
        self.strands = strands;
        self
    }
    /// `sense`, `reach` and `release` together.
    pub fn with_distances(mut self, sense: f32, reach: f32, release: f32) -> Self {
        self.sense = sense;
        self.reach = reach;
        self.release = release.max(reach);
        self
    }
}

/// Where a [`Venom`] is in its cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VenomState {
    /// At rest; nothing in range.
    Idle,
    /// The pointer is within `sense`: the surface swells and feelers reach.
    Seeking,
    /// Within `reach`: the strands are lunging for the grips.
    Grabbing,
    /// Latched on; they follow the pointer until it passes `release`.
    Holding,
    /// Let go, and slumping back into the body.
    Snapping,
}

/// The effect. Add it to a [`Stage`](crate::fx::Stage) with any component, or
/// drive it yourself through [`Effect`].
pub struct Venom {
    pub style: VenomStyle,
    sim: Sim,
}

impl Venom {
    pub fn new(style: VenomStyle) -> Self {
        Self { style, sim: Sim::new() }
    }

    pub fn state(&self) -> VenomState {
        self.sim.state()
    }
}

impl Default for Venom {
    fn default() -> Self {
        Self::new(VenomStyle::default())
    }
}

impl Effect for Venom {
    fn update(&mut self, input: &EffectInput) {
        self.sim.cursor = input.cursor;
        self.sim.sync_shape(input.shape);
        self.sim.step(&self.style, input.dt.clamp(1.0 / 240.0, 1.0 / 30.0), input.pointer);
    }

    fn surface(&self) -> Option<Surface> {
        let sim = &self.sim;
        let style = &self.style;
        let holding = sim.holding();
        let seeking = sim.mode == Mode::Seek;
        let thin = sim.thin();
        let mut surface = Surface::new_with_thickness(sim.shape.clone(), style.look, sim.inradius.min(style.bevel));
        surface.lean = sim.lean;

        let grow = if seeking { 0.55 + 0.45 * sim.seek_f } else { 1.0 };
        for st in sim.list.iter().filter(|s| !s.hair) {
            let chain = (0..NP)
                .map(|i| {
                    let f = i as f32 / (NP - 1) as f32;
                    // thick where it leaves the body, a fine thread at the hand
                    let mut r = st.width * style.strand_width * (3.0 - 1.9 * f) * thin * grow;
                    // a slow pulse travels from the body toward the hand, as if it is feeding
                    if holding {
                        r *= 1.0 + 0.24 * (f * 9.0 - sim.time * 6.0 + st.phase).sin() * (f * std::f32::consts::PI).sin();
                    }
                    if sim.mode == Mode::Snap {
                        r *= 1.15; // gathering up as it slumps back
                    }
                    Blob { center: Vector2::new(st.x[i], st.y[i]), radius: r }
                })
                .collect();
            surface.strands.push(chain);
        }
        for sp in &sim.spikes {
            let p = sp.age / sp.life;
            let up = if p < 0.15 {
                1.0 - (1.0 - p / 0.15).powi(3)
            } else if p > 0.45 {
                (1.0 - (p - 0.45) / 0.55).max(0.0)
            } else {
                1.0
            };
            let len = sp.len * if p > 0.45 { up * up * (3.0 - 2.0 * up) } else { up };
            if len <= 0.6 {
                continue;
            }
            let b = sim.outline.at(sp.s);
            let sway = (sp.age * 14.0).sin() * (-sp.age * 3.0).exp() * 0.35;
            let ang = b.normal.y.atan2(b.normal.x) + sp.tilt + sway;
            surface.spikes.push(Spike {
                base: b.point - b.normal * 2.0,
                tip: b.point + Vector2::new(ang.cos(), ang.sin()) * len,
                radius: 2.6 * (up + 0.2).min(1.0),
            });
        }
        for d in &sim.drops {
            surface.drops.push(Blob { center: Vector2::new(d.x, d.y), radius: d.r });
        }
        let mlen = sim.mound.max(0.0);
        if mlen > 0.3 {
            let e = sim.outline.at(sim.mound_s);
            let r = (9.0 + sim.spread_px * 0.32) * (sim.mound / 6.0).min(1.0);
            surface.mound = Some(Mound {
                from: e.point - e.normal * 6.0,
                to: e.point + sim.mound_dir * mlen,
                base_radius: r,
                tip_radius: r * 0.45,
            });
        }
        Some(surface)
    }

    /// The fine hairs and web threads, over the element's content.
    fn paint(&self, o: &mut Overlay, dpr: f32) {
        let sim = &self.sim;
        if sim.list.is_empty() || !(sim.holding() || sim.mode == Mode::Snap) {
            return;
        }
        let px = |x: f32, y: f32| Vector2::new(x * dpr, y * dpr);
        o.no_fill();
        o.stroke_alpha(self.style.look.base_color(0.8, 0.0), 0.9);
        for st in sim.list.iter().filter(|s| s.hair) {
            let pts: Vec<Vector2> = smooth_strand(st.x, st.y).iter().map(|p| px(p.x, p.y)).collect();
            draw::stroke(o, &pts, 0.35 * sim.thin() * self.style.strand_width * dpr);
        }
        if sim.mode == Mode::Hold {
            o.stroke_alpha(self.style.look.base_color(0.8, 0.0), 0.6);
            for &(a, b, i, j, kink) in &sim.webs {
                let (Some(sa), Some(sb)) = (sim.list.get(a), sim.list.get(b)) else { continue };
                let mid = Vector2::new((sa.x[i] + sb.x[j]) / 2.0 + kink, (sa.y[i] + sb.y[j]) / 2.0 - kink.abs() * 0.4);
                draw::stroke(o, &[px(sa.x[i], sa.y[i]), mid * dpr, px(sb.x[j], sb.y[j])], 0.25 * dpr);
            }
        }
    }

    fn decorate_cursor(&self, decor: &mut CursorDecor) {
        for (g, &c) in self.sim.coat.iter().enumerate() {
            if c > 0.0 {
                decor.coat(g, c, self.style.look);
            }
        }
    }

    /// The last stretch of each strand lies over the cursor, as one taper.
    fn paint_over_cursor(&self, o: &mut Overlay, dpr: f32) {
        let sim = &self.sim;
        if !sim.holding() {
            return;
        }
        o.no_stroke();
        for st in &sim.list {
            let gap = (Vector2::new(st.ex, st.ey) - sim.grip_at(st.grip)).length();
            let on = (1.0 - gap / 14.0).clamp(0.0, 1.0);
            if on <= 0.0 {
                continue;
            }
            let from = NP - 4;
            let pts: Vec<Vector2> = (from..NP).map(|i| Vector2::new(st.x[i], st.y[i]) * dpr).collect();
            let radii: Vec<f32> = (from..NP)
                .map(|i| {
                    let f = i as f32 / (NP - 1) as f32;
                    let w = (st.width * self.style.strand_width * (3.0 - 1.9 * f) * sim.thin() * 2.0).max(0.6);
                    w * if st.hair { 0.45 } else { 1.0 } * 0.5 * dpr
                })
                .collect();
            o.stroke_alpha(self.style.look.base_color(0.8, 0.0), on);
            draw::taper(o, &pts, &radii);
        }
    }
}

/// The polyline a canvas draws for `moveTo(p0)`, `quadraticCurveTo` through
/// the midpoints, `lineTo(last)` — sampled.
fn smooth_strand(x: [f32; NP], y: [f32; NP]) -> Vec<Vector2> {
    let mut out = vec![Vector2::new(x[0], y[0])];
    let mut from = out[0];
    for i in 1..NP - 1 {
        let c = Vector2::new(x[i], y[i]);
        let to = Vector2::new((x[i] + x[i + 1]) / 2.0, (y[i] + y[i + 1]) / 2.0);
        for k in 1..=4 {
            let t = k as f32 / 4.0;
            out.push(from * ((1.0 - t) * (1.0 - t)) + c * (2.0 * t * (1.0 - t)) + to * (t * t));
        }
        from = to;
    }
    out.push(Vector2::new(x[NP - 1], y[NP - 1]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx::{Cursor, Shape};

    #[test]
    fn it_reaches_grabs_holds_and_lets_go() {
        let shape = Shape::pill(Vector2::new(400.0, 300.0), 240.0, 64.0);
        let mut venom = Venom::default();
        let dt = 1.0 / 60.0;
        let run = |venom: &mut Venom, p: Option<Vector2>, secs: f32| {
            for i in 0..(secs / dt) as usize {
                let input = EffectInput { shape: &shape, dt, time: i as f32 * dt, pointer: p, cursor: Cursor::default(), hovered: false, pressed: false };
                venom.update(&input);
            }
        };
        run(&mut venom, None, 0.5);
        assert_eq!(venom.state(), VenomState::Idle);
        run(&mut venom, Some(Vector2::new(560.0, 30.0)), 0.5);
        assert_eq!(venom.state(), VenomState::Seeking);
        run(&mut venom, Some(Vector2::new(520.0, 200.0)), 1.0);
        assert_eq!(venom.state(), VenomState::Holding);
        let surface = venom.surface().unwrap();
        assert_eq!(surface.strands.len(), 7);
        assert!(surface.mound.is_some());
        let mut decor = CursorDecor::default();
        venom.decorate_cursor(&mut decor);
        assert!(decor.coats.iter().any(|c| c.is_some()), "holding coats the glove");
        run(&mut venom, Some(Vector2::new(900.0, 0.0)), 0.1);
        assert_eq!(venom.state(), VenomState::Snapping);
        assert!(!venom.surface().unwrap().spikes.is_empty());
        run(&mut venom, None, 3.0);
        assert_eq!(venom.state(), VenomState::Idle);
    }
}
