//! What stands in for the pointer: [`Cursor`], and what effects may leave on
//! it ([`CursorDecor`]).

use super::draw::bounds;
use super::Look;
use crate::curves::parse_svg_subpaths;
use crate::math::{Color, Vector2};
use crate::overlay::Overlay;
use crate::sdf::{self, d2};

/// The pointer as the stage draws it, and the points on it an effect can
/// grab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cursor {
    /// A cartoon glove with its fingertip on the pointer, `scale` CSS pixels
    /// per unit of its 32 x 36 drawing. Its grips are the index tip, the three
    /// knuckles, the thumb, the index joint, the palm and the cuff. The system
    /// cursor is hidden over the stage.
    Glove { scale: f32 },
    /// Nothing drawn and the system cursor kept; the grips cluster round the
    /// hotspot.
    Hotspot,
}

impl Default for Cursor {
    fn default() -> Self {
        Cursor::Glove { scale: 1.55 }
    }
}

/// The glove's fingertip, in its own units.
const HOT: Vector2 = Vector2::new(11.5, 1.0);
/// The glove's grips, in its own units.
const GLOVE_GRIPS: [[f32; 2]; Cursor::GRIPS] = [
    [11.5, 3.6], [16.2, 13.2], [20.6, 14.0], [24.9, 15.8], [5.6, 18.0], [11.5, 10.0], [18.5, 22.0], [18.7, 31.6],
];
/// The hotspot's, in CSS pixels from it.
const HOTSPOT_GRIPS: [[f32; 2]; Cursor::GRIPS] = [
    [0.0, 0.0], [5.0, 7.0], [-4.0, 6.0], [8.0, 13.0], [-2.0, 12.0], [3.0, 3.0], [6.0, 18.0], [0.0, 16.0],
];

impl Cursor {
    /// How many grips every cursor has.
    pub const GRIPS: usize = 8;

    /// Grip `i` (mod [`GRIPS`](Self::GRIPS)) with the pointer at `hotspot`.
    pub fn grip(&self, hotspot: Vector2, i: usize) -> Vector2 {
        let i = i % Self::GRIPS;
        match self {
            Cursor::Glove { scale } => {
                let g = GLOVE_GRIPS[i];
                hotspot + (Vector2::new(g[0], g[1]) - HOT) * *scale
            }
            Cursor::Hotspot => hotspot + Vector2::new(HOTSPOT_GRIPS[i][0], HOTSPOT_GRIPS[i][1]),
        }
    }

    /// The middle of what is drawn — where distances to the cursor are
    /// measured from, so a big hand counts as near when its palm is.
    pub fn center(&self, hotspot: Vector2) -> Vector2 {
        match self {
            Cursor::Glove { scale } => hotspot + (Vector2::new(16.0, 18.0) - HOT) * *scale,
            Cursor::Hotspot => hotspot,
        }
    }

    /// Whether the system cursor should be hidden while this one is drawn.
    pub fn hides_system_cursor(&self) -> bool {
        matches!(self, Cursor::Glove { .. })
    }

    /// Draw the cursor at `hotspot` with whatever effects left on it.
    pub fn paint(&self, o: &mut Overlay, hotspot: Vector2, decor: &CursorDecor, time: f32, dpr: f32) {
        if let Cursor::Glove { scale } = self {
            paint_glove(o, hotspot, *scale, decor, time, dpr);
        }
    }
}

/// What effects leave on the cursor this frame: for each grip, how far a
/// coat of their surface has crept over it (0 to 1) and in what look. Where
/// two effects coat the same grip, the thicker coat shows.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CursorDecor {
    pub coats: [Option<(f32, Look)>; Cursor::GRIPS],
}

impl CursorDecor {
    /// Coat grip `grip` to `amount` in `look`, unless something already
    /// coats it more.
    pub fn coat(&mut self, grip: usize, amount: f32, look: Look) {
        let slot = &mut self.coats[grip % Cursor::GRIPS];
        if amount > slot.map_or(0.0, |c| c.0) {
            *slot = Some((amount.min(1.0), look));
        }
    }
}

// The glove, in a 32 x 36 box.
const GLOVE: &str = "M9 14V3.6a2.5 2.5 0 0 1 5 0V12.2a2.2 2.2 0 0 1 4.4 0.4a2.2 2.2 0 0 1 4.4 0.8a2.2 2.2 0 0 1 4.2 1.6V22c0 4-2.6 7-6 7H14.6c-3 0-5.4-1.8-6.6-4.4L4.1 18.6a2.3 2.3 0 0 1 3.6-2.8L9 17.4Z";
const GLOVE_LINES: &str = "M18.4 12.6V17.2M22.8 13.4V17.6M14 12.2V16.6";
const CUFF: &str = "M12.6 28.6h12.2a1.6 1.6 0 0 1 1.6 1.6v3.4a1.6 1.6 0 0 1-1.6 1.6H12.6a1.6 1.6 0 0 1-1.6-1.6v-3.4a1.6 1.6 0 0 1 1.6-1.6Z";
const CUFF_LINES: &str = "M15.4 29.2v5.4M18.7 29.2v5.4M22 29.2v5.4";

/// The glove's paths, flattened once, in glove units.
struct GloveArt {
    glove: Vec<Vector2>,
    cuff: Vec<Vector2>,
    lines: Vec<Vec<Vector2>>,
}

fn flatten(d: &str) -> Vec<Vec<Vector2>> {
    parse_svg_subpaths(d)
        .unwrap_or_default()
        .iter()
        .map(|path| {
            let mut out: Vec<Vector2> = Vec::new();
            for c in &path.curve_path.curves {
                for p in c.get_points(10) {
                    if out.last().is_none_or(|q| (*q - p).length() > 0.05) {
                        out.push(p);
                    }
                }
            }
            out
        })
        .collect()
}

fn art() -> &'static GloveArt {
    static ART: std::sync::OnceLock<GloveArt> = std::sync::OnceLock::new();
    ART.get_or_init(|| {
        let mut lines = flatten(CUFF_LINES);
        lines.extend(flatten(GLOVE_LINES));
        GloveArt {
            glove: flatten(GLOVE).into_iter().next().unwrap_or_default(),
            cuff: flatten(CUFF).into_iter().next().unwrap_or_default(),
            lines,
        }
    })
}

fn paint_glove(o: &mut Overlay, hotspot: Vector2, sc: f32, decor: &CursorDecor, time: f32, dpr: f32) {
    let art = art();
    let to_px = |g: Vector2, dx: f32, dy: f32| (hotspot + (g + Vector2::new(dx, dy) - HOT) * sc) * dpr;
    let place = |pts: &[Vector2], dx: f32, dy: f32| pts.iter().map(|&g| to_px(g, dx, dy)).collect::<Vec<_>>();
    o.no_stroke();
    // a soft drop shadow, then the white glove and cuff
    o.fill_alpha(Color::from_hex(0x000000), 0.15);
    o.polygon(&place(&art.cuff, 2.5, 4.5));
    o.polygon(&place(&art.glove, 2.5, 4.5));
    let glove = place(&art.glove, 0.0, 0.0);
    let cuff = place(&art.cuff, 0.0, 0.0);
    o.fill(Color::from_hex(0xffffff));
    o.polygon(&cuff);
    o.polygon(&glove);

    // a coat spreading out from each grip: a wobbling blob intersected with
    // the hand, which is the clip a canvas would use
    let hand = |p: Vector2| sdf::union(d2::polygon(p, &glove), d2::polygon(p, &cuff));
    for (g, coat) in decor.coats.iter().enumerate() {
        let Some((c, look)) = *coat else { continue };
        if c < 0.01 {
            continue;
        }
        let e = c * c * (3.0 - 2.0 * c);
        let [gx0, gy0] = GLOVE_GRIPS[g];
        // it creeps from the grip toward the middle of the hand as it spreads
        let centre = Vector2::new(gx0 + (17.0 - gx0) * e * 0.35, gy0 + (20.0 - gy0) * e * 0.35);
        let r = 2.4 + e * 7.5;
        let blob = |scale: f32, shift: Vector2| -> Vec<Vector2> {
            (0..28)
                .map(|k| {
                    let a = k as f32 / 28.0 * std::f32::consts::TAU;
                    let rr = r
                        * scale
                        * (1.0 + 0.17 * (4.0 * a + g as f32 * 1.7 + time * 2.2).sin()
                            + 0.08 * (7.0 * a - time * 3.1 + g as f32).sin());
                    to_px(centre + shift + Vector2::new(a.cos() * rr, a.sin() * rr), 0.0, 0.0)
                })
                .collect()
        };
        // a radial gradient from a highlight up and to the left, in three steps
        let highlight = Vector2::new(-0.35 * r, -0.4 * r);
        let layers = [
            (1.0, Vector2::ZERO, look.base_color(0.7, 0.0)),
            (0.45, highlight * 0.6, look.base_color(1.0, 0.05)),
            (0.18, highlight * 0.9, look.tint_color(0.25, 0.0)),
        ];
        for (scale, shift, colour) in layers {
            let shape = blob(scale, shift);
            let (x0, y0, x1, y1) = bounds(&shape);
            o.fill(colour);
            o.distance_field(x0, y0, x1, y1, |p| sdf::intersection(d2::polygon(p, &shape), hand(p)));
        }
    }

    // ink outlines
    o.no_fill();
    o.stroke(Color::from_hex(0x0b0b0b));
    o.stroke_weight(1.9 * sc * dpr);
    o.polygon(&cuff);
    o.polygon(&glove);
    o.stroke_weight(1.1 * sc * dpr);
    for line in &art.lines {
        o.polyline(&place(line, 0.0, 0.0));
    }
    o.no_stroke();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grips_sit_on_the_glove() {
        let c = Cursor::Glove { scale: 2.0 };
        let hot = Vector2::new(100.0, 100.0);
        // the index tip grip is just below the hotspot
        let tip = c.grip(hot, 0);
        assert!((tip.x - 100.0).abs() < 1e-4 && (tip.y - 105.2).abs() < 1e-4);
        assert!(c.center(hot).y > tip.y);
        assert_eq!(Cursor::Hotspot.grip(hot, 0), hot);
        let mut d = CursorDecor::default();
        d.coat(3, 0.5, Look::GOLD);
        d.coat(3, 0.2, Look::CHROME);
        assert_eq!(d.coats[3], Some((0.5, Look::GOLD)));
    }
}
