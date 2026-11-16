//! G-code toolpaths, read the way three.js r165's
//! `examples/jsm/loaders/GCodeLoader.js` draws them: each move a line
//! segment, green where the extruder pushes filament and red where the head
//! only travels, split into layers at every new Z that extrudes, and the
//! whole group turned −90° about X so the printer's Z-up bed lies in
//! three.js's Y-up world.
//!
//! Where three.js stops, this keeps going:
//!
//! - **Arcs.** `G2`/`G3` in all three planes (`G17`–`G19`), by centre offset
//!   (`I` `J` `K`) or radius (`R`), helical when the third axis moves too,
//!   tessellated to [`GcodeLoader::arc_tolerance`]. three.js ignores them.
//! - **Modes that last.** three.js forgets `G91` after the first move (its
//!   state object is replaced by the move's); here `G90`/`G91` stay until
//!   changed, `M82`/`M83` set the extruder alone, and `G20` reads inches
//!   (converted to millimetres).
//! - **`G92` as an offset.** Setting a position moves the coordinate frame,
//!   not the head, so the drawing does not jump; `G28` homes and clears it.
//! - **Extruding is per move.** A move extrudes when it advances E. three.js
//!   keeps the flag from an earlier move in some orders and drops it in
//!   others.
//! - **Machines without E.** A file whose moves never mention E (CNC, laser,
//!   pen plotter) counts feed moves (`G1`–`G3`) as work and `G0` as travel;
//!   after an `M5` turns the spindle or laser off, feed moves travel until an
//!   `M3`/`M4` turns it on again.
//! - **Real-world syntax.** Words need no spaces between them (`G1X10Y5`);
//!   `;` and `( … )` comments, `N` line numbers, `*` checksums and `%` tape
//!   marks are dropped; a line of bare coordinates repeats the last motion
//!   command, as CNC controllers do.
//!
//! Zero-length moves — a retraction in place — draw nothing. Commands that
//! do not move the head (temperatures, fans, tool changes) are counted in
//! [`GcodeToolpath::skipped`].
//!
//! ```
//! use threers::loaders::GcodeLoader;
//!
//! let path = GcodeLoader::default().parse(
//!     "G21 G90 M82\n\
//!      G1 Z0.2 F3000\n\
//!      G1 X10 Y0 E1 ; first line\n\
//!      G0 X10 Y10\n",
//! );
//! // Layer 0 holds the travel up to Z 0.2, as in three.js.
//! assert_eq!(path.layers.len(), 2);
//! assert_eq!(path.layers[1].z, 0.2);
//! assert_eq!(path.layers[1].extruded.len(), 6); // one segment
//! ```

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::core::{BufferAttribute, BufferGeometry, LineSegments, Object3D, ObjectId};
use crate::materials::{LineBasicMaterial, Material};
use crate::math::{Color, Quaternion};
use crate::Scene;

/// three.js's `pathMaterial` colour: travel moves.
pub const GCODE_TRAVEL_COLOR: u32 = 0xff0000;
/// three.js's `extrudingMaterial` colour: extruding (or cutting) moves.
pub const GCODE_EXTRUDED_COLOR: u32 = 0x00ff00;

/// Reads G-code into a [`GcodeToolpath`]. `GCodeLoader` in three.js.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcodeLoader {
    /// three.js `splitLayer`: [`GcodeToolpath::add_to_scene`] makes two
    /// objects per layer instead of two for the whole file.
    pub split_layer: bool,
    /// The most an arc's chords may stray from the arc, in millimetres.
    pub arc_tolerance: f64,
}

impl Default for GcodeLoader {
    fn default() -> Self {
        Self {
            split_layer: false,
            arc_tolerance: 0.01,
        }
    }
}

/// One layer: the segments drawn while the head was at it. Positions are
/// millimetres in the machine frame (Z up), two vertices per segment, as
/// three.js's `layer.vertex` and `layer.pathVertex` hold them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GcodeLayer {
    /// The Z of the first extruding move that opened the layer (or of the
    /// first move at all, for travel before anything extrudes).
    pub z: f64,
    /// Extruding (or cutting) segments.
    pub extruded: Vec<f32>,
    /// Travel segments.
    pub travel: Vec<f32>,
}

/// What [`GcodeLoader::parse`] found.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GcodeToolpath {
    pub layers: Vec<GcodeLayer>,
    /// Whether any move mentioned E. Without it, extruded means a feed move
    /// with the spindle or laser on (see the module docs).
    pub has_extrusion: bool,
    /// Commands that do not move the head, by name (`M104`, `T1`, …), and how
    /// many times each came; also `"malformed"` for lines with a word that
    /// did not parse, which are skipped whole.
    pub skipped: BTreeMap<String, usize>,
    /// [`GcodeLoader::split_layer`], carried to [`add_to_scene`](Self::add_to_scene).
    pub split_layer: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Motion {
    Rapid,
    Linear,
    Clockwise,
    CounterClockwise,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Plane {
    Xy,
    Zx,
    Yz,
}

impl Plane {
    /// The two axes the arc turns in, in the order that makes `G2`
    /// clockwise seen from the positive third axis, then that third axis.
    fn axes(self) -> (usize, usize, usize) {
        match self {
            Plane::Xy => (0, 1, 2),
            Plane::Zx => (2, 0, 1),
            Plane::Yz => (1, 2, 0),
        }
    }
}

/// One line's words, letter upper-cased, in order.
type Words = Vec<(char, f64)>;

/// Splits a line into words, dropping comments, line numbers and checksums.
/// `None` when a word's number does not parse.
fn words(line: &str) -> Option<Words> {
    let mut out = Words::new();
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        match c {
            b';' => break,
            b'*' => break,
            b'(' => {
                while i < bytes.len() && bytes[i] != b')' {
                    i += 1;
                }
                i += 1;
            }
            b'%' => i += 1,
            c if c.is_ascii_whitespace() => i += 1,
            c if c.is_ascii_alphabetic() => {
                let letter = (c as char).to_ascii_uppercase();
                i += 1;
                while i < bytes.len() && bytes[i] == b' ' {
                    i += 1;
                }
                let start = i;
                while i < bytes.len()
                    && (bytes[i].is_ascii_digit() || matches!(bytes[i], b'.' | b'-' | b'+'))
                {
                    i += 1;
                }
                let text = &line[start..i];
                // A bare letter (`G28 X`) names an axis without a value.
                let value = if text.is_empty() {
                    f64::NAN
                } else {
                    text.parse().ok()?
                };
                if letter != 'N' {
                    out.push((letter, value));
                }
                // Marlin's messages and file names: the rest is free text.
                if letter == 'M' && [23.0, 28.0, 30.0, 32.0, 117.0, 118.0, 928.0].contains(&value) {
                    break;
                }
            }
            _ => return None,
        }
    }
    Some(out)
}

/// A G or M code as written, `G1` for `G01`, `G91.1` for `G91.1`.
fn code_name(letter: char, v: f64) -> String {
    if v.fract() == 0.0 {
        format!("{letter}{}", v as i64)
    } else {
        format!("{letter}{v}")
    }
}

struct Machine {
    /// Head position in machine millimetres; `pos[3]` is E.
    pos: [f64; 4],
    /// `G92`: logical = machine − offset.
    offset: [f64; 4],
    relative: bool,
    relative_e: bool,
    /// Millimetres per unit: 1, or 25.4 under `G20`.
    scale: f64,
    plane: Plane,
    motion: Option<Motion>,
    /// `M3`/`M4` on, `M5` off, `None` before either.
    spindle: Option<bool>,
}

struct Builder {
    layers: Vec<GcodeLayer>,
}

impl Builder {
    fn segment(&mut self, a: [f64; 3], b: [f64; 3], extruding: bool) {
        if a == b {
            return;
        }
        if self.layers.is_empty() {
            self.layers.push(GcodeLayer {
                z: a[2],
                ..Default::default()
            });
        }
        let layer = self.layers.last_mut().unwrap();
        let v = if extruding {
            &mut layer.extruded
        } else {
            &mut layer.travel
        };
        v.extend(a.iter().chain(&b).map(|&x| x as f32));
    }
}

impl GcodeLoader {
    /// Parse G-code text. Never fails: what it cannot read is counted in
    /// [`GcodeToolpath::skipped`].
    pub fn parse(&self, src: &str) -> GcodeToolpath {
        let lines: Vec<Option<Words>> = src.lines().map(words).collect();
        let has_extrusion = lines.iter().flatten().any(|w| {
            w.iter().any(|&(l, _)| l == 'E')
                && w.iter().any(|&(l, v)| l == 'G' && (0.0..=3.0).contains(&v))
        });
        let mut m = Machine {
            pos: [0.0; 4],
            offset: [0.0; 4],
            relative: false,
            relative_e: false,
            scale: 1.0,
            plane: Plane::Xy,
            motion: None,
            spindle: None,
        };
        let mut b = Builder { layers: Vec::new() };
        let mut skipped = BTreeMap::new();
        for line in &lines {
            let Some(w) = line else {
                *skipped.entry("malformed".to_string()).or_insert(0) += 1;
                continue;
            };
            self.line(w, &mut m, &mut b, has_extrusion, &mut skipped);
        }
        GcodeToolpath {
            layers: b.layers,
            has_extrusion,
            skipped,
            split_layer: self.split_layer,
        }
    }

    fn line(
        &self,
        w: &Words,
        m: &mut Machine,
        b: &mut Builder,
        has_extrusion: bool,
        skipped: &mut BTreeMap<String, usize>,
    ) {
        let get = |letter: char| {
            w.iter()
                .find(|&&(l, v)| l == letter && !v.is_nan())
                .map(|&(_, v)| v)
        };
        let mut motion = None;
        let mut set_position = false;
        let mut home = false;
        for &(letter, v) in w {
            match letter {
                'G' => match v {
                    0.0 => motion = Some(Motion::Rapid),
                    1.0 => motion = Some(Motion::Linear),
                    2.0 => motion = Some(Motion::Clockwise),
                    3.0 => motion = Some(Motion::CounterClockwise),
                    17.0 => m.plane = Plane::Xy,
                    18.0 => m.plane = Plane::Zx,
                    19.0 => m.plane = Plane::Yz,
                    20.0 => m.scale = 25.4,
                    21.0 => m.scale = 1.0,
                    28.0 => home = true,
                    // Marlin's G90/G91 set the extruder too; M82/M83 after
                    // them set it alone.
                    90.0 => (m.relative, m.relative_e) = (false, false),
                    91.0 => (m.relative, m.relative_e) = (true, true),
                    92.0 => set_position = true,
                    // Arc centres are always incremental here.
                    91.1 => {}
                    _ => *skipped.entry(code_name('G', v)).or_insert(0) += 1,
                },
                'M' => match v {
                    3.0 | 4.0 => m.spindle = Some(true),
                    5.0 => m.spindle = Some(false),
                    82.0 => m.relative_e = false,
                    83.0 => m.relative_e = true,
                    _ => *skipped.entry(code_name('M', v)).or_insert(0) += 1,
                },
                'T' => *skipped.entry(code_name('T', v)).or_insert(0) += 1,
                _ => {}
            }
        }
        let axes = ['X', 'Y', 'Z', 'E'];
        if set_position {
            // G92 with no axes zeroes them all.
            let any = axes.iter().any(|&a| get(a).is_some());
            for (i, &a) in axes.iter().enumerate() {
                if let Some(v) = get(a).or((!any).then_some(0.0)) {
                    m.offset[i] = m.pos[i] - v * m.scale;
                }
            }
            return;
        }
        if home {
            // A bare `G28` homes everything; `G28 X Y` (bare letters) those.
            let named: Vec<usize> = (0..3)
                .filter(|&i| w.iter().any(|&(l, _)| l == axes[i]))
                .collect();
            let homed = if named.is_empty() {
                vec![0, 1, 2]
            } else {
                named
            };
            let from = m.pos;
            for i in homed {
                m.pos[i] = 0.0;
                m.offset[i] = 0.0;
            }
            b.segment(head(from), head(m.pos), false);
            return;
        }
        if motion.is_some() {
            m.motion = motion;
        }
        // Bare coordinates repeat the last motion; coordinates after another
        // command (`M203 E25`, `G4 P5`) are that command's parameters.
        let moves = (0..4).any(|i| get(axes[i]).is_some())
            && !w.iter().any(|&(l, _)| matches!(l, 'G' | 'M' | 'T'));
        let Some(motion) = (if motion.is_some() || moves {
            m.motion
        } else {
            None
        }) else {
            return;
        };
        let mut target = m.pos;
        for i in 0..4 {
            if let Some(v) = get(axes[i]) {
                let relative = if i == 3 { m.relative_e } else { m.relative };
                target[i] = if relative {
                    m.pos[i] + v * m.scale
                } else {
                    v * m.scale + m.offset[i]
                };
            }
        }
        let extruding = if has_extrusion {
            target[3] > m.pos[3]
        } else {
            motion != Motion::Rapid && m.spindle != Some(false)
        };
        // three.js opens a layer at each extruding move whose Z differs
        // from the current layer's.
        if extruding && b.layers.last().is_none_or(|l| l.z != target[2]) {
            b.layers.push(GcodeLayer {
                z: target[2],
                ..Default::default()
            });
        }
        match motion {
            Motion::Rapid | Motion::Linear => b.segment(head(m.pos), head(target), extruding),
            Motion::Clockwise | Motion::CounterClockwise => {
                let offsets = [get('I'), get('J'), get('K')].map(|o| o.map(|v| v * m.scale));
                let radius = get('R').map(|r| r * m.scale);
                let cw = motion == Motion::Clockwise;
                for (a, c) in arc(
                    m.pos,
                    target,
                    m.plane,
                    cw,
                    offsets,
                    radius,
                    self.arc_tolerance,
                ) {
                    b.segment(a, c, extruding);
                }
            }
        }
        m.pos = target;
    }
}

fn head(p: [f64; 4]) -> [f64; 3] {
    [p[0], p[1], p[2]]
}

/// The chords of an arc from `from` to `to`, `None`-free: an arc whose
/// centre cannot be placed comes back as the straight move.
fn arc(
    from: [f64; 4],
    to: [f64; 4],
    plane: Plane,
    clockwise: bool,
    offsets: [Option<f64>; 3],
    radius: Option<f64>,
    tolerance: f64,
) -> Vec<([f64; 3], [f64; 3])> {
    let (p, q, l) = plane.axes();
    let (p1, q1, p2, q2) = (from[p], from[q], to[p], to[q]);
    let center = if let Some(r) = radius {
        // Marlin's construction: the centre on the perpendicular bisector,
        // left or right of the chord by direction and the sign of R.
        let (dx, dy) = (p2 - p1, q2 - q1);
        let d = dx.hypot(dy);
        if d == 0.0 || r == 0.0 {
            None
        } else {
            let h = (r * r - d * d * 0.25).max(0.0).sqrt();
            let e = if clockwise ^ (r < 0.0) { -1.0 } else { 1.0 };
            let (mx, my) = ((p1 + p2) * 0.5, (q1 + q2) * 0.5);
            Some((mx + e * h * (-dy / d), my + e * h * (dx / d)))
        }
    } else {
        let o = |axis: usize| offsets[axis].unwrap_or(0.0);
        let (op, oq) = (o(p), o(q));
        (op != 0.0 || oq != 0.0).then_some((p1 + op, q1 + oq))
    };
    let Some((cp, cq)) = center else {
        return vec![(head(from), head(to))];
    };
    let r0 = (p1 - cp).hypot(q1 - cq);
    let r1 = (p2 - cp).hypot(q2 - cq);
    let a0 = (q1 - cq).atan2(p1 - cp);
    let a1 = (q2 - cq).atan2(p2 - cp);
    let tau = std::f64::consts::TAU;
    let mut sweep = a1 - a0;
    // Endpoints that meet make a full circle (the R form never does).
    let closed = (p1 - p2).hypot(q1 - q2) <= 1e-9 * r0.max(1.0);
    if clockwise {
        if sweep >= 0.0 || closed {
            sweep -= tau;
        }
    } else if sweep <= 0.0 || closed {
        sweep += tau;
    }
    let rmax = r0.max(r1);
    let step = if rmax > tolerance {
        2.0 * (1.0 - tolerance / rmax).acos()
    } else {
        tau / 4.0
    };
    let n = ((sweep.abs() / step).ceil() as usize).clamp(1, 4096);
    let point = |t: f64| -> [f64; 3] {
        if t == 1.0 {
            return head(to);
        }
        let a = a0 + sweep * t;
        let r = r0 + (r1 - r0) * t;
        let mut v = [0.0; 3];
        v[p] = cp + r * a.cos();
        v[q] = cq + r * a.sin();
        v[l] = from[l] + (to[l] - from[l]) * t;
        v
    };
    let mut out = Vec::with_capacity(n);
    let mut prev = head(from);
    for i in 1..=n {
        let next = point(i as f64 / n as f64);
        out.push((prev, next));
        prev = next;
    }
    out
}

impl GcodeToolpath {
    /// Every extruding segment, all layers in order.
    pub fn extruded(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|l| l.extruded.iter().copied())
            .collect()
    }

    /// Every travel segment, all layers in order.
    pub fn travel(&self) -> Vec<f32> {
        self.layers
            .iter()
            .flat_map(|l| l.travel.iter().copied())
            .collect()
    }

    /// The group three.js's `GCodeLoader.parse` returns: named `gcode`,
    /// rotated −90° about X, holding `LineSegments` named `layer{i}` —
    /// extruded (green) then travel (red) for each layer under
    /// [`split_layer`](Self::split_layer), else one pair named
    /// `layer{layer count}` for the whole file.
    pub fn to_objects(&self) -> (Object3D, Vec<Object3D>) {
        let extruded = Arc::new(Material::Line(LineBasicMaterial::new(Color::from_hex(
            GCODE_EXTRUDED_COLOR,
        ))));
        let travel = Arc::new(Material::Line(LineBasicMaterial::new(Color::from_hex(
            GCODE_TRAVEL_COLOR,
        ))));
        let object = |vertices: Vec<f32>, material: &Arc<Material>, i: usize| {
            let mut g = BufferGeometry::new();
            g.set_attribute("position", BufferAttribute::new(vertices, 3));
            let mut o =
                Object3D::line_segments(LineSegments::from_arc(Arc::new(g), material.clone()));
            o.name = format!("layer{i}");
            o
        };
        let mut children = Vec::new();
        if self.split_layer {
            for (i, layer) in self.layers.iter().enumerate() {
                children.push(object(layer.extruded.clone(), &extruded, i));
                children.push(object(layer.travel.clone(), &travel, i));
            }
        } else {
            let n = self.layers.len();
            children.push(object(self.extruded(), &extruded, n));
            children.push(object(self.travel(), &travel, n));
        }
        let mut group = Object3D::group();
        group.name = "gcode".into();
        group.quaternion = Quaternion::from_euler_xyz(-std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        (group, children)
    }

    /// Add [`to_objects`](Self::to_objects) to `scene`; returns the group.
    pub fn add_to_scene(&self, scene: &mut Scene) -> ObjectId {
        let (group, children) = self.to_objects();
        let gid = scene.add(group);
        for c in children {
            let id = scene.arena.insert(c);
            scene.arena.add_child(gid, id);
        }
        gid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> GcodeToolpath {
        GcodeLoader::default().parse(src)
    }

    fn segments(v: &[f32]) -> Vec<[[f32; 3]; 2]> {
        v.chunks(6)
            .map(|c| [[c[0], c[1], c[2]], [c[3], c[4], c[5]]])
            .collect()
    }

    #[test]
    fn three_js_layers_and_colours() {
        let p = parse(
            "G1 Z0.2\n\
             G1 X10 E1\n\
             G0 X0 Y10\n\
             G1 Z0.4\n\
             G1 X10 E2\n",
        );
        // Travel before anything extrudes opens a layer at its own Z.
        assert_eq!(p.layers.len(), 3);
        assert_eq!(p.layers[0].z, 0.0);
        assert_eq!(
            segments(&p.layers[0].travel),
            vec![[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]]]
        );
        assert!(p.layers[0].extruded.is_empty());
        assert_eq!(p.layers[1].z, 0.2);
        assert_eq!(
            segments(&p.layers[1].travel),
            vec![
                [[10.0, 0.0, 0.2], [0.0, 10.0, 0.2]],
                [[0.0, 10.0, 0.2], [0.0, 10.0, 0.4]]
            ]
        );
        assert_eq!(
            segments(&p.layers[1].extruded),
            vec![[[0.0, 0.0, 0.2], [10.0, 0.0, 0.2]]]
        );
        assert_eq!(p.layers[2].z, 0.4);
        assert_eq!(
            segments(&p.layers[2].extruded),
            vec![[[0.0, 10.0, 0.4], [10.0, 10.0, 0.4]]]
        );
        assert!(p.has_extrusion);
    }

    #[test]
    fn relative_modes_persist() {
        // three.js loses G91 after the first move; it holds here.
        let p = parse("G91\nG1 X1 E1\nG1 X1 E1\nM82\nG1 X1 E3\nG1 X1 E3\n");
        let e = segments(&p.extruded());
        assert_eq!(e.len(), 3);
        assert_eq!(e[2], [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]);
        assert_eq!(
            segments(&p.travel()),
            vec![[[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]]
        );
    }

    #[test]
    fn g92_offsets_the_frame_without_moving() {
        let p = parse("G1 X10 E1\nG92 X0 E0\nG1 X5 E1\n");
        let e = segments(&p.extruded());
        assert_eq!(e[1], [[10.0, 0.0, 0.0], [15.0, 0.0, 0.0]]);
        // G92 E0 then extruding from 0 still counts.
        assert_eq!(e.len(), 2);
    }

    #[test]
    fn retraction_in_place_draws_nothing() {
        let p = parse("G1 X1 E1\nG1 E0.2\nG1 E1\nG0 X2\n");
        assert_eq!(p.extruded().len(), 6);
        assert_eq!(p.travel().len(), 6);
    }

    #[test]
    fn compact_cnc_syntax_and_modal_motion() {
        let p = parse(
            "%\n(header)\nN10 G21G90G0X0Y0Z5\nN20 G1Z-1F100*57\nX10\nY10 ; feed continues\nG0Z5\nM5\nG1X0\n%\n",
        );
        assert!(!p.has_extrusion);
        let e = segments(&p.extruded());
        assert_eq!(
            e,
            vec![
                [[0.0, 0.0, 5.0], [0.0, 0.0, -1.0]],
                [[0.0, 0.0, -1.0], [10.0, 0.0, -1.0]],
                [[10.0, 0.0, -1.0], [10.0, 10.0, -1.0]],
            ]
        );
        // The opening rapid, G0 Z5 and the feed move after M5 all travel.
        assert_eq!(segments(&p.travel()).len(), 3);
        assert!(p.skipped.is_empty(), "{:?}", p.skipped);
    }

    #[test]
    fn inches() {
        let p = parse("G20\nG1 X1 E0.1\n");
        assert_eq!(segments(&p.extruded())[0][1], [25.4, 0.0, 0.0]);
    }

    fn on_circle(v: &[f32], c: [f32; 2], r: f32) {
        for s in segments(v) {
            for pt in s {
                let d = (pt[0] - c[0]).hypot(pt[1] - c[1]);
                assert!((d - r).abs() < 1e-4, "{pt:?} is {d} from the centre");
            }
        }
    }

    #[test]
    fn arcs_by_centre_and_radius() {
        // Quarter circle CCW from (10,0) to (0,10) round the origin.
        let ij = parse("G0 X10\nG3 X0 Y10 I-10 J0 E1\n");
        let r = parse("G0 X10\nG3 X0 Y10 R10 E1\n");
        for p in [&ij, &r] {
            let e = p.extruded();
            on_circle(&e, [0.0, 0.0], 10.0);
            let s = segments(&e);
            assert!(s.len() > 4);
            assert_eq!(s[0][0], [10.0, 0.0, 0.0]);
            assert_eq!(s.last().unwrap()[1], [0.0, 10.0, 0.0]);
            // Counter-clockwise: through the first quadrant.
            assert!(s.iter().all(|s| s[1][0] >= -1e-6 && s[1][1] >= -1e-6));
        }
        // Clockwise the long way round from the same points, through x < 0.
        let cw = parse("G0 X10\nG2 X0 Y10 I-10 J0 E1\n");
        assert!(segments(&cw.extruded()).iter().any(|s| s[1][0] < -9.0));
        // The R form picks the short arc for R > 0 and the long one for R < 0.
        let long = parse("G0 X10\nG3 X0 Y10 R-10 E1\n");
        let s = segments(&long.extruded());
        on_circle(&long.extruded(), [10.0, 10.0], 10.0);
        assert!(s.iter().any(|s| s[1][0] > 19.0));
    }

    #[test]
    fn full_circle_and_helix() {
        let p = parse("G0 X5\nG2 X5 Y0 Z2 I-5 E1\n");
        let s = segments(&p.extruded());
        on_circle(&p.extruded(), [0.0, 0.0], 5.0);
        assert_eq!(s.last().unwrap()[1], [5.0, 0.0, 2.0]);
        // Clockwise: the first step goes to negative y.
        assert!(s[0][1][1] < 0.0);
        // Z rises steadily.
        assert!(s.windows(2).all(|w| w[1][1][2] > w[0][1][2]));
        // Chords stay within tolerance of the arc.
        let step = 2.0 * (1.0 - 0.01f64 / 5.0).acos();
        assert_eq!(s.len(), (std::f64::consts::TAU / step).ceil() as usize);
    }

    #[test]
    fn arc_in_the_zx_plane() {
        // G18: Z the first axis, so Z to X is counter-clockwise seen from +Y.
        let p = parse("G18\nG0 Z10\nG3 Z0 X10 K-10 E1\n");
        let s = segments(&p.extruded());
        for seg in &s {
            let d = seg[1][0].hypot(seg[1][2]);
            assert!((d - 10.0).abs() < 1e-4);
            assert_eq!(seg[1][1], 0.0);
        }
        assert!(s.iter().all(|s| s[1][0] >= -1e-6 && s[1][2] >= -1e-6));
    }

    #[test]
    fn parameters_are_not_moves() {
        let p = parse("G1 X1 E1\nM203 E25\nG4 P500\nX2 E2\n");
        assert_eq!(segments(&p.extruded()).len(), 2);
        assert!(p.travel().is_empty());
    }

    #[test]
    fn skipped_and_malformed() {
        let p = parse("M104 S200\nM104 S210\nT1\nG1 X1 E1\nG1 X# Y2\nG29\nM117 Printing... X9\n");
        assert_eq!(p.skipped["M104"], 2);
        assert_eq!(p.skipped["M117"], 1);
        assert_eq!(p.extruded().len(), 6);
        assert_eq!(p.skipped["T1"], 1);
        assert_eq!(p.skipped["G29"], 1);
        assert_eq!(p.skipped["malformed"], 1);
    }

    #[test]
    fn scene_objects() {
        let src = "G1 X10 Z0.2 E1\nG1 Z0.4\nG1 X0 E2\n";
        let p = parse(src);
        let (group, children) = p.to_objects();
        assert_eq!(group.name, "gcode");
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].name, "layer2");
        let split = GcodeLoader {
            split_layer: true,
            ..Default::default()
        }
        .parse(src);
        let (_, children) = split.to_objects();
        let names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["layer0", "layer0", "layer1", "layer1"]);
        let mut scene = Scene::new();
        let id = split.add_to_scene(&mut scene);
        assert_eq!(scene.arena.get(id).unwrap().children.len(), 4);
    }
}
