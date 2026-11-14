//! A flat UI in a headset: a [`Panel`] stands in front of the viewer and
//! turns the inputs of any [`XrSource`](super::XrSource) (a [`Frame`]) into what a screen would give the
//! UI — pointer moves and presses, wheel turns, touches — in the UI's own
//! pixels. [`PanelPass`](super::PanelPass) draws it.
//!
//! A controller is a mouse: its ray is the pointer (hovering included), the
//! trigger the button, the thumbstick the wheel, and squeezing brings the
//! panel back in front of the viewer. An input that exists only while it is
//! used (Apple Vision Pro's look-and-pinch, a phone's tap in AR) is a
//! finger: the pinch is a touch, so dragging scrolls and swipes.
//!
//! ```
//! use threers::xr::{Panel, PanelInput};
//! use threers::Vector3;
//!
//! let mut panel = Panel::new(1280.0, 800.0);
//! let head = threers::Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)).elements;
//! panel.place_facing(&head);
//! // A ray straight at the panel's middle lands in the UI's middle.
//! let plane = panel.plane().unwrap();
//! let (u, v, _) = plane.hit(Vector3::new(0.0, plane.center.y, 0.0), Vector3::new(0.0, 0.0, -1.0)).unwrap();
//! assert_eq!(panel.ui_point(u, v), (640.0, 400.0));
//! ```

use std::collections::HashMap;

use super::input::{XrEventKind, XrInput};
use super::Frame;
use crate::math::Vector3;

/// Where the panel is: its middle, unit vectors to its right and up, and
/// half its width and height in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelPlane {
    pub center: Vector3,
    pub right: Vector3,
    pub up: Vector3,
    pub half: (f32, f32),
}

impl PanelPlane {
    /// Where a ray from `origin` along `dir` crosses the plane, as a
    /// fraction of the panel from its top left (outside 0..1 off its edges),
    /// and how far along the ray; `None` behind it or parallel.
    pub fn project(&self, origin: Vector3, dir: Vector3) -> Option<(f32, f32, f32)> {
        let normal = self.right.cross(self.up);
        let d = dir.dot(normal);
        if d.abs() < 1e-6 {
            return None;
        }
        let t = (self.center - origin).dot(normal) / d;
        if t <= 0.0 {
            return None;
        }
        let p = origin + dir * t - self.center;
        Some((
            0.5 + p.dot(self.right) / (2.0 * self.half.0),
            0.5 - p.dot(self.up) / (2.0 * self.half.1),
            t,
        ))
    }

    /// [`project`](Self::project), only where it lands on the panel.
    pub fn hit(&self, origin: Vector3, dir: Vector3) -> Option<(f32, f32, f32)> {
        self.project(origin, dir)
            .filter(|&(u, v, _)| (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v))
    }
}

/// What the UI is told, in its pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PanelInput {
    Move(f32, f32),
    Down(f32, f32),
    Up(f32, f32),
    /// At `(x, y)`, by `(dx, dy)` pixels.
    Wheel(f32, f32, f32, f32),
    /// A finger `id`: down, moved, lifted, or taken away (the input left).
    TouchStart(u64, f32, f32),
    TouchMove(u64, f32, f32),
    TouchEnd(u64, f32, f32),
    TouchCancel(u64, f32, f32),
}

/// What is drawn of a pointer: a ray from `from` to `to`, and a dot where it
/// meets the panel (`pressed` while it presses).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pointer {
    pub from: Vector3,
    pub to: Vector3,
    pub ray: bool,
    pub dot: Option<(f32, f32)>,
    pub pressed: bool,
}

/// A UI on a panel in a headset. See the module docs.
#[derive(Debug, Clone)]
pub struct Panel {
    /// The UI's size in its pixels.
    pub size: (f32, f32),
    /// Metres a UI pixel (1.25 mm: a 1280-pixel UI is 1.6 m wide).
    pub metres_per_pixel: f32,
    /// How far in front of the head it stands, and how far below the eyes
    /// its middle is.
    pub distance: f32,
    pub drop: f32,
    /// How far its top leans back, radians: a panel below the eyes reads
    /// best turned up to face them (`atan(drop / distance)`). 0: upright.
    pub tilt: f32,
    /// How far to the right of straight ahead it stands (negative: the
    /// left), metres; it turns to face the head from there.
    pub side: f32,
    /// How long a ray that hits nothing is drawn.
    pub reach: f32,
    plane: Option<PanelPlane>,
    recenter: bool,
    /// The controller driving the mouse, whether its trigger is down, and
    /// where the mouse last was.
    active: Option<usize>,
    pressed: bool,
    mouse: Option<(f32, f32)>,
    /// Pinches held, as touches: each one's last point.
    touches: HashMap<usize, (f32, f32)>,
    /// Fingertips touching the panel (a hand's poke): the slot's last point.
    pokes: HashMap<usize, (f32, f32)>,
    /// Each hand's fingertip distance in front of the panel last frame.
    poke_was: HashMap<usize, f32>,
}

impl Panel {
    /// A panel for a UI `width × height` pixels; placed in front of the
    /// head on the first frame.
    pub fn new(width: f32, height: f32) -> Panel {
        Panel {
            size: (width, height),
            metres_per_pixel: 0.00125,
            distance: 1.3,
            drop: 0.15,
            tilt: 0.0,
            side: 0.0,
            reach: 3.0,
            plane: None,
            recenter: false,
            active: None,
            pressed: false,
            mouse: None,
            touches: HashMap::new(),
            pokes: HashMap::new(),
            poke_was: HashMap::new(),
        }
    }

    /// Where it stands, once placed.
    pub fn plane(&self) -> Option<PanelPlane> {
        self.plane
    }

    /// Stands it upright in front of `head` (a pose, column-major), facing it.
    pub fn place_facing(&mut self, head: &[f32; 16]) {
        let at = Vector3::new(head[12], head[13], head[14]);
        let ahead = Vector3::new(-head[8], 0.0, -head[10]);
        let ahead = if ahead.length() > 1e-3 {
            ahead.normalize()
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        };
        let up = Vector3::new(0.0, 1.0, 0.0);
        let half = (
            self.size.0 * self.metres_per_pixel / 2.0,
            self.size.1 * self.metres_per_pixel / 2.0,
        );
        let across = ahead.cross(up).normalize();
        let center = at + ahead * self.distance + across * self.side - up * self.drop;
        // facing the head from where it stands, level
        let toward = Vector3::new(center.x - at.x, 0.0, center.z - at.z);
        let ahead = if toward.length() > 1e-3 {
            toward.normalize()
        } else {
            ahead
        };
        let right = ahead.cross(up).normalize();
        let up = up * self.tilt.cos() + ahead * self.tilt.sin();
        self.plane = Some(PanelPlane {
            center,
            right,
            up,
            half,
        });
    }

    /// Brings it back in front of the head on the next frame.
    pub fn recenter(&mut self) {
        self.recenter = true;
    }

    /// A point on the panel (0..1 from its top left) in UI pixels.
    pub fn ui_point(&self, u: f32, v: f32) -> (f32, f32) {
        (u * self.size.0, v * self.size.1)
    }

    /// This frame's input for the UI and pointers to draw, from `frame`'s
    /// head, inputs and events (see [`Frame::take`]).
    pub fn step(&mut self, frame: &Frame) -> (Vec<PanelInput>, Vec<Pointer>) {
        let (events, inputs) = (frame.events.clone(), frame.inputs.clone());
        if let Some(head) = frame.head
            && (self.plane.is_none() || self.recenter)
        {
            self.place_facing(&head);
            self.recenter = false;
        }
        let Some(plane) = self.plane else {
            return (vec![], vec![]);
        };
        let by_slot: HashMap<usize, &XrInput> = inputs.iter().map(|i| (i.slot, i)).collect();
        let transient = |slot: usize| {
            by_slot
                .get(&slot)
                .is_some_and(|i| i.target_ray_mode.is_transient())
        };
        let project = |slot: usize| {
            by_slot
                .get(&slot)
                .and_then(|i| i.ray_line())
                .and_then(|(o, d)| plane.project(o, d))
        };
        let on = |p: Option<(f32, f32, f32)>| {
            p.filter(|&(u, v, _)| (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v))
        };
        let mut out = vec![];
        for e in events {
            let held = self.touches.contains_key(&e.slot);
            match e.kind {
                XrEventKind::SelectStart if transient(e.slot) => {
                    if let Some((u, v, _)) = on(project(e.slot)) {
                        let (x, y) = self.ui_point(u, v);
                        out.push(PanelInput::TouchStart(e.slot as u64, x, y));
                        self.touches.insert(e.slot, (x, y));
                    }
                }
                XrEventKind::SelectEnd | XrEventKind::Disconnected if held => {
                    let (x, y) = self.touches.remove(&e.slot).unwrap_or_default();
                    let id = e.slot as u64;
                    out.push(if e.kind == XrEventKind::SelectEnd {
                        PanelInput::TouchEnd(id, x, y)
                    } else {
                        PanelInput::TouchCancel(id, x, y)
                    });
                }
                XrEventKind::SelectStart => {
                    self.active = Some(e.slot);
                    if let Some((u, v, _)) = on(project(e.slot)) {
                        let (x, y) = self.ui_point(u, v);
                        out.push(PanelInput::Move(x, y));
                        out.push(PanelInput::Down(x, y));
                        self.mouse = Some((x, y));
                        self.pressed = true;
                    }
                }
                XrEventKind::SelectEnd | XrEventKind::Disconnected
                    if self.active == Some(e.slot) && self.pressed =>
                {
                    let (x, y) = self.mouse.unwrap_or_default();
                    out.push(PanelInput::Up(x, y));
                    self.pressed = false;
                }
                XrEventKind::SqueezeStart => self.recenter = true,
                _ => {}
            }
        }
        // Touches follow their pinch, off the edges too (a drag carries on).
        let size = self.size;
        for (slot, last) in self.touches.iter_mut() {
            if let Some((u, v, _)) = project(*slot) {
                let p = (u * size.0, v * size.1);
                if p != *last {
                    out.push(PanelInput::TouchMove(*slot as u64, p.0, p.1));
                    *last = p;
                }
            }
        }
        // The mouse: the pressed controller, else whichever points at the panel.
        let hits = |slot: usize| on(project(slot)).is_some();
        if !self.pressed && !self.active.is_some_and(hits) {
            self.active = inputs
                .iter()
                .filter(|i| !i.target_ray_mode.is_transient())
                .map(|i| i.slot)
                .find(|&s| hits(s));
        }
        if let Some(slot) = self.active {
            let at = if self.pressed {
                project(slot)
            } else {
                on(project(slot))
            };
            if let Some((u, v, _)) = at {
                let p = self.ui_point(u, v);
                if self.mouse != Some(p) {
                    out.push(PanelInput::Move(p.0, p.1));
                    self.mouse = Some(p);
                }
            }
            if let (Some(i), Some((x, y))) = (by_slot.get(&slot), self.mouse) {
                let (dx, dy) = i.thumbstick();
                let dead = |v: f32| if v.abs() > 0.2 { v * 24.0 } else { 0.0 };
                if dead(dx) != 0.0 || dead(dy) != 0.0 {
                    out.push(PanelInput::Wheel(x, y, dead(dx), dead(dy)));
                }
            }
        }
        let mut pointers: Vec<Pointer> = self.poke(&inputs, &plane, &mut out);
        let rays = inputs
            .iter()
            .filter_map(|i| {
                let (from, dir) = i.ray_line()?;
                let hit = plane.hit(from, dir);
                let finger = i.target_ray_mode.is_transient();
                let touching = self.touches.contains_key(&i.slot);
                Some(Pointer {
                    from,
                    to: from + dir * hit.map_or(self.reach, |h| h.2),
                    // A pinch shows no ray, as the system draws none.
                    ray: !finger,
                    dot: hit.map(|(u, v, _)| (u, v)).filter(|_| !finger || touching),
                    pressed: (self.pressed && self.active == Some(i.slot)) || touching,
                })
            })
            .collect::<Vec<_>>();
        pointers.extend(rays);
        (out, pointers)
    }

    /// Direct touch: each tracked hand's index fingertip pressing the panel
    /// like a screen. Within [`POKE_HOVER`] in front of it a dot shows where
    /// it would touch; crossing the surface (from in front) is a touch,
    /// moving along it drags, and pulling back past [`POKE_LIFT`] — or
    /// going [`POKE_THROUGH`] behind, or off its edges — lifts it.
    fn poke(
        &mut self,
        inputs: &[XrInput],
        plane: &PanelPlane,
        out: &mut Vec<PanelInput>,
    ) -> Vec<Pointer> {
        let normal = plane.right.cross(plane.up);
        let mut pointers = vec![];
        let mut seen = vec![];
        for i in inputs {
            let Some((joints, _)) = &i.joints else {
                continue;
            };
            let Some(m) = joints.get(INDEX_TIP * 16..INDEX_TIP * 16 + 16) else {
                continue;
            };
            seen.push(i.slot);
            let tip = Vector3::new(m[12], m[13], m[14]);
            let rel = tip - plane.center;
            let depth = rel.dot(normal);
            let (u, v) = (
                0.5 + rel.dot(plane.right) / (2.0 * plane.half.0),
                0.5 - rel.dot(plane.up) / (2.0 * plane.half.1),
            );
            let inside = (0.0..=1.0).contains(&u) && (0.0..=1.0).contains(&v);
            let p = self.ui_point(u, v);
            let id = POKE_ID + i.slot as u64;
            let was = self.poke_was.insert(i.slot, depth);
            match self.pokes.get(&i.slot).copied() {
                Some(last) => {
                    if !inside || depth > POKE_LIFT || depth < -POKE_THROUGH {
                        self.pokes.remove(&i.slot);
                        out.push(PanelInput::TouchEnd(id, last.0, last.1));
                    } else if p != last {
                        self.pokes.insert(i.slot, p);
                        out.push(PanelInput::TouchMove(id, p.0, p.1));
                    }
                }
                // A touch begins only as the fingertip crosses the surface
                // from in front (not a hand rising from behind or below).
                None if inside
                    && depth <= 0.0
                    && was.is_some_and(|w| w > 0.0)
                    && depth > -POKE_THROUGH =>
                {
                    self.pokes.insert(i.slot, p);
                    out.push(PanelInput::TouchStart(id, p.0, p.1));
                }
                None => {}
            }
            let touching = self.pokes.contains_key(&i.slot);
            if inside && (touching || (0.0..POKE_HOVER).contains(&depth)) {
                pointers.push(Pointer {
                    from: tip,
                    to: tip - normal * depth,
                    ray: false,
                    dot: Some((u, v)),
                    pressed: touching,
                });
            }
        }
        // Hands gone mid-touch: cancelled.
        for slot in self
            .pokes
            .keys()
            .copied()
            .filter(|s| !seen.contains(s))
            .collect::<Vec<_>>()
        {
            let (x, y) = self.pokes.remove(&slot).unwrap_or_default();
            out.push(PanelInput::TouchCancel(POKE_ID + slot as u64, x, y));
        }
        self.poke_was.retain(|s, _| seen.contains(s));
        pointers
    }
}

/// [`HAND_JOINTS`](super::HAND_JOINTS)' index fingertip.
const INDEX_TIP: usize = 9;
/// Pokes' touch ids, apart from pinches' (which are their slots).
const POKE_ID: u64 = 1 << 32;
/// How near in front of the panel a fingertip shows where it would touch,
/// how far back it lifts a touch, and how far through it before a touch is
/// let go (metres).
pub const POKE_HOVER: f32 = 0.05;
pub const POKE_LIFT: f32 = 0.015;
pub const POKE_THROUGH: f32 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::{Matrix4, Quaternion};
    use crate::xr::{TargetRayMode, XrEvent, XrEye, XrSource};

    /// A scripted headset: a head, inputs and events set by the test.
    #[derive(Default)]
    struct Fake {
        head: Option<[f32; 16]>,
        inputs: Vec<XrInput>,
        events: Vec<XrEvent>,
    }

    impl XrSource for Fake {
        fn head(&self) -> Option<[f32; 16]> {
            self.head
        }
        fn eyes(&self, _: &Matrix4) -> Vec<XrEye> {
            vec![]
        }
        fn framebuffer_size(&self) -> (u32, u32) {
            (0, 0)
        }
        fn inputs(&self) -> Vec<XrInput> {
            self.inputs.clone()
        }
        fn take_events(&mut self) -> Vec<XrEvent> {
            std::mem::take(&mut self.events)
        }
        fn see_through(&self) -> bool {
            false
        }
    }

    fn pointing_at(slot: usize, mode: TargetRayMode, x: f32, y: f32) -> XrInput {
        XrInput {
            slot,
            target_ray_mode: mode,
            ray: Some(Matrix4::translation(Vector3::new(x, y, 0.0)).elements),
            ..Default::default()
        }
    }

    /// Pixels to the hundredth, so the tests read in whole numbers.
    fn rounded(v: Vec<PanelInput>) -> Vec<PanelInput> {
        let r = |x: f32| (x * 100.0).round() / 100.0;
        v.into_iter()
            .map(|i| match i {
                PanelInput::Move(x, y) => PanelInput::Move(r(x), r(y)),
                PanelInput::Down(x, y) => PanelInput::Down(r(x), r(y)),
                PanelInput::Up(x, y) => PanelInput::Up(r(x), r(y)),
                PanelInput::Wheel(x, y, dx, dy) => PanelInput::Wheel(r(x), r(y), r(dx), r(dy)),
                PanelInput::TouchStart(id, x, y) => PanelInput::TouchStart(id, r(x), r(y)),
                PanelInput::TouchMove(id, x, y) => PanelInput::TouchMove(id, r(x), r(y)),
                PanelInput::TouchEnd(id, x, y) => PanelInput::TouchEnd(id, r(x), r(y)),
                PanelInput::TouchCancel(id, x, y) => PanelInput::TouchCancel(id, r(x), r(y)),
            })
            .collect()
    }

    fn headset() -> (Panel, Fake) {
        let fake = Fake {
            head: Some(Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)).elements),
            ..Default::default()
        };
        (Panel::new(1280.0, 800.0), fake)
    }

    #[test]
    fn the_plane_stands_ahead_and_turns_with_the_head() {
        let mut p = Panel::new(1280.0, 800.0);
        p.place_facing(&Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)).elements);
        let plane = p.plane().unwrap();
        assert!((plane.center - Vector3::new(0.0, 1.45, -1.3)).length() < 1e-5);
        assert!((plane.half.0 - 0.8).abs() < 1e-6 && (plane.half.1 - 0.5).abs() < 1e-6);
        let (u, v, _) = plane
            .hit(Vector3::new(0.4, 1.7, 0.0), Vector3::new(0.0, 0.0, -1.0))
            .unwrap();
        assert_eq!(p.ui_point(u, v), (960.0, 200.0));
        assert!(plane
            .hit(Vector3::new(2.0, 1.45, 0.0), Vector3::new(0.0, 0.0, -1.0))
            .is_none());
        assert!(
            plane
                .hit(Vector3::new(0.0, 1.45, 0.0), Vector3::new(0.0, 0.0, 1.0))
                .is_none(),
            "behind"
        );
        // Turned 90° right: the panel stands at +X.
        let turned = Matrix4::translation(Vector3::new(0.0, 1.6, 0.0)).multiply(
            &Matrix4::from_quaternion(Quaternion::from_axis_angle(
                Vector3::new(0.0, 1.0, 0.0),
                -std::f32::consts::FRAC_PI_2,
            )),
        );
        p.place_facing(&turned.elements);
        let plane = p.plane().unwrap();
        assert!(
            (plane.center - Vector3::new(1.3, 1.45, 0.0)).length() < 1e-4,
            "{:?}",
            plane.center
        );
        assert!((plane.right - Vector3::new(0.0, 0.0, 1.0)).length() < 1e-4);
    }

    #[test]
    fn a_controller_hovers_clicks_and_scrolls() {
        let (mut p, mut x) = headset();
        x.inputs = vec![pointing_at(0, TargetRayMode::TrackedPointer, 0.0, 1.45)];
        let (out, pointers) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(out, [PanelInput::Move(640.0, 400.0)], "hover");
        assert!(pointers[0].ray && pointers[0].dot.is_some() && !pointers[0].pressed);
        x.events = vec![XrEvent {
            kind: XrEventKind::SelectStart,
            slot: 0,
        }];
        let (out, pointers) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(
            out,
            [
                PanelInput::Move(640.0, 400.0),
                PanelInput::Down(640.0, 400.0)
            ]
        );
        assert!(pointers[0].pressed);
        // Dragged off the edge while pressed, then let go: up where it went.
        x.inputs = vec![pointing_at(0, TargetRayMode::TrackedPointer, 1.0, 1.45)];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(out, [PanelInput::Move(1440.0, 400.0)]);
        x.events = vec![XrEvent {
            kind: XrEventKind::SelectEnd,
            slot: 0,
        }];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(out, [PanelInput::Up(1440.0, 400.0)]);
        // The thumbstick.
        x.inputs = vec![XrInput {
            axes: vec![0.0, 0.0, 0.0, 0.5],
            ..pointing_at(0, TargetRayMode::TrackedPointer, 0.0, 1.45)
        }];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(
            out,
            [
                PanelInput::Move(640.0, 400.0),
                PanelInput::Wheel(640.0, 400.0, 0.0, 12.0)
            ]
        );
    }

    #[test]
    fn a_pinch_is_a_touch_and_shows_no_ray() {
        let (mut p, mut x) = headset();
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        // Apple Vision Pro: the input appears with the pinch.
        x.inputs = vec![pointing_at(3, TargetRayMode::TransientPointer, 0.0, 1.45)];
        x.events = vec![
            XrEvent {
                kind: XrEventKind::Connected,
                slot: 3,
            },
            XrEvent {
                kind: XrEventKind::SelectStart,
                slot: 3,
            },
        ];
        let (out, pointers) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(
            out,
            [PanelInput::TouchStart(3, 640.0, 400.0)],
            "no hover, no mouse"
        );
        assert!(!pointers[0].ray && pointers[0].pressed);
        x.inputs = vec![pointing_at(3, TargetRayMode::TransientPointer, 0.0, 1.55)];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(out, [PanelInput::TouchMove(3, 640.0, 320.0)], "a drag up");
        x.events = vec![
            XrEvent {
                kind: XrEventKind::SelectEnd,
                slot: 3,
            },
            XrEvent {
                kind: XrEventKind::Disconnected,
                slot: 3,
            },
        ];
        x.inputs = vec![];
        let (out, pointers) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        let out = rounded(out);
        assert_eq!(out, [PanelInput::TouchEnd(3, 640.0, 320.0)]);
        assert!(pointers.is_empty());
    }

    /// A hand whose index fingertip is at `(x, y, z)` (the other joints
    /// at the origin).
    fn hand_at(slot: usize, x: f32, y: f32, z: f32) -> XrInput {
        let mut joints = vec![0.0; 25 * 16];
        joints[INDEX_TIP * 16..INDEX_TIP * 16 + 16]
            .copy_from_slice(&Matrix4::translation(Vector3::new(x, y, z)).elements);
        XrInput {
            slot,
            joints: Some((joints, vec![0.008; 25])),
            ..Default::default()
        }
    }

    #[test]
    fn a_fingertip_pokes_the_panel_like_a_screen() {
        let (mut p, mut x) = headset();
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        // The panel's plane is at z = -1.3, facing +z; its middle at y 1.45.
        let z = -1.3;
        x.inputs = vec![hand_at(7, 0.0, 1.45, z + 0.03)];
        let (out, pointers) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert!(out.is_empty(), "hovering touches nothing: {out:?}");
        assert!(
            pointers.iter().any(|q| q.dot.is_some() && !q.pressed),
            "a dot where it would touch"
        );
        x.inputs = vec![hand_at(7, 0.0, 1.45, z - 0.003)];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert_eq!(
            rounded(out),
            [PanelInput::TouchStart(POKE_ID + 7, 640.0, 400.0)]
        );
        x.inputs = vec![hand_at(7, 0.0, 1.55, z - 0.003)];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert_eq!(
            rounded(out),
            [PanelInput::TouchMove(POKE_ID + 7, 640.0, 320.0)],
            "a drag up"
        );
        // Pulled back a little stays down (no chatter); further lifts it.
        x.inputs = vec![hand_at(7, 0.0, 1.55, z + 0.01)];
        assert!(p.step(&Frame::take(&mut x, 0.0, 0.0)).0.is_empty());
        x.inputs = vec![hand_at(7, 0.0, 1.55, z + 0.03)];
        let (out, _) = p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert_eq!(
            rounded(out),
            [PanelInput::TouchEnd(POKE_ID + 7, 640.0, 320.0)]
        );
    }

    #[test]
    fn a_hand_from_behind_does_not_poke() {
        let (mut p, mut x) = headset();
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        x.inputs = vec![hand_at(2, 0.0, 1.45, -1.32)];
        assert!(
            p.step(&Frame::take(&mut x, 0.0, 0.0)).0.is_empty(),
            "first seen behind"
        );
        x.inputs = vec![hand_at(2, 0.0, 1.45, -1.31)];
        assert!(
            p.step(&Frame::take(&mut x, 0.0, 0.0)).0.is_empty(),
            "still behind: never crossed from in front"
        );
    }

    #[test]
    fn squeezing_brings_the_panel_back() {
        let (mut p, mut x) = headset();
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        x.head = Some(Matrix4::translation(Vector3::new(2.0, 1.6, 0.0)).elements);
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert!(
            (p.plane().unwrap().center.x).abs() < 1e-5,
            "stays put as the head moves"
        );
        x.events = vec![XrEvent {
            kind: XrEventKind::SqueezeStart,
            slot: 0,
        }];
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        p.step(&Frame::take(&mut x, 0.0, 0.0));
        assert!((p.plane().unwrap().center.x - 2.0).abs() < 1e-5);
    }
}
