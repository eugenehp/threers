//! 3D effects for a flat UI: a box (and everything in it) turned about its
//! horizontal and vertical axes, lifted toward the viewer, and seen in
//! perspective; lit by a glare that slides with the turn; casting a shadow
//! that grows and shifts with its lift; with layers inside it that move by
//! their depth (parallax).
//!
//! On top of that: [`Input::tilt_on_hover`] (a card leaning toward the
//! pointer or finger and settling back when it leaves), [`flip_card`] (two
//! faces turned over by a binding), and [`Ui::carousel_3d`] (a strip whose
//! items turn away from its middle as it scrolls).
//!
//! A 2D painter cannot draw true perspective, so each box is drawn through
//! the 2D transform nearest its projection, taken at its middle: it narrows
//! as it turns (foreshortening) and grows as it comes forward, but its near
//! edge is not drawn taller than its far one. Painters without transforms
//! draw it flat. Hit testing goes by the untransformed boxes.
//!
//! ```
//! use threers_ui::depth::Transform3d;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let card = ui.add_box(root, style! { width: 200px; height: 100px; });
//! ui.set_transform3d(card, Transform3d { rotate_y: 60f32.to_radians(), ..Transform3d::default() });
//! let m = ui.affine_of(card).unwrap();
//! assert!((m[0] - 0.5).abs() < 0.05, "half as wide, turned 60°: {m:?}");
//! ```

use std::cell::Cell;
use std::rc::Rc;

use crate::anim::Spring;
use crate::animation::Property;
use crate::binding::Binding;
use crate::input::{Event, Input, Phase};
use crate::tree::{NodeId, Ui};
use crate::{style, Rect, Role, Semantics};

/// How a box is turned and lifted. Angles in radians.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform3d {
    /// About the horizontal axis: positive tips the top away.
    pub rotate_x: f32,
    /// About the vertical axis: positive turns the right side away.
    pub rotate_y: f32,
    /// Toward the viewer, in pixels.
    pub depth: f32,
    /// Larger (or smaller), about its middle.
    pub scale: f32,
    /// The viewer's distance, in pixels: less is more dramatic.
    pub perspective: f32,
    /// A light glancing off it, sliding with the turn.
    pub glare: bool,
    /// A shadow under it, as far below as it is lifted.
    pub shadow: bool,
    /// Not drawn while its back faces the viewer (one face of a card that
    /// flips).
    pub hide_backface: bool,
    /// Turned about this point, as fractions of its width and height
    /// (`(0.5, 0.5)`, its middle, by default; `(0.0, 0.5)` its left edge, as
    /// a door or a cube's face).
    pub origin: (f32, f32),
    /// In its own plane, in radians (a card tipped as it is swiped away).
    pub rotate_z: f32,
    /// Darker as it turns away from the light (see [`Light`]).
    pub shade: bool,
    /// Moved sideways in 3D after it is turned, in pixels (a cube's face
    /// swinging round the cube's middle).
    pub translate_x: f32,
    /// Moved down (up, negative) in 3D after it is turned, in pixels: a
    /// box hopping or bobbing (see [`motion`](crate::motion)).
    pub translate_y: f32,
}

impl Default for Transform3d {
    fn default() -> Self {
        Self {
            rotate_x: 0.0,
            rotate_y: 0.0,
            depth: 0.0,
            scale: 1.0,
            perspective: 800.0,
            glare: false,
            shadow: false,
            hide_backface: false,
            origin: (0.5, 0.5),
            rotate_z: 0.0,
            shade: false,
            translate_x: 0.0,
            translate_y: 0.0,
        }
    }
}

impl Transform3d {
    /// Where a point `(x, y)` of a box `rect` is seen, turned, lifted and
    /// in perspective (about the window point it pivots on).
    pub fn project(&self, rect: Rect, x: f32, y: f32) -> (f32, f32) {
        let (px, py) = (rect.x + rect.width * self.origin.0, rect.y + rect.height * self.origin.1);
        let p = self.perspective.max(1.0);
        let (x, y) = (x - px, y - py);
        // In its plane, then about its horizontal axis, then its vertical.
        let (sz, cz) = self.rotate_z.sin_cos();
        let (x, y) = (x * cz - y * sz, x * sz + y * cz);
        let (sx, cxr) = self.rotate_x.sin_cos();
        let (sy, cyr) = self.rotate_y.sin_cos();
        let (y1, z1) = (y * cxr, y * sx);
        let (x2, z2) = (x * cyr + z1 * sy, -x * sy + z1 * cyr);
        // Seen from in front of the box's middle.
        let (cx, cy) = (rect.x + rect.width / 2.0 - px, rect.y + rect.height / 2.0 - py);
        let x2 = x2 + self.translate_x;
        let y1 = y1 + self.translate_y;
        let z = z2 + self.depth;
        let f = p / (p - z).max(1.0) * self.scale;
        (px + cx + (x2 - cx) * f, py + cy + (y1 - cy) * f)
    }

    /// The 2D transform nearest it, for a box `rect`: `[a, b, c, d, e, f]`
    /// (exact at its middle and along its middle lines).
    pub fn affine(&self, rect: Rect) -> [f32; 6] {
        let (cx, cy) = (rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
        let (hw, hh) = (rect.width.max(1.0) / 2.0, rect.height.max(1.0) / 2.0);
        let o = self.project(rect, cx, cy);
        let (l, r) = (self.project(rect, cx - hw, cy), self.project(rect, cx + hw, cy));
        let (t, b) = (self.project(rect, cx, cy - hh), self.project(rect, cx, cy + hh));
        let (a, bb) = ((r.0 - l.0) / (2.0 * hw), (r.1 - l.1) / (2.0 * hw));
        let (c, d) = ((b.0 - t.0) / (2.0 * hh), (b.1 - t.1) / (2.0 * hh));
        [a, bb, c, d, o.0 - (a * cx + c * cy), o.1 - (bb * cx + d * cy)]
    }

    /// The projective transform taking the box `rect` exactly to where its
    /// corners are seen, `[h11, h12, h13, h21, h22, h23, h31, h32, h33]`
    /// (x' = (h11·x + h12·y + h13) / (h31·x + h32·y + h33), and so for y'):
    /// for painters that draw true perspective.
    pub fn homography(&self, rect: Rect) -> [f32; 9] {
        let (x0, y0, w, h) = (rect.x, rect.y, rect.width.max(1e-3), rect.height.max(1e-3));
        let q = [
            self.project(rect, x0, y0),
            self.project(rect, x0 + w, y0),
            self.project(rect, x0 + w, y0 + h),
            self.project(rect, x0, y0 + h),
        ];
        homography_to(q, (x0, y0, w, h))
    }

    /// The way its face points, a unit vector (z toward the viewer, y down).
    pub fn normal(&self) -> [f32; 3] {
        let (sx, cx) = self.rotate_x.sin_cos();
        let (sy, cy) = self.rotate_y.sin_cos();
        let (y, z) = (-sx, cx);
        [z * sy, y, z * cy]
    }

    /// Whether its front faces the viewer.
    pub fn facing(&self) -> bool {
        self.rotate_x.cos() * self.rotate_y.cos() >= 0.0
    }
}

impl Ui {
    /// Turns, lifts and scales `node` (and what is in it) in 3D.
    pub fn set_transform3d(&mut self, node: NodeId, t: Transform3d) {
        self.env.transforms3d.insert(node, t);
    }

    pub fn transform3d(&self, node: NodeId) -> Option<Transform3d> {
        self.env.transforms3d.get(&node).copied()
    }

    pub fn clear_transform3d(&mut self, node: NodeId) {
        self.env.transforms3d.remove(&node);
    }

    /// The 2D transform `node` is painted through, if it has a 3D one.
    pub fn affine_of(&self, node: NodeId) -> Option<[f32; 6]> {
        Some(self.transform3d(node)?.affine(self.rect(node)))
    }

    /// Makes `node` a layer `depth` pixels in front of (negative: behind)
    /// the box around it: when that box tilts, it moves by its depth, so
    /// near layers slide one way and far ones the other.
    pub fn set_parallax(&mut self, node: NodeId, depth: f32) {
        self.env.parallax.insert(node, depth);
    }

    /// Whether `node` is drawn: not a box turned away whose back is hidden.
    /// Its own turn is judged as projected, perspective and all (a face
    /// turned a right angle off to one side of the eye shows its back, as
    /// a sliver); turns of the boxes around it add to the angle.
    pub(crate) fn faces_viewer(&self, node: NodeId) -> bool {
        let Some(t) = self.transform3d(node).filter(|t| t.hide_backface) else { return true };
        let (mut rx, mut ry) = (0.0, 0.0);
        for a in self.ancestors(node).skip(1) {
            if let Some(at) = self.transform3d(a) {
                rx += at.rotate_x;
                ry += at.rotate_y;
            }
        }
        let r = self.rect(node);
        let p = [(r.x, r.y), (r.x + r.width, r.y), (r.x + r.width, r.y + r.height), (r.x, r.y + r.height)].map(|(x, y)| t.project(r, x, y));
        let area: f32 = (0..4).map(|i| p[i].0 * p[(i + 1) % 4].1 - p[(i + 1) % 4].0 * p[i].1).sum();
        // Facing: wound as the box itself is, by more than a sliver; or the
        // other way, when the boxes around it have turned it over.
        let sliver = 1e-3 * (r.width * r.height).max(1.0);
        if rx.cos() * ry.cos() >= 0.0 { area > sliver } else { area < -sliver }
    }

    /// The transform to paint `node` through, its own and its parallax:
    /// exact (projective) and nearest 2D.
    pub(crate) fn paint_transform(&self, node: NodeId) -> Option<([f32; 9], [f32; 6])> {
        let r = self.rect(node);
        // A solid box turns in its own shader, walls and all: drawn here
        // unturned (rolled, moved and scaled still).
        let solid = self.env.solids.contains_key(&node);
        let own = self.transform3d(node).map(|t| if solid { Transform3d { rotate_x: 0.0, rotate_y: 0.0, ..t } } else { t }).map(|t| (t.homography(r), t.affine(r)));
        let shift = self.env.parallax.get(&node).and_then(|&z| {
            // The nearest box around it that turns.
            let t = self.ancestors(node).skip(1).find_map(|a| self.transform3d(a))?;
            Some((t.rotate_y.sin() * z, -t.rotate_x.sin() * z))
        });
        let identity = ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        match (own, shift) {
            (None, None) => None,
            (own, shift) => {
                let (mut h, mut m) = own.unwrap_or(identity);
                if let Some((dx, dy)) = shift {
                    // Moved after the projection.
                    for k in 0..3 {
                        h[k] += dx * h[6 + k];
                        h[3 + k] += dy * h[6 + k];
                    }
                    m[4] += dx;
                    m[5] += dy;
                }
                Some((h, m))
            }
        }
    }

    /// Where light falls from, for the glare and shading of turned boxes.
    pub fn set_light(&mut self, light: Light) {
        self.env.light = light;
    }

    pub fn light(&self) -> Light {
        self.env.light
    }

    /// Blurs the boxes inside `scope` lifted or sunk away from `focus` (a
    /// depth, in pixels): by `strength` pixels of blur for each 100 pixels
    /// of depth away, as a camera's depth of field does. `None` turns it
    /// off there. (An [environment](crate::environment) value: a scope
    /// inside sets its own.)
    pub fn set_depth_of_field(&mut self, scope: NodeId, focus_and_strength: Option<(f32, f32)>) {
        self.set_environment(scope, DepthOfField(focus_and_strength));
    }

    /// The blur `node` is painted with, from its depth.
    pub(crate) fn depth_blur(&self, node: NodeId) -> Option<f32> {
        let t = self.transform3d(node)?;
        let (focus, strength) = self.environment::<DepthOfField>(node)?.0?;
        let r = (t.depth - focus).abs() / 100.0 * strength;
        (r >= 0.25).then_some(r.min(20.0))
    }

    /// How the device is held (a phone's `deviceorientation`: `beta`, its
    /// tilt forward, and `gamma`, sideways, in degrees), for
    /// [`Input::tilt_with_device`].
    pub fn set_device_tilt(&mut self, beta: f32, gamma: f32) {
        self.env.device_tilt = (beta, gamma);
    }

    pub fn device_tilt(&self) -> (f32, f32) {
        self.env.device_tilt
    }

    /// Under a lifted box: a shadow as far below as it is lifted, cast away
    /// from the light.
    pub(crate) fn paint_lift_shadow(&self, node: NodeId, painter: &mut dyn crate::Painter, opacity: f32) {
        let Some(t) = self.transform3d(node).filter(|t| t.shadow) else { return };
        let r = self.rect(node);
        let lift = t.depth.max(0.0) + 4.0;
        let radius = self.style(node).border_radius.resolve(Some(r.width.min(r.height))).unwrap_or(0.0);
        // As large as the box is drawn: a lifted box is drawn nearer, so
        // larger, and its shadow with it (a little smaller, as the light
        // is above).
        let p = t.perspective.max(1.0);
        let k = t.scale * p / (p - t.depth).max(1.0) * 0.96;
        let l = self.env.light;
        // Offset and softened by the lift, gently: a shadow far below a box
        // reads as a smudge rather than a shadow.
        let drop = lift.sqrt() * 2.2;
        let rect = Rect {
            x: r.x + r.width * (1.0 - k) / 2.0 - t.rotate_y.sin() * lift - l.x * drop,
            y: r.y + r.height * (1.0 - k) / 2.0 + drop + t.rotate_x.sin() * lift - l.y * drop * 0.5,
            width: r.width * k,
            height: r.height * k,
        };
        let alpha = if t.depth > 0.0 { (0.16 + lift * 0.002).min(0.3) } else { 0.1 };
        painter.shadow(rect, radius, 4.0 + drop * 1.6, [0.0, 0.0, 0.0, alpha * opacity]);
    }

    /// Over a shaded box: darker by how far it has turned from the light.
    pub(crate) fn paint_shade(&self, node: NodeId, painter: &mut dyn crate::Painter, opacity: f32) {
        let Some(t) = self.transform3d(node).filter(|t| t.shade) else { return };
        let (n, l) = (t.normal(), self.env.light.direction());
        let lit = |n: [f32; 3]| (n[0] * l[0] + n[1] * l[1] + n[2] * l[2]).max(0.0);
        // Darker than it is flat on, never lighter; a face turned edge on
        // goes dark.
        let dark = ((lit([0.0, 0.0, 1.0]) - lit(n)) * 0.9 + (1.0 - n[2].abs()) * 0.3).clamp(0.0, 0.75) * self.env.light.intensity;
        if dark < 0.01 {
            return;
        }
        let r = self.rect(node);
        let radius = self.style(node).border_radius.resolve(Some(r.width.min(r.height))).unwrap_or(0.0);
        painter.fill_rounded_rect(r, radius, radius, [0.0, 0.0, 0.0, dark * opacity]);
    }

    /// Over a box with glare: a band of light where the turn catches it,
    /// strongest as the face turns toward the light.
    pub(crate) fn paint_glare(&self, node: NodeId, painter: &mut dyn crate::Painter, opacity: f32) {
        let Some(t) = self.transform3d(node).filter(|t| t.glare) else { return };
        let (n, l) = (t.normal(), self.env.light.direction());
        // Halfway between the light and the viewer: the shine is brightest
        // where the face points there.
        let h = {
            let v = [l[0], l[1], l[2] + 1.0];
            let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v.map(|c| c / m)
        };
        let spec = |n: [f32; 3]| (n[0] * h[0] + n[1] * h[1] + n[2] * h[2]).max(0.0).powi(12);
        let strength = ((spec(n) - spec([0.0, 0.0, 1.0])).max(0.0) * 3.0 + (t.rotate_y.sin().abs() + t.rotate_x.sin().abs()) * 0.5).min(1.0)
            * self.env.light.intensity;
        if strength < 0.01 {
            return;
        }
        let r = self.rect(node);
        let radius = self.style(node).border_radius.resolve(Some(r.width.min(r.height))).unwrap_or(0.0);
        // Across the face, toward where the light is.
        let at = (0.5 + (h[0] - n[0]) * 1.4).clamp(-0.2, 1.2);
        let a = 0.35 * strength * opacity;
        painter.push_clip_rounded(r, radius, radius);
        painter.fill_linear_gradient(r, &[(at - 0.45, [1.0, 1.0, 1.0, 0.0]), (at, [1.0, 1.0, 1.0, a]), (at + 0.45, [1.0, 1.0, 1.0, 0.0])], 0.0);
        painter.pop_clip();
    }

    /// The box under `(x, y)` with boxes turned in 3D hit where they are
    /// drawn: the point is taken back through each transform around a box
    /// into its own (unturned) place before it is tested.
    pub(crate) fn hit_through_transforms(&self, x: f32, y: f32) -> Option<NodeId> {
        let index = || self.hits.get_or_init(|| crate::layers::HitIndex::build(self)).hit(self, x, y);
        // The turned boxes that show (from the few that have transforms, not
        // a walk of the tree: this runs on every pointer move), in tree
        // order, so the last painted is tried first below.
        let mut turned: Vec<(Vec<usize>, NodeId)> = self
            .env
            .transforms3d
            .keys()
            .chain(self.env.parallax.keys())
            .copied()
            .filter(|&n| self.contains(n) && self.ancestors(n).all(|a| self.style(a).display != crate::Display::None))
            .map(|n| {
                // Its path from the root, child indices: tree order.
                let mut path: Vec<usize> = self
                    .ancestors(n)
                    .filter_map(|a| self.parent(a).map(|p| self.children(p).iter().position(|&c| c == a).unwrap_or(0)))
                    .collect();
                path.reverse();
                (path, n)
            })
            .collect();
        turned.sort();
        turned.dedup_by_key(|t| t.1);
        let turned: Vec<NodeId> = turned.into_iter().map(|(_, n)| n).collect();
        // The point in a box's own place: back through each transform from
        // the outermost.
        let local = |n: NodeId| -> Option<(f32, f32)> {
            let mut chain: Vec<NodeId> = self.ancestors(n).filter(|&a| self.paint_transform(a).is_some()).collect();
            chain.reverse();
            let mut p = (x, y);
            for a in chain {
                let m = invert_affine(self.paint_transform(a)?.1)?;
                p = apply_affine(m, p.0, p.1);
            }
            Some(p)
        };
        // What floats over everything (a menu, a popover) is hit first, as
        // it is painted last, unless it is itself turned.
        let flat = index();
        if let Some(h) = flat
            && self.ancestors(h).any(|a| self.nodes[a.0].deferred.is_some())
            && !self.ancestors(h).any(|a| self.paint_transform(a).is_some())
        {
            return Some(h);
        }
        for &n in turned.iter().rev() {
            if !self.faces_viewer(n) {
                continue;
            }
            let Some((lx, ly)) = local(n) else { continue };
            let r = self.rect(n);
            if lx < r.x || ly < r.y || lx >= r.x + r.width || ly >= r.y + r.height {
                continue;
            }
            if let Some(h) = self.hit_walk_except(n, lx, ly, None) {
                return Some(h);
            }
        }
        // Not on any turned box where it is drawn: what is under, leaving
        // out a turned box the unturned point would have hit.
        let h = flat?;
        match self.ancestors(h).find(|&a| self.paint_transform(a).is_some()) {
            None => Some(h),
            Some(t) => self.hit_test_except(x, y, t),
        }
    }

    /// Turns the items of `strip` (a box that scrolls sideways) away from
    /// its middle as they move off it, smaller and further back: a 3D
    /// carousel. Kept up after each layout.
    pub fn carousel_3d(&mut self, strip: NodeId) {
        if !self.env.carousels.contains(&strip) {
            self.env.carousels.push(strip);
        }
    }

    /// Rolls the items of `list` (a box that scrolls down) over a drum as
    /// they move off its middle, as a date picker's wheel does. Kept up
    /// after each layout.
    pub fn cylinder_3d(&mut self, list: NodeId) {
        if !self.env.cylinders.contains(&list) {
            self.env.cylinders.push(list);
        }
    }

    /// Shows `node` mirrored below itself, fading into what is behind it,
    /// as on a glossy floor: `strength` its opacity at the edge (0 to 1),
    /// `height` how much of it shows (a fraction of its height). `None`
    /// takes it away.
    pub fn set_reflection(&mut self, node: NodeId, strength_and_height: Option<(f32, f32)>) {
        match strength_and_height {
            Some(v) => self.env.reflections.insert(node, v),
            None => self.env.reflections.remove(&node),
        };
    }

    /// The reflection of `node`, painted after it (from `paint_node`).
    pub(crate) fn paint_reflection(&self, node: NodeId, painter: &mut dyn crate::Painter, opacity: f32) {
        let Some(&(strength, height)) = self.env.reflections.get(&node) else { return };
        if self.env.reflecting.get() {
            return;
        }
        let r = self.rect(node);
        let gap = 2.0;
        let shown = r.height * height.clamp(0.0, 1.0);
        let floor = r.y + r.height + gap;
        // Mirrored about the line under it.
        painter.push_clip(Rect { x: r.x - 2.0, y: floor, width: r.width + 4.0, height: shown });
        painter.push_transform([1.0, 0.0, 0.0, -1.0, 0.0, 2.0 * (r.y + r.height) + gap]);
        self.env.reflecting.set(true);
        self.paint_node_pub(node, painter, opacity * strength.clamp(0.0, 1.0));
        self.env.reflecting.set(false);
        painter.pop_transform();
        // Faded into what is behind it: bands of its colour, thicker down.
        let behind = self.ancestors(node).skip(1).find_map(|a| self.style(a).background).unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let bands = 16;
        for k in 0..bands {
            let t = (k as f32 + 0.5) / bands as f32;
            let band = Rect { x: r.x - 2.0, y: floor + shown * k as f32 / bands as f32, width: r.width + 4.0, height: shown / bands as f32 + 0.5 };
            painter.fill_rect(band, [behind[0], behind[1], behind[2], t.powf(0.7)]);
        }
        painter.pop_clip();
    }

    /// The cylinders' items placed by where they now are.
    pub(crate) fn place_cylinders(&mut self) {
        let lists: Vec<NodeId> = self.env.cylinders.iter().copied().filter(|&s| self.contains(s)).collect();
        self.env.cylinders = lists.clone();
        for l in lists {
            let r = self.rect(l);
            let mid = r.y + r.height / 2.0;
            let radius = (r.height / 2.0).max(1.0);
            for c in self.children(l).to_vec() {
                let cr = self.rect(c);
                // How far round the drum: its offset as an angle.
                let a = ((cr.y + cr.height / 2.0 - mid) / radius).clamp(-1.4, 1.4);
                let t = Transform3d {
                    rotate_x: -a,
                    depth: -(1.0 - a.cos()) * radius,
                    perspective: radius * 4.0,
                    shade: true,
                    hide_backface: true,
                    ..self.transform3d(c).unwrap_or_default()
                };
                self.env.transforms3d.insert(c, t);
            }
        }
    }

    /// The carousels' items placed by where they now are.
    pub(crate) fn place_carousels(&mut self) {
        let strips: Vec<NodeId> = self.env.carousels.iter().copied().filter(|&s| self.contains(s)).collect();
        self.env.carousels = strips.clone();
        for s in strips {
            let r = self.rect(s);
            let mid = r.x + r.width / 2.0;
            for c in self.children(s).to_vec() {
                let cr = self.rect(c);
                let d = ((cr.x + cr.width / 2.0 - mid) / (r.width / 2.0).max(1.0)).clamp(-1.5, 1.5);
                let t = Transform3d {
                    rotate_y: -d * 0.9,
                    depth: -d.abs() * 60.0,
                    scale: 1.0 - d.abs() * 0.12,
                    perspective: 500.0,
                    ..self.transform3d(c).unwrap_or_default()
                };
                self.env.transforms3d.insert(c, t);
            }
        }
    }
}

impl<S: 'static> Input<S> {
    /// Makes `node` lean toward the pointer (or finger) over it, by up to
    /// `max_degrees`, lifted a little with a glare and a shadow, and settle
    /// back flat when it leaves. Reduced motion leaves it flat.
    pub fn tilt_on_hover(&mut self, ui: &mut Ui, node: NodeId, max_degrees: f32) {
        let base = ui.transform3d(node).unwrap_or(Transform3d { glare: true, shadow: true, ..Transform3d::default() });
        ui.set_transform3d(node, base);
        let max = max_degrees.to_radians();
        let tilted = Rc::new(Cell::new(false));
        let t = tilted.clone();
        input_on_move(self, node, move |ui, x, y| {
            if ui.reduce_motion() {
                return;
            }
            let r = ui.rect(node);
            let (u, v) = (((x - r.x) / r.width.max(1.0)) * 2.0 - 1.0, ((y - r.y) / r.height.max(1.0)) * 2.0 - 1.0);
            // The side under the pointer comes toward it, on springs: it
            // follows the pointer with the speed it has.
            ui.spring_to(node, Property::RotateY, -u.clamp(-1.0, 1.0) * max, Spring::snappy());
            ui.spring_to(node, Property::RotateX, v.clamp(-1.0, 1.0) * max, Spring::snappy());
            ui.spring_to(node, Property::Depth, 18.0, Spring::snappy());
            t.set(true);
        });
        self.on_after_dispatch(move |input, ui, _| {
            if !ui.contains(node) {
                return false;
            }
            let over = input.hovered().is_some_and(|h| ui.contains(h) && ui.ancestors(h).any(|a| a == node));
            if tilted.get() && !over {
                tilted.set(false);
                // Back flat, with a little wobble.
                for p in [Property::RotateX, Property::RotateY, Property::Depth] {
                    ui.spring_to(node, p, 0.0, Spring::new(180.0, 14.0, 1.0));
                }
            }
            true
        });
    }
}

impl<S: 'static> Input<S> {
    /// Moves the light (see [`Light`]) with the pointer over the window, so
    /// glare and shading follow it.
    pub fn light_follows_pointer(&mut self) {
        let mut last = (f32::NAN, f32::NAN);
        self.on_after_dispatch(move |input, ui, _| {
            let p = input.pointer_position();
            // No pointer yet: no place to put the light.
            if p != last && p.0.is_finite() && p.1.is_finite() {
                last = p;
                let w = ui.rect(ui.root());
                let (x, y) = ((p.0 / w.width.max(1.0)) * 2.0 - 1.0, (p.1 / w.height.max(1.0)) * 2.0 - 1.0);
                ui.set_light(Light { x: x.clamp(-1.0, 1.0), y: y.clamp(-1.0, 1.0), ..ui.light() });
            }
            true
        });
    }

    /// Leans `node` with the device (see [`Ui::set_device_tilt`]), by up
    /// to `max_degrees`, as a phone's parallax wallpaper does, measured from
    /// how it was held when this began. Reduced motion leaves it flat.
    pub fn tilt_with_device(&mut self, ui: &mut Ui, node: NodeId, max_degrees: f32) {
        if ui.transform3d(node).is_none() {
            ui.set_transform3d(node, Transform3d { glare: true, shadow: true, ..Transform3d::default() });
        }
        let max = max_degrees.to_radians();
        let mut rest: Option<(f32, f32)> = None;
        let mut last = (f32::NAN, f32::NAN);
        self.on_after_dispatch(move |_, ui, _| {
            if !ui.contains(node) {
                return false;
            }
            let (beta, gamma) = ui.device_tilt();
            if (beta, gamma) == last || ui.reduce_motion() {
                return true;
            }
            last = (beta, gamma);
            let (b0, g0) = *rest.get_or_insert((beta, gamma));
            let k = 30f32.to_radians();
            let rx = ((beta - b0).to_radians() / k).clamp(-1.0, 1.0) * max;
            let ry = ((gamma - g0).to_radians() / k).clamp(-1.0, 1.0) * max;
            ui.spring_to(node, Property::RotateX, -rx, Spring::gentle());
            ui.spring_to(node, Property::RotateY, ry, Spring::gentle());
            true
        });
    }
}

fn input_on_move<S: 'static>(input: &mut Input<S>, node: NodeId, mut f: impl FnMut(&mut Ui, f32, f32) + 'static) {
    input.on(node, Phase::Bubble, move |_, ui, e, _| {
        if let Event::PointerMove { x, y } = *e {
            f(ui, x, y);
        }
    });
}


/// A card with two faces, turned over (about its vertical axis) when
/// `flipped` changes: by a click, a tap, Space or Enter on it, or the app.
pub struct FlipCard {
    pub card: NodeId,
    pub front: NodeId,
    pub back: NodeId,
}

/// A card under `parent` (give it a size), `front` and `back` filled by the
/// closures; `label` names it for assistive technology, which hears the
/// face showing as its value and only that face's content.
#[allow(clippy::too_many_arguments)]
pub fn flip_card<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    flipped: Binding<S, bool>,
    front: impl FnOnce(&mut Ui, NodeId),
    back: impl FnOnce(&mut Ui, NodeId),
) -> FlipCard {
    let card = ui.add_box(parent, style! { position: relative; cursor: pointer; });
    let face = style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; box-sizing: border-box; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; };
    let f = ui.add_box(card, face.clone());
    let b = ui.add_box(card, face);
    for n in [f, b] {
        ui.token_style(n, &[("background", crate::environment::Token::Raised), ("color", crate::environment::Token::Text)]);
    }
    front(ui, f);
    back(ui, b);
    ui.set_transform3d(card, Transform3d { shadow: true, glare: true, ..Transform3d::default() });
    ui.set_transform3d(f, Transform3d { hide_backface: true, ..Transform3d::default() });
    // The back is turned half round within the card: when the card turns
    // over, it faces the viewer the right way round.
    ui.set_transform3d(b, Transform3d { rotate_y: std::f32::consts::PI, hide_backface: true, ..Transform3d::default() });
    let mut sem = Semantics::new(Role::Button).label(label);
    sem.description = Some("Flips the card".into());
    ui.set_semantics(card, sem);
    ui.focus_visible_style(card, "box-shadow: 0 0 0 2px #93c5fd");
    let fl = flipped.clone();
    input.on_click(card, move |s, _, _| {
        let now = !fl.get(s);
        fl.set(s, now);
    });
    let show = move |ui: &mut Ui, over: bool, animate: bool| {
        let to = if over { std::f32::consts::PI } else { 0.0 };
        if animate && !ui.reduce_motion() {
            // A spring: turned back half way, it reverses from where it is.
            ui.spring_to(card, Property::RotateY, to, Spring::new(120.0, 16.0, 1.0));
        } else if let Some(mut t) = ui.transform3d(card) {
            t.rotate_y = to;
            ui.set_transform3d(card, t);
        }
        // Only the face showing is read.
        ui.semantics_mut(f).hidden = over;
        ui.semantics_mut(b).hidden = !over;
        ui.semantics_mut(card).value = Some(if over { "Back" } else { "Front" }.into());
    };
    let mut shown = flipped.get(state);
    show(ui, shown, false);
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(card) {
            return false;
        }
        let now = flipped.get(s);
        if now != shown {
            shown = now;
            show(ui, now, true);
        }
        true
    });
    FlipCard { card, front: f, back: b }
}

/// The projective map from the box `(x, y, w, h)` to the quad `q` (its
/// corners clockwise from the top left), Heckbert's square-to-quad.
pub(crate) fn homography_to(q: [(f32, f32); 4], (x0, y0, w, h): (f32, f32, f32, f32)) -> [f32; 9] {
    let [(ax, ay), (bx, by), (cx, cy), (dx, dy)] = q.map(|(x, y)| (x as f64, y as f64));
    let (sx, sy) = (ax - bx + cx - dx, ay - by + cy - dy);
    let (dx1, dy1, dx2, dy2) = (bx - cx, by - cy, dx - cx, dy - cy);
    let den = dx1 * dy2 - dx2 * dy1;
    let (g, hh) = if den.abs() < 1e-12 { (0.0, 0.0) } else { ((sx * dy2 - dx2 * sy) / den, (dx1 * sy - sx * dy1) / den) };
    let (a, b, c) = (bx - ax + g * bx, dx - ax + hh * dx, ax);
    let (d, e, f) = (by - ay + g * by, dy - ay + hh * dy, ay);
    // After (x, y) ↦ ((x − x0) / w, (y − y0) / h).
    let (w, h, x0, y0) = (w as f64, h as f64, x0 as f64, y0 as f64);
    [
        a / w,
        b / h,
        c - a * x0 / w - b * y0 / h,
        d / w,
        e / h,
        f - d * x0 / w - e * y0 / h,
        g / w,
        hh / h,
        1.0 - g * x0 / w - hh * y0 / h,
    ]
    .map(|v| v as f32)
}

/// Where a projective map `h` takes `(x, y)`.
pub fn apply_homography(h: [f32; 9], x: f32, y: f32) -> (f32, f32) {
    let w = h[6] * x + h[7] * y + h[8];
    ((h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w)
}

/// Where a 2D transform `m` takes `(x, y)`.
pub fn apply_affine(m: [f32; 6], x: f32, y: f32) -> (f32, f32) {
    (m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5])
}

/// The 2D transform undoing `m`, if it can be undone.
pub fn invert_affine(m: [f32; 6]) -> Option<[f32; 6]> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-9 {
        return None;
    }
    let (a, b, c, d) = (m[3] / det, -m[1] / det, -m[2] / det, m[0] / det);
    Some([a, b, c, d, -(a * m[4] + c * m[5]), -(b * m[4] + d * m[5])])
}

/// Where a light falls from, for the glare and shading of boxes turned in
/// 3D: `x` and `y` lean it right and down of straight on, as a pointer
/// over the window would (−1 to 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Light {
    pub x: f32,
    pub y: f32,
    /// How strong its glare and shading are (1 by default).
    pub intensity: f32,
}

impl Default for Light {
    fn default() -> Self {
        // From the top left, as a window's light usually is.
        Self { x: -0.4, y: -0.6, intensity: 1.0 }
    }
}

impl Light {
    /// Its direction, a unit vector toward it.
    pub fn direction(&self) -> [f32; 3] {
        let v = [self.x, self.y, 1.0];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|c| c / n)
    }
}

/// The depth of field in effect inside a box (see
/// [`Ui::set_depth_of_field`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthOfField(pub Option<(f32, f32)>);

/// Which way a card was swiped off a [`card_deck`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwipeDirection {
    Left,
    Right,
}

/// A stack of cards, the top one dragged off to either side (it tips as it
/// goes, and flies off past a third of its width, or springs back), the
/// next coming up from under it. The cards are `cards`' first entries;
/// a swiped one is taken out of it and `on_swipe` hears which way. The
/// arrow keys swipe the top card, and assistive technology is offered
/// "Swipe left" and "Swipe right" on it.
pub struct CardDeck {
    pub deck: NodeId,
}

/// Cards showing at once.
const DECK_DEPTH: usize = 3;

pub fn card_deck<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    parent: NodeId,
    label: &str,
    cards: Binding<S, Vec<String>>,
    build: impl Fn(&mut Ui, NodeId, &str) + 'static,
    on_swipe: impl FnMut(&mut S, &str, SwipeDirection) + 'static,
) -> CardDeck {
    use std::cell::RefCell;
    let deck = ui.add_box(parent, style! { position: relative; });
    ui.set_semantics(deck, Semantics::new(Role::List).label(label).tab_index(0));
    ui.focus_visible_style(deck, "box-shadow: 0 0 0 2px #93c5fd");
    // The cards built, top first.
    let built: Rc<RefCell<Vec<(String, NodeId)>>> = Rc::default();
    let drag: Rc<Cell<Option<(f32, f32)>>> = Rc::default();
    let on_swipe = Rc::new(RefCell::new(on_swipe));
    // The top card off to `dir`: out of the state, flying off.
    let fly = {
        let (built, cards, on_swipe) = (built.clone(), cards.clone(), on_swipe.clone());
        Rc::new(move |s: &mut S, ui: &mut Ui, dir: SwipeDirection| {
            let top = {
                let mut b = built.borrow_mut();
                if b.is_empty() {
                    return;
                }
                b.remove(0)
            };
            let w = ui.rect(deck).width.max(1.0);
            let sign = if dir == SwipeDirection::Right { 1.0 } else { -1.0 };
            let spring = Spring::new(120.0, 18.0, 1.0);
            ui.spring_to(top.1, Property::Left, sign * w * 1.6, spring);
            if let Some(t) = ui.transform3d(top.1) {
                ui.set_transform3d(top.1, Transform3d { rotate_z: t.rotate_z, ..t });
            }
            ui.remove_with(top.1, crate::transitions::Transition::Fade, 0.4);
            let mut list = cards.get(s);
            if let Some(i) = list.iter().position(|c| *c == top.0) {
                list.remove(i);
            }
            cards.set(s, list);
            (on_swipe.borrow_mut())(s, &top.0, dir);
        })
    };
    // Dragged: the top card follows, tipping; let go, it goes or comes back.
    let (b, d, f) = (built.clone(), drag.clone(), fly.clone());
    input.on(deck, Phase::Bubble, move |s, ui, e, cx| {
        let Some(&(_, top)) = b.borrow().first() else { return };
        match *e {
            Event::PointerDown { x, y, .. } => {
                d.set(Some((x, y)));
                cx.capture_pointer();
            }
            Event::PointerMove { x, y } => {
                if let Some((x0, y0)) = d.get() {
                    let w = ui.rect(deck).width.max(1.0);
                    let (dx, dy) = (x - x0, y - y0);
                    ui.update_style(top, |st| {
                        st.left = crate::Length::Px(dx);
                        st.top = crate::Length::Px(dy * 0.3);
                    });
                    if let Some(t) = ui.transform3d(top) {
                        ui.set_transform3d(top, Transform3d { rotate_z: dx / w * 0.35, ..t });
                    }
                }
            }
            Event::PointerUp { x, .. } => {
                if let Some((x0, _)) = d.take() {
                    let w = ui.rect(deck).width.max(1.0);
                    let dx = x - x0;
                    if dx.abs() > w / 3.0 {
                        f(s, ui, if dx > 0.0 { SwipeDirection::Right } else { SwipeDirection::Left });
                    } else {
                        let spring = Spring::snappy();
                        ui.spring_to(top, Property::Left, 0.0, spring);
                        ui.spring_to(top, Property::Top, 0.0, spring);
                        if let Some(t) = ui.transform3d(top) {
                            ui.set_transform3d(top, Transform3d { rotate_z: 0.0, ..t });
                        }
                    }
                }
            }
            _ => {}
        }
    });
    let f = fly.clone();
    input.on_key(deck, move |s, ui, key, _, _| {
        let dir = match key {
            "ArrowLeft" => SwipeDirection::Left,
            "ArrowRight" => SwipeDirection::Right,
            _ => return false,
        };
        f(s, ui, dir);
        true
    });
    // The cards showing kept as the state's first ones, stacked.
    let build = Rc::new(build);
    let mut offered: Option<NodeId> = None;
    // The cards as last stacked (a fling takes one out between syncs).
    let mut stacked: Vec<NodeId> = vec![];
    input.on_after_dispatch(move |input, ui, s| {
        if !ui.contains(deck) {
            return false;
        }
        let want: Vec<String> = cards.get(s).into_iter().take(DECK_DEPTH).collect();
        let changed = {
            let mut b = built.borrow_mut();
            b.retain(|(id, n)| want.contains(id) && ui.contains(*n));
            for id in &want {
                if !b.iter().any(|(k, _)| k == id) {
                    let card = ui.add_box(deck, style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; box-sizing: border-box; border-radius: 14px; overflow: hidden; });
                    ui.token_style(card, &[("background", crate::environment::Token::Raised), ("color", crate::environment::Token::Text)]);
                    build(ui, card, id);
                    ui.set_semantics(card, Semantics::new(Role::ListItem));
                    ui.combine_accessibility(card);
                    ui.set_transform3d(card, Transform3d { shadow: true, scale: 0.9, ..Transform3d::default() });
                    b.push((id.clone(), card));
                }
            }
            // In the state's order.
            b.sort_by_key(|(id, _)| want.iter().position(|w| w == id));
            let now: Vec<NodeId> = b.iter().map(|&(_, n)| n).collect();
            let changed = now != stacked;
            stacked = now;
            changed
        };
        if changed {
            let b = built.borrow();
            let n = b.len();
            // Each to the front of the deck's children, top card first: the
            // deepest ends up first, the top card last of them, and a card
            // flying off after it, over everything.
            for &(_, card) in b.iter() {
                ui.move_child(deck, card, 0);
            }
            for (k, &(_, card)) in b.iter().enumerate() {
                // Those under the top one smaller and lower.
                ui.spring_to(card, Property::Top, k as f32 * 10.0, Spring::snappy());
                if let Some(t) = ui.transform3d(card) {
                    let scale = 1.0 - k as f32 * 0.05;
                    if (t.scale - scale).abs() > 1e-4 {
                        ui.set_transform3d(card, Transform3d { scale, ..t });
                    }
                }
                ui.semantics_mut(card).hidden = k > 0;
                ui.semantics_mut(card).position = Some((k + 1, n));
            }
        }
        // The top card offers the swipes to assistive technology.
        let top = built.borrow().first().map(|&(_, n)| n);
        if top != offered {
            offered = top;
            if let Some(t) = top {
                let f = fly.clone();
                input.on_accessibility_action(ui, t, &["Swipe left", "Swipe right"], move |s, ui, a| {
                    f(s, ui, if a == "Swipe right" { SwipeDirection::Right } else { SwipeDirection::Left });
                });
            }
        }
        true
    });
    CardDeck { deck }
}

/// A cube of faces turned to the one `selected` names, about its vertical
/// axis: each face a side of the cube, the turn sprung (it can be sent
/// on before it arrives). The arrow keys turn it a face at a time;
/// assistive technology reads the face showing, "2 of 4".
pub struct Cube {
    pub cube: NodeId,
    pub faces: Vec<NodeId>,
}

/// A cube under `parent` (give it a size) of `faces` faces, each filled by
/// `fill` (given its index), turned to `selected`.
#[allow(clippy::too_many_arguments)]
pub fn cube<S: 'static>(
    ui: &mut Ui,
    input: &mut Input<S>,
    state: &S,
    parent: NodeId,
    label: &str,
    faces: usize,
    selected: Binding<S, usize>,
    fill: impl Fn(&mut Ui, NodeId, usize),
) -> Cube {
    let n = faces.max(1);
    let cube = ui.add_box(parent, style! { position: relative; });
    ui.set_semantics(cube, Semantics::new(Role::Group).label(label).tab_index(0));
    ui.focus_visible_style(cube, "box-shadow: 0 0 0 2px #93c5fd");
    let face_nodes: Vec<NodeId> = (0..n)
        .map(|k| {
            let f = ui.add_box(cube, style! { position: absolute; left: 0px; top: 0px; width: 100%; height: 100%; box-sizing: border-box; border-radius: 10px; overflow: hidden; });
            ui.token_style(f, &[("background", crate::environment::Token::Raised), ("color", crate::environment::Token::Text)]);
            fill(ui, f, k);
            ui.set_semantics(f, Semantics::new(Role::Group).label(format!("{} of {n}", k + 1)));
            f
        })
        .collect();
    // What turns: an angle, on a node of its own that is never drawn.
    let driver = ui.add_box(cube, style! { display: none; });
    let step = std::f32::consts::TAU / n as f32;
    ui.set_transform3d(driver, Transform3d { rotate_y: -(selected.get(state).min(n - 1) as f32) * step, ..Transform3d::default() });
    let sel = selected.clone();
    input.on_key(cube, move |s, _, key, _, _| {
        let k = sel.get(s);
        let to = match key {
            "ArrowRight" => (k + 1) % n,
            "ArrowLeft" => (k + n - 1) % n,
            _ => return false,
        };
        sel.set(s, to);
        true
    });
    let faces2 = face_nodes.clone();
    let mut aimed: Option<usize> = None;
    input.on_after_dispatch(move |_, ui, s| {
        if !ui.contains(cube) {
            return false;
        }
        let want = selected.get(s).min(n - 1);
        if aimed != Some(want) {
            // The nearest way round to it.
            let now = ui.transform3d(driver).map_or(0.0, |t| t.rotate_y);
            let mut target = -(want as f32) * step;
            while target - now > std::f32::consts::PI {
                target -= std::f32::consts::TAU;
            }
            while now - target > std::f32::consts::PI {
                target += std::f32::consts::TAU;
            }
            ui.spring_to(driver, Property::RotateY, target, Spring::new(140.0, 20.0, 1.0));
            aimed = Some(want);
        }
        // Each face where the angle puts it round the cube's middle.
        let angle = ui.transform3d(driver).map_or(0.0, |t| t.rotate_y);
        let r = ui.rect(cube);
        // From the middle to a face: the apothem of the cube's polygon.
        let d = r.width / 2.0 / (std::f32::consts::PI / n as f32).tan().max(1e-3);
        for (k, &f) in faces2.iter().enumerate() {
            let a = angle + k as f32 * step;
            ui.set_transform3d(
                f,
                Transform3d {
                    rotate_y: a,
                    translate_x: d * a.sin(),
                    depth: d * (a.cos() - 1.0),
                    perspective: r.width.max(1.0) * 2.5,
                    hide_backface: true,
                    shade: true,
                    ..Transform3d::default()
                },
            );
            ui.semantics_mut(f).hidden = k != want;
        }
        // Painted far faces first, so the nearest covers them as they turn.
        let mut by_depth: Vec<(f32, NodeId)> = faces2.iter().enumerate().map(|(k, &f)| ((angle + k as f32 * step).cos(), f)).collect();
        by_depth.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (i, &(_, f)) in by_depth.iter().enumerate() {
            if ui.children(cube).get(i) != Some(&f) {
                ui.move_child(cube, f, i);
            }
        }
        ui.semantics_mut(cube).value = Some(format!("{} of {n}", want + 1));
        true
    });
    Cube { cube, faces: face_nodes }
}

/// A split-flap display: a character a cell, each cut across its middle,
/// whose top half falls over its hinge to show the next one, as on a
/// station's board or a flip clock. Made by [`split_flap`]; given what to
/// show with [`SplitFlap::set`].
pub struct SplitFlap {
    pub node: NodeId,
    cells: Vec<FlapCell>,
    shown: Vec<char>,
}

/// One character's halves: the new top and the old bottom stay still; the
/// old top falls away over the middle, then the new bottom falls into place.
struct FlapCell {
    top: NodeId,
    bottom: NodeId,
    falling: NodeId,
    landing: NodeId,
}

/// A split-flap display of `cells` characters, each `width` × `height`,
/// under `parent`; `label` names it for assistive technology, which hears
/// what it shows as its value (the halves are hidden from it).
pub fn split_flap(ui: &mut Ui, parent: NodeId, label: &str, cells: usize, width: f32, height: f32) -> SplitFlap {
    use crate::environment::Token;
    let node = ui.add_box(parent, style! { display: flex; gap: 4px; flex-shrink: 0; });
    let mut sem = Semantics::new(Role::Status).label(label);
    sem.live = true;
    ui.set_semantics(node, sem);
    let half = height / 2.0;
    let mut out = vec![];
    for _ in 0..cells {
        let cell = ui.add_box(node, style! { position: relative; border-radius: 6px; overflow: hidden; flex-shrink: 0; });
        ui.update_style(cell, |s| {
            s.width = crate::Length::Px(width);
            s.height = crate::Length::Px(height);
        });
        // A half: a box as wide as the cell and half as tall, showing the
        // top or the bottom of a character as tall as the cell.
        let mut half_box = |lower: bool| {
            let h = ui.add_box(cell, style! { position: absolute; left: 0px; overflow: hidden; });
            ui.token_style(h, &[("background", Token::Raised), ("color", Token::Text)]);
            ui.update_style(h, |s| {
                s.width = crate::Length::Px(width);
                s.height = crate::Length::Px(half);
                s.top = crate::Length::Px(if lower { half } else { 0.0 });
            });
            let glyph = ui.add_box(h, style! { position: absolute; left: 0px; display: flex; align-items: center; justify-content: center; font-weight: bold; });
            ui.update_style(glyph, |s| {
                s.width = crate::Length::Px(width);
                s.height = crate::Length::Px(height);
                s.top = crate::Length::Px(if lower { -half } else { 0.0 });
                s.font_size = height * 0.62;
            });
            ui.add_text(glyph, " ");
            ui.semantics_mut(h).hidden = true;
            h
        };
        let top = half_box(false);
        let bottom = half_box(true);
        let falling = half_box(false);
        let landing = half_box(true);
        // The seam.
        let seam = ui.add_box(cell, style! { position: absolute; left: 0px; height: 1px; background: rgba(0,0,0,0.45); });
        ui.update_style(seam, |s| {
            s.width = crate::Length::Px(width);
            s.top = crate::Length::Px(half - 0.5);
        });
        // The falling half turns on its lower edge, the landing one on its
        // upper; both darken as they turn from the light.
        let hinge = |origin_y: f32, rotate_x: f32| Transform3d { origin: (0.5, origin_y), rotate_x, perspective: height * 6.0, shade: true, ..Transform3d::default() };
        ui.set_transform3d(falling, hinge(1.0, -FLAT));
        ui.set_transform3d(landing, hinge(0.0, 0.0));
        out.push(FlapCell { top, bottom, falling, landing });
    }
    SplitFlap { node, cells: out, shown: vec![' '; cells] }
}

/// A quarter turn, a little short of edge-on so the box stays drawable.
const FLAT: f32 = std::f32::consts::FRAC_PI_2 * 0.999;

impl SplitFlap {
    /// Shows `text` (padded or cut to the cells): each cell whose character
    /// changes flips, the cells one after another, left to right.
    pub fn set(&mut self, ui: &mut Ui, text: &str) {
        use crate::animation::{Animation, Value};
        use threers_math::easing::Easing;
        let want: Vec<char> = text.chars().chain(std::iter::repeat(' ')).take(self.cells.len()).collect();
        if want == self.shown {
            return;
        }
        ui.semantics_mut(self.node).value = Some(text.trim().to_string());
        let glyph = |ui: &Ui, half: NodeId| ui.children(ui.children(half)[0])[0];
        let fall = 0.16;
        let mut order = 0.0;
        for (i, cell) in self.cells.iter().enumerate() {
            let (old, new) = (self.shown[i].to_string(), want[i].to_string());
            if old == new {
                continue;
            }
            for (half, s) in [(cell.top, &new), (cell.bottom, &old), (cell.falling, &old), (cell.landing, &new)] {
                let g = glyph(ui, half);
                ui.set_text(g, s.as_str());
            }
            if ui.reduce_motion() {
                for (n, angle) in [(cell.falling, -FLAT), (cell.landing, 0.0)] {
                    let t = ui.transform3d(n).unwrap_or_default();
                    ui.set_transform3d(n, Transform3d { rotate_x: angle, ..t });
                }
                continue;
            }
            let delay = order * 0.04;
            order += 1.0;
            let mut down = Animation::new(Property::RotateX, Value::Number(0.0), Value::Number(-FLAT), fall).easing(Easing::QuadIn);
            down.delay = delay;
            ui.animate(cell.falling, down);
            let mut land = Animation::new(Property::RotateX, Value::Number(FLAT), Value::Number(0.0), fall * 1.2).easing(Easing::BounceOut);
            land.delay = delay + fall;
            ui.animate(cell.landing, land);
        }
        self.shown = want;
    }

    /// What it shows (or is turning to).
    pub fn text(&self) -> String {
        self.shown.iter().collect()
    }
}

#[cfg(test)]
mod split_flap_tests {
    use super::*;
    use crate::FixedWidthText;

    #[test]
    fn a_changed_character_falls_and_lands() {
        let mut ui = Ui::new();
        let root = ui.root();
        let mut board = split_flap(&mut ui, root, "Departures", 3, 40.0, 60.0);
        board.set(&mut ui, "AB");
        ui.layout(400.0, 200.0, &mut FixedWidthText(7.0));
        assert_eq!(board.text(), "AB ");
        assert_eq!(ui.semantics(board.node).and_then(|s| s.value.as_deref()), Some("AB"));
        for _ in 0..60 {
            ui.step(1.0 / 60.0);
        }
        // Mid-flip: the old top half on its way down, the new bottom not
        // yet down.
        board.set(&mut ui, "AC");
        let c = &board.cells[1];
        for _ in 0..6 {
            ui.step(1.0 / 60.0);
        }
        let falling = ui.transform3d(c.falling).unwrap().rotate_x;
        let landing = ui.transform3d(c.landing).unwrap().rotate_x;
        assert!(falling < -0.2 && falling > -FLAT, "falling: {falling}");
        assert!((landing - FLAT).abs() < 1e-3, "waiting to land: {landing}");
        // The first cell kept its character, and did not move.
        assert!((ui.transform3d(board.cells[0].falling).unwrap().rotate_x + FLAT).abs() < 1e-3);
        for _ in 0..60 {
            ui.step(1.0 / 60.0);
        }
        assert!(ui.transform3d(c.landing).unwrap().rotate_x.abs() < 1e-3, "landed");
        // What each half shows: the new top and, landed, the new bottom.
        let text = |half: NodeId| ui.text_lines_of(ui.children(ui.children(half)[0])[0]).concat();
        assert_eq!((text(c.top), text(c.landing)), ("C".to_string(), "C".to_string()));
    }
}

#[cfg(test)]
mod backface_tests {
    use super::*;
    use crate::style;

    #[test]
    fn a_face_turned_square_to_one_side_shows_its_back_and_is_hidden() {
        let mut ui = Ui::new();
        let root = ui.root();
        let f = ui.add_box(root, style! { width: 200px; height: 100px; });
        ui.layout(400.0, 300.0, &mut crate::FixedWidthText(7.0));
        // As the cube puts its faces: turned by `a` about its middle, `d`
        // out from the cube's centre.
        let d = 100.0;
        let face = |a: f32| Transform3d { rotate_y: a, translate_x: d * a.sin(), depth: d * (a.cos() - 1.0), perspective: 500.0, hide_backface: true, ..Transform3d::default() };
        for (a, shows) in [(0.0, true), (0.6, true), (-0.6, true), (std::f32::consts::FRAC_PI_2, false), (-std::f32::consts::FRAC_PI_2, false), (std::f32::consts::PI, false)] {
            ui.set_transform3d(f, face(a));
            assert_eq!(ui.faces_viewer(f), shows, "turned {a}");
        }
    }
}
