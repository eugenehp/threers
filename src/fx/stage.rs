//! A set of elements over a scene: [`Stage`].

use std::any::Any;
use std::sync::Arc;

use super::component::{ComponentInput, Layout};
use super::surface::{Backdrop, Surface};
use super::{Component, Cursor, CursorDecor, Effect, EffectInput, Shape};
use crate::core::{ObjectId, ObjectKind};
use crate::materials::Material;
use crate::math::Vector2;
use crate::overlay::Overlay;
use crate::scene::Scene;

/// An element on a [`Stage`]. Stays valid until the element is removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ElementId(usize);

/// What happened to an element this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// The pointer moved onto it.
    Enter,
    /// The pointer moved off it.
    Leave,
    /// The pointer went down over it.
    Press,
    /// The pointer came up after a press on it, wherever it is now.
    Release,
    /// The pointer came up over it after a press on it.
    Click,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub element: ElementId,
    pub kind: EventKind,
}

/// One frame's input, in CSS pixels.
#[derive(Debug, Clone, Copy)]
pub struct StageInput {
    /// The viewport's size.
    pub viewport: Vector2,
    /// Device pixels per CSS pixel; the overlay is in device pixels.
    pub dpr: f32,
    /// Seconds since the last frame.
    pub dt: f32,
    /// The pointer, or `None` when there is none (it left the window, a
    /// touch lifted).
    pub pointer: Option<Vector2>,
    pub pointer_down: bool,
}

struct Element {
    component: Box<dyn Component>,
    effect: Box<dyn Effect>,
    quad: ObjectId,
    hovered: bool,
    pressed: bool,
}

/// Elements — a [`Component`] each, wearing an [`Effect`] — over a scene.
///
/// Each frame [`update`](Self::update) lays the elements out, hit-tests the
/// pointer and raises [`Event`]s, steps every effect, puts each effect's
/// [`Surface`] on its quad in the scene, and paints the overlay in layers:
/// every component's content, then every effect's detail, then the cursor —
/// once, carrying whatever the effects left on it — then what effects lay
/// over the cursor. Elements are drawn in the order they were added.
///
/// ```no_run
/// use threers::fx::{Button, Stage, Venom};
/// use threers::sketch::{sketch, Frame, SketchOptions};
///
/// sketch(SketchOptions::new("fx"), |s| {
///     let mut stage = Stage::new();
///     let button = stage.add(s.scene, Button::new("hello"), Venom::default());
///     move |f: &mut Frame| {
///         for event in stage.sketch_frame(f) {
///             println!("{event:?}");
///         }
///     }
/// });
/// ```
pub struct Stage {
    elements: Vec<Option<Element>>,
    /// Quads of removed elements, reused by the next added.
    spare: Vec<ObjectId>,
    cursor: Cursor,
    backdrop: Option<(Backdrop, ObjectId)>,
    time: f32,
    was_down: bool,
}

impl Default for Stage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage {
    /// A stage with the glove cursor and no backdrop.
    pub fn new() -> Self {
        Self { elements: Vec::new(), spare: Vec::new(), cursor: Cursor::default(), backdrop: None, time: 0.0, was_down: false }
    }

    pub fn with_cursor(mut self, cursor: Cursor) -> Self {
        self.cursor = cursor;
        self
    }

    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    pub fn set_cursor(&mut self, cursor: Cursor) {
        self.cursor = cursor;
    }

    /// Draw `backdrop` behind everything, filling the viewport.
    pub fn set_backdrop(&mut self, scene: &mut Scene, backdrop: Backdrop) {
        let material = backdrop.material(Vector2::new(1.0, 1.0), 1.0);
        match &mut self.backdrop {
            Some((b, _)) => *b = backdrop,
            None => {
                let id = scene.add_mesh(Surface::quad(), Material::Shader(material));
                if let Some(obj) = scene.get_mut(id) {
                    obj.render_order = -1_000_000;
                }
                self.backdrop = Some((backdrop, id));
            }
        }
    }

    /// Add an element: `component` wearing `effect`, drawn after every
    /// element already on the stage.
    pub fn add(&mut self, scene: &mut Scene, component: impl Component, effect: impl Effect) -> ElementId {
        let quad = self.spare.pop().unwrap_or_else(|| {
            let placeholder = Surface::new(Shape::circle(Vector2::ZERO, 1.0), super::Look::LACQUER);
            let id = scene.add_mesh(Surface::quad(), Material::Shader(placeholder.material(Vector2::new(1.0, 1.0), 1.0)));
            if let Some(obj) = scene.get_mut(id) {
                // late, so the elements blend over the scene, in order
                obj.render_order = 1_000_000 + self.elements.len() as i32;
                obj.visible = false;
            }
            id
        });
        let element = Element { component: Box::new(component), effect: Box::new(effect), quad, hovered: false, pressed: false };
        if let Some(i) = self.elements.iter().position(|e| e.is_none()) {
            self.elements[i] = Some(element);
            ElementId(i)
        } else {
            self.elements.push(Some(element));
            ElementId(self.elements.len() - 1)
        }
    }

    /// Take an element off the stage. Its quad is hidden and kept for the
    /// next one added.
    pub fn remove(&mut self, scene: &mut Scene, id: ElementId) {
        if let Some(e) = self.elements.get_mut(id.0).and_then(|e| e.take()) {
            if let Some(obj) = scene.get_mut(e.quad) {
                obj.visible = false;
            }
            self.spare.push(e.quad);
        }
    }

    /// Every element on the stage, in drawing order.
    pub fn ids(&self) -> impl Iterator<Item = ElementId> + '_ {
        self.elements.iter().enumerate().filter(|(_, e)| e.is_some()).map(|(i, _)| ElementId(i))
    }

    fn get(&self, id: ElementId) -> Option<&Element> {
        self.elements.get(id.0).and_then(|e| e.as_ref())
    }

    fn get_mut(&mut self, id: ElementId) -> Option<&mut Element> {
        self.elements.get_mut(id.0).and_then(|e| e.as_mut())
    }

    /// The element's component, if it is a `T`.
    pub fn component<T: Component>(&self, id: ElementId) -> Option<&T> {
        let c: &dyn Any = self.get(id)?.component.as_ref();
        c.downcast_ref()
    }

    pub fn component_mut<T: Component>(&mut self, id: ElementId) -> Option<&mut T> {
        let c: &mut dyn Any = self.get_mut(id)?.component.as_mut();
        c.downcast_mut()
    }

    /// The element's effect, if it is a `T` — to read its state or change
    /// its style while it runs.
    pub fn effect<T: Effect>(&self, id: ElementId) -> Option<&T> {
        let e: &dyn Any = self.get(id)?.effect.as_ref();
        e.downcast_ref()
    }

    pub fn effect_mut<T: Effect>(&mut self, id: ElementId) -> Option<&mut T> {
        let e: &mut dyn Any = self.get_mut(id)?.effect.as_mut();
        e.downcast_mut()
    }

    /// Swap the element's effect for another; the component stays.
    pub fn set_effect(&mut self, id: ElementId, effect: impl Effect) {
        if let Some(e) = self.get_mut(id) {
            e.effect = Box::new(effect);
        }
    }

    /// The element's shape as of the last update.
    pub fn shape(&self, id: ElementId) -> Option<Shape> {
        self.get(id).map(|e| e.component.shape())
    }

    /// Whether the system cursor should be hidden: the stage draws its own
    /// and the pointer is present.
    pub fn hides_system_cursor(&self, input: &StageInput) -> bool {
        input.pointer.is_some() && self.cursor.hides_system_cursor()
    }

    /// One frame: lay out, hit-test, step the effects, put their surfaces in
    /// the scene and paint the overlay. Returns what happened.
    pub fn update(&mut self, scene: &mut Scene, overlay: &mut Overlay, input: StageInput) -> Vec<Event> {
        self.run(scene, overlay, input, None)
    }

    /// [`update`](Self::update) with a pointer of its own for each element —
    /// for scripted demos and tests, where several elements should be caught
    /// in different moments at once. Each element's cursor is drawn with it.
    /// Events are not raised.
    pub fn update_each(
        &mut self,
        scene: &mut Scene,
        overlay: &mut Overlay,
        input: StageInput,
        pointer_for: &dyn Fn(ElementId) -> Option<Vector2>,
    ) {
        self.run(scene, overlay, input, Some(pointer_for));
    }

    fn run(
        &mut self,
        scene: &mut Scene,
        overlay: &mut Overlay,
        input: StageInput,
        pointer_for: Option<&dyn Fn(ElementId) -> Option<Vector2>>,
    ) -> Vec<Event> {
        let dt = input.dt.clamp(0.0, 0.25);
        self.time += dt;
        let mut events = Vec::new();
        let down_edge = input.pointer_down && !self.was_down;
        let up_edge = !input.pointer_down && self.was_down;
        self.was_down = input.pointer_down;
        let (cursor, time) = (self.cursor, self.time);

        if let Some((backdrop, id)) = &self.backdrop {
            set_material(scene, *id, Material::Shader(backdrop.material(input.viewport, input.dpr)));
        }

        // ---- lay out, hit-test, step ----
        for (i, slot) in self.elements.iter_mut().enumerate() {
            let Some(e) = slot else { continue };
            let id = ElementId(i);
            e.component.layout(&mut Layout { viewport: input.viewport, overlay });
            let pointer = match pointer_for {
                Some(f) => f(id),
                None => input.pointer,
            };
            if pointer_for.is_none() && e.component.interactive() {
                let over = pointer.is_some_and(|p| e.component.shape().contains(p));
                let mut raise = |kind| events.push(Event { element: id, kind });
                if over != e.hovered {
                    raise(if over { EventKind::Enter } else { EventKind::Leave });
                    e.hovered = over;
                }
                if down_edge && over {
                    e.pressed = true;
                    raise(EventKind::Press);
                }
                if up_edge && e.pressed {
                    e.pressed = false;
                    raise(EventKind::Release);
                    if over {
                        raise(EventKind::Click);
                    }
                }
            }
            e.component.update(&ComponentInput { dt, hovered: e.hovered, pressed: e.pressed });
            let shape = e.component.shape();
            e.effect.update(&EffectInput { shape: &shape, dt, time, pointer, cursor, hovered: e.hovered, pressed: e.pressed });
            match e.effect.surface() {
                Some(surface) => {
                    set_material(scene, e.quad, Material::Shader(surface.material(input.viewport, input.dpr)));
                    set_visible(scene, e.quad, true);
                }
                None => set_visible(scene, e.quad, false),
            }
        }

        // ---- paint, in layers ----
        let dpr = input.dpr;
        let live: Vec<&Element> = self.elements.iter().flatten().collect();
        for e in &live {
            e.component.paint(overlay, dpr);
        }
        match pointer_for {
            None => {
                for e in &live {
                    e.effect.paint(overlay, dpr);
                }
                if let Some(p) = input.pointer {
                    let mut decor = CursorDecor::default();
                    for e in &live {
                        e.effect.decorate_cursor(&mut decor);
                    }
                    cursor.paint(overlay, p, &decor, time, dpr);
                }
                for e in &live {
                    e.effect.paint_over_cursor(overlay, dpr);
                }
            }
            Some(f) => {
                for (i, slot) in self.elements.iter().enumerate() {
                    let Some(e) = slot else { continue };
                    e.effect.paint(overlay, dpr);
                    if let Some(p) = f(ElementId(i)) {
                        let mut decor = CursorDecor::default();
                        e.effect.decorate_cursor(&mut decor);
                        cursor.paint(overlay, p, &decor, time, dpr);
                    }
                    e.effect.paint_over_cursor(overlay, dpr);
                }
            }
        }
        events
    }

    /// [`update`](Self::update) from a [`sketch`](crate::sketch) frame: its
    /// pointer, clock and overlay, and the system cursor hidden while the
    /// stage draws its own.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sketch_frame(&mut self, f: &mut crate::sketch::Frame) -> Vec<Event> {
        let dpr = f.scale_factor;
        let input = StageInput {
            viewport: Vector2::new(f.width as f32 / dpr, f.height as f32 / dpr),
            dpr,
            dt: f.delta,
            pointer: f.pointer_inside.then(|| f.pointer * (1.0 / dpr)),
            pointer_down: f.pointer_down,
        };
        f.cursor_visible = !self.hides_system_cursor(&input);
        self.update(f.scene, f.overlay, input)
    }
}

fn set_material(scene: &mut Scene, id: ObjectId, material: Material) {
    if let Some(obj) = scene.get_mut(id) {
        if let ObjectKind::Mesh(mesh) = &mut obj.kind {
            mesh.material = Arc::new(material);
        }
    }
}

fn set_visible(scene: &mut Scene, id: ObjectId, visible: bool) {
    if let Some(obj) = scene.get_mut(id) {
        obj.visible = visible;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fx::{Button, Figure, Gloss, Venom, VenomState};

    fn input(pointer: Option<Vector2>, down: bool) -> StageInput {
        StageInput { viewport: Vector2::new(800.0, 600.0), dpr: 1.0, dt: 1.0 / 60.0, pointer, pointer_down: down }
    }

    #[test]
    fn a_click_is_a_press_and_a_release_over_the_element() {
        let mut scene = Scene::new();
        let mut overlay = Overlay::new(800, 600);
        let mut stage = Stage::new();
        let button = stage.add(&mut scene, Button::new("go"), Gloss::default());
        let centre = Some(Vector2::new(400.0, 300.0));
        let kinds = |events: Vec<Event>| events.into_iter().map(|e| e.kind).collect::<Vec<_>>();
        assert!(stage.update(&mut scene, &mut overlay, input(None, false)).is_empty());
        assert_eq!(kinds(stage.update(&mut scene, &mut overlay, input(centre, false))), [EventKind::Enter]);
        assert_eq!(kinds(stage.update(&mut scene, &mut overlay, input(centre, true))), [EventKind::Press]);
        for _ in 0..10 {
            stage.update(&mut scene, &mut overlay, input(centre, true));
        }
        assert!(stage.component::<Button>(button).unwrap().scale() < 0.99, "it gives while held");
        assert_eq!(kinds(stage.update(&mut scene, &mut overlay, input(centre, false))), [EventKind::Release, EventKind::Click]);
        // pressed, dragged off, released: no click
        stage.update(&mut scene, &mut overlay, input(centre, true));
        let off = Some(Vector2::new(10.0, 10.0));
        assert_eq!(kinds(stage.update(&mut scene, &mut overlay, input(off, true))), [EventKind::Leave]);
        assert_eq!(kinds(stage.update(&mut scene, &mut overlay, input(off, false))), [EventKind::Release]);
    }

    #[test]
    fn effects_are_reachable_and_quads_are_reused() {
        let mut scene = Scene::new();
        let mut overlay = Overlay::new(800, 600);
        let mut stage = Stage::new();
        let a = stage.add(&mut scene, Figure::new(Shape::circle(Vector2::new(200.0, 200.0), 40.0)), Venom::default());
        let b = stage.add(&mut scene, Figure::new(Shape::circle(Vector2::new(600.0, 200.0), 40.0)).decorative(), Gloss::default());
        for _ in 0..60 {
            stage.update(&mut scene, &mut overlay, input(Some(Vector2::new(220.0, 120.0)), false));
        }
        assert_eq!(stage.effect::<Venom>(a).unwrap().state(), VenomState::Holding);
        assert!(stage.effect::<Venom>(b).is_none(), "b wears gloss");
        assert!(stage.effect_mut::<Gloss>(b).is_some());
        assert!(overlay.touched(), "the glove and hairs were painted");
        let quad = stage.get(a).unwrap().quad;
        stage.remove(&mut scene, a);
        assert!(!scene.get(quad).unwrap().visible);
        let c = stage.add(&mut scene, Figure::new(Shape::circle(Vector2::new(400.0, 400.0), 40.0)), Gloss::default());
        assert_eq!(stage.get(c).unwrap().quad, quad, "the removed element's quad was reused");
        assert_eq!(stage.ids().collect::<Vec<_>>(), vec![c, b]);
    }
}
