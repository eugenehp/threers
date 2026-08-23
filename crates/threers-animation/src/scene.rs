//! Driving a `threers` scene graph.
//!
//! [`Animator`] is the join between the timing primitives in this crate and the
//! nodes they move. It owns a set of clips, each pointing at one node and one
//! property, and writes them all in a single call per frame.

use crate::easing::Easing;
use crate::tween::{Repeat, Tween};
use threers::core::{ObjectArena, ObjectId};
use threers::math::{Quaternion, Vector3};

/// Which property of a node a clip drives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Property {
    Position(Vector3, Vector3),
    Rotation(Quaternion, Quaternion),
    Scale(Vector3, Vector3),
}

/// Handle to a running clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClipId(usize);

#[derive(Debug, Clone)]
struct Clip {
    object: ObjectId,
    property: Property,
    position: Tween<Vector3>,
    rotation: Tween<Quaternion>,
    removed: bool,
}

/// Animates scene nodes.
///
/// ```no_run
/// use threers_animation::prelude::*;
/// use threers::core::{Object3D, ObjectArena};
///
/// let mut arena = ObjectArena::new();
/// let node = arena.insert(Object3D::group());
///
/// let mut animator = Animator::new();
/// animator.move_to(node, Vector3::ZERO, Vector3::new(0.0, 5.0, 0.0), 1.5)
///     .easing(Easing::BackOut);
///
/// // once per frame
/// animator.update(1.0 / 60.0, &mut arena);
/// ```
#[derive(Debug, Clone, Default)]
pub struct Animator {
    clips: Vec<Clip>,
    /// Global rate. `0.5` is slow motion; `0` freezes everything.
    pub speed: f32,
}

impl Animator {
    pub fn new() -> Self {
        Self {
            clips: Vec::new(),
            speed: 1.0,
        }
    }

    /// Animate a node's position.
    pub fn move_to(
        &mut self,
        object: ObjectId,
        from: Vector3,
        to: Vector3,
        duration: f32,
    ) -> ClipHandle<'_> {
        self.push(object, Property::Position(from, to), duration)
    }

    /// Animate a node's rotation, along the shortest arc.
    pub fn rotate_to(
        &mut self,
        object: ObjectId,
        from: Quaternion,
        to: Quaternion,
        duration: f32,
    ) -> ClipHandle<'_> {
        self.push(object, Property::Rotation(from, to), duration)
    }

    /// Animate a node's scale.
    pub fn scale_to(
        &mut self,
        object: ObjectId,
        from: Vector3,
        to: Vector3,
        duration: f32,
    ) -> ClipHandle<'_> {
        self.push(object, Property::Scale(from, to), duration)
    }

    fn push(&mut self, object: ObjectId, property: Property, duration: f32) -> ClipHandle<'_> {
        let (position, rotation) = match property {
            Property::Position(a, b) | Property::Scale(a, b) => (
                Tween::new(a, b, duration),
                Tween::new(Quaternion::identity(), Quaternion::identity(), duration),
            ),
            Property::Rotation(a, b) => (
                Tween::new(Vector3::ZERO, Vector3::ZERO, duration),
                Tween::new(a, b, duration),
            ),
        };
        self.clips.push(Clip {
            object,
            property,
            position,
            rotation,
            removed: false,
        });
        let index = self.clips.len() - 1;
        ClipHandle {
            animator: self,
            index,
        }
    }

    /// Advance every clip and write the results onto the scene.
    ///
    /// Finished clips are dropped, so an animator left running does not grow
    /// without bound.
    pub fn update(&mut self, dt: f32, arena: &mut ObjectArena) {
        if !dt.is_finite() {
            return;
        }
        let dt = dt * self.speed;

        for clip in &mut self.clips {
            if clip.removed {
                continue;
            }
            let Some(object) = arena.get_mut(clip.object) else {
                // The node is gone; drop the clip rather than leaking it.
                clip.removed = true;
                continue;
            };

            match clip.property {
                Property::Position(..) => {
                    object.position = clip.position.update(dt);
                }
                Property::Scale(..) => {
                    object.scale = clip.position.update(dt);
                }
                Property::Rotation(..) => {
                    object.quaternion = clip.rotation.update(dt);
                }
            }
            object.update_matrix();

            let finished = match clip.property {
                Property::Rotation(..) => clip.rotation.is_finished(),
                _ => clip.position.is_finished(),
            };
            if finished {
                clip.removed = true;
            }
        }

        self.clips.retain(|c| !c.removed);
    }

    /// Clips still running.
    pub fn len(&self) -> usize {
        self.clips.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clips.is_empty()
    }

    /// Stop everything immediately, leaving nodes where they are.
    pub fn clear(&mut self) {
        self.clips.clear();
    }

    /// Stop the clips driving one node, leaving the rest alone.
    pub fn cancel(&mut self, object: ObjectId) {
        self.clips.retain(|c| c.object != object);
    }

    /// Whether any clip is driving this node.
    pub fn is_animating(&self, object: ObjectId) -> bool {
        self.clips.iter().any(|c| c.object == object && !c.removed)
    }
}

/// Fluent configuration for the clip just added.
pub struct ClipHandle<'a> {
    animator: &'a mut Animator,
    index: usize,
}

impl ClipHandle<'_> {
    pub fn easing(self, easing: Easing) -> Self {
        self.configure(|c| {
            c.position = c.position.easing(easing);
            c.rotation = c.rotation.easing(easing);
        })
    }

    pub fn delay(self, seconds: f32) -> Self {
        self.configure(|c| {
            c.position = c.position.delay(seconds);
            c.rotation = c.rotation.delay(seconds);
        })
    }

    pub fn repeat(self, repeat: Repeat) -> Self {
        self.configure(|c| {
            c.position = c.position.repeat(repeat);
            c.rotation = c.rotation.repeat(repeat);
        })
    }

    pub fn yoyo(self, yoyo: bool) -> Self {
        self.configure(|c| {
            c.position = c.position.yoyo(yoyo);
            c.rotation = c.rotation.yoyo(yoyo);
        })
    }

    /// The handle for this clip, for cancelling it later.
    pub fn id(self) -> ClipId {
        ClipId(self.index)
    }

    fn configure(self, f: impl FnOnce(&mut Clip)) -> Self {
        if let Some(clip) = self.animator.clips.get_mut(self.index) {
            f(clip);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use threers::core::Object3D;

    fn scene() -> (ObjectArena, ObjectId) {
        let mut arena = ObjectArena::new();
        let node = arena.insert(Object3D::group());
        (arena, node)
    }

    #[test]
    fn a_position_clip_moves_the_node_and_then_retires() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        a.move_to(node, Vector3::ZERO, Vector3::new(0.0, 10.0, 0.0), 1.0);
        assert_eq!(a.len(), 1);

        a.update(0.5, &mut arena);
        let y = arena.get(node).unwrap().position.y;
        assert!((y - 5.0).abs() < 1e-3, "halfway should be 5, got {y}");

        a.update(0.6, &mut arena);
        assert_eq!(arena.get(node).unwrap().position.y, 10.0);
        assert!(a.is_empty(), "a finished clip should be dropped");
    }

    #[test]
    fn rotation_and_scale_clips_drive_their_own_properties() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        let target = Quaternion::from_axis_angle(Vector3::UP, 1.0);
        a.rotate_to(node, Quaternion::identity(), target, 1.0);
        a.update(1.5, &mut arena);
        assert!(arena.get(node).unwrap().quaternion.dot(target).abs() > 0.99);
        // Position was untouched.
        assert_eq!(arena.get(node).unwrap().position, Vector3::ZERO);

        let mut a = Animator::new();
        a.scale_to(node, Vector3::ONE, Vector3::new(2.0, 2.0, 2.0), 1.0);
        a.update(1.5, &mut arena);
        assert_eq!(arena.get(node).unwrap().scale, Vector3::new(2.0, 2.0, 2.0));
    }

    #[test]
    fn several_clips_on_different_nodes_run_at_once() {
        let mut arena = ObjectArena::new();
        let a_node = arena.insert(Object3D::group());
        let b_node = arena.insert(Object3D::group());

        let mut a = Animator::new();
        a.move_to(a_node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 1.0);
        a.move_to(b_node, Vector3::ZERO, Vector3::new(0.0, 0.0, 5.0), 2.0);
        a.update(1.0, &mut arena);

        assert_eq!(arena.get(a_node).unwrap().position.x, 1.0);
        assert!((arena.get(b_node).unwrap().position.z - 2.5).abs() < 1e-3);
        assert_eq!(a.len(), 1, "only the shorter clip should have retired");
    }

    #[test]
    fn a_deleted_node_does_not_keep_its_clip_alive() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        a.move_to(node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 10.0);
        arena.remove_child(node);
        // Simulate the node disappearing from under the animator.
        let mut empty = ObjectArena::new();
        a.update(0.1, &mut empty);
        assert!(a.is_empty(), "the clip should have been dropped");
    }

    #[test]
    fn speed_scales_everything_and_zero_freezes_it() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        a.move_to(node, Vector3::ZERO, Vector3::new(10.0, 0.0, 0.0), 1.0);

        a.speed = 0.5;
        a.update(1.0, &mut arena);
        assert!((arena.get(node).unwrap().position.x - 5.0).abs() < 1e-3);

        a.speed = 0.0;
        let held = arena.get(node).unwrap().position.x;
        a.update(10.0, &mut arena);
        assert_eq!(arena.get(node).unwrap().position.x, held);
    }

    #[test]
    fn cancel_stops_one_node_without_touching_the_others() {
        let mut arena = ObjectArena::new();
        let a_node = arena.insert(Object3D::group());
        let b_node = arena.insert(Object3D::group());
        let mut a = Animator::new();
        a.move_to(a_node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 10.0);
        a.move_to(b_node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 10.0);

        assert!(a.is_animating(a_node));
        a.cancel(a_node);
        assert!(!a.is_animating(a_node));
        assert!(a.is_animating(b_node));
        assert_eq!(a.len(), 1);
    }

    #[test]
    fn a_looping_yoyo_clip_never_retires() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        a.move_to(node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 1.0)
            .repeat(Repeat::Forever)
            .yoyo(true);
        for _ in 0..200 {
            a.update(0.05, &mut arena);
        }
        assert_eq!(a.len(), 1);
        let x = arena.get(node).unwrap().position.x;
        assert!((0.0..=1.0).contains(&x), "yoyo left its range: {x}");
    }

    #[test]
    fn nonsense_frame_times_are_ignored() {
        let (mut arena, node) = scene();
        let mut a = Animator::new();
        a.move_to(node, Vector3::ZERO, Vector3::new(1.0, 0.0, 0.0), 1.0);
        a.update(f32::NAN, &mut arena);
        a.update(f32::INFINITY, &mut arena);
        assert!(arena.get(node).unwrap().position.x.is_finite());
    }
}
