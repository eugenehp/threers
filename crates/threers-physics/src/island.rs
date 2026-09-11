//! Connected components of the constraint graph.
//!
//! Two bodies are in the same island if a contact, a joint or a tendon couples
//! them, so an island is a set of bodies that can only be solved — and can only
//! come to rest — together. A tendon couples its whole route at once, being one
//! constraint over all of it.
//!
//! # Why sleeping needs this
//!
//! Judging rest per body is wrong in a way that shows. A box at the top of a
//! settling stack can be momentarily still while the stack beneath it is not,
//! fall asleep, and then hang in the air when the support below it shifts.
//! Rest is a property of the whole connected group, and this is what identifies
//! the group.
//!
//! # Static bodies are boundaries, not members
//!
//! A fixed body is deliberately *not* merged into its neighbours' island. The
//! ground touches everything, so merging through it would collapse an entire
//! scene into one island — which would then never sleep, because something
//! somewhere is always moving.

use crate::body::{BodySet, BodyId};
use crate::contact::ContactSet;
use crate::joint::JointSet;

/// Islands of a world, rebuilt each step.
///
/// The buffers are reused between steps; nothing here allocates in the steady
/// state.
#[derive(Debug, Default, Clone)]
pub struct IslandSet {
    /// Union-find parent, indexed by body slot.
    parent: Vec<u32>,
    rank: Vec<u8>,
    /// Slot -> dense island index, or `u32::MAX` for a body in no island.
    island_of: Vec<u32>,
    /// Island members, concatenated.
    members: Vec<u32>,
    /// `(start, len)` into `members`, one per island.
    ranges: Vec<(u32, u32)>,
}

impl IslandSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Rebuild from the current contacts, joints and tendons.
    pub fn build(
        &mut self,
        bodies: &BodySet,
        contacts: &ContactSet,
        joints: &JointSet,
        tendons: &crate::tendon::TendonSet,
    ) {
        let slots = bodies.slot_count();
        self.parent.clear();
        self.parent.extend(0..slots as u32);
        self.rank.clear();
        self.rank.resize(slots, 0);

        // Only a body the solver can move propagates coupling. Anything else is
        // an anchor: it constrains its neighbours without joining them.
        let couples = |slot: u32| -> bool {
            bodies
                .by_index(slot as usize)
                .is_some_and(|b| b.is_dynamic() && b.enabled)
        };

        for manifold in contacts.iter() {
            if manifold.is_sensor || !manifold.touching {
                continue;
            }
            let (a, b) = (
                manifold.key.body_a.index() as u32,
                manifold.key.body_b.index() as u32,
            );
            if couples(a) && couples(b) {
                self.union(a, b);
            }
        }

        for (_, joint) in joints.iter() {
            if !joint.is_active() {
                continue;
            }
            let (a, b) = (joint.body_a.index() as u32, joint.body_b.index() as u32);
            if couples(a) && couples(b) {
                self.union(a, b);
            }
            // A coupling on a moving carrier is one constraint over three
            // bodies, so the carrier has to solve with the pair rather than
            // beside it. In a real train it is already in the island — the
            // pinion runs in a bearing in the case — but a carrier held only by
            // this coupling would otherwise be left out of it.
            #[cfg(feature = "mechanism")]
            if let crate::joint::JointKind::Gear {
                carrier: Some(c), ..
            } = &joint.kind
            {
                let c = c.index() as u32;
                if couples(c) {
                    if couples(a) {
                        self.union(a, c);
                    }
                    if couples(b) {
                        self.union(b, c);
                    }
                }
            }
        }

        // A tendon is one constraint over its whole route, so every movable
        // body it touches has to end up in the same island — solving half of
        // one route in one island and half in another would have two solvers
        // writing the same velocities from half the row each.
        for (_, tendon) in tendons.iter() {
            if !tendon.is_active() {
                continue;
            }
            let mut anchor: Option<u32> = None;
            for node in &tendon.path {
                let Some(body) = node.body() else {
                    continue;
                };
                let slot = body.index() as u32;
                if !couples(slot) {
                    continue;
                }
                match anchor {
                    Some(first) => self.union(first, slot),
                    None => anchor = Some(slot),
                }
            }
        }

        // Flatten into dense islands. Only bodies that can move get one; a
        // fixed body belongs to no island by construction.
        self.island_of.clear();
        self.island_of.resize(slots, u32::MAX);
        self.members.clear();
        self.ranges.clear();

        // First pass: assign a dense index to each distinct root.
        let mut root_to_island: Vec<u32> = vec![u32::MAX; slots];
        for slot in 0..slots as u32 {
            if !couples(slot) {
                continue;
            }
            let root = self.find(slot);
            if root_to_island[root as usize] == u32::MAX {
                root_to_island[root as usize] = self.ranges.len() as u32;
                self.ranges.push((0, 0));
            }
            self.island_of[slot as usize] = root_to_island[root as usize];
        }

        // Second pass: counting sort the members so each island is contiguous.
        for slot in 0..slots {
            let island = self.island_of[slot];
            if island != u32::MAX {
                self.ranges[island as usize].1 += 1;
            }
        }
        let mut offset = 0u32;
        for range in &mut self.ranges {
            range.0 = offset;
            offset += range.1;
            range.1 = 0; // reused as a fill cursor
        }
        self.members.resize(offset as usize, 0);
        for slot in 0..slots {
            let island = self.island_of[slot];
            if island == u32::MAX {
                continue;
            }
            let range = &mut self.ranges[island as usize];
            self.members[(range.0 + range.1) as usize] = slot as u32;
            range.1 += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Body slots making up one island.
    pub fn island(&self, index: usize) -> &[u32] {
        match self.ranges.get(index) {
            Some(&(start, len)) => &self.members[start as usize..(start + len) as usize],
            None => &[],
        }
    }

    pub fn islands(&self) -> impl Iterator<Item = &[u32]> {
        (0..self.ranges.len()).map(move |i| self.island(i))
    }

    /// Which island a body belongs to, if any.
    pub fn island_of(&self, body: BodyId) -> Option<usize> {
        self.island_of
            .get(body.index())
            .copied()
            .filter(|&i| i != u32::MAX)
            .map(|i| i as usize)
    }

    /// Whether two bodies would be solved together.
    pub fn same_island(&self, a: BodyId, b: BodyId) -> bool {
        match (self.island_of(a), self.island_of(b)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    fn find(&mut self, mut slot: u32) -> u32 {
        // Path halving: flattens the tree without a second pass.
        while self.parent[slot as usize] != slot {
            let grandparent = self.parent[self.parent[slot as usize] as usize];
            self.parent[slot as usize] = grandparent;
            slot = grandparent;
        }
        slot
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        // Union by rank, so the trees stay shallow.
        let (small, large) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (ra, rb)
        } else {
            (rb, ra)
        };
        self.parent[small as usize] = large;
        if self.rank[small as usize] == self.rank[large as usize] {
            self.rank[large as usize] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::joint::Joint;
    use crate::prelude::*;

    /// Step a world far enough for contacts to exist, then build islands.
    fn islands_of(world: &mut World, steps: usize) -> IslandSet {
        for _ in 0..steps {
            world.step_fixed();
        }
        let mut islands = IslandSet::new();
        islands.build(
            world.bodies(),
            world.contacts(),
            world.joints(),
            world.tendons(),
        );
        islands
    }

    #[test]
    fn separate_piles_are_separate_islands() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));

        // Two stacks, far apart. They touch the ground but not each other.
        let mut left = Vec::new();
        let mut right = Vec::new();
        for i in 0..3 {
            left.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(-10.0, 0.5 + i as f32 * 1.01, 0.0))
                    .can_sleep(false),
            ));
            right.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(10.0, 0.5 + i as f32 * 1.01, 0.0))
                    .can_sleep(false),
            ));
        }

        let islands = islands_of(&mut world, 120);
        assert_eq!(islands.len(), 2, "the two stacks should not have merged");
        assert!(islands.same_island(left[0], left[2]));
        assert!(islands.same_island(right[0], right[2]));
        assert!(!islands.same_island(left[0], right[0]));
    }

    #[test]
    fn the_ground_does_not_merge_everything_into_one_island() {
        // The failure this rule exists to prevent: a static body touches
        // everything, so propagating through it would give one giant island
        // that can never sleep.
        let mut world = World::new();
        let ground = world.add_body(RigidBody::fixed().shape(Shape::ground()));
        let mut bodies = Vec::new();
        for i in 0..8 {
            bodies.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.4))
                    .translation(Vector3::new(i as f32 * 5.0, 0.4, 0.0))
                    .can_sleep(false),
            ));
        }
        let islands = islands_of(&mut world, 120);
        assert_eq!(islands.len(), 8, "each ball should be its own island");
        // The ground itself is in none.
        assert!(islands.island_of(ground).is_none());
    }

    #[test]
    fn a_joint_couples_two_bodies_that_never_touch() {
        let mut world = World::new();
        let a = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(-5.0, 0.0, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        let b = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.3))
                .translation(Vector3::new(5.0, 0.0, 0.0))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        world.add_joint(Joint::distance(a, b, Vector3::ZERO, Vector3::ZERO, 10.0));

        let islands = islands_of(&mut world, 5);
        assert_eq!(islands.len(), 1);
        assert!(islands.same_island(a, b), "the joint should have coupled them");
    }

    #[test]
    fn a_chain_of_contacts_forms_one_island() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
        let mut stack = Vec::new();
        for i in 0..6 {
            stack.push(world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(0.0, 0.5 + i as f32 * 1.01, 0.0))
                    .can_sleep(false),
            ));
        }
        let islands = islands_of(&mut world, 180);
        assert_eq!(islands.len(), 1, "a stack is one island");
        assert_eq!(islands.island(0).len(), 6);
        // Top and bottom are transitively connected.
        assert!(islands.same_island(stack[0], stack[5]));
    }

    #[test]
    fn sensors_do_not_couple_bodies() {
        let mut world = World::new();
        let a = world.add_body(
            RigidBody::dynamic()
                .collider(Collider::new(Shape::ball(1.0)).sensor(true))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        let b = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        let islands = islands_of(&mut world, 5);
        assert!(
            !islands.same_island(a, b),
            "a sensor overlap is not a physical coupling"
        );
    }

    #[test]
    fn an_empty_or_contactless_world_produces_no_islands() {
        let mut world = World::new();
        let mut islands = IslandSet::new();
        islands.build(
            world.bodies(),
            world.contacts(),
            world.joints(),
            world.tendons(),
        );
        assert!(islands.is_empty());

        // A single floating body is its own island once it exists.
        let solo = world.add_body(
            RigidBody::dynamic()
                .shape(Shape::ball(0.5))
                .gravity_scale(0.0)
                .can_sleep(false),
        );
        let islands = islands_of(&mut world, 2);
        assert_eq!(islands.len(), 1);
        assert!(islands.island_of(solo).is_some());
    }

    #[test]
    fn every_body_appears_in_exactly_one_island() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
        for i in 0..30 {
            world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::ball(0.35))
                    .translation(Vector3::new(
                        (i % 6) as f32 * 0.7 - 2.0,
                        0.4 + (i / 6) as f32 * 0.75,
                        0.0,
                    ))
                    .can_sleep(false),
            );
        }
        let islands = islands_of(&mut world, 200);

        let mut seen = std::collections::HashSet::new();
        let mut total = 0;
        for island in islands.islands() {
            assert!(!island.is_empty(), "an empty island was emitted");
            for &slot in island {
                assert!(seen.insert(slot), "slot {slot} is in two islands");
                total += 1;
            }
        }
        // Every dynamic body, and nothing else.
        let dynamic = world.bodies().iter().filter(|(_, b)| b.is_dynamic()).count();
        assert_eq!(total, dynamic);
    }

    #[test]
    fn rebuilding_is_stable_and_reusable() {
        let mut world = World::new();
        world.add_body(RigidBody::fixed().shape(Shape::ground()).friction(0.8));
        for i in 0..4 {
            world.add_body(
                RigidBody::dynamic()
                    .shape(Shape::cuboid(0.5, 0.5, 0.5))
                    .translation(Vector3::new(0.0, 0.5 + i as f32 * 1.01, 0.0))
                    .can_sleep(false),
            );
        }
        for _ in 0..180 {
            world.step_fixed();
        }
        let mut islands = IslandSet::new();
        islands.build(
            world.bodies(),
            world.contacts(),
            world.joints(),
            world.tendons(),
        );
        let first: Vec<Vec<u32>> = islands.islands().map(|i| i.to_vec()).collect();
        // Building again over the same state must give the same answer, and
        // must not accumulate.
        islands.build(
            world.bodies(),
            world.contacts(),
            world.joints(),
            world.tendons(),
        );
        let second: Vec<Vec<u32>> = islands.islands().map(|i| i.to_vec()).collect();
        assert_eq!(first, second);
    }
}
