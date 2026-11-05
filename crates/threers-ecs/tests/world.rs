//! Components, queries, despawning and the rows they leave behind.

use std::collections::HashMap;
use threers::core::Object3D;
use threers_ecs::prelude::*;
use threers_ecs::Term;

#[derive(Debug, PartialEq, Clone, Copy)]
struct Health(i32);
impl Component for Health {}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Speed(f32);
impl Component for Speed {}

struct Enemy;
impl Component for Enemy {}

struct Frozen;
impl Component for Frozen {}

fn spawn(w: &mut World) -> Entity {
    w.spawn(Object3D::group())
}

#[test]
fn a_node_added_through_the_scene_is_an_entity() {
    let mut w = World::new();
    let id = w.scene.add(Object3D::group());
    assert!(w.insert(id, Health(3)));
    assert_eq!(w.get::<Health>(id), Some(&Health(3)));
}

#[test]
fn insert_takes_one_component_or_a_tuple_and_replaces() {
    let mut w = World::new();
    let id = spawn(&mut w);
    w.insert(id, Health(1));
    w.insert(id, (Health(2), Speed(5.0), Enemy));
    assert_eq!(w.get::<Health>(id), Some(&Health(2)));
    assert_eq!(w.get::<Speed>(id), Some(&Speed(5.0)));
    assert!(w.has::<Enemy>(id));
    assert_eq!(w.entities::<&Health>().len(), 1);
}

#[test]
fn remove_hands_the_component_back() {
    let mut w = World::new();
    let id = spawn(&mut w);
    assert_eq!(w.remove::<Health>(id), None);
    w.insert(id, Health(1));
    assert_eq!(w.remove::<Health>(id), Some(Health(1)));
    assert!(!w.has::<Health>(id));
}

#[test]
fn each_writes_through_to_the_node_and_the_component() {
    let mut w = World::new();
    let id = spawn(&mut w);
    w.insert(id, (Health(1), Speed(2.0)));
    w.each::<(Node, &mut Health, &Speed)>(|(node, health, speed)| {
        node.position.x = speed.0;
        health.0 += 1;
    });
    assert_eq!(w.scene.get(id).unwrap().position.x, 2.0);
    assert_eq!(w.get::<Health>(id), Some(&Health(2)));
}

#[test]
fn each_matches_only_entities_with_everything() {
    let mut w = World::new();
    let both = spawn(&mut w);
    let health_only = spawn(&mut w);
    let speed_only = spawn(&mut w);
    w.insert(both, (Health(1), Speed(1.0)));
    w.insert(health_only, Health(1));
    w.insert(speed_only, Speed(1.0));
    assert_eq!(w.entities::<(&Health, &Speed)>(), vec![both]);
}

#[test]
fn filters_and_optional_terms() {
    let mut w = World::new();
    let plain = spawn(&mut w);
    let enemy = spawn(&mut w);
    let frozen = spawn(&mut w);
    w.insert(plain, Health(1));
    w.insert(enemy, (Health(2), Enemy));
    w.insert(frozen, (Health(3), Enemy, Frozen));

    assert_eq!(
        w.entities::<(&Health, With<Enemy>, Without<Frozen>)>(),
        vec![enemy]
    );
    assert_eq!(w.entities::<(&Health, Without<Enemy>)>(), vec![plain]);

    let mut seen = Vec::new();
    w.each::<(&Health, Option<&Enemy>)>(|(health, enemy)| seen.push((health.0, enemy.is_some())));
    seen.sort();
    assert_eq!(seen, vec![(1, false), (2, true), (3, true)]);

    w.each::<(&Health, Option<&mut Speed>)>(|(_, speed)| assert!(speed.is_none()));
}

#[test]
fn a_query_with_no_required_component_visits_every_node() {
    let mut w = World::new();
    let id = spawn(&mut w);
    // The scene root is a node too.
    assert_eq!(w.entities::<NodeRef>().len(), 2);
    assert_eq!(w.entities::<Without<Enemy>>().len(), 2);
    w.insert(id, Enemy);
    assert_eq!(w.entities::<Without<Enemy>>(), vec![w.scene.root]);
}

/// Results are the same whichever list a query walks, so only this notices if it
/// walks the long one: a lookup of the one `Player` would cost as much as the scene.
#[test]
fn a_query_walks_its_rarest_component() {
    fn walked<Q: Term>(w: &mut World) -> usize {
        // SAFETY: only the length is read, and nothing else touches the world.
        let state = unsafe { Q::state(w) };
        Q::candidates(&state).map_or(0, |ids| ids.len())
    }

    let mut w = World::new();
    for _ in 0..5 {
        let id = spawn(&mut w);
        w.insert(id, Health(1));
    }
    let id = spawn(&mut w);
    w.insert(id, (Health(1), Enemy));

    assert_eq!(walked::<(&Health, With<Enemy>)>(&mut w), 1);
    assert_eq!(walked::<(With<Enemy>, &Health)>(&mut w), 1);
    assert_eq!(walked::<(&Health, Without<Enemy>)>(&mut w), 6);
}

#[test]
fn a_component_nothing_has_matches_nothing() {
    let mut w = World::new();
    spawn(&mut w);
    assert!(w.entities::<&Health>().is_empty());
    assert!(w.entities::<(NodeRef, With<Health>)>().is_empty());
}

#[test]
#[should_panic(expected = "twice")]
fn naming_a_component_twice_is_refused() {
    let mut w = World::new();
    w.each::<(&mut Health, &Health)>(|_| {});
}

#[test]
#[should_panic(expected = "twice")]
fn naming_the_node_twice_is_refused() {
    let mut w = World::new();
    w.each::<(Node, Node)>(|_| {});
}

/// A filter reads a row's bookkeeping, not the value being written.
#[test]
fn a_filter_on_a_written_component_is_allowed() {
    let mut w = World::new();
    w.each::<(&mut Health, With<Health>)>(|_| {});
}

#[test]
fn despawn_takes_descendants_and_their_components() {
    let mut w = World::new();
    let parent = spawn(&mut w);
    let child = w.spawn_under(parent, Object3D::group());
    let grandchild = w.spawn_under(child, Object3D::group());
    let other = spawn(&mut w);
    for id in [parent, child, grandchild, other] {
        w.insert(id, Health(1));
    }

    assert!(w.despawn(parent));

    for id in [parent, child, grandchild] {
        assert!(!w.is_alive(id));
        assert!(!w.has::<Health>(id));
    }
    assert_eq!(w.entities::<&Health>(), vec![other]);
    assert_eq!(w.scene.arena.nodes.len(), 2);
    let root = w.scene.root;
    assert_eq!(w.scene.get(root).unwrap().children, vec![other]);
}

#[test]
fn despawning_a_child_leaves_the_parent() {
    let mut w = World::new();
    let parent = spawn(&mut w);
    let child = w.spawn_under(parent, Object3D::group());
    w.despawn(child);
    assert!(w.is_alive(parent));
    assert!(w.scene.get(parent).unwrap().children.is_empty());
}

#[test]
fn the_root_survives_despawn_and_a_dead_id_is_refused() {
    let mut w = World::new();
    let root = w.scene.root;
    let id = spawn(&mut w);
    assert!(!w.despawn(root));
    assert!(w.is_alive(root) && w.is_alive(id));
    assert!(w.despawn(id));
    assert!(!w.despawn(id));
    assert!(!w.insert(id, Health(1)));
    assert!(w.entities::<&Health>().is_empty());
}

#[test]
fn a_reused_slot_does_not_inherit_components() {
    let mut w = World::new();
    let old = spawn(&mut w);
    w.insert(old, Health(1));
    w.despawn(old);
    let new = spawn(&mut w);
    assert_ne!(old, new);
    assert!(!w.has::<Health>(new));
    assert!(!w.has::<Health>(old));
}

#[test]
fn a_node_deleted_behind_the_world_is_skipped_then_pruned() {
    let mut w = World::new();
    let gone = spawn(&mut w);
    let kept = spawn(&mut w);
    w.insert(gone, Health(1));
    w.insert(kept, Health(2));

    w.scene.arena.delete(gone);
    assert_eq!(w.entities::<&Health>(), vec![kept]);

    // The freed slot is reused while the old row is still there.
    let new = spawn(&mut w);
    w.insert(new, Health(3));
    let mut seen = Vec::new();
    w.each::<&Health>(|health| seen.push(health.0));
    seen.sort();
    assert_eq!(seen, vec![2, 3]);

    w.prune();
    assert_eq!(w.get::<Health>(new), Some(&Health(3)));
    assert_eq!(w.get::<Health>(kept), Some(&Health(2)));
    assert_eq!(w.get::<Health>(gone), None);
    assert_eq!(w.entities::<&Health>().len(), 2);
}

#[test]
fn a_data_entity_is_out_of_the_scene_tree() {
    let mut w = World::new();
    let quest = w.spawn_data();
    w.insert(quest, Health(1));
    let mut walked = Vec::new();
    let root = w.scene.root;
    w.scene.arena.traverse(root, &mut |id, _| walked.push(id));
    assert_eq!(walked, vec![root]);
    assert_eq!(w.entities::<&Health>(), vec![quest]);
    assert!(w.despawn(quest));
}

#[test]
fn zero_sized_components_are_stored_per_entity() {
    let mut w = World::new();
    let ids: Vec<_> = (0..4).map(|_| spawn(&mut w)).collect();
    w.insert(ids[1], Enemy);
    w.insert(ids[3], Enemy);
    w.remove::<Enemy>(ids[1]);
    assert_eq!(w.entities::<With<Enemy>>(), vec![ids[3]]);
}

#[test]
fn resources_can_be_lent_to_a_query() {
    #[derive(Default)]
    struct Score(i32);

    let mut w = World::new();
    assert!(w.get_resource::<Score>().is_none());
    w.insert_resource(Score::default());
    let id = spawn(&mut w);
    w.insert(id, Health(7));

    let total = w.with_resource::<Score, _>(|w, score| {
        w.each::<&Health>(|health| score.0 += health.0);
        score.0
    });

    assert_eq!(total, 7);
    assert_eq!(w.resource::<Score>().0, 7);
    w.resource_mut::<Score>().0 = 1;
    assert_eq!(w.remove_resource::<Score>().map(|s| s.0), Some(1));
}

#[test]
#[should_panic(expected = "no resource")]
fn a_missing_resource_names_itself() {
    World::new().resource::<u32>();
}

#[test]
fn commands_from_a_query_apply_in_order_at_flush() {
    let mut w = World::new();
    let dead = spawn(&mut w);
    let alive = spawn(&mut w);
    w.insert(dead, Health(0));
    w.insert(alive, Health(1));

    let commands = w.commands();
    w.each::<(Entity, &Health)>(|(id, health)| {
        if health.0 <= 0 {
            commands.despawn(id);
            // Queued after the despawn, so it finds the entity gone.
            commands.insert(id, Speed(1.0));
        } else {
            commands.spawn(Object3D::group(), Enemy);
            commands.remove::<Health>(id);
        }
    });

    assert!(w.is_alive(dead));
    w.flush();
    assert!(!w.is_alive(dead));
    assert!(w.entities::<&Speed>().is_empty());
    assert!(!w.has::<Health>(alive));
    assert_eq!(w.entities::<With<Enemy>>().len(), 1);
}

#[test]
fn a_command_can_queue_another() {
    let mut w = World::new();
    let id = spawn(&mut w);
    let commands = w.commands();
    let later = commands.clone();
    commands.add(move |_| later.insert(id, Health(1)));
    w.flush();
    assert!(w.has::<Health>(id));
}

#[derive(Default)]
struct Goodbyes(Vec<(Entity, i32)>);

fn world_with_hook() -> World {
    let mut w = World::new();
    w.insert_resource(Goodbyes::default());
    w.on_remove::<Health>(|w, id, health| w.resource_mut::<Goodbyes>().0.push((id, health.0)));
    w
}

fn goodbyes(w: &World) -> &[(Entity, i32)] {
    &w.resource::<Goodbyes>().0
}

#[test]
fn on_remove_runs_for_remove_and_still_returns_the_value() {
    let mut w = world_with_hook();
    let id = spawn(&mut w);
    w.insert(id, Health(1));
    assert_eq!(w.remove::<Health>(id), Some(Health(1)));
    assert_eq!(goodbyes(&w), [(id, 1)]);
    assert_eq!(w.remove::<Health>(id), None);
    assert_eq!(goodbyes(&w).len(), 1);
}

#[test]
fn on_remove_runs_for_the_value_a_replacement_displaces() {
    let mut w = world_with_hook();
    let id = spawn(&mut w);
    w.insert(id, Health(1));
    assert!(goodbyes(&w).is_empty());
    w.insert(id, Health(2));
    assert_eq!(goodbyes(&w), [(id, 1)]);
    assert_eq!(w.get::<Health>(id), Some(&Health(2)));
}

#[test]
fn on_remove_runs_for_a_whole_despawned_subtree_with_the_entity_gone() {
    let mut w = world_with_hook();
    let parent = spawn(&mut w);
    let child = w.spawn_under(parent, Object3D::group());
    w.insert(parent, Health(1));
    w.insert(child, Health(2));
    w.on_remove::<Health>(|w, id, _| assert!(!w.is_alive(id)));

    w.despawn(parent);

    let mut seen = goodbyes(&w).to_vec();
    seen.sort_by_key(|g| g.1);
    assert_eq!(seen, [(parent, 1), (child, 2)]);
}

#[test]
fn on_remove_runs_for_rows_prune_finds() {
    let mut w = world_with_hook();
    let id = spawn(&mut w);
    w.insert(id, Health(1));
    w.scene.arena.delete(id);
    assert!(goodbyes(&w).is_empty());
    w.prune();
    assert_eq!(goodbyes(&w), [(id, 1)]);
}

#[test]
fn a_hook_may_change_the_world_and_every_hook_runs() {
    let mut w = world_with_hook();
    w.on_remove::<Health>(|w, _, _| {
        let id = w.spawn(Object3D::group());
        w.insert(id, Enemy);
    });
    let id = spawn(&mut w);
    w.insert(id, Health(1));
    w.despawn(id);
    assert_eq!(goodbyes(&w).len(), 1);
    assert_eq!(w.entities::<With<Enemy>>().len(), 1);
}

/// Random operations against a plain map. Removal moves rows about, so a wrong
/// index here shows up as one entity reading another's component.
#[test]
fn stores_agree_with_a_plain_map_under_random_operations() {
    let mut state = 0x9e37_79b9_u64;
    let mut next = move |n: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % n
    };

    for _ in 0..if cfg!(miri) { 2 } else { 40 } {
        let mut w = World::new();
        let mut live: Vec<Entity> = Vec::new();
        let mut model: HashMap<Entity, i32> = HashMap::new();

        for step in 0..if cfg!(miri) { 150 } else { 1500 } {
            let value = step;
            match next(8) {
                0 | 1 => live.push(spawn(&mut w)),
                2 | 3 if !live.is_empty() => {
                    let id = live[next(live.len())];
                    w.insert(id, Health(value));
                    model.insert(id, value);
                }
                4 if !live.is_empty() => {
                    let id = live[next(live.len())];
                    assert_eq!(w.remove::<Health>(id).map(|h| h.0), model.remove(&id));
                }
                5 if !live.is_empty() => {
                    let id = live.swap_remove(next(live.len()));
                    w.despawn(id);
                    model.remove(&id);
                }
                6 if !live.is_empty() => {
                    let id = live.swap_remove(next(live.len()));
                    w.scene.arena.delete(id);
                    model.remove(&id);
                }
                7 => w.prune(),
                _ => {}
            }

            let mut seen = HashMap::new();
            w.each::<(Entity, &Health)>(|(id, health)| {
                assert!(seen.insert(id, health.0).is_none(), "visited twice");
            });
            assert_eq!(seen, model);
            for &id in &live {
                assert_eq!(w.get::<Health>(id).map(|h| h.0), model.get(&id).copied());
            }
        }
    }
}
