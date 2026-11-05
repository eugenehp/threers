//! What a query may name, and what it may mark: the rules the `unsafe` in
//! `query.rs` rests on. Run under Miri too; see `docs/ecs.md`.

use threers::prelude::*;
use threers_ecs::prelude::*;

struct A(u32);
impl Component for A {}
struct B;
impl Component for B {}

#[derive(Default)]
struct Seen(usize);

fn world_of(n: u32) -> World {
    let mut w = World::new();
    for i in 0..n {
        let e = w.spawn(Object3D::group());
        w.insert(e, A(i));
    }
    w
}

#[test]
#[should_panic(expected = "twice")]
fn a_node_cannot_be_borrowed_both_ways() {
    let mut w = world_of(1);
    w.each::<(Node, NodeRef)>(|_| {});
}

#[test]
#[should_panic(expected = "twice")]
fn a_component_cannot_be_borrowed_twice() {
    let mut w = world_of(1);
    w.each::<(&A, &mut A)>(|_| {});
}

#[test]
fn a_filter_can_name_what_the_query_borrows() {
    let mut w = world_of(3);
    let mut sum = 0;
    w.each::<(&A, Changed<A>)>(|(a, _)| sum += a.0);
    assert_eq!(sum, 3);
    w.each::<(&mut A, With<A>)>(|(a, _)| a.0 += 1);
    w.each::<(With<A>, &mut A)>(|(_, a)| a.0 += 1);
    w.each::<(Added<A>, &mut A, Changed<A>)>(|(_, a, _)| a.0 += 1);
    let mut after = 0;
    w.each_ref::<&A>(|a| after += a.0);
    assert_eq!(after, 12);
    assert_eq!(w.count::<(With<A>, Without<A>)>(), 0);
}

fn writer(w: &mut World) {
    w.each::<(&mut A, Without<B>)>(|(a, _)| a.0 += 1);
}

fn reader(w: &mut World) {
    let n = w.count::<Changed<A>>();
    w.resource_mut::<Seen>().0 = n;
}

/// A row a later term turns down was never written, so it did not change.
#[test]
fn a_row_the_query_turned_down_is_not_changed() {
    let mut app = App::new();
    app.world.insert_resource(Seen(0));
    app.system(writer);
    app.system(System::new(reader).after(writer));
    for i in 0..10 {
        let e = app.world.spawn(Object3D::group());
        app.world.insert(e, A(0));
        if i > 0 {
            app.world.insert(e, B);
        }
    }
    app.frame(0.016);
    app.frame(0.016);
    assert_eq!(app.world.resource::<Seen>().0, 1);
}

#[test]
fn a_whole_arena_query_can_run_again() {
    let mut w = world_of(4);
    for _ in 0..3 {
        let mut n = 0;
        w.each::<Node>(|node| {
            node.position.x += 1.0;
            n += 1;
        });
        assert!(n >= 4);
    }
}

fn counting(n: u32) -> impl FnMut(&mut World) {
    move |w| w.resource_mut::<Vec<u32>>().push(n)
}

#[test]
fn closures_of_one_type_are_separate_systems() {
    let mut app = App::new();
    app.world.insert_resource(Vec::<u32>::new());
    for n in 0..3 {
        app.system(counting(n));
    }
    app.frame(0.016);
    assert_eq!(*app.world.resource::<Vec<u32>>(), [0, 1, 2]);
}

#[derive(Clone)]
struct Ping;

/// Each has its own place in the queue, so each hears every event.
#[test]
fn each_closure_reads_events_for_itself() {
    fn listener() -> impl FnMut(&mut World) {
        let marker = 1u8;
        move |w| {
            let heard = w.events::<Ping>().len();
            w.resource_mut::<Seen>().0 += heard * marker as usize;
        }
    }
    let mut app = App::new();
    app.world.insert_resource(Seen(0));
    app.system(listener());
    app.system(listener());
    app.world.send(Ping);
    app.frame(0.016);
    assert_eq!(app.world.resource::<Seen>().0, 2);
}

fn one(w: &mut World) {
    w.resource_mut::<Vec<u32>>().push(1);
}
fn two(w: &mut World) {
    w.resource_mut::<Vec<u32>>().push(2);
}

#[test]
fn function_pointers_are_separate_systems() {
    let mut app = App::new();
    app.world.insert_resource(Vec::<u32>::new());
    let list: [fn(&mut World); 2] = [one, two];
    for f in list {
        app.system(f);
    }
    app.frame(0.016);
    assert_eq!(*app.world.resource::<Vec<u32>>(), [1, 2]);
}

#[test]
#[should_panic(expected = "added twice")]
fn the_same_function_twice_is_a_mistake() {
    let mut s = Schedule::new();
    s.add(one).add(one);
}

#[test]
fn ordering_against_a_repeated_type_waits_for_all_of_it() {
    fn last(w: &mut World) {
        w.resource_mut::<Vec<u32>>().push(9);
    }
    let mut app = App::new();
    app.world.insert_resource(Vec::<u32>::new());
    app.system(System::new(last).after(counting(0)));
    app.system(counting(1));
    app.system(counting(2));
    app.frame(0.016);
    assert_eq!(*app.world.resource::<Vec<u32>>(), [1, 2, 9]);
}

#[test]
fn par_query_visits_what_each_would() {
    let n = if cfg!(miri) { 50 } else { 5000 };
    let mut w = world_of(n);
    for (i, e) in w.entities::<With<A>>().into_iter().enumerate() {
        if i % 2 == 0 {
            w.insert(e, B);
        }
    }
    let hits = std::sync::atomic::AtomicUsize::new(0);
    w.par_query::<(&mut A, With<B>)>(|(a, _)| {
        a.0 = 7;
        hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });
    let half = n as usize / 2;
    assert_eq!(hits.into_inner(), half);
    let mut sevens = 0;
    w.each_ref::<&A>(|a| sevens += (a.0 == 7) as usize);
    assert!((half..=half + 1).contains(&sevens));
}

#[test]
fn single_finds_the_one_and_says_when_there_is_none() {
    let mut w = world_of(3);
    assert!(w.single::<(Entity, With<B>)>().is_none());
    let e = w.spawn(Object3D::group());
    w.insert(e, (A(9), B));
    let (found, a, _) = w.single::<(Entity, &A, With<B>)>().unwrap();
    assert_eq!((found, a.0), (e, 9));
    // Whole-arena queries take the same path with no list to go by.
    assert!(w.single::<NodeRef>().is_some());
}

#[test]
#[should_panic(expected = "twice")]
fn single_holds_to_the_same_rule() {
    let w = world_of(1);
    let _ = w.single::<(&A, &A)>();
}

#[test]
fn iter_is_each_ref_as_an_iterator() {
    let mut w = world_of(4);
    let e = w.spawn(Object3D::group());
    w.insert(e, (A(100), B));
    let sum: u32 = w.iter::<&A>().map(|a| a.0).sum();
    assert_eq!(sum, 106);
    assert_eq!(w.iter::<(Entity, &A, With<B>)>().next().unwrap().0, e);
    assert_eq!(w.iter::<(&A, Without<B>)>().count(), 4);
    assert!(w.iter::<NodeRef>().count() >= 5);
    // Two at once: nothing here writes.
    let pairs = w
        .iter::<&A>()
        .zip(w.iter::<&A>())
        .filter(|(x, y)| x.0 == y.0)
        .count();
    assert_eq!(pairs, 5);
}

#[test]
fn iter_skips_rows_of_deleted_nodes() {
    let mut w = world_of(3);
    let gone = w.entities::<With<A>>()[1];
    w.scene.arena.nodes.remove(gone);
    assert_eq!(w.iter::<&A>().count(), 2);
}

#[test]
fn read_answers_for_one_entity() {
    let mut w = world_of(1);
    let id = w.entities::<With<A>>()[0];
    assert!(w.read::<&A>(id).is_some());
    assert!(w.read::<(&A, With<B>)>(id).is_none());
    assert!(w.read::<(&A, Without<B>)>(id).is_some());
    w.despawn(id);
    assert!(w.read::<&A>(id).is_none());
}

#[test]
fn update_changes_one_and_marks_only_it() {
    let mut w = world_of(3);
    let ids = w.entities::<With<A>>();
    w.clear_trackers();
    assert_eq!(
        w.update::<&mut A, _>(ids[1], |a| {
            a.0 = 50;
            a.0 + 1
        }),
        Some(51)
    );
    assert_eq!(w.get::<A>(ids[1]).unwrap().0, 50);
    assert_eq!(w.entities::<Changed<A>>(), vec![ids[1]]);
    // A row turned down is left alone, and unmarked.
    w.clear_trackers();
    assert_eq!(w.update::<(&mut A, With<B>), _>(ids[0], |_| ()), None);
    assert_eq!(w.count::<Changed<A>>(), 0);
}

#[test]
fn world_transform_goes_through_parents() {
    let mut w = World::new();
    let turned = Quaternion::from_axis_angle(Vector3::UP, std::f32::consts::FRAC_PI_2);
    let parent = w
        .object(Object3D::group())
        .at(Vector3::new(10.0, 0.0, 0.0))
        .rotated(turned)
        .id();
    let child = w.spawn_under(parent, Object3D::group());
    w.node_mut(child).unwrap().position = Vector3::new(0.0, 0.0, -2.0);
    let (at, rotation) = w.world_transform(child).unwrap();
    // Turned left, the parent's -Z points to -X.
    assert!(at.distance_to(Vector3::new(8.0, 0.0, 0.0)) < 1e-4, "{at:?}");
    assert!((rotation.w - turned.w).abs() < 1e-4);
}

struct Lent(u32);

/// A panic inside `with_resource` gives the resource back, and the world can
/// lend it again.
#[test]
fn a_panic_while_lent_puts_the_resource_back() {
    let mut w = World::new();
    w.insert_resource(Lent(1));
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        w.with_resource::<Lent, _>(|w, lent| {
            lent.0 += 1;
            w.insert_resource(5u8);
            panic!("mid-borrow");
        });
    }));
    assert!(outcome.is_err());
    assert_eq!(w.resource::<Lent>().0, 2);
    w.with_resource::<Lent, _>(|_, lent| lent.0 += 1);
    assert_eq!(w.resource::<Lent>().0, 3);
}
