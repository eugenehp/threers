//! Reflection, saves, prefabs and the inspector.

use threers::core::Object3D;
use threers::math::Vector3;
use threers_ecs::prelude::*;
use threers_ecs::{reflect, FieldValue, Reflect};

#[derive(Default, Debug, PartialEq, Clone)]
struct Stats {
    hp: f32,
    level: u32,
    alive: bool,
    title: String,
    home: Vector3,
    friend: Option<Entity>,
}
impl Component for Stats {}
reflect!(Stats {
    hp,
    level,
    alive,
    title,
    home,
    friend
});

#[derive(Default, Debug, PartialEq)]
struct Speed(f32);
impl Component for Speed {}
reflect!(Speed { 0 });

#[derive(Default, Debug, PartialEq)]
struct Tag;
impl Component for Tag {}
reflect!(Tag {});

#[derive(Default, Debug, PartialEq)]
struct Score {
    points: u64,
}
reflect!(Score { points });

struct Unsaved;
impl Component for Unsaved {}

fn world() -> World {
    let mut w = World::new();
    w.register::<Stats>()
        .register::<Speed>()
        .register::<Tag>()
        .register_resource::<Score>();
    w
}

fn named(w: &mut World, name: &str) -> Entity {
    let id = w.spawn(Object3D::group());
    w.scene.get_mut(id).unwrap().name = name.to_owned();
    id
}

#[test]
fn reflect_reads_and_writes_by_name() {
    let mut s = Stats::default();
    assert!(s.set_field("hp", &FieldValue::Number(3.5)));
    assert!(s.set_field("title", &FieldValue::Text("hero".into())));
    assert_eq!(s.hp, 3.5);
    assert_eq!(s.title, "hero");
    assert!(!s.set_field("nope", &FieldValue::Bool(true)));
    assert!(!s.set_field("hp", &FieldValue::Bool(true)), "wrong kind");
    let names: Vec<_> = s.fields().iter().map(|f| f.0).collect();
    assert_eq!(names, ["hp", "level", "alive", "title", "home", "friend"]);
    assert_eq!(Stats::type_name(), "Stats");
    assert_eq!(Speed(2.0).fields()[0], ("0", FieldValue::Number(2.0)));
}

#[test]
fn the_inspector_lists_registered_components_only() {
    let mut w = world();
    let id = w.spawn(Object3D::group());
    w.insert(id, (Speed(1.0), Tag, Unsaved));
    let names: Vec<_> = w.inspect(id).iter().map(|c| c.0).collect();
    assert_eq!(names, ["Speed", "Tag"]);
    assert_eq!(w.inspectable(), [id]);
}

#[test]
fn a_round_trip_keeps_every_kind_of_field() {
    let mut w = world();
    let a = named(&mut w, "a");
    let b = named(&mut w, "b");
    let stats = Stats {
        hp: 1.5,
        level: 7,
        alive: true,
        title: "with \"quotes\" and\nlines".to_owned(),
        home: Vector3::new(1.0, 2.0, 3.0),
        friend: Some(b),
    };
    w.insert(a, stats.clone());
    w.insert(b, Speed(4.0));
    w.scene.get_mut(a).unwrap().position = Vector3::new(5.0, 6.0, 7.0);
    w.scene.get_mut(a).unwrap().scale = Vector3::new(2.0, 2.0, 2.0);
    w.insert_resource(Score { points: 99 });
    let save = w.save();

    let mut other = world();
    other.load(&save).expect("load");
    let [mut a2, mut b2] = [None, None];
    other.each_ref::<(Entity, NodeRef)>(|(id, node)| match node.name.as_str() {
        "a" => a2 = Some(id),
        "b" => b2 = Some(id),
        _ => {}
    });
    let (a2, b2) = (a2.unwrap(), b2.unwrap());
    let loaded = other.get::<Stats>(a2).unwrap();
    assert_eq!(
        *loaded,
        Stats {
            friend: Some(b2),
            ..stats
        }
    );
    assert_eq!(other.scene.get(a2).unwrap().position.y, 6.0);
    assert_eq!(other.scene.get(a2).unwrap().scale.x, 2.0);
    assert_eq!(other.get::<Speed>(b2), Some(&Speed(4.0)));
    assert_eq!(other.resource::<Score>().points, 99);
}

#[test]
fn load_finds_named_nodes_and_clears_what_the_save_lacks() {
    let mut w = world();
    let kept = named(&mut w, "kept");
    w.insert(kept, Speed(1.0));
    let save = w.save();

    w.insert(kept, Tag);
    let extra = named(&mut w, "extra");
    w.insert(extra, Speed(2.0));
    let scenery = named(&mut w, "scenery");

    w.load(&save).expect("load");
    assert!(w.is_alive(kept), "the same node, found by name");
    assert!(!w.has::<Tag>(kept), "added since the save");
    assert!(!w.is_alive(extra), "not in the save");
    assert!(
        w.is_alive(scenery),
        "nothing registered, so not the save's business"
    );
}

#[test]
fn a_bad_save_changes_nothing() {
    let mut w = world();
    let id = named(&mut w, "x");
    w.insert(id, Speed(1.0));
    let good = w.save();

    for bad in [
        "not json",
        "{}",
        r#"{"version":2,"entities":[],"resources":{}}"#,
        &good.replace("Speed", "Renamed"),
    ] {
        assert!(w.load(bad).is_err(), "{bad}");
        assert_eq!(w.get::<Speed>(id), Some(&Speed(1.0)));
    }
}

#[test]
fn after_load_hooks_run() {
    let mut w = world();
    w.after_load(|w| {
        w.insert_resource(true);
    });
    let save = w.save();
    w.load(&save).unwrap();
    assert!(*w.resource::<bool>());
}

#[test]
fn a_prefab_copies_a_subtree_with_components_and_references() {
    let mut w = world();
    let top = named(&mut w, "top");
    let child = w.spawn_under(top, Object3D::group());
    w.scene.get_mut(child).unwrap().position.x = 2.0;
    let outside = named(&mut w, "outside");
    w.insert(child, Speed(3.0));
    w.insert(
        top,
        Stats {
            friend: Some(child),
            ..Stats::default()
        },
    );
    let prefab = w.capture_prefab(top).unwrap();
    let data = w.spawn_data();
    assert!(w.capture_prefab(data).is_some());

    let copy = w.spawn_prefab(&prefab);
    let second = w.spawn_prefab_under(outside, &prefab);

    assert_ne!(copy, top);
    assert_eq!(w.scene.get(second).unwrap().parent, Some(outside));
    let kids = w.scene.get(copy).unwrap().children.clone();
    assert_eq!(kids.len(), 1);
    assert_ne!(kids[0], child);
    assert_eq!(w.scene.get(kids[0]).unwrap().position.x, 2.0);
    assert_eq!(w.get::<Speed>(kids[0]), Some(&Speed(3.0)));
    // A reference inside the prefab points at the copy, not the original.
    assert_eq!(w.get::<Stats>(copy).unwrap().friend, Some(kids[0]));
    // The original is untouched.
    assert_eq!(w.scene.get(top).unwrap().children, [child]);
}

#[test]
fn a_prefab_setup_runs_on_each_copy() {
    let mut w = world();
    let top = w.spawn(Object3D::group());
    let prefab = w.capture_prefab(top).unwrap().with_setup(|w, id| {
        w.insert(id, Tag);
    });
    let copy = w.spawn_prefab(&prefab);
    assert!(w.has::<Tag>(copy));
    assert!(!w.has::<Tag>(top));
}
