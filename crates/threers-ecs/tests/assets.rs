//! Loading real files, reloading them, and real `.glb` bytes.
#![recursion_limit = "512"]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use threers_ecs::assets::*;
use threers_ecs::prelude::*;
use threers_ecs::reflect;
use threers_ecs::testing::TestGame;

struct Text(String);
impl Asset for Text {
    fn parse(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes).map_err(|e| e.to_string())?;
        if text.starts_with("bad") {
            return Err("refused".to_owned());
        }
        Ok(Text(text.to_owned()))
    }
}

fn temp(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("threers-ecs-assets-{}-{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn game(dir: &Path) -> TestGame {
    let dir = dir.to_path_buf();
    TestGame::new(move |app| {
        app.plugin(AssetsPlugin::new(dir));
        app.world.resource_mut::<Assets>().hot_reload = true;
    })
}

/// Run frames until `done`, in real time, since files arrive on a thread.
fn until(game: &mut TestGame, mut done: impl FnMut(&mut TestGame) -> bool) {
    let start = Instant::now();
    while !done(game) {
        assert!(start.elapsed() < Duration::from_secs(10), "timed out");
        game.frames(1);
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_file_arrives_without_blocking_and_reports_it() {
    let dir = temp("load");
    std::fs::write(dir.join("a.txt"), "hello").unwrap();
    let mut g = game(&dir);
    let handle = g.world().resource_mut::<Assets>().load::<Text>("a.txt");
    assert!(g.resource::<Assets>().is_loading());
    until(&mut g, |g| g.resource::<Assets>().get(&handle).is_some());
    assert_eq!(g.resource::<Assets>().get(&handle).unwrap().0, "hello");
    assert!(!g.resource::<Assets>().is_loading());
    let loaded = g.world().events::<AssetLoaded>();
    assert_eq!(
        loaded,
        [AssetLoaded {
            path: "a.txt".into(),
            reloaded: false
        }]
    );
}

#[test]
fn loading_twice_is_one_asset() {
    let dir = temp("twice");
    std::fs::write(dir.join("a.txt"), "x").unwrap();
    let mut g = game(&dir);
    let a = g.world().resource_mut::<Assets>().load::<Text>("a.txt");
    let b = g.world().resource_mut::<Assets>().load::<Text>("a.txt");
    until(&mut g, |g| g.resource::<Assets>().get(&a).is_some());
    let (x, y) = (
        g.resource::<Assets>().get(&a).unwrap(),
        g.resource::<Assets>().get(&b).unwrap(),
    );
    assert!(std::sync::Arc::ptr_eq(&x, &y));
    assert_eq!(g.world().events::<AssetLoaded>().len(), 1);
}

#[test]
fn a_missing_file_fails_with_its_name() {
    let dir = temp("missing");
    let mut g = game(&dir);
    let handle = g.world().resource_mut::<Assets>().load::<Text>("gone.txt");
    until(&mut g, |g| g.resource::<Assets>().error(&handle).is_some());
    assert!(g
        .resource::<Assets>()
        .error(&handle)
        .unwrap()
        .contains("gone.txt"));
    assert!(g.resource::<Assets>().get(&handle).is_none());
    assert_eq!(g.world().events::<AssetFailed>().len(), 1);
    assert!(!g.resource::<Assets>().is_loading());
}

#[test]
fn a_changed_file_reloads_behind_the_same_handle() {
    let dir = temp("reload");
    let path = dir.join("a.txt");
    std::fs::write(&path, "one").unwrap();
    let mut g = game(&dir);
    let handle = g.world().resource_mut::<Assets>().load::<Text>("a.txt");
    until(&mut g, |g| g.resource::<Assets>().get(&handle).is_some());

    // Let the watcher record the first timestamp before changing the file.
    let mark = g.resource::<Time>().real_elapsed;
    until(&mut g, |g| g.resource::<Time>().real_elapsed > mark + 0.3);
    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(&path, "two").unwrap();

    until(&mut g, |g| {
        g.resource::<Assets>().get(&handle).unwrap().0 == "two"
    });
    assert_eq!(g.resource::<Assets>().version(&handle), 2);
    let events = g.world().events::<AssetLoaded>();
    assert!(events.iter().any(|e| e.reloaded), "{events:?}");
}

#[test]
fn a_bad_edit_keeps_the_last_good_version() {
    let dir = temp("bad");
    let path = dir.join("a.txt");
    std::fs::write(&path, "good").unwrap();
    let mut g = game(&dir);
    let handle = g.world().resource_mut::<Assets>().load::<Text>("a.txt");
    until(&mut g, |g| g.resource::<Assets>().get(&handle).is_some());
    let mark = g.resource::<Time>().real_elapsed;
    until(&mut g, |g| g.resource::<Time>().real_elapsed > mark + 0.3);
    std::thread::sleep(Duration::from_millis(20));
    std::fs::write(&path, "bad edit").unwrap();
    until(&mut g, |g| g.resource::<Assets>().error(&handle).is_some());
    assert_eq!(g.resource::<Assets>().get(&handle).unwrap().0, "good");
}

/// A real binary glTF: nodes with `extras`, a hierarchy and no buffers.
fn glb(json: &str) -> Vec<u8> {
    let mut chunk = json.as_bytes().to_vec();
    while !chunk.len().is_multiple_of(4) {
        chunk.push(b' ');
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(20 + chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&chunk);
    out
}

const LEVEL: &str = r#"{
  "asset": {"version": "2.0"},
  "scene": 0,
  "scenes": [{"nodes": [0]}],
  "nodes": [
    {"name": "door", "children": [1], "translation": [1, 2, 3],
     "extras": {"Health": {"hp": 5}, "Marker": true, "Other": "ignored"}},
    {"name": "hinge", "translation": [0, 1, 0]}
  ]
}"#;

#[derive(Default, Debug, PartialEq)]
struct Health {
    hp: f32,
}
impl Component for Health {}
reflect!(Health { hp });

#[derive(Default)]
struct Marker;
impl Component for Marker {}
reflect!(Marker {});

fn level(dir: &Path) -> (TestGame, Entity) {
    let mut g = game(dir);
    g.world().register::<Health>().register::<Marker>();
    let anchor = g.world().spawn_scene("level.glb");
    (g, anchor)
}

fn find(g: &mut TestGame, name: &str) -> Option<Entity> {
    let mut found = None;
    g.world().each_ref::<(Entity, NodeRef)>(|(id, node)| {
        if node.name == name {
            found = Some(id);
        }
    });
    found
}

#[test]
fn a_scene_lands_under_its_anchor_with_components_from_extras() {
    let dir = temp("scene");
    std::fs::write(dir.join("level.glb"), glb(LEVEL)).unwrap();
    let (mut g, anchor) = level(&dir);
    until(&mut g, |g| !g.world().events::<SceneReady>().is_empty());

    let door = find(&mut g, "door").expect("door");
    let hinge = find(&mut g, "hinge").expect("hinge");
    assert_eq!(g.world().scene.get(door).unwrap().parent, Some(anchor));
    assert_eq!(g.world().scene.get(hinge).unwrap().parent, Some(door));
    assert_eq!(g.world().scene.get(door).unwrap().position.y, 2.0);
    assert_eq!(g.world().get::<Health>(door), Some(&Health { hp: 5.0 }));
    assert!(g.world().has::<Marker>(door));
    assert!(!g.world().has::<Health>(hinge));
}

#[test]
fn editing_a_level_swaps_it_in_place() {
    let dir = temp("hot");
    let path = dir.join("level.glb");
    std::fs::write(&path, glb(LEVEL)).unwrap();
    let (mut g, anchor) = level(&dir);
    until(&mut g, |g| find(g, "door").is_some());
    let mark = g.resource::<Time>().real_elapsed;
    until(&mut g, |g| g.resource::<Time>().real_elapsed > mark + 0.3);
    std::thread::sleep(Duration::from_millis(20));

    std::fs::write(
        &path,
        glb(&LEVEL
            .replace("door", "gate")
            .replace("\"hp\": 5", "\"hp\": 9")),
    )
    .unwrap();
    until(&mut g, |g| find(g, "gate").is_some());
    assert!(find(&mut g, "door").is_none());
    let gate = find(&mut g, "gate").unwrap();
    assert_eq!(g.world().get::<Health>(gate), Some(&Health { hp: 9.0 }));
    assert_eq!(g.world().scene.get(anchor).unwrap().children, [gate]);
}

#[test]
fn a_scene_that_cannot_load_shows_a_marker_not_a_crash() {
    let dir = temp("nofile");
    let (mut g, anchor) = level(&dir);
    until(&mut g, |g| !g.world().events::<AssetFailed>().is_empty());
    g.frames(2);
    assert_eq!(g.world().scene.get(anchor).unwrap().children.len(), 1);

    // Fixing the file replaces the marker.
    std::fs::write(dir.join("level.glb"), glb(LEVEL)).unwrap();
    let scene = g
        .world()
        .get::<SceneInstance>(anchor)
        .unwrap()
        .scene
        .clone();
    let bytes = glb(LEVEL);
    g.world()
        .resource_mut::<Assets>()
        .insert_bytes::<SceneAsset>(scene.path(), bytes);
    until(&mut g, |g| find(g, "door").is_some());
    assert_eq!(g.world().scene.get(anchor).unwrap().children.len(), 1);
}

#[test]
fn garbage_is_refused_as_a_scene() {
    let dir = temp("garbage");
    std::fs::write(dir.join("level.glb"), b"not a glb").unwrap();
    let (mut g, _) = level(&dir);
    until(&mut g, |g| !g.world().events::<AssetFailed>().is_empty());
}

struct WhereParsed(std::thread::ThreadId);
impl Asset for WhereParsed {
    fn parse(_: &[u8]) -> Result<Self, String> {
        Ok(WhereParsed(std::thread::current().id()))
    }
}

#[test]
fn parsing_stays_off_the_game_thread() {
    let dir = temp("thread");
    std::fs::write(dir.join("a"), b"x").unwrap();
    let mut g = game(&dir);
    let read = g.world().resource_mut::<Assets>().load::<WhereParsed>("a");
    let given = g
        .world()
        .resource_mut::<Assets>()
        .insert_bytes::<WhereParsed>("b", vec![1]);
    until(&mut g, |g| !g.resource::<Assets>().is_loading());
    let here = std::thread::current().id();
    assert_ne!(g.resource::<Assets>().get(&read).unwrap().0, here);
    assert_ne!(g.resource::<Assets>().get(&given).unwrap().0, here);
}

#[test]
fn a_second_copy_of_a_scene_gets_nodes_of_its_own() {
    let dir = temp("copies");
    std::fs::write(dir.join("level.glb"), glb(LEVEL)).unwrap();
    let (mut g, first) = level(&dir);
    let second = g.world().spawn_scene("level.glb");
    let mut ready = 0;
    until(&mut g, |g| {
        ready += g.world().events::<SceneReady>().len();
        ready >= 2
    });
    let kids = |g: &mut TestGame, id| g.world().scene.get(id).unwrap().children.clone();
    let (a, b) = (kids(&mut g, first), kids(&mut g, second));
    assert_eq!((a.len(), b.len()), (1, 1));
    assert_ne!(a[0], b[0]);
}

#[test]
fn embedded_bytes_need_no_file() {
    let dir = temp("embedded");
    let mut g = game(&dir);
    let handle = g
        .world()
        .resource_mut::<Assets>()
        .insert_bytes::<Bytes>("built-in", vec![1, 2, 3]);
    until(&mut g, |g| g.resource::<Assets>().get(&handle).is_some());
    assert_eq!(g.resource::<Assets>().get(&handle).unwrap().0, [1, 2, 3]);
}
