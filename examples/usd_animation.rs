//! USD animation, out and back in again.
//!
//! Builds a small animated scene, writes it as a `.usda` layer and a `.usdz`
//! archive, then reads its own output back and prints what survived. If a
//! `.usdc`, `.usda` or `.usdz` path is given on the command line it reads that
//! instead — the same entry point handles all three, told apart by content
//! rather than by file name.
//!
//!     cargo run --release --example usd_animation
//!     cargo run --release --example usd_animation -- some_asset.usdc

use threers::animation::{AnimationClip, KeyframeTrack, TrackTarget};
use threers::core::{Mesh, Object3D, ObjectArena};
use threers::geometries::BoxGeometry;
use threers::loaders::{animated_scene_to_usda, animated_scene_to_usdz, UsdLoader};
use threers::materials::{Material, StandardMaterial};
use threers::math::{Quaternion, Vector3};

fn main() {
    if let Some(path) = std::env::args().nth(1) {
        inspect(&path);
        return;
    }

    let (arena, roots, clips) = scene();
    let text = animated_scene_to_usda(&arena, &roots, &clips);
    let archive = animated_scene_to_usdz(&arena, &roots, &clips, &[]);

    let _ = std::fs::create_dir_all("out");
    std::fs::write("out/usd_animation.usda", &text).unwrap();
    std::fs::write("out/usd_animation.usdz", &archive).unwrap();
    println!(
        "wrote out/usd_animation.usda ({} bytes) and out/usd_animation.usdz ({} bytes)",
        text.len(),
        archive.len()
    );

    // Read back what we just wrote, through the same path any other file takes.
    println!("\n--- reading out/usd_animation.usdz back");
    inspect("out/usd_animation.usdz");

    println!(
        "\nBoth files open in usdview, and the .usdz plays in Quick Look.\n\
         `usdcat --out x.usdc out/usd_animation.usda` converts it to the binary\n\
         form, which this crate reads too:\n\
         `cargo run --example usd_animation -- x.usdc`"
    );
}

/// A cube that travels, spins and pulses, plus a static plinth under it so the
/// output has something that does *not* animate.
fn scene() -> (ObjectArena, Vec<threers::core::ObjectId>, Vec<AnimationClip>) {
    let mut arena = ObjectArena::new();

    let mut cube = Object3D::mesh(Mesh::new(
        BoxGeometry::new(1.0, 1.0, 1.0),
        Material::Standard(StandardMaterial::default()),
    ));
    cube.name = "Cube".into();
    let cube = arena.insert(cube);

    let mut plinth = Object3D::mesh(Mesh::new(
        BoxGeometry::new(4.0, 0.2, 4.0),
        Material::Standard(StandardMaterial::default()),
    ));
    plinth.name = "Plinth".into();
    plinth.position = Vector3::new(0.0, -1.0, 0.0);
    let plinth = arena.insert(plinth);

    // Three tracks on one object, each keyed on its own frames — USD stores
    // them as `timeSamples` on the three transform ops, and reading them back
    // re-merges them onto one time line.
    let clip = AnimationClip::new(
        "spin",
        2.0,
        vec![
            KeyframeTrack::vector(
                cube,
                TrackTarget::Position,
                vec![0.0, 1.0, 2.0],
                vec![
                    Vector3::new(-3.0, 0.0, 0.0),
                    Vector3::new(0.0, 2.0, 0.0),
                    Vector3::new(3.0, 0.0, 0.0),
                ],
            ),
            KeyframeTrack::quaternion(
                cube,
                TrackTarget::Quaternion,
                vec![0.0, 2.0],
                vec![
                    Quaternion::identity(),
                    Quaternion::from_axis_angle(Vector3::new(0.0, 1.0, 0.0), std::f32::consts::PI),
                ],
            ),
            KeyframeTrack::vector(
                cube,
                TrackTarget::Scale,
                vec![0.0, 0.5, 1.5, 2.0],
                vec![
                    Vector3::new(1.0, 1.0, 1.0),
                    Vector3::new(1.4, 1.4, 1.4),
                    Vector3::new(0.6, 0.6, 0.6),
                    Vector3::new(1.0, 1.0, 1.0),
                ],
            ),
        ],
    );

    (arena, vec![cube, plinth], vec![clip])
}

fn inspect(path: &str) {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };
    let scene = match UsdLoader::parse(&bytes) {
        Ok(scene) => scene,
        Err(e) => {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        }
    };

    println!(
        "{path}: {} root prims, {} metres per unit, {} up",
        scene.roots.len(),
        scene.meters_per_unit,
        scene.up_axis
    );
    for root in &scene.roots {
        print_object(&scene, *root, 1);
    }

    if scene.animations.is_empty() {
        println!("  (no animation)");
        return;
    }
    for clip in &scene.animations {
        println!(
            "\n  clip {:?}: {:.3}s, {} tracks",
            clip.name,
            clip.duration,
            clip.tracks.len()
        );
        for track in &clip.tracks {
            let name = scene
                .arena
                .get(track.object)
                .map(|o| o.name.clone())
                .unwrap_or_default();
            println!(
                "    {name:<10} {:<12} {} keys over {:.2}s",
                format!("{:?}", track.target),
                track.times.len(),
                track.times.last().copied().unwrap_or(0.0)
            );
        }
    }
}

fn print_object(scene: &threers::loaders::UsdScene, id: threers::core::ObjectId, depth: usize) {
    let Some(object) = scene.arena.get(id) else {
        return;
    };
    let kind = match &object.kind {
        threers::core::ObjectKind::Mesh(mesh) => format!(
            "Mesh ({} verts)",
            mesh.geometry
                .get_attribute("position")
                .map(|a| a.count())
                .unwrap_or(0)
        ),
        _ => "Group".to_string(),
    };
    println!(
        "{:indent$}{} — {kind}",
        "",
        object.name,
        indent = depth * 2
    );
    for child in &object.children {
        print_object(scene, *child, depth + 1);
    }
}
