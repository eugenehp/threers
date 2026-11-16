//! A scene written with `threers_ios::write_usdz` opened by RealityKit
//! itself (`Entity(contentsOf:)`, what AR Quick Look draws with): its
//! meshes found, their materials physically based, their maps attached.
//! `tests/usdz_check.swift` is compiled for it with the system's swiftc.
#![cfg(target_os = "macos")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use threers_ios::threers::core::{Mesh, Object3D};
use threers_ios::threers::geometries::{BoxGeometry, SphereGeometry};
use threers_ios::threers::materials::{Material, PhysicalMaterial, StandardMaterial};
use threers_ios::threers::math::{Color, Vector3};
use threers_ios::threers::scene::Scene;
use threers_ios::threers::textures::{Texture, TextureFormat};

fn tmp() -> PathBuf {
    PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("realitykit-usdz")
}

/// The checker, compiled once.
fn checker() -> PathBuf {
    let dir = tmp();
    std::fs::create_dir_all(&dir).unwrap();
    let exe = dir.join("usdz_check");
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/usdz_check.swift");
    let status = Command::new("xcrun")
        .args(["--sdk", "macosx", "swiftc", "-O", "-parse-as-library", "-target", "arm64-apple-macos15.0"])
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("xcrun swiftc: this test needs Xcode's command-line tools");
    assert!(status.success(), "compiling {}", src.display());
    exe
}

#[test]
fn realitykit_opens_a_written_scene_with_its_maps() {
    let mut scene = Scene::new();
    let n = 8u32;
    let px: Vec<u8> = (0..n * n).flat_map(|i| if i % 2 == 0 { [255, 0, 0, 255] } else { [0, 0, 255, 255] }).collect();
    let lit = StandardMaterial {
        map: Some(Arc::new(Texture::new(n, n, TextureFormat::Rgba8UnormSrgb, px))),
        normal_map: Some(Arc::new(Texture::new(n, n, TextureFormat::Rgba8Unorm, [128, 128, 255, 255].repeat((n * n) as usize)))),
        ..Default::default()
    };
    let mut boxed = Object3D::mesh(Mesh::new(BoxGeometry::new(1.0, 1.0, 1.0), Material::Standard(lit)));
    boxed.name = "Box".into();
    scene.add(boxed);
    let coat = PhysicalMaterial {
        color: Color::new(0.8, 0.1, 0.1),
        clearcoat: 1.0,
        ..Default::default()
    };
    let mut ball = Object3D::mesh(Mesh::new(SphereGeometry::new(0.5, 16, 8), Material::Physical(coat)));
    ball.name = "Ball".into();
    ball.position = Vector3::new(1.5, 0.0, 0.0);
    scene.add(ball);

    let path = tmp().join("scene.usdz");
    std::fs::create_dir_all(tmp()).unwrap();
    threers_ios::write_usdz(&mut scene, &path).unwrap();
    let out = Command::new(checker()).arg(&path).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "RealityKit could not open it: {text}{}", String::from_utf8_lossy(&out.stderr));
    let models: Vec<&str> = text.lines().filter(|l| l.starts_with("model")).collect();
    assert_eq!(models.len(), 2, "{text}");
    let boxed = models.iter().find(|l| l.contains("Box")).unwrap_or_else(|| panic!("{text}"));
    assert!(boxed.contains("triangles=12"), "{boxed}");
    assert!(boxed.contains("pbr+color+normal"), "{boxed}");
    let ball = models.iter().find(|l| l.contains("Ball")).unwrap_or_else(|| panic!("{text}"));
    assert!(ball.contains("pbr"), "{ball}");
}
