//! three.js's per-material `depthTest`, `depthWrite`, `depthFunc` and
//! `colorWrite`, on every kind of draw.
//!
//! An orthographic camera looks down −z at a red wall (z = +0.5, nearer) and a
//! green subject behind it (z = −0.5) over a blue background. Both are opaque,
//! so three.js draws them in `renderOrder`: the wall first, the subject second.
//! What lands on the centre pixel says which state took effect:
//!
//! - the subject with `depthTest: false` shows through the wall (WebGL turns
//!   writes off with the test, so nothing it draws occludes later draws);
//! - a wall with `depthWrite: false` lets the subject through;
//! - a wall with `colorWrite: false` is invisible yet still hides the subject,
//!   leaving the background;
//! - `depthFunc` on the subject: `Greater` and `Always` pass behind the wall,
//!   `Never` never passes;
//! - the same plane drawn twice at one depth: three.js's default `LessEqual`
//!   lets the second draw win, `Less` keeps the first, `Equal` passes,
//!   `NotEqual` fails.
//!
//! The subject is drawn as each kind of object three.js has — mesh, line,
//! points, sprite, instanced mesh, skinned mesh — since each takes its own
//! pipeline. Shared by `tests/material_state.rs` (wgpu) and
//! `tests/backend_parity.rs` (Metal) through `#[path]`.
#![allow(dead_code)]

use std::sync::Arc;
use threers::cameras::{Camera, OrthographicCamera};
use threers::core::{
    Bone, BufferAttribute, BufferGeometry, InstancedMesh, LineSegments, Mesh, Object3D, ObjectId,
    Points, Skeleton, SkinnedMesh, Sprite,
};
use threers::geometries::PlaneGeometry;
use threers::materials::{
    BasicMaterial, DepthFunc, LineBasicMaterial, Material, PointsMaterial, RenderState,
    SpriteMaterial,
};
use threers::math::{Color, Matrix4, Vector3};
use threers::scene::Scene;

pub const N: u32 = 32;
pub const WALL: u32 = 0xff0000;
pub const SUBJECT: u32 = 0x00ff00;
pub const BACKGROUND: u32 = 0x0000ff;

pub fn camera() -> OrthographicCamera {
    let mut cam = OrthographicCamera::new(-1.0, 1.0, 1.0, -1.0, 0.1, 10.0);
    cam.position = Vector3::new(0.0, 0.0, 2.0);
    cam.target = Vector3::ZERO;
    cam
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Mesh,
    Line,
    Points,
    Sprite,
    Instanced,
    Skinned,
}

pub const KINDS: [Kind; 6] = [
    Kind::Mesh,
    Kind::Line,
    Kind::Points,
    Kind::Sprite,
    Kind::Instanced,
    Kind::Skinned,
];

/// One configuration and the colour three.js leaves at the centre.
pub struct Case {
    pub name: &'static str,
    pub wall: RenderState,
    pub subject: RenderState,
    /// The subject at the wall's depth rather than behind it.
    pub coplanar: bool,
    pub expect: u32,
}

fn st(f: impl FnOnce(&mut RenderState)) -> RenderState {
    let mut s = RenderState::default();
    f(&mut s);
    s
}

pub fn cases() -> Vec<Case> {
    let d = RenderState::default;
    let func = |f: DepthFunc| st(|s| s.depth_func = f);
    vec![
        Case { name: "defaults: the wall hides the subject", wall: d(), subject: d(), coplanar: false, expect: WALL },
        Case { name: "subject depthTest false", wall: d(), subject: st(|s| s.depth_test = false), coplanar: false, expect: SUBJECT },
        Case { name: "wall depthWrite false", wall: st(|s| s.depth_write = Some(false)), subject: d(), coplanar: false, expect: SUBJECT },
        Case { name: "wall colorWrite false", wall: st(|s| s.color_write = false), subject: d(), coplanar: false, expect: BACKGROUND },
        Case { name: "subject depthFunc Greater", wall: d(), subject: func(DepthFunc::Greater), coplanar: false, expect: SUBJECT },
        Case { name: "subject depthFunc Always", wall: d(), subject: func(DepthFunc::Always), coplanar: false, expect: SUBJECT },
        Case { name: "subject depthFunc Never", wall: d(), subject: func(DepthFunc::Never), coplanar: false, expect: WALL },
        Case { name: "subject depthFunc GreaterEqual", wall: d(), subject: func(DepthFunc::GreaterEqual), coplanar: false, expect: SUBJECT },
    ]
}

/// The coplanar cases: only meaningful where the two draws produce the very
/// same depths, which is two plain meshes through the same vertex path.
pub fn coplanar_cases() -> Vec<Case> {
    let d = RenderState::default;
    let func = |f: DepthFunc| st(|s| s.depth_func = f);
    vec![
        Case { name: "coplanar, default LessEqual", wall: d(), subject: d(), coplanar: true, expect: SUBJECT },
        Case { name: "coplanar, Less", wall: d(), subject: func(DepthFunc::Less), coplanar: true, expect: WALL },
        Case { name: "coplanar, Equal", wall: d(), subject: func(DepthFunc::Equal), coplanar: true, expect: SUBJECT },
        Case { name: "coplanar, NotEqual", wall: d(), subject: func(DepthFunc::NotEqual), coplanar: true, expect: WALL },
    ]
}

fn plane_material(hex: u32, state: RenderState) -> Material {
    let mut m = BasicMaterial::new(Color::from_hex(hex));
    m.render_state = state;
    m.into()
}

/// The subject, drawn as `kind`, at depth `z`.
fn subject(kind: Kind, state: RenderState, z: f32) -> Object3D {
    let mut o = match kind {
        Kind::Mesh => Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), plane_material(SUBJECT, state))),
        Kind::Line => {
            // Through the centre row of pixels: row N/2's centre is at
            // y = 1 − (N/2 + ½)·2/N.
            let y = 1.0 - (N as f32 / 2.0 + 0.5) * 2.0 / N as f32;
            let mut g = BufferGeometry::new();
            g.set_attribute("position", BufferAttribute::new(vec![-2.0, y, 0.0, 2.0, y, 0.0], 3));
            let mut m = LineBasicMaterial::new(Color::from_hex(SUBJECT));
            m.render_state = state;
            Object3D::line_segments(LineSegments::new(g, Material::Line(m)))
        }
        Kind::Points => {
            let mut g = BufferGeometry::new();
            g.set_attribute("position", BufferAttribute::new(vec![0.0, 0.0, 0.0], 3));
            let mut m = PointsMaterial::new(Color::from_hex(SUBJECT), 200.0);
            m.render_state = state;
            Object3D::points(Points::new(g, Material::Points(m)))
        }
        Kind::Sprite => {
            let mut m = SpriteMaterial::new(Color::from_hex(SUBJECT));
            m.render_state = state;
            let mut o = Object3D::sprite(Sprite::new(Material::Sprite(m)));
            o.scale = Vector3::new(2.5, 2.5, 1.0);
            o
        }
        Kind::Instanced => {
            let mut im = InstancedMesh::from_arc(
                Arc::new(PlaneGeometry::new(2.5, 2.5)),
                Arc::new(plane_material(SUBJECT, state)),
                1,
            );
            im.set_matrix_at(0, Matrix4::identity());
            Object3D::instanced_mesh(im)
        }
        Kind::Skinned => {
            let mut g = PlaneGeometry::new(2.5, 2.5);
            let n = g.get_attribute("position").unwrap().count();
            g.set_attribute("joint", BufferAttribute::new(vec![0.0; n * 4], 4));
            g.set_attribute("weight", BufferAttribute::new([1.0, 0.0, 0.0, 0.0].repeat(n), 4));
            let mut skeleton = Skeleton::new(vec![Bone {
                node: ObjectId::default(),
                inverse_bind: Matrix4::identity(),
            }]);
            skeleton.bone_matrices = vec![Matrix4::identity()];
            Object3D::skinned_mesh(SkinnedMesh::new(g, plane_material(SUBJECT, state), skeleton))
        }
    };
    o.position = Vector3::new(0.0, 0.0, z);
    o.render_order = 1;
    o
}

pub fn scene(kind: Kind, case: &Case) -> Scene {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(BACKGROUND);
    let mut wall = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), plane_material(WALL, case.wall)));
    wall.position = Vector3::new(0.0, 0.0, 0.5);
    wall.render_order = 0;
    scene.add(wall);
    scene.add(subject(kind, case.subject, if case.coplanar { 0.5 } else { -0.5 }));
    scene
}

fn rgb(hex: u32) -> [i32; 3] {
    [((hex >> 16) & 0xff) as i32, ((hex >> 8) & 0xff) as i32, (hex & 0xff) as i32]
}

/// Which of the three colours the centre pixel is nearest.
pub fn classify(rgba: &[u8]) -> (u32, [u8; 3]) {
    let i = ((N / 2 * N + N / 2) * 4) as usize;
    let got = [rgba[i], rgba[i + 1], rgba[i + 2]];
    let dist = |hex: u32| {
        let c = rgb(hex);
        (0..3).map(|k| (got[k] as i32 - c[k]).pow(2)).sum::<i32>()
    };
    let best = [WALL, SUBJECT, BACKGROUND].into_iter().min_by_key(|&h| dist(h)).unwrap();
    (best, got)
}

fn name(hex: u32) -> &'static str {
    match hex {
        WALL => "wall",
        SUBJECT => "subject",
        _ => "background",
    }
}

/// Every case for `kinds`, plus the coplanar ones on a plain mesh, through
/// `render`; returns the failures.
pub fn check(
    backend: &str,
    kinds: &[Kind],
    render: &mut dyn FnMut(&mut Scene, &dyn Camera) -> Vec<u8>,
) -> Vec<String> {
    let cam = camera();
    let mut failures = Vec::new();
    let mut run = |kind: Kind, case: &Case, failures: &mut Vec<String>| {
        let rgba = render(&mut scene(kind, case), &cam);
        let (got, px) = classify(&rgba);
        eprintln!("{backend} {kind:?} {}: {} {px:?}", case.name, name(got));
        if got != case.expect {
            failures.push(format!(
                "{backend} {kind:?} — {}: centre is the {} ({px:?}), three.js draws the {}",
                case.name,
                name(got),
                name(case.expect)
            ));
        }
    };
    for &kind in kinds {
        for case in cases() {
            run(kind, &case, &mut failures);
        }
    }
    for case in coplanar_cases() {
        run(Kind::Mesh, &case, &mut failures);
    }
    failures
}
