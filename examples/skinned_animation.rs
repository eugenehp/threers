//! Skinning and the animation mixer: three tentacles, each a tapered tube
//! bound to a chain of eight bones, sway in one clip and then cross-fade
//! into a second that curls them over. Four frames side by side in
//! `out/skinned_animation.png`: swaying, swaying, half way through the
//! cross-fade, curled.
//!
//! The rig is built here rather than loaded, so the example needs no asset:
//! bones are scene nodes, each vertex is weighted between the two bones
//! nearest it, and the clips are keyframed quaternion tracks on the bones.
//!
//! ```bash
//! cargo run --release --example skinned_animation
//! ```

use std::f32::consts::{PI, TAU};
use threers::animation::{AnimationClip, AnimationMixer, KeyframeTrack, TrackTarget};
use threers::core::{Bone, BufferAttribute, Skeleton, SkinnedMesh};
use threers::lights::Light;
use threers::prelude::*;

const BONES: usize = 8;
const SEGMENT: f32 = 0.34;
const W: u32 = 360;
const H: u32 = 420;

/// A tube standing on `base`, its bones a chain up its middle. Returns the
/// bones, root first.
fn tentacle(scene: &mut Scene, base: Vector3, color: u32) -> Vec<ObjectId> {
    let height = SEGMENT * BONES as f32;
    let mut bones = Vec::with_capacity(BONES);
    let mut parent = None;
    for i in 0..BONES {
        let mut b = Object3D::group();
        b.position = if i == 0 {
            base
        } else {
            Vector3::new(0.0, SEGMENT, 0.0)
        };
        let id = match parent {
            None => scene.add(b),
            Some(p) => scene.add_to(p, b),
        };
        bones.push(id);
        parent = Some(id);
    }
    // Tapered, and moved from centred on the origin to standing on `base`.
    let mut g = CylinderGeometry::new(0.07, 0.26, height, 40, 96, false, 0.0, TAU);
    let mut pos = g.get_attribute("position").unwrap().clone();
    let count = pos.count();
    let mut joints = Vec::with_capacity(count * 4);
    let mut weights = Vec::with_capacity(count * 4);
    for v in 0..count {
        // In the bind pose, where the bones are: standing on `base`.
        let y = pos.array[v * 3 + 1] + height / 2.0;
        pos.array[v * 3] += base.x;
        pos.array[v * 3 + 1] = y + base.y;
        pos.array[v * 3 + 2] += base.z;
        // Between the bone below and the one above, eased so a joint bends
        // as a curve rather than a crease.
        let f = (y / SEGMENT).clamp(0.0, (BONES - 1) as f32);
        let i = (f.floor() as usize).min(BONES - 2);
        let t = f - i as f32;
        let t = t * t * (3.0 - 2.0 * t);
        joints.extend([i as f32, (i + 1) as f32, 0.0, 0.0]);
        weights.extend([1.0 - t, t, 0.0, 0.0]);
    }
    g.set_attribute("position", pos);
    g.set_attribute("joint", BufferAttribute::new(joints, 4));
    g.set_attribute("weight", BufferAttribute::new(weights, 4));
    // Bind pose: bone i sits `i` segments up from the base.
    let skeleton = Skeleton::new(
        (0..BONES)
            .map(|i| Bone {
                node: bones[i],
                inverse_bind: Matrix4::translation(
                    -(base + Vector3::new(0.0, SEGMENT * i as f32, 0.0)),
                ),
            })
            .collect(),
    );
    let mut m = StandardMaterial::new(Color::from_hex(color));
    m.roughness = 0.45;
    m.metalness = 0.0;
    let mut body = Object3D::skinned_mesh(SkinnedMesh::new(g, Material::Standard(m), skeleton));
    body.cast_shadow = true;
    body.receive_shadow = true;
    scene.add(body);
    bones
}

/// One quaternion track per bone, sampled from `angle(bone, time)` about
/// `axis` over a two-second loop.
fn clip(
    name: &str,
    bones: &[ObjectId],
    axis: Vector3,
    angle: impl Fn(usize, f32) -> f32,
) -> AnimationClip {
    let keys = 33;
    let duration = 2.0;
    let times: Vec<f32> = (0..keys)
        .map(|k| duration * k as f32 / (keys - 1) as f32)
        .collect();
    let tracks = bones
        .iter()
        .enumerate()
        .map(|(i, &bone)| {
            let values = times
                .iter()
                .map(|&t| Quaternion::from_axis_angle(axis, angle(i, t)))
                .collect();
            KeyframeTrack::quaternion(bone, TrackTarget::Quaternion, times.clone(), values)
        })
        .collect();
    AnimationClip::new(name, duration, tracks)
}

fn main() {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(0x10161e);
    scene.add_light(HemisphereLight::new(
        Color::from_hex(0x9fc4ff),
        Color::from_hex(0x2a2018),
        0.6,
    ));
    let mut sun = DirectionalLight::new(Color::from_hex(0xfff0dc), 2.6);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 8.0;
    sun.shadow.normal_bias = 0.02;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(-3.0, 6.0, 4.0);
    scene.add(light);
    let mut floor_m = StandardMaterial::new(Color::from_hex(0x3a4250));
    floor_m.roughness = 0.9;
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(30.0, 30.0), floor_m.into()));
    floor.rotate_x(-PI / 2.0);
    floor.receive_shadow = true;
    scene.add(floor);

    let mut mixer = AnimationMixer::new();
    let mut sways = Vec::new();
    let mut curls = Vec::new();
    for (k, (x, color)) in [(-1.3f32, 0xe0604a), (0.0, 0xf0b040), (1.3, 0x4ab0a0)]
        .into_iter()
        .enumerate()
    {
        let bones = tentacle(&mut scene, Vector3::new(x, 0.0, 0.0), color);
        let phase = k as f32 * 1.1;
        // A travelling wave: each bone a little behind the one below it.
        let sway = clip("sway", &bones, Vector3::new(0.0, 0.0, 1.0), move |i, t| {
            let reach = if i == 0 { 0.15 } else { 0.22 };
            reach * (TAU * t / 2.0 - 0.7 * i as f32 + phase).sin()
        });
        // Curled over towards the camera, breathing a little.
        let curl = clip("curl", &bones, Vector3::new(1.0, 0.0, 0.0), move |i, t| {
            let breathe = 1.0 + 0.08 * (TAU * t / 2.0 + phase).sin();
            if i == 0 {
                0.1
            } else {
                0.24 * breathe
            }
        });
        let s = mixer.clip_action(sway);
        mixer.actions[s].play();
        sways.push(s);
        curls.push(mixer.clip_action(curl));
    }

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .build()
        .expect("a GPU adapter");
    hr.set_msaa(4);
    let mut camera = PerspectiveCamera::new(42.0, W as f32 / H as f32, 0.1, 50.0);
    camera.position = Vector3::new(0.0, 2.3, 7.4);
    camera.look_at(Vector3::new(0.0, 1.35, 0.0));

    // Frames at these times; the cross-fade runs from 1.2 s to 2.4 s.
    let shots = [0.3f32, 0.8, 1.8, 3.2];
    let fade_at = 1.2;
    let dt = 1.0 / 60.0;
    let mut t = 0.0f32;
    let mut faded = false;
    let mut sheet = vec![0u8; (W * shots.len() as u32 * H * 4) as usize];
    for (slot, &shot) in shots.iter().enumerate() {
        while t < shot {
            if !faded && t >= fade_at {
                for (&s, &c) in sways.iter().zip(&curls) {
                    mixer.cross_fade(s, c, 1.2);
                }
                faded = true;
            }
            mixer.update(&mut scene, dt);
            t += dt;
        }
        let px = hr.render_to_rgba(&mut scene, &camera);
        let stride = W * shots.len() as u32;
        for y in 0..H {
            let row = &px[(y * W * 4) as usize..((y + 1) * W * 4) as usize];
            let at = ((y * stride + slot as u32 * W) * 4) as usize;
            sheet[at..at + row.len()].copy_from_slice(row);
        }
    }
    std::fs::create_dir_all("out").unwrap();
    std::fs::write(
        "out/skinned_animation.png",
        encode_png(W * shots.len() as u32, H, &sheet),
    )
    .unwrap();
    println!("wrote out/skinned_animation.png (sway, sway, cross-fading, curled)");
}
