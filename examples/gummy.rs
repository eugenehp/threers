//! The gummy creatures, natively: a jellyfish (or `--octopus`) simulated on
//! `threers::softbody`, held at rest, then pulled by a tentacle until it tears,
//! then left a moment to start growing back — and the frame drawn with the
//! wgpu renderer and, with `--features metal`, threers' own Metal renderer.
//!
//! ```text
//! cargo run --release --example gummy
//! cargo run --release --example gummy --features metal -- --octopus
//! ```
//!
//! Writes `out/gummy_wgpu.png` (and `out/gummy_metal.png`). The same creatures
//! run in the browser in `web/examples/jelly.html`, on WebGPU or WebGL2.

use std::f32::consts::PI;

use threers::lights::Light;
use threers::materials::TransparencyMode;
use threers::math::Vector3;
use threers::prelude::*;
use threers::sdf::{Sdf3, Shape};
use threers::softbody::{carry, ChainSpec, Creature, Event, Marks, ShapeSpec, SoftBody};
use threers::ToneMapping;

include!("gummy_creatures.inc");

const W: u32 = 960;
const H: u32 = 600;

/// A body pulled by its first part until it tears, then left to regrow a while.
fn simulate(c: Creature, rest: usize, after: usize) -> (SoftBody, Option<usize>) {
    let mut sim = c.into_soft_body(7);
    for _ in 0..rest {
        sim.step(1.0 / 60.0);
    }
    let slot = sim.leg_slots[0][0];
    let p = sim.pivot_now(0);
    let hand = [p[0] * 1.6, p[1] + 0.02, p[2] * 1.6];
    sim.start_grab(hand, slot);
    let mut torn = None;
    for frame in 1..=240 {
        if let Some(g) = sim.grab.as_mut() {
            let k = frame as f64;
            g.target = [hand[0] * (1.0 + 0.012 * k), hand[1] + 0.004 * k, hand[2] * (1.0 + 0.012 * k)];
        }
        sim.step(1.0 / 60.0);
        if sim.events.drain(..).any(|e| matches!(e, Event::Tear { .. })) {
            torn = Some(frame);
            break;
        }
    }
    sim.grab = None;
    for _ in 0..after {
        sim.step(1.0 / 60.0);
        sim.events.clear();
    }
    (sim, torn)
}

fn geometry(pos: Vec<f32>, nrm: Vec<f32>, idx: Vec<u32>) -> BufferGeometry {
    let mut g = BufferGeometry::new();
    g.set_attribute("position", BufferAttribute::new(pos, 3));
    g.set_attribute("normal", BufferAttribute::new(nrm, 3));
    g.set_index(idx);
    g
}

/// A tapered capsule for every chain segment, as the browser draws them.
fn chain_geometry(sim: &SoftBody) -> Option<BufferGeometry> {
    let c = &sim.chains;
    let (sides, rings) = (7usize, 6usize);
    let (mut pos, mut nrm, mut idx) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..c.parent.len() {
        let Ok(p) = usize::try_from(c.parent[i]) else { continue };
        let a = Vector3::new(sim.gx[3 * p], sim.gx[3 * p + 1], sim.gx[3 * p + 2]);
        let b = Vector3::new(sim.gx[3 * i], sim.gx[3 * i + 1], sim.gx[3 * i + 2]);
        let d = (b - a).normalize();
        let h = if d.y.abs() > 0.9 { Vector3::new(1.0, 0.0, 0.0) } else { Vector3::new(0.0, 1.0, 0.0) };
        let e1 = d.cross(h).normalize();
        let e2 = d.cross(e1);
        let base = (pos.len() / 3) as u32;
        // rings from one cap's pole to the other's
        for r in 0..rings {
            let s = r as f32 / (rings - 1) as f32;
            let (t, ring, cap) = match r {
                0 => (0.0, 0.0, -1.0),
                1 => (0.0, 0.7, -0.7),
                2 => (0.0, 1.0, 0.0),
                3 => (1.0, 1.0, 0.0),
                4 => (1.0, 0.7, 0.7),
                _ => (1.0, 0.0, 1.0),
            };
            let _ = s;
            let rad = c.seg_radius[i] + (c.radius[i] - c.seg_radius[i]) * t;
            for k in 0..sides {
                let ang = k as f32 / sides as f32 * 2.0 * PI;
                let o = e1 * (ang.cos() * ring) + e2 * (ang.sin() * ring) + d * cap;
                let q = a + (b - a) * t + o * rad;
                pos.extend([q.x, q.y, q.z]);
                let n = o.normalize();
                nrm.extend([n.x, n.y, n.z]);
            }
        }
        for r in 0..rings as u32 - 1 {
            for k in 0..sides as u32 {
                let (s, k1) = (sides as u32, (k + 1) % sides as u32);
                let (i0, i1, i2, i3) = (base + r * s + k, base + r * s + k1, base + (r + 1) * s + k, base + (r + 1) * s + k1);
                idx.extend([i0, i2, i1, i1, i2, i3]);
            }
        }
    }
    (!idx.is_empty()).then(|| geometry(pos, nrm, idx))
}

fn scene(sim: &SoftBody, skins: &[threers::sdf::SurfaceNets], octopus: bool) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0xe4e1dc);
    s.environment = Some(threers::extras::RoomEnvironment::pmrem(128));
    s.add_light(HemisphereLight::new(Color::from_hex(0xffffff), Color::from_hex(0xd8d2c8), 0.9));
    let mut sun = DirectionalLight::new(Color::WHITE, 2.2);
    sun.cast_shadow = true;
    sun.shadow.map_size = 2048;
    sun.shadow.camera_size = 2.6;
    sun.shadow.camera_near = 0.5;
    sun.shadow.camera_far = 14.0;
    sun.shadow.radius = 4.0;
    let mut light = Object3D::light(Light::Directional(sun));
    light.position = Vector3::new(-2.3, 5.6, -3.0);
    s.add(light);
    let mut floor = Object3D::mesh(Mesh::new(PlaneGeometry::new(40.0, 40.0), Material::Standard(StandardMaterial::new(Color::from_hex(0xe4e1dc)))));
    floor.rotate_x(-PI / 2.0);
    floor.receive_shadow = true;
    s.add(floor);

    // the gel: a moon jelly's clear blue, or a common octopus's red
    let (tint, depth) = if octopus { (Color::new(0.95, 0.38, 0.28), 3.8) } else { (Color::new(0.8, 0.88, 0.98), 2.2) };
    let gel = || {
        let m = PhysicalMaterial::new(tint)
            .with_roughness(0.05)
            .with_transmission(1.0, 1.42, 0.4)
            .with_attenuation(tint, 1.0 / depth)
            .with_transparency(TransparencyMode::Refract);
        Material::Physical(m)
    };
    for (bi, b) in sim.blocks.iter().enumerate() {
        if !b.active {
            continue;
        }
        let topo = if b.leg < 0 { 0 } else { b.leg as usize + 1 };
        let skin = &skins[topo];
        let emb = if topo == 0 { &sim.body.skin } else { &sim.limbs[topo - 1].part.skin };
        let rest_n: Vec<f32> = skin.normal.iter().flat_map(|n| [n.x, n.y, n.z]).collect();
        let (mut pos, mut nrm) = (vec![0.0f32; rest_n.len()], vec![0.0f32; rest_n.len()]);
        carry(sim, topo, emb, &rest_n, bi, &mut pos, &mut nrm);
        let mut m = Object3D::mesh(Mesh::new(geometry(pos, nrm, skin.index.clone()), gel()));
        m.cast_shadow = true;
        s.add(m);
    }
    // the jellyfish's gonads, inside the bell and carried with it
    if !octopus {
        let rings = Shape::union_all(
            (0..4).map(|k| {
                let a = k as f32 * PI / 2.0;
                Shape::torus(0.085, 0.02).translate(Vector3::new(0.15 * a.cos(), 0.34, 0.15 * a.sin()))
            })
            .collect(),
            0.0,
        );
        let (lo, hi) = rings.bounds().unwrap();
        let g = rings.to_surface_nets(0.007, lo - Vector3::new(0.01, 0.01, 0.01), hi + Vector3::new(0.01, 0.01, 0.01));
        let pts: Vec<f32> = g.position.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
        let rest_n: Vec<f32> = g.normal.iter().flat_map(|n| [n.x, n.y, n.z]).collect();
        let emb = sim.body.embed(&pts);
        let (mut pos, mut nrm) = (vec![0.0f32; pts.len()], vec![0.0f32; pts.len()]);
        carry(sim, 0, &emb, &rest_n, 0, &mut pos, &mut nrm);
        let mut m = StandardMaterial::new(Color::new(0.86, 0.42, 0.82));
        m.roughness = 0.55;
        s.add(Object3D::mesh(Mesh::new(geometry(pos, nrm, g.index), Material::Standard(m))));
    }
    if let Some(g) = chain_geometry(sim) {
        let m = PhysicalMaterial::new(Color::new(0.86, 0.9, 1.0))
            .with_roughness(0.05)
            .with_transmission(1.0, 1.42, 0.05)
            .with_attenuation(Color::new(0.86, 0.9, 1.0), 0.1)
            .with_transparency(TransparencyMode::Refract);
        let mut o = Object3D::mesh(Mesh::new(g, Material::Physical(m)));
        o.cast_shadow = true;
        s.add(o);
    }
    // the goo the tear threw
    for i in 0..sim.d_count {
        let mut o = Object3D::mesh(Mesh::new(SphereGeometry::new(sim.dr[i], 16, 10), gel()));
        o.position = Vector3::new(sim.dp[3 * i], sim.dp[3 * i + 1], sim.dp[3 * i + 2]);
        o.cast_shadow = true;
        s.add(o);
    }
    s
}

fn camera(octopus: bool) -> PerspectiveCamera {
    let (yaw, pitch, dist): (f32, f32, f32) = if octopus { (0.55, 0.72, 4.2) } else { (0.6, 0.78, 4.0) };
    let target = Vector3::new(0.0, 0.12, 0.0);
    let mut c = PerspectiveCamera::new(30.0, W as f32 / H as f32, 0.1, 60.0);
    c.position = target + Vector3::new(yaw.sin() * pitch.cos(), pitch.sin(), yaw.cos() * pitch.cos()) * dist;
    c.look_at(target);
    c
}

fn main() {
    let octopus = std::env::args().any(|a| a == "--octopus");
    let c = if octopus { octopus_creature() } else { jellyfish() };
    let skins = c.skins.clone();
    let (sim, torn) = simulate(c, 60, 50);
    match torn {
        Some(f) => println!("simulated: the {} came away {f} frames into the pull", sim.limbs[0].name),
        None => println!("simulated: nothing tore"),
    }
    let cam = camera(octopus);
    std::fs::create_dir_all("out").expect("create out/");

    let mut hr = HeadlessRenderer::builder()
        .size(W, H)
        .color_format(threers::wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .expect("a wgpu adapter");
    hr.renderer().set_tone_mapping(ToneMapping::Neutral, 1.0);
    hr.set_msaa(4);
    let wgpu_frame = hr.render_to_rgba_resolved(&mut scene(&sim, &skins, octopus), &cam);
    std::fs::write("out/gummy_wgpu.png", encode_png(W, H, &wgpu_frame)).expect("write png");
    println!("wrote out/gummy_wgpu.png (wgpu)");

    #[cfg(feature = "metal")]
    {
        let mut metal = threers::metal::MetalHeadlessRenderer::builder().size(W, H).build().expect("a Metal device");
        metal.renderer().set_tone_mapping(ToneMapping::Neutral, 1.0);
        let frame = metal.render_to_rgba(&mut scene(&sim, &skins, octopus), &cam).expect("a Metal frame");
        std::fs::write("out/gummy_metal.png", encode_png(W, H, &frame)).expect("write png");
        let diff = frame.iter().zip(&wgpu_frame).map(|(a, b)| (*a as f64 - *b as f64).abs()).sum::<f64>() / frame.len() as f64;
        println!("wrote out/gummy_metal.png (Metal); mean difference from the wgpu frame {diff:.2} levels of 255");
    }
}

fn octopus_creature() -> Creature {
    octopus()
}
