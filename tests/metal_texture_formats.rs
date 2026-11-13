//! Every texture format on the Metal backend, against the wgpu one.
//!
//! The Metal backend names its pixel formats itself, by number. One of those
//! numbers was wrong for as long as it existed — BC7 sRGB as 145, which is no
//! format at all, so every BC7 texture failed to create — and nothing noticed,
//! because nothing drew one. Here each format is created, and drawn on a plane
//! through both backends, which must agree. wgpu's own sampling of these
//! formats is held to the hardware decode in `tests/texture_formats.rs`.

#![cfg(metal_backend)]

use std::sync::Arc;
use threers::cameras::PerspectiveCamera;
use threers::core::{Mesh, Object3D};
use threers::geometries::PlaneGeometry;
use threers::materials::BasicMaterial;
use threers::math::{Color, Vector3};
use threers::metal::{MetalDevice, MetalError, MetalHeadlessRenderer};
use threers::scene::Scene;
use threers::textures::{AstcBlock, Texture, TextureFilter, TextureFormat};

const N: u32 = 96;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A texture of `format`: random blocks, or for uncompressed formats random
/// in-range texels. Nearest sampling and no mip chain, so both backends read
/// exactly level 0's texels.
fn texture(format: TextureFormat, rng: &mut Rng) -> Texture {
    let (bw, bh) = format.block_dim();
    let (w, h) = (bw * (64 / bw).max(1), bh * (64 / bh).max(1));
    let data = if format.is_block_compressed() {
        let mut d: Vec<u8> = (0..format.data_len(w, h))
            .map(|_| rng.next() as u8)
            .collect();
        if let TextureFormat::Astc { .. } = format {
            // Random ASTC bits are nearly all malformed (magenta), which would
            // compare equal however the format were named. Void-extent blocks
            // of random colours are valid at every footprint.
            for b in d.as_chunks_mut::<16>().0 {
                let colour = u128::from_le_bytes(*b) & !((1u128 << 64) - 1);
                *b = (colour | 0x1fc | (3 << 10) | (((1u128 << 52) - 1) << 12)).to_le_bytes();
            }
        }
        d
    } else {
        let n = (w * h) as usize * format.channels();
        let vals: Vec<f32> = (0..n)
            .map(|_| (rng.next() % 1000) as f32 / 1000.0)
            .collect();
        format.pack_f32(&vals).unwrap()
    };
    let mut t = Texture::new(w, h, format, data);
    t.flip_y = false;
    t.mag_filter = TextureFilter::Nearest;
    t.min_filter = TextureFilter::Nearest;
    t.mipmap_filter = None;
    t
}

fn scene(tex: &Texture) -> Scene {
    let mut s = Scene::new();
    s.background = Color::from_hex(0x000000);
    let mut mat = BasicMaterial::new(Color::WHITE);
    mat.map = Some(Arc::new(tex.clone()));
    s.add(Object3D::mesh(Mesh::new(
        PlaneGeometry::new(40.0, 40.0),
        mat.into(),
    )));
    s
}

fn camera() -> PerspectiveCamera {
    let fov: f32 = 30.0;
    let d = N as f32 / (2.0 * (fov.to_radians() / 2.0).tan());
    let mut c = PerspectiveCamera::new(fov, 1.0, 0.1, 1000.0);
    c.position = Vector3::new(0.0, 0.0, d);
    c.look_at(Vector3::ZERO);
    c
}

/// Mean colour of 8x8 tiles, in [0, 1].
fn tiles(px: &[u8]) -> Vec<f64> {
    const G: usize = 12;
    let step = N as usize / G;
    let mut out = Vec::with_capacity(G * G * 3);
    for ty in 0..G {
        for tx in 0..G {
            for c in 0..3 {
                let mut sum = 0.0;
                for y in ty * step..(ty + 1) * step {
                    for x in tx * step..(tx + 1) * step {
                        sum += px[(y * N as usize + x) * 4 + c] as f64 / 255.0;
                    }
                }
                out.push(sum / (step * step) as f64);
            }
        }
    }
    out
}

fn formats() -> Vec<TextureFormat> {
    TextureFormat::all()
        .into_iter()
        .filter(|f| match f {
            // Three ASTC footprints stand for the fourteen: the smallest, an
            // uneven one and the largest.
            TextureFormat::Astc { block, .. } => {
                matches!(
                    block,
                    AstcBlock::B4x4 | AstcBlock::B10x6 | AstcBlock::B12x12
                )
            }
            _ => true,
        })
        .collect()
}

#[test]
fn every_format_creates_on_metal() {
    let device = match MetalDevice::new() {
        Ok(d) => d,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device");
            return;
        }
        Err(e) => panic!("{e:?}"),
    };
    let mut rng = Rng(5);
    let mut failed = Vec::new();
    for format in TextureFormat::all() {
        if let Err(e) =
            threers::metal::resources::upload_texture(&device, &texture(format, &mut rng))
        {
            failed.push(format!("{format:?}: {e:?}"));
        }
    }
    assert!(
        failed.is_empty(),
        "could not create:\n{}",
        failed.join("\n")
    );
}

#[test]
fn every_format_draws_on_metal_as_on_wgpu() {
    let Some(mut w) = threers::HeadlessRenderer::builder().size(N, N).build().ok() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = match MetalHeadlessRenderer::builder().size(N, N).build() {
        Ok(m) => m,
        Err(MetalError::NoDevice) => {
            eprintln!("skipping: no Metal device");
            return;
        }
        Err(e) => panic!("metal renderer: {e:?}"),
    };
    let mut rng = Rng(11);
    let mut failed = Vec::new();
    for format in formats() {
        let tex = texture(format, &mut rng);
        let want = tiles(&w.render_to_rgba(&mut scene(&tex), &camera()));
        let got = tiles(
            &m.render_to_rgba(&mut scene(&tex), &camera())
                .expect("metal render"),
        );
        let rms = (want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / want.len() as f64)
            .sqrt();
        let lit = want.iter().sum::<f64>() / want.len() as f64;
        if rms > 0.02 || lit < 0.02 {
            failed.push(format!(
                "{format:?}: {rms:.4} rms from wgpu (wgpu mean {lit:.3})"
            ));
        }
    }
    assert!(
        failed.is_empty(),
        "Metal draws these differently:\n{}",
        failed.join("\n")
    );
}
