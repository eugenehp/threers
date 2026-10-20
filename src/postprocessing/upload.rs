//! Textures a pass samples rather than renders — TexturePass's map, a LUT,
//! SMAA's tables, CubeTexturePass's faces — brought onto the GPU as
//! half-float targets, so every post-fx kind reads them the way it reads a
//! rendered buffer: linear, through the same bindings.

use std::sync::Arc;

use super::target::{Target, INTERMEDIATE_FORMAT};
use crate::renderer::Renderer;
use crate::textures::{CubeTexture, Texture, TextureFormat};

/// `data` (`format`) as linear RGBA texels. `None` for a block-compressed
/// format, which has no texels to read here.
pub fn texels(format: TextureFormat, data: &[u8], count: usize) -> Option<Vec<[f32; 4]>> {
    let mut out = Vec::with_capacity(count);
    match format {
        TextureFormat::Rgba8UnormSrgb => {
            for p in data.chunks_exact(4).take(count) {
                let d = crate::utils::srgb::srgb_to_linear;
                out.push([d(p[0]), d(p[1]), d(p[2]), p[3] as f32 / 255.0]);
            }
        }
        TextureFormat::Rgba8Unorm => {
            for p in data.chunks_exact(4).take(count) {
                out.push([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                    p[3] as f32 / 255.0,
                ]);
            }
        }
        TextureFormat::R8Unorm => {
            for &v in data.iter().take(count) {
                let v = v as f32 / 255.0;
                out.push([v, v, v, 1.0]);
            }
        }
        TextureFormat::Rgba16Float => {
            for p in data.chunks_exact(8).take(count) {
                let h = |i: usize| {
                    crate::renderer::gpu_texture::f16_bits_to_f32(u16::from_le_bytes([p[i], p[i + 1]]))
                };
                out.push([h(0), h(2), h(4), h(6)]);
            }
        }
        _ => return None,
    }
    (out.len() == count).then_some(out)
}

/// A texture as a pass binds it: the render target behind it when it is one
/// (`external_rt_id`), else its pixels uploaded once into a half-float target
/// kept in `slot` (re-uploaded when the texture changes).
pub fn texture_target(
    renderer: &mut Renderer,
    tex: &Arc<Texture>,
    slot: &mut Option<(u64, Target)>,
) -> Option<Target> {
    if let Some(id) = tex.external_rt_id {
        return renderer
            .registered_render_target(id)
            .map(|rt| Target::from_registered(id, rt));
    }
    let key = tex.id ^ (Arc::as_ptr(&tex.data) as u64).rotate_left(17);
    if let Some((k, t)) = slot {
        if *k == key && t.size() == (tex.width.max(1), tex.height.max(1)) {
            return Some(t.clone());
        }
    }
    let count = (tex.width * tex.height) as usize;
    let texels = texels(tex.format, &tex.data, count)?;
    let t = match slot.take() {
        Some((_, t)) if t.size() == (tex.width.max(1), tex.height.max(1)) => t,
        Some((_, t)) => {
            renderer.unregister_render_target(t.id());
            Target::new(renderer, tex.width, tex.height, INTERMEDIATE_FORMAT)
        }
        None => Target::new(renderer, tex.width, tex.height, INTERMEDIATE_FORMAT),
    };
    renderer.upload_render_target_f16(t.id(), &texels);
    *slot = Some((key, t.clone()));
    Some(t)
}

/// A cube texture's six faces side by side (+x, -x, +y, -y, +z, -z), each
/// `size × size` and in the order its data is stored — what kind 66 reads.
pub fn cube_atlas(
    renderer: &mut Renderer,
    cube: &Arc<CubeTexture>,
    slot: &mut Option<(usize, Target)>,
) -> Option<Target> {
    let key = Arc::as_ptr(cube) as usize;
    if let Some((k, t)) = slot {
        if *k == key {
            return Some(t.clone());
        }
    }
    let size = cube.size.max(1);
    let n = (size * size) as usize;
    let mut faces: Vec<Vec<[f32; 4]>> = Vec::with_capacity(6);
    for f in 0..6 {
        let face = match &cube.faces_f32 {
            Some(hdr) => hdr[f]
                .chunks_exact(4)
                .take(n)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
            None => texels(cube.format, &cube.faces[f], n)?,
        };
        if face.len() != n {
            return None;
        }
        faces.push(face);
    }
    let w = size * 6;
    let mut atlas = vec![[0f32; 4]; (w * size) as usize];
    for (f, face) in faces.iter().enumerate() {
        for y in 0..size {
            for x in 0..size {
                atlas[(y * w + f as u32 * size + x) as usize] = face[(y * size + x) as usize];
            }
        }
    }
    if let Some((_, old)) = slot.take() {
        renderer.unregister_render_target(old.id());
    }
    let t = Target::new(renderer, w, size, INTERMEDIATE_FORMAT);
    renderer.upload_render_target_f16(t.id(), &atlas);
    *slot = Some((key, t.clone()));
    Some(t)
}
