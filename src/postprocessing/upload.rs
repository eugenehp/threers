//! Textures a pass samples rather than renders — TexturePass's map, a LUT,
//! SMAA's tables, CubeTexturePass's faces — brought onto the GPU as
//! half-float targets, so every post-fx kind reads them the way it reads a
//! rendered buffer: linear, through the same bindings.

use std::sync::Arc;

use super::target::{Target, INTERMEDIATE_FORMAT};
use super::PostBackend;
use crate::textures::{CubeTexture, Texture, TextureFormat};

/// `data` (a `width x height` image in `format`) as linear RGBA texels:
/// block formats decoded, the sRGB curve undone, one channel splat across RGB
/// and two read as red and green. `None` when the bytes are too few.
pub fn texels(
    format: TextureFormat,
    data: &[u8],
    width: u32,
    height: u32,
) -> Option<Vec<[f32; 4]>> {
    let count = (width.max(1) * height.max(1)) as usize;
    let decoded;
    let (format, data) = if format.is_block_compressed() {
        decoded = crate::textures::codec::decode(format, data, width, height)?;
        (crate::textures::codec::decoded_format(format), &decoded[..])
    } else {
        (format, data)
    };
    let v = format.texels_f32(data)?;
    let n = format.channels();
    if v.len() < count * n {
        return None;
    }
    let srgb = format.is_srgb();
    let d = |x: f32| {
        if srgb {
            crate::utils::srgb::srgb_to_linear((x * 255.0).round() as u8)
        } else {
            x
        }
    };
    Some(
        (0..count)
            .map(|i| {
                let t = &v[i * n..i * n + n];
                match n {
                    1 => [t[0], t[0], t[0], 1.0],
                    2 => [t[0], t[1], 0.0, 1.0],
                    _ => [d(t[0]), d(t[1]), d(t[2]), t[3]],
                }
            })
            .collect(),
    )
}

/// A texture as a pass binds it: the render target behind it when it is one
/// (`external_rt_id`), else its pixels uploaded once into a half-float target
/// kept in `slot` (re-uploaded when the texture changes).
pub fn texture_target(
    renderer: &mut (impl PostBackend + ?Sized),
    tex: &Arc<Texture>,
    slot: &mut Option<(u64, Target)>,
) -> Option<Target> {
    if let Some(id) = tex.external_rt_id {
        return renderer.registered_target(id);
    }
    let key = tex.id ^ (Arc::as_ptr(&tex.data) as u64).rotate_left(17);
    if let Some((k, t)) = slot
        && *k == key
        && t.size() == (tex.width.max(1), tex.height.max(1))
    {
        return Some(t.clone());
    }
    let texels = texels(tex.format, &tex.data, tex.width, tex.height)?;
    let t = match slot.take() {
        Some((_, t)) if t.size() == (tex.width.max(1), tex.height.max(1)) => t,
        Some((_, t)) => {
            renderer.unregister_render_target(t.id());
            renderer.new_target(tex.width, tex.height, INTERMEDIATE_FORMAT)
        }
        None => renderer.new_target(tex.width, tex.height, INTERMEDIATE_FORMAT),
    };
    renderer.upload_render_target_f16(t.id(), &texels);
    *slot = Some((key, t.clone()));
    Some(t)
}

/// A cube texture's six faces side by side (+x, -x, +y, -y, +z, -z), each
/// `size × size` and in the order its data is stored — what kind 66 reads.
pub fn cube_atlas(
    renderer: &mut (impl PostBackend + ?Sized),
    cube: &Arc<CubeTexture>,
    slot: &mut Option<(usize, Target)>,
) -> Option<Target> {
    let key = Arc::as_ptr(cube) as usize;
    if let Some((k, t)) = slot
        && *k == key
    {
        return Some(t.clone());
    }
    let size = cube.size.max(1);
    let n = (size * size) as usize;
    let mut faces: Vec<Vec<[f32; 4]>> = Vec::with_capacity(6);
    for f in 0..6 {
        let face = match &cube.faces_f32 {
            Some(hdr) => hdr[f]
                .as_chunks::<4>()
                .0
                .iter()
                .take(n)
                .map(|c| [c[0], c[1], c[2], c[3]])
                .collect(),
            None => texels(cube.format, &cube.faces[f], cube.size, cube.size)?,
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
    let t = renderer.new_target(w, size, INTERMEDIATE_FORMAT);
    renderer.upload_render_target_f16(t.id(), &atlas);
    *slot = Some((key, t.clone()));
    Some(t)
}
