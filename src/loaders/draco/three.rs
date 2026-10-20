//! A decoded Draco mesh as three.js r165's `DRACOLoader` hands it back.
//!
//! `DRACOLoader` asks Google's decoder for the first attribute of each
//! semantic type — `POSITION`, `NORMAL`, `COLOR`, `TEX_COORD`, named
//! `position`, `normal`, `color` and `uv` — as `Float32Array`s, which turns a
//! normalised integer attribute into `[0, 1]` (or `[-1, 1]`) floats and leaves
//! other integers as their values; adds the face index; and, when loading a
//! file (`load` / `parse`), decodes sRGB vertex colours to linear.

use super::attributes::{AttributeValues, DecodedAttribute};
use super::{decode_mesh, DracoError};
use crate::core::{BufferAttribute, BufferGeometry};

/// Draco `GeometryAttribute::Type` → three.js attribute name, in
/// `DRACOLoader.defaultAttributeIDs` order.
const NAMES: [(u8, &str); 4] = [(0, "position"), (1, "normal"), (2, "color"), (3, "uv")];

/// The largest value of a Draco `DataType`, for normalised integers.
fn max_value(data_type: u8) -> Option<f32> {
    Some(match data_type {
        1 => i8::MAX as f32,
        2 => u8::MAX as f32,
        3 => i16::MAX as f32,
        4 => u16::MAX as f32,
        5 => i32::MAX as f32,
        6 => u32::MAX as f32,
        _ => return None,
    })
}

/// Draco's `ConvertValue<float>`: integers divided by their type's maximum
/// when the attribute is normalised, taken as they are otherwise.
fn as_float(a: &DecodedAttribute) -> Vec<f32> {
    match (&a.values, a.descriptor.normalized, max_value(a.descriptor.data_type)) {
        (AttributeValues::Integer(v), true, Some(max)) => v.iter().map(|&x| x as f32 / max).collect(),
        _ => a.as_f32(),
    }
}

/// three.js `Color.convertSRGBToLinear` (`SRGBToLinear`) per channel.
fn srgb_to_linear(c: f32) -> f32 {
    let c = c as f64;
    (if c < 0.04045 {
        c * 0.0773993808
    } else {
        (c * 0.9478672986 + 0.0521327014).powf(2.4)
    }) as f32
}

/// Decode `bytes` as `DRACOLoader` does. `srgb_colors` is its
/// `vertexColorSpace === SRGBColorSpace` (true for `load`/`parse`): the
/// first three channels of `color` are then converted to linear.
pub fn decode_three(bytes: &[u8], srgb_colors: bool) -> Result<BufferGeometry, DracoError> {
    let mesh = decode_mesh(bytes)?;
    let mut g = BufferGeometry::new();
    for (att_type, name) in NAMES {
        let Some(a) = mesh.attribute(att_type) else { continue };
        let size = a.num_components().max(1);
        let mut values = as_float(a);
        if name == "color" && srgb_colors && size >= 3 {
            for px in values.chunks_exact_mut(size) {
                for c in &mut px[..3] {
                    *c = srgb_to_linear(*c);
                }
            }
        }
        g.set_attribute(name, BufferAttribute::new(values, size));
    }
    g.set_index(mesh.indices());
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bunny_has_positions_and_a_full_index() {
        let g = decode_three(include_bytes!("testdata/bunny.drc"), true).unwrap();
        let pos = g.get_attribute("position").unwrap();
        assert_eq!(pos.item_size, 3);
        let index = g.index.as_ref().unwrap();
        assert_eq!(index.len() % 3, 0);
        assert!(index.iter().all(|&i| (i as usize) < pos.count()));
    }

    #[test]
    fn srgb_conversion_matches_three() {
        // three.js: SRGBToLinear(0.5) = 0.21404114048223255.
        assert!((srgb_to_linear(0.5) as f64 - 0.21404114048223255).abs() < 1e-7);
        assert_eq!(srgb_to_linear(0.0), 0.0);
    }
}
