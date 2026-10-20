//! The browser's constructive solid geometry: one boolean evaluation by the
//! Rust port of three-bvh-csg (`crate::csg`), for `web/bvh-csg-impl.js`,
//! which keeps three-bvh-csg's scene-graph classes (`Brush`, `Operation`,
//! `OperationGroup`) and its `Evaluator`'s group and material bookkeeping in
//! JavaScript, as the library has them.

use js_sys::{Array, Float32Array, Float64Array, Object, Reflect, Uint32Array};
use wasm_bindgen::prelude::*;

use crate::core::{BufferAttribute, BufferGeometry, GeometryGroup};
use crate::csg::{CsgBrush, CsgEvaluator};

fn get(obj: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(obj, &JsValue::from_str(key))
}

/// A brush from its plain description: `{ attributes: { name: { array,
/// itemSize } }, index: Uint32Array | null, groups: [{ start, count }],
/// matrixWorld: Float64Array(16) }`.
fn brush_from(desc: &JsValue, what: &str) -> Result<CsgBrush, JsValue> {
    let err = |m: &str| JsValue::from_str(&format!("csgEvaluate: {what}: {m}"));
    let mut geometry = BufferGeometry::new();
    let attributes = get(desc, "attributes")?;
    if !attributes.is_object() {
        return Err(err("no attributes"));
    }
    for key in Object::keys(attributes.unchecked_ref::<Object>()).iter() {
        let name = key.as_string().unwrap_or_default();
        let attr = get(&attributes, &name)?;
        let array = get(&attr, "array")?;
        let item_size = get(&attr, "itemSize")?.as_f64().unwrap_or(0.0) as usize;
        if item_size == 0 {
            return Err(err(&format!("'{name}' has no itemSize")));
        }
        let data = Float32Array::new(&array).to_vec();
        geometry.set_attribute(&name, BufferAttribute::new(data, item_size));
    }
    if geometry.get_attribute("position").is_none() {
        return Err(err("no 'position' attribute"));
    }
    let index = get(desc, "index")?;
    if !index.is_null() && !index.is_undefined() {
        geometry.set_index(Uint32Array::new(&index).to_vec());
    }
    let groups = get(desc, "groups")?;
    if Array::is_array(&groups) {
        for g in Array::from(&groups).iter() {
            let n = |k: &str| get(&g, k).ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
            let count = n("count");
            geometry.groups.push(GeometryGroup {
                start: n("start") as u32,
                count: if count.is_finite() {
                    count as u32
                } else {
                    u32::MAX
                },
                material_index: n("materialIndex") as u32,
            });
        }
    }
    let m = Float64Array::new(&get(desc, "matrixWorld")?).to_vec();
    if m.len() != 16 {
        return Err(err("matrixWorld is not 16 numbers"));
    }
    let mut brush = CsgBrush::new(geometry);
    brush.set_matrix_world(std::array::from_fn(|i| m[i]));
    Ok(brush)
}

/// three-bvh-csg's boolean step: `a` against `b` under each of `operations`
/// (its `ADDITION` … `HOLLOW_INTERSECTION` constants), with `useGroups` and
/// the `attributes` to carry (each must exist on `a`). Returns, per
/// operation, the groups in order — group `g` from `a`'s group `g`, then
/// `b`'s from `a.groups.length || 1` on — each `{ name: Float32Array }` of
/// world-space triangles (non-indexed, three vertices each).
#[wasm_bindgen(js_name = csgEvaluate)]
pub fn csg_evaluate(
    a: JsValue,
    b: JsValue,
    operations: Vec<u8>,
    use_groups: bool,
    attributes: Vec<String>,
) -> Result<Array, JsValue> {
    if let Some(&op) = operations
        .iter()
        .find(|&&op| op > crate::csg::HOLLOW_INTERSECTION)
    {
        return Err(JsValue::from_str(&format!(
            "csgEvaluate: unknown operation {op}"
        )));
    }
    let mut brush_a = brush_from(&a, "a")?;
    let mut brush_b = brush_from(&b, "b")?;
    let mut evaluator = CsgEvaluator::new();
    evaluator.use_groups = use_groups;
    let names: Vec<&str> = attributes.iter().map(String::as_str).collect();
    let results = evaluator
        .evaluate_groups(&mut brush_a, &mut brush_b, &operations, &names)
        .map_err(|e| JsValue::from_str(&e))?;
    let out = Array::new();
    for groups in results {
        let js_groups = Array::new();
        for group in groups {
            let obj = Object::new();
            for (name, _size, data) in group {
                Reflect::set(
                    &obj,
                    &JsValue::from_str(&name),
                    &Float32Array::from(data.as_slice()),
                )?;
            }
            js_groups.push(&obj);
        }
        out.push(&js_groups);
    }
    Ok(out)
}
