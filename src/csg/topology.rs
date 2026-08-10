//! Ordered triangle keys for comparing CSG soups to JS reference bins.
//!
//! A [`TriKey`] is nine `hashVertex3` ints (three corners × xyz), preserving winding.

use std::collections::HashSet;

use crate::core::BufferGeometry;

use super::triangle_utils::hash_vertex3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TriKey(pub (i32, i32, i32, i32, i32, i32, i32, i32, i32));

pub fn tri_key_from_soup(pos: &[f32], tri: usize) -> TriKey {
    let i = tri * 9;
    let a = crate::math::Vector3::new(pos[i], pos[i + 1], pos[i + 2]);
    let b = crate::math::Vector3::new(pos[i + 3], pos[i + 4], pos[i + 5]);
    let c = crate::math::Vector3::new(pos[i + 6], pos[i + 7], pos[i + 8]);
    let ha = hash_vertex3(a);
    let hb = hash_vertex3(b);
    let hc = hash_vertex3(c);
    TriKey((ha.0, ha.1, ha.2, hb.0, hb.1, hb.2, hc.0, hc.1, hc.2))
}

pub fn triangle_keys(geom: &BufferGeometry) -> HashSet<TriKey> {
    let pos = geom.get_attribute("position").expect("position");
    let mut keys = HashSet::new();
    for t in 0..pos.count() / 3 {
        keys.insert(tri_key_from_soup(&pos.array, t));
    }
    keys
}

#[derive(Debug)]
pub struct TopologyOverlap {
    pub native_tris: usize,
    pub reference_tris: usize,
    pub shared: usize,
    pub only_native: usize,
    pub only_reference: usize,
}

pub fn topology_overlap(native: &BufferGeometry, reference: &BufferGeometry) -> TopologyOverlap {
    let nk = triangle_keys(native);
    let rk = triangle_keys(reference);
    let shared = nk.intersection(&rk).count();
    TopologyOverlap {
        native_tris: nk.len(),
        reference_tris: rk.len(),
        shared,
        only_native: nk.difference(&rk).count(),
        only_reference: rk.difference(&nk).count(),
    }
}

#[cfg(test)]
pub fn assert_topology_partial_overlap(
    native: &HashSet<TriKey>,
    reference: &HashSet<TriKey>,
    min_partial_ratio: f64,
    label: &str,
) {
    let o = tri_key_overlap(native, reference);
    let ratio = o.partial_shared as f64 / native.len().max(1) as f64;
    assert!(
        ratio >= min_partial_ratio,
        "{label}: partial (2-vert) {}/{} native tris ({:.1}% < {:.1}% min)",
        o.partial_shared,
        native.len(),
        ratio * 100.0,
        min_partial_ratio * 100.0
    );
}

pub fn assert_topology_overlap(
    native: &BufferGeometry,
    reference: &BufferGeometry,
    min_shared_ratio: f64,
    label: &str,
) {
    let o = topology_overlap(native, reference);
    if min_shared_ratio >= 1.0 {
        assert_eq!(
            o.only_native, 0,
            "{label}: only_native={} (shared {}/{})",
            o.only_native, o.shared, o.reference_tris
        );
        assert_eq!(
            o.only_reference, 0,
            "{label}: only_reference={} (shared {}/{})",
            o.only_reference, o.shared, o.reference_tris
        );
        return;
    }
    let ratio = o.shared as f64 / o.reference_tris.max(1) as f64;
    assert!(
        ratio >= min_shared_ratio,
        "{label}: shared {}/{} ref tris ({:.1}% < {:.1}% min); only_native={} only_ref={}",
        o.shared,
        o.reference_tris,
        ratio * 100.0,
        min_shared_ratio * 100.0,
        o.only_native,
        o.only_reference
    );
}

#[cfg(test)]
fn tri_key_verts(k: TriKey) -> [[i32; 3]; 3] {
    let t = k.0;
    [[t.0, t.1, t.2], [t.3, t.4, t.5], [t.6, t.7, t.8]]
}

#[cfg(test)]
fn shared_vert_count(a: TriKey, b: TriKey) -> usize {
    let av = tri_key_verts(a);
    let bv = tri_key_verts(b);
    av.iter().filter(|v| bv.contains(v)).count()
}

#[derive(Debug)]
#[cfg(test)]
pub struct TriKeyOverlap {
    pub exact_shared: usize,
    /// Native keys with >=2 vertex hashes matching some reference key.
    pub partial_shared: usize,
}

#[cfg(test)]
pub fn tri_key_overlap(native: &HashSet<TriKey>, reference: &HashSet<TriKey>) -> TriKeyOverlap {
    let exact_shared = native.intersection(reference).count();
    let mut partial_shared = exact_shared;
    for nk in native.difference(reference) {
        let best = reference
            .iter()
            .map(|rk| shared_vert_count(*nk, *rk))
            .max()
            .unwrap_or(0);
        if best >= 2 {
            partial_shared += 1;
        }
    }
    TriKeyOverlap {
        exact_shared,
        partial_shared,
    }
}
