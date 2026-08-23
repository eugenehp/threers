use std::collections::HashMap;

use crate::core::BufferGeometry;

use super::geometry_prep::{index_at, tri_count};
use super::triangle_utils::hash_vertex3;

#[derive(Debug)]
pub struct HalfEdgeMap {
    pub data: Vec<i32>,
}

impl HalfEdgeMap {
    pub fn from_geometry(geometry: &BufferGeometry) -> Self {
        let count = tri_count(geometry);
        let pos = geometry.get_attribute("position").expect("position");
        let mut data = vec![-1i32; count * 3];
        let mut map: HashMap<(i32, i32, i32, i32, i32, i32), usize> = HashMap::new();

        for tri in 0..count {
            let mut hashes = [(0i32, 0i32, 0i32); 3];
            // `e` is the edge number, passed to `index_at` as well as indexing.
            #[allow(clippy::needless_range_loop)]
            for e in 0..3 {
                let vi = index_at(geometry, tri, e);
                let v = super::geometry_prep::read_position(pos, vi);
                hashes[e] = hash_vertex3(v);
            }
            for e in 0..3 {
                let next_e = (e + 1) % 3;
                let reverse = (
                    hashes[next_e].0,
                    hashes[next_e].1,
                    hashes[next_e].2,
                    hashes[e].0,
                    hashes[e].1,
                    hashes[e].2,
                );
                let index = tri * 3 + e;
                if let Some(other) = map.remove(&reverse) {
                    data[index] = other as i32;
                    data[other] = index as i32;
                } else {
                    let forward = (
                        hashes[e].0,
                        hashes[e].1,
                        hashes[e].2,
                        hashes[next_e].0,
                        hashes[next_e].1,
                        hashes[next_e].2,
                    );
                    map.insert(forward, index);
                }
            }
        }
        Self { data }
    }

    pub fn sibling_triangle_index(&self, tri_index: usize, edge_index: usize) -> i32 {
        let other = self.data[tri_index * 3 + edge_index];
        if other == -1 {
            -1
        } else {
            other / 3
        }
    }
}
