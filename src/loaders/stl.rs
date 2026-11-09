use crate::core::{BufferAttribute, BufferGeometry};

/// STL parser (ASCII + binary). Output is a non-indexed flat-shaded
/// triangle list (positions + normals).
pub struct StlLoader;

impl StlLoader {
    pub fn parse_ascii(src: &str) -> BufferGeometry {
        let mut positions: Vec<f32> = Vec::new();
        let mut normals: Vec<f32> = Vec::new();
        let mut current_normal = [0.0f32; 3];
        let mut tri_buf: Vec<f32> = Vec::with_capacity(9);
        for line in src.lines() {
            let l = line.trim();
            if let Some(rest) = l.strip_prefix("facet normal ") {
                let nums: Vec<f32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() == 3 {
                    current_normal = [nums[0], nums[1], nums[2]];
                }
                tri_buf.clear();
            } else if let Some(rest) = l.strip_prefix("vertex ") {
                let nums: Vec<f32> = rest
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if nums.len() == 3 {
                    tri_buf.extend_from_slice(&nums);
                }
            } else if l == "endfacet" {
                if tri_buf.len() == 9 {
                    positions.extend_from_slice(&tri_buf);
                    for _ in 0..3 {
                        normals.extend_from_slice(&current_normal);
                    }
                }
                tri_buf.clear();
            }
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g
    }

    pub fn parse_binary(bytes: &[u8]) -> BufferGeometry {
        // Layout: 80-byte header, u32 triangle count, then 50 bytes per triangle.
        if bytes.len() < 84 {
            return BufferGeometry::new();
        }
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        let mut positions = Vec::with_capacity(count * 9);
        let mut normals = Vec::with_capacity(count * 9);
        let mut p = 84;
        for _ in 0..count {
            if p + 50 > bytes.len() {
                break;
            }
            let read_f = |o: usize| {
                f32::from_le_bytes([
                    bytes[p + o],
                    bytes[p + o + 1],
                    bytes[p + o + 2],
                    bytes[p + o + 3],
                ])
            };
            let nx = read_f(0);
            let ny = read_f(4);
            let nz = read_f(8);
            for v in 0..3 {
                let off = 12 + v * 12;
                positions.extend_from_slice(&[read_f(off), read_f(off + 4), read_f(off + 8)]);
                normals.extend_from_slice(&[nx, ny, nz]);
            }
            p += 50;
        }
        let mut g = BufferGeometry::new();
        g.set_attribute("position", BufferAttribute::new(positions, 3));
        g.set_attribute("normal", BufferAttribute::new(normals, 3));
        g
    }

    /// Auto-detect ASCII vs binary based on the header.
    pub fn parse(bytes: &[u8]) -> BufferGeometry {
        if bytes.len() >= 5
            && &bytes[0..5] == b"solid"
            && let Ok(s) = std::str::from_utf8(bytes)
        {
            return Self::parse_ascii(s);
        }
        Self::parse_binary(bytes)
    }
}
