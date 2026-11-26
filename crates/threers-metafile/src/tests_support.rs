//! Small metafiles made by hand, for the tests and the examples.

fn rec(out: &mut Vec<u8>, kind: u32, params: &[u8]) {
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&((8 + params.len()) as u32).to_le_bytes());
    out.extend_from_slice(params);
}

fn le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// An inch square at 96 pixels an inch: a red rectangle, a blue line, "Hi".
pub fn sample_emf() -> Vec<u8> {
    let mut out = Vec::new();
    // the header: bounds, a frame of 25.4 mm, " EMF", the device 96 dpi
    let mut h = vec![0u8; 80];
    h[0..16].copy_from_slice(&le(&[0, 0, 95, 95]));
    h[16..32].copy_from_slice(&le(&[0, 0, 2540, 2540]));
    h[32..36].copy_from_slice(&0x464D_4520u32.to_le_bytes());
    h[64..72].copy_from_slice(&le(&[960, 960]));
    h[72..80].copy_from_slice(&le(&[254, 254]));
    rec(&mut out, 1, &h);
    // a red brush (1), a blue pen (2)
    rec(&mut out, 39, &le(&[1, 0, 0x0000FF, 0]));
    rec(&mut out, 38, &le(&[2, 0, 2, 0, 0xFF0000]));
    rec(&mut out, 37, &le(&[1]));
    rec(&mut out, 37, &0x8000_0008u32.to_le_bytes());
    rec(&mut out, 43, &le(&[10, 10, 50, 40]));
    rec(&mut out, 37, &le(&[2]));
    rec(&mut out, 27, &le(&[0, 90]));
    rec(&mut out, 54, &le(&[90, 90]));
    // "Hi" at (20, 70)
    let mut t = vec![0u8; 76];
    t[28..32].copy_from_slice(&20i32.to_le_bytes());
    t[32..36].copy_from_slice(&70i32.to_le_bytes());
    t[36..40].copy_from_slice(&2u32.to_le_bytes());
    t[40..44].copy_from_slice(&84u32.to_le_bytes());
    t.extend_from_slice(&[b'H', 0, b'i', 0]);
    rec(&mut out, 84, &t);
    rec(&mut out, 14, &le(&[0, 0, 0]));
    out
}

/// A placeable WMF an inch by half an inch (1440 units an inch): a filled
/// rectangle and a line.
pub fn sample_wmf() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0x9AC6_CDD7u32.to_le_bytes());
    out.extend_from_slice(&[0, 0]);
    for v in [0i16, 0, 1440, 720] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&1440u16.to_le_bytes());
    out.extend_from_slice(&[0; 6]);
    // the header: memory, 9 words, version 3
    for v in [1u16, 9, 0x300, 0, 0, 2, 0, 0, 0] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut rec = |f: u16, words: &[i16], extra: &[u8]| {
        let size = (6 + 2 * words.len() + extra.len()) / 2;
        out.extend_from_slice(&(size as u32).to_le_bytes());
        out.extend_from_slice(&f.to_le_bytes());
        out.extend_from_slice(&extra[..]);
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
    };
    // a green brush, selected
    rec(0x02FC, &[], &[0, 0, 0, 0xFF, 0, 0, 0, 0]);
    rec(0x012D, &[0], &[]);
    rec(0x041B, &[600, 1000, 100, 100], &[]);
    rec(0x0214, &[700, 0], &[]);
    rec(0x0213, &[700, 1440], &[]);
    rec(0, &[], &[]);
    out
}
