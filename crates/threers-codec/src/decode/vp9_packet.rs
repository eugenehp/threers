//! VP9 packet-level parsing that demuxers need without the decoder (the
//! decoder re-exports these as `vp9::{packet_shows, split_superframe}`).

use crate::decode::bits::BitReader;

/// Whether a packet (frame or superframe) shows a picture: some frame in it
/// has `show_existing_frame` or `show_frame` set.
pub fn packet_shows(data: &[u8]) -> bool {
    split_superframe(data).iter().any(|f| {
        let mut r = BitReader::new(f);
        if f.is_empty() || r.u(2) != 2 {
            return false;
        }
        let profile = r.u(1) | (r.u(1) << 1);
        if profile == 3 {
            r.skip(1);
        }
        if r.flag() {
            return true; // show_existing_frame
        }
        r.skip(1); // frame_type
        r.flag() // show_frame
    })
}

/// Split a VP9 superframe (ffmpeg `vp9_superframe_split`) into its frames.
pub fn split_superframe(data: &[u8]) -> Vec<&[u8]> {
    if let Some(&marker) = data.last() {
        if marker & 0xE0 == 0xC0 {
            let nbytes = 1 + ((marker >> 3) & 3) as usize;
            let n = 1 + (marker & 7) as usize;
            let idx_size = 2 + nbytes * n;
            if data.len() >= idx_size && data[data.len() - idx_size] == marker {
                let mut sizes = Vec::with_capacity(n);
                let mut p = data.len() - idx_size + 1;
                for _ in 0..n {
                    let mut s = 0usize;
                    for b in 0..nbytes {
                        s |= (data[p + b] as usize) << (8 * b);
                    }
                    p += nbytes;
                    sizes.push(s);
                }
                let total: usize = sizes.iter().sum();
                if total <= data.len() - idx_size {
                    let mut out = Vec::new();
                    let mut off = 0;
                    for s in sizes {
                        out.push(&data[off..off + s]);
                        off += s;
                    }
                    return out;
                }
            }
        }
    }
    vec![data]
}
