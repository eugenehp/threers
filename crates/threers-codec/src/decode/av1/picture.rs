//! Decoded AV1 pictures: three `u16` planes (all bit depths), allocated to
//! a 128-aligned size like dav1d's default allocator, so block writes past
//! the visible edge stay in bounds.

use super::obu::Layout;

#[derive(Clone)]
pub struct Picture {
    pub planes: [Vec<u16>; 3],
    /// Luma, chroma stride in pixels.
    pub stride: [usize; 2],
    pub w: i32,
    pub h: i32,
    pub layout: Layout,
    pub bpc: u32,
}

impl Picture {
    pub fn new(w: i32, h: i32, layout: Layout, bpc: u32) -> Self {
        let aw = ((w + 127) & !127) as usize;
        let ah = ((h + 127) & !127) as usize;
        let ss_hor = (layout == Layout::I420 || layout == Layout::I422) as usize;
        let ss_ver = (layout == Layout::I420) as usize;
        let ys = aw;
        let uvs = if layout == Layout::I400 { 0 } else { aw >> ss_hor };
        Picture {
            planes: [vec![0; ys * ah], vec![0; uvs * (ah >> ss_ver)], vec![0; uvs * (ah >> ss_ver)]],
            stride: [ys, uvs],
            w,
            h,
            layout,
            bpc,
        }
    }

    #[inline]
    pub fn ss(&self, pl: usize) -> (i32, i32) {
        if pl == 0 {
            (0, 0)
        } else {
            ((self.layout == Layout::I420 || self.layout == Layout::I422) as i32, (self.layout == Layout::I420) as i32)
        }
    }

    #[inline]
    pub fn pstride(&self, pl: usize) -> usize {
        self.stride[(pl != 0) as usize]
    }
}
