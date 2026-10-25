//! Images packed into one texture, for sprites and image quads.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Where an image sits in a [`TextureAtlas`], in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasRect {
    /// `[x, y, width, height]` as floats — what a rect column holds.
    pub fn to_array(self) -> [f32; 4] {
        [
            self.x as f32,
            self.y as f32,
            self.width as f32,
            self.height as f32,
        ]
    }
}

/// Why an image did not go in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AtlasError {
    /// Wider than the atlas, with its gutters.
    TooWide { width: u32, atlas: u32 },
    /// The atlas would have to grow past its maximum height.
    Full { max_height: u32 },
    /// The pixel slice is not `width * height * 4` bytes.
    WrongLength { expected: usize, got: usize },
    /// No image of that name to [`write`](TextureAtlas::write) over.
    Missing(String),
}

impl std::fmt::Display for AtlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtlasError::TooWide { width, atlas } => {
                write!(f, "a {width} px image does not fit a {atlas} px atlas")
            }
            AtlasError::Full { max_height } => write!(f, "the atlas is full at {max_height} px"),
            AtlasError::WrongLength { expected, got } => {
                write!(f, "{got} bytes of RGBA where {expected} were expected")
            }
            AtlasError::Missing(name) => write!(f, "no image named {name:?} in the atlas"),
        }
    }
}

impl std::error::Error for AtlasError {}

/// An atlas several layers draw from: one texture for every tile's icons.
pub type SharedAtlas = Arc<RwLock<TextureAtlas>>;

struct AtlasGpu {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    levels: u32,
    version: u64,
}

/// RGBA images packed onto shelves of one growable texture.
///
/// Every image keeps a gutter of transparent pixels around it, wide enough
/// that linear filtering — and every mip level the atlas keeps — never reads
/// a neighbour. The height doubles as images arrive, up to
/// [`max_height`](Self::max_height); rects are in pixels, so they stay valid
/// as it grows, and the marks divide by the current size on the GPU.
///
/// Images are packed in arrival order and never moved; [`clear`](Self::clear)
/// starts over. A tile cache that evicts should keep one atlas per
/// generation, or rebuild when it fills.
///
/// ```
/// use threers::marks::TextureAtlas;
/// let mut atlas = TextureAtlas::new(256);
/// let pin = atlas.insert("pin", 16, 24, &vec![255; 16 * 24 * 4]).unwrap();
/// assert_eq!(atlas.get("pin"), Some(pin));
/// assert_eq!((pin.width, pin.height), (16, 24));
/// ```
pub struct TextureAtlas {
    width: u32,
    height: u32,
    /// The tallest the atlas may grow.
    pub max_height: u32,
    gutter: u32,
    levels: u32,
    srgb: bool,
    pixels: Vec<u8>,
    shelf_x: u32,
    shelf_y: u32,
    shelf_h: u32,
    entries: HashMap<String, AtlasRect>,
    version: u64,
    /// Regions changed since the last upload, `(x0, y0, x1, y1)`, each
    /// uploaded on its own: two images at opposite corners send two images,
    /// not the whole atlas between them.
    dirty: Vec<(u32, u32, u32, u32)>,
    gpu: Option<AtlasGpu>,
}

impl std::fmt::Debug for TextureAtlas {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureAtlas")
            .field("size", &(self.width, self.height))
            .field("images", &self.entries.len())
            .finish()
    }
}

impl TextureAtlas {
    /// An atlas `width` pixels wide, sRGB, with a 2 px gutter and no mip
    /// levels beyond the base — the setting for images drawn near their own
    /// size. See [`with_mip_levels`](Self::with_mip_levels) for icons drawn
    /// much smaller.
    pub fn new(width: u32) -> Self {
        Self {
            width: width.max(1),
            height: 64,
            max_height: 8192,
            gutter: 2,
            levels: 1,
            srgb: true,
            pixels: vec![0; width.max(1) as usize * 64 * 4],
            shelf_x: 0,
            shelf_y: 0,
            shelf_h: 0,
            entries: HashMap::new(),
            version: 1,
            dirty: Vec::new(),
            gpu: None,
        }
    }

    /// Keep `levels` mip levels (at least 1), widening the gutter so the
    /// smallest still separates images. Call before inserting.
    pub fn with_mip_levels(mut self, levels: u32) -> Self {
        assert!(self.entries.is_empty(), "set mip levels before inserting");
        self.levels = levels.clamp(1, 8);
        self.gutter = 1 << (self.levels - 1).max(1);
        self
    }

    /// Hold linear data (masks, distance fields, data images) rather than
    /// sRGB colour.
    pub fn linear(mut self) -> Self {
        self.srgb = false;
        self
    }

    /// Whether the texture decodes sRGB (else its bytes are read as they are).
    pub fn is_srgb(&self) -> bool {
        self.srgb
    }

    /// Wrap for sharing between layers.
    pub fn shared(self) -> SharedAtlas {
        Arc::new(RwLock::new(self))
    }

    /// Pack an image of `rgba` bytes (straight alpha, row-major, top row
    /// first) under `name`, replacing the name's earlier image (whose pixels
    /// stay where they were, unreferenced).
    pub fn insert(
        &mut self,
        name: impl Into<String>,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<AtlasRect, AtlasError> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(AtlasError::WrongLength {
                expected,
                got: rgba.len(),
            });
        }
        let g = self.gutter;
        if width + 2 * g > self.width {
            return Err(AtlasError::TooWide {
                width,
                atlas: self.width,
            });
        }
        if self.shelf_x + width + 2 * g > self.width {
            self.shelf_y += self.shelf_h;
            self.shelf_x = 0;
            self.shelf_h = 0;
        }
        let need = self.shelf_y + height + 2 * g;
        if need > self.max_height {
            return Err(AtlasError::Full {
                max_height: self.max_height,
            });
        }
        while need > self.height {
            self.height = (self.height * 2).min(self.max_height);
            self.pixels
                .resize(self.width as usize * self.height as usize * 4, 0);
        }
        let (x, y) = (self.shelf_x + g, self.shelf_y + g);
        self.blit(x, y, width, height, rgba);
        self.shelf_x += width + 2 * g;
        self.shelf_h = self.shelf_h.max(height + 2 * g);
        let rect = AtlasRect {
            x,
            y,
            width,
            height,
        };
        self.entries.insert(name.into(), rect);
        self.version += 1;
        Ok(rect)
    }

    /// Put an image at `(x, y)` exactly as given — no gutter, its edges not
    /// repeated — for an atlas laid out by someone else's rules: a font
    /// atlas packed as deck.gl packs it, whose mip levels then match. The
    /// atlas grows (in powers of two) to hold it; later
    /// [`insert`](Self::insert)s pack below it.
    pub fn insert_at(
        &mut self,
        name: impl Into<String>,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> Result<AtlasRect, AtlasError> {
        let expected = width as usize * height as usize * 4;
        if rgba.len() != expected {
            return Err(AtlasError::WrongLength { expected, got: rgba.len() });
        }
        if x + width > self.width {
            return Err(AtlasError::TooWide { width: x + width, atlas: self.width });
        }
        let need = y + height;
        if need > self.max_height {
            return Err(AtlasError::Full { max_height: self.max_height });
        }
        while need > self.height {
            self.height = (self.height * 2).min(self.max_height);
            self.pixels.resize(self.width as usize * self.height as usize * 4, 0);
        }
        let aw = self.width as usize;
        for row in 0..height as usize {
            let src = row * width as usize * 4;
            let dst = ((y as usize + row) * aw + x as usize) * 4;
            self.pixels[dst..dst + width as usize * 4].copy_from_slice(&rgba[src..src + width as usize * 4]);
        }
        if width > 0 && height > 0 {
            self.mark_dirty((x, y, x + width, y + height));
        }
        // Shelves continue below everything placed.
        if need > self.shelf_y + self.shelf_h {
            self.shelf_y = need;
            self.shelf_x = 0;
            self.shelf_h = 0;
        }
        let rect = AtlasRect { x, y, width, height };
        self.entries.insert(name.into(), rect);
        self.version += 1;
        Ok(rect)
    }

    /// At least `height` pixels tall (rounded up to a power of two): a
    /// layout made elsewhere sized for its own rows.
    pub fn reserve_height(&mut self, height: u32) {
        while height > self.height && self.height < self.max_height {
            self.height = (self.height * 2).min(self.max_height);
            self.pixels.resize(self.width as usize * self.height as usize * 4, 0);
            self.version += 1;
        }
    }

    /// Replace the pixels of the image packed as `name`, which keeps its
    /// place and size: a map tile cache reusing its slots, an icon redrawn.
    /// Only that image goes to the GPU at the next upload, and layers keep
    /// their bind groups (the atlas's [`version`](Self::version) does not move).
    pub fn write(&mut self, name: &str, rgba: &[u8]) -> Result<AtlasRect, AtlasError> {
        let rect = self
            .entries
            .get(name)
            .copied()
            .ok_or_else(|| AtlasError::Missing(name.to_owned()))?;
        let expected = rect.width as usize * rect.height as usize * 4;
        if rgba.len() != expected {
            return Err(AtlasError::WrongLength {
                expected,
                got: rgba.len(),
            });
        }
        self.blit(rect.x, rect.y, rect.width, rect.height, rgba);
        Ok(rect)
    }

    /// Copy a `w × h` image to `(x, y)` with its edge texels repeated into
    /// the gutter around it, and mark the lot for upload.
    ///
    /// The repeated edge is what keeps filtering at the image's border from
    /// blending in the gutter's transparent black — the dark seam between
    /// images laid edge to edge, as map tiles and opaque sprites are.
    fn blit(&mut self, x: u32, y: u32, w: u32, h: u32, rgba: &[u8]) {
        if w == 0 || h == 0 {
            return;
        }
        let g = self.gutter;
        let aw = self.width as usize;
        for row in 0..(h + 2 * g) as usize {
            let sy = row.saturating_sub(g as usize).min(h as usize - 1);
            let dst_row = (y as usize - g as usize + row) * aw;
            for col in 0..(w + 2 * g) as usize {
                let sx = col.saturating_sub(g as usize).min(w as usize - 1);
                let src = (sy * w as usize + sx) * 4;
                let dst = (dst_row + x as usize - g as usize + col) * 4;
                self.pixels[dst..dst + 4].copy_from_slice(&rgba[src..src + 4]);
            }
        }
        self.mark_dirty((x - g, y - g, x + w + g, y + h + g));
    }

    fn mark_dirty(&mut self, d: (u32, u32, u32, u32)) {
        self.dirty.push(d);
        // Past a few dozen regions, one upload of their bounds is cheaper
        // than the calls.
        if self.dirty.len() > 64 {
            let u = self.dirty.iter().fold(self.dirty[0], |a, b| {
                (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
            });
            self.dirty = vec![u];
        }
    }

    pub fn get(&self, name: &str) -> Option<AtlasRect> {
        self.entries.get(name).copied()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current size in pixels.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// The packed pixels, RGBA, row-major.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Changes on every insert and clear: layers compare it to know their
    /// bind group is stale.
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Forget every image and start packing from the top again.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.pixels.iter_mut().for_each(|p| *p = 0);
        self.shelf_x = 0;
        self.shelf_y = 0;
        self.shelf_h = 0;
        self.version += 1;
        self.dirty = vec![(0, 0, self.width, self.height)];
    }

    fn format(&self) -> wgpu::TextureFormat {
        if self.srgb {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        }
    }

    /// The texture, created or brought up to date: a new texture when the
    /// atlas grew, otherwise only the changed pixels (every level, with
    /// mips). Returns the view and the generation it belongs to, which
    /// changes whenever a bind group holding the old view must be rebuilt.
    pub fn texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> (wgpu::TextureView, u64) {
        let size = (self.width, self.height);
        let max = device.limits().max_texture_dimension_2d;
        let fresh = self
            .gpu
            .as_ref()
            .is_none_or(|g| g.size != size || g.levels != self.levels);
        if fresh {
            let levels = self
                .levels
                .min(32 - size.0.min(size.1).max(1).leading_zeros());
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("threers texture atlas"),
                size: wgpu::Extent3d {
                    width: size.0.min(max),
                    height: size.1.min(max),
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            self.gpu = Some(AtlasGpu {
                texture,
                view,
                size,
                levels: self.levels,
                version: self.version,
            });
            self.dirty = vec![(0, 0, size.0, size.1)];
        }
        let dirty = std::mem::take(&mut self.dirty);
        if self
            .gpu
            .as_ref()
            .is_some_and(|g| g.texture.mip_level_count() > 1)
        {
            // A mip chain is rebuilt from the whole level, so once for all
            // of them.
            if let Some(&first) = dirty.first() {
                let u = dirty.iter().fold(first, |a, b| {
                    (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3))
                });
                self.upload(queue, u);
            }
        } else {
            for d in dirty {
                self.upload(queue, d);
            }
        }
        let g = self.gpu.as_mut().unwrap();
        if fresh {
            g.version = self.version;
        }
        (g.view.clone(), g.version)
    }

    fn upload(&self, queue: &wgpu::Queue, (x0, y0, x1, y1): (u32, u32, u32, u32)) {
        let g = self.gpu.as_ref().unwrap();
        let levels = g.texture.mip_level_count();
        // Level 0 straight from the packed pixels; each further level a box
        // filter of the one above, premultiplied so transparent gutters do
        // not darken the edges.
        // Level 0 is read from the packed pixels in place; only a mip chain
        // needs a level of its own to downsample into.
        let mut owned: Option<Vec<u8>> = None;
        let (mut w, mut h) = (self.width, self.height);
        let (mut rx0, mut ry0, mut rx1, mut ry1) = (x0, y0, x1.max(x0 + 1), y1.max(y0 + 1));
        for mip in 0..levels {
            let rw = (rx1 - rx0).min(w - rx0.min(w));
            let rh = (ry1 - ry0).min(h - ry0.min(h));
            if rw > 0 && rh > 0 {
                let mut rows = Vec::with_capacity((rw * rh * 4) as usize);
                let level: &[u8] = owned.as_deref().unwrap_or(&self.pixels);
                for y in ry0..ry0 + rh {
                    let s = ((y * w + rx0) * 4) as usize;
                    rows.extend_from_slice(&level[s..s + (rw * 4) as usize]);
                }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &g.texture,
                        mip_level: mip,
                        origin: wgpu::Origin3d {
                            x: rx0,
                            y: ry0,
                            z: 0,
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &rows,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rw * 4),
                        rows_per_image: Some(rh),
                    },
                    wgpu::Extent3d {
                        width: rw,
                        height: rh,
                        depth_or_array_layers: 1,
                    },
                );
            }
            if mip + 1 == levels {
                break;
            }
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let level: &[u8] = owned.as_deref().unwrap_or(&self.pixels);
            // A linear atlas holds data, or sRGB bytes to be read as a
            // WebGL texture is: its levels plain means of the texels, as
            // `generateMipmap` makes them.
            let straight = !self.srgb;
            let mut next = vec![0u8; (nw * nh * 4) as usize];
            for y in 0..nh {
                for x in 0..nw {
                    let mut acc = [0u32; 4];
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                        let p = &level[((sy * w + sx) * 4) as usize..][..4];
                        let a = if straight { 1 } else { p[3] as u32 };
                        acc[0] += p[0] as u32 * a;
                        acc[1] += p[1] as u32 * a;
                        acc[2] += p[2] as u32 * a;
                        acc[3] += if straight { p[3] as u32 } else { a };
                    }
                    let o = ((y * nw + x) * 4) as usize;
                    for c in 0..3 {
                        next[o + c] = if straight {
                            ((acc[c] + 2) / 4) as u8
                        } else {
                            acc[c].checked_div(acc[3]).unwrap_or(0) as u8
                        };
                    }
                    next[o + 3] = if straight { ((acc[3] + 2) / 4) as u8 } else { (acc[3] / 4) as u8 };
                }
            }
            owned = Some(next);
            (w, h) = (nw, nh);
            (rx0, ry0) = (rx0 / 2, ry0 / 2);
            (rx1, ry1) = (rx1.div_ceil(2), ry1.div_ceil(2));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `write` replaces an image where it lies, gutter and all, and refuses
    /// names it does not hold and wrong sizes.
    #[test]
    fn write_replaces_an_image_in_place() {
        let mut a = TextureAtlas::new(64);
        let r = a.insert("tile", 2, 2, &[10u8; 16]).unwrap();
        let version = a.version();
        let w = a.write("tile", &[200u8; 16]).unwrap();
        assert_eq!((w.x, w.y), (r.x, r.y));
        assert_eq!(a.version(), version, "no new bind group needed");
        let px = |x: u32, y: u32| a.pixels()[((y * 64 + x) * 4) as usize];
        assert_eq!(px(r.x, r.y), 200);
        assert_eq!(px(r.x - 1, r.y - 1), 200, "the gutter follows");
        assert!(matches!(
            a.write("nope", &[0; 16]),
            Err(AtlasError::Missing(_))
        ));
        assert!(matches!(
            a.write("tile", &[0; 4]),
            Err(AtlasError::WrongLength { .. })
        ));
    }

    /// The gutter around an image repeats its edge texels, so a filtered
    /// read at the edge never blends in black.
    #[test]
    fn the_gutter_repeats_the_edge() {
        let mut a = TextureAtlas::new(64);
        let red_green: Vec<u8> = [[255, 0, 0, 255], [0, 255, 0, 255]].repeat(2).concat();
        let r = a.insert("rg", 2, 2, &red_green).unwrap();
        let px = |x: u32, y: u32| {
            let o = ((y * 64 + x) * 4) as usize;
            [
                a.pixels()[o],
                a.pixels()[o + 1],
                a.pixels()[o + 2],
                a.pixels()[o + 3],
            ]
        };
        // Left of the red column red, right of the green column green, and
        // the corners the corner texels.
        assert_eq!(px(r.x - 1, r.y), [255, 0, 0, 255]);
        assert_eq!(px(r.x + 2, r.y + 1), [0, 255, 0, 255]);
        assert_eq!(px(r.x - 2, r.y - 2), [255, 0, 0, 255]);
        assert_eq!(px(r.x + 3, r.y + 3), [0, 255, 0, 255]);
        // An empty image packs without touching pixels.
        assert!(a.insert("empty", 0, 0, &[]).is_ok());
    }

    fn solid(w: u32, h: u32, c: [u8; 4]) -> Vec<u8> {
        c.iter()
            .copied()
            .cycle()
            .take((w * h * 4) as usize)
            .collect()
    }

    #[test]
    fn packs_in_shelves_with_gutters_and_grows() {
        let mut a = TextureAtlas::new(64);
        let r1 = a
            .insert("a", 20, 10, &solid(20, 10, [255, 0, 0, 255]))
            .unwrap();
        let r2 = a
            .insert("b", 20, 30, &solid(20, 30, [0, 255, 0, 255]))
            .unwrap();
        // Third does not fit beside them (2 + 20 + 2 twice is 48; 24 more is 72).
        let r3 = a
            .insert("c", 20, 70, &solid(20, 70, [0, 0, 255, 255]))
            .unwrap();
        assert_eq!((r1.x, r1.y), (2, 2));
        assert_eq!((r2.x, r2.y), (26, 2));
        assert_eq!((r3.x, r3.y), (2, 36), "next shelf, under the tallest");
        assert_eq!(a.size(), (64, 128), "doubled to fit");
        // Pixels landed, and each image's own gutter repeats its edge: where
        // two meet, each side of the boundary is its own image's colour.
        let px = |x: u32, y: u32| &a.pixels()[((y * 64 + x) * 4) as usize..][..4];
        assert_eq!(px(r2.x, r2.y), &[0, 255, 0, 255]);
        assert_eq!(px(r2.x - 1, r2.y), &[0, 255, 0, 255]);
        assert_eq!(px(r1.x + 21, r1.y), &[255, 0, 0, 255]);
        assert_eq!(px(r3.x + 5, r3.y + 69), &[0, 0, 255, 255]);
    }

    #[test]
    fn errors_say_why() {
        let mut a = TextureAtlas::new(32);
        a.max_height = 64;
        assert!(matches!(
            a.insert("w", 40, 1, &solid(40, 1, [0; 4])),
            Err(AtlasError::TooWide { .. })
        ));
        assert!(matches!(
            a.insert("l", 2, 2, &[0; 3]),
            Err(AtlasError::WrongLength { .. })
        ));
        assert!(matches!(
            a.insert("f", 8, 80, &solid(8, 80, [0; 4])),
            Err(AtlasError::Full { .. })
        ));
        let v = a.version();
        a.insert("ok", 8, 8, &solid(8, 8, [1; 4])).unwrap();
        assert!(a.version() > v);
    }

    #[test]
    fn mip_levels_widen_the_gutter() {
        let mut a = TextureAtlas::new(64).with_mip_levels(3);
        let r = a.insert("x", 8, 8, &solid(8, 8, [255; 4])).unwrap();
        assert_eq!(
            (r.x, r.y),
            (4, 4),
            "a 4 px gutter is still a pixel at level 2"
        );
    }
}
