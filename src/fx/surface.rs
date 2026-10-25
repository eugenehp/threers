//! A liquid surface drawn from one distance field: [`Surface`].
//!
//! The shape, plus whatever an effect grows out of it — a mound pulled toward
//! a point, strands, spikes, droplets — all joined with a smooth union whose
//! blend weight carries a local *thickness* through every fillet. The shading
//! treats that thickness as the radius of a round profile, so the shape reads
//! as a pillow and a strand as a tube, and colours it with a [`Look`].
//!
//! An effect builds a `Surface` each frame and the [`Stage`](super::Stage)
//! turns it into a [`ShaderMaterial`] on a quad that shades only the box the
//! surface occupies.

use super::{Look, Shape};
use crate::core::BufferGeometry;
use crate::geometries::PlaneGeometry;
use crate::materials::{RenderState, ShaderMaterial};
use crate::math::Vector2;

/// Strands a surface can carry.
pub const MAX_STRANDS: usize = 7;
/// Points per strand; shorter strands are padded with their last point.
pub const STRAND_POINTS: usize = 12;
pub const MAX_SPIKES: usize = 8;
pub const MAX_DROPS: usize = 6;
/// Thinnest a strand is drawn, in CSS pixels. Below a pixel the shader takes
/// a strand's coverage as a stroke's — the share of the pixel it covers — so
/// a tip can thin to a thread and fade, rather than having to stop at the
/// width (0.55 in the original) where edge coverage starts to break it up.
pub const MIN_RADIUS: f32 = 0.3;

/// A disc: a strand point, a droplet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Blob {
    pub center: Vector2,
    pub radius: f32,
}

/// A tapered spike from `base` (radius `radius`) to a sharp `tip`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spike {
    pub base: Vector2,
    pub tip: Vector2,
    pub radius: f32,
}

/// The surface swelling toward something: a tapered capsule from `from`
/// (radius `base_radius`) to `to` (`tip_radius`), melted into the shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mound {
    pub from: Vector2,
    pub to: Vector2,
    pub base_radius: f32,
    pub tip_radius: f32,
}

/// One frame of a liquid surface. See the [module docs](self).
#[derive(Debug, Clone, PartialEq)]
pub struct Surface {
    pub shape: Shape,
    pub look: Look,
    /// Depth of the shape's round profile, in CSS pixels. [`Surface::new`]
    /// sets the shape's inradius capped at 32 — a large shape gets a pillowed
    /// edge and a flat top. Computing the inradius of a polygon is not free,
    /// so an effect rebuilding its surface every frame should keep it.
    pub thickness: f32,
    /// -1..1: shifts the reflections sideways, as if the studio leant with
    /// the pointer.
    pub lean: f32,
    pub mound: Option<Mound>,
    /// Chains of tapered capsules, at most [`MAX_STRANDS`] of at most
    /// [`STRAND_POINTS`] points. Pieces of one strand join hard (no beads at
    /// the joints); strands melt into each other a little and into the shape
    /// a lot.
    pub strands: Vec<Vec<Blob>>,
    pub spikes: Vec<Spike>,
    pub drops: Vec<Blob>,
}

impl Surface {
    pub fn new(shape: Shape, look: Look) -> Self {
        let thickness = shape.inradius().min(32.0);
        Self { shape, look, thickness, lean: 0.0, mound: None, strands: Vec::new(), spikes: Vec::new(), drops: Vec::new() }
    }

    /// [`new`](Self::new) with the profile depth given rather than measured.
    pub fn new_with_thickness(shape: Shape, look: Look, thickness: f32) -> Self {
        Self { shape, look, thickness, lean: 0.0, mound: None, strands: Vec::new(), spikes: Vec::new(), drops: Vec::new() }
    }

    /// Top-left and bottom-right of everything on the surface.
    pub fn bounds(&self) -> (Vector2, Vector2) {
        let (mut lo, mut hi) = self.shape.bounds();
        let mut grow = |p: Vector2, r: f32| {
            lo = lo.min(p - Vector2::new(r, r));
            hi = hi.max(p + Vector2::new(r, r));
        };
        for s in &self.strands {
            for b in s {
                grow(b.center, b.radius);
            }
        }
        for s in &self.spikes {
            grow(s.tip, 0.0);
            grow(s.base, s.radius);
        }
        for d in &self.drops {
            grow(d.center, d.radius);
        }
        if let Some(m) = &self.mound {
            grow(m.to, m.tip_radius);
        }
        (lo, hi)
    }

    /// The quad a surface is drawn on: big enough that no camera culls it.
    pub fn quad() -> BufferGeometry {
        PlaneGeometry::new(2.0e6, 2.0e6)
    }

    /// The material for this frame, for a viewport `viewport` CSS pixels
    /// across at `dpr` device pixels each. It blends over what is beneath and
    /// shades only the box round the surface and its shadow.
    pub fn material(&self, viewport: Vector2, dpr: f32) -> ShaderMaterial {
        let mut pts = vec![0.0f32; MAX_STRANDS * STRAND_POINTS * 4];
        let strands = self.strands.iter().filter(|s| !s.is_empty()).take(MAX_STRANDS);
        let mut count = 0;
        for (si, s) in strands.enumerate() {
            count += 1;
            for i in 0..STRAND_POINTS {
                let b = s[i.min(s.len() - 1)];
                let o = (si * STRAND_POINTS + i) * 4;
                pts[o..o + 4].copy_from_slice(&[b.center.x, b.center.y, b.radius.max(MIN_RADIUS), 1.0]);
            }
        }
        let mut s1 = vec![0.0f32; MAX_SPIKES * 8 + MAX_DROPS * 4];
        for (k, sp) in self.spikes.iter().take(MAX_SPIKES).enumerate() {
            s1[k * 4..k * 4 + 4].copy_from_slice(&[sp.base.x, sp.base.y, sp.tip.x, sp.tip.y]);
            let o = MAX_SPIKES * 4 + k * 4;
            s1[o] = sp.radius;
            s1[o + 1] = 1.0;
        }
        for (k, d) in self.drops.iter().take(MAX_DROPS).enumerate() {
            let o = MAX_SPIKES * 8 + k * 4;
            s1[o..o + 4].copy_from_slice(&[d.center.x, d.center.y, d.radius, 1.0]);
        }
        let (kind, shape_slot, corner, poly) = match &self.shape {
            Shape::RoundRect { center, half, radius } => {
                let r = radius.max(0.0).min(half.x.min(half.y));
                (0.0, [center.x, center.y, half.x, half.y], r, Vec::new())
            }
            Shape::Circle { center, radius } => (1.0, [center.x, center.y, *radius, 0.0], 0.0, Vec::new()),
            Shape::Polygon { points, radius } => {
                (2.0, [0.0; 4], *radius, points.iter().flat_map(|p| [p.x, p.y]).collect::<Vec<f32>>())
            }
        };
        let look = &self.look;
        let (lo, hi) = self.bounds();
        // room for the soft shadow, cast down and to the right
        let pad = 34.0;
        let mut data = vec![[0.0f32; 4]; 16];
        data[0] = [viewport.x.max(1.0), viewport.y.max(1.0), dpr, count as f32];
        data[1] = shape_slot;
        data[3] = [0.0, 0.0, self.lean, 0.0];
        if let Some(m) = &self.mound {
            data[2] = [m.from.x, m.from.y, m.to.x, m.to.y];
            data[3][0] = m.base_radius;
            data[3][1] = m.tip_radius;
        }
        data[4] = [lo.x - pad, lo.y - pad, hi.x + pad, hi.y + pad + 10.0];
        data[5] = [kind, corner, self.thickness.max(1.0), (poly.len() / 2) as f32];
        data[6] = [look.base[0], look.base[1], look.base[2], look.f0];
        data[7] = [look.tint[0], look.tint[1], look.tint[2], look.glints];
        data[8] = [look.softbox, look.window, look.fill, look.floor];
        data[9] = [look.shadow, look.front, 0.0, 0.0];
        ShaderMaterial {
            vertex: Some(VERTEX.to_string()),
            side: 2,
            tone_mapped: false,
            render_state: RenderState { depth_test: false, depth_write: Some(false), ..RenderState::DEFAULT },
            ..ShaderMaterial::new(source())
        }
        .with_transparent(true)
        .with_data(data)
        .with_storage0(pts)
        .with_storage1(s1)
        .with_storage2(poly)
    }
}

/// A full-viewport vertical gradient behind the elements, as CSS
/// `linear-gradient(180deg, top, middle 55%, bottom)` draws it. Drawn by the
/// surface shader, so every backend that draws a surface draws this.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Backdrop {
    pub top: [f32; 3],
    pub middle: [f32; 3],
    pub bottom: [f32; 3],
}

impl Backdrop {
    /// The pale studio grey the venom button sits on.
    pub const STUDIO: Backdrop = Backdrop {
        top: [245.0 / 255.0, 244.0 / 255.0, 246.0 / 255.0],
        middle: [239.0 / 255.0, 238.0 / 255.0, 241.0 / 255.0],
        bottom: [228.0 / 255.0, 227.0 / 255.0, 230.0 / 255.0],
    };

    /// From three `0xRRGGBB` colours.
    pub fn gradient(top: u32, middle: u32, bottom: u32) -> Self {
        let c = |v: u32| [((v >> 16) & 255) as f32 / 255.0, ((v >> 8) & 255) as f32 / 255.0, (v & 255) as f32 / 255.0];
        Self { top: c(top), middle: c(middle), bottom: c(bottom) }
    }

    pub fn solid(rgb: u32) -> Self {
        Self::gradient(rgb, rgb, rgb)
    }

    /// The material for a `viewport` CSS pixels across at `dpr`. Opaque.
    pub fn material(&self, viewport: Vector2, dpr: f32) -> ShaderMaterial {
        let mut data = vec![[0.0f32; 4]; 16];
        data[0] = [viewport.x.max(1.0), viewport.y.max(1.0), dpr, 0.0];
        data[4] = [0.0, 0.0, viewport.x.max(1.0), viewport.y.max(1.0)];
        data[5] = [3.0, 0.0, 1.0, 0.0];
        data[6] = [self.top[0], self.top[1], self.top[2], 0.0];
        data[7] = [self.middle[0], self.middle[1], self.middle[2], 0.0];
        data[8] = [self.bottom[0], self.bottom[1], self.bottom[2], 0.0];
        ShaderMaterial {
            vertex: Some(VERTEX.to_string()),
            side: 2,
            tone_mapped: false,
            render_state: RenderState { depth_test: false, depth_write: Some(false), ..RenderState::DEFAULT },
            ..ShaderMaterial::new(source())
        }
        .with_data(data)
    }
}

/// Floats in [`metal_bytes`]: the sixteen uniform slots, the strand points,
/// spikes and drops, and the polygon — each padded to its maximum.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub(crate) const METAL_FLOATS: usize =
    16 * 4 + MAX_STRANDS * STRAND_POINTS * 4 + MAX_SPIKES * 8 + MAX_DROPS * 4 + super::shape::MAX_POLYGON_POINTS * 2;

/// Whether `material` is a surface (or a backdrop) drawn by this module —
/// what the Metal renderer, which runs no WGSL, recognises to draw it with its
/// own port of the shader.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub(crate) fn is_surface(material: &ShaderMaterial) -> bool {
    material.vertex.as_deref() == Some(VERTEX)
}

/// A surface material's uniforms and arrays in one block, laid out as the
/// Metal port's `FxSurfaceData` reads them. Under Metal's 4 KB limit for
/// `setBytes`, so no buffer is needed.
#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
pub(crate) fn metal_bytes(material: &ShaderMaterial) -> Box<[f32; METAL_FLOATS]> {
    let mut out = Box::new([0.0f32; METAL_FLOATS]);
    let mut at = 0;
    for slot in material.data_slots() {
        out[at..at + 4].copy_from_slice(&slot);
        at += 4;
    }
    for (src, len) in [
        (&material.storage0, MAX_STRANDS * STRAND_POINTS * 4),
        (&material.storage1, MAX_SPIKES * 8 + MAX_DROPS * 4),
        (&material.storage2, super::shape::MAX_POLYGON_POINTS * 2),
    ] {
        let n = src.len().min(len);
        out[at..at + n].copy_from_slice(&src[..n]);
        at += len;
    }
    out
}

const FRAGMENT: &str = include_str!("surface.wgsl");
const VERTEX: &str = include_str!("surface_vs.wgsl");

/// The whole fragment source, built once.
fn source() -> String {
    static SOURCE: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SOURCE.get_or_init(|| format!("{}\n{}", crate::sdf::WGSL, FRAGMENT)).clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_material_carries_the_shape_and_its_box() {
        let mut s = Surface::new(Shape::circle(Vector2::new(100.0, 80.0), 30.0), Look::GOLD);
        s.strands.push(vec![Blob { center: Vector2::new(140.0, 20.0), radius: 2.0 }; 3]);
        let m = s.material(Vector2::new(800.0, 600.0), 2.0);
        let d = m.data_slots();
        assert_eq!(d[0], [800.0, 600.0, 2.0, 1.0]);
        assert_eq!(d[5][0], 1.0, "kind: circle");
        assert!(d[4][0] < 70.0 && d[4][1] < 0.0 && d[4][2] > 140.0 && d[4][3] > 110.0, "{:?}", d[4]);
        assert_eq!(d[6][..3], Look::GOLD.base);
        // a short strand is padded with its last point
        assert_eq!(m.storage0[(STRAND_POINTS - 1) * 4], 140.0);
    }

    /// Spliced into the real preamble the way the renderer does it, with
    /// storage buffers and with the WebGL2 texture fallback, and validated —
    /// so a name the preamble does not have fails here rather than at
    /// pipeline creation.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_shader_validates_against_the_preamble() {
        use crate::renderer::{custom_preamble_lines, custom_shader_module_source, validate_custom_shader};
        for storage in [true, false] {
            let src = custom_shader_module_source(&source(), Some(VERTEX), storage, false, 1);
            validate_custom_shader(&src, custom_preamble_lines()).unwrap_or_else(|e| panic!("storage={storage}: {e}"));
        }
    }
}
