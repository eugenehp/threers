//! A creature from fields: a soft body, parts glued on that can tear away and
//! regrow, and particle chains rooted in the body — everything
//! [`SoftBody::new`](super::SoftBody::new) takes, built the way the gummy
//! creatures demo's `creature-kit.js` builds it.
//!
//! [`Creature::from_shapes`] takes [`Shape`](crate::sdf::Shape)s and meshes
//! their skins by surface nets; [`Creature::from_fields`] takes any field and
//! any skin points, which is what the parity test feeds both sides.

use threers_softbody::js_math::{self as js, Lcg};
use threers_softbody::{Bounds, Chains, Landmarks, Limb, Part, SoftBody};
use crate::math::Vector3;
use crate::sdf::{Sdf3, Shape, SurfaceNets};

/// A field the lattice is built over, with the box it is built in.
pub struct Field<'a> {
    pub f: &'a dyn Fn(f64, f64, f64) -> f64,
    pub bounds: Bounds,
    /// Lattice spacing.
    pub h: f64,
    /// The skin points to bind to it, xyz.
    pub skin: Vec<f32>,
}

/// A chain: its rest points from root (in the body) to tip, radius from `r0`
/// to `r1`, and how firmly it holds its shape.
#[derive(Debug, Clone)]
pub struct ChainSpec {
    pub points: Vec<[f64; 3]>,
    pub r0: f64,
    pub r1: f64,
    pub stiff: f64,
}

/// Rest points the solver reads: three for the head's frame, two for the size.
#[derive(Debug, Clone, Copy)]
pub struct Marks {
    pub head: [[f64; 3]; 3],
    pub span: [[f64; 3]; 2],
}

/// A built creature: the parts, ready for [`SoftBody::new`](super::SoftBody::new).
pub struct Creature {
    pub body: Part,
    pub limbs: Vec<Limb>,
    pub chains: Chains,
    pub landmarks: Landmarks,
    /// The skins, when built from shapes: the body's first, then each part's.
    pub skins: Vec<SurfaceNets>,
}

fn nearest(p: &[f32], q: [f64; 3]) -> usize {
    let (mut best, mut bd) = (usize::MAX, f64::INFINITY);
    for i in 0..p.len() / 3 {
        let d = (p[3 * i] as f64 - q[0]) * (p[3 * i] as f64 - q[0])
            + (p[3 * i + 1] as f64 - q[1]) * (p[3 * i + 1] as f64 - q[1])
            + (p[3 * i + 2] as f64 - q[2]) * (p[3 * i + 2] as f64 - q[2]);
        if d < bd {
            bd = d;
            best = i;
        }
    }
    best
}

impl Creature {
    /// Its soft body, the goo drawn from a generator seeded with `seed`.
    pub fn into_soft_body(self, seed: u32) -> SoftBody {
        SoftBody::new(&self.body, self.limbs, self.chains, self.landmarks, Box::new(Lcg(seed)))
    }

    /// Build from fields. `body_f` is the body's field again, for the glue:
    /// a part node holds on to the body where it is within 0.025 of it.
    pub fn from_fields(
        body: Field,
        parts: Vec<(String, Field)>,
        chains: &[ChainSpec],
        marks: Marks,
    ) -> Result<Creature, String> {
        let body_f = body.f;
        let body_part = Part::new(body.f, &body.bounds, body.h, &body.skin);
        let bn = body_part.lattice.nodes.clone();
        let mut limbs = Vec::new();
        for (name, field) in parts {
            let part = Part::new(field.f, &field.bounds, field.h, &field.skin);
            let n = &part.lattice.nodes;
            let (mut glue_part, mut glue_body) = (Vec::new(), Vec::new());
            for a in 0..n.len() / 3 {
                if body_f(n[3 * a] as f64, n[3 * a + 1] as f64, n[3 * a + 2] as f64) > 0.025 {
                    continue;
                }
                let mut near: Vec<(f64, u32)> = Vec::new();
                for j in 0..bn.len() / 3 {
                    let d = js::hypot3(
                        n[3 * a] as f64 - bn[3 * j] as f64,
                        n[3 * a + 1] as f64 - bn[3 * j + 1] as f64,
                        n[3 * a + 2] as f64 - bn[3 * j + 2] as f64,
                    );
                    if d < 0.12 {
                        near.push((d, j as u32));
                    }
                }
                near.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
                for &(_, j) in near.iter().take(3) {
                    glue_part.push(a as u32);
                    glue_body.push(j);
                }
            }
            if glue_part.is_empty() {
                return Err(format!("the {name} does not reach into the body"));
            }
            // the pivot: the mean of the glued nodes, each counted once, in
            // the order they were first glued
            let mut uniq: Vec<u32> = Vec::new();
            for &a in &glue_part {
                if !uniq.contains(&a) {
                    uniq.push(a);
                }
            }
            let mut pivot = [0.0f64; 3];
            for &a in &uniq {
                for c in 0..3 {
                    pivot[c] += n[3 * a as usize + c] as f64 / uniq.len() as f64;
                }
            }
            let anchor = nearest(&bn, pivot);
            limbs.push(Limb { part, name, glue_part, glue_body, pivot, anchor });
        }
        let chains = Self::chains(chains, &body_part);
        let landmarks = Landmarks {
            head: marks.head.map(|p| nearest(&bn, p)),
            snout: nearest(&bn, marks.span[0]),
            tail: nearest(&bn, marks.span[1]),
        };
        Ok(Creature { body: body_part, limbs, chains, landmarks, skins: Vec::new() })
    }

    fn chains(specs: &[ChainSpec], body: &Part) -> Chains {
        let mut c = Chains::default();
        let mut rad: Vec<f64> = Vec::new();
        for s in specs {
            let n = s.points.len().saturating_sub(1).max(1);
            let mut root = -1i32;
            for (k, p) in s.points.iter().enumerate() {
                let t = k as f64 / n as f64;
                let id = c.rest.len() / 3;
                c.rest.extend(p.map(|v| v as f32));
                let r = s.r0 + (s.r1 - s.r0) * t;
                rad.push(r);
                c.radius.push(r as f32);
                c.parent.push(if k == 0 { -1 } else { id as i32 - 1 });
                c.anchor.push(if k == 0 { -1 } else { root });
                c.stiff.push(if k == 0 { 0.0 } else { (s.stiff * (1.0 - 0.5 * t)) as f32 });
                c.seg_radius.push(if k == 0 { 0.0 } else { rad[id - 1] as f32 });
                if k == 0 {
                    root = id as i32;
                    c.roots.push(id as i32);
                }
            }
        }
        let root_pts: Vec<f32> = c.roots.iter().flat_map(|&r| {
            let r = r as usize;
            [c.rest[3 * r], c.rest[3 * r + 1], c.rest[3 * r + 2]]
        }).collect();
        c.root_emb = body.embed(&root_pts);
        c
    }

    /// Build from shapes: each skin meshed by surface nets at `cell`, the
    /// lattice at `h`. A part is stopped just inside the body (0.03), so its
    /// skin is a closed shape of its own.
    pub fn from_shapes(body: &ShapeSpec, parts: &[(String, ShapeSpec)], chains: &[ChainSpec], marks: Marks) -> Result<Creature, String> {
        let inner = body.shape.clone().round(-0.03);
        let prepared: Vec<(String, Shape, f32, f64)> = parts
            .iter()
            .map(|(name, p)| (name.clone(), p.shape.clone().subtract(inner.clone(), 0.0), p.cell, p.h))
            .collect();
        let mut skins = Vec::new();
        let mut build = |shape: &Shape, cell: f32, h: f64| -> Result<(Bounds, Vec<f32>), String> {
            let (lo, hi) = shape.bounds().ok_or("a body or part shape has no bounds")?;
            let pad = cell as f64 * 3.0;
            let bounds = Bounds {
                min: [lo.x as f64 - pad, -0.03, lo.z as f64 - pad],
                max: [hi.x as f64 + pad, hi.y as f64 + pad, hi.z as f64 + pad],
            };
            let f32v = |v: [f64; 3]| Vector3::new(v[0] as f32, v[1] as f32, v[2] as f32);
            let m = shape.to_surface_nets(cell, f32v(bounds.min), f32v(bounds.max));
            let skin = m.position.iter().flat_map(|p| [p.x, p.y, p.z]).collect();
            skins.push(m);
            let _ = h;
            Ok((bounds, skin))
        };
        let (bb, bskin) = build(&body.shape, body.cell, body.h)?;
        let mut part_fields = Vec::new();
        for (name, shape, cell, h) in &prepared {
            let (b, skin) = build(shape, *cell, *h)?;
            part_fields.push((name.clone(), shape, b, *h, skin));
        }
        fn field_of(s: &Shape) -> impl Fn(f64, f64, f64) -> f64 + '_ {
            move |x, y, z| s.distance(Vector3::new(x as f32, y as f32, z as f32)) as f64
        }
        let bf = field_of(&body.shape);
        let pfs: Vec<_> = part_fields.iter().map(|(_, s, ..)| field_of(s)).collect();
        let mut c = Creature::from_fields(
            Field { f: &bf, bounds: bb, h: body.h, skin: bskin },
            part_fields
                .iter()
                .zip(&pfs)
                .map(|((name, _, b, h, skin), f)| (name.clone(), Field { f, bounds: *b, h: *h, skin: skin.clone() }))
                .collect(),
            chains,
            marks,
        )?;
        c.skins = skins;
        Ok(c)
    }
}

/// A body or part to build from a [`Shape`]: skin cell and lattice spacing.
#[derive(Debug, Clone)]
pub struct ShapeSpec {
    pub shape: Shape,
    pub cell: f32,
    pub h: f64,
}
