//! AP203/AP214 advanced B-rep → `Body`.
//!
//! The inverse of [`mod@super::export`], and held to the same rule: an entity that
//! cannot be mapped is *reported*, not guessed at. A caller that gets an empty
//! [`ImportReport::skipped`] knows the `Body` is the whole file.
//!
//! Reading is the harder direction, because a file need not have been written
//! here. Three things vary in practice and are handled:
//!
//! * **Trimming.** An `EDGE_CURVE` names a curve and two vertices; which *arc*
//!   of a circle is meant follows from the vertices and the sense, not from the
//!   curve. A full circle names the same vertex twice.
//! * **Placement conventions.** A `CONICAL_SURFACE` is placed at a reference
//!   circle of some radius, not at its apex, so the apex has to be recovered.
//! * **Complex instances.** Rational B-splines and units are written as several
//!   entity names juxtaposed, which [`super::part21`] already keeps intact.

use std::collections::HashMap;

use crate::brep::body::Edge;
use crate::brep::{Body, Face, Surface, TrimLoop};
use crate::nurbs::{v3, NurbsSurface, V3, V4};

use super::part21::{parse, Entity, ParseError, StepFile, Value};

/// What the import read, and what it could not.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImportReport {
    pub faces: usize,
    pub edges: usize,
    pub vertices: usize,
    /// Entity names encountered where a surface or curve was expected but which
    /// have no mapping here, with how many times each occurred.
    pub skipped: Vec<(String, usize)>,
}

impl ImportReport {
    /// Whether every face in the file came through.
    pub fn is_complete(&self) -> bool {
        self.skipped.is_empty()
    }
}

/// Why a file could not be read as a solid.
#[derive(Debug, Clone, PartialEq)]
pub enum ImportError {
    /// The exchange syntax itself is malformed.
    Syntax(ParseError),
    /// Parsed, but with no `CLOSED_SHELL` — nothing here is a solid.
    NoSolid,
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Syntax(e) => write!(f, "{e}"),
            ImportError::NoSolid => f.write_str("no CLOSED_SHELL in the file"),
        }
    }
}

impl std::error::Error for ImportError {}

impl From<ParseError> for ImportError {
    fn from(e: ParseError) -> Self {
        ImportError::Syntax(e)
    }
}

/// Read the first solid in an exchange file.
///
/// `tolerance` decides when two vertices are the same one. A file's vertices are
/// already shared by reference, so this only matters for files whose writer
/// duplicated them — which is common enough to be worth doing.
pub fn import(text: &str, tolerance: f64) -> Result<(Body, ImportReport), ImportError> {
    let file = parse(text)?;
    Reader::new(&file, tolerance).solid()
}

struct Reader<'a> {
    index: HashMap<u64, &'a Entity>,
    file: &'a StepFile,
    tolerance: f64,
    /// Welded by position, so a file that repeats a corner still closes.
    vertices: Vec<V3>,
    by_position: HashMap<(i64, i64, i64), usize>,
    skipped: HashMap<String, usize>,
}

impl<'a> Reader<'a> {
    fn new(file: &'a StepFile, tolerance: f64) -> Self {
        Reader {
            index: file.index(),
            file,
            tolerance: tolerance.max(1e-12),
            vertices: Vec::new(),
            by_position: HashMap::new(),
            skipped: HashMap::new(),
        }
    }

    fn get(&self, v: &Value) -> Option<&'a Entity> {
        self.index.get(&v.as_ref()?).copied()
    }

    fn skip(&mut self, name: &str) {
        *self.skipped.entry(name.to_string()).or_insert(0) += 1;
    }

    fn vertex(&mut self, p: V3) -> usize {
        let q = |x: f64| (x / (self.tolerance * 0.5)).round() as i64;
        let key = (q(p[0]), q(p[1]), q(p[2]));
        if let Some(i) = self.by_position.get(&key) {
            return *i;
        }
        self.vertices.push(p);
        self.by_position.insert(key, self.vertices.len() - 1);
        self.vertices.len() - 1
    }

    // ---- geometry ------------------------------------------------------

    fn cartesian(&self, e: &Entity) -> Option<V3> {
        let c = e.part("CARTESIAN_POINT")?.get(1)?.as_list()?;
        Some([
            c.first()?.as_real()?,
            c.get(1)?.as_real()?,
            c.get(2).and_then(|x| x.as_real()).unwrap_or(0.0),
        ])
    }

    fn direction(&self, e: &Entity) -> Option<V3> {
        let c = e.part("DIRECTION")?.get(1)?.as_list()?;
        v3::normalize([
            c.first()?.as_real()?,
            c.get(1)?.as_real()?,
            c.get(2).and_then(|x| x.as_real()).unwrap_or(0.0),
        ])
    }

    /// `(origin, axis, x_dir)` from an `AXIS2_PLACEMENT_3D`.
    ///
    /// Both directions are optional in the schema: a placement with `$` for its
    /// reference direction means "any perpendicular will do".
    fn placement(&self, e: &Entity) -> Option<(V3, V3, V3)> {
        let a = e.part("AXIS2_PLACEMENT_3D")?;
        let origin = self.cartesian(self.get(a.get(1)?)?)?;
        let axis = a
            .get(2)
            .and_then(|v| self.get(v))
            .and_then(|d| self.direction(d))
            .unwrap_or([0.0, 0.0, 1.0]);
        let x = a
            .get(3)
            .and_then(|v| self.get(v))
            .and_then(|d| self.direction(d))
            .unwrap_or_else(|| {
                let alt = if axis[0].abs() < 0.9 {
                    [1.0, 0.0, 0.0]
                } else {
                    [0.0, 1.0, 0.0]
                };
                v3::normalize(v3::cross(axis, alt)).unwrap_or([1.0, 0.0, 0.0])
            });
        Some((origin, axis, x))
    }

    fn surface(&mut self, e: &Entity) -> Option<Surface> {
        if let Some(a) = e.part("PLANE") {
            let (o, n, x) = self.placement(self.get(a.get(1)?)?)?;
            return Some(Surface::plane(o, n).with_x_dir(x));
        }
        if let Some(a) = e.part("CYLINDRICAL_SURFACE") {
            let (o, n, x) = self.placement(self.get(a.get(1)?)?)?;
            let r = a.get(2)?.as_real()?;
            return Some(Surface::cylinder(o, n, r).with_x_dir(x));
        }
        if let Some(a) = e.part("SPHERICAL_SURFACE") {
            let (c, n, x) = self.placement(self.get(a.get(1)?)?)?;
            let r = a.get(2)?.as_real()?;
            return Some(Surface::sphere(c, r).with_axis(n).with_x_dir(x));
        }
        if let Some(a) = e.part("CONICAL_SURFACE") {
            let (o, n, x) = self.placement(self.get(a.get(1)?)?)?;
            let radius = a.get(2)?.as_real()?;
            let half_angle = a.get(3)?.as_real()?;
            // The placement sits on a reference circle of `radius`, which is the
            // apex only when that radius is zero. Walk back down the axis to
            // find it, since `Surface::Cone` is apex-based.
            let apex = if half_angle.abs() > 1e-12 && radius.abs() > 1e-12 {
                v3::sub(o, v3::scale(n, radius / half_angle.tan()))
            } else {
                o
            };
            return Some(Surface::cone(apex, n, half_angle).with_x_dir(x));
        }
        if let Some(a) = e.part("TOROIDAL_SURFACE") {
            let (c, n, x) = self.placement(self.get(a.get(1)?)?)?;
            let major = a.get(2)?.as_real()?;
            let minor = a.get(3)?.as_real()?;
            return Some(Surface::torus(c, n, major, minor).with_x_dir(x));
        }
        if e.is("B_SPLINE_SURFACE") || e.is("B_SPLINE_SURFACE_WITH_KNOTS") {
            return self.bspline_surface(e).map(Surface::nurbs);
        }
        let name = if e.name.is_empty() {
            e.args
                .iter()
                .find_map(|a| match a {
                    Value::Typed(n, _) => Some(n.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "complex".into())
        } else {
            e.name.clone()
        };
        self.skip(&name);
        None
    }

    fn bspline_surface(&self, e: &Entity) -> Option<NurbsSurface> {
        // The two spellings put the same data in different places: a simple
        // `B_SPLINE_SURFACE_WITH_KNOTS` carries everything, while the rational
        // form splits it across the parts of a complex instance.
        let base = e
            .part("B_SPLINE_SURFACE")
            .or_else(|| e.part("B_SPLINE_SURFACE_WITH_KNOTS"))?;
        let simple = e.part("B_SPLINE_SURFACE").is_none();
        let off = if simple { 1 } else { 0 };

        let degree_u = base.get(off)?.as_int()? as usize;
        let degree_v = base.get(off + 1)?.as_int()? as usize;
        let rows = base.get(off + 2)?.as_list()?;

        let knots = e.part("B_SPLINE_SURFACE_WITH_KNOTS")?;
        // In the complex form the knot part starts at its own index 0; in the
        // simple form it continues the same argument list.
        let k = if simple { 8 } else { 0 };
        let mult_u = knots.get(k)?.as_list()?;
        let mult_v = knots.get(k + 1)?.as_list()?;
        let knot_u = knots.get(k + 2)?.as_list()?;
        let knot_v = knots.get(k + 3)?.as_list()?;

        let expand = |ks: &[Value], ms: &[Value]| -> Option<Vec<f64>> {
            let mut out = Vec::new();
            for (kv, mv) in ks.iter().zip(ms) {
                let k = kv.as_real()?;
                for _ in 0..mv.as_int()?.max(0) {
                    out.push(k);
                }
            }
            Some(out)
        };
        let knots_u = expand(knot_u, mult_u)?;
        let knots_v = expand(knot_v, mult_v)?;

        let weights = e
            .part("RATIONAL_B_SPLINE_SURFACE")
            .and_then(|w| w.first())
            .and_then(|w| w.as_list());

        let n_u = rows.len();
        let n_v = rows.first()?.as_list()?.len();
        let mut control: Vec<V4> = Vec::with_capacity(n_u * n_v);
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_list()?;
            if row.len() != n_v {
                return None; // a ragged net is not a surface
            }
            for (j, cell) in row.iter().enumerate() {
                let p = self.cartesian(self.get(cell)?)?;
                let w = weights
                    .and_then(|ws| ws.get(i))
                    .and_then(|r| r.as_list())
                    .and_then(|r| r.get(j))
                    .and_then(|x| x.as_real())
                    .unwrap_or(1.0);
                control.push([p[0] * w, p[1] * w, p[2] * w, w]);
            }
        }
        NurbsSurface::from_homogeneous(degree_u, degree_v, knots_u, knots_v, n_u, n_v, control).ok()
    }

    /// Sample an `EDGE_CURVE` into the polyline the B-rep stores.
    ///
    /// A `Body`'s edge is a vertex chain, so a circle has to be sampled — but
    /// only *here*, at the boundary, and the surfaces it bounds stay exact. The
    /// arc is chosen by the edge's own endpoints, which is the only thing that
    /// distinguishes it from the rest of the circle.
    fn edge_points(&mut self, e: &Entity, tolerance: f64) -> Option<(Vec<V3>, bool)> {
        let a = e.part("EDGE_CURVE")?;
        let start =
            self.cartesian(self.get(self.get(a.get(1)?)?.part("VERTEX_POINT")?.get(1)?)?)?;
        let end = self.cartesian(self.get(self.get(a.get(2)?)?.part("VERTEX_POINT")?.get(1)?)?)?;
        let geom = self.get(a.get(3)?)?;
        let same_sense = a.get(4).and_then(|s| s.as_bool()).unwrap_or(true);
        let closed = v3::dist(start, end) < self.tolerance;

        if geom.is("LINE") {
            return Some((vec![start, end], false));
        }
        if geom.is("POLYLINE") {
            let pts = geom.part("POLYLINE")?.get(1)?.as_list()?;
            let mut out: Vec<V3> = pts
                .iter()
                .filter_map(|p| self.get(p).and_then(|e| self.cartesian(e)))
                .collect();
            if !same_sense {
                out.reverse();
            }
            let closed = out.len() > 2 && v3::dist(out[0], *out.last()?) < self.tolerance;
            return Some((out, closed));
        }
        if let Some(c) = geom.part("CIRCLE") {
            let (center, axis, x) = self.placement(self.get(c.get(1)?)?)?;
            let radius = c.get(2)?.as_real()?;
            let y = v3::cross(axis, x);
            let angle = |p: V3| -> f64 {
                let d = v3::sub(p, center);
                v3::dot(d, y).atan2(v3::dot(d, x))
            };
            let t0 = angle(start);
            let mut sweep = if closed {
                std::f64::consts::TAU
            } else {
                let mut s = angle(end) - t0;
                while s <= 0.0 {
                    s += std::f64::consts::TAU;
                }
                s
            };
            if !same_sense {
                sweep -= std::f64::consts::TAU;
            }
            let n = segments(radius, sweep.abs(), tolerance);
            let pts: Vec<V3> = (0..=n)
                .map(|i| {
                    let t = t0 + sweep * i as f64 / n as f64;
                    v3::add(
                        center,
                        v3::add(
                            v3::scale(x, radius * t.cos()),
                            v3::scale(y, radius * t.sin()),
                        ),
                    )
                })
                .collect();
            return Some((pts, closed));
        }

        let name = if geom.name.is_empty() {
            "complex".to_string()
        } else {
            geom.name.clone()
        };
        self.skip(&name);
        // The endpoints are still known, so the topology survives even when the
        // curve's shape does not. Reporting it is what keeps that honest.
        Some((vec![start, end], false))
    }

    fn solid(mut self) -> Result<(Body, ImportReport), ImportError> {
        let Some(shell) = self
            .file
            .all("CLOSED_SHELL")
            .next()
            .or_else(|| self.file.all("OPEN_SHELL").next())
        else {
            return Err(ImportError::NoSolid);
        };
        let face_refs: Vec<Value> = shell
            .part("CLOSED_SHELL")
            .or_else(|| shell.part("OPEN_SHELL"))
            .and_then(|a| a.get(1))
            .and_then(|l| l.as_list())
            .map(|l| l.to_vec())
            .unwrap_or_default();

        let mut surfaces: Vec<Surface> = Vec::new();
        let mut faces: Vec<Face> = Vec::new();
        // The pole vertices each face names, parallel to `faces`.
        let mut poles: Vec<Vec<V3>> = Vec::new();
        // Which faces named bounds but no outer one.
        let mut complements: Vec<bool> = Vec::new();
        // Each face's bounds, in file order, with each edge use's direction.
        let mut walks: Vec<Vec<Vec<(usize, bool)>>> = Vec::new();
        let mut edges: Vec<Edge> = Vec::new();
        // One `EDGE_CURVE` used by two faces must become *one* `Edge`, or the
        // solid arrives open along every seam. The same holds for a surface: two
        // faces cut from one `SPHERICAL_SURFACE` must share it, or nothing can
        // tell that between them they cover it exactly once.
        let mut edge_of_entity: HashMap<u64, usize> = HashMap::new();
        let mut surface_of_entity: HashMap<u64, usize> = HashMap::new();

        let tolerance = self.tolerance;
        for fr in &face_refs {
            let Some(fe) = self.get(fr) else { continue };
            let Some(a) = fe.part("ADVANCED_FACE").or_else(|| fe.part("FACE_SURFACE")) else {
                continue;
            };
            let Some(se) = a.get(2).and_then(|v| self.get(v)) else {
                continue;
            };
            let se_id = se.id;
            let surface_index = match surface_of_entity.get(&se_id) {
                Some(i) => *i,
                None => {
                    let Some(surface) = self.surface(se) else {
                        continue;
                    };
                    surfaces.push(surface);
                    surface_of_entity.insert(se_id, surfaces.len() - 1);
                    surfaces.len() - 1
                }
            };
            let flipped = !a.get(3).and_then(|s| s.as_bool()).unwrap_or(true);

            let mut face_edges = Vec::new();
            let mut face_poles: Vec<V3> = Vec::new();
            let mut face_walks: Vec<Vec<(usize, bool)>> = Vec::new();
            // A face with bounds but no *outer* one is everything except what
            // those bounds enclose — see `complement_loops`.
            let mut has_outer = false;
            for bound in a.get(1).and_then(|l| l.as_list()).unwrap_or(&[]).to_vec() {
                let Some(be) = self.get(&bound) else { continue };
                let outer = be.part("FACE_OUTER_BOUND");
                has_outer |= outer.is_some();
                let Some(ba) = outer.or_else(|| be.part("FACE_BOUND")) else {
                    continue;
                };
                let Some(le) = ba.get(1).and_then(|v| self.get(v)) else {
                    continue;
                };
                // A `VERTEX_LOOP` is a bound of one point: the pole a face
                // runs up to, where the iso-curve collapses and there is no
                // edge to carry it. It contributes no edge, only the parameter
                // its vertex sits at — without which the face's extent stops at
                // its rim and the cap between rim and pole is lost.
                if let Some(va) = le.part("VERTEX_LOOP") {
                    if let Some(p) = va
                        .get(1)
                        .and_then(|v| self.get(v))
                        .and_then(|e| e.part("VERTEX_POINT"))
                        .and_then(|vp| vp.get(1))
                        .and_then(|r| self.get(r))
                        .and_then(|c| self.cartesian(c))
                    {
                        face_poles.push(p);
                    }
                    continue;
                }
                let Some(la) = le.part("EDGE_LOOP") else {
                    continue;
                };
                // Kept in the order the file gives, with each use's direction. A
                // face that meets itself along a seam names one edge twice, and
                // the only thing telling the two apart is where they sit in the
                // walk.
                let mut walk: Vec<(usize, bool)> = Vec::new();
                for oe in la.get(1).and_then(|l| l.as_list()).unwrap_or(&[]).to_vec() {
                    let Some(oriented) = self.get(&oe) else {
                        continue;
                    };
                    let Some(oa) = oriented.part("ORIENTED_EDGE") else {
                        continue;
                    };
                    let Some(ec_ref) = oa.get(3).and_then(|v| v.as_ref()) else {
                        continue;
                    };
                    let forward = oriented
                        .part("ORIENTED_EDGE")
                        .and_then(|o| o.get(4))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    if let Some(&i) = edge_of_entity.get(&ec_ref) {
                        // Twice is allowed, and means a seam: `EdgeFaceCount`
                        // counts uses rather than distinct faces so it can.
                        if face_edges.iter().filter(|&&x| x == i).count() < 2 {
                            face_edges.push(i);
                        }
                        walk.push((i, forward));
                        continue;
                    }
                    let Some(ec) = self.index.get(&ec_ref).copied() else {
                        continue;
                    };
                    let Some((pts, closed)) = self.edge_points(ec, tolerance) else {
                        continue;
                    };
                    let vs: Vec<usize> = pts.into_iter().map(|p| self.vertex(p)).collect();
                    edges.push(Edge {
                        // The surface this side of the edge lies on; the other
                        // is filled in below, once its second user is known. A
                        // seam cut into one closed surface has the same surface
                        // on both sides, which is correct and not a defect.
                        surfaces: (surface_index, surface_index),
                        vertices: vs,
                        closed,
                    });
                    edge_of_entity.insert(ec_ref, edges.len() - 1);
                    face_edges.push(edges.len() - 1);
                    walk.push((edges.len() - 1, forward));
                }
                if !walk.is_empty() {
                    face_walks.push(walk);
                }
            }

            // A file says nothing about periodicity — it is a property of the
            // surface, and the face inherits it. Without this a cylindrical face
            // tessellates as a slit rather than a tube, and the solid arrives
            // open along a seam the file never mentioned.
            faces.push(Face {
                surface: surface_index,
                edges: face_edges,
                flipped,
                // Derived by `from_parts` from the vertices the edges reach,
                // which is where a STEP face's extent actually lives.
                u_range: (0.0, 0.0),
                v_range: (0.0, 0.0),
                u_wraps: false,
                v_wraps: false,
                loops: None,
            });
            poles.push(face_poles);
            complements.push(!has_outer);
            walks.push(face_walks);
        }

        // Recover each face's parameter extent from the vertices its own edges
        // reach. A STEP face carries no parameter range — its extent *is* its
        // boundary — so this is where that implicit fact is made explicit, and
        // without it a face tessellates as a zero-area sliver.
        // Patches cut from one closed surface are ambiguous from their boundary
        // alone: the two halves of a seam-split sphere are bounded by the *same*
        // two meridians, so both derive the same range and one of them is wrong.
        // What settles it is that the pieces tile the surface exactly once —
        // so a range already claimed on this surface means take the complement.
        let mut claimed: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();
        for (fi, face) in faces.iter_mut().enumerate() {
            let surface = &surfaces[face.surface];
            let (pu, pv) = surface.periodic();
            let mut uu: Vec<f64> = Vec::new();
            let mut vv: Vec<f64> = Vec::new();
            for &e in &face.edges {
                for &vi in &edges[e].vertices {
                    let Some((a, b)) = self.vertices.get(vi).and_then(|p| surface.invert(*p))
                    else {
                        continue;
                    };
                    uu.push(a);
                    vv.push(b);
                }
            }
            // The pole's `u` is meaningless — every `u` maps to the same
            // point — so only its `v` joins the extent.
            for p in poles.get(fi).into_iter().flatten() {
                if let Some((_, b)) = surface.invert(*p) {
                    vv.push(b);
                }
            }
            let (mut ur, uw) = extent(&uu, pu);
            let (vr, vw) = extent(&vv, pv);
            if pu && !uw {
                let taken = claimed.entry(face.surface).or_default();
                if taken
                    .iter()
                    .any(|t| (t.0 - ur.0).abs() < 1e-9 && (t.1 - ur.1).abs() < 1e-9)
                {
                    ur = (ur.1, ur.0 + std::f64::consts::TAU);
                }
                taken.push(ur);
            }
            face.u_range = ur;
            face.v_range = vr;
            face.u_wraps = uw;
            face.v_wraps = vw;
        }

        // A face that names one edge twice meets itself along a seam, and its
        // region is a band the parameter extent cannot state: both passes along
        // the seam invert to the same `u`, so taken as points they collapse and
        // the band has no area. Walking the bound *in order* and unwrapping `u`
        // against the point before puts the second pass a full turn from the
        // first, which is where it belongs.
        for (fi, face) in faces.iter_mut().enumerate() {
            let seam = (1..face.edges.len()).any(|i| face.edges[i..].contains(&face.edges[i - 1]));
            if !seam {
                continue;
            }
            let surface = surfaces[face.surface].clone();
            let (pu, _) = surface.periodic();
            let mut rings: Vec<TrimLoop> = Vec::new();
            for walk in walks.get(fi).into_iter().flatten() {
                let mut vs: Vec<usize> = Vec::new();
                for &(e, forward) in walk {
                    let Some(edge) = edges.get(e) else { continue };
                    let seq: Vec<usize> = if forward {
                        edge.vertices.clone()
                    } else {
                        edge.vertices.iter().rev().copied().collect()
                    };
                    for v in seq {
                        if vs.last() != Some(&v) {
                            vs.push(v);
                        }
                    }
                }
                if vs.len() > 1 && vs.first() == vs.last() {
                    vs.pop();
                }
                if vs.len() < 3 {
                    continue;
                }
                let mut uv: Vec<[f64; 2]> = Vec::with_capacity(vs.len());
                let mut anchor: Option<f64> = None;
                for &v in &vs {
                    let Some(p) = self.vertices.get(v).copied() else {
                        uv.clear();
                        break;
                    };
                    let Some((mut u, w)) = surface.invert(p) else {
                        uv.clear();
                        break;
                    };
                    if pu {
                        if let Some(a) = anchor {
                            u -= std::f64::consts::TAU * ((u - a) / std::f64::consts::TAU).round();
                        }
                        anchor = Some(u);
                    }
                    uv.push([u, w]);
                }
                if uv.len() != vs.len() {
                    continue;
                }
                let area = shoelace(&uv);
                rings.push(TrimLoop {
                    vertices: vs,
                    uv,
                    area,
                });
            }
            if rings.is_empty() {
                continue;
            }
            // The outer ring is the one enclosing the most; holes run the other
            // way, as the fill relies on.
            let widest = rings
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.area.abs().total_cmp(&b.1.area.abs()))
                .map(|(i, _)| i)
                .unwrap_or(0);
            let sign = rings[widest].area.signum();
            for (i, r) in rings.iter_mut().enumerate() {
                let want = if i == widest { sign } else { -sign };
                if r.area.signum() != want {
                    r.vertices.reverse();
                    r.uv.reverse();
                    r.area = -r.area;
                }
            }
            if sign < 0.0 {
                for r in rings.iter_mut() {
                    r.vertices.reverse();
                    r.uv.reverse();
                    r.area = -r.area;
                }
            }
            let us: Vec<f64> = rings
                .iter()
                .flat_map(|r| r.uv.iter().map(|p| p[0]))
                .collect();
            let vsv: Vec<f64> = rings
                .iter()
                .flat_map(|r| r.uv.iter().map(|p| p[1]))
                .collect();
            let mn = |x: &[f64]| x.iter().cloned().fold(f64::MAX, f64::min);
            let mx = |x: &[f64]| x.iter().cloned().fold(f64::MIN, f64::max);
            face.u_range = (mn(&us), mx(&us));
            face.v_range = (mn(&vsv), mx(&vsv));
            face.u_wraps = false;
            face.v_wraps = false;
            face.loops = Some(rings);
        }

        // A ring that does not go the whole way round in `u` encloses a disk
        // in the parameters, and that disk — not the box around it — is the
        // face. A circle sitting side-on to a sphere's axis is such a ring; the
        // extent pass above can only give it a rectangle, which covers ground
        // the face does not. Faces whose ring *does* wrap are the ordinary case
        // and keep their extent: a rim at constant `v` is a boundary of a band,
        // not of a disk.
        for (fi, face) in faces.iter_mut().enumerate() {
            if complements.get(fi).copied().unwrap_or(false)
                || face.edges.len() != 1
                || face.u_wraps
                || !matches!(surfaces[face.surface], Surface::Sphere { .. })
            {
                continue;
            }
            let surface = surfaces[face.surface].clone();
            let vids = edges[face.edges[0]].vertices.clone();
            let mut uv: Vec<[f64; 2]> = Vec::new();
            let mut vs: Vec<usize> = Vec::new();
            let mut ok = true;
            for vi in vids {
                let Some(p) = self.vertices.get(vi).copied() else {
                    ok = false;
                    break;
                };
                let Some((a, b)) = surface.invert(p) else {
                    ok = false;
                    break;
                };
                uv.push([a, b]);
                vs.push(vi);
            }
            if !ok || vs.len() < 3 {
                continue;
            }
            if vs.first() == vs.last() {
                vs.pop();
                uv.pop();
            }
            let area = shoelace(&uv);
            if area < 0.0 {
                vs.reverse();
                uv.reverse();
            }
            face.loops = Some(vec![TrimLoop {
                vertices: vs,
                uv,
                area: area.abs(),
            }]);
        }

        // A face that named bounds but no *outer* one is everything except what
        // those bounds enclose. Its region is not a parameter rectangle — a
        // sphere minus a side-on disk reaches every `u` and both poles — so it
        // cannot be stated by an extent at all, and the extent pass above has
        // just given it the disk's own tiny one. Rebuild it as the kernel's own
        // booleans state such a face: the surface's whole outline as the outer
        // ring, and each bound as a hole in it.
        for (fi, face) in faces.iter_mut().enumerate() {
            if !complements.get(fi).copied().unwrap_or(false) || face.edges.is_empty() {
                continue;
            }
            let surface = surfaces[face.surface].clone();
            if !matches!(surface, Surface::Sphere { .. }) {
                continue;
            }
            use std::f64::consts::{FRAC_PI_2, PI};
            let (u0, u1, v0, v1) = (-PI, PI, -FRAC_PI_2, FRAC_PI_2);

            let mut uv: Vec<[f64; 2]> = Vec::new();
            let side = |a: [f64; 2], b: [f64; 2], out: &mut Vec<[f64; 2]>| {
                for i in 0..48 {
                    let t = f64::from(i) / 48.0;
                    out.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
                }
            };
            side([u0, v0], [u1, v0], &mut uv);
            side([u1, v0], [u1, v1], &mut uv);
            side([u1, v1], [u0, v1], &mut uv);
            side([u0, v1], [u0, v0], &mut uv);
            // A pole is one point however many parameters name it. Left in, the
            // duplicates are zero-length edges the ear clip may make anything of.
            let same = |a: &[f64; 2], b: &[f64; 2]| {
                v3::dist(surface.point(a[0], a[1]), surface.point(b[0], b[1])) <= tolerance * 0.25
            };
            uv.dedup_by(|a, b| same(a, b));
            while uv.len() > 2 && same(&uv[0], &uv[uv.len() - 1]) {
                uv.pop();
            }
            let pts: Vec<V3> = uv.iter().map(|p| surface.point(p[0], p[1])).collect();
            let vs: Vec<usize> = pts.into_iter().map(|p| self.vertex(p)).collect();
            let area = shoelace(&uv);
            let mut rings = vec![TrimLoop {
                vertices: vs,
                uv,
                area,
            }];

            let mut ok = true;
            for &e in &face.edges {
                let vids = edges[e].vertices.clone();
                let mut huv: Vec<[f64; 2]> = Vec::new();
                let mut hvs: Vec<usize> = Vec::new();
                for vi in vids {
                    let Some(p) = self.vertices.get(vi).copied() else {
                        ok = false;
                        break;
                    };
                    let Some((a, b)) = surface.invert(p) else {
                        ok = false;
                        break;
                    };
                    huv.push([a, b]);
                    hvs.push(vi);
                }
                if !ok {
                    break;
                }
                if hvs.len() > 1 && hvs.first() == hvs.last() {
                    hvs.pop();
                    huv.pop();
                }
                let a = shoelace(&huv);
                // A hole runs against the outer ring.
                if (a > 0.0) == (area > 0.0) {
                    hvs.reverse();
                    huv.reverse();
                }
                rings.push(TrimLoop {
                    vertices: hvs,
                    uv: huv,
                    area: -a.abs() * area.signum(),
                });
            }
            if !ok {
                continue;
            }
            face.u_range = (u0, u1);
            face.v_range = (v0, v1);
            face.u_wraps = true;
            face.v_wraps = false;
            face.loops = Some(rings);
        }

        // Give each edge its second surface, so the pair is available to the
        // kernel the way an authored body's would be.
        for face in faces.iter() {
            for &e in &face.edges {
                if edges[e].surfaces.0 == edges[e].surfaces.1 && edges[e].surfaces.0 != face.surface
                {
                    edges[e].surfaces.1 = face.surface;
                }
            }
        }

        let report = ImportReport {
            faces: faces.len(),
            edges: edges.len(),
            vertices: self.vertices.len(),
            skipped: {
                let mut v: Vec<(String, usize)> = self.skipped.into_iter().collect();
                v.sort();
                v
            },
        };
        // `from_parts` derives each face's parameter range from the vertices its
        // edges reach, which is exactly what the file leaves implicit.
        //
        // Which cannot say that a face has *holes*, and that is what stops a
        // rod crossed by a rod surviving a round trip: it goes out as an outline
        // with two holes where the other rod passes and comes back a whole
        // cylinder, open by 418 edges.
        //
        // The obvious repair is wrong. `face_loops` will chain this face's edges
        // into rings, and asking it whenever a face has more than one ring turns
        // an *annulus* inside out — a cylinder between two rims has two rings and
        // neither encloses the other, so the smaller is taken for a hole and the
        // solid collapses: a ball with a bore came back 19.9 against 132.3, with
        // nothing reported. A ring is a hole when it is *inside* another, which
        // is a containment question and not an area comparison.
        //
        // Asking that — outer ring by area, holes only where every point of a
        // ring lies inside it, and the face left alone otherwise — is green
        // everywhere and does not fix the rod. That wall is **both** an annulus
        // and holed: its four edges chain into two rims, neither inside the
        // other, plus the two rings the crossing rod cut. The nesting test
        // rightly declines, and the face comes back without its holes anyway.
        //
        // Its region is the rectangle *between* the rims with two holes in it,
        // and no chaining of edges will produce that outline — the kernel
        // synthesises it (`parameter_outline`) rather than deriving it. The
        // importer would have to do the same — but only where the outer
        // boundary really is not derivable.
        //
        // Synthesising it wherever a face has an interior ring is wrong and
        // costs three STEP tests: a plate with a bore has an outer ring that
        // touches its extent and a bore ring inside, and its extent is only the
        // bounding box of its edges — so a rectangle replaces a boundary that
        // was already right, and the plate grows to its bounding box.
        //
        // What separates the two is whether the non-interior ring *encloses* the
        // interior ones. A plate's outer ring does; a band's rims do not, being
        // lines across the rectangle rather than loops around it. So: enclose,
        // and the edges already say the boundary; do not, and it has to be
        // synthesised.
        //
        // Both rules together, chosen that way, are green on their own — and
        // with the kernel's narrower trace box they leave the rod *worse*: 839
        // open edges against 418 for no import rules at all. So the synthesised
        // rectangle is not what that wall's region is either. Its rims run round
        // a cylinder, so in the parameters they are two lines across a rectangle
        // whose `u` sides are the same seam — and a plain four-sided outline
        // says nothing about that. The kernel's `parameter_outline` reuses the
        // rims themselves for two of its sides for exactly this reason.
        //
        // Except that nothing needs synthesising. Measured on that wall, with
        // the kernel's narrower box:
        //
        //     F0 cylinder edges=4  rings (pts, off-edge) [(34, 0), (146, 0), (147, 0)]
        //
        // Every ring is made **entirely of edges**, the 34-point outline
        // included — `materialise_seams` gave the seam an edge, so the outline
        // has one too. The export writes a face whose boundary really is its
        // edges, and `AmbiguousRegion` rightly does not fire. So what fails is
        // re-chaining those edges here, and the difference is visible:
        //
        //     OUT  e0 147v  e1 148v  e2 18v  e3 18v
        //          rings [(34, outer), (146, hole), (147, hole)]
        //     BACK e0 101v  e1 101v  e2 148v  e3 147v
        //          rings [(100, ..), (100, ..), (147, outer), (146, ..)]
        //
        // The original chains its two 18-point rims into one 34-point outer.
        // Read back, the rims arrive as 101-point circles and stay *two* rings,
        // so there are four rings and no outer — and the largest by area is a
        // traced hole. Any rule that takes the largest ring for the outer picks
        // a hole on this body, which is why every one tried so far has failed.
        //
        // Not because the rims meet: `face_loops` chains *segments* by shared
        // vertices, and measured, neither pair shares any —
        //
        //     OUT  rims 16 and 16 vertices, share 0
        //     BACK rims 100 and 100 vertices, share 0
        //
        // — so the kernel's 34-point outer is not its two rims joined, and what
        // makes one ring there and two here has a simpler answer than any of
        // them: `face_loops` returns a face's **stored** loops when it has any,
        // and only derives when it does not. Chaining both bodies' segments by
        // hand gives *four* rings each —
        //
        //     OUT  raw rings [147, 148, 17, 17]   face_loops [34, 146, 147]
        //     BACK raw rings [101, 101, 148, 147] face_loops [100, 100, 147, 146]
        //
        // — so the kernel's 34-point outer was never chained from its edges. It
        // is the outline the boolean stored, which runs rim, seam, rim, seam.
        // Its vertices are edge vertices, which is why nothing reads as
        // off-edge, but its *order* is one no chaining of those edges produces.
        //
        // So an imported face cannot recover that boundary by deriving, however
        // the rings are classified. It has to be given one.
        Ok((
            Body::from_parts(surfaces, self.vertices, edges, faces),
            report,
        ))
    }
}

/// The extent a face occupies in one parameter direction, and whether it wraps.
///
/// A STEP face carries no parameter range — its extent *is* its boundary — so
/// this is where that implicit fact is made explicit. Two cases have to be told
/// apart, and min/max alone cannot:
///
/// * A rim circle whose samples run all the way round. Its min and max fall one
///   sample short of the period, and using them leaves the face open by that
///   sliver.
/// * A face that is only *part* of a periodic surface, such as one half of a
///   seam-split sphere. Its samples cover half the circle, and forcing the full
///   period there makes each half claim the whole sphere.
///
/// The largest gap between consecutive samples separates them: a full wrap has
/// no gap, and a partial face has exactly one large one.
/// Signed area of a closed polygon in parameter space.
fn shoelace(uv: &[[f64; 2]]) -> f64 {
    let n = uv.len();
    if n < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        a += uv[i][0] * uv[j][1] - uv[j][0] * uv[i][1];
    }
    a * 0.5
}

fn extent(values: &[f64], periodic: bool) -> ((f64, f64), bool) {
    if values.is_empty() {
        return ((0.0, 0.0), false);
    }
    let lo = values.iter().cloned().fold(f64::MAX, f64::min);
    let hi = values.iter().cloned().fold(f64::MIN, f64::max);
    if !periodic {
        return ((lo, hi), false);
    }
    use std::f64::consts::{PI, TAU};
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut gap: f64 = sorted[0] - (sorted[sorted.len() - 1] - TAU);
    for w in sorted.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    if gap < TAU / 8.0 {
        ((-PI, PI), true)
    } else {
        ((lo, hi), false)
    }
}

/// Segments needed for an arc to stay within `tolerance` of its chord.
fn segments(radius: f64, sweep: f64, tolerance: f64) -> usize {
    if !radius.is_finite() || radius <= 0.0 || sweep <= 0.0 {
        return 1;
    }
    let per = 2.0 * (1.0 - (tolerance / radius).min(1.0)).acos().max(1e-6);
    ((sweep / per).ceil() as usize).clamp(4, 4096)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::export::export;

    /// Volume of a body's tessellation, by the divergence theorem — the check
    /// that a round trip preserved the *shape*, not merely the entity counts.
    fn volume(body: &Body, tolerance: f64) -> f64 {
        let mut b = body.clone();
        b.refine_edges(tolerance);
        let (mesh, _) = b.tessellate(tolerance);
        let pos = &mesh.get_attribute("position").unwrap().array;
        let Some(idx) = mesh.index.as_ref() else {
            return 0.0;
        };
        let v = |i: u32| -> V3 {
            let o = i as usize * 3;
            [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
        };
        idx.chunks_exact(3)
            .map(|t| {
                let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
                v3::dot(a, v3::cross(b, c)) / 6.0
            })
            .sum::<f64>()
            .abs()
    }

    fn round_trip(body: &Body, tolerance: f64) -> (Body, ImportReport) {
        let (text, ex) = export(body, "part", tolerance);
        assert!(ex.skipped.is_empty(), "export dropped {:?}", ex.skipped);
        let (back, im) = import(&text, tolerance).expect("the file we just wrote");
        assert!(im.skipped.is_empty(), "import dropped {:?}", im.skipped);
        (back, im)
    }

    #[test]
    fn a_cuboid_survives_a_round_trip() {
        let body = Body::cuboid([2.0, 3.0, 4.0]);
        let (back, report) = round_trip(&body, 1e-6);
        assert_eq!(report.faces, 6);
        assert_eq!(report.edges, 12);
        assert_eq!(report.vertices, 8, "the corners must weld back together");
        assert!(
            (volume(&back, 1e-3) - 24.0).abs() < 1e-6,
            "{}",
            volume(&back, 1e-3)
        );
    }

    #[test]
    fn a_cylinder_comes_back_as_a_cylinder() {
        // Not as a mesh of one: the surface must survive, or the format has
        // bought us nothing over STL.
        let body = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0);
        let (back, _) = round_trip(&body, 1e-4);
        assert_eq!(
            back.surfaces()
                .iter()
                .filter(|s| s.kind() == "cylinder")
                .count(),
            1
        );
        assert_eq!(
            back.surfaces()
                .iter()
                .filter(|s| s.kind() == "plane")
                .count(),
            2
        );

        let expected = std::f64::consts::PI * 4.0 * 5.0;
        let got = volume(&back, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.01,
            "volume {got}, expected {expected}"
        );
    }

    #[test]
    fn each_analytic_surface_round_trips_to_itself() {
        for (body, kind) in [
            (Body::cuboid([1.0, 1.0, 1.0]), "plane"),
            (
                Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0),
                "cylinder",
            ),
            (Body::sphere([0.0; 3], 1.5), "sphere"),
            (Body::cone([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0), "cone"),
            (Body::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0), "torus"),
        ] {
            let (text, ex) = export(&body, "s", 1e-4);
            assert!(ex.skipped.is_empty(), "{kind}: {:?}", ex.skipped);
            let (back, im) = import(&text, 1e-4).unwrap();
            assert!(im.skipped.is_empty(), "{kind}: {:?}", im.skipped);
            assert!(
                back.surfaces().iter().any(|s| s.kind() == kind),
                "{kind} did not come back; got {:?}",
                back.surfaces().iter().map(|s| s.kind()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn a_cone_recovers_its_apex_from_a_reference_circle() {
        // AP203 places a cone at a circle of some radius, not at its apex. A
        // reader that takes the placement *as* the apex puts the cone in the
        // wrong place — silently, since the axis and angle still look right.
        let text = "\
ISO-10303-21;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,4.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=CONICAL_SURFACE('',#4,2.,0.7853981633974483);
#6=CLOSED_SHELL('',(#7));
#7=ADVANCED_FACE('',(),#5,.T.);
ENDSEC;
END-ISO-10303-21;
";
        let (body, _) = import(text, 1e-6).unwrap();
        let Surface::Cone { apex, .. } = &body.surfaces()[0] else {
            panic!("not a cone: {:?}", body.surfaces()[0].kind());
        };
        // radius 2 at a 45° half-angle is 2 above the apex, so z = 4 - 2 = 2.
        assert!(v3::dist(*apex, [0.0, 0.0, 2.0]) < 1e-9, "{apex:?}");
    }

    #[test]
    fn a_shared_edge_is_read_as_one_edge() {
        // Two faces naming the same `EDGE_CURVE` must produce a single `Edge`.
        // Reading it twice is how an imported solid ends up open along a seam
        // that the file said was closed.
        let (text, _) = export(&Body::cuboid([1.0, 1.0, 1.0]), "unit", 1e-6);
        let (body, report) = import(&text, 1e-6).unwrap();
        assert_eq!(report.edges, 12, "a box has twelve, not twenty-four");
        for (i, _) in body.edges().iter().enumerate() {
            let users = body.faces().iter().filter(|f| f.edges.contains(&i)).count();
            assert_eq!(users, 2, "edge {i} used by {users} faces");
        }
    }

    #[test]
    fn a_boolean_result_survives_the_round_trip() {
        use crate::brep::BooleanOp;
        let plate = Body::cuboid([10.0, 8.0, 2.0]);
        let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0);
        let cut = plate.boolean(&drill, BooleanOp::Difference, 1e-3).unwrap();

        let (back, report) = round_trip(&cut, 1e-3);
        assert_eq!(report.faces, 7);
        assert_eq!(
            back.surfaces()
                .iter()
                .filter(|s| s.kind() == "cylinder")
                .count(),
            1,
            "the bore is still a cylinder"
        );
        let expected = 160.0 - std::f64::consts::PI * 4.0 * 2.0;
        let got = volume(&back, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.02,
            "volume {got}, expected {expected}"
        );
    }

    #[test]
    fn a_file_with_no_shell_is_refused_rather_than_returning_an_empty_solid() {
        let e = import("ISO-10303-21;\nDATA;\n#1=PLANE('',$);\nENDSEC;\n", 1e-6).unwrap_err();
        assert_eq!(e, ImportError::NoSolid);
    }

    #[test]
    fn a_surface_with_no_mapping_is_reported_not_dropped_silently() {
        let text = "\
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=AXIS2_PLACEMENT_3D('',#1,$,$);
#3=SURFACE_OF_LINEAR_EXTRUSION('',#2,#2);
#4=CLOSED_SHELL('',(#5));
#5=ADVANCED_FACE('',(),#3,.T.);
ENDSEC;
";
        let (body, report) = import(text, 1e-6).unwrap();
        assert_eq!(body.faces().len(), 0);
        assert_eq!(
            report.skipped,
            vec![("SURFACE_OF_LINEAR_EXTRUSION".to_string(), 1)]
        );
        assert!(!report.is_complete());
    }

    #[test]
    fn malformed_input_reports_where_it_broke() {
        let e = import("DATA;#1=A(;ENDSEC;", 1e-6).unwrap_err();
        assert!(matches!(e, ImportError::Syntax(_)), "{e}");
    }

    #[test]
    fn an_arc_is_sampled_finely_enough_to_meet_the_tolerance() {
        // The sagitta of one segment must stay under the tolerance, or a bore
        // imports visibly faceted.
        for (r, tol) in [(1.0, 1e-3), (100.0, 1e-3), (0.5, 1e-5)] {
            let n = segments(r, std::f64::consts::TAU, tol);
            let half = std::f64::consts::PI / n as f64;
            let sagitta = r * (1.0 - half.cos());
            assert!(sagitta <= tol * 1.001, "r={r} tol={tol} n={n} → {sagitta}");
        }
    }
}
