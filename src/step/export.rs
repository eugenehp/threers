//! `Body` → AP203/AP214 advanced B-rep.
//!
//! The mapping is deliberately literal: a [`Surface::Cylinder`] becomes a
//! `CYLINDRICAL_SURFACE`, not a mesh of it, and a circular seam becomes a
//! `CIRCLE`, not a polyline through its sample points. That is the whole reason
//! to have a B-rep — a bore that arrives in a CAD system as a cylinder can be
//! re-dimensioned, and one that arrives as 200 triangles cannot.
//!
//! Edge geometry is recovered from the edge's own vertices rather than trusted
//! from elsewhere: points that lie on a common line become a `LINE`, points on a
//! common circle become a `CIRCLE`, and anything else becomes a `POLYLINE`.
//! Recovering it means an edge exports exactly whatever it actually is, whether
//! it came from an authored primitive, a boolean seam, or a file.

use std::collections::{HashMap, HashSet};

use crate::brep::{Body, Surface};
use crate::nurbs::{v3, V3};

use super::part21::{Entity, StepFile, Value};

/// Something the mapping could not carry.
///
/// Reported rather than approximated: a caller that gets an empty list knows the
/// file is the whole solid, and one that does not knows exactly what is missing.
#[derive(Debug, Clone, PartialEq)]
pub enum Unsupported {
    /// A surface with no AP203 counterpart.
    Surface { face: usize, kind: &'static str },
    /// A face whose bounding edges do not chain into closed loops, so it has no
    /// well-defined boundary to write.
    OpenFaceBoundary { face: usize },
    /// A face with no edges at all, on a surface whose seam could not be
    /// synthesised — there is nothing to bound it with.
    UnboundedFace { face: usize },
    /// A face bounded partly by its own seam — an edge it uses twice.
    ///
    /// The file could state this: an `EDGE_LOOP` naming one `EDGE_CURVE` in
    /// both directions is ordinary practice, and the reader here now handles
    /// it, walking each bound in order and unwrapping `u` so the two passes
    /// along the seam land at opposite ends of the parameter domain rather than
    /// on top of each other. What is missing is on the writing side: rings are
    /// built by chaining the face's edges into closed loops independently, so a
    /// boundary that walks two holes *and* the seam between them comes out as
    /// three loops — the two holes, and the seam as a loop of no area. The
    /// rings have to come from the face's stored trim loops, in their order,
    /// and until they do this is not written.
    SeamBoundary { face: usize },
    /// A face whose trim loops name points none of its edges carry, so the
    /// edge loops written here would not describe the same region.
    ///
    /// A closed curve on a closed surface bounds *two* regions and does not say
    /// which of them the face is: one circle on a sphere is the boundary of the
    /// cap and equally of everything else. Where the two are told apart by a
    /// trim loop rather than by the edges, an AP203 face built from those edges
    /// is ambiguous — and the receiving system is free to read the other side.
    AmbiguousRegion { face: usize },
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::Surface { face, kind } => {
                write!(f, "face {face}: no AP203 entity for a {kind} surface")
            }
            Unsupported::OpenFaceBoundary { face } => {
                write!(f, "face {face}: bounding edges do not close")
            }
            Unsupported::UnboundedFace { face } => {
                write!(f, "face {face}: no edges and no synthesisable seam")
            }
            Unsupported::SeamBoundary { face } => {
                write!(f, "face {face}: bounded partly by its own seam")
            }
            Unsupported::AmbiguousRegion { face } => {
                write!(
                    f,
                    "face {face}: its edges bound two regions and do not say which"
                )
            }
        }
    }
}

/// What the export carried, and what it did not.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ExportReport {
    pub faces: usize,
    pub edges: usize,
    pub vertices: usize,
    /// Faces written with an exact analytic surface.
    pub analytic_faces: usize,
    /// Edges written as a `LINE` or `CIRCLE` rather than a `POLYLINE`.
    pub analytic_edges: usize,
    pub skipped: Vec<Unsupported>,
}

impl ExportReport {
    /// Whether every face and every edge was carried exactly.
    pub fn is_exact(&self) -> bool {
        self.skipped.is_empty()
            && self.analytic_faces == self.faces
            && self.analytic_edges == self.edges
    }
}

/// Write a solid as an ISO 10303-21 exchange file.
///
/// `name` names the product. `tolerance` is the fit tolerance for deciding that
/// an edge's vertices lie on a line or a circle — the same number used to
/// tessellate is the right one to pass.
pub fn export(body: &Body, name: &str, tolerance: f64) -> (String, ExportReport) {
    let mut w = Writer::new(tolerance);
    let report = w.body(body, name);
    (w.finish(name).to_string(), report)
}

// --------------------------------------------------------------------------

struct Writer {
    next: u64,
    data: Vec<Entity>,
    tolerance: f64,
    /// Deduplicated by quantised coordinates: a box has 8 corners, not 24, and a
    /// reader that welds by identity should see the same topology we meant.
    points: HashMap<(i64, i64, i64), u64>,
    directions: HashMap<(i64, i64, i64), u64>,
    /// The `CLOSED_SHELL` members.
    face_ids: Vec<u64>,
}

impl Writer {
    fn new(tolerance: f64) -> Self {
        Writer {
            next: 0,
            data: Vec::new(),
            tolerance: tolerance.max(1e-12),
            points: HashMap::new(),
            directions: HashMap::new(),
            face_ids: Vec::new(),
        }
    }

    fn add(&mut self, name: &str, args: Vec<Value>) -> u64 {
        self.next += 1;
        self.data.push(Entity {
            id: self.next,
            name: name.to_string(),
            args,
        });
        self.next
    }

    fn complex(&mut self, parts: Vec<Value>) -> u64 {
        self.next += 1;
        self.data.push(Entity {
            id: self.next,
            name: String::new(),
            args: parts,
        });
        self.next
    }

    fn key(&self, p: V3) -> (i64, i64, i64) {
        let q = |x: f64| (x / (self.tolerance * 0.5)).round() as i64;
        (q(p[0]), q(p[1]), q(p[2]))
    }

    fn point(&mut self, p: V3) -> u64 {
        if let Some(id) = self.points.get(&self.key(p)) {
            return *id;
        }
        let id = self.add(
            "CARTESIAN_POINT",
            vec![
                Value::Text(String::new()),
                Value::List(p.iter().map(|x| Value::Real(*x)).collect()),
            ],
        );
        self.points.insert(self.key(p), id);
        id
    }

    fn direction(&mut self, d: V3) -> u64 {
        let d = v3::normalize(d).unwrap_or([0.0, 0.0, 1.0]);
        if let Some(id) = self.directions.get(&self.key(d)) {
            return *id;
        }
        let id = self.add(
            "DIRECTION",
            vec![
                Value::Text(String::new()),
                Value::List(d.iter().map(|x| Value::Real(*x)).collect()),
            ],
        );
        self.directions.insert(self.key(d), id);
        id
    }

    /// `AXIS2_PLACEMENT_3D` — an origin, a z axis, and a reference x.
    fn placement(&mut self, origin: V3, axis: V3, x_dir: V3) -> u64 {
        let o = self.point(origin);
        let z = self.direction(axis);
        let x = self.direction(orthogonalise(x_dir, axis));
        self.add(
            "AXIS2_PLACEMENT_3D",
            vec![
                Value::Text(String::new()),
                Value::Ref(o),
                Value::Ref(z),
                Value::Ref(x),
            ],
        )
    }

    fn surface(&mut self, s: &Surface) -> Option<u64> {
        let id = match s {
            Surface::Plane {
                origin,
                normal,
                x_dir,
            } => {
                let ax = self.placement(*origin, *normal, *x_dir);
                self.add("PLANE", vec![Value::Text(String::new()), Value::Ref(ax)])
            }
            Surface::Cylinder {
                origin,
                axis,
                x_dir,
                radius,
            } => {
                let ax = self.placement(*origin, *axis, *x_dir);
                self.add(
                    "CYLINDRICAL_SURFACE",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(ax),
                        Value::Real(*radius),
                    ],
                )
            }
            Surface::Sphere {
                center,
                axis,
                x_dir,
                radius,
            } => {
                let ax = self.placement(*center, *axis, *x_dir);
                self.add(
                    "SPHERICAL_SURFACE",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(ax),
                        Value::Real(*radius),
                    ],
                )
            }
            Surface::Cone {
                apex,
                axis,
                x_dir,
                half_angle,
            } => {
                // AP203 places a cone by a *reference circle*, not by its apex:
                // the placement sits where the radius is `radius`. Putting the
                // placement at the apex makes that radius zero, which is legal
                // and unambiguous, and is what round-trips.
                let ax = self.placement(*apex, *axis, *x_dir);
                self.add(
                    "CONICAL_SURFACE",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(ax),
                        Value::Real(0.0),
                        Value::Real(*half_angle),
                    ],
                )
            }
            Surface::Torus {
                center,
                axis,
                x_dir,
                major,
                minor,
            } => {
                let ax = self.placement(*center, *axis, *x_dir);
                self.add(
                    "TOROIDAL_SURFACE",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(ax),
                        Value::Real(*major),
                        Value::Real(*minor),
                    ],
                )
            }
            Surface::Nurbs(n) => self.nurbs_surface(n)?,
        };
        Some(id)
    }

    /// `B_SPLINE_SURFACE_WITH_KNOTS`, wrapped in the rational complex instance
    /// when any weight differs from 1.
    fn nurbs_surface(&mut self, n: &crate::nurbs::NurbsSurface) -> Option<u64> {
        let (nu, nv) = (n.n_u(), n.n_v());
        let mut rows = Vec::with_capacity(nu);
        let mut weights = Vec::with_capacity(nu);
        let mut rational = false;
        for i in 0..nu {
            let mut row = Vec::with_capacity(nv);
            let mut wrow = Vec::with_capacity(nv);
            for j in 0..nv {
                let (p, w) = (n.control_point(i, j), n.weight(i, j));
                let id = self.point(p);
                row.push(Value::Ref(id));
                if (w - 1.0).abs() > 1e-12 {
                    rational = true;
                }
                wrow.push(Value::Real(w));
            }
            rows.push(Value::List(row));
            weights.push(Value::List(wrow));
        }

        let (uk, um) = compress(n.knots_u());
        let (vk, vm) = compress(n.knots_v());
        let spline = vec![
            Value::Text(String::new()),
            Value::Int(n.degree_u() as i64),
            Value::Int(n.degree_v() as i64),
            Value::List(rows),
            Value::Enum("UNSPECIFIED".into()),
            Value::Enum("F".into()),
            Value::Enum("F".into()),
            Value::Enum("F".into()),
            Value::List(um.iter().map(|m| Value::Int(*m as i64)).collect()),
            Value::List(vm.iter().map(|m| Value::Int(*m as i64)).collect()),
            Value::List(uk.iter().map(|k| Value::Real(*k)).collect()),
            Value::List(vk.iter().map(|k| Value::Real(*k)).collect()),
            Value::Enum("UNSPECIFIED".into()),
        ];

        Some(if rational {
            // The rational form has no entity of its own: it is the intersection
            // of several, written as a complex instance.
            let mut with_knots = spline.clone();
            with_knots.remove(0); // the name lives on B_SPLINE_SURFACE only
            self.complex(vec![
                Value::Typed("BOUNDED_SURFACE".into(), vec![]),
                Value::Typed("B_SPLINE_SURFACE".into(), spline[1..8].to_vec()),
                Value::Typed("B_SPLINE_SURFACE_WITH_KNOTS".into(), spline[8..].to_vec()),
                Value::Typed("GEOMETRIC_REPRESENTATION_ITEM".into(), vec![]),
                Value::Typed(
                    "RATIONAL_B_SPLINE_SURFACE".into(),
                    vec![Value::List(weights)],
                ),
                Value::Typed(
                    "REPRESENTATION_ITEM".into(),
                    vec![Value::Text(String::new())],
                ),
                Value::Typed("SURFACE".into(), vec![]),
            ])
        } else {
            self.add("B_SPLINE_SURFACE_WITH_KNOTS", spline)
        })
    }

    /// The geometry of one edge, from its own vertices.
    ///
    /// Returns the entity and whether it is analytic, so the report can say how
    /// much of the solid survived exactly.
    fn curve(&mut self, pts: &[V3], closed: bool) -> (u64, bool) {
        if let Some((origin, dir)) = fit_line(pts, self.tolerance) {
            let d = self.direction(dir);
            let o = self.point(origin);
            let vec = self.add(
                "VECTOR",
                vec![Value::Text(String::new()), Value::Ref(d), Value::Real(1.0)],
            );
            let id = self.add(
                "LINE",
                vec![Value::Text(String::new()), Value::Ref(o), Value::Ref(vec)],
            );
            return (id, true);
        }
        if let Some(c) = fit_circle(pts, closed, self.tolerance) {
            let ax = self.placement(c.center, c.axis, c.x_dir);
            let id = self.add(
                "CIRCLE",
                vec![
                    Value::Text(String::new()),
                    Value::Ref(ax),
                    Value::Real(c.radius),
                ],
            );
            return (id, true);
        }
        // Faithful to what the edge actually is, and honest that it is sampled.
        let ids: Vec<Value> = pts.iter().map(|p| Value::Ref(self.point(*p))).collect();
        let id = self.add(
            "POLYLINE",
            vec![Value::Text(String::new()), Value::List(ids)],
        );
        (id, false)
    }

    fn body(&mut self, body: &Body, _name: &str) -> ExportReport {
        let mut report = ExportReport {
            vertices: body.vertices().len(),
            ..Default::default()
        };

        // Vertices, shared by index so the topology survives.
        let vertex_ids: Vec<u64> = body
            .vertices()
            .iter()
            .map(|p| {
                let pt = self.point(*p);
                self.add(
                    "VERTEX_POINT",
                    vec![Value::Text(String::new()), Value::Ref(pt)],
                )
            })
            .collect();

        // Edges, shared by index for the same reason: two faces naming one
        // `EDGE_CURVE` is what tells a reader the solid is closed there.
        let mut edge_ids: Vec<Option<u64>> = Vec::with_capacity(body.edges().len());
        for e in body.edges() {
            let pts: Vec<V3> = e
                .vertices
                .iter()
                .filter_map(|&v| body.vertices().get(v).copied())
                .collect();
            if pts.len() < 2 {
                edge_ids.push(None);
                continue;
            }
            let (geom, analytic) = self.curve(&pts, e.closed);
            report.edges += 1;
            if analytic {
                report.analytic_edges += 1;
            }
            let a = vertex_ids[e.vertices[0]];
            let b = vertex_ids[*e.vertices.last().unwrap()];
            let id = self.add(
                "EDGE_CURVE",
                vec![
                    Value::Text(String::new()),
                    Value::Ref(a),
                    Value::Ref(b),
                    Value::Ref(geom),
                    Value::Enum("T".into()),
                ],
            );
            edge_ids.push(Some(id));
        }

        for (fi, face) in body.faces().iter().enumerate() {
            let surface = &body.surfaces()[face.surface];
            let Some(surf_id) = self.surface(surface) else {
                report.skipped.push(Unsupported::Surface {
                    face: fi,
                    kind: surface.kind(),
                });
                continue;
            };

            // The region a face occupies is not always determined by the
            // edges around it, and AP203 writes the edges. Where the face says
            // so itself — a trim loop naming points no edge of this face
            // carries — writing those edges would state a boundary two regions
            // share and leave the reader to pick one. It picked the empty one:
            // a ball differenced with a ball came back as a solid of no volume,
            // with nothing reported at either end.
            let mut complement = false;
            if let Some(loops) = face.loops.as_ref().filter(|l| !l.is_empty()) {
                let named: HashSet<usize> = face
                    .edges
                    .iter()
                    .flat_map(|&e| body.edges()[e].vertices.iter().copied())
                    .collect();
                let off = loops
                    .iter()
                    .filter(|l| l.vertices.iter().any(|v| !named.contains(v)))
                    .count();
                // One ring off the edges and the rest on them is the shape of a
                // face that is *everything except* what its edges enclose: the
                // off-edge ring is the parameter outline, which is not a
                // boundary of the solid at all, and the on-edge rings are the
                // real ones, as holes. AP203 says exactly that by giving the
                // face no `FACE_OUTER_BOUND` — a bound that is not the outer
                // one is a hole, and the face runs past it to the far side of
                // the surface.
                //
                // Only on a sphere for now, because the reader has to rebuild
                // the region from the surface's own extent and that extent is
                // hardcoded there; anything else still declines.
                if off == 1 && loops.len() > 1 && matches!(surface, Surface::Sphere { .. }) {
                    complement = true;
                } else if off > 0 {
                    report
                        .skipped
                        .push(Unsupported::AmbiguousRegion { face: fi });
                    continue;
                }
            }

            if face.edges.is_empty() {
                match self.seam_faces(fi, face, surface, surf_id) {
                    Ok(ids) => {
                        report.faces += ids.len();
                        if !matches!(surface, Surface::Nurbs(_)) {
                            report.analytic_faces += ids.len();
                        }
                        // The seam edges are counted as analytic because they
                        // are: an iso-curve of a quadric is a circle or a line.
                        report.edges += ids.len();
                        report.analytic_edges += ids.len();
                        self.face_ids.extend(ids);
                    }
                    Err(u) => report.skipped.push(u),
                }
                continue;
            }

            let rings = match self.face_rings(body, fi, face, surface, &edge_ids, &vertex_ids) {
                Ok(r) => r,
                Err(u) => {
                    report.skipped.push(u);
                    continue;
                }
            };

            // A pole is a boundary no edge can state: at a sphere's axis or a
            // cone's apex the whole iso-curve collapses to one point, so there
            // is no curve to be an edge.
            //
            // Safe here only because of what has already been turned away. A
            // pole at the end of `v_range` is a *3D* boundary when the face
            // stops there, and merely a corner of the parameter rectangle when
            // the face runs past it — a sphere minus a side-on cap reaches
            // `v = ±π/2` at both ends and contains both poles in its interior.
            // Stating those as bounds would invent two boundaries the solid does
            // not have. Every such face is an `AmbiguousRegion` above, so what
            // reaches here always has a rectangle for a region and a pole it
            // really does end at. If that decline is ever lifted, this needs the
            // trim loop to say which of the two it is; the ranges cannot. AP203 has `VERTEX_LOOP` for exactly
            // this — a bound holding a single vertex — and without it the
            // reader has only the face's other rim to work from, which says
            // where the face *ends* and nothing about where it begins. A
            // spherical cap came back with `v` running from its rim to its rim:
            // zero extent, no triangles, and the volume quietly gone.
            let (pv0, pv1) = face.v_range;
            let (pu0, pu1) = face.u_range;
            let mut poles = Vec::new();
            // Only where the face's region *is* the rectangle its ranges
            // describe, which is exactly when it states no loops of its own. A
            // face that states loops is bounded by them, and its `v_range` ends
            // are corners of the parameter domain rather than places the solid
            // stops — a cap sitting side-on to the axis has a stale full-sphere
            // range and two ends that look degenerate, and is bounded by neither
            // pole. Written otherwise it gains two boundaries it does not have.
            let ends = [pv0, pv1];
            let ends: &[f64] = if complement || face.loops.is_some() {
                &[]
            } else {
                &ends
            };
            for &v in ends {
                let at = surface.point(pu0, v);
                if v3::dist(at, surface.point(0.5 * (pu0 + pu1), v)) > self.tolerance * 1e-3 {
                    continue;
                }
                let pid = self.point(at);
                let vid = self.add(
                    "VERTEX_POINT",
                    vec![Value::Text(String::new()), Value::Ref(pid)],
                );
                poles.push(self.add(
                    "VERTEX_LOOP",
                    vec![Value::Text(String::new()), Value::Ref(vid)],
                ));
            }

            // Exactly one bound is the outer one. On a face that is an annulus —
            // a cylinder between two rims — neither ring encloses the other, and
            // the choice is arbitrary but must still be made; longest wins.
            let mut bounds = Vec::with_capacity(rings.len());
            for (i, (loop_id, _)) in rings.iter().enumerate() {
                bounds.push(Value::Ref(self.add(
                    if i == 0 && !complement {
                        "FACE_OUTER_BOUND"
                    } else {
                        "FACE_BOUND"
                    },
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(*loop_id),
                        Value::Enum("T".into()),
                    ],
                )));
            }
            // Never the outer bound: a point encloses nothing.
            for loop_id in poles {
                bounds.push(Value::Ref(self.add(
                    "FACE_BOUND",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(loop_id),
                        Value::Enum("T".into()),
                    ],
                )));
            }

            let id = self.add(
                "ADVANCED_FACE",
                vec![
                    Value::Text(String::new()),
                    Value::List(bounds),
                    Value::Ref(surf_id),
                    Value::Enum(if face.flipped { "F" } else { "T" }.into()),
                ],
            );
            self.face_ids.push(id);
            report.faces += 1;
            if !matches!(surface, Surface::Nurbs(_)) {
                report.analytic_faces += 1;
            }
        }
        report
    }

    /// The `EDGE_LOOP`s bounding one face, longest first.
    fn face_rings(
        &mut self,
        body: &Body,
        fi: usize,
        face: &crate::brep::Face,
        _surface: &Surface,
        edge_ids: &[Option<u64>],
        vertex_ids: &[u64],
    ) -> Result<Vec<(u64, f64)>, Unsupported> {
        let mut chains: Vec<(Vec<(usize, bool)>, f64)> = Vec::new();

        // A face that names an edge twice meets itself along a seam, and its
        // boundary is one ring that walks the holes *and* the seam between
        // them. Chaining edges into closed rings on their own cannot produce
        // it, so for these the ring the face already states is read off, and
        // its edges matched into it in that order.
        //
        // The ring does not begin where an edge does — it begins wherever the
        // arrangement left it — and an `EDGE_LOOP` is cyclic, so the walk starts
        // at a vertex some edge really ends at.
        if (1..face.edges.len()).any(|i| face.edges[i..].contains(&face.edges[i - 1])) {
            let Some(loops) = face.loops.as_ref().filter(|l| !l.is_empty()) else {
                return Err(Unsupported::SeamBoundary { face: fi });
            };
            let mut found: Vec<(Vec<(usize, bool)>, f64)> = Vec::new();
            for ring in loops.iter() {
                let n = ring.vertices.len();
                if n < 3 {
                    return Err(Unsupported::SeamBoundary { face: fi });
                }
                let ends: HashSet<usize> = face
                    .edges
                    .iter()
                    .filter_map(|&e| body.edges().get(e))
                    .filter_map(|e| Some([*e.vertices.first()?, *e.vertices.last()?]))
                    .flatten()
                    .collect();
                let Some(chain) = (0..n)
                    .filter(|&k| ends.contains(&ring.vertices[k]))
                    .find_map(|offset| {
                        walk_ring(body, face, edge_ids, n, &|k| {
                            ring.vertices[(offset + k) % n]
                        })
                    })
                else {
                    return Err(Unsupported::SeamBoundary { face: fi });
                };
                found.push((chain, ring.area.abs()));
            }
            found.sort_by(|a, b| b.1.total_cmp(&a.1));
            chains = found;
        }

        // Closed edges each stand alone; open ones chain end to end.
        // Taken before the loop: the chaining below *also* pushes to `chains`,
        // one per closed edge, so asking whether it is empty part-way through
        // says something else entirely — and dropped every edge after the first
        // closed one when it did.
        let from_loops = !chains.is_empty();
        let mut open: Vec<usize> = Vec::new();
        for &e in &face.edges {
            if from_loops {
                break;
            }
            let Some(edge) = body.edges().get(e) else {
                continue;
            };
            if edge_ids.get(e).copied().flatten().is_none() {
                continue;
            }
            let ends_meet = edge.closed || edge.vertices.first() == edge.vertices.last();
            if ends_meet {
                chains.push((vec![(e, true)], perimeter(body, e)));
            } else {
                open.push(e);
            }
        }

        while let Some(first) = open.pop() {
            let mut chain = vec![(first, true)];
            let start = body.edges()[first].vertices[0];
            let mut head = *body.edges()[first].vertices.last().unwrap();
            let mut length = perimeter(body, first);
            while head != start {
                let Some(pos) = open.iter().position(|&e| {
                    let v = &body.edges()[e].vertices;
                    v[0] == head || *v.last().unwrap() == head
                }) else {
                    return Err(Unsupported::OpenFaceBoundary { face: fi });
                };
                let e = open.remove(pos);
                let v = &body.edges()[e].vertices;
                let forward = v[0] == head;
                head = if forward { *v.last().unwrap() } else { v[0] };
                length += perimeter(body, e);
                chain.push((e, forward));
            }
            chains.push((chain, length));
        }

        chains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::with_capacity(chains.len());
        for (chain, length) in chains {
            let oriented: Vec<Value> = chain
                .iter()
                .map(|&(e, forward)| {
                    let id = self.add(
                        "ORIENTED_EDGE",
                        vec![
                            Value::Text(String::new()),
                            Value::Derived,
                            Value::Derived,
                            Value::Ref(edge_ids[e].unwrap()),
                            Value::Enum(if forward { "T" } else { "F" }.into()),
                        ],
                    );
                    Value::Ref(id)
                })
                .collect();
            let id = self.add(
                "EDGE_LOOP",
                vec![Value::Text(String::new()), Value::List(oriented)],
            );
            out.push((id, length));
        }
        let _ = vertex_ids;
        Ok(out)
    }

    /// Emit a face that has no edges, by cutting it at its parametric seams.
    ///
    /// A closed surface has no STEP face: `ADVANCED_FACE` is a *bounded* region,
    /// and a whole sphere has no boundary to give it. So cut it — once per
    /// direction that wraps, giving two faces for a sphere and four for a torus
    /// — and let the pieces share the cut edges, which is what keeps the solid
    /// closed across a seam that only exists because the format needs one.
    ///
    /// The cuts are iso-curves, and on every analytic surface here an iso-curve
    /// is a circle or a line, so nothing is lost by making them.
    fn seam_faces(
        &mut self,
        fi: usize,
        face: &crate::brep::Face,
        surface: &Surface,
        surf_id: u64,
    ) -> Result<Vec<u64>, Unsupported> {
        if matches!(surface, Surface::Nurbs(_)) {
            return Err(Unsupported::UnboundedFace { face: fi });
        }
        let (u0, u1) = face.u_range;
        let (v0, v1) = face.v_range;
        if u1 <= u0 || v1 <= v0 || !u0.is_finite() || !v1.is_finite() {
            return Err(Unsupported::UnboundedFace { face: fi });
        }
        let (pu, pv) = surface.periodic();
        let us: Vec<f64> = if pu || face.u_wraps {
            vec![u0, 0.5 * (u0 + u1), u1]
        } else {
            vec![u0, u1]
        };
        let vs: Vec<f64> = if pv || face.v_wraps {
            vec![v0, 0.5 * (v0 + v1), v1]
        } else {
            vec![v0, v1]
        };

        // One `EDGE_CURVE` per cut, shared by the two cells it separates. Two
        // cells that each sampled their own copy would coincide but not join.
        //
        // The last cut in a wrapping direction *is* the first one — u₁ ≡ u₀ on a
        // closed surface — so it has to fold onto the same key, or a sphere
        // comes back with three meridians where it has two.
        let (nu, nv) = (us.len() - 1, vs.len() - 1);
        let (u_cyclic, v_cyclic) = (us.len() > 2, vs.len() > 2);
        let fold = move |dir: u8, a: usize, b: usize| -> (u8, usize, usize) {
            if dir == b'u' {
                (dir, if a == nu && u_cyclic { 0 } else { a }, b)
            } else {
                (dir, a, if b == nv && v_cyclic { 0 } else { b })
            }
        };
        let mut cache: HashMap<(u8, usize, usize), Option<u64>> = HashMap::new();
        let mut faces = Vec::new();

        for i in 0..us.len() - 1 {
            for j in 0..vs.len() - 1 {
                let mut oriented = Vec::new();
                // Round the cell: along v0, up the far u, back along v1, down
                // the near u. A degenerate side — a pole, where an iso-curve
                // collapses to a point — simply is not there.
                let sides: [(u8, usize, usize, bool); 4] = [
                    (b'v', i, j, true),
                    (b'u', i + 1, j, true),
                    (b'v', i, j + 1, false),
                    (b'u', i, j, false),
                ];
                for (dir, a, b, forward) in sides {
                    let key = fold(dir, a, b);
                    let id = match cache.get(&key) {
                        Some(v) => *v,
                        None => {
                            let pts: Vec<V3> = (0..=32)
                                .map(|k| {
                                    let t = k as f64 / 32.0;
                                    if dir == b'u' {
                                        surface.point(us[a], vs[b] + (vs[b + 1] - vs[b]) * t)
                                    } else {
                                        surface.point(us[a] + (us[a + 1] - us[a]) * t, vs[b])
                                    }
                                })
                                .collect();
                            let made = if degenerate(&pts) {
                                None
                            } else {
                                let sv = self.point(pts[0]);
                                let ev = self.point(*pts.last().unwrap());
                                let va = self.add(
                                    "VERTEX_POINT",
                                    vec![Value::Text(String::new()), Value::Ref(sv)],
                                );
                                let vb = self.add(
                                    "VERTEX_POINT",
                                    vec![Value::Text(String::new()), Value::Ref(ev)],
                                );
                                let (geom, _) = self.curve(&pts, false);
                                Some(self.add(
                                    "EDGE_CURVE",
                                    vec![
                                        Value::Text(String::new()),
                                        Value::Ref(va),
                                        Value::Ref(vb),
                                        Value::Ref(geom),
                                        Value::Enum("T".into()),
                                    ],
                                ))
                            };
                            cache.insert(key, made);
                            made
                        }
                    };
                    let Some(ec) = id else { continue };
                    let oe = self.add(
                        "ORIENTED_EDGE",
                        vec![
                            Value::Text(String::new()),
                            Value::Derived,
                            Value::Derived,
                            Value::Ref(ec),
                            Value::Enum(if forward { "T" } else { "F" }.into()),
                        ],
                    );
                    oriented.push(Value::Ref(oe));
                }
                if oriented.is_empty() {
                    return Err(Unsupported::UnboundedFace { face: fi });
                }
                let loop_id = self.add(
                    "EDGE_LOOP",
                    vec![Value::Text(String::new()), Value::List(oriented)],
                );
                let bound = self.add(
                    "FACE_OUTER_BOUND",
                    vec![
                        Value::Text(String::new()),
                        Value::Ref(loop_id),
                        Value::Enum("T".into()),
                    ],
                );
                faces.push(self.add(
                    "ADVANCED_FACE",
                    vec![
                        Value::Text(String::new()),
                        Value::List(vec![Value::Ref(bound)]),
                        Value::Ref(surf_id),
                        Value::Enum(if face.flipped { "F" } else { "T" }.into()),
                    ],
                ));
            }
        }
        Ok(faces)
    }

    /// The AP203 product/context boilerplate, and the file around it.
    fn finish(mut self, name: &str) -> StepFile {
        let shell = self.add(
            "CLOSED_SHELL",
            vec![
                Value::Text(String::new()),
                Value::List(self.face_ids.iter().map(|f| Value::Ref(*f)).collect()),
            ],
        );
        let brep = self.add(
            "MANIFOLD_SOLID_BREP",
            vec![Value::Text(name.into()), Value::Ref(shell)],
        );

        let metre = self.complex(vec![
            Value::Typed("LENGTH_UNIT".into(), vec![]),
            Value::Typed("NAMED_UNIT".into(), vec![Value::Derived]),
            Value::Typed(
                "SI_UNIT".into(),
                vec![Value::Omitted, Value::Enum("METRE".into())],
            ),
        ]);
        let radian = self.complex(vec![
            Value::Typed("NAMED_UNIT".into(), vec![Value::Derived]),
            Value::Typed("PLANE_ANGLE_UNIT".into(), vec![]),
            Value::Typed(
                "SI_UNIT".into(),
                vec![Value::Omitted, Value::Enum("RADIAN".into())],
            ),
        ]);
        let steradian = self.complex(vec![
            Value::Typed("NAMED_UNIT".into(), vec![Value::Derived]),
            Value::Typed(
                "SI_UNIT".into(),
                vec![Value::Omitted, Value::Enum("STERADIAN".into())],
            ),
            Value::Typed("SOLID_ANGLE_UNIT".into(), vec![]),
        ]);
        let uncertainty = self.add(
            "UNCERTAINTY_MEASURE_WITH_UNIT",
            vec![
                Value::Typed("LENGTH_MEASURE".into(), vec![Value::Real(self.tolerance)]),
                Value::Ref(metre),
                Value::Text("distance_accuracy_value".into()),
                Value::Text("confusion accuracy".into()),
            ],
        );
        let context = self.complex(vec![
            Value::Typed(
                "GEOMETRIC_REPRESENTATION_CONTEXT".into(),
                vec![Value::Int(3)],
            ),
            Value::Typed(
                "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT".into(),
                vec![Value::List(vec![Value::Ref(uncertainty)])],
            ),
            Value::Typed(
                "GLOBAL_UNIT_ASSIGNED_CONTEXT".into(),
                vec![Value::List(vec![
                    Value::Ref(metre),
                    Value::Ref(radian),
                    Value::Ref(steradian),
                ])],
            ),
            Value::Typed(
                "REPRESENTATION_CONTEXT".into(),
                vec![Value::Text(String::new()), Value::Text(String::new())],
            ),
        ]);

        let origin = self.point([0.0; 3]);
        let z = self.direction([0.0, 0.0, 1.0]);
        let x = self.direction([1.0, 0.0, 0.0]);
        let axis = self.add(
            "AXIS2_PLACEMENT_3D",
            vec![
                Value::Text(String::new()),
                Value::Ref(origin),
                Value::Ref(z),
                Value::Ref(x),
            ],
        );
        let shape = self.add(
            "ADVANCED_BREP_SHAPE_REPRESENTATION",
            vec![
                Value::Text(name.into()),
                Value::List(vec![Value::Ref(axis), Value::Ref(brep)]),
                Value::Ref(context),
            ],
        );

        let app = self.add(
            "APPLICATION_CONTEXT",
            vec![Value::Text(
                "core data for automotive mechanical design processes".into(),
            )],
        );
        self.add(
            "APPLICATION_PROTOCOL_DEFINITION",
            vec![
                Value::Text("international standard".into()),
                Value::Text("automotive_design".into()),
                Value::Int(2000),
                Value::Ref(app),
            ],
        );
        let pctx = self.add(
            "PRODUCT_CONTEXT",
            vec![
                Value::Text(String::new()),
                Value::Ref(app),
                Value::Text("mechanical".into()),
            ],
        );
        let pdctx = self.add(
            "PRODUCT_DEFINITION_CONTEXT",
            vec![
                Value::Text("part definition".into()),
                Value::Ref(app),
                Value::Text("design".into()),
            ],
        );
        let product = self.add(
            "PRODUCT",
            vec![
                Value::Text(name.into()),
                Value::Text(name.into()),
                Value::Text(String::new()),
                Value::List(vec![Value::Ref(pctx)]),
            ],
        );
        let formation = self.add(
            "PRODUCT_DEFINITION_FORMATION",
            vec![
                Value::Text(String::new()),
                Value::Text(String::new()),
                Value::Ref(product),
            ],
        );
        let definition = self.add(
            "PRODUCT_DEFINITION",
            vec![
                Value::Text("design".into()),
                Value::Text(String::new()),
                Value::Ref(formation),
                Value::Ref(pdctx),
            ],
        );
        let pds = self.add(
            "PRODUCT_DEFINITION_SHAPE",
            vec![
                Value::Text(String::new()),
                Value::Text(String::new()),
                Value::Ref(definition),
            ],
        );
        self.add(
            "SHAPE_DEFINITION_REPRESENTATION",
            vec![Value::Ref(pds), Value::Ref(shape)],
        );

        StepFile {
            header: vec![
                Entity {
                    id: 1,
                    name: "FILE_DESCRIPTION".into(),
                    args: vec![
                        Value::List(vec![Value::Text(String::new())]),
                        Value::Text("2;1".into()),
                    ],
                },
                Entity {
                    id: 2,
                    name: "FILE_NAME".into(),
                    args: vec![
                        Value::Text(format!("{name}.step")),
                        // No clock is read here: a timestamp would make the same
                        // solid export to a different file every time, which
                        // breaks the round-trip tests and any content hash.
                        Value::Text(String::new()),
                        Value::List(vec![Value::Text(String::new())]),
                        Value::List(vec![Value::Text(String::new())]),
                        Value::Text(concat!("threers ", env!("CARGO_PKG_VERSION")).into()),
                        Value::Text(String::new()),
                        Value::Text(String::new()),
                    ],
                },
                Entity {
                    id: 3,
                    name: "FILE_SCHEMA".into(),
                    args: vec![Value::List(vec![Value::Text(
                        "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }".into(),
                    )])],
                },
            ],
            data: self.data,
        }
    }
}

// --------------------------------------------------------------------------
// Geometry recovery
// --------------------------------------------------------------------------

/// Collapse a knot vector to the distinct knots and their multiplicities, which
/// is how STEP writes one.
fn compress(knots: &[f64]) -> (Vec<f64>, Vec<usize>) {
    let mut ks: Vec<f64> = Vec::new();
    let mut ms: Vec<usize> = Vec::new();
    for &k in knots {
        match ks.last() {
            Some(&last) if (k - last).abs() < 1e-12 => *ms.last_mut().unwrap() += 1,
            _ => {
                ks.push(k);
                ms.push(1);
            }
        }
    }
    (ks, ms)
}

fn orthogonalise(x: V3, axis: V3) -> V3 {
    let a = v3::normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let p = v3::sub(x, v3::scale(a, v3::dot(x, a)));
    v3::normalize(p).unwrap_or_else(|| {
        let alt = if a[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        v3::normalize(v3::cross(a, alt)).unwrap_or([1.0, 0.0, 0.0])
    })
}

/// Whether every sample sits at the same place — a pole, where an iso-curve
/// collapses and there is no edge to write.
fn degenerate(pts: &[V3]) -> bool {
    pts.windows(2).all(|w| v3::dist(w[0], w[1]) < 1e-9)
}

/// Match a face's edges into a ring, in the ring's own order.
///
/// Consecutive edges share a vertex the ring names once, so each match advances
/// by one less than the edge's length. `None` if the ring turns out not to be
/// made of this face's edges, which is a decline rather than a guess.
fn walk_ring(
    body: &Body,
    face: &crate::brep::Face,
    edge_ids: &[Option<u64>],
    n: usize,
    at: &dyn Fn(usize) -> usize,
) -> Option<Vec<(usize, bool)>> {
    let mut chain: Vec<(usize, bool)> = Vec::new();
    let mut pos = 0usize;
    while pos < n {
        let mut step = None;
        for &e in &face.edges {
            let edge = body.edges().get(e)?;
            if edge_ids.get(e).copied().flatten().is_none() {
                continue;
            }
            let v = &edge.vertices;
            let l = v.len();
            if l < 2 || pos + l - 1 > n {
                continue;
            }
            if (0..l).all(|k| at(pos + k) == v[k]) {
                step = Some((e, true, l));
                break;
            }
            if (0..l).all(|k| at(pos + k) == v[l - 1 - k]) {
                step = Some((e, false, l));
                break;
            }
        }
        let (e, forward, l) = step?;
        chain.push((e, forward));
        pos += l - 1;
    }
    (pos == n && !chain.is_empty()).then_some(chain)
}

fn perimeter(body: &Body, edge: usize) -> f64 {
    let v = &body.edges()[edge].vertices;
    v.windows(2)
        .filter_map(|w| {
            Some(v3::dist(
                *body.vertices().get(w[0])?,
                *body.vertices().get(w[1])?,
            ))
        })
        .sum()
}

/// `(origin, direction)` if every point lies on one line.
fn fit_line(pts: &[V3], tolerance: f64) -> Option<(V3, V3)> {
    if pts.len() < 2 {
        return None;
    }
    let a = pts[0];
    let b = *pts.last()?;
    let dir = v3::normalize(v3::sub(b, a))?;
    for p in pts {
        let d = v3::sub(*p, a);
        let along = v3::dot(d, dir);
        if v3::dist(d, v3::scale(dir, along)) > tolerance {
            return None;
        }
    }
    Some((a, dir))
}

struct FittedCircle {
    center: V3,
    axis: V3,
    x_dir: V3,
    radius: f64,
}

/// A circle through the points, if they all lie on one.
///
/// Three points determine it; the rest are the check. Deriving it rather than
/// carrying it means an edge exports as a `CIRCLE` whatever produced it — an
/// authored rim, a boolean seam, or a file that was read back in.
fn fit_circle(pts: &[V3], closed: bool, tolerance: f64) -> Option<FittedCircle> {
    if pts.len() < 3 {
        return None;
    }
    // Three well-separated samples, so a nearly-straight arc does not decide the
    // plane from noise.
    let a = pts[0];
    let b = pts[pts.len() / 3];
    let c = pts[2 * pts.len() / 3];
    let ab = v3::sub(b, a);
    let ac = v3::sub(c, a);
    let n = v3::cross(ab, ac);
    let n2 = v3::dot(n, n);
    if n2 < 1e-24 {
        return None;
    }
    // Circumcentre of a triangle, in vector form.
    let alpha = v3::dot(ac, ac) * v3::dot(ab, v3::sub(ab, ac)) / (2.0 * n2);
    let beta = v3::dot(ab, ab) * v3::dot(ac, v3::sub(ac, ab)) / (2.0 * n2);
    let center = v3::add(a, v3::add(v3::scale(ab, alpha), v3::scale(ac, beta)));
    let radius = v3::dist(center, a);
    if !radius.is_finite() || radius <= tolerance {
        return None;
    }
    let axis = v3::normalize(n)?;
    for p in pts {
        if (v3::dist(center, *p) - radius).abs() > tolerance {
            return None;
        }
        if v3::dot(v3::sub(*p, center), axis).abs() > tolerance {
            return None;
        }
    }
    let x_dir = v3::normalize(v3::sub(a, center))?;
    // Wind the circle the way the samples run, so `same_sense` stays `.T.` and
    // the arc a reader trims out is the one that was meant.
    let y = v3::cross(axis, x_dir);
    let second = v3::sub(pts[1.min(pts.len() - 1)], center);
    let axis = if v3::dot(second, y) < 0.0 {
        v3::scale(axis, -1.0)
    } else {
        axis
    };
    let _ = closed;
    Some(FittedCircle {
        center,
        axis,
        x_dir,
        radius,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::step::part21::{parse, real};

    #[test]
    fn a_cuboid_exports_as_six_planes() {
        let (text, report) = export(&Body::cuboid([2.0, 3.0, 4.0]), "box", 1e-6);
        assert_eq!(report.faces, 6);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        assert!(report.is_exact(), "{report:?}");

        let f = parse(&text).unwrap();
        assert_eq!(f.all("PLANE").count(), 6);
        assert_eq!(f.all("ADVANCED_FACE").count(), 6);
        assert_eq!(f.all("CLOSED_SHELL").count(), 1);
        // A box has eight corners, and welding by position is what keeps a
        // reader from seeing six unconnected quads.
        assert_eq!(f.all("VERTEX_POINT").count(), 8);
        assert_eq!(f.all("EDGE_CURVE").count(), 12);
        // Every edge is straight, so none should have become a polyline.
        assert_eq!(f.all("LINE").count(), 12);
        assert_eq!(f.all("POLYLINE").count(), 0);
    }

    #[test]
    fn a_cylinder_keeps_its_surface_and_its_rims() {
        let (text, report) = export(
            &Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0),
            "rod",
            1e-6,
        );
        assert!(report.is_exact(), "{report:?}");
        let f = parse(&text).unwrap();

        assert_eq!(f.all("CYLINDRICAL_SURFACE").count(), 1);
        assert_eq!(f.all("PLANE").count(), 2);
        // The rims are circles, not two hundred short segments.
        assert_eq!(f.all("CIRCLE").count(), 2);
        assert_eq!(f.all("POLYLINE").count(), 0);

        let cyl = f.all("CYLINDRICAL_SURFACE").next().unwrap();
        assert_eq!(cyl.args[2].as_real(), Some(2.0));
    }

    #[test]
    fn the_analytic_surfaces_each_get_their_own_entity() {
        for (body, entity) in [
            (Body::sphere([0.0; 3], 1.5), "SPHERICAL_SURFACE"),
            (
                Body::torus([0.0; 3], [0.0, 0.0, 1.0], 3.0, 1.0),
                "TOROIDAL_SURFACE",
            ),
            (
                Body::cone([0.0; 3], [0.0, 0.0, 1.0], 2.0, 4.0),
                "CONICAL_SURFACE",
            ),
        ] {
            let (text, report) = export(&body, "s", 1e-6);
            assert!(report.skipped.is_empty(), "{entity}: {:?}", report.skipped);
            let f = parse(&text).unwrap();
            assert_eq!(f.all(entity).count(), 1, "{entity} missing from {text}");
        }
    }

    #[test]
    fn a_face_with_a_hole_writes_two_bounds() {
        // The outer boundary and the hole are both `FACE_BOUND`s of one face,
        // and exactly one of them is the outer one.
        let body = Body::plate_with_hole([6.0, 6.0, 1.0], 1.5).unwrap();
        let (text, report) = export(&body, "plate", 1e-6);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        let f = parse(&text).unwrap();

        // The two plate faces the bore passes through, and the bore itself,
        // which is an annulus between its two rims.
        let holed = f
            .all("ADVANCED_FACE")
            .filter(|e| e.args[1].as_list().is_some_and(|b| b.len() == 2))
            .count();
        assert_eq!(holed, 3);
        assert_eq!(
            f.all("FACE_OUTER_BOUND").count(),
            f.all("ADVANCED_FACE").count()
        );
    }

    #[test]
    fn the_file_is_a_valid_part21_document() {
        let (text, _) = export(&Body::cuboid([1.0, 1.0, 1.0]), "unit", 1e-6);
        assert!(text.starts_with("ISO-10303-21;"), "{text}");
        assert!(text.trim_end().ends_with("END-ISO-10303-21;"));

        let f = parse(&text).unwrap();
        assert_eq!(f.header.len(), 3);
        assert_eq!(f.all("SHAPE_DEFINITION_REPRESENTATION").count(), 1);
        assert_eq!(f.all("APPLICATION_PROTOCOL_DEFINITION").count(), 1);

        // Every reference resolves. A dangling `#n` is the single most common
        // way a written file fails in another system.
        let index = f.index();
        for e in &f.data {
            for r in refs(&e.args) {
                assert!(index.contains_key(&r), "#{} references missing #{r}", e.id);
            }
        }
    }

    fn refs(args: &[Value]) -> Vec<u64> {
        let mut out = Vec::new();
        for a in args {
            match a {
                Value::Ref(r) => out.push(*r),
                Value::List(v) | Value::Typed(_, v) => out.extend(refs(v)),
                _ => {}
            }
        }
        out
    }

    #[test]
    fn exporting_is_deterministic() {
        // No clock, no hash iteration order: the same solid must produce the
        // same bytes, or nothing downstream can be content-addressed.
        let body = Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 1.0, 2.0);
        let (a, _) = export(&body, "rod", 1e-6);
        let (b, _) = export(&body, "rod", 1e-6);
        assert_eq!(a, b);
    }

    #[test]
    fn a_boolean_result_exports_with_its_surfaces_intact() {
        use crate::brep::BooleanOp;
        let plate = Body::cuboid([10.0, 8.0, 2.0]);
        let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0);
        let cut = plate
            .boolean(&drill, BooleanOp::Difference, 1e-3)
            .expect("closed form");

        let (text, report) = export(&cut, "cut", 1e-3);
        assert!(report.skipped.is_empty(), "{:?}", report.skipped);
        let f = parse(&text).unwrap();
        // The bore survives as a cylinder. This is the whole point: a mesh
        // export would have made it two hundred triangles.
        assert_eq!(f.all("CYLINDRICAL_SURFACE").count(), 1);
        assert_eq!(f.all("PLANE").count(), 6);
        // And its seams as circles.
        assert_eq!(f.all("CIRCLE").count(), 2);
    }

    #[test]
    fn a_straight_run_of_points_is_a_line_and_a_round_one_is_a_circle() {
        let straight: Vec<V3> = (0..5).map(|i| [i as f64, 2.0 * i as f64, 0.0]).collect();
        assert!(fit_line(&straight, 1e-9).is_some());
        assert!(fit_circle(&straight, false, 1e-9).is_none());

        let round: Vec<V3> = (0..16)
            .map(|i| {
                let t = std::f64::consts::TAU * i as f64 / 16.0;
                [3.0 * t.cos() + 1.0, 3.0 * t.sin() - 2.0, 5.0]
            })
            .collect();
        assert!(fit_line(&round, 1e-9).is_none());
        let c = fit_circle(&round, true, 1e-9).unwrap();
        assert!((c.radius - 3.0).abs() < 1e-9, "{}", c.radius);
        assert!(v3::dist(c.center, [1.0, -2.0, 5.0]) < 1e-9);

        // A point off the circle disqualifies the whole run rather than being
        // averaged away — approximating here would silently move an edge.
        let mut bent = round.clone();
        bent[4][2] += 0.01;
        assert!(fit_circle(&bent, true, 1e-6).is_none());
    }

    #[test]
    fn a_knot_vector_compresses_to_distinct_knots_and_multiplicities() {
        let (k, m) = compress(&[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]);
        assert_eq!(k, vec![0.0, 0.5, 1.0]);
        assert_eq!(m, vec![3, 1, 3]);
    }

    #[test]
    fn reals_are_written_in_the_form_step_requires() {
        let (text, _) = export(&Body::cuboid([1.0, 1.0, 1.0]), "unit", 1e-6);
        // A coordinate of exactly 1 must be `1.`, not `1` — the second is an
        // integer token where the schema wants a real.
        assert!(text.contains("0.5"), "{text}");
        assert!(
            !text.contains("(1,0.5,0.5)"),
            "an integer where a real belongs"
        );
        assert!(text.contains("0.5,0.5,0.5"), "{text}");
        assert_eq!(real(1.0), "1.");
    }
}
