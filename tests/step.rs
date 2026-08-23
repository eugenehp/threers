//! STEP import/export, from outside the crate.
//!
//! The unit tests check the pieces; these check the promise — that a solid can
//! leave as a file and come back as the same solid, with its surfaces intact.

#![cfg(feature = "step")]

use threers::brep::{Body, BooleanOp};
use threers::step::{export, import, parse, Unsupported, Value};

/// Volume of a body's tessellation, by the divergence theorem.
fn volume(body: &Body, tolerance: f64) -> f64 {
    let mut b = body.clone();
    b.refine_edges(tolerance);
    let (mesh, report) = b.tessellate(tolerance);
    assert!(report.is_closed(), "{} open edges", report.boundary_edges);
    // An empty mesh is vacuously closed, and a body that exported to nothing
    // would otherwise sail through every volume check in this file.
    assert!(report.triangles > 0, "nothing was tessellated");
    let pos = &mesh.get_attribute("position").unwrap().array;
    let idx = mesh.index.as_ref().unwrap();
    let v = |i: u32| -> [f64; 3] {
        let o = i as usize * 3;
        [pos[o] as f64, pos[o + 1] as f64, pos[o + 2] as f64]
    };
    idx.chunks_exact(3)
        .map(|t| {
            let (a, b, c) = (v(t[0]), v(t[1]), v(t[2]));
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        })
        .sum::<f64>()
        .abs()
}

#[test]
fn a_solid_survives_a_file_round_trip() {
    for (body, expected) in [
        (Body::cuboid([2.0, 3.0, 4.0]), 24.0),
        (
            Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 5.0),
            std::f64::consts::PI * 4.0 * 5.0,
        ),
        (
            Body::sphere([0.0; 3], 2.0),
            4.0 / 3.0 * std::f64::consts::PI * 8.0,
        ),
    ] {
        let (text, ex) = export(&body, "part", 1e-4);
        assert!(ex.skipped.is_empty(), "export dropped {:?}", ex.skipped);
        let (back, im) = import(&text, 1e-4).expect("our own file");
        assert!(im.skipped.is_empty(), "import dropped {:?}", im.skipped);

        let got = volume(&back, 1e-3);
        assert!(
            (got - expected).abs() / expected < 0.01,
            "volume {got}, expected {expected}"
        );
    }
}

#[test]
fn a_bore_arrives_as_a_cylinder_not_as_triangles() {
    // The whole reason to write STEP instead of STL: the receiving system gets
    // a cylinder it can re-dimension, not two hundred facets it cannot.
    let plate = Body::cuboid([10.0, 8.0, 2.0]);
    let drill = Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0);
    let cut = plate
        .boolean(&drill, BooleanOp::Difference, 1e-3)
        .expect("closed form");

    let (text, ex) = export(&cut, "plate", 1e-3);
    assert!(ex.skipped.is_empty(), "{:?}", ex.skipped);

    let f = parse(&text).unwrap();
    assert_eq!(f.all("CYLINDRICAL_SURFACE").count(), 1);
    assert_eq!(f.all("CIRCLE").count(), 2, "the two seams");
    assert_eq!(f.all("POLYLINE").count(), 0, "nothing was sampled");

    let (back, im) = import(&text, 1e-3).unwrap();
    assert!(im.skipped.is_empty(), "{:?}", im.skipped);
    assert_eq!(
        back.surfaces()
            .iter()
            .filter(|s| s.kind() == "cylinder")
            .count(),
        1
    );

    let expected = 160.0 - std::f64::consts::PI * 4.0 * 2.0;
    let got = volume(&back, 1e-3);
    assert!(
        (got - expected).abs() / expected < 0.02,
        "volume {got}, expected {expected}"
    );
}

#[test]
fn an_imported_solid_can_be_used_again() {
    // The point of importing into a `Body` rather than a mesh: what comes back
    // is a solid, so it can be cut again.
    let (text, _) = export(&Body::cuboid([10.0, 10.0, 4.0]), "blank", 1e-3);
    let (blank, _) = import(&text, 1e-3).unwrap();

    let drill = Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 2.0, 12.0);
    let cut = blank
        .boolean(&drill, BooleanOp::Difference, 1e-3)
        .expect("an imported solid is still a solid");

    let expected = 400.0 - std::f64::consts::PI * 4.0 * 4.0;
    let got = volume(&cut, 1e-3);
    assert!(
        (got - expected).abs() / expected < 0.02,
        "volume {got}, expected {expected}"
    );
}

#[test]
fn a_part_built_by_several_cuts_round_trips() {
    // The whole stack in one line of modelling: three bores cut one after
    // another, written out, read back, and still the same solid — with all
    // three bores arriving as cylinders.
    let mut part = Body::cuboid([40.0, 24.0, 4.0]);
    for x in [-14.0f64, 0.0, 14.0] {
        let drill = Body::cylinder([x, 0.0, -4.0], [0.0, 0.0, 1.0], 2.5, 12.0);
        part = part.boolean(&drill, BooleanOp::Difference, 1e-3).unwrap();
    }

    let (text, ex) = export(&part, "drilled-plate", 1e-3);
    assert!(ex.skipped.is_empty(), "{:?}", ex.skipped);
    assert!(ex.is_exact(), "{ex:?}");

    let (back, im) = import(&text, 1e-3).unwrap();
    assert!(im.skipped.is_empty(), "{:?}", im.skipped);
    assert_eq!(
        back.surfaces()
            .iter()
            .filter(|s| s.kind() == "cylinder")
            .count(),
        3
    );
    let before = volume(&part, 1e-3);
    let after = volume(&back, 1e-3);
    assert!(
        (before - after).abs() / before < 1e-9,
        "{before} out, {after} back"
    );
}

#[test]
fn every_reference_in_a_written_file_resolves() {
    // A dangling `#n` is the commonest way a written file fails in another
    // system, and it is invisible until something else tries to read it.
    let body = Body::plate_with_hole([6.0, 6.0, 1.0], 1.5).unwrap();
    let (text, _) = export(&body, "plate", 1e-4);
    let f = parse(&text).unwrap();
    let index = f.index();

    fn refs(args: &[Value], out: &mut Vec<u64>) {
        for a in args {
            match a {
                Value::Ref(r) => out.push(*r),
                Value::List(v) | Value::Typed(_, v) => refs(v, out),
                _ => {}
            }
        }
    }
    let mut count = 0;
    for e in &f.data {
        let mut rs = Vec::new();
        refs(&e.args, &mut rs);
        for r in rs {
            assert!(index.contains_key(&r), "#{} references missing #{r}", e.id);
            count += 1;
        }
    }
    assert!(count > 50, "only {count} references — is the file empty?");
}

#[test]
fn the_file_names_the_schema_it_claims_to_be() {
    let (text, _) = export(&Body::cuboid([1.0, 1.0, 1.0]), "unit", 1e-6);
    let f = parse(&text).unwrap();
    let schema = f
        .header
        .iter()
        .find(|e| e.name == "FILE_SCHEMA")
        .expect("a header without a schema is not readable anywhere");
    let names = schema.args[0].as_list().unwrap();
    assert!(
        names[0].as_text().unwrap().contains("10303"),
        "{:?}",
        names[0]
    );
    assert_eq!(f.all("APPLICATION_PROTOCOL_DEFINITION").count(), 1);
    assert_eq!(f.all("SHAPE_DEFINITION_REPRESENTATION").count(), 1);
}

#[test]
fn a_file_written_by_hand_reads() {
    // Not everything is round-tripped from our own writer. This is the shape a
    // real exporter produces: separate placements, `$` for an optional
    // reference direction, and reals in mixed spellings.
    let text = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=AXIS2_PLACEMENT_3D('',#1,#2,$);
#4=CYLINDRICAL_SURFACE('',#3,3.0);
#5=CARTESIAN_POINT('',(3.,0.,0.));
#6=VERTEX_POINT('',#5);
#7=CIRCLE('',#3,3.);
#8=EDGE_CURVE('',#6,#6,#7,.T.);
#9=ORIENTED_EDGE('',*,*,#8,.T.);
#10=EDGE_LOOP('',(#9));
#11=FACE_OUTER_BOUND('',#10,.T.);
#12=ADVANCED_FACE('',(#11),#4,.T.);
#13=CLOSED_SHELL('',(#12));
ENDSEC;
END-ISO-10303-21;
";
    let (body, report) = import(text, 1e-4).expect("a hand-written file");
    assert!(report.is_complete(), "{:?}", report.skipped);
    assert_eq!(body.faces().len(), 1);
    assert_eq!(body.surfaces()[0].kind(), "cylinder");
    // `$` for the reference direction means "any perpendicular", not "invalid".
    assert_eq!(report.edges, 1);
}

#[test]
fn a_face_that_is_everything_except_its_own_boundary_survives() {
    // Two balls differenced leave a face that is a whole sphere *minus* a disk.
    // Its boundary is one circle, and that circle bounds the disk just as well —
    // so a file that only says "this circle" says both, and the reader picks.
    // It picked the empty one: the round trip came back a solid of no volume,
    // with `skipped` empty at *both* ends.
    //
    // AP203 already distinguishes them. A `FACE_OUTER_BOUND` is the edge of the
    // face; a plain `FACE_BOUND` is a hole, and a face with only holes is
    // everything else on the surface. Written that way it round-trips.
    let a = Body::sphere([0.0; 3], 3.0);
    let b = Body::sphere([2.0, 0.0, 0.0], 2.0);
    let cut = a
        .boolean(&b, BooleanOp::Difference, 1e-3)
        .expect("two spheres have a closed form");

    let (text, ex) = export(&cut, "two-balls", 1e-3);
    assert!(ex.skipped.is_empty(), "{:?}", ex.skipped);
    let f = parse(&text).unwrap();
    assert_eq!(
        f.all("FACE_OUTER_BOUND").count(),
        0,
        "neither face is bounded by what it encloses"
    );
    assert!(f.all("FACE_BOUND").count() >= 2, "both holes are written");

    let (back, im) = import(&text, 1e-3).unwrap();
    assert!(im.skipped.is_empty(), "{:?}", im.skipped);
    let (before, after) = (volume(&cut, 1e-3), volume(&back, 1e-3));
    assert!(
        (before - after).abs() / before < 1e-3,
        "{before} out, {after} back"
    );

    // What still cannot be stated is still named rather than written.
    let boxed = a
        .boolean(
            &Body::cuboid([2.0, 2.0, 8.0]),
            BooleanOp::Intersection,
            1e-3,
        )
        .expect("closed form");
    let (_, ex) = export(&boxed, "ball-and-box", 1e-3);
    assert!(
        ex.skipped
            .iter()
            .any(|u| matches!(u, Unsupported::AmbiguousRegion { .. })),
        "what cannot be stated is named: {:?}",
        ex.skipped
    );
}

#[test]
fn a_face_that_runs_up_to_a_pole_keeps_its_cap() {
    // At a sphere's axis the iso-curve collapses to a point, so there is no
    // curve for an edge to be, and a cap's boundary is its rim *and* the pole.
    // Written with only the rim, the reader has nothing to say where the face
    // begins: a cap came back with `v` running from its rim to its rim — zero
    // extent, no triangles — and a third of the solid's volume disappeared with
    // `skipped` empty at both ends. AP203's answer is `VERTEX_LOOP`, a bound of
    // one vertex.
    let ball = Body::sphere([0.0; 3], 3.0);
    let bore = Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0);
    let plug = ball
        .boolean(&bore, BooleanOp::Intersection, 1e-3)
        .expect("a sphere and a cylinder have a closed form");

    let (text, ex) = export(&plug, "plug", 1e-3);
    assert!(ex.skipped.is_empty(), "{:?}", ex.skipped);
    let f = parse(&text).unwrap();
    assert_eq!(f.all("VERTEX_LOOP").count(), 2, "one bound per pole");

    let (back, im) = import(&text, 1e-3).unwrap();
    assert!(im.skipped.is_empty(), "{:?}", im.skipped);
    let (before, after) = (volume(&plug, 1e-3), volume(&back, 1e-3));
    assert!(
        (before - after).abs() / before < 1e-3,
        "{before} out, {after} back"
    );
}

#[test]
fn no_solid_leaves_through_a_file_and_comes_back_a_different_size() {
    // The guard for a whole class of bug, twice found by hand: a file that
    // claims to be the solid and is not. Both times `skipped` was empty at both
    // ends, so nothing downstream had any way to know — once because a face's
    // edges bound two regions and the reader took the empty one, once because a
    // cap's pole is not a curve and the face came back with no extent.
    //
    // The rule is not "everything round-trips". It is that a result either
    // comes back the size it left, or the export says what it could not carry.
    // Silence and a wrong answer together is the one outcome ruled out.
    let pairs: Vec<(&str, Body, Body)> = vec![
        (
            "plate and bore",
            Body::cuboid([10.0, 8.0, 2.0]),
            Body::cylinder([0.0, 0.0, -3.0], [0.0, 0.0, 1.0], 2.0, 6.0),
        ),
        (
            "box and box",
            Body::cuboid([4.0, 4.0, 4.0]),
            Body::cuboid([3.0, 3.0, 3.0]),
        ),
        (
            "ball and ball",
            Body::sphere([0.0; 3], 3.0),
            Body::sphere([2.0, 0.0, 0.0], 2.0),
        ),
        (
            "ball and bore",
            Body::sphere([0.0; 3], 3.0),
            Body::cylinder([0.0, 0.0, -6.0], [0.0, 0.0, 1.0], 1.0, 12.0),
        ),
        (
            "rod across rod",
            Body::cylinder([0.0; 3], [0.0, 0.0, 1.0], 2.0, 8.0),
            Body::cylinder([-6.0, 0.0, 4.0], [1.0, 0.0, 0.0], 1.0, 12.0),
        ),
        (
            "box and ball",
            Body::cuboid([4.0, 4.0, 4.0]),
            Body::sphere([2.0, 0.0, 0.0], 2.0),
        ),
        (
            "ball and box",
            Body::sphere([0.0; 3], 3.0),
            Body::cuboid([2.0, 2.0, 8.0]),
        ),
        (
            "box and corner box",
            Body::cuboid([4.0, 4.0, 4.0]),
            Body::cuboid([3.0, 3.0, 3.0])
                .translated([1.0, 1.0, 1.0])
                .expect("a translate keeps every surface"),
        ),
        (
            "plate and blind bore",
            Body::cuboid([6.0, 6.0, 2.0]),
            Body::cylinder([0.0, 0.0, -4.0], [0.0, 0.0, 1.0], 1.5, 8.0),
        ),
    ];

    let (mut checked, mut declined) = (0, 0);
    for (name, a, b) in &pairs {
        for op in [
            BooleanOp::Union,
            BooleanOp::Difference,
            BooleanOp::Intersection,
        ] {
            let Ok(result) = a.boolean(b, op, 1e-3) else {
                continue; // the kernel declining is its own business
            };
            let mut probe = result.clone();
            probe.refine_edges(1e-3);
            if probe.tessellate(1e-3).1.triangles == 0 {
                continue; // an empty intersection is nothing to carry
            }
            let before = volume(&result, 1e-3);

            let (text, ex) = export(&result, "part", 1e-3);
            if !ex.skipped.is_empty() {
                declined += 1;
                continue;
            }
            let (back, im) = import(&text, 1e-3)
                .unwrap_or_else(|e| panic!("{name} {op:?}: wrote a file it cannot read: {e:?}"));
            assert!(im.skipped.is_empty(), "{name} {op:?}: {:?}", im.skipped);

            let after = volume(&back, 1e-3);
            assert!(
                (before - after).abs() / before < 1e-3,
                "{name} {op:?}: left as {before}, came back as {after}, and nothing was reported"
            );
            checked += 1;
        }
    }
    // Not an assertion about the ratio — only that the corpus is doing work.
    assert!(
        checked >= 14,
        "only {checked} results round-tripped ({declined} declined) — is the corpus still building solids?"
    );
}

#[test]
fn a_face_that_meets_itself_along_a_seam_reads() {
    // How other systems write a closed surface: one face, and one `EDGE_CURVE`
    // named twice in its loop — up the seam, round the top, down the same seam,
    // round the bottom. Nothing this crate exports looks like this yet, so
    // without a file written by hand the reader would go untested.
    //
    // The difficulty is that both passes along the seam invert to the same `u`.
    // Taken as points they sit on top of each other and the face has no area;
    // the bound has to be walked in order with `u` unwrapped against the point
    // before, which puts the second pass a full turn from the first.
    let text = "\
ISO-10303-21;
HEADER;
FILE_DESCRIPTION((''),'2;1');
FILE_NAME('','',(''),(''),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.,0.,0.));
#2=DIRECTION('',(0.,0.,1.));
#3=DIRECTION('',(1.,0.,0.));
#4=AXIS2_PLACEMENT_3D('',#1,#2,#3);
#5=CYLINDRICAL_SURFACE('',#4,1.);
#6=CARTESIAN_POINT('',(1.,0.,0.));
#7=VERTEX_POINT('',#6);
#8=CARTESIAN_POINT('',(1.,0.,2.));
#9=VERTEX_POINT('',#8);
#11=VECTOR('',#2,1.);
#10=LINE('',#6,#11);
#12=EDGE_CURVE('',#7,#9,#10,.T.);
#13=CIRCLE('',#4,1.);
#14=EDGE_CURVE('',#7,#7,#13,.T.);
#15=CARTESIAN_POINT('',(0.,0.,2.));
#16=AXIS2_PLACEMENT_3D('',#15,#2,#3);
#17=CIRCLE('',#16,1.);
#18=EDGE_CURVE('',#9,#9,#17,.T.);
#19=ORIENTED_EDGE('',*,*,#12,.T.);
#20=ORIENTED_EDGE('',*,*,#18,.T.);
#21=ORIENTED_EDGE('',*,*,#12,.F.);
#22=ORIENTED_EDGE('',*,*,#14,.F.);
#23=EDGE_LOOP('',(#19,#20,#21,#22));
#24=FACE_OUTER_BOUND('',#23,.T.);
#25=ADVANCED_FACE('',(#24),#5,.T.);
#26=CLOSED_SHELL('',(#25));
ENDSEC;
END-ISO-10303-21;
";
    let (body, report) = import(text, 1e-3).expect("a hand-written seamed face");
    assert!(report.is_complete(), "{:?}", report.skipped);
    assert_eq!(body.faces().len(), 1);

    let face = &body.faces()[0];
    let seam = face
        .edges
        .iter()
        .find(|&&e| face.edges.iter().filter(|&&x| x == e).count() == 2)
        .copied()
        .expect("the seam is named twice by the one face");
    assert_eq!(body.edges()[seam].surfaces.0, body.edges()[seam].surfaces.1);

    // The region is the whole parameter rectangle: a full turn by the height.
    let rings = face
        .loops
        .as_ref()
        .expect("a seamed face states its region");
    let area: f64 = rings.iter().map(|r| r.area).sum();
    let expected = std::f64::consts::TAU * 2.0;
    assert!(
        (area.abs() - expected).abs() / expected < 0.05,
        "region area {area}, expected about {expected} — did the two passes along the seam collapse?"
    );
}

#[test]
fn a_band_bounded_by_its_own_seam_survives_a_file_round_trip() {
    // A square column through a ball leaves the sphere as a *band*: its
    // boundary walks the top hole, down the seam, the bottom hole, and back up
    // the same seam. Three things had to be true at once for that to survive.
    //
    // The seam had to be an edge — the two passes along it are the same 51
    // vertices in opposite order, so the curve was already there and shared.
    // The edges crossing the seam had to be cut at it, because in three
    // dimensions such an edge is one curve and the ring uses it as two. And the
    // writer had to take the ring the face states rather than chain the edges
    // into closed loops of their own, which produced the two holes plus the
    // seam as a loop of no area.
    let ball = Body::sphere([0.0; 3], 3.0);
    let column = Body::cuboid([2.0, 2.0, 8.0]);
    let cut = ball
        .boolean(&column, BooleanOp::Difference, 1e-3)
        .expect("a sphere and a box have a closed form");

    // The band's face names one edge twice: that is what a seam is.
    let face = cut
        .faces()
        .iter()
        .find(|f| (1..f.edges.len()).any(|i| f.edges[i..].contains(&f.edges[i - 1])))
        .expect("the sphere's face runs along its own seam");
    assert!(face.edges.len() >= 14, "{} edges", face.edges.len());

    let (text, ex) = export(&cut, "ball-and-column", 1e-3);
    assert!(ex.skipped.is_empty(), "{:?}", ex.skipped);

    let (back, im) = import(&text, 1e-3).unwrap();
    assert!(im.skipped.is_empty(), "{:?}", im.skipped);
    let (before, after) = (volume(&cut, 1e-3), volume(&back, 1e-3));
    assert!(
        (before - after).abs() / before < 1e-3,
        "{before} out, {after} back"
    );
}
