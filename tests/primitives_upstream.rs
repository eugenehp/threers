//! threers's primitives against three r165's, bit for bit.
//!
//! `tests/parity/scenes/rust/primitives-upstream.txt` is three.js's own
//! geometry classes built with a spread of parameters
//! (`tests/parity/export-primitives-upstream.mjs`). Each is built here through
//! threers with the same parameters and must have the same index, the same
//! position, normal and uv floats to the bit, and the same groups. Near is not
//! enough: a vertex a hair off an axis decides which side of a ray or a
//! cutting plane it falls on — a ray's hits, a CSG's cuts.

use threers::core::BufferGeometry;
use threers::geometries::{
    BoxGeometry, CapsuleGeometry, CircleGeometry, ConeGeometry, CylinderGeometry,
    DodecahedronGeometry, IcosahedronGeometry, LatheGeometry, OctahedronGeometry, PlaneGeometry,
    RingGeometry, SphereGeometry, TetrahedronGeometry, TorusGeometry, TorusKnotGeometry,
};
use threers::math::Vector2;

const FIXTURE: &str = include_str!("parity/scenes/rust/primitives-upstream.txt");

fn base64(s: &str) -> Vec<u8> {
    let val = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    };
    let bytes: Vec<u8> = s.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            n |= val(c) << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        if chunk.len() > 2 {
            out.push((n >> 8) as u8);
        }
        if chunk.len() > 3 {
            out.push(n as u8);
        }
    }
    out
}

fn words(s: &str) -> Vec<u32> {
    base64(s)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Case {
    name: String,
    args: String,
    index: Option<Vec<u32>>,
    attrs: Vec<(String, usize, Vec<u32>)>,
    groups: Vec<(u32, u32, u32)>,
}

fn parse() -> Vec<Case> {
    let mut out = Vec::new();
    let mut case: Option<Case> = None;
    for line in FIXTURE.lines() {
        let mut f = line.splitn(2, ' ');
        let (key, rest) = (f.next().unwrap_or(""), f.next().unwrap_or(""));
        match key {
            "geom" => {
                let (name, args) = rest.split_once(' ').unwrap();
                case = Some(Case {
                    name: name.into(),
                    args: args.into(),
                    index: None,
                    attrs: vec![],
                    groups: vec![],
                });
            }
            "index" => case.as_mut().unwrap().index = (rest != "-").then(|| words(rest)),
            "attr" => {
                let v: Vec<&str> = rest.split(' ').collect();
                case.as_mut().unwrap().attrs.push((
                    v[0].into(),
                    v[1].parse().unwrap(),
                    words(v[2]),
                ));
            }
            "group" => {
                let v: Vec<u32> = rest.split(' ').map(|x| x.parse().unwrap()).collect();
                case.as_mut().unwrap().groups.push((v[0], v[1], v[2]));
            }
            "end" => out.push(case.take().unwrap()),
            _ => {}
        }
    }
    out
}

/// The numbers in a JSON array (nested arrays flattened).
fn numbers(json: &str) -> Vec<f64> {
    json.split(|c: char| {
        !(c.is_ascii_digit()
            || c == '.'
            || c == '-'
            || c == 'e'
            || c == 'E'
            || c == 'f'
            || c == 'a'
            || c == 'l'
            || c == 's'
            || c == 't'
            || c == 'r'
            || c == 'u')
    })
    .filter(|t| !t.is_empty())
    .map(|t| match t {
        "true" => 1.0,
        "false" => 0.0,
        t => t.parse().unwrap(),
    })
    .collect()
}

fn build(name: &str, args: &str) -> BufferGeometry {
    let a = numbers(args);
    let f = |i: usize| a[i] as f32;
    let n = |i: usize| a[i] as usize;
    match name {
        "BoxGeometry" => {
            BoxGeometry::with_segments(f(0), f(1), f(2), a[3] as u32, a[4] as u32, a[5] as u32)
        }
        "SphereGeometry" => SphereGeometry::with_range(f(0), n(1), n(2), f(3), f(4), f(5), f(6)),
        "CylinderGeometry" => {
            CylinderGeometry::new(f(0), f(1), f(2), n(3), n(4), a[5] != 0.0, f(6), f(7))
        }
        "ConeGeometry" => ConeGeometry::new(f(0), f(1), n(2), n(3), a[4] != 0.0, f(5), f(6)),
        "PlaneGeometry" => PlaneGeometry::with_segments(f(0), f(1), n(2), n(3)),
        "CircleGeometry" => CircleGeometry::new(f(0), n(1), f(2), f(3)),
        "RingGeometry" => RingGeometry::new(f(0), f(1), n(2), n(3), f(4), f(5)),
        "TorusGeometry" => TorusGeometry::new(f(0), f(1), n(2), n(3), f(4)),
        "TorusKnotGeometry" => {
            TorusKnotGeometry::new(f(0), f(1), n(2), n(3), a[4] as u32, a[5] as u32)
        }
        "TetrahedronGeometry" => TetrahedronGeometry::new(f(0), n(1)),
        "OctahedronGeometry" => OctahedronGeometry::new(f(0), n(1)),
        "IcosahedronGeometry" => IcosahedronGeometry::new(f(0), n(1)),
        "DodecahedronGeometry" => DodecahedronGeometry::new(f(0), n(1)),
        "CapsuleGeometry" => CapsuleGeometry::new(f(0), f(1), n(2), n(3)),
        "LatheGeometry" => {
            // `[[x, y], …], segments, phiStart, phiLength`
            let k = a.len() - 3;
            let points: Vec<Vector2> = a[..k]
                .chunks(2)
                .map(|p| Vector2::new(p[0] as f32, p[1] as f32))
                .collect();
            LatheGeometry::new(&points, a[k] as usize, a[k + 1] as f32, a[k + 2] as f32)
        }
        other => panic!("no builder for {other}"),
    }
}

#[test]
fn every_primitive_is_three_js_bit_for_bit() {
    let cases = parse();
    assert!(cases.len() >= 33, "fixture holds {} cases", cases.len());
    let mut failures = Vec::new();
    for c in &cases {
        let g = build(&c.name, &c.args);
        let mut problems = Vec::new();
        let ours_index = g.index.clone();
        if ours_index != c.index {
            problems.push(format!(
                "index ({} vs three.js {})",
                ours_index.as_ref().map_or(0, |i| i.len()),
                c.index.as_ref().map_or(0, |i| i.len())
            ));
        }
        for (name, size, want) in &c.attrs {
            match g.get_attribute(name) {
                None => problems.push(format!("no {name}")),
                Some(attr) => {
                    let got: Vec<u32> = attr.array.iter().map(|v| v.to_bits()).collect();
                    if attr.item_size != *size || got.len() != want.len() {
                        problems.push(format!(
                            "{name}: {} values vs three.js {}",
                            got.len(),
                            want.len()
                        ));
                    } else {
                        let differ = got.iter().zip(want).filter(|(a, b)| a != b).count();
                        if differ != 0 {
                            problems
                                .push(format!("{name}: {differ} of {} values differ", want.len()));
                        }
                    }
                }
            }
        }
        let groups: Vec<(u32, u32, u32)> = g
            .groups
            .iter()
            .map(|gr| (gr.start, gr.count, gr.material_index))
            .collect();
        if groups != c.groups {
            problems.push(format!("groups {groups:?} vs three.js {:?}", c.groups));
        }
        if !problems.is_empty() {
            failures.push(format!("{}{}: {}", c.name, c.args, problems.join("; ")));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} differ from three.js:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}
