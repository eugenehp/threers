//! The polygon operations against the C++ library they follow, built with
//! clang for arm64 at `-O3` with a `z` per point: 400 random cases of
//! booleans to paths and trees (open paths and a `z` callback included),
//! offsets and simplification, whose output must match to the point.

use std::io::{BufRead, Write};

use threers_math::polygon_ops::*;

fn read_paths<'a>(tokens: &mut impl Iterator<Item = &'a str>) -> Paths64 {
    let n: usize = tokens.next().unwrap().parse().unwrap();

    (0..n)
        .map(|_| {
            let m: usize = tokens.next().unwrap().parse().unwrap();

            (0..m)
                .map(|_| {
                    let x = tokens.next().unwrap().parse().unwrap();
                    let y = tokens.next().unwrap().parse().unwrap();
                    let z = tokens.next().unwrap().parse().unwrap();
                    Point64::with_z(x, y, z)
                })
                .collect()
        })
        .collect()
}

fn print_path(out: &mut impl Write, path: &[Point64]) {
    write!(out, "{}", path.len()).unwrap();

    for pt in path {
        write!(out, " {},{},{}", pt.x, pt.y, pt.z).unwrap();
    }

    writeln!(out).unwrap();
}

fn print_tree(out: &mut impl Write, node: &PolyPath64, depth: usize) {
    for child in &node.childs {
        write!(out, "{depth} ").unwrap();
        print_path(out, &child.polygon);
        print_tree(out, child, depth + 1);
    }
}

#[allow(dead_code)]

/// Every case of `input`, its result to `out`.
pub fn run(input: impl BufRead, out: &mut impl Write) {
    for line in input.lines() {
        let line = line.unwrap();
        let mut tokens = line.split_whitespace();

        match tokens.next() {
            Some("P") => {
                let epsilon: f64 = tokens.next().unwrap().parse().unwrap();
                let closed = tokens.next().unwrap() != "0";
                let paths = read_paths(&mut tokens);
                let result = simplify_paths(&paths, epsilon, closed);
                writeln!(out, "simplified {}", result.len()).unwrap();

                for path in &result {
                    print_path(out, path);
                }

                continue;
            }
            Some("F") => {
                let mut num = || tokens.next().unwrap().parse::<f64>().unwrap();
                let (delta, miter, arc_tol) = (num(), num(), num());
                let (tree, preserve, reverse, groups) =
                    (num() != 0.0, num() != 0.0, num() != 0.0, num() as usize);
                let mut c = PathOffset::new(miter, arc_tol, preserve, reverse);

                for _ in 0..groups {
                    let jt = [
                        JoinType::Square,
                        JoinType::Bevel,
                        JoinType::Round,
                        JoinType::Miter,
                    ][tokens.next().unwrap().parse::<usize>().unwrap()];
                    let et = [
                        EndType::Polygon,
                        EndType::Joined,
                        EndType::Butt,
                        EndType::Square,
                        EndType::Round,
                    ][tokens.next().unwrap().parse::<usize>().unwrap()];
                    let paths = read_paths(&mut tokens);
                    c.add_paths(&paths, jt, et);
                }

                if tree {
                    let solution = c.execute_tree(delta);
                    writeln!(out, "tree 1").unwrap();
                    print_tree(out, &solution, 0);
                } else {
                    let solution = c.execute_paths(delta);
                    writeln!(out, "paths 1 {}", solution.len()).unwrap();

                    for path in &solution {
                        print_path(out, path);
                    }
                }

                writeln!(out, "open 0").unwrap();
                continue;
            }
            Some("C") => {}
            _ => continue,
        }

        let mut int = || tokens.next().unwrap().parse::<i32>().unwrap();
        let ct = [
            ClipType::NoClip,
            ClipType::Intersection,
            ClipType::Union,
            ClipType::Difference,
            ClipType::Xor,
        ][int() as usize];
        let fr = [
            FillRule::EvenOdd,
            FillRule::NonZero,
            FillRule::Positive,
            FillRule::Negative,
        ][int() as usize];
        let tree = int() != 0;
        let preserve = int() != 0;
        let reverse = int() != 0;

        tokens.next();
        let subjects = read_paths(&mut tokens);
        tokens.next();
        let open_subjects = read_paths(&mut tokens);
        tokens.next();
        let clips = read_paths(&mut tokens);

        let mut c = Boolean64::new();
        c.preserve_collinear = preserve;
        c.reverse_solution = reverse;
        c.set_z_callback(Box::new(|e1bot, e1top, e2bot, e2top, pt| {
            // as the C++ computes it, wrapping
            pt.z =
                pt.z.wrapping_mul(7)
                    .wrapping_add(e1bot.z.wrapping_mul(1000003))
                    .wrapping_add(e1top.z.wrapping_mul(1009))
                    .wrapping_add(e2bot.z.wrapping_mul(101))
                    .wrapping_add(e2top.z.wrapping_mul(11))
                    .wrapping_add(1);
        }));
        c.add_subject(&subjects);
        c.add_open_subject(&open_subjects);
        c.add_clip(&clips);

        let open;

        if tree {
            let (ok, solution, o) = c.execute_tree(ct, fr);
            open = o;
            writeln!(out, "tree {}", i32::from(ok)).unwrap();
            print_tree(out, &solution, 0);
        } else {
            let (ok, closed, o) = c.execute_paths(ct, fr);
            open = o;
            writeln!(out, "paths {} {}", i32::from(ok), closed.len()).unwrap();

            for path in &closed {
                print_path(out, path);
            }
        }

        writeln!(out, "open {}", open.len()).unwrap();

        for path in &open {
            print_path(out, path);
        }
    }
}

#[test]
fn the_same_as_the_cpp_original() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("polygon_ops");
    let input = std::fs::read(dir.join("cases.txt")).unwrap();
    let expected = std::fs::read_to_string(dir.join("expected.txt")).unwrap();

    let mut out = Vec::new();
    run(&input[..], &mut out);
    let out = String::from_utf8(out).unwrap();

    // case by case, to say which
    let split = |text: &str| -> Vec<String> {
        let mut cases: Vec<String> = Vec::new();

        for line in text.lines() {
            if line.starts_with("paths ")
                || line.starts_with("tree ")
                || line.starts_with("simplified ")
            {
                cases.push(String::new());
            }

            let last = cases.last_mut().expect("a case starts the output");
            last.push_str(line);
            last.push('\n');
        }

        cases
    };

    let (ours, theirs) = (split(&out), split(&expected));
    assert_eq!(ours.len(), theirs.len());

    for (i, (a, b)) in ours.iter().zip(&theirs).enumerate() {
        assert_eq!(a, b, "case {i}");
    }
}
