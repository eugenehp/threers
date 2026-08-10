//! A **Rust DSL** for authoring [`Solid`](crate::Solid) CSG trees — the same
//! models the `.scad` interpreter produces, but written directly in Rust with
//! full type-checking, editor autocomplete, and native control flow (loops,
//! functions, generics) instead of a string program.
//!
//! Two layers; reach for whichever reads better, and mix them freely.
//!
//! # Fluent (postfix)
//!
//! Transforms and booleans chain as methods on [`Solid`](crate::Solid). Reads
//! inside-out, like normal Rust:
//!
//! ```ignore
//! use threers::{cube, cylinder, sphere};
//!
//! let part = cube([30.0, 30.0, 30.0])
//!     .difference(cylinder(40.0, 8.0).rotate([90.0, 0.0, 0.0]))
//!     .union(sphere(5.0).translate([0.0, 0.0, 15.0]));
//! let stl = part.to_stl();
//! ```
//!
//! Rotations here are in **degrees** ([`Solid::rotate`](crate::Solid::rotate) /
//! [`rotate_axis`](crate::Solid::rotate_axis)), matching OpenSCAD; [`mirror`],
//! [`translate`], and [`scale`] round out the OpenSCAD transform set.
//!
//! [`mirror`]: crate::Solid::mirror
//! [`translate`]: crate::Solid::translate
//! [`scale`]: crate::Solid::scale
//!
//! # Declarative (`scad!`)
//!
//! An OpenSCAD-*shaped* tree: prefix operators, `(args)`, and `{ … }` child
//! blocks. Reads outside-in, exactly like a `.scad` program — but it's real Rust
//! that expands to the fluent builders at compile time (zero runtime parsing):
//!
//! ```ignore
//! use threers::scad;
//!
//! let part = scad! {
//!     difference() {
//!         cube([30.0, 30.0, 30.0]);
//!         translate([0.0, 0.0, -1.0]) { cylinder(40.0, 8.0); }
//!         union() {
//!             sphere(5.0);
//!             translate([0.0, 0.0, 12.0]) { sphere(5.0); }
//!         }
//!     }
//! };
//! ```
//!
//! Grammar (each *node* ends in `;` unless it takes a `{ … }` child block):
//! - `expr;` — a **leaf**: any Rust expression that evaluates to a `Solid`. That
//!   covers primitive calls (`cube([20.0,20.0,20.0]);`, `cylinder(40.0, 8.0);`),
//!   fluent chains on them (`cylinder(40.0, 8.0).rotate([90.0,0.0,0.0]);`), bare
//!   variables (`hub;`), and the `solid(x);` splice. Leaves resolve in the
//!   calling scope, so bring primitives in with `use threers::{cube, cylinder,
//!   sphere, …}`. Arguments are ordinary Rust — positional, not `h=40, r=8`.
//! - `union() { … }`, `difference() { … }`, `intersection() { … }`, `hull() { … }`
//!   — booleans (and the convex hull) over their child nodes.
//! - `translate(v) { … }`, `rotate(v) { … }`, `scale(v) { … }`, `mirror(v) { … }`
//!   — a transform applied to the implicit union of its children.
//! - `solid(expr);` — the **escape hatch**: splice any Rust `Solid` expression
//!   (a variable, a function result, a loop-built value) straight into the tree.
//!
//! Multiple top-level nodes are an implicit union, as in OpenSCAD.
//!
//! # Mixing in Rust control flow
//!
//! Because every builder returns a plain [`Solid`](crate::Solid) value, native
//! loops and functions compose without ceremony — build a `Vec<Solid>` and fold
//! it with `union!`, or splice it into a `scad!` tree via `solid(…)`:
//!
//! ```ignore
//! use threers::{cube, cylinder, scad, union};
//!
//! // A bolt circle, built with a normal `for` loop.
//! let holes: Vec<_> = (0..6)
//!     .map(|i| {
//!         let a = i as f32 * 60.0;
//!         cylinder(10.0, 1.5).rotate([90.0, 0.0, 0.0]).rotate([0.0, 0.0, a]).translate([12.0, 0.0, 0.0])
//!     })
//!     .collect();
//!
//! let plate = scad! {
//!     difference() {
//!         cylinder(4.0, 20.0);
//!         solid(union!(holes));   // splice the loop-built ring back in
//!     }
//! };
//! # let _ = plate;
//! ```

// Everything the DSL needs lives on `Solid` / the crate free functions already;
// this module is the docs home plus the macros below (which `#[macro_export]`
// hoists to the crate root regardless).

/// Build a [`Solid`](crate::Solid) from an OpenSCAD-shaped tree. See the
/// [module docs](self) for the grammar.
///
/// ```ignore
/// use threers::scad;
/// let m = scad! {
///     difference() {
///         cube([20.0, 20.0, 20.0]);
///         translate([0.0, 0.0, -1.0]) { cylinder(30.0, 6.0); }
///     }
/// };
/// ```
#[macro_export]
macro_rules! scad {
    ($($body:tt)*) => { $crate::__scad_group!($($body)*) };
}

// The muncher is split across small helper macros (rather than one `@`-tagged
// macro) on purpose: with a single macro, any node the rules *don't* match would
// fall through to the public `($($body:tt)*)` arm and recurse forever. Separate
// macros have no such catch-all, so a malformed node is a clean compile error.

/// Internal: collapse a child-node sequence to one [`Solid`](crate::Solid),
/// unioning when there's more than one (OpenSCAD's implicit-union semantics).
#[doc(hidden)]
#[macro_export]
macro_rules! __scad_group {
    ($($b:tt)*) => {{
        let mut __nodes: ::std::vec::Vec<$crate::Solid> = $crate::__scad_list!([] $($b)*);
        if __nodes.len() == 1 {
            __nodes.pop().unwrap()
        } else {
            $crate::union_all(__nodes)
        }
    }};
}

/// Internal: accumulate a node sequence into a `Vec<Solid>`.
#[doc(hidden)]
#[macro_export]
macro_rules! __scad_list {
    // done — emit the collected siblings.
    ([$($acc:expr,)*]) => { ::std::vec![$($acc),*] };
    // a parent op with a `{ … }` child block (boolean or transform keyword).
    ([$($acc:expr,)*] $p:ident ( $($a:tt)* ) { $($body:tt)* } $($rest:tt)*) => {
        $crate::__scad_list!([$($acc,)* $crate::__scad_parent!($p ( $($a)* ) { $($body)* }),] $($rest)*)
    };
    // any other node is a `<expr>;` leaf — a primitive call, a fluent chain, a
    // `solid(x)` splice, or a bare `Solid` variable. Evaluated in the caller's
    // scope, so primitives just need to be in scope (`use threers::{cube, …}`).
    ([$($acc:expr,)*] $e:expr ; $($rest:tt)*) => {
        $crate::__scad_list!([$($acc,)* $e,] $($rest)*)
    };
}

/// Internal: dispatch a `keyword(args) { children }` node to its builder.
#[doc(hidden)]
#[macro_export]
macro_rules! __scad_parent {
    (union ( ) { $($b:tt)* }) => { $crate::union_all($crate::__scad_list!([] $($b)*)) };
    (difference ( ) { $($b:tt)* }) => { $crate::difference_all($crate::__scad_list!([] $($b)*)) };
    (intersection ( ) { $($b:tt)* }) => { $crate::intersection_all($crate::__scad_list!([] $($b)*)) };
    (hull ( ) { $($b:tt)* }) => { $crate::hull($crate::__scad_list!([] $($b)*)) };
    (translate ( $t:expr ) { $($b:tt)* }) => { $crate::__scad_group!($($b)*).translate($t) };
    (rotate ( $t:expr ) { $($b:tt)* })    => { $crate::__scad_group!($($b)*).rotate($t) };
    (scale ( $t:expr ) { $($b:tt)* })     => { $crate::__scad_group!($($b)*).scale($t) };
    (mirror ( $t:expr ) { $($b:tt)* })    => { $crate::__scad_group!($($b)*).mirror($t) };
}

/// n-ary union — `union![a, b, c]` or `union!(vec_of_solids)`.
///
/// The comma form folds its arguments; the single-argument form takes any
/// `IntoIterator<Item = Solid>` (a `Vec`, an iterator, …), so it pairs naturally
/// with a `for`/`map` that builds a list of parts.
#[macro_export]
macro_rules! union {
    ($iter:expr $(,)?) => { $crate::union_all(::std::iter::IntoIterator::into_iter($iter).collect()) };
    ($first:expr, $($rest:expr),+ $(,)?) => { $crate::union_all(::std::vec![$first, $($rest),+]) };
}

/// n-ary difference — `difference![base, hole1, hole2]` (`base − rest…`), or
/// `difference!(vec_of_solids)`.
#[macro_export]
macro_rules! difference {
    ($iter:expr $(,)?) => { $crate::difference_all(::std::iter::IntoIterator::into_iter($iter).collect()) };
    ($first:expr, $($rest:expr),+ $(,)?) => { $crate::difference_all(::std::vec![$first, $($rest),+]) };
}

/// n-ary intersection — `intersection![a, b, c]` or `intersection!(vec_of_solids)`.
#[macro_export]
macro_rules! intersection {
    ($iter:expr $(,)?) => { $crate::intersection_all(::std::iter::IntoIterator::into_iter($iter).collect()) };
    ($first:expr, $($rest:expr),+ $(,)?) => { $crate::intersection_all(::std::vec![$first, $($rest),+]) };
}

/// Convex hull of solids — `hull![a, b, c]` or `hull!(vec_of_solids)`.
#[macro_export]
macro_rules! hull {
    ($iter:expr $(,)?) => { $crate::hull(::std::iter::IntoIterator::into_iter($iter).collect()) };
    ($first:expr, $($rest:expr),+ $(,)?) => { $crate::hull(::std::vec![$first, $($rest),+]) };
}

#[cfg(test)]
mod tests {
    use crate::{cube, cylinder, difference_all, solid, sphere, Solid};

    /// Signed volume of a mesh (Σ of tetrahedron volumes over its triangles);
    /// magnitude is orientation-independent, so it's a robust model fingerprint.
    fn vol(s: Solid) -> f32 {
        let g = s.to_geometry_exact();
        let pos = match g.get_attribute("position") {
            Some(a) => a.array.clone(),
            None => return 0.0,
        };
        let v = |i: usize| [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]];
        let tri = |a: [f32; 3], b: [f32; 3], c: [f32; 3]| {
            (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0
        };
        let mut total = 0.0;
        if let Some(idx) = &g.index {
            for t in idx.chunks_exact(3) {
                total += tri(v(t[0] as usize), v(t[1] as usize), v(t[2] as usize));
            }
        } else {
            for k in 0..pos.len() / 9 {
                total += tri(v(k * 3), v(k * 3 + 1), v(k * 3 + 2));
            }
        }
        total.abs()
    }

    #[test]
    fn scad_matches_fluent() {
        // The declarative tree lowers to exactly the fluent builders, so the two
        // spellings of the same model must have the same volume.
        let fluent = cube([20.0, 20.0, 20.0]).difference(cylinder(30.0, 6.0));
        let declarative = scad! {
            difference() {
                cube([20.0, 20.0, 20.0]);
                cylinder(30.0, 6.0);
            }
        };
        let (a, b) = (vol(fluent), vol(declarative));
        assert!((a - b).abs() < 1.0, "scad! {b} != fluent {a}");
        // And it's actually a box with a hole: less than the solid 8000.
        assert!(b < 8000.0 && b > 5000.0, "unexpected drilled-box volume {b}");
    }

    #[test]
    fn nested_and_implicit_union() {
        // Nested ops + a transform over multiple children (implicit union).
        let m = scad! {
            union() {
                cube([10.0, 10.0, 10.0]);
                translate([20.0, 0.0, 0.0]) {
                    sphere(5.0);
                    translate([0.0, 0.0, 8.0]) { sphere(5.0); }
                }
            }
        };
        // cube (1000) + two spheres (~523 each), all disjoint → additive.
        let v = vol(m);
        assert!((v - (1000.0 + 2.0 * 523.6)).abs() < 60.0, "volume {v}");
    }

    #[test]
    fn escape_hatch_splices_rust_values() {
        // Build a part with a Rust loop, then splice it into a `scad!` tree.
        let pins: Vec<Solid> = (0..4)
            .map(|i| cube([2.0, 2.0, 2.0]).translate([i as f32 * 6.0, 0.0, 0.0]))
            .collect();
        let spliced = scad! {
            union() {
                solid(union!(pins));
                translate([0.0, 10.0, 0.0]) { cube([2.0, 2.0, 2.0]); }
            }
        };
        // 4 pins + 1 extra cube, all disjoint 2³ boxes → 5 × 8 = 40.
        let v = vol(spliced);
        assert!((v - 40.0).abs() < 1.0, "spliced volume {v}");
    }

    #[test]
    fn variadic_boolean_macros() {
        // `difference![base, holes…]` == difference_all(vec![…]).
        let holes = [
            cylinder(30.0, 2.0).translate([-6.0, 0.0, 0.0]),
            cylinder(30.0, 2.0).translate([6.0, 0.0, 0.0]),
        ];
        let macro_form = difference![cube([20.0, 20.0, 20.0]), holes[0].clone(), holes[1].clone()];
        let fn_form = difference_all(vec![
            cube([20.0, 20.0, 20.0]),
            holes[0].clone(),
            holes[1].clone(),
        ]);
        assert!((vol(macro_form) - vol(fn_form)).abs() < 1.0);
    }

    #[test]
    fn hull_of_solids() {
        // Convex hull of two spaced cubes: a solid enclosing both, so its volume
        // exceeds the two 8-unit cubes alone (the connecting "bridge" fills in).
        let both = vol(cube([2.0, 2.0, 2.0]).union(cube([2.0, 2.0, 2.0]).translate([8.0, 0.0, 0.0])));
        let hulled = vol(scad! {
            hull() {
                cube([2.0, 2.0, 2.0]);
                translate([8.0, 0.0, 0.0]) { cube([2.0, 2.0, 2.0]); }
            }
        });
        assert!(hulled > both + 20.0, "hull {hulled} should exceed the two cubes {both}");
        // The `hull!` macro agrees with the scad! form.
        let m = vol(hull![cube([2.0, 2.0, 2.0]), cube([2.0, 2.0, 2.0]).translate([8.0, 0.0, 0.0])]);
        assert!((m - hulled).abs() < 1.0, "hull! {m} != scad! hull {hulled}");
    }

    #[test]
    fn mirror_and_rotate_preserve_volume() {
        // Rigid motions and reflections don't change volume.
        let base = cube([8.0, 4.0, 2.0]);
        assert!((vol(base.clone()) - 64.0).abs() < 1e-2);
        assert!((vol(base.clone().rotate([37.0, 12.0, 90.0])) - 64.0).abs() < 1e-2);
        assert!((vol(base.clone().rotate_axis(45.0, [1.0, 1.0, 0.0])) - 64.0).abs() < 1e-2);
        assert!((vol(base.mirror([1.0, 0.0, 0.0])) - 64.0).abs() < 1e-2);
    }
}
