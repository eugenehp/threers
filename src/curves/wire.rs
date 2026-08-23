//! Wires and cables: the curves a slack conductor actually takes.
//!
//! # Which curve is right
//!
//! A wire's shape is decided by whichever energy dominates it, and there are two
//! different answers depending on where it is.
//!
//! **On the ground**, a slack cable is heavy and limp: gravity dominates and
//! bending stiffness is negligible. It hangs in a [`Catenary3`] — `cosh`, the
//! shape of a chain, exact for an inextensible line with no bending resistance.
//! Everyone knows this one, and for a spacecraft harness it is **wrong**.
//!
//! **In free fall** there is no gravity to hang from. What is left is the
//! conductor's own bending stiffness, and the cable settles into the shape that
//! minimises stored bending energy subject to where its ends are clamped:
//!
//! ```text
//!     minimise  ∫ κ² ds     with p, t fixed at both ends and ∫ ds fixed
//! ```
//!
//! That is Euler's **elastica** — [`Elastica3`] — and it is the curve a harness
//! takes between tie-downs on orbit. It is also the curve one takes on a bench
//! whenever it is stiff enough that its own weight does not matter, which for a
//! shielded bundle over a short span is most of the time.
//!
//! The difference is not cosmetic. A catenary sags toward gravity and has its
//! maximum curvature at the low point; an elastica is symmetric about the chord
//! and puts its curvature wherever the end tangents demand. Routing clearance,
//! the swept volume across a moving joint, and the bending radius the conductor
//! must survive all come off the wrong curve if the wrong one is used.
//!
//! # Why not just a spline
//!
//! A Catmull-Rom or Bézier through waypoints is a modelling convenience: it
//! interpolates what someone drew. It has no relationship to what a cable does
//! between those points, and it will happily produce a bend radius no conductor
//! could take. The elastica is the curve the physics picks; the spline is the
//! curve the mouse picked. Both are useful, and they are not the same object.
//!
//! # Slack
//!
//! Arc length is an input, not an output. A cable cut to exactly the distance
//! between its ends is a straight line under tension, which is the one thing a
//! harness must never be: it has to survive the joint moving, thermal
//! contraction, and assembly tolerance. `slack` is that extra length, and it is
//! what makes the curve bow at all.

use crate::curves::Curve3;
use crate::math::Vector3;

/// A cable hanging under gravity: `y = a·cosh(x/a)`.
///
/// The classical answer, and the right one only when weight dominates bending
/// stiffness. Kept because it is correct on the ground and because having it
/// beside [`Elastica3`] makes the choice explicit rather than accidental.
///
/// `sag` is the drop at mid-span below the chord, in model units. The shape
/// parameter `a` follows from it and the span; there is no closed form for `a`
/// given sag, so it is found by bisection, which converges in a few dozen steps
/// and is done once at construction.
pub struct Catenary3 {
    p0: Vector3,
    p1: Vector3,
    /// Direction the cable hangs, normalised. Usually -Y.
    down: Vector3,
    a: f32,
    span: f32,
}

impl Catenary3 {
    pub fn new(p0: Vector3, p1: Vector3, down: Vector3, sag: f32) -> Self {
        let chord = p1 - p0;
        let down = if down.length() > 1e-9 {
            down.normalize()
        } else {
            Vector3::new(0.0, -1.0, 0.0)
        };
        // horizontal component of the span, i.e. across the hang direction
        let along = chord - down * chord.dot(down);
        let span = along.length().max(1e-6);
        // sag(a) = a·(cosh(span/2a) − 1); monotonically DECREASING in a, so
        // bisect on a between something taut and something very slack.
        let (mut lo, mut hi) = (1e-3f32, span * 1e3);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let s = mid * ((span / (2.0 * mid)).cosh() - 1.0);
            if s > sag {
                lo = mid
            } else {
                hi = mid
            }
        }
        Self {
            p0,
            p1,
            down,
            a: 0.5 * (lo + hi),
            span,
        }
    }
}

impl Curve3 for Catenary3 {
    fn get_point(&self, t: f32) -> Vector3 {
        let t = t.clamp(0.0, 1.0);
        let chord = self.p1 - self.p0;
        let straight = self.p0 + chord * t;
        // depth below the chord at parameter t, zero at both ends
        let x = (t - 0.5) * self.span;
        let d = self.a * ((x / self.a).cosh() - (self.span / (2.0 * self.a)).cosh());
        straight + self.down * (-d)
    }
}

/// Euler's elastica: the minimum-bending-energy curve between two clamped ends.
///
/// This is the shape a stiff cable takes when gravity is not what is holding it
/// up — a harness on orbit, or any span short enough that the conductor's own
/// stiffness beats its weight.
///
/// # How it is solved
///
/// Not in closed form. The exact elastica is a Jacobi elliptic function, and
/// inverting it for "these two endpoints, these two tangents, this arc length"
/// is a shooting problem with no reliable single solution. What is done instead
/// is the same thing a cable solver does: discretise into segments of fixed
/// length, clamp the ends, and relax.
///
/// Each iteration does two things in turn:
///
///   1. **smooth** — move every interior node toward the midpoint of its
///      neighbours, which is the gradient of the discrete bending energy
///      `Σ |p[i-1] − 2p[i] + p[i+1]|²`;
///   2. **project** — restore every segment to its rest length, twice per
///      iteration in opposite directions so the error does not pile up at one
///      end.
///
/// That is position-based dynamics with one constraint, and it converges to the
/// discrete elastica. It is stable at any slack, including zero, where it
/// straightens to the chord as it should.
///
/// The tangent clamps are enforced by holding the FIRST TWO and LAST TWO nodes:
/// a node pair fixes both a position and a direction, which is what "clamped"
/// means. Holding only the end node would leave the tangent free and give the
/// pinned elastica instead — a different curve, and the wrong one for a cable
/// leaving a connector backshell.
pub struct Elastica3 {
    pts: Vec<Vector3>,
}

impl Elastica3 {
    /// `slack` is arc length BEYOND the straight-line distance, in model units.
    /// Zero gives the chord; a harness wants a few percent of span.
    ///
    /// `t0` and `t1` point *along* the cable, out of each end: the tangent at
    /// the start and the tangent at the finish, both in the direction of travel.
    pub fn clamped(
        p0: Vector3,
        t0: Vector3,
        p1: Vector3,
        t1: Vector3,
        slack: f32,
        samples: usize,
    ) -> Self {
        let n = samples.max(8);
        let chord = (p1 - p0).length();
        let total = chord + slack.max(0.0);
        let seg = total / (n - 1) as f32;
        let t0 = if t0.length() > 1e-9 {
            t0.normalize()
        } else {
            (p1 - p0).normalize()
        };
        let t1 = if t1.length() > 1e-9 {
            t1.normalize()
        } else {
            (p1 - p0).normalize()
        };

        // A STRAIGHT SLACK CABLE IS AN UNSTABLE EQUILIBRIUM.
        //
        // With both tangents along the chord the Hermite start IS the chord, and
        // a straight line is a minimum of bending energy: the smoothing step has
        // nothing to correct, so the relaxation never picks a direction to bow
        // into and the extra length has nowhere to go. Asked for 130 mm between
        // points 100 apart, the first version of this returned 100 mm of arc and
        // a dead-straight cable.
        //
        // That is Euler buckling, and it is physics rather than a solver defect:
        // the straight state is an equilibrium, just not a stable one. A real
        // cable leaves it the moment anything is not perfectly symmetric. So the
        // seed carries a half-sine bulge across the chord, scaled by the slack,
        // which is the first buckling mode and the shape it actually takes.
        //
        // Direction: whatever the end tangents lean toward, or any perpendicular
        // if they lean nowhere.
        let chord_dir = if chord > 1e-9 {
            (p1 - p0) * (1.0 / chord)
        } else {
            t0
        };
        let lean = (t0 - chord_dir * t0.dot(chord_dir)) + (t1 - chord_dir * t1.dot(chord_dir));
        let bulge = if lean.length() > 1e-6 {
            lean.normalize()
        } else {
            let seed = if chord_dir.x.abs() < 0.9 {
                Vector3::new(1.0, 0.0, 0.0)
            } else {
                Vector3::new(0.0, 1.0, 0.0)
            };
            (seed - chord_dir * seed.dot(chord_dir)).normalize()
        };
        // Amplitude of a half-sine whose arc length exceeds its chord by `slack`:
        // for y = A sin(pi x / L), the excess is about pi^2 A^2 / (4 L).
        let amp = (4.0 * slack.max(0.0) * chord.max(1e-6)
            / (std::f32::consts::PI * std::f32::consts::PI))
            .sqrt();

        // Start from a cubic Hermite with the given end tangents, plus that
        // buckling seed.
        let scale = total;
        let mut pts: Vec<Vector3> = (0..n)
            .map(|i| {
                let u = i as f32 / (n - 1) as f32;
                let (u2, u3) = (u * u, u * u * u);
                let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
                let h10 = u3 - 2.0 * u2 + u;
                let h01 = -2.0 * u3 + 3.0 * u2;
                let h11 = u3 - u2;
                let base = p0 * h00 + t0 * (h10 * scale) + p1 * h01 + t1 * (h11 * scale);
                base + bulge * (amp * (u * std::f32::consts::PI).sin())
            })
            .collect();

        // The four clamped nodes: ends, and one in from each end set by the
        // tangent. Those two are what make it clamped rather than pinned.
        let a0 = p0;
        let a1 = p0 + t0 * seg;
        let b1 = p1 - t1 * seg;
        let b0 = p1;

        // ITERATIONS SCALE WITH n². Smoothing is diffusion, and information
        // crosses a chain of n nodes in O(n²) sweeps -- a fixed budget converges
        // a coarse curve and leaves a fine one half-relaxed. The symptom is a
        // shape that changes with the sample count: this returned bend radii of
        // 5.25, 5.95 and 3.62 mm for the same wire at 40, 80 and 160 samples,
        // which is not a wire with three shapes, it is a solver stopping early.
        let iters = (n * n / 2).clamp(2000, 200_000);
        for _ in 0..iters {
            // 1. bending: Laplacian smoothing of the interior
            let prev = pts.clone();
            for i in 1..n - 1 {
                let mid = (prev[i - 1] + prev[i + 1]) * 0.5;
                // 0.08, not 0.35. Smoothing SHORTENS the curve and the length
                // projection lengthens it; run them at comparable strength and
                // they settle at a compromise that is neither -- 119.7 mm of arc
                // where 130 was asked for. A cable is inextensible: length is a
                // hard constraint and bending is what relaxes inside it, so the
                // smoothing has to be the weaker of the two by a wide margin.
                pts[i] = prev[i] + (mid - prev[i]) * 0.08;
            }
            // 2. clamps
            pts[0] = a0;
            pts[1] = a1;
            pts[n - 2] = b1;
            pts[n - 1] = b0;
            // 3. inextensibility, swept both ways so the residual does not
            //    accumulate at one end
            for _ in 0..12 {
                for i in 1..n {
                    let d = pts[i] - pts[i - 1];
                    let l = d.length().max(1e-9);
                    let fix = d * ((l - seg) / l);
                    if i > 1 && i < n - 1 {
                        pts[i] = pts[i] - fix * 0.5;
                        pts[i - 1] = pts[i - 1] + fix * 0.5;
                    } else if i == 1 {
                        pts[i] = pts[i] - fix;
                    } else {
                        pts[i - 1] = pts[i - 1] + fix;
                    }
                }
                pts[0] = a0;
                pts[1] = a1;
                pts[n - 2] = b1;
                pts[n - 1] = b0;
            }
        }
        Self { pts }
    }

    /// The relaxed polyline, if the caller wants the samples rather than the
    /// curve — a physics solver seeding a cable, for instance.
    pub fn points(&self) -> &[Vector3] {
        &self.pts
    }

    /// Tightest bend on the curve, as a RADIUS. The number a conductor is
    /// specified against: every wire has a minimum bend radius, usually a
    /// multiple of its diameter, and violating it is a fatigue failure rather
    /// than an immediate one — which is exactly the sort of thing that does not
    /// show up in a render.
    /// `baseline` is the arc length the curvature is measured OVER, and it is
    /// not optional.
    ///
    /// Measured between adjacent nodes this returned 1.60 mm at 40 samples and
    /// 0.66 at 80 -- halving with the spacing, which is the signature of noise
    /// rather than curvature. A relaxed polyline has a fraction of a degree of
    /// wobble left at every node; over a short enough baseline that wobble IS
    /// the answer, and the number reported is the discretisation, not the wire.
    ///
    /// A conductor does not care about a wobble finer than its own diameter, so
    /// the baseline to use is a small multiple of the outside diameter. Then the
    /// number means what it says and does not move when the sampling changes.
    pub fn min_bend_radius_over(&self, baseline: f32) -> f32 {
        let n = self.pts.len();
        if n < 3 {
            return f32::INFINITY;
        }
        // how many nodes span the baseline
        let seg = (self.pts[1] - self.pts[0]).length().max(1e-9);
        let k = ((baseline / seg).round() as usize).clamp(1, (n - 1) / 2);
        let mut worst = f32::INFINITY;
        for i in k..n - k {
            let (a, b, c) = (self.pts[i - k], self.pts[i], self.pts[i + k]);
            let (u, v) = (b - a, c - b);
            let (lu, lv) = (u.length(), v.length());
            if lu < 1e-9 || lv < 1e-9 {
                continue;
            }
            let cross = u.cross(v).length();
            if cross < 1e-12 {
                continue;
            }
            let r = (lu * lv * (c - a).length()) / (2.0 * cross);
            if r < worst {
                worst = r;
            }
        }
        worst
    }

    /// Adjacent-node curvature. Kept for the polyline itself; for a WIRE use
    /// [`Elastica3::min_bend_radius_over`] with the conductor's diameter, or this number
    /// will change every time the sample count does.
    pub fn min_bend_radius(&self) -> f32 {
        let mut worst = f32::INFINITY;
        for w in self.pts.windows(3) {
            let (a, b, c) = (w[0], w[1], w[2]);
            let (u, v) = (b - a, c - b);
            let (lu, lv) = (u.length(), v.length());
            if lu < 1e-9 || lv < 1e-9 {
                continue;
            }
            // menger curvature of the triple
            let cross = u.cross(v).length();
            if cross < 1e-12 {
                continue;
            }
            let r = (lu * lv * (c - a).length()) / (2.0 * cross);
            if r < worst {
                worst = r;
            }
        }
        worst
    }
}

impl Curve3 for Elastica3 {
    fn get_point(&self, t: f32) -> Vector3 {
        let n = self.pts.len();
        let s = t.clamp(0.0, 1.0) * (n - 1) as f32;
        let i = (s.floor() as usize).min(n - 1);
        let j = (i + 1).min(n - 1);
        let f = s - i as f32;
        self.pts[i] + (self.pts[j] - self.pts[i]) * f
    }
}

/// A cable crossing a joint that turns: the service loop.
///
/// A conductor cannot cross a revolute joint on a straight line — the joint
/// would stretch it. There are exactly two ways to do it and they are different
/// curves:
///
///   * a **loop** in the plane of rotation, which absorbs the turn as bending
///     and sweeps a large volume as it opens and closes;
///   * a **helix on the axis**, which absorbs the turn as TWIST over its own
///     length and whose envelope is a cylinder — the same cylinder at every
///     joint angle.
///
/// The second is what this builds. Its envelope not changing with the joint
/// angle is the whole point: it can be checked for interference once, statically,
/// instead of swept through every pose. That property is why a spacecraft hinge
/// carries its harness on the pin rather than round it.
///
/// `turns` is how much the joint rotates, in turns; `length` is how much axial
/// run the twist is taken over. Torsion per unit length is `turns / length`, and
/// that — not the total angle — is what the bundle has to survive.
pub struct ServiceHelix3 {
    origin: Vector3,
    axis: Vector3,
    radial: Vector3,
    radius: f32,
    length: f32,
    turns: f32,
}

impl ServiceHelix3 {
    pub fn new(origin: Vector3, axis: Vector3, radius: f32, length: f32, turns: f32) -> Self {
        let axis = if axis.length() > 1e-9 {
            axis.normalize()
        } else {
            Vector3::new(0.0, 0.0, 1.0)
        };
        // any vector not parallel to the axis, to start the radial frame
        let seed = if axis.x.abs() < 0.9 {
            Vector3::new(1.0, 0.0, 0.0)
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };
        let radial = (seed - axis * seed.dot(axis)).normalize();
        Self {
            origin,
            axis,
            radial,
            radius,
            length,
            turns,
        }
    }

    /// Twist rate in degrees per unit length — what the bundle is specified
    /// against, and the number that says whether the capsule is long enough.
    pub fn twist_rate(&self) -> f32 {
        if self.length.abs() < 1e-9 {
            f32::INFINITY
        } else {
            self.turns * 360.0 / self.length
        }
    }
}

impl Curve3 for ServiceHelix3 {
    fn get_point(&self, t: f32) -> Vector3 {
        let t = t.clamp(0.0, 1.0);
        let ang = t * self.turns * std::f32::consts::TAU;
        let bi = self.axis.cross(self.radial);
        self.origin
            + self.axis * (self.length * t)
            + self.radial * (self.radius * ang.cos())
            + bi * (self.radius * ang.sin())
    }
}

// ---------------------------------------------------------------------------
// A wire is two things, not one
// ---------------------------------------------------------------------------

/// A conductor and the insulation around it.
///
/// Modelling a wire as a single tube is modelling the jacket and forgetting the
/// metal, and the two are not interchangeable. They have different diameters,
/// different densities, different colours and different jobs: the conductor
/// carries the current and sets the resistance and the mass, the insulation sets
/// the outside diameter that has to fit through the hole and the colour that
/// says which wire it is. A harness mass estimate off jacket volume at copper
/// density is wrong by the ratio of their areas, which for a fine gauge is most
/// of the wire.
///
/// # Gauge
///
/// AWG is geometric, not arbitrary: each step is a fixed ratio, and
///
/// ```text
///     d(mm) = 0.127 · 92^((36 − n)/39)
/// ```
///
/// so 36 AWG is 0.127 mm by definition and every other size follows. Taking the
/// gauge rather than a radius means the number in the model is the number on the
/// drawing, and the diameter cannot drift from it.
#[derive(Debug, Clone, Copy)]
pub struct WireSpec {
    /// AWG. 20–24 is typical for spacecraft signal and low-power runs.
    pub gauge: f32,
    /// Radial wall thickness of the insulation, in model units. PTFE at this
    /// scale is usually 0.15–0.25 mm.
    pub insulation: f32,
    /// What the jacket says this wire is. Colour is not decoration on a
    /// harness — it is how the right wire gets landed on the right pin.
    pub code: WireCode,
}

/// Standard insulation colours, as meanings rather than as pigments.
///
/// A model that stores "red" has to be read by someone who knows what red meant
/// on this vehicle. A model that stores [`WireCode::Positive`] does not, and it
/// can be checked: a positive lead landing on a return pin is a fault a
/// colour-blind file cannot notice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WireCode {
    /// Power positive. Red, by near-universal convention.
    Positive,
    /// Power return. Black.
    Return,
    /// Chassis / structure bond. Green.
    Bond,
    /// Signal pair, first of two. White.
    SignalA,
    /// Signal pair, second of two. Blue.
    SignalB,
    /// Screen or drain. Bare or grey.
    Screen,
}

impl WireCode {
    /// Linear-space RGB for the jacket, as a renderer wants it.
    pub fn rgb(self) -> [f32; 3] {
        match self {
            WireCode::Positive => [0.55, 0.05, 0.05],
            WireCode::Return => [0.04, 0.04, 0.045],
            WireCode::Bond => [0.06, 0.30, 0.10],
            WireCode::SignalA => [0.80, 0.80, 0.82],
            WireCode::SignalB => [0.08, 0.16, 0.50],
            WireCode::Screen => [0.35, 0.36, 0.38],
        }
    }

    /// The name that would be written on a wire list.
    pub fn name(self) -> &'static str {
        match self {
            WireCode::Positive => "red / +28V",
            WireCode::Return => "black / return",
            WireCode::Bond => "green / bond",
            WireCode::SignalA => "white / signal A",
            WireCode::SignalB => "blue / signal B",
            WireCode::Screen => "grey / screen",
        }
    }
}

impl WireSpec {
    pub fn new(gauge: f32, insulation: f32, code: WireCode) -> Self {
        Self {
            gauge,
            insulation,
            code,
        }
    }

    /// Conductor diameter from the gauge, in mm.
    pub fn conductor_diameter(&self) -> f32 {
        0.127 * 92f32.powf((36.0 - self.gauge) / 39.0)
    }

    pub fn conductor_radius(&self) -> f32 {
        self.conductor_diameter() * 0.5
    }

    /// Outside diameter — what has to fit through the grommet.
    pub fn outer_diameter(&self) -> f32 {
        self.conductor_diameter() + 2.0 * self.insulation
    }

    /// Mass per unit length, in kg/mm for lengths in mm: copper core at
    /// 8960 kg/m³ and PTFE jacket at 2200. Stranded conductors are about 5%
    /// short of solid area for the same nominal gauge, which is included --
    /// the strands do not fill their own envelope.
    pub fn mass_per_mm(&self) -> f32 {
        let rc = self.conductor_radius();
        let ro = self.outer_diameter() * 0.5;
        let a_cu = std::f32::consts::PI * rc * rc * 0.95;
        let a_ins = std::f32::consts::PI * (ro * ro - rc * rc);
        (a_cu * 8960.0 + a_ins * 2200.0) * 1e-9
    }

    /// DC resistance per unit length, ohms per mm, annealed copper at 20 °C.
    /// The reason gauge is a design choice and not a drawing detail: a return
    /// run sized for fit rather than for drop is how a 28 V bus arrives at 26.
    pub fn ohms_per_mm(&self) -> f32 {
        let rc = self.conductor_radius();
        let a = std::f32::consts::PI * rc * rc * 0.95;
        1.68e-8 / (a * 1e-6) * 1e-3
    }

    /// The metal, as a tube along the path.
    pub fn conductor(&self, path: &dyn Curve3, segments: usize) -> crate::BufferGeometry {
        crate::geometries::TubeGeometry::new(path, segments, self.conductor_radius(), 8, false)
    }

    /// The jacket, as a tube along the same path. Drawn as a solid rod rather
    /// than an annulus: nothing sees the inside of an intact insulation, and an
    /// annulus doubles the triangles for a surface that is never visible.
    pub fn jacket(&self, path: &dyn Curve3, segments: usize) -> crate::BufferGeometry {
        crate::geometries::TubeGeometry::new(path, segments, self.outer_diameter() * 0.5, 10, false)
    }
}

/// A pair carrying power out and back.
///
/// One wire does not power anything. A solar panel string needs a positive and
/// a return, and they are routed TOGETHER — twisted or at least adjacent —
/// because the loop area between them is what couples into everything else. A
/// model that draws one lead has drawn half a circuit and none of the loop.
pub fn power_pair(gauge: f32, insulation: f32) -> [WireSpec; 2] {
    [
        WireSpec::new(gauge, insulation, WireCode::Positive),
        WireSpec::new(gauge, insulation, WireCode::Return),
    ]
}
