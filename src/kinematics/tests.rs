use super::*;

/// A serial chain of revolute joints, as the forward kinematics to solve
/// against. Joint `i` turns about `axes[i]` in its own frame and is followed by
/// a translation `links[i]`; the tool sits `tool_at` beyond the last link and
/// points along `tool_dir`.
struct Chain {
    axes: Vec<V3>,
    links: Vec<V3>,
    tool_at: V3,
    tool_dir: V3,
}

type M3 = [[f64; 3]; 3];

fn rot(axis: V3, deg: f64) -> M3 {
    let a = unit(axis);
    let (s, c) = deg.to_radians().sin_cos();
    let t = 1.0 - c;
    [
        [
            t * a[0] * a[0] + c,
            t * a[0] * a[1] - s * a[2],
            t * a[0] * a[2] + s * a[1],
        ],
        [
            t * a[0] * a[1] + s * a[2],
            t * a[1] * a[1] + c,
            t * a[1] * a[2] - s * a[0],
        ],
        [
            t * a[0] * a[2] - s * a[1],
            t * a[1] * a[2] + s * a[0],
            t * a[2] * a[2] + c,
        ],
    ]
}

fn mul(a: M3, b: M3) -> M3 {
    let mut o = [[0.0; 3]; 3];
    for (r, row) in o.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    o
}

fn apply(m: M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

impl Chain {
    fn tool(&self, q: &[f64]) -> (V3, V3) {
        let mut r: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut p: V3 = [0.0; 3];
        for ((axis, &angle), link) in self.axes.iter().zip(q.iter()).zip(self.links.iter()) {
            r = mul(r, rot(*axis, angle));
            let d = apply(r, *link);
            p = [p[0] + d[0], p[1] + d[1], p[2] + d[2]];
        }
        let t = apply(r, self.tool_at);
        (
            [p[0] + t[0], p[1] + t[1], p[2] + t[2]],
            apply(r, self.tool_dir),
        )
    }

    /// Three joints about +Y with links along +Z: everything happens in the XZ
    /// plane, so the answers can be checked by hand. Reach 900, and the first
    /// two links alone reach 700 — which is what decides whether a given point
    /// can be approached straight down.
    fn planar() -> Self {
        Chain {
            axes: vec![[0.0, 1.0, 0.0]; 3],
            links: vec![[0.0, 0.0, 400.0], [0.0, 0.0, 300.0], [0.0, 0.0, 200.0]],
            tool_at: [0.0; 3],
            tool_dir: [0.0, 0.0, 1.0],
        }
    }

    /// Seven joints, alternating roll about the link and pitch across it — the
    /// usual shape of a redundant arm, and the case IK exists for.
    ///
    /// Every roll axis lies along the link that follows it, so the rolls move
    /// no metal by themselves and this reduces, for a target in the XZ plane,
    /// to a planar 3R arm of 600, 600 and 750 standing on a 300 riser. That is
    /// what makes its envelope checkable by hand: a point can be approached
    /// straight down only if it sits 750 below somewhere the first two segments
    /// can put the wrist.
    fn redundant() -> Self {
        Chain {
            axes: (0..7)
                .map(|i| {
                    if i % 2 == 0 {
                        [0.0, 0.0, 1.0]
                    } else {
                        [0.0, 1.0, 0.0]
                    }
                })
                .collect(),
            links: vec![[0.0, 0.0, 300.0]; 7],
            tool_at: [0.0, 0.0, 150.0],
            tool_dir: [0.0, 0.0, 1.0],
        }
    }

    /// Two joints and no links: the tool point never moves, so only the
    /// orientation residual can drive a solve. Isolates what a position error
    /// would otherwise mask.
    fn gimbal() -> Self {
        Chain {
            axes: vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
            links: vec![[0.0; 3], [0.0; 3]],
            tool_at: [0.0; 3],
            tool_dir: [0.0, 0.0, 1.0],
        }
    }
}

const DOWN: V3 = [0.0, 0.0, -1.0];

#[test]
fn reaches_a_point_it_can_reach() {
    let c = Chain::planar();
    let s = Ik::default().solve(
        &[10.0, 20.0, 30.0],
        &Goal::Point([500.0, 0.0, 300.0]),
        |q| c.tool(q),
    );
    assert_eq!(s.stop, Stop::Converged, "{s:?}");
    assert!(s.position_err < 1e-3, "{s:?}");
}

/// When a square approach is possible it must come out square AND on target,
/// not one traded against the other. This is what separate residual rows buy,
/// and what a weighted scalar cost gets wrong.
#[test]
fn holds_the_tool_square_while_reaching() {
    let c = Chain::planar();
    // Wrist lands at [400,0,400], 566 from a base that reaches 700, so a
    // vertical approach genuinely exists here.
    let g = Goal::PointAlong {
        at: [400.0, 0.0, 200.0],
        along: DOWN,
    };
    let s = Ik::default().solve(&[10.0, 20.0, 30.0], &g, |q| c.tool(q));
    assert_eq!(s.stop, Stop::Converged, "{s:?}");
    assert!(s.within(1e-3, 1e-3), "{s:?}");
}

/// When a square approach is NOT possible the miss has to be reported rather
/// than absorbed by leaning, and `rot_weight` is the dial that says who pays.
///
/// The companion `Goal::Point` solve proves the point itself is reachable, so
/// everything the constrained solve gives up is the constraint talking and not
/// a short arm.
#[test]
fn an_infeasible_approach_reports_a_miss() {
    let c = Chain::planar();
    let at = [690.0, 0.0, 100.0];
    let seed = [10.0, 20.0, 30.0];

    let free = Ik::default().solve(&seed, &Goal::Point(at), |q| c.tool(q));
    assert!(free.position_err < 1e-3, "reachable when free: {free:?}");

    // Straight down here needs the wrist at [690,0,300] — 752.4 from a base
    // that stretches 700, so it is 52.4 short and no setting can conjure that
    // away.
    let lean = Ik::default().solve(&seed, &Goal::PointAlong { at, along: DOWN }, |q| c.tool(q));
    let square = Ik {
        rot_weight: 40_000.0,
        ..Default::default()
    }
    .solve(&seed, &Goal::PointAlong { at, along: DOWN }, |q| c.tool(q));

    assert!(
        lean.position_err > 10.0,
        "must not claim it arrived: {lean:?}"
    );
    assert!(
        square.axis_err < lean.axis_err,
        "weight must tighten the axis"
    );
    assert!(square.axis_err < 0.01, "{square:?}");
    // Held exactly square, the miss is the geometric shortfall and nothing else.
    assert!(
        (square.position_err - 52.4).abs() < 0.5,
        "expected the 52.4 shortfall, got {square:?}"
    );
}

/// Why the orientation residual is a difference of unit vectors and not the
/// cross product the textbook reaches for.
///
/// `axis x along` vanishes when the tool points exactly BACKWARDS as surely as
/// when it points the right way, so a solver built on it reads a reversed tool
/// as solved and never turns it round. This chain has no position error to
/// mask that, so the seed here is exactly the trap.
#[test]
fn escapes_an_axis_pointing_exactly_backwards() {
    let g = Chain::gimbal();
    let (_, a) = g.tool(&[0.0, 0.0]);
    assert!(
        dot(a, DOWN) < -0.999,
        "the seed must be anti-parallel to be a test"
    );

    // The trap made concrete rather than asserted in a comment: at this pose
    // the cross-product residual is identically zero — a converged answer,
    // pointing 180 degrees wrong — while the difference residual sits at its
    // maximum of 2 and pushes hardest.
    let cross = [
        a[1] * DOWN[2] - a[2] * DOWN[1],
        a[2] * DOWN[0] - a[0] * DOWN[2],
        a[0] * DOWN[1] - a[1] * DOWN[0],
    ];
    assert_eq!(norm(cross), 0.0);
    assert!((norm([a[0] - DOWN[0], a[1] - DOWN[1], a[2] - DOWN[2]]) - 2.0).abs() < 1e-12);

    let s = Ik::default().solve(
        &[0.0, 0.0],
        &Goal::PointAlong {
            at: [0.0; 3],
            along: DOWN,
        },
        |q| g.tool(q),
    );
    assert_eq!(s.stop, Stop::Converged, "{s:?}");
    assert!(s.axis_err < 1e-3, "{s:?}");
}

/// Seeding each station from the last is what makes a sequence of solves a
/// trajectory. A redundant arm has a continuum of answers at every station, so
/// if consecutive solves picked from it independently the arm would snap
/// between elbow-up and elbow-down mid-move with every pose looking correct.
#[test]
fn consecutive_solves_stay_on_one_branch() {
    let c = Chain::redundant();
    let ik = Ik::default();
    let station = |x: f64| Goal::PointAlong {
        at: [x, 0.0, 300.0],
        along: DOWN,
    };

    // Settle onto a branch first; the seed below is arbitrary and the jump onto
    // the arc is not what this is measuring.
    let mut q = ik
        .solve(
            &[20.0, -40.0, 15.0, 50.0, -10.0, 30.0, 5.0],
            &station(300.0),
            |q| c.tool(q),
        )
        .joints;

    let (mut worst_step, mut worst_pos, mut worst_axis) = (0.0f64, 0.0f64, 0.0f64);
    for i in 1..=40 {
        let s = ik.solve(&q, &station(300.0 + 500.0 * i as f64 / 40.0), |q| c.tool(q));
        for (a, b) in q.iter().zip(&s.joints) {
            worst_step = worst_step.max((a - b).abs());
        }
        worst_pos = worst_pos.max(s.position_err);
        worst_axis = worst_axis.max(s.axis_err);
        q = s.joints;
    }
    assert!(worst_pos < 1e-3, "worst position error {worst_pos}");
    assert!(worst_axis < 1e-3, "worst axis error {worst_axis}");
    // 12.5 apart: a branch change shows up here as tens of degrees between two
    // neighbouring stations, not the couple this needs.
    assert!(
        worst_step < 5.0,
        "joint jumped {worst_step:.1} deg between stations"
    );
}

#[test]
fn never_leaves_the_joint_limits() {
    let c = Chain::redundant();
    let ik = Ik {
        limits: (-45.0, 45.0),
        ..Default::default()
    };
    // Out of reach for so tight a range, to make it push against them.
    let g = Goal::PointAlong {
        at: [1800.0, 400.0, 200.0],
        along: DOWN,
    };
    let s = ik.solve(&[0.0; 7], &g, |q| c.tool(q));
    for (i, v) in s.joints.iter().enumerate() {
        assert!(
            (-45.0..=45.0).contains(v),
            "joint {i} at {v} left its limits"
        );
    }
}

/// Never report success it did not achieve. The arm reaches 900 and the target
/// is 5000 out, so the only honest answer is the fully extended pose and a
/// 4100 miss.
#[test]
fn admits_when_a_target_is_out_of_reach() {
    let c = Chain::planar();
    let s = Ik::default().solve(&[10.0, 20.0, 30.0], &Goal::Point([5000.0, 0.0, 0.0]), |q| {
        c.tool(q)
    });
    assert_ne!(s.stop, Stop::Converged, "{s:?}");
    assert!((s.position_err - 4100.0).abs() < 0.1, "{s:?}");
    // Extended straight at it: 90 degrees at the shoulder, nothing after.
    assert!((s.joints[0] - 90.0).abs() < 0.5, "{s:?}");
    assert!(s.joints[1].abs() < 0.5 && s.joints[2].abs() < 0.5, "{s:?}");
}

/// A seed at a singularity — this chain fully extended, where four of its seven
/// joints move nothing at all — is the caller's mistake to avoid, but it must
/// degrade rather than lie. It turns the tool over; it does not quite arrive,
/// and it says which by not reporting `Converged`.
#[test]
fn a_singular_seed_degrades_without_lying() {
    let c = Chain::redundant();
    let g = Goal::PointAlong {
        at: [600.0, 0.0, 300.0],
        along: DOWN,
    };
    let s = Ik {
        max_iters: 200,
        ..Default::default()
    }
    .solve(&[0.0; 7], &g, |q| c.tool(q));
    assert_ne!(s.stop, Stop::Converged, "{s:?}");
    assert!(s.axis_err < 1.0, "should still turn the tool over: {s:?}");
    assert!(s.position_err < 25.0, "{s:?}");
}

#[test]
fn a_point_goal_leaves_the_axis_unreported() {
    let c = Chain::planar();
    let s = Ik::default().solve(
        &[10.0, 20.0, 30.0],
        &Goal::Point([500.0, 0.0, 300.0]),
        |q| c.tool(q),
    );
    assert_eq!(
        s.axis_err, 0.0,
        "an unconstrained axis has no error to report"
    );
    assert!(s.within(1e-3, 0.0));
}

#[test]
fn an_unnormalised_goal_axis_is_fine() {
    let c = Chain::planar();
    let seed = [10.0, 20.0, 30.0];
    let at = [400.0, 0.0, 200.0];
    let a = Ik::default().solve(
        &seed,
        &Goal::PointAlong {
            at,
            along: [0.0, 0.0, -37.5],
        },
        |q| c.tool(q),
    );
    let b = Ik::default().solve(&seed, &Goal::PointAlong { at, along: DOWN }, |q| c.tool(q));
    for (x, y) in a.joints.iter().zip(&b.joints) {
        assert!((x - y).abs() < 1e-9, "{a:?} vs {b:?}");
    }
}
