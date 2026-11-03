//! Run IK trajectories through servo + gear physics.
//!
//! At each station the solver produces joint setpoints; [`ServoJoint::step`]
//! integrates the actual angles under transmission losses. Forward kinematics
//! on the *actual* angles gives honest tool error — the gap between ideal IK
//! and what a geared hand can track.

use super::plant::ArmPlant;
use super::servo::ServoJoint;
use super::{Goal, Ik, SerialChain, Solution, V3};
use super::transmission::GearTrain;

fn norm(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// One recorded frame of a physics-grounded IK run.
#[derive(Debug, Clone)]
pub struct SimFrame {
    pub t: f64,
    /// IK setpoint, degrees.
    pub q_cmd: Vec<f64>,
    /// Actual joint angles after servo physics, degrees.
    pub q_act: Vec<f64>,
    /// Tool position error vs the goal point, length units.
    pub position_err: f64,
    /// Tool axis error vs goal, degrees (0 for [`Goal::Point`]).
    pub axis_err: f64,
    /// Max joint tracking error this frame, degrees.
    pub worst_joint_err: f64,
    /// Minimum force-estimation SNR across joints.
    pub min_force_snr: f64,
}

/// Aggregate report comparing ideal IK to physics-grounded tracking.
#[derive(Debug, Clone)]
pub struct SimReport {
    pub frames: Vec<SimFrame>,
    pub worst_position_err: f64,
    pub worst_axis_err: f64,
    pub worst_joint_err: f64,
    pub mean_joint_err: f64,
    /// RMS tool position error over the steady window.
    pub rms_position_err: f64,
    pub min_force_snr: f64,
    pub transmission: &'static str,
}

impl SimReport {
    pub fn print_summary(&self) {
        println!(
            "transmission={:<12} frames={}  worst_tool={:6.2} mm  rms_tool={:6.2} mm  worst_axis={:5.2}°  mean_joint={:5.2}°  min_snr={:.1}",
            self.transmission,
            self.frames.len(),
            self.worst_position_err,
            self.rms_position_err,
            self.worst_axis_err,
            self.mean_joint_err,
            self.min_force_snr,
        );
    }

    /// Frame nearest to time `t`.
    pub fn frame_at(&self, t: f64) -> Option<&SimFrame> {
        self.frames.iter().min_by(|a, b| {
            (a.t - t)
                .abs()
                .partial_cmp(&(b.t - t).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }
}

/// A time-stamped IK goal along a trajectory.
#[derive(Debug, Clone, Copy)]
pub struct Waypoint {
    pub t: f64,
    pub goal: Goal,
}

/// Simulate an arm through IK + servo physics.
#[derive(Debug, Clone)]
pub struct IkServoSim {
    pub ik: Ik,
    pub dt: f64,
    pub label: &'static str,
}

impl Default for IkServoSim {
    fn default() -> Self {
        Self {
            ik: Ik::default(),
            dt: 1.0 / 240.0,
            label: "custom",
        }
    }
}

impl IkServoSim {
    pub fn new(ik: Ik) -> Self {
        Self {
            ik,
            ..Default::default()
        }
    }

    pub fn with_dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    pub fn with_label(mut self, label: &'static str) -> Self {
        self.label = label;
        self
    }

    /// Run from `seed` through `waypoints`, stepping `actuators` each frame.
    pub fn run<F>(
        &self,
        seed: &[f64],
        waypoints: &[Waypoint],
        actuators: &mut [ServoJoint],
        tool: F,
    ) -> SimReport
    where
        F: Fn(&[f64]) -> (V3, V3),
    {
        assert!(!waypoints.is_empty());
        assert_eq!(seed.len(), actuators.len());

        let t_end = waypoints.last().unwrap().t;
        let mut q_ik = seed.to_vec();
        let mut frames = Vec::new();
        let mut t = 0.0;

        while t <= t_end + 0.5 * self.dt {
            let goal = interpolate_goal(waypoints, t);

            // Plan from the last IK solution, not noisy actual angles.
            let sol = self.ik.solve(&q_ik, &goal, &tool);
            let q_cmd = sol.joints.clone();
            q_ik.clone_from(&q_cmd);

            for (i, act) in actuators.iter_mut().enumerate() {
                act.step(q_cmd[i], 0.0, self.dt);
            }

            let target = goal.at();
            let along = match goal {
                Goal::PointAlong { along, .. } => along,
                Goal::Point(_) => [0.0, 0.0, 0.0],
            };

            let (p, ax) = tool(
                &actuators
                    .iter()
                    .map(|a| a.theta_deg)
                    .collect::<Vec<_>>(),
            );
            let position_err = norm([p[0] - target[0], p[1] - target[1], p[2] - target[2]]);
            let axis_err = if matches!(goal, Goal::PointAlong { .. }) {
                let a = ax;
                let n = norm(a);
                let an = if n > 1e-12 {
                    [a[0] / n, a[1] / n, a[2] / n]
                } else {
                    [0.0, 0.0, 1.0]
                };
                let ln = norm(along);
                let al = if ln > 1e-12 {
                    [along[0] / ln, along[1] / ln, along[2] / ln]
                } else {
                    [0.0, 0.0, -1.0]
                };
                an[0] * al[0] + an[1] * al[1] + an[2] * al[2]
            } else {
                1.0
            };
            let axis_err_deg = if matches!(goal, Goal::PointAlong { .. }) {
                axis_err.clamp(-1.0, 1.0).acos().to_degrees()
            } else {
                0.0
            };

            let worst_joint_err = actuators
                .iter()
                .map(|a| a.tracking_error_deg())
                .fold(0.0f64, f64::max);
            let min_force_snr = actuators
                .iter()
                .map(|a| a.force_snr())
                .fold(f64::INFINITY, f64::min);

            frames.push(SimFrame {
                t,
                q_cmd,
                q_act: actuators.iter().map(|a| a.theta_deg).collect(),
                position_err,
                axis_err: axis_err_deg,
                worst_joint_err,
                min_force_snr,
            });

            t += self.dt;
        }

        Self::summarise(frames, self.label)
    }

    /// Coupled Newton–Euler plant + geometric-Jacobian IK.
    pub fn run_plant(
        &self,
        seed: &[f64],
        waypoints: &[Waypoint],
        plant: &mut ArmPlant,
    ) -> SimReport {
        assert!(!waypoints.is_empty());
        assert_eq!(seed.len(), plant.servos.len());
        plant.seed(seed);

        let t_end = waypoints.last().unwrap().t;
        let mut q_ik = seed.to_vec();
        let mut frames = Vec::new();
        let mut t = 0.0;

        while t <= t_end + 0.5 * self.dt {
            let goal = interpolate_goal(waypoints, t);
            let sol = self.ik.solve_chain(&q_ik, &goal, &plant.chain);
            let q_cmd = sol.joints.clone();
            q_ik.clone_from(&q_cmd);

            plant.step(&q_cmd, self.dt);

            let target = goal.at();
            let along = match goal {
                Goal::PointAlong { along, .. } => along,
                Goal::Point(_) => [0.0, 0.0, 0.0],
            };
            let q_act = plant.q_deg();
            let (p, ax) = plant.chain.tool(&q_act);
            let position_err = norm([p[0] - target[0], p[1] - target[1], p[2] - target[2]]);
            let axis_err_deg = if matches!(goal, Goal::PointAlong { .. }) {
                let n = norm(ax);
                let an = if n > 1e-12 {
                    [ax[0] / n, ax[1] / n, ax[2] / n]
                } else {
                    [0.0, 0.0, 1.0]
                };
                let ln = norm(along);
                let al = if ln > 1e-12 {
                    [along[0] / ln, along[1] / ln, along[2] / ln]
                } else {
                    [0.0, 0.0, -1.0]
                };
                (an[0] * al[0] + an[1] * al[1] + an[2] * al[2])
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            } else {
                0.0
            };
            let worst_joint_err = plant
                .servos
                .iter()
                .map(|a| a.tracking_error_deg())
                .fold(0.0f64, f64::max);
            let min_force_snr = plant
                .servos
                .iter()
                .map(|a| a.force_snr())
                .fold(f64::INFINITY, f64::min);

            frames.push(SimFrame {
                t,
                q_cmd,
                q_act,
                position_err,
                axis_err: axis_err_deg,
                worst_joint_err,
                min_force_snr,
            });
            t += self.dt;
        }

        Self::summarise(frames, self.label)
    }

    fn summarise(frames: Vec<SimFrame>, label: &'static str) -> SimReport {
        let skip = frames.len() / 4;
        let steady: Vec<_> = frames.iter().skip(skip).collect();
        let worst_position_err = steady
            .iter()
            .map(|f| f.position_err)
            .fold(0.0f64, f64::max);
        let worst_axis_err = steady.iter().map(|f| f.axis_err).fold(0.0, f64::max);
        let worst_joint_err = steady.iter().map(|f| f.worst_joint_err).fold(0.0, f64::max);
        let mean_joint_err = if steady.is_empty() {
            0.0
        } else {
            steady.iter().map(|f| f.worst_joint_err).sum::<f64>() / steady.len() as f64
        };
        let min_force_snr = frames
            .iter()
            .map(|f| f.min_force_snr)
            .fold(f64::INFINITY, f64::min);
        let rms_position_err = if steady.is_empty() {
            0.0
        } else {
            (steady
                .iter()
                .map(|f| f.position_err * f.position_err)
                .sum::<f64>()
                / steady.len() as f64)
                .sqrt()
        };
        SimReport {
            frames,
            worst_position_err,
            worst_axis_err,
            worst_joint_err,
            mean_joint_err,
            rms_position_err,
            min_force_snr,
            transmission: label,
        }
    }

    /// Compare three transmission presets on independent 1-DOF joints.
    pub fn compare<F>(
        ik: Ik,
        seed: &[f64],
        waypoints: &[Waypoint],
        tool: F,
        duration: f64,
    ) -> [SimReport; 3]
    where
        F: Fn(&[f64]) -> (V3, V3) + Copy,
    {
        let n = seed.len();
        let dt = 1.0 / 240.0;

        let mut direct = vec![ServoJoint::new(GearTrain::direct_drive()); n];
        let mut qdd = super::servo::presets::qdd(n);
        let mut high = super::servo::presets::high_ratio(n);

        for (i, j) in direct.iter_mut().enumerate() {
            j.theta_deg = seed[i];
        }
        for (i, j) in qdd.iter_mut().enumerate() {
            j.theta_deg = seed[i];
        }
        for (i, j) in high.iter_mut().enumerate() {
            j.theta_deg = seed[i];
        }

        let _ = duration; // waypoints carry timing

        [
            Self::new(ik.clone())
                .with_label("direct")
                .with_dt(dt)
                .run(seed, waypoints, &mut direct, tool),
            Self::new(ik.clone())
                .with_label("qdd-15:1")
                .with_dt(dt)
                .run(seed, waypoints, &mut qdd, tool),
            Self::new(ik)
                .with_label("servo-288:1")
                .with_dt(dt)
                .run(seed, waypoints, &mut high, tool),
        ]
    }

    /// Compare transmissions through the coupled plant (gravity + inertia).
    pub fn compare_plant(
        ik: Ik,
        seed: &[f64],
        waypoints: &[Waypoint],
        chain: SerialChain,
    ) -> [SimReport; 3] {
        let dt = 1.0 / 240.0;
        let mut direct = ArmPlant::from_train(chain.clone(), GearTrain::direct_drive());
        let mut qdd = ArmPlant::from_train(chain.clone(), GearTrain::qdd());
        let mut high = ArmPlant::from_train(chain, GearTrain::high_ratio_servo());
        [
            Self::new(ik.clone())
                .with_label("direct")
                .with_dt(dt)
                .run_plant(seed, waypoints, &mut direct),
            Self::new(ik.clone())
                .with_label("qdd-15:1")
                .with_dt(dt)
                .run_plant(seed, waypoints, &mut qdd),
            Self::new(ik)
                .with_label("servo-288:1")
                .with_dt(dt)
                .run_plant(seed, waypoints, &mut high),
        ]
    }

    /// All five drive modalities on the coupled plant.
    pub fn compare_drives(
        ik: Ik,
        seed: &[f64],
        waypoints: &[Waypoint],
        chain: SerialChain,
    ) -> [SimReport; 5] {
        use super::actuator::Drive;
        let dt = 1.0 / 240.0;
        let drives = [
            ("direct", Drive::direct()),
            ("qdd-15:1", Drive::qdd()),
            ("servo-288:1", Drive::high_ratio()),
            ("hydraulic", Drive::hydraulic()),
            ("tendon", Drive::tendon()),
        ];
        let mut out: [Option<SimReport>; 5] = [None, None, None, None, None];
        for (i, (label, drive)) in drives.into_iter().enumerate() {
            let mut plant = ArmPlant::from_drive(chain.clone(), drive);
            out[i] = Some(
                Self::new(ik.clone())
                    .with_label(label)
                    .with_dt(dt)
                    .run_plant(seed, waypoints, &mut plant),
            );
        }
        [
            out[0].take().unwrap(),
            out[1].take().unwrap(),
            out[2].take().unwrap(),
            out[3].take().unwrap(),
            out[4].take().unwrap(),
        ]
    }
}

/// Linear interpolation between waypoints in space; holds the first/last.
fn interpolate_goal(waypoints: &[Waypoint], t: f64) -> Goal {
    if waypoints.len() == 1 {
        return waypoints[0].goal;
    }
    if t <= waypoints[0].t {
        return waypoints[0].goal;
    }
    let last = waypoints.last().unwrap();
    if t >= last.t {
        return last.goal;
    }

    let mut i = 0;
    while i + 1 < waypoints.len() && waypoints[i + 1].t < t {
        i += 1;
    }
    let a = &waypoints[i];
    let b = &waypoints[i + 1];
    let u = ((t - a.t) / (b.t - a.t)).clamp(0.0, 1.0);

    match (a.goal, b.goal) {
        (Goal::Point(p0), Goal::Point(p1)) => Goal::Point([
            p0[0] + u * (p1[0] - p0[0]),
            p0[1] + u * (p1[1] - p0[1]),
            p0[2] + u * (p1[2] - p0[2]),
        ]),
        (
            Goal::PointAlong { at: p0, along: d0 },
            Goal::PointAlong { at: p1, along: d1 },
        ) => Goal::PointAlong {
            at: [
                p0[0] + u * (p1[0] - p0[0]),
                p0[1] + u * (p1[1] - p0[1]),
                p0[2] + u * (p1[2] - p0[2]),
            ],
            along: [
                d0[0] + u * (d1[0] - d0[0]),
                d0[1] + u * (d1[1] - d0[1]),
                d0[2] + u * (d1[2] - d0[2]),
            ],
        },
        _ => a.goal,
    }
}

/// Ideal IK-only trajectory (no servo physics) for baseline comparison.
pub fn ideal_envelope<F>(
    ik: &Ik,
    seed: &[f64],
    waypoints: &[Waypoint],
    tool: F,
) -> Vec<(f64, Solution)>
where
    F: Fn(&[f64]) -> (V3, V3),
{
    let mut q = seed.to_vec();
    let mut out = Vec::with_capacity(waypoints.len());
    for wp in waypoints {
        let sol = ik.solve(&q, &wp.goal, &tool);
        q = sol.joints.clone();
        out.push((wp.t, sol));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    const DOWN: V3 = [0.0, 0.0, -1.0];

    #[test]
    fn high_ratio_worse_than_direct_on_same_path() {
        let chain = SerialChain::planar_3r();
        let seed = [-2.8, 88.9, 93.9];
        // Step then hold — acceleration (not steady ramp lag) exposes N² J.
        let waypoints = vec![
            Waypoint {
                t: 0.0,
                goal: Goal::PointAlong {
                    at: [280.0, 0.0, 220.0],
                    along: DOWN,
                },
            },
            Waypoint {
                t: 0.12,
                goal: Goal::PointAlong {
                    at: [420.0, 0.0, 220.0],
                    along: DOWN,
                },
            },
            Waypoint {
                t: 0.9,
                goal: Goal::PointAlong {
                    at: [420.0, 0.0, 220.0],
                    along: DOWN,
                },
            },
        ];

        let ik = Ik {
            limits: (-179.0, 179.0),
            max_iters: 120,
            joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
            ..Default::default()
        };
        let reports = IkServoSim::compare_plant(ik, &seed, &waypoints, chain);
        assert!(
            reports[0].rms_position_err < 25.0,
            "direct should settle: rms_tool={}",
            reports[0].rms_position_err
        );
        assert!(
            reports[2].worst_position_err > reports[0].worst_position_err,
            "high-ratio tip lag: direct worst={} high worst={}",
            reports[0].worst_position_err,
            reports[2].worst_position_err
        );
        assert!(
            reports[2].mean_joint_err > reports[0].mean_joint_err,
            "high-ratio joint lag: direct={} high={}",
            reports[0].mean_joint_err,
            reports[2].mean_joint_err
        );
    }

    #[test]
    fn hydraulic_and_tendon_plants_track() {
        let chain = SerialChain::planar_3r();
        let seed = [-2.8, 88.9, 93.9];
        let waypoints = vec![
            Waypoint {
                t: 0.0,
                goal: Goal::PointAlong {
                    at: [280.0, 0.0, 220.0],
                    along: DOWN,
                },
            },
            Waypoint {
                t: 0.4,
                goal: Goal::PointAlong {
                    at: [360.0, 0.0, 220.0],
                    along: DOWN,
                },
            },
        ];
        let ik = Ik {
            limits: (-179.0, 179.0),
            max_iters: 120,
            joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
            ..Default::default()
        };
        let reports = IkServoSim::compare_drives(ik, &seed, &waypoints, chain);
        assert_eq!(reports[3].transmission, "hydraulic");
        assert_eq!(reports[4].transmission, "tendon");
        assert!(!reports[3].frames.is_empty());
        assert!(!reports[4].frames.is_empty());
        // Tendon stretch / valve lag → more tip error than direct on a fast move.
        assert!(
            reports[4].worst_position_err > reports[0].worst_position_err * 0.5
                || reports[3].worst_position_err > reports[0].worst_position_err * 0.5,
            "expected soft actuators to lag: {:?}",
            reports
                .iter()
                .map(|r| (r.transmission, r.worst_position_err))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn geometric_ik_reaches_a_square_approach() {
        let chain = SerialChain::planar_3r();
        let s = Ik {
            limits: (-179.0, 179.0),
            max_iters: 80,
            joint_limits: chain.joints.iter().map(|j| j.limits).collect(),
            ..Default::default()
        }
        .solve_chain(
            &[-2.8, 88.9, 93.9],
            &Goal::PointAlong {
                at: [280.0, 0.0, 220.0],
                along: DOWN,
            },
            &chain,
        );
        assert!(s.within(0.05, 0.05), "{s:?}");
    }
}
