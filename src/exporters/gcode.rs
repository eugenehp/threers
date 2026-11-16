//! G-code toolpaths from polylines, for pen plotters, laser cutters and
//! 2.5D CNC: each path a rapid to its start, the tool down (a plunge to the
//! path's Z under [`GcodeOptions::safe_z`], and the
//! [`tool_on`](GcodeOptions::tool_on) lines), feed moves through its
//! points, and the tool up again. This writes motion, not slices: a mesh for
//! a 3D printer needs a slicer.
//!
//! [`crate::loaders::GcodeLoader`] reads it back, feed moves as the green
//! "extruded" segments.
//!
//! ```
//! use threers::exporters::{write_gcode, GcodeOptions, GcodePath};
//! use threers::loaders::GcodeLoader;
//!
//! let square = GcodePath::closed_2d(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
//! let text = write_gcode(&[square], &GcodeOptions::default());
//! let read = GcodeLoader::default().parse(&text);
//! assert_eq!(read.extruded().len(), 4 * 6 + 6); // four sides, and the plunge
//! ```

use std::fmt::Write;

/// One polyline. [`write_gcode`] cuts it in the order given.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GcodePath {
    pub points: Vec<[f64; 3]>,
}

impl GcodePath {
    pub fn new(points: Vec<[f64; 3]>) -> Self {
        Self { points }
    }

    /// An open 2D polyline at Z 0.
    pub fn open_2d(points: &[[f64; 2]]) -> Self {
        Self::new(points.iter().map(|&[x, y]| [x, y, 0.0]).collect())
    }

    /// A closed 2D polygon at Z 0: the first point repeats at the end.
    pub fn closed_2d(points: &[[f64; 2]]) -> Self {
        let mut p = Self::open_2d(points);
        if let Some(&first) = p.points.first()
            && p.points.last() != Some(&first)
        {
            p.points.push(first);
        }
        p
    }
}

/// Units the file declares (`G21` / `G20`). Coordinates are written as
/// given; this does not convert them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GcodeUnits {
    #[default]
    Millimeters,
    Inches,
}

/// How [`write_gcode`] writes a toolpath.
#[derive(Debug, Clone, PartialEq)]
pub struct GcodeOptions {
    pub units: GcodeUnits,
    /// Feed rate for cutting moves, units per minute.
    pub feed_rate: f64,
    /// Feed rate for the plunge into each path; `None` uses `feed_rate`.
    pub plunge_rate: Option<f64>,
    /// `Some(z)`: lift to `z` between paths and plunge to each path's own Z.
    /// `None`: rapid straight to each path's start in 3D (a laser, or a
    /// plotter that lifts its pen through `tool_on` / `tool_off`).
    pub safe_z: Option<f64>,
    /// Lines written once the tool is at a path's start (`M3 S1000`, a pen
    /// servo command), and after its end (`M5`).
    pub tool_on: Vec<String>,
    pub tool_off: Vec<String>,
    /// Lines after the preamble and before the end; a `;` comment each, so
    /// none of them can be a command.
    pub comments: Vec<String>,
    /// Lines written once, after the preamble and before the first move
    /// (a router's `M3 S18000` and a `G4 P2` to let it come up to speed:
    /// unlike [`tool_on`](Self::tool_on), before any plunge).
    pub start: Vec<String>,
    /// Lines written once, after the last path, before `M2` (`M5`).
    pub end: Vec<String>,
    /// `Some(step)`: a one-point path (a hole) is drilled in steps of
    /// `step` down from Z 0, the tool lifted to [`safe_z`](Self::safe_z)
    /// after each to clear the chips and brought rapidly back to just above
    /// the depth reached: a peck cycle in plain moves, for controllers
    /// without `G83`. Needs `safe_z`; `None` plunges straight down.
    pub peck: Option<f64>,
    /// Digits after the decimal point; trailing zeros are dropped.
    pub decimals: usize,
}

impl Default for GcodeOptions {
    fn default() -> Self {
        Self {
            units: GcodeUnits::Millimeters,
            feed_rate: 1000.0,
            plunge_rate: None,
            safe_z: Some(5.0),
            tool_on: Vec::new(),
            tool_off: Vec::new(),
            comments: Vec::new(),
            start: Vec::new(),
            end: Vec::new(),
            peck: None,
            decimals: 3,
        }
    }
}

fn number(out: &mut String, v: f64, decimals: usize) {
    let mut s = format!("{v:.decimals$}");
    if s.contains('.') {
        let trimmed = s.trim_end_matches('0').trim_end_matches('.').len();
        s.truncate(trimmed);
    }
    if s == "-0" {
        s = "0".into();
    }
    out.push_str(&s);
}

/// The G-code for `paths`: `G21`/`G20`, `G90`, `G17`, the
/// [`start`](GcodeOptions::start) lines, then each path in turn, ending with
/// the tool lifted to [`safe_z`](GcodeOptions::safe_z), the
/// [`end`](GcodeOptions::end) lines and `M2`.
/// Paths with no points are skipped; one point is a plunge and nothing else
/// (a drilled hole).
pub fn write_gcode(paths: &[GcodePath], options: &GcodeOptions) -> String {
    let d = options.decimals;
    let mut out = String::new();
    for c in &options.comments {
        for line in c.lines() {
            let _ = writeln!(out, "; {line}");
        }
    }
    out.push_str(match options.units {
        GcodeUnits::Millimeters => "G21\n",
        GcodeUnits::Inches => "G20\n",
    });
    out.push_str("G90\nG17\n");
    for line in &options.start {
        let _ = writeln!(out, "{line}");
    }
    let axes = |out: &mut String, p: [f64; 3], which: [bool; 3]| {
        for (i, letter) in ['X', 'Y', 'Z'].iter().enumerate() {
            if which[i] {
                out.push(' ');
                out.push(*letter);
                number(out, p[i], d);
            }
        }
    };
    let feed = |out: &mut String, f: f64| {
        out.push_str(" F");
        number(out, f, d);
    };
    if let Some(z) = options.safe_z {
        out.push_str("G0");
        axes(&mut out, [0.0, 0.0, z], [false, false, true]);
        out.push('\n');
    }
    for path in paths {
        let Some(&start) = path.points.first() else {
            continue;
        };
        out.push_str("G0");
        match options.safe_z {
            Some(_) => axes(&mut out, start, [true, true, false]),
            None => axes(&mut out, start, [true, true, true]),
        }
        out.push('\n');
        let plunge = options.plunge_rate.unwrap_or(options.feed_rate);
        match (options.safe_z, options.peck) {
            (Some(safe), Some(step)) if path.points.len() == 1 && step > 0.0 && start[2] < 0.0 => {
                // Peck: down a step, out to clear, back to just above.
                let mut reached = 0.0f64;
                while reached > start[2] {
                    if reached < 0.0 {
                        out.push_str("G0");
                        axes(&mut out, [0.0, 0.0, reached + PECK_CLEARANCE], [false, false, true]);
                        out.push('\n');
                    }
                    reached = (reached - step).max(start[2]);
                    out.push_str("G1");
                    axes(&mut out, [0.0, 0.0, reached], [false, false, true]);
                    feed(&mut out, plunge);
                    out.push('\n');
                    if reached > start[2] {
                        out.push_str("G0");
                        axes(&mut out, [0.0, 0.0, safe], [false, false, true]);
                        out.push('\n');
                    }
                }
            }
            (Some(_), _) => {
                out.push_str("G1");
                axes(&mut out, start, [false, false, true]);
                feed(&mut out, plunge);
                out.push('\n');
            }
            (None, _) => {}
        }
        for line in &options.tool_on {
            let _ = writeln!(out, "{line}");
        }
        // F is modal: the plunge set it, unless there was none or it was
        // the plunge rate.
        let mut need_feed = options.safe_z.is_none() || options.plunge_rate.is_some();
        let mut prev = start;
        for &p in &path.points[1..] {
            let changed = [p[0] != prev[0], p[1] != prev[1], p[2] != prev[2]];
            if changed == [false; 3] {
                continue;
            }
            out.push_str("G1");
            axes(&mut out, p, changed);
            if need_feed {
                feed(&mut out, options.feed_rate);
                need_feed = false;
            }
            out.push('\n');
            prev = p;
        }
        for line in &options.tool_off {
            let _ = writeln!(out, "{line}");
        }
        if let Some(z) = options.safe_z {
            out.push_str("G0");
            axes(&mut out, [0.0, 0.0, z], [false, false, true]);
            out.push('\n');
        }
    }
    for line in &options.end {
        let _ = writeln!(out, "{line}");
    }
    out.push_str("M2\n");
    out
}

/// How far above the depth already drilled a peck comes back down rapidly.
const PECK_CLEARANCE: f64 = 0.2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loaders::GcodeLoader;

    #[test]
    fn square_with_lift() {
        let square = GcodePath::closed_2d(&[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]);
        let text = write_gcode(&[square], &GcodeOptions::default());
        assert_eq!(
            text,
            "G21\nG90\nG17\nG0 Z5\nG0 X0 Y0\nG1 Z0 F1000\nG1 X10\nG1 Y10\nG1 X0\nG1 Y0\nG0 Z5\nM2\n"
        );
    }

    #[test]
    fn laser_without_lift() {
        let opts = GcodeOptions {
            safe_z: None,
            tool_on: vec!["M3 S1000".into()],
            tool_off: vec!["M5".into()],
            comments: vec!["cut\nlist".into()],
            decimals: 2,
            ..Default::default()
        };
        let line = GcodePath::open_2d(&[[1.0 / 3.0, -0.0001], [2.0, 2.0]]);
        let text = write_gcode(&[line], &opts);
        assert_eq!(
            text,
            "; cut\n; list\nG21\nG90\nG17\nG0 X0.33 Y0 Z0\nM3 S1000\nG1 X2 Y2 F1000\nM5\nM2\n"
        );
    }

    #[test]
    fn duplicate_points_are_dropped() {
        let p = GcodePath::new(vec![[0.0; 3], [0.0; 3], [1.0, 0.0, 0.0]]);
        let text = write_gcode(&[p], &GcodeOptions::default());
        assert!(text.contains("G1 Z0 F1000\nG1 X1\nG0 Z5"), "{text}");
    }

    #[test]
    fn a_router_starts_its_spindle_before_the_first_plunge() {
        let opts = GcodeOptions {
            start: vec!["M3 S18000".into(), "G4 P2".into()],
            end: vec!["M5".into()],
            ..Default::default()
        };
        let p = GcodePath::new(vec![[1.0, 1.0, -1.0], [2.0, 1.0, -1.0]]);
        assert_eq!(
            write_gcode(&[p], &opts),
            "G21\nG90\nG17\nM3 S18000\nG4 P2\nG0 Z5\nG0 X1 Y1\nG1 Z-1 F1000\nG1 X2\nG0 Z5\nM5\nM2\n"
        );
    }

    #[test]
    fn holes_are_pecked() {
        let opts = GcodeOptions { peck: Some(1.0), plunge_rate: Some(200.0), ..Default::default() };
        let hole = GcodePath::new(vec![[3.0, 4.0, -2.5]]);
        assert_eq!(
            write_gcode(std::slice::from_ref(&hole), &opts),
            "G21\nG90\nG17\nG0 Z5\nG0 X3 Y4\n\
             G1 Z-1 F200\nG0 Z5\nG0 Z-0.8\nG1 Z-2 F200\nG0 Z5\nG0 Z-1.8\nG1 Z-2.5 F200\n\
             G0 Z5\nM2\n"
        );
        // Read back, the cut reaches the hole's depth and no deeper.
        let opts = GcodeOptions { start: vec!["M3".into()], ..opts };
        let read = GcodeLoader::default().parse(&write_gcode(&[hole], &opts));
        let deepest = read.extruded().chunks(3).map(|v| v[2]).fold(f32::MAX, f32::min);
        assert_eq!(deepest, -2.5);
        // A path of more than one point is cut as before, pecking or not.
        let line = GcodePath::new(vec![[0.0, 0.0, -2.0], [5.0, 0.0, -2.0]]);
        assert!(write_gcode(&[line], &opts).contains("G1 Z-2 F200\nG1 X5 F1000"));
    }

    #[test]
    fn round_trip_through_the_loader() {
        let paths = [
            GcodePath::closed_2d(&[[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]]),
            GcodePath::new(vec![[20.0, 0.0, -1.0], [30.0, 0.0, -2.0]]),
            GcodePath::default(),
        ];
        for safe_z in [Some(3.0), None] {
            let opts = GcodeOptions {
                safe_z,
                tool_on: vec!["M3".into()],
                tool_off: vec!["M5".into()],
                ..Default::default()
            };
            let read = GcodeLoader::default().parse(&write_gcode(&paths, &opts));
            assert!(!read.has_extrusion);
            assert!(read.skipped.contains_key("M2"));
            let cut: Vec<[f32; 6]> = read
                .extruded()
                .chunks(6)
                .map(|c| c.try_into().unwrap())
                .filter(|s: &[f32; 6]| s[2] == s[5])
                .collect();
            // The level cuts are the triangle's three sides; the plunges and
            // the sloped line are the rest.
            assert_eq!(cut.len(), 3, "{safe_z:?}");
            let all: Vec<f32> = read.extruded();
            let last = &all[all.len() - 6..];
            assert_eq!(last, [20.0, 0.0, -1.0, 30.0, 0.0, -2.0]);
        }
    }
}
