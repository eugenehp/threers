//! Wide lines, polylines and arcs.

use super::{Column, ColumnFormat, KindUniforms, Mark, MarkCommon, MarkKind, PrepareContext};

/// Lines of any width between pairs of points, straight or arched —
/// deck.gl's `LineLayer` and `ArcLayer`, and `PathLayer` through
/// [`set_paths`](Self::set_paths).
///
/// Each row is one segment from a source to a target. Width is per row (the
/// `widths` column) times [`MarkStyle::size_scale`](super::MarkStyle::size_scale),
/// in pixels or world units, and the colour runs from the row's source colour
/// to its target colour.
///
/// The GPU draws no line wider than one pixel by itself (WebGPU has no line
/// width at all), so every segment is a quad expanded in screen space, with
/// round caps and an antialiased edge computed per fragment.
///
/// Polylines become one segment per edge, each knowing its neighbours, and
/// meet at mitred joints: the two quads share the joint's bisector, so a
/// translucent line is not darker where they meet. Turns sharper than 120°
/// would make long spikes; there both ends get round caps instead, which
/// overlap. Closed rings (last point equal to the first) join all round.
#[derive(Debug)]
pub struct SegmentMarks {
    pub common: MarkCommon,
    sources: Column,
    targets: Column,
    /// Per row: the width, and the arc's height as a multiple of
    /// [`arc_height`](Self::arc_height).
    widths: Column,
    /// The parts of `widths` as set, empty when not per row: widths, arc
    /// heights, and dashes (`[dash, gap]` in line widths).
    width_rows: Vec<f32>,
    height_rows: Vec<f32>,
    offset_rows: Vec<f32>,
    dash_rows: Vec<[f32; 2]>,
    colors: Column,
    target_colors: Column,
    /// Per row: the previous point of its path minus the source and the
    /// distance along the path to the source; the next point minus the
    /// target and the whole path's length. Zero where the path ends. One buffer
    /// for all of it, to stay inside WebGPU's eight vertex buffers with both
    /// custom columns bound.
    neighbours: Column,
    /// Height of the arc's peak as a fraction of the segment's length. 0 is a
    /// straight line; 0.5 is a tall arch.
    pub arc_height: f32,
    /// Pieces each arc is drawn in. Ignored (one piece) when `arc_height` is
    /// 0, unless [`arc_strip`](Self::arc_strip) is on.
    pub arc_segments: u32,
    /// The direction arcs rise in, world space.
    pub arc_up: [f32; 3],
    /// The arc's shape between its ends.
    pub arc_profile: ArcProfile,
    /// Space an arc's pieces closer together toward its ends (by
    /// smoothstep) instead of evenly — what a profile steep at the ends,
    /// like [`ArcProfile::Ellipse`], needs. deck.gl's `ArcLayer` does this.
    pub arc_ease_ends: bool,
    /// Draw a hard-edged arc as deck.gl's `ArcLayer` does: one triangle
    /// strip, each vertex pushed across the clip-space direction to its
    /// neighbour (the previous one's, the first's next), without joints.
    /// Flat arcs too keep their pieces: each vertex's width then follows
    /// perspective, as deck.gl's do.
    pub arc_strip: bool,
    /// Antialiased edges ramped as deck.gl ramps them — a `smoothstep`
    /// over a pixel of the distance to the edge — rather than by the share
    /// of the pixel the stroke covers (the default, truer for lines
    /// thinner than a pixel).
    pub smoothstep_edges: bool,
    /// The centre of a globe the segments lie on: they then follow great
    /// circles (in [`arc_segments`](Self::arc_segments) pieces, even with no
    /// height) and arcs rise away from the centre instead of along
    /// [`arc_up`](Self::arc_up).
    pub globe: Option<[f64; 3]>,
    globe_rel: [f32; 3],
    /// Dashes along straight segments and paths: `[dash, gap]`, each in
    /// line widths (or [`dash_unit_px`](Self::dash_unit_px)). Rows with
    /// their own ([`set_dashes`](Self::set_dashes)) take theirs. Dash ends
    /// are antialiased along the line, periods under a pixel fade to their
    /// duty cycle, and round caps round them — deck.gl's
    /// `PathStyleExtension`.
    pub dash: Option<[f32; 2]>,
    /// Where the pattern is measured from: each segment's start, or the
    /// path's (continuing across its joints, the default).
    pub dash_mode: DashMode,
    /// Fit a whole number of dash periods to each segment (or, in
    /// [`DashMode::Path`], the whole path), starting and ending half a dash
    /// in — deck.gl's `dashJustified`.
    pub dash_justified: bool,
    /// Dash lengths in this many pixels each rather than line widths
    /// (deck.gl's `dashUnits` other than `'widths'`, converted to pixels).
    pub dash_unit_px: Option<f32>,
    /// Measure dashes along lines that face the camera (not
    /// [`flat`](Self::flat)) by their world length at this many pixels per
    /// world unit, rather than as drawn — deck.gl's billboarded paths,
    /// measured at the view's nominal scale.
    pub dash_nominal_px_per_unit: Option<f32>,
    /// Lean every [`ArcProfile::Ellipse`] arc this many degrees about its
    /// chord, its peak to the left of the way from source to target
    /// (deck.gl's `getTilt`; 90 lays it on the ground). Not on a globe.
    pub arc_tilt: f32,
    /// The ends of lines and paths.
    pub cap: LineCap,
    /// The joints of paths.
    pub join: LineJoin,
    /// Lie flat on the ground (world y up) instead of facing the camera:
    /// widths are world widths across the ground, foreshortened as the
    /// ground is (deck.gl's paths without `billboard`); on a
    /// [`globe`](Self::globe), flat on its surface. Not for arcs.
    pub flat: bool,
    /// Mitre joints however sharp, cut flat this many half widths past
    /// the outside of the joint's centre plus one (deck.gl's `miterLimit`:
    /// 0 is close to a bevel). `None`: mitres up to a 120° turn, rounded
    /// beyond.
    pub miter_limit: Option<f32>,
    /// A casing this many pixels wide inside the line's edge, in
    /// [`outline_color`](Self::outline_color): roads, borders.
    pub outline_width: f32,
    /// sRGB, straight alpha.
    pub outline_color: [u8; 4],
    /// For segments made by [`set_paths`](Self::set_paths): the first segment
    /// of each path, so a segment's row maps back to its path.
    path_starts: Vec<u32>,
}

/// Where a dash pattern starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashMode {
    /// Afresh at each segment (deck.gl's `dashMode: 'segment'`).
    Segment,
    /// At the path's start, continuing across its joints (deck.gl's
    /// `dashMode: 'path'`).
    #[default]
    Path,
}

/// The shape of an arc between its ends, each peaking at
/// [`arc_height`](SegmentMarks::arc_height) times the chord.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ArcProfile {
    /// `4 t (1 - t)`.
    #[default]
    Parabola,
    /// `2 sqrt(t (1 - t))`, a half ellipse, steep at the ends — deck.gl's
    /// `ArcLayer`.
    Ellipse,
}

/// How a line ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineCap {
    /// Flat, at the end point.
    Butt,
    /// A half disc past the end point.
    #[default]
    Round,
    /// A half square past the end point.
    Square,
}

/// How a path turns at its joints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineJoin {
    /// Mitred, or rounded where the turn is too sharp to mitre.
    #[default]
    Miter,
    /// A disc at every joint (deck.gl's `jointRounded`).
    Round,
}

/// Polylines flattened to segments, as plain data, built on any thread —
/// what [`SegmentMarks::set_path_mesh`] uploads.
///
/// Each segment knows its neighbours (for mitred joints) and its distance
/// along its path (for dashes), and positions are local to `origin`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PathMesh {
    pub origin: [f64; 3],
    pub sources: Vec<[f32; 3]>,
    pub targets: Vec<[f32; 3]>,
    /// Per segment: previous point minus source, distance along the path to
    /// the source; next point minus target, the whole path's length.
    pub neighbours: Vec<[f32; 8]>,
    /// The first segment of each path.
    pub path_starts: Vec<u32>,
}

impl PathMesh {
    /// Flatten `paths`, local to the middle of their bounds.
    pub fn build(paths: &[Vec<[f64; 3]>]) -> Self {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in paths.iter().flatten() {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        let origin = if lo[0].is_finite() {
            [
                (lo[0] + hi[0]) * 0.5,
                (lo[1] + hi[1]) * 0.5,
                (lo[2] + hi[2]) * 0.5,
            ]
        } else {
            [0.0; 3]
        };
        Self::build_around(paths, origin)
    }

    /// Flatten `paths`, local to `origin`.
    pub fn build_around(paths: &[Vec<[f64; 3]>], origin: [f64; 3]) -> Self {
        let rel = |p: [f64; 3], from: [f64; 3]| {
            [
                (p[0] - from[0]) as f32,
                (p[1] - from[1]) as f32,
                (p[2] - from[2]) as f32,
            ]
        };
        let len = |a: [f64; 3], b: [f64; 3]| {
            ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
        };
        let mut m = Self {
            origin,
            ..Default::default()
        };
        for p in paths {
            m.path_starts.push(m.sources.len() as u32);
            let edges = p.len().saturating_sub(1);
            let closed = edges >= 3 && p.first() == p.last();
            let mut along = 0.0f64;
            for k in 0..edges {
                let (s, t) = (p[k], p[k + 1]);
                m.sources.push(rel(s, origin));
                m.targets.push(rel(t, origin));
                let before = match k {
                    0 if closed => Some(p[edges - 1]),
                    0 => None,
                    _ => Some(p[k - 1]),
                };
                let after = match k + 2 {
                    j if j <= edges => Some(p[j]),
                    _ if closed => Some(p[1]),
                    _ => None,
                };
                let b = before.map_or([0.0; 3], |q| rel(q, s));
                let a = after.map_or([0.0; 3], |q| rel(q, t));
                m.neighbours.push([b[0], b[1], b[2], along as f32, a[0], a[1], a[2], 0.0]);
                along += len(s, t);
            }
            let first = m.neighbours.len() - edges;
            for n in &mut m.neighbours[first..] {
                n[7] = along as f32;
            }
        }
        m
    }

    /// As bytes for a Web Worker boundary, little-endian.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"TPA1");
        for v in self.origin {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&(self.sources.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.path_starts.len() as u32).to_le_bytes());
        let floats = self
            .sources
            .iter()
            .flatten()
            .chain(self.targets.iter().flatten())
            .chain(self.neighbours.iter().flatten());
        for f in floats {
            out.extend_from_slice(&f.to_le_bytes());
        }
        for s in &self.path_starts {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    /// [`to_bytes`](Self::to_bytes) back.
    pub fn from_bytes(b: &[u8]) -> Result<Self, String> {
        let take = |at: &mut usize, n: usize| -> Result<&[u8], String> {
            let s = b.get(*at..*at + n).ok_or("PathMesh bytes end early")?;
            *at += n;
            Ok(s)
        };
        let mut at = 0;
        if take(&mut at, 4)? != b"TPA1" {
            return Err("not a PathMesh".into());
        }
        let mut origin = [0.0; 3];
        for o in &mut origin {
            *o = f64::from_le_bytes(take(&mut at, 8)?.try_into().unwrap());
        }
        let n = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap()) as usize;
        let np = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap()) as usize;
        let mut f = |count: usize| -> Result<Vec<f32>, String> {
            (0..count)
                .map(|_| Ok(f32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap())))
                .collect()
        };
        let s = f(n * 3)?;
        let t = f(n * 3)?;
        let nb = f(n * 8)?;
        let path_starts = (0..np)
            .map(|_| Ok(u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap())))
            .collect::<Result<Vec<_>, String>>()?;
        if at != b.len() {
            return Err("PathMesh bytes run on".into());
        }
        Ok(Self {
            origin,
            sources: s.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            targets: t.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(),
            neighbours: nb.chunks(8).map(|c| c.try_into().unwrap()).collect(),
            path_starts,
        })
    }
}

impl Default for SegmentMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl SegmentMarks {
    pub fn new() -> Self {
        let mut widths = Column::float(4);
        widths.set_default_f32(&[1.0, 1.0, 0.0, 0.0]);
        Self {
            common: MarkCommon::default(),
            sources: Column::new(ColumnFormat::PositionF64),
            targets: Column::new(ColumnFormat::PositionF64),
            widths,
            width_rows: Vec::new(),
            height_rows: Vec::new(),
            offset_rows: Vec::new(),
            dash_rows: Vec::new(),
            colors: Column::new(ColumnFormat::Unorm8x4),
            target_colors: Column::new(ColumnFormat::Unorm8x4),
            neighbours: Column::new(ColumnFormat::Float4x2),
            arc_height: 0.0,
            arc_segments: 32,
            arc_up: [0.0, 1.0, 0.0],
            arc_profile: ArcProfile::Parabola,
            arc_ease_ends: false,
            arc_strip: false,
            smoothstep_edges: false,
            globe: None,
            globe_rel: [0.0; 3],
            dash: None,
            dash_mode: DashMode::Path,
            dash_justified: false,
            dash_unit_px: None,
            dash_nominal_px_per_unit: None,
            arc_tilt: 0.0,
            cap: LineCap::Round,
            join: LineJoin::Miter,
            flat: false,
            miter_limit: None,
            outline_width: 0.0,
            outline_color: [0, 0, 0, 255],
            path_starts: Vec::new(),
        }
    }

    /// Source and target of every segment. Both must be the same length.
    pub fn set_segments(&mut self, sources: &[[f64; 3]], targets: &[[f64; 3]]) {
        assert_eq!(sources.len(), targets.len(), "one target per source");
        self.common.local_origin = None;
        self.sources.set_positions(sources);
        self.targets.set_positions(targets);
        self.neighbours.set_f32(&[]);
        self.path_starts.clear();
    }

    /// [`set_segments`](Self::set_segments) with both ends relative to
    /// `origin`, at half the memory.
    pub fn set_local_segments(
        &mut self,
        origin: [f64; 3],
        sources: &[[f32; 3]],
        targets: &[[f32; 3]],
    ) {
        assert_eq!(sources.len(), targets.len(), "one target per source");
        self.common.local_origin = Some(origin);
        self.sources.set_local_positions(sources);
        self.targets.set_local_positions(targets);
        self.neighbours.set_f32(&[]);
        self.path_starts.clear();
    }

    /// [`set_paths`](Self::set_paths) with points relative to `origin`.
    pub fn set_local_paths(&mut self, origin: [f64; 3], paths: &[Vec<[f32; 3]>]) {
        let abs: Vec<Vec<[f64; 3]>> = paths
            .iter()
            .map(|p| {
                p.iter()
                    .map(|q| {
                        [
                            origin[0] + q[0] as f64,
                            origin[1] + q[1] as f64,
                            origin[2] + q[2] as f64,
                        ]
                    })
                    .collect()
            })
            .collect();
        self.set_path_mesh(PathMesh::build_around(&abs, origin));
    }

    /// Polylines, one segment per edge. Colours and widths set afterwards are
    /// per *segment*; [`set_path_colors`](Self::set_path_colors) and
    /// [`set_path_widths`](Self::set_path_widths) set them per path.
    pub fn set_paths(&mut self, paths: &[Vec<[f64; 3]>]) {
        self.set_path_mesh(PathMesh::build(paths));
    }

    /// Paths prepared by [`PathMesh::build`], on any thread.
    pub fn set_path_mesh(&mut self, mesh: PathMesh) {
        self.common.local_origin = Some(mesh.origin);
        self.sources.set_local_positions(&mesh.sources);
        self.targets.set_local_positions(&mesh.targets);
        self.neighbours
            .set_f32(bytemuck::cast_slice::<[f32; 8], f32>(&mesh.neighbours));
        self.path_starts = mesh.path_starts;
    }

    /// The segment rows of path `path`, when built with
    /// [`set_paths`](Self::set_paths).
    pub fn path_rows(&self, path: usize) -> Option<std::ops::Range<u32>> {
        let start = *self.path_starts.get(path)?;
        let end = self.path_starts.get(path + 1).copied().unwrap_or(self.sources.len() as u32);
        Some(start..end)
    }

    /// Highlight paths `paths` (consecutive) whole, or nothing.
    pub fn highlight_paths(&mut self, paths: Option<std::ops::Range<usize>>) {
        let rows = paths.and_then(|p| Some(self.path_rows(p.start)?.start..self.path_rows(p.end.max(p.start + 1) - 1)?.end));
        let s = &mut self.common.style;
        s.highlight = rows.as_ref().map(|r| r.start);
        s.highlight_len = rows.map_or(1, |r| (r.end - r.start).max(1));
    }

    /// The path a segment row belongs to, when built with
    /// [`set_paths`](Self::set_paths).
    pub fn path_of(&self, segment: u32) -> Option<usize> {
        if self.path_starts.is_empty() || segment as usize >= self.sources.len() {
            return None;
        }
        Some(self.path_starts.partition_point(|&s| s <= segment) - 1)
    }

    fn per_path<T: Copy>(&self, values: &[T]) -> Vec<T> {
        let n = self.sources.len();
        let mut out = Vec::with_capacity(n);
        for (i, &start) in self.path_starts.iter().enumerate() {
            let end = self.path_starts.get(i + 1).copied().unwrap_or(n as u32);
            let v = values[i.min(values.len() - 1)];
            out.extend(std::iter::repeat_n(v, (end - start) as usize));
        }
        out
    }

    /// One colour per path, for paths from [`set_paths`](Self::set_paths).
    pub fn set_path_colors(&mut self, colors: &[[u8; 4]]) {
        let c = self.per_path(colors);
        self.colors.set_rgba(&c);
        self.target_colors.set_rgba(&c);
    }

    /// One width per path.
    pub fn set_path_widths(&mut self, widths: &[f32]) {
        let w = self.per_path(widths);
        self.set_widths(&w);
    }

    /// Pieces each segment is drawn in.
    fn pieces(&self) -> u32 {
        if self.arc_height == 0.0 && self.globe.is_none() && !self.arc_strip {
            1
        } else {
            self.arc_segments.clamp(1, 512)
        }
    }

    pub fn set_widths(&mut self, w: &[f32]) {
        self.width_rows = w.to_vec();
        self.pack_widths();
    }
    pub fn update_widths(&mut self, start: usize, w: &[f32]) {
        if start + w.len() > self.width_rows.len() || self.widths.len() != self.width_rows.len().max(self.height_rows.len()).max(self.offset_rows.len()).max(self.dash_rows.len()) {
            // Not a plain update of what is there: pack it all again.
            let n = self.width_rows.len().max(start + w.len());
            self.width_rows.resize(n, 1.0);
            self.width_rows[start..start + w.len()].copy_from_slice(w);
            self.pack_widths();
            return;
        }
        self.width_rows[start..start + w.len()].copy_from_slice(w);
        let packed: Vec<f32> = (start..start + w.len()).flat_map(|i| self.packed_row(i)).collect();
        self.widths.update_f32(start, &packed);
    }
    /// Each arc's height as a multiple of [`arc_height`](Self::arc_height)
    /// (deck.gl's per-row `getHeight`); 1 for rows without one.
    pub fn set_arc_heights(&mut self, h: &[f32]) {
        self.height_rows = h.to_vec();
        self.pack_widths();
    }
    /// Each row's own dash, `[dash, gap]` in line widths (`[0, 0]`: the
    /// layer's [`dash`](Self::dash)); empty for none.
    pub fn set_dashes(&mut self, d: &[[f32; 2]]) {
        self.dash_rows = d.to_vec();
        self.pack_widths();
    }
    /// Each row's line moved sideways by this many widths, right of its
    /// direction of travel (negative: left) — deck.gl's `getOffset`. Not
    /// with arcs: it takes the place of their heights. Empty for none.
    pub fn set_offsets(&mut self, o: &[f32]) {
        self.offset_rows = o.to_vec();
        self.pack_widths();
    }
    /// Per path, for paths from [`set_paths`](Self::set_paths).
    pub fn set_path_offsets(&mut self, o: &[f32]) {
        let o = if o.is_empty() { Vec::new() } else { self.per_path(o) };
        self.set_offsets(&o);
    }
    /// Per path, for paths from [`set_paths`](Self::set_paths).
    pub fn set_path_dashes(&mut self, d: &[[f32; 2]]) {
        let d = self.per_path(d);
        self.set_dashes(&d);
    }
    fn packed_row(&self, i: usize) -> [f32; 4] {
        let d = self.dash_rows.get(i).copied().unwrap_or([0.0, 0.0]);
        let y = if self.offset_rows.is_empty() {
            self.height_rows.get(i).copied().unwrap_or(1.0)
        } else {
            self.offset_rows.get(i).copied().unwrap_or(0.0)
        };
        [self.width_rows.get(i).copied().unwrap_or(1.0), y, d[0], d[1]]
    }
    fn pack_widths(&mut self) {
        let n = self.width_rows.len().max(self.height_rows.len()).max(self.offset_rows.len()).max(self.dash_rows.len());
        if n == 0 {
            self.widths.set_f32(&[]);
            return;
        }
        let packed: Vec<f32> = (0..n).flat_map(|i| self.packed_row(i)).collect();
        self.widths.set_f32(&packed);
    }
    /// Colour at each segment's source — and at its target too, unless
    /// [`set_target_colors`](Self::set_target_colors) says otherwise.
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
        self.target_colors.set_rgba(c);
    }
    pub fn set_target_colors(&mut self, c: &[[u8; 4]]) {
        self.target_colors.set_rgba(c);
    }
    pub fn update_colors(&mut self, start: usize, c: &[[u8; 4]]) {
        self.colors.update_rgba(start, c);
        self.target_colors.update_rgba(start, c);
    }
    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
        self.target_colors.set_default_rgba(c);
    }
}

impl Mark for SegmentMarks {
    fn kind(&self) -> MarkKind {
        MarkKind::Segment
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        self.sources.len().min(self.targets.len())
    }
    fn vertices_per_instance(&self) -> u32 {
        6 * self.pieces()
    }
    fn kind_uniforms(&self) -> KindUniforms {
        (
            [
                self.arc_height,
                self.pieces() as f32,
                // Bit 0: the profile; bit 1: eased pieces; bit 2: flat.
                (match self.arc_profile {
                    ArcProfile::Parabola => 0u32,
                    ArcProfile::Ellipse => 1,
                } | (u32::from(self.arc_ease_ends) << 1)
                    | (u32::from(self.flat && self.globe.is_none()) << 2)
                    | (u32::from(!self.offset_rows.is_empty()) << 4)
                    | (u32::from(self.flat && self.globe.is_some() && self.arc_height == 0.0) << 5)
                    | (u32::from(self.arc_strip) << 6)
                    | (u32::from(self.smoothstep_edges) << 7)) as f32,
                0.0,
            ],
            match self.globe {
                Some(_) => [self.globe_rel[0], self.globe_rel[1], self.globe_rel[2], 1.0],
                None => [self.arc_up[0], self.arc_up[1], self.arc_up[2], 0.0],
            },
            // The casing's colour.
            super::points::srgb_u8_to_linear(self.outline_color),
        )
    }
    fn kind_extra(&self) -> [f32; 4] {
        let [dash, gap] = self.dash.unwrap_or([0.0, 0.0]);
        let cap = match self.cap {
            LineCap::Butt => 0.0,
            LineCap::Round => 1.0,
            LineCap::Square => 3.0,
        } + if self.join == LineJoin::Round { 4.0 } else { 0.0 }
            // Bits 3 and up: the mitre limit in eighths, plus one; 0 for none.
            + 8.0 * self.miter_limit.map_or(0.0, |l| (l.max(0.0) * 8.0).round() + 1.0);
        [
            dash.max(0.0),
            gap.max(0.0),
            self.outline_width.max(0.0),
            cap,
        ]
    }
    /// x: pixels per dash unit (0: line widths); y: bit 0 the whole path,
    /// bit 1 justified; z: nominal pixels per world unit (0: as drawn); w:
    /// the arcs' tilt, radians.
    fn kind_material(&self) -> [f32; 4] {
        [
            self.dash_unit_px.map_or(0.0, |p| p.max(1e-6)),
            (u32::from(self.dash_mode == DashMode::Path) | (u32::from(self.dash_justified) << 1)) as f32,
            self.dash_nominal_px_per_unit.map_or(0.0, |k| k.max(0.0)),
            self.arc_tilt.to_radians(),
        ]
    }
    fn prepare(&mut self, ctx: &PrepareContext) {
        if let Some(c) = self.globe {
            let o = ctx.view.origin.position;
            self.globe_rel = [
                (c[0] - o[0]) as f32,
                (c[1] - o[1]) as f32,
                (c[2] - o[2]) as f32,
            ];
        }
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![
            &mut self.sources,
            &mut self.targets,
            &mut self.widths,
            &mut self.colors,
            &mut self.target_colors,
            &mut self.neighbours,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The neighbours column as (prev, next), each a flat run of triples.
    fn split(s: &SegmentMarks) -> (Vec<f32>, Vec<f32>) {
        let all: Vec<f32> = bytemuck::pod_collect_to_vec(s.neighbours.bytes());
        let (mut prev, mut next) = (Vec::new(), Vec::new());
        for row in all.chunks(8) {
            prev.extend_from_slice(&row[..3]);
            next.extend_from_slice(&row[4..7]);
        }
        (prev, next)
    }

    #[test]
    fn closed_rings_join_all_round() {
        let mut s = SegmentMarks::new();
        let square = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        s.set_paths(&[square]);
        let (prev, next) = split(&s);
        // The first edge's predecessor is the last edge's source, and the
        // last edge's successor is the first edge's target.
        assert_eq!(&prev[0..3], &[0.0, 1.0, 0.0]);
        assert_eq!(&next[9..12], &[1.0, 0.0, 0.0]);
        assert!(prev.chunks(3).chain(next.chunks(3)).all(|v| v != [0.0; 3]));
    }

    #[test]
    fn paths_flatten_to_edges_and_map_back() {
        let mut s = SegmentMarks::new();
        s.set_paths(&[
            vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            vec![[5.0; 3]],
            vec![[0.0; 3], [0.0, 1.0, 0.0]],
        ]);
        assert_eq!(s.instance_count(), 3);
        assert_eq!(s.path_of(0), Some(0));
        assert_eq!(s.path_of(1), Some(0));
        // The single-point path has no segments; row 2 is the third path.
        assert_eq!(s.path_of(2), Some(2));
        assert_eq!(s.path_of(3), None);
        // Neighbours: the first edge has none before it, the middle joint is
        // shared, and the second path's lone edge is free at both ends.
        let (prev, next) = split(&s);
        assert_eq!(prev, [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(next, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        s.set_path_colors(&[[1, 0, 0, 255], [2, 0, 0, 255], [3, 0, 0, 255]]);
        assert_eq!(
            s.colors.bytes(),
            &[1, 0, 0, 255, 1, 0, 0, 255, 3, 0, 0, 255]
        );
    }
}

#[cfg(test)]
mod offset_tests {
    use super::*;

    #[test]
    fn offsets_take_the_height_lane() {
        let mut s = SegmentMarks::new();
        s.set_paths(&[vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]);
        s.set_widths(&[]);
        s.set_path_offsets(&[2.0]);
        let w: Vec<f32> = bytemuck::pod_collect_to_vec(s.widths.bytes());
        assert_eq!(w, vec![1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]);
        assert_eq!(s.kind_uniforms().0[2] as u32 & 16, 16);
    }
}
