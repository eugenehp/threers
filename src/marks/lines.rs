//! Lines a pixel wide drawn as the GPU's own line primitives — what WebGL's
//! `gl.LINES` gives deck.gl's wireframes — rather than as quads: the
//! rasterizer decides which pixels a line lights, as it does there.

use super::shader::CustomShader;
use super::{Column, ColumnFormat, Mark, MarkCommon, MarkKind};

const SOURCE: &str = r#"
struct LineIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) a_hi : vec3<f32>,
    @location(1) a_lo : vec3<f32>,
    @location(2) b_hi : vec3<f32>,
    @location(3) b_lo : vec3<f32>,
    @location(4) color : vec4<f32>,
    /*CUSTOM_IN*/
};

struct LineOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
    @location(4) world : vec3<f32>,
};

@vertex
fn vs_main(in : LineIn) -> LineOut {
    var i : MarkInstance;
    i.position = mark_relative(in.a_hi, in.a_lo);
    i.position2 = mark_relative(in.b_hi, in.b_lo);
    i.size = 1.0;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);
    var out : LineOut;
    let p = select(i.position, i.position2, in.vid == 1u);
    out.clip = select(mark_order(frame.view_proj * vec4<f32>(p, 1.0)), mark_hidden(), i.visible <= 0.0);
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    out.world = p;
    return out;
}

@fragment
fn fs_main(in : LineOut) -> @location(0) vec4<f32> {
    if (mark_culled(in.world)) {
        discard;
    }
    let c = hook_color(in.color, in.custom0, in.custom1);
    return mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
}

@fragment
fn fs_pick(in : LineOut) -> @location(0) vec2<u32> {
    if (mark_culled(in.world)) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

/// The same lines as SwiftShader's parallelograms ([`LineMarks::parallelogram`]).
const QUAD_SOURCE: &str = r#"
struct LineIn {
    @builtin(vertex_index) vid : u32,
    @builtin(instance_index) iid : u32,
    @location(0) a_hi : vec3<f32>,
    @location(1) a_lo : vec3<f32>,
    @location(2) b_hi : vec3<f32>,
    @location(3) b_lo : vec3<f32>,
    @location(4) color : vec4<f32>,
    /*CUSTOM_IN*/
};

struct LineOut {
    @builtin(position) clip : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) custom0 : vec4<f32>,
    @location(2) custom1 : vec4<f32>,
    @location(3) @interpolate(flat) index : u32,
    @location(4) world : vec3<f32>,
    @location(5) @interpolate(flat) plane : vec4<f32>,
    @location(6) @interpolate(flat) plane_origin : vec4<f32>,
};

@vertex
fn vs_main(in : LineIn) -> LineOut {
    var i : MarkInstance;
    i.position = mark_relative(in.a_hi, in.a_lo);
    i.position2 = mark_relative(in.b_hi, in.b_lo);
    i.size = 1.0;
    i.color = mark_decode(in.color);
    i.color2 = i.color;
    i.custom0 = /*CUSTOM0*/;
    i.custom1 = /*CUSTOM1*/;
    i.visible = 1.0;
    i.index = in.iid;
    i = hook_instance(i);
    var out : LineOut;
    // SwiftShader's line as ANGLE asks for it (Bresenham): each end moved
    // half a pixel each way along the minor axis, by its own w — its setup's
    // arithmetic, in GL's clip space (before any flip). x-major exactly when
    // its "right" or "left" cases hold: `(dx > -dy) == (dx > dy)`.
    let ca = frame.view_proj * vec4<f32>(i.position, 1.0);
    let cb = frame.view_proj * vec4<f32>(i.position2, 1.0);
    let half_w = frame.viewport.x * 0.5;
    let half_h = frame.viewport.y * 0.5;
    let dx = half_w * (cb.x / cb.w - ca.x / ca.w);
    let dy = half_h * (cb.y / cb.w - ca.y / ca.w);
    let x_major = (dx > -dy) == (dx > dy);
    let c = mark_corner_signed(mark_quad_corner(in.vid));
    let far = c.x > 0.0;
    let base = select(ca, cb, far);
    let across = select(vec2<f32>(0.5 * base.w / half_w, 0.0), vec2<f32>(0.0, 0.5 * base.w / half_h), x_major);
    let off = across * c.y;
    let hidden = i.visible <= 0.0 || ca.w <= 0.0 || cb.w <= 0.0 || (dx == 0.0 && dy == 0.0);
    out.clip = select(mark_order(vec4<f32>(base.xy + off, base.z, base.w)), mark_hidden(), hidden);
    if (frame.raster.z > 0.5) {
        let plane = mark_ss_line_plane(mark_order(ca), mark_order(cb));
        out.plane = plane.abc;
        out.plane_origin = plane.origin;
    }
    out.color = i.color;
    out.custom0 = i.custom0;
    out.custom1 = i.custom1;
    out.index = i.index;
    out.world = select(i.position, i.position2, far);
    return out;
}

struct LineFrag {
    @location(0) color : vec4<f32>,
    @builtin(frag_depth) depth : f32,
}

@fragment
fn fs_main(in : LineOut) -> LineFrag {
    if (mark_culled(in.world)) {
        discard;
    }
    let c = hook_color(in.color, in.custom0, in.custom1);
    var o : LineFrag;
    o.color = mark_out(mark_highlight(vec4<f32>(c.rgb, c.a * layer.style2.w), in.index));
    o.depth = mark_ss_depth(in.plane, in.plane_origin, in.clip);
    return o;
}

@fragment
fn fs_pick(in : LineOut) -> @location(0) vec2<u32> {
    if (mark_culled(in.world)) {
        discard;
    }
    return vec2<u32>(layer.ids.x, in.index);
}
"#;

/// Segments drawn one pixel wide by the GPU's line rasterizer, a colour
/// each — deck.gl's wireframes (`gl.LINES`).
///
/// Row `k` is segment `k` for picking and the hooks' `MarkInstance`, whose
/// `position` and `position2` are its ends.
#[derive(Debug)]
pub struct LineMarks {
    pub common: MarkCommon,
    sources: Column,
    targets: Column,
    colors: Column,
    /// Each line a parallelogram — the segment's ends moved half a pixel
    /// each way along its minor axis, by SwiftShader's own arithmetic —
    /// rather than the GPU's line primitive: how SwiftShader draws the
    /// Bresenham lines ANGLE asks for (and so a WebGL `gl.LINES` there), its
    /// depth plane too under
    /// [`MarkView::swiftshader_depth`](super::MarkView::swiftshader_depth).
    pub parallelogram: bool,
    shader: CustomShader,
    quad_shader: CustomShader,
}

impl Default for LineMarks {
    fn default() -> Self {
        Self::new()
    }
}

impl LineMarks {
    pub fn new() -> Self {
        let mut common = MarkCommon::default();
        common.style.depth_write = true;
        let mut colors = Column::new(ColumnFormat::Unorm8x4);
        colors.set_default_rgba([0, 0, 0, 255]);
        Self {
            common,
            sources: Column::new(ColumnFormat::PositionF64),
            targets: Column::new(ColumnFormat::PositionF64),
            colors,
            parallelogram: false,
            shader: CustomShader::new("threers line marks", SOURCE, 5),
            quad_shader: CustomShader::new("threers line marks (parallelograms)", QUAD_SOURCE, 5),
        }
    }

    /// Replace every segment: from `sources[k]` to `targets[k]`, absolute
    /// world positions.
    pub fn set_segments(&mut self, sources: &[[f64; 3]], targets: &[[f64; 3]]) {
        let n = sources.len().min(targets.len());
        self.sources.set_positions(&sources[..n]);
        self.targets.set_positions(&targets[..n]);
    }

    /// sRGB colour per segment, straight alpha.
    pub fn set_colors(&mut self, c: &[[u8; 4]]) {
        self.colors.set_rgba(c);
    }

    pub fn set_default_color(&mut self, c: [u8; 4]) {
        self.colors.set_default_rgba(c);
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl Mark for LineMarks {
    fn kind(&self) -> MarkKind {
        self.custom_shader().map_or(self.shader.kind(), |s| s.kind())
    }
    fn common(&self) -> &MarkCommon {
        &self.common
    }
    fn common_mut(&mut self) -> &mut MarkCommon {
        &mut self.common
    }
    fn instance_count(&self) -> usize {
        self.sources.len()
    }
    fn vertices_per_instance(&self) -> u32 {
        if self.parallelogram { 6 } else { 2 }
    }
    fn line_list(&self) -> bool {
        !self.parallelogram
    }
    fn columns(&mut self) -> Vec<&mut Column> {
        vec![&mut self.sources, &mut self.targets, &mut self.colors]
    }
    fn custom_shader(&self) -> Option<&CustomShader> {
        Some(if self.parallelogram { &self.quad_shader } else { &self.shader })
    }
}
