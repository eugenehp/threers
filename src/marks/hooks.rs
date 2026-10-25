//! Hook points: WGSL a caller splices into a mark's shaders.
//!
//! Every mark's vertex stage assembles a [`MarkInstance`](self#the-instance)
//! from its columns and passes it through `hook_instance` before projecting
//! it; every colour stage passes its result through `hook_color`. Both are
//! identity functions unless a [`ShaderHooks`] replaces them, so a hook can
//! move, resize, recolour or hide instances without forking the mark's shader.
//!
//! # The instance
//!
//! ```wgsl
//! struct MarkInstance {
//!     position  : vec3<f32>, // origin-relative world position
//!     position2 : vec3<f32>, // segments: the far end; otherwise = position
//!     size      : f32,       // radius / width / weight, in the layer's units
//!     color     : vec4<f32>, // linear RGB, straight alpha
//!     color2    : vec4<f32>, // segments: colour at the far end
//!     custom0   : vec4<f32>, // the layer's custom columns, zero-padded
//!     custom1   : vec4<f32>,
//!     visible   : f32,       // set to 0 to drop the instance
//!     index     : u32,       // the instance's row
//! };
//! ```
//!
//! `layer.params[0..4]` are four `vec4<f32>` the caller sets per frame with
//! [`MarkStyle::params`](super::MarkStyle::params) — the place for a filter
//! range or an animation time, which change without rebuilding a pipeline.
//!
//! Filtering this way is what keeps a time slider over a million rows
//! interactive: the rows never leave the GPU, and moving the slider writes
//! sixteen bytes.

use std::hash::{Hash, Hasher};

/// The identity hooks, used for whichever of the two a [`ShaderHooks`] leaves
/// unset.
pub(crate) const DEFAULT_INSTANCE: &str =
    "fn hook_instance(i: MarkInstance) -> MarkInstance { return i; }\n";
pub(crate) const DEFAULT_COLOR: &str =
    "fn hook_color(color: vec4<f32>, custom0: vec4<f32>, custom1: vec4<f32>) -> vec4<f32> { return color; }\n";

/// WGSL that replaces a mark's hook functions.
///
/// Each field is the *whole* function, and may be preceded by helpers of its
/// own:
///
/// ```
/// use threers::marks::ShaderHooks;
/// let hooks = ShaderHooks {
///     instance: Some(
///         "fn hook_instance(i: MarkInstance) -> MarkInstance {\n\
///              var o = i;\n\
///              o.size = i.size * (1.0 + 0.5 * sin(layer.params[0].x + f32(i.index)));\n\
///              return o;\n\
///          }".into(),
///     ),
///     color: None,
/// };
/// # #[cfg(not(target_arch = "wasm32"))]
/// assert!(hooks.validate().is_ok());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ShaderHooks {
    /// `fn hook_instance(i: MarkInstance) -> MarkInstance`, in the vertex stage.
    pub instance: Option<String>,
    /// `fn hook_color(color: vec4<f32>, custom0: vec4<f32>, custom1: vec4<f32>) -> vec4<f32>`,
    /// in the fragment stage, on linear straight-alpha colour.
    pub color: Option<String>,
}

impl ShaderHooks {
    /// No hooks.
    pub fn none() -> Self {
        Self::default()
    }

    /// Drop every instance whose custom value is outside `layer.params[param].xy`.
    ///
    /// `component` counts across both custom columns: `0..4` is `custom0.xyzw`,
    /// `4..8` is `custom1.xyzw`. Set the range with
    /// `style.params[param] = [lo, hi, 0.0, 0.0]`; both ends are inclusive.
    pub fn range_filter(component: usize, param: usize) -> Self {
        Self::range_filters(&[(component, param)])
    }

    /// [`range_filter`](Self::range_filter) on several values at once: an
    /// instance is drawn only when every one of them is in range.
    pub fn range_filters(filters: &[(usize, usize)]) -> Self {
        let mut body =
            String::from("fn hook_instance(i: MarkInstance) -> MarkInstance {\n    var o = i;\n");
        for &(component, param) in filters {
            assert!(component < 8, "custom component {component} is not in 0..8");
            assert!(param < 4, "param {param} is not in 0..4");
            let column = if component < 4 { "custom0" } else { "custom1" };
            let lane = ["x", "y", "z", "w"][component % 4];
            body.push_str(&format!(
                "    {{\n        let v = i.{column}.{lane};\n        let r = layer.params[{param}];\n        if (v < r.x || v > r.y) {{ o.visible = 0.0; }}\n    }}\n"
            ));
        }
        body.push_str("    return o;\n}\n");
        Self {
            instance: Some(body),
            color: None,
        }
    }

    /// Fade each instance by its custom value `component` (counted as in
    /// [`range_filter`](Self::range_filter)): its colour's alpha is
    /// multiplied by it, and an instance faded below 1e-4 is not drawn.
    /// deck.gl's `CollisionFilterExtension` fades with this.
    pub fn fade(component: usize) -> Self {
        assert!(component < 8, "custom component {component} is not in 0..8");
        let column = if component < 4 { "custom0" } else { "custom1" };
        let lane = ["x", "y", "z", "w"][component % 4];
        Self {
            instance: Some(format!(
                "fn hook_instance(i: MarkInstance) -> MarkInstance {{\n    var o = i;\n    let f = i.{column}.{lane};\n    o.color.a *= f;\n    o.color2.a *= f;\n    if (f < 0.0001) {{ o.visible = 0.0; }}\n    return o;\n}}\n"
            )),
            color: None,
        }
    }

    /// Animated trails over [`SegmentMarks`](super::SegmentMarks) — deck.gl's
    /// `TripsLayer`.
    ///
    /// Each segment carries the time its source and its target are reached,
    /// as custom components `times` and `times + 1` (counted as in
    /// [`range_filter`](Self::range_filter)). With
    /// `style.params[param] = [now, trail, 0.0, 0.0]`, every segment is cut
    /// to the part between `now - trail` and `now`, so the head sits exactly
    /// where `now` is, and fades from opaque at the head to clear at the tail.
    /// Advancing `now` is a uniform write, however many trips there are.
    ///
    /// ```
    /// use threers::marks::{SegmentMarks, ShaderHooks};
    /// let mut trip = SegmentMarks::new();
    /// trip.set_paths(&[vec![[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 0.0, -10.0]]]);
    /// // Reached at t = 0, 1 and 2: per segment, (source time, target time).
    /// trip.common.set_custom(0, 2, &[0.0, 1.0, 1.0, 2.0]);
    /// trip.common.hooks = ShaderHooks::trail(0, 0);
    /// trip.common.style.params[0] = [1.5, 0.75, 0.0, 0.0];
    /// ```
    pub fn trail(times: usize, param: usize) -> Self {
        assert!(
            times < 7,
            "custom components {times} and {} are not both in 0..8",
            times + 1
        );
        assert!(param < 4, "param {param} is not in 0..4");
        let lane = |c: usize| {
            format!(
                "i.{}.{}",
                if c < 4 { "custom0" } else { "custom1" },
                ["x", "y", "z", "w"][c % 4]
            )
        };
        let (t0, t1) = (lane(times), lane(times + 1));
        Self {
            instance: Some(format!(
                "fn hook_instance(i: MarkInstance) -> MarkInstance {{
    var o = i;
    let t0 = {t0};
    let t1 = {t1};
    let now = layer.params[{param}].x;
    let trail = max(layer.params[{param}].y, 1e-6);
    let tail = now - trail;
    if (t0 > now || t1 < tail) {{
        o.visible = 0.0;
        return o;
    }}
    // Cut the segment to [tail, now], then fade by how long ago each end was.
    let span = max(t1 - t0, 1e-9);
    let a = clamp((tail - t0) / span, 0.0, 1.0);
    let b = clamp((now - t0) / span, 0.0, 1.0);
    o.position = mix(i.position, i.position2, a);
    o.position2 = mix(i.position, i.position2, b);
    o.color.a = i.color.a * clamp((mix(t0, t1, a) - tail) / trail, 0.0, 1.0);
    o.color2.a = i.color2.a * clamp((mix(t0, t1, b) - tail) / trail, 0.0, 1.0);
    return o;
}}
"
            )),
            color: None,
        }
    }

    /// Animate rows from where they were to where they are — clusters
    /// opening, points moving between time steps.
    ///
    /// Custom components `offset..offset + 3` hold each row's old position
    /// minus its new one ([`transition_offsets`] computes them), and, with
    /// `size`, component `size` its old size. `style.params[param].x` runs
    /// from 0 (all old) to 1 (all new); ease it on the CPU as you like. The
    /// rows' columns hold the new state throughout, so finishing is setting
    /// the clock to 1 — or dropping the hook. Applies to `position2` too, so
    /// segments move whole.
    ///
    /// ```
    /// use threers::marks::{transition_offsets, PointMarks, ShaderHooks};
    /// let (old, new) = ([[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]]);
    /// let mut pts = PointMarks::new();
    /// pts.set_positions(&new);
    /// pts.common.set_custom(0, 3, &transition_offsets(&old, &new));
    /// pts.common.hooks = ShaderHooks::transition(0, None, 1);
    /// pts.common.style.params[1] = [0.25, 0.0, 0.0, 0.0]; // a quarter of the way
    /// ```
    pub fn transition(offset: usize, size: Option<usize>, param: usize) -> Self {
        assert!(
            offset + 3 <= 8,
            "custom components {offset}..{} are not in 0..8",
            offset + 3
        );
        assert!(param < 4, "param {param} is not in 0..4");
        let lane = |c: usize| {
            format!(
                "i.{}.{}",
                if c < 4 { "custom0" } else { "custom1" },
                ["x", "y", "z", "w"][c % 4]
            )
        };
        let d = format!(
            "vec3<f32>({}, {}, {})",
            lane(offset),
            lane(offset + 1),
            lane(offset + 2)
        );
        let size_line = size
            .map(|c| {
                assert!(c < 8, "custom component {c} is not in 0..8");
                format!("    o.size = mix({}, i.size, t);\n", lane(c))
            })
            .unwrap_or_default();
        Self {
            instance: Some(format!(
                "fn hook_instance(i: MarkInstance) -> MarkInstance {{
    var o = i;
    let t = clamp(layer.params[{param}].x, 0.0, 1.0);
    let back = {d} * (1.0 - t);
    o.position = i.position + back;
    o.position2 = i.position2 + back;
{size_line}    return o;
}}
"
            )),
            color: None,
        }
    }

    /// Both hooks in sequence: `self`'s instance hook, then `then`'s, and the
    /// same for colour. A transition with a filter, a trail with a
    /// recolouring.
    pub fn combine(&self, then: &ShaderHooks) -> ShaderHooks {
        let chain = |a: &Option<String>, b: &Option<String>, name: &str, call: &str| match (a, b) {
            (None, None) => None,
            (Some(x), None) | (None, Some(x)) => Some(x.clone()),
            (Some(x), Some(y)) => Some(format!(
                "{}\n{}\n{}",
                x.replace(&format!("fn {name}("), &format!("fn {name}_first(")),
                y.replace(&format!("fn {name}("), &format!("fn {name}_second(")),
                call
            )),
        };
        ShaderHooks {
            instance: chain(
                &self.instance,
                &then.instance,
                "hook_instance",
                "fn hook_instance(i: MarkInstance) -> MarkInstance {\n    return hook_instance_second(hook_instance_first(i));\n}\n",
            ),
            color: chain(
                &self.color,
                &then.color,
                "hook_color",
                "fn hook_color(color: vec4<f32>, custom0: vec4<f32>, custom1: vec4<f32>) -> vec4<f32> {\n    return hook_color_second(hook_color_first(color, custom0, custom1), custom0, custom1);\n}\n",
            ),
        }
    }

    /// Both hooks as WGSL, defaults filled in.
    pub(crate) fn source(&self) -> String {
        let mut s = String::new();
        s.push_str(self.instance.as_deref().unwrap_or(DEFAULT_INSTANCE));
        s.push('\n');
        s.push_str(self.color.as_deref().unwrap_or(DEFAULT_COLOR));
        s.push('\n');
        s
    }

    pub(crate) fn key(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Compile-check the hooks against every kind of mark, without a GPU.
    ///
    /// Line numbers in an error are counted from the start of the assembled
    /// module; the message says where the hook source begins. Always `Ok` on
    /// `wasm32`, where naga is not built in — see
    /// [`ShaderMaterial::VALIDATION_AVAILABLE`](crate::ShaderMaterial::VALIDATION_AVAILABLE).
    pub fn validate(&self) -> Result<(), String> {
        for kind in super::shader::HOOKED_KINDS {
            let source = super::shader::module_source(*kind, self, [4, 4]);
            let offset = super::shader::hook_line(&source);
            crate::renderer::validate_custom_shader(&source, 0).map_err(|e| {
                format!("{kind:?} mark: {e}\n(hook source starts at line {offset} of the module)")
            })?;
        }
        Ok(())
    }
}

/// Old position minus new, three floats a row — the custom column
/// [`ShaderHooks::transition`] reads. Rows past `from`'s end start where they
/// end up.
pub fn transition_offsets(from: &[[f64; 3]], to: &[[f64; 3]]) -> Vec<f32> {
    to.iter()
        .enumerate()
        .flat_map(|(i, t)| {
            let f = from.get(i).unwrap_or(t);
            [
                (f[0] - t[0]) as f32,
                (f[1] - t[1]) as f32,
                (f[2] - t[2]) as f32,
            ]
        })
        .collect()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_filters_compile_for_every_mark() {
        ShaderHooks::none().validate().unwrap();
        ShaderHooks::range_filter(0, 0).validate().unwrap();
        ShaderHooks::fade(4).validate().unwrap();
        ShaderHooks::range_filter(0, 0).combine(&ShaderHooks::fade(4)).validate().unwrap();
        ShaderHooks::range_filters(&[(1, 0), (6, 3)])
            .validate()
            .unwrap();
        ShaderHooks::trail(0, 0).validate().unwrap();
        ShaderHooks::transition(0, Some(3), 1).validate().unwrap();
        ShaderHooks::transition(5, None, 0)
            .combine(&ShaderHooks::range_filter(0, 2))
            .validate()
            .unwrap();
        ShaderHooks::trail(3, 2).validate().unwrap();
    }

    #[test]
    fn a_broken_hook_is_reported_not_swallowed() {
        let hooks = ShaderHooks {
            instance: Some(
                "fn hook_instance(i: MarkInstance) -> MarkInstance { return 1.0; }".into(),
            ),
            color: None,
        };
        let err = hooks.validate().unwrap_err();
        assert!(err.contains("hook source starts at line"), "{err}");
    }
}
