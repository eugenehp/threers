//! WGSL for your own [shader effects](crate::shader): noise, colour,
//! shapes by distance, and the fills the indicators use (fire, water,
//! plasma, frost). Every name starts `ui_`, out of the way of yours.
//!
//! - [`NOISE`]: `ui_hash`, `ui_noise` (smooth value noise), `ui_fbm` (three
//!   octaves), `ui_voronoi` (cells: nearest, next nearest, which).
//! - [`COLOR`]: `ui_ramp3` (through three colours), `ui_to_linear`,
//!   `ui_to_srgb`, `ui_hsl`, `ui_blackbody`.
//! - [`SDF`]: `ui_cover` (a distance as pixel coverage), `ui_segment`,
//!   `ui_rbox` (a rounded box), `ui_ring`.
//! - [`FX`] (needs noise and colour): `ui_fire`, `ui_water`, `ui_plasma`,
//!   `ui_frost`, each `(along, up, half, env, cover, u, now, c0, c1, c2) ->
//!   vec4` — the [`Fill::Custom`](crate::indicators::Fill::Custom)
//!   signature, so a custom fill can start from one.
//!
//! Put them in front of a shader with [`with`], [`Shader::with_libraries`],
//! or the macro's `use` form:
//!
//! ```
//! use threers_ui::ui_shader;
//!
//! // A card burning from the bottom.
//! let burning = ui_shader! {
//!     use Fx;
//!     fn burning(p: vec2<f32>, color: vec4<f32>, size: vec2<f32>, time: f32) -> vec4<f32> {
//!         let c0 = vec4<f32>(0.5, 0.05, 0.02, 1.0);
//!         let c1 = vec4<f32>(1.0, 0.45, 0.05, 1.0);
//!         let c2 = vec4<f32>(1.0, 0.9, 0.3, 1.0);
//!         let flame = ui_fire(p.x, size.y - p.y - 12.0, 8.0, 1.0, 0.0, 0.0, time, c0, c1, c2);
//!         return flame + color * (1.0 - flame.a);
//!     }
//! };
//! assert_eq!(burning.error(), None);
//! assert_eq!(burning.function(), "burning");
//! ```

/// Noise: `ui_hash`, `ui_noise`, `ui_fbm`, `ui_voronoi`.
pub const NOISE: &str = include_str!("shader_lib/noise.wgsl");
/// Colour: `ui_ramp3`, `ui_to_linear`, `ui_to_srgb`, `ui_hsl`, `ui_blackbody`.
pub const COLOR: &str = include_str!("shader_lib/color.wgsl");
/// Shapes by distance: `ui_cover`, `ui_segment`, `ui_rbox`, `ui_ring`.
pub const SDF: &str = include_str!("shader_lib/sdf.wgsl");
/// Fills: `ui_fire`, `ui_water`, `ui_plasma`, `ui_frost` (needs [`NOISE`]
/// and [`COLOR`]).
pub const FX: &str = include_str!("shader_lib/fx.wgsl");

/// One part of the library.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Library {
    Noise,
    Color,
    Sdf,
    Fx,
}

impl Library {
    /// Every part, each after what it needs.
    pub const ALL: [Library; 4] = [Library::Noise, Library::Color, Library::Sdf, Library::Fx];

    pub fn source(self) -> &'static str {
        match self {
            Library::Noise => NOISE,
            Library::Color => COLOR,
            Library::Sdf => SDF,
            Library::Fx => FX,
        }
    }

    /// The parts it calls.
    pub fn needs(self) -> &'static [Library] {
        match self {
            Library::Fx => &[Library::Noise, Library::Color],
            _ => &[],
        }
    }
}

/// `source` with `libraries` in front, each once and after what it needs.
pub fn with(libraries: &[Library], source: &str) -> String {
    let mut wanted: Vec<Library> = vec![];
    for &l in libraries {
        for &n in l.needs().iter().chain(std::iter::once(&l)) {
            if !wanted.contains(&n) {
                wanted.push(n);
            }
        }
    }
    // In the library's own order, so what is needed comes first.
    let mut out = String::new();
    for l in Library::ALL.into_iter().filter(|l| wanted.contains(l)) {
        out.push_str(l.source());
        out.push('\n');
    }
    out.push_str(source);
    out
}

impl crate::shader::Shader {
    /// A shader from `source` with `libraries` in front (see
    /// [`shader_lib`](self)).
    pub fn with_libraries(libraries: &[Library], source: &str) -> crate::shader::Shader {
        crate::shader::Shader::new(with(libraries, source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parts_come_once_after_what_they_need() {
        let s = with(
            &[Library::Fx, Library::Noise],
            "fn f(p: vec2<f32>) -> vec4<f32> { return vec4<f32>(ui_noise(p)); }",
        );
        assert_eq!(s.matches("fn ui_noise(").count(), 1);
        assert!(s.find("fn ui_noise(").unwrap() < s.find("fn ui_fire(").unwrap());
        assert!(s.find("fn ui_ramp3(").unwrap() < s.find("fn ui_fire(").unwrap());
        assert!(!s.contains("fn ui_rbox("), "only what was asked for");
        let shader = crate::shader::Shader::new(s);
        assert_eq!(
            shader.function(),
            "f",
            "the user's function stays the effect"
        );
    }
}
