use crate::math::{Matrix4, Vector3};

/// Shadow-map configuration carried by lights that can cast shadows.
/// Mirrors three.js's `LightShadow` for the parts the renderer reads.
///
/// These DO drive pixels: the renderer runs a depth pre-pass per casting light
/// and samples it with percentage-closer filtering over a disc `radius` texels
/// wide. (An earlier version of this note said otherwise, long after it
/// stopped being true, which is worth more scepticism than a comment usually
/// earns.)
#[derive(Debug, Clone, Copy)]
pub struct ShadowSettings {
    /// The directional shadow map's size in texels (the map is reallocated
    /// when this changes). Its resolution on the ground is this over
    /// `camera_size`: halve the frustum and every texel covers half the
    /// distance. Several lights of a kind share one map array, at the
    /// largest size any asks for, shrunk to fit a budget of two 4096²
    /// maps' worth of texels per kind.
    pub map_size: u32,
    pub bias: f32,
    pub normal_bias: f32,
    /// Orthographic frustum size (in world units) around the light target.
    pub camera_size: f32,
    pub camera_near: f32,
    pub camera_far: f32,
    /// three.js's `LightShadow.radius`: how soft the shadow's edge is, as the
    /// radius in shadow-map texels of the disc it is filtered over. 1 (the
    /// default) anti-aliases the edge; larger values blur it into a penumbra;
    /// 0 is a single sharp comparison.
    pub radius: f32,
    /// Cascaded shadow maps, for a directional light: 1 (the default) is one
    /// map over the fixed cube `camera_size` describes; 2 to 4 split the
    /// camera's view into that many depth ranges, each with a map fitted to
    /// it, so the near ground gets fine texels and the far ground coarse
    /// ones. Cascades follow the camera, so a moving camera redraws them.
    pub cascades: u32,
    /// How far from the camera the cascades reach, world units (0: the
    /// camera's far plane). Beyond it the light casts no shadow.
    pub cascade_distance: f32,
    /// Where the splits fall, 0 (evenly spaced) to 1 (logarithmic, more
    /// resolution near the camera). 0.6 by default.
    pub cascade_split: f32,
    /// The light's size, for [`ShadowType::Pcss`]: the wider the light, the
    /// softer a shadow's edge far from its caster. World units for point and
    /// spot lights (0.5 by default); for a directional light, its angular
    /// size in radians (0.02 by default, a softened sun). 0 picks those.
    pub light_size: f32,
    /// This light's filter, in place of the renderer's
    /// ([`Renderer::set_shadow_type`](crate::Renderer::set_shadow_type)).
    pub filter: Option<ShadowType>,
}

/// How shadow maps are filtered — three.js's `renderer.shadowMap.type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadowType {
    /// One comparison: hard, aliased edges (`BasicShadowMap`).
    Basic,
    /// Twelve comparisons over a disc `radius` texels wide (`PCFShadowMap`,
    /// the default).
    #[default]
    Pcf,
    /// Twenty-four comparisons over twice the disc, at least two texels:
    /// smoother and softer (`PCFSoftShadowMap`).
    PcfSoft,
    /// Percentage-closer soft shadows: the penumbra widens with the distance
    /// from the caster, by `ShadowSettings::light_size`. Sharp where an
    /// object meets the ground, soft further off. Also what three.js's
    /// `VSMShadowMap` maps to.
    Pcss,
}

impl ShadowType {
    /// The shader's number for it (`ShadowRecord.info.w`).
    pub(crate) fn code(self) -> u32 {
        match self {
            ShadowType::Basic => 0,
            ShadowType::Pcf => 1,
            ShadowType::PcfSoft => 2,
            ShadowType::Pcss => 3,
        }
    }
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            map_size: 4096,
            bias: 0.0,
            normal_bias: 0.0,
            camera_size: 5.0,
            camera_near: 0.1,
            camera_far: 500.0,
            radius: 1.0,
            cascades: 1,
            cascade_distance: 0.0,
            cascade_split: 0.6,
            light_size: 0.0,
            filter: None,
        }
    }
}

/// How many shadow-map texels several lights of one kind share before each
/// map shrinks: 32M, two 4096² maps. One light gets the size it asks for.
pub(crate) const SHADOW_TEXEL_BUDGET: u64 = 32 << 20;

/// The side of each map when `lights` lights of a kind share an array: what
/// they asked for (clamped to 256..8192, a power of two), halved until the
/// array fits [`SHADOW_TEXEL_BUDGET`]. Both renderers size their arrays by it.
pub(crate) fn shadow_layer_size(requested: u32, lights: usize) -> u32 {
    let mut size = requested.clamp(256, 8192).next_power_of_two();
    while lights > 1 && size > 256 && lights as u64 * (size as u64).pow(2) > SHADOW_TEXEL_BUDGET {
        size /= 2;
    }
    size
}

/// How many lights can cast shadows at once (`ShadowRecord`s in the shader).
/// Sixteen keeps the frame uniforms inside WebGL2's 16 KB uniform-block
/// floor; a light past it lights unshadowed.
pub(crate) const MAX_SHADOW_RECORDS: usize = 16;

/// A light that casts this frame, in the order its record takes.
#[derive(Clone, Copy)]
pub(crate) enum Caster {
    Dir {
        dir: Vector3,
        settings: ShadowSettings,
    },
    Spot {
        pos: Vector3,
        dir: Vector3,
        angle: f32,
        settings: ShadowSettings,
    },
    Point {
        pos: Vector3,
        radius: f32,
        settings: ShadowSettings,
    },
}

impl Caster {
    pub(crate) fn settings(&self) -> &ShadowSettings {
        match self {
            Caster::Dir { settings, .. }
            | Caster::Spot { settings, .. }
            | Caster::Point { settings, .. } => settings,
        }
    }

    /// The layers it takes in its kind's map array.
    pub(crate) fn layers(&self) -> usize {
        match self {
            Caster::Dir { settings, .. } => settings.cascades.clamp(1, 4) as usize,
            Caster::Spot { .. } => 1,
            Caster::Point { .. } => 6,
        }
    }
}

/// Cascaded shadow maps (`ShadowSettings::cascades`): the camera's view cut
/// into depth ranges, each with an orthographic light camera fitted around
/// it. Returns each cascade's matrix and its far edge as a view depth.
///
/// Each range's eight corners are bounded by a sphere, not a box: a box
/// turns with the camera and so changes size, which makes the shadow
/// shimmer, and a sphere does not. The sphere's centre is then snapped to
/// the map's texel grid in the light's frame, so a moving camera slides the
/// map by whole texels and the shadow's edge stands still.
pub(crate) fn cascade_frusta(
    view: &Matrix4,
    projection: &Matrix4,
    dir: Vector3,
    settings: &ShadowSettings,
    map_size: u32,
) -> Vec<([f32; 16], f32)> {
    let n = settings.cascades.clamp(1, 4) as usize;
    let inv_proj = projection.invert();
    // The view's depth range, from its projection: depth 0 and 1 back to
    // view space (either way round, so reversed depth reads the same).
    let (d0, d1) = (
        -Vector3::new(0.0, 0.0, 0.0).apply_matrix4(&inv_proj).z,
        -Vector3::new(0.0, 0.0, 1.0).apply_matrix4(&inv_proj).z,
    );
    let (near, far) = (d0.min(d1).max(1e-4), d0.max(d1));
    let reach = if settings.cascade_distance > 0.0 {
        settings.cascade_distance.min(far)
    } else {
        far
    };
    let reach = reach.max(near * 1.01);
    let lambda = settings.cascade_split.clamp(0.0, 1.0);
    let split = |i: usize| {
        let f = i as f32 / n as f32;
        let uniform = near + (reach - near) * f;
        let log = near * (reach / near).powf(f);
        uniform + (log - uniform) * lambda
    };
    // A ray through each corner of the view, in view space: two points on it,
    // where any view depth can be found (perspective and orthographic alike).
    let inv_view = view.invert();
    let rays: Vec<(Vector3, Vector3)> = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
        .iter()
        .map(|&(x, y)| {
            (
                Vector3::new(x, y, 0.25).apply_matrix4(&inv_proj),
                Vector3::new(x, y, 0.75).apply_matrix4(&inv_proj),
            )
        })
        .collect();
    let at_depth = |a: Vector3, b: Vector3, d: f32| {
        let dz = b.z - a.z;
        let t = if dz.abs() < 1e-12 {
            0.0
        } else {
            (-d - a.z) / dz
        };
        (a + (b - a) * t).apply_matrix4(&inv_view)
    };
    let dir = dir.normalize();
    let up = if dir.y.abs() > 0.99 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::UP
    };
    // The light's frame, at the origin: for snapping, which only needs its
    // rotation.
    let light_frame = Matrix4::look_at(Vector3::ZERO, dir, up);
    let light_frame_inv = light_frame.invert();
    (0..n)
        .map(|i| {
            let (d0, d1) = (split(i), split(i + 1));
            let corners: Vec<Vector3> = rays
                .iter()
                .flat_map(|&(a, b)| [at_depth(a, b, d0), at_depth(a, b, d1)])
                .collect();
            let centre = corners.iter().fold(Vector3::ZERO, |acc, &c| acc + c)
                * (1.0 / corners.len() as f32);
            let radius = corners
                .iter()
                .map(|&c| (c - centre).length())
                .fold(0.0f32, f32::max);
            // Rounded up to a sixteenth, so float noise does not resize it.
            let radius = (radius * 16.0).ceil() / 16.0;
            let texel = 2.0 * radius / map_size.max(1) as f32;
            let mut c = centre.apply_matrix4(&light_frame);
            c.x = (c.x / texel).floor() * texel;
            c.y = (c.y / texel).floor() * texel;
            let centre = c.apply_matrix4(&light_frame_inv);
            // Back along the light far enough to catch casters outside the
            // range that throw shadows into it.
            let eye = centre - dir * (radius * 10.0);
            let view = Matrix4::look_at(eye, centre, up);
            let proj = Matrix4::orthographic(-radius, radius, radius, -radius, 0.0, radius * 12.0);
            (proj.multiply(&view).elements, d1)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::shadow_layer_size;

    #[test]
    fn several_lights_share_a_texel_budget() {
        assert_eq!(
            shadow_layer_size(8192, 1),
            8192,
            "one light gets what it asks for"
        );
        assert_eq!(shadow_layer_size(4096, 2), 4096);
        assert_eq!(shadow_layer_size(4096, 4), 2048);
        assert_eq!(shadow_layer_size(4096, 8), 2048);
        assert_eq!(
            shadow_layer_size(1000, 1),
            1024,
            "rounded up to a power of two"
        );
        assert_eq!(shadow_layer_size(100, 3), 256, "and never below 256");
    }
}
