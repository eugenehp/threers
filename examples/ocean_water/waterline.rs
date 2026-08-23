//! The waterline: what the sea does to the *lens* when the camera is in it.
//!
//! Everything else in this example draws the ocean. This draws the boundary —
//! the moment half the frame is under water and half is not, which is the one
//! shot no amount of surface shading will get you. Three things happen there,
//! and none of them are properties of the water surface:
//!
//! - the **waterline** itself, which is wave-shaped rather than straight,
//!   because the wave in front of the lens is centimetres away and its own
//!   profile is what cuts the frame;
//! - the **meniscus**, the bright curved band where surface tension pulls water
//!   up against the glass — full of bubbles, collecting light from a wide cone,
//!   and the single most recognisable part of the effect;
//! - **droplets**, which stay on the lens for a few seconds after it comes back
//!   up and run down it as they age.
//!
//! # Why an overlay and not the surface shader
//!
//! The water disc is real geometry, so it does produce a real waterline — but it
//! is cut by the near plane, and pulling the near plane in far enough to keep it
//! costs depth precision across a 20 km scene. This is a sphere of radius 1.2 m
//! centred on the camera instead: every view direction hits it, so no aspect or
//! field-of-view arithmetic is needed, and it is transparent rather than
//! screen-space, so it does not write depth and sorts last of everything.
//!
//! It samples the *same* cascades as the surface, through the same shared code,
//! so the line it draws is the wave that is actually there.

use std::sync::Arc;

use threers::{
    Material, Mesh, Object3D, ObjectId, ObjectKind, PlaneGeometry, Quaternion, Scene,
    ShaderMaterial, Vector3,
};

use crate::ocean_fft::Cascades;
use crate::shader::COMMON_WGSL;

/// How far in front of the camera the quad sits, metres.
///
/// Barely past the near plane, and for a reason. A shell *around* the camera is
/// the obvious shape — every direction hits it, no orientation arithmetic — but
/// a sphere of radius r has view-space depth `r·cos θ`, so at the corners of a
/// wide frustum it dips back inside the near plane and gets clipped, leaving the
/// overlay as a rounded porthole in the middle of the frame. A quad facing the
/// camera has the same depth everywhere on it, so it clips nowhere.
///
/// Sitting 5 cm past the near plane also means nothing can get between it and
/// the lens: geometry closer than that is already clipped away.
const STANDOFF: f32 = 1.05;

/// Seconds of droplets after the lens leaves the water.
const DRY_TIME: f32 = 3.5;

pub struct Waterline {
    pub object: ObjectId,
    /// Off for `--no-waterline`, so the lens effect can be compared against the
    /// surface shading underneath it.
    pub enabled: bool,
    /// Counts down from 1 after the camera surfaces; drives the droplets.
    wetness: f32,
    slots: Vec<[f32; 4]>,
}

impl Waterline {
    pub fn new(
        scene: &mut Scene,
        slots: Vec<[f32; 4]>,
        peak_wavelength: f32,
        cascades: [Arc<wgpu::TextureView>; 3],
        cas: &Cascades,
    ) -> Waterline {
        let src = format!("{COMMON_WGSL}{BODY}").replace(
            "CASCADE_BODY_PLACEHOLDER",
            &crate::waves_gpu::cascade_wgsl(
                "u_tex1",
                "u_tex2",
                "u_tex3",
                "u_samp",
                peak_wavelength,
                cas,
            ),
        );
        let material: Material = ShaderMaterial::new(src)
            .with_data(slots.clone())
            // Screen-space, which here is about *ordering* rather than about
            // reading the capture. The water is a screen-space surface and is
            // drawn in the renderer's glass pass; an ordinary transparent is
            // drawn in the main pass, which runs *before* that. A film on the
            // lens that the water then paints over is not a film on the lens.
            // Screen-space puts this in the same pass, and `render_order` below
            // puts it after the surface within it.
            //
            // It alpha-blends (the renderer blends screen-space customs) and it
            // writes depth, which is harmless: nothing is drawn after it.
            .with_screen_space(true)
            // Double-sided: the quad is oriented at the camera every frame, but
            // one frame of lag during a fast spin would otherwise blink it out.
            .with_side(2)
            .with_textures(vec![
                cascades[0].clone(),
                cascades[0].clone(),
                cascades[1].clone(),
                cascades[2].clone(),
            ])
            .into();

        // A unit quad, scaled to the frustum every frame.
        let mut obj = Object3D::mesh(Mesh::new(PlaneGeometry::new(1.0, 1.0), material));
        // Last within its class, so the surface is drawn before the lens is.
        obj.render_order = 1000;
        // `refract_capture` stays true. Opting out sounds right — this is a lens
        // effect, not scenery — but the renderer's overlay pass has no pipeline
        // for custom materials, so opting out means it is skipped in the main
        // pass *and* unsupported in the overlay pass, and draws nowhere at all.
        obj.visible = false;
        let object = scene.add(obj);

        Waterline {
            object,
            enabled: true,
            wetness: 0.0,
            slots,
        }
    }

    /// Follow the camera and update the film on the lens.
    ///
    /// `submersion` is signed metres of water over the camera, the same quantity
    /// the surface shader is given: positive under, negative above.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        scene: &mut Scene,
        camera_position: Vector3,
        camera_target: Vector3,
        fov: f32,
        aspect: f32,
        t: f32,
        dt: f32,
        submersion: f32,
    ) {
        if submersion > 0.0 {
            self.wetness = 1.0;
        } else {
            self.wetness = (self.wetness - dt / DRY_TIME).max(0.0);
        }
        // Nothing to draw once the lens is dry and the surface is out of reach.
        let active = self.enabled && (submersion > -0.45 || self.wetness > 0.002);

        self.slots[0][0] = t;
        self.slots[10][2] = self.wetness;
        self.slots[14][2] = submersion;

        if let Some(obj) = scene.get_mut(self.object) {
            obj.visible = active;
            if !active {
                return;
            }
            // Sit the quad square in front of the camera, sized to the frustum
            // at that distance with a little margin for the rotation lagging by
            // a frame.
            let fwd = (camera_target - camera_position).normalize();
            obj.position = camera_position + fwd * STANDOFF;
            let h = 2.0 * STANDOFF * (fov * 0.5).tan() * 1.08;
            obj.scale = Vector3::new(h * aspect, h, 1.0);
            obj.quaternion = face_camera(fwd);
            if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                    sm.data.clone_from(&self.slots);
                }
            }
        }
    }

    /// Rebind after a rebuild: new cascades, new preset, same object.
    pub fn reslot(&mut self, slots: Vec<[f32; 4]>) {
        self.slots = slots;
    }
}

/// Rotation taking the quad's local axes to (camera right, camera up, back
/// toward the camera).
///
/// Written out rather than composed from two `from_axis_angle` calls because
/// aligning one axis leaves the roll about it free, and a screen-filling quad
/// with an arbitrary roll does not fill the screen.
fn face_camera(fwd: Vector3) -> Quaternion {
    let world_up = Vector3::new(0.0, 1.0, 0.0);
    // Straight up or down leaves `right` undefined; any horizontal reference
    // will do there, since the roll is then genuinely arbitrary.
    let mut right = fwd.cross(world_up);
    if right.length() < 1e-4 {
        right = Vector3::new(1.0, 0.0, 0.0);
    }
    let right = right.normalize();
    let up = right.cross(fwd).normalize();
    let back = -fwd;

    // Shepperd's method: pick the largest diagonal term to divide by, so the
    // square root never lands on a near-zero denominator.
    let (m00, m10, m20) = (right.x, right.y, right.z);
    let (m01, m11, m21) = (up.x, up.y, up.z);
    let (m02, m12, m22) = (back.x, back.y, back.z);
    let trace = m00 + m11 + m22;
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        Quaternion::new((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s)
    } else if m00 > m11 && m00 > m22 {
        let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
        Quaternion::new(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
        Quaternion::new((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
        Quaternion::new((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    }
}

const BODY: &str = r#"
CASCADE_BODY_PLACEHOLDER

// Effective radius of the front element, metres. This is what sets how wide the
// straddling window is: the waterline exists on the glass only while the surface
// passes through it.
const LENS_R: f32 = 0.075;

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let cam = frame.camera_position.xyz;
    let dir = normalize(in.world_pos - cam);

    // Coarse early-out on the CPU's own estimate of where the surface is. It is
    // built from the swell cascade's dominant modes only, so it is good to a few
    // tens of centimetres — hence the metre of margin before trusting it.
    if (abs(u_submersion()) > 1.2 && u_wetness() < 0.002) { discard; }

    // The surface *at the camera*, from the same shared code the mesh and the
    // pixels use. Uniform across the frame, but it has to be evaluated here: the
    // CPU's estimate has no shoaling and no mid or fine cascade in it, and a
    // waterline is a centimetre-scale event.
    let s = sample_surface(cam.xz, 0.05);
    let n = s.normal;
    // Positive while the lens is under water.
    let lens = s.disp.y - cam.y;

    // Where this pixel's corner of the glass sits relative to the surface. The
    // surface is taken as a plane through the lens over a few centimetres, which
    // it is: `dot(dir, n) / n.y` is how fast the plane recedes along this ray,
    // and it is what tilts the line when the camera rolls or the water does.
    //
    // Probing some distance along the ray instead — the obvious thing — asks a
    // different question, and answers it wrongly: with the lens 20 cm under, it
    // reports a "waterline" wherever a ray happens to break the surface, which
    // is a bright arc across the sky rather than a lens that is simply under.
    let sub = lens - LENS_R * dot(dir, n) / max(n.y, 0.25);

    // Under the line: a thin film. Light, deliberately — the surface shader's
    // own underwater branch and the scene fog already draw what the water does
    // to the view, and a third layer of it is a wash over the frame.
    let submerged_film = 1.0 - smoothstep(0.20, 0.75, lens);
    let under = smoothstep(-0.004, 0.004, sub) * submerged_film;
    // The meniscus: surface tension pulls a curved lens of water up against the
    // glass. It is bright because it gathers light from a wide cone and because
    // it is full of bubbles.
    let rim = 1.0 - smoothstep(0.0, LENS_R * 0.5, abs(sub));
    // Drops are on the *outside* of the glass, so they belong to the moment the
    // lens is back out of the water, not to being under it.
    let drop_gate = clamp(-lens * 5.0, 0.0, 1.0) * u_wetness();

    let sun_col = u_sun_color() * u_sun_intensity();
    let tint = u_scatter() * sun_col * 2.4;

    var col = tint;
    var alpha = under * 0.22;

    // The rim reads as foam rather than as clear water.
    let foam = u_foam_color() * (sun_col * 0.55 + tint * 0.5);
    col = mix(col, foam, rim * rim);
    alpha = max(alpha, rim * rim * 0.85);

    // Droplets, left on the glass after the lens comes back up.
    if (drop_gate > 0.002) {
        let suv = in.clip_pos.xy / frame.viewport_size.xy;
        // One cell per drop, so they cannot pile into a uniform smear.
        let cell = suv * vec2<f32>(19.0, 11.0);
        let id = floor(cell);
        let r = hash21(id);
        let r2 = hash21(id + vec2<f32>(3.1, 7.7));
        // Drops run down the glass as they age.
        let drip = vec2<f32>(0.0, (1.0 - u_wetness()) * (0.3 + 0.5 * r));
        let c = fract(cell) - vec2<f32>(0.3 + 0.4 * r, 0.3 + 0.4 * r2) + drip;
        let rad = 0.10 + 0.15 * r;
        // `step` leaves rather more than half the cells empty, which is what a
        // few large drops look like instead of a regular stipple.
        let drop = (1.0 - smoothstep(rad * 0.55, rad, length(c * vec2<f32>(1.0, 1.35))))
                 * step(0.5, r2);
        if (drop > 0.001) {
            col = mix(col, tint * 1.5 + u_foam_color() * 0.14, 0.55);
            alpha = max(alpha, drop * drop_gate * 0.55);
        }
    }

    if (alpha < 0.003) { discard; }
    return vec4<f32>(framebuffer_encode(soft_clip(col * u_exposure())), clamp(alpha, 0.0, 1.0));
}
"#;
