//! Scene assembly, the per-frame update, and the runtime API.
//!
//! # Reconfiguring at runtime
//!
//! Some of the knobs here are genuinely cheap and some are not, and this module
//! keeps the distinction honest rather than presenting them as though they were
//! all the same:
//!
//! - **Free** — sky, exposure, foam, choppiness. Uniform slots. No allocation.
//! - **Cheap** — wind. Rewrites `h0` and the mode table, ~a millisecond, no
//!   pipeline touched. See [`OceanFft::set_spectrum`].
//! - **A rebuild** — preset, quality, cascade resolution, max scale. The tile
//!   sizes and the peak wavenumber are substituted into every consumer's WGSL,
//!   and the mesh density is a different mesh, so these recreate the world.
//!   Measured in Safari: 360 ms for a preset, 610 ms for the lattice, 650 ms for
//!   a jump to `max`. Nearly all of that is shader compilation. That is fine for
//!   something a person clicks and would be absurd per frame, which is why the
//!   two tiers above exist.

use std::sync::Arc;

use threers::lights::Light;
use threers::scene::FogParams;
use threers::{
    AmbientLight, BufferGeometry, Color, DirectionalLight, HemisphereLight, Material, Mesh,
    Object3D, ObjectId, ObjectKind, PerspectiveCamera, Quaternion, Renderer, Scene, SphereGeometry,
    StandardMaterial, Vector3,
};

use crate::grid::{self, Disc};
use crate::ocean_fft::{Cascades, OceanFft};
use crate::particles::{Kind, Particles};
use crate::preset::{Preset, Quality};
use crate::probe::Probe;
use crate::shader;
use crate::spectrum::{Float, Pose, SeaState};
use crate::surface_state::{SurfaceState, WakeSource};
use crate::terrain;
use crate::waterline::Waterline;
use crate::waves_gpu::WaveCompute;

/// Outer radius of the water disc, metres. Comfortably past the haze, so its
/// edge is never the thing you see.
const DISC_RADIUS: f32 = 6000.0;
/// Innermost ring radius — the finest detail, directly under the viewer.
const DISC_INNER: f32 = 1.6;
/// Screen-space footprint per metre of view distance, feeding the band limit.
const FOOTPRINT_SCALE: f32 = 0.0022;

/// The boat's circular course, and how fast it runs it. Chosen so the wake is
/// long enough to see the wedge open out but the course still fits inside the
/// swell cascade's tile.
const BOAT_RADIUS: f32 = 210.0;
const BOAT_SPEED: f32 = 7.5;

/// Surface height and slope from the cascades' dominant modes.
///
/// The fallback for buoyancy, used until the GPU probe's first answer lands.
/// It is an approximation of the rendered surface — no shoaling, swell cascade
/// only — but of the long waves that actually move a hull it is a good one.
fn fft_sample(fft: &Option<OceanFft>, x: f32, z: f32, t: f32) -> (f32, [f32; 2]) {
    match fft {
        Some(f) => f.sample(x, z, t),
        None => (0.0, [0.0; 2]),
    }
}

/// One probe answer turned back into the (height, slope) pair buoyancy wants.
///
/// The probe reports a normal; a height field's normal is `(-dy/dx, 1, -dy/dz)`
/// normalised, so the slopes come back out by dividing through by `y`.
fn tap_from_probe(v: [f32; 4]) -> (f32, [f32; 2]) {
    let ny = if v[2].abs() < 1e-4 { 1e-4 } else { v[2] };
    (v[0], [-v[1] / ny, -v[3] / ny])
}

pub struct World {
    pub scene: Scene,
    pub camera: PerspectiveCamera,
    pub sea: SeaState,
    /// Kept so the compute pass can be rebuilt and so nothing else bumps the
    /// geometry version — see [`crate::waves_gpu`].
    pub water_geometry: Arc<BufferGeometry>,
    pub base: Vec<[f32; 4]>,
    water_id: ObjectId,
    sky_id: ObjectId,
    seabed_id: ObjectId,
    buoy_id: ObjectId,
    boat_id: ObjectId,
    /// The sun as a scene light, kept addressable so `set_sky` can re-aim it.
    key_id: ObjectId,
    buoy: Float,
    boat: Float,
    /// Where the boat is and how fast — the velocity is what turns its wake from
    /// a stirred patch into a Kelvin wedge.
    boat_pose: ([f32; 2], [f32; 2]),
    /// Fog as the preset asked for it, restored whenever the camera surfaces.
    surface_fog: FogParams,
    /// True while the camera is below the waterline.
    pub submerged: bool,
    /// Linear-space colour everything fades to underwater.
    underwater_tint: [f32; 3],
    /// Foam and wake memory. See [`crate::surface_state`].
    pub state: Option<SurfaceState>,
    /// The cascades themselves. See [`crate::ocean_fft`].
    pub fft: Option<OceanFft>,
    /// Spray, rain and underwater motes. See [`crate::particles`].
    pub particles: Vec<Particles>,
    /// The film on the lens at the waterline. See [`crate::waterline`].
    pub waterline: Option<Waterline>,
    /// Buoyancy read back from the rendered surface. See [`crate::probe`].
    pub probe: Option<Probe>,
    pub preset: Preset,
    pub quality: Quality,
    pub cas: Cascades,
    aspect: f32,
    last_time: f32,
    /// Where the disc sat when the vertices were last written.
    centre: [f32; 2],
}

impl World {
    /// Build with the cascade lattice the quality tier implies.
    ///
    /// `#[allow(dead_code)]` because the desktop example goes through
    /// [`World::build_with`] to honour `--cascade` and `--max-scale`; this is the
    /// entry point the browser build uses.
    #[allow(dead_code)]
    pub fn build(preset: &Preset, quality: &Quality, aspect: f32) -> World {
        World::build_with(
            preset,
            quality,
            aspect,
            Cascades::default().with_resolution(quality.cascade_n),
        )
    }

    /// As [`World::build`], with the cascade lattice given explicitly — which is
    /// what `--max-scale` and the runtime API need.
    pub fn build_with(preset: &Preset, quality: &Quality, aspect: f32, cas: Cascades) -> World {
        let sea = SeaState::from_preset(preset);
        let Disc { geometry, base } =
            grid::build(quality.rings, quality.sectors, DISC_INNER, DISC_RADIUS);

        let mut scene = Scene::new();
        let tint = Color::new(
            preset.horizon_tint[0],
            preset.horizon_tint[1],
            preset.horizon_tint[2],
        );
        scene.background = tint;
        // Linear fog over the same range the water's horizon blend uses, so the
        // two reach the sky together.
        let scene_fog = FogParams {
            color: tint,
            near: preset.haze_near * 0.25,
            far: preset.haze_far,
            density: 0.0,
            mode: 1,
        };
        scene.fog = scene_fog;

        let sun = shader::sun_direction(preset.sun_elevation, preset.sun_azimuth);
        let sun = Vector3::new(sun[0], sun[1], sun[2]);
        let slots = shader::slots(
            preset,
            sea.significant_height,
            FOOTPRINT_SCALE,
            cas.extent(),
        );

        // The dome runs the *same* shader as the water's reflection term. threers
        // ships a Preetham `SkyMaterial` that would do the job, but it tone-maps
        // on its own terms; sharing one fragment is what keeps the horizon
        // invisible. It has to enclose the water disc, since unlike
        // `SkyMaterial` this one writes real depth.
        let sky_id = scene.add(Object3D::mesh(Mesh::new(
            grid::invert_winding(SphereGeometry::new(9000.0, 48, 32)),
            shader::sky_material(slots.clone()),
        )));

        // Sea floor. Its material needs the cascades (for caustics), which do not
        // exist until `attach_waves`, so the mesh goes in now and is re-materialed
        // there.
        let seabed_id = scene.add(Object3D::mesh(Mesh::new(
            terrain::build_seabed(terrain::SEABED_RADIUS, 180, 320),
            StandardMaterial::new(Color::new(
                preset.sand_color[0],
                preset.sand_color[1],
                preset.sand_color[2],
            ))
            .with_roughness(0.95)
            .into(),
        )));

        let water_geometry = Arc::new(geometry);
        let water_id = scene.add(Object3D::mesh(Mesh::from_arc(
            water_geometry.clone(),
            Arc::new(shader::water_material(slots, sea.peak_wavelength, &cas)),
        )));

        // Frame the shot from the sun rather than from a fixed vector: the
        // glitter path only exists between the viewer and the sun, and every
        // preset puts the sun somewhere else.
        let sun_h = {
            let h = Vector3::new(sun.x, 0.0, sun.z);
            if h.length() < 1e-3 {
                Vector3::new(0.0, 0.0, 1.0)
            } else {
                h.normalize()
            }
        };
        let side = Vector3::new(-sun_h.z, 0.0, sun_h.x);

        // A buoy, floating on the same field the surface is drawn from.
        let buoy_pos = sun_h * -150.0 + side * 52.0;
        let buoy = Float {
            anchor: [buoy_pos.x, buoy_pos.z],
            radius: 2.2,
            stiffness: 0.7,
        };
        let buoy_id = scene.add(Object3D::mesh(Mesh::new(
            SphereGeometry::new(2.2, 24, 16),
            StandardMaterial::new(Color::from_hex(0xd63b2a))
                .with_roughness(0.55)
                .with_metalness(0.05)
                .into(),
        )));

        // And a boat, because a wake with nothing making it is a feature you
        // cannot see. It runs a wide circle at a speed that opens the Kelvin
        // wedge out to something worth looking at.
        let boat = Float {
            anchor: [BOAT_RADIUS, 0.0],
            radius: 4.0,
            stiffness: 0.55,
        };
        let mut boat_obj = Object3D::mesh(Mesh::new(
            SphereGeometry::new(3.0, 20, 12),
            StandardMaterial::new(Color::from_hex(0xf2f0ea))
                .with_roughness(0.5)
                .with_metalness(0.05)
                .into(),
        ));
        // A sphere squashed into something hull-shaped: narrow, shallow, long.
        boat_obj.scale = Vector3::new(0.42, 0.4, 1.5);
        let boat_id = scene.add(boat_obj);

        // Exposure is applied inside the water and sky fragments, but the island
        // and the buoy are ordinary lit materials that never see it. Folding it
        // into the light intensities is what keeps a night scene from having a
        // noon island sitting in it.
        let lit = preset.sun_intensity * preset.exposure;

        // Hemisphere rather than plain ambient: with the sun on the horizon the
        // sky is doing nearly all the work above the waterline, and a
        // directionless fill leaves the island a flat cut-out.
        scene.add_light(HemisphereLight::new(
            Color::new(
                0.55 + 0.45 * preset.sun_color[0],
                0.62 + 0.38 * preset.sun_color[1],
                0.85 + 0.15 * preset.sun_color[2],
            ),
            Color::new(
                preset.scatter_color[0] * 2.0,
                preset.scatter_color[1] * 2.0,
                preset.scatter_color[2] * 2.0,
            ),
            0.7 * lit.max(0.28),
        ));
        scene.add_light(AmbientLight::new(
            Color::from_hex(0x404a5a),
            0.15 * lit.max(0.28),
        ));
        // `direction` points from the light toward the scene, hence the negation.
        let mut key = Object3D::light(
            DirectionalLight::new(
                Color::new(
                    preset.sun_color[0],
                    preset.sun_color[1],
                    preset.sun_color[2],
                ),
                lit * 1.6,
            )
            .with_direction(-sun),
        );
        key.position = sun * 500.0;
        let key_id = scene.add(key);

        let mut camera = PerspectiveCamera::new(55.0, aspect, 1.0, 20000.0);
        camera.position = sun_h * -preset.camera_distance
            + side * (preset.camera_distance * 0.36)
            + Vector3::new(0.0, preset.camera_height, 0.0);
        camera.look_at(Vector3::new(0.0, 3.0, 0.0));

        World {
            scene,
            camera,
            sea,
            water_geometry,
            base,
            water_id,
            sky_id,
            seabed_id,
            buoy_id,
            boat_id,
            key_id,
            buoy,
            boat,
            boat_pose: ([BOAT_RADIUS, 0.0], [0.0, 0.0]),
            surface_fog: scene_fog,
            submerged: false,
            underwater_tint: [
                preset.scatter_color[0] * 1.6,
                preset.scatter_color[1] * 1.6,
                preset.scatter_color[2] * 1.6,
            ],
            state: None,
            fft: None,
            particles: Vec::new(),
            waterline: None,
            probe: None,
            preset: *preset,
            quality: *quality,
            cas,
            aspect,
            last_time: 0.0,
            centre: [0.0, 0.0],
        }
    }

    /// Build the compute pass that owns this world's water vertices. The
    /// geometry has to be on the GPU first, which is what `upload_geometry`
    /// forces without waiting for a draw.
    pub fn attach_waves(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut Renderer,
    ) -> WaveCompute {
        renderer.upload_geometry(&self.water_geometry);
        let fft = OceanFft::new(
            device,
            queue,
            &self.preset,
            self.sea.significant_height,
            self.cas,
        );
        // Shoaling needs the dominant wavenumber, so every consumer of the
        // cascades is told the sea's peak wavelength.
        let pw = self.sea.peak_wavelength;
        let waves = WaveCompute::new(device, renderer, &self.water_geometry, &self.base, &fft, pw);
        self.state = Some(SurfaceState::new(device, &fft, pw));
        self.probe = Some(Probe::new(device, &fft, pw));

        // Spray needs the cascades to find its crests, so the systems are built
        // after them. Counts are modest on purpose: these are accents, and a sea
        // that is more spray than water reads as fog.
        let p = self.preset;
        if p.spray > 0.0 {
            self.particles.push(Particles::new(
                device,
                renderer,
                &mut self.scene,
                &fft,
                Kind::Spray,
                12288,
                0.75,
                65.0,
                7.5,
                1.7,
                [1.0, 1.0, 1.0, 0.85 * p.spray],
                pw,
            ));
        }
        if p.rain > 0.0 {
            self.particles.push(Particles::new(
                device,
                renderer,
                &mut self.scene,
                &fft,
                Kind::Rain,
                12288,
                0.045,
                32.0,
                24.0,
                11.0,
                [0.72, 0.78, 0.88, 0.75 * p.rain],
                pw,
            ));
        }
        // Motes exist only below the waterline, and are hidden above it.
        self.particles.push(Particles::new(
            device,
            renderer,
            &mut self.scene,
            &fft,
            Kind::Motes,
            4096,
            0.045,
            22.0,
            0.0,
            1.0,
            [0.80, 0.88, 0.92, 0.65],
            pw,
        ));

        // Now that the cascades exist, give the floor the shader that reads them.
        let slots = shader::slots(
            &self.preset,
            self.sea.significant_height,
            FOOTPRINT_SCALE,
            self.cas.extent(),
        );
        let cascades = [
            fft.cascade_views[0].clone(),
            fft.cascade_views[1].clone(),
            fft.cascade_views[2].clone(),
        ];
        if let Some(obj) = self.scene.get_mut(self.seabed_id) {
            if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                mesh.material = Arc::new(shader::seabed_material(
                    slots.clone(),
                    pw,
                    cascades.clone(),
                    &self.cas,
                ));
            }
        }

        // The lens overlay goes in last, so it is last in the transparent sort
        // as well as nearest to the camera.
        self.waterline = Some(Waterline::new(
            &mut self.scene,
            slots,
            pw,
            cascades,
            &self.cas,
        ));

        self.fft = Some(fft);
        waves
    }

    // -----------------------------------------------------------------------
    // Runtime API
    // -----------------------------------------------------------------------

    /// Rebuild the world around a new preset, quality tier or cascade lattice,
    /// keeping the camera exactly where it is.
    ///
    /// This is the expensive path — a few hundred milliseconds, mostly spent
    /// compiling shaders — and it is expensive for a reason: the tile sizes and
    /// the peak wavenumber are substituted into every consumer's WGSL, and the
    /// mesh density is a different mesh. Everything that *can* be changed
    /// without it — sky, wind, exposure, foam, choppiness — has its own method
    /// below and does not come through here.
    pub fn reconfigure(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &mut Renderer,
        preset: &Preset,
        quality: &Quality,
        cas: Cascades,
    ) -> WaveCompute {
        let (pos, target) = (self.camera.position, self.camera.target);
        let mut next = World::build_with(preset, quality, self.aspect, cas);
        next.camera.position = pos;
        next.camera.look_at(target);
        next.last_time = self.last_time;
        let waves = next.attach_waves(device, queue, renderer);
        *self = next;
        waves
    }

    /// Replace the sea state's look without touching the spectrum.
    ///
    /// Every one of these is a uniform slot, so this is a memcpy into three
    /// materials and costs nothing. `elevation`/`azimuth` are degrees, matching
    /// three.js `Sky`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_sky(
        &mut self,
        elevation: f32,
        azimuth: f32,
        turbidity: f32,
        rayleigh: f32,
        mie: f32,
        mie_g: f32,
        clouds: f32,
    ) {
        self.preset.sun_elevation = elevation;
        self.preset.sun_azimuth = azimuth;
        self.preset.turbidity = turbidity.max(1.0);
        self.preset.rayleigh = rayleigh.max(0.0);
        self.preset.mie_coefficient = mie.max(0.0);
        self.preset.mie_directional_g = mie_g.clamp(0.0, 0.99);
        self.preset.clouds = clouds.clamp(0.0, 1.0);
        self.apply_slots();
        self.relight();
    }

    /// Turn the wind. Direction in radians, speed in m/s.
    ///
    /// Wind enters this ocean in exactly one place — the spectrum seed — so this
    /// rewrites `h0` and nothing else. The shoaling wavenumber baked into the
    /// shaders is left alone: it is derived from the peak wavelength, which
    /// moves by a few percent under a wind change and is only used to decide how
    /// a wave feels the bottom.
    pub fn set_wind(&mut self, queue: &wgpu::Queue, direction: f32, speed: f32) {
        self.preset.wind_dir = direction;
        self.preset.wind_speed = speed.clamp(0.5, 40.0);
        self.sea = SeaState::from_preset(&self.preset);
        if let Some(fft) = &mut self.fft {
            fft.set_spectrum(queue, &self.preset, self.sea.significant_height);
        }
        self.apply_slots();
    }

    /// Artistic knobs, all of them uniform slots.
    pub fn set_look(&mut self, exposure: f32, choppiness: f32, foam: f32) {
        self.preset.exposure = exposure.max(0.0);
        self.preset.choppiness = choppiness.clamp(0.0, 2.0);
        self.preset.foam_amount = foam.clamp(0.0, 3.0);
        self.apply_slots();
        self.relight();
    }

    /// Cut a circular hole in the water surface.
    ///
    /// Slot 15 is read by the water shader as `xy` centre, `z` radius, `w`
    /// feather, and a radius of zero disables it. The shader has always
    /// supported this — it is how a hull or a dock sits *in* the sea rather than
    /// the sea sitting inside it — but nothing drove it until now.
    pub fn set_water_mask(&mut self, centre: [f32; 2], radius: f32, feather: f32) {
        if let Some(obj) = self.scene.get_mut(self.water_id) {
            if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                    sm.data[15] = [centre[0], centre[1], radius, feather];
                }
            }
        }
    }

    /// Height and slope of the rendered surface at a world point.
    ///
    /// The probe's answer where there is one — which is the surface as drawn,
    /// shoaling included — and the dominant modes otherwise.
    pub fn sample_surface(&self, x: f32, z: f32, t: f32) -> (f32, [f32; 2]) {
        fft_sample(&self.fft, x, z, t)
    }

    /// Push the current preset into every material that reads it.
    fn apply_slots(&mut self) {
        let slots = shader::slots(
            &self.preset,
            self.sea.significant_height,
            FOOTPRINT_SCALE,
            self.cas.extent(),
        );
        for id in [self.water_id, self.sky_id, self.seabed_id] {
            if let Some(obj) = self.scene.get_mut(id) {
                if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                    if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                        // Slot 0's time and slot 11's disc centre are written per
                        // frame; leave whatever the last frame put there.
                        let (time, centre) = (sm.data[0][0], sm.data[11]);
                        sm.data.clone_from(&slots);
                        sm.data[0][0] = time;
                        sm.data[11] = centre;
                    }
                }
            }
        }
        if let Some(w) = &mut self.waterline {
            w.reslot(slots);
        }
        let tint = Color::new(
            self.preset.horizon_tint[0],
            self.preset.horizon_tint[1],
            self.preset.horizon_tint[2],
        );
        self.scene.background = tint;
        self.surface_fog = FogParams {
            color: tint,
            near: self.preset.haze_near * 0.25,
            far: self.preset.haze_far,
            density: 0.0,
            mode: 1,
        };
        self.underwater_tint = [
            self.preset.scatter_color[0] * 1.6,
            self.preset.scatter_color[1] * 1.6,
            self.preset.scatter_color[2] * 1.6,
        ];
    }

    /// Re-aim the key light after the sun or the exposure moves.
    ///
    /// The island and the floats are ordinary lit materials: the water's own sun
    /// comes from a uniform, but theirs comes from here. The hemisphere and
    /// ambient fills are not re-aimed — they have no direction to re-aim, and
    /// their level only matters at the extremes a preset switch would rebuild
    /// through anyway.
    fn relight(&mut self) {
        let s = shader::sun_direction(self.preset.sun_elevation, self.preset.sun_azimuth);
        let sun = Vector3::new(s[0], s[1], s[2]);
        let lit = self.preset.sun_intensity * self.preset.exposure;
        if let Some(obj) = self.scene.get_mut(self.key_id) {
            obj.position = sun * 500.0;
            if let ObjectKind::Light(Light::Directional(d)) = &mut obj.kind {
                d.direction = -sun;
                d.intensity = lit * 1.6;
                d.color = Color::new(
                    self.preset.sun_color[0],
                    self.preset.sun_color[1],
                    self.preset.sun_color[2],
                );
            }
        }
    }

    /// Advance to time `t` and encode the frame's displacement.
    ///
    /// Everything the CPU does here is O(waves) or O(1): fold the time term into
    /// the component table, write two small buffers, move two objects. The
    /// vertices are the GPU's business and never come back.
    pub fn update(
        &mut self,
        t: f32,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        waves: &WaveCompute,
    ) {
        // Follow the camera in the horizontal plane only — the disc is a
        // sampling pattern, not part of the scene.
        self.centre = [self.camera.position.x, self.camera.position.z];
        let dt = (t - self.last_time).clamp(0.0, 0.1);
        self.last_time = t;

        // The boat's course. Closed-form in `t` like everything else here, so it
        // does not drift and two machines agree frame for frame.
        let omega = BOAT_SPEED / BOAT_RADIUS;
        let a = t * omega;
        self.boat_pose = (
            [BOAT_RADIUS * a.cos(), BOAT_RADIUS * a.sin()],
            [-BOAT_SPEED * a.sin(), BOAT_SPEED * a.cos()],
        );
        self.boat.anchor = self.boat_pose.0;

        // One encoder for every compute pass this frame. They were six separate
        // encoders and six submits, and a submit is not free — but they cannot
        // share a single *pass*: the cascades are written as storage textures and
        // then read as sampled ones, and wgpu resolves that transition at pass
        // boundaries. So: many passes, one submission.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ocean frame"),
        });

        // Cascades first: everything else this frame reads them.
        if let Some(fft) = &self.fft {
            fft.dispatch(queue, &mut encoder, t, self.preset.choppiness);
        }
        waves.dispatch(queue, &mut encoder, self.centre);

        // Buoyancy, asked of the surface that is actually being drawn. The
        // answer arrives a frame or two later; see `crate::probe`.
        let mut query = Vec::with_capacity(Float::TAPS * 2);
        query.extend_from_slice(&self.buoy.query_points());
        query.extend_from_slice(&self.boat.query_points());
        if let Some(probe) = &mut self.probe {
            probe.dispatch(queue, &mut encoder, &query);
        }
        let answers = self.probe.as_ref().map(|p| p.read()).unwrap_or_default();
        let resolve = |f: &Float, slot: usize| -> Pose {
            if answers.len() >= (slot + 1) * Float::TAPS {
                let taps: Vec<_> = answers[slot * Float::TAPS..(slot + 1) * Float::TAPS]
                    .iter()
                    .map(|v| tap_from_probe(*v))
                    .collect();
                f.resolve_from(&taps)
            } else {
                f.resolve(|x, z| fft_sample(&self.fft, x, z, t))
            }
        };
        let buoy_pose = resolve(&self.buoy, 0);
        let boat_pose = resolve(&self.boat, 1);

        // Foam and wake memory. The boat is what makes the Kelvin wedge; the
        // buoy only stirs the patch it sits in.
        let mut foam_view = None;
        if let Some(state) = &mut self.state {
            let wind = self.preset.wind_dir;
            state.dispatch(
                queue,
                &mut encoder,
                dt,
                [
                    wind.cos() * self.preset.wind_speed * 0.06,
                    wind.sin() * self.preset.wind_speed * 0.06,
                ],
                self.preset.foam_threshold,
                self.preset.foam_softness,
                self.preset.foam_decay,
                &[
                    WakeSource {
                        position: [buoy_pose.position[0], buoy_pose.position[2]],
                        velocity: [0.0, 0.0],
                        strength: 0.55,
                        radius: 7.0,
                        _pad: [0.0; 2],
                    },
                    WakeSource {
                        position: self.boat_pose.0,
                        velocity: self.boat_pose.1,
                        strength: 0.95,
                        radius: 5.0,
                        _pad: [0.0; 2],
                    },
                ],
            );
            foam_view = Some(state.current());
        }
        let cascade_views = self.fft.as_ref().map(|f| {
            (
                f.cascade_views[0].clone(),
                f.cascade_views[1].clone(),
                f.cascade_views[2].clone(),
            )
        });

        if let Some(obj) = self.scene.get_mut(self.water_id) {
            obj.position = Vector3::new(self.centre[0], 0.0, self.centre[1]);
            if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                // Uniquely held, so this mutates in place. Note that the
                // *geometry* is deliberately left alone: touching it would bump
                // its version, the renderer would re-upload, and the compute
                // pass's buffer handle would go stale.
                if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                    sm.data[0][0] = t;
                    sm.data[11] = [self.centre[0], self.centre[1], 0.0, 0.0];
                    // Ping-pong: the view the shader reads alternates each frame.
                    if let (Some(v), Some(fft)) = (foam_view, cascade_views) {
                        sm.textures = vec![v, fft.0, fft.1, fft.2];
                    }
                }
            }
        }
        // The clouds drift, so the dome and the floor need the clock too.
        for id in [self.sky_id, self.seabed_id] {
            if let Some(obj) = self.scene.get_mut(id) {
                if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                    if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                        sm.data[0][0] = t;
                    }
                }
            }
        }

        // Particles. The camera's screen axes are what turn a point into a quad
        // that faces it.
        let fwd = (self.camera.target - self.camera.position).normalize();
        let right = fwd.cross(Vector3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(fwd).normalize();
        let wind = [
            self.preset.wind_dir.cos() * self.preset.wind_speed * 0.35,
            self.preset.wind_dir.sin() * self.preset.wind_speed * 0.35,
        ];
        for sys in &self.particles {
            sys.dispatch(
                queue,
                &mut encoder,
                self.camera.position,
                right,
                up,
                wind,
                dt,
                t,
            );
        }
        queue.submit([encoder.finish()]);
        // Only now is the copy this frame queued actually on its way.
        if let Some(probe) = &mut self.probe {
            probe.after_submit();
        }

        self.place_float(self.buoy_id, &buoy_pose, 0.5, None);
        self.place_float(self.boat_id, &boat_pose, 0.9, Some(self.boat_pose.1));

        // Above or below? One evaluation of the dominant modes, not a collision
        // query against 51 000 triangles.
        let (surf, _) = fft_sample(&self.fft, self.camera.position.x, self.camera.position.z, t);
        let submersion = surf - self.camera.position.y;
        self.submerged = submersion > 0.0;
        // Underwater, visibility collapses and everything goes the colour of the
        // water. The island and the buoy are ordinary lit materials, so this is
        // the only way they hear about it.
        for sys in &self.particles {
            // Motes are suspended *in* the water; spray and rain are above it.
            let want = if sys.kind() == Kind::Motes {
                self.submerged
            } else {
                !self.submerged
            };
            if let Some(obj) = self.scene.get_mut(sys.object) {
                obj.visible = want;
            }
        }
        self.scene.fog = if self.submerged {
            let c = self.underwater_tint;
            FogParams {
                color: Color::new(c[0], c[1], c[2]),
                near: 0.5,
                far: 90.0,
                density: 0.0,
                mode: 1,
            }
        } else {
            self.surface_fog
        };
        if let Some(obj) = self.scene.get_mut(self.water_id) {
            if let ObjectKind::Mesh(mesh) = &mut obj.kind {
                if let Material::Shader(sm) = Arc::make_mut(&mut mesh.material) {
                    sm.data[14][2] = submersion;
                }
            }
        }

        // The film on the lens, last of all: it needs this frame's submersion.
        let (cam, target) = (self.camera.position, self.camera.target);
        let (fov, aspect) = (self.camera.fov, self.camera.aspect);
        if let Some(mut w) = self.waterline.take() {
            w.update(&mut self.scene, cam, target, fov, aspect, t, dt, submersion);
            self.waterline = Some(w);
        }
    }

    /// Sit an object on the water: heave, tilt, and — for something under way —
    /// a heading.
    fn place_float(&mut self, id: ObjectId, pose: &Pose, lift: f32, heading: Option<[f32; 2]>) {
        let Some(obj) = self.scene.get_mut(id) else {
            return;
        };
        obj.position = Vector3::new(pose.position[0], pose.position[1] + lift, pose.position[2]);
        let up = Vector3::new(0.0, 1.0, 0.0);
        let tilt = Vector3::new(pose.up[0], pose.up[1], pose.up[2]);
        let axis = up.cross(tilt);
        let len = axis.length();
        let mut q = Quaternion::identity();
        if len > 1e-5 {
            let angle = up.dot(tilt).clamp(-1.0, 1.0).acos();
            q = Quaternion::from_axis_angle(axis * (1.0 / len), angle);
        }
        if let Some(v) = heading {
            // Yaw first, then the tilt the water imposes on it: a hull points
            // where it is going and rolls about that.
            let yaw = Quaternion::from_axis_angle(up, (-v[1]).atan2(v[0]));
            q = q.multiply(yaw);
        }
        obj.quaternion = q;
    }
}
