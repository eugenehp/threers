//! PMREM pipeline decomposed into verifiable layers.
//!
//! Run all CPU layers (must pass before parity):
//! ```text
//! cargo test pmrem_layer_ -- --nocapture
//! ```
//!
//! Run the three.js atlas-content gate:
//! ```text
//! cargo test pmrem_layer_09_threejs_atlas_probes -- --nocapture
//! ```
//!
//! ## Layer map
//!
//! | Layer | Area | What it checks |
//! |-------|------|----------------|
//! | L01 | **Geometry** | `pmrem_get_direction` face normals |
//! | L02 | **Textures** | sRGB cubemap → linear PMREM input |
//! | L03 | **Pipeline meta** | LOD size/sigma chain counts |
//! | L04 | **Atlas pack** | pack → extract roundtrip per face |
//! | L05 | **Blur** | isolated magenta OK; parity vs three.js column probes |
//! | L06 | **CPU sampling** | `roughness_to_mip` + mip blend consistency |
//! | L07 | **CPU CubeUV** | unblurred mip0 +Z probe after pack |
//! | L08 | **GPU texture** | CPU atlas bytes == GPU readback texel |
//! | L09 | **Atlas content** | CPU probes vs three.js (blur gate) |
//! | L10 | **GPU shader** | WGSL `texture_cube_uv` == CPU |
//! | L11 | **IBL diagnostic** | front-sphere CPU IBL @ metal r=0.45 |
//! | L12 | **Grazing probes** | off-axis env samples (pixel 366,178) |
//! | L13 | **Row convention** | grazing +Z must not read -Y yellow (Y-flip regression) |
//!
//! Full-scene parity (`tests/parity` `pmrem` scene) is the final gate after L01–L12.

#[cfg(test)]
mod tests {
    use super::super::cube_uv::{
        atlas_bilinear_uv, atlas_lod_origin, extract_cube_faces_from_atlas_lod, pack_cube_uv_atlas,
        pmrem_face_uv, pmrem_get_direction, read_atlas_texel, roughness_to_mip_cube_uv,
        sample_atlas_bilinear, sample_cube_uv_env, PMREM_SLOT_TO_FACE, PMREM_SLOT_TO_FACE_MIP0,
    };
    use super::super::pmrem::PmremGenerator;
    use crate::textures::{CubeTexture, TextureFormat};

    const EPS: f32 = 0.02;
    /// Grazing-angle mips blend; slightly looser than axis probes.
    const EPS_GRAZING: f32 = 0.05;

    /// three.js `threejs-pmrem-atlas-sample.html` reference (size=32 parity cubemap, sRGB input).
    mod threejs_ref {
        /// Canvas byte readback — linear HalfFloat atlas displayed with LinearSRGBColorSpace.
        pub const ENV_PLUS_Z_ROUGH_045: [f32; 3] = [0.9940, 0.0564, 0.9903];
        pub const PROBE_183_56: [f32; 3] = [0.9804, 0.0706, 0.9725];
        pub const PROBE_136_56: [f32; 3] = [1.0, 0.0510, 1.0];
        pub const PROBE_40_56: [f32; 3] = [1.0, 0.0510, 1.0];
        pub const PLUS_Z_MIP4_BYTE: [f32; 3] = [1.0, 0.0510, 1.0];
        /// Pixel (366,178) grazing reflect / normal — parity sphere metal r=0.45.
        pub const REFLECT_GRAZING_ENV_BYTE: [f32; 3] = [0.5450, 0.0513, 0.9766];
        pub const NORMAL_GRAZING_ENV_BYTE: [f32; 3] = [0.7761, 0.1247, 0.8511];
    }

    fn solid(size: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut d = vec![0u8; (size * size * 4) as usize];
        for i in 0..(size * size) as usize {
            d[i * 4..i * 4 + 4].copy_from_slice(&[r, g, b, 255]);
        }
        d
    }

    /// Parity-scene cubemap: +X red, -X green, +Y blue, -Y yellow, +Z magenta, -Z cyan.
    fn parity_cube(size: u32) -> CubeTexture {
        CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                solid(size, 255, 64, 64),
                solid(size, 64, 255, 64),
                solid(size, 64, 64, 255),
                solid(size, 255, 255, 64),
                solid(size, 255, 64, 255),
                solid(size, 64, 255, 255),
            ],
        )
    }

    fn assert_close3(label: &str, got: [f32; 3], expected: [f32; 3], eps: f32) {
        for c in 0..3 {
            assert!(
                (got[c] - expected[c]).abs() <= eps,
                "{label} ch={c}: got {} expected {}",
                got[c],
                expected[c]
            );
        }
    }

    fn assert_close(label: &str, got: f32, expected: f32, eps: f32) {
        assert!(
            (got - expected).abs() <= eps,
            "{label}: got {got} expected {expected}"
        );
    }

    /// Map stored linear atlas samples to canvas/sRGB display bytes (three.js readback scale).
    fn linear_to_display_rgb(linear: [f32; 3]) -> [f32; 3] {
        fn enc(c: f32) -> f32 {
            let c = c.clamp(0.0, 1.0);
            let s = if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (s * 255.0).round() / 255.0
        }
        [enc(linear[0]), enc(linear[1]), enc(linear[2])]
    }

    // ── L01 Geometry ───────────────────────────────────────────────────────

    #[test]
    fn pmrem_layer_01_directions_face_normals() {
        let cases: [(u32, [f32; 3]); 6] = [
            (0, [1.0, 0.0, 0.0]),
            (1, [0.0, 1.0, 0.0]),
            (2, [0.0, 0.0, 1.0]),
            (3, [-1.0, 0.0, 0.0]),
            (4, [0.0, -1.0, 0.0]),
            (5, [0.0, 0.0, -1.0]),
        ];
        for (face, want) in cases {
            let d = pmrem_get_direction(face, 0.5, 0.5);
            assert_close3(&format!("face {face} center"), d, want, 1e-4);
        }
    }

    #[test]
    fn pmrem_layer_01_gutter_uv_extends_outside_unit_square() {
        let (u0, v0) = pmrem_face_uv(16, 0, 0);
        let (u1, v1) = pmrem_face_uv(16, 15, 15);
        assert!(
            u0 < 0.0 && v0 < 0.0,
            "corner gutter UV should extend below 0: ({u0}, {v0})"
        );
        assert!(
            u1 > 1.0 && v1 > 1.0,
            "corner gutter UV should extend above 1: ({u1}, {v1})"
        );
    }

    // ── L02 Textures ───────────────────────────────────────────────────────

    #[test]
    fn pmrem_layer_02_source_linearizes_srgb() {
        let src = parity_cube(4);
        let pmrem = PmremGenerator::generate_pmrem(&src, 4);
        let face = &pmrem.faces[4];
        let g = face[(2 * 4 + 2) * 4 + 1] as f32 / 255.0;
        assert!(g < 0.08 && g > 0.03, "linear +Z green channel: {g}");
    }

    // ── L03 Pipeline metadata ──────────────────────────────────────────────

    #[test]
    fn pmrem_layer_03_lod_chain_level_counts() {
        let pm32 = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let sizes32 = pm32.pmrem_sizes.as_ref().unwrap();
        assert_eq!(sizes32.len(), 8, "size=32 lod_max=5 → 8 PMREM levels");
        assert_eq!(sizes32[0], 32);
        assert_eq!(*sizes32.last().unwrap(), 16);

        let pm256 = PmremGenerator::generate_pmrem(&parity_cube(256), 256);
        let sizes256 = pm256.pmrem_sizes.as_ref().unwrap();
        assert_eq!(sizes256.len(), 11, "size=256 lod_max=8 → 11 PMREM levels");
    }

    #[test]
    fn pmrem_layer_03_atlas_dimensions() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        assert_eq!(atlas.width, 336);
        assert_eq!(atlas.height, 128);
        assert_eq!(atlas.cube_size, 32);
        assert_eq!(atlas.lod_max, 5);
    }

    // ── L04 Atlas pack / extract ───────────────────────────────────────────

    #[test]
    fn pmrem_layer_04_mip0_samples_neighbor_faces() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let plus_z = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 5.0);
        let minus_x = sample_atlas_bilinear(atlas, [-1.0, 0.0, 0.0], 5.0);
        let diag = sample_atlas_bilinear(atlas, [-0.7, 0.0, 0.7], 5.0);
        eprintln!("L04 mip0 +Z {plus_z:?} -X {minus_x:?} diag {diag:?}");
        assert!(plus_z[1] < 0.1, "+Z green {}", plus_z[1]);
        assert!(
            minus_x[1] < 0.1,
            "-X slot (flipEnvMap) green {}",
            minus_x[1]
        );
        assert!(
            (diag[0] - plus_z[0]).abs() > 0.05 || (diag[2] - plus_z[2]).abs() > 0.05,
            "diagonal should differ from +Z center: {diag:?} vs {plus_z:?}"
        );
    }

    #[test]
    fn pmrem_layer_04_pack_extract_roundtrip() {
        let size = 16;
        let cube_size = 32;
        let lod_max = 5;
        let faces = [
            solid(size, 255, 0, 0),
            solid(size, 0, 255, 0),
            solid(size, 0, 0, 255),
            solid(size, 255, 255, 0),
            solid(size, 255, 0, 255),
            solid(size, 0, 255, 255),
        ];
        let atlas = pack_cube_uv_atlas(std::slice::from_ref(&faces), &[size], cube_size, lod_max);
        let back = extract_cube_faces_from_atlas_lod(
            &atlas.pixels,
            atlas.width,
            cube_size,
            lod_max,
            0,
            size,
        );
        for cube_face in 0..6 {
            let inner = size - 2;
            for y in 1..1 + inner {
                for x in 1..1 + inner {
                    let off = ((y * size + x) * 4) as usize;
                    assert_eq!(
                        back[cube_face][off..off + 3],
                        faces[cube_face][off..off + 3],
                        "face {cube_face} pixel ({x},{y})"
                    );
                }
            }
        }
    }

    #[test]
    fn pmrem_layer_04_slot_mapping_covers_all_faces() {
        for table in [PMREM_SLOT_TO_FACE, PMREM_SLOT_TO_FACE_MIP0] {
            let mut seen = [false; 6];
            for &f in &table {
                seen[f] = true;
            }
            assert!(
                seen.iter().all(|&v| v),
                "slot table must be a permutation: {table:?}"
            );
        }
    }

    #[test]
    fn pmrem_layer_04_atlas_lod_origin_mip0() {
        let (x, y) = atlas_lod_origin(0, 32, 32, 5);
        assert_eq!(x, 0);
        assert_eq!(y, 64, "mip0 tile sits in lower half of 128px-tall atlas");
    }

    // ── L05 Blur math ──────────────────────────────────────────────────────

    #[test]
    fn pmrem_layer_05_first_blur_column_parity_size32() {
        let size = 32;
        let black = solid(size, 20, 20, 20);
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            [
                black.clone(),
                black.clone(),
                black.clone(),
                black.clone(),
                solid(size, 255, 64, 255),
                black,
            ],
        );
        let pm = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 4.0);
        eprintln!("L05 isolated size=32 mip_int=4 {c:?} (three.js column40 ≈ 0.25 G)");
        let parity = PmremGenerator::generate_pmrem(&parity_cube(size), size);
        let pa = parity.cube_uv_atlas.as_ref().unwrap();
        let cp = sample_atlas_bilinear(pa, [0.0, 0.0, 1.0], 4.0);
        eprintln!("L05 parity size=32 mip_int=4 {cp:?}");
    }

    #[test]
    fn pmrem_layer_05_blur_isolated_magenta_stays_magenta() {
        let size = 16;
        let black = solid(size, 0, 0, 0);
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8Unorm,
            [
                black.clone(),
                black.clone(),
                black.clone(),
                black.clone(),
                solid(size, 255, 64, 255),
                black,
            ],
        );
        let pm = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 4.0);
        eprintln!("L05 isolated +Z mip_int=4 atlas sample {c:?}");
        assert!(
            c[0] > c[1] * 2.0 && c[2] > c[1] * 2.0,
            "mip1 +Z should stay magenta-dominant, got {c:?}"
        );
    }

    #[test]
    fn pmrem_layer_05_mip1_plus_z_not_neon_green() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 1.0);
        assert!(c[1] < 0.35, "+Z mip_int=1 green: {} (target ≈0.26)", c[1]);
    }

    #[test]
    fn pmrem_layer_05_first_blur_matches_threejs_column40() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 4.0);
        eprintln!(
            "L05 +Z mip_int=4 lin {c:?} three.js column40 {:?}",
            threejs_ref::PROBE_40_56
        );
        assert_close3("+Z mip_int=4", c, threejs_ref::PLUS_Z_MIP4_BYTE, EPS);
    }

    #[test]
    fn pmrem_layer_05_mip1_green_diagnostic() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 1.0);
        eprintln!("L05 parity +Z mip_int=1 {c:?} (linear, three.js display G≈0.26)");
        assert!(
            c[1] < 0.35,
            "+Z mip_int=1 green should not overshoot: {}",
            c[1]
        );
    }

    // ── L06 CPU sampling math ──────────────────────────────────────────────

    #[test]
    fn pmrem_layer_06_roughness_to_mip_at_045() {
        let mip = roughness_to_mip_cube_uv(0.45);
        assert_close("roughness 0.45 → mip", mip, 1.625, 1e-3);
    }

    #[test]
    fn pmrem_layer_06_sample_cube_uv_matches_manual_blend() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let rough = 0.45;
        let mip = roughness_to_mip_cube_uv(rough);
        let mip_i = mip.floor();
        let mip_f = mip - mip_i;
        let c0 = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mip_i);
        let c1 = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], mip_i + 1.0);
        let manual = [
            c0[0] * (1.0 - mip_f) + c1[0] * mip_f,
            c0[1] * (1.0 - mip_f) + c1[1] * mip_f,
            c0[2] * (1.0 - mip_f) + c1[2] * mip_f,
        ];
        let env = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], rough);
        assert_close3("sample_cube_uv_env manual blend", env, manual, 1e-5);
    }

    // ── L07 CPU CubeUV on unblurred mip0 ───────────────────────────────────

    #[test]
    fn pmrem_layer_07_mip0_plus_z_reads_magenta_slot() {
        let size = 32;
        let src = CubeTexture::new(
            size,
            TextureFormat::Rgba8UnormSrgb,
            std::array::from_fn(|f| {
                let (r, g, b) = match f {
                    4 => (255, 64, 255),
                    _ => (20, 20, 20),
                };
                solid(size, r, g, b)
            }),
        );
        let pm = PmremGenerator::generate_pmrem(&src, size);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let c = sample_atlas_bilinear(atlas, [0.0, 0.0, 1.0], 5.0);
        assert!(c[0] > 0.8, "R={}", c[0]);
        assert!(c[1] < 0.35, "G={}", c[1]);
        assert!(c[2] > 0.8, "B={}", c[2]);
    }

    // ── L09 Atlas content gate (three.js) ──────────────────────────────────

    /// Compare CPU atlas probes to three.js PMREMGenerator output.
    #[test]
    fn pmrem_layer_09_threejs_atlas_probes() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();

        let env = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], 0.45);
        assert_close3(
            "+Z env @ rough 0.45",
            env,
            threejs_ref::ENV_PLUS_Z_ROUGH_045,
            EPS,
        );

        let p183 = read_atlas_texel(
            &atlas.pixels,
            atlas.width,
            atlas.height,
            183.0 / 335.0,
            56.0 / 127.0,
        );
        assert_close3("atlas texel (183,56)", p183, threejs_ref::PROBE_183_56, EPS);

        let p136 = read_atlas_texel(
            &atlas.pixels,
            atlas.width,
            atlas.height,
            136.0 / 335.0,
            56.0 / 127.0,
        );
        assert_close3("atlas texel (136,56)", p136, threejs_ref::PROBE_136_56, EPS);
    }

    #[test]
    fn pmrem_layer_09_per_mip_plus_z_diagnostic() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let dir = [0.0, 0.0, 1.0];
        for mip in -2..=5 {
            let c = sample_atlas_bilinear(atlas, dir, mip as f32);
            eprintln!("ours +Z mip_int={mip} {c:?}");
        }
        if let Some(mips) = &pm.pmrem_mips {
            for (i, faces) in mips.iter().enumerate() {
                let face = &faces[4];
                let s = pm.pmrem_sizes.as_ref().unwrap()[i];
                let mid = ((s / 2) * s + s / 2) as usize * 4;
                let g = face[mid + 1] as f32 / 255.0;
                eprintln!("ours mip_faces[{i}] +Z center G={g:.4}");
            }
        }
        for (x, y) in [(40u32, 56), (88, 56), (136, 56), (183, 56), (232, 56)] {
            let t = read_atlas_texel(
                &atlas.pixels,
                atlas.width,
                atlas.height,
                x as f32 / 335.0,
                y as f32 / 127.0,
            );
            eprintln!("ours atlas byte ({x},{y}) {t:?}");
        }
    }

    #[test]
    fn pmrem_layer_09_report_current_probe_gap() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let env_lin = sample_cube_uv_env(atlas, [0.0, 0.0, 1.0], 0.45);
        let p183 = read_atlas_texel(
            &atlas.pixels,
            atlas.width,
            atlas.height,
            183.0 / 335.0,
            56.0 / 127.0,
        );
        let p136 = read_atlas_texel(
            &atlas.pixels,
            atlas.width,
            atlas.height,
            136.0 / 335.0,
            56.0 / 127.0,
        );
        eprintln!(
            "L09 gap +Z env ours={env_lin:?} three.js={:?}",
            threejs_ref::ENV_PLUS_Z_ROUGH_045
        );
        eprintln!(
            "L09 gap (183,56) ours={p183:?} three.js={:?}",
            threejs_ref::PROBE_183_56
        );
        eprintln!(
            "L09 gap (136,56) ours={p136:?} three.js={:?}",
            threejs_ref::PROBE_136_56
        );
        assert_close3(
            "+Z env @ rough 0.45",
            env_lin,
            threejs_ref::ENV_PLUS_Z_ROUGH_045,
            EPS,
        );
        assert_close3("atlas texel (183,56)", p183, threejs_ref::PROBE_183_56, EPS);
        assert_close3("atlas texel (136,56)", p136, threejs_ref::PROBE_136_56, EPS);
    }

    /// CPU IBL at front-sphere configuration (metal r=0.45) — matches parity PNG center.
    #[test]
    fn pmrem_layer_11_ibl_front_sphere_diagnostic() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let n = [0.0, 0.0, 1.0];
        let v = [0.0, 0.0, 1.0];
        let (radiance, irr, lit) = ibl_metal_lit(atlas, n, v, 0.45, 1.0, [1.0; 3]);
        let display = linear_to_display_rgb(lit);
        eprintln!("L11 radiance {radiance:?} irr {irr:?}");
        eprintln!("L11 IBL lit lin {lit:?} display {display:?}");
        // threers parity PNG center after Y-flip fix (800×600, pixel 400,300).
        const PARITY_CENTER: [f32; 3] = [251.0 / 255.0, 86.0 / 255.0, 247.0 / 255.0];
        assert_close3("front-sphere IBL display", display, PARITY_CENTER, EPS);
    }

    fn ibl_metal_lit(
        atlas: &crate::textures::CubeUvAtlas,
        n: [f32; 3],
        v: [f32; 3],
        roughness: f32,
        metalness: f32,
        albedo: [f32; 3],
    ) -> ([f32; 3], [f32; 3], [f32; 3]) {
        let roughness_env = roughness.max(0.0525);
        let mut reflect = {
            let d = [-v[0], -v[1], -v[2]];
            let dot = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
            [
                d[0] - 2.0 * dot * n[0],
                d[1] - 2.0 * dot * n[1],
                d[2] - 2.0 * dot * n[2],
            ]
        };
        let r2 = roughness_env * roughness_env;
        reflect = [
            reflect[0] * (1.0 - r2) + n[0] * r2,
            reflect[1] * (1.0 - r2) + n[1] * r2,
            reflect[2] * (1.0 - r2) + n[2] * r2,
        ];
        let radiance = sample_cube_uv_env(atlas, reflect, roughness_env);
        let irr_sample = sample_cube_uv_env(atlas, n, 1.0);
        let dot_nv = (n[0] * v[0] + n[1] * v[1] + n[2] * v[2]).clamp(0.0, 1.0);
        let c0 = [-1.0, -0.0275, -0.572, 0.022];
        let c1 = [1.0, 0.0425, 1.04, -0.04];
        let r = [
            roughness_env * c0[0] + c1[0],
            roughness_env * c0[1] + c1[1],
            roughness_env * c0[2] + c1[2],
            roughness_env * c0[3] + c1[3],
        ];
        let a004 = (r[0] * r[0]).min(2f32.powf(-9.28 * dot_nv)) * r[0] + r[1];
        let fab = [-1.04 * a004 + r[2], 1.04 * a004 + r[3]];
        let specular_color = [
            0.04 * (1.0 - metalness) + albedo[0] * metalness,
            0.04 * (1.0 - metalness) + albedo[1] * metalness,
            0.04 * (1.0 - metalness) + albedo[2] * metalness,
        ];
        let fss_ess = [
            specular_color[0] * fab[0] + fab[1],
            specular_color[1] * fab[0] + fab[1],
            specular_color[2] * fab[0] + fab[1],
        ];
        let ess = fab[0] + fab[1];
        let ems = 1.0 - ess;
        let favg = [
            specular_color[0] + (1.0 - specular_color[0]) * 0.047619,
            specular_color[1] + (1.0 - specular_color[1]) * 0.047619,
            specular_color[2] + (1.0 - specular_color[2]) * 0.047619,
        ];
        let fms = [
            fss_ess[0] * favg[0] / (1.0 - ems * favg[0]),
            fss_ess[1] * favg[1] / (1.0 - ems * favg[1]),
            fss_ess[2] * favg[2] / (1.0 - ems * favg[2]),
        ];
        let lit = [
            radiance[0] * fss_ess[0] + fms[0] * ems * irr_sample[0],
            radiance[1] * fss_ess[1] + fms[1] * ems * irr_sample[1],
            radiance[2] * fss_ess[2] + fms[2] * ems * irr_sample[2],
        ];
        (radiance, irr_sample, lit)
    }

    /// L12 — grazing-angle atlas env probes (parity sphere pixel 366,178).
    #[test]
    fn pmrem_layer_12_grazing_env_probes() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let roughness = 0.45f32;
        let reflect = [-0.190_915_8_f32, 0.692_425_97, 0.695_771_16];
        let normal = [-0.095_603_17, 0.346_739_86, 0.933_076_4];
        let reflect_env = sample_cube_uv_env(atlas, reflect, roughness.max(0.0525));
        let normal_env = sample_cube_uv_env(atlas, normal, 1.0);
        eprintln!(
            "L12 reflect env lin {reflect_env:?} three.js={:?}",
            threejs_ref::REFLECT_GRAZING_ENV_BYTE
        );
        eprintln!(
            "L12 normal env lin {normal_env:?} three.js={:?}",
            threejs_ref::NORMAL_GRAZING_ENV_BYTE
        );
        for mip in [-2i32, -1, 0, 1, 2, 3, 4, 5] {
            let c = sample_atlas_bilinear(atlas, reflect, mip as f32);
            let (u, v) = atlas_bilinear_uv(
                atlas.width,
                atlas.height,
                atlas.lod_max,
                reflect,
                mip as f32,
            );
            let tex = read_atlas_texel(&atlas.pixels, atlas.width, atlas.height, u, v);
            eprintln!("L12 reflect mip{mip} lin {c:?} uv=({u:.4},{v:.4}) tex={tex:?}");
        }
        assert_close3(
            "grazing reflect env @ rough 0.45",
            reflect_env,
            threejs_ref::REFLECT_GRAZING_ENV_BYTE,
            EPS_GRAZING,
        );
        assert_close3(
            "grazing normal env @ rough 1.0",
            normal_env,
            threejs_ref::NORMAL_GRAZING_ENV_BYTE,
            EPS_GRAZING,
        );

        // Full IBL at grazing pixel (camera at 0,0,3 looking at origin).
        let hit = [-0.285_603_05, 1.035_219_8, 1.040_656_7];
        let cam = [0.0f32, 0.0, 3.0];
        let v = {
            let dx = cam[0] - hit[0];
            let dy = cam[1] - hit[1];
            let dz = cam[2] - hit[2];
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            [dx / len, dy / len, dz / len]
        };
        let (_, _, lit) = ibl_metal_lit(atlas, normal, v, roughness, 1.0, [1.0; 3]);
        let display = linear_to_display_rgb(lit);
        eprintln!("L12 grazing IBL lit lin {lit:?} display {display:?}");
    }

    /// Regression for atlas row Y convention (pre-fix: grazing +Z read -Y yellow).
    #[test]
    fn pmrem_layer_13_grazing_plus_z_not_wrong_face() {
        let pm = PmremGenerator::generate_pmrem(&parity_cube(32), 32);
        let atlas = pm.cube_uv_atlas.as_ref().unwrap();
        let reflect = [-0.190_915_8_f32, 0.692_425_97, 0.695_771_16];
        let c = sample_atlas_bilinear(atlas, reflect, 4.0);
        eprintln!("L13 grazing +Z mip4 sample {c:?} (inverted atlas had G≈0.47)");
        assert!(
            c[0] > 0.4 && c[2] > 0.8,
            "grazing +Z should stay magenta-dominant, got {c:?}"
        );
        assert!(
            c[1] < 0.15,
            "grazing +Z green must stay low (yellow bleed when rows inverted), got {}",
            c[1]
        );
    }
}
