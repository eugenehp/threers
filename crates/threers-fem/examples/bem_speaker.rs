//! A closed loudspeaker cabinet with one driver, by the boundary element
//! method: the sound it radiates at a few frequencies, its directivity, and
//! its radiated power — with no mesh of the air at all.
//!
//! The cabinet is a 20 × 30 × 20 cm box; the driver is a 12 cm square piston
//! on the front face moving at 1 mm/s. Air at 20 °C.
//!
//! ```sh
//! cargo run --release -p threers-fem --example bem_speaker
//! ```

use threers::geometries::BoxGeometry;
use threers_fem::bem::helmholtz::spl;
use threers_fem::{Complex, Helmholtz, TriSurface};

fn main() {
    let (rho, c) = (1.204, 343.0);
    let cabinet = BoxGeometry::with_segments(0.2, 0.3, 0.2, 10, 15, 10);
    let surface = TriSurface::from_geometry(&cabinet).expect("closed box");
    println!("cabinet: {} triangles", surface.triangles.len());

    // Front face is +z; the piston is its central 12 cm square.
    let piston = |p: [f64; 3], n: [f64; 3]| n[2] > 0.5 && p[0].abs() < 0.06 && p[1].abs() < 0.06;
    let velocity = 1e-3;

    println!("{:>7}  {:>9}  {:>9}  {:>9}  {:>10}", "f (Hz)", "on-axis", "side", "behind", "power (W)");
    for f in [100.0, 300.0, 1000.0, 2000.0] {
        let t = std::time::Instant::now();
        let h = Helmholtz::acoustic(&surface, f, c).expect("problem");
        let s = h
            .radiate_velocity(rho, c, |p, n| if piston(p, n) { Complex::from(velocity) } else { Complex::ZERO })
            .expect("solve");
        // SPL at 1 m in front, to the side and behind.
        let at = |dir: [f64; 3]| spl(s.pressure_at(dir));
        println!(
            "{f:>7.0}  {:>6.1} dB  {:>6.1} dB  {:>6.1} dB  {:>10.3e}   ({:.1} el/λ, {:.2} s)",
            at([0.0, 0.0, 1.0]),
            at([1.0, 0.0, 0.0]),
            at([0.0, 0.0, -1.0]),
            s.sound_power(rho, c),
            h.elements_per_wavelength(),
            t.elapsed().as_secs_f64()
        );
    }

    // Directivity at 2 kHz in the horizontal plane, from the far field.
    let h = Helmholtz::acoustic(&surface, 2000.0, c).expect("problem");
    let s = h
        .radiate_velocity(rho, c, |p, n| if piston(p, n) { Complex::from(velocity) } else { Complex::ZERO })
        .expect("solve");
    let on_axis = s.far_field([0.0, 0.0, 1.0]).abs();
    println!("\ndirectivity at 2 kHz (dB re on-axis):");
    for deg in (0..=180).step_by(30) {
        let a = (deg as f64).to_radians();
        let level = 20.0 * (s.far_field([a.sin(), 0.0, a.cos()]).abs() / on_axis).log10();
        let bar = "#".repeat(((level + 30.0).max(0.0)) as usize);
        println!("  {deg:>3}°  {level:>6.1}  {bar}");
    }
}
