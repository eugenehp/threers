//! Integration tests for cuboct metamaterials (Jenett et al. Sci. Adv. 2020).

use threers::{
    ChiralRule, Cuboct, CuboctAssembly, CuboctAssemblyPlan, CuboctFrame, FrameMaterial,
    Lattice, LatticeKind,
};

#[test]
fn continuum_lattice_builds() {
    let geom = Lattice::new(LatticeKind::Cuboct(Cuboct::Auxetic))
        .size(threers::Vector3::new(20.0, 20.0, 20.0))
        .cells([3, 3, 3])
        .shape(0.18)
        .resolution(22)
        .fit_relative_density(0.18)
        .build();
    let tris = geom.index.as_ref().map(|i| i.len() / 3).unwrap_or(0);
    assert!(tris > 1000, "triangles={tris}");
}

#[test]
fn frame_rigid_stiffens_with_cell_count() {
    let e1 = CuboctFrame::new([1, 1, 1], Cuboct::Rigid, 1.0, 0.0)
        .compress_z(0.01)
        .effective_modulus;
    let e2 = CuboctFrame::new([2, 2, 2], Cuboct::Rigid, 1.0, 0.0)
        .compress_z(0.01)
        .effective_modulus;
    assert!(e1 > 0.0 && e2 > e1, "E*1={e1} E*2={e2}");
}

#[test]
fn frame_auxetic_softer_than_rigid() {
    let rigid = CuboctFrame::new([2, 2, 2], Cuboct::Rigid, 1.0, 0.0).compress_z(0.05);
    let aux = CuboctFrame::new([2, 2, 2], Cuboct::Auxetic, 1.0, 0.2).compress_z(0.05);
    assert!(aux.effective_modulus < rigid.effective_modulus);
}

#[test]
fn assembly_export_and_automation() {
    let dir = std::env::temp_dir().join("threers_cuboct_export");
    let _ = std::fs::remove_dir_all(&dir);
    let asm = CuboctAssembly::new(Cuboct::Compliant)
        .pitch(12.0)
        .cells([1, 1, 1])
        .mold_ready(true)
        .resolution(12);
    assert_eq!(asm.export_stl_dir(&dir).unwrap(), 6);
    assert_eq!(asm.export_obj_dir(&dir).unwrap(), 6);
    assert!(dir.join("part_0000.stl").exists());
    let plan = CuboctAssemblyPlan::from_assembly(&asm);
    assert_eq!(
        plan.steps
            .iter()
            .filter(|s| matches!(s, threers::CuboctAssemblyStep::PickPart { .. }))
            .count(),
        6
    );
}

#[test]
fn heterogeneous_column_program() {
    let m = 4i32;
    let asm = CuboctAssembly::new(Cuboct::ChiralCcw)
        .pitch(10.0)
        .cells([1, 1, m as usize])
        .chiral_rule(ChiralRule::R1)
        .program(move |_, _, k| Cuboct::column_half(k, m))
        .resolution(14);
    assert_eq!(asm.parts().len(), m as usize * 6);
    let frame = CuboctFrame::with_program(
        [1, 1, m as usize],
        1.0,
        0.12,
        Some(ChiralRule::R1),
        move |_, _, k| Cuboct::column_half(k, m),
    )
    .material(FrameMaterial {
        e: 70e9,
        a: 1e-6,
        i: 1e-12,
    });
    let r = frame.compress_z(0.04);
    assert!(r.effective_modulus > 0.0);
}
