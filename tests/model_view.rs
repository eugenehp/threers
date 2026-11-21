//! [`ModelView`] puts the scene it draws into its image box: the frame just
//! drawn by default, the frame before when `pipelined`.
//!
//! Skipped when no GPU adapter is available.

use threers::math::Vector3;
use threers::style;
use threers::ui::model3d::Orbit;
use threers::ui::model_view::ModelView;
use threers::ui::{FixedWidthText, NodeKind, Ui};
use threers::{Color, Scene};

/// The box's picture: its size and first pixel.
fn shown(ui: &Ui, view: &ModelView) -> ((u32, u32), [u8; 3]) {
    let NodeKind::Image { image } = ui.kind(view.node) else {
        panic!("an image node");
    };
    ((image.width, image.height), [image.rgba[0], image.rgba[1], image.rgba[2]])
}

fn draw(ui: &mut Ui, view: &mut ModelView, hex: u32) -> Result<(), String> {
    let mut scene = Scene::new();
    scene.background = Color::from_hex(hex);
    view.render(ui, &mut scene, &Orbit::default(), Vector3::new(0.0, 0.0, 0.0), 1.0)
}

fn setup() -> (Ui, ModelView) {
    let mut ui = Ui::new();
    let root = ui.root();
    let view = ModelView::new(&mut ui, root, style! { width: 40px; height: 20px; });
    ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
    (ui, view)
}

#[test]
fn a_model_view_shows_the_frame_it_drew() {
    let (mut ui, mut view) = setup();
    if let Err(e) = draw(&mut ui, &mut view, 0xff0000) {
        return eprintln!("skipping: {e}");
    }
    assert_eq!(shown(&ui, &view), ((40, 20), [255, 0, 0]));
    draw(&mut ui, &mut view, 0x00ff00).unwrap();
    assert_eq!(shown(&ui, &view), ((40, 20), [0, 255, 0]));
}

#[test]
fn a_pipelined_model_view_shows_the_frame_before() {
    let (mut ui, mut view) = setup();
    view.pipelined = true;
    if let Err(e) = draw(&mut ui, &mut view, 0xff0000) {
        return eprintln!("skipping: {e}");
    }
    // Nothing back yet: the blank placeholder stays.
    assert_eq!(shown(&ui, &view).0, (1, 1));
    draw(&mut ui, &mut view, 0x00ff00).unwrap();
    assert_eq!(shown(&ui, &view), ((40, 20), [255, 0, 0]));
    draw(&mut ui, &mut view, 0x0000ff).unwrap();
    assert_eq!(shown(&ui, &view), ((40, 20), [0, 255, 0]));

    // A resize: the old picture stays one frame, then the new size shows.
    ui.set_style(view.node, style! { width: 30px; height: 30px; });
    ui.layout(100.0, 100.0, &mut FixedWidthText(7.0));
    draw(&mut ui, &mut view, 0x808080).unwrap();
    assert_eq!(shown(&ui, &view), ((40, 20), [0, 255, 0]));
    draw(&mut ui, &mut view, 0x000000).unwrap();
    assert_eq!(shown(&ui, &view), ((30, 30), [128, 128, 128]));
}
