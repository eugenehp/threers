//! The retained UI layer, drawn on the HUD and clicked through `Input`.

use threers::math::Vector2;
use threers::ui::{Length, NodeId, Style};
use threers_ecs::hud::{Hud, UiClicked, UiLayer, UiPlugin};
use threers_ecs::prelude::*;
use threers_ecs::testing::TestGame;

fn game() -> (TestGame, NodeId) {
    let mut button = None;
    let game = TestGame::new(|app| {
        app.resource(Hud::new(320, 180));
        app.plugin(UiPlugin);
        let layer = app.world.resource_mut::<UiLayer>();
        let root = layer.ui.root();
        button = Some(layer.ui.add_box(
            root,
            Style {
                width: Length::Px(100.0),
                height: Length::Px(50.0),
                background: Some([0.9, 0.2, 0.2, 1.0]),
                ..Style::default()
            },
        ));
    });
    (game, button.unwrap())
}

fn click(game: &mut TestGame, x: f32, y: f32) {
    let input = game.input();
    input.move_mouse(Vector2::new(x, y));
    input.press(MouseButton::Left);
    game.frames(1);
    game.input().release(MouseButton::Left);
    game.frames(1);
}

#[test]
fn a_box_is_drawn_onto_the_hud() {
    let (mut game, _) = game();
    game.frames(1);
    let hud = game.resource::<Hud>();
    let (width, _) = hud.size();
    let pixel = |x: usize, y: usize| {
        let at = (y * width as usize + x) * 4;
        hud.pixels()[at + 3]
    };
    assert!(pixel(50, 25) > 200, "inside");
    assert_eq!(pixel(200, 100), 0, "outside");
}

#[test]
fn a_click_on_a_node_reports_that_node() {
    let (mut game, button) = game();
    game.frames(1);
    click(&mut game, 50.0, 25.0);
    assert_eq!(game.world().events::<UiClicked>(), [UiClicked(button)]);
}

#[test]
fn a_click_outside_every_box_reports_only_what_is_there() {
    let (mut game, button) = game();
    game.frames(1);
    click(&mut game, 300.0, 170.0);
    let clicks = game.world().events::<UiClicked>();
    assert!(!clicks.contains(&UiClicked(button)), "{clicks:?}");
}

#[test]
fn holding_the_button_is_not_a_click_until_it_comes_up() {
    let (mut game, _) = game();
    game.frames(1);
    let input = game.input();
    input.move_mouse(Vector2::new(50.0, 25.0));
    input.press(MouseButton::Left);
    game.frames(3);
    assert!(game.world().events::<UiClicked>().is_empty());
}
