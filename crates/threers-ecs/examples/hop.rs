//! A small platform game: collect the coins, don't fall.
//!
//! WASD, the arrows or the left stick to move; space or the south button to jump;
//! escape to quit. Leave out the `gamepad` feature to build without pad support.
//!
//! ```text
//! cargo run -p threers-ecs --features physics,window,gamepad,audio --example hop
//! ```

use threers_ecs::dev::DevTools;
use threers_ecs::prelude::*;

pub const SPEED: f32 = 5.0;
pub const JUMP: f32 = 7.0;
pub const START: Vector3 = Vector3 {
    x: 0.0,
    y: 1.5,
    z: 0.0,
};

pub const COIN_HEIGHT: f32 = 3.0;

/// Centre of each platform; they are all the same size.
pub const PLATFORMS: [[f32; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [0.0, 1.0, -7.0],
    [6.0, 1.5, -7.0],
    [6.0, 2.5, -14.0],
];

#[derive(Default)]
pub struct Player;
impl Component for Player {}
reflect!(Player {});

#[derive(Default)]
pub struct Coin;
impl Component for Coin {}
reflect!(Coin {});

#[derive(Default)]
pub struct Score {
    pub coins: u32,
    pub falls: u32,
}
reflect!(Score { coins, falls });

/// Sound effects, made rather than loaded.
pub struct Sounds {
    jump: threers::audio::Audio,
    coin: threers::audio::Audio,
}

fn play(w: &mut World, pick: fn(&Sounds) -> &threers::audio::Audio) {
    let sound = pick(w.resource::<Sounds>()).clone();
    if let Some(out) = w.get_resource_mut::<AudioOut>() {
        out.play(&sound);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    Playing,
    Won,
}
impl GameState for Phase {}

fn control(w: &mut World) {
    let input = w.resource::<Input>();
    let direction = input.axis2("move_x", "move_y");
    let jump = input.just_pressed("jump");
    let mut jumped = false;
    w.each::<(&mut Character, With<Player>)>(|(character, _)| {
        character.walk = Vector3::new(direction.x, 0.0, -direction.y) * SPEED;
        if jump && character.grounded {
            character.jump(JUMP);
            jumped = true;
        }
    });
    if jumped {
        play(w, |sounds| &sounds.jump);
    }
}

fn collect(w: &mut World) {
    for hit in w.events::<Collision>().iter().filter(|hit| hit.started) {
        let involved = [hit.a, hit.b].into_iter().flatten();
        let coin = involved.clone().find(|&id| w.has::<Coin>(id));
        let player = involved.clone().any(|id| w.has::<Player>(id));
        if let (Some(coin), true) = (coin, player) {
            w.commands().despawn(coin);
            let score = w.resource_mut::<Score>();
            score.coins += 1;
            if score.coins as usize == PLATFORMS.len() {
                w.set_state(Phase::Won);
            }
            play(w, |sounds| &sounds.coin);
        }
    }
}

fn hud(w: &mut World) {
    let score = w.resource::<Score>();
    let line = format!("coins {} / {}", score.coins, PLATFORMS.len());
    let won = w.state::<Phase>() == Phase::Won;
    with_hud(w, |hud| {
        hud.fill(Color::WHITE);
        hud.text_size(28.0);
        hud.text(24.0, 44.0, &line);
        if won {
            hud.text_size(56.0);
            hud.text(24.0, 110.0, "You win!");
        }
    });
}

/// Back to the start. `teleport`, because walking there would cross the level in
/// one step and hit everything on the way.
fn respawn(w: &mut World) {
    let mut fell = 0;
    w.each::<&mut Character>(|character| {
        if character.position().y < -10.0 {
            character.teleport(START);
            fell += 1;
        }
    });
    w.resource_mut::<Score>().falls += fell;
}

fn follow(w: &mut World) {
    let mut focus = None;
    w.each::<(Node, With<Player>)>(|(node, _)| focus = Some(node.position));
    if let Some(focus) = focus {
        let camera = &mut w.resource_mut::<MainCamera>().0;
        camera.position = focus + Vector3::new(0.0, 6.0, 10.0);
        camera.look_at(focus);
    }
}

fn quit(w: &mut World) {
    if w.resource::<Input>().just_pressed("quit") {
        w.insert_resource(Exit);
    }
}

/// Everything but the window, so the game can also be played by a test.
pub fn build(app: &mut App) {
    app.plugin(PhysicsPlugin::default())
        .plugin(DevTools)
        .state(Phase::Playing)
        .resource(Score::default())
        .resource(Sounds {
            jump: tone(330.0, 0.12, 0.3),
            coin: tone(990.0, 0.15, 0.3),
        })
        .resource(MainCamera(PerspectiveCamera::new(
            50.0,
            16.0 / 9.0,
            0.1,
            500.0,
        )))
        .fixed_system(
            System::new(control)
                .before(character_step)
                .in_state(Phase::Playing),
        )
        .fixed_system(System::new(collect).after(physics_step))
        .fixed_system(
            System::new(respawn)
                .after(character_step)
                .before(physics_step),
        )
        .system(System::new(follow).after(physics_sync))
        .system(hud)
        .system(quit);

    let input = app.world.resource_mut::<Input>();
    input
        .bind("jump", Key::Space)
        .bind("jump", PadButton::South);
    input.bind("quit", Key::Escape);
    input.bind_axis("move_x", Key::KeyA, Key::KeyD);
    input.bind_axis("move_x", Key::ArrowLeft, Key::ArrowRight);
    input.bind_analog("move_x", PadAxis::LeftX);
    input.bind_axis("move_y", Key::KeyS, Key::KeyW);
    input.bind_axis("move_y", Key::ArrowDown, Key::ArrowUp);
    input.bind_analog("move_y", PadAxis::LeftY);

    app.plugin(
        TouchControls::new()
            .stick("move_x", "move_y")
            .button(TouchButton::new("jump", "A", 0.88, 0.75)),
    );

    let w = &mut app.world;
    w.register::<Player>()
        .register::<Coin>()
        .register_resource::<Score>();
    w.scene.background = Color::from_hex(0x87b5d8);
    let mut sun = DirectionalLight::new(Color::WHITE, 3.0);
    sun.direction = Vector3::new(-1.0, -2.0, -1.5).normalize();
    w.scene.add_light(sun);
    w.scene.add_light(AmbientLight::new(Color::WHITE, 0.6));

    for (index, [x, y, z]) in PLATFORMS.into_iter().enumerate() {
        let platform = w.scene.add_mesh(
            BoxGeometry::new(5.0, 1.0, 5.0),
            StandardMaterial::new(Color::from_hex(0x4f8f5b)).into(),
        );
        w.scene.get_mut(platform).unwrap().position = Vector3::new(x, y, z);
        w.attach_body(
            platform,
            RigidBody::fixed().shape(Shape::cuboid(2.5, 0.5, 2.5)),
        );

        let coin = w.scene.add_mesh(
            BoxGeometry::new(0.5, 0.5, 0.5),
            StandardMaterial::new(Color::from_hex(0xffcc33)).into(),
        );
        // Out of reach from the ground, so each one takes a jump.
        w.scene.get_mut(coin).unwrap().position = Vector3::new(x, y + COIN_HEIGHT, z);
        w.scene.get_mut(coin).unwrap().name = format!("coin{index}");
        w.insert(coin, Coin);
        w.attach_body(
            coin,
            RigidBody::fixed().collider(Collider::new(Shape::ball(0.6)).sensor(true)),
        );
    }

    let player = w.scene.add_mesh(
        CapsuleGeometry::new(0.4, 0.8, 8, 16),
        StandardMaterial::new(Color::from_hex(0xd9483b)).into(),
    );
    let node = w.scene.get_mut(player).unwrap();
    node.position = START;
    node.name = "player".to_owned();
    w.insert(player, Player);
    w.attach_character(
        player,
        CharacterController::new(Shape::capsule(0.4, 0.4))
            .max_slope_degrees(45.0)
            .snap_to_ground(0.2),
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() {
    let mut app = App::new();
    build(&mut app);
    app.plugin(threers_ecs::audio::AudioPlugin::Device);
    run(app, WindowOptions::new("hop"));
}
