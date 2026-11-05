//! The finished game from docs/ecs-tutorial.md: roll a ball around an arena,
//! collect the orbs before the clock runs out.
//!
//! cargo run -p threers-ecs --release --features physics,window,audio --example tutorial

use threers_ecs::prelude::*;

// ---- step 2: what things are --------------------------------------------------

struct Player;
impl Component for Player {}

struct Orb;
impl Component for Orb {}

/// What the round is up to.
pub struct Round {
    pub collected: u32,
    pub clock: Timer,
}

// ---- step 6: menus and states ------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Phase {
    Playing,
    Won,
    Lost,
}
impl GameState for Phase {}

/// Numbers to tune while it runs: edit tuning/ball.json and save.
#[derive(Default)]
pub struct Tuning {
    pub push: f32,
}
reflect!(Tuning { push });

pub const ORBS: u32 = 8;
pub const SECONDS: f32 = 60.0;

// ---- step 3: input and systems -----------------------------------------------

/// Push the ball the way the keys say, relative to the camera.
fn roll(w: &mut World) {
    let input = w.resource::<Input>();
    let push = input.axis2("x", "z");
    let strength = w.resource::<Tuning>().push.max(15.0);
    let dt = w.resource::<Time>().delta;
    let Some((id, _)) = w.single::<(Entity, With<Player>)>() else {
        return;
    };
    if let Some(ball) = w.body_mut(id) {
        ball.apply_impulse(Vector3::new(push.x, 0.0, -push.y) * strength * dt);
    }
}

// ---- step 4: physics and events ----------------------------------------------

/// An orb the player touches is an orb collected. Only the player's touch: an
/// orb sits on the floor, and that is a collision too.
fn collect(w: &mut World) {
    let mut touches = Vec::new();
    w.read_events::<Collision>(|hit| {
        if let (true, Some(a), Some(b)) = (hit.started, hit.a, hit.b) {
            touches.push((a, b));
        }
    });
    for (a, b) in touches {
        let id = if w.has::<Player>(a) {
            b
        } else if w.has::<Player>(b) {
            a
        } else {
            continue;
        };
        if w.has::<Orb>(id) {
            w.despawn(id);
            w.resource_mut::<Round>().collected += 1;
            if let Some(out) = w.get_resource_mut::<AudioOut>() {
                out.play(&tone(990.0, 0.12, 0.3));
            }
        }
    }
    if w.resource::<Round>().collected == ORBS {
        w.set_state(Phase::Won);
    }
}

fn countdown(w: &mut World) {
    let dt = w.resource::<Time>().delta;
    if w.resource_mut::<Round>().clock.tick(dt) {
        w.set_state(Phase::Lost);
    }
}

/// Orbs bob, so they read as things to take.
fn bob(w: &mut World) {
    let t = w.resource::<Time>().elapsed as f32;
    w.each::<(Node, With<Orb>)>(|(node, _)| {
        node.position.y = 1.0 + (t * 2.0 + node.position.x).sin() * 0.2;
    });
}

// ---- step 5: what the player sees --------------------------------------------

fn hud(w: &mut World) {
    let round = w.resource::<Round>();
    let line = format!(
        "{} / {ORBS}   {:.0} s",
        round.collected,
        round.clock.remaining()
    );
    let phase = w.state::<Phase>();
    with_hud(w, |hud| {
        hud.fill(Color::WHITE);
        hud.text_size(30.0);
        hud.text(24.0, 44.0, &line);
        let big = match phase {
            Phase::Won => "You got them all",
            Phase::Lost => "Out of time",
            Phase::Playing => return,
        };
        hud.text_size(54.0);
        hud.text(24.0, 120.0, big);
    });
}

// ---- step 1: putting it together ---------------------------------------------

/// Everything but the window, so a test can play it.
pub fn build(app: &mut App) {
    app.plugin(PhysicsPlugin::default())
        .plugin(
            ChaseCamera::new()
                .distance(12.0)
                .height(9.0)
                .look_at(0.0, 0.0),
        )
        .state(Phase::Playing)
        .tune::<Tuning>("tuning/ball.json")
        .resource(Round {
            collected: 0,
            clock: Timer::once(SECONDS),
        })
        .fixed_system(
            System::new(roll)
                .in_state(Phase::Playing)
                .before(physics_step),
        )
        .fixed_system(System::new(collect).after(physics_step))
        .fixed_system(System::new(countdown).in_state(Phase::Playing))
        .system(bob)
        .system(hud);

    let input = app.world.resource_mut::<Input>();
    input
        .bind_axis("x", Key::KeyA, Key::KeyD)
        .bind_axis("x", Key::ArrowLeft, Key::ArrowRight);
    input
        .bind_axis("z", Key::KeyS, Key::KeyW)
        .bind_axis("z", Key::ArrowDown, Key::ArrowUp);
    input
        .bind_analog("x", PadAxis::LeftX)
        .bind_analog("z", PadAxis::LeftY);

    let w = &mut app.world;
    w.scene.background = Color::from_hex(0x1d2733);
    w.sun(Vector3::new(-1.0, -2.0, -1.0), 3.0, 60.0);
    w.scene.add_light(AmbientLight::new(Color::WHITE, 0.5));

    // A floor with a raised rim, built in code and solid.
    let mut arena = MeshBuilder::new();
    arena.grid(40.0, 40.0, 40, 40, |x, z| {
        let edge = x.abs().max(z.abs());
        if edge > 17.0 {
            (edge - 17.0) * 1.5
        } else {
            0.0
        }
    });
    w.mesh(arena, StandardMaterial::new(Color::from_hex(0x3c6e71)))
        .shadows(false, true)
        .solid();

    w.mesh(
        SphereGeometry::new(0.6, 24, 16),
        StandardMaterial::new(Color::from_hex(0xffcc33)),
    )
    .at(Vector3::new(0.0, 1.0, 0.0))
    .shadows(true, false)
    .body(
        RigidBody::dynamic()
            .shape(Shape::ball(0.6))
            .mass(1.0)
            .linear_damping(0.3),
    )
    .with((Player, CameraTarget));

    for i in 0..ORBS {
        let angle = i as f32 / ORBS as f32 * std::f32::consts::TAU;
        let at = Vector3::new(angle.cos() * 11.0, 1.0, angle.sin() * 11.0);
        w.mesh(
            IcosahedronGeometry::new(0.45, 0),
            StandardMaterial::new(Color::from_hex(0x7fdbff))
                .with_emissive(Color::from_hex(0x2a7fbf), 1.0),
        )
        .at(at)
        .with(Orb)
        .body(RigidBody::kinematic().collider(Collider::new(Shape::ball(0.6)).sensor(true)));
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[allow(dead_code)]
fn main() {
    let mut app = App::new();
    build(&mut app);
    app.plugin(AudioPlugin::Device)
        .plugin(threers_ecs::dev::DevTools);
    run(app, WindowOptions::new("orbs"));
}
