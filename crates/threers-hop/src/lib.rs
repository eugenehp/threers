//! `hop` in a browser: the same game as the native example, on a page's canvas.
//!
//! ```sh
//! python3 scripts/hop-browser-test.py
//! ```
#![cfg(target_arch = "wasm32")]

#[allow(dead_code)]
#[path = "../../threers-ecs/examples/hop.rs"]
mod hop;

#[allow(dead_code)]
#[path = "../../threers-ecs/examples/race/main.rs"]
mod race;

/// What the race page reads, for a beacon or a test.
fn publish_race(w: &mut World) {
    let time = *w.resource::<Time>();
    let speed = w
        .single::<(&threers_ecs::car::Car, With<race::Player>)>()
        .map_or(0.0, |(car, _)| car.speed);
    let backend = w.resource::<threers_ecs::web::GraphicsBackend>().0.clone();
    let phase = format!("{:?}", w.state::<race::Phase>());
    let state = format!(
        r#"{{"frame":{},"steps":{},"speed":{speed},"phase":"{phase}","backend":"{backend}"}}"#,
        time.frame, time.fixed_steps
    );
    STATE.with(|s| *s.borrow_mut() = state);
}

/// Start the race on `canvas`.
#[wasm_bindgen]
pub async fn start_race(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    let mut app = App::new();
    race::build(&mut app);
    app.plugin(AudioPlugin::Device)
        .plugin(AssetsPlugin::new(""));
    use_font(&mut app.world, "sans.ttf");
    app.system(publish_race);
    threers_ecs::web::run(app, canvas).await
}

use std::cell::RefCell;
use threers_ecs::assets::{use_font, Assets, AssetsPlugin, Handle};
use threers_ecs::audio::{AudioPlugin, SoundAsset};
use threers_ecs::prelude::*;
use threers_ecs::saves::SaveSlots;
use threers_ecs::tasks::{Jobs, Task};
use threers_ecs::web::WorkerPool;
use wasm_bindgen::prelude::*;

thread_local! {
    static STATE: RefCell<String> = const { RefCell::new(String::new()) };
}

fn publish(w: &mut World) {
    let time = *w.resource::<Time>();
    let mut player = (0.0, 0.0, 0.0);
    w.each_ref::<(NodeRef, With<hop::Player>)>(|(node, _)| {
        player = (node.position.x, node.position.y, node.position.z);
    });
    let score = w.resource::<hop::Score>();
    let backend = &w.resource::<threers_ecs::web::GraphicsBackend>().0;
    let (frames, peak) = w.resource::<threers_ecs::web::WebAudioMeter>().0.get();
    let sound = w
        .resource::<Assets>()
        .get(&w.resource::<Fetched>().0)
        .and_then(|asset| asset.0.buffer.as_ref().map(|samples| samples.len()))
        .unwrap_or(0);
    let state = format!(
        r#"{{"frame":{},"steps":{},"coins":{},"x":{},"y":{},"z":{},"backend":"{}","audio_frames":{},"audio_peak":{},"fetched_samples":{}}}"#,
        time.frame,
        time.fixed_steps,
        score.coins,
        player.0,
        player.1,
        player.2,
        backend,
        frames,
        peak,
        sound
    );
    STATE.with(|s| *s.borrow_mut() = state);
}

/// A sound file brought in over `fetch`, to show that path works.
struct Fetched(Handle<SoundAsset>);

/// Count primes below a number. Slow on purpose, so a test can see frames go on.
fn primes(input: &[u8]) -> Vec<u8> {
    let limit = u32::from_le_bytes([input[0], input[1], input[2], input[3]]);
    let count = (2..limit)
        .filter(|n| (2..).take_while(|d| d * d <= *n).all(|d| n % d != 0))
        .count() as u32;
    count.to_le_bytes().to_vec()
}

fn jobs() -> Jobs {
    Jobs::new().register("primes", primes)
}

/// What a worker calls.
#[wasm_bindgen]
pub fn run_job(name: &str, input: &[u8]) -> Result<Vec<u8>, JsValue> {
    jobs()
        .run_here(name, input)
        .map_err(|error| error.to_string().into())
}

/// Spread a sum over rayon's pool and report how many threads took part, from a
/// task, so the main thread never has to wait for it.
#[cfg(feature = "threads")]
fn start_parallel(w: &mut World) {
    use rayon::prelude::*;
    use std::collections::HashSet;
    use std::sync::Mutex;

    let task = w.resource::<threers_ecs::tasks::Tasks>().spawn(|| {
        let threads = Mutex::new(HashSet::new());
        let total: u64 = (0..400_000_u64)
            .into_par_iter()
            .map(|n| {
                if n % 1000 == 0 {
                    threads
                        .lock()
                        .unwrap()
                        .insert(rayon::current_thread_index());
                }
                (0..200).fold(n, |a, b| a.wrapping_mul(31).wrapping_add(b))
            })
            .reduce(|| 0, u64::wrapping_add);
        (threads.into_inner().unwrap().len(), total)
    });
    w.insert_resource(Parallel(task));
}

#[cfg(feature = "threads")]
struct Parallel(Task<(usize, u64)>);

/// Hand rayon's threads to the page.
#[cfg(feature = "threads")]
pub use threers::wasm::init_thread_pool;

/// A job in flight, and the frame it started on.
struct Running(Option<(Task<Vec<u8>>, u64)>);

/// Start and collect a worker job when asked.
fn work(w: &mut World) {
    let frame = w.resource::<Time>().frame;
    #[cfg(feature = "threads")]
    {
        if REQUEST.with(|r| r.borrow().as_deref() == Some("parallel")) {
            REQUEST.with(|r| *r.borrow_mut() = None);
            start_parallel(w);
        }
        let result = w
            .get_resource_mut::<Parallel>()
            .and_then(|running| running.0.poll());
        if let Some(result) = result {
            let text = match result {
                Ok((threads, _)) => format!("{threads} threads"),
                Err(error) => format!("error: {error}"),
            };
            ANSWER.with(|a| *a.borrow_mut() = text);
        }
    }
    let start = REQUEST.with(|r| r.borrow().as_deref() == Some("job"));
    if start {
        REQUEST.with(|r| *r.borrow_mut() = None);
        let task = w
            .resource::<WorkerPool>()
            .run("primes", &5_000_000_u32.to_le_bytes());
        w.resource_mut::<Running>().0 = Some((task, frame));
    }
    let done = w
        .resource_mut::<Running>()
        .0
        .as_mut()
        .and_then(|(task, began)| task.poll().map(|result| (result, *began)));
    if let Some((result, began)) = done {
        w.resource_mut::<Running>().0 = None;
        let text = match result {
            Ok(bytes) => format!(
                "{} primes, {} frames",
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                frame - began
            ),
            Err(error) => format!("error: {error}"),
        };
        ANSWER.with(|a| *a.borrow_mut() = text);
    }
}

/// Start the game on `canvas`.
#[wasm_bindgen]
pub async fn start(canvas: web_sys::HtmlCanvasElement) -> Result<(), JsValue> {
    let mut app = App::new();
    hop::build(&mut app);
    app.plugin(AudioPlugin::Device)
        .plugin(AssetsPlugin::new(""));
    let fetched = app
        .world
        .resource_mut::<Assets>()
        .load::<SoundAsset>("tone.ogg");
    app.resource(Fetched(fetched));
    use_font(&mut app.world, "sans.ttf");
    let pool = WorkerPool::new("./pkg/threers_hop.js", 2)?;
    app.resource(pool).resource(Running(None));
    #[cfg(feature = "threads")]
    app.resource(threers_ecs::tasks::Tasks::new());
    app.system(publish)
        .system(work)
        .system(System::new(serve).after(work));
    threers_ecs::web::run(app, canvas).await
}

thread_local! {
    static REQUEST: RefCell<Option<String>> = const { RefCell::new(None) };
    static ANSWER: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Runs what the page asked for between frames.
fn serve(w: &mut World) {
    let Some(request) = REQUEST.with(|r| r.borrow_mut().take()) else {
        return;
    };
    let slots = SaveSlots::new("hop");
    let answer = match request.as_str() {
        "save" => slots.write(w, "test").map(|()| "saved".to_owned()),
        "load" => slots.read(w, "test").map(|()| "loaded".to_owned()),
        "list" => Ok(slots.list().join(",")),
        other => Err(format!("unknown request {other}")),
    };
    ANSWER.with(|a| *a.borrow_mut() = answer.unwrap_or_else(|e| format!("error: {e}")));
}

/// Ask the running game for "save", "load" or "list".
#[wasm_bindgen]
pub fn request(what: String) {
    ANSWER.with(|a| a.borrow_mut().clear());
    REQUEST.with(|r| *r.borrow_mut() = Some(what));
}

/// What the last request came to, once it has.
#[wasm_bindgen]
pub fn answer() -> String {
    ANSWER.with(|a| a.borrow().clone())
}

/// The game as JSON, for a test to read.
#[wasm_bindgen]
pub fn state() -> String {
    STATE.with(|s| s.borrow().clone())
}
