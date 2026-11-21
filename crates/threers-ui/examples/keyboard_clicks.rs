//! The on-screen keyboard's three clicks — a letter's, Backspace's, and the
//! other keys' — played on the default audio device, then a burst as fast
//! typing makes them.
//!
//! ```text
//! cargo run -p threers-ui --features sound --example keyboard_clicks
//! ```

use std::thread::sleep;
use std::time::Duration;

use threers_ui::virtual_keyboard::{ClickPlayer, KeySound};

fn main() -> Result<(), String> {
    let player = ClickPlayer::open()?;
    for sound in [KeySound::Letter, KeySound::Delete, KeySound::Modifier] {
        println!("{sound:?}");
        for _ in 0..3 {
            player.play(sound, 0.8);
            sleep(Duration::from_millis(350));
        }
        sleep(Duration::from_millis(400));
    }
    println!("typing");
    for i in 0..24 {
        let s = if i % 7 == 6 {
            KeySound::Modifier
        } else {
            KeySound::Letter
        };
        player.play(s, 0.6);
        sleep(Duration::from_millis(70 + (i * 37 % 60) as u64));
    }
    sleep(Duration::from_millis(300));
    Ok(())
}
