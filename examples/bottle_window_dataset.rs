//! Dataset generation moved to `/Users/Shared/exp-bottle`.
//!
//! ```text
//! cd /Users/Shared/exp-bottle
//! cargo run --release
//! ```

fn main() {
    eprintln!(
        "bottle_window_dataset now lives in /Users/Shared/exp-bottle\n\
         cd /Users/Shared/exp-bottle && SAMPLES=256 SPP=16 cargo run --release"
    );
    std::process::exit(1);
}
