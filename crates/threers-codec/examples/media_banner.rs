//! Print what `codec::ffcompat` predicts ffmpeg's banner says for a video:
//! `fps`, `tbr` (r_frame_rate), `Duration:` and the start time — the values
//! moviepy parses.
//!
//!   media_banner FILE
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&a[1]).expect("read");
    match threers_codec::ffcompat::reader::VideoReader::open(data) {
        Ok(r) => {
            let m = &r.mov;
            println!(
                "fps {} tbr {}/{} duration {:.2} start_us {} frames {}",
                m.banner_fps, m.r_frame_rate.0, m.r_frame_rate.1, m.banner_duration, m.start_us, m.pts.len()
            );
        }
        Err(e) => println!("ERROR {e}"),
    }
}
