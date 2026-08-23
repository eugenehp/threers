fn main() {
    for p in [
        "/System/Library/Fonts/SFCompact.ttf",
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/Supplemental/DIN Condensed Bold.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Verdana Bold.ttf",
    ] {
        let Ok(b) = std::fs::read(p) else {
            println!("{p:<58} missing");
            continue;
        };
        match threers::captions::CaptionFont::from_ttf_bytes(&b) {
            Ok(f) => {
                let m = f.metrics(48.0);
                println!(
                    "{p:<58} OK  ascent {:.1} line {:.1}",
                    m.ascent,
                    m.line_height()
                );
            }
            Err(e) => println!("{p:<58} FAIL {e:?}"),
        }
    }
}
