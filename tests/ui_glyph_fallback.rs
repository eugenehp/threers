//! The UI face falls back to the system's symbol fonts (and, asked, CJK)
//! for characters it lacks, so they draw instead of the missing-glyph box.
#![cfg(all(target_os = "macos", feature = "captions"))]

use threers::captions::CaptionFont;

#[test]
fn symbols_and_cjk_fall_back() {
    assert!(CaptionFont::ui().fallback_count() > 0, "the system's symbol fonts are there");
    for c in ['✕', '⋯', '⟳', '⌨', '⏎', '⇧', '⌘'] {
        let mut f = CaptionFont::ui();
        assert!(!f.glyph(c, 16.0).is_blank(), "{c} draws");
    }
    let t = std::time::Instant::now();
    let n = CaptionFont::use_system_cjk_fallback();
    eprintln!("CJK fallback: {n} faces in {:?}", t.elapsed());
    assert!(n > 0);
    let mut f = CaptionFont::ui();
    for c in ['日', '本', '中', '한'] {
        assert!(!f.glyph(c, 16.0).is_blank(), "{c} draws");
    }
    // a monospace face keeps them too
    let mut mono = CaptionFont::from_ttf_bytes(&std::fs::read("/System/Library/Fonts/Monaco.ttf").unwrap())
        .unwrap()
        .with_ui_fallbacks();
    assert!(!mono.glyph('⋯', 16.0).is_blank());
    assert!(!mono.glyph('日', 16.0).is_blank());
}
