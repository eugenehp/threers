//! The English weights in `weights/` load and read swipes.

use threers_swipe::synth::{self, Rng, SynthConfig};
use threers_swipe::{Decoder, KeyboardGeometry, Lexicon, Model, ModelConfig};
use threers_ui::virtual_keyboard::{KeyboardConfig, KeyboardLayout};

#[test]
fn the_english_weights_read_made_up_swipes() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/weights/swipe-en-small.safetensors"
    );
    let model = Model::load(path).unwrap();
    assert_eq!(model.config, ModelConfig::small());
    let words = [
        "the",
        "people",
        "between",
        "keyboard",
        "swipe",
        "typing",
        "quick",
        "brown",
        "world",
        "hello",
        "answer",
        "picture",
        "language",
        "history",
        "government",
        "university",
    ];
    let lexicon = Lexicon::from_counts(words.map(|w| (w, 1.0)));
    let keyboard =
        KeyboardGeometry::from_config(&KeyboardConfig::iphone(), &KeyboardLayout::english(), 390.0);
    let decoder = Decoder::new(&keyboard, &lexicon);
    let mut rng = Rng::new(3);
    let mut right = 0;
    for w in words {
        let s = synth::swipe(&keyboard, w, &SynthConfig::default(), &mut rng).unwrap();
        right += (decoder
            .decode(&model, &s, 1)
            .first()
            .map(|c| c.word.as_str())
            == Some(w)) as usize;
    }
    assert!(right >= 14, "{right} of {}", words.len());
}
