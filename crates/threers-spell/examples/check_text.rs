//! What autocorrect would change in a text: `cargo run -p threers-spell --release --example check_text -- FILE`
fn main() {
    let text = std::fs::read_to_string(std::env::args().nth(1).expect("a file")).unwrap();
    let s = threers_spell::Speller::new(true);
    let mut n = 0;
    let mut seen = std::collections::BTreeMap::new();
    for w in threers_spell::words_in(&text) {
        n += 1;
        if let Some(f) = s.autocorrect(w) {
            *seen.entry(format!("{w} -> {f}")).or_insert(0) += 1;
        }
    }
    eprintln!("{n} words, {} would change:", seen.len());
    for (k, c) in seen {
        eprintln!("  {k} ({c})");
    }
}
