//! Writes `data/en.words` from a word list, one word a line (as
//! `/usr/share/dict/web2`, Webster's Second International, public domain):
//! lowercase ASCII words of letters, sorted, deduplicated, front-coded (a
//! byte of the length shared with the word before, then the rest, then a
//! newline).
//!
//! `cargo run -p threers-spell --example build_words -- /usr/share/dict/web2 crates/threers-spell/data/en.words`

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (from, to) = (&args[1], &args[2]);
    let text = std::fs::read_to_string(from).expect("the word list");
    let mut words: Vec<String> = text
        .lines()
        .map(|w| w.trim().to_ascii_lowercase())
        .filter(|w| !w.is_empty() && w.bytes().all(|b| b.is_ascii_lowercase()))
        .collect();
    words.sort();
    words.dedup();
    let out = threers_spell::front_code(&words);
    std::fs::write(to, &out).expect("write");
    eprintln!("{} words, {} bytes", words.len(), out.len());
}
