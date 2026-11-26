//! `cargo run -p threers-metafile --example convert -- in.emf out.pdf|out.svg`

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (Some(input), Some(output)) = (a.get(1), a.get(2)) else {
        eprintln!("usage: convert in.emf|in.wmf out.pdf|out.svg");
        std::process::exit(2);
    };
    let d = threers_metafile::read(&std::fs::read(input).expect("read")).unwrap_or_else(|e| {
        eprintln!("{input}: {e}");
        std::process::exit(1)
    });
    println!("{input}: {:.1} × {:.1} pt, {} things drawn", d.width, d.height, d.ops.len());
    if output.ends_with(".svg") {
        std::fs::write(output, d.to_svg()).expect("write");
    } else {
        std::fs::write(output, d.to_pdf()).expect("write");
    }
}
