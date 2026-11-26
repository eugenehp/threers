//! A face's glyphs' unhinted outlines as `FT_Outline_Decompose` walks
//! them, to compare with FreeType's: `outline font index size gid…`
//! (`size` in 26.6 points at 72 dpi).
use threers_font::glyph::Segment;
use threers_font::{Face, Glyph, Size};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let face = Face::new(std::fs::read(&args[1]).expect("font"), args[2].parse().unwrap()).expect("face");
    let size = Size::char_size(&face, 0, args[3].parse().unwrap(), 72, 72);
    for gid in args[4..].iter().filter_map(|a| a.parse::<u16>().ok()) {
        println!("G {gid}");
        let Ok(g) = Glyph::load(&face, &size, gid) else {
            println!("E");
            continue;
        };
        for s in g.outline.decompose().unwrap_or_default() {
            match s {
                Segment::MoveTo(p) => println!("M {} {}", p.0, p.1),
                Segment::LineTo(p) => println!("L {} {}", p.0, p.1),
                Segment::ConicTo(c, p) => println!("Q {} {} {} {}", c.0, c.1, p.0, p.1),
                Segment::CubicTo(a, b, p) => println!("C {} {} {} {} {} {}", a.0, a.1, b.0, b.1, p.0, p.1),
            }
        }
        println!("A {}", g.advance);
    }
}
