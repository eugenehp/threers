//! The node tables the browser fetches.
//!
//! One file per attribute rather than one interleaved struct: the viewer uploads
//! positions and colours as separate GPU attributes and filters on the rest, so
//! splitting them means each `fetch` lands directly in the typed array that
//! wants it, with no unpacking pass. All little-endian, which is what
//! `Float32Array` and friends read on every platform a browser ships on.

use std::fs;
use std::io::{self, Write};
use std::path::Path;

use crate::tree::Tree;

/// Vocabulary for one categorical field, ordered by frequency so index 0 is the
/// commonest value — the palette's first and most legible slot lands where it
/// does the most good.
struct Vocab {
    names: Vec<String>,
    counts: Vec<usize>,
}

fn vocab<'a>(values: impl Iterator<Item = &'a str>) -> Vocab {
    let mut at: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut counts: Vec<(String, usize)> = Vec::new();
    for v in values {
        match at.get(v) {
            Some(&i) => counts[i].1 += 1,
            None => {
                at.insert(v.to_string(), counts.len());
                counts.push((v.to_string(), 1));
            }
        }
    }
    counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Vocab {
        names: counts.iter().map(|c| c.0.clone()).collect(),
        counts: counts.iter().map(|c| c.1).collect(),
    }
}

fn index_of(v: &Vocab, name: &str) -> usize {
    v.names.iter().position(|n| n == name).unwrap_or(0)
}

/// What `meta.json` carries: the vocabularies the legend is built from, the
/// bounding boxes the viewer centres on, and an honest account of what is *not*
/// in the tables.
pub struct Meta {
    pub root: String,
    pub json: String,
}

pub fn write_tables(out: &Path, root: &Path, trees: &[Tree]) -> io::Result<Meta> {
    let mut per_tree = Vec::new();

    for t in trees {
        let dir = out.join(&t.key);
        fs::create_dir_all(&dir)?;
        let n = t.nodes.len();

        let roles = vocab(t.nodes.iter().map(|x| x.role.as_str()));
        let nts = vocab(t.nodes.iter().map(|x| x.nt.as_str()));
        let classes = vocab(t.nodes.iter().map(|x| x.class.as_str()));

        let mut pos = Vec::with_capacity(n * 12);
        let mut role = Vec::with_capacity(n);
        let mut nt = Vec::with_capacity(n);
        let mut cls = Vec::with_capacity(n * 2);
        let mut syn = Vec::with_capacity(n * 4);
        let mut ids = Vec::with_capacity(n * 8);
        let mut labels = String::with_capacity(n * 24);

        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];

        for (i, node) in t.nodes.iter().enumerate() {
            for d in 0..3 {
                pos.extend_from_slice(&node.pos[d].to_le_bytes());
                lo[d] = lo[d].min(node.pos[d]);
                hi[d] = hi[d].max(node.pos[d]);
            }
            role.push(index_of(&roles, &node.role) as u8);
            nt.push(index_of(&nts, &node.nt) as u8);
            cls.extend_from_slice(&(index_of(&classes, &node.class) as u16).to_le_bytes());
            syn.extend_from_slice(&node.synapses.to_le_bytes());
            ids.extend_from_slice(&node.id.to_le_bytes());
            if i > 0 {
                labels.push('\n');
            }
            labels.push_str(&node.label);
        }

        write(&dir.join("pos.f32"), &pos)?;
        write(&dir.join("role.u8"), &role)?;
        write(&dir.join("nt.u8"), &nt)?;
        write(&dir.join("cls.u16"), &cls)?;
        write(&dir.join("syn.u32"), &syn)?;
        write(&dir.join("ids.u64"), &ids)?;
        write(&dir.join("labels.txt"), labels.as_bytes())?;

        per_tree.push(tree_json(t, n, &lo, &hi, &roles, &nts, &classes));
    }

    let total: usize = trees.iter().map(|t| t.nodes.len()).sum();
    let json = format!(
        "{{\n \"root\": {},\n \"total\": {},\n \"trees\": [\n{}\n ]\n}}\n",
        jstr(&root.display().to_string()),
        total,
        per_tree.join(",\n")
    );
    fs::create_dir_all(out)?;
    write(&out.join("meta.json"), json.as_bytes())?;

    Ok(Meta { root: root.display().to_string(), json })
}

fn tree_json(
    t: &Tree,
    n: usize,
    lo: &[f32; 3],
    hi: &[f32; 3],
    roles: &Vocab,
    nts: &Vocab,
    classes: &Vocab,
) -> String {
    // Only the first thousand unplaceable ids: enough to check any of them by
    // hand, and the count beside it is the number that matters.
    let ids: Vec<String> = t.unplaceable.iter().take(1000).map(|i| i.to_string()).collect();
    let mut f: Vec<String> = Vec::new();
    f.push(format!("\"key\": {}", jstr(&t.key)));
    f.push(format!("\"title\": {}", jstr(&t.title)));
    f.push(format!("\"subtitle\": {}", jstr(&t.subtitle)));
    f.push(format!("\"count\": {}", n));
    f.push(format!("\"total_in_index\": {}", t.indexed));
    f.push(format!("\"from_index\": {}", t.from_index));
    f.push(format!("\"from_skeleton\": {}", t.from_skeleton));
    f.push(format!("\"unplaceable\": {}", t.unplaceable.len()));
    f.push(format!("\"unplaceable_ids\": [{}]", ids.join(",")));
    f.push(format!("\"pos_source\": {}", jstr(&t.pos_source)));
    f.push(format!("\"bbox\": {{\"min\": [{}], \"max\": [{}]}}", jnums(lo), jnums(hi)));
    f.push(format!("\"roles\": [{}]", jstrs(&roles.names)));
    f.push(format!("\"role_counts\": [{}]", jusize(&roles.counts)));
    f.push(format!("\"nts\": [{}]", jstrs(&nts.names)));
    f.push(format!("\"nt_counts\": [{}]", jusize(&nts.counts)));
    f.push(format!("\"classes\": [{}]", jstrs(&classes.names)));
    f.push(format!("\"class_counts\": [{}]", jusize(&classes.counts)));
    format!("  {{\n   {}\n  }}", f.join(",\n   "))
}

fn write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::File::create(path)?;
    f.write_all(bytes)
}

fn jstr(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn jstrs(v: &[String]) -> String {
    v.iter().map(|s| jstr(s)).collect::<Vec<_>>().join(",")
}

fn jusize(v: &[usize]) -> String {
    v.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(",")
}

fn jnums(v: &[f32; 3]) -> String {
    v.iter().map(|n| format!("{}", n)).collect::<Vec<_>>().join(",")
}
