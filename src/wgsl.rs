//! WGSL with its comments taken out at compile time.
//!
//! threers' shaders are written with their comments — what a pass reads, why
//! a constant has the value it does — and kept that way in the sources. The
//! GPU's compiler throws the comments away, so a build that carries them
//! (every wasm module, whose data segment holds the shader text) ships tens
//! of kB of prose for nothing. [`wgsl!`](crate::wgsl!) strips a constant
//! source during const evaluation: only the stripped text is ever used at
//! run time, so only it is emitted into the binary.
//!
//! Line comments go with any blanks before them, and block comments (which
//! nest, in WGSL) with the blanks before them; the line breaks stay, so a
//! compile error's line number still points at the source's line. Code that
//! splices into a shader looks for a token that is not a comment
//! (`@@SHARED_FUNCTIONS@@`, say), since comments do not survive.

/// The bytes [`strip`] writes for `src`.
pub const fn stripped_len(src: &str) -> usize {
    run(src.as_bytes(), &mut [], false)
}

/// `src` without its comments, `N` bytes long — `N` must be
/// [`stripped_len`]`(src)`. Use [`wgsl!`](crate::wgsl!), which works it out.
pub const fn strip<const N: usize>(src: &str) -> [u8; N] {
    let mut out = [0u8; N];
    let n = run(src.as_bytes(), &mut out, true);
    assert!(n == N, "wgsl::strip: N is not stripped_len(src)");
    out
}

/// The stripping, once for both: counts the bytes kept, and writes them to
/// `out` when `write` is set.
const fn run(s: &[u8], out: &mut [u8], write: bool) -> usize {
    let mut i = 0;
    let mut n = 0;
    // The start of a run of spaces or tabs not yet written: blanks before a
    // comment or a line break are dropped, any others written as they were.
    let mut blank: Option<usize> = None;
    while i < s.len() {
        let c = s[i];
        if c == b' ' || c == b'\t' {
            if blank.is_none() {
                blank = Some(i);
            }
            i += 1;
            continue;
        }
        let line = c == b'/' && i + 1 < s.len() && s[i + 1] == b'/';
        let block = c == b'/' && i + 1 < s.len() && s[i + 1] == b'*';
        if let Some(b) = blank {
            if !(line || block || c == b'\n' || c == b'\r') {
                let mut k = b;
                while k < i {
                    if write {
                        out[n] = s[k];
                    }
                    n += 1;
                    k += 1;
                }
            }
            blank = None;
        }
        if line {
            while i < s.len() && s[i] != b'\n' && s[i] != b'\r' {
                i += 1;
            }
        } else if block {
            i += 2;
            let mut depth = 1;
            let mut breaks = false;
            while i < s.len() && depth > 0 {
                if s[i] == b'/' && i + 1 < s.len() && s[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if s[i] == b'*' && i + 1 < s.len() && s[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    if s[i] == b'\n' {
                        breaks = true;
                        if write {
                            out[n] = b'\n';
                        }
                        n += 1;
                    }
                    i += 1;
                }
            }
            // `a/* … */b` is two tokens: keep them apart.
            if !breaks {
                if write {
                    out[n] = b' ';
                }
                n += 1;
            }
        } else {
            if write {
                out[n] = c;
            }
            n += 1;
            i += 1;
        }
    }
    n
}

/// A constant WGSL source — a literal, `include_str!`, `concat!` of those, or
/// any other constant `&str` expression — as a `&'static str` without its
/// comments, stripped at compile time (see [`wgsl`](crate::wgsl)).
///
/// ```
/// const SHADER: &str = threers::wgsl!(
///     "// The whole pass.\nfn f() -> f32 { return 1.0; } // one\n"
/// );
/// assert_eq!(SHADER, "\nfn f() -> f32 { return 1.0; }\n");
/// ```
#[macro_export]
macro_rules! wgsl {
    ($src:expr $(,)?) => {{
        const SRC: &str = $src;
        #[allow(long_running_const_eval)]
        const LEN: usize = $crate::wgsl::stripped_len(SRC);
        #[allow(long_running_const_eval)]
        const OUT: [u8; LEN] = $crate::wgsl::strip::<LEN>(SRC);
        const STR: &str = match ::core::str::from_utf8(&OUT) {
            Ok(s) => s,
            Err(_) => panic!("wgsl!: stripping split a character"),
        };
        STR
    }};
}

#[cfg(test)]
mod tests {
    fn strip(src: &str) -> String {
        let mut out = vec![0u8; super::stripped_len(src)];
        let n = super::run(src.as_bytes(), &mut out, true);
        assert_eq!(n, out.len());
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn line_comments_go_with_the_blanks_before_them() {
        assert_eq!(strip("let a = 1; // one\n  // two\nlet b = 2;"), "let a = 1;\n\nlet b = 2;");
        assert_eq!(strip("// only"), "");
        assert_eq!(strip("a // x\r\nb"), "a\r\nb");
    }

    #[test]
    fn block_comments_nest_and_keep_line_breaks() {
        assert_eq!(strip("a/* x */b"), "a b");
        assert_eq!(strip("a /* x /* y */ z */ b"), "a  b");
        assert_eq!(strip("a /* one\ntwo */b"), "a\nb");
    }

    #[test]
    fn code_is_untouched() {
        let src = "@group(0) @binding(0) var<uniform> u : vec4<f32>;\nfn f(x : f32) -> f32 { return x / 2.0 * 3.0; }\n";
        assert_eq!(strip(src), src);
        assert_eq!(strip("  indented\n\ttabbed\n"), "  indented\n\ttabbed\n");
    }

    #[test]
    fn non_ascii_comments_strip_whole() {
        assert_eq!(strip("x // ½ — ü\ny"), "x\ny");
    }

    #[test]
    fn shipped_shaders_carry_no_comments_but_keep_their_splice_tokens() {
        use crate::renderer::shader::{MATERIAL_MAPS_MARKER, SHADER_SOURCE, SHARED_FUNCTIONS_MARKER};
        for (name, src) in [
            ("SHADER_SOURCE", SHADER_SOURCE),
            ("marks PREAMBLE", crate::marks::shader::PREAMBLE),
            ("fullscreen VS", crate::renderer::fullscreen::VS),
        ] {
            assert!(!src.contains("//") && !src.contains("/*"), "{name} still has comments");
        }
        assert_eq!(SHADER_SOURCE.matches(MATERIAL_MAPS_MARKER).count(), 1);
        assert_eq!(SHADER_SOURCE.matches(SHARED_FUNCTIONS_MARKER).count(), 1);
        assert_eq!(crate::renderer::shader::postfx_shader().matches("@@").count(), 0);
    }

    #[test]
    fn the_macro_strips_at_compile_time() {
        const A: &str = crate::wgsl!("fn f() {} // gone\n");
        assert_eq!(A, "fn f() {}\n");
        const B: &str = crate::wgsl!(concat!("a // x\n", "b"));
        assert_eq!(B, "a\nb");
    }
}
