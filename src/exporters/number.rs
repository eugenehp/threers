//! `Number.prototype.toString()` for an `f64`, as ECMAScript specifies it
//! (`Number::toString`, radix 10).

/// Print `v` exactly as JavaScript's `String(v)` does: the shortest digits
/// that round-trip, positional between `1e-7` and `1e21`, exponential
/// (`1.5e-7`, `1e+21`) outside, `0` for both zeroes, `NaN`, `Infinity`.
pub fn js_number(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v == 0.0 {
        return "0".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let mut out = String::new();
    let a = if v < 0.0 {
        out.push('-');
        -v
    } else {
        v
    };
    // Rust's `{:e}` with no precision prints the shortest round-tripping
    // digits, which is the digit string ECMAScript asks for.
    let sci = format!("{a:e}");
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    // ECMAScript's n: the value is 0.d₁d₂…d_k × 10ⁿ.
    let n = exp + 1;
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if n > 0 { '+' } else { '-' });
        out.push_str(&(n - 1).abs().to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::js_number;

    #[test]
    fn matches_javascript() {
        // Expected strings are `String(x)` in node.
        let cases: &[(f64, &str)] = &[
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (-2.5, "-2.5"),
            (0.1f32 as f64, "0.10000000149011612"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (123456789012345680000.0, "123456789012345680000"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (-3.4028234663852886e38, "-3.4028234663852886e+38"),
            (0.000123, "0.000123"),
            (5e-324, "5e-324"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (12.5, "12.5"),
            (100.0, "100"),
        ];
        for (v, want) in cases {
            assert_eq!(js_number(*v), *want, "{v:e}");
        }
    }
}
