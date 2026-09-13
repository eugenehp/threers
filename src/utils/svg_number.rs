/// Format a coordinate for an SVG attribute: fixed-point, trailing zeros
/// trimmed.
///
/// `1.50` and `3.00` become `1.5` and `3`, which over the few hundred thousand
/// coordinates a vector render emits is a real fraction of the file. Negative
/// zero comes back as `0`, and a non-finite value as `0` rather than the
/// `NaN` that would make the document unparseable.
pub fn format_svg_number(v: f32, precision: usize) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    let s = format!("{:.*}", precision, v);
    if !s.contains('.') {
        return s;
    }
    let t = s.trim_end_matches('0').trim_end_matches('.');
    match t {
        "" | "-" | "-0" => "0".into(),
        _ => t.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_guards() {
        assert_eq!(format_svg_number(1.5, 2), "1.5");
        assert_eq!(format_svg_number(3.0, 2), "3");
        assert_eq!(format_svg_number(-0.001, 2), "0");
        assert_eq!(format_svg_number(f32::NAN, 2), "0");
        assert_eq!(format_svg_number(f32::INFINITY, 2), "0");
        assert_eq!(format_svg_number(12.3456, 2), "12.35");
        assert_eq!(format_svg_number(-12.3456, 3), "-12.346");
        assert_eq!(format_svg_number(0.0, 4), "0");
    }
}
