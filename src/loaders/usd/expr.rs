//! Variable expressions in asset paths and variant selections.
//!
//! A layer can declare variables and then write them into the places a path or
//! a selection goes:
//!
//! ```text
//! (
//!     expressionVariables = {
//!         string COLOR = "red"
//!     }
//! )
//! def Scope "Thing" (
//!     references = @`"${COLOR}_asset.usda"`@
//! )
//! ```
//!
//! That is how one shot layer drives which of several assets a whole stage
//! pulls in, without editing every reference. The backticks mark an expression;
//! everything else is the path it looks like.
//!
//! # What is evaluated
//!
//! String literals and `${VAR}` substitution, which is what the overwhelming
//! majority of authored expressions are. USD's expression language also has
//! functions — `if`, `eq`, `and` — and an expression using one is left alone
//! rather than guessed at, so it fails to resolve loudly instead of resolving
//! to the wrong asset quietly.

use std::collections::HashMap;

use super::value::UsdValue;

/// The variables in scope, by name.
pub type Variables = HashMap<String, String>;

/// Read a layer's `expressionVariables`.
pub fn variables_of(metadata: &[(String, UsdValue)]) -> Variables {
    let mut out = Variables::new();
    for (key, value) in metadata {
        if key
            .split_once(' ')
            .map(|(_, rest)| rest)
            .unwrap_or(key)
            != "expressionVariables"
        {
            continue;
        }
        if let UsdValue::Dict(entries) = value {
            for (name, value) in entries {
                if let Some(text) = value.as_str() {
                    out.insert(name.clone(), text.to_string());
                }
            }
        }
    }
    out
}

/// Whether some text is an expression rather than a literal.
pub fn is_expression(text: &str) -> bool {
    text.len() >= 2 && text.starts_with('`') && text.ends_with('`')
}

/// Evaluate an expression, or return the text unchanged if it is not one.
///
/// Returns `None` when the text *is* an expression but uses something beyond
/// substitution — a function call, an undefined variable — because resolving
/// half of it would name a file nobody meant.
pub fn evaluate(text: &str, variables: &Variables) -> Option<String> {
    if !is_expression(text) {
        return Some(text.to_string());
    }
    let body = text[1..text.len() - 1].trim();

    // A bare quoted string, possibly with substitutions in it.
    let inner = match (body.starts_with('"'), body.ends_with('"'), body.len() >= 2) {
        (true, true, true) => &body[1..body.len() - 1],
        // Not a string literal: a function call or something else this does
        // not evaluate.
        _ => return None,
    };

    let mut out = String::with_capacity(inner.len());
    let mut rest = inner;
    while let Some(at) = rest.find("${") {
        out.push_str(&rest[..at]);
        let after = &rest[at + 2..];
        let end = after.find('}')?;
        let name = &after[..end];
        // An undefined variable is not an empty string; it is a mistake.
        out.push_str(variables.get(name)?);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vars(pairs: &[(&str, &str)]) -> Variables {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn a_plain_path_is_left_alone() {
        let v = vars(&[]);
        assert_eq!(evaluate("asset.usda", &v).as_deref(), Some("asset.usda"));
        // A backtick in the middle is not an expression.
        assert_eq!(evaluate("a`b.usda", &v).as_deref(), Some("a`b.usda"));
    }

    #[test]
    fn a_variable_is_substituted() {
        let v = vars(&[("COLOR", "red")]);
        assert_eq!(
            evaluate("`\"${COLOR}_asset.usda\"`", &v).as_deref(),
            Some("red_asset.usda")
        );
        // Several, and in the middle.
        let v = vars(&[("SHOT", "010"), ("DEPT", "anim")]);
        assert_eq!(
            evaluate("`\"shots/${SHOT}/${DEPT}.usda\"`", &v).as_deref(),
            Some("shots/010/anim.usda")
        );
    }

    /// An undefined variable is a mistake, not an empty string: resolving it
    /// away would name a file nobody meant and open it without complaint.
    #[test]
    fn an_undefined_variable_does_not_resolve() {
        assert_eq!(evaluate("`\"${MISSING}.usda\"`", &vars(&[])), None);
    }

    /// The expression language has more in it than substitution, and guessing
    /// at the rest is worse than declining.
    #[test]
    fn anything_beyond_substitution_is_declined() {
        let v = vars(&[("A", "1")]);
        assert_eq!(evaluate("`if(eq(${A}, 1), \"x.usda\", \"y.usda\")`", &v), None);
        assert_eq!(evaluate("`${A}`", &v), None, "not a string literal");
    }

    #[test]
    fn an_unterminated_substitution_does_not_resolve() {
        let v = vars(&[("A", "1")]);
        assert_eq!(evaluate("`\"${A\"`", &v), None);
    }

    #[test]
    fn variables_are_read_from_the_layer() {
        let layer = super::super::parse::parse(
            r#"#usda 1.0
(
    expressionVariables = {
        string COLOR = "red"
        string SHOT = "010"
    }
)
"#,
        )
        .unwrap();
        let v = variables_of(&layer.metadata);
        assert_eq!(v.get("COLOR").map(String::as_str), Some("red"));
        assert_eq!(v.get("SHOT").map(String::as_str), Some("010"));
    }
}
