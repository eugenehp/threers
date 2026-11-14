//! Themes from `tailwind.config`-shaped JSON.

use std::collections::HashMap;

use serde_json::Value;

use super::theme::{rem_text, Paint, Tailwind};

/// A length in config text (`0.25rem`, `4px`, `4`) in pixels.
fn config_px(v: &str, rem: f32) -> Option<f32> {
    let v = v.trim();
    if let Some(r) = v.strip_suffix("rem") {
        return r.trim().parse::<f32>().ok().map(|r| r * rem);
    }
    v.strip_suffix("px").unwrap_or(v).trim().parse().ok()
}

fn colour(name: &str, css: &str) -> Result<crate::style::Rgba, String> {
    crate::css::color(css).ok_or_else(|| format!("colour `{name}`: `{css}` is not a colour"))
}

/// Colours under `prefix`: a CSS colour, `[light, dark]`, or an object of
/// shades (`DEFAULT` the bare name).
fn colours(prefix: &str, v: &Value, out: &mut HashMap<String, Paint>) -> Result<(), String> {
    match v {
        Value::String(s) => {
            out.insert(prefix.to_owned(), Paint::fixed(colour(prefix, s)?));
        }
        Value::Array(a) => match a.as_slice() {
            [Value::String(light), Value::String(dark)] => {
                out.insert(
                    prefix.to_owned(),
                    Paint::adaptive(colour(prefix, light)?, colour(prefix, dark)?),
                );
            }
            _ => return Err(format!("colour `{prefix}`: an array is `[light, dark]`")),
        },
        Value::Object(o) => {
            for (k, v) in o {
                let key = match (k.as_str(), prefix) {
                    ("DEFAULT", p) => p.to_owned(),
                    (k, "") => k.to_owned(),
                    (k, p) => format!("{p}-{k}"),
                };
                colours(&key, v, out)?;
            }
        }
        _ => {
            return Err(format!(
                "colour `{prefix}` is not a string, `[light, dark]` or an object"
            ))
        }
    }
    Ok(())
}

impl Tailwind {
    /// Reads scales from JSON shaped as a `tailwind.config`'s `theme`:
    /// what `theme` has replaces the scale it names, and what
    /// `theme.extend` has adds to it (a bare object is read as `theme`).
    /// Read are `colors` (nested shades, `DEFAULT`, and — this UI's own —
    /// `["light", "dark"]` for a colour that switches with the
    /// appearance), `spacing`, `screens`, `fontSize` (a size, or
    /// `[size, lineHeight]` or `[size, {"lineHeight": …}]`), `fontFamily`
    /// (the first of a list), `fontWeight`, `lineHeight`, `borderRadius`,
    /// `boxShadow`, `blur`, `maxWidth` and `perspective`, with lengths in
    /// `px` or `rem`; and `components`: `{"btn": "px-4 py-2 rounded"}`.
    ///
    /// ```
    /// use threers_ui::tailwind::Tailwind;
    /// use threers_ui::states::Appearance;
    ///
    /// let mut tw = Tailwind::default();
    /// tw.extend_json(r##"{
    ///     "theme": {
    ///         "extend": {
    ///             "colors": {
    ///                 "brand": { "DEFAULT": "#ff6600", "dark": "#cc5200" },
    ///                 "surface": ["#ffffff", "#18181b"]
    ///             },
    ///             "spacing": { "18": "4.5rem" },
    ///             "components": { "card": "p-4 rounded-lg bg-surface shadow" }
    ///         }
    ///     }
    /// }"##).unwrap();
    /// let s = tw.style("card text-brand-dark w-18").unwrap();
    /// assert_eq!(s.width, threers_ui::Length::Px(72.0));
    /// assert_eq!(s.background, Some([1.0; 4]));
    /// let dark = tw.style_in("card", Appearance::Dark).unwrap();
    /// assert_eq!(dark.background, threers_ui::css::color("#18181b"));
    /// ```
    pub fn extend_json(&mut self, json: &str) -> Result<(), String> {
        let root: Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
        let theme = root.get("theme").unwrap_or(&root);
        let obj = theme.as_object().ok_or("the theme is not an object")?;
        for (key, value) in obj {
            if key != "extend" {
                self.read_scale(key, value, true)?;
            }
        }
        if let Some(extend) = obj.get("extend") {
            for (key, value) in extend.as_object().ok_or("`extend` is not an object")? {
                self.read_scale(key, value, false)?;
            }
        }
        Ok(())
    }

    /// [`Tailwind::default`] with [`extend_json`](Self::extend_json).
    pub fn from_json(json: &str) -> Result<Tailwind, String> {
        let mut t = Tailwind::default();
        t.extend_json(json)?;
        Ok(t)
    }

    fn read_scale(&mut self, key: &str, value: &Value, replace: bool) -> Result<(), String> {
        let rem = self.rem;
        if key == "colors" {
            if replace {
                self.colors.clear();
            }
            return colours("", value, &mut self.colors);
        }
        let obj = value
            .as_object()
            .ok_or_else(|| format!("`{key}` is not an object"))?;
        let text = |k: &str, v: &Value| -> Result<String, String> {
            match v {
                Value::String(s) => Ok(s.clone()),
                Value::Number(n) => Ok(n.to_string()),
                _ => Err(format!("`{key}.{k}` is not a string")),
            }
        };
        let length = |k: &str, v: &Value| -> Result<f32, String> {
            let t = text(k, v)?;
            config_px(&t, rem)
                .ok_or_else(|| format!("`{key}.{k}`: `{t}` is not a length in px or rem"))
        };
        // `DEFAULT` is the bare class (`rounded`, `shadow`).
        let name = |k: &str| {
            if k == "DEFAULT" {
                String::new()
            } else {
                k.to_owned()
            }
        };
        fn fill<V>(map: &mut HashMap<String, V>, replace: bool, entries: Vec<(String, V)>) {
            if replace {
                map.clear();
            }
            map.extend(entries);
        }
        let lengths = |named: bool| -> Result<Vec<(String, f32)>, String> {
            obj.iter()
                .map(|(k, v)| Ok((if named { name(k) } else { k.clone() }, length(k, v)?)))
                .collect()
        };
        match key {
            "spacing" => fill(&mut self.spacing, replace, lengths(false)?),
            "borderRadius" => fill(&mut self.radii, replace, lengths(true)?),
            "blur" => fill(&mut self.blurs, replace, lengths(true)?),
            "maxWidth" => fill(&mut self.max_widths, replace, lengths(false)?),
            "perspective" => fill(&mut self.perspectives, replace, lengths(false)?),
            "boxShadow" => fill(
                &mut self.shadows,
                replace,
                obj.iter()
                    .map(|(k, v)| Ok((name(k), text(k, v)?)))
                    .collect::<Result<_, String>>()?,
            ),
            "components" => fill(
                &mut self.components,
                replace,
                obj.iter()
                    .map(|(k, v)| Ok((k.clone(), text(k, v)?)))
                    .collect::<Result<_, String>>()?,
            ),
            "lineHeight" => fill(
                &mut self.line_heights,
                replace,
                obj.iter()
                    .map(|(k, v)| Ok((k.clone(), rem_text(&text(k, v)?, rem))))
                    .collect::<Result<_, String>>()?,
            ),
            "fontWeight" => {
                let weights = obj
                    .iter()
                    .map(|(k, v)| {
                        let w = text(k, v)?;
                        Ok((
                            k.clone(),
                            w.parse()
                                .map_err(|_| format!("`fontWeight.{k}`: `{w}` is not a weight"))?,
                        ))
                    })
                    .collect::<Result<_, String>>()?;
                fill(&mut self.font_weights, replace, weights);
            }
            "fontFamily" => {
                let families = obj
                    .iter()
                    .map(|(k, v)| {
                        let first = match v {
                            Value::Array(a) => a
                                .first()
                                .map(|f| text(k, f))
                                .transpose()?
                                .unwrap_or_default(),
                            v => text(k, v)?.split(',').next().unwrap_or("").to_owned(),
                        };
                        Ok((
                            k.clone(),
                            first
                                .trim()
                                .trim_matches(|c| c == '"' || c == '\'')
                                .to_owned(),
                        ))
                    })
                    .collect::<Result<_, String>>()?;
                fill(&mut self.font_families, replace, families);
            }
            "fontSize" => {
                let sizes = obj
                    .iter()
                    .map(|(k, v)| {
                        let (size, lh) = match v {
                            Value::Array(a) => {
                                let size = length(
                                    k,
                                    a.first()
                                        .ok_or_else(|| format!("`fontSize.{k}` is empty"))?,
                                )?;
                                let lh = match a.get(1) {
                                    Some(Value::Object(o)) => {
                                        o.get("lineHeight").map(|l| text(k, l)).transpose()?
                                    }
                                    Some(l) => Some(text(k, l)?),
                                    None => None,
                                };
                                (size, lh)
                            }
                            v => (length(k, v)?, None),
                        };
                        Ok((k.clone(), (size, lh.map(|l| rem_text(&l, rem)))))
                    })
                    .collect::<Result<_, String>>()?;
                fill(&mut self.font_sizes, replace, sizes);
            }
            "screens" => {
                let screens = lengths(false)?;
                if replace {
                    self.screens.clear();
                }
                for (k, w) in screens {
                    *self = std::mem::take(self).screen(&k, w);
                }
            }
            _ => return Err(format!("`{key}` is not a scale this UI reads")),
        }
        Ok(())
    }
}
