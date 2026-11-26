//! Gradients, glass and centring, for any element (SwiftUI's shape styles,
//! `.glassEffect` and frame alignment).

use super::El;
use crate::environment::Token;
use crate::fill::{Fill, Gradient, Paint};
use crate::material::Material;

impl<S: 'static> El<S> {
    /// A gradient behind it (`.background(LinearGradient(…))`), over any
    /// colour it has there.
    pub fn background_gradient(self, g: Gradient) -> Self {
        self.css(&format!("background: {}", g.css()))
    }

    /// Its text in a gradient, along each text in it
    /// (`.foregroundStyle(gradient)`); what is drawn (an icon) takes the
    /// first colour.
    pub fn foreground_gradient(mut self, g: Gradient) -> Self {
        if let Some(first) = g.first_css() {
            self = self.css(&format!("color: {first}"));
        }
        self.text_gradient = Some(g);
        self
    }

    /// A border of a gradient round it, `width` thick, its corners
    /// `radius` round, over the theme's `behind` colour.
    pub fn border_gradient(self, g: Gradient, width: f32, radius: f32, behind: Token) -> Self {
        El::div(&format!(
            "padding: {width}px; border-radius: {radius}px; display: flex; flex-direction: column"
        ))
        .background_gradient(g)
        .child(
            self.css(&format!(
                "border-radius: {}px; align-self: stretch",
                (radius - width).max(0.0)
            ))
            .background(behind),
        )
    }

    /// Everything in it drawn in the accent is drawn in `g` instead: a
    /// switch, a progress bar, a default button, the chosen of a set
    /// (`.tint(gradient)`).
    pub fn tint(self, g: Gradient) -> Self {
        self.fill(Token::Accent, Fill::Gradient(g.clone()))
            .fill(Token::Prominent, Fill::Gradient(g))
    }

    /// Backgrounds of `token` in it painted as `fill`.
    pub fn fill(mut self, token: Token, fill: Fill) -> Self {
        self.tints.retain(|(t, _)| *t != token);
        self.tints.push((token, fill));
        self
    }

    /// Glass: what is behind it, blurred, under a little of the theme's
    /// surface, with a lit edge (`.glassEffect()`).
    pub fn glass(self, material: Material) -> Self {
        self.glass_tinted(material, Token::Surface)
    }

    /// [`glass`](Self::glass) tinted in `token` (`.glassEffect(.regular.tint(…))`).
    pub fn glass_tinted(self, material: Material, token: Token) -> Self {
        let (blur, saturate, _, opacity) = material.recipe();
        self.css(&format!(
            "backdrop-filter: blur({blur}px) saturate({saturate}); \
             background: color-mix(in srgb, var(--{}) {}%, transparent); \
             border: 1px solid; border-color: color-mix(in srgb, var(--text) 16%, transparent)",
            token.css_name(),
            (opacity * 100.0).round()
        ))
    }

    /// In the middle of what it is in, across and down.
    pub fn center(self) -> Self {
        self.center_horizontally().center_vertically()
    }

    /// In the middle across.
    pub fn center_horizontally(self) -> Self {
        self.css("margin-left: auto; margin-right: auto; align-self: auto; flex-grow: 0")
    }

    /// In the middle, down.
    pub fn center_vertically(self) -> Self {
        self.css("margin-top: auto; margin-bottom: auto; align-self: auto")
    }

    /// What is in it in its middle, across and down, its text too.
    pub fn centers_content(self) -> Self {
        self.css("display: flex; align-items: center; justify-content: center; text-align: center")
    }

    /// It and what is in it read `direction` (SwiftUI's
    /// `.environment(\\.layoutDirection, …)`).
    pub fn layout_direction(self, direction: crate::environment::LayoutDirection) -> Self {
        self.css(match direction {
            crate::environment::LayoutDirection::LeftToRight => "direction: ltr",
            crate::environment::LayoutDirection::RightToLeft => "direction: rtl",
        })
    }

    /// It and what is in it read as its first letter does: for what people
    /// write (a message, a note), in whatever language.
    pub fn reads_as_written(self) -> Self {
        self.css("direction: auto")
    }

    /// Drawn the other way round where it reads right to left (an arrow,
    /// a chevron).
    pub fn flips_for_right_to_left(self) -> Self {
        self.css("rtl-flip: on")
    }

    /// Its lines of text centred.
    pub fn center_text(self) -> Self {
        self.css("text-align: center")
    }
}

impl Gradient {
    fn first_css(&self) -> Option<String> {
        self.first().map(|p| match p {
            Paint::Token(t) => format!("var(--{})", t.css_name()),
            Paint::Css(c) => c.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::environment::Token;
    use crate::fill::{Fill, Gradient};
    use crate::input::Input;
    use crate::material::Material;
    use crate::view::{
        button, gauge, hstack, pill, progress_view, stepper, text, toggle, vstack, El, Tint, View,
    };
    use crate::{FixedWidthText, NodeId, Ui};

    fn show(els: Vec<El<()>>) -> Ui {
        let mut ui = Ui::new();
        let mut input = Input::new();
        let root = ui.root();
        let mut v = View::new(
            &mut ui,
            root,
            "display: flex; flex-direction: column; width: 400px; height: 300px",
        );
        v.update_with(&mut ui, &mut input, &(), els);
        ui.layout(400.0, 300.0, &mut FixedWidthText(7.0));
        ui
    }

    fn nodes(ui: &Ui) -> Vec<NodeId> {
        let mut out = vec![];
        let mut stack = vec![ui.root()];
        while let Some(n) = stack.pop() {
            stack.extend_from_slice(ui.children(n));
            out.push(n);
        }
        out
    }

    fn gradients(ui: &Ui) -> usize {
        nodes(ui)
            .iter()
            .filter(|n| !ui.style(**n).background_gradient.is_empty())
            .count()
    }

    fn parts() -> Vec<El<()>> {
        vec![
            toggle("On", true, |_, _| {}),
            progress_view(Some(0.5)),
            button("Go", |_| {}).button_style(crate::components::ButtonKind::Primary),
        ]
    }

    /// Switch, progress and default button: each takes a tint, from the
    /// element they are in or from the theme.
    #[test]
    fn components_take_a_gradient_from_around_them_or_the_theme() {
        let g = || Gradient::linear(90.0, ["#7c3aed", "#ec4899"]);
        assert_eq!(gradients(&show(parts())), 0);
        assert_eq!(gradients(&show(vec![vstack(4.0, parts()).tint(g())])), 3);
        let mut ui = show(parts());
        for t in [Token::Accent, Token::Prominent] {
            ui.set_token_fill(t, Some(Fill::Gradient(g())));
        }
        assert_eq!(gradients(&ui), 3);
        ui.set_token_fill(Token::Accent, None);
        assert_eq!(gradients(&ui), 1);
    }

    #[test]
    fn any_element_takes_a_gradient_and_glass() {
        let g = || Gradient::of(Token::Accent);
        let els: Vec<El<()>> = vec![
            text("a"),
            hstack(0.0, [text("b")]),
            pill("c", Tint::Accent),
            stepper("1", ("-", "+"), None, None),
            gauge(0.5, "d", "d", 40.0),
        ];
        let n = els.len();
        let ui = show(
            els.into_iter()
                .map(|e| e.background_gradient(g()))
                .collect(),
        );
        assert_eq!(gradients(&ui), n);
        let ui = show(vec![
            text("a").glass(Material::Regular),
            hstack(0.0, [text("b")]).glass_tinted(Material::Thin, Token::Accent),
        ]);
        let glass: Vec<_> = nodes(&ui)
            .into_iter()
            .filter(|n| ui.style(*n).backdrop_blur > 0.0)
            .collect();
        assert_eq!(glass.len(), 2);
        assert!(glass.iter().all(|n| ui
            .style(*n)
            .background
            .is_some_and(|c| c[3] > 0.2 && c[3] < 0.95)));
        let mut ui = show(vec![text("a").background(Token::Surface)]);
        ui.set_token_fill(Token::Surface, Some(Fill::Glass(Material::Thick)));
        assert!(nodes(&ui)
            .iter()
            .any(|n| ui.style(*n).backdrop_blur == 30.0));
    }

    #[test]
    fn gradients_in_every_shape_are_read() {
        for g in [
            Gradient::linear(45.0, [Token::Accent, Token::Success]),
            Gradient::radial(["#ffffff", "rgba(0, 0, 0, 0.2)"]),
            Gradient::angular(90.0, ["red", "blue"]).at([0.2, 0.7]),
            Gradient::of(Token::Warning),
            Gradient::linear(0.0, ["#123456"]),
        ] {
            let ui = show(vec![text("x").background_gradient(g.clone())]);
            assert_eq!(gradients(&ui), 1, "{}", g.css());
        }
    }

    #[test]
    fn text_takes_a_gradient_across_it() {
        let ui =
            show(vec![text("abc").foreground_gradient(Gradient::linear(
                90.0,
                ["#ff0000", "#0000ff"],
            ))]);
        let t = nodes(&ui)
            .into_iter()
            .find(|n| matches!(ui.kind(*n), crate::NodeKind::Text(_)))
            .unwrap();
        let runs = ui.text_runs(t).unwrap();
        let c = |i: usize| runs[i].color.unwrap();
        assert!(c(0)[0] > 0.9 && c(2)[2] > 0.9 && c(1)[0] > 0.2 && c(1)[2] > 0.2);
    }

    #[test]
    fn anything_is_centred_in_what_it_is_in() {
        for el in [
            text::<()>("xx"),
            pill("xx", Tint::Accent),
            stepper("1", ("-", "+"), None, None),
            progress_view(Some(0.5)).css("width: 100px"),
            text("xx").fill_width(),
        ] {
            let ui = show(vec![El::div(
                "display: flex; flex-direction: column; width: 400px; height: 300px",
            )
            .child(el.center())]);
            let holder = nodes(&ui)
                .into_iter()
                .find(|n| {
                    ui.rect(*n).height == 300.0
                        && ui.children(*n).len() == 1
                        && ui.rect(ui.children(*n)[0]).height < 200.0
                })
                .unwrap();
            let r = ui.rect(ui.children(holder)[0]);
            assert!((r.x + r.width / 2.0 - 200.0).abs() < 1.0, "{r:?}");
            assert!((r.y + r.height / 2.0 - 150.0).abs() < 1.0, "{r:?}");
        }
        let ui = show(vec![El::div("width: 400px; height: 300px")
            .child(text("xx"))
            .centers_content()]);
        let t = nodes(&ui)
            .into_iter()
            .find(|n| matches!(ui.kind(*n), crate::NodeKind::Text(_)))
            .unwrap();
        let r = ui.rect(t);
        assert!(
            (r.x + r.width / 2.0 - 200.0).abs() < 1.0 && (r.y + r.height / 2.0 - 150.0).abs() < 1.0
        );
    }
}
