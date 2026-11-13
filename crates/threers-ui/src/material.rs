//! Materials: a box that shows what is behind it, blurred and made more
//! colourful, through a translucent tint, as frosted glass does. A sheet, a
//! menu or a bar over content keeps the content's colour without its
//! detail, so what is on the material stays legible over anything.
//!
//! [`Ui::set_material`] gives a box one of a few thicknesses (more blur
//! and a denser tint as they thicken), tinted for the light and the dark
//! appearance. In CSS terms it is `backdrop-filter: blur(…) saturate(…)`
//! over a translucent `background`, which can also be set directly.
//!
//! Painters that cannot read back what they painted (the 2D canvas, a GPU
//! pass drawing straight into a window's texture) show the tint alone.
//!
//! ```
//! use threers_ui::material::Material;
//! use threers_ui::{style, Ui};
//!
//! let mut ui = Ui::new();
//! let root = ui.root();
//! let bar = ui.add_box(root, style! { height: 44px; });
//! ui.set_material(bar, Some(Material::Bar));
//! assert!(ui.style(bar).backdrop_blur > 0.0);
//! ```

use crate::states::Appearance;
use crate::tree::{NodeId, Ui};

/// How thick a material is: how much it blurs and how densely it tints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    /// Barely there: most of the colour behind shows.
    UltraThin,
    Thin,
    Regular,
    /// Little shows through but colour.
    Thick,
    /// For a toolbar or a tab bar over scrolling content.
    Bar,
}

impl Material {
    /// Blur (standard deviation, px), saturation, and the tint's opacity
    /// over white (light) and near-black (dark).
    fn recipe(self) -> (f32, f32, f32, f32) {
        match self {
            Material::UltraThin => (10.0, 1.8, 0.28, 0.32),
            Material::Thin => (16.0, 1.8, 0.45, 0.48),
            Material::Regular => (24.0, 1.8, 0.62, 0.62),
            Material::Thick => (30.0, 1.8, 0.78, 0.76),
            Material::Bar => (20.0, 1.8, 0.75, 0.72),
        }
    }
}

impl Ui {
    /// Makes `node` a material (see the [module docs](self)), or plain
    /// again with `None` (its backdrop and tint removed).
    pub fn set_material(&mut self, node: NodeId, material: Option<Material>) {
        let Some(m) = material else {
            self.update_style(node, |s| {
                s.backdrop_blur = 0.0;
                s.backdrop_saturate = 1.0;
                s.background = None;
            });
            return;
        };
        let (blur, saturate, light, dark) = m.recipe();
        let light = format!("background: rgba(250, 250, 252, {light})");
        let dark = format!("background: rgba(28, 28, 32, {dark})");
        self.update_style(node, |s| {
            s.backdrop_blur = blur;
            s.backdrop_saturate = saturate;
        });
        self.appearance_style(node, Appearance::Light, &light);
        self.appearance_style(node, Appearance::Dark, &dark);
    }
}
