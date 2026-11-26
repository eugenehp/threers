//! Which parts of an app's own title bar move its window.

use crate::{NodeId, Role, Ui};

/// Whether a press at (`x`, `y`) is on `bar` and on nothing in it that is
/// pressed itself (a button, a field, a tab): it moves the window.
pub fn grabs(ui: &Ui, bar: NodeId, x: f32, y: f32) -> bool {
    let Some(hit) = ui.hit_test(x, y) else {
        return false;
    };
    for n in ui.ancestors(hit) {
        if n == bar {
            return true;
        }
        let takes = ui.semantics(n).is_some_and(|s| {
            !matches!(
                s.role,
                Role::None
                    | Role::Generic
                    | Role::Group
                    | Role::StaticText
                    | Role::Heading(_)
                    | Role::Image
                    | Role::Status
                    | Role::Region
                    | Role::Toolbar
            )
        });
        if takes {
            return false;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedWidthText, Semantics};

    #[test]
    fn the_bar_moves_the_window_and_what_is_pressed_in_it_does_not() {
        let mut ui = Ui::new();
        let root = ui.root();
        let bar = ui.add_box(
            root,
            crate::style! { display: flex; height: 40px; width: 300px; },
        );
        let button = ui.add_box(bar, crate::style! { width: 50px; height: 40px; });
        ui.set_semantics(button, Semantics::new(Role::Button).label("Go"));
        let title = ui.add_box(bar, crate::style! { width: 80px; height: 40px; });
        ui.set_semantics(title, Semantics::new(Role::StaticText).label("App"));
        ui.layout(300.0, 200.0, &mut FixedWidthText(7.0));
        assert!(!grabs(&ui, bar, 10.0, 10.0), "a button");
        assert!(grabs(&ui, bar, 90.0, 10.0), "the title");
        assert!(grabs(&ui, bar, 250.0, 10.0), "the bar");
        assert!(!grabs(&ui, bar, 90.0, 100.0), "the page");
    }
}
