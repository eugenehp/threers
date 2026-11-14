//! A layout's accents row: French's ê, ô, œ… on the keyboard, typed like
//! its other letters (capital after Shift).

use threers_ui::input::Input;
use threers_ui::virtual_keyboard::{KeyboardLayout, VirtualKeyboard};
use threers_ui::{style, FixedWidthText, Role, Semantics, TextField, Ui};

#[test]
fn french_has_its_accents() {
    let mut ui = Ui::new();
    let mut input: Input<TextField> = Input::new();
    let mut field = TextField::new("");
    let root = ui.root();
    let b = ui.add_box(root, style! { height: 20px; });
    ui.set_semantics(b, Semantics::new(Role::TextBox).label("Answer"));
    input.on_key(b, |f: &mut TextField, _, key, m, _| {
        !matches!(f.key(key, m, |_| true), threers_ui::FieldKey::Ignored)
    });
    let kb = VirtualKeyboard::new(&mut ui, &mut input);
    kb.set_enabled(true);
    kb.set_layouts(&mut ui, vec![KeyboardLayout::french()]);
    input.set_focus(&mut ui, &mut field, Some(b));
    input.tick(&mut ui, &mut field, 0.0);
    ui.layout(400.0, 700.0, &mut FixedWidthText(7.0));
    assert!(kb.showing(&ui));
    // The accents show as keys…
    let keys: Vec<String> = ui
        .accessibility_tree()
        .into_iter()
        .filter(|a| a.role == Role::Button)
        .map(|a| a.name)
        .collect();
    for k in ["ê", "ô", "œ", "«", "é", "ç"] {
        assert!(
            keys.iter().any(|n| n == k),
            "{k} not on the keyboard: {keys:?}"
        );
    }
    // …and type as letters do.
    for k in ["ê", "t", "r", "e"] {
        kb.press(&mut ui, &mut input, &mut field, k);
    }
    assert_eq!(field.text, "être");
    // English has none of them.
    assert!(KeyboardLayout::english().accents.is_empty());
}
