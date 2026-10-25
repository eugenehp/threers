//! The accessibility mirror in a real browser:
//! `FEATURES=ui TEST=ui_a11y_dom scripts/wasm-browser-test.sh`.
#![cfg(all(feature = "ui", target_arch = "wasm32"))]

use threers::ui::a11y_dom::{A11yAction, DomAccessibility};
use threers::ui::controls::{button, checkbox, range};
use threers::ui::{Edges, FixedWidthText, Role, Semantics, Ui};
use wasm_bindgen::JsCast;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};
use web_sys::{HtmlCanvasElement, HtmlElement};

wasm_bindgen_test_configure!(run_in_browser);

fn canvas() -> HtmlCanvasElement {
    let document = web_sys::window().unwrap().document().unwrap();
    let holder = document.create_element("div").unwrap();
    holder.set_attribute("style", "position:relative;width:400px;height:200px").unwrap();
    document.body().unwrap().append_child(&holder).unwrap();
    let c: HtmlCanvasElement = document.create_element("canvas").unwrap().dyn_into().unwrap();
    c.set_attribute("style", "position:absolute;left:0;top:0;width:400px;height:200px").unwrap();
    holder.append_child(&c).unwrap();
    c
}

/// Every control is in the page with its role, name and state, on its
/// box; text is there to read; focus, clicks and keys come back as actions;
/// a change updates the element and a removal takes it away.
#[wasm_bindgen_test]
fn the_mirror_is_what_a_screen_reader_reads() {
    let c = canvas();
    let mut ui = Ui::new();
    let root = ui.root();
    let font = threers::style! { font-family: Arial; font-size: 13px; color: black; };
    let mut text = FixedWidthText(7.0);
    let heading = ui.add_box(root, font.clone());
    ui.add_text(heading, "Render settings");
    ui.set_semantics(heading, Semantics::new(Role::Heading(2)));
    let apply = button(&mut ui, root, "Apply", &font, &mut text);
    let wire = checkbox(&mut ui, root, true, 13.0, [0.46, 0.46, 0.46, 1.0], Edges::ZERO);
    ui.semantics_mut(wire).label = Some("Wireframe".into());
    let slider = range(&mut ui, root, 0.25, false);
    ui.semantics_mut(slider).label = Some("Exposure".into());
    let note = ui.add_box(root, font.clone());
    let ready_text = ui.add_text(note, "Ready");
    ui.layout(400.0, 200.0, &mut text);

    let mut mirror = DomAccessibility::new(&c).expect("a mirror");
    mirror.update(&ui);
    let attr = |m: &DomAccessibility, id, k: &str| m.element(id).unwrap().get_attribute(k);
    assert_eq!(attr(&mirror, heading, "role").as_deref(), Some("heading"));
    assert_eq!(attr(&mirror, heading, "aria-level").as_deref(), Some("2"));
    assert_eq!(attr(&mirror, heading, "aria-label").as_deref(), Some("Render settings"));
    assert_eq!(attr(&mirror, apply, "role").as_deref(), Some("button"));
    assert_eq!(attr(&mirror, apply, "aria-label").as_deref(), Some("Apply"));
    assert_eq!(attr(&mirror, apply, "tabindex").as_deref(), Some("0"));
    assert_eq!(attr(&mirror, wire, "aria-checked").as_deref(), Some("true"));
    assert_eq!(attr(&mirror, slider, "aria-valuenow").as_deref(), Some("25"));
    assert_eq!(attr(&mirror, slider, "aria-label").as_deref(), Some("Exposure"));
    let ready = mirror.element(ready_text).expect("the text, mirrored (its box has no meaning of its own)").clone();
    assert_eq!(ready.text_content().as_deref(), Some("Ready"));
    assert_eq!(ready.get_attribute("role"), None, "plain text needs no role");
    // On the control's box.
    let r = mirror.element(apply).unwrap().get_bounding_client_rect();
    let canvas_rect = c.get_bounding_client_rect();
    assert!((r.left() - canvas_rect.left() - ui.rect(apply).x as f64).abs() < 0.5);
    assert!((r.width() - ui.rect(apply).width as f64).abs() < 0.5);

    // Focus and presses come back.
    mirror.focus(apply);
    let active = web_sys::window().unwrap().document().unwrap().active_element().unwrap();
    assert_eq!(&active, mirror.element(apply).unwrap());
    mirror.element(wire).unwrap().dyn_ref::<HtmlElement>().unwrap().click();
    let key = |k: &str| web_sys::KeyboardEvent::new_with_keyboard_event_init_dict(
        "keydown",
        &{ let i = web_sys::KeyboardEventInit::new(); i.set_key(k); i.set_bubbles(true); i.set_cancelable(true); i },
    ).unwrap();
    mirror.element(slider).unwrap().dispatch_event(&key("ArrowRight")).unwrap();
    mirror.element(apply).unwrap().dispatch_event(&key("Enter")).unwrap();
    assert_eq!(
        mirror.take_actions(),
        vec![A11yAction::Focus(apply), A11yAction::Activate(wire), A11yAction::Increment(slider), A11yAction::Activate(apply)]
    );

    // Unticked: the same element, its state changed. Removed: gone.
    let element = mirror.element(wire).unwrap().clone();
    ui.semantics_mut(wire).checked = Some(false);
    ui.remove(note);
    mirror.update(&ui);
    assert_eq!(mirror.element(wire), Some(&element));
    assert_eq!(element.get_attribute("aria-checked").as_deref(), Some("false"));
    assert!(mirror.element(ready_text).is_none());
    assert!(!ready.is_connected());
}
