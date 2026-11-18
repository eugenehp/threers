//! Two of these near each other (or two on one Mac) find each other and
//! trade messages: `cargo run -p threers-ecs --features nearby --example nearby -- NAME`.

#[cfg(feature = "nearby")]
fn main() {
    use threers_ecs::nearby::Nearby;
    use threers_ecs::share::Link;
    let name = std::env::args().nth(1).unwrap_or_else(|| "device".into());
    let mut link = Nearby::start("threers-demo", &name).expect("MultipeerConnectivity");
    for i in 0..40 {
        link.send(&format!("{name} says {i}"));
        for message in link.receive() {
            println!("{name} heard: {message}  (joined: {:?})", link.peers());
        }
        // A window's event loop runs the main run loop; here, run it by hand.
        unsafe { CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, 0) };
    }
}

#[cfg(feature = "nearby")]
#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFRunLoopDefaultMode: *const std::ffi::c_void;
    fn CFRunLoopRunInMode(mode: *const std::ffi::c_void, seconds: f64, once: u8) -> i32;
}

#[cfg(not(feature = "nearby"))]
fn main() {
    eprintln!("needs --features nearby");
}
