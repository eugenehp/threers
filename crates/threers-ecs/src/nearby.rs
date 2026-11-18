//! A [`Link`] to the same game on a device nearby, through Apple's
//! MultipeerConnectivity: iPads, iPhones and Macs find each other over
//! Wi-Fi, peer-to-peer Wi-Fi or Bluetooth, with no network, server or
//! address to type. Every device advertises the game and browses for it;
//! when two find each other, the one whose name sorts first invites the
//! other, which accepts. Messages then go to every device joined.
//!
//! ```no_run
//! use threers_ecs::nearby::Nearby;
//! use threers_ecs::share::Link;
//! let mut link = Nearby::start("my-game", "Pilot").unwrap();
//! link.send("hello");
//! for message in link.receive() { /* … */ }
//! ```
//!
//! Its news comes on the main run loop, which a window's event loop runs;
//! without a window, run it by hand (`examples/nearby.rs` does).
//!
//! An iOS app needs two `Info.plist` keys for it: `NSLocalNetworkUsageDescription`
//! (why, shown when asked), and `NSBonjourServices` listing
//! `_<service>._tcp` and `_<service>._udp`. Elsewhere than Apple's
//! platforms [`Nearby::start`] gives `None`.

#[cfg(target_vendor = "apple")]
pub use apple::Nearby;

/// Where there is no MultipeerConnectivity: never starts.
#[cfg(not(target_vendor = "apple"))]
pub struct Nearby;

#[cfg(not(target_vendor = "apple"))]
impl Nearby {
    pub fn start(_service: &str, _name: &str) -> Option<Nearby> {
        None
    }
}

#[cfg(not(target_vendor = "apple"))]
impl crate::share::Link for Nearby {
    fn send(&mut self, _message: &str) {}
    fn receive(&mut self) -> Vec<String> {
        Vec::new()
    }
}

/// Whether `service` is one MultipeerConnectivity takes: 1 to 15 lowercase
/// ASCII letters, digits and hyphens, not starting or ending with a hyphen.
pub fn valid_service(service: &str) -> bool {
    (1..=15).contains(&service.len())
        && service
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !service.starts_with('-')
        && !service.ends_with('-')
}

#[cfg(target_vendor = "apple")]
mod apple {
    use crate::share::Link;
    use block2::DynBlock;
    use objc2::rc::{Allocated, Retained};
    use objc2::runtime::{AnyClass, AnyObject, Bool, NSObject, NSObjectProtocol};
    use objc2::{AnyThread, DefinedClass, define_class, msg_send};
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::sync::{Arc, Mutex};

    #[link(name = "MultipeerConnectivity", kind = "framework")]
    unsafe extern "C" {}
    #[link(name = "Foundation", kind = "framework")]
    unsafe extern "C" {}

    /// `NSUTF8StringEncoding`
    const UTF8: usize = 4;
    /// `MCSessionSendDataReliable`: in order, every one.
    const RELIABLE: isize = 0;
    /// `MCSessionSendDataUnreliable`: may be lost or overtaken.
    const UNRELIABLE: isize = 1;
    /// `MCEncryptionRequired`
    const ENCRYPTED: isize = 1;
    /// `MCSessionStateConnected`
    const CONNECTED: isize = 2;
    /// Seconds an invitation waits for an answer.
    const INVITE_SECONDS: f64 = 20.0;

    /// What the delegate hears, from MultipeerConnectivity's own queues.
    #[derive(Default)]
    struct Heard {
        inbox: VecDeque<String>,
        /// Names of the devices joined.
        peers: Vec<String>,
    }

    struct Ivars {
        heard: Arc<Mutex<Heard>>,
        /// The `MCSession`, to hand an invitation; set once made.
        session: Mutex<Option<Retained<AnyObject>>>,
        /// This device's name, to decide who invites.
        name: String,
    }

    fn string(text: &str) -> Retained<AnyObject> {
        let class = AnyClass::get(c"NSString").expect("Foundation");
        unsafe {
            let made: Allocated<AnyObject> = msg_send![class, alloc];
            let made: Option<Retained<AnyObject>> = msg_send![
                made,
                initWithBytes: text.as_ptr() as *const c_void,
                length: text.len(),
                encoding: UTF8
            ];
            made.expect("NSString")
        }
    }

    fn rust_string(ns: &AnyObject) -> String {
        unsafe {
            let utf8: *const std::ffi::c_char = msg_send![ns, UTF8String];
            if utf8.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(utf8).to_string_lossy().into_owned()
        }
    }

    fn peer_name(peer: &AnyObject) -> String {
        let name: Option<Retained<AnyObject>> = unsafe { msg_send![peer, displayName] };
        name.map(|n| rust_string(&n)).unwrap_or_default()
    }

    define_class!(
        // SAFETY: NSObject has no subclassing requirements; no Drop.
        #[unsafe(super(NSObject))]
        #[ivars = Ivars]
        struct Delegate;

        unsafe impl NSObjectProtocol for Delegate {}

        // MCSessionDelegate
        impl Delegate {
            #[unsafe(method(session:peer:didChangeState:))]
            fn changed(&self, _session: &AnyObject, peer: &AnyObject, state: isize) {
                let name = peer_name(peer);
                if let Ok(mut heard) = self.ivars().heard.lock() {
                    heard.peers.retain(|p| *p != name);
                    if state == CONNECTED {
                        heard.peers.push(name);
                    }
                }
            }

            #[unsafe(method(session:didReceiveData:fromPeer:))]
            fn received(&self, _session: &AnyObject, data: &AnyObject, _peer: &AnyObject) {
                let text = unsafe {
                    let bytes: *const c_void = msg_send![data, bytes];
                    let length: usize = msg_send![data, length];
                    if bytes.is_null() {
                        return;
                    }
                    String::from_utf8_lossy(std::slice::from_raw_parts(bytes as *const u8, length)).into_owned()
                };
                if let Ok(mut heard) = self.ivars().heard.lock() {
                    heard.inbox.push_back(text);
                }
            }

            #[unsafe(method(session:didReceiveStream:withName:fromPeer:))]
            fn stream(&self, _s: &AnyObject, _stream: &AnyObject, _name: &AnyObject, _peer: &AnyObject) {}

            #[unsafe(method(session:didStartReceivingResourceWithName:fromPeer:withProgress:))]
            fn resource_started(&self, _s: &AnyObject, _name: &AnyObject, _peer: &AnyObject, _p: &AnyObject) {}

            #[unsafe(method(session:didFinishReceivingResourceWithName:fromPeer:atURL:withError:))]
            fn resource_finished(
                &self,
                _s: &AnyObject,
                _name: &AnyObject,
                _peer: &AnyObject,
                _url: *mut AnyObject,
                _error: *mut AnyObject,
            ) {
            }
        }

        // MCNearbyServiceAdvertiserDelegate
        impl Delegate {
            #[unsafe(method(advertiser:didReceiveInvitationFromPeer:withContext:invitationHandler:))]
            fn invited(
                &self,
                _advertiser: &AnyObject,
                _peer: &AnyObject,
                _context: *mut AnyObject,
                handler: &DynBlock<dyn Fn(Bool, *mut AnyObject)>,
            ) {
                let session = self.ivars().session.lock().ok().and_then(|s| s.clone());
                match session {
                    Some(session) => handler.call((Bool::YES, Retained::as_ptr(&session) as *mut AnyObject)),
                    None => handler.call((Bool::NO, std::ptr::null_mut())),
                }
            }
        }

        // MCNearbyServiceBrowserDelegate
        impl Delegate {
            #[unsafe(method(browser:foundPeer:withDiscoveryInfo:))]
            fn found(&self, browser: &AnyObject, peer: &AnyObject, _info: *mut AnyObject) {
                // One invites, the other waits to be invited: not both.
                if self.ivars().name.as_str() >= peer_name(peer).as_str() {
                    return;
                }
                let Some(session) = self.ivars().session.lock().ok().and_then(|s| s.clone()) else {
                    return;
                };
                let none: *mut AnyObject = std::ptr::null_mut();
                unsafe {
                    let _: () = msg_send![
                        browser,
                        invitePeer: peer,
                        toSession: &*session,
                        withContext: none,
                        timeout: INVITE_SECONDS
                    ];
                }
            }

            #[unsafe(method(browser:lostPeer:))]
            fn lost(&self, _browser: &AnyObject, _peer: &AnyObject) {}
        }
    );

    /// Joined to the same game's other devices nearby. Dropping it leaves.
    pub struct Nearby {
        session: Retained<AnyObject>,
        advertiser: Retained<AnyObject>,
        browser: Retained<AnyObject>,
        _delegate: Retained<Delegate>,
        heard: Arc<Mutex<Heard>>,
    }

    impl Nearby {
        /// Advertise `service` (see [`valid_service`](super::valid_service))
        /// as `name`, and join every device nearby doing the same. Two devices
        /// should not share a name; a short random tag is added to it.
        /// `None` if MultipeerConnectivity is missing or `service` is not one
        /// it takes.
        pub fn start(service: &str, name: &str) -> Option<Nearby> {
            if !super::valid_service(service) {
                return None;
            }
            let tag = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| (d.subsec_micros() + std::process::id()) % 10_000)
                .unwrap_or(0);
            // A display name is at most 63 bytes of UTF-8.
            let mut name: String = name.chars().take(40).collect();
            name.push_str(&format!(" {tag:04}"));

            let peer_class = AnyClass::get(c"MCPeerID")?;
            let session_class = AnyClass::get(c"MCSession")?;
            let advertiser_class = AnyClass::get(c"MCNearbyServiceAdvertiser")?;
            let browser_class = AnyClass::get(c"MCNearbyServiceBrowser")?;
            let heard = Arc::new(Mutex::new(Heard::default()));
            let delegate = Delegate::alloc().set_ivars(Ivars {
                heard: heard.clone(),
                session: Mutex::new(None),
                name: name.clone(),
            });
            let delegate: Retained<Delegate> = unsafe { msg_send![super(delegate), init] };
            let none: *mut AnyObject = std::ptr::null_mut();
            unsafe {
                let peer: Allocated<AnyObject> = msg_send![peer_class, alloc];
                let peer: Option<Retained<AnyObject>> = msg_send![peer, initWithDisplayName: &*string(&name)];
                let peer = peer?;
                let session: Allocated<AnyObject> = msg_send![session_class, alloc];
                let session: Option<Retained<AnyObject>> = msg_send![
                    session,
                    initWithPeer: &*peer,
                    securityIdentity: none,
                    encryptionPreference: ENCRYPTED
                ];
                let session = session?;
                let _: () = msg_send![&*session, setDelegate: &*delegate];
                *delegate.ivars().session.lock().ok()? = Some(session.clone());

                let service = string(service);
                let advertiser: Allocated<AnyObject> = msg_send![advertiser_class, alloc];
                let advertiser: Option<Retained<AnyObject>> = msg_send![
                    advertiser,
                    initWithPeer: &*peer,
                    discoveryInfo: none,
                    serviceType: &*service
                ];
                let advertiser = advertiser?;
                let _: () = msg_send![&*advertiser, setDelegate: &*delegate];
                let _: () = msg_send![&*advertiser, startAdvertisingPeer];

                let browser: Allocated<AnyObject> = msg_send![browser_class, alloc];
                let browser: Option<Retained<AnyObject>> =
                    msg_send![browser, initWithPeer: &*peer, serviceType: &*service];
                let browser = browser?;
                let _: () = msg_send![&*browser, setDelegate: &*delegate];
                let _: () = msg_send![&*browser, startBrowsingForPeers];

                Some(Nearby {
                    session,
                    advertiser,
                    browser,
                    _delegate: delegate,
                    heard,
                })
            }
        }

    }

    impl Nearby {
        fn send_with(&mut self, message: &str, mode: isize) {
            unsafe {
                let peers: Retained<AnyObject> = msg_send![&*self.session, connectedPeers];
                let count: usize = msg_send![&*peers, count];
                if count == 0 {
                    return;
                }
                let data_class = AnyClass::get(c"NSData").expect("Foundation");
                let data: Retained<AnyObject> = msg_send![
                    data_class,
                    dataWithBytes: message.as_ptr() as *const c_void,
                    length: message.len()
                ];
                let mut error: *mut AnyObject = std::ptr::null_mut();
                let _: Bool = msg_send![
                    &*self.session,
                    sendData: &*data,
                    toPeers: &*peers,
                    withMode: mode,
                    error: &mut error
                ];
            }
        }
    }

    impl Link for Nearby {
        fn send(&mut self, message: &str) {
            self.send_with(message, RELIABLE);
        }

        fn send_lossy(&mut self, message: &str) {
            self.send_with(message, UNRELIABLE);
        }

        fn receive(&mut self) -> Vec<String> {
            self.heard
                .lock()
                .map(|mut h| h.inbox.drain(..).collect())
                .unwrap_or_default()
        }

        /// The names of the devices joined now.
        fn peers(&self) -> Vec<String> {
            self.heard.lock().map(|h| h.peers.clone()).unwrap_or_default()
        }
    }

    impl Drop for Nearby {
        fn drop(&mut self) {
            unsafe {
                let _: () = msg_send![&*self.advertiser, stopAdvertisingPeer];
                let _: () = msg_send![&*self.browser, stopBrowsingForPeers];
                let _: () = msg_send![&*self.session, disconnect];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_names_follow_apple_rules() {
        assert!(valid_service("spacearm"));
        assert!(valid_service("my-game-2"));
        assert!(!valid_service(""));
        assert!(!valid_service("Space"));
        assert!(!valid_service("-lead"));
        assert!(!valid_service("much-too-long-name"));
    }
}
