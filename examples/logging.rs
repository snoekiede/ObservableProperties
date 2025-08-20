use observable_property::ObservableProperty;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

fn main() {
    // Initialize logging (enable the crate feature `logging` to see logs)
    env_logger::init();

    let prop = ObservableProperty::new(0);

    // Observer that panics once to demonstrate error logging
    let panics_once = {
        let called = Arc::new(AtomicBool::new(false));
        let called2 = called.clone();
        Arc::new(move |_: &i32, _: &i32| {
            if !called2.swap(true, Ordering::SeqCst) {
                panic!("simulated observer panic");
            }
        })
    };

    // Normal observer
    let normal = Arc::new(|old: &i32, new: &i32| println!("observed: {old} -> {new}"));

    prop.subscribe(panics_once).unwrap();
    prop.subscribe(normal).unwrap();

    // This will log an error from the panicking observer (when feature `logging` is enabled)
    prop.set(1).unwrap();
    prop.set(2).unwrap();
}
