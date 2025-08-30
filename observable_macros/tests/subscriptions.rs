use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

#[test]
fn subscribe_to_derived_field_notifications() {
    #[derive(observable_macros::Observable)]
    struct Counter {
        pub count: usize,
    }

    let raw = Counter { count: 0 };
    let obs = raw.into_observable();

    let notified = Arc::new(AtomicUsize::new(0));
    let n = notified.clone();

    let id = obs.count.subscribe(Arc::new(move |_old, new| {
        // increment notification counter and assert new value
        n.fetch_add(1, Ordering::SeqCst);
        assert_eq!(*new, 1usize);
    })).expect("subscribe failed");

    // trigger notification
    obs.count.set(1).expect("set failed");

    assert_eq!(notified.load(Ordering::SeqCst), 1);

    // unsubscribe should succeed
    obs.count.unsubscribe(id).expect("unsubscribe failed");
}

#[test]
fn subscribe_to_attribute_generated_field_notifications() {
    #[observable_macros::observable]
    struct Item {
        pub value: i32,
    }

    let it = Item::new(0i32);

    let notified = Arc::new(AtomicUsize::new(0));
    let n = notified.clone();

    let id = it.value.subscribe(Arc::new(move |_old, new| {
        n.fetch_add(1, Ordering::SeqCst);
        assert_eq!(*new, 42i32);
    })).expect("subscribe failed");

    it.value.set(42).expect("set failed");

    assert_eq!(notified.load(Ordering::SeqCst), 1);

    it.value.unsubscribe(id).expect("unsubscribe failed");
}

