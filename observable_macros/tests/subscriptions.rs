use std::sync::{Arc, atomic::{AtomicUsize, Ordering}, Mutex};
use std::thread;
use std::time::{Duration, Instant};

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

#[test]
fn test_set_async_behavior() {
    #[observable_macros::observable]
    struct AsyncItem {
        pub value: i32,
    }

    let it = AsyncItem::new(0);

    let notified = Arc::new(AtomicUsize::new(0));
    let n1 = notified.clone();
    let n2 = notified.clone();

    // slow observer
    let _id1 = it.value.subscribe(Arc::new(move |_old, _new| {
        thread::sleep(Duration::from_millis(50));
        n1.fetch_add(1, Ordering::SeqCst);
    })).expect("subscribe failed");

    // fast observer
    let _id2 = it.value.subscribe(Arc::new(move |_old, _new| {
        n2.fetch_add(1, Ordering::SeqCst);
    })).expect("subscribe failed");

    // Call set_async and ensure it returns quickly
    let start = Instant::now();
    it.value.set_async(7).expect("set_async failed");
    let duration = start.elapsed();
    assert!(duration < Duration::from_millis(20), "set_async took too long: {:?}", duration);

    // Wait for observers to finish with timeout
    let wait_start = Instant::now();
    while notified.load(Ordering::SeqCst) < 2 && wait_start.elapsed() < Duration::from_millis(500) {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(notified.load(Ordering::SeqCst), 2);
}

#[test]
fn test_observer_panic_isolated() {
    #[derive(observable_macros::Observable)]
    struct Punctured {
        pub v: i32,
    }

    let raw = Punctured { v: 0 };
    let obs = raw.into_observable();

    let counter = Arc::new(AtomicUsize::new(0));
    let c = counter.clone();

    // observer that panics
    let _p = obs.v.subscribe(Arc::new(move |_old, _new| {
        panic!("observer panic for test");
    })).expect("subscribe failed");

    // observer that increments
    let _ok = obs.v.subscribe(Arc::new(move |_old, _new| {
        c.fetch_add(1, Ordering::SeqCst);
    })).expect("subscribe failed");

    // This should not panic; the panic inside observer is caught
    obs.v.set(1).expect("set failed");
    assert_eq!(counter.load(Ordering::SeqCst), 1);

    // Test set_async also isolates panics
    let counter2 = Arc::new(AtomicUsize::new(0));
    let c2 = counter2.clone();
    let _ok2 = obs.v.subscribe(Arc::new(move |_old, _new| {
        c2.fetch_add(1, Ordering::SeqCst);
    })).expect("subscribe failed");

    obs.v.set_async(2).expect("set_async failed");
    // wait for async observers
    let wait_start = Instant::now();
    while counter2.load(Ordering::SeqCst) < 1 && wait_start.elapsed() < Duration::from_millis(500) {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(counter2.load(Ordering::SeqCst), 1);
}

#[test]
fn test_concurrent_subscribe_unsubscribe() {
    #[observable_macros::observable]
    struct Crowd {
        pub field: usize,
    }

    let p = Crowd::new(0usize);

    let subscribe_count = 50usize;
    let ids = Arc::new(Mutex::new(Vec::with_capacity(subscribe_count)));
    let notified = Arc::new(AtomicUsize::new(0));

    // Spawn threads to subscribe concurrently
    let mut handles = Vec::new();
    for _ in 0..subscribe_count {
        let p_clone = p.clone();
        let ids_clone = ids.clone();
        let notified_clone = notified.clone();
        handles.push(thread::spawn(move || {
            let id = p_clone.field.subscribe(Arc::new(move |_old, _new| {
                notified_clone.fetch_add(1, Ordering::SeqCst);
            })).expect("subscribe failed");
            ids_clone.lock().unwrap().push(id);
        }));
    }

    for h in handles { h.join().expect("subscribe thread failed"); }

    // Trigger notifications
    p.field.set(1).expect("set failed");
    assert_eq!(notified.load(Ordering::SeqCst), subscribe_count);

    // Unsubscribe concurrently
    let mut handles = Vec::new();
    for id in ids.lock().unwrap().iter().cloned() {
        let p_clone = p.clone();
        handles.push(thread::spawn(move || {
            p_clone.field.unsubscribe(id).expect("unsubscribe failed");
        }));
    }
    for h in handles { h.join().expect("unsubscribe thread failed"); }

    // Reset counter and set again should not notify
    notified.store(0, Ordering::SeqCst);
    p.field.set(2).expect("set failed");
    assert_eq!(notified.load(Ordering::SeqCst), 0);
}

#[test]
fn test_unsubscribe_edge_cases() {
    #[observable_macros::observable]
    struct E {
        pub v: i32,
    }

    let e = E::new(0);

    // Unsubscribe non-existing id
    let res = e.v.unsubscribe(9999).expect("unsubscribe returned error");
    assert!(!res);

    // Subscribe and unsubscribe twice
    let id = e.v.subscribe(Arc::new(|_, _| {})).expect("subscribe failed");
    let first = e.v.unsubscribe(id).expect("unsubscribe failed");
    assert!(first);
    let second = e.v.unsubscribe(id).expect("unsubscribe failed");
    assert!(!second);
}

#[test]
fn test_many_concurrent_set_async() {
    #[observable_macros::observable]
    struct Big {
        pub v: i32,
    }

    let b = Big::new(0);

    let observers = 5usize;
    let threads = 20usize;
    let calls_per_thread = 10usize;
    let expected_notifications = observers * threads * calls_per_thread;

    let notified = Arc::new(AtomicUsize::new(0));
    for _ in 0..observers {
        let n = notified.clone();
        b.v.subscribe(Arc::new(move |_old, _new| {
            n.fetch_add(1, Ordering::SeqCst);
        })).expect("subscribe failed");
    }

    let mut handles = Vec::new();
    for t in 0..threads {
        let b_clone = b.clone();
        handles.push(thread::spawn(move || {
            for i in 0..calls_per_thread {
                let _ = b_clone.v.set_async((t * calls_per_thread + i) as i32);
            }
        }));
    }

    for h in handles { h.join().expect("thread failed"); }

    // Wait until all notifications observed or timeout
    let start = Instant::now();
    while notified.load(Ordering::SeqCst) < expected_notifications && start.elapsed() < Duration::from_secs(5) {
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(notified.load(Ordering::SeqCst), expected_notifications);
}

#[test]
fn test_filtered_observers_with_async_sets() {
    #[observable_macros::observable]
    struct F {
        pub val: i32,
    }

    let f = F::new(0);

    let even_notified = Arc::new(AtomicUsize::new(0));
    let n_even = even_notified.clone();

    // Subscribe filtered observer: only notify when new is even and greater than old
    let _id = f.val.subscribe_filtered(
        Arc::new(move |_old, _new| {
            n_even.fetch_add(1, Ordering::SeqCst);
        }),
        |old, new| (new % 2 == 0) && (new > old)
    ).expect("subscribe_filtered failed");

    // Trigger a series of async sets (0->1->2->3->4)
    for v in 1..=4 {
        f.val.set_async(v).expect("set_async failed");
    }

    // Expect notifications for values 2 and 4 only
    let start = Instant::now();
    while even_notified.load(Ordering::SeqCst) < 2 && start.elapsed() < Duration::from_secs(2) {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(even_notified.load(Ordering::SeqCst), 2);
}

#[test]
fn test_stress_subscribe_unsubscribe_heavy_load() {
    #[observable_macros::observable]
    struct S {
        pub x: i32,
    }

    let s = S::new(0);
    let threads = 10usize;
    let iters = 200usize;

    let success_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..threads {
        let s_clone = s.clone();
        let success = success_count.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..iters {
                // subscribe
                let id = match s_clone.x.subscribe(Arc::new(|_o, _n| {})) {
                    Ok(id) => id,
                    Err(_) => continue,
                };
                // set
                let _ = s_clone.x.set(1);
                // unsubscribe
                let _ = s_clone.x.unsubscribe(id);
                success.fetch_add(1, Ordering::SeqCst);
            }
        }));
    }

    for h in handles { h.join().expect("thread failed"); }

    // Each successful loop increments success_count once
    assert_eq!(success_count.load(Ordering::SeqCst), threads * iters);
}
