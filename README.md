# Observable Property

A thread-safe observable property implementation for Rust that allows you to observe changes to values across multiple threads.

## Features

* **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access

* **Observer pattern**: Subscribe to property changes with callbacks

* **Filtered observers**: Only notify when specific conditions are met

* **Async notifications**: Non-blocking observer notifications with background threads

* **Panic isolation**: Observer panics don't crash the system

* **Type-safe**: Generic implementation; most methods only need `Clone`, async paths need `Send + 'static`

* **Zero mandatory dependencies**: Standard library only; optional `logging` feature uses the `log` crate

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
observable-property = "0.1.0"
```

Optional logging support (feature-flagged):

```toml
[dependencies]
observable-property = { version = "0.1.0", features = ["logging"] }

[dependencies.env_logger]
version = "0.11" # or latest
```

Then initialize a logger in your binary (tests/examples often use `env_logger`):

```rust
env_logger::init();
```

## Usage

### Basic Example

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

// Create an observable property
let property = ObservableProperty::new(42);

// Subscribe to changes
let observer_id = property.subscribe(Arc::new(|old_value, new_value| {
    println!("Value changed from {} to {}", old_value, new_value);
})).unwrap();

// Change the value (triggers observer)
property.set(100).unwrap(); // Prints: Value changed from 42 to 100

// Unsubscribe when done
property.unsubscribe(observer_id).unwrap();
```

### Multi-threading Example

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;

let property = Arc::new(ObservableProperty::new(0));
let property_clone = property.clone();

// Subscribe from one thread
property.subscribe(Arc::new(|old, new| {
    println!("Value changed: {} -> {}", old, new);
})).unwrap();

// Modify from another thread
thread::spawn(move || {
    property_clone.set(42).unwrap();
}).join().unwrap();
```

### Filtered Observers

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

let counter = ObservableProperty::new(0);

// Only notify when value increases
let observer_id = counter.subscribe_filtered(
    Arc::new(|old, new| println!("Increased: {} -> {}", old, new)),
    |old, new| new > old
).unwrap();

counter.set(5).unwrap();  // Triggers observer: "Increased: 0 -> 5"
counter.set(3).unwrap();  // Does NOT trigger observer
counter.set(10).unwrap(); // Triggers observer: "Increased: 3 -> 10"
```

### Immediate Subscription

Invoke an observer right away with the current value (useful for UI/state init):

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

let prop = ObservableProperty::new(String::from("ready"));

prop.subscribe_immediate(Arc::new(|old, new| {
    // old == new == "ready" on first call
    println!("initial: {old} / {new}");
})).unwrap();
```

### Async Notifications

For observers that might perform time-consuming operations, use async notifications to avoid blocking:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::time::Duration;

let property = ObservableProperty::new(0);

property.subscribe(Arc::new(|old, new| {
    // This slow observer won't block the caller
    std::thread::sleep(Duration::from_millis(100));
    println!("Slow observer: {} -> {}", old, new);
})).unwrap();

// This returns immediately even though observer is slow
property.set_async(42).unwrap();
```

### Change Utilities

Update atomically from the current value and notify once:

```rust
let prop = ObservableProperty::new(10);
let (old, new) = prop.update(|v| v + 5).unwrap();
assert_eq!((old, new), (10, 15));
```

Avoid redundant notifications when the value is unchanged:

```rust
let updated = prop.set_if_changed(15).unwrap();
assert!(!updated); // value was already 15
```

### Lock Helpers

Run closures under locks without cloning the whole value:

```rust
let len = prop.with_read(|s: &String| s.len()).unwrap();

prop.with_write(|v: &mut String| v.push_str("!"))
    .unwrap();
```

### Observer Management

```rust
let count = prop.observer_count().unwrap();
let removed = prop.clear_observers().unwrap();
assert_eq!(removed, count);
```

### Complex Types

Observable properties work with any type that implements the required traits:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

#[derive(Clone, Debug)]
struct Person {
    name: String,
    age: u32,
}

let person_property = ObservableProperty::new(Person {
    name: "Alice".to_string(),
    age: 30,
});

person_property.subscribe(Arc::new(|old_person, new_person| {
    println!("Person changed: {:?} -> {:?}", old_person, new_person);
})).unwrap();

person_property.set(Person {
    name: "Alice".to_string(),
    age: 31,
}).unwrap();
```

## Error Handling

All operations return `Result` types with descriptive errors:

```rust
use observable_property::{ObservableProperty, PropertyError};

let property = ObservableProperty::new(42);

match property.get() {
    Ok(value) => println!("Current value: {}", value),
    Err(PropertyError::PoisonedLock) => println!("Lock was poisoned!"),
    Err(e) => println!("Other error: {}", e),
}
```

## Performance Considerations

* **Read operations** are very fast and can be performed concurrently from multiple threads

* **Write operations** are serialized but optimize for quick lock release

* **Synchronous notifications** block the setter until all observers complete

* **Asynchronous notifications** return immediately and run observers in background threads

* **Observer panics** are isolated and won't affect other observers or crash the system

* **Logging** of observer panics is available when enabling the `logging` feature (uses the `log` facade)

## Examples

Run the included examples to see more usage patterns:

```bash
# Basic usage example
cargo run --example basic

# Multithreaded usage with performance comparisons
cargo run --example multithreaded

# Logging example (enable feature)
cargo run --example logging --features logging
```

## Safety

This crate is designed with safety as a primary concern:

* Thread-safe access patterns prevent data races

* Observer panics are caught and isolated

* Lock poisoning is properly handled and reported

* No unsafe code is used

## API overview

| Area | Method | Notes |
|------|--------|-------|
| Read | `get()` | Clones current value |
| Read | `with_read(f)` | Borrowed read without cloning |
| Write | `set(v)` | Sync notify |
| Write | `set_async(v)` | Async notify in batches |
| Write | `update(f)` | Compute-and-set under one lock, notify once |
| Write | `set_if_changed(v)` | No-op if equal (PartialEq) |
| Obs | `subscribe(cb)` | Returns ObserverId |
| Obs | `subscribe_filtered(cb, pred)` | Predicate-gated |
| Obs | `subscribe_immediate(cb)` | Calls once immediately |
| Obs | `unsubscribe(id)` | Returns bool |
| Obs | `observer_count()` | Count observers |
| Obs | `clear_observers()` | Remove all observers |

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](https://www.google.com/search?q=LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

* MIT license ([LICENSE-MIT](https://www.google.com/search?q=LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
