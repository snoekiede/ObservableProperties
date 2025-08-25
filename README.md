# Observable Property

A thread-safe observable property implementation for Rust that allows you to observe changes to values across multiple threads.

## Features

* **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access

* **Observer pattern**: Subscribe to property changes with callbacks

* **Filtered observers**: Only notify when specific conditions are met

* **Async notifications**: Non-blocking observer notifications with background threads

* **Panic isolation**: Observer panics don't crash the system

* **Type-safe**: Generic implementation works with any `Clone + Send + Sync + 'static` type

* **Zero dependencies**: Uses only Rust standard library

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
observable-property = "0.2.1"
```

## Usage

### Basic Example

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    // Create an observable property
    let property = ObservableProperty::new(42);

    // Subscribe to changes
    let observer_id = property.subscribe(Arc::new(|old_value, new_value| {
        println!("Value changed from {} to {}", old_value, new_value);
    })).map_err(|e| {
        eprintln!("Failed to subscribe: {}", e);
        e
    })?;

    // Change the value (triggers observer)
    property.set(100).map_err(|e| {
        eprintln!("Failed to set value: {}", e);
        e
    })?; // Prints: Value changed from 42 to 100

    // Unsubscribe when done
    property.unsubscribe(observer_id).map_err(|e| {
        eprintln!("Failed to unsubscribe: {}", e);
        e
    })?;
    
    Ok(())
}
```

### Multi-threading Example

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;

fn main() -> Result<(), observable_property::PropertyError> {
    let property = Arc::new(ObservableProperty::new(0));
    let property_clone = property.clone();

    // Subscribe from one thread
    property.subscribe(Arc::new(|old, new| {
        println!("Value changed: {} -> {}", old, new);
    })).map_err(|e| {
        eprintln!("Failed to subscribe: {}", e);
        e
    })?;

    // Modify from another thread
    thread::spawn(move || {
        if let Err(e) = property_clone.set(42) {
            eprintln!("Failed to set value: {}", e);
        }
    }).join().expect("Thread panicked");
    
    Ok(())
}
```

### Filtered Observers

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    let counter = ObservableProperty::new(0);

    // Only notify when value increases
    let observer_id = counter.subscribe_filtered(
        Arc::new(|old, new| println!("Increased: {} -> {}", old, new)),
        |old, new| new > old
    ).map_err(|e| {
        eprintln!("Failed to subscribe: {}", e);
        e
    })?;

    counter.set(5).map_err(|e| {
        eprintln!("Failed to set value: {}", e);
        e
    })?;  // Triggers observer: "Increased: 0 -> 5"
    
    counter.set(3).map_err(|e| {
        eprintln!("Failed to set value: {}", e);
        e
    })?;  // Does NOT trigger observer
    
    counter.set(10).map_err(|e| {
        eprintln!("Failed to set value: {}", e);
        e
    })?; // Triggers observer: "Increased: 3 -> 10"
    
    Ok(())
}
```

### Async Notifications

For observers that might perform time-consuming operations, use async notifications to avoid blocking:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    let property = ObservableProperty::new(0);

    property.subscribe(Arc::new(|old, new| {
        // This slow observer won't block the caller
        std::thread::sleep(Duration::from_millis(100));
        println!("Slow observer: {} -> {}", old, new);
    })).map_err(|e| {
        eprintln!("Failed to subscribe: {}", e);
        e
    })?;

    // This returns immediately even though observer is slow
    property.set_async(42).map_err(|e| {
        eprintln!("Failed to set value asynchronously: {}", e);
        e
    })?;
    
    Ok(())
}
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

fn main() -> Result<(), observable_property::PropertyError> {
    let person_property = ObservableProperty::new(Person {
        name: "Alice".to_string(),
        age: 30,
    });

    person_property.subscribe(Arc::new(|old_person, new_person| {
        println!("Person changed: {:?} -> {:?}", old_person, new_person);
    })).map_err(|e| {
        eprintln!("Failed to subscribe: {}", e);
        e
    })?;

    person_property.set(Person {
        name: "Alice".to_string(),
        age: 31,
    }).map_err(|e| {
        eprintln!("Failed to set person: {}", e);
        e
    })?;
    
    Ok(())
}
```

## Error Handling

All operations return `Result` types with descriptive errors:

```rust
use observable_property::{ObservableProperty, PropertyError};

fn main() -> Result<(), PropertyError> {
    let property = ObservableProperty::new(42);

    match property.get() {
        Ok(value) => println!("Current value: {}", value),
        Err(PropertyError::PoisonedLock) => println!("Lock was poisoned!"),
        Err(e) => println!("Other error: {}", e),
    }

    Ok(())
}
```

## Performance Considerations

* **Read operations** are very fast and can be performed concurrently from multiple threads

* **Write operations** are serialized but optimize for quick lock release

* **Synchronous notifications** block the setter until all observers complete

* **Asynchronous notifications** return immediately and run observers in background threads

* **Observer panics** are isolated and won't affect other observers or crash the system

## Examples

Run the included examples to see more usage patterns:

```bash
# Basic usage example
cargo run --example basic

# Multithreaded usage with performance comparisons
cargo run --example multithreaded
```

## Safety

This crate is designed with safety as a primary concern:

* Thread-safe access patterns prevent data races

* Observer panics are caught and isolated

* Lock poisoning is properly handled and reported

* No unsafe code is used

## Disclaimer

This crate is provided "as is", without warranty of any kind, express or implied. The authors and contributors are not responsible for any damages or liability arising from the use of this software. While efforts have been made to ensure the crate functions correctly, it may contain bugs or issues in certain scenarios. Users should thoroughly test the crate in their specific environment before deploying to production.

Performance characteristics may vary depending on system configuration, observer complexity, and concurrency patterns. The observer pattern implementation may introduce overhead in systems with very high frequency property changes or large numbers of observers.

By using this crate, you acknowledge that you have read and understood this disclaimer.

## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](https://www.google.com/search?q=LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)

* MIT license ([LICENSE-MIT](https://www.google.com/search?q=LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
