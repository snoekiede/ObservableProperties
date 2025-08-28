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
    
    // Continue with other work while observers run in background
    println!("This prints immediately!");
    
    Ok(())
}
```

## Error Handling

The library uses a comprehensive error system for robust error handling:

```rust
use observable_property::{ObservableProperty, PropertyError};
use std::sync::Arc;

fn example() -> Result<(), PropertyError> {
    let property = ObservableProperty::new(42);
    
    match property.subscribe(Arc::new(|old, new| {
        println!("Value: {} -> {}", old, new);
    })) {
        Ok(observer_id) => {
            // Successfully subscribed
            property.set(100)?;
            property.unsubscribe(observer_id)?;
        }
        Err(PropertyError::PoisonedLock) => {
            eprintln!("Property lock was poisoned by a panic in another thread");
        }
        Err(PropertyError::WriteLockError { context }) => {
            eprintln!("Failed to acquire write lock: {}", context);
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }
    
    Ok(())
}
```

## Performance Considerations

- **Observers**: Each observer is called sequentially. For heavy computations, use `set_async()` to run observers in background threads.
- **Lock contention**: The property uses a single `RwLock` internally. Consider having fewer, larger properties rather than many small ones.
- **Memory**: Observer functions are stored as `Arc<dyn Fn>` and kept until unsubscribed.

## Thread Safety

All operations are thread-safe:
- Multiple threads can read the property value simultaneously
- Only one thread can modify the property at a time
- Observer notifications happen outside the lock to minimize contention
- Observer panics are isolated and don't affect other observers or the property

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
