# Observable Property

A thread-safe observable property implementation for Rust that allows you to observe changes to values across multiple threads. Built with comprehensive error handling and no `unwrap()` calls for maximum reliability.

## Features

* **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access
* **Observer pattern**: Subscribe to property changes with callbacks
* **RAII subscriptions**: Automatic cleanup with subscription guards (no manual unsubscribe needed)
* **Filtered observers**: Only notify when specific conditions are met
* **Async notifications**: Non-blocking observer notifications with background threads
* **Panic isolation**: Observer panics don't crash the system
* **Robust error handling**: Comprehensive error handling with descriptive error messages
* **Production-ready**: No `unwrap()` calls - all errors are handled gracefully
* **Type-safe**: Generic implementation works with any `Clone + Send + Sync + 'static` type
* **Zero dependencies**: Uses only Rust standard library

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
observable-property = "0.3.0"
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

### RAII Subscriptions (Recommended)

For automatic cleanup without manual unsubscribe calls, use RAII subscriptions:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    let property = ObservableProperty::new(0);

    {
        // Create RAII subscription - automatically cleaned up when dropped
        let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
            println!("Value changed: {} -> {}", old, new);
        })).map_err(|e| {
            eprintln!("Failed to create subscription: {}", e);
            e
        })?;

        property.set(42).map_err(|e| {
            eprintln!("Failed to set value: {}", e);
            e
        })?; // Prints: "Value changed: 0 -> 42"

        // Subscription automatically unsubscribes when leaving this scope
    }

    // No observer active anymore
    property.set(100).map_err(|e| {
        eprintln!("Failed to set value: {}", e);
        e
    })?; // No output

    Ok(())
}
```

### Filtered RAII Subscriptions

Combine filtering with automatic cleanup:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    let temperature = ObservableProperty::new(20.0);

    {
        // Monitor temperature increases > 5 degrees with automatic cleanup
        let _heat_warning = temperature.subscribe_filtered_with_subscription(
            Arc::new(|old, new| {
                println!("🔥 Heat warning! {:.1}°C -> {:.1}°C", old, new);
            }),
            |old, new| new > old && (new - old) > 5.0
        ).map_err(|e| {
            eprintln!("Failed to create heat warning subscription: {}", e);
            e
        })?;

        temperature.set(22.0).map_err(|e| {
            eprintln!("Failed to set temperature: {}", e);
            e
        })?; // No warning (only 2°C increase)
        
        temperature.set(28.0).map_err(|e| {
            eprintln!("Failed to set temperature: {}", e);
            e
        })?; // Triggers warning (6°C increase)

        // Subscription automatically cleaned up here
    }

    temperature.set(35.0).map_err(|e| {
        eprintln!("Failed to set temperature: {}", e);
        e
    })?; // No warning (subscription was cleaned up)

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

    let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
        // This slow observer won't block the caller
        std::thread::sleep(Duration::from_millis(100));
        println!("Slow observer: {} -> {}", old, new);
    })).map_err(|e| {
        eprintln!("Failed to create subscription: {}", e);
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

The library uses a comprehensive error system for robust, production-ready error handling. **All operations are designed to fail gracefully** with meaningful error messages - there are no `unwrap()` calls that can cause unexpected panics.

### Error Types

The library provides detailed error information through the `PropertyError` enum:

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

### Graceful Degradation

The library is designed to handle edge cases gracefully:

- **Poisoned locks**: When a thread panics while holding a lock, the property becomes "poisoned." All subsequent operations return clear error messages instead of panicking
- **Observer panics**: If an observer function panics, it's isolated - other observers continue to work normally
- **Thread safety**: All error conditions are thread-safe and don't cause data races or undefined behavior
- **Resource cleanup**: RAII subscriptions clean up properly even when locks are poisoned or other errors occur

```rust
// Even if a lock is poisoned, operations fail gracefully
match property.subscribe_with_subscription(observer) {
    Ok(_subscription) => println!("Successfully subscribed"),
    Err(PropertyError::PoisonedLock) => {
        // Handle gracefully - no panics, clear error message
        eprintln!("Property is in an invalid state due to a previous panic");
        // Can still safely continue program execution
    }
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Subscription Management

The library provides two approaches for managing observer subscriptions:

### Manual Management
```rust
let observer_id = property.subscribe(observer)?;
// ... use the property
property.unsubscribe(observer_id)?; // Manual cleanup required
```

### RAII Management (Recommended)
```rust
let _subscription = property.subscribe_with_subscription(observer)?;
// ... use the property
// Automatic cleanup when _subscription goes out of scope
```

**Benefits of RAII subscriptions:**
- ✅ No manual cleanup required
- ✅ Exception-safe (cleanup happens even if code panics)
- ✅ Works across thread boundaries
- ✅ Prevents observer leaks in complex control flow

## Performance Considerations

- **Observers**: Each observer is called sequentially. For heavy computations, use `set_async()` to run observers in background threads.
- **Lock contention**: The property uses a single `RwLock` internally. Consider having fewer, larger properties rather than many small ones.
- **Memory**: Observer functions are stored as `Arc<dyn Fn>` and kept until unsubscribed or subscription is dropped.
- **RAII overhead**: Subscription objects have minimal overhead (just an ID and Arc reference).

## Thread Safety

All operations are thread-safe with comprehensive error handling:
- Multiple threads can read the property value simultaneously
- Only one thread can modify the property at a time
- Observer notifications happen outside the lock to minimize contention
- Observer panics are isolated and don't affect other observers or the property
- RAII subscriptions can be created and dropped from any thread
- **Poisoned locks are handled gracefully** - subscriptions clean up without panicking
- **No `unwrap()` calls** - all potential failure points use proper error handling
- **Fail-safe design** - errors never cause undefined behavior or crashes

## Best Practices

### Use RAII Subscriptions
Prefer `subscribe_with_subscription()` and `subscribe_filtered_with_subscription()` over manual subscription management:

```rust
// ✅ Recommended: RAII subscription
let _subscription = property.subscribe_with_subscription(observer)?;
// Automatically cleaned up

// ❌ Discouraged: Manual management (error-prone)
let id = property.subscribe(observer)?;
property.unsubscribe(id)?; // Easy to forget or miss in error paths
```

### Scoped Subscriptions
Use block scoping for temporary subscriptions:

```rust
{
    let _temp_subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
        println!("Temporary monitoring: {} -> {}", old, new);
    }))?;
    
    // Do some work with monitoring active
    property.set(42)?;
    
    // Subscription automatically ends here
}
// No more monitoring
```

### Lightweight Observers
Keep observer functions lightweight for better performance:

```rust
// ✅ Good: Lightweight observer
let _subscription = property.subscribe_with_subscription(Arc::new(|_, new| {
    log::info!("Value updated to {}", new);
}))?;

// ❌ Avoid: Heavy computation in observer
let _subscription = property.subscribe_with_subscription(Arc::new(|_, new| {
    // This blocks all other observers!
    expensive_computation(*new);
}))?;

// ✅ Better: Use async for heavy work
property.set_async(new_value)?; // Non-blocking
```

### Comprehensive Error Handling
The library is production-ready with robust error handling. Always handle potential errors:

**Key benefits:**
- ✅ No `unwrap()` calls that could panic unexpectedly
- ✅ Clear, descriptive error messages for debugging
- ✅ Graceful degradation in all error conditions
- ✅ Thread-safe error handling

```rust
match property.subscribe_with_subscription(observer) {
    Ok(_subscription) => {
        // Use subscription
    }
    Err(PropertyError::PoisonedLock) => {
        // Handle poisoned lock scenario
        eprintln!("Property is in invalid state");
    }
    Err(e) => {
        eprintln!("Failed to create subscription: {}", e);
    }
}
```

## Migration from Manual to RAII Subscriptions

If you're upgrading from manual subscription management:

```rust
// Before (manual management)
let observer_id = property.subscribe(Arc::new(|old, new| {
    println!("Value: {} -> {}", old, new);
}))?;
// ... do work
property.unsubscribe(observer_id)?;

// After (RAII management)  
let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    println!("Value: {} -> {}", old, new);
}))?;
// ... do work
// Automatic cleanup!
```

## Recent Improvements

### v0.2.1 - Enhanced Error Handling & Production Readiness

- 🔧 **Eliminated all `unwrap()` calls**: Replaced with proper error handling using `expect()` with descriptive messages
- 🛡️ **Enhanced robustness**: All error conditions now provide clear, actionable error messages
- 🧪 **Improved testing**: 40+ unit tests and 26+ documentation tests ensure reliability
- 🔒 **Better poisoned lock handling**: Graceful degradation when locks are poisoned by panics
- 📈 **Production ready**: Suitable for production environments with comprehensive error handling
- 🚀 **Performance**: No runtime performance impact from improved error handling
- 📚 **Better debugging**: Clear error context helps identify issues quickly

The library now follows Rust best practices for error handling, making it more reliable and easier to debug in production environments.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This software is provided "as-is" without any express or implied warranties. While every effort has been made to ensure reliability and correctness, the authors and contributors make no guarantees regarding the software's performance, suitability for any particular purpose, or freedom from defects. Use this library at your own risk.

Users are responsible for:
- Testing the library thoroughly in their specific use cases
- Implementing appropriate error handling and validation
- Ensuring the library meets their performance and reliability requirements

The comprehensive error handling and extensive test suite are designed to promote reliability, but do not constitute a warranty or guarantee of correctness.

## License

This project is licensed under either the:

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
