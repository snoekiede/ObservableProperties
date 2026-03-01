# Observable-Property 

## Features

* **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access
* **Observer pattern**: Subscribe to property changes with callbacks
* **RAII subscriptions**: Automatic cleanup with subscription guards (no manual unsubscribe needed)
* **Filtered observers**: Only notify when specific conditions are met
* **Computed properties**: Automatically recompute derived values when dependencies change
* **Debouncing**: Delay notifications until changes stop for a specified duration
* **Throttling**: Rate-limit notifications to at most once per time interval
* **Async notifications**: Non-blocking observer notifications with background threads
* **Configurable threading**: Customize thread pool size for async notifications via `with_max_threads()`
* **Panic isolation**: Observer panics don't crash the system
* **Robust error handling**: Comprehensive error handling with descriptive error messages
* **Production-ready**: No `unwrap()` calls - all errors are handled gracefully
* **Type-safe**: Generic implementation works with any `Clone + Send + Sync + 'static' type
* **Zero dependencies**: Uses only Rust standard library
* **Exhaustively tested**: 235 tests (128 unit + 107 doc tests) with 100% feature coverage

A thread-safe observable property implementation for Rust that allows you to observe changes to values across multiple threads. Built with comprehensive error handling and no `unwrap()` calls for maximum reliability.

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
observable-property = "0.4.2"
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

### Computed Properties

Computed properties automatically update when their dependencies change. Perfect for derived values:

```rust
use observable_property::{ObservableProperty, computed};
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    // Create source properties
    let width = Arc::new(ObservableProperty::new(10));
    let height = Arc::new(ObservableProperty::new(5));

    // Create computed property for area
    let area = computed(
        vec![width.clone(), height.clone()],
        |values| values[0] * values[1]
    )?;

    // Subscribe to changes in the computed property
    area.subscribe(Arc::new(|old, new| {
        println!("Area changed from {} to {}", old, new);
    }))?;

    println!("Initial area: {}", area.get()?); // 50

    // Change width - area updates automatically
    width.set(20)?;
    std::thread::sleep(std::time::Duration::from_millis(10));
    println!("New area: {}", area.get()?); // 100

    // Change height - area updates automatically
    height.set(8)?;
    std::thread::sleep(std::time::Duration::from_millis(10));
    println!("New area: {}", area.get()?); // 160

    Ok(())
}
```

### Debouncing

Debouncing delays notifications until changes stop for a specified duration. Perfect for auto-save, search-as-you-type, and form validation:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    let search_query = ObservableProperty::new("".to_string());

    // Only search after user stops typing for 300ms
    let _subscription = search_query.subscribe_debounced(
        Arc::new(|_old, new| {
            if !new.is_empty() {
                println!("Searching for: {}", new);
                // Perform expensive API call here
            }
        }),
        Duration::from_millis(300)
    )?;

    // Rapid typing - no searches triggered yet
    search_query.set("r".to_string())?;
    search_query.set("ru".to_string())?;
    search_query.set("rus".to_string())?;
    search_query.set("rust".to_string())?;

    // Wait for debounce
    std::thread::sleep(Duration::from_millis(400));
    // Now search executes once with "rust"
    
    Ok(())
}
```

### Throttling

Throttling rate-limits notifications to at most once per interval. Perfect for scroll events, mouse tracking, and API rate limiting:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    let scroll_position = ObservableProperty::new(0);

    // Update UI at most every 100ms, even if scrolling continuously
    let _subscription = scroll_position.subscribe_throttled(
        Arc::new(|_old, new| {
            println!("Updating UI for scroll position: {}", new);
        }),
        Duration::from_millis(100)
    )?;

    // Rapid scroll events (e.g., 60fps = ~16ms per frame)
    for i in 1..=20 {
        scroll_position.set(i * 10)?;
        std::thread::sleep(Duration::from_millis(16));
    }
    // UI updates happen less frequently than scroll events
    
    Ok(())
}
```

**Key Differences:**
- **Debouncing**: Waits for changes to stop, then notifies once with the final value
- **Throttling**: Notifies periodically during continuous changes, firing first immediately

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

### Configurable Threading

Customize the thread pool size for async notifications based on your system requirements:

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    // For high-throughput systems (more CPU cores)
    let high_perf_property = ObservableProperty::with_max_threads(0, 8);
    
    // For resource-constrained systems (embedded/mobile)
    let low_resource_property = ObservableProperty::with_max_threads(42, 1);
    
    // For I/O-heavy observers (network/database operations)
    let io_heavy_property = ObservableProperty::with_max_threads("data".to_string(), 16);

    // Use like any other property
    let _subscription = high_perf_property.subscribe_with_subscription(Arc::new(|old, new| {
        println!("High performance: {} -> {}", old, new);
    }))?;

    // Async notifications will use the configured thread pool
    high_perf_property.set_async(100)?;
    
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

### v0.4.1 - Debouncing & Throttling

- ⏱️ **Debouncing support**: New `subscribe_debounced()` delays notifications until changes stop
- 🎯 **Throttling support**: New `subscribe_throttled()` rate-limits notifications to maximum frequency
- 🔍 **Perfect for UX**: Auto-save, search-as-you-type, form validation, scroll handlers
- 📈 **Performance optimization**: Reduce unnecessary observer calls during rapid changes
- 🧪 **Comprehensive tests**: 13 new tests covering debouncing and throttling behavior
- 📚 **Rich documentation**: Detailed examples for auto-save, search, scroll events, and more
- 🔒 **Thread-safe**: Both features work seamlessly in multi-threaded environments

### v0.4.2 - Exhaustive Test Coverage

- 🧪 **Exhaustive test suite**: 235 total tests (128 unit tests + 107 doc tests) with 100% feature coverage
- ✅ **All features tested**: Comprehensive tests for validation, custom equality, history, event logging, transformations, bidirectional binding, metrics, persistence, and async features
- 🎯 **Production confidence**: Every public API thoroughly tested with both happy paths and error conditions
- 📊 **Quality assurance**: All tests passing, ensuring reliability and correctness
- 🔍 **Edge cases covered**: Tests include concurrent access, error handling, thread safety, and resource limits

### v0.3.2 - Configurable Threading & Enhanced Documentation

- ⚙️ **Configurable thread pools**: New `with_max_threads()` constructor allows custom thread limits for async notifications
- 📖 **Comprehensive documentation**: Complete API documentation with examples, use cases, and performance guidance
- 🧪 **Expanded test coverage**: Foundation for exhaustive testing with unit and documentation tests
- 🎯 **Performance tuning**: Fine-tune async notification performance for different system requirements
- 🔧 **Better async control**: Optimize for CPU-bound, I/O-bound, or resource-constrained environments
- 📚 **Rich examples**: Detailed code examples for high-throughput, embedded, and network-heavy scenarios

### v0.2.1 - Enhanced Error Handling & Production Readiness

- 🔧 **Eliminated all `unwrap()` calls**: Replaced with proper error handling using `expect()` with descriptive messages
- 🛡️ **Enhanced robustness**: All error conditions now provide clear, actionable error messages
- 🧪 **Improved testing**: Foundation for comprehensive test suite ensuring reliability
- 🔒 **Better poisoned lock handling**: Graceful degradation when locks are poisoned by panics
- 📈 **Production ready**: Suitable for production environments with comprehensive error handling
- 🚀 **Performance**: No runtime performance impact from improved error handling
- 📚 **Better debugging**: Clear error context helps identify issues quickly

The library now provides both robust error handling and configurable performance tuning, making it suitable for a wide range of production environments from embedded systems to high-throughput servers.

## Testing

This library has **exhaustive test coverage** with 235 total tests ensuring reliability and correctness:

### Test Statistics
- **128 unit tests** - Comprehensive testing of all functionality
- **107 documentation tests** - Every public API example is tested
- **100% passing** - All tests consistently pass

### Coverage Areas

#### Core Features (100% Covered)
- ✅ Property creation and basic operations
- ✅ Observer subscription and notification
- ✅ RAII subscription automatic cleanup
- ✅ Thread safety and concurrent access
- ✅ Error handling and graceful degradation
- ✅ Async notifications with background threads

#### Advanced Features (100% Covered)
- ✅ **Validation** (`with_validator`) - Valid/invalid values, error handling
- ✅ **Custom Equality** (`with_equality`) - Epsilon tolerance, case-insensitive strings
- ✅ **History Tracking** (`with_history`) - Undo/redo, bounded buffers
- ✅ **Event Logging** (`with_event_log`) - Complete audit trails with timestamps
- ✅ **Property Transformations** (`map`) - Type conversions, chaining
- ✅ **Atomic Modifications** (`modify`) - With validators, complex updates
- ✅ **Bidirectional Binding** - Two-way sync, loop prevention
- ✅ **Performance Metrics** - Observer call counts, notification timing
- ✅ **Persistence** - Load/save, auto-save on changes
- ✅ **Filtering** - Conditional observers with predicates
- ✅ **Debouncing & Throttling** - Rate limiting, delay notifications
- ✅ **Computed Properties** - Automatic dependency tracking

#### Edge Cases & Error Conditions
- ✅ Lock poisoning recovery
- ✅ Observer panics isolation
- ✅ Observer limit enforcement
- ✅ Concurrent subscription management
- ✅ Weak observer cleanup
- ✅ Empty history/event log handling
- ✅ Invalid initial values
- ✅ Thread ID wraparound

### Running Tests

```bash
# Run all tests (unit + doc tests)
cargo test

# Run only unit tests
cargo test --lib

# Run only documentation tests
cargo test --doc

# Run with output
cargo test -- --nocapture
```

All tests complete in ~7 seconds and demonstrate both correct behavior and robust error handling.

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
