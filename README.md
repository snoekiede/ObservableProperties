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
* **Convenient macros**: Automatic property wrapping and struct generation

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
observable-property = "0.2.1"
```

## Usage

### Basic Example with Macros

```rust
use observable_property::{observable, ObservableProperty};
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    // Create an observable property using the convenient macro
    let counter = observable!(42);
    let name = observable!("Alice".to_string());

    // Subscribe to changes
    let counter_id = counter.subscribe(Arc::new(|old_value, new_value| {
        println!("Counter changed from {} to {}", old_value, new_value);
    }))?;

    let name_id = name.subscribe(Arc::new(|old_value, new_value| {
        println!("Name changed from '{}' to '{}'", old_value, new_value);
    }))?;

    // Change the values (triggers observers)
    counter.set(100)?; // Prints: Counter changed from 42 to 100
    name.set("Bob".to_string())?; // Prints: Name changed from 'Alice' to 'Bob'

    // Clean up
    counter.unsubscribe(counter_id)?;
    name.unsubscribe(name_id)?;
    
    Ok(())
}
```

### Struct with Observable Fields

```rust
use observable_property::{Observable, observable, ObservableProperty};
use std::sync::Arc;

#[derive(Observable)]
struct Person {
    #[observable]
    name: ObservableProperty<String>,
    #[observable]
    age: ObservableProperty<i32>,
    #[observable]
    salary: ObservableProperty<f64>,
    // Regular field (not observable)
    id: u64,
}

fn main() -> Result<(), observable_property::PropertyError> {
    let person = Person {
        name: observable!("Alice".to_string()),
        age: observable!(30),
        salary: observable!(50000.0),
        id: 12345,
    };

    // Subscribe to individual properties using generated methods
    let name_id = person.subscribe_name(Arc::new(|old, new| {
        println!("Name changed: '{}' -> '{}'", old, new);
    }))?;

    let age_id = person.subscribe_age(Arc::new(|old, new| {
        println!("Age changed: {} -> {}", old, new);
    }))?;

    // Subscribe with filtering - only notify when salary increases
    let salary_id = person.subscribe_salary_filtered(
        Arc::new(|old, new| {
            println!("Salary increased: ${:.2} -> ${:.2}", old, new);
        }),
        |old, new| new > old
    )?;

    // Use generated setter methods
    person.set_name("Bob".to_string())?;
    person.set_age(31)?;
    person.set_salary(55000.0)?; // Will trigger filtered observer
    person.set_salary(54000.0)?; // Will NOT trigger filtered observer

    // Use generated getter methods
    println!("Current name: {}", person.get_name()?);
    println!("Current age: {}", person.get_age()?);
    println!("Current salary: ${:.2}", person.get_salary()?);
    
    // Regular fields work normally
    println!("ID: {}", person.id);

    // Clean up
    person.unsubscribe_name(name_id)?;
    person.unsubscribe_age(age_id)?;
    person.unsubscribe_salary(salary_id)?;
    
    Ok(())
}
```

### Traditional Usage (Without Macros)

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    // Create an observable property manually
    let property = ObservableProperty::new(42);

    // Subscribe to changes
    let observer_id = property.subscribe(Arc::new(|old_value, new_value| {
        println!("Value changed from {} to {}", old_value, new_value);
    }))?;

    // Change the value (triggers observer)
    property.set(100)?; // Prints: Value changed from 42 to 100

    // Unsubscribe when done
    property.unsubscribe(observer_id)?;
    
    Ok(())
}
```

### Multi-threading Example

```rust
use observable_property::{observable, ObservableProperty};
use std::sync::Arc;
use std::thread;

fn main() -> Result<(), observable_property::PropertyError> {
    let property = Arc::new(observable!(0));
    let property_clone = property.clone();

    // Subscribe from one thread
    property.subscribe(Arc::new(|old, new| {
        println!("Value changed: {} -> {}", old, new);
    }))?;

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
use observable_property::observable;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    let counter = observable!(0);

    // Only notify when value increases
    let observer_id = counter.subscribe_filtered(
        Arc::new(|old, new| println!("Increased: {} -> {}", old, new)),
        |old, new| new > old
    )?;

    counter.set(5)?;  // Triggers observer: "Increased: 0 -> 5"
    counter.set(3)?;  // Does NOT trigger observer
    counter.set(10)?; // Triggers observer: "Increased: 3 -> 10"
    
    counter.unsubscribe(observer_id)?;
    Ok(())
}
```

### Async Notifications

For observers that might perform time-consuming operations, use async notifications to avoid blocking:

```rust
use observable_property::observable;
use std::sync::Arc;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    let property = observable!(0);

    property.subscribe(Arc::new(|old, new| {
        // This slow observer won't block the caller when using set_async
        std::thread::sleep(Duration::from_millis(100));
        println!("Slow observer: {} -> {}", old, new);
    }))?;

    // This returns immediately even though observer is slow
    property.set_async(42)?;
    
    // Continue with other work while observers run in background
    println!("This prints immediately!");
    
    Ok(())
}
```

## Available Macros

### `observable!(value)`

Creates an `ObservableProperty` from any value:

```rust
let counter = observable!(0);
let name = observable!("Alice".to_string());
let config = observable!(MyStruct::default());
```

### `#[derive(Observable)]`

Generates convenient methods for structs with observable fields:

```rust
#[derive(Observable)]
struct MyStruct {
    #[observable]
    field1: ObservableProperty<String>,
    #[observable] 
    field2: ObservableProperty<i32>,
    regular_field: bool, // Not observable
}
```

**Generated methods for each `#[observable]` field:**
- `get_field1()` -> `Result<String, PropertyError>`
- `set_field1(value: String)` -> `Result<(), PropertyError>`
- `subscribe_field1(observer)` -> `Result<ObserverId, PropertyError>`
- `subscribe_field1_filtered(observer, filter)` -> `Result<ObserverId, PropertyError>`
- `unsubscribe_field1(id)` -> `Result<bool, PropertyError>`

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
- **Macros**: The convenience macros have zero runtime overhead - they generate the same code you'd write manually.

## Thread Safety

All operations are thread-safe:
- Multiple threads can read the property value simultaneously
- Only one thread can modify the property at a time
- Observer notifications happen outside the lock to minimize contention
- Observer panics are isolated and don't affect other observers or the property

## Features

This crate supports the following feature flags:

- `macros` (default): Enables the convenient `observable!` macro and `Observable` derive macro
- Without macros: Only the core `ObservableProperty` type is available

```toml
# Disable macros if you only want the core functionality
[dependencies]
observable-property = { version = "0.2.1", default-features = false }
```

## Examples

Check out the [examples directory](examples/) for more comprehensive usage examples:

- [`basic.rs`](examples/basic.rs) - Basic usage without macros
- [`multithreaded.rs`](examples/multithreaded.rs) - Multi-threaded scenarios
- [`macros_demo.rs`](examples/macros_demo.rs) - Comprehensive macro usage examples

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
