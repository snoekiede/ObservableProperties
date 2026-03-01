# Change Coalescing

## Overview

Change coalescing allows you to batch multiple property updates and send a single notification with the final value. This is useful for bulk updates where intermediate values don't matter.

## API

### `begin_update()`

Starts a batch update, suppressing observer notifications until `end_update()` is called.

```rust
pub fn begin_update(&self) -> Result<(), PropertyError>
```

### `end_update()`

Ends a batch update, sending a single notification from the initial value to the final value.

```rust
pub fn end_update(&self) -> Result<(), PropertyError>
```

## Features

- **Nested Batches**: Supports nested `begin_update()`/`end_update()` calls
- **Thread-Safe**: Works correctly across multiple threads
- **Works with Both `set()` and `set_async()`**: Both methods respect batching
- **Error Handling**: Returns error if `end_update()` is called without matching `begin_update()`

## Examples

### Basic Usage

```rust
use observable_property::ObservableProperty;
use std::sync::Arc;

let property = ObservableProperty::new(0);

property.subscribe(Arc::new(|old, new| {
    println!("Value changed: {} -> {}", old, new);
}))?;

// Begin batch update
property.begin_update()?;

// Multiple changes - no notifications yet
property.set(10)?;
property.set(20)?;
property.set(30)?;

// End batch - single notification from 0 to 30
property.end_update()?;
// Output: "Value changed: 0 -> 30"
```

### Nested Batches

```rust
property.begin_update()?;  // Outer batch
property.set(10)?;

property.begin_update()?;  // Inner batch
property.set(20)?;
property.set(30)?;
property.end_update()?;    // End inner (no notification yet)

property.set(40)?;
property.end_update()?;    // End outer - notification: 0 -> 40
```

### With Async Updates

```rust
property.begin_update()?;
property.set(10)?;
property.set_async(20)?;  // Also suppressed
property.set(30)?;
property.end_update()?;   // Single notification: 0 -> 30
```

## Implementation Details

- Batching state is tracked with `batch_depth` counter
- Initial value is stored when first `begin_update()` is called
- Notifications are suppressed while `batch_depth > 0`
- Observer calls happen only when outermost batch completes
- Compatible with all existing features (history, metrics, weak observers, etc.)

## Benefits

1. **Performance**: Reduces notification overhead for bulk updates
2. **Atomicity**: Observers see only the final state change
3. **Simplicity**: Clear API for batch operations
4. **Flexibility**: Works with nested updates and multi-threading

## See Also

- Example: [examples/update_batch.rs](examples/update_batch.rs)
- Tests: Search for `test_change_coalescing_*` in [src/lib.rs](src/lib.rs)
- Alternative API: `update_batch()` method for functional-style batching
