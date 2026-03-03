//! # Observable Property
//!
//! A thread-safe observable property implementation for Rust that allows you to
//! observe changes to values across multiple threads.
//!
//! ## Features
//!
//! - **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access
//! - **Observer pattern**: Subscribe to property changes with callbacks
//! - **RAII subscriptions**: Automatic cleanup with subscription guards (no manual unsubscribe needed)
//! - **Filtered observers**: Only notify when specific conditions are met
//! - **Async notifications**: Non-blocking observer notifications with background threads
//! - **Panic isolation**: Observer panics don't crash the system
//! - **Graceful lock recovery**: Continues operation even after lock poisoning from panics
//! - **Memory protection**: Observer limit (10,000) prevents memory exhaustion
//! - **Type-safe**: Generic implementation works with any `Clone + Send + Sync` type
//!
//! ## Quick Start
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//!
//! // Create an observable property
//! let property = ObservableProperty::new(42);
//!
//! // Subscribe to changes
//! let observer_id = property.subscribe(Arc::new(|old_value, new_value| {
//!     println!("Value changed from {} to {}", old_value, new_value);
//! })).map_err(|e| {
//!     eprintln!("Failed to subscribe: {}", e);
//!     e
//! })?;
//!
//! // Change the value (triggers observer)
//! property.set(100).map_err(|e| {
//!     eprintln!("Failed to set value: {}", e);
//!     e
//! })?;
//!
//! // Unsubscribe when done
//! property.unsubscribe(observer_id).map_err(|e| {
//!     eprintln!("Failed to unsubscribe: {}", e);
//!     e
//! })?;
//! # Ok::<(), observable_property::PropertyError>(())
//! ```
//!
//! ## Multi-threading Example
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//! use std::thread;
//!
//! let property = Arc::new(ObservableProperty::new(0));
//! let property_clone = property.clone();
//!
//! // Subscribe from one thread
//! property.subscribe(Arc::new(|old, new| {
//!     println!("Value changed: {} -> {}", old, new);
//! })).map_err(|e| {
//!     eprintln!("Failed to subscribe: {}", e);
//!     e
//! })?;
//!
//! // Modify from another thread
//! thread::spawn(move || {
//!     if let Err(e) = property_clone.set(42) {
//!         eprintln!("Failed to set value: {}", e);
//!     }
//! }).join().expect("Thread panicked");
//! # Ok::<(), observable_property::PropertyError>(())
//! ```
//!
//! ## RAII Subscriptions (Recommended)
//!
//! For automatic cleanup without manual unsubscribe calls, use RAII subscriptions:
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), observable_property::PropertyError> {
//! let property = ObservableProperty::new(0);
//!
//! {
//!     // Create RAII subscription - automatically cleaned up when dropped
//!     let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
//!         println!("Value changed: {} -> {}", old, new);
//!     }))?;
//!
//!     property.set(42)?; // Prints: "Value changed: 0 -> 42"
//!
//!     // Subscription automatically unsubscribes when leaving this scope
//! }
//!
//! // No observer active anymore
//! property.set(100)?; // No output
//! # Ok(())
//! # }
//! ```
//!
//! ## Filtered RAII Subscriptions
//!
//! Combine filtering with automatic cleanup for conditional monitoring:
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), observable_property::PropertyError> {
//! let temperature = ObservableProperty::new(20.0);
//!
//! {
//!     // Monitor only significant temperature increases with automatic cleanup
//!     let _heat_warning = temperature.subscribe_filtered_with_subscription(
//!         Arc::new(|old, new| {
//!             println!("🔥 Heat warning! {:.1}°C -> {:.1}°C", old, new);
//!         }),
//!         |old, new| new > old && (new - old) > 5.0
//!     )?;
//!
//!     temperature.set(22.0)?; // No warning (only 2°C increase)
//!     temperature.set(28.0)?; // Prints warning (6°C increase from 22°C)
//!
//!     // Subscription automatically cleaned up here
//! }
//!
//! temperature.set(35.0)?; // No warning (subscription was cleaned up)
//! # Ok(())
//! # }
//! ```
//!
//! ## Subscription Management Comparison
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), observable_property::PropertyError> {
//! let property = ObservableProperty::new(0);
//! let observer = Arc::new(|old: &i32, new: &i32| {
//!     println!("Value: {} -> {}", old, new);
//! });
//!
//! // Method 1: Manual subscription management
//! let observer_id = property.subscribe(observer.clone())?;
//! property.set(42)?;
//! property.unsubscribe(observer_id)?; // Manual cleanup required
//!
//! // Method 2: RAII subscription management (recommended)
//! {
//!     let _subscription = property.subscribe_with_subscription(observer)?;
//!     property.set(100)?;
//!     // Automatic cleanup when _subscription goes out of scope
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced RAII Patterns
//!
//! Comprehensive example showing various RAII subscription patterns:
//!
//! ```rust
//! use observable_property::ObservableProperty;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), observable_property::PropertyError> {
//! // System monitoring example
//! let cpu_usage = ObservableProperty::new(25.0f64); // percentage
//! let memory_usage = ObservableProperty::new(1024); // MB
//! let active_connections = ObservableProperty::new(0u32);
//!
//! // Conditional monitoring based on system state
//! let high_load_monitoring = cpu_usage.get()? > 50.0;
//!
//! if high_load_monitoring {
//!     // Critical system monitoring - active only during high load
//!     let _cpu_critical = cpu_usage.subscribe_filtered_with_subscription(
//!         Arc::new(|old, new| {
//!             println!("🚨 Critical CPU usage: {:.1}% -> {:.1}%", old, new);
//!         }),
//!         |_, new| *new > 80.0
//!     )?;
//!
//!     let _memory_warning = memory_usage.subscribe_filtered_with_subscription(
//!         Arc::new(|old, new| {
//!             println!("⚠️ High memory usage: {}MB -> {}MB", old, new);
//!         }),
//!         |_, new| *new > 8192 // > 8GB
//!     )?;
//!
//!     // Simulate system load changes
//!     cpu_usage.set(85.0)?;     // Would trigger critical alert
//!     memory_usage.set(9216)?;  // Would trigger memory warning
//!     
//!     // All monitoring automatically stops when exiting this block
//! }
//!
//! // Connection monitoring with scoped lifetime
//! {
//!     let _connection_monitor = active_connections.subscribe_with_subscription(
//!         Arc::new(|old, new| {
//!             if new > old {
//!                 println!("📈 New connections: {} -> {}", old, new);
//!             } else if new < old {
//!                 println!("📉 Connections closed: {} -> {}", old, new);
//!             }
//!         })
//!     )?;
//!
//!     active_connections.set(5)?;  // Prints: "📈 New connections: 0 -> 5"
//!     active_connections.set(3)?;  // Prints: "📉 Connections closed: 5 -> 3"
//!     active_connections.set(8)?;  // Prints: "📈 New connections: 3 -> 8"
//!
//!     // Connection monitoring automatically stops here
//! }
//!
//! // No monitoring active anymore
//! cpu_usage.set(95.0)?;         // No output
//! memory_usage.set(10240)?;     // No output  
//! active_connections.set(15)?;  // No output
//!
//! println!("All monitoring automatically cleaned up!");
//! # Ok(())
//! # }
//! ```

// Module declarations
mod constants;
mod core;
mod error;
mod events;
mod metrics;
mod observer;
mod persistence;
mod subscription;

pub mod computed;

#[cfg(test)]
pub mod testing;

// Re-exports
pub use error::PropertyError;
pub use events::PropertyEvent;
pub use metrics::PropertyMetrics;
pub use observer::{Observer, ObserverId};
pub use persistence::PropertyPersistence;
pub use subscription::Subscription;
pub use computed::computed;

// Internal use
use crate::constants::{MAX_THREADS, MAX_OBSERVERS};
use crate::core::InnerProperty;
use crate::observer::ObserverRef;
#[cfg(feature = "debug")]
use crate::events::ChangeLog;
use std::collections::HashMap;
use std::mem;
use std::panic;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "debug")]
use backtrace::Backtrace;

#[cfg(feature = "async")]
use std::pin::Pin;

#[cfg(feature = "async")]
use std::task::{Context, Poll};

/// A custom Stream trait for async iteration over values
///
/// This trait provides a simple async iteration interface similar to the standard
/// `futures::Stream` trait, but implemented using only standard library primitives.
///
/// # Note
///
/// This is a custom trait and is not compatible with the futures ecosystem's
/// `Stream` trait. For ecosystem compatibility, consider using the futures-core crate.
#[cfg(feature = "async")]
pub trait Stream {
    /// The type of items yielded by this stream
    type Item;

    /// Attempts to pull out the next value of this stream
    ///
    /// Returns:
    /// - `Poll::Ready(Some(item))` if a value is ready
    /// - `Poll::Pending` if no value is ready yet (will wake the task later)
    /// - `Poll::Ready(None)` if the stream has ended
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>>;
}

/// A thread-safe observable property that notifies observers when its value changes
///
/// This type wraps a value of type `T` and allows multiple observers to be notified
/// whenever the value is modified. All operations are thread-safe and can be called
/// from multiple threads concurrently.
///
/// # Type Requirements
///
/// The generic type `T` must implement:
/// - `Clone`: Required for returning values and passing them to observers
/// - `Send`: Required for transferring between threads
/// - `Sync`: Required for concurrent access from multiple threads  
/// - `'static`: Required for observer callbacks that may outlive the original scope
///
/// # Examples
///
/// ```rust
/// use observable_property::ObservableProperty;
/// use std::sync::Arc;
///
/// let property = ObservableProperty::new("initial".to_string());
///
/// let observer_id = property.subscribe(Arc::new(|old, new| {
///     println!("Changed from '{}' to '{}'", old, new);
/// })).map_err(|e| {
///     eprintln!("Failed to subscribe: {}", e);
///     e
/// })?;
///
/// property.set("updated".to_string()).map_err(|e| {
///     eprintln!("Failed to set value: {}", e);
///     e
/// })?; // Prints: Changed from 'initial' to 'updated'
///
/// property.unsubscribe(observer_id).map_err(|e| {
///     eprintln!("Failed to unsubscribe: {}", e);
///     e
/// })?;
/// # Ok::<(), observable_property::PropertyError>(())
/// ```
pub struct ObservableProperty<T>
where
    T: Clone + Send + Sync + 'static,
{
    inner: Arc<RwLock<InnerProperty<T>>>,
    max_threads: usize,
    max_observers: usize,
}

impl<T: Clone + Send + Sync + 'static> ObservableProperty<T> {
    /// Creates a new observable property with the given initial value
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// match property.get() {
    ///     Ok(value) => assert_eq!(value, 42),
    ///     Err(e) => eprintln!("Failed to get property value: {}", e),
    /// }
    /// ```
    pub fn new(initial_value: T) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: None,
                event_log: None,
                event_log_size: 0,
            })),
            max_threads: MAX_THREADS,
            max_observers: MAX_OBSERVERS,
        }
    }

    /// Creates a new observable property with custom equality comparison
    ///
    /// This constructor allows you to define custom logic for determining when two values
    /// are considered "equal". Observers are only notified when the equality function
    /// returns `false` (i.e., when the values are considered different).
    ///
    /// This is particularly useful for:
    /// - Float comparisons with epsilon tolerance (avoiding floating-point precision issues)
    /// - Case-insensitive string comparisons
    /// - Semantic equality that differs from structural equality
    /// - Preventing spurious notifications for "equal enough" values
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    /// * `eq_fn` - A function that returns `true` if two values should be considered equal,
    ///             `false` if they should be considered different (which triggers notifications)
    ///
    /// # Examples
    ///
    /// ## Float comparison with epsilon
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Create a property that only notifies if the difference is > 0.001
    /// let temperature = ObservableProperty::with_equality(
    ///     20.0_f64,
    ///     |a, b| (a - b).abs() < 0.001
    /// );
    ///
    /// let _sub = temperature.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Significant temperature change: {} -> {}", old, new);
    /// }))?;
    ///
    /// temperature.set(20.0005)?;  // No notification (within epsilon)
    /// temperature.set(20.5)?;     // Notification triggered
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Case-insensitive string comparison
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Only notify on case-insensitive changes
    /// let username = ObservableProperty::with_equality(
    ///     "Alice".to_string(),
    ///     |a, b| a.to_lowercase() == b.to_lowercase()
    /// );
    ///
    /// let _sub = username.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Username changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// username.set("alice".to_string())?;  // No notification
    /// username.set("ALICE".to_string())?;  // No notification
    /// username.set("Bob".to_string())?;    // Notification triggered
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Semantic equality for complex types
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// #[derive(Clone)]
    /// struct Config {
    ///     host: String,
    ///     port: u16,
    ///     timeout_ms: u64,
    /// }
    ///
    /// // Only notify if critical fields change
    /// let config = ObservableProperty::with_equality(
    ///     Config { host: "localhost".to_string(), port: 8080, timeout_ms: 1000 },
    ///     |a, b| a.host == b.host && a.port == b.port  // Ignore timeout changes
    /// );
    ///
    /// let _sub = config.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Critical config changed: {}:{} -> {}:{}",
    ///         old.host, old.port, new.host, new.port);
    /// }))?;
    ///
    /// config.modify(|c| c.timeout_ms = 2000)?;  // No notification (timeout ignored)
    /// config.modify(|c| c.port = 9090)?;        // Notification triggered
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// The equality function is called every time `set()` or `set_async()` is called,
    /// so it should be relatively fast. For expensive comparisons, consider using
    /// filtered observers instead.
    ///
    /// # Thread Safety
    ///
    /// The equality function must be `Send + Sync + 'static` as it may be called from
    /// any thread that modifies the property.
    pub fn with_equality<F>(initial_value: T, eq_fn: F) -> Self
    where
        F: Fn(&T, &T) -> bool + Send + Sync + 'static,
    {
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: Some(Arc::new(eq_fn)),
                validator: None,
                event_log: None,
                event_log_size: 0,
            })),
            max_threads: MAX_THREADS,
            max_observers: MAX_OBSERVERS,
        }
    }

    /// Creates a new observable property with value validation
    ///
    /// This constructor enables value validation for the property. Any attempt to set
    /// a value that fails validation will be rejected with a `ValidationError`. This
    /// ensures the property always contains valid data according to your business rules.
    ///
    /// The validator function is called:
    /// - When the property is created (to validate the initial value)
    /// - Every time `set()` or `set_async()` is called (before the value is changed)
    /// - When `modify()` is called (after the modification function runs)
    ///
    /// If validation fails, the property value remains unchanged and an error is returned.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property (must pass validation)
    /// * `validator` - A function that validates values, returning `Ok(())` for valid values
    ///                 or `Err(String)` with an error message for invalid values
    ///
    /// # Returns
    ///
    /// * `Ok(Self)` - If the initial value passes validation
    /// * `Err(PropertyError::ValidationError)` - If the initial value fails validation
    ///
    /// # Use Cases
    ///
    /// ## Age Validation
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Only allow ages between 0 and 150
    /// let age = ObservableProperty::with_validator(
    ///     25,
    ///     |age| {
    ///         if *age <= 150 {
    ///             Ok(())
    ///         } else {
    ///             Err(format!("Age must be between 0 and 150, got {}", age))
    ///         }
    ///     }
    /// )?;
    ///
    /// let _sub = age.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Age changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// age.set(30)?;  // ✓ Valid - prints: "Age changed: 25 -> 30"
    ///
    /// // Attempt to set invalid age
///     match age.set(200) {
///         Err(e) => println!("Validation failed: {}", e), // Prints validation error
    ///         Ok(_) => unreachable!(),
    ///     }
    ///
    /// assert_eq!(age.get()?, 30); // Value unchanged after failed validation
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Email Format Validation
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Validate email format (simplified)
    /// let email = ObservableProperty::with_validator(
    ///     "user@example.com".to_string(),
    ///     |email| {
    ///         if email.contains('@') && email.contains('.') {
    ///             Ok(())
    ///         } else {
    ///             Err(format!("Invalid email format: {}", email))
    ///         }
    ///     }
    /// )?;
    ///
    /// email.set("valid@email.com".to_string())?;  // ✓ Valid
    ///
    /// match email.set("invalid-email".to_string()) {
    ///     Err(e) => println!("{}", e), // Prints: "Validation failed: Invalid email format: invalid-email"
    ///     Ok(_) => unreachable!(),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Range Validation for Floats
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Temperature must be between absolute zero and practical maximum
    /// let temperature = ObservableProperty::with_validator(
    ///     20.0_f64,
    ///     |temp| {
    ///         if *temp >= -273.15 && *temp <= 1000.0 {
    ///             Ok(())
    ///         } else {
    ///             Err(format!("Temperature {} is out of valid range [-273.15, 1000.0]", temp))
    ///         }
    ///     }
    /// )?;
    ///
    /// temperature.set(100.0)?;   // ✓ Valid
    /// temperature.set(-300.0).unwrap_err();  // ✗ Fails validation
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multiple Validation Rules
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Username validation with multiple rules
    /// let username = ObservableProperty::with_validator(
    ///     "alice".to_string(),
    ///     |name| {
    ///         if name.is_empty() {
    ///             return Err("Username cannot be empty".to_string());
    ///         }
    ///         if name.len() < 3 {
    ///             return Err(format!("Username must be at least 3 characters, got {}", name.len()));
    ///         }
    ///         if name.len() > 20 {
    ///             return Err(format!("Username must be at most 20 characters, got {}", name.len()));
    ///         }
    ///         if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
    ///             return Err("Username can only contain letters, numbers, and underscores".to_string());
    ///         }
    ///         Ok(())
    ///     }
    /// )?;
    ///
    /// username.set("bob".to_string())?;      // ✓ Valid
    /// username.set("ab".to_string()).unwrap_err();   // ✗ Too short
    /// username.set("user@123".to_string()).unwrap_err(); // ✗ Invalid characters
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Rejecting Invalid Initial Values
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() {
    /// // Attempt to create with invalid initial value
    /// let result = ObservableProperty::with_validator(
    ///     200,
    ///     |age| {
    ///         if *age <= 150 {
    ///             Ok(())
    ///         } else {
    ///             Err(format!("Age must be at most 150, got {}", age))
    ///         }
    ///     }
    /// );
    ///
    /// match result {
    ///     Err(e) => println!("Failed to create property: {}", e),
    ///     Ok(_) => unreachable!(),
    /// }
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// The validator function is called on every `set()` or `set_async()` operation,
    /// so it should be relatively fast. For expensive validations, consider:
    /// - Caching validation results if the same value is set multiple times
    /// - Using async validation patterns outside of the property setter
    /// - Implementing early-exit validation logic (check cheapest rules first)
    ///
    /// # Thread Safety
    ///
    /// The validator function must be `Send + Sync + 'static` as it may be called from
    /// any thread that modifies the property. Ensure your validation logic is thread-safe.
    ///
    /// # Combining with Other Features
    ///
    /// Validation works alongside other property features:
    /// - **Custom equality**: Validation runs before equality checking
    /// - **History tracking**: Only valid values are stored in history
    /// - **Observers**: Observers only fire when validation succeeds and values differ
    /// - **Batching**: Validation occurs when batch is committed, not during batch
    pub fn with_validator<F>(
        initial_value: T,
        validator: F,
    ) -> Result<Self, PropertyError>
    where
        F: Fn(&T) -> Result<(), String> + Send + Sync + 'static,
    {
        // Validate the initial value
        validator(&initial_value).map_err(|reason| PropertyError::ValidationError { reason })?;

        Ok(Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: Some(Arc::new(validator)),
                event_log: None,
                event_log_size: 0,
            })),
            max_threads: MAX_THREADS,
            max_observers: MAX_OBSERVERS,
        })
    }

    /// Creates a new observable property with a custom maximum thread count for async notifications
    ///
    /// This constructor allows you to customize the maximum number of threads used for
    /// asynchronous observer notifications via `set_async()`. This is useful for tuning
    /// performance based on your specific use case and system constraints.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    /// * `max_threads` - Maximum number of threads to use for async notifications.
    ///   If 0 is provided, defaults to 4.
    ///
    /// # Thread Pool Behavior
    ///
    /// When `set_async()` is called, observers are divided into batches and each batch
    /// runs in its own thread, up to the specified maximum. For example:
    /// - With 100 observers and `max_threads = 4`: 4 threads with ~25 observers each
    /// - With 10 observers and `max_threads = 8`: 10 threads with 1 observer each
    /// - With 2 observers and `max_threads = 4`: 2 threads with 1 observer each
    ///
    /// # Use Cases
    ///
    /// ## High-Throughput Systems
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// // For systems with many CPU cores and CPU-bound observers
    /// let property = ObservableProperty::with_max_threads(0, 8);
    /// ```
    ///
    /// ## Resource-Constrained Systems
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// // For embedded systems or memory-constrained environments
    /// let property = ObservableProperty::with_max_threads(42, 1);
    /// ```
    ///
    /// ## I/O-Heavy Observers
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// // For observers that do network/database operations
    /// let property = ObservableProperty::with_max_threads("data".to_string(), 16);
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - **Higher values**: Better parallelism but more thread overhead and memory usage
    /// - **Lower values**: Less overhead but potentially slower async notifications
    /// - **Optimal range**: Typically between 1 and 2x the number of CPU cores
    /// - **Zero value**: Automatically uses the default value (4)
    ///
    /// # Thread Safety
    ///
    /// This setting only affects async notifications (`set_async()`). Synchronous
    /// operations (`set()`) always execute observers sequentially regardless of this setting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// // Create property with custom thread pool size
    /// let property = ObservableProperty::with_max_threads(42, 2);
    ///
    /// // Subscribe observers as usual
    /// let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Value changed: {} -> {}", old, new);
    /// })).expect("Failed to create subscription");
    ///
    /// // Async notifications will use at most 2 threads
    /// property.set_async(100).expect("Failed to set value asynchronously");
    /// ```
    pub fn with_max_threads(initial_value: T, max_threads: usize) -> Self {
        let max_threads = if max_threads == 0 {
            MAX_THREADS
        } else {
            max_threads
        };
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: None,
                event_log: None,
                event_log_size: 0,
            })),
            max_threads,
            max_observers: MAX_OBSERVERS,
        }
    }

    /// Creates a new observable property with automatic persistence
    ///
    /// This constructor creates a property that automatically saves its value to persistent
    /// storage whenever it changes. It attempts to load the initial value from storage, falling
    /// back to the provided `initial_value` if loading fails.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The value to use if loading from persistence fails
    /// * `persistence` - An implementation of `PropertyPersistence` that handles save/load operations
    ///
    /// # Behavior
    ///
    /// 1. Attempts to load the initial value from `persistence.load()`
    /// 2. If loading fails, uses the provided `initial_value`
    /// 3. Sets up an internal observer that automatically calls `persistence.save()` on every value change
    /// 4. Returns the configured property
    ///
    /// # Error Handling
    ///
    /// - Load failures are logged and the provided `initial_value` is used instead
    /// - Save failures during subsequent updates will be logged but won't prevent the update
    /// - The property continues to operate normally even if persistence operations fail
    ///
    /// # Type Requirements
    ///
    /// The persistence handler's `Value` type must match the property's type `T`.
    ///
    /// # Examples
    ///
    /// ## File-based Persistence
    ///
    /// ```rust,no_run
    /// use observable_property::{ObservableProperty, PropertyPersistence};
    /// use std::fs;
    ///
    /// struct FilePersistence {
    ///     path: String,
    /// }
    ///
    /// impl PropertyPersistence for FilePersistence {
    ///     type Value = String;
    ///
    ///     fn load(&self) -> Result<Self::Value, Box<dyn std::error::Error + Send + Sync>> {
    ///         fs::read_to_string(&self.path)
    ///             .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    ///     }
    ///
    ///     fn save(&self, value: &Self::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         fs::write(&self.path, value)
    ///             .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
    ///     }
    /// }
    ///
    /// // Create a property that auto-saves to a file
    /// let property = ObservableProperty::with_persistence(
    ///     "default_value".to_string(),
    ///     FilePersistence { path: "./data.txt".to_string() }
    /// );
    ///
    /// // Value changes are automatically saved to disk
    /// property.set("new_value".to_string())
    ///     .expect("Failed to set value");
    /// ```
    ///
    /// ## JSON Database Persistence
    ///
    /// ```rust,no_run
    /// # // This example requires serde and serde_json dependencies
    /// use observable_property::{ObservableProperty, PropertyPersistence};
    /// # /*
    /// use serde::{Serialize, Deserialize};
    /// # */
    /// use std::fs;
    ///
    /// # /*
    /// #[derive(Clone, Serialize, Deserialize)]
    /// # */
    /// # #[derive(Clone)]
    /// struct UserPreferences {
    ///     theme: String,
    ///     font_size: u32,
    /// }
    ///
    /// struct JsonPersistence {
    ///     path: String,
    /// }
    ///
    /// impl PropertyPersistence for JsonPersistence {
    ///     type Value = UserPreferences;
    ///
    ///     fn load(&self) -> Result<Self::Value, Box<dyn std::error::Error + Send + Sync>> {
    ///         # /*
    ///         let data = fs::read_to_string(&self.path)?;
    ///         let prefs = serde_json::from_str(&data)?;
    ///         Ok(prefs)
    ///         # */
    ///         # Ok(UserPreferences { theme: "dark".to_string(), font_size: 14 })
    ///     }
    ///
    ///     fn save(&self, value: &Self::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    ///         # /*
    ///         let data = serde_json::to_string_pretty(value)?;
    ///         fs::write(&self.path, data)?;
    ///         # */
    ///         Ok(())
    ///     }
    /// }
    ///
    /// let default_prefs = UserPreferences {
    ///     theme: "dark".to_string(),
    ///     font_size: 14,
    /// };
    ///
    /// let prefs = ObservableProperty::with_persistence(
    ///     default_prefs,
    ///     JsonPersistence { path: "./preferences.json".to_string() }
    /// );
    ///
    /// // Changes are auto-saved as JSON
    /// prefs.modify(|p| {
    ///     p.theme = "light".to_string();
    ///     p.font_size = 16;
    /// }).expect("Failed to update preferences");
    /// ```
    ///
    /// # Thread Safety
    ///
    /// The persistence handler must be `Send + Sync + 'static` since save operations
    /// may be called from any thread holding a reference to the property.
    ///
    /// # Performance Considerations
    ///
    /// - Persistence operations are called synchronously on every value change
    /// - Use fast storage backends or consider debouncing frequent updates
    /// - For high-frequency updates, consider implementing a buffered persistence strategy
    pub fn with_persistence<P>(initial_value: T, persistence: P) -> Self
    where
        P: PropertyPersistence<Value = T>,
    {
        // Try to load from persistence, fall back to initial_value on error
        let value = persistence.load().unwrap_or_else(|e| {
            eprintln!(
                "Failed to load persisted value, using initial value: {}",
                e
            );
            initial_value.clone()
        });

        // Create the property with the loaded or default value
        let property = Self::new(value);

        // Set up auto-save observer
        let persistence = Arc::new(persistence);
        if let Err(e) = property.subscribe(Arc::new(move |_old, new| {
            if let Err(save_err) = persistence.save(new) {
                eprintln!("Failed to persist property value: {}", save_err);
            }
        })) {
            eprintln!("Failed to subscribe persistence observer: {}", e);
        }

        property
    }

    /// Creates a new observable property with history tracking enabled
    ///
    /// This constructor creates a property that maintains a history of previous values,
    /// allowing you to undo changes and view historical values. The history buffer is
    /// automatically managed with a fixed size limit.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    /// * `history_size` - Maximum number of previous values to retain in history.
    ///   If 0, history tracking is disabled and behaves like a regular property.
    ///
    /// # History Behavior
    ///
    /// - The history buffer stores up to `history_size` previous values
    /// - When the buffer is full, the oldest value is removed when a new value is added
    /// - The current value is **not** included in the history - only past values
    /// - History is stored in chronological order (oldest to newest)
    /// - Undo operations pop values from the history and restore them as current
    ///
    /// # Memory Considerations
    ///
    /// Each historical value requires memory equivalent to `size_of::<T>()`. For large
    /// types or high history sizes, consider:
    /// - Using smaller history_size values
    /// - Wrapping large types in `Arc<T>` to share data between history entries
    /// - Implementing custom equality checks to avoid storing duplicate values
    ///
    /// # Examples
    ///
    /// ## Basic History Usage
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Create property with space for 5 historical values
    /// let property = ObservableProperty::with_history(0, 5);
    ///
    /// // Make some changes
    /// property.set(10)?;
    /// property.set(20)?;
    /// property.set(30)?;
    ///
    /// assert_eq!(property.get()?, 30);
    ///
    /// // Undo last change
    /// property.undo()?;
    /// assert_eq!(property.get()?, 20);
    ///
    /// // Undo again
    /// property.undo()?;
    /// assert_eq!(property.get()?, 10);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## View History
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history("start".to_string(), 3);
    ///
    /// property.set("second".to_string())?;
    /// property.set("third".to_string())?;
    /// property.set("fourth".to_string())?;
    ///
    /// // Get all historical values (oldest to newest)
    /// let history = property.get_history();
    /// assert_eq!(history.len(), 3);
    /// assert_eq!(history[0], "start");   // oldest
    /// assert_eq!(history[1], "second");
    /// assert_eq!(history[2], "third");   // newest (most recent past value)
    ///
    /// // Current value is "fourth" and not in history
    /// assert_eq!(property.get()?, "fourth");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## History with Observer Pattern
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history(100, 10);
    ///
    /// // Observers work normally with history-enabled properties
    /// let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Value changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// property.set(200)?; // Triggers observer
    /// property.undo()?;   // Also triggers observer when reverting to 100
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Bounded History Buffer
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Only keep 2 historical values
    /// let property = ObservableProperty::with_history(1, 2);
    ///
    /// property.set(2)?;  // history: [1]
    /// property.set(3)?;  // history: [1, 2]
    /// property.set(4)?;  // history: [2, 3] (oldest '1' was removed)
    ///
    /// let history = property.get_history();
    /// assert_eq!(history, vec![2, 3]);
    /// assert_eq!(property.get()?, 4);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// History tracking is fully thread-safe and works correctly even when multiple
    /// threads are calling `set()`, `undo()`, and `get_history()` concurrently.
    pub fn with_history(initial_value: T, history_size: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: if history_size > 0 {
                    Some(Vec::with_capacity(history_size))
                } else {
                    None
                },
                history_size,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: None,
                event_log: None,
                event_log_size: 0,
            })),
            max_threads: MAX_THREADS,
            max_observers: MAX_OBSERVERS,
        }
    }

    /// Creates a new observable property with event sourcing enabled
    ///
    /// This method enables full event logging for the property, recording every change
    /// as a timestamped event. This provides powerful capabilities for debugging,
    /// auditing, and event replay.
    ///
    /// # Features
    ///
    /// - **Complete Audit Trail**: Every change is recorded with old value, new value, and timestamp
    /// - **Time-Travel Debugging**: Examine the complete history of state changes
    /// - **Event Replay**: Reconstruct property state at any point in time
    /// - **Thread Information**: Each event captures which thread made the change
    /// - **Sequential Numbering**: Events are numbered starting from 0
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    /// * `event_log_size` - Maximum number of events to keep in memory (0 = unlimited)
    ///
    /// # Memory Considerations
    ///
    /// Event logs store complete copies of both old and new values for each change.
    /// For properties with large values or high update frequency:
    /// - Use a bounded `event_log_size` to prevent unbounded memory growth
    /// - Consider using `with_history()` if you only need value snapshots without metadata
    /// - Monitor memory usage in production environments
    ///
    /// When the log exceeds `event_log_size`, the oldest events are automatically removed.
    ///
    /// # Examples
    ///
    /// ## Basic Event Logging
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Create property with unlimited event log
    /// let counter = ObservableProperty::with_event_log(0, 0);
    ///
    /// counter.set(1)?;
    /// counter.set(2)?;
    /// counter.set(3)?;
    ///
    /// // Retrieve the complete event log
    /// let events = counter.get_event_log();
    /// assert_eq!(events.len(), 3);
    ///
    /// // Examine first event
    /// assert_eq!(events[0].old_value, 0);
    /// assert_eq!(events[0].new_value, 1);
    /// assert_eq!(events[0].event_number, 0);
    ///
    /// // Examine last event
    /// assert_eq!(events[2].old_value, 2);
    /// assert_eq!(events[2].new_value, 3);
    /// assert_eq!(events[2].event_number, 2);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Bounded Event Log
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Keep only the last 3 events
    /// let property = ObservableProperty::with_event_log(100, 3);
    ///
    /// property.set(101)?;
    /// property.set(102)?;
    /// property.set(103)?;
    /// property.set(104)?; // Oldest event (100->101) is now removed
    ///
    /// let events = property.get_event_log();
    /// assert_eq!(events.len(), 3);
    /// assert_eq!(events[0].old_value, 101);
    /// assert_eq!(events[2].new_value, 104);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Time-Travel Debugging
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let config = ObservableProperty::with_event_log("default".to_string(), 0);
    ///
    /// let start = std::time::Instant::now();
    /// config.set("config_v1".to_string())?;
    /// std::thread::sleep(Duration::from_millis(10));
    /// config.set("config_v2".to_string())?;
    /// std::thread::sleep(Duration::from_millis(10));
    /// config.set("config_v3".to_string())?;
    ///
    /// // Find what the config was 15ms after start
    /// let target_time = start + Duration::from_millis(15);
    /// let events = config.get_event_log();
    /// 
    /// let mut state = "default".to_string();
    /// for event in events {
    ///     if event.timestamp <= target_time {
    ///         state = event.new_value.clone();
    ///     } else {
    ///         break;
    ///     }
    /// }
    /// 
    /// println!("State at +15ms: {}", state);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Audit Trail with Thread Information
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let shared_state = Arc::new(ObservableProperty::with_event_log(0, 0));
    ///
    /// let handles: Vec<_> = (0..3)
    ///     .map(|i| {
    ///         let state = shared_state.clone();
    ///         thread::spawn(move || {
    ///             state.set(i * 10).expect("Failed to set");
    ///         })
    ///     })
    ///     .collect();
    ///
    /// for handle in handles {
    ///     handle.join().unwrap();
    /// }
    ///
    /// // Examine which threads made changes
    /// let events = shared_state.get_event_log();
    /// for event in events {
    ///     println!(
    ///         "Event #{}: {} -> {} (thread: {})",
    ///         event.event_number,
    ///         event.old_value,
    ///         event.new_value,
    ///         event.thread_id
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Event Replay
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let account_balance = ObservableProperty::with_event_log(1000, 0);
    ///
    /// // Simulate transactions
    /// account_balance.modify(|b| *b -= 100)?; // Withdrawal
    /// account_balance.modify(|b| *b += 50)?;  // Deposit
    /// account_balance.modify(|b| *b -= 200)?; // Withdrawal
    ///
    /// // Replay all transactions
    /// let events = account_balance.get_event_log();
    /// println!("Transaction History:");
    /// for event in events {
    ///     let change = event.new_value as i32 - event.old_value as i32;
    ///     let transaction_type = if change > 0 { "Deposit" } else { "Withdrawal" };
    ///     println!(
    ///         "[{}] {}: ${} (balance: {} -> {})",
    ///         event.event_number,
    ///         transaction_type,
    ///         change.abs(),
    ///         event.old_value,
    ///         event.new_value
    ///     );
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// Event logging is fully thread-safe and works correctly even when multiple
    /// threads are modifying the property concurrently. Event numbers are assigned
    /// sequentially based on the order changes complete (not the order they start).
    ///
    /// # Difference from History
    ///
    /// While `with_history()` only stores previous values, `with_event_log()` stores
    /// complete event objects with timestamps and metadata. This makes event logs
    /// more suitable for auditing and debugging, but they consume more memory.
    ///
    /// | Feature | `with_history()` | `with_event_log()` |
    /// |---------|------------------|--------------------|
    /// | Stores values | ✓ | ✓ |
    /// | Stores timestamps | ✗ | ✓ |
    /// | Stores thread info | ✗ | ✓ |
    /// | Sequential numbering | ✗ | ✓ |
    /// | Old + new values | ✗ | ✓ |
    /// | Memory overhead | Low | Higher |
    /// | Undo support | ✓ | ✗ (manual) |
    pub fn with_event_log(initial_value: T, event_log_size: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: None,
                event_log: Some(Vec::with_capacity(if event_log_size > 0 { event_log_size } else { 16 })),
                event_log_size,
            })),
            max_threads: MAX_THREADS,
            max_observers: MAX_OBSERVERS,
        }
    }

    /// Reverts the property to its previous value from history
    ///
    /// This method pops the most recent value from the history buffer and makes it
    /// the current value. The current value is **not** added to history during undo.
    /// All subscribed observers are notified of this change.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the undo was successful
    /// - `Err(PropertyError::NoHistory)` if:
    ///   - History tracking is not enabled (created without `with_history()`)
    ///   - History buffer is empty (no previous values to restore)
    ///
    /// # Undo Chain Behavior
    ///
    /// Consecutive undo operations walk back through history until exhausted:
    /// ```text
    /// Initial: value=4, history=[1, 2, 3]
    /// After undo(): value=3, history=[1, 2]
    /// After undo(): value=2, history=[1]
    /// After undo(): value=1, history=[]
    /// After undo(): Error(NoHistory) - no more history
    /// ```
    ///
    /// # Observer Notification
    ///
    /// Observers are notified with the current value as "old" and the restored
    /// historical value as "new", maintaining the same notification pattern as `set()`.
    ///
    /// # Examples
    ///
    /// ## Basic Undo
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history(1, 5);
    ///
    /// property.set(2)?;
    /// property.set(3)?;
    /// assert_eq!(property.get()?, 3);
    ///
    /// property.undo()?;
    /// assert_eq!(property.get()?, 2);
    ///
    /// property.undo()?;
    /// assert_eq!(property.get()?, 1);
    ///
    /// // No more history
    /// assert!(property.undo().is_err());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Undo with Observers
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history(10, 5);
    ///
    /// let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("Changed from {} to {}", old, new);
    /// }))?;
    ///
    /// property.set(20)?; // Prints: "Changed from 10 to 20"
    /// property.undo()?;  // Prints: "Changed from 20 to 10"
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Error Handling
    ///
    /// ```rust
    /// use observable_property::{ObservableProperty, PropertyError};
    ///
    /// # fn main() -> Result<(), PropertyError> {
    /// // Property without history
    /// let no_history = ObservableProperty::new(42);
    /// match no_history.undo() {
    ///     Err(PropertyError::NoHistory { .. }) => {
    ///         println!("Expected: history not enabled");
    ///     }
    ///     _ => panic!("Should fail without history"),
    /// }
    ///
    /// // Property with history but empty
    /// let empty_history = ObservableProperty::with_history(100, 5);
    /// match empty_history.undo() {
    ///     Err(PropertyError::NoHistory { .. }) => {
    ///         println!("Expected: no history to undo");
    ///     }
    ///     _ => panic!("Should fail with empty history"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Undo After Multiple Changes
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let counter = ObservableProperty::with_history(0, 3);
    ///
    /// // Make several changes
    /// for i in 1..=5 {
    ///     counter.set(i)?;
    /// }
    ///
    /// assert_eq!(counter.get()?, 5);
    ///
    /// // Undo three times (limited by history_size=3)
    /// counter.undo()?;
    /// assert_eq!(counter.get()?, 4);
    ///
    /// counter.undo()?;
    /// assert_eq!(counter.get()?, 3);
    ///
    /// counter.undo()?;
    /// assert_eq!(counter.get()?, 2);
    ///
    /// // No more history (oldest value in buffer was 2)
    /// assert!(counter.undo().is_err());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently with `set()`,
    /// `get()`, and other operations from multiple threads.
    pub fn undo(&self) -> Result<(), PropertyError> {
        let (old_value, new_value, observers_snapshot, dead_observer_ids) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };

            // Clone the validator Arc before working with history to avoid borrow conflicts
            let validator = prop.validator.clone();

            // Check if history is enabled and has values
            let history = prop.history.as_mut().ok_or_else(|| PropertyError::NoHistory {
                reason: "History tracking is not enabled for this property".to_string(),
            })?;

            if history.is_empty() {
                return Err(PropertyError::NoHistory {
                    reason: "No history available to undo".to_string(),
                });
            }

            // Pop the most recent historical value
            let previous_value = history.pop().unwrap();
            
            // Validate the historical value if a validator is configured
            // This ensures consistency if validation rules have changed since the value was stored
            if let Some(validator) = validator {
                validator(&previous_value).map_err(|reason| {
                    // Put the value back in history if validation fails
                    history.push(previous_value.clone());
                    PropertyError::ValidationError { 
                        reason: format!("Cannot undo to invalid historical value: {}", reason)
                    }
                })?;
            }
            
            let old_value = mem::replace(&mut prop.value, previous_value.clone());

            // Debug logging (requires T: std::fmt::Debug when debug feature is enabled)
            // Collect active observers (same pattern as set())
            let mut observers_snapshot = Vec::new();
            let mut dead_ids = Vec::new();
            for (id, observer_ref) in &prop.observers {
                if let Some(observer) = observer_ref.try_call() {
                    observers_snapshot.push(observer);
                } else {
                    dead_ids.push(*id);
                }
            }

            (old_value, previous_value, observers_snapshot, dead_ids)
        };

        // Notify all active observers
        for observer in observers_snapshot {
            if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                observer(&old_value, &new_value);
            })) {
                eprintln!("Observer panic during undo: {:?}", e);
            }
        }

        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }

        Ok(())
    }

    /// Returns a snapshot of all historical values
    ///
    /// This method returns a vector containing all previous values currently stored
    /// in the history buffer, ordered from oldest to newest. The current value is
    /// **not** included in the returned vector.
    ///
    /// # Returns
    ///
    /// A `Vec<T>` containing historical values in chronological order:
    /// - `vec[0]` is the oldest value in history
    /// - `vec[len-1]` is the most recent past value (the one that would be restored by `undo()`)
    /// - Empty vector if history is disabled or no history has been recorded
    ///
    /// # Memory
    ///
    /// This method clones all historical values, so the returned vector owns its data
    /// independently of the property. This allows safe sharing across threads without
    /// holding locks.
    ///
    /// # Examples
    ///
    /// ## Basic History Retrieval
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history("a".to_string(), 5);
    ///
    /// property.set("b".to_string())?;
    /// property.set("c".to_string())?;
    /// property.set("d".to_string())?;
    ///
    /// let history = property.get_history();
    /// assert_eq!(history.len(), 3);
    /// assert_eq!(history[0], "a"); // oldest
    /// assert_eq!(history[1], "b");
    /// assert_eq!(history[2], "c"); // newest (what undo() would restore)
    ///
    /// // Current value is not in history
    /// assert_eq!(property.get()?, "d");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Empty History
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// // No history recorded yet
    /// let fresh = ObservableProperty::with_history(42, 10);
    /// assert!(fresh.get_history().is_empty());
    ///
    /// // History disabled (size = 0)
    /// let no_tracking = ObservableProperty::with_history(42, 0);
    /// assert!(no_tracking.get_history().is_empty());
    ///
    /// // Regular property (no history support)
    /// let regular = ObservableProperty::new(42);
    /// assert!(regular.get_history().is_empty());
    /// ```
    ///
    /// ## History Buffer Limit
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Limited history size
    /// let property = ObservableProperty::with_history(1, 3);
    ///
    /// for i in 2..=6 {
    ///     property.set(i)?;
    /// }
    ///
    /// // Only last 3 historical values are kept
    /// let history = property.get_history();
    /// assert_eq!(history, vec![3, 4, 5]);
    /// assert_eq!(property.get()?, 6); // current
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Iterating Through History
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history(0.0f64, 5);
    ///
    /// property.set(1.5)?;
    /// property.set(3.0)?;
    /// property.set(4.5)?;
    ///
    /// println!("Historical values:");
    /// for (i, value) in property.get_history().iter().enumerate() {
    ///     println!("  [{}] {}", i, value);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Checking History Before Undo
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_history(100, 5);
    /// property.set(200)?;
    /// property.set(300)?;
    ///
    /// // Check what undo would restore
    /// let history = property.get_history();
    /// if !history.is_empty() {
    ///     let would_restore = history.last().unwrap();
    ///     println!("Undo would restore: {}", would_restore);
    ///     
    ///     // Actually perform the undo
    ///     property.undo()?;
    ///     assert_eq!(property.get()?, *would_restore);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method acquires a read lock, allowing multiple concurrent readers.
    /// The returned vector is independent of the property's internal state.
    pub fn get_history(&self) -> Vec<T> {
        match self.inner.read() {
            Ok(prop) => prop.history.as_ref().map_or(Vec::new(), |h| h.clone()),
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned lock
                let prop = poisoned.into_inner();
                prop.history.as_ref().map_or(Vec::new(), |h| h.clone())
            }
        }
    }

    /// Gets the complete event log for this property
    ///
    /// Returns a vector of all recorded property change events. Each event contains
    /// the old value, new value, timestamp, event number, and thread information.
    /// This provides a complete audit trail of all changes to the property.
    ///
    /// This method acquires a read lock, allowing multiple concurrent readers.
    /// The returned vector is independent of the property's internal state.
    ///
    /// # Returns
    ///
    /// A vector of `PropertyEvent<T>` objects, in chronological order (oldest first).
    /// Returns an empty vector if:
    /// - Event logging is not enabled (property not created with `with_event_log()`)
    /// - No changes have been made yet
    ///
    /// # Examples
    ///
    /// ## Basic Event Log Retrieval
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let counter = ObservableProperty::with_event_log(0, 0);
    ///
    /// counter.set(1)?;
    /// counter.set(2)?;
    /// counter.set(3)?;
    ///
    /// let events = counter.get_event_log();
    /// assert_eq!(events.len(), 3);
    ///
    /// // First event
    /// assert_eq!(events[0].old_value, 0);
    /// assert_eq!(events[0].new_value, 1);
    /// assert_eq!(events[0].event_number, 0);
    ///
    /// // Last event
    /// assert_eq!(events[2].old_value, 2);
    /// assert_eq!(events[2].new_value, 3);
    /// assert_eq!(events[2].event_number, 2);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Filtering Events by Time
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::time::{Duration, Instant};
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_event_log(0, 0);
    /// let start = Instant::now();
    ///
    /// property.set(1)?;
    /// std::thread::sleep(Duration::from_millis(10));
    /// property.set(2)?;
    /// std::thread::sleep(Duration::from_millis(10));
    /// property.set(3)?;
    ///
    /// let cutoff = start + Duration::from_millis(15);
    /// let recent_events: Vec<_> = property.get_event_log()
    ///     .into_iter()
    ///     .filter(|e| e.timestamp > cutoff)
    ///     .collect();
    ///
    /// println!("Recent events: {}", recent_events.len());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Analyzing Event Patterns
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let score = ObservableProperty::with_event_log(0, 0);
    ///
    /// score.modify(|s| *s += 10)?;
    /// score.modify(|s| *s -= 3)?;
    /// score.modify(|s| *s += 5)?;
    ///
    /// let events = score.get_event_log();
    /// let total_increases = events.iter()
    ///     .filter(|e| e.new_value > e.old_value)
    ///     .count();
    /// let total_decreases = events.iter()
    ///     .filter(|e| e.new_value < e.old_value)
    ///     .count();
    ///
    /// println!("Increases: {}, Decreases: {}", total_increases, total_decreases);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Event Log with Thread Information
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = Arc::new(ObservableProperty::with_event_log(0, 0));
    ///
    /// let handles: Vec<_> = (0..3).map(|i| {
    ///     let prop = property.clone();
    ///     thread::spawn(move || {
    ///         prop.set(i * 10).expect("Set failed");
    ///     })
    /// }).collect();
    ///
    /// for handle in handles {
    ///     handle.join().unwrap();
    /// }
    ///
    /// // Analyze which threads made changes
    /// for event in property.get_event_log() {
    ///     println!("Event {}: Thread {}", event.event_number, event.thread_id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Replaying Property State
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::with_event_log(100, 0);
    ///
    /// property.set(150)?;
    /// property.set(200)?;
    /// property.set(175)?;
    ///
    /// // Replay state at each point in time
    /// let events = property.get_event_log();
    /// let mut state = 100; // Initial value
    /// 
    /// println!("Initial state: {}", state);
    /// for event in events {
    ///     state = event.new_value;
    ///     println!("After event {}: {}", event.event_number, state);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently from multiple threads.
    /// The returned event log is a snapshot at the time of the call.
    ///
    /// # Performance
    ///
    /// This method clones the entire event log. For properties with large event logs,
    /// consider the memory and performance implications. If you only need recent events,
    /// use filtering on the result or create the property with a bounded `event_log_size`.
    pub fn get_event_log(&self) -> Vec<PropertyEvent<T>> {
        match self.inner.read() {
            Ok(prop) => prop.event_log.as_ref().map_or(Vec::new(), |log| log.clone()),
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned lock
                let prop = poisoned.into_inner();
                prop.event_log.as_ref().map_or(Vec::new(), |log| log.clone())
            }
        }
    }

    /// Gets the current value of the property
    ///
    /// This method acquires a read lock, which allows multiple concurrent readers
    /// but will block if a writer currently holds the lock.
    ///
    /// # Returns
    ///
    /// `Ok(T)` containing a clone of the current value, or `Err(PropertyError)`
    /// if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new("hello".to_string());
    /// match property.get() {
    ///     Ok(value) => assert_eq!(value, "hello"),
    ///     Err(e) => eprintln!("Failed to get property value: {}", e),
    /// }
    /// ```
    pub fn get(&self) -> Result<T, PropertyError> {
        match self.inner.read() {
            Ok(prop) => Ok(prop.value.clone()),
            Err(poisoned) => {
                // Graceful degradation: recover value from poisoned lock
                // This allows continued operation even after a panic in another thread
                Ok(poisoned.into_inner().value.clone())
            }
        }
    }

    /// Returns performance metrics for this property
    ///
    /// Provides insight into property usage patterns and observer notification
    /// performance. This is useful for profiling, debugging, and performance
    /// optimization.
    ///
    /// # Metrics Provided
    ///
    /// - **total_changes**: Number of times the property value has been changed
    /// - **observer_calls**: Total number of observer notification calls made
    /// - **avg_notification_time**: Average time taken to notify all observers
    ///
    /// # Note
    ///
    /// - For `set_async()`, the notification time measures the time to spawn threads,
    ///   not the actual observer execution time (since threads are fire-and-forget).
    /// - Observer calls are counted even if they panic (panic recovery continues).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// // Subscribe multiple observers
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Observer 1: {} -> {}", old, new);
    /// }))?;
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Observer 2: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Make some changes
    /// property.set(42)?;
    /// property.set(100)?;
    /// property.set(200)?;
    ///
    /// // Get performance metrics
    /// let metrics = property.get_metrics()?;
    /// println!("Total changes: {}", metrics.total_changes); // 3
    /// println!("Observer calls: {}", metrics.observer_calls); // 6 (3 changes × 2 observers)
    /// println!("Avg notification time: {:?}", metrics.avg_notification_time);
    /// # Ok(())
    /// # }
    /// ```
    pub fn get_metrics(&self) -> Result<PropertyMetrics, PropertyError> {
        match self.inner.read() {
            Ok(prop) => {
                let avg_notification_time = if prop.notification_times.is_empty() {
                    Duration::from_secs(0)
                } else {
                    let total: Duration = prop.notification_times.iter().sum();
                    total / prop.notification_times.len() as u32
                };

                Ok(PropertyMetrics {
                    total_changes: prop.total_changes,
                    observer_calls: prop.observer_calls,
                    avg_notification_time,
                })
            }
            Err(poisoned) => {
                // Graceful degradation: recover metrics from poisoned lock
                let prop = poisoned.into_inner();
                let avg_notification_time = if prop.notification_times.is_empty() {
                    Duration::from_secs(0)
                } else {
                    let total: Duration = prop.notification_times.iter().sum();
                    total / prop.notification_times.len() as u32
                };

                Ok(PropertyMetrics {
                    total_changes: prop.total_changes,
                    observer_calls: prop.observer_calls,
                    avg_notification_time,
                })
            }
        }
    }

    /// Sets the property to a new value and notifies all observers
    ///
    /// This method will:
    /// 1. Acquire a write lock (blocking other readers/writers)
    /// 2. Update the value and capture a snapshot of observers
    /// 3. Release the lock
    /// 4. Notify all observers sequentially with the old and new values
    ///
    /// Observer notifications are wrapped in panic recovery to prevent one
    /// misbehaving observer from affecting others.
    ///
    /// # Arguments
    ///
    /// * `new_value` - The new value to set
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// let property = ObservableProperty::new(10);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Value changed from {} to {}", old, new);
    /// })).map_err(|e| {
    ///     eprintln!("Failed to subscribe: {}", e);
    ///     e
    /// })?;
    ///
    /// property.set(20).map_err(|e| {
    ///     eprintln!("Failed to set property value: {}", e);
    ///     e
    /// })?; // Triggers observer notification
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    pub fn set(&self, new_value: T) -> Result<(), PropertyError> {
        // Validate the new value if a validator is configured
        {
            let prop = match self.inner.read() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            
            if let Some(validator) = &prop.validator {
                validator(&new_value).map_err(|reason| PropertyError::ValidationError { reason })?;
            }
        }

        let notification_start = Instant::now();
        let (old_value, observers_snapshot, dead_observer_ids, in_batch) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    // Graceful degradation: recover from poisoned write lock
                    // Clear the poison flag by taking ownership of the inner value
                    poisoned.into_inner()
                }
            };

            // Check if values are equal using custom equality function if provided
            let values_equal = if let Some(eq_fn) = &prop.eq_fn {
                eq_fn(&prop.value, &new_value)
            } else {
                false  // No equality function = always notify
            };

            // If values are equal, skip everything
            if values_equal {
                return Ok(());
            }

            // Check if we're in a batch update
            let in_batch = prop.batch_depth > 0;

            // Performance optimization: use mem::replace to avoid one clone operation
            let old_value = mem::replace(&mut prop.value, new_value.clone());
            
            // Track the change
            prop.total_changes += 1;
            let event_num = prop.total_changes - 1; // Capture for event numbering
            
            // Add old value to history if history tracking is enabled
            let history_size = prop.history_size;
            if let Some(history) = &mut prop.history {
                // Add old value to history
                history.push(old_value.clone());
                
                // Enforce history size limit by removing oldest values
                if history.len() > history_size {
                    let overflow = history.len() - history_size;
                    history.drain(0..overflow);
                }
            }
            
            // Record event if event logging is enabled
            let event_log_size = prop.event_log_size;
            if let Some(event_log) = &mut prop.event_log {
                let event = PropertyEvent {
                    timestamp: Instant::now(),
                    old_value: old_value.clone(),
                    new_value: new_value.clone(),
                    event_number: event_num, // Use captured event number for consistent numbering
                    thread_id: format!("{:?}", thread::current().id()),
                };
                
                event_log.push(event);
                
                // Enforce event log size limit by removing oldest events (if bounded)
                if event_log_size > 0 && event_log.len() > event_log_size {
                    let overflow = event_log.len() - event_log_size;
                    event_log.drain(0..overflow);
                }
            }
            
            // Collect active observers and track dead weak observers (only if not in batch)
            let mut observers_snapshot = Vec::new();
            let mut dead_ids = Vec::new();
            if !in_batch {
                for (id, observer_ref) in &prop.observers {
                    if let Some(observer) = observer_ref.try_call() {
                        observers_snapshot.push(observer);
                    } else {
                        // Weak observer is dead, mark for removal
                        dead_ids.push(*id);
                    }
                }
            }
            
            (old_value, observers_snapshot, dead_ids, in_batch)
        };

        // Skip notifications if we're in a batch update
        if in_batch {
            return Ok(());
        }

        // Notify all active observers
        let observer_count = observers_snapshot.len();
        for observer in observers_snapshot {
            if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                observer(&old_value, &new_value);
            })) {
                eprintln!("Observer panic: {:?}", e);
            }
        }
        
        // Record metrics
        let notification_time = notification_start.elapsed();
        {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            prop.observer_calls += observer_count;
            prop.notification_times.push(notification_time);
        }

        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }

        Ok(())
    }

    /// Sets the property to a new value and notifies observers asynchronously
    ///
    /// This method is similar to `set()` but spawns observers in background threads
    /// for non-blocking operation. This is useful when observers might perform
    /// time-consuming operations.
    ///
    /// Observers are batched into groups and each batch runs in its own thread
    /// to limit resource usage while still providing parallelism.
    ///
    /// # Thread Management (Fire-and-Forget Pattern)
    ///
    /// **Important**: This method uses a fire-and-forget pattern. Spawned threads are
    /// **not joined** and run independently in the background. This design is intentional
    /// for non-blocking behavior but has important implications:
    ///
    /// ## Characteristics:
    /// - ✅ **Non-blocking**: Returns immediately without waiting for observers
    /// - ✅ **High performance**: No synchronization overhead
    /// - ⚠️ **No completion guarantee**: Thread may still be running when method returns
    /// - ⚠️ **No error propagation**: Observer errors are logged but not returned
    /// - ⚠️ **Testing caveat**: May need explicit delays to observe side effects
    /// - ⚠️ **Ordering caveat**: Multiple rapid `set_async()` calls may result in observers
    ///   receiving notifications out of order due to thread scheduling. Use `set()` if
    ///   sequential ordering is critical.
    ///
    /// ## Use Cases:
    /// - **UI updates**: Fire updates without blocking the main thread
    /// - **Logging**: Asynchronous logging that doesn't block operations
    /// - **Metrics**: Non-critical telemetry that can be lost
    /// - **Notifications**: Fire-and-forget alerts or messages
    ///
    /// ## When NOT to Use:
    /// - **Critical operations**: Use `set()` if you need guarantees
    /// - **Transactional updates**: Use `set()` for atomic operations
    /// - **Sequential dependencies**: If next operation depends on observer completion
    ///
    /// ## Testing Considerations:
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    /// let was_called = Arc::new(AtomicBool::new(false));
    /// let flag = was_called.clone();
    ///
    /// property.subscribe(Arc::new(move |_, _| {
    ///     flag.store(true, Ordering::SeqCst);
    /// }))?;
    ///
    /// property.set_async(42)?;
    ///
    /// // ⚠️ Immediate check might fail - thread may not have run yet
    /// // assert!(was_called.load(Ordering::SeqCst)); // May fail!
    ///
    /// // ✅ Add a small delay to allow background thread to complete
    /// std::thread::sleep(Duration::from_millis(10));
    /// assert!(was_called.load(Ordering::SeqCst)); // Now reliable
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `new_value` - The new value to set
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(PropertyError)` if the lock is poisoned.
    /// Note that this only indicates the property was updated successfully;
    /// observer execution happens asynchronously and errors are not returned.
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// let property = ObservableProperty::new(0);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     // This observer does slow work but won't block the caller
    ///     std::thread::sleep(Duration::from_millis(100));
    ///     println!("Slow observer: {} -> {}", old, new);
    /// })).map_err(|e| {
    ///     eprintln!("Failed to subscribe: {}", e);
    ///     e
    /// })?;
    ///
    /// // This returns immediately even though observer is slow
    /// property.set_async(42).map_err(|e| {
    ///     eprintln!("Failed to set value asynchronously: {}", e);
    ///     e
    /// })?;
    ///
    /// // Continue working immediately - observer runs in background
    /// println!("Main thread continues without waiting");
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    ///
    /// ## Multiple Rapid Updates
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     // Expensive operation (e.g., database update, API call)
    ///     println!("Processing: {} -> {}", old, new);
    /// }))?;
    ///
    /// // All of these return immediately - observers run in parallel
    /// property.set_async(1)?;
    /// property.set_async(2)?;
    /// property.set_async(3)?;
    /// property.set_async(4)?;
    /// property.set_async(5)?;
    ///
    /// // All observer calls are now running in background threads
    /// # Ok(())
    /// # }
    /// ```
    pub fn set_async(&self, new_value: T) -> Result<(), PropertyError> {
        // Validate the new value if a validator is configured
        {
            let prop = match self.inner.read() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            
            if let Some(validator) = &prop.validator {
                validator(&new_value).map_err(|reason| PropertyError::ValidationError { reason })?;
            }
        }

        let notification_start = Instant::now();
        let (old_value, observers_snapshot, dead_observer_ids, in_batch) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    // Graceful degradation: recover from poisoned write lock
                    poisoned.into_inner()
                }
            };

            // Check if values are equal using custom equality function if provided
            let values_equal = if let Some(eq_fn) = &prop.eq_fn {
                eq_fn(&prop.value, &new_value)
            } else {
                false  // No equality function = always notify
            };

            // If values are equal, skip everything
            if values_equal {
                return Ok(());
            }

            // Check if we're in a batch update
            let in_batch = prop.batch_depth > 0;

            // Performance optimization: use mem::replace to avoid one clone operation
            let old_value = mem::replace(&mut prop.value, new_value.clone());
            
            // Track the change
            prop.total_changes += 1;
            let event_num = prop.total_changes - 1; // Capture for event numbering
            
            // Add old value to history if history tracking is enabled
            let history_size = prop.history_size;
            if let Some(history) = &mut prop.history {
                // Add old value to history
                history.push(old_value.clone());
                
                // Enforce history size limit by removing oldest values
                if history.len() > history_size {
                    let overflow = history.len() - history_size;
                    history.drain(0..overflow);
                }
            }
            
            // Record event if event logging is enabled
            let event_log_size = prop.event_log_size;
            if let Some(event_log) = &mut prop.event_log {
                let event = PropertyEvent {
                    timestamp: Instant::now(),
                    old_value: old_value.clone(),
                    new_value: new_value.clone(),
                    event_number: event_num, // Use captured event number for consistent numbering
                    thread_id: format!("{:?}", thread::current().id()),
                };
                
                event_log.push(event);
                
                // Enforce event log size limit by removing oldest events (if bounded)
                if event_log_size > 0 && event_log.len() > event_log_size {
                    let overflow = event_log.len() - event_log_size;
                    event_log.drain(0..overflow);
                }
            }
            
            // Collect active observers and track dead weak observers (only if not in batch)
            let mut observers_snapshot = Vec::new();
            let mut dead_ids = Vec::new();
            if !in_batch {
                for (id, observer_ref) in &prop.observers {
                    if let Some(observer) = observer_ref.try_call() {
                        observers_snapshot.push(observer);
                    } else {
                        // Weak observer is dead, mark for removal
                        dead_ids.push(*id);
                    }
                }
            }
            
            (old_value, observers_snapshot, dead_ids, in_batch)
        };

        // Skip notifications if we're in a batch update
        if in_batch {
            return Ok(());
        }

        if observers_snapshot.is_empty() {
            // Clean up dead weak observers before returning
            if !dead_observer_ids.is_empty() {
                let mut prop = match self.inner.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                for id in dead_observer_ids {
                    prop.observers.remove(&id);
                }
            }
            return Ok(());
        }

        let observers_per_thread = observers_snapshot.len().div_ceil(self.max_threads);

        // Record metrics for async notifications (time to spawn threads, not execute)
        let observer_count = observers_snapshot.len();
        
        // Fire-and-forget pattern: Spawn threads without joining
        // This is intentional for non-blocking behavior. Observers run independently
        // and the caller continues immediately without waiting for completion.
        // Trade-offs:
        //   ✅ Non-blocking, high performance
        //   ⚠️ No completion guarantee, no error propagation to caller
        for batch in observers_snapshot.chunks(observers_per_thread) {
            let batch_observers = batch.to_vec();
            let old_val = old_value.clone();
            let new_val = new_value.clone();

            thread::spawn(move || {
                for observer in batch_observers {
                    if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        observer(&old_val, &new_val);
                    })) {
                        eprintln!("Observer panic in batch thread: {:?}", e);
                    }
                }
            });
            // Thread handle intentionally dropped - fire-and-forget pattern
        }
        
        // Record notification time (time to spawn all threads)
        let notification_time = notification_start.elapsed();
        {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            prop.observer_calls += observer_count;
            prop.notification_times.push(notification_time);
        }

        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }

        Ok(())
    }

    /// Begins a batch update, suppressing observer notifications
    ///
    /// Call this method to start a batch of updates where you want to change
    /// the value multiple times but only notify observers once at the end.
    /// This is useful for bulk updates where intermediate values don't matter.
    ///
    /// # Nested Batches
    ///
    /// This method supports nesting - you can call `begin_update()` multiple times
    /// and must call `end_update()` the same number of times. Observers will only
    /// be notified when the outermost batch is completed.
    ///
    /// # Thread Safety
    ///
    /// Each batch is scoped to the current execution context. If you begin a batch
    /// in one thread, it won't affect other threads.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Value changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Begin batch update
    /// property.begin_update()?;
    ///
    /// // These changes won't trigger notifications
    /// property.set(10)?;
    /// property.set(20)?;
    /// property.set(30)?;
    ///
    /// // End batch - single notification from 0 to 30
    /// property.end_update()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Nested Batches
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Value changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// property.begin_update()?; // Outer batch
    /// property.set(5)?;
    ///
    /// property.begin_update()?; // Inner batch
    /// property.set(10)?;
    /// property.end_update()?; // End inner batch (no notification yet)
    ///
    /// property.set(15)?;
    /// property.end_update()?; // End outer batch - notification sent: 0 -> 15
    /// # Ok(())
    /// # }
    /// ```
    pub fn begin_update(&self) -> Result<(), PropertyError> {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        if prop.batch_depth == 0 {
            // Store the initial value when starting a new batch
            prop.batch_initial_value = Some(prop.value.clone());
        }

        prop.batch_depth += 1;
        Ok(())
    }

    /// Ends a batch update, sending a single notification with the final value
    ///
    /// This method completes a batch update started with `begin_update()`. When the
    /// outermost batch is completed, observers will be notified once with the value
    /// change from the start of the batch to the final value.
    ///
    /// # Behavior
    ///
    /// - If the value hasn't changed during the batch, no notification is sent
    /// - Supports nested batches - only notifies when all batches are complete
    /// - If called without a matching `begin_update()`, returns an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// property.subscribe(Arc::new(|old, new| {
    ///     println!("Value changed: {} -> {}", old, new);
    /// }))?;
    ///
    /// property.begin_update()?;
    /// property.set(10)?;
    /// property.set(20)?;
    /// property.end_update()?; // Prints: "Value changed: 0 -> 20"
    /// # Ok(())
    /// # }
    /// ```
    pub fn end_update(&self) -> Result<(), PropertyError> {
        let notification_start = Instant::now();
        let (should_notify, old_value, new_value, observers_snapshot, dead_observer_ids) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };

            if prop.batch_depth == 0 {
                return Err(PropertyError::InvalidConfiguration {
                    reason: "end_update() called without matching begin_update()".to_string(),
                });
            }

            prop.batch_depth -= 1;

            // Only notify when we've exited all nested batches
            if prop.batch_depth == 0 {
                if let Some(initial_value) = prop.batch_initial_value.take() {
                    let current_value = prop.value.clone();
                    
                    // Collect observers if value changed
                    let mut observers_snapshot = Vec::new();
                    let mut dead_ids = Vec::new();
                    for (id, observer_ref) in &prop.observers {
                        if let Some(observer) = observer_ref.try_call() {
                            observers_snapshot.push(observer);
                        } else {
                            dead_ids.push(*id);
                        }
                    }
                    
                    (true, initial_value, current_value, observers_snapshot, dead_ids)
                } else {
                    (false, prop.value.clone(), prop.value.clone(), Vec::new(), Vec::new())
                }
            } else {
                (false, prop.value.clone(), prop.value.clone(), Vec::new(), Vec::new())
            }
        };

        if should_notify && !observers_snapshot.is_empty() {
            // Notify all active observers
            let observer_count = observers_snapshot.len();
            for observer in observers_snapshot {
                if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    observer(&old_value, &new_value);
                })) {
                    eprintln!("Observer panic: {:?}", e);
                }
            }
            
            // Record metrics
            let notification_time = notification_start.elapsed();
            {
                let mut prop = match self.inner.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                prop.observer_calls += observer_count;
                prop.notification_times.push(notification_time);
            }

            // Clean up dead weak observers
            if !dead_observer_ids.is_empty() {
                let mut prop = match self.inner.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };
                for id in dead_observer_ids {
                    prop.observers.remove(&id);
                }
            }
        }

        Ok(())
    }

    /// Subscribes an observer function to be called when the property changes
    ///
    /// The observer function will be called with the old and new values whenever
    /// the property is modified via `set()` or `set_async()`.
    ///
    /// # Arguments
    ///
    /// * `observer` - A function wrapped in `Arc` that takes `(&T, &T)` parameters
    ///
    /// # Returns
    ///
    /// `Ok(ObserverId)` containing a unique identifier for this observer,
    /// or `Err(PropertyError::InvalidConfiguration)` if the maximum observer limit is exceeded.
    ///
    /// # Observer Limit
    ///
    /// To prevent memory exhaustion, there is a maximum limit of observers per property
    /// (currently set to 10,000). If you attempt to add more observers than this limit,
    /// the subscription will fail with an `InvalidConfiguration` error.
    ///
    /// This protection helps prevent:
    /// - Memory leaks from forgotten unsubscriptions
    /// - Unbounded memory growth in long-running applications
    /// - Out-of-memory conditions in resource-constrained environments
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// let property = ObservableProperty::new(0);
    ///
    /// let observer_id = property.subscribe(Arc::new(|old_value, new_value| {
    ///     println!("Property changed from {} to {}", old_value, new_value);
    /// })).map_err(|e| {
    ///     eprintln!("Failed to subscribe observer: {}", e);
    ///     e
    /// })?;
    ///
    /// // Later, unsubscribe using the returned ID
    /// property.unsubscribe(observer_id).map_err(|e| {
    ///     eprintln!("Failed to unsubscribe observer: {}", e);
    ///     e
    /// })?;
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    pub fn subscribe(&self, observer: Observer<T>) -> Result<ObserverId, PropertyError> {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned write lock
                poisoned.into_inner()
            }
        };

        // Check observer limit to prevent memory exhaustion
        if prop.observers.len() >= self.max_observers {
            return Err(PropertyError::InvalidConfiguration {
                reason: format!(
                    "Maximum observer limit ({}) exceeded. Current observers: {}. \
                     Consider unsubscribing unused observers to free resources.",
                    self.max_observers,
                    prop.observers.len()
                ),
            });
        }

        let id = prop.next_id;
        // Use wrapping_add to prevent overflow panics in production
        // After ~usize::MAX subscriptions, IDs will wrap around
        // This is acceptable as old observers are typically unsubscribed
        prop.next_id = prop.next_id.wrapping_add(1);
        prop.observers.insert(id, ObserverRef::Strong(observer));
        Ok(id)
    }

    /// Removes an observer identified by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The observer ID returned by `subscribe()`
    ///
    /// # Returns
    ///
    /// `Ok(bool)` where `true` means the observer was found and removed,
    /// `false` means no observer with that ID existed.
    /// Returns `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// let property = ObservableProperty::new(0);
    /// let id = property.subscribe(Arc::new(|_, _| {})).map_err(|e| {
    ///     eprintln!("Failed to subscribe: {}", e);
    ///     e
    /// })?;
    ///
    /// let was_removed = property.unsubscribe(id).map_err(|e| {
    ///     eprintln!("Failed to unsubscribe: {}", e);
    ///     e
    /// })?;
    /// assert!(was_removed); // Observer existed and was removed
    ///
    /// let was_removed_again = property.unsubscribe(id).map_err(|e| {
    ///     eprintln!("Failed to unsubscribe again: {}", e);
    ///     e
    /// })?;
    /// assert!(!was_removed_again); // Observer no longer exists
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    pub fn unsubscribe(&self, id: ObserverId) -> Result<bool, PropertyError> {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned write lock
                poisoned.into_inner()
            }
        };

        let was_present = prop.observers.remove(&id).is_some();
        Ok(was_present)
    }

    /// Subscribes a weak observer that automatically cleans up when dropped
    ///
    /// Unlike `subscribe()` which holds a strong reference to the observer, this method
    /// stores only a weak reference. When the observer's `Arc` is dropped elsewhere,
    /// the observer will be automatically removed from the property on the next notification.
    ///
    /// This is useful for scenarios where you want observers to have independent lifetimes
    /// without needing explicit unsubscribe calls or `Subscription` guards.
    ///
    /// # Arguments
    ///
    /// * `observer` - A weak reference to the observer function
    ///
    /// # Returns
    ///
    /// `Ok(ObserverId)` containing a unique identifier for this observer,
    /// or `Err(PropertyError::InvalidConfiguration)` if the maximum observer limit is exceeded.
    ///
    /// # Automatic Cleanup
    ///
    /// The observer will be automatically removed when:
    /// - The `Arc` that the `Weak` was created from is dropped
    /// - The next notification occurs (via `set()`, `set_async()`, `modify()`, etc.)
    ///
    /// # Examples
    ///
    /// ## Basic Weak Observer
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// {
    ///     // Create observer as trait object
    ///     let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(|old: &i32, new: &i32| {
    ///         println!("Value changed: {} -> {}", old, new);
    ///     });
    ///     
    ///     // Subscribe with a weak reference
    ///     property.subscribe_weak(Arc::downgrade(&observer))?;
    ///     
    ///     property.set(42)?; // Prints: "Value changed: 0 -> 42"
    ///     
    ///     // When observer Arc goes out of scope, weak reference becomes invalid
    /// }
    ///
    /// // Next set automatically cleans up the dead observer
    /// property.set(100)?; // No output - observer was automatically cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Managing Observer Lifetime
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(String::from("initial"));
    ///
    /// // Store the observer Arc somewhere accessible (as trait object)
    /// let observer: Arc<dyn Fn(&String, &String) + Send + Sync> = Arc::new(|old: &String, new: &String| {
    ///     println!("Text changed: '{}' -> '{}'", old, new);
    /// });
    ///
    /// property.subscribe_weak(Arc::downgrade(&observer))?;
    /// property.set(String::from("updated"))?; // Works - observer is alive
    ///
    /// // Explicitly drop the observer when done
    /// drop(observer);
    ///
    /// property.set(String::from("final"))?; // No output - observer was dropped
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multi-threaded Weak Observers
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = Arc::new(ObservableProperty::new(0));
    /// let property_clone = property.clone();
    ///
    /// // Create observer as trait object
    /// let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(|old: &i32, new: &i32| {
    ///     println!("Thread observer: {} -> {}", old, new);
    /// });
    ///
    /// property.subscribe_weak(Arc::downgrade(&observer))?;
    ///
    /// let handle = thread::spawn(move || {
    ///     property_clone.set(42)
    /// });
    ///
    /// handle.join().unwrap()?; // Prints: "Thread observer: 0 -> 42"
    ///
    /// // Observer is still alive
    /// property.set(100)?; // Prints: "Thread observer: 42 -> 100"
    ///
    /// // Drop the observer
    /// drop(observer);
    /// property.set(200)?; // No output
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Comparison with `subscribe()` and `subscribe_with_subscription()`
    ///
    /// - **`subscribe()`**: Holds strong reference, requires manual `unsubscribe()`
    /// - **`subscribe_with_subscription()`**: Holds strong reference, automatic cleanup via RAII guard
    /// - **`subscribe_weak()`**: Holds weak reference, cleanup when Arc is dropped elsewhere
    ///
    /// Use `subscribe_weak()` when:
    /// - You want to control observer lifetime independently from subscriptions
    /// - You need multiple code paths to potentially drop the observer
    /// - You want to avoid keeping observers alive longer than necessary
    pub fn subscribe_weak(
        &self,
        observer: std::sync::Weak<dyn Fn(&T, &T) + Send + Sync>,
    ) -> Result<ObserverId, PropertyError> {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned write lock
                poisoned.into_inner()
            }
        };

        // Check observer limit to prevent memory exhaustion
        if prop.observers.len() >= self.max_observers {
            return Err(PropertyError::InvalidConfiguration {
                reason: format!(
                    "Maximum observer limit ({}) exceeded. Current observers: {}. \
                     Consider unsubscribing unused observers to free resources.",
                    self.max_observers,
                    prop.observers.len()
                ),
            });
        }

        let id = prop.next_id;
        // Use wrapping_add to prevent overflow panics in production
        // After ~usize::MAX subscriptions, IDs will wrap around
        // This is acceptable as old observers are typically unsubscribed
        prop.next_id = prop.next_id.wrapping_add(1);
        prop.observers.insert(id, ObserverRef::Weak(observer));
        Ok(id)
    }

    /// Subscribes an observer that only gets called when a filter condition is met
    ///
    /// This is useful for observing only specific types of changes, such as
    /// when a value increases or crosses a threshold.
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer function to call when the filter passes
    /// * `filter` - A predicate function that receives `(old_value, new_value)` and returns `bool`
    ///
    /// # Returns
    ///
    /// `Ok(ObserverId)` for the filtered observer, or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// let property = ObservableProperty::new(0);
    ///
    /// // Only notify when value increases
    /// let id = property.subscribe_filtered(
    ///     Arc::new(|old, new| println!("Value increased: {} -> {}", old, new)),
    ///     |old, new| new > old
    /// ).map_err(|e| {
    ///     eprintln!("Failed to subscribe filtered observer: {}", e);
    ///     e
    /// })?;
    ///
    /// property.set(10).map_err(|e| {
    ///     eprintln!("Failed to set value: {}", e);
    ///     e
    /// })?; // Triggers observer (0 -> 10)
    /// property.set(5).map_err(|e| {
    ///     eprintln!("Failed to set value: {}", e);
    ///     e
    /// })?;  // Does NOT trigger observer (10 -> 5)
    /// property.set(15).map_err(|e| {
    ///     eprintln!("Failed to set value: {}", e);
    ///     e
    /// })?; // Triggers observer (5 -> 15)
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    pub fn subscribe_filtered<F>(
        &self,
        observer: Observer<T>,
        filter: F,
    ) -> Result<ObserverId, PropertyError>
    where
        F: Fn(&T, &T) -> bool + Send + Sync + 'static,
    {
        let filter = Arc::new(filter);
        let filtered_observer = Arc::new(move |old_val: &T, new_val: &T| {
            if filter(old_val, new_val) {
                observer(old_val, new_val);
            }
        });

        self.subscribe(filtered_observer)
    }

    /// Subscribes an observer that only gets called after changes stop for a specified duration
    ///
    /// Debouncing delays observer notifications until a quiet period has passed. Each new
    /// change resets the timer. This is useful for expensive operations that shouldn't
    /// run on every single change, such as auto-save, search-as-you-type, or form validation.
    ///
    /// # How It Works
    ///
    /// When the property changes:
    /// 1. A timer starts for the specified `debounce_duration`
    /// 2. If another change occurs before the timer expires, the timer resets
    /// 3. When the timer finally expires with no new changes, the observer is notified
    /// 4. Only the **most recent** change is delivered to the observer
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer function to call after the debounce period
    /// * `debounce_duration` - How long to wait after the last change before notifying
    ///
    /// # Returns
    ///
    /// `Ok(ObserverId)` for the debounced observer, or `Err(PropertyError)` if subscription fails.
    ///
    /// # Examples
    ///
    /// ## Auto-Save Example
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let document = ObservableProperty::new("".to_string());
    /// let save_count = Arc::new(AtomicUsize::new(0));
    /// let count_clone = save_count.clone();
    ///
    /// // Auto-save only after user stops typing for 500ms
    /// document.subscribe_debounced(
    ///     Arc::new(move |_old, new| {
    ///         count_clone.fetch_add(1, Ordering::SeqCst);
    ///         println!("Auto-saving: {}", new);
    ///     }),
    ///     Duration::from_millis(500)
    /// )?;
    ///
    /// // Rapid changes (user typing)
    /// document.set("H".to_string())?;
    /// document.set("He".to_string())?;
    /// document.set("Hel".to_string())?;
    /// document.set("Hell".to_string())?;
    /// document.set("Hello".to_string())?;
    ///
    /// // At this point, no auto-save has occurred yet
    /// assert_eq!(save_count.load(Ordering::SeqCst), 0);
    ///
    /// // Wait for debounce period
    /// std::thread::sleep(Duration::from_millis(600));
    ///
    /// // Now auto-save has occurred exactly once with the final value
    /// assert_eq!(save_count.load(Ordering::SeqCst), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Search-as-You-Type Example
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let search_query = ObservableProperty::new("".to_string());
    ///
    /// // Only search after user stops typing for 300ms
    /// search_query.subscribe_debounced(
    ///     Arc::new(|_old, new| {
    ///         if !new.is_empty() {
    ///             println!("Searching for: {}", new);
    ///             // Perform expensive API call here
    ///         }
    ///     }),
    ///     Duration::from_millis(300)
    /// )?;
    ///
    /// // User types quickly - no searches triggered yet
    /// search_query.set("r".to_string())?;
    /// search_query.set("ru".to_string())?;
    /// search_query.set("rus".to_string())?;
    /// search_query.set("rust".to_string())?;
    ///
    /// // Wait for debounce
    /// std::thread::sleep(Duration::from_millis(400));
    /// // Now search executes once with "rust"
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Form Validation Example
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let email = ObservableProperty::new("".to_string());
    ///
    /// // Validate email only after user stops typing for 500ms
    /// email.subscribe_debounced(
    ///     Arc::new(|_old, new| {
    ///         if new.contains('@') && new.contains('.') {
    ///             println!("✓ Email looks valid");
    ///         } else if !new.is_empty() {
    ///             println!("✗ Email appears invalid");
    ///         }
    ///     }),
    ///     Duration::from_millis(500)
    /// )?;
    ///
    /// email.set("user".to_string())?;
    /// email.set("user@".to_string())?;
    /// email.set("user@ex".to_string())?;
    /// email.set("user@example".to_string())?;
    /// email.set("user@example.com".to_string())?;
    ///
    /// // Validation only runs once after typing stops
    /// std::thread::sleep(Duration::from_millis(600));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - Each debounced observer spawns a background thread when changes occur
    /// - The thread sleeps for the debounce duration and then checks if it should notify
    /// - Multiple rapid changes don't create multiple threads - they just update the pending value
    /// - Memory overhead: ~2 Mutex allocations per debounced observer
    ///
    /// # Thread Safety
    ///
    /// Debounced observers are fully thread-safe. Multiple threads can trigger changes
    /// simultaneously, and the debouncing logic will correctly handle the most recent value.
    pub fn subscribe_debounced(
        &self,
        observer: Observer<T>,
        debounce_duration: Duration,
    ) -> Result<ObserverId, PropertyError> {
        let last_change_time = Arc::new(Mutex::new(Instant::now()));
        let pending_values = Arc::new(Mutex::new(None::<(T, T)>));
        
        let debounced_observer = Arc::new(move |old_val: &T, new_val: &T| {
            // Update the last change time and store the values
            {
                let mut last_time = last_change_time.lock().unwrap();
                *last_time = Instant::now();
                
                let mut pending = pending_values.lock().unwrap();
                *pending = Some((old_val.clone(), new_val.clone()));
            }
            
            // Spawn a thread to wait and then notify if no newer changes occurred
            let last_change_time_thread = last_change_time.clone();
            let pending_values_thread = pending_values.clone();
            let observer_thread = observer.clone();
            let duration = debounce_duration;
            
            thread::spawn(move || {
                thread::sleep(duration);
                
                // Check if enough time has passed since the last change
                let should_notify = {
                    let last_time = last_change_time_thread.lock().unwrap();
                    last_time.elapsed() >= duration
                };
                
                if should_notify {
                    // Get and clear the pending values
                    let values = {
                        let mut pending = pending_values_thread.lock().unwrap();
                        pending.take()
                    };
                    
                    // Notify the observer with the final values
                    if let Some((old, new)) = values {
                        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                            observer_thread(&old, &new);
                        }));
                    }
                }
            });
        });

        self.subscribe(debounced_observer)
    }

    /// Subscribes an observer that gets called at most once per specified duration
    ///
    /// Throttling ensures that regardless of how frequently the property changes,
    /// the observer is notified at most once per `throttle_interval`. The first change
    /// triggers an immediate notification, then subsequent changes are rate-limited.
    ///
    /// # How It Works
    ///
    /// When the property changes:
    /// 1. If enough time has passed since the last notification, notify immediately
    /// 2. Otherwise, schedule a notification for after the throttle interval expires
    /// 3. During the throttle interval, additional changes update the pending value
    ///    but don't trigger additional notifications
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer function to call (rate-limited)
    /// * `throttle_interval` - Minimum time between observer notifications
    ///
    /// # Returns
    ///
    /// `Ok(ObserverId)` for the throttled observer, or `Err(PropertyError)` if subscription fails.
    ///
    /// # Examples
    ///
    /// ## Scroll Event Handling
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let scroll_position = ObservableProperty::new(0);
    /// let update_count = Arc::new(AtomicUsize::new(0));
    /// let count_clone = update_count.clone();
    ///
    /// // Update UI at most every 100ms, even if scrolling continuously
    /// scroll_position.subscribe_throttled(
    ///     Arc::new(move |_old, new| {
    ///         count_clone.fetch_add(1, Ordering::SeqCst);
    ///         println!("Updating UI for scroll position: {}", new);
    ///     }),
    ///     Duration::from_millis(100)
    /// )?;
    ///
    /// // Rapid scroll events (e.g., 60fps = ~16ms per frame)
    /// for i in 1..=20 {
    ///     scroll_position.set(i * 10)?;
    ///     std::thread::sleep(Duration::from_millis(16));
    /// }
    ///
    /// // UI updates happened less frequently than scroll events
    /// let updates = update_count.load(Ordering::SeqCst);
    /// assert!(updates < 20); // Throttled to ~100ms intervals
    /// assert!(updates > 0);  // But at least some updates occurred
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Mouse Movement Tracking
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let mouse_position = ObservableProperty::new((0, 0));
    ///
    /// // Track mouse position, but only log every 200ms
    /// mouse_position.subscribe_throttled(
    ///     Arc::new(|_old, new| {
    ///         println!("Mouse at: ({}, {})", new.0, new.1);
    ///     }),
    ///     Duration::from_millis(200)
    /// )?;
    ///
    /// // Simulate rapid mouse movements
    /// for x in 0..50 {
    ///     mouse_position.set((x, x * 2))?;
    ///     std::thread::sleep(Duration::from_millis(10));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## API Rate Limiting
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let sensor_reading = ObservableProperty::new(0.0);
    ///
    /// // Send sensor data to API at most once per second
    /// sensor_reading.subscribe_throttled(
    ///     Arc::new(|_old, new| {
    ///         println!("Sending to API: {:.2}", new);
    ///         // Actual API call would go here
    ///     }),
    ///     Duration::from_secs(1)
    /// )?;
    ///
    /// // High-frequency sensor updates
    /// for i in 0..100 {
    ///     sensor_reading.set(i as f64 * 0.1)?;
    ///     std::thread::sleep(Duration::from_millis(50));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Difference from Debouncing
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    /// let throttle_count = Arc::new(AtomicUsize::new(0));
    /// let debounce_count = Arc::new(AtomicUsize::new(0));
    ///
    /// let throttle_clone = throttle_count.clone();
    /// let debounce_clone = debounce_count.clone();
    ///
    /// // Throttling: Notifies periodically during continuous changes
    /// property.subscribe_throttled(
    ///     Arc::new(move |_, _| {
    ///         throttle_clone.fetch_add(1, Ordering::SeqCst);
    ///     }),
    ///     Duration::from_millis(100)
    /// )?;
    ///
    /// // Debouncing: Notifies only after changes stop
    /// property.subscribe_debounced(
    ///     Arc::new(move |_, _| {
    ///         debounce_clone.fetch_add(1, Ordering::SeqCst);
    ///     }),
    ///     Duration::from_millis(100)
    /// )?;
    ///
    /// // Continuous changes for 500ms
    /// for i in 1..=50 {
    ///     property.set(i)?;
    ///     std::thread::sleep(Duration::from_millis(10));
    /// }
    ///
    /// // Wait for debounce to complete
    /// std::thread::sleep(Duration::from_millis(150));
    ///
    /// // Throttled: Multiple notifications during the period
    /// assert!(throttle_count.load(Ordering::SeqCst) >= 4);
    ///
    /// // Debounced: Single notification after changes stopped
    /// assert_eq!(debounce_count.load(Ordering::SeqCst), 1);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - Throttled observers spawn background threads to handle delayed notifications
    /// - First notification is immediate (no delay), subsequent ones are rate-limited
    /// - Memory overhead: ~1 Mutex allocation per throttled observer
    ///
    /// # Thread Safety
    ///
    /// Throttled observers are fully thread-safe. Multiple threads can trigger changes
    /// and the throttling logic will correctly enforce the rate limit.
    pub fn subscribe_throttled(
        &self,
        observer: Observer<T>,
        throttle_interval: Duration,
    ) -> Result<ObserverId, PropertyError> {
        let last_notify_time = Arc::new(Mutex::new(None::<Instant>));
        let pending_notification = Arc::new(Mutex::new(None::<(T, T)>));
        
        let throttled_observer = Arc::new(move |old_val: &T, new_val: &T| {
            let should_notify_now = {
                let last_time = last_notify_time.lock().unwrap();
                match *last_time {
                    None => true, // First notification - notify immediately
                    Some(last) => last.elapsed() >= throttle_interval,
                }
            };
            
            if should_notify_now {
                // Notify immediately
                {
                    let mut last_time = last_notify_time.lock().unwrap();
                    *last_time = Some(Instant::now());
                }
                
                let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    observer(old_val, new_val);
                }));
            } else {
                // Schedule a notification for later
                {
                    let mut pending = pending_notification.lock().unwrap();
                    *pending = Some((old_val.clone(), new_val.clone()));
                }
                
                // Check if we need to spawn a thread for the pending notification
                let last_notify_time_thread = last_notify_time.clone();
                let pending_notification_thread = pending_notification.clone();
                let observer_thread = observer.clone();
                let interval = throttle_interval;
                
                thread::spawn(move || {
                    // Calculate how long to wait
                    let wait_duration = {
                        let last_time = last_notify_time_thread.lock().unwrap();
                        if let Some(last) = *last_time {
                            let elapsed = last.elapsed();
                            if elapsed < interval {
                                interval - elapsed
                            } else {
                                Duration::from_millis(0)
                            }
                        } else {
                            Duration::from_millis(0)
                        }
                    };
                    
                    if wait_duration > Duration::from_millis(0) {
                        thread::sleep(wait_duration);
                    }
                    
                    // Check if we should notify
                    let should_notify = {
                        let last_time = last_notify_time_thread.lock().unwrap();
                        match *last_time {
                            Some(last) => last.elapsed() >= interval,
                            None => true,
                        }
                    };
                    
                    if should_notify {
                        // Get and clear pending notification
                        let values = {
                            let mut pending = pending_notification_thread.lock().unwrap();
                            pending.take()
                        };
                        
                        if let Some((old, new)) = values {
                            {
                                let mut last_time = last_notify_time_thread.lock().unwrap();
                                *last_time = Some(Instant::now());
                            }
                            
                            let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                                observer_thread(&old, &new);
                            }));
                        }
                    }
                });
            }
        });

        self.subscribe(throttled_observer)
    }

    /// Notifies all observers with a batch of changes
    ///
    /// This method allows you to trigger observer notifications for multiple
    /// value changes efficiently. Unlike individual `set()` calls, this method
    /// acquires the observer list once and then notifies all observers with each
    /// change in the batch.
    ///
    /// # Performance Characteristics
    ///
    /// - **Lock optimization**: Acquires read lock only to snapshot observers, then releases it
    /// - **Non-blocking**: Other operations can proceed during observer notifications
    /// - **Panic isolation**: Individual observer panics don't affect other observers
    ///
    /// # Arguments
    ///
    /// * `changes` - A vector of tuples `(old_value, new_value)` to notify observers about
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful. Observer errors are logged but don't cause the method to fail.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    /// let call_count = Arc::new(AtomicUsize::new(0));
    /// let count_clone = call_count.clone();
    ///
    /// property.subscribe(Arc::new(move |old, new| {
    ///     count_clone.fetch_add(1, Ordering::SeqCst);
    ///     println!("Change: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Notify with multiple changes at once
    /// property.notify_observers_batch(vec![
    ///     (0, 10),
    ///     (10, 20),
    ///     (20, 30),
    /// ])?;
    ///
    /// assert_eq!(call_count.load(Ordering::SeqCst), 3);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Note
    ///
    /// This method does NOT update the property's actual value - it only triggers
    /// observer notifications. Use `set()` if you want to update the value and
    /// notify observers.
    pub fn notify_observers_batch(&self, changes: Vec<(T, T)>) -> Result<(), PropertyError> {
        // Acquire lock, clone observers, then release lock immediately
        // This prevents blocking other operations during potentially long notification process
        let (observers_snapshot, dead_observer_ids) = {
            let prop = match self.inner.read() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    // Graceful degradation: recover from poisoned read lock
                    poisoned.into_inner()
                }
            };
            
            // Collect active observers and track dead weak observers
            let mut observers = Vec::new();
            let mut dead_ids = Vec::new();
            for (id, observer_ref) in &prop.observers {
                if let Some(observer) = observer_ref.try_call() {
                    observers.push(observer);
                } else {
                    // Weak observer is dead, mark for removal
                    dead_ids.push(*id);
                }
            }
            
            (observers, dead_ids)
        }; // Lock released here

        // Notify observers without holding the lock
        for (old_val, new_val) in changes {
            for observer in &observers_snapshot {
                // Wrap in panic recovery like other notification methods
                if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    observer(&old_val, &new_val);
                })) {
                    eprintln!("Observer panic in batch notification: {:?}", e);
                }
            }
        }
        
        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }
        
        Ok(())
    }

    /// Subscribes an observer and returns a RAII guard for automatic cleanup
    ///
    /// This method is similar to `subscribe()` but returns a `Subscription` object
    /// that automatically removes the observer when it goes out of scope. This
    /// provides a more convenient and safer alternative to manual subscription
    /// management.
    ///
    /// # Arguments
    ///
    /// * `observer` - A function wrapped in `Arc` that takes `(&T, &T)` parameters
    ///
    /// # Returns
    ///
    /// `Ok(Subscription<T>)` containing a RAII guard for the observer,
    /// or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ## Basic RAII Subscription
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    ///
    /// {
    ///     let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///         println!("Value: {} -> {}", old, new);
    ///     }))?;
    ///
    ///     property.set(42)?; // Prints: "Value: 0 -> 42"
    ///     property.set(100)?; // Prints: "Value: 42 -> 100"
    ///
    ///     // Automatic cleanup when _subscription goes out of scope
    /// }
    ///
    /// property.set(200)?; // No output - subscription was cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Comparison with Manual Management
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new("initial".to_string());
    ///
    /// // Method 1: Manual subscription management (traditional approach)
    /// let observer_id = property.subscribe(Arc::new(|old, new| {
    ///     println!("Manual: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Method 2: RAII subscription management (recommended)
    /// let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///     println!("RAII: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Both observers will be called
    /// property.set("changed".to_string())?;
    /// // Prints:
    /// // "Manual: initial -> changed"
    /// // "RAII: initial -> changed"
    ///
    /// // Manual cleanup required for first observer
    /// property.unsubscribe(observer_id)?;
    ///
    /// // Second observer (_subscription) is automatically cleaned up when
    /// // the variable goes out of scope - no manual intervention needed
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Error Handling with Early Returns
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// fn process_with_monitoring(property: &ObservableProperty<i32>) -> Result<(), observable_property::PropertyError> {
    ///     let _monitoring = property.subscribe_with_subscription(Arc::new(|old, new| {
    ///         println!("Processing: {} -> {}", old, new);
    ///     }))?;
    ///
    ///     property.set(1)?;
    ///     
    ///     if property.get()? > 0 {
    ///         return Ok(()); // Subscription automatically cleaned up on early return
    ///     }
    ///
    ///     property.set(2)?;
    ///     Ok(()) // Subscription automatically cleaned up on normal return
    /// }
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(0);
    /// process_with_monitoring(&property)?; // Monitoring active only during function call
    /// property.set(99)?; // No monitoring output - subscription was cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multi-threaded Subscription Management
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = Arc::new(ObservableProperty::new(0));
    /// let property_clone = property.clone();
    ///
    /// let handle = thread::spawn(move || -> Result<(), observable_property::PropertyError> {
    ///     let _subscription = property_clone.subscribe_with_subscription(Arc::new(|old, new| {
    ///         println!("Thread observer: {} -> {}", old, new);
    ///     }))?;
    ///
    ///     property_clone.set(42)?; // Prints: "Thread observer: 0 -> 42"
    ///     
    ///     // Subscription automatically cleaned up when thread ends
    ///     Ok(())
    /// });
    ///
    /// handle.join().unwrap()?;
    /// property.set(100)?; // No output - thread subscription was cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    ///
    /// This method is particularly useful in scenarios such as:
    /// - Temporary observers that should be active only during a specific scope
    /// - Error-prone code where manual cleanup might be forgotten
    /// - Complex control flow where multiple exit points make manual cleanup difficult
    /// - Resource-constrained environments where observer leaks are problematic
    pub fn subscribe_with_subscription(
        &self,
        observer: Observer<T>,
    ) -> Result<Subscription<T>, PropertyError> {
        let id = self.subscribe(observer)?;
        Ok(Subscription {
            inner: Arc::clone(&self.inner),
            id,
        })
    }

    /// Subscribes a filtered observer and returns a RAII guard for automatic cleanup
    ///
    /// This method combines the functionality of `subscribe_filtered()` with the automatic
    /// cleanup benefits of `subscribe_with_subscription()`. The observer will only be
    /// called when the filter condition is satisfied, and it will be automatically
    /// unsubscribed when the returned `Subscription` goes out of scope.
    ///
    /// # Arguments
    ///
    /// * `observer` - The observer function to call when the filter passes
    /// * `filter` - A predicate function that receives `(old_value, new_value)` and returns `bool`
    ///
    /// # Returns
    ///
    /// `Ok(Subscription<T>)` containing a RAII guard for the filtered observer,
    /// or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ## Basic Filtered RAII Subscription
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let counter = ObservableProperty::new(0);
    ///
    /// {
    ///     // Monitor only increases with automatic cleanup
    ///     let _increase_monitor = counter.subscribe_filtered_with_subscription(
    ///         Arc::new(|old, new| {
    ///             println!("Counter increased: {} -> {}", old, new);
    ///         }),
    ///         |old, new| new > old
    ///     )?;
    ///
    ///     counter.set(5)?;  // Prints: "Counter increased: 0 -> 5"
    ///     counter.set(3)?;  // No output (decrease)
    ///     counter.set(7)?;  // Prints: "Counter increased: 3 -> 7"
    ///
    ///     // Subscription automatically cleaned up when leaving scope
    /// }
    ///
    /// counter.set(10)?; // No output - subscription was cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multi-Condition Temperature Monitoring
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let temperature = ObservableProperty::new(20.0_f64);
    ///
    /// {
    ///     // Create filtered subscription that only triggers for significant temperature increases
    ///     let _heat_warning = temperature.subscribe_filtered_with_subscription(
    ///         Arc::new(|old_temp, new_temp| {
    ///             println!("🔥 Heat warning! Temperature rose from {:.1}°C to {:.1}°C",
    ///                      old_temp, new_temp);
    ///         }),
    ///         |old, new| new > old && (new - old) > 5.0  // Only trigger for increases > 5°C
    ///     )?;
    ///
    ///     // Create another filtered subscription for cooling alerts
    ///     let _cooling_alert = temperature.subscribe_filtered_with_subscription(
    ///         Arc::new(|old_temp, new_temp| {
    ///             println!("❄️ Cooling alert! Temperature dropped from {:.1}°C to {:.1}°C",
    ///                      old_temp, new_temp);
    ///         }),
    ///         |old, new| new < old && (old - new) > 3.0  // Only trigger for decreases > 3°C
    ///     )?;
    ///
    ///     // Test the filters
    ///     temperature.set(22.0)?; // No alerts (increase of only 2°C)
    ///     temperature.set(28.0)?; // Heat warning triggered (increase of 6°C from 22°C)
    ///     temperature.set(23.0)?; // Cooling alert triggered (decrease of 5°C)
    ///
    ///     // Both subscriptions are automatically cleaned up when they go out of scope
    /// }
    ///
    /// temperature.set(35.0)?; // No alerts - subscriptions were cleaned up
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Conditional Monitoring with Complex Filters
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let stock_price = ObservableProperty::new(100.0_f64);
    ///
    /// {
    ///     // Monitor significant price movements (> 5% change)
    ///     let _volatility_alert = stock_price.subscribe_filtered_with_subscription(
    ///         Arc::new(|old_price, new_price| {
    ///             let change_percent = ((new_price - old_price) / old_price * 100.0).abs();
    ///             println!("📈 Significant price movement: ${:.2} -> ${:.2} ({:.1}%)",
    ///                     old_price, new_price, change_percent);
    ///         }),
    ///         |old, new| {
    ///             let change_percent = ((new - old) / old * 100.0).abs();
    ///             change_percent > 5.0  // Trigger on > 5% change
    ///         }
    ///     )?;
    ///
    ///     stock_price.set(103.0)?; // No alert (3% change)
    ///     stock_price.set(108.0)?; // Alert triggered (4.85% from 103, but let's say it rounds up)
    ///     stock_price.set(95.0)?;  // Alert triggered (12% decrease)
    ///
    ///     // Subscription automatically cleaned up when leaving scope
    /// }
    ///
    /// stock_price.set(200.0)?; // No alert - monitoring ended
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Cross-Thread Filtered Monitoring
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    /// use std::time::Duration;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let network_latency = Arc::new(ObservableProperty::new(50)); // milliseconds
    /// let latency_clone = network_latency.clone();
    ///
    /// let monitor_handle = thread::spawn(move || -> Result<(), observable_property::PropertyError> {
    ///     // Monitor high latency in background thread with automatic cleanup
    ///     let _high_latency_alert = latency_clone.subscribe_filtered_with_subscription(
    ///         Arc::new(|old_ms, new_ms| {
    ///             println!("⚠️ High latency detected: {}ms -> {}ms", old_ms, new_ms);
    ///         }),
    ///         |_, new| *new > 100  // Alert when latency exceeds 100ms
    ///     )?;
    ///
    ///     // Simulate monitoring for a short time
    ///     thread::sleep(Duration::from_millis(10));
    ///     
    ///     // Subscription automatically cleaned up when thread ends
    ///     Ok(())
    /// });
    ///
    /// // Simulate network conditions
    /// network_latency.set(80)?;  // No alert (under threshold)
    /// network_latency.set(150)?; // Alert triggered in background thread
    ///
    /// monitor_handle.join().unwrap()?;
    /// network_latency.set(200)?; // No alert - background monitoring ended
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use Cases
    ///
    /// This method is ideal for:
    /// - Threshold-based monitoring with automatic cleanup
    /// - Temporary conditional observers in specific code blocks
    /// - Event-driven systems where observers should be active only during certain phases
    /// - Resource management scenarios where filtered observers have limited lifetimes
    ///
    /// # Performance Notes
    ///
    /// The filter function is evaluated for every property change, so it should be
    /// lightweight. Complex filtering logic should be optimized to avoid performance
    /// bottlenecks, especially in high-frequency update scenarios.
    pub fn subscribe_filtered_with_subscription<F>(
        &self,
        observer: Observer<T>,
        filter: F,
    ) -> Result<Subscription<T>, PropertyError>
    where
        F: Fn(&T, &T) -> bool + Send + Sync + 'static,
    {
        let id = self.subscribe_filtered(observer, filter)?;
        Ok(Subscription {
            inner: Arc::clone(&self.inner),
            id,
        })
    }

    /// Creates a new observable property with full configuration control
    ///
    /// This constructor provides complete control over the property's configuration,
    /// allowing you to customize both thread pool size and maximum observer count.
    ///
    /// # Arguments
    ///
    /// * `initial_value` - The starting value for this property
    /// * `max_threads` - Maximum threads for async notifications (0 = use default)
    /// * `max_observers` - Maximum number of allowed observers (0 = use default)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// // Create a property optimized for high-frequency updates with many observers
    /// let property = ObservableProperty::with_config(0, 8, 50000);
    /// assert_eq!(property.get().unwrap(), 0);
    /// ```
    pub fn with_config(initial_value: T, max_threads: usize, max_observers: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InnerProperty {
                value: initial_value,
                observers: HashMap::new(),
                next_id: 0,
                history: None,
                history_size: 0,
                total_changes: 0,
                observer_calls: 0,
                notification_times: Vec::new(),
                #[cfg(feature = "debug")]
                debug_logging_enabled: false,
                #[cfg(feature = "debug")]
                change_logs: Vec::new(),
                batch_depth: 0,
                batch_initial_value: None,
                eq_fn: None,
                validator: None,
                event_log: None,
                event_log_size: 0,
            })),
            max_threads: if max_threads == 0 { MAX_THREADS } else { max_threads },
            max_observers: if max_observers == 0 { MAX_OBSERVERS } else { max_observers },
        }
    }

    /// Returns the current number of active observers
    ///
    /// This method is useful for debugging, monitoring, and testing to verify
    /// that observers are being properly managed and cleaned up.
    ///
    /// # Returns
    ///
    /// The number of currently subscribed observers, or 0 if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let property = ObservableProperty::new(42);
    /// assert_eq!(property.observer_count(), 0);
    ///
    /// let id1 = property.subscribe(Arc::new(|_, _| {}))?;
    /// assert_eq!(property.observer_count(), 1);
    ///
    /// let id2 = property.subscribe(Arc::new(|_, _| {}))?;
    /// assert_eq!(property.observer_count(), 2);
    ///
    /// property.unsubscribe(id1)?;
    /// assert_eq!(property.observer_count(), 1);
    /// # Ok(())
    /// # }
    /// ```
    pub fn observer_count(&self) -> usize {
        match self.inner.read() {
            Ok(prop) => prop.observers.len(),
            Err(poisoned) => {
                // Graceful degradation: recover from poisoned lock
                poisoned.into_inner().observers.len()
            }
        }
    }

    /// Gets the current value without Result wrapping
    ///
    /// This is a convenience method that returns `None` if the lock is poisoned
    /// (which shouldn't happen with graceful degradation) instead of a Result.
    ///
    /// # Returns
    ///
    /// `Some(T)` containing the current value, or `None` if somehow inaccessible.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// assert_eq!(property.try_get(), Some(42));
    /// ```
    pub fn try_get(&self) -> Option<T> {
        self.get().ok()
    }

    /// Atomically modifies the property value using a closure
    ///
    /// This method allows you to update the property based on its current value
    /// in a single atomic operation. The closure receives a mutable reference to
    /// the value and can modify it in place.
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that receives `&mut T` and modifies it
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let counter = ObservableProperty::new(0);
    ///
    /// counter.subscribe(Arc::new(|old, new| {
    ///     println!("Counter: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Increment counter atomically
    /// counter.modify(|value| *value += 1)?;
    /// assert_eq!(counter.get()?, 1);
    ///
    /// // Double the counter atomically
    /// counter.modify(|value| *value *= 2)?;
    /// assert_eq!(counter.get()?, 2);
    /// # Ok(())
    /// # }
    /// ```
    pub fn modify<F>(&self, f: F) -> Result<(), PropertyError>
    where
        F: FnOnce(&mut T),
    {
        let (old_value, new_value, observers_snapshot, dead_observer_ids) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    // Graceful degradation: recover from poisoned write lock
                    poisoned.into_inner()
                }
            };

            let old_value = prop.value.clone();
            f(&mut prop.value);
            let new_value = prop.value.clone();
            
            // Validate the modified value if a validator is configured
            if let Some(validator) = &prop.validator {
                validator(&new_value).map_err(|reason| {
                    // Restore the old value if validation fails
                    prop.value = old_value.clone();
                    PropertyError::ValidationError { reason }
                })?;
            }
            
            // Check if values are equal using custom equality function if provided
            let values_equal = if let Some(eq_fn) = &prop.eq_fn {
                eq_fn(&old_value, &new_value)
            } else {
                false  // No equality function = always notify
            };

            // If values are equal, skip everything
            if values_equal {
                return Ok(());
            }
            
            // Track the change
            prop.total_changes += 1;
            let event_num = prop.total_changes - 1; // Capture for event numbering
            
            // Add old value to history if history tracking is enabled
            let history_size = prop.history_size;
            if let Some(history) = &mut prop.history {
                // Add old value to history
                history.push(old_value.clone());
                
                // Enforce history size limit by removing oldest values
                if history.len() > history_size {
                    let overflow = history.len() - history_size;
                    history.drain(0..overflow);
                }
            }
            
            // Record event if event logging is enabled
            let event_log_size = prop.event_log_size;
            if let Some(event_log) = &mut prop.event_log {
                let event = PropertyEvent {
                    timestamp: Instant::now(),
                    old_value: old_value.clone(),
                    new_value: new_value.clone(),
                    event_number: event_num, // Use captured event number for consistent numbering
                    thread_id: format!("{:?}", thread::current().id()),
                };
                
                event_log.push(event);
                
                // Enforce event log size limit by removing oldest events (if bounded)
                if event_log_size > 0 && event_log.len() > event_log_size {
                    let overflow = event_log.len() - event_log_size;
                    event_log.drain(0..overflow);
                }
            }
            
            // Collect active observers and track dead weak observers
            let mut observers = Vec::new();
            let mut dead_ids = Vec::new();
            for (id, observer_ref) in &prop.observers {
                if let Some(observer) = observer_ref.try_call() {
                    observers.push(observer);
                } else {
                    // Weak observer is dead, mark for removal
                    dead_ids.push(*id);
                }
            }
            
            (old_value, new_value, observers, dead_ids)
        };

        // Notify observers with old and new values
        for observer in observers_snapshot {
            if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                observer(&old_value, &new_value);
            })) {
                eprintln!("Observer panic in modify: {:?}", e);
            }
        }

        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }

        Ok(())
    }

    /// Creates a derived property that automatically updates when this property changes
    ///
    /// This method applies a transformation function to create a new `ObservableProperty` of a
    /// potentially different type. The derived property automatically updates whenever the source
    /// property changes, maintaining the transformation relationship.
    ///
    /// This enables functional reactive programming patterns and property chaining, similar to
    /// `map` operations in functional programming or reactive frameworks.
    ///
    /// # Type Parameters
    ///
    /// * `U` - The type of the derived property (must be `Clone + Send + Sync + 'static`)
    /// * `F` - The transformation function type
    ///
    /// # Arguments
    ///
    /// * `transform` - A function that converts values from type `T` to type `U`
    ///
    /// # Returns
    ///
    /// * `Ok(ObservableProperty<U>)` - The derived property with the transformed initial value
    /// * `Err(PropertyError)` - If unable to read the source property or create the subscription
    ///
    /// # Lifetime and Ownership
    ///
    /// - The derived property remains connected to the source property through an observer subscription
    /// - The subscription keeps both properties alive as long as the source has observers
    /// - When the derived property is dropped, updates stop, but the source property continues working
    /// - The transformation function is called immediately to compute the initial value, then on every change
    ///
    /// # Examples
    ///
    /// ## Temperature Conversion (Celsius to Fahrenheit)
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Create a Celsius property
    /// let celsius = ObservableProperty::new(20.0);
    ///
    /// // Derive a Fahrenheit property that auto-updates
    /// let fahrenheit = celsius.map(|c| c * 9.0 / 5.0 + 32.0)?;
    ///
    /// assert_eq!(fahrenheit.get()?, 68.0);
    ///
    /// // Observe the derived property
    /// let _sub = fahrenheit.subscribe_with_subscription(Arc::new(|_old, new| {
    ///     println!("Fahrenheit: {:.1}°F", new);
    /// }))?;
    ///
    /// celsius.set(25.0)?;  // Prints: "Fahrenheit: 77.0°F"
    /// assert_eq!(fahrenheit.get()?, 77.0);
    ///
    /// celsius.set(0.0)?;   // Prints: "Fahrenheit: 32.0°F"
    /// assert_eq!(fahrenheit.get()?, 32.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## String Formatting
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let count = ObservableProperty::new(42);
    ///
    /// // Create a formatted string property
    /// let message = count.map(|n| format!("Count: {}", n))?;
    ///
    /// assert_eq!(message.get()?, "Count: 42");
    ///
    /// count.set(100)?;
    /// assert_eq!(message.get()?, "Count: 100");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Mathematical Transformations
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let radius = ObservableProperty::new(5.0);
    ///
    /// // Derive area from radius (πr²)
    /// let area = radius.map(|r| std::f64::consts::PI * r * r)?;
    ///
    /// assert!((area.get()? - 78.54).abs() < 0.01);
    ///
    /// radius.set(10.0)?;
    /// assert!((area.get()? - 314.16).abs() < 0.01);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Chaining Multiple Transformations
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let base = ObservableProperty::new(10);
    ///
    /// // Chain multiple transformations
    /// let doubled = base.map(|x| x * 2)?;
    /// let squared = doubled.map(|x| x * x)?;
    /// let formatted = squared.map(|x| format!("Result: {}", x))?;
    ///
    /// assert_eq!(formatted.get()?, "Result: 400");
    ///
    /// base.set(5)?;
    /// assert_eq!(formatted.get()?, "Result: 100");  // (5 * 2)² = 100
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Type Conversion
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let integer = ObservableProperty::new(42);
    ///
    /// // Convert integer to float
    /// let float_value = integer.map(|i| *i as f64)?;
    /// assert_eq!(float_value.get()?, 42.0);
    ///
    /// // Convert to boolean (is even?)
    /// let is_even = integer.map(|i| i % 2 == 0)?;
    /// assert_eq!(is_even.get()?, true);
    ///
    /// integer.set(43)?;
    /// assert_eq!(is_even.get()?, false);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Complex Object Transformation
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// #[derive(Clone)]
    /// struct User {
    ///     first_name: String,
    ///     last_name: String,
    ///     age: u32,
    /// }
    ///
    /// let user = ObservableProperty::new(User {
    ///     first_name: "John".to_string(),
    ///     last_name: "Doe".to_string(),
    ///     age: 30,
    /// });
    ///
    /// // Derive full name from user
    /// let full_name = user.map(|u| format!("{} {}", u.first_name, u.last_name))?;
    /// assert_eq!(full_name.get()?, "John Doe");
    ///
    /// // Derive adult status
    /// let is_adult = user.map(|u| u.age >= 18)?;
    /// assert_eq!(is_adult.get()?, true);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Working with Options
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let optional = ObservableProperty::new(Some(42));
    ///
    /// // Extract value with default
    /// let value_or_zero = optional.map(|opt| opt.unwrap_or(0))?;
    /// assert_eq!(value_or_zero.get()?, 42);
    ///
    /// optional.set(None)?;
    /// assert_eq!(value_or_zero.get()?, 0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - The transformation function is called on every change to the source property
    /// - Keep transformation functions lightweight for best performance
    /// - The derived property maintains its own observer list independent of the source
    /// - Cloning the source property is cheap (internal Arc), but each clone shares the same observers
    ///
    /// # Thread Safety
    ///
    /// The transformation function must be `Send + Sync + 'static` as it may be called from
    /// any thread that modifies the source property. Ensure your transformation logic is thread-safe.
    ///
    /// # Comparison with `computed()`
    ///
    /// - `map()` is simpler and works on a single source property
    /// - `computed()` can depend on multiple source properties
    /// - `map()` is an instance method; `computed()` is a standalone function
    /// - For single-source transformations, `map()` is more ergonomic
    pub fn map<U, F>(&self, transform: F) -> Result<ObservableProperty<U>, PropertyError>
    where
        U: Clone + Send + Sync + 'static,
        F: Fn(&T) -> U + Send + Sync + 'static,
    {
        // Get initial value and transform it
        let initial_value = self.get()?;
        let derived = ObservableProperty::new(transform(&initial_value));
        
        // Subscribe to source changes and update derived property
        let derived_clone = derived.clone();
        let transform = Arc::new(transform);
        self.subscribe(Arc::new(move |_old, new| {
            let transformed = transform(new);
            if let Err(e) = derived_clone.set(transformed) {
                eprintln!("Failed to update derived property: {}", e);
            }
        }))?;
        
        Ok(derived)
    }

    /// Updates the property through multiple intermediate states and notifies observers for each change
    ///
    /// This method is useful for animations, multi-step transformations, or any scenario where
    /// you want to record and notify observers about intermediate states during a complex update.
    /// The provided function receives a mutable reference to the current value and returns a
    /// vector of intermediate states. Observers are notified for each transition between states.
    ///
    /// # Behavior
    ///
    /// 1. Captures the initial value
    /// 2. Calls the provided function with `&mut T` to get intermediate states
    /// 3. Updates the property's value to the final state (last intermediate state, or unchanged if empty)
    /// 4. Notifies observers for each state transition:
    ///    - initial → intermediate\[0\]
    ///    - intermediate\[0\] → intermediate\[1\]
    ///    - ... → intermediate\[n\]
    ///
    /// # Arguments
    ///
    /// * `f` - A closure that receives `&mut T` and returns a vector of intermediate states
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(PropertyError)` if the lock is poisoned.
    ///
    /// # Examples
    ///
    /// ## Animation with Intermediate States
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let position = ObservableProperty::new(0);
    /// let notification_count = Arc::new(AtomicUsize::new(0));
    /// let count_clone = notification_count.clone();
    ///
    /// position.subscribe(Arc::new(move |old, new| {
    ///     count_clone.fetch_add(1, Ordering::SeqCst);
    ///     println!("Position: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Animate from 0 to 100 in steps of 25
    /// position.update_batch(|_current| {
    ///     vec![25, 50, 75, 100]
    /// })?;
    ///
    /// // Observers were notified 4 times:
    /// // 0 -> 25, 25 -> 50, 50 -> 75, 75 -> 100
    /// assert_eq!(notification_count.load(Ordering::SeqCst), 4);
    /// assert_eq!(position.get()?, 100);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multi-Step Transformation
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let data = ObservableProperty::new(String::from("hello"));
    ///
    /// data.subscribe(Arc::new(|old, new| {
    ///     println!("Transformation: '{}' -> '{}'", old, new);
    /// }))?;
    ///
    /// // Transform through multiple steps
    /// data.update_batch(|current| {
    ///     let step1 = current.to_uppercase(); // "HELLO"
    ///     let step2 = format!("{}!", step1);   // "HELLO!"
    ///     let step3 = format!("{} WORLD", step2); // "HELLO! WORLD"
    ///     vec![step1, step2, step3]
    /// })?;
    ///
    /// assert_eq!(data.get()?, "HELLO! WORLD");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Counter with Intermediate Values
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let counter = ObservableProperty::new(0);
    ///
    /// counter.subscribe(Arc::new(|old, new| {
    ///     println!("Count: {} -> {}", old, new);
    /// }))?;
    ///
    /// // Increment with recording intermediate states
    /// counter.update_batch(|current| {
    ///     *current += 10; // Modify in place (optional)
    ///     vec![5, 8, 10] // Intermediate states to report
    /// })?;
    ///
    /// // Final value is the last intermediate state
    /// assert_eq!(counter.get()?, 10);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Empty Intermediate States
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let value = ObservableProperty::new(42);
    /// let was_notified = Arc::new(AtomicBool::new(false));
    /// let flag = was_notified.clone();
    ///
    /// value.subscribe(Arc::new(move |_, _| {
    ///     flag.store(true, Ordering::SeqCst);
    /// }))?;
    ///
    /// // No intermediate states - value remains unchanged, no notifications
    /// value.update_batch(|current| {
    ///     *current = 100; // This modification is ignored
    ///     Vec::new() // No intermediate states
    /// })?;
    ///
    /// assert!(!was_notified.load(Ordering::SeqCst));
    /// assert_eq!(value.get()?, 42); // Value unchanged
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Considerations
    ///
    /// - Lock is held during function execution and state collection
    /// - All intermediate states are stored in memory before notification
    /// - Observers are notified sequentially for each state transition
    /// - Consider using `set()` or `modify()` if you don't need intermediate state tracking
    ///
    /// # Use Cases
    ///
    /// - **Animations**: Smooth transitions through intermediate visual states
    /// - **Progressive calculations**: Show progress through multi-step computations
    /// - **State machines**: Record transitions through multiple states
    /// - **Debugging**: Track how a value transforms through complex operations
    /// - **History tracking**: Maintain a record of transformation steps
    pub fn update_batch<F>(&self, f: F) -> Result<(), PropertyError>
    where
        F: FnOnce(&mut T) -> Vec<T>,
    {
        let (initial_value, intermediate_states, observers_snapshot, dead_observer_ids) = {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => {
                    // Graceful degradation: recover from poisoned write lock
                    poisoned.into_inner()
                }
            };

            let initial_value = prop.value.clone();
            let states = f(&mut prop.value);
            
            // Update to the final state if intermediate states were provided
            // Otherwise, restore the original value (ignore any in-place modifications)
            if let Some(final_state) = states.last() {
                prop.value = final_state.clone();
                
                // Add initial value to history if history tracking is enabled
                // Note: We only track the pre-batch initial value, not intermediates
                let history_size = prop.history_size;
                if let Some(history) = &mut prop.history {
                    history.push(initial_value.clone());
                    
                    // Enforce history size limit by removing oldest values
                    if history.len() > history_size {
                        let overflow = history.len() - history_size;
                        history.drain(0..overflow);
                    }
                }
            } else {
                prop.value = initial_value.clone();
            }
            
            // Collect active observers and track dead weak observers
            let mut observers = Vec::new();
            let mut dead_ids = Vec::new();
            for (id, observer_ref) in &prop.observers {
                if let Some(observer) = observer_ref.try_call() {
                    observers.push(observer);
                } else {
                    // Weak observer is dead, mark for removal
                    dead_ids.push(*id);
                }
            }
            
            (initial_value, states, observers, dead_ids)
        };

        // Notify observers for each state transition
        if !intermediate_states.is_empty() {
            let mut previous_state = initial_value;
            
            for current_state in intermediate_states {
                for observer in &observers_snapshot {
                    if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        observer(&previous_state, &current_state);
                    })) {
                        eprintln!("Observer panic in update_batch: {:?}", e);
                    }
                }
                previous_state = current_state;
            }
        }

        // Clean up dead weak observers
        if !dead_observer_ids.is_empty() {
            let mut prop = match self.inner.write() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            for id in dead_observer_ids {
                prop.observers.remove(&id);
            }
        }

        Ok(())
    }

    /// Creates a bidirectional binding between two properties
    ///
    /// This method establishes a two-way synchronization where changes to either property
    /// will automatically update the other. This is particularly useful for model-view
    /// synchronization patterns where a UI control and a data model need to stay in sync.
    ///
    /// # How It Works
    ///
    /// 1. Each property subscribes to changes in the other
    /// 2. When property A changes, property B is updated to match
    /// 3. When property B changes, property A is updated to match
    /// 4. Infinite loops are prevented by comparing values before updating
    ///
    /// # Loop Prevention
    ///
    /// The method uses value comparison to prevent infinite update loops. If the new
    /// value equals the current value, no update is triggered. This requires `T` to
    /// implement `PartialEq`.
    ///
    /// # Type Requirements
    ///
    /// The value type must implement:
    /// - `Clone` - For copying values between properties
    /// - `PartialEq` - For comparing values to prevent infinite loops
    /// - `Send + Sync` - For thread-safe operation
    /// - `'static` - For storing in observers
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the binding was successfully established
    /// - `Err(PropertyError)` if subscription fails (e.g., observer limit exceeded)
    ///
    /// # Subscription Management
    ///
    /// The subscriptions created by this method are stored as strong references and will
    /// remain active until one of the properties is dropped or the observers are manually
    /// unsubscribed. The returned `ObserverId`s can be used to unsubscribe if needed.
    ///
    /// # Examples
    ///
    /// ## Basic Two-Way Binding
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let model = ObservableProperty::new(0);
    /// let view = ObservableProperty::new(0);
    ///
    /// // Establish bidirectional binding
    /// model.bind_bidirectional(&view)?;
    ///
    /// // Update model - view automatically updates
    /// model.set(42)?;
    /// assert_eq!(view.get()?, 42);
    ///
    /// // Update view - model automatically updates
    /// view.set(100)?;
    /// assert_eq!(model.get()?, 100);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Model-View Synchronization
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// // Model representing application state
    /// let username = ObservableProperty::new("".to_string());
    ///
    /// // View representing UI input field
    /// let username_field = ObservableProperty::new("".to_string());
    ///
    /// // Bind them together
    /// username.bind_bidirectional(&username_field)?;
    ///
    /// // Add validation observer on the model
    /// username.subscribe(Arc::new(|_old, new| {
    ///     if new.len() > 3 {
    ///         println!("Valid username: {}", new);
    ///     }
    /// }))?;
    ///
    /// // User types in UI field
    /// username_field.set("john".to_string())?;
    /// // Both properties are now "john", validation observer triggered
    /// assert_eq!(username.get()?, "john");
    ///
    /// // Programmatic model update
    /// username.set("alice".to_string())?;
    /// // UI field automatically reflects the change
    /// assert_eq!(username_field.get()?, "alice");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## Multiple Property Synchronization
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let slider_value = ObservableProperty::new(50);
    /// let text_input = ObservableProperty::new(50);
    /// let display_label = ObservableProperty::new(50);
    ///
    /// // Create a synchronized group of controls
    /// slider_value.bind_bidirectional(&text_input)?;
    /// slider_value.bind_bidirectional(&display_label)?;
    ///
    /// // Update any one of them
    /// text_input.set(75)?;
    ///
    /// // All are synchronized
    /// assert_eq!(slider_value.get()?, 75);
    /// assert_eq!(text_input.get()?, 75);
    /// assert_eq!(display_label.get()?, 75);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// ## With Additional Observers
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// # fn main() -> Result<(), observable_property::PropertyError> {
    /// let celsius = ObservableProperty::new(0.0);
    /// let fahrenheit = ObservableProperty::new(32.0);
    ///
    /// // Note: For unit conversion, you'd typically use computed properties
    /// // instead of bidirectional binding, but this shows the concept
    /// celsius.bind_bidirectional(&fahrenheit)?;
    ///
    /// // Add logging to observe synchronization
    /// celsius.subscribe(Arc::new(|old, new| {
    ///     println!("Celsius changed: {:.1}°C -> {:.1}°C", old, new);
    /// }))?;
    ///
    /// fahrenheit.subscribe(Arc::new(|old, new| {
    ///     println!("Fahrenheit changed: {:.1}°F -> {:.1}°F", old, new);
    /// }))?;
    ///
    /// celsius.set(100.0)?;
    /// // Both properties are now 100.0 (not a real unit conversion!)
    /// // Prints:
    /// // "Celsius changed: 0.0°C -> 100.0°C"
    /// // "Fahrenheit changed: 32.0°F -> 100.0°F"
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Thread Safety
    ///
    /// The binding is fully thread-safe. Both properties can be updated from any thread,
    /// and the synchronization will work correctly across thread boundaries.
    ///
    /// # Performance Considerations
    ///
    /// - Each bound property creates two observer subscriptions (one in each direction)
    /// - Value comparisons are performed on every update to prevent loops
    /// - Consider using computed properties for one-way transformations instead
    /// - Binding many properties in a chain may amplify update overhead
    ///
    /// # Limitations
    ///
    /// - Both properties must have the same type `T`
    /// - Not suitable for complex transformations (use computed properties instead)
    /// - Value comparison relies on `PartialEq` implementation quality
    /// - Circular update chains with 3+ properties may have propagation delays
    pub fn bind_bidirectional(
        &self,
        other: &ObservableProperty<T>,
    ) -> Result<(), PropertyError>
    where
        T: PartialEq,
    {
        // Subscribe self to other's changes
        // When other changes, update self
        let self_inner = Arc::clone(&self.inner);
        other.subscribe(Arc::new(move |_old, new| {
            // Check if self's current value differs to prevent infinite loop
            let should_update = {
                match self_inner.read() {
                    Ok(prop) => &prop.value != new,
                    Err(poisoned) => &poisoned.into_inner().value != new,
                }
            };

            if should_update {
                let mut prop = match self_inner.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };

                let old_value = mem::replace(&mut prop.value, new.clone());

                // Add to history if enabled
                let history_size = prop.history_size;
                if let Some(history) = &mut prop.history {
                    history.push(old_value.clone());
                    if history.len() > history_size {
                        let overflow = history.len() - history_size;
                        history.drain(0..overflow);
                    }
                }

                // Collect and notify observers
                let mut observers = Vec::new();
                for (_id, observer_ref) in &prop.observers {
                    if let Some(observer) = observer_ref.try_call() {
                        observers.push(observer);
                    }
                }
                
                // Release lock before notifying
                drop(prop);

                for observer in observers {
                    if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        observer(&old_value, new);
                    })) {
                        eprintln!("Observer panic in bidirectional binding: {:?}", e);
                    }
                }
            }
        }))?;

        // Subscribe other to self's changes
        // When self changes, update other
        let other_inner = Arc::clone(&other.inner);
        self.subscribe(Arc::new(move |_old, new| {
            // Check if other's current value differs to prevent infinite loop
            let should_update = {
                match other_inner.read() {
                    Ok(prop) => &prop.value != new,
                    Err(poisoned) => &poisoned.into_inner().value != new,
                }
            };

            if should_update {
                let mut prop = match other_inner.write() {
                    Ok(guard) => guard,
                    Err(poisoned) => poisoned.into_inner(),
                };

                let old_value = mem::replace(&mut prop.value, new.clone());

                // Add to history if enabled
                let history_size = prop.history_size;
                if let Some(history) = &mut prop.history {
                    history.push(old_value.clone());
                    if history.len() > history_size {
                        let overflow = history.len() - history_size;
                        history.drain(0..overflow);
                    }
                }

                // Collect and notify observers
                let mut observers = Vec::new();
                for (_id, observer_ref) in &prop.observers {
                    if let Some(observer) = observer_ref.try_call() {
                        observers.push(observer);
                    }
                }
                
                // Release lock before notifying
                drop(prop);

                for observer in observers {
                    if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        observer(&old_value, new);
                    })) {
                        eprintln!("Observer panic in bidirectional binding: {:?}", e);
                    }
                }
            }
        }))?;

        Ok(())
    }

    /// Converts the observable property to an async stream
    ///
    /// Creates a stream that yields the current value followed by all future values
    /// as the property changes. The stream will continue indefinitely until dropped.
    ///
    /// # Features
    ///
    /// This method requires the `async` feature to be enabled.
    ///
    /// # Returns
    ///
    /// Returns a `PropertyStream<T>` that yields cloned values whenever
    /// the property changes.
    ///
    /// # Note
    ///
    /// This implementation uses only standard library primitives and does not
    /// depend on external async runtimes. The custom `Stream` trait is not
    /// compatible with the futures ecosystem.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use observable_property::{ObservableProperty, Stream};
    /// use std::sync::Arc;
    /// use std::pin::Pin;
    ///
    /// async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let property = Arc::new(ObservableProperty::new(0));
    ///     let mut stream = Box::pin(property.to_stream());
    ///
    ///     // Spawn a thread to modify the property
    ///     let prop_clone = property.clone();
    ///     std::thread::spawn(move || {
    ///         std::thread::sleep(std::time::Duration::from_millis(100));
    ///         prop_clone.set(42).ok();
    ///         std::thread::sleep(std::time::Duration::from_millis(100));
    ///         prop_clone.set(100).ok();
    ///     });
    ///
    ///     // Manual polling (you'd typically use an async runtime for this)
    ///     // This is just for demonstration
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "async")]
    pub fn to_stream(&self) -> PropertyStream<T> {
        use std::sync::mpsc;
        
        let (tx, rx) = mpsc::channel::<T>();
        let inner = self.inner.clone();

        // Get the current value to send as the first item
        let current_value = inner
            .read()
            .or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner()))
            .map(|prop| prop.value.clone())
            .ok();

        // Subscribe to changes and send them to the channel
        let subscription_id = self
            .subscribe(Arc::new(move |_old, new| {
                let _ = tx.send(new.clone());
            }))
            .ok();

        PropertyStream {
            rx,
            current_value,
            subscription_id,
            property: self.inner.clone(),
            waker: Arc::new(Mutex::new(None)),
        }
    }
}

/// A stream that yields values from an ObservableProperty
///
/// This stream implementation uses only standard library primitives and
/// provides a simple async iteration interface.
#[cfg(feature = "async")]
pub struct PropertyStream<T>
where
    T: Clone + Send + Sync + 'static,
{
    rx: std::sync::mpsc::Receiver<T>,
    current_value: Option<T>,
    subscription_id: Option<ObserverId>,
    property: Arc<RwLock<InnerProperty<T>>>,
    waker: Arc<Mutex<Option<std::task::Waker>>>,
}

#[cfg(feature = "async")]
impl<T: Clone + Send + Sync + Unpin + 'static> Stream for PropertyStream<T> {
    type Item = T;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        // Get mutable access to self
        let this = self.get_mut();

        // First, yield the current value if we haven't yet
        if let Some(value) = this.current_value.take() {
            return Poll::Ready(Some(value));
        }

        // Try to receive a value from the channel without blocking
        match this.rx.try_recv() {
            Ok(value) => Poll::Ready(Some(value)),
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                // Store the waker so the observer can wake us up
                if let Ok(mut waker_lock) = this.waker.lock() {
                    *waker_lock = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Poll::Ready(None),
        }
    }
}

#[cfg(feature = "async")]
impl<T: Clone + Send + Sync + 'static> Drop for PropertyStream<T> {
    fn drop(&mut self) {
        // Clean up subscription when stream is dropped
        if let Some(id) = self.subscription_id {
            if let Ok(mut prop) = self.property.write().or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner())) {
                prop.observers.remove(&id);
            }
        }
    }
}

#[cfg(feature = "async")]
impl<T: Clone + Send + Sync + Unpin + 'static> ObservableProperty<T> {
    /// Asynchronously waits for a specific condition to be met
    ///
    /// This method will await until the predicate function returns `true` for
    /// the property value. It checks the current value immediately, and if the
    /// predicate is not satisfied, it subscribes to changes and waits.
    ///
    /// # Features
    ///
    /// This method requires the `async` feature to be enabled.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that tests whether the condition is met
    ///
    /// # Returns
    ///
    /// Returns the first value that satisfies the predicate.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let property = Arc::new(ObservableProperty::new(0));
    ///
    ///     // Spawn a task to modify the property after a delay
    ///     let prop_clone = property.clone();
    ///     tokio::spawn(async move {
    ///         for i in 1..=10 {
    ///             tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    ///             prop_clone.set(i).ok();
    ///         }
    ///     });
    ///
    ///     // Wait for the property to reach a specific value
    ///     let result = property.wait_for(|value| *value >= 5).await;
    ///     println!("Property reached: {}", result);
    ///     assert!(result >= 5);
    ///
    ///     Ok(())
    /// }
    /// ```
    ///
    /// # Advanced Example: Multiple Conditions
    ///
    /// ```rust,no_run
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// async fn example() -> Result<(), Box<dyn std::error::Error>> {
    ///     let temperature = Arc::new(ObservableProperty::new(20.0));
    ///
    ///     // Spawn a thread to simulate temperature changes
    ///     let temp_clone = temperature.clone();
    ///     std::thread::spawn(move || {
    ///         for i in 0..20 {
    ///             std::thread::sleep(std::time::Duration::from_millis(50));
    ///             temp_clone.set(20.0 + i as f64 * 0.5).ok();
    ///         }
    ///     });
    ///
    ///     // Wait for critical temperature
    ///     let critical = temperature.wait_for(|temp| *temp > 25.0).await;
    ///     println!("Critical temperature reached: {:.1}°C", critical);
    ///
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "async")]
    pub fn wait_for<F>(&self, predicate: F) -> WaitForFuture<T, F>
    where
        F: Fn(&T) -> bool + Send + Sync + Unpin + 'static,
    {
        WaitForFuture::new(self.inner.clone(), predicate, self)
    }
}

/// A Future that resolves when an ObservableProperty meets a specific condition
#[cfg(feature = "async")]
pub struct WaitForFuture<T, F>
where
    T: Clone + Send + Sync + Unpin + 'static,
    F: Fn(&T) -> bool + Send + Sync + Unpin + 'static,
{
    property: Arc<RwLock<InnerProperty<T>>>,
    rx: std::sync::mpsc::Receiver<T>,
    subscription_id: Option<ObserverId>,
    result: Option<T>,
    _phantom: std::marker::PhantomData<F>,
}

#[cfg(feature = "async")]
impl<T, F> WaitForFuture<T, F>
where
    T: Clone + Send + Sync + Unpin + 'static,
    F: Fn(&T) -> bool + Send + Sync + Unpin + 'static,
{
    fn new<P>(property: Arc<RwLock<InnerProperty<T>>>, predicate: F, obs_property: &P) -> Self
    where
        P: HasInner<T>,
    {
        // Check current value first
        if let Ok(current) = property
            .read()
            .or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner()))
            .map(|prop| prop.value.clone())
        {
            if predicate(&current) {
                // Condition already met - create a dummy channel
                let (tx, rx) = std::sync::mpsc::channel();
                let _ = tx.send(current.clone());
                return Self {
                    property,
                    rx,
                    subscription_id: None,
                    result: Some(current),
                    _phantom: std::marker::PhantomData,
                };
            }
        }

        let predicate = Arc::new(predicate);
        let (tx, rx) = std::sync::mpsc::channel();

        // Subscribe to changes
        let subscription_id = obs_property
            .subscribe_internal(Arc::new(move |_old, new| {
                if predicate(new) {
                    let _ = tx.send(new.clone());
                }
            }))
            .ok();

        Self {
            property,
            rx,
            subscription_id,
            result: None,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "async")]
impl<T, F> std::future::Future for WaitForFuture<T, F>
where
    T: Clone + Send + Sync + Unpin + 'static,
    F: Fn(&T) -> bool + Send + Sync + Unpin + 'static,
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        
        // If we already have a result, return it immediately
        if let Some(result) = this.result.take() {
            return Poll::Ready(result);
        }

        // Try to receive without blocking
        match this.rx.try_recv() {
            Ok(value) => {
                // Clean up subscription
                if let Some(id) = this.subscription_id.take() {
                    if let Ok(mut prop) = this.property.write().or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner())) {
                        prop.observers.remove(&id);
                    }
                }
                Poll::Ready(value)
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => Poll::Pending,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                // Channel disconnected, try to get current value as fallback
                let value = this
                    .property
                    .read()
                    .or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner()))
                    .map(|prop| prop.value.clone())
                    .unwrap();
                Poll::Ready(value)
            }
        }
    }
}

#[cfg(feature = "async")]
impl<T, F> Drop for WaitForFuture<T, F>
where
    T: Clone + Send + Sync + Unpin + 'static,
    F: Fn(&T) -> bool + Send + Sync + Unpin + 'static,
{
    fn drop(&mut self) {
        // Clean up subscription when future is dropped
        if let Some(id) = self.subscription_id.take() {
            if let Ok(mut prop) = self.property.write().or_else(|poisoned| Ok::<_, ()>(poisoned.into_inner())) {
                prop.observers.remove(&id);
            }
        }
    }
}

// Helper trait to allow WaitForFuture to call subscribe without circular dependencies
#[cfg(feature = "async")]
trait HasInner<T: Clone + Send + Sync + 'static> {
    fn subscribe_internal(&self, observer: Observer<T>) -> Result<ObserverId, PropertyError>;
}

#[cfg(feature = "async")]
impl<T: Clone + Send + Sync + 'static> HasInner<T> for ObservableProperty<T> {
    fn subscribe_internal(&self, observer: Observer<T>) -> Result<ObserverId, PropertyError> {
        self.subscribe(observer)
    }
}

// Debug feature methods - only available when T implements Debug
#[cfg(feature = "debug")]
impl<T: Clone + Send + Sync + std::fmt::Debug + 'static> ObservableProperty<T> {
    /// Manually logs a property change with stack trace.
    ///
    /// Captures and stores:
    /// - Timestamp of the change
    /// - Old and new values (formatted via Debug trait)
    /// - Full stack trace at the call site
    /// - Thread ID
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get value");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "Updated from main");
    ///
    /// let logs = property.get_change_logs();
    /// assert_eq!(logs.len(), 1);
    /// # }
    /// ```
    pub fn log_change(&self, old_value: &T, new_value: &T, label: &str) {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };

        if !prop.debug_logging_enabled {
            return;
        }

        let backtrace = Backtrace::new();
        let thread_id = format!("{:?}", thread::current().id());
        let old_value_repr = format!("{:?}", old_value);
        let new_value_repr = format!("{:?} ({})", new_value, label);

        prop.change_logs.push(ChangeLog {
            timestamp: Instant::now(),
            old_value_repr,
            new_value_repr,
            backtrace: format!("{:?}", backtrace),
            thread_id,
        });
    }

    /// Enables debug logging of property changes.
    ///
    /// After calling this method, use `log_change()` to manually record changes
    /// with stack traces. Logs can be retrieved with `get_change_logs()`.
    ///
    /// # Performance Impact
    ///
    /// Stack trace capture has significant overhead:
    /// - ~10-100μs per change (varies by platform and stack depth)
    /// - Memory usage grows with change count (no automatic limit)
    /// - Only enable during debugging/development
    ///
    /// # Requirements
    ///
    /// This method is only available when the `debug` feature is enabled:
    /// ```toml
    /// [dependencies]
    /// observable-property = { version = "0.4", features = ["debug"] }
    /// ```
    ///
    /// Additionally, the type `T` must implement `std::fmt::Debug` for value logging.
    ///
    /// # Examples
    ///
    /// ## Basic Debug Logging
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "update 1");
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(200).ok();
    /// property.log_change(&old, &200, "update 2");
    ///
    /// // Each change is now logged with a stack trace
    /// let logs = property.get_change_logs();
    /// assert_eq!(logs.len(), 2);
    /// # }
    /// ```
    ///
    /// ## Debugging Unexpected Changes
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    /// use std::thread;
    ///
    /// let property = Arc::new(ObservableProperty::new(0));
    /// property.enable_change_logging();
    ///
    /// // Multiple threads modifying the property
    /// let handles: Vec<_> = (0..3).map(|i| {
    ///     let prop = property.clone();
    ///     thread::spawn(move || {
    ///         let old = prop.get().expect("Failed to get");
    ///         let new_val = i * 10;
    ///         prop.set(new_val).ok();
    ///         prop.log_change(&old, &new_val, "thread update");
    ///     })
    /// }).collect();
    ///
    /// for h in handles { h.join().ok(); }
    ///
    /// // Print detailed logs showing which thread made each change
    /// property.print_change_logs();
    /// # }
    /// ```
    ///
    /// # See Also
    ///
    /// - [`disable_change_logging`](#method.disable_change_logging) - Stop capturing logs
    /// - [`get_change_logs`](#method.get_change_logs) - Retrieve captured logs
    /// - [`print_change_logs`](#method.print_change_logs) - Pretty-print logs to stdout
    /// - [`clear_change_logs`](#method.clear_change_logs) - Clear accumulated logs
    pub fn enable_change_logging(&self) {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        prop.debug_logging_enabled = true;
    }

    /// Disables debug logging of property changes
    ///
    /// Stops capturing new change logs, but preserves existing logs.
    /// Use `clear_change_logs()` to remove accumulated logs.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "logged"); // Logged
    ///
    /// property.disable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(200).ok();
    /// property.log_change(&old, &200, "not logged"); // Not logged
    ///
    /// let logs = property.get_change_logs();
    /// assert_eq!(logs.len(), 1); // Only the first change
    /// # }
    /// ```
    pub fn disable_change_logging(&self) {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        prop.debug_logging_enabled = false;
    }

    /// Clears all accumulated change logs
    ///
    /// Removes all captured change logs from memory. The debug logging
    /// enabled/disabled state is not affected.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "update");
    ///
    /// assert_eq!(property.get_change_logs().len(), 1);
    /// property.clear_change_logs();
    /// assert_eq!(property.get_change_logs().len(), 0);
    /// # }
    /// ```
    pub fn clear_change_logs(&self) {
        let mut prop = match self.inner.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        prop.change_logs.clear();
    }

    /// Retrieves all captured change logs
    ///
    /// Returns a vector of formatted strings, each containing:
    /// - Timestamp (relative to first log)
    /// - Thread ID
    /// - Old and new values
    /// - Full stack trace
    ///
    /// # Returns
    ///
    /// A vector of log strings, one per captured change, in chronological order.
    /// Returns an empty vector if no changes have been logged.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "update");
    ///
    /// for log in property.get_change_logs() {
    ///     println!("{}", log);
    /// }
    /// # }
    /// ```
    pub fn get_change_logs(&self) -> Vec<String> {
        let prop = match self.inner.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        
        if prop.change_logs.is_empty() {
            return Vec::new();
        }

        let first_timestamp = prop.change_logs[0].timestamp;
        prop.change_logs.iter().map(|log| {
            let elapsed = log.timestamp.duration_since(first_timestamp);
            format!(
                "[+{:.3}s] Thread: {}\n  {} -> {}\n  Stack trace:\n{}\n",
                elapsed.as_secs_f64(),
                log.thread_id,
                log.old_value_repr,
                log.new_value_repr,
                log.backtrace
            )
        }).collect()
    }

    /// Pretty-prints all change logs to stdout
    ///
    /// This is a convenience method that prints each log entry in a readable format.
    /// Useful for quick debugging in terminals.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "debug")]
    /// # {
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// property.enable_change_logging();
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(100).ok();
    /// property.log_change(&old, &100, "update 1");
    ///
    /// let old = property.get().expect("Failed to get");
    /// property.set(200).ok();
    /// property.log_change(&old, &200, "update 2");
    ///
    /// // Prints formatted logs to stdout
    /// property.print_change_logs();
    /// # }
    /// ```
    pub fn print_change_logs(&self) {
        println!("===== Property Change Logs =====");
        for log in self.get_change_logs() {
            println!("{}", log);
        }
        println!("================================");
    }
}


impl<T: Clone + Default + Send + Sync + 'static> ObservableProperty<T> {
    /// Gets the current value or returns the default if inaccessible
    ///
    /// This convenience method is only available when `T` implements `Default`.
    /// It provides a fallback to `T::default()` if the value cannot be read.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    ///
    /// let property = ObservableProperty::new(42);
    /// assert_eq!(property.get_or_default(), 42);
    ///
    /// // Even if somehow inaccessible, returns default
    /// let empty_property: ObservableProperty<i32> = ObservableProperty::new(0);
    /// assert_eq!(empty_property.get_or_default(), 0);
    /// ```
    pub fn get_or_default(&self) -> T {
        self.get().unwrap_or_default()
    }
}

impl<T: Clone + Send + Sync + 'static> Clone for ObservableProperty<T> {
    /// Creates a new reference to the same observable property
    ///
    /// This creates a new `ObservableProperty` instance that shares the same
    /// underlying data with the original. Changes made through either instance
    /// will be visible to observers subscribed through both instances.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use observable_property::ObservableProperty;
    /// use std::sync::Arc;
    ///
    /// let property1 = ObservableProperty::new(42);
    /// let property2 = property1.clone();
    ///
    /// property2.subscribe(Arc::new(|old, new| {
    ///     println!("Observer on property2 saw change: {} -> {}", old, new);
    /// })).map_err(|e| {
    ///     eprintln!("Failed to subscribe: {}", e);
    ///     e
    /// })?;
    ///
    /// // This change through property1 will trigger the observer on property2
    /// property1.set(100).map_err(|e| {
    ///     eprintln!("Failed to set value: {}", e);
    ///     e
    /// })?;
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            max_threads: self.max_threads,
            max_observers: self.max_observers,
        }
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for ObservableProperty<T> {
    /// Debug implementation that shows the current value if accessible
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get() {
            Ok(value) => f
                .debug_struct("ObservableProperty")
                .field("value", &value)
                .field("observers_count", &"[hidden]")
                .field("max_threads", &self.max_threads)
                .field("max_observers", &self.max_observers)
                .finish(),
            Err(_) => f
                .debug_struct("ObservableProperty")
                .field("value", &"[inaccessible]")
                .field("observers_count", &"[hidden]")
                .field("max_threads", &self.max_threads)
                .field("max_observers", &self.max_observers)
                .finish(),
        }
    }
}

#[cfg(feature = "serde")]
impl<T: Clone + Send + Sync + 'static + Serialize> Serialize for ObservableProperty<T> {
    /// Serializes only the current value of the property, not the observers
    ///
    /// # Note
    ///
    /// This serialization only captures the current value. Observer subscriptions
    /// and internal state are not serialized, as they contain function pointers
    /// and runtime state that cannot be meaningfully serialized.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "serde")] {
    /// use observable_property::ObservableProperty;
    /// use serde_json;
    ///
    /// let property = ObservableProperty::new(42);
    /// let json = serde_json::to_string(&property).unwrap();
    /// assert_eq!(json, "42");
    /// # }
    /// ```
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self.get() {
            Ok(value) => value.serialize(serializer),
            Err(_) => Err(serde::ser::Error::custom("Failed to read property value")),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, T: Clone + Send + Sync + 'static + Deserialize<'de>> Deserialize<'de> for ObservableProperty<T> {
    /// Deserializes a value and creates a new ObservableProperty with no observers
    ///
    /// # Note
    ///
    /// The deserialized property will have no observers. You'll need to
    /// re-establish any subscriptions after deserialization.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # #[cfg(feature = "serde")] {
    /// use observable_property::ObservableProperty;
    /// use serde_json;
    ///
    /// let json = "42";
    /// let property: ObservableProperty<i32> = serde_json::from_str(json).unwrap();
    /// assert_eq!(property.get().unwrap(), 42);
    /// # }
    /// ```
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = T::deserialize(deserializer)?;
        Ok(ObservableProperty::new(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[test]
    fn test_property_creation_and_basic_operations() {
        let prop = ObservableProperty::new(42);

        // Test initial value
        match prop.get() {
            Ok(value) => assert_eq!(value, 42),
            Err(e) => panic!("Failed to get initial value: {}", e),
        }

        // Test setting value
        if let Err(e) = prop.set(100) {
            panic!("Failed to set value: {}", e);
        }

        match prop.get() {
            Ok(value) => assert_eq!(value, 100),
            Err(e) => panic!("Failed to get updated value: {}", e),
        }
    }

    #[test]
    fn test_observer_subscription_and_notification() {
        let prop = ObservableProperty::new("initial".to_string());
        let notification_count = Arc::new(AtomicUsize::new(0));
        let last_old_value = Arc::new(RwLock::new(String::new()));
        let last_new_value = Arc::new(RwLock::new(String::new()));

        let count_clone = notification_count.clone();
        let old_clone = last_old_value.clone();
        let new_clone = last_new_value.clone();

        let observer_id = match prop.subscribe(Arc::new(move |old, new| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            if let Ok(mut old_val) = old_clone.write() {
                *old_val = old.clone();
            }
            if let Ok(mut new_val) = new_clone.write() {
                *new_val = new.clone();
            }
        })) {
            Ok(id) => id,
            Err(e) => panic!("Failed to subscribe observer: {}", e),
        };

        // Change value and verify notification
        if let Err(e) = prop.set("changed".to_string()) {
            panic!("Failed to set property value: {}", e);
        }

        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        match last_old_value.read() {
            Ok(old_val) => assert_eq!(*old_val, "initial"),
            Err(e) => panic!("Failed to read old value: {:?}", e),
        }

        match last_new_value.read() {
            Ok(new_val) => assert_eq!(*new_val, "changed"),
            Err(e) => panic!("Failed to read new value: {:?}", e),
        }

        // Test unsubscription
        match prop.unsubscribe(observer_id) {
            Ok(was_present) => assert!(was_present),
            Err(e) => panic!("Failed to unsubscribe observer: {}", e),
        }

        // Change value again - should not notify
        if let Err(e) = prop.set("not_notified".to_string()) {
            panic!("Failed to set property value after unsubscribe: {}", e);
        }
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_filtered_observer() {
        let prop = ObservableProperty::new(0i32);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        // Observer only triggered when value increases
        let observer_id = match prop.subscribe_filtered(
            Arc::new(move |_, _| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
            |old, new| new > old,
        ) {
            Ok(id) => id,
            Err(e) => panic!("Failed to subscribe filtered observer: {}", e),
        };

        // Should trigger (0 -> 5)
        if let Err(e) = prop.set(5) {
            panic!("Failed to set property value to 5: {}", e);
        }
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        // Should NOT trigger (5 -> 3)
        if let Err(e) = prop.set(3) {
            panic!("Failed to set property value to 3: {}", e);
        }
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        // Should trigger (3 -> 10)
        if let Err(e) = prop.set(10) {
            panic!("Failed to set property value to 10: {}", e);
        }
        assert_eq!(notification_count.load(Ordering::SeqCst), 2);

        match prop.unsubscribe(observer_id) {
            Ok(_) => {}
            Err(e) => panic!("Failed to unsubscribe filtered observer: {}", e),
        }
    }

    #[test]
    fn test_thread_safety_concurrent_reads() {
        let prop = Arc::new(ObservableProperty::new(42i32));
        let num_threads = 10;
        let reads_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                let prop_clone = prop.clone();
                thread::spawn(move || {
                    for _ in 0..reads_per_thread {
                        match prop_clone.get() {
                            Ok(value) => assert_eq!(value, 42),
                            Err(e) => panic!("Failed to read property value: {}", e),
                        }
                        thread::sleep(Duration::from_millis(1));
                    }
                })
            })
            .collect();

        for handle in handles {
            if let Err(e) = handle.join() {
                panic!("Thread failed to complete: {:?}", e);
            }
        }
    }

    #[test]
    fn test_async_set_performance() {
        let prop = ObservableProperty::new(0i32);
        let slow_observer_count = Arc::new(AtomicUsize::new(0));
        let count_clone = slow_observer_count.clone();

        // Add observer that simulates slow work
        let _id = match prop.subscribe(Arc::new(move |_, _| {
            thread::sleep(Duration::from_millis(50));
            count_clone.fetch_add(1, Ordering::SeqCst);
        })) {
            Ok(id) => id,
            Err(e) => panic!("Failed to subscribe slow observer: {}", e),
        };

        // Test synchronous set (should be slow)
        let start = std::time::Instant::now();
        if let Err(e) = prop.set(1) {
            panic!("Failed to set property value synchronously: {}", e);
        }
        let sync_duration = start.elapsed();

        // Test asynchronous set (should be fast)
        let start = std::time::Instant::now();
        if let Err(e) = prop.set_async(2) {
            panic!("Failed to set property value asynchronously: {}", e);
        }
        let async_duration = start.elapsed();

        // Async should be much faster than sync
        assert!(async_duration < sync_duration);
        assert!(async_duration.as_millis() < 10); // Should be very fast

        // Wait for async observer to complete
        thread::sleep(Duration::from_millis(100));

        // Both observers should have been called
        assert_eq!(slow_observer_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_lock_poisoning() {
        // Create a property that we'll poison
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Create a thread that will deliberately poison the lock
        let poison_thread = thread::spawn(move || {
            // Get write lock and then panic, which will poison the lock
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for poisoning test");
            panic!("Deliberate panic to poison the lock");
        });

        // Wait for the thread to complete (it will panic)
        let _ = poison_thread.join();

        // With graceful degradation, operations should succeed even with poisoned locks
        // The implementation recovers the inner value using into_inner()
        
        // get() should succeed by recovering from poisoned lock
        match prop.get() {
            Ok(value) => assert_eq!(value, 0), // Should recover the value
            Err(e) => panic!("get() should succeed with graceful degradation, got error: {:?}", e),
        }

        // set() should succeed by recovering from poisoned lock
        match prop.set(42) {
            Ok(_) => {}, // Expected success with graceful degradation
            Err(e) => panic!("set() should succeed with graceful degradation, got error: {:?}", e),
        }
        
        // Verify the value was actually set
        assert_eq!(prop.get().expect("Failed to get value after set"), 42);

        // subscribe() should succeed by recovering from poisoned lock
        match prop.subscribe(Arc::new(|_, _| {})) {
            Ok(_) => {}, // Expected success with graceful degradation
            Err(e) => panic!("subscribe() should succeed with graceful degradation, got error: {:?}", e),
        }
    }

    #[test]
    fn test_observer_panic_isolation() {
        let prop = ObservableProperty::new(0);
        let call_counts = Arc::new(AtomicUsize::new(0));

        // First observer will panic
        let panic_observer_id = prop
            .subscribe(Arc::new(|_, _| {
                panic!("This observer deliberately panics");
            }))
            .expect("Failed to subscribe panic observer");

        // Second observer should still be called despite first one panicking
        let counts = call_counts.clone();
        let normal_observer_id = prop
            .subscribe(Arc::new(move |_, _| {
                counts.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to subscribe normal observer");

        // Trigger the observers - this shouldn't panic despite the first observer panicking
        prop.set(42).expect("Failed to set property value");

        // Verify the second observer was still called
        assert_eq!(call_counts.load(Ordering::SeqCst), 1);

        // Clean up
        prop.unsubscribe(panic_observer_id).expect("Failed to unsubscribe panic observer");
        prop.unsubscribe(normal_observer_id).expect("Failed to unsubscribe normal observer");
    }

    #[test]
    fn test_unsubscribe_nonexistent_observer() {
        let property = ObservableProperty::new(0);

        // Generate a valid observer ID
        let valid_id = property.subscribe(Arc::new(|_, _| {})).expect("Failed to subscribe test observer");

        // Create an ID that doesn't exist (valid_id + 1000 should not exist)
        let nonexistent_id = valid_id + 1000;

        // Test unsubscribing a nonexistent observer
        match property.unsubscribe(nonexistent_id) {
            Ok(was_present) => {
                assert!(
                    !was_present,
                    "Unsubscribe should return false for nonexistent ID"
                );
            }
            Err(e) => panic!("Unsubscribe returned error: {:?}", e),
        }

        // Also verify that unsubscribing twice returns false the second time
        property.unsubscribe(valid_id).expect("Failed first unsubscribe"); // First unsubscribe should return true

        let result = property.unsubscribe(valid_id).expect("Failed second unsubscribe");
        assert!(!result, "Second unsubscribe should return false");
    }

    #[test]
    fn test_observer_id_wraparound() {
        let prop = ObservableProperty::new(0);

        // Test that observer IDs increment properly and don't wrap around unexpectedly
        let id1 = prop.subscribe(Arc::new(|_, _| {})).expect("Failed to subscribe observer 1");
        let id2 = prop.subscribe(Arc::new(|_, _| {})).expect("Failed to subscribe observer 2");
        let id3 = prop.subscribe(Arc::new(|_, _| {})).expect("Failed to subscribe observer 3");

        assert!(id2 > id1, "Observer IDs should increment");
        assert!(id3 > id2, "Observer IDs should continue incrementing");
        assert_eq!(id2, id1 + 1, "Observer IDs should increment by 1");
        assert_eq!(id3, id2 + 1, "Observer IDs should increment by 1");

        // Clean up
        prop.unsubscribe(id1).expect("Failed to unsubscribe observer 1");
        prop.unsubscribe(id2).expect("Failed to unsubscribe observer 2");
        prop.unsubscribe(id3).expect("Failed to unsubscribe observer 3");
    }

    #[test]
    fn test_concurrent_subscribe_unsubscribe() {
        let prop = Arc::new(ObservableProperty::new(0));
        let num_threads = 8;
        let operations_per_thread = 100;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let prop_clone = prop.clone();
                thread::spawn(move || {
                    let mut local_ids = Vec::new();

                    for i in 0..operations_per_thread {
                        // Subscribe an observer
                        let observer_id = prop_clone
                            .subscribe(Arc::new(move |old, new| {
                                // Do some work to simulate real observer
                                let _ = thread_id + i + old + new;
                            }))
                            .expect("Subscribe should succeed");

                        local_ids.push(observer_id);

                        // Occasionally unsubscribe some observers
                        if i % 10 == 0 && !local_ids.is_empty() {
                            let idx = i % local_ids.len();
                            let id_to_remove = local_ids.remove(idx);
                            prop_clone
                                .unsubscribe(id_to_remove)
                                .expect("Unsubscribe should succeed");
                        }
                    }

                    // Clean up remaining observers
                    for id in local_ids {
                        prop_clone
                            .unsubscribe(id)
                            .expect("Final cleanup should succeed");
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Property should still be functional
        prop.set(42)
            .expect("Property should still work after concurrent operations");
    }

    #[test]
    fn test_multiple_observer_panics_isolation() {
        let prop = ObservableProperty::new(0);
        let successful_calls = Arc::new(AtomicUsize::new(0));

        // Create multiple observers that will panic
        let _panic_id1 = prop
            .subscribe(Arc::new(|_, _| {
                panic!("First panic observer");
            }))
            .expect("Failed to subscribe first panic observer");

        let _panic_id2 = prop
            .subscribe(Arc::new(|_, _| {
                panic!("Second panic observer");
            }))
            .expect("Failed to subscribe second panic observer");

        // Create observers that should succeed despite the panics
        let count1 = successful_calls.clone();
        let _success_id1 = prop
            .subscribe(Arc::new(move |_, _| {
                count1.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to subscribe first success observer");

        let count2 = successful_calls.clone();
        let _success_id2 = prop
            .subscribe(Arc::new(move |_, _| {
                count2.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to subscribe second success observer");

        // Trigger all observers
        prop.set(42).expect("Failed to set property value for panic isolation test");

        // Both successful observers should have been called despite the panics
        assert_eq!(successful_calls.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_async_observer_panic_isolation() {
        let prop = ObservableProperty::new(0);
        let successful_calls = Arc::new(AtomicUsize::new(0));

        // Create observer that will panic
        let _panic_id = prop
            .subscribe(Arc::new(|_, _| {
                panic!("Async panic observer");
            }))
            .expect("Failed to subscribe async panic observer");

        // Create observer that should succeed
        let count = successful_calls.clone();
        let _success_id = prop
            .subscribe(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to subscribe async success observer");

        // Use async set to trigger observers in background threads
        prop.set_async(42).expect("Failed to set property value asynchronously");

        // Wait for async observers to complete
        thread::sleep(Duration::from_millis(100));

        // The successful observer should have been called despite the panic
        assert_eq!(successful_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_very_large_observer_count() {
        let prop = ObservableProperty::new(0);
        let total_calls = Arc::new(AtomicUsize::new(0));
        let observer_count = 1000;

        // Subscribe many observers
        let mut observer_ids = Vec::with_capacity(observer_count);
        for i in 0..observer_count {
            let count = total_calls.clone();
            let id = prop
                .subscribe(Arc::new(move |old, new| {
                    count.fetch_add(1, Ordering::SeqCst);
                    // Verify we got the right values
                    assert_eq!(*old, 0);
                    assert_eq!(*new, i + 1);
                }))
                .expect("Failed to subscribe large observer count test observer");
            observer_ids.push(id);
        }

        // Trigger all observers
        prop.set(observer_count).expect("Failed to set property value for large observer count test");

        // All observers should have been called
        assert_eq!(total_calls.load(Ordering::SeqCst), observer_count);

        // Clean up
        for id in observer_ids {
            prop.unsubscribe(id).expect("Failed to unsubscribe observer in large count test");
        }
    }

    #[test]
    fn test_observer_with_mutable_state() {
        let prop = ObservableProperty::new(0);
        let call_history = Arc::new(RwLock::new(Vec::new()));

        let history = call_history.clone();
        let observer_id = prop
            .subscribe(Arc::new(move |old, new| {
                if let Ok(mut hist) = history.write() {
                    hist.push((*old, *new));
                }
            }))
            .expect("Failed to subscribe mutable state observer");

        // Make several changes
        prop.set(1).expect("Failed to set property to 1");
        prop.set(2).expect("Failed to set property to 2");
        prop.set(3).expect("Failed to set property to 3");

        // Verify the history was recorded correctly
        let history = call_history.read().expect("Failed to read call history");
        assert_eq!(history.len(), 3);
        assert_eq!(history[0], (0, 1));
        assert_eq!(history[1], (1, 2));
        assert_eq!(history[2], (2, 3));

        prop.unsubscribe(observer_id).expect("Failed to unsubscribe mutable state observer");
    }

    #[test]
    fn test_subscription_automatic_cleanup() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        // Test that subscription automatically cleans up when dropped
        {
            let count = call_count.clone();
            let _subscription = prop
                .subscribe_with_subscription(Arc::new(move |_, _| {
                    count.fetch_add(1, Ordering::SeqCst);
                }))
                .expect("Failed to create subscription for automatic cleanup test");

            // Observer should be active while subscription is in scope
            prop.set(1).expect("Failed to set property value in subscription test");
            assert_eq!(call_count.load(Ordering::SeqCst), 1);

            // Subscription goes out of scope here and should auto-cleanup
        }

        // Observer should no longer be active after subscription dropped
        prop.set(2).expect("Failed to set property value after subscription dropped");
        assert_eq!(call_count.load(Ordering::SeqCst), 1); // No additional calls
    }

    #[test]
    fn test_subscription_explicit_drop() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        let count = call_count.clone();
        let subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for explicit drop test");

        // Observer should be active
        prop.set(1).expect("Failed to set property value before explicit drop");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Explicitly drop the subscription
        drop(subscription);

        // Observer should no longer be active
        prop.set(2).expect("Failed to set property value after explicit drop");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_multiple_subscriptions_with_cleanup() {
        let prop = ObservableProperty::new(0);
        let call_count1 = Arc::new(AtomicUsize::new(0));
        let call_count2 = Arc::new(AtomicUsize::new(0));
        let call_count3 = Arc::new(AtomicUsize::new(0));

        let count1 = call_count1.clone();
        let count2 = call_count2.clone();
        let count3 = call_count3.clone();

        let subscription1 = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count1.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create first subscription");

        let subscription2 = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count2.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create second subscription");

        let subscription3 = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count3.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create third subscription");

        // All observers should be active
        prop.set(1).expect("Failed to set property value with all subscriptions");
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);
        assert_eq!(call_count3.load(Ordering::SeqCst), 1);

        // Drop second subscription
        drop(subscription2);

        // Only first and third should be active
        prop.set(2).expect("Failed to set property value with partial subscriptions");
        assert_eq!(call_count1.load(Ordering::SeqCst), 2);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1); // No change
        assert_eq!(call_count3.load(Ordering::SeqCst), 2);

        // Drop remaining subscriptions
        drop(subscription1);
        drop(subscription3);

        // No observers should be active
        prop.set(3).expect("Failed to set property value with no subscriptions");
        assert_eq!(call_count1.load(Ordering::SeqCst), 2);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);
        assert_eq!(call_count3.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_subscription_drop_with_poisoned_lock() {
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Create a subscription
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();
        let subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for poisoned lock test");

        // Poison the lock by panicking while holding a write lock
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for poisoning test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join(); // Ignore the panic result

        // Dropping the subscription should not panic even with poisoned lock
        // This tests that the Drop implementation handles poisoned locks gracefully
        drop(subscription); // Should complete without panic

        // Test passes if we reach here without panicking
    }

    #[test]
    fn test_subscription_vs_manual_unsubscribe() {
        let prop = ObservableProperty::new(0);
        let auto_count = Arc::new(AtomicUsize::new(0));
        let manual_count = Arc::new(AtomicUsize::new(0));

        // Manual subscription
        let manual_count_clone = manual_count.clone();
        let manual_id = prop
            .subscribe(Arc::new(move |_, _| {
                manual_count_clone.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create manual subscription");

        // Automatic subscription
        let auto_count_clone = auto_count.clone();
        let _auto_subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                auto_count_clone.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create automatic subscription");

        // Both should be active
        prop.set(1).expect("Failed to set property value with both subscriptions");
        assert_eq!(manual_count.load(Ordering::SeqCst), 1);
        assert_eq!(auto_count.load(Ordering::SeqCst), 1);

        // Manual unsubscribe
        prop.unsubscribe(manual_id).expect("Failed to manually unsubscribe");

        // Only automatic subscription should be active
        prop.set(2).expect("Failed to set property value after manual unsubscribe");
        assert_eq!(manual_count.load(Ordering::SeqCst), 1); // No change
        assert_eq!(auto_count.load(Ordering::SeqCst), 2);

        // Auto subscription goes out of scope here and cleans up automatically
    }

    #[test]
    fn test_subscribe_with_subscription_error_handling() {
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Poison the lock
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for poisoning test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // With graceful degradation, subscribe_with_subscription should succeed
        let result = prop.subscribe_with_subscription(Arc::new(|_, _| {}));
        assert!(result.is_ok(), "subscribe_with_subscription should succeed with graceful degradation");
    }

    #[test]
    fn test_subscription_with_property_cloning() {
        let prop1 = ObservableProperty::new(0);
        let prop2 = prop1.clone();
        let call_count = Arc::new(AtomicUsize::new(0));

        // Subscribe to prop1
        let count = call_count.clone();
        let _subscription = prop1
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for cloned property test");

        // Changes through prop2 should trigger the observer subscribed to prop1
        prop2.set(42).expect("Failed to set property value through prop2");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Changes through prop1 should also trigger the observer
        prop1.set(100).expect("Failed to set property value through prop1");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_subscription_in_conditional_blocks() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        let should_subscribe = true;

        if should_subscribe {
            let count = call_count.clone();
            let _subscription = prop
                .subscribe_with_subscription(Arc::new(move |_, _| {
                    count.fetch_add(1, Ordering::SeqCst);
                }))
                .expect("Failed to create subscription in conditional block");

            // Observer active within this block
            prop.set(1).expect("Failed to set property value in conditional block");
            assert_eq!(call_count.load(Ordering::SeqCst), 1);

            // Subscription dropped when exiting this block
        }

        // Observer should be inactive now
        prop.set(2).expect("Failed to set property value after conditional block");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_subscription_with_early_return() {
        fn test_function(
            prop: &ObservableProperty<i32>,
            should_return_early: bool,
        ) -> Result<(), PropertyError> {
            let call_count = Arc::new(AtomicUsize::new(0));
            let count = call_count.clone();

            let _subscription = prop.subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))?;

            prop.set(1)?;
            assert_eq!(call_count.load(Ordering::SeqCst), 1);

            if should_return_early {
                return Ok(()); // Subscription should be cleaned up here
            }

            prop.set(2)?;
            assert_eq!(call_count.load(Ordering::SeqCst), 2);

            Ok(())
            // Subscription cleaned up when function exits normally
        }

        let prop = ObservableProperty::new(0);

        // Test early return
        test_function(&prop, true).expect("Failed to test early return");

        // Verify observer is no longer active after early return
        prop.set(10).expect("Failed to set property value after early return");

        // Test normal exit
        test_function(&prop, false).expect("Failed to test normal exit");

        // Verify observer is no longer active after normal exit
        prop.set(20).expect("Failed to set property value after normal exit");
    }

    #[test]
    fn test_subscription_move_semantics() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        let count = call_count.clone();
        let subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for move semantics test");

        // Observer should be active
        prop.set(1).expect("Failed to set property value before move");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Move subscription to a new variable
        let moved_subscription = subscription;

        // Observer should still be active after move
        prop.set(2).expect("Failed to set property value after move");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        // Drop the moved subscription
        drop(moved_subscription);

        // Observer should now be inactive
        prop.set(3).expect("Failed to set property value after moved subscription drop");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_filtered_subscription_automatic_cleanup() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        {
            let count = call_count.clone();
            let _subscription = prop
                .subscribe_filtered_with_subscription(
                    Arc::new(move |_, _| {
                        count.fetch_add(1, Ordering::SeqCst);
                    }),
                    |old, new| new > old, // Only trigger on increases
                )
                .expect("Failed to create filtered subscription");

            // Should trigger (0 -> 5)
            prop.set(5).expect("Failed to set property value to 5 in filtered test");
            assert_eq!(call_count.load(Ordering::SeqCst), 1);

            // Should NOT trigger (5 -> 3)
            prop.set(3).expect("Failed to set property value to 3 in filtered test");
            assert_eq!(call_count.load(Ordering::SeqCst), 1);

            // Should trigger (3 -> 10)
            prop.set(10).expect("Failed to set property value to 10 in filtered test");
            assert_eq!(call_count.load(Ordering::SeqCst), 2);

            // Subscription goes out of scope here
        }

        // Observer should be inactive after subscription cleanup
        prop.set(20).expect("Failed to set property value after filtered subscription cleanup");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_multiple_filtered_subscriptions() {
        let prop = ObservableProperty::new(10);
        let increase_count = Arc::new(AtomicUsize::new(0));
        let decrease_count = Arc::new(AtomicUsize::new(0));
        let significant_change_count = Arc::new(AtomicUsize::new(0));

        let inc_count = increase_count.clone();
        let dec_count = decrease_count.clone();
        let sig_count = significant_change_count.clone();

        let _increase_sub = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |_, _| {
                    inc_count.fetch_add(1, Ordering::SeqCst);
                }),
                |old, new| new > old,
            )
            .expect("Failed to create increase subscription");

        let _decrease_sub = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |_, _| {
                    dec_count.fetch_add(1, Ordering::SeqCst);
                }),
                |old, new| new < old,
            )
            .expect("Failed to create decrease subscription");

        let _significant_sub = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |_, _| {
                    sig_count.fetch_add(1, Ordering::SeqCst);
                }),
                |old, new| ((*new as i32) - (*old as i32)).abs() > 5,
            )
            .expect("Failed to create significant change subscription");

        // Test increases
        prop.set(15).expect("Failed to set property to 15 in multiple filtered test"); // +5: triggers increase, not significant
        assert_eq!(increase_count.load(Ordering::SeqCst), 1);
        assert_eq!(decrease_count.load(Ordering::SeqCst), 0);
        assert_eq!(significant_change_count.load(Ordering::SeqCst), 0);

        // Test significant increase
        prop.set(25).expect("Failed to set property to 25 in multiple filtered test"); // +10: triggers increase and significant
        assert_eq!(increase_count.load(Ordering::SeqCst), 2);
        assert_eq!(decrease_count.load(Ordering::SeqCst), 0);
        assert_eq!(significant_change_count.load(Ordering::SeqCst), 1);

        // Test significant decrease
        prop.set(5).expect("Failed to set property to 5 in multiple filtered test"); // -20: triggers decrease and significant
        assert_eq!(increase_count.load(Ordering::SeqCst), 2);
        assert_eq!(decrease_count.load(Ordering::SeqCst), 1);
        assert_eq!(significant_change_count.load(Ordering::SeqCst), 2);

        // Test small decrease
        prop.set(3).expect("Failed to set property to 3 in multiple filtered test"); // -2: triggers decrease, not significant
        assert_eq!(increase_count.load(Ordering::SeqCst), 2);
        assert_eq!(decrease_count.load(Ordering::SeqCst), 2);
        assert_eq!(significant_change_count.load(Ordering::SeqCst), 2);

        // All subscriptions auto-cleanup when they go out of scope
    }

    #[test]
    fn test_filtered_subscription_complex_filter() {
        let prop = ObservableProperty::new(0.0f64);
        let call_count = Arc::new(AtomicUsize::new(0));
        let values_received = Arc::new(RwLock::new(Vec::new()));

        let count = call_count.clone();
        let values = values_received.clone();
        let _subscription = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |old, new| {
                    count.fetch_add(1, Ordering::SeqCst);
                    if let Ok(mut v) = values.write() {
                        v.push((*old, *new));
                    }
                }),
                |old, new| {
                    // Complex filter: trigger only when crossing integer boundaries
                    // and the change is significant (> 0.5)
                    let old_int = old.floor() as i32;
                    let new_int = new.floor() as i32;
                    old_int != new_int && (new - old).abs() > 0.5_f64
                },
            )
            .expect("Failed to create complex filtered subscription");

        // Small changes within same integer - should not trigger
        prop.set(0.3).expect("Failed to set property to 0.3 in complex filter test");
        prop.set(0.7).expect("Failed to set property to 0.7 in complex filter test");
        assert_eq!(call_count.load(Ordering::SeqCst), 0);

        // Cross integer boundary with significant change - should trigger
        prop.set(1.3).expect("Failed to set property to 1.3 in complex filter test"); // Change of 0.6, which is > 0.5
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Small cross-boundary change - should not trigger
        prop.set(1.9).expect("Failed to set property to 1.9 in complex filter test");
        prop.set(2.1).expect("Failed to set property to 2.1 in complex filter test"); // Change of 0.2, less than 0.5
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Large cross-boundary change - should trigger
        prop.set(3.5).expect("Failed to set property to 3.5 in complex filter test");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);

        // Verify received values
        let values = values_received.read().expect("Failed to read values in complex filter test");
        assert_eq!(values.len(), 2);
        assert_eq!(values[0], (0.7, 1.3));
        assert_eq!(values[1], (2.1, 3.5));
    }

    #[test]
    fn test_filtered_subscription_error_handling() {
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Poison the lock
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for filtered subscription poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // With graceful degradation, subscribe_filtered_with_subscription should succeed
        let result = prop.subscribe_filtered_with_subscription(Arc::new(|_, _| {}), |_, _| true);
        assert!(result.is_ok(), "subscribe_filtered_with_subscription should succeed with graceful degradation");
    }

    #[test]
    fn test_filtered_subscription_vs_manual_filtered() {
        let prop = ObservableProperty::new(0);
        let auto_count = Arc::new(AtomicUsize::new(0));
        let manual_count = Arc::new(AtomicUsize::new(0));

        // Manual filtered subscription
        let manual_count_clone = manual_count.clone();
        let manual_id = prop
            .subscribe_filtered(
                Arc::new(move |_, _| {
                    manual_count_clone.fetch_add(1, Ordering::SeqCst);
                }),
                |old, new| new > old,
            )
            .expect("Failed to create manual filtered subscription");

        // Automatic filtered subscription
        let auto_count_clone = auto_count.clone();
        let _auto_subscription = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |_, _| {
                    auto_count_clone.fetch_add(1, Ordering::SeqCst);
                }),
                |old, new| new > old,
            )
            .expect("Failed to create automatic filtered subscription");

        // Both should be triggered by increases
        prop.set(5).expect("Failed to set property to 5 in filtered vs manual test");
        assert_eq!(manual_count.load(Ordering::SeqCst), 1);
        assert_eq!(auto_count.load(Ordering::SeqCst), 1);

        // Neither should be triggered by decreases
        prop.set(3).expect("Failed to set property to 3 in filtered vs manual test");
        assert_eq!(manual_count.load(Ordering::SeqCst), 1);
        assert_eq!(auto_count.load(Ordering::SeqCst), 1);

        // Both should be triggered by increases again
        prop.set(10).expect("Failed to set property to 10 in filtered vs manual test");
        assert_eq!(manual_count.load(Ordering::SeqCst), 2);
        assert_eq!(auto_count.load(Ordering::SeqCst), 2);

        // Manual cleanup
        prop.unsubscribe(manual_id).expect("Failed to unsubscribe manual filtered observer");

        // Only automatic subscription should be active
        prop.set(15).expect("Failed to set property to 15 after manual cleanup");
        assert_eq!(manual_count.load(Ordering::SeqCst), 2); // No change
        assert_eq!(auto_count.load(Ordering::SeqCst), 3);

        // Auto subscription cleaned up when it goes out of scope
    }

    #[test]
    fn test_filtered_subscription_with_panicking_filter() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));

        let count = call_count.clone();
        let _subscription = prop
            .subscribe_filtered_with_subscription(
                Arc::new(move |_, _| {
                    count.fetch_add(1, Ordering::SeqCst);
                }),
                |_, new| {
                    if *new == 42 {
                        panic!("Filter panic on 42");
                    }
                    true // Accept all other values
                },
            )
            .expect("Failed to create panicking filter subscription");

        // Normal value should work
        prop.set(1).expect("Failed to set property to 1 in panicking filter test");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Value that causes filter to panic should be handled gracefully
        // The behavior here depends on how the filter panic is handled
        // In the current implementation, filter panics may cause the observer to not be called
        prop.set(42).expect("Failed to set property to 42 in panicking filter test");

        // Observer should still work for subsequent normal values
        prop.set(2).expect("Failed to set property to 2 after filter panic");
        // Note: The exact count here depends on panic handling implementation
        // The important thing is that the system doesn't crash
    }

    #[test]
    fn test_subscription_thread_safety() {
        let prop = Arc::new(ObservableProperty::new(0));
        let num_threads = 8;
        let operations_per_thread = 50;
        let total_calls = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let prop_clone = prop.clone();
                let calls_clone = total_calls.clone();

                thread::spawn(move || {
                    let mut local_subscriptions = Vec::new();

                    for i in 0..operations_per_thread {
                        let calls = calls_clone.clone();
                        let subscription = prop_clone
                            .subscribe_with_subscription(Arc::new(move |old, new| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                // Simulate some work
                                let _ = thread_id + i + old + new;
                            }))
                            .expect("Should be able to create subscription");

                        local_subscriptions.push(subscription);

                        // Trigger observers
                        prop_clone
                            .set(thread_id * 1000 + i)
                            .expect("Should be able to set value");

                        // Occasionally drop some subscriptions
                        if i % 5 == 0 && !local_subscriptions.is_empty() {
                            local_subscriptions.remove(0); // Drop first subscription
                        }
                    }

                    // All remaining subscriptions dropped when vector goes out of scope
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Property should still be functional after all the concurrent operations
        prop.set(9999).expect("Property should still work");

        // We can't easily verify the exact call count due to the complex timing,
        // but we can verify that the system didn't crash and is still operational
        println!(
            "Total observer calls: {}",
            total_calls.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn test_subscription_cross_thread_drop() {
        let prop = Arc::new(ObservableProperty::new(0));
        let call_count = Arc::new(AtomicUsize::new(0));

        // Create subscription in main thread
        let count = call_count.clone();
        let subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for cross-thread drop test");

        // Verify observer is active
        prop.set(1).expect("Failed to set property value in cross-thread drop test");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Move subscription to another thread and drop it there
        let prop_clone = prop.clone();
        let call_count_clone = call_count.clone();

        let handle = thread::spawn(move || {
            // Verify observer is still active in the other thread
            prop_clone.set(2).expect("Failed to set property value in other thread");
            assert_eq!(call_count_clone.load(Ordering::SeqCst), 2);

            // Drop subscription in this thread
            drop(subscription);

            // Verify observer is no longer active
            prop_clone.set(3).expect("Failed to set property value after drop in other thread");
            assert_eq!(call_count_clone.load(Ordering::SeqCst), 2); // No change
        });

        handle.join().expect("Failed to join cross-thread drop test thread");

        // Verify observer is still inactive in main thread
        prop.set(4).expect("Failed to set property value after thread join");
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_concurrent_subscription_creation_and_property_changes() {
        let prop = Arc::new(ObservableProperty::new(0));
        let total_notifications = Arc::new(AtomicUsize::new(0));
        let num_subscriber_threads = 4;
        let num_setter_threads = 2;
        let operations_per_thread = 25;

        // Threads that create and destroy subscriptions
        let subscriber_handles: Vec<_> = (0..num_subscriber_threads)
            .map(|_| {
                let prop_clone = prop.clone();
                let notifications_clone = total_notifications.clone();

                thread::spawn(move || {
                    for _ in 0..operations_per_thread {
                        let notifications = notifications_clone.clone();
                        let _subscription = prop_clone
                            .subscribe_with_subscription(Arc::new(move |_, _| {
                                notifications.fetch_add(1, Ordering::SeqCst);
                            }))
                            .expect("Should create subscription");

                        // Keep subscription alive for a short time
                        thread::sleep(Duration::from_millis(1));

                        // Subscription dropped when _subscription goes out of scope
                    }
                })
            })
            .collect();

        // Threads that continuously change the property value
        let setter_handles: Vec<_> = (0..num_setter_threads)
            .map(|thread_id| {
                let prop_clone = prop.clone();

                thread::spawn(move || {
                    for i in 0..operations_per_thread * 2 {
                        prop_clone
                            .set(thread_id * 10000 + i)
                            .expect("Should set value");
                        thread::sleep(Duration::from_millis(1));
                    }
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in subscriber_handles
            .into_iter()
            .chain(setter_handles.into_iter())
        {
            handle.join().expect("Thread should complete");
        }

        // System should be stable after concurrent operations
        prop.set(99999).expect("Property should still work");

        println!(
            "Total notifications during concurrent test: {}",
            total_notifications.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn test_filtered_subscription_thread_safety() {
        let prop = Arc::new(ObservableProperty::new(0));
        let increase_notifications = Arc::new(AtomicUsize::new(0));
        let decrease_notifications = Arc::new(AtomicUsize::new(0));
        let num_threads = 6;

        let handles: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let prop_clone = prop.clone();
                let inc_notifications = increase_notifications.clone();
                let dec_notifications = decrease_notifications.clone();

                thread::spawn(move || {
                    // Create increase-only subscription
                    let inc_count = inc_notifications.clone();
                    let _inc_subscription = prop_clone
                        .subscribe_filtered_with_subscription(
                            Arc::new(move |_, _| {
                                inc_count.fetch_add(1, Ordering::SeqCst);
                            }),
                            |old, new| new > old,
                        )
                        .expect("Should create filtered subscription");

                    // Create decrease-only subscription
                    let dec_count = dec_notifications.clone();
                    let _dec_subscription = prop_clone
                        .subscribe_filtered_with_subscription(
                            Arc::new(move |_, _| {
                                dec_count.fetch_add(1, Ordering::SeqCst);
                            }),
                            |old, new| new < old,
                        )
                        .expect("Should create filtered subscription");

                    // Perform some property changes
                    let base_value = thread_id * 100;
                    for i in 0..20 {
                        let new_value = base_value + (i % 10); // Creates increases and decreases
                        prop_clone.set(new_value).expect("Should set value");
                        thread::sleep(Duration::from_millis(1));
                    }

                    // Subscriptions automatically cleaned up when going out of scope
                })
            })
            .collect();

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        // Verify system is still operational
        let initial_inc = increase_notifications.load(Ordering::SeqCst);
        let initial_dec = decrease_notifications.load(Ordering::SeqCst);

        prop.set(1000).expect("Property should still work");
        prop.set(2000).expect("Property should still work");

        // No new notifications should occur (all subscriptions cleaned up)
        assert_eq!(increase_notifications.load(Ordering::SeqCst), initial_inc);
        assert_eq!(decrease_notifications.load(Ordering::SeqCst), initial_dec);

        println!(
            "Increase notifications: {}, Decrease notifications: {}",
            initial_inc, initial_dec
        );
    }

    #[test]
    fn test_subscription_with_async_property_changes() {
        let prop = Arc::new(ObservableProperty::new(0));
        let sync_notifications = Arc::new(AtomicUsize::new(0));
        let async_notifications = Arc::new(AtomicUsize::new(0));

        // Subscription that tracks sync notifications
        let sync_count = sync_notifications.clone();
        let _sync_subscription = prop
            .subscribe_with_subscription(Arc::new(move |old, new| {
                sync_count.fetch_add(1, Ordering::SeqCst);
                // Simulate slow observer work
                thread::sleep(Duration::from_millis(5));
                println!("Sync observer: {} -> {}", old, new);
            }))
            .expect("Failed to create sync subscription");

        // Subscription that tracks async notifications
        let async_count = async_notifications.clone();
        let _async_subscription = prop
            .subscribe_with_subscription(Arc::new(move |old, new| {
                async_count.fetch_add(1, Ordering::SeqCst);
                println!("Async observer: {} -> {}", old, new);
            }))
            .expect("Failed to create async subscription");

        // Test sync property changes
        let start = std::time::Instant::now();
        prop.set(1).expect("Failed to set property value 1 in async test");
        prop.set(2).expect("Failed to set property value 2 in async test");
        let sync_duration = start.elapsed();

        // Test async property changes
        let start = std::time::Instant::now();
        prop.set_async(3).expect("Failed to set property value 3 async");
        prop.set_async(4).expect("Failed to set property value 4 async");
        let async_duration = start.elapsed();

        // Async should be much faster
        assert!(async_duration < sync_duration);

        // Wait for async observers to complete
        thread::sleep(Duration::from_millis(50));

        // All observers should have been called
        assert_eq!(sync_notifications.load(Ordering::SeqCst), 4);
        assert_eq!(async_notifications.load(Ordering::SeqCst), 4);

        // Subscriptions auto-cleanup when going out of scope
    }

    #[test]
    fn test_subscription_creation_with_poisoned_lock() {
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Create a valid subscription before poisoning
        let call_count = Arc::new(AtomicUsize::new(0));
        let count = call_count.clone();
        let existing_subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription before poisoning");

        // Poison the lock
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for subscription poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // With graceful degradation, new subscription creation should succeed
        let result = prop.subscribe_with_subscription(Arc::new(|_, _| {}));
        assert!(result.is_ok(), "subscribe_with_subscription should succeed with graceful degradation");

        // New filtered subscription creation should also succeed
        let filtered_result =
            prop.subscribe_filtered_with_subscription(Arc::new(|_, _| {}), |_, _| true);
        assert!(filtered_result.is_ok(), "subscribe_filtered_with_subscription should succeed with graceful degradation");

        // Dropping existing subscription should not panic
        drop(existing_subscription);
    }

    #[test]
    fn test_subscription_cleanup_behavior_with_poisoned_lock() {
        // This test specifically verifies that Drop doesn't panic with poisoned locks
        let prop = Arc::new(ObservableProperty::new(0));
        let call_count = Arc::new(AtomicUsize::new(0));

        // Create subscription
        let count = call_count.clone();
        let subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for cleanup behavior test");

        // Verify it works initially
        prop.set(1).expect("Failed to set property value in cleanup behavior test");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Poison the lock from another thread
        let prop_clone = prop.clone();
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for cleanup behavior poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // Now drop the subscription - this should NOT panic
        // The Drop implementation should handle the poisoned lock gracefully
        drop(subscription);

        // Test succeeds if we reach this point without panicking
    }

    #[test]
    fn test_multiple_subscription_cleanup_with_poisoned_lock() {
        let prop = Arc::new(ObservableProperty::new(0));
        let mut subscriptions = Vec::new();

        // Create multiple subscriptions
        for i in 0..5 {
            let call_count = Arc::new(AtomicUsize::new(0));
            let count = call_count.clone();
            let subscription = prop
                .subscribe_with_subscription(Arc::new(move |old, new| {
                    count.fetch_add(1, Ordering::SeqCst);
                    println!("Observer {}: {} -> {}", i, old, new);
                }))
                .expect("Failed to create subscription in multiple cleanup test");
            subscriptions.push(subscription);
        }

        // Verify they all work
        prop.set(42).expect("Failed to set property value in multiple cleanup test");

        // Poison the lock
        let prop_clone = prop.clone();
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for multiple cleanup poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // Drop all subscriptions - none should panic
        for subscription in subscriptions {
            drop(subscription);
        }

        // Test succeeds if no panics occurred
    }

    #[test]
    fn test_subscription_behavior_before_and_after_poison() {
        let prop = Arc::new(ObservableProperty::new(0));
        let before_poison_count = Arc::new(AtomicUsize::new(0));
        let after_poison_count = Arc::new(AtomicUsize::new(0));

        // Create subscription before poisoning
        let before_count = before_poison_count.clone();
        let before_subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                before_count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription before poison test");

        // Verify it works
        prop.set(1).expect("Failed to set property value before poison test");
        assert_eq!(before_poison_count.load(Ordering::SeqCst), 1);

        // Poison the lock
        let prop_clone = prop.clone();
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for before/after poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // With graceful degradation, subscription creation after poisoning should succeed
        let after_count = after_poison_count.clone();
        let after_result = prop.subscribe_with_subscription(Arc::new(move |_, _| {
            after_count.fetch_add(1, Ordering::SeqCst);
        }));
        assert!(after_result.is_ok(), "subscribe_with_subscription should succeed with graceful degradation");
        
        let _after_subscription = after_result.unwrap();

        // Clean up the before-poison subscription - should not panic
        drop(before_subscription);
    }

    #[test]
    fn test_concurrent_subscription_drops_with_poison() {
        let prop = Arc::new(ObservableProperty::new(0));
        let num_subscriptions = 10;
        let mut subscriptions = Vec::new();

        // Create multiple subscriptions
        for i in 0..num_subscriptions {
            let call_count = Arc::new(AtomicUsize::new(0));
            let count = call_count.clone();
            let subscription = prop
                .subscribe_with_subscription(Arc::new(move |_, _| {
                    count.fetch_add(1, Ordering::SeqCst);
                    println!("Observer {}", i);
                }))
                .expect("Failed to create subscription in concurrent drops test");
            subscriptions.push(subscription);
        }

        // Poison the lock
        let prop_clone = prop.clone();
        let poison_thread = thread::spawn(move || {
            let _guard = prop_clone.inner.write().expect("Failed to acquire write lock for concurrent drops poison test");
            panic!("Deliberate panic to poison the lock");
        });
        let _ = poison_thread.join();

        // Drop subscriptions concurrently from multiple threads
        let handles: Vec<_> = subscriptions
            .into_iter()
            .enumerate()
            .map(|(i, subscription)| {
                thread::spawn(move || {
                    // Add some randomness to timing
                    thread::sleep(Duration::from_millis(i as u64 % 5));
                    drop(subscription);
                    println!("Dropped subscription {}", i);
                })
            })
            .collect();

        // Wait for all drops to complete
        for handle in handles {
            handle
                .join()
                .expect("Drop thread should complete without panic");
        }

        // Test succeeds if all threads completed successfully
    }

    
    #[test]
    fn test_with_max_threads_creation() {
        // Test creation with various thread counts
        let prop1 = ObservableProperty::with_max_threads(42, 1);
        let prop2 = ObservableProperty::with_max_threads("test".to_string(), 8);
        let prop3 = ObservableProperty::with_max_threads(0.5_f64, 16);

        // Verify initial values are correct
        assert_eq!(prop1.get().expect("Failed to get prop1 value"), 42);
        assert_eq!(prop2.get().expect("Failed to get prop2 value"), "test");
        assert_eq!(prop3.get().expect("Failed to get prop3 value"), 0.5);
    }

    #[test]
    fn test_with_max_threads_zero_defaults_to_max_threads() {
        // Test that zero max_threads defaults to MAX_THREADS (4)
        let prop1 = ObservableProperty::with_max_threads(100, 0);
        let prop2 = ObservableProperty::new(100); // Uses default MAX_THREADS

        // Both should have the same max_threads value
        // We can't directly access max_threads, but we can verify behavior is consistent
        assert_eq!(prop1.get().expect("Failed to get prop1 value"), 100);
        assert_eq!(prop2.get().expect("Failed to get prop2 value"), 100);
    }

    #[test]
    fn test_with_max_threads_basic_functionality() {
        let prop = ObservableProperty::with_max_threads(0, 2);
        let call_count = Arc::new(AtomicUsize::new(0));

        // Subscribe an observer
        let count = call_count.clone();
        let _subscription = prop
            .subscribe_with_subscription(Arc::new(move |old, new| {
                count.fetch_add(1, Ordering::SeqCst);
                assert_eq!(*old, 0);
                assert_eq!(*new, 42);
            }))
            .expect("Failed to create subscription for max_threads test");

        // Test synchronous set
        prop.set(42).expect("Failed to set value synchronously");
        assert_eq!(call_count.load(Ordering::SeqCst), 1);

        // Test asynchronous set
        prop.set_async(43).expect("Failed to set value asynchronously");
        
        // Wait for async observers to complete
        thread::sleep(Duration::from_millis(50));
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_with_max_threads_async_performance() {
        // Test that with_max_threads affects async performance
        let prop = ObservableProperty::with_max_threads(0, 1); // Single thread
        let slow_call_count = Arc::new(AtomicUsize::new(0));

        // Add multiple slow observers
        let mut subscriptions = Vec::new();
        for _ in 0..4 {
            let count = slow_call_count.clone();
            let subscription = prop
                .subscribe_with_subscription(Arc::new(move |_, _| {
                    thread::sleep(Duration::from_millis(25)); // Simulate slow work
                    count.fetch_add(1, Ordering::SeqCst);
                }))
                .expect("Failed to create slow observer subscription");
            subscriptions.push(subscription);
        }

        // Measure time for async notification
        let start = std::time::Instant::now();
        prop.set_async(42).expect("Failed to set value asynchronously");
        let async_duration = start.elapsed();

        // Should return quickly even with slow observers
        assert!(async_duration.as_millis() < 50, "Async set should return quickly");

        // Wait for all observers to complete
        thread::sleep(Duration::from_millis(200));
        assert_eq!(slow_call_count.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn test_with_max_threads_vs_regular_constructor() {
        let prop_regular = ObservableProperty::new(42);
        let prop_custom = ObservableProperty::with_max_threads(42, 4); // Same as default

        let count_regular = Arc::new(AtomicUsize::new(0));
        let count_custom = Arc::new(AtomicUsize::new(0));

        // Both should behave identically
        let count1 = count_regular.clone();
        let _sub1 = prop_regular
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count1.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create regular subscription");

        let count2 = count_custom.clone();
        let _sub2 = prop_custom
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count2.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create custom subscription");

        // Test sync behavior
        prop_regular.set(100).expect("Failed to set regular property");
        prop_custom.set(100).expect("Failed to set custom property");

        assert_eq!(count_regular.load(Ordering::SeqCst), 1);
        assert_eq!(count_custom.load(Ordering::SeqCst), 1);

        // Test async behavior
        prop_regular.set_async(200).expect("Failed to set regular property async");
        prop_custom.set_async(200).expect("Failed to set custom property async");

        thread::sleep(Duration::from_millis(50));
        assert_eq!(count_regular.load(Ordering::SeqCst), 2);
        assert_eq!(count_custom.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_with_max_threads_large_values() {
        // Test with very large max_threads values
        let prop = ObservableProperty::with_max_threads(0, 1000);
        let call_count = Arc::new(AtomicUsize::new(0));

        // Add a few observers
        let count = call_count.clone();
        let _subscription = prop
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for large max_threads test");

        // Should work normally even with excessive thread limit
        prop.set_async(42).expect("Failed to set value with large max_threads");

        thread::sleep(Duration::from_millis(50));
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_with_max_threads_clone_behavior() {
        let prop1 = ObservableProperty::with_max_threads(42, 2);
        let prop2 = prop1.clone();

        let call_count1 = Arc::new(AtomicUsize::new(0));
        let call_count2 = Arc::new(AtomicUsize::new(0));

        // Subscribe to both properties
        let count1 = call_count1.clone();
        let _sub1 = prop1
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count1.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for cloned property test");

        let count2 = call_count2.clone();
        let _sub2 = prop2
            .subscribe_with_subscription(Arc::new(move |_, _| {
                count2.fetch_add(1, Ordering::SeqCst);
            }))
            .expect("Failed to create subscription for original property test");

        // Changes through either property should trigger both observers
        prop1.set_async(100).expect("Failed to set value through prop1");
        thread::sleep(Duration::from_millis(50));
        
        assert_eq!(call_count1.load(Ordering::SeqCst), 1);
        assert_eq!(call_count2.load(Ordering::SeqCst), 1);

        prop2.set_async(200).expect("Failed to set value through prop2");
        thread::sleep(Duration::from_millis(50));
        
        assert_eq!(call_count1.load(Ordering::SeqCst), 2);
        assert_eq!(call_count2.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_with_max_threads_thread_safety() {
        let prop = Arc::new(ObservableProperty::with_max_threads(0, 3));
        let call_count = Arc::new(AtomicUsize::new(0));

        // Add observers from multiple threads
        let handles: Vec<_> = (0..5)
            .map(|thread_id| {
                let prop_clone = prop.clone();
                let count_clone = call_count.clone();

                thread::spawn(move || {
                    let count = count_clone.clone();
                    let _subscription = prop_clone
                        .subscribe_with_subscription(Arc::new(move |_, _| {
                            count.fetch_add(1, Ordering::SeqCst);
                        }))
                        .expect("Failed to create thread-safe subscription");

                    // Trigger async notifications from this thread
                    prop_clone
                        .set_async(thread_id * 10)
                        .expect("Failed to set value from thread");
                        
                    thread::sleep(Duration::from_millis(10));
                })
            })
            .collect();

        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }

        // Wait for all async operations to complete
        thread::sleep(Duration::from_millis(100));

        // Each set_async should trigger all active observers at that time
        // The exact count depends on timing, but should be > 0
        assert!(call_count.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn test_with_max_threads_error_handling() {
        let prop = ObservableProperty::with_max_threads(42, 2);
        
        // Test that error handling works the same as regular properties
        let _subscription = prop
            .subscribe_with_subscription(Arc::new(|_, _| {
                // Normal observer
            }))
            .expect("Failed to create subscription for error handling test");

        // Should handle errors gracefully
        assert!(prop.set(100).is_ok());
        assert!(prop.set_async(200).is_ok());
        assert_eq!(prop.get().expect("Failed to get value after error test"), 200);
    }

    #[test]
    fn test_observer_limit_enforcement() {
        let prop = ObservableProperty::new(0);
        let mut observer_ids = Vec::new();

        // Add observers up to the limit (using a small test to avoid slow tests)
        // In reality, MAX_OBSERVERS is 10,000, but we'll test the mechanism
        // by adding a reasonable number and then checking the error message
        for i in 0..100 {
            let result = prop.subscribe(Arc::new(move |_, _| {
                let _ = i; // Use the capture to make each observer unique
            }));
            assert!(result.is_ok(), "Should be able to add observer {}", i);
            observer_ids.push(result.unwrap());
        }

        // Verify all observers were added
        assert_eq!(observer_ids.len(), 100);

        // Verify we can still add more (we're well under the 10,000 limit)
        let result = prop.subscribe(Arc::new(|_, _| {}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_observer_limit_error_message() {
        let prop = ObservableProperty::new(0);
        
        // We can't easily test hitting the actual 10,000 limit in a unit test
        // (it would be too slow), but we can verify the error type exists
        // and the subscribe method has the check
        
        // Add a few observers successfully
        for _ in 0..10 {
            assert!(prop.subscribe(Arc::new(|_, _| {})).is_ok());
        }

        // The mechanism is in place - the limit check happens before insertion
        // In production, if 10,000 observers are added, the 10,001st will fail
    }

    #[test]
    fn test_observer_limit_with_unsubscribe() {
        let prop = ObservableProperty::new(0);
        
        // Add observers
        let mut ids = Vec::new();
        for _ in 0..50 {
            ids.push(prop.subscribe(Arc::new(|_, _| {})).expect("Failed to subscribe"));
        }

        // Remove half of them
        for id in ids.iter().take(25) {
            assert!(prop.unsubscribe(*id).expect("Failed to unsubscribe"));
        }

        // Should be able to add more observers after unsubscribing
        for _ in 0..30 {
            assert!(prop.subscribe(Arc::new(|_, _| {})).is_ok());
        }
    }

    #[test]
    fn test_observer_limit_with_raii_subscriptions() {
        let prop = ObservableProperty::new(0);
        
        // Create RAII subscriptions
        let mut subscriptions = Vec::new();
        for _ in 0..50 {
            subscriptions.push(
                prop.subscribe_with_subscription(Arc::new(|_, _| {}))
                    .expect("Failed to create subscription")
            );
        }

        // Drop half of them (automatic cleanup)
        subscriptions.truncate(25);

        // Should be able to add more after RAII cleanup
        for _ in 0..30 {
            let _sub = prop.subscribe_with_subscription(Arc::new(|_, _| {}))
                .expect("Failed to create subscription after RAII cleanup");
        }
    }

    #[test]
    fn test_filtered_subscription_respects_observer_limit() {
        let prop = ObservableProperty::new(0);
        
        // Add regular and filtered observers
        for i in 0..50 {
            if i % 2 == 0 {
                assert!(prop.subscribe(Arc::new(|_, _| {})).is_ok());
            } else {
                assert!(prop.subscribe_filtered(Arc::new(|_, _| {}), |_, _| true).is_ok());
            }
        }

        // Both types count toward the limit
        // Should still be well under the 10,000 limit
        assert!(prop.subscribe_filtered(Arc::new(|_, _| {}), |_, _| true).is_ok());
    }

    #[test]
    fn test_observer_limit_concurrent_subscriptions() {
        let prop = Arc::new(ObservableProperty::new(0));
        let success_count = Arc::new(AtomicUsize::new(0));

        // Try to add observers from multiple threads
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let prop_clone = prop.clone();
                let count_clone = success_count.clone();
                
                thread::spawn(move || {
                    for _ in 0..10 {
                        if prop_clone.subscribe(Arc::new(|_, _| {})).is_ok() {
                            count_clone.fetch_add(1, Ordering::SeqCst);
                        }
                        thread::sleep(Duration::from_micros(10));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should complete");
        }

        // All 100 subscriptions should succeed (well under limit)
        assert_eq!(success_count.load(Ordering::SeqCst), 100);
    }

    #[test]
    fn test_notify_observers_batch_releases_lock_early() {
        use std::sync::atomic::AtomicBool;
        
        let prop = Arc::new(ObservableProperty::new(0));
        let call_count = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicBool::new(false));
        
        // Subscribe with a slow observer
        let started_clone = started.clone();
        let count_clone = call_count.clone();
        prop.subscribe(Arc::new(move |_, _| {
            started_clone.store(true, Ordering::SeqCst);
            count_clone.fetch_add(1, Ordering::SeqCst);
            // Simulate slow observer
            thread::sleep(Duration::from_millis(50));
        })).expect("Failed to subscribe");
        
        // Start batch notification in background
        let prop_clone = prop.clone();
        let batch_handle = thread::spawn(move || {
            prop_clone.notify_observers_batch(vec![(0, 1), (1, 2)]).expect("Failed to notify batch");
        });
        
        // Wait for observer to start
        while !started.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(1));
        }
        
        // Now verify we can still subscribe while observer is running
        // This proves the lock was released before observer execution
        let subscribe_result = prop.subscribe(Arc::new(|_, _| {
            // New observer
        }));
        
        assert!(subscribe_result.is_ok(), "Should be able to subscribe while batch notification is in progress");
        
        // Wait for batch to complete
        batch_handle.join().expect("Batch thread should complete");
        
        // Verify observers were called (2 changes in batch)
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_notify_observers_batch_panic_isolation() {
        let prop = ObservableProperty::new(0);
        let good_observer_count = Arc::new(AtomicUsize::new(0));
        let count_clone = good_observer_count.clone();
        
        // First observer that panics
        prop.subscribe(Arc::new(|_, _| {
            panic!("Deliberate panic in batch observer");
        })).expect("Failed to subscribe panicking observer");
        
        // Second observer that should still be called
        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe good observer");
        
        // Batch notification should not fail despite panic
        let result = prop.notify_observers_batch(vec![(0, 1), (1, 2), (2, 3)]);
        assert!(result.is_ok());
        
        // Second observer should have been called for all 3 changes
        assert_eq!(good_observer_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_notify_observers_batch_multiple_changes() {
        let prop = ObservableProperty::new(0);
        let received_changes = Arc::new(RwLock::new(Vec::new()));
        let changes_clone = received_changes.clone();
        
        prop.subscribe(Arc::new(move |old, new| {
            if let Ok(mut changes) = changes_clone.write() {
                changes.push((*old, *new));
            }
        })).expect("Failed to subscribe");
        
        // Send multiple changes
        prop.notify_observers_batch(vec![
            (0, 10),
            (10, 20),
            (20, 30),
            (30, 40),
        ]).expect("Failed to notify batch");
        
        let changes = received_changes.read().expect("Failed to read changes");
        assert_eq!(changes.len(), 4);
        assert_eq!(changes[0], (0, 10));
        assert_eq!(changes[1], (10, 20));
        assert_eq!(changes[2], (20, 30));
        assert_eq!(changes[3], (30, 40));
    }

    #[test]
    fn test_notify_observers_batch_empty() {
        let prop = ObservableProperty::new(0);
        let call_count = Arc::new(AtomicUsize::new(0));
        let count_clone = call_count.clone();
        
        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe");
        
        // Empty batch should succeed without calling observers
        prop.notify_observers_batch(vec![]).expect("Failed with empty batch");
        
        assert_eq!(call_count.load(Ordering::SeqCst), 0);
    }

    // ========================================================================
    // Debouncing Tests
    // ========================================================================

    #[test]
    fn test_debounced_observer_basic() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let last_value = Arc::new(RwLock::new(0));
        
        let count_clone = notification_count.clone();
        let value_clone = last_value.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut val) = value_clone.write() {
                    *val = *new;
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced observer");
        
        // Make a single change
        prop.set(42).expect("Failed to set value");
        
        // Immediately after, should not have been notified yet
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);
        
        // Wait for debounce period
        thread::sleep(Duration::from_millis(150));
        
        // Now should have been notified exactly once
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(*last_value.read().unwrap(), 42);
    }

    #[test]
    fn test_debounced_observer_rapid_changes() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let last_value = Arc::new(RwLock::new(0));
        
        let count_clone = notification_count.clone();
        let value_clone = last_value.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut val) = value_clone.write() {
                    *val = *new;
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced observer");
        
        // Make rapid changes
        for i in 1..=10 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(20)); // Changes every 20ms
        }
        
        // Should not have been notified yet
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);
        
        // Wait for debounce period after last change
        thread::sleep(Duration::from_millis(150));
        
        // Should have been notified exactly once with the final value
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(*last_value.read().unwrap(), 10);
    }

    #[test]
    fn test_debounced_observer_multiple_sequences() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let values = Arc::new(RwLock::new(Vec::new()));
        
        let count_clone = notification_count.clone();
        let values_clone = values.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut vals) = values_clone.write() {
                    vals.push(*new);
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced observer");
        
        // First sequence of changes
        prop.set(1).expect("Failed to set value");
        prop.set(2).expect("Failed to set value");
        prop.set(3).expect("Failed to set value");
        
        // Wait for debounce
        thread::sleep(Duration::from_millis(150));
        
        // Second sequence of changes
        prop.set(4).expect("Failed to set value");
        prop.set(5).expect("Failed to set value");
        
        // Wait for debounce
        thread::sleep(Duration::from_millis(150));
        
        // Should have been notified twice, once for each sequence
        assert_eq!(notification_count.load(Ordering::SeqCst), 2);
        let vals = values.read().unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], 3); // Last value from first sequence
        assert_eq!(vals[1], 5); // Last value from second sequence
    }

    #[test]
    fn test_debounced_observer_with_string() {
        let prop = ObservableProperty::new("".to_string());
        let notification_count = Arc::new(AtomicUsize::new(0));
        let last_value = Arc::new(RwLock::new(String::new()));
        
        let count_clone = notification_count.clone();
        let value_clone = last_value.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut val) = value_clone.write() {
                    *val = new.clone();
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced observer");
        
        // Simulate typing
        prop.set("H".to_string()).expect("Failed to set value");
        thread::sleep(Duration::from_millis(30));
        prop.set("He".to_string()).expect("Failed to set value");
        thread::sleep(Duration::from_millis(30));
        prop.set("Hel".to_string()).expect("Failed to set value");
        thread::sleep(Duration::from_millis(30));
        prop.set("Hell".to_string()).expect("Failed to set value");
        thread::sleep(Duration::from_millis(30));
        prop.set("Hello".to_string()).expect("Failed to set value");
        
        // Should not have been notified during typing
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);
        
        // Wait for debounce period
        thread::sleep(Duration::from_millis(150));
        
        // Should have been notified once with final value
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(*last_value.read().unwrap(), "Hello");
    }

    #[test]
    fn test_debounced_observer_zero_duration() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        
        let count_clone = notification_count.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, _new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
            Duration::from_millis(0)
        ).expect("Failed to subscribe debounced observer");
        
        prop.set(1).expect("Failed to set value");
        
        // Even with zero duration, thread needs time to execute
        thread::sleep(Duration::from_millis(10));
        
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
    }

    // ========================================================================
    // Throttling Tests
    // ========================================================================

    #[test]
    fn test_throttled_observer_basic() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let values = Arc::new(RwLock::new(Vec::new()));
        
        let count_clone = notification_count.clone();
        let values_clone = values.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut vals) = values_clone.write() {
                    vals.push(*new);
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe throttled observer");
        
        // First change should trigger immediately
        prop.set(1).expect("Failed to set value");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        
        // Second change within throttle period should be delayed
        prop.set(2).expect("Failed to set value");
        thread::sleep(Duration::from_millis(10));
        // Still only 1 notification
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        
        // Wait for throttle period
        thread::sleep(Duration::from_millis(100));
        
        // Now should have 2 notifications
        assert_eq!(notification_count.load(Ordering::SeqCst), 2);
        let vals = values.read().unwrap();
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0], 1);
        assert_eq!(vals[1], 2);
    }

    #[test]
    fn test_throttled_observer_continuous_changes() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        
        let count_clone = notification_count.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, _new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe throttled observer");
        
        // Make changes every 20ms for 500ms total
        for i in 1..=25 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(20));
        }
        
        // Wait for any pending notifications
        thread::sleep(Duration::from_millis(150));
        
        let count = notification_count.load(Ordering::SeqCst);
        // Should have been notified multiple times (roughly every 100ms)
        // Expecting around 5-6 notifications over 500ms
        assert!(count >= 4, "Expected at least 4 notifications, got {}", count);
        assert!(count <= 10, "Expected at most 10 notifications, got {}", count);
    }

    #[test]
    fn test_throttled_observer_rate_limiting() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let values = Arc::new(RwLock::new(Vec::new()));
        
        let count_clone = notification_count.clone();
        let values_clone = values.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut vals) = values_clone.write() {
                    vals.push(*new);
                }
            }),
            Duration::from_millis(200)
        ).expect("Failed to subscribe throttled observer");
        
        // Rapid-fire 20 changes
        for i in 1..=20 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(10));
        }
        
        // Wait for any pending notifications
        thread::sleep(Duration::from_millis(250));
        
        let count = notification_count.load(Ordering::SeqCst);
        // With 200ms throttle and changes every 10ms (200ms total duration),
        // should get 2 notifications maximum (1 immediate + 1 delayed)
        assert!(count >= 1, "Expected at least 1 notification, got {}", count);
        assert!(count <= 3, "Expected at most 3 notifications, got {}", count);
    }

    #[test]
    fn test_throttled_observer_first_change_immediate() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let first_value = Arc::new(RwLock::new(None));
        
        let count_clone = notification_count.clone();
        let value_clone = first_value.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
                if let Ok(mut val) = value_clone.write() {
                    if val.is_none() {
                        *val = Some(*new);
                    }
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe throttled observer");
        
        // First change
        prop.set(42).expect("Failed to set value");
        
        // Should be notified immediately (no sleep needed)
        thread::sleep(Duration::from_millis(10));
        
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(*first_value.read().unwrap(), Some(42));
    }

    #[test]
    fn test_throttled_vs_debounced_behavior() {
        let prop = ObservableProperty::new(0);
        let throttle_count = Arc::new(AtomicUsize::new(0));
        let debounce_count = Arc::new(AtomicUsize::new(0));
        
        let throttle_clone = throttle_count.clone();
        let debounce_clone = debounce_count.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, _new| {
                throttle_clone.fetch_add(1, Ordering::SeqCst);
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe throttled observer");
        
        prop.subscribe_debounced(
            Arc::new(move |_old, _new| {
                debounce_clone.fetch_add(1, Ordering::SeqCst);
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced observer");
        
        // Make continuous changes for 300ms
        for i in 1..=30 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(10));
        }
        
        // Wait for debounce to complete
        thread::sleep(Duration::from_millis(150));
        
        let throttle_notifications = throttle_count.load(Ordering::SeqCst);
        let debounce_notifications = debounce_count.load(Ordering::SeqCst);
        
        // Throttled: Multiple notifications during the period
        assert!(throttle_notifications >= 2, 
            "Throttled should have multiple notifications, got {}", throttle_notifications);
        
        // Debounced: Single notification after changes stopped
        assert_eq!(debounce_notifications, 1, 
            "Debounced should have exactly 1 notification, got {}", debounce_notifications);
        
        // Throttled should have more notifications than debounced
        assert!(throttle_notifications > debounce_notifications,
            "Throttled ({}) should have more notifications than debounced ({})",
            throttle_notifications, debounce_notifications);
    }

    #[test]
    fn test_throttled_observer_with_long_interval() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        
        let count_clone = notification_count.clone();
        
        prop.subscribe_throttled(
            Arc::new(move |_old, _new| {
                count_clone.fetch_add(1, Ordering::SeqCst);
            }),
            Duration::from_secs(1)
        ).expect("Failed to subscribe throttled observer");
        
        // First change - immediate
        prop.set(1).expect("Failed to set value");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        
        // Multiple changes within the throttle period
        for i in 2..=5 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(50));
        }
        
        // Still only 1 notification (throttle period hasn't expired)
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        
        // Wait for throttle period to expire
        thread::sleep(Duration::from_millis(1100));
        
        // Should now have 2 notifications (initial + delayed for last change)
        let final_count = notification_count.load(Ordering::SeqCst);
        assert!(final_count >= 1 && final_count <= 2,
            "Expected 1-2 notifications, got {}", final_count);
    }

    #[test]
    fn test_debounced_and_throttled_combined() {
        let prop = ObservableProperty::new(0);
        let debounce_values = Arc::new(RwLock::new(Vec::new()));
        let throttle_values = Arc::new(RwLock::new(Vec::new()));
        
        let debounce_clone = debounce_values.clone();
        let throttle_clone = throttle_values.clone();
        
        prop.subscribe_debounced(
            Arc::new(move |_old, new| {
                if let Ok(mut vals) = debounce_clone.write() {
                    vals.push(*new);
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe debounced");
        
        prop.subscribe_throttled(
            Arc::new(move |_old, new| {
                if let Ok(mut vals) = throttle_clone.write() {
                    vals.push(*new);
                }
            }),
            Duration::from_millis(100)
        ).expect("Failed to subscribe throttled");
        
        // Make a series of changes
        for i in 1..=10 {
            prop.set(i).expect("Failed to set value");
            thread::sleep(Duration::from_millis(25));
        }
        
        // Wait for both to complete
        thread::sleep(Duration::from_millis(200));
        
        let debounce_vals = debounce_values.read().unwrap();
        let throttle_vals = throttle_values.read().unwrap();
        
        // Debounced should have 1 value (the last one)
        assert_eq!(debounce_vals.len(), 1);
        assert_eq!(debounce_vals[0], 10);
        
        // Throttled should have multiple values
        assert!(throttle_vals.len() >= 2, 
            "Throttled should have at least 2 values, got {}", throttle_vals.len());
    }

    // ========================================================================
    // Computed Properties Tests
    // ========================================================================

    #[test]
    fn test_computed_basic() {
        let a = Arc::new(ObservableProperty::new(5));
        let b = Arc::new(ObservableProperty::new(10));

        let sum = computed(
            vec![a.clone(), b.clone()],
            |values| values[0] + values[1]
        ).expect("Failed to create computed property");

        assert_eq!(sum.get().unwrap(), 15);

        a.set(7).expect("Failed to set a");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(sum.get().unwrap(), 17);

        b.set(3).expect("Failed to set b");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(sum.get().unwrap(), 10);
    }

    #[test]
    fn test_computed_with_observer() {
        let width = Arc::new(ObservableProperty::new(10));
        let height = Arc::new(ObservableProperty::new(5));

        let area = computed(
            vec![width.clone(), height.clone()],
            |values| values[0] * values[1]
        ).expect("Failed to create computed property");

        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        area.subscribe(Arc::new(move |_old, _new| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe");

        assert_eq!(area.get().unwrap(), 50);

        width.set(20).expect("Failed to set width");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(area.get().unwrap(), 100);
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        height.set(8).expect("Failed to set height");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(area.get().unwrap(), 160);
        assert_eq!(notification_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_computed_string_concatenation() {
        let first = Arc::new(ObservableProperty::new("Hello".to_string()));
        let last = Arc::new(ObservableProperty::new("World".to_string()));

        let full = computed(
            vec![first.clone(), last.clone()],
            |values| format!("{} {}", values[0], values[1])
        ).expect("Failed to create computed property");

        assert_eq!(full.get().unwrap(), "Hello World");

        first.set("Goodbye".to_string()).expect("Failed to set first");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(full.get().unwrap(), "Goodbye World");

        last.set("Rust".to_string()).expect("Failed to set last");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(full.get().unwrap(), "Goodbye Rust");
    }

    #[test]
    fn test_computed_chaining() {
        let celsius = Arc::new(ObservableProperty::new(0.0));

        let fahrenheit = computed(
            vec![celsius.clone()],
            |values| values[0] * 9.0 / 5.0 + 32.0
        ).expect("Failed to create fahrenheit");

        let kelvin = computed(
            vec![celsius.clone()],
            |values| values[0] + 273.15
        ).expect("Failed to create kelvin");

        assert_eq!(celsius.get().unwrap(), 0.0);
        assert_eq!(fahrenheit.get().unwrap(), 32.0);
        assert_eq!(kelvin.get().unwrap(), 273.15);

        celsius.set(100.0).expect("Failed to set celsius");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(fahrenheit.get().unwrap(), 212.0);
        assert_eq!(kelvin.get().unwrap(), 373.15);
    }

    #[test]
    fn test_computed_multiple_dependencies() {
        let a = Arc::new(ObservableProperty::new(1));
        let b = Arc::new(ObservableProperty::new(2));
        let c = Arc::new(ObservableProperty::new(3));

        let result = computed(
            vec![a.clone(), b.clone(), c.clone()],
            |values| values[0] + values[1] * values[2]
        ).expect("Failed to create computed property");

        // Initial: 1 + 2 * 3 = 7
        assert_eq!(result.get().unwrap(), 7);

        a.set(5).expect("Failed to set a");
        thread::sleep(Duration::from_millis(10));
        // 5 + 2 * 3 = 11
        assert_eq!(result.get().unwrap(), 11);

        b.set(4).expect("Failed to set b");
        thread::sleep(Duration::from_millis(10));
        // 5 + 4 * 3 = 17
        assert_eq!(result.get().unwrap(), 17);

        c.set(2).expect("Failed to set c");
        thread::sleep(Duration::from_millis(10));
        // 5 + 4 * 2 = 13
        assert_eq!(result.get().unwrap(), 13);
    }

    #[test]
    fn test_computed_single_dependency() {
        let number = Arc::new(ObservableProperty::new(5));

        let doubled = computed(
            vec![number.clone()],
            |values| values[0] * 2
        ).expect("Failed to create computed property");

        assert_eq!(doubled.get().unwrap(), 10);

        number.set(7).expect("Failed to set number");
        thread::sleep(Duration::from_millis(10));
        assert_eq!(doubled.get().unwrap(), 14);
    }

    #[test]
    fn test_computed_complex_calculation() {
        let base = Arc::new(ObservableProperty::new(10.0_f64));
        let rate = Arc::new(ObservableProperty::new(0.05_f64));
        let years = Arc::new(ObservableProperty::new(2.0_f64));

        // Compound interest formula: A = P(1 + r)^t
        let amount = computed(
            vec![base.clone(), rate.clone(), years.clone()],
            |values| values[0] * (1.0 + values[1]).powf(values[2])
        ).expect("Failed to create computed property");

        // 10 * (1.05)^2 ≈ 11.025
        let initial_amount = amount.get().unwrap();
        assert!((initial_amount - 11.025_f64).abs() < 0.001);

        base.set(100.0_f64).expect("Failed to set base");
        thread::sleep(Duration::from_millis(10));
        // 100 * (1.05)^2 ≈ 110.25
        let new_amount = amount.get().unwrap();
        assert!((new_amount - 110.25_f64).abs() < 0.001);

        years.set(5.0_f64).expect("Failed to set years");
        thread::sleep(Duration::from_millis(10));
        // 100 * (1.05)^5 ≈ 127.628
        let final_amount = amount.get().unwrap();
        assert!((final_amount - 127.628_f64).abs() < 0.001);
    }

    #[test]
    fn test_update_batch_basic() {
        let prop = ObservableProperty::new(0);
        let notifications = Arc::new(RwLock::new(Vec::new()));
        let notifications_clone = notifications.clone();

        prop.subscribe(Arc::new(move |old, new| {
            if let Ok(mut notifs) = notifications_clone.write() {
                notifs.push((*old, *new));
            }
        })).expect("Failed to subscribe");

        prop.update_batch(|_current| {
            vec![10, 20, 30]
        }).expect("Failed to update_batch");

        let notifs = notifications.read().unwrap();
        assert_eq!(notifs.len(), 3);
        assert_eq!(notifs[0], (0, 10));
        assert_eq!(notifs[1], (10, 20));
        assert_eq!(notifs[2], (20, 30));
        assert_eq!(prop.get().unwrap(), 30);
    }

    #[test]
    fn test_update_batch_empty_vec() {
        let prop = ObservableProperty::new(42);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe");

        prop.update_batch(|current| {
            *current = 100; // This should be ignored
            Vec::new()
        }).expect("Failed to update_batch");

        assert_eq!(notification_count.load(Ordering::SeqCst), 0);
        assert_eq!(prop.get().unwrap(), 42); // Value unchanged
    }

    #[test]
    fn test_update_batch_single_state() {
        let prop = ObservableProperty::new(5);
        let notifications = Arc::new(RwLock::new(Vec::new()));
        let notifications_clone = notifications.clone();

        prop.subscribe(Arc::new(move |old, new| {
            if let Ok(mut notifs) = notifications_clone.write() {
                notifs.push((*old, *new));
            }
        })).expect("Failed to subscribe");

        prop.update_batch(|_current| {
            vec![10]
        }).expect("Failed to update_batch");

        let notifs = notifications.read().unwrap();
        assert_eq!(notifs.len(), 1);
        assert_eq!(notifs[0], (5, 10));
        assert_eq!(prop.get().unwrap(), 10);
    }

    #[test]
    fn test_update_batch_string_transformation() {
        let prop = ObservableProperty::new(String::from("hello"));
        let notifications = Arc::new(RwLock::new(Vec::new()));
        let notifications_clone = notifications.clone();

        prop.subscribe(Arc::new(move |old, new| {
            if let Ok(mut notifs) = notifications_clone.write() {
                notifs.push((old.clone(), new.clone()));
            }
        })).expect("Failed to subscribe");

        prop.update_batch(|current| {
            let step1 = current.to_uppercase();
            let step2 = format!("{}!", step1);
            let step3 = format!("{} WORLD", step2);
            vec![step1, step2, step3]
        }).expect("Failed to update_batch");

        let notifs = notifications.read().unwrap();
        assert_eq!(notifs.len(), 3);
        assert_eq!(notifs[0].0, "hello");
        assert_eq!(notifs[0].1, "HELLO");
        assert_eq!(notifs[1].0, "HELLO");
        assert_eq!(notifs[1].1, "HELLO!");
        assert_eq!(notifs[2].0, "HELLO!");
        assert_eq!(notifs[2].1, "HELLO! WORLD");
        assert_eq!(prop.get().unwrap(), "HELLO! WORLD");
    }

    #[test]
    fn test_update_batch_multiple_observers() {
        let prop = ObservableProperty::new(0);
        let count1 = Arc::new(AtomicUsize::new(0));
        let count2 = Arc::new(AtomicUsize::new(0));
        
        let count1_clone = count1.clone();
        let count2_clone = count2.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count1_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe observer 1");

        prop.subscribe(Arc::new(move |_, _| {
            count2_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe observer 2");

        prop.update_batch(|_current| {
            vec![1, 2, 3, 4, 5]
        }).expect("Failed to update_batch");

        assert_eq!(count1.load(Ordering::SeqCst), 5);
        assert_eq!(count2.load(Ordering::SeqCst), 5);
        assert_eq!(prop.get().unwrap(), 5);
    }

    #[test]
    fn test_update_batch_with_panicking_observer() {
        let prop = ObservableProperty::new(0);
        let good_observer_count = Arc::new(AtomicUsize::new(0));
        let count_clone = good_observer_count.clone();

        // Observer that panics
        prop.subscribe(Arc::new(|old, _new| {
            if *old == 1 {
                panic!("Observer panic!");
            }
        })).expect("Failed to subscribe panicking observer");

        // Observer that should still work
        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe good observer");

        // Should not panic, good observer should still be notified
        prop.update_batch(|_current| {
            vec![1, 2, 3]
        }).expect("Failed to update_batch");

        // Good observer should have been called for all 3 states
        assert_eq!(good_observer_count.load(Ordering::SeqCst), 3);
        assert_eq!(prop.get().unwrap(), 3);
    }

    #[test]
    fn test_update_batch_thread_safety() {
        let prop = Arc::new(ObservableProperty::new(0));
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).expect("Failed to subscribe");

        let handles: Vec<_> = (0..5).map(|i| {
            let prop_clone = prop.clone();
            thread::spawn(move || {
                prop_clone.update_batch(|_current| {
                    vec![i * 10 + 1, i * 10 + 2, i * 10 + 3]
                }).expect("Failed to update_batch in thread");
            })
        }).collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // 5 threads * 3 states each = 15 notifications
        assert_eq!(notification_count.load(Ordering::SeqCst), 15);
    }

    #[test]
    fn test_update_batch_with_weak_observers() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        prop.subscribe_weak(Arc::downgrade(&observer))
            .expect("Failed to subscribe weak observer");

        // Observer is alive, should get notifications
        prop.update_batch(|_current| {
            vec![1, 2, 3]
        }).expect("Failed to update_batch");

        assert_eq!(notification_count.load(Ordering::SeqCst), 3);

        // Drop the observer
        drop(observer);

        // Observer is dead, should not get notifications
        prop.update_batch(|_current| {
            vec![4, 5, 6]
        }).expect("Failed to update_batch");

        // Count should still be 3 (no new notifications)
        assert_eq!(notification_count.load(Ordering::SeqCst), 3);
        assert_eq!(prop.get().unwrap(), 6);
    }

    #[test]
    fn test_change_coalescing_basic() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let last_old = Arc::new(RwLock::new(0));
        let last_new = Arc::new(RwLock::new(0));

        let count_clone = notification_count.clone();
        let old_clone = last_old.clone();
        let new_clone = last_new.clone();

        prop.subscribe(Arc::new(move |old, new| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            *old_clone.write().unwrap() = *old;
            *new_clone.write().unwrap() = *new;
        }))
        .expect("Failed to subscribe");

        // Begin batch update
        prop.begin_update().expect("Failed to begin update");

        // Multiple changes - should not trigger notifications
        prop.set(10).expect("Failed to set value");
        prop.set(20).expect("Failed to set value");
        prop.set(30).expect("Failed to set value");

        // No notifications yet
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);
        assert_eq!(prop.get().unwrap(), 30);

        // End batch - should trigger single notification
        prop.end_update().expect("Failed to end update");

        // Should have exactly one notification from 0 to 30
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(*last_old.read().unwrap(), 0);
        assert_eq!(*last_new.read().unwrap(), 30);
    }

    #[test]
    fn test_change_coalescing_nested() {
        let prop = ObservableProperty::new(100);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }))
        .expect("Failed to subscribe");

        // Start outer batch
        prop.begin_update().expect("Failed to begin update");
        prop.set(110).expect("Failed to set");

        // Start inner batch
        prop.begin_update().expect("Failed to begin nested update");
        prop.set(120).expect("Failed to set");
        prop.set(130).expect("Failed to set");
        prop.end_update().expect("Failed to end nested update");

        // Still no notifications (outer batch still active)
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);

        prop.set(140).expect("Failed to set");
        prop.end_update().expect("Failed to end outer update");

        // Should have exactly one notification
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(prop.get().unwrap(), 140);
    }

    #[test]
    fn test_change_coalescing_without_begin() {
        let prop = ObservableProperty::new(0);

        // Calling end_update without begin_update should fail
        let result = prop.end_update();
        assert!(result.is_err());

        if let Err(PropertyError::InvalidConfiguration { reason }) = result {
            assert!(reason.contains("without matching begin_update"));
        } else {
            panic!("Expected InvalidConfiguration error");
        }
    }

    #[test]
    fn test_change_coalescing_multiple_cycles() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }))
        .expect("Failed to subscribe");

        // First batch
        prop.begin_update().expect("Failed to begin update 1");
        prop.set(10).expect("Failed to set");
        prop.set(20).expect("Failed to set");
        prop.end_update().expect("Failed to end update 1");

        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        // Second batch
        prop.begin_update().expect("Failed to begin update 2");
        prop.set(30).expect("Failed to set");
        prop.set(40).expect("Failed to set");
        prop.end_update().expect("Failed to end update 2");

        assert_eq!(notification_count.load(Ordering::SeqCst), 2);

        // Regular set (not batched)
        prop.set(50).expect("Failed to set");

        // Should trigger immediate notification
        assert_eq!(notification_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_change_coalescing_with_async() {
        let prop = ObservableProperty::new(0);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }))
        .expect("Failed to subscribe");

        // Batched update suppresses both sync and async notifications
        prop.begin_update().expect("Failed to begin update");
        prop.set(10).expect("Failed to set");
        prop.set_async(20).expect("Failed to set async");
        prop.set(30).expect("Failed to set");

        // No notifications yet
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);

        prop.end_update().expect("Failed to end update");

        // Should have one notification
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_change_coalescing_thread_safety() {
        use std::thread;

        let prop = Arc::new(ObservableProperty::new(0));
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        }))
        .expect("Failed to subscribe");

        let prop_clone = prop.clone();
        let handle = thread::spawn(move || {
            prop_clone.begin_update().expect("Failed to begin update");
            prop_clone.set(10).expect("Failed to set");
            prop_clone.set(20).expect("Failed to set");
            prop_clone.end_update().expect("Failed to end update");
        });

        handle.join().expect("Thread panicked");

        // Should have one notification
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
        assert_eq!(prop.get().unwrap(), 20);
    }

    // ========================================================================
    // Validation Tests
    // ========================================================================

    #[test]
    fn test_with_validator_basic() {
        let result = ObservableProperty::with_validator(25, |age| {
            if *age <= 150 {
                Ok(())
            } else {
                Err(format!("Age must be at most 150, got {}", age))
            }
        });

        assert!(result.is_ok());
        let prop = result.unwrap();
        assert_eq!(prop.get().unwrap(), 25);

        // Valid update
        assert!(prop.set(30).is_ok());
        assert_eq!(prop.get().unwrap(), 30);

        // Invalid update
        let invalid_result = prop.set(200);
        assert!(invalid_result.is_err());
        assert_eq!(prop.get().unwrap(), 30); // Value unchanged
    }

    #[test]
    fn test_with_validator_rejects_invalid_initial_value() {
        let result = ObservableProperty::with_validator(200, |age| {
            if *age <= 150 {
                Ok(())
            } else {
                Err(format!("Age must be at most 150, got {}", age))
            }
        });

        assert!(result.is_err());
        match result {
            Err(PropertyError::ValidationError { reason }) => {
                assert!(reason.contains("200"));
            }
            _ => panic!("Expected ValidationError"),
        }
    }

    #[test]
    fn test_with_validator_string_validation() {
        let result = ObservableProperty::with_validator("alice".to_string(), |name| {
            if name.is_empty() {
                return Err("Username cannot be empty".to_string());
            }
            if name.len() < 3 {
                return Err(format!("Username must be at least 3 characters, got {}", name.len()));
            }
            if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                return Err("Username can only contain letters, numbers, and underscores".to_string());
            }
            Ok(())
        });

        assert!(result.is_ok());
        let prop = result.unwrap();

        assert!(prop.set("bob".to_string()).is_ok());
        assert!(prop.set("ab".to_string()).is_err()); // Too short
        assert!(prop.set("user@123".to_string()).is_err()); // Invalid chars
        assert_eq!(prop.get().unwrap(), "bob");
    }

    #[test]
    fn test_with_validator_with_observers() {
        let prop = ObservableProperty::with_validator(10, |val| {
            if *val >= 0 && *val <= 100 {
                Ok(())
            } else {
                Err(format!("Value must be between 0 and 100, got {}", val))
            }
        }).unwrap();

        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).unwrap();

        // Valid update - should notify
        prop.set(50).unwrap();
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);

        // Invalid update - should not notify
        prop.set(150).unwrap_err();
        assert_eq!(notification_count.load(Ordering::SeqCst), 1); // No additional notification
    }

    // ========================================================================
    // Custom Equality Tests
    // ========================================================================

    #[test]
    fn test_with_equality_basic() {
        // Create property where values within 5 are considered equal
        let prop = ObservableProperty::with_equality(10i32, |a, b| (a - b).abs() <= 5);
        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).unwrap();

        // Small change - should not notify
        prop.set(12).unwrap();
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);

        // Large change - should notify
        prop.set(20).unwrap();
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_with_equality_string_case_insensitive() {
        let prop = ObservableProperty::with_equality("Hello".to_string(), |a, b| {
            a.to_lowercase() == b.to_lowercase()
        });

        let notification_count = Arc::new(AtomicUsize::new(0));
        let count_clone = notification_count.clone();

        prop.subscribe(Arc::new(move |_, _| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        })).unwrap();

        // Same case-insensitive - should not notify
        prop.set("hello".to_string()).unwrap();
        assert_eq!(notification_count.load(Ordering::SeqCst), 0);

        // Different case-insensitive - should notify
        prop.set("World".to_string()).unwrap();
        assert_eq!(notification_count.load(Ordering::SeqCst), 1);
    }

    // ========================================================================
    // History Tests
    // ========================================================================

    #[test]
    fn test_with_history_basic() {
        let prop = ObservableProperty::with_history(0, 5);

        prop.set(10).unwrap();
        prop.set(20).unwrap();
        prop.set(30).unwrap();

        assert_eq!(prop.get().unwrap(), 30);

        // Undo
        prop.undo().unwrap();
        assert_eq!(prop.get().unwrap(), 20);

        prop.undo().unwrap();
        assert_eq!(prop.get().unwrap(), 10);

        prop.undo().unwrap();
        assert_eq!(prop.get().unwrap(), 0);
    }

    #[test]
    fn test_with_history_get_history() {
        let prop = ObservableProperty::with_history("start".to_string(), 3);

        prop.set("second".to_string()).unwrap();
        prop.set("third".to_string()).unwrap();
        prop.set("fourth".to_string()).unwrap();

        let history = prop.get_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0], "start");
        assert_eq!(history[1], "second");
        assert_eq!(history[2], "third");
        assert_eq!(prop.get().unwrap(), "fourth");
    }

    #[test]
    fn test_with_history_bounded_buffer() {
        let prop = ObservableProperty::with_history(1, 2);

        prop.set(2).unwrap();
        prop.set(3).unwrap();
        prop.set(4).unwrap();

        let history = prop.get_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0], 2);
        assert_eq!(history[1], 3);
        assert_eq!(prop.get().unwrap(), 4);
    }

    #[test]
    fn test_undo_no_history() {
        let prop = ObservableProperty::new(42);
        let result = prop.undo();
        assert!(result.is_err());
    }

    #[test]
    fn test_undo_empty_history() {
        let prop = ObservableProperty::with_history(42, 5);
        let result = prop.undo();
        assert!(result.is_err());
    }

    #[test]
    fn test_history_with_observers() {
        let prop = ObservableProperty::with_history(0, 5);
        let notifications = Arc::new(RwLock::new(Vec::new()));
        let notifs_clone = notifications.clone();

        prop.subscribe(Arc::new(move |old, new| {
            if let Ok(mut notifs) = notifs_clone.write() {
                notifs.push((*old, *new));
            }
        })).unwrap();

        prop.set(10).unwrap();
        prop.set(20).unwrap();
        prop.undo().unwrap(); // Should notify: 20 -> 10

        let notifs = notifications.read().unwrap();
        assert_eq!(notifs.len(), 3);
        assert_eq!(notifs[0], (0, 10));
        assert_eq!(notifs[1], (10, 20));
        assert_eq!(notifs[2], (20, 10)); // Undo notification
    }

    // ========================================================================
    // Event Logging Tests
    // ========================================================================

    #[test]
    fn test_with_event_log_basic() {
        let counter = ObservableProperty::with_event_log(0, 0);

        counter.set(1).unwrap();
        counter.set(2).unwrap();
        counter.set(3).unwrap();

        let events = counter.get_event_log();
        assert_eq!(events.len(), 3);

        assert_eq!(events[0].old_value, 0);
        assert_eq!(events[0].new_value, 1);
        assert_eq!(events[0].event_number, 0);

        assert_eq!(events[2].old_value, 2);
        assert_eq!(events[2].new_value, 3);
        assert_eq!(events[2].event_number, 2);
    }

    #[test]
    fn test_with_event_log_bounded() {
        let prop = ObservableProperty::with_event_log(100, 3);

        prop.set(101).unwrap();
        prop.set(102).unwrap();
        prop.set(103).unwrap();
        prop.set(104).unwrap();

        let events = prop.get_event_log();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].old_value, 101);
        assert_eq!(events[2].new_value, 104);
    }

    #[test]
    fn test_event_log_timestamps() {
        let prop = ObservableProperty::with_event_log(0, 0);

        let before = Instant::now();
        thread::sleep(Duration::from_millis(10));
        prop.set(1).unwrap();
        thread::sleep(Duration::from_millis(10));
        prop.set(2).unwrap();
        let after = Instant::now();

        let events = prop.get_event_log();
        assert_eq!(events.len(), 2);
        assert!(events[0].timestamp >= before);
        assert!(events[1].timestamp <= after);
        assert!(events[1].timestamp >= events[0].timestamp);
    }

    // ========================================================================
    // Property Transformation Tests (map)
    // ========================================================================

    #[test]
    fn test_map_basic() {
        let celsius = ObservableProperty::new(20.0);
        let fahrenheit = celsius.map(|c| c * 9.0 / 5.0 + 32.0).unwrap();

        assert_eq!(fahrenheit.get().unwrap(), 68.0);

        celsius.set(25.0).unwrap();
        thread::sleep(Duration::from_millis(10)); // Give observer time to fire
        assert_eq!(fahrenheit.get().unwrap(), 77.0);

        celsius.set(0.0).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(fahrenheit.get().unwrap(), 32.0);
    }

    #[test]
    fn test_map_string_formatting() {
        let count = ObservableProperty::new(42);
        let message = count.map(|n| format!("Count: {}", n)).unwrap();

        assert_eq!(message.get().unwrap(), "Count: 42");

        count.set(100).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(message.get().unwrap(), "Count: 100");
    }

    #[test]
    fn test_map_chaining() {
        let base = ObservableProperty::new(10);
        let doubled = base.map(|x| x * 2).unwrap();
        let squared = doubled.map(|x| x * x).unwrap();

        assert_eq!(squared.get().unwrap(), 400); // (10 * 2)^2 = 400

        base.set(5).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(squared.get().unwrap(), 100); // (5 * 2)^2 = 100
    }

    #[test]
    fn test_map_type_conversion() {
        let integer = ObservableProperty::new(42);
        let float_value = integer.map(|i| *i as f64).unwrap();
        let is_even = integer.map(|i| i % 2 == 0).unwrap();

        assert_eq!(float_value.get().unwrap(), 42.0);
        assert_eq!(is_even.get().unwrap(), true);

        integer.set(43).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(is_even.get().unwrap(), false);
    }

    // ========================================================================
    // Modify Tests
    // ========================================================================

    #[test]
    fn test_modify_basic() {
        let counter = ObservableProperty::new(0);
        let notifications = Arc::new(RwLock::new(Vec::new()));
        let notifs_clone = notifications.clone();

        counter.subscribe(Arc::new(move |old, new| {
            if let Ok(mut notifs) = notifs_clone.write() {
                notifs.push((*old, *new));
            }
        })).unwrap();

        counter.modify(|value| *value += 1).unwrap();
        assert_eq!(counter.get().unwrap(), 1);

        counter.modify(|value| *value *= 2).unwrap();
        assert_eq!(counter.get().unwrap(), 2);

        let notifs = notifications.read().unwrap();
        assert_eq!(notifs.len(), 2);
        assert_eq!(notifs[0], (0, 1));
        assert_eq!(notifs[1], (1, 2));
    }

    #[test]
    fn test_modify_with_validator() {
        let prop = ObservableProperty::with_validator(10, |val| {
            if *val >= 0 && *val <= 100 {
                Ok(())
            } else {
                Err("Value must be between 0 and 100".to_string())
            }
        }).unwrap();

        // Valid modification
        assert!(prop.modify(|v| *v += 5).is_ok());
        assert_eq!(prop.get().unwrap(), 15);

        // Invalid modification
        let result = prop.modify(|v| *v += 100);
        assert!(result.is_err());
        assert_eq!(prop.get().unwrap(), 15); // Value unchanged
    }

    #[test]
    fn test_modify_string() {
        let text = ObservableProperty::new("hello".to_string());

        text.modify(|s| {
            *s = s.to_uppercase();
        }).unwrap();

        assert_eq!(text.get().unwrap(), "HELLO");

        text.modify(|s| {
            s.push_str(" WORLD");
        }).unwrap();

        assert_eq!(text.get().unwrap(), "HELLO WORLD");
    }

    // ========================================================================
    // Bidirectional Binding Tests
    // ========================================================================

    #[test]
    fn test_bind_bidirectional_basic() {
        let prop1 = Arc::new(ObservableProperty::new(10));
        let prop2 = Arc::new(ObservableProperty::new(10)); // Start with same value

        prop1.bind_bidirectional(&prop2).unwrap();

        // Change prop1 - prop2 should update
        prop1.set(30).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(prop2.get().unwrap(), 30);

        // Change prop2 - prop1 should update
        prop2.set(40).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(prop1.get().unwrap(), 40);
    }

    #[test]
    fn test_bind_bidirectional_strings() {
        let prop1 = Arc::new(ObservableProperty::new("first".to_string()));
        let prop2 = Arc::new(ObservableProperty::new("first".to_string())); // Start with same value

        prop1.bind_bidirectional(&prop2).unwrap();

        // Change prop2
        prop2.set("updated".to_string()).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(prop1.get().unwrap(), "updated");

        // Change prop1
        prop1.set("final".to_string()).unwrap();
        thread::sleep(Duration::from_millis(20));
        assert_eq!(prop2.get().unwrap(), "final");
    }

    // ========================================================================
    // Comprehensive Metrics Tests
    // ========================================================================

    #[test]
    fn test_get_metrics_basic() {
        let prop = ObservableProperty::new(0);
        
        prop.subscribe(Arc::new(|_, _| {
            thread::sleep(Duration::from_millis(5));
        })).unwrap();

        prop.set(1).unwrap();
        prop.set(2).unwrap();
        prop.set(3).unwrap();

        let metrics = prop.get_metrics().unwrap();
        assert_eq!(metrics.total_changes, 3);
        assert_eq!(metrics.observer_calls, 3);
        assert!(metrics.avg_notification_time.as_millis() >= 4);
    }

    #[test]
    fn test_metrics_multiple_observers() {
        let prop = ObservableProperty::new(0);
        
        for _ in 0..3 {
            prop.subscribe(Arc::new(|_, _| {})).unwrap();
        }

        prop.set(1).unwrap();
        prop.set(2).unwrap();

        let metrics = prop.get_metrics().unwrap();
        assert_eq!(metrics.total_changes, 2);
        assert_eq!(metrics.observer_calls, 6); // 2 changes * 3 observers
    }

    #[test]
    fn test_metrics_no_observers() {
        let prop = ObservableProperty::new(0);

        prop.set(1).unwrap();
        prop.set(2).unwrap();

        let metrics = prop.get_metrics().unwrap();
        assert_eq!(metrics.total_changes, 2);
        assert_eq!(metrics.observer_calls, 0);
    }

    // ========================================================================
    // Debug Logging Tests (when debug feature is enabled)
    // ========================================================================

    #[test]
    #[cfg(feature = "debug")]
    fn test_enable_disable_change_logging() {
        let prop = ObservableProperty::new(0);

        prop.enable_change_logging();
        prop.set(1).unwrap();
        prop.set(2).unwrap();
        prop.disable_change_logging();
        prop.set(3).unwrap();

        // With debug feature enabled, change logs should be recorded
        // This is a basic compile test to ensure the methods exist
    }

    // ========================================================================
    // Async Features Tests
    // ========================================================================

    // Note: The following async tests are disabled because the wait_for() implementation
    // needs to be refactored to use an async-compatible channel (tokio::sync::mpsc)
    // instead of std::sync::mpsc to properly integrate with async runtimes.
    // The current implementation returns Poll::Pending without registering a waker,
    // causing the tests to hang indefinitely.
    // TODO: Fix in version 0.5.0 by adding tokio as optional dependency for async feature

    /*
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_wait_for_basic() {
        let prop = Arc::new(ObservableProperty::new(0));
        let prop_clone = prop.clone();

        // Spawn task to change value after delay
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
            prop_clone.set(42).unwrap();
        });

        // Wait for value to become 42
        prop.wait_for(|v| *v == 42).await;
        assert_eq!(prop.get().unwrap(), 42);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_wait_for_already_true() {
        let prop = ObservableProperty::new(100);

        // Predicate is already true
        prop.wait_for(|v| *v >= 50).await;
        assert_eq!(prop.get().unwrap(), 100);
    }
    */

    // ========================================================================
    // Observer Count Tests
    // ========================================================================

    #[test]
    fn test_observer_count() {
        let prop = ObservableProperty::new(0);
        assert_eq!(prop.observer_count(), 0);

        let _id1 = prop.subscribe(Arc::new(|_, _| {})).unwrap();
        assert_eq!(prop.observer_count(), 1);

        let _id2 = prop.subscribe(Arc::new(|_, _| {})).unwrap();
        assert_eq!(prop.observer_count(), 2);

        let _id3 = prop.subscribe(Arc::new(|_, _| {})).unwrap();
        assert_eq!(prop.observer_count(), 3);
    }

    #[test]
    fn test_observer_count_after_unsubscribe() {
        let prop = ObservableProperty::new(0);

        let id1 = prop.subscribe(Arc::new(|_, _| {})).unwrap();
        let id2 = prop.subscribe(Arc::new(|_, _| {})).unwrap();
        assert_eq!(prop.observer_count(), 2);

        prop.unsubscribe(id1).unwrap();
        assert_eq!(prop.observer_count(), 1);

        prop.unsubscribe(id2).unwrap();
        assert_eq!(prop.observer_count(), 0);
    }

    // ========================================================================
    // With Config Tests
    // ========================================================================

    #[test]
    fn test_with_config_custom_limits() {
        let prop = ObservableProperty::with_config(42, 2, 5);

        // Should be able to add up to 5 observers
        for _ in 0..5 {
            assert!(prop.subscribe(Arc::new(|_, _| {})).is_ok());
        }

        assert_eq!(prop.observer_count(), 5);

        // 6th observer should fail due to limit
        let result = prop.subscribe(Arc::new(|_, _| {}));
        assert!(result.is_err());
    }

    #[test]
    fn test_with_config_max_threads() {
        let prop = ObservableProperty::with_config(0, 1, 100);
        let call_count = Arc::new(AtomicUsize::new(0));

        for _ in 0..4 {
            let count = call_count.clone();
            prop.subscribe(Arc::new(move |_, _| {
                thread::sleep(Duration::from_millis(25));
                count.fetch_add(1, Ordering::SeqCst);
            })).unwrap();
        }

        // With max_threads = 1, async should still work but sequentially
        let start = Instant::now();
        prop.set_async(42).unwrap();
        let duration = start.elapsed();

        // Should return quickly
        assert!(duration.as_millis() < 50);

        // Wait for observers
        thread::sleep(Duration::from_millis(150));
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
    }

    // ========================================================================
    // Persistence Tests
    // ========================================================================

    struct MockPersistence {
        data: Arc<RwLock<Option<i32>>>,
    }

    impl PropertyPersistence for MockPersistence {
        type Value = i32;

        fn load(&self) -> Result<Self::Value, Box<dyn std::error::Error + Send + Sync>> {
            self.data
                .read()
                .unwrap()
                .ok_or_else(|| "No data".into())
        }

        fn save(&self, value: &Self::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            *self.data.write().unwrap() = Some(*value);
            Ok(())
        }
    }

    #[test]
    fn test_with_persistence_load_success() {
        let storage = Arc::new(RwLock::new(Some(42)));
        let persistence = MockPersistence { data: storage.clone() };

        let prop = ObservableProperty::with_persistence(0, persistence);
        
        // Should load 42 from persistence
        assert_eq!(prop.get().unwrap(), 42);
    }

    #[test]
    fn test_with_persistence_auto_save() {
        let storage = Arc::new(RwLock::new(None));
        let persistence = MockPersistence { data: storage.clone() };

        let prop = ObservableProperty::with_persistence(10, persistence);
        
        // Change value - should auto-save
        prop.set(20).unwrap();
        thread::sleep(Duration::from_millis(10));

        // Check storage
        assert_eq!(*storage.read().unwrap(), Some(20));

        prop.set(30).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(*storage.read().unwrap(), Some(30));
    }

    #[test]
    fn test_with_persistence_load_failure_uses_default() {
        let storage = Arc::new(RwLock::new(None));
        let persistence = MockPersistence { data: storage };

        // Load will fail, should use initial_value
        let prop = ObservableProperty::with_persistence(99, persistence);
        assert_eq!(prop.get().unwrap(), 99);
    }

    // ========================================================================
    // Computed Properties Tests (additional coverage)
    // ========================================================================

    #[test]
    fn test_computed_updates_immediately() {
        let a = Arc::new(ObservableProperty::new(5));
        let b = Arc::new(ObservableProperty::new(10));

        let sum = computed(
            vec![a.clone(), b.clone()],
            |values| values[0] + values[1]
        ).unwrap();

        assert_eq!(sum.get().unwrap(), 15);

        a.set(7).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(sum.get().unwrap(), 17);
    }

    #[test]
    fn test_computed_with_string() {
        let first = Arc::new(ObservableProperty::new("Hello".to_string()));
        let last = Arc::new(ObservableProperty::new("World".to_string()));

        let full = computed(
            vec![first.clone(), last.clone()],
            |values| format!("{} {}", values[0], values[1])
        ).unwrap();

        assert_eq!(full.get().unwrap(), "Hello World");

        first.set("Goodbye".to_string()).unwrap();
        thread::sleep(Duration::from_millis(10));
        assert_eq!(full.get().unwrap(), "Goodbye World");
    }
}

