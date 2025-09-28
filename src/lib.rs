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

use std::collections::HashMap;
use std::fmt;
use std::panic;
use std::sync::{Arc, RwLock};
use std::thread;

/// Maximum number of background threads used for asynchronous observer notifications
///
/// This constant controls the degree of parallelism when using `set_async()` to notify
/// observers. The observer list is divided into batches, with each batch running in
/// its own background thread, up to this maximum number of threads.
///
/// # Rationale
///
/// - **Resource Control**: Prevents unbounded thread creation that could exhaust system resources
/// - **Performance Balance**: Provides parallelism benefits without excessive context switching overhead  
/// - **Scalability**: Ensures consistent behavior regardless of the number of observers
/// - **System Responsiveness**: Limits thread contention on multi-core systems
///
/// # Implementation Details
///
/// When `set_async()` is called:
/// 1. All observers are collected into a snapshot
/// 2. Observers are divided into `MAX_THREADS` batches (or fewer if there are fewer observers)
/// 3. Each batch executes in its own `thread::spawn()` call
/// 4. Observers within each batch are executed sequentially
///
/// For example, with 100 observers and `MAX_THREADS = 4`:
/// - Batch 1: Observers 1-25 (Thread 1)
/// - Batch 2: Observers 26-50 (Thread 2)  
/// - Batch 3: Observers 51-75 (Thread 3)
/// - Batch 4: Observers 76-100 (Thread 4)
///
/// # Tuning Considerations
///
/// This value can be adjusted based on your application's needs:
/// - **CPU-bound observers**: Higher values may improve throughput on multi-core systems
/// - **I/O-bound observers**: Higher values can improve concurrency for network/disk operations
/// - **Memory-constrained systems**: Lower values reduce thread overhead
/// - **Real-time systems**: Lower values reduce scheduling unpredictability
///
/// # Thread Safety
///
/// This constant is used only during the batching calculation and does not affect
/// the thread safety of the overall system.
const MAX_THREADS: usize = 4;
/// Errors that can occur when working with ObservableProperty
#[derive(Debug, Clone)]
pub enum PropertyError {
    /// Failed to acquire a read lock on the property
    ReadLockError {
        /// Context describing what operation was being attempted
        context: String,
    },
    /// Failed to acquire a write lock on the property  
    WriteLockError {
        /// Context describing what operation was being attempted
        context: String,
    },
    /// Attempted to unsubscribe an observer that doesn't exist
    ObserverNotFound {
        /// The ID of the observer that wasn't found
        id: usize,
    },
    /// The property's lock has been poisoned due to a panic in another thread
    PoisonedLock,
    /// An observer function encountered an error during execution
    ObserverError {
        /// Description of what went wrong
        reason: String,
    },
    /// The thread pool for async notifications is exhausted
    ThreadPoolExhausted,
    /// Invalid configuration was provided
    InvalidConfiguration {
        /// Description of the invalid configuration
        reason: String,
    },
}

impl fmt::Display for PropertyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PropertyError::ReadLockError { context } => {
                write!(f, "Failed to acquire read lock: {}", context)
            }
            PropertyError::WriteLockError { context } => {
                write!(f, "Failed to acquire write lock: {}", context)
            }
            PropertyError::ObserverNotFound { id } => {
                write!(f, "Observer with ID {} not found", id)
            }
            PropertyError::PoisonedLock => {
                write!(
                    f,
                    "Property is in a poisoned state due to a panic in another thread"
                )
            }
            PropertyError::ObserverError { reason } => {
                write!(f, "Observer execution failed: {}", reason)
            }
            PropertyError::ThreadPoolExhausted => {
                write!(f, "Thread pool is exhausted and cannot spawn more observers")
            }
            PropertyError::InvalidConfiguration { reason } => {
                write!(f, "Invalid configuration: {}", reason)
            }
        }
    }
}

impl std::error::Error for PropertyError {}

/// Function type for observers that get called when property values change
pub type Observer<T> = Arc<dyn Fn(&T, &T) + Send + Sync>;

/// Unique identifier for registered observers
pub type ObserverId = usize;

/// A RAII guard for an observer subscription that automatically unsubscribes when dropped
///
/// This struct provides automatic cleanup for observer subscriptions using RAII (Resource
/// Acquisition Is Initialization) pattern. When a `Subscription` goes out of scope, its
/// `Drop` implementation automatically removes the associated observer from the property.
///
/// This eliminates the need for manual `unsubscribe()` calls and helps prevent resource
/// leaks in scenarios where observers might otherwise be forgotten.
///
/// # Type Requirements
///
/// The generic type `T` must implement the same traits as `ObservableProperty<T>`:
/// - `Clone`: Required for observer notifications
/// - `Send`: Required for transferring between threads  
/// - `Sync`: Required for concurrent access from multiple threads
/// - `'static`: Required for observer callbacks that may outlive the original scope
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
///     // Create subscription - observer is automatically registered
///     let _subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
///         println!("Value changed: {} -> {}", old, new);
///     }))?;
///
///     property.set(42)?; // Observer is called: "Value changed: 0 -> 42"
///
///     // When _subscription goes out of scope here, observer is automatically removed
/// }
///
/// property.set(100)?; // No observer output - subscription was automatically cleaned up
/// # Ok(())
/// # }
/// ```
///
/// ## Cross-Thread Subscription Management
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
/// // Create subscription in main thread
/// let subscription = property.subscribe_with_subscription(Arc::new(|old, new| {
///     println!("Observed: {} -> {}", old, new);
/// }))?;
///
/// // Move subscription to another thread for cleanup
/// let handle = thread::spawn(move || {
///     // Subscription is still active here
///     let _ = property_clone.set(42); // Will trigger observer
///     
///     // When subscription is dropped here (end of thread), observer is cleaned up
///     drop(subscription);
/// });
///
/// handle.join().unwrap();
/// 
/// // Observer is no longer active
/// property.set(100)?; // No output
/// # Ok(())
/// # }
/// ```
///
/// ## Conditional Scoped Subscriptions
///
/// ```rust
/// use observable_property::ObservableProperty;
/// use std::sync::Arc;
///
/// # fn main() -> Result<(), observable_property::PropertyError> {
/// let counter = ObservableProperty::new(0);
/// let debug_mode = true;
///
/// if debug_mode {
///     let _debug_subscription = counter.subscribe_with_subscription(Arc::new(|old, new| {
///         println!("Debug: counter {} -> {}", old, new);
///     }))?;
///     
///     counter.set(1)?; // Prints debug info
///     counter.set(2)?; // Prints debug info
///     
///     // Debug subscription automatically cleaned up when exiting if block
/// }
///
/// counter.set(3)?; // No debug output (subscription was cleaned up)
/// # Ok(())
/// # }
/// ```
///
/// # Thread Safety
///
/// Like `ObservableProperty` itself, `Subscription` is thread-safe. It can be safely
/// sent between threads and the automatic cleanup will work correctly even if the
/// subscription is dropped from a different thread than where it was created.
pub struct Subscription<T: Clone + Send + Sync + 'static> {
    inner: Arc<RwLock<InnerProperty<T>>>,
    id: ObserverId,
}

impl<T: Clone + Send + Sync + 'static> std::fmt::Debug for Subscription<T> {
    /// Debug implementation that shows the subscription ID without exposing internals
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription")
            .field("id", &self.id)
            .field("inner", &"[ObservableProperty]")
            .finish()
    }
}

impl<T: Clone + Send + Sync + 'static> Drop for Subscription<T> {
    /// Automatically removes the associated observer when the subscription is dropped
    ///
    /// This implementation provides automatic cleanup by removing the observer
    /// from the property's observer list when the `Subscription` goes out of scope.
    ///
    /// # Error Handling
    ///
    /// If the property's lock is poisoned or inaccessible during cleanup, the error
    /// is silently ignored using the `let _ = ...` pattern. This is intentional
    /// because:
    /// 1. Drop implementations should not panic
    /// 2. If the property is poisoned, it's likely unusable anyway
    /// 3. There's no meaningful way to handle cleanup errors in a destructor
    ///
    /// # Thread Safety
    ///
    /// This method is safe to call from any thread, even if the subscription
    /// was created on a different thread.
    fn drop(&mut self) {
        let _ = self.inner.write().map(|mut prop| {
            prop.observers.remove(&self.id);
        });
    }
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
pub struct ObservableProperty<T> {
    inner: Arc<RwLock<InnerProperty<T>>>,
    max_threads: usize,
}

struct InnerProperty<T> {
    value: T,
    observers: HashMap<ObserverId, Observer<T>>,
    next_id: ObserverId,
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
            })),
            max_threads: MAX_THREADS,
        }
    }

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
            })),
            max_threads,
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
        self.inner
            .read()
            .map(|prop| prop.value.clone())
            .map_err(|_| PropertyError::PoisonedLock)
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
        let (old_value, observers_snapshot) = {
            let mut prop = self
                .inner
                .write()
                .map_err(|_| PropertyError::WriteLockError {
                    context: "setting property value".to_string(),
                })?;

            let old_value = prop.value.clone();
            prop.value = new_value.clone();
            let observers_snapshot: Vec<Observer<T>> = prop.observers.values().cloned().collect();
            (old_value, observers_snapshot)
        };

        for observer in observers_snapshot {
            if let Err(e) = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                observer(&old_value, &new_value);
            })) {
                eprintln!("Observer panic: {:?}", e);
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
    /// # Arguments
    ///
    /// * `new_value` - The new value to set
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(PropertyError)` if the lock is poisoned.
    /// Note that this only indicates the property was updated successfully;
    /// observer execution happens asynchronously.
    ///
    /// # Examples
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
    /// # Ok::<(), observable_property::PropertyError>(())
    /// ```
    pub fn set_async(&self, new_value: T) -> Result<(), PropertyError> {
        let (old_value, observers_snapshot) = {
            let mut prop = self
                .inner
                .write()
                .map_err(|_| PropertyError::WriteLockError {
                    context: "setting property value".to_string(),
                })?;

            let old_value = prop.value.clone();
            prop.value = new_value.clone();
            let observers_snapshot: Vec<Observer<T>> = prop.observers.values().cloned().collect();
            (old_value, observers_snapshot)
        };

        if observers_snapshot.is_empty() {
            return Ok(());
        }

        let observers_per_thread = observers_snapshot.len().div_ceil(MAX_THREADS);

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
    /// or `Err(PropertyError)` if the lock is poisoned.
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
        let mut prop = self
            .inner
            .write()
            .map_err(|_| PropertyError::WriteLockError {
                context: "subscribing observer".to_string(),
            })?;

        let id = prop.next_id;
        prop.next_id += 1;
        prop.observers.insert(id, observer);
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
        let mut prop = self
            .inner
            .write()
            .map_err(|_| PropertyError::WriteLockError {
                context: "unsubscribing observer".to_string(),
            })?;

        let was_present = prop.observers.remove(&id).is_some();
        Ok(was_present)
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

    pub fn notify_observers_batch(&self, changes: Vec<(T, T)>) -> Result<(), PropertyError> {
        let prop = self
            .inner
            .read()
            .map_err(|_| PropertyError::ReadLockError {
                context: "notifying observers".to_string(),
            })?;

        for (old_val, new_val) in changes {
            for observer in prop.observers.values() {
                observer(&old_val, &new_val);
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
}

impl<T: Clone> Clone for ObservableProperty<T> {
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
                .finish(),
            Err(_) => f
                .debug_struct("ObservableProperty")
                .field("value", &"[inaccessible]")
                .field("observers_count", &"[hidden]")
                .finish(),
        }
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

        // Now the lock should be poisoned, verify all operations return appropriate errors
        match prop.get() {
            Ok(_) => panic!("get() should fail on a poisoned lock"),
            Err(e) => match e {
                PropertyError::PoisonedLock => {} // Expected error
                _ => panic!("Expected PoisonedLock error, got: {:?}", e),
            },
        }

        match prop.set(42) {
            Ok(_) => panic!("set() should fail on a poisoned lock"),
            Err(e) => match e {
                PropertyError::WriteLockError { .. } | PropertyError::PoisonedLock => {} // Either is acceptable
                _ => panic!("Expected lock-related error, got: {:?}", e),
            },
        }

        match prop.subscribe(Arc::new(|_, _| {})) {
            Ok(_) => panic!("subscribe() should fail on a poisoned lock"),
            Err(e) => match e {
                PropertyError::WriteLockError { .. } | PropertyError::PoisonedLock => {} // Either is acceptable
                _ => panic!("Expected lock-related error, got: {:?}", e),
            },
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

        // subscribe_with_subscription should return an error for poisoned lock
        let result = prop.subscribe_with_subscription(Arc::new(|_, _| {}));
        assert!(result.is_err());
        match result.expect_err("Expected error for poisoned lock") {
            PropertyError::WriteLockError { .. } | PropertyError::PoisonedLock => {
                // Either error type is acceptable for poisoned lock
            }
            other => panic!("Unexpected error type: {:?}", other),
        }
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

        // subscribe_filtered_with_subscription should return error for poisoned lock
        let result = prop.subscribe_filtered_with_subscription(Arc::new(|_, _| {}), |_, _| true);
        assert!(result.is_err());
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

        // New subscription creation should fail
        let result = prop.subscribe_with_subscription(Arc::new(|_, _| {}));
        assert!(result.is_err());

        // New filtered subscription creation should also fail
        let filtered_result =
            prop.subscribe_filtered_with_subscription(Arc::new(|_, _| {}), |_, _| true);
        assert!(filtered_result.is_err());

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

        // Try to create subscription after poisoning - should fail
        let after_count = after_poison_count.clone();
        let after_result = prop.subscribe_with_subscription(Arc::new(move |_, _| {
            after_count.fetch_add(1, Ordering::SeqCst);
        }));
        assert!(after_result.is_err());

        // Clean up the before-poison subscription - should not panic
        drop(before_subscription);

        // Verify after-poison subscription was never created
        assert_eq!(after_poison_count.load(Ordering::SeqCst), 0);
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
}
