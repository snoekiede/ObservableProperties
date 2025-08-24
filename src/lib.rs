//! # Observable Property
//!
//! A thread-safe observable property implementation for Rust that allows you to
//! observe changes to values across multiple threads.
//!
//! ## Features
//!
//! - **Thread-safe**: Uses `Arc<RwLock<>>` for safe concurrent access
//! - **Observer pattern**: Subscribe to property changes with callbacks
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

use std::collections::HashMap;
use std::fmt;
use std::panic;
use std::sync::{Arc, RwLock};
use std::thread;

/// Errors that can occur when working with ObservableProperty
#[derive(Debug, Clone)]
pub enum PropertyError {
    /// Failed to acquire a read lock on the property
    ReadLockError { 
        /// Context describing what operation was being attempted
        context: String 
    },
    /// Failed to acquire a write lock on the property  
    WriteLockError { 
        /// Context describing what operation was being attempted
        context: String 
    },
    /// Attempted to unsubscribe an observer that doesn't exist
    ObserverNotFound { 
        /// The ID of the observer that wasn't found
        id: usize 
    },
    /// The property's lock has been poisoned due to a panic in another thread
    PoisonedLock,
    /// An observer function encountered an error during execution
    ObserverError { 
        /// Description of what went wrong
        reason: String 
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
        }
    }
}

impl std::error::Error for PropertyError {}

/// Function type for observers that get called when property values change
pub type Observer<T> = Arc<dyn Fn(&T, &T) + Send + Sync>;

/// Unique identifier for registered observers
pub type ObserverId = usize;

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

        const MAX_THREADS: usize = 4;
        let observers_per_thread = (observers_snapshot.len() + MAX_THREADS - 1) / MAX_THREADS;

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

    /// Removes an observer by its ID
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
    /// let id = property.subscribe(Arc::new(|_, _| {})).unwrap();
    ///
    /// let was_removed = property.unsubscribe(id).unwrap();
    /// assert!(was_removed); // Observer existed and was removed
    ///
    /// let was_removed_again = property.unsubscribe(id).unwrap();
    /// assert!(!was_removed_again); // Observer no longer exists
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
    /// ).unwrap();
    ///
    /// property.set(10).unwrap(); // Triggers observer (0 -> 10)
    /// property.set(5).unwrap();  // Does NOT trigger observer (10 -> 5)
    /// property.set(15).unwrap(); // Triggers observer (5 -> 15)
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
    /// })).unwrap();
    ///
    /// // This change through property1 will trigger the observer on property2
    /// property1.set(100).unwrap();
    /// ```
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Clone + std::fmt::Debug + Send + Sync + 'static> std::fmt::Debug for ObservableProperty<T> {
    /// Debug implementation that shows the current value if accessible
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.get() {
            Ok(value) => f.debug_struct("ObservableProperty")
                .field("value", &value)
                .field("observers_count", &"[hidden]")
                .finish(),
            Err(_) => f.debug_struct("ObservableProperty")
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
            |old, new| new > old
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
            Ok(_) => {},
            Err(e) => panic!("Failed to unsubscribe filtered observer: {}", e),
        }
    }

    #[test]
    fn test_thread_safety_concurrent_reads() {
        let prop = Arc::new(ObservableProperty::new(42i32));
        let num_threads = 10;
        let reads_per_thread = 100;
        
        let handles: Vec<_> = (0..num_threads).map(|_| {
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
        }).collect();
        
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
            let _guard = prop_clone.inner.write().unwrap();
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
            }
        }

        match prop.set(42) {
            Ok(_) => panic!("set() should fail on a poisoned lock"),
            Err(e) => match e {
                PropertyError::WriteLockError { .. } | PropertyError::PoisonedLock => {} // Either is acceptable
                _ => panic!("Expected lock-related error, got: {:?}", e),
            }
        }

        match prop.subscribe(Arc::new(|_, _| {})) {
            Ok(_) => panic!("subscribe() should fail on a poisoned lock"),
            Err(e) => match e {
                PropertyError::WriteLockError { .. } | PropertyError::PoisonedLock => {} // Either is acceptable
                _ => panic!("Expected lock-related error, got: {:?}", e),
            }
        }
    }

    #[test]
    fn test_observer_panic_isolation() {
        let prop = ObservableProperty::new(0);
        let call_counts = Arc::new(AtomicUsize::new(0));

        // First observer will panic
        let panic_observer_id = prop.subscribe(Arc::new(|_, _| {
            panic!("This observer deliberately panics");
        })).unwrap();

        // Second observer should still be called despite first one panicking
        let counts = call_counts.clone();
        let normal_observer_id = prop.subscribe(Arc::new(move |_, _| {
            counts.fetch_add(1, Ordering::SeqCst);
        })).unwrap();

        // Trigger the observers - this shouldn't panic despite the first observer panicking
        prop.set(42).unwrap();

        // Verify the second observer was still called
        assert_eq!(call_counts.load(Ordering::SeqCst), 1);

        // Clean up
        prop.unsubscribe(panic_observer_id).unwrap();
        prop.unsubscribe(normal_observer_id).unwrap();
    }

    #[test]
    fn test_observer_id_edge_cases() {
        let prop = ObservableProperty::new(0);
        let mut observer_ids = Vec::new();

        // Helper function to generate many observer IDs
        // Using a separate function instead of a closure to avoid borrow issues
        fn subscribe_observers(prop: &ObservableProperty<i32>, count: usize) -> Vec<ObserverId> {
            let mut ids = Vec::with_capacity(count);
            for _ in 0..count {
                let id = prop.subscribe(Arc::new(|_, _| {})).unwrap();
                ids.push(id);
            }
            ids
        }

        // 1. Test very large number of observers (testing ID generation)
        let new_ids = subscribe_observers(&prop, 1000);
        observer_ids.extend(new_ids);

        // 2. Verify all IDs are unique
        let unique_count = observer_ids.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(unique_count, observer_ids.len(), "Observer IDs should be unique");

        // 3. Test unsubscribing all observers
        for id in &observer_ids {
            let result = prop.unsubscribe(*id).unwrap();
            assert!(result, "Unsubscribe should return true for valid ID");
        }

        // 4. Create one more observer to ensure ID generation works after mass unsubscribe
        let new_id = prop.subscribe(Arc::new(|_, _| {})).unwrap();

        // Observe internal state to confirm next_id didn't reset (would require exposing internal state)
        // Instead, we just verify the new ID doesn't duplicate any previous ID
        assert!(!observer_ids.contains(&new_id), "New ID should not duplicate any previous ID");
    }

    #[test]
    fn test_complex_custom_types() {
        #[derive(Clone, Debug, PartialEq)]
        struct ComplexType {
            name: String,
            values: Vec<i32>,
            metadata: HashMap<String, String>,
        }

        let initial = ComplexType {
            name: "Initial".to_string(),
            values: vec![1, 2, 3],
            metadata: {
                let mut map = HashMap::new();
                map.insert("created".to_string(), "now".to_string());
                map
            },
        };

        let property = ObservableProperty::new(initial.clone());
        let received_values = Arc::new(RwLock::new(Vec::new()));

        let values_clone = received_values.clone();
        let observer_id = property.subscribe(Arc::new(move |_, new| {
            if let Ok(mut values) = values_clone.write() {
                values.push(new.clone());
            }
        })).unwrap();

        let updated = ComplexType {
            name: "Updated".to_string(),
            values: vec![4, 5, 6],
            metadata: {
                let mut map = HashMap::new();
                map.insert("created".to_string(), "now".to_string());
                map.insert("updated".to_string(), "later".to_string());
                map
            },
        };

        // Set the new value
        property.set(updated.clone()).unwrap();

        // Verify the observer received the correct value
        let received = received_values.read().unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(&received[0], &updated);

        // Verify get() returns the updated value
        let current = property.get().unwrap();
        assert_eq!(current, updated);

        property.unsubscribe(observer_id).unwrap();
    }

    #[test]
    fn test_unsubscribe_nonexistent_observer() {
        let property = ObservableProperty::new(0);

        // Generate a valid observer ID
        let valid_id = property.subscribe(Arc::new(|_, _| {})).unwrap();

        // Create an ID that doesn't exist (valid_id + 1000 should not exist)
        let nonexistent_id = valid_id + 1000;

        // Test unsubscribing a nonexistent observer
        match property.unsubscribe(nonexistent_id) {
            Ok(was_present) => {
                assert!(!was_present, "Unsubscribe should return false for nonexistent ID");
            },
            Err(e) => panic!("Unsubscribe returned error: {:?}", e),
        }

        // Also verify that unsubscribing twice returns false the second time
        property.unsubscribe(valid_id).unwrap(); // First unsubscribe should return true

        let result = property.unsubscribe(valid_id).unwrap();
        assert!(!result, "Second unsubscribe should return false");
    }
}
