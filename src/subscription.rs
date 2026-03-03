//! RAII subscription guards for automatic observer cleanup

use std::sync::{Arc, RwLock};
use crate::observer::ObserverId;
use crate::core::InnerProperty;

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
/// # Thread Safety
///
/// Like `ObservableProperty` itself, `Subscription` is thread-safe. It can be safely
/// sent between threads and the automatic cleanup will work correctly even if the
/// subscription is dropped from a different thread than where it was created.
pub struct Subscription<T: Clone + Send + Sync + 'static> {
    pub(crate) inner: Arc<RwLock<InnerProperty<T>>>,
    pub(crate) id: ObserverId,
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
        // Graceful degradation: always attempt to clean up, even from poisoned locks
        match self.inner.write() {
            Ok(mut guard) => {
                guard.observers.remove(&self.id);
            }
            Err(poisoned) => {
                // Recover from poisoned lock to ensure cleanup happens
                let mut guard = poisoned.into_inner();
                guard.observers.remove(&self.id);
            }
        }
    }
}
