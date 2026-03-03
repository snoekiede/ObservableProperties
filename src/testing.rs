//! Test helper utilities for working with `ObservableProperty` in tests.

use std::sync::Arc;
use std::sync::mpsc::{channel, Receiver};
use std::time::Duration;
use crate::{ObservableProperty, Subscription};

/// Blocks the current thread until the next notification from the property.
///
/// This is useful for synchronizing test code with property changes.
/// Times out after 5 seconds to prevent tests from hanging indefinitely.
///
/// # Panics
///
/// Panics if:
/// - The subscription fails
/// - No notification is received within 5 seconds
///
/// # Example
///
/// ```
/// use observable_property::{ObservableProperty, testing::await_notification};
/// use std::thread;
///
/// let property = ObservableProperty::new(0);
///
/// thread::spawn({
///     let prop = property.clone();
///     move || {
///         thread::sleep(std::time::Duration::from_millis(100));
///         prop.set(42).unwrap();
///     }
/// });
///
/// await_notification(&property);
/// assert_eq!(property.get().unwrap(), 42);
/// ```
pub fn await_notification<T: Clone + Send + Sync + 'static>(
    property: &ObservableProperty<T>,
) {
    let (tx, rx) = channel();
    
    let _subscription = property
        .subscribe_with_subscription(Arc::new(move |_old, _new| {
            let _ = tx.send(());
        }))
        .expect("Failed to subscribe for await_notification");
    
    rx.recv_timeout(Duration::from_secs(5))
        .expect("Timeout waiting for notification");
}

/// Records all changes to a property for test assertions.
///
/// Returns a `ChangeCollector` that captures old and new values
/// whenever the property changes.
///
/// # Example
///
/// ```
/// use observable_property::{ObservableProperty, testing::collect_changes};
///
/// let property = ObservableProperty::new(0);
/// let collector = collect_changes(&property);
///
/// property.set(1).unwrap();
/// property.set(2).unwrap();
/// property.set(3).unwrap();
///
/// let changes = collector.changes();
/// assert_eq!(changes.len(), 3);
/// assert_eq!(changes[0], (0, 1));
/// assert_eq!(changes[1], (1, 2));
/// assert_eq!(changes[2], (2, 3));
/// ```
pub fn collect_changes<T: Clone + Send + Sync + 'static>(
    property: &ObservableProperty<T>,
) -> ChangeCollector<T> {
    ChangeCollector::new(property)
}

/// Collects changes to an observable property for testing purposes.
///
/// This struct maintains a subscription to a property and records
/// all changes (old value, new value) in a thread-safe manner.
pub struct ChangeCollector<T: Clone + Send + Sync + 'static> {
    receiver: Receiver<(T, T)>,
    _subscription: Subscription<T>,
}

impl<T: Clone + Send + Sync + 'static> ChangeCollector<T> {
    /// Creates a new `ChangeCollector` that subscribes to the given property.
    fn new(property: &ObservableProperty<T>) -> Self {
        let (tx, rx) = channel();
        
        let subscription = property
            .subscribe_with_subscription(Arc::new(move |old, new| {
                let _ = tx.send((old.clone(), new.clone()));
            }))
            .expect("Failed to subscribe for collect_changes");
        
        ChangeCollector {
            receiver: rx,
            _subscription: subscription,
        }
    }

    /// Returns all collected changes as a vector of (old_value, new_value) tuples.
    ///
    /// This method drains all available changes from the internal receiver.
    pub fn changes(&self) -> Vec<(T, T)> {
        let mut changes = Vec::new();
        while let Ok(change) = self.receiver.try_recv() {
            changes.push(change);
        }
        changes
    }

    /// Waits for at least `count` changes to occur, with a timeout.
    ///
    /// Returns all collected changes once the count is reached.
    /// Times out after 5 seconds.
    ///
    /// # Panics
    ///
    /// Panics if the expected number of changes is not received within the timeout.
    pub fn wait_for_changes(&self, count: usize) -> Vec<(T, T)> {
        let mut changes = Vec::new();
        let timeout = Duration::from_secs(5);
        let start = std::time::Instant::now();
        
        while changes.len() < count {
            if start.elapsed() > timeout {
                panic!(
                    "Timeout waiting for {} changes, only received {}",
                    count,
                    changes.len()
                );
            }
            
            match self.receiver.recv_timeout(Duration::from_millis(100)) {
                Ok(change) => changes.push(change),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(e) => panic!("Error receiving change: {}", e),
            }
        }
        
        changes
    }

    /// Returns the total number of changes currently collected.
    pub fn count(&self) -> usize {
        self.changes().len()
    }
}
