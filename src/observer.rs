//! Observer types and implementations

use std::sync::Arc;

/// Function type for observers that get called when property values change
pub type Observer<T> = Arc<dyn Fn(&T, &T) + Send + Sync>;

/// Unique identifier for registered observers
pub type ObserverId = usize;

/// Reference to an observer, either strong or weak
pub(crate) enum ObserverRef<T: Clone + Send + Sync + 'static> {
    /// Strong reference to an observer (keeps the observer alive)
    Strong(Observer<T>),
    /// Weak reference to an observer (automatically cleaned up when dropped)
    Weak(std::sync::Weak<dyn Fn(&T, &T) + Send + Sync>),
}

impl<T: Clone + Send + Sync + 'static> ObserverRef<T> {
    /// Attempts to get a callable observer from this reference
    ///
    /// For strong references, always returns Some. For weak references, attempts
    /// to upgrade and returns None if the observer has been dropped.
    pub(crate) fn try_call(&self) -> Option<Observer<T>> {
        match self {
            ObserverRef::Strong(arc) => Some(arc.clone()),
            ObserverRef::Weak(weak) => weak.upgrade(),
        }
    }
}
