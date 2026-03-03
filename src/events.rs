//! Event tracking for observable properties

use std::time::Instant;

/// Represents a single change event in the property's history
///
/// This struct captures all details of a value change for event sourcing,
/// enabling time-travel debugging, audit logs, and event replay capabilities.
#[derive(Clone, Debug)]
pub struct PropertyEvent<T: Clone> {
    /// When the change occurred
    pub timestamp: Instant,
    /// The value before the change
    pub old_value: T,
    /// The value after the change
    pub new_value: T,
    /// Sequential event number (starts at 0 for first change)
    pub event_number: usize,
    /// Thread that triggered the change (for debugging)
    pub thread_id: String,
}

/// Information about a property change with stack trace
#[cfg(feature = "debug")]
#[derive(Clone)]
pub(crate) struct ChangeLog {
    pub(crate) timestamp: Instant,
    pub(crate) old_value_repr: String,
    pub(crate) new_value_repr: String,
    pub(crate) backtrace: String,
    pub(crate) thread_id: String,
}
