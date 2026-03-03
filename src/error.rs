//! Error types for observable properties

use std::fmt;

/// Errors that can occur when working with ObservableProperty
///
/// # Note on Lock Poisoning
///
/// This implementation uses **graceful degradation** for poisoned locks. When a lock
/// is poisoned (typically due to a panic in an observer or another thread), the
/// library automatically recovers the inner value using [`PoisonError::into_inner()`](std::sync::PoisonError::into_inner).
///
/// This means:
/// - **All operations continue to work** even after a lock is poisoned
/// - No `ReadLockError`, `WriteLockError`, or `PoisonedLock` errors will occur in practice
/// - The system remains operational and observers continue to function
///
/// The error variants are kept for backward compatibility and potential future use cases,
/// but with the current implementation, poisoned locks are transparent to users.
///
/// # Production Benefit
///
/// This approach ensures maximum availability and resilience in production systems where
/// a misbehaving observer shouldn't bring down the entire property system.
#[derive(Debug, Clone)]
pub enum PropertyError {
    /// Failed to acquire a read lock on the property
    ///
    /// **Note**: With graceful degradation, this error is unlikely to occur in practice
    /// as poisoned locks are automatically recovered.
    ReadLockError {
        /// Context describing what operation was being attempted
        context: String,
    },
    /// Failed to acquire a write lock on the property
    ///
    /// **Note**: With graceful degradation, this error is unlikely to occur in practice
    /// as poisoned locks are automatically recovered.
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
    ///
    /// **Note**: With graceful degradation, this error will not occur in practice
    /// as the implementation automatically recovers from poisoned locks.
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
    /// Failed to persist the property value
    PersistenceError {
        /// Description of the persistence error
        reason: String,
    },
    /// Attempted to undo when no history is available
    NoHistory {
        /// Description of the issue
        reason: String,
    },
    /// Validation failed for the provided value
    ValidationError {
        /// Description of why validation failed
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
            PropertyError::PersistenceError { reason } => {
                write!(f, "Persistence failed: {}", reason)
            }
            PropertyError::NoHistory { reason } => {
                write!(f, "No history available: {}", reason)
            }
            PropertyError::ValidationError { reason } => {
                write!(f, "Validation failed: {}", reason)
            }
        }
    }
}

impl std::error::Error for PropertyError {}
