//! Core property structures

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use crate::observer::{ObserverId, ObserverRef};
use crate::events::PropertyEvent;
#[cfg(feature = "debug")]
use crate::events::ChangeLog;

/// Internal property state
pub(crate) struct InnerProperty<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub(crate) value: T,
    pub(crate) observers: HashMap<ObserverId, ObserverRef<T>>,
    pub(crate) next_id: ObserverId,
    pub(crate) history: Option<Vec<T>>,
    pub(crate) history_size: usize,
    // Metrics tracking
    pub(crate) total_changes: usize,
    pub(crate) observer_calls: usize,
    pub(crate) notification_times: Vec<Duration>,
    // Debug tracking
    #[cfg(feature = "debug")]
    pub(crate) debug_logging_enabled: bool,
    #[cfg(feature = "debug")]
    pub(crate) change_logs: Vec<ChangeLog>,
    // Change coalescing
    pub(crate) batch_depth: usize,
    pub(crate) batch_initial_value: Option<T>,
    // Custom equality function
    pub(crate) eq_fn: Option<Arc<dyn Fn(&T, &T) -> bool + Send + Sync>>,
    // Validator function
    pub(crate) validator: Option<Arc<dyn Fn(&T) -> Result<(), String> + Send + Sync>>,
    // Event sourcing
    pub(crate) event_log: Option<Vec<PropertyEvent<T>>>,
    pub(crate) event_log_size: usize,
}
