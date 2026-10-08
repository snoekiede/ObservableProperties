//! Core property structures

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
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
    pub(crate) notification_time_nanos: u128,
    pub(crate) notification_count: u128,
    pub(crate) active_async_workers: usize,
    // Debug tracking
    #[cfg(feature = "debug")]
    pub(crate) debug_logging_enabled: bool,
    #[cfg(feature = "debug")]
    pub(crate) change_logs: Vec<ChangeLog>,
    // Change coalescing
    pub(crate) batch_depth: usize,
    pub(crate) batch_initial_value: Option<T>,
    pub(crate) batch_changed: bool,
    // Custom equality function
    pub(crate) eq_fn: Option<Arc<dyn Fn(&T, &T) -> bool + Send + Sync>>,
    // Validator function
    pub(crate) validator: Option<Arc<dyn Fn(&T) -> Result<(), String> + Send + Sync>>,
    // Event sourcing
    pub(crate) event_log: Option<Vec<PropertyEvent<T>>>,
    pub(crate) event_log_size: usize,
}

impl<T: Clone + Send + Sync + 'static> InnerProperty<T> {
    pub(crate) fn record_notification(&mut self, observer_calls: usize, elapsed: Duration) {
        self.observer_calls = self.observer_calls.saturating_add(observer_calls);
        self.notification_time_nanos = self
            .notification_time_nanos
            .saturating_add(elapsed.as_nanos());
        self.notification_count = self.notification_count.saturating_add(1);
    }

    pub(crate) fn record_change(&mut self, old_value: &T, new_value: &T, store_history: bool) {
        self.total_changes = self.total_changes.saturating_add(1);

        if store_history && let Some(history) = &mut self.history {
            history.push(old_value.clone());
            if history.len() > self.history_size {
                let overflow = history.len() - self.history_size;
                history.drain(0..overflow);
            }
        }

        if let Some(event_log) = &mut self.event_log {
            event_log.push(PropertyEvent {
                timestamp: Instant::now(),
                old_value: old_value.clone(),
                new_value: new_value.clone(),
                event_number: self.total_changes.saturating_sub(1),
                thread_id: format!("{:?}", thread::current().id()),
            });

            if self.event_log_size > 0 && event_log.len() > self.event_log_size {
                let overflow = event_log.len() - self.event_log_size;
                event_log.drain(0..overflow);
            }
        }
    }
}
