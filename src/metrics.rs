//! Performance metrics for observable properties

use std::time::Duration;

/// Performance metrics for an observable property
///
/// This struct provides insight into property usage patterns and observer
/// notification performance, useful for debugging and performance optimization.
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
/// property.set(42)?;
/// property.set(100)?;
///
/// let metrics = property.get_metrics()?;
/// println!("Total changes: {}", metrics.total_changes);
/// println!("Observer calls: {}", metrics.observer_calls);
/// println!("Avg notification time: {:?}", metrics.avg_notification_time);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PropertyMetrics {
    /// Total number of times the property value has changed
    pub total_changes: usize,
    /// Total number of observer calls (notification events)
    pub observer_calls: usize,
    /// Average time taken to notify all observers per change
    pub avg_notification_time: Duration,
}
