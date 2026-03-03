//! Computed properties that automatically update based on dependencies

use std::sync::Arc;
use crate::{ObservableProperty, PropertyError};

/// Creates a computed property that automatically updates when dependencies change
///
/// A computed property is an observable property whose value is derived from one or more
/// other observable properties. Whenever any of the dependencies change, the computed
/// property automatically recalculates its value using the provided compute function.
///
/// # Type Parameters
///
/// * `T` - The type of the dependency properties
/// * `U` - The type of the computed property
/// * `F` - The compute function type that transforms dependency values into the computed value
///
/// # Arguments
///
/// * `dependencies` - A vector of observable properties that this computed property depends on
/// * `compute_fn` - A function that takes current values from all dependencies and returns the computed value
///
/// # Returns
///
/// Returns `Ok(Arc<ObservableProperty<U>>)` containing the computed property, or
/// `Err(PropertyError)` if initial value computation fails.
///
/// # Examples
///
/// ## Simple Computed Property
///
/// ```rust
/// use observable_property::{ObservableProperty, computed};
/// use std::sync::Arc;
///
/// # fn main() -> Result<(), observable_property::PropertyError> {
/// let width = Arc::new(ObservableProperty::new(10));
/// let height = Arc::new(ObservableProperty::new(20));
///
/// let area = computed(
///     vec![width.clone(), height.clone()],
///     |values| values[0] * values[1]
/// )?;
///
/// assert_eq!(area.get()?, 200);
///
/// // When dependencies change, computed value updates automatically
/// width.set(15)?;
/// std::thread::sleep(std::time::Duration::from_millis(10));
/// assert_eq!(area.get()?, 300);
/// # Ok(())
/// # }
/// ```
///
/// ## Complex Multi-Dependency Computation
///
/// ```rust
/// use observable_property::{ObservableProperty, computed};
/// use std::sync::Arc;
///
/// # fn main() -> Result<(), observable_property::PropertyError> {
/// // Temperature conversion system with multiple dependencies
/// let celsius = Arc::new(ObservableProperty::new(0.0));
///
/// let fahrenheit = computed(
///     vec![celsius.clone()],
///     |values| values[0] * 9.0 / 5.0 + 32.0
/// )?;
///
/// let kelvin = computed(
///     vec![celsius.clone()],
///     |values| values[0] + 273.15
/// )?;
///
/// assert_eq!(celsius.get()?, 0.0);
/// assert_eq!(fahrenheit.get()?, 32.0);
/// assert_eq!(kelvin.get()?, 273.15);
///
/// celsius.set(100.0)?;
/// std::thread::sleep(std::time::Duration::from_millis(10));
/// assert_eq!(fahrenheit.get()?, 212.0);
/// assert_eq!(kelvin.get()?, 373.15);
/// # Ok(())
/// # }
/// ```
///
/// # Thread Safety
///
/// Computed properties are fully thread-safe. Updates happen asynchronously in response to
/// dependency changes, and proper synchronization ensures the computed value is always
/// based on the current dependency values at the time of computation.
///
/// # Performance Considerations
///
/// - The compute function is called every time any dependency changes
/// - For expensive computations, consider using `subscribe_debounced` or `subscribe_throttled`
///   on the dependencies before computing
/// - The computed property uses async notifications, so there may be a small delay between
///   a dependency change and the computed value update
pub fn computed<T, U, F>(
    dependencies: Vec<Arc<ObservableProperty<T>>>,
    compute_fn: F,
) -> Result<Arc<ObservableProperty<U>>, PropertyError>
where
    T: Clone + Send + Sync + 'static,
    U: Clone + Send + Sync + 'static,
    F: Fn(&[T]) -> U + Send + Sync + 'static,
{
    // Collect initial values from all dependencies
    let initial_values: Result<Vec<T>, PropertyError> = 
        dependencies.iter().map(|dep| dep.get()).collect();
    let initial_values = initial_values?;
    
    // Compute initial value
    let initial_computed = compute_fn(&initial_values);
    
    // Create the computed property
    let computed_property = Arc::new(ObservableProperty::new(initial_computed));
    
    // Wrap compute_fn in Arc for sharing across multiple subscriptions
    let compute_fn = Arc::new(compute_fn);
    
    // Subscribe to each dependency
    for dependency in dependencies.iter() {
        let deps_clone = dependencies.clone();
        let computed_clone = computed_property.clone();
        let compute_fn_clone = compute_fn.clone();
        
        // Subscribe to this dependency
        dependency.subscribe(Arc::new(move |_old, _new| {
            // When any dependency changes, collect all current values
            let current_values: Result<Vec<T>, PropertyError> = 
                deps_clone.iter().map(|dep| dep.get()).collect();
            
            if let Ok(values) = current_values {
                // Recompute the value
                let new_computed = compute_fn_clone(&values);
                
                // Update the computed property
                if let Err(e) = computed_clone.set(new_computed) {
                    eprintln!("Error updating computed property: {}", e);
                }
            }
        }))?;
    }
    
    Ok(computed_property)
}
