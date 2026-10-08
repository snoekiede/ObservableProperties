//! Computed properties that automatically update based on dependencies

use std::sync::{Arc, Weak};
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
    
    let weak_dependencies: Vec<Weak<ObservableProperty<T>>> =
        dependencies.iter().map(Arc::downgrade).collect();

    // The computed property owns dependencies and subscription guards. Callbacks
    // hold weak references so the dependency graph cannot retain the output.
    for dependency in dependencies.iter() {
        let weak_dependencies = weak_dependencies.clone();
        let computed_weak = Arc::downgrade(&computed_property);
        let compute_fn_clone = compute_fn.clone();
        
        dependency.subscribe_owned(&computed_property, Arc::new(move |_old, _new| {
            let Some(dependencies) = weak_dependencies
                .iter()
                .map(Weak::upgrade)
                .collect::<Option<Vec<_>>>()
            else {
                return;
            };

            let current_values: Result<Vec<T>, PropertyError> =
                dependencies.iter().map(|dependency| dependency.get()).collect();

            if let (Ok(values), Some(computed_property)) = (current_values, computed_weak.upgrade()) {
                let new_computed = compute_fn_clone(&values);

                if let Err(e) = computed_property.set(new_computed) {
                    eprintln!("Error updating computed property: {}", e);
                }
            }
        }))?;

        computed_property.retain_resource(dependency.clone());
    }
    
    Ok(computed_property)
}
