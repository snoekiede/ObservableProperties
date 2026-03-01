use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Observable Property Metrics Demo ===\n");

    // Create an observable property
    let property = ObservableProperty::new(0);

    // Subscribe multiple observers
    println!("Subscribing observers...");
    property.subscribe(Arc::new(|old, new| {
        println!("  Observer 1: {} -> {}", old, new);
    }))?;

    property.subscribe(Arc::new(|old, new| {
        println!("  Observer 2: {} -> {}", old, new);
        // Simulate some work
        thread::sleep(Duration::from_micros(100));
    }))?;

    property.subscribe(Arc::new(|old, new| {
        println!("  Observer 3: {} -> {}", old, new);
    }))?;

    println!("\nMaking changes to property...\n");

    // Make some changes
    property.set(10)?;
    property.set(20)?;
    property.set(30)?;
    property.set(40)?;
    property.set(50)?;

    // Get and display metrics
    println!("\n=== Performance Metrics ===");
    let metrics = property.get_metrics()?;
    println!("Total changes: {}", metrics.total_changes);
    println!("Observer calls: {}", metrics.observer_calls);
    println!("Average notification time: {:?}", metrics.avg_notification_time);

    // Calculate some derived metrics
    if metrics.total_changes > 0 {
        let avg_observers_per_change = metrics.observer_calls / metrics.total_changes;
        println!("Average observers per change: {}", avg_observers_per_change);
    }

    println!("\n=== Testing Async Notifications ===\n");

    // Test async notifications
    println!("Making async changes...");
    property.set_async(60)?;
    property.set_async(70)?;
    property.set_async(80)?;

    // Give async operations time to complete
    thread::sleep(Duration::from_millis(50));

    // Get updated metrics
    let metrics = property.get_metrics()?;
    println!("\n=== Updated Metrics (after async) ===");
    println!("Total changes: {}", metrics.total_changes);
    println!("Observer calls: {}", metrics.observer_calls);
    println!("Average notification time: {:?}", metrics.avg_notification_time);

    Ok(())
}
