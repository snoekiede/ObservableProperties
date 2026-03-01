use observable_property::ObservableProperty;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Observable Property - Batched Updates Example ===\n");

    // Example 1: Change Coalescing with begin_update/end_update
    println!("Example 1: Change Coalescing (begin_update/end_update)");
    println!("-------------------------------------------------------");
    let property = ObservableProperty::new(0);
    let notification_count = Arc::new(AtomicUsize::new(0));
    let count_clone = notification_count.clone();

    property.subscribe(Arc::new(move |old, new| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        println!("  Notification: {} -> {}", old, new);
    }))?;

    // Begin batch update
    property.begin_update()?;
    
    // Multiple changes - no notifications yet
    property.set(10)?;
    property.set(20)?;
    property.set(30)?;
    
    // End batch - single notification from 0 to 30
    property.end_update()?;
    
    println!("  Final value: {}", property.get()?);
    println!("  Total notifications: {} (should be 1)\n", notification_count.load(Ordering::SeqCst));

    // Example 2: Nested batch updates
    println!("Example 2: Nested batch updates");
    println!("--------------------------------");
    let nested = ObservableProperty::new(100);
    
    nested.subscribe(Arc::new(|old, new| {
        println!("  Notification: {} -> {}", old, new);
    }))?;

    nested.begin_update()?;  // Outer batch
    nested.set(110)?;
    
    nested.begin_update()?;  // Inner batch
    nested.set(120)?;
    nested.set(130)?;
    nested.end_update()?;    // End inner (no notification yet)
    
    nested.set(140)?;
    nested.end_update()?;    // End outer - notification: 100 -> 140
    
    println!("  Final value: {}\n", nested.get()?);

    // Example 3: No-op batch (value returns to original)
    println!("Example 3: No-op batch (value returns to original)");
    println!("---------------------------------------------------");
    let unchanged = ObservableProperty::new(42);
    let notify_count = Arc::new(AtomicUsize::new(0));
    let count = notify_count.clone();
    
    unchanged.subscribe(Arc::new(move |old, new| {
        count.fetch_add(1, Ordering::SeqCst);
        println!("  Notification: {} -> {}", old, new);
    }))?;

    unchanged.begin_update()?;
    unchanged.set(50)?;
    unchanged.set(42)?;  // Back to original
    unchanged.end_update()?;  // Still notifies (42 -> 42) - library doesn't check equality
    
    println!("  Notifications: {} (always 1 since library doesn't require PartialEq)\n", notify_count.load(Ordering::SeqCst));

    println!("=== update_batch() Examples ===\n");

    println!("=== update_batch() Examples ===\n");

    // Example 4: Animation through intermediate positions
    println!("Example 4: Animation with intermediate states");
    println!("----------------------------------------------");
    let position = ObservableProperty::new(0);
    let notification_count = Arc::new(AtomicUsize::new(0));
    let count_clone = notification_count.clone();

    position.subscribe(Arc::new(move |old, new| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        println!("  Position: {} -> {}", old, new);
    }))?;

    // Animate from 0 to 100 in steps
    position.update_batch(|_current| {
        vec![25, 50, 75, 100]
    })?;

    println!("  Final position: {}", position.get()?);
    println!("  Total notifications: {}\n", notification_count.load(Ordering::SeqCst));

    // Example 5: Multi-step string transformation
    println!("Example 5: Multi-step string transformation");
    println!("--------------------------------------------");
    let text = ObservableProperty::new(String::from("hello"));

    text.subscribe(Arc::new(|old, new| {
        println!("  Transformation: '{}' -> '{}'", old, new);
    }))?;

    text.update_batch(|current| {
        let step1 = current.to_uppercase();
        let step2 = format!("{}!", step1);
        let step3 = format!("{} WORLD", step2);
        vec![step1, step2, step3]
    })?;

    println!("  Final text: '{}'\n", text.get()?);

    // Example 6: Counter increment with intermediate values
    println!("Example 6: Counter with intermediate values");
    println!("--------------------------------------------");
    let counter = ObservableProperty::new(0);

    counter.subscribe(Arc::new(|old, new| {
        println!("  Count: {} -> {}", old, new);
    }))?;

    counter.update_batch(|_current| {
        // Return intermediate states from 0 to 10
        (1..=10).collect()
    })?;

    println!("  Final count: {}\n", counter.get()?);

    // Example 7: Temperature changes with gradual increase
    println!("Example 7: Temperature gradual increase");
    println!("----------------------------------------");
    let temperature = ObservableProperty::new(20.0);

    temperature.subscribe(Arc::new(|old, new| {
        println!("  Temperature: {:.1}°C -> {:.1}°C", old, new);
    }))?;

    temperature.update_batch(|_current| {
        // Gradually increase from 20 to 25 degrees
        vec![21.0, 22.0, 23.0, 24.0, 25.0]
    })?;

    println!("  Final temperature: {:.1}°C\n", temperature.get()?);

    // Example 8: Empty batch (no changes)
    println!("Example 8: Empty batch (no notifications)");
    println!("------------------------------------------");
    let value = ObservableProperty::new(42);
    let was_notified = Arc::new(AtomicUsize::new(0));
    let flag = was_notified.clone();

    value.subscribe(Arc::new(move |old, new| {
        flag.fetch_add(1, Ordering::SeqCst);
        println!("  Value changed: {} -> {}", old, new);
    }))?;

    value.update_batch(|current| {
        *current = 100; // This modification is ignored since we return empty vec
        Vec::new()
    })?;

    println!("  Was notified: {} times", was_notified.load(Ordering::SeqCst));
    println!("  Value remains: {}\n", value.get()?);

    // Example 9: Fibonacci sequence generation
    println!("Example 9: Fibonacci sequence generation");
    println!("-----------------------------------------");
    let fib = ObservableProperty::new(1);

    fib.subscribe(Arc::new(|old, new| {
        println!("  Fibonacci: {} -> {}", old, new);
    }))?;

    fib.update_batch(|_current| {
        let mut sequence = vec![1, 2];
        for i in 2..8 {
            let next = sequence[i - 1] + sequence[i - 2];
            sequence.push(next);
        }
        sequence
    })?;

    println!("  Final Fibonacci number: {}\n", fib.get()?);

    println!("=== All examples completed successfully! ===");
    Ok(())
}
