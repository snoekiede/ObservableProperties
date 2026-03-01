use observable_property::ObservableProperty;
use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Observable Property - Batched Updates Example ===\n");

    // Example 1: Animation through intermediate positions
    println!("Example 1: Animation with intermediate states");
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

    // Example 2: Multi-step string transformation
    println!("Example 2: Multi-step string transformation");
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

    // Example 3: Counter increment with intermediate values
    println!("Example 3: Counter with intermediate values");
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

    // Example 4: Temperature changes with gradual increase
    println!("Example 4: Temperature gradual increase");
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

    // Example 5: Empty batch (no changes)
    println!("Example 5: Empty batch (no notifications)");
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

    // Example 6: Fibonacci sequence generation
    println!("Example 6: Fibonacci sequence generation");
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
