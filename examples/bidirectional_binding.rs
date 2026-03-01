use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Bidirectional Binding Example ===\n");

    // Example 1: Basic Model-View Synchronization
    println!("Example 1: Basic Model-View Binding");
    println!("-------------------------------------");
    
    let model_value = ObservableProperty::new(0);
    let view_value = ObservableProperty::new(0);

    // Add observers to see synchronization in action
    model_value.subscribe(Arc::new(|old, new| {
        println!("Model changed: {} -> {}", old, new);
    }))?;

    view_value.subscribe(Arc::new(|old, new| {
        println!("View changed: {} -> {}", old, new);
    }))?;

    // Establish bidirectional binding
    model_value.bind_bidirectional(&view_value)?;

    println!("\nUpdating model to 42:");
    model_value.set(42)?;
    println!("Model value: {}", model_value.get()?);
    println!("View value: {}", view_value.get()?);

    println!("\nUpdating view to 100:");
    view_value.set(100)?;
    println!("Model value: {}", model_value.get()?);
    println!("View value: {}", view_value.get()?);

    // Example 2: String Synchronization (Form Input)
    println!("\n\nExample 2: Form Input Binding");
    println!("------------------------------");
    
    let username_model = ObservableProperty::new("".to_string());
    let username_input = ObservableProperty::new("".to_string());

    // Add validation observer
    username_model.subscribe(Arc::new(|_old, new| {
        if new.len() >= 3 {
            println!("✓ Valid username: '{}'", new);
        } else if !new.is_empty() {
            println!("✗ Username too short: '{}'", new);
        }
    }))?;

    username_model.bind_bidirectional(&username_input)?;

    println!("\nUser types 'a':");
    username_input.set("a".to_string())?;

    println!("\nUser types 'al':");
    username_input.set("al".to_string())?;

    println!("\nUser types 'alice':");
    username_input.set("alice".to_string())?;

    println!("\nProgrammatic update from model:");
    username_model.set("bob".to_string())?;
    println!("Input field shows: '{}'", username_input.get()?);

    // Example 3: Multiple Control Synchronization
    println!("\n\nExample 3: Multiple Control Binding");
    println!("------------------------------------");
    
    let slider = ObservableProperty::new(50);
    let number_input = ObservableProperty::new(50);
    let display_label = ObservableProperty::new(50);

    // Add observer to show updates
    let update_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&update_count);
    display_label.subscribe(Arc::new(move |_old, new| {
        let count = counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        println!("Display updated #{}: {}", count, new);
    }))?;

    // Bind all controls together
    slider.bind_bidirectional(&number_input)?;
    slider.bind_bidirectional(&display_label)?;

    println!("\nUser moves slider to 75:");
    slider.set(75)?;

    println!("\nUser types 25 in number input:");
    number_input.set(25)?;

    println!("\nAll values are synchronized:");
    println!("  Slider: {}", slider.get()?);
    println!("  Number Input: {}", number_input.get()?);
    println!("  Display Label: {}", display_label.get()?);

    // Example 4: No Infinite Loop Demonstration
    println!("\n\nExample 4: Loop Prevention");
    println!("--------------------------");
    
    let prop_a = ObservableProperty::new(10);
    let prop_b = ObservableProperty::new(10);

    let update_a = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let update_b = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let counter_a = Arc::clone(&update_a);
    let counter_b = Arc::clone(&update_b);

    prop_a.subscribe(Arc::new(move |_old, new| {
        let count = counter_a.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        println!("Property A observer called #{}: {}", count, new);
    }))?;

    prop_b.subscribe(Arc::new(move |_old, new| {
        let count = counter_b.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        println!("Property B observer called #{}: {}", count, new);
    }))?;

    prop_a.bind_bidirectional(&prop_b)?;

    println!("\nSetting prop_a to 20:");
    prop_a.set(20)?;
    println!("Update counts - A: {}, B: {}", 
             update_a.load(std::sync::atomic::Ordering::SeqCst),
             update_b.load(std::sync::atomic::Ordering::SeqCst));

    println!("\nSetting prop_b to 20 again (same value):");
    prop_b.set(20)?;
    println!("Update counts - A: {}, B: {} (no additional updates!)", 
             update_a.load(std::sync::atomic::Ordering::SeqCst),
             update_b.load(std::sync::atomic::Ordering::SeqCst));

    println!("\n=== All examples completed successfully! ===");
    Ok(())
}
