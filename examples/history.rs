use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Observable Property History Example ===\n");

    // Create a property with history tracking enabled (max 5 historical values)
    let counter = ObservableProperty::with_history(0, 5);

    // Subscribe to observe changes
    let _subscription = counter.subscribe_with_subscription(Arc::new(|old, new| {
        println!("  Value changed: {} -> {}", old, new);
    }))?;

    println!("Initial value: {}", counter.get()?);
    println!();

    // Make some changes
    println!("Making changes...");
    counter.set(10)?;
    counter.set(20)?;
    counter.set(30)?;
    counter.set(40)?;
    println!();

    // View history
    println!("Current value: {}", counter.get()?);
    println!("History: {:?}", counter.get_history());
    println!();

    // Undo operations
    println!("Performing undo operations...");
    counter.undo()?;
    println!("After 1st undo - Current: {}, History: {:?}", counter.get()?, counter.get_history());
    
    counter.undo()?;
    println!("After 2nd undo - Current: {}, History: {:?}", counter.get()?, counter.get_history());
    
    counter.undo()?;
    println!("After 3rd undo - Current: {}, History: {:?}", counter.get()?, counter.get_history());
    println!();

    // Make more changes after undo
    println!("Making new changes after undo...");
    counter.set(25)?;
    counter.set(35)?;
    println!("Current: {}, History: {:?}", counter.get()?, counter.get_history());
    println!();

    // Test history size limit
    println!("Testing history size limit (max 5 values)...");
    for i in 0..10 {
        counter.set(100 + i)?;
    }
    println!("After 10 changes:");
    println!("  Current: {}", counter.get()?);
    println!("  History (should have max 5 values): {:?}", counter.get_history());
    println!("  History length: {}", counter.get_history().len());
    println!();

    // Test undo with empty history
    println!("Testing undo with different scenarios...");
    while counter.get_history().len() > 0 {
        counter.undo()?;
    }
    println!("After undoing all history:");
    println!("  Current: {}", counter.get()?);
    println!("  History: {:?}", counter.get_history());
    
    // Try to undo when history is empty
    match counter.undo() {
        Ok(_) => println!("  Unexpected: undo succeeded"),
        Err(e) => println!("  Expected error: {}", e),
    }
    println!();

    // Demonstrate history with strings
    println!("=== History with String values ===");
    let document = ObservableProperty::with_history("Start".to_string(), 3);
    
    let _doc_subscription = document.subscribe_with_subscription(Arc::new(|old, new| {
        println!("  Document: '{}' -> '{}'", old, new);
    }))?;

    document.set("Chapter 1".to_string())?;
    document.set("Chapter 1 and 2".to_string())?;
    document.set("Complete Book".to_string())?;

    println!("Current: '{}'", document.get()?);
    println!("History: {:?}", document.get_history());
    println!();

    println!("Reverting changes...");
    document.undo()?;
    println!("After undo: '{}', History: {:?}", document.get()?, document.get_history());
    
    document.undo()?;
    println!("After undo: '{}', History: {:?}", document.get()?, document.get_history());
    println!();

    // Compare with regular property (no history)
    println!("=== Comparison: Regular property (no history) ===");
    let regular = ObservableProperty::new(100);
    regular.set(200)?;
    regular.set(300)?;
    
    println!("Regular property - Current: {}", regular.get()?);
    println!("Regular property - History: {:?} (should be empty)", regular.get_history());
    
    match regular.undo() {
        Ok(_) => println!("Unexpected: undo succeeded on regular property"),
        Err(e) => println!("Expected: {}", e),
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
