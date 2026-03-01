//! Example demonstrating the debug logging feature with stack traces
//!
//! This example shows how to use the debug feature to track property changes
//! with full stack traces. This is invaluable for debugging unexpected
//! modifications in complex applications.
//!
//! To run this example:
//! ```bash
//! cargo run --example debug_logging --features debug
//! ```

#[cfg(feature = "debug")]
use observable_property::ObservableProperty;
#[cfg(feature = "debug")]
use std::{sync::Arc, thread};

#[cfg(feature = "debug")]
fn modify_value_from_function(property: &ObservableProperty<i32>, value: i32) {
    let old = property.get().expect("Failed to get value");
    property.set(value).expect("Failed to set value");
    property.log_change(&old, &value, "function call");
}

#[cfg(feature = "debug")]
fn main() {
    println!("=== Observable Property Debug Logging Demo ===\n");

    // Create a property and enable debug logging
    let property = ObservableProperty::new(0);
    property.enable_change_logging();

    println!("1. Making changes from the main thread...");
    let old = property.get().expect("Failed to get value");
    property.set(10).expect("Failed to set to 10");
    property.log_change(&old, &10, "main thread");
    
    let old = property.get().expect("Failed to get value");
    property.set(20).expect("Failed to set to 20");
    property.log_change(&old, &20, "main thread");
    
    println!("2. Making a change through a helper function...");
    modify_value_from_function(&property, 30);

    println!("3. Making changes from multiple threads...\n");
    let property_arc = Arc::new(property);
    let mut handles = vec![];

    for i in 0..3 {
        let prop = property_arc.clone();
        let handle = thread::spawn(move || {
            let old = prop.get().expect("Failed to get value");
            let new = 100 + i * 10;
            prop.set(new).expect("Failed to set in thread");
            prop.log_change(&old, &new, &format!("thread {}", i));
        });
        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Give threads time to complete
    thread::sleep(std::time::Duration::from_millis(50));

    println!("\n=== Change Logs with Stack Traces ===\n");
    property_arc.print_change_logs();

    println!("\n=== Programmatic Access to Logs ===");
    let logs = property_arc.get_change_logs();
    println!("Total changes logged: {}", logs.len());
    
    println!("\n=== Clearing Logs ===");
    property_arc.clear_change_logs();
    println!("Logs after clearing: {}", property_arc.get_change_logs().len());
    
    println!("\n=== Disabling Logging ===");
    property_arc.disable_change_logging();
    let old = property_arc.get().expect("Failed to get value");
    property_arc.set(999).expect("Failed to set after disabling");
    property_arc.log_change(&old, &999, "after disable"); // Won't be logged
    println!("Logs after disabling (should still be 0): {}", property_arc.get_change_logs().len());
}

#[cfg(not(feature = "debug"))]
fn main() {
    println!("This example requires the 'debug' feature to be enabled.");
    println!("Run with: cargo run --example debug_logging --features debug");
}
