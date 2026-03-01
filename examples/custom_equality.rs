//! Example demonstrating custom equality comparison for ObservableProperty
//!
//! This example shows various use cases where custom equality functions are useful:
//! 1. Float comparison with epsilon tolerance
//! 2. Case-insensitive string comparison
//! 3. Semantic equality for complex types
//!
//! Run with: cargo run --example custom_equality

use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Custom Equality Comparison Examples ===\n");

    // Example 1: Float comparison with epsilon tolerance
    println!("1. Float Comparison with Epsilon");
    println!("   (Only notifies if difference > 0.001)\n");
    
    let temperature = ObservableProperty::with_equality(
        20.0_f64,
        |a, b| (a - b).abs() < 0.001
    );

    let _temp_sub = temperature.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   🔥 Temperature changed: {:.4}°C -> {:.4}°C", old, new);
    }))?;

    println!("   Setting to 20.0005 (within epsilon):");
    temperature.set(20.0005)?;
    println!("   ✓ No notification (within threshold)\n");

    println!("   Setting to 20.5 (outside epsilon):");
    temperature.set(20.5)?;
    
    println!("   Setting to 20.50001 (within epsilon):");
    temperature.set(20.50001)?;
    println!("   ✓ No notification (within threshold)\n");

    println!("   Setting to 22.0 (outside epsilon):");
    temperature.set(22.0)?;
    println!();

    // Example 2: Case-insensitive string comparison
    println!("2. Case-Insensitive String Comparison\n");
    
    let username = ObservableProperty::with_equality(
        "Alice".to_string(),
        |a, b| a.to_lowercase() == b.to_lowercase()
    );

    let _user_sub = username.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   👤 Username changed: '{}' -> '{}'", old, new);
    }))?;

    println!("   Setting to 'alice' (same case-insensitive):");
    username.set("alice".to_string())?;
    println!("   ✓ No notification\n");

    println!("   Setting to 'ALICE' (same case-insensitive):");
    username.set("ALICE".to_string())?;
    println!("   ✓ No notification\n");

    println!("   Setting to 'Bob' (different):");
    username.set("Bob".to_string())?;
    
    println!("   Setting to 'bob' (same case-insensitive):");
    username.set("bob".to_string())?;
    println!("   ✓ No notification\n");

    println!("   Setting to 'Charlie' (different):");
    username.set("Charlie".to_string())?;
    println!();

    // Example 3: Semantic equality for complex types
    println!("3. Semantic Equality for Complex Types");
    println!("   (Ignoring timeout_ms field)\n");

    #[derive(Clone, Debug)]
    struct ServerConfig {
        host: String,
        port: u16,
        timeout_ms: u64,
    }

    let config = ObservableProperty::with_equality(
        ServerConfig {
            host: "localhost".to_string(),
            port: 8080,
            timeout_ms: 1000,
        },
        |a, b| {
            // Only compare host and port, ignore timeout
            a.host == b.host && a.port == b.port
        }
    );

    let _config_sub = config.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   ⚙️  Critical config changed:");
        println!("      Old: {}:{} (timeout: {}ms)", old.host, old.port, old.timeout_ms);
        println!("      New: {}:{} (timeout: {}ms)", new.host, new.port, new.timeout_ms);
    }))?;

    println!("   Changing timeout to 2000ms:");
    config.modify(|c| c.timeout_ms = 2000)?;
    println!("   ✓ No notification (timeout is ignored)\n");

    println!("   Changing timeout to 5000ms:");
    config.modify(|c| c.timeout_ms = 5000)?;
    println!("   ✓ No notification (timeout is ignored)\n");

    println!("   Changing port to 9090:");
    config.modify(|c| c.port = 9090)?;
    
    println!("   Changing timeout to 10000ms:");
    config.modify(|c| c.timeout_ms = 10000)?;
    println!("   ✓ No notification (timeout is ignored)\n");

    println!("   Changing host to '192.168.1.1':");
    config.modify(|c| c.host = "192.168.1.1".to_string())?;
    
    // Example 4: Percentage change threshold
    println!("\n4. Percentage Change Threshold");
    println!("   (Only notifies if change > 5%)\n");

    let stock_price = ObservableProperty::with_equality(
        100.0_f64,
        |a, b| {
            let change_percent = ((b - a).abs() / a) * 100.0;
            change_percent < 5.0
        }
    );

    let _stock_sub = stock_price.subscribe_with_subscription(Arc::new(|old, new| {
        let change_percent = ((new - old) / old) * 100.0;
        println!("   📈 Significant price change: ${:.2} -> ${:.2} ({:+.2}%)", 
                 old, new, change_percent);
    }))?;

    println!("   Setting to $102 (2% change):");
    stock_price.set(102.0)?;
    println!("   ✓ No notification (< 5% threshold)\n");

    println!("   Setting to $104 (4% change from original $100):");
    stock_price.set(104.0)?;
    println!("   ✓ No notification (< 5% threshold)\n");

    println!("   Setting to $106 (6% change from original $100):");
    stock_price.set(106.0)?;

    println!("   Setting to $108 (~1.9% change from $106):");
    stock_price.set(108.0)?;
    println!("   ✓ No notification (< 5% threshold)\n");

    println!("   Setting to $120 (11% change from $108):");
    stock_price.set(120.0)?;

    println!("\n=== All Examples Complete ===");
    
    Ok(())
}
