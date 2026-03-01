use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Event Sourcing Demo ===\n");

    // Example 1: Basic event logging
    println!("1. Basic Event Logging:");
    println!("   Creating a counter with unlimited event log...");
    let counter = ObservableProperty::with_event_log(0, 0);

    counter.set(10)?;
    counter.set(20)?;
    counter.set(15)?;

    let events = counter.get_event_log();
    println!("   Total events recorded: {}", events.len());
    for event in &events {
        println!(
            "   Event #{}: {} -> {} (at {:?})",
            event.event_number,
            event.old_value,
            event.new_value,
            event.timestamp.elapsed()
        );
    }

    // Example 2: Bounded event log
    println!("\n2. Bounded Event Log (keeps only last 3 events):");
    let bounded = ObservableProperty::with_event_log(100, 3);

    for i in 1..=6 {
        bounded.set(100 + i * 10)?;
    }

    let events = bounded.get_event_log();
    println!("   Kept {} events (oldest was removed)", events.len());
    for event in &events {
        println!(
            "   Event #{}: {} -> {}",
            event.event_number, event.old_value, event.new_value
        );
    }

    // Example 3: Time-travel debugging
    println!("\n3. Time-Travel Debugging:");
    let config = ObservableProperty::with_event_log("default".to_string(), 0);

    let start = std::time::Instant::now();
    config.set("config_v1".to_string())?;
    thread::sleep(Duration::from_millis(50));
    config.set("config_v2".to_string())?;
    thread::sleep(Duration::from_millis(50));
    config.set("config_v3".to_string())?;

    println!("   Current config: {:?}", config.get()?);

    // Find what the config was 60ms after start
    let target_time = start + Duration::from_millis(60);
    let events = config.get_event_log();

    let mut state = "default".to_string();
    for event in &events {
        if event.timestamp <= target_time {
            state = event.new_value.clone();
        } else {
            break;
        }
    }
    println!("   Config at +60ms was: {}", state);

    // Example 4: Multi-threaded audit trail
    println!("\n4. Multi-threaded Audit Trail:");
    let shared = Arc::new(ObservableProperty::with_event_log(0, 0));

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let prop = shared.clone();
            thread::spawn(move || {
                for j in 0..2 {
                    let _ = prop.set(i * 100 + j * 10);
                    thread::sleep(Duration::from_millis(5));
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let events = shared.get_event_log();
    println!("   Total events from {} threads: {}", 3, events.len());
    for event in &events {
        println!(
            "   Event #{}: {} -> {} (thread: {})",
            event.event_number, event.old_value, event.new_value, event.thread_id
        );
    }

    // Example 5: Transaction replay
    println!("\n5. Transaction Replay:");
    let account = ObservableProperty::with_event_log(1000, 0);

    println!("   Initial balance: ${}", account.get()?);

    // Simulate transactions
    account.modify(|b| *b -= 100)?; // Withdrawal
    account.modify(|b| *b += 50)?; // Deposit
    account.modify(|b| *b -= 200)?; // Withdrawal
    account.modify(|b| *b += 300)?; // Deposit

    println!("   Final balance: ${}", account.get()?);

    println!("\n   Transaction History:");
    let events = account.get_event_log();
    for event in &events {
        let change = event.new_value as i32 - event.old_value as i32;
        let transaction_type = if change > 0 { "Deposit  " } else { "Withdrawal" };
        println!(
            "   [{}] {}: ${:4} (balance: ${} -> ${})",
            event.event_number,
            transaction_type,
            change.abs(),
            event.old_value,
            event.new_value
        );
    }

    // Example 6: Event log with observers
    println!("\n6. Event Log with Observers:");
    let monitored = ObservableProperty::with_event_log(0, 0);

    let _subscription = monitored.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   🔔 Observer notified: {} -> {}", old, new);
    }))?;

    monitored.set(5)?;
    monitored.set(10)?;

    println!("   Events and observer notifications are both recorded!");
    let events = monitored.get_event_log();
    println!("   Event log has {} entries", events.len());

    // Example 7: Analyzing patterns
    println!("\n7. Analyzing Event Patterns:");
    let score = ObservableProperty::with_event_log(0, 0);

    score.modify(|s| *s += 10)?;
    score.modify(|s| *s += 5)?;
    score.modify(|s| *s -= 3)?;
    score.modify(|s| *s += 8)?;
    score.modify(|s| *s -= 2)?;

    let events = score.get_event_log();
    let increases = events.iter().filter(|e| e.new_value > e.old_value).count();
    let decreases = events.iter().filter(|e| e.new_value < e.old_value).count();
    let total_gain: i32 = events
        .iter()
        .map(|e| e.new_value - e.old_value)
        .filter(|&change| change > 0)
        .sum();
    let total_loss: i32 = events
        .iter()
        .map(|e| e.old_value - e.new_value)
        .filter(|&change| change > 0)
        .sum();

    println!("   Score changes:");
    println!("   - Increases: {} (total: +{})", increases, total_gain);
    println!("   - Decreases: {} (total: -{})", decreases, total_loss);
    println!("   - Final score: {}", score.get()?);

    println!("\n=== Demo Complete ===");
    Ok(())
}
