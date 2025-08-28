use observable_property::{Observable, observable};
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Observable Property Macro Demo ===\n");

    // 1. Basic observable! macro usage
    println!("1. Using observable! macro:");
    let counter = observable!(0);
    let name = observable!("Alice".to_string());

    let counter_id = counter.subscribe(Arc::new(|old, new| {
        println!("   Counter changed: {} -> {}", old, new);
    }))?;

    let name_id = name.subscribe(Arc::new(|old, new| {
        println!("   Name changed: '{}' -> '{}'", old, new);
    }))?;

    counter.set(42)?;
    name.set("Bob".to_string())?;

    counter.unsubscribe(counter_id)?;
    name.unsubscribe(name_id)?;

    println!();

    // 2. Struct with Observable derive macro - NEW CLEAN API!
    println!("2. Using Observable derive macro:");

    #[derive(Observable)]
    struct Person {
        #[observable]
        name: observable_property::ObservableProperty<String>,
        #[observable]
        age: observable_property::ObservableProperty<i32>,
        #[observable]
        salary: observable_property::ObservableProperty<f64>,
        // Regular field (not observable)
        id: u64,
    }


    let person = Person::new(
        "Charlie".to_string(),  // ← Direct values!
        25,                     // ← Direct values!
        50000.0,               // ← Direct values!
        12345,                 // ← Regular field
    );

    // Subscribe to individual properties using generated methods
    let name_sub_id = person.subscribe_name(Arc::new(|old, new| {
        println!("   Person's name changed: '{}' -> '{}'", old, new);
    }))?;

    let age_sub_id = person.subscribe_age(Arc::new(|old, new| {
        println!("   Person's age changed: {} -> {}", old, new);
    }))?;

    let salary_sub_id = person.subscribe_salary_filtered(
        Arc::new(|old, new| {
            println!("   Salary increased: ${:.2} -> ${:.2}", old, new);
        }),
        |old, new| new > old
    )?;

    // Use generated setter methods
    person.set_name("David".to_string())?;
    person.set_age(26)?;
    person.set_salary(55000.0)?; // Will trigger filtered observer
    person.set_salary(54000.0)?; // Will NOT trigger filtered observer

    // Use generated getter methods
    println!("   Current name: {}", person.get_name()?);
    println!("   Current age: {}", person.get_age()?);
    println!("   Current salary: ${:.2}", person.get_salary()?);
    println!("   ID (regular field): {}", person.id);

    // Clean up
    person.unsubscribe_name(name_sub_id)?;
    person.unsubscribe_age(age_sub_id)?;
    person.unsubscribe_salary(salary_sub_id)?;

    println!();

    // 3. Complex observable structures
    println!("3. Complex observable structures:");

    #[derive(Observable)]
    struct Config {
        #[observable]
        debug_mode: observable_property::ObservableProperty<bool>,
        #[observable]
        max_connections: observable_property::ObservableProperty<usize>,
        #[observable]
        server_name: observable_property::ObservableProperty<String>,
        version: String, // Not observable
    }

    let config = Config::new(
        false,                          // debug_mode
        100,                           // max_connections
        "localhost".to_string(),       // server_name
        "1.0.0".to_string(),          // version (regular field)
    );

    let debug_id = config.subscribe_debug_mode(Arc::new(|old, new| {
        println!("   Debug mode: {} -> {}", old, new);
    }))?;

    let connections_id = config.subscribe_max_connections_filtered(
        Arc::new(|old, new| {
            println!("   Max connections increased: {} -> {}", old, new);
        }),
        |old, new| new > old
    )?;

    config.set_debug_mode(true)?;
    config.set_max_connections(150)?; // Will trigger filtered observer
    config.set_max_connections(120)?; // Will NOT trigger filtered observer
    config.set_server_name("production.example.com".to_string())?;

    println!("   Final config:");
    println!("     Debug: {}", config.get_debug_mode()?);
    println!("     Max connections: {}", config.get_max_connections()?);
    println!("     Server: {}", config.get_server_name()?);
    println!("     Version: {}", config.version);

    config.unsubscribe_debug_mode(debug_id)?;
    config.unsubscribe_max_connections(connections_id)?;

    println!("\n=== Demo Complete ===");
    Ok(())
}
