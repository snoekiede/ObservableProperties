use observable_property::{computed, ObservableProperty};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Computed Properties Example ===\n");

    // Example 1: Simple math computation
    println!("1. Rectangle Area Computation:");
    let width = Arc::new(ObservableProperty::new(10));
    let height = Arc::new(ObservableProperty::new(5));

    let area = computed(vec![width.clone(), height.clone()], |values| {
        values[0] * values[1]
    })?;

    // Subscribe to changes in the computed property
    area.subscribe(Arc::new(|old, new| {
        println!("   Area changed: {} -> {}", old, new);
    }))?;

    println!("   Initial area: {}", area.get()?);
    
    width.set(20)?;
    thread::sleep(Duration::from_millis(10));
    println!("   After width=20: {}", area.get()?);

    height.set(8)?;
    thread::sleep(Duration::from_millis(10));
    println!("   After height=8: {}", area.get()?);

    // Example 2: String concatenation
    println!("\n2. Full Name Computation:");
    let first_name = Arc::new(ObservableProperty::new("John".to_string()));
    let last_name = Arc::new(ObservableProperty::new("Doe".to_string()));

    let full_name = computed(
        vec![first_name.clone(), last_name.clone()],
        |values| format!("{} {}", values[0], values[1]),
    )?;

    full_name.subscribe(Arc::new(|_old, new| {
        println!("   Full name updated: {}", new);
    }))?;

    println!("   Initial name: {}", full_name.get()?);
    
    first_name.set("Jane".to_string())?;
    thread::sleep(Duration::from_millis(10));

    last_name.set("Smith".to_string())?;
    thread::sleep(Duration::from_millis(10));

    // Example 3: Price with tax computation
    println!("\n3. Total Price with Tax:");
    let subtotal = Arc::new(ObservableProperty::new(100.0_f64));
    let tax_rate = Arc::new(ObservableProperty::new(0.08_f64)); // 8% tax

    let total = computed(
        vec![subtotal.clone(), tax_rate.clone()],
        |values| values[0] * (1.0 + values[1]),
    )?;

    total.subscribe(Arc::new(|old, new| {
        println!("   Total price changed: ${:.2} -> ${:.2}", old, new);
    }))?;

    println!("   Initial total: ${:.2}", total.get()?);
    
    subtotal.set(200.0)?;
    thread::sleep(Duration::from_millis(10));

    tax_rate.set(0.10)?; // Change to 10%
    thread::sleep(Duration::from_millis(10));

    // Example 4: Multiple dependencies
    println!("\n4. Expression: a + b * c");
    let a = Arc::new(ObservableProperty::new(1));
    let b = Arc::new(ObservableProperty::new(2));
    let c = Arc::new(ObservableProperty::new(3));

    let result = computed(
        vec![a.clone(), b.clone(), c.clone()],
        |values| values[0] + values[1] * values[2],
    )?;

    result.subscribe(Arc::new(|old, new| {
        println!("   Result changed: {} -> {}", old, new);
    }))?;

    println!("   Initial: 1 + 2 * 3 = {}", result.get()?);
    
    a.set(5)?;
    thread::sleep(Duration::from_millis(10));
    println!("   After a=5: 5 + 2 * 3 = {}", result.get()?);

    b.set(4)?;
    thread::sleep(Duration::from_millis(10));
    println!("   After b=4: 5 + 4 * 3 = {}", result.get()?);

    // Example 5: Chained computed properties
    println!("\n5. Temperature Conversions:");
    let celsius = Arc::new(ObservableProperty::new(0.0_f64));

    let fahrenheit = computed(vec![celsius.clone()], |values| {
        values[0] * 9.0 / 5.0 + 32.0
    })?;

    let kelvin = computed(vec![celsius.clone()], |values| values[0] + 273.15)?;

    celsius.subscribe(Arc::new(|_, new| {
        println!("   Celsius updated: {:.1}°C", new);
    }))?;

    fahrenheit.subscribe(Arc::new(|_, new| {
        println!("   Fahrenheit updated: {:.1}°F", new);
    }))?;

    kelvin.subscribe(Arc::new(|_, new| {
        println!("   Kelvin updated: {:.2}K", new);
    }))?;

    println!("   Initial: {:.1}°C = {:.1}°F = {:.2}K", 
             celsius.get()?, fahrenheit.get()?, kelvin.get()?);

    celsius.set(100.0)?;
    thread::sleep(Duration::from_millis(10));

    celsius.set(-40.0)?;
    thread::sleep(Duration::from_millis(10));

    println!("\n=== Example Complete ===");
    Ok(())
}
