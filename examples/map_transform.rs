//! Example demonstrating the `map()` transformation feature
//!
//! This example shows how to create derived properties that automatically
//! update when the source property changes, enabling functional reactive
//! programming patterns.
//!
//! Run with: cargo run --example map_transform

use observable_property::ObservableProperty;
use std::sync::Arc;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Map Transformation Examples ===\n");

    // Example 1: Temperature conversion (Celsius to Fahrenheit)
    println!("1. Temperature Conversion (Celsius → Fahrenheit)");
    let celsius = ObservableProperty::new(20.0);
    let fahrenheit = celsius.map(|c| c * 9.0 / 5.0 + 32.0)?;

    println!("   Initial: {}°C = {}°F", celsius.get()?, fahrenheit.get()?);

    // Subscribe to observe changes
    let _f_sub = fahrenheit.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   Fahrenheit changed: {:.1}°F → {:.1}°F", old, new);
    }))?;

    celsius.set(25.0)?;
    println!("   Current: {}°C = {}°F", celsius.get()?, fahrenheit.get()?);

    celsius.set(0.0)?;
    println!("   Current: {}°C = {}°F", celsius.get()?, fahrenheit.get()?);

    celsius.set(100.0)?;
    println!("   Current: {}°C = {}°F\n", celsius.get()?, fahrenheit.get()?);

    // Example 2: Chained transformations
    println!("2. Chained Transformations (base → doubled → squared → formatted)");
    let base = ObservableProperty::new(5);
    let doubled = base.map(|x| x * 2)?;
    let squared = doubled.map(|x| x * x)?;
    let formatted = squared.map(|x| format!("Result: {}", x))?;

    println!("   Base: {} → {}", base.get()?, formatted.get()?);

    base.set(10)?;
    println!("   Base: {} → {}\n", base.get()?, formatted.get()?);

    // Example 3: Type conversions
    println!("3. Type Conversions");
    let integer = ObservableProperty::new(42);
    
    let float_value = integer.map(|i| *i as f64)?;
    let is_even = integer.map(|i| i % 2 == 0)?;
    let string_repr = integer.map(|i| format!("The number is {}", i))?;

    println!("   Integer: {}", integer.get()?);
    println!("   As float: {}", float_value.get()?);
    println!("   Is even: {}", is_even.get()?);
    println!("   As string: {}", string_repr.get()?);

    integer.set(43)?;
    println!("\n   Integer: {}", integer.get()?);
    println!("   As float: {}", float_value.get()?);
    println!("   Is even: {}", is_even.get()?);
    println!("   As string: {}\n", string_repr.get()?);

    // Example 4: Mathematical transformations
    println!("4. Mathematical Transformations (Radius → Area & Circumference)");
    let radius = ObservableProperty::new(5.0);
    let area = radius.map(|r| std::f64::consts::PI * r * r)?;
    let circumference = radius.map(|r| 2.0 * std::f64::consts::PI * r)?;

    println!("   Radius: {:.2} units", radius.get()?);
    println!("   Area: {:.2} square units", area.get()?);
    println!("   Circumference: {:.2} units", circumference.get()?);

    radius.set(10.0)?;
    println!("\n   Radius: {:.2} units", radius.get()?);
    println!("   Area: {:.2} square units", area.get()?);
    println!("   Circumference: {:.2} units\n", circumference.get()?);

    // Example 5: Working with complex objects
    println!("5. Complex Object Transformations");

    #[derive(Clone, Debug)]
    struct User {
        first_name: String,
        last_name: String,
        age: u32,
    }

    let user = ObservableProperty::new(User {
        first_name: "John".to_string(),
        last_name: "Doe".to_string(),
        age: 30,
    });

    let full_name = user.map(|u| format!("{} {}", u.first_name, u.last_name))?;
    let is_adult = user.map(|u| u.age >= 18)?;
    let greeting = user.map(|u| format!("Hello, {} {}!", u.first_name, u.last_name))?;

    println!("   User: {:?}", user.get()?);
    println!("   Full name: {}", full_name.get()?);
    println!("   Is adult: {}", is_adult.get()?);
    println!("   Greeting: {}", greeting.get()?);

    user.modify(|u| {
        u.first_name = "Jane".to_string();
        u.age = 16;
    })?;

    println!("\n   User: {:?}", user.get()?);
    println!("   Full name: {}", full_name.get()?);
    println!("   Is adult: {}", is_adult.get()?);
    println!("   Greeting: {}\n", greeting.get()?);

    // Example 6: String operations
    println!("6. String Operations");
    let text = ObservableProperty::new("hello world".to_string());
    
    let uppercase = text.map(|s| s.to_uppercase())?;
    let word_count = text.map(|s| s.split_whitespace().count())?;
    let char_count = text.map(|s| s.len())?;
    let reversed = text.map(|s| s.chars().rev().collect::<String>())?;

    println!("   Original: {}", text.get()?);
    println!("   Uppercase: {}", uppercase.get()?);
    println!("   Word count: {}", word_count.get()?);
    println!("   Char count: {}", char_count.get()?);
    println!("   Reversed: {}", reversed.get()?);

    text.set("Rust is awesome!".to_string())?;
    println!("\n   Original: {}", text.get()?);
    println!("   Uppercase: {}", uppercase.get()?);
    println!("   Word count: {}", word_count.get()?);
    println!("   Char count: {}", char_count.get()?);
    println!("   Reversed: {}\n", reversed.get()?);

    // Example 7: Options and error handling
    println!("7. Working with Options");
    let optional = ObservableProperty::new(Some(42));
    
    let value_or_zero = optional.map(|opt| opt.unwrap_or(0))?;
    let is_some = optional.map(|opt| opt.is_some())?;
    let doubled_if_some = optional.map(|opt| opt.map(|v| v * 2))?;

    println!("   Optional: {:?}", optional.get()?);
    println!("   Value or zero: {}", value_or_zero.get()?);
    println!("   Is some: {}", is_some.get()?);
    println!("   Doubled if some: {:?}", doubled_if_some.get()?);

    optional.set(None)?;
    println!("\n   Optional: {:?}", optional.get()?);
    println!("   Value or zero: {}", value_or_zero.get()?);
    println!("   Is some: {}", is_some.get()?);
    println!("   Doubled if some: {:?}\n", doubled_if_some.get()?);

    // Example 8: Collections
    println!("8. Collection Transformations");
    let numbers = ObservableProperty::new(vec![1, 2, 3, 4, 5]);
    
    let sum = numbers.map(|v| v.iter().sum::<i32>())?;
    let count = numbers.map(|v| v.len())?;
    let first = numbers.map(|v| v.first().copied())?;
    let evens = numbers.map(|v| v.iter().filter(|&&x| x % 2 == 0).count())?;

    println!("   Numbers: {:?}", numbers.get()?);
    println!("   Sum: {}", sum.get()?);
    println!("   Count: {}", count.get()?);
    println!("   First: {:?}", first.get()?);
    println!("   Even count: {}", evens.get()?);

    numbers.set(vec![10, 20, 30])?;
    println!("\n   Numbers: {:?}", numbers.get()?);
    println!("   Sum: {}", sum.get()?);
    println!("   Count: {}", count.get()?);
    println!("   First: {:?}", first.get()?);
    println!("   Even count: {}\n", evens.get()?);

    // Example 9: Observer on derived property
    println!("9. Observers on Derived Properties");
    let distance = ObservableProperty::new(100.0); // meters
    let distance_km = distance.map(|m| m / 1000.0)?;

    let _sub = distance_km.subscribe_with_subscription(Arc::new(|old, new| {
        println!("   Distance changed: {:.2} km → {:.2} km", old, new);
    }))?;

    distance.set(500.0)?;
    distance.set(2500.0)?;
    distance.set(5000.0)?;
    println!();

    // Example 10: Multiple maps from same source
    println!("10. Multiple Derived Properties from Same Source");
    let price = ObservableProperty::new(100.0_f64);
    
    let with_tax = price.map(|p| p * 1.20)?;  // 20% tax
    let discounted = price.map(|p| p * 0.80)?;  // 20% discount
    let rounded = price.map(|p| (*p).round())?;

    println!("   Base price: ${:.2}", price.get()?);
    println!("   With tax: ${:.2}", with_tax.get()?);
    println!("   Discounted: ${:.2}", discounted.get()?);
    println!("   Rounded: ${:.2}", rounded.get()?);

    price.set(99.99)?;
    println!("\n   Base price: ${:.2}", price.get()?);
    println!("   With tax: ${:.2}", with_tax.get()?);
    println!("   Discounted: ${:.2}", discounted.get()?);
    println!("   Rounded: ${:.2}", rounded.get()?);

    println!("\n=== All Map Examples Complete ===");
    
    Ok(())
}
