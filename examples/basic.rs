//! Basic example demonstrating observable property usage
//!
//! Run with: cargo run --example basic

use observable_property::ObservableProperty;
use std::sync::Arc;

/// A simple person struct that demonstrates observable properties
#[derive(Clone, Debug)]
struct Person {
    name: ObservableProperty<String>,
    age: ObservableProperty<i32>,
}

impl Person {
    fn new(name: String, age: i32) -> Self {
        Self {
            name: ObservableProperty::new(name),
            age: ObservableProperty::new(age),
        }
    }

    fn get_name(&self) -> Result<String, observable_property::PropertyError> {
        self.name.get()
    }

    fn set_name(&self, new_name: String) -> Result<(), observable_property::PropertyError> {
        self.name.set(new_name)
    }

    fn get_age(&self) -> Result<i32, observable_property::PropertyError> {
        self.age.get()
    }

    fn set_age(&self, new_age: i32) -> Result<(), observable_property::PropertyError> {
        self.age.set(new_age)
    }

    fn celebrate_birthday(&self) -> Result<(), observable_property::PropertyError> {
        let current_age = self.age.get()?;
        self.age.set(current_age + 1)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Observable Property Basic Example ===\n");

    // Create a person
    let person = Person::new("Alice".to_string(), 25);

    // Subscribe to name changes
    let name_observer_id = person.name.subscribe(Arc::new(|old_name, new_name| {
        println!("📝 Name changed: '{}' → '{}'", old_name, new_name);
    }))?;

    // Subscribe to age changes
    let age_observer_id = person.age.subscribe(Arc::new(|old_age, new_age| {
        println!("🎂 Age changed: {} → {}", old_age, new_age);
    }))?;

    // Subscribe to only significant age changes (milestones)
    let milestone_observer_id = person.age.subscribe_filtered(
        Arc::new(|_old_age, new_age| {
            println!("🎉 Milestone reached! {} is now a special age: {}", 
                    if *new_age >= 18 { "Adult" } else { "Child" }, new_age);
        }),
        |old_age, new_age| {
            // Notify on milestone ages
            let milestones = [18, 21, 30, 40, 50, 65];
            milestones.contains(new_age) || 
            (milestones.contains(old_age) && !milestones.contains(new_age))
        }
    )?;

    println!("Initial state:");
    println!("  Name: {}", person.get_name()?);
    println!("  Age: {}\n", person.get_age()?);

    // Demonstrate property changes
    println!("Making changes...\n");

    // Change name
    person.set_name("Alice Johnson".to_string())?;

    // Age up a few times
    person.celebrate_birthday()?;
    person.celebrate_birthday()?;
    person.celebrate_birthday()?;

    // Change to milestone age
    person.set_age(21)?;

    // Direct property access
    println!("\nDirect property access:");
    let simple_property = ObservableProperty::new(100);
    
    simple_property.subscribe(Arc::new(|old, new| {
        println!("💰 Value changed: {} → {}", old, new);
    }))?;

    simple_property.set(200)?;
    simple_property.set(150)?;

    // Cleanup observers
    person.name.unsubscribe(name_observer_id)?;
    person.age.unsubscribe(age_observer_id)?;
    person.age.unsubscribe(milestone_observer_id)?;

    println!("\n✅ Example completed successfully!");
    Ok(())
}
