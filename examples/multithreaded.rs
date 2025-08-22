use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

// --- Person struct and impl (from basic.rs, adapted) ---
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
// --- End Person struct and impl ---

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Observable Property Multithreaded Person Example ===\n");

    let person = Arc::new(Person::new("Bob".to_string(), 30));

    // Subscribe to name changes
    let name_observer_id = person.name.subscribe(Arc::new(|old_name, new_name| {
        println!("📝 Name changed: '{}' → '{}'", old_name, new_name);
    }))?;

    // Subscribe to age changes
    let age_observer_id = person.age.subscribe(Arc::new(|old_age, new_age| {
        println!("🎂 Age changed: {} → {}", old_age, new_age);
    }))?;

    // Spawn threads to update name and age concurrently
    let person_clone1 = person.clone();
    let handle1 = thread::spawn(move || {
        for i in 0..3 {
            let new_name = format!("Bob #{}", i + 1);
            person_clone1.set_name(new_name).unwrap();
            thread::sleep(Duration::from_millis(50));
        }
    });

    let person_clone2 = person.clone();
    let handle2 = thread::spawn(move || {
        for _ in 0..3 {
            person_clone2.celebrate_birthday().unwrap();
            thread::sleep(Duration::from_millis(30));
        }
    });

    handle1.join().unwrap();
    handle2.join().unwrap();

    println!("\nFinal state:");
    println!("  Name: {}", person.get_name()?);
    println!("  Age: {}", person.get_age()?);

    // Cleanup observers
    person.name.unsubscribe(name_observer_id)?;
    person.age.unsubscribe(age_observer_id)?;

    println!("\n✅ Multithreaded Person example completed successfully!");
    Ok(())
}