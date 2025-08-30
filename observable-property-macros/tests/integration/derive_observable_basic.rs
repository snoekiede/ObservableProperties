use observable_property::{Observable, observable, ObservableProperty};
use std::sync::Arc;

#[derive(Observable)]
struct Person {
    #[observable]
    name: ObservableProperty<String>,        // Changed to direct type instead of ObservableProperty<String>
    #[observable]
    age: ObservableProperty<i32>,            // Changed to direct type instead of ObservableProperty<i32>
    id: u64,
}

fn main() {
    let person = Person::new("Alice".to_string(), 25, 12345);

    // Test getter methods
    assert_eq!(person.get_name().unwrap(), "Alice");
    assert_eq!(person.get_age().unwrap(), 25);
    assert_eq!(person.id, 12345);

    // Test setter methods
    person.set_name("Bob".to_string()).unwrap();
    person.set_age(30).unwrap();

    assert_eq!(person.get_name().unwrap(), "Bob");
    assert_eq!(person.get_age().unwrap(), 30);

    // Test subscription
    let name_id = person.subscribe_name(Arc::new(|old, new| {
        println!("Name changed: {} -> {}", old, new);
    })).unwrap();

    person.set_name("Charlie".to_string()).unwrap();

    // Test unsubscription
    person.unsubscribe_name(name_id).unwrap();
}
