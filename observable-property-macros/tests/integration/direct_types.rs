use observable_property::{Observable, ObservableProperty};
use std::sync::Arc;

// Example of using #[observable] with ObservableProperty<T>
#[derive(Observable)]
pub struct User {
    #[observable]
    name: ObservableProperty<String>,
    #[observable]
    age: ObservableProperty<i32>,

    // Regular field (not observable)
    id: u64,
}

// The main function will be called by the test
pub fn main() {
    let user = User::new("Alice".to_string(), 30, 12345);

    // Test getter methods
    assert_eq!(user.get_name().unwrap(), "Alice");
    assert_eq!(user.get_age().unwrap(), 30);
    assert_eq!(user.id, 12345);

    // Test setter methods
    user.set_name("Bob".to_string()).unwrap();
    user.set_age(40).unwrap();

    assert_eq!(user.get_name().unwrap(), "Bob");
    assert_eq!(user.get_age().unwrap(), 40);

    // Test subscription
    let name_id = user.subscribe_name(Arc::new(|old, new| {
        println!("Name changed: {} -> {}", old, new);
    })).unwrap();

    user.set_name("Charlie".to_string()).unwrap();

    // Test unsubscription
    user.unsubscribe_name(name_id).unwrap();

    // No need to return anything
}
