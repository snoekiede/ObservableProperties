// Integration tests for the Observable procedural macros
// These tests use the macros as intended, not by calling macro functions directly.

use observable_property::ObservableProperty;
use observable_property_macros::*;

fn main() {
    // Main function required for trybuild tests
    // The actual test logic is in the test functions
}

#[test]
fn test_observable_macro_simple_value() {
    let prop = observable!(42);
    assert_eq!(prop.get().unwrap(), 42);
    prop.set(100).unwrap();
    assert_eq!(prop.get().unwrap(), 100);
}

#[test]
fn test_observable_macro_string_value() {
    let prop = observable!("hello".to_string());
    assert_eq!(prop.get().unwrap(), "hello");
    prop.set("world".to_string()).unwrap();
    assert_eq!(prop.get().unwrap(), "world");
}

#[test]
fn test_derive_observable_simple_struct() {
    #[derive(Observable)]
    struct Person {
        #[observable]
        name: String, // Changed to direct type
        #[observable]
        age: i32,     // Changed to direct type
        id: u64,
    }

    let person = Person::new("Alice".to_string(), 30, 1);
    assert_eq!(person.get_name().unwrap(), "Alice");
    assert_eq!(person.get_age().unwrap(), 30);
    assert_eq!(person.id, 1);
    person.set_name("Bob".to_string()).unwrap();
    assert_eq!(person.get_name().unwrap(), "Bob");
}

#[test]
fn test_derive_observable_constructor_parameters() {
    #[derive(Observable)]
    struct Config {
        #[observable]
        debug: bool,      // Changed to direct type
        #[observable]
        count: usize,     // Changed to direct type
        version: String,
    }

    let config = Config::new(true, 42, "v1.0".to_string());
    assert_eq!(config.get_debug().unwrap(), true);
    assert_eq!(config.get_count().unwrap(), 42);
    assert_eq!(config.version, "v1.0");
}
