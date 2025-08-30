#[test]
fn test_integration_tests() {
    let t = trybuild::TestCases::new();

    // Tests that should compile successfully
    t.pass("tests/integration/*.rs");

    // Tests that should fail to compile (UI tests)
    t.compile_fail("tests/ui/*.rs");
}

#[test]
fn test_derive_observable_with_generics() {
    use observable_property::{Observable, ObservableProperty};
    use std::marker::PhantomData;

    #[derive(Observable)]
    struct GenericStruct<T: Clone + Send + Sync + 'static> {
        #[observable]
        value: ObservableProperty<T>,
        phantom: PhantomData<T>,
    }

    let instance = GenericStruct::<i32>::new(42, PhantomData);
    assert_eq!(instance.get_value().unwrap(), 42);
    instance.set_value(100).unwrap();
    assert_eq!(instance.get_value().unwrap(), 100);
}

#[test]
fn test_observable_with_complex_expressions() {
    use observable_property::observable;

    // Test with function calls
    let len_obs = observable!("hello".len());
    assert_eq!(len_obs.get().unwrap(), 5);

    // Test with arithmetic
    let calc_obs = observable!(10 + 20 * 2);
    assert_eq!(calc_obs.get().unwrap(), 50);

    // Test with method chains
    let chain_obs = observable!(vec![1, 2, 3].into_iter().sum::<i32>());
    assert_eq!(chain_obs.get().unwrap(), 6);
}

#[test]
fn test_multiple_observable_structs() {
    use observable_property::{Observable, ObservableProperty};
    use std::sync::Arc;

    // When using #[observable], we need to use ObservableProperty<T> explicitly
    #[derive(Observable)]
    pub struct User {
        #[observable]
        name: ObservableProperty<String>,
        #[observable]
        email: ObservableProperty<String>,
    }

    #[derive(Observable)]
    pub struct Settings {
        #[observable]
        theme: ObservableProperty<String>,
        #[observable]
        notifications: ObservableProperty<bool>,
    }

    // Create instances using the generated constructor
    let user = User::new("Alice".to_string(), "alice@example.com".to_string());
    let settings = Settings::new("dark".to_string(), true);

    // Test that both structs work as expected
    assert_eq!(user.get_name().unwrap(), "Alice");
    assert_eq!(settings.get_theme().unwrap(), "dark");

    // Test subscription
    let _user_id = user.subscribe_name(Arc::new(|old, new| {
        println!("Name changed: {} -> {}", old, new);
    })).unwrap();

    // Test setting values
    user.set_name("Bob".to_string()).unwrap();
    settings.set_theme("light".to_string()).unwrap();

    assert_eq!(user.get_name().unwrap(), "Bob");
    assert_eq!(settings.get_theme().unwrap(), "light");
}
