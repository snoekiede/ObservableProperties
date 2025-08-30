use observable_property::observable;

fn main() {
    // Test basic observable macro with different types
    let counter = observable!(0);
    let name = observable!("Alice".to_string());
    let flag = observable!(true);
    let data = observable!(vec![1, 2, 3]);

    // Test that they work as expected
    assert_eq!(counter.get().unwrap(), 0);
    assert_eq!(name.get().unwrap(), "Alice");
    assert_eq!(flag.get().unwrap(), true);
    assert_eq!(data.get().unwrap(), vec![1, 2, 3]);

    // Test setting values
    counter.set(42).unwrap();
    name.set("Bob".to_string()).unwrap();
    flag.set(false).unwrap();
    data.set(vec![4, 5, 6]).unwrap();

    assert_eq!(counter.get().unwrap(), 42);
    assert_eq!(name.get().unwrap(), "Bob");
    assert_eq!(flag.get().unwrap(), false);
    assert_eq!(data.get().unwrap(), vec![4, 5, 6]);
}
