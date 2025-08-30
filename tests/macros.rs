// Use fully-qualified macro paths to invoke macros from the proc-macro crate without `use` imports.

#[test]
fn test_attribute_macro_wraps_fields() {
    #[observable_macros::observable]
    struct Person {
        pub name: String,
        pub age: usize,
    }

    let p = Person::new("Alice".to_string(), 30);

    // Verify fields are wrapped
    assert_eq!(p.name.get().unwrap(), "Alice".to_string());
    assert_eq!(p.age.get().unwrap(), 30usize);

    p.age.set(31).unwrap();
    assert_eq!(p.age.get().unwrap(), 31usize);
}

#[test]
fn test_derive_macro_creates_observable_struct_and_into() {
    #[derive(observable_macros::Observable)]
    struct RawPerson {
        pub name: String,
        pub age: usize,
    }

    let rp = RawPerson { name: "Bob".to_string(), age: 25 };

    // into_observable should be available and produce RawPersonObservable
    let obs = rp.into_observable();

    // RawPersonObservable has fields of ObservableProperty
    assert_eq!(obs.name.get().unwrap(), "Bob".to_string());
    assert_eq!(obs.age.get().unwrap(), 25usize);

    obs.age.set(26).unwrap();
    assert_eq!(obs.age.get().unwrap(), 26usize);
}

