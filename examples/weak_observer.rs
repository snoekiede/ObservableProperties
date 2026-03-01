use observable_property::ObservableProperty;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() -> Result<(), observable_property::PropertyError> {
    println!("=== Weak Observer Example ===\n");

    // Example 1: Basic weak observer with automatic cleanup
    println!("Example 1: Basic weak observer");
    {
        let property = ObservableProperty::new(0);

        {
            // Create an observer Arc as a trait object
            let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(|old: &i32, new: &i32| {
                println!("  Observer called: {} -> {}", old, new);
            });

            // Subscribe with a weak reference
            let _id = property.subscribe_weak(Arc::downgrade(&observer))?;

            property.set(10)?; // Observer is active
            property.set(20)?; // Observer is still active

            // When observer Arc goes out of scope here, weak reference becomes invalid
        }

        println!("  Observer Arc dropped, next set will clean it up:");
        property.set(30)?; // No output - observer was automatically cleaned up
        println!("  (No observer called - it was automatically cleaned up)\n");
    }

    // Example 2: Compare with strong observer (for contrast)
    println!("Example 2: Strong observer (for comparison)");
    {
        let property = ObservableProperty::new(0);

        {
            let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(|old: &i32, new: &i32| {
                println!("  Strong observer: {} -> {}", old, new);
            });

            let _id = property.subscribe(observer)?;

            property.set(10)?; // Observer is active
            property.set(20)?; // Observer is still active
        }

        property.set(30)?; // Strong observer still active!
        println!("  (Strong observer keeps working even after Arc goes out of scope)\n");
    }

    // Example 3: Multiple weak observers with independent lifetimes
    println!("Example 3: Multiple weak observers with independent lifetimes");
    {
        let property = ObservableProperty::new(String::from("start"));

        let observer1: Arc<dyn Fn(&String, &String) + Send + Sync> = Arc::new(|old: &String, new: &String| {
            println!("  Observer 1: '{}' -> '{}'", old, new);
        });

        let observer2: Arc<dyn Fn(&String, &String) + Send + Sync> = Arc::new(|old: &String, new: &String| {
            println!("  Observer 2: '{}' -> '{}'", old, new);
        });

        let observer3: Arc<dyn Fn(&String, &String) + Send + Sync> = Arc::new(|old: &String, new: &String| {
            println!("  Observer 3: '{}' -> '{}'", old, new);
        });

        property.subscribe_weak(Arc::downgrade(&observer1))?;
        property.subscribe_weak(Arc::downgrade(&observer2))?;
        property.subscribe_weak(Arc::downgrade(&observer3))?;

        println!("  All three observers active:");
        property.set(String::from("step1"))?;

        // Drop observer2
        drop(observer2);

        println!("\n  Observer 2 dropped:");
        property.set(String::from("step2"))?;

        // Drop observer1
        drop(observer1);

        println!("\n  Observer 1 dropped:");
        property.set(String::from("step3"))?;

        // Drop observer3
        drop(observer3);

        println!("\n  Observer 3 dropped:");
        property.set(String::from("step4"))?;
        println!("  (No observers left)\n");
    }

    // Example 4: Weak observers in multi-threaded context
    println!("Example 4: Weak observers in multi-threaded context");
    {
        let property = Arc::new(ObservableProperty::new(0));
        let property_clone = property.clone();

        let observer: Arc<dyn Fn(&i32, &i32) + Send + Sync> = Arc::new(|old: &i32, new: &i32| {
            println!("  Thread observer: {} -> {}", old, new);
        });

        property.subscribe_weak(Arc::downgrade(&observer))?;

        // Spawn a thread that will modify the property
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            let _ = property_clone.set(42);
            thread::sleep(Duration::from_millis(50));
            let _ = property_clone.set(100);
        });

        // Keep the observer alive for a bit
        thread::sleep(Duration::from_millis(200));

        // Drop the observer
        drop(observer);

        // Give time for thread to finish
        handle.join().unwrap();

        // This set will clean up the dead observer
        property.set(200)?;
        println!("  (Observer was cleaned up after being dropped)\n");
    }

    // Example 5: Practical use case - conditional monitoring
    println!("Example 5: Conditional monitoring");
    {
        let temperature = ObservableProperty::new(20.0_f64);

        println!("  Initial temperature: 20.0°C");
        temperature.set(22.0)?;

        // Enable monitoring when temperature gets high
        if temperature.get()? > 25.0 {
            println!("  Temperature above 25°C - enabling monitor");
        } else {
            println!("  Temperature rising to 28.0°C - enabling monitor");
            temperature.set(28.0)?;

            let monitor: Arc<dyn Fn(&f64, &f64) + Send + Sync> = Arc::new(|old: &f64, new: &f64| {
                println!("  🔥 High temp alert: {:.1}°C -> {:.1}°C", old, new);
            });

            temperature.subscribe_weak(Arc::downgrade(&monitor))?;

            temperature.set(30.0)?;
            temperature.set(32.0)?;

            // Disable monitoring by dropping the monitor
            println!("\n  Cooling down - disabling monitor");
            drop(monitor);
        }

        temperature.set(25.0)?;
        println!("  (No alerts - monitor was automatically cleaned up)\n");
    }

    println!("=== Weak Observer Examples Complete ===");
    Ok(())
}
