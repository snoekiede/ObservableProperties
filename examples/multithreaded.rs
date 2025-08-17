//! Multithreaded example demonstrating thread-safe observable properties
//!
//! Run with: cargo run --example multithreaded

use observable_property::ObservableProperty;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::sync::atomic::{AtomicUsize, Ordering};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Observable Property Multithreaded Example ===\n");

    // Shared counter with multiple observers
    let counter = Arc::new(ObservableProperty::new(0i32));
    let notification_count = Arc::new(AtomicUsize::new(0));

    // Add observers that track notifications
    let count_clone = notification_count.clone();
    let observer1_id = counter.subscribe(Arc::new(move |old, new| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        println!("🔴 Observer 1: {} → {}", old, new);
    }))?;

    let count_clone = notification_count.clone();
    let observer2_id = counter.subscribe(Arc::new(move |old, new| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        println!("🔵 Observer 2: {} → {}", old, new);
    }))?;

    // Filtered observer for even numbers only
    let count_clone = notification_count.clone();
    let even_observer_id = counter.subscribe_filtered(
        Arc::new(move |old, new| {
            count_clone.fetch_add(1, Ordering::SeqCst);
            println!("🟢 Even Observer: {} → {} (new value is even!)", old, new);
        }),
        |_old, new| new % 2 == 0
    )?;

    println!("📊 Starting concurrent updates...\n");

    // Test 1: Multiple threads updating the counter
    {
        let num_threads = 4;
        let updates_per_thread = 5;
        let barrier = Arc::new(Barrier::new(num_threads));

        let handles: Vec<_> = (0..num_threads).map(|thread_id| {
            let counter_clone = counter.clone();
            let barrier_clone = barrier.clone();

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();

                for i in 0..updates_per_thread {
                    let new_value = (thread_id * 100 + i) as i32;
                    
                    if let Err(e) = counter_clone.set(new_value) {
                        eprintln!("Thread {} failed to set counter: {}", thread_id, e);
                    } else {
                        println!("🧵 Thread {} set counter to {}", thread_id, new_value);
                    }

                    // Small delay to see interleaving
                    thread::sleep(Duration::from_millis(10));
                }
            })
        }).collect();

        // Wait for all threads to complete
        for handle in handles {
            if let Err(e) = handle.join() {
                eprintln!("Thread panicked: {:?}", e);
            }
        }
    }

    thread::sleep(Duration::from_millis(100)); // Let notifications settle

    println!("\n📈 Current counter value: {}", counter.get()?);
    println!("🔔 Total notifications sent: {}\n", notification_count.load(Ordering::SeqCst));

    // Test 2: Performance comparison between sync and async notifications
    println!("⚡ Performance Test: Sync vs Async notifications\n");

    // Create a property with a slow observer
    let perf_counter = ObservableProperty::new(0i32);
    let slow_notifications = Arc::new(AtomicUsize::new(0));
    let slow_count_clone = slow_notifications.clone();

    perf_counter.subscribe(Arc::new(move |_old, _new| {
        // Simulate slow observer work
        thread::sleep(Duration::from_millis(20));
        slow_count_clone.fetch_add(1, Ordering::SeqCst);
    }))?;

    // Test synchronous updates (will block)
    let start = Instant::now();
    for i in 1..=3 {
        perf_counter.set(i)?;
    }
    let sync_duration = start.elapsed();
    println!("🐌 Sync updates (3 changes): {:?}", sync_duration);

    // Test asynchronous updates (won't block)
    let start = Instant::now();
    for i in 4..=6 {
        perf_counter.set_async(i)?;
    }
    let async_duration = start.elapsed();
    println!("🚀 Async updates (3 changes): {:?}", async_duration);

    println!("⚡ Speedup: {:.1}x faster", 
             sync_duration.as_secs_f64() / async_duration.as_secs_f64());

    // Wait for async observers to complete
    thread::sleep(Duration::from_millis(100));
    println!("🔔 Slow observer notifications: {}\n", slow_notifications.load(Ordering::SeqCst));

    // Test 3: Concurrent reads while updating
    println!("📚 Concurrent reads test...\n");
    {
        let read_counter = Arc::new(ObservableProperty::new(1000i32));
        let read_count = Arc::new(AtomicUsize::new(0));
        let num_readers = 8;
        let reads_per_reader = 50;

        // Start reader threads
        let reader_handles: Vec<_> = (0..num_readers).map(|reader_id| {
            let counter_clone = read_counter.clone();
            let count_clone = read_count.clone();

            thread::spawn(move || {
                for _ in 0..reads_per_reader {
                    match counter_clone.get() {
                        Ok(value) => {
                            count_clone.fetch_add(1, Ordering::SeqCst);
                            if value % 100 == 0 {
                                println!("📖 Reader {} saw value: {}", reader_id, value);
                            }
                        }
                        Err(e) => eprintln!("Reader {} error: {}", reader_id, e),
                    }
                    thread::sleep(Duration::from_millis(1));
                }
            })
        }).collect();

        // Start writer thread
        let writer_handle = {
            let counter_clone = read_counter.clone();
            thread::spawn(move || {
                for i in 1..=20 {
                    let new_value = 1000 + i * 50;
                    if let Err(e) = counter_clone.set(new_value) {
                        eprintln!("Writer error: {}", e);
                    } else {
                        println!("✍️  Writer set value to: {}", new_value);
                    }
                    thread::sleep(Duration::from_millis(25));
                }
            })
        };

        // Wait for all threads
        for handle in reader_handles {
            if let Err(e) = handle.join() {
                eprintln!("Reader thread panicked: {:?}", e);
            }
        }

        if let Err(e) = writer_handle.join() {
            eprintln!("Writer thread panicked: {:?}", e);
        }

        println!("📊 Total successful reads: {}", read_count.load(Ordering::SeqCst));
    }

    // Cleanup
    counter.unsubscribe(observer1_id)?;
    counter.unsubscribe(observer2_id)?;
    counter.unsubscribe(even_observer_id)?;

    println!("\n✅ Multithreaded example completed successfully!");
    Ok(())
}
