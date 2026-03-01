//! Example demonstrating async stream and wait_for functionality (std-only)
//!
//! This example showcases the async features of ObservableProperty using only
//! standard library primitives (no external async runtime required):
//! - Converting a property to an async stream
//! - Waiting for specific conditions with wait_for
//!
//! Run with: cargo run --example async_stream --features async
//!
//! Note: This example requires the `async` feature to be enabled.
//! The implementation uses only std library primitives and does not depend
//! on external async runtimes like tokio.

use observable_property::ObservableProperty;
use std::sync::Arc;
use std::time::Duration;
use std::pin::Pin;
use std::task::{Context, Poll, Wake, Waker};
use std::future::Future;

#[cfg(not(feature = "async"))]
fn main() {
    println!("This example requires the 'async' feature to be enabled.");
    println!("Run with: cargo run --example async_stream --features async");
}

#[cfg(feature = "async")]
fn main() {
    println!("=== Observable Property Async Stream Example (std-only) ===\n");

    // Example 1: wait_for demonstration
    println!("Example 1: Waiting for specific condition");
    println!("------------------------------------------");
    
    let counter = Arc::new(ObservableProperty::new(0));
    
    // Spawn a thread to increment the counter
    let counter_clone = counter.clone();
    let handle = std::thread::spawn(move || {
        for i in 1..=10 {
            std::thread::sleep(Duration::from_millis(100));
            counter_clone.set(i).ok();
            println!("  Counter set to: {}", i);
        }
    });

    // Create the future
    let future = counter.wait_for(|value| *value >= 5);
    
    // Manual polling loop (in a real application, you'd use an async runtime)
    let result = block_on(future);
    println!("  ✓ Counter reached: {}\n", result);

    handle.join().expect("Thread panicked");

    // Example 2: Temperature monitoring
    println!("Example 2: Temperature threshold monitoring");
    println!("-------------------------------------------");
    
    let temperature = Arc::new(ObservableProperty::new(20.0));
    
    // Spawn a thread to simulate temperature changes
    let temp_clone = temperature.clone();
    let handle2 = std::thread::spawn(move || {
        for i in 0..15 {
            std::thread::sleep(Duration::from_millis(50));
            let new_temp = 20.0 + i as f64 * 0.8;
            temp_clone.set(new_temp).ok();
        }
    });

    // Wait for critical temperature
    println!("  Waiting for temperature > 28°C...");
    let future = temperature.wait_for(|temp| *temp > 28.0);
    let critical = block_on(future);
    println!("  ✓ Critical temperature reached: {:.1}°C\n", critical);

    handle2.join().expect("Thread panicked");

    // Example 3: Status monitoring
    println!("Example 3: Waiting for ready status");
    println!("------------------------------------");
    
    let status = Arc::new(ObservableProperty::new("initializing".to_string()));
    
    // Spawn a thread to change status
    let status_clone = status.clone();
    let handle3 = std::thread::spawn(move || {
        let states = vec!["loading", "processing", "validating", "ready"];
        
        for state in states {
            std::thread::sleep(Duration::from_millis(200));
            println!("  Status: {}", state);
            status_clone.set(state.to_string()).ok();
        }
    });

    // Wait for ready state
    println!("  Waiting for 'ready' status...");
    let future = status.wait_for(|s| s == "ready");
    let ready_status = block_on(future);
    println!("  ✓ Status achieved: {}\n", ready_status);

    handle3.join().expect("Thread panicked");

    println!("=== All Examples Completed Successfully ===\n");
    println!("Note: This implementation uses only std library primitives.");
    println!("For a full async ecosystem experience, consider using futures-rs or tokio.");
}

/// A simple block_on implementation for running futures to completion
///
/// This is a minimal executor for demonstration purposes. In production,
/// you should use a proper async runtime like tokio or async-std.
#[cfg(feature = "async")]
fn block_on<F: Future>(mut future: F) -> F::Output {
    use std::sync::Arc;
    use std::thread;

    let mut future = unsafe { Pin::new_unchecked(&mut future) };
    
    // Create a waker that does nothing (busy-wait polling)
    struct DummyWaker;
    impl Wake for DummyWaker {
        fn wake(self: Arc<Self>) {}
    }
    
    let waker = Waker::from(Arc::new(DummyWaker));
    let mut context = Context::from_waker(&waker);

    // Poll in a loop until ready
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                // Busy wait with a small sleep to avoid CPU spinning
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
}
