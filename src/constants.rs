//! Constants used throughout the observable property library

/// Default maximum number of background notification workers per property
///
/// This constant sets the default background worker limit for `set_async()`. The limit
/// is shared across concurrent calls and clones of a property. When all worker slots
/// are occupied, new notifications run on the calling thread.
///
/// # Rationale
///
/// - **Resource Control**: Prevents unbounded thread creation that could exhaust system resources
/// - **Performance Balance**: Provides parallelism benefits without excessive context switching overhead  
/// - **Scalability**: Ensures consistent behavior regardless of the number of observers
/// - **System Responsiveness**: Limits thread contention on multi-core systems
///
/// # Implementation Details
///
/// When `set_async()` is called:
/// 1. All observers are collected into a snapshot
/// 2. Observers are divided into `MAX_THREADS` batches (or fewer if there are fewer observers)
/// 3. Each batch executes in its own background worker, up to the per-property limit
/// 4. Observers within each batch are executed sequentially
///
/// For example, with 100 observers and `MAX_THREADS = 4`:
/// - Batch 1: Observers 1-25 (Thread 1)
/// - Batch 2: Observers 26-50 (Thread 2)  
/// - Batch 3: Observers 51-75 (Thread 3)
/// - Batch 4: Observers 76-100 (Thread 4)
///
/// # Tuning Considerations
///
/// This value can be adjusted based on your application's needs:
/// - **CPU-bound observers**: Higher values may improve throughput on multi-core systems
/// - **I/O-bound observers**: Higher values can improve concurrency for network/disk operations
/// - **Memory-constrained systems**: Lower values reduce thread overhead
/// - **Real-time systems**: Lower values reduce scheduling unpredictability
///
/// # Thread Safety
///
/// This constant bounds background worker concurrency; it does not affect the
/// thread safety of the overall system.
pub const MAX_THREADS: usize = 4;

/// Maximum number of observers allowed per property instance
///
/// This limit prevents memory exhaustion from unbounded observer registration.
/// Once this limit is reached, attempts to add more observers will fail with
/// an `InvalidConfiguration` error.
///
/// # Rationale
///
/// - **Memory Protection**: Prevents unbounded memory growth from observer accumulation
/// - **Resource Management**: Ensures predictable memory usage in long-running applications
/// - **Early Detection**: Catches potential memory leaks from forgotten unsubscriptions
/// - **System Stability**: Prevents out-of-memory conditions in constrained environments
///
/// # Tuning Considerations
///
/// This value can be adjusted based on your application's needs:
/// - **High-frequency properties**: May need fewer observers to avoid notification overhead
/// - **Low-frequency properties**: Can safely support more observers
/// - **Memory-constrained systems**: Lower values prevent memory pressure
/// - **Development/testing**: Higher values may be useful for comprehensive test coverage
///
/// # Default Value
///
/// The default of 10,000 observers provides generous headroom for most applications while
/// still preventing pathological cases. In practice, most properties have fewer than 100
/// observers.
///
/// # Example Scenarios
///
/// - **User sessions**: 1,000 concurrent users, each with 5 properties = ~5,000 observers
/// - **IoT devices**: 5,000 devices, each with 1-2 observers = ~10,000 observers
/// - **Monitoring system**: 100 metrics, each with 20 dashboard widgets = ~2,000 observers
pub const MAX_OBSERVERS: usize = 10_000;
