# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.4] - Unreleased

### Added
- All-features test run to CI.
- `serde_json` as a development dependency for serde documentation tests.

### Changed
- Bound async, debounced, and throttled notification workers across concurrent calls and clones of a property; saturated dispatch and thread-spawn failures fall back to caller-thread notification.
- Store notification timing metrics as constant-space aggregates instead of retaining every sample.
- Serialize persistence callbacks and save the latest committed property value.
- Run `set()` and `set_async()` validators outside the property lock.

### Fixed
- Wake async streams and `wait_for()` futures after notifications, and release their subscriptions when complete.
- Release map, computed, and bidirectional-binding subscriptions when their owning properties are dropped.
- Validate batch updates before commit, roll back closure panics, record each intermediate transition, and suppress notifications for empty batches.
- Record undo and modify transitions in event logs, change counts, and notification metrics while preserving undo-history semantics.
- Restore the previous value if a `modify()` closure, validator, or equality function panics before resuming the panic.

## [0.4.2] - 2024-01-XX

### Added
- Exhaustive test suite with 235 tests (128 unit + 107 doc tests)
- Validation support with `with_validator()`
- Custom equality comparison with `with_equality()`
- History tracking with `with_history()` and `undo()`
- Event logging with `with_event_log()` and `get_event_log()`
- Property transformations with `map()`
- Atomic modifications with `modify()`
- Bidirectional binding with `bind_bidirectional()`
- Comprehensive metrics with `get_metrics()`
- Debounced observers with `subscribe_debounced()`
- Throttled observers with `subscribe_throttled()`
- Computed properties with `computed()`
- Change coalescing with `begin_update()`/`end_update()`
- Batch notifications with `update_batch()`
- Weak observer support with `subscribe_weak()`

### Changed
- Improved lock poisoning recovery with graceful degradation
- Enhanced documentation with extensive examples

## [0.4.1] - 2024-01-XX
- Previous version

## [0.4.0] - 2024-01-XX
- Initial release