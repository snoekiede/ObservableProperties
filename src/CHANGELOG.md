# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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