use observable_property::{Observable, observable, ObservableProperty};
use std::sync::Arc;

#[derive(Observable)]
struct Config {
    #[observable]
    debug_mode: ObservableProperty<bool>,
    #[observable]
    max_connections: ObservableProperty<usize>,
    #[observable]
    server_name: ObservableProperty<String>,
    version: String, // Not observable
}

fn main() {
    let config = Config::new(
        false,
        100,
        "localhost".to_string(),
        "1.0.0".to_string(),
    );

    // Test filtered subscription
    let _debug_id = config.subscribe_debug_mode_filtered(
        Arc::new(|old, new| {
            println!("Debug mode enabled: {} -> {}", old, new);
        }),
        |old, new| !old && *new // Only trigger when debug is turned on
    ).unwrap();

    config.set_debug_mode(false).unwrap(); // Should not trigger
    config.set_debug_mode(true).unwrap();  // Should trigger

    // Test complex types
    assert_eq!(config.get_debug_mode().unwrap(), true);
    assert_eq!(config.get_max_connections().unwrap(), 100);
    assert_eq!(config.get_server_name().unwrap(), "localhost");
    assert_eq!(config.version, "1.0.0");
}
