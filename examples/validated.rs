//! Example demonstrating validated properties
//!
//! This example shows how to use the `with_validator` constructor to create
//! properties that enforce validation rules on their values.
//!
//! Validators are extracted as named, reusable functions, errors are typed via
//! [`ValidationError`], and all output goes through the `tracing` crate.

use observable_property::{ObservableProperty, PropertyError};
use std::sync::Arc;
use tracing::{error, info, warn};

// ---------------------------------------------------------------------------
// Typed validation errors
// ---------------------------------------------------------------------------

/// Domain-level validation errors returned by the validators in this module.
#[derive(Debug)]
pub enum ValidationError {
    AgeOutOfRange(i32),
    InvalidEmail(String),
    UsernameEmpty,
    UsernameTooShort { len: usize },
    UsernameTooLong { len: usize },
    UsernameInvalidChars,
    TemperatureOutOfRange(f64),
    CounterOutOfRange(i32),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AgeOutOfRange(v) =>
                write!(f, "Age must be between 0 and 150, got {v}"),
            Self::InvalidEmail(e) =>
                write!(f, "Invalid email format: {e}"),
            Self::UsernameEmpty =>
                write!(f, "Username cannot be empty"),
            Self::UsernameTooShort { len } =>
                write!(f, "Username must be at least 3 characters, got {len}"),
            Self::UsernameTooLong { len } =>
                write!(f, "Username must be at most 20 characters, got {len}"),
            Self::UsernameInvalidChars =>
                write!(f, "Username can only contain letters, numbers, and underscores"),
            Self::TemperatureOutOfRange(v) =>
                write!(f, "Temperature {v} is out of valid range [-273.15, 1000.0]"),
            Self::CounterOutOfRange(v) =>
                write!(f, "Counter must be between 0 and 100, got {v}"),
        }
    }
}

impl std::error::Error for ValidationError {}

// ---------------------------------------------------------------------------
// Named, reusable validator functions
// ---------------------------------------------------------------------------

fn validate_age(age: &i32) -> Result<(), String> {
    if *age >= 0 && *age <= 150 {
        Ok(())
    } else {
        Err(ValidationError::AgeOutOfRange(*age).to_string())
    }
}

/// Validates an email address.
///
/// Checks that the address has a non-empty local part, an `@` separator, and a
/// domain containing at least one `.` that is not at the start or end.
fn validate_email(email: &String) -> Result<(), String> {
    let valid = email.find('@').map_or(false, |at| {
        let local = &email[..at];
        let domain = &email[at + 1..];
        !local.is_empty()
            && domain.contains('.')
            && !domain.starts_with('.')
            && !domain.ends_with('.')
            && domain.len() > 2
    });
    if valid {
        Ok(())
    } else {
        Err(ValidationError::InvalidEmail(email.clone()).to_string())
    }
}

fn validate_username(name: &String) -> Result<(), String> {
    if name.is_empty() {
        return Err(ValidationError::UsernameEmpty.to_string());
    }
    if name.len() < 3 {
        return Err(ValidationError::UsernameTooShort { len: name.len() }.to_string());
    }
    if name.len() > 20 {
        return Err(ValidationError::UsernameTooLong { len: name.len() }.to_string());
    }
    if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return Err(ValidationError::UsernameInvalidChars.to_string());
    }
    Ok(())
}

fn validate_temperature(temp: &f64) -> Result<(), String> {
    if *temp >= -273.15 && *temp <= 1000.0 {
        Ok(())
    } else {
        Err(ValidationError::TemperatureOutOfRange(*temp).to_string())
    }
}

fn validate_counter(value: &i32) -> Result<(), String> {
    if *value >= 0 && *value <= 100 {
        Ok(())
    } else {
        Err(ValidationError::CounterOutOfRange(*value).to_string())
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<(), PropertyError> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    info!("=== Validated Properties Example ===");

    // Example 1: Age validation (0-150)
    info!("1. Age Validation (range: 0-150)");
    let age = ObservableProperty::with_validator(25, validate_age)?;
    info!(value = age.get()?, "Initial age");

    let _sub = age.subscribe_with_subscription(Arc::new(|old, new| {
        info!(old, new, "Age changed");
    }))?;

    age.set(30)?;
    info!(value = age.get()?, "Current age");

    match age.set(200) {
        Ok(_) => error!("Should have failed validation!"),
        Err(e) => warn!(error = %e, "Validation correctly rejected"),
    }
    info!(value = age.get()?, "Age after failed validation");

    // Example 2: Email format validation
    info!("2. Email Format Validation");
    let email =
        ObservableProperty::with_validator("user@example.com".to_string(), validate_email)?;
    info!(value = %email.get()?, "Initial email");

    email.set("valid@email.com".to_string())?;
    info!(value = %email.get()?, "Updated email");

    match email.set("invalid-email".to_string()) {
        Ok(_) => error!("Should have failed validation!"),
        Err(e) => warn!(error = %e, "Validation correctly rejected"),
    }

    // Example 3: Username validation (multiple rules)
    info!("3. Username Validation (3-20 chars, alphanumeric + underscore)");
    let username = ObservableProperty::with_validator("alice".to_string(), validate_username)?;
    info!(value = %username.get()?, "Initial username");

    username.set("bob_123".to_string())?;
    info!(value = %username.get()?, "Changed username");

    match username.set("ab".to_string()) {
        Ok(_) => error!("Should have failed!"),
        Err(e) => warn!(error = %e, "Rejected (too short)"),
    }
    match username.set("user@123".to_string()) {
        Ok(_) => error!("Should have failed!"),
        Err(e) => warn!(error = %e, "Rejected (invalid chars)"),
    }

    // Example 4: Temperature range validation
    info!("4. Temperature Range Validation (-273.15 to 1000.0°C)");
    let temperature =
        ObservableProperty::with_validator(20.0_f64, validate_temperature)?;
    info!(value = temperature.get()?, "Initial temperature (°C)");

    temperature.set(100.0)?;
    info!(value = temperature.get()?, "Heated to (°C)");

    temperature.set(-50.0)?;
    info!(value = temperature.get()?, "Cooled to (°C)");

    match temperature.set(-300.0) {
        Ok(_) => error!("Should have failed!"),
        Err(e) => warn!(error = %e, "Rejected (below absolute zero)"),
    }

    // Example 5: Invalid initial value is rejected
    info!("5. Rejecting Invalid Initial Values");
    match ObservableProperty::with_validator(200, validate_age) {
        Ok(_) => error!("Should have failed to create!"),
        Err(e) => warn!(error = %e, "Creation correctly rejected"),
    }

    // Example 6: modify() with validation — atomically rolls back on failure
    info!("6. Validation with modify()");
    let counter = ObservableProperty::with_validator(5, validate_counter)?;
    info!(value = counter.get()?, "Initial counter");

    counter.modify(|val| *val += 10)?;
    info!(value = counter.get()?, "After increment");

    match counter.modify(|val| *val += 200) {
        Ok(_) => error!("Should have failed validation!"),
        Err(e) => {
            warn!(error = %e, "Modification rejected; value atomically rolled back");
            info!(value = counter.get()?, "Counter after rollback");
        }
    }

    info!("=== All validation examples completed successfully ===");
    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests for validator functions
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_age ---

    #[test]
    fn age_accepts_zero() {
        assert!(validate_age(&0).is_ok());
    }

    #[test]
    fn age_accepts_150() {
        assert!(validate_age(&150).is_ok());
    }

    #[test]
    fn age_rejects_negative() {
        assert!(validate_age(&-1).is_err());
    }

    #[test]
    fn age_rejects_above_150() {
        assert!(validate_age(&151).is_err());
    }

    // --- validate_email ---

    #[test]
    fn email_accepts_valid() {
        assert!(validate_email(&"user@example.com".to_string()).is_ok());
    }

    #[test]
    fn email_rejects_missing_at() {
        assert!(validate_email(&"nodomain.com".to_string()).is_err());
    }

    #[test]
    fn email_rejects_missing_dot_in_domain() {
        assert!(validate_email(&"user@nodot".to_string()).is_err());
    }

    #[test]
    fn email_rejects_empty_local_part() {
        assert!(validate_email(&"@example.com".to_string()).is_err());
    }

    #[test]
    fn email_rejects_dot_at_start_of_domain() {
        assert!(validate_email(&"user@.example.com".to_string()).is_err());
    }

    // --- validate_username ---

    #[test]
    fn username_accepts_valid() {
        assert!(validate_username(&"alice_123".to_string()).is_ok());
    }

    #[test]
    fn username_rejects_empty() {
        assert!(validate_username(&String::new()).is_err());
    }

    #[test]
    fn username_rejects_too_short() {
        assert!(validate_username(&"ab".to_string()).is_err());
    }

    #[test]
    fn username_rejects_too_long() {
        assert!(validate_username(&"a".repeat(21)).is_err());
    }

    #[test]
    fn username_rejects_invalid_chars() {
        assert!(validate_username(&"user@name".to_string()).is_err());
    }

    #[test]
    fn username_accepts_boundary_lengths() {
        assert!(validate_username(&"abc".to_string()).is_ok());
        assert!(validate_username(&"a".repeat(20)).is_ok());
    }

    // --- validate_temperature ---

    #[test]
    fn temperature_accepts_absolute_zero() {
        assert!(validate_temperature(&-273.15).is_ok());
    }

    #[test]
    fn temperature_accepts_max() {
        assert!(validate_temperature(&1000.0).is_ok());
    }

    #[test]
    fn temperature_rejects_below_absolute_zero() {
        assert!(validate_temperature(&-274.0).is_err());
    }

    #[test]
    fn temperature_rejects_above_max() {
        assert!(validate_temperature(&1001.0).is_err());
    }

    // --- validate_counter ---

    #[test]
    fn counter_accepts_boundaries() {
        assert!(validate_counter(&0).is_ok());
        assert!(validate_counter(&100).is_ok());
    }

    #[test]
    fn counter_rejects_negative() {
        assert!(validate_counter(&-1).is_err());
    }

    #[test]
    fn counter_rejects_above_100() {
        assert!(validate_counter(&101).is_err());
    }
}
