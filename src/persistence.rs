//! Persistence trait for observable properties

/// Trait for implementing property value persistence
///
/// This trait allows custom persistence strategies for ObservableProperty values,
/// enabling automatic save/load functionality to various storage backends like
/// disk files, databases, cloud storage, etc.
///
/// # Examples
///
/// ```rust
/// use observable_property::PropertyPersistence;
/// use std::fs;
///
/// struct FilePersistence {
///     path: String,
/// }
///
/// impl PropertyPersistence for FilePersistence {
///     type Value = String;
///
///     fn load(&self) -> Result<Self::Value, Box<dyn std::error::Error + Send + Sync>> {
///         fs::read_to_string(&self.path)
///             .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
///     }
///
///     fn save(&self, value: &Self::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
///         fs::write(&self.path, value)
///             .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
///     }
/// }
/// ```
pub trait PropertyPersistence: Send + Sync + 'static {
    /// The type of value being persisted
    type Value: Clone + Send + Sync + 'static;

    /// Load the value from persistent storage
    ///
    /// # Returns
    ///
    /// Returns the loaded value or an error if loading fails
    fn load(&self) -> Result<Self::Value, Box<dyn std::error::Error + Send + Sync>>;

    /// Save the value to persistent storage
    ///
    /// # Arguments
    ///
    /// * `value` - The value to persist
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if successful, or an error if saving fails
    fn save(&self, value: &Self::Value) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
