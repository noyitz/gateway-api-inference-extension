use std::any::Any;
use std::collections::HashMap;

use crate::error::PluginError;

pub struct CycleState {
    storage: HashMap<&'static str, Box<dyn Any + Send + Sync>>,
}

impl CycleState {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    pub fn write<T: Any + Send + Sync>(&mut self, key: &'static str, value: T) {
        self.storage.insert(key, Box::new(value));
    }

    pub fn read<T: Any + Send + Sync>(&self, key: &'static str) -> Result<&T, PluginError> {
        let val = self
            .storage
            .get(key)
            .ok_or_else(|| PluginError::internal(format!("cycle state key '{key}' not found")))?;
        val.downcast_ref::<T>().ok_or_else(|| {
            PluginError::internal(format!("cycle state key '{key}' has unexpected type"))
        })
    }

    pub fn try_read<T: Any + Send + Sync>(&self, key: &'static str) -> Option<&T> {
        self.storage.get(key)?.downcast_ref::<T>()
    }

    pub fn delete(&mut self, key: &'static str) {
        self.storage.remove(key);
    }
}

impl Default for CycleState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_read_string() {
        let mut cs = CycleState::new();
        cs.write("provider", "anthropic".to_string());
        let val: &String = cs.read("provider").unwrap();
        assert_eq!(val, "anthropic");
    }

    #[test]
    fn write_and_read_different_types() {
        let mut cs = CycleState::new();
        cs.write("name", "test-model".to_string());
        cs.write("count", 42u64);

        assert_eq!(cs.read::<String>("name").unwrap(), "test-model");
        assert_eq!(cs.read::<u64>("count").unwrap(), &42u64);
    }

    #[test]
    fn read_missing_key_returns_error() {
        let cs = CycleState::new();
        let result = cs.read::<String>("missing");
        assert!(result.is_err());
    }

    #[test]
    fn read_wrong_type_returns_error() {
        let mut cs = CycleState::new();
        cs.write("key", "a string".to_string());
        let result = cs.read::<u64>("key");
        assert!(result.is_err());
    }

    #[test]
    fn try_read_returns_none_for_missing() {
        let cs = CycleState::new();
        assert!(cs.try_read::<String>("missing").is_none());
    }

    #[test]
    fn delete_removes_key() {
        let mut cs = CycleState::new();
        cs.write("key", "value".to_string());
        cs.delete("key");
        assert!(cs.try_read::<String>("key").is_none());
    }

    #[test]
    fn overwrite_key() {
        let mut cs = CycleState::new();
        cs.write("key", "first".to_string());
        cs.write("key", "second".to_string());
        assert_eq!(cs.read::<String>("key").unwrap(), "second");
    }
}
