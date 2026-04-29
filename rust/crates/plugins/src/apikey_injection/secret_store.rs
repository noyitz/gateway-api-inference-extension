use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

#[derive(Clone)]
pub struct SecretStore {
    inner: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
}

impl SecretStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn add_or_update(
        &self,
        key: &str,
        credentials: HashMap<String, String>,
    ) -> Result<(), String> {
        if credentials.is_empty() {
            self.delete(key);
            return Err(format!("secret '{}' has no data fields", key));
        }

        for (field, value) in &credentials {
            if value.is_empty() {
                self.delete(key);
                return Err(format!("secret '{}' has empty field '{}'", key, field));
            }
        }

        self.inner.write().insert(key.to_string(), credentials);
        Ok(())
    }

    pub fn delete(&self, key: &str) {
        self.inner.write().remove(key);
    }

    pub fn get(&self, key: &str) -> Option<HashMap<String, String>> {
        self.inner.read().get(key).cloned()
    }
}

impl Default for SecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_and_get() {
        let store = SecretStore::new();
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "sk-123".to_string());
        store.add_or_update("ns/secret1", creds).unwrap();

        let result = store.get("ns/secret1").unwrap();
        assert_eq!(result["api-key"], "sk-123");
    }

    #[test]
    fn empty_credentials_rejected() {
        let store = SecretStore::new();
        let err = store
            .add_or_update("ns/empty", HashMap::new())
            .unwrap_err();
        assert!(err.contains("no data fields"));
    }

    #[test]
    fn empty_field_value_rejected() {
        let store = SecretStore::new();
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), String::new());
        let err = store.add_or_update("ns/bad", creds).unwrap_err();
        assert!(err.contains("empty field"));
    }

    #[test]
    fn delete_removes() {
        let store = SecretStore::new();
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "sk-123".to_string());
        store.add_or_update("ns/s", creds).unwrap();
        store.delete("ns/s");
        assert!(store.get("ns/s").is_none());
    }
}
