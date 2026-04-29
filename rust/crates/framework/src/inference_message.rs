use std::collections::{HashMap, HashSet};

use serde_json::Value;

pub struct InferenceMessage {
    pub headers: HashMap<String, String>,
    pub body: Value,
    mutated_headers: HashMap<String, String>,
    removed_headers: HashSet<String>,
    body_mutated: bool,
}

impl InferenceMessage {
    pub fn new() -> Self {
        Self {
            headers: HashMap::new(),
            body: Value::Null,
            mutated_headers: HashMap::new(),
            removed_headers: HashSet::new(),
            body_mutated: false,
        }
    }

    pub fn with_headers_and_body(headers: HashMap<String, String>, body: Value) -> Self {
        Self {
            headers,
            body,
            mutated_headers: HashMap::new(),
            removed_headers: HashSet::new(),
            body_mutated: false,
        }
    }

    pub fn set_header(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        self.removed_headers.remove(&key);
        self.headers.insert(key.clone(), value.clone());
        self.mutated_headers.insert(key, value);
    }

    pub fn remove_header(&mut self, key: &str) {
        self.headers.remove(key);
        self.mutated_headers.remove(key);
        self.removed_headers.insert(key.to_string());
    }

    pub fn set_body(&mut self, body: Value) {
        self.body = body;
        self.body_mutated = true;
    }

    pub fn set_body_field(&mut self, key: impl Into<String>, value: Value) {
        if let Value::Object(ref mut map) = self.body {
            map.insert(key.into(), value);
            self.body_mutated = true;
        }
    }

    pub fn remove_body_field(&mut self, key: &str) {
        if let Value::Object(ref mut map) = self.body {
            if map.remove(key).is_some() {
                self.body_mutated = true;
            }
        }
    }

    pub fn body_mutated(&self) -> bool {
        self.body_mutated
    }

    pub fn mark_body_mutated(&mut self) {
        self.body_mutated = true;
    }

    pub fn mutated_headers(&self) -> &HashMap<String, String> {
        &self.mutated_headers
    }

    pub fn removed_headers(&self) -> Vec<String> {
        self.removed_headers.iter().cloned().collect()
    }
}

impl Default for InferenceMessage {
    fn default() -> Self {
        Self::new()
    }
}

pub struct InferenceRequest {
    pub inner: InferenceMessage,
}

impl InferenceRequest {
    pub fn new() -> Self {
        Self {
            inner: InferenceMessage::new(),
        }
    }

    pub fn with_headers_and_body(headers: HashMap<String, String>, body: Value) -> Self {
        Self {
            inner: InferenceMessage::with_headers_and_body(headers, body),
        }
    }
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for InferenceRequest {
    type Target = InferenceMessage;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for InferenceRequest {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub struct InferenceResponse {
    pub inner: InferenceMessage,
}

impl InferenceResponse {
    pub fn new() -> Self {
        Self {
            inner: InferenceMessage::new(),
        }
    }

    pub fn with_headers_and_body(headers: HashMap<String, String>, body: Value) -> Self {
        Self {
            inner: InferenceMessage::with_headers_and_body(headers, body),
        }
    }
}

impl Default for InferenceResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for InferenceResponse {
    type Target = InferenceMessage;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl std::ops::DerefMut for InferenceResponse {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn set_header_tracks_mutation() {
        let mut msg = InferenceMessage::new();
        msg.set_header("x-api-key", "secret123");

        assert_eq!(msg.headers.get("x-api-key").unwrap(), "secret123");
        assert_eq!(msg.mutated_headers().get("x-api-key").unwrap(), "secret123");
        assert!(msg.removed_headers().is_empty());
    }

    #[test]
    fn remove_header_tracks_removal() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer old".to_string());
        let mut msg = InferenceMessage::with_headers_and_body(headers, Value::Null);

        msg.remove_header("authorization");
        assert!(!msg.headers.contains_key("authorization"));
        assert!(msg.removed_headers().contains(&"authorization".to_string()));
    }

    #[test]
    fn set_then_remove_header() {
        let mut msg = InferenceMessage::new();
        msg.set_header("key", "value");
        assert!(msg.mutated_headers().contains_key("key"));

        msg.remove_header("key");
        assert!(!msg.mutated_headers().contains_key("key"));
        assert!(msg.removed_headers().contains(&"key".to_string()));
    }

    #[test]
    fn remove_then_set_header() {
        let mut headers = HashMap::new();
        headers.insert("key".to_string(), "old".to_string());
        let mut msg = InferenceMessage::with_headers_and_body(headers, Value::Null);

        msg.remove_header("key");
        msg.set_header("key", "new");

        assert_eq!(msg.mutated_headers().get("key").unwrap(), "new");
        assert!(!msg.removed_headers().contains(&"key".to_string()));
    }

    #[test]
    fn set_body_marks_mutation() {
        let mut msg = InferenceMessage::new();
        assert!(!msg.body_mutated());

        msg.set_body(json!({"model": "gpt-4"}));
        assert!(msg.body_mutated());
        assert_eq!(msg.body["model"], "gpt-4");
    }

    #[test]
    fn set_body_field() {
        let mut msg = InferenceMessage::new();
        msg.set_body(json!({"model": "gpt-4"}));
        msg.set_body_field("temperature", json!(0.7));

        assert_eq!(msg.body["temperature"], 0.7);
    }

    #[test]
    fn remove_body_field() {
        let mut msg = InferenceMessage::new();
        msg.body = json!({"model": "gpt-4", "stream": true});
        assert!(!msg.body_mutated());

        msg.remove_body_field("stream");
        assert!(msg.body_mutated());
        assert!(msg.body.get("stream").is_none());
    }

    #[test]
    fn remove_body_field_nonexistent_does_not_mark_mutated() {
        let mut msg = InferenceMessage::new();
        msg.body = json!({"model": "gpt-4"});
        msg.remove_body_field("nonexistent");
        assert!(!msg.body_mutated());
    }

    #[test]
    fn inference_request_deref() {
        let mut req = InferenceRequest::new();
        req.set_header(":path", "/v1/chat/completions");
        req.set_body(json!({"model": "gpt-4"}));

        assert_eq!(req.headers.get(":path").unwrap(), "/v1/chat/completions");
        assert!(req.body_mutated());
    }

    #[test]
    fn inference_response_deref() {
        let mut resp = InferenceResponse::new();
        resp.set_header("content-type", "application/json");

        assert_eq!(
            resp.mutated_headers().get("content-type").unwrap(),
            "application/json"
        );
    }
}
