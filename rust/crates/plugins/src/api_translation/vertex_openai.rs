use std::collections::HashMap;

use ipp_framework::error::PluginError;
use serde_json::Value;

use super::field_stripper::ResponseFieldStripper;
use super::translator::{TranslateRequestResult, Translator};

pub struct VertexOpenAiTranslator {
    path: String,
    stripper: ResponseFieldStripper,
}

impl VertexOpenAiTranslator {
    pub fn new(project: &str, location: &str, endpoint: &str) -> Self {
        Self {
            path: format!(
                "/v1/projects/{}/locations/{}/endpoints/{}/chat/completions",
                project, location, endpoint
            ),
            stripper: ResponseFieldStripper::new(&["usage.extra_properties"]),
        }
    }
}

impl Translator for VertexOpenAiTranslator {
    fn translate_request(&self, body: &Value) -> Result<TranslateRequestResult, PluginError> {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");
        if model.is_empty() {
            return Err(PluginError::bad_request("model field is required"));
        }

        let messages = body.get("messages").and_then(Value::as_array);
        if messages.is_none_or(|arr| arr.is_empty()) {
            return Err(PluginError::bad_request(
                "messages field is required and must not be empty",
            ));
        }

        let mut headers = HashMap::new();
        headers.insert(":path".to_string(), self.path.clone());
        headers.insert("content-type".to_string(), "application/json".to_string());

        Ok(TranslateRequestResult {
            body: None,
            headers_to_mutate: headers,
            headers_to_remove: Vec::new(),
        })
    }

    fn translate_response(&self, body: &mut Value, _model: &str) -> Result<bool, PluginError> {
        Ok(self.stripper.strip(body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_translator() -> VertexOpenAiTranslator {
        VertexOpenAiTranslator::new("my-project", "us-central1", "openapi")
    }

    #[test]
    fn request_path_includes_project_location_endpoint() {
        let body = json!({
            "model": "google/gemini-2.0-flash",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = test_translator().translate_request(&body).unwrap();
        assert!(result.body.is_none());
        assert_eq!(
            result.headers_to_mutate[":path"],
            "/v1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions"
        );
    }

    #[test]
    fn response_strips_extra_properties() {
        let mut body = json!({
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15,
                "extra_properties": {"cached_content_token_count": 0}
            }
        });
        let mutated = test_translator()
            .translate_response(&mut body, "gemini")
            .unwrap();
        assert!(mutated);
        assert!(body["usage"].get("extra_properties").is_none());
        assert_eq!(body["usage"]["prompt_tokens"], 5);
    }

    #[test]
    fn response_no_mutation_when_clean() {
        let mut body = json!({
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        });
        let mutated = test_translator()
            .translate_response(&mut body, "gemini")
            .unwrap();
        assert!(!mutated);
    }

    #[test]
    fn missing_model() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        assert!(test_translator().translate_request(&body).is_err());
    }

    #[test]
    fn empty_messages() {
        let body = json!({"model": "gemini", "messages": []});
        assert!(test_translator().translate_request(&body).is_err());
    }
}
