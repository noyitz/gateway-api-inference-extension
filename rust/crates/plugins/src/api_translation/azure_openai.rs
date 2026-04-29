use std::collections::HashMap;

use ipp_framework::error::PluginError;
use serde_json::Value;

use super::field_stripper::ResponseFieldStripper;
use super::translator::{TranslateRequestResult, Translator};

const AZURE_CHAT_COMPLETIONS_PATH: &str = "/openai/v1/chat/completions";

pub struct AzureOpenAiTranslator {
    stripper: ResponseFieldStripper,
}

impl AzureOpenAiTranslator {
    pub fn new() -> Self {
        Self {
            stripper: ResponseFieldStripper::new(&[
                "prompt_filter_results",
                "choices[].content_filter_results",
            ]),
        }
    }
}

impl Default for AzureOpenAiTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl Translator for AzureOpenAiTranslator {
    fn translate_request(&self, body: &Value) -> Result<TranslateRequestResult, PluginError> {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");
        if model.is_empty() {
            return Err(PluginError::bad_request("model field is required"));
        }

        let mut headers = HashMap::new();
        headers.insert(":path".to_string(), AZURE_CHAT_COMPLETIONS_PATH.to_string());
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

    #[test]
    fn request_path_rewrite() {
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = AzureOpenAiTranslator::new()
            .translate_request(&body)
            .unwrap();
        assert!(result.body.is_none());
        assert_eq!(
            result.headers_to_mutate[":path"],
            "/openai/v1/chat/completions"
        );
    }

    #[test]
    fn response_strips_filter_results() {
        let mut body = json!({
            "id": "chatcmpl-123",
            "prompt_filter_results": [{"index": 0}],
            "choices": [
                {"index": 0, "content_filter_results": {"hate": {"filtered": false}}, "message": {"content": "Hi"}}
            ]
        });
        let mutated = AzureOpenAiTranslator::new()
            .translate_response(&mut body, "gpt-4o")
            .unwrap();
        assert!(mutated);
        assert!(body.get("prompt_filter_results").is_none());
        assert!(body["choices"][0].get("content_filter_results").is_none());
        assert!(body["choices"][0].get("message").is_some());
    }

    #[test]
    fn response_no_mutation_when_clean() {
        let mut body = json!({
            "id": "chatcmpl-123",
            "choices": [{"index": 0, "message": {"content": "Hi"}}]
        });
        let mutated = AzureOpenAiTranslator::new()
            .translate_response(&mut body, "gpt-4o")
            .unwrap();
        assert!(!mutated);
    }

    #[test]
    fn missing_model() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        assert!(AzureOpenAiTranslator::new()
            .translate_request(&body)
            .is_err());
    }
}
