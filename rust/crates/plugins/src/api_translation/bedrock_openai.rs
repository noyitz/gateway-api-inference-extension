use std::collections::HashMap;

use ipp_framework::error::PluginError;
use serde_json::Value;

use super::translator::{TranslateRequestResult, Translator};

const BEDROCK_OPENAI_PATH: &str = "/v1/chat/completions";

pub struct BedrockOpenAiTranslator;

impl Translator for BedrockOpenAiTranslator {
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
        headers.insert(":path".to_string(), BEDROCK_OPENAI_PATH.to_string());
        headers.insert("content-type".to_string(), "application/json".to_string());

        Ok(TranslateRequestResult {
            body: None,
            headers_to_mutate: headers,
            headers_to_remove: Vec::new(),
        })
    }

    fn translate_response(&self, _body: &mut Value, _model: &str) -> Result<bool, PluginError> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basic_request() {
        let body = json!({
            "model": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = BedrockOpenAiTranslator.translate_request(&body).unwrap();
        assert!(result.body.is_none());
        assert_eq!(result.headers_to_mutate[":path"], "/v1/chat/completions");
        assert_eq!(result.headers_to_mutate["content-type"], "application/json");
    }

    #[test]
    fn missing_model() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        assert!(BedrockOpenAiTranslator.translate_request(&body).is_err());
    }

    #[test]
    fn empty_messages() {
        let body = json!({"model": "some-model", "messages": []});
        assert!(BedrockOpenAiTranslator.translate_request(&body).is_err());
    }

    #[test]
    fn response_noop() {
        let mut body = json!({"choices": []});
        assert!(!BedrockOpenAiTranslator
            .translate_response(&mut body, "m")
            .unwrap());
    }
}
