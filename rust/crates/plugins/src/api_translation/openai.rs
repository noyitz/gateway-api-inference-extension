use std::collections::HashMap;

use ipp_framework::error::PluginError;
use serde_json::Value;

use super::translator::{TranslateRequestResult, Translator};

const OPENAI_PATH: &str = "/v1/chat/completions";

pub struct OpenAiTranslator;

impl Translator for OpenAiTranslator {
    fn translate_request(&self, body: &Value) -> Result<TranslateRequestResult, PluginError> {
        let model = body.get("model").and_then(Value::as_str).unwrap_or("");
        if model.is_empty() {
            return Err(PluginError::bad_request("model field is required"));
        }

        let messages = body.get("messages").and_then(Value::as_array);
        match messages {
            None => {
                return Err(PluginError::bad_request(
                    "messages field is required and must not be empty",
                ));
            }
            Some(arr) if arr.is_empty() => {
                return Err(PluginError::bad_request(
                    "messages field is required and must not be empty",
                ));
            }
            _ => {}
        }

        let mut headers = HashMap::new();
        headers.insert(":path".to_string(), OPENAI_PATH.to_string());

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
    fn basic_chat() {
        let body = json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}]
        });

        let result = OpenAiTranslator.translate_request(&body).unwrap();
        assert!(result.body.is_none());
        assert_eq!(result.headers_to_mutate[":path"], "/v1/chat/completions");
        assert!(result.headers_to_remove.is_empty());
    }

    #[test]
    fn missing_model() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let err = OpenAiTranslator.translate_request(&body).unwrap_err();
        assert!(err.msg.contains("model"));
    }

    #[test]
    fn empty_messages() {
        let body = json!({
            "model": "gpt-4o-mini",
            "messages": []
        });
        let err = OpenAiTranslator.translate_request(&body).unwrap_err();
        assert!(err.msg.contains("messages"));
    }

    #[test]
    fn missing_messages() {
        let body = json!({"model": "gpt-4o-mini"});
        let err = OpenAiTranslator.translate_request(&body).unwrap_err();
        assert!(err.msg.contains("messages"));
    }

    #[test]
    fn response_noop() {
        let mut body = json!({"choices": []});
        let mutated = OpenAiTranslator
            .translate_response(&mut body, "gpt-4o-mini")
            .unwrap();
        assert!(!mutated);
    }
}
