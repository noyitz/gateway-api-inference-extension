use std::collections::HashMap;

use ipp_framework::cycle_state::CycleState;
use ipp_framework::error::PluginError;
use ipp_framework::inference_message::{InferenceRequest, InferenceResponse};
use ipp_framework::plugin::{RequestProcessor, ResponseProcessor};
use ipp_framework::{provider, state_keys};

use super::anthropic::AnthropicTranslator;
use super::azure_openai::AzureOpenAiTranslator;
use super::bedrock_openai::BedrockOpenAiTranslator;
use super::openai::OpenAiTranslator;
use super::translator::Translator;
use super::vertex_openai::VertexOpenAiTranslator;

#[derive(Clone)]
pub struct VertexOpenAiConfig {
    pub project: String,
    pub location: String,
    pub endpoint: String,
}

pub struct ApiTranslationPlugin {
    providers: HashMap<String, Box<dyn Translator>>,
}

impl ApiTranslationPlugin {
    pub fn new(vertex_config: Option<VertexOpenAiConfig>) -> Result<Self, PluginError> {
        let mut providers: HashMap<String, Box<dyn Translator>> = HashMap::new();

        providers.insert(provider::OPENAI.to_string(), Box::new(OpenAiTranslator));
        providers.insert(
            provider::ANTHROPIC.to_string(),
            Box::new(AnthropicTranslator),
        );
        providers.insert(
            provider::AZURE_OPENAI.to_string(),
            Box::new(AzureOpenAiTranslator::new()),
        );
        providers.insert(
            provider::BEDROCK_OPENAI.to_string(),
            Box::new(BedrockOpenAiTranslator),
        );

        if let Some(cfg) = vertex_config {
            if cfg.project.is_empty() || cfg.location.is_empty() || cfg.endpoint.is_empty() {
                return Err(PluginError::bad_request(
                    "vertexOpenAI config requires non-empty project, location, and endpoint",
                ));
            }
            providers.insert(
                provider::VERTEX_OPENAI.to_string(),
                Box::new(VertexOpenAiTranslator::new(
                    &cfg.project,
                    &cfg.location,
                    &cfg.endpoint,
                )),
            );
        }

        Ok(Self { providers })
    }
}

impl RequestProcessor for ApiTranslationPlugin {
    fn name(&self) -> &str {
        "api-translation"
    }

    fn process_request(
        &self,
        cycle_state: &mut CycleState,
        request: &mut InferenceRequest,
    ) -> Result<(), PluginError> {
        let provider_name = match cycle_state.try_read::<String>(state_keys::PROVIDER) {
            Some(p) if !p.is_empty() => p.clone(),
            _ => return Ok(()),
        };

        let translator = self.providers.get(&provider_name).ok_or_else(|| {
            PluginError::bad_request(format!("unsupported provider - '{provider_name}'"))
        })?;

        let result = translator.translate_request(&request.body)?;

        if let Some(body) = result.body {
            request.set_body(body);
        }

        for (key, value) in &result.headers_to_mutate {
            request.set_header(key.clone(), value.clone());
        }
        for key in &result.headers_to_remove {
            request.remove_header(key);
        }

        request.remove_header("authorization");

        Ok(())
    }
}

impl ResponseProcessor for ApiTranslationPlugin {
    fn name(&self) -> &str {
        "api-translation"
    }

    fn process_response(
        &self,
        cycle_state: &mut CycleState,
        response: &mut InferenceResponse,
    ) -> Result<(), PluginError> {
        let provider_name = match cycle_state.try_read::<String>(state_keys::PROVIDER) {
            Some(p) if !p.is_empty() => p.clone(),
            _ => return Ok(()),
        };

        let translator = self.providers.get(&provider_name).ok_or_else(|| {
            PluginError::bad_request(format!("unsupported provider - '{provider_name}'"))
        })?;

        let model = cycle_state
            .try_read::<String>(state_keys::MODEL)
            .cloned()
            .unwrap_or_default();

        let mutated = translator.translate_response(&mut response.body, &model)?;

        if mutated {
            response.mark_body_mutated();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn plugin() -> ApiTranslationPlugin {
        ApiTranslationPlugin::new(Some(VertexOpenAiConfig {
            project: "test-project".to_string(),
            location: "us-central1".to_string(),
            endpoint: "openapi".to_string(),
        }))
        .unwrap()
    }

    fn make_request(model: &str) -> InferenceRequest {
        let body = json!({
            "model": model,
            "messages": [{"role": "user", "content": format!("hello from {model}")}]
        });
        let mut req = InferenceRequest::new();
        req.set_body(body);
        req
    }

    // --- Request tests matching E2E flow ---

    #[test]
    fn no_provider_in_cycle_state_is_passthrough() {
        let p = plugin();
        let mut cs = CycleState::new();
        let mut req = make_request("some-model");
        p.process_request(&mut cs, &mut req).unwrap();
        // Body should not have been re-set (original set_body from make_request is the only mutation)
    }

    #[test]
    fn openai_provider_request() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());
        let mut req = make_request("gpt-4o");

        p.process_request(&mut cs, &mut req).unwrap();

        assert_eq!(req.headers.get(":path").unwrap(), "/v1/chat/completions");
        assert!(!req.headers.contains_key("authorization"));
    }

    #[test]
    fn anthropic_provider_request() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "anthropic".to_string());
        let mut req = make_request("claude-3-5-sonnet-20241022");

        p.process_request(&mut cs, &mut req).unwrap();

        assert_eq!(req.headers.get(":path").unwrap(), "/v1/messages");
        assert_eq!(req.headers.get("anthropic-version").unwrap(), "2023-06-01");
        assert!(req.body_mutated());
        assert_eq!(req.body["model"], "claude-3-5-sonnet-20241022");
        assert!(req.body.get("messages").is_some());
        assert!(req.body.get("max_tokens").is_some());
    }

    #[test]
    fn azure_provider_request() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "azure-openai".to_string());
        let mut req = make_request("gpt-4o");

        p.process_request(&mut cs, &mut req).unwrap();

        assert_eq!(
            req.headers.get(":path").unwrap(),
            "/openai/v1/chat/completions"
        );
    }

    #[test]
    fn bedrock_provider_request() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "bedrock-openai".to_string());
        let mut req = make_request("us.anthropic.claude-3-5-sonnet-20241022-v2:0");

        p.process_request(&mut cs, &mut req).unwrap();

        assert_eq!(req.headers.get(":path").unwrap(), "/v1/chat/completions");
    }

    #[test]
    fn vertex_openai_provider_request() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "vertex-openai".to_string());
        let mut req = make_request("google/gemini-2.0-flash");

        p.process_request(&mut cs, &mut req).unwrap();

        assert_eq!(
            req.headers.get(":path").unwrap(),
            "/v1/projects/test-project/locations/us-central1/endpoints/openapi/chat/completions"
        );
    }

    #[test]
    fn unsupported_provider_error() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "unknown-provider".to_string());
        let mut req = make_request("some-model");

        let err = p.process_request(&mut cs, &mut req).unwrap_err();
        assert!(err.msg.contains("unsupported provider"));
    }

    #[test]
    fn authorization_header_removed() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());
        let mut req = make_request("gpt-4o");
        req.set_header("authorization", "Bearer user-token");

        p.process_request(&mut cs, &mut req).unwrap();

        assert!(!req.headers.contains_key("authorization"));
    }

    // --- Response tests ---

    #[test]
    fn anthropic_response_translated_to_openai() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "anthropic".to_string());
        cs.write(state_keys::MODEL, "claude-3-5-sonnet-20241022".to_string());

        let mut resp = InferenceResponse::new();
        resp.set_body(json!({
            "id": "msg_123",
            "type": "message",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        }));

        p.process_response(&mut cs, &mut resp).unwrap();

        assert!(resp.body_mutated());
        assert_eq!(resp.body["object"], "chat.completion");
        assert!(resp.body.get("choices").is_some());
        let choices = resp.body["choices"].as_array().unwrap();
        assert!(!choices.is_empty());
        assert_eq!(choices[0]["message"]["content"], "Hello!");
        assert_eq!(resp.body["model"], "claude-3-5-sonnet-20241022");
    }

    #[test]
    fn openai_response_passthrough() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());

        let mut resp = InferenceResponse::new();
        resp.set_body(json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"content": "Hi"}}]
        }));
        // Reset body_mutated flag by creating fresh
        let mut resp2 = InferenceResponse::new();
        resp2.inner.body = json!({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"content": "Hi"}}]
        });

        p.process_response(&mut cs, &mut resp2).unwrap();
        assert!(!resp2.body_mutated());
    }

    #[test]
    fn azure_response_strips_filter_results() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "azure-openai".to_string());

        let mut resp = InferenceResponse::new();
        resp.inner.body = json!({
            "id": "chatcmpl-123",
            "prompt_filter_results": [{"index": 0}],
            "choices": [
                {"index": 0, "content_filter_results": {"hate": "safe"}, "message": {"content": "Hi"}}
            ]
        });

        p.process_response(&mut cs, &mut resp).unwrap();
        assert!(resp.body_mutated());
        assert!(resp.body.get("prompt_filter_results").is_none());
        assert!(resp.body["choices"][0]
            .get("content_filter_results")
            .is_none());
    }

    #[test]
    fn vertex_response_strips_extra_properties() {
        let p = plugin();
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "vertex-openai".to_string());

        let mut resp = InferenceResponse::new();
        resp.inner.body = json!({
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "extra_properties": {"cached": true}
            }
        });

        p.process_response(&mut cs, &mut resp).unwrap();
        assert!(resp.body_mutated());
        assert!(resp.body["usage"].get("extra_properties").is_none());
    }

    #[test]
    fn no_provider_response_passthrough() {
        let p = plugin();
        let mut cs = CycleState::new();
        let mut resp = InferenceResponse::new();
        resp.inner.body = json!({"choices": []});

        p.process_response(&mut cs, &mut resp).unwrap();
        assert!(!resp.body_mutated());
    }
}
