pub mod auth;
pub mod reconciler;
pub mod secret_store;

use std::collections::HashMap;

use ipp_framework::cycle_state::CycleState;
use ipp_framework::error::PluginError;
use ipp_framework::inference_message::InferenceRequest;
use ipp_framework::plugin::RequestProcessor;
use ipp_framework::{provider, state_keys};

use auth::SimpleAuthGenerator;
use secret_store::SecretStore;

pub struct ApiKeyInjectionPlugin {
    generators: HashMap<String, SimpleAuthGenerator>,
    store: SecretStore,
}

impl ApiKeyInjectionPlugin {
    pub fn new(store: SecretStore) -> Self {
        let mut generators = HashMap::new();
        generators.insert(
            provider::OPENAI.to_string(),
            SimpleAuthGenerator {
                header_name: "Authorization".to_string(),
                header_value_prefix: "Bearer ".to_string(),
            },
        );
        generators.insert(
            provider::ANTHROPIC.to_string(),
            SimpleAuthGenerator {
                header_name: "x-api-key".to_string(),
                header_value_prefix: String::new(),
            },
        );
        generators.insert(
            provider::AZURE_OPENAI.to_string(),
            SimpleAuthGenerator {
                header_name: "api-key".to_string(),
                header_value_prefix: String::new(),
            },
        );
        generators.insert(
            provider::VERTEX_OPENAI.to_string(),
            SimpleAuthGenerator {
                header_name: "Authorization".to_string(),
                header_value_prefix: "Bearer ".to_string(),
            },
        );
        generators.insert(
            provider::BEDROCK_OPENAI.to_string(),
            SimpleAuthGenerator {
                header_name: "Authorization".to_string(),
                header_value_prefix: "Bearer ".to_string(),
            },
        );

        Self { generators, store }
    }
}

impl RequestProcessor for ApiKeyInjectionPlugin {
    fn name(&self) -> &str {
        "apikey-injection"
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

        let creds_name = cycle_state
            .try_read::<String>(state_keys::CREDS_REF_NAME)
            .cloned()
            .unwrap_or_default();
        if creds_name.is_empty() {
            return Err(PluginError::internal(format!(
                "provider '{}' is missing credentialRef",
                provider_name
            )));
        }

        let creds_namespace = cycle_state
            .try_read::<String>(state_keys::CREDS_REF_NAMESPACE)
            .cloned()
            .unwrap_or_default();
        if creds_namespace.is_empty() {
            return Err(PluginError::internal(format!(
                "provider '{}' is missing credentialRef namespace",
                provider_name
            )));
        }

        let secret_key = format!("{}/{}", creds_namespace, creds_name);
        let credentials = self.store.get(&secret_key).ok_or_else(|| {
            PluginError::internal(format!(
                "provider '{}' credentials not found",
                provider_name
            ))
        })?;

        let generator = self.generators.get(&provider_name).ok_or_else(|| {
            PluginError::internal(format!("unsupported provider - '{}'", provider_name))
        })?;

        let auth_headers = generator.generate_auth_headers(&credentials).map_err(|e| {
            PluginError::internal(format!(
                "failed to generate auth headers for provider '{}': {}",
                provider_name, e
            ))
        })?;

        for (key, value) in auth_headers {
            request.set_header(key, value);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup_store() -> SecretStore {
        let store = SecretStore::new();
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "sk-openai-123".to_string());
        store.add_or_update("bbr-e2e/e2e-openai", creds).unwrap();

        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "sk-ant-123".to_string());
        store.add_or_update("bbr-e2e/e2e-anthropic", creds).unwrap();

        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "azure-key-123".to_string());
        store.add_or_update("bbr-e2e/e2e-azure", creds).unwrap();

        store
    }

    #[test]
    fn openai_bearer_token_injected() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());
        cs.write(state_keys::CREDS_REF_NAME, "e2e-openai".to_string());
        cs.write(state_keys::CREDS_REF_NAMESPACE, "bbr-e2e".to_string());

        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert_eq!(
            req.headers.get("Authorization").unwrap(),
            "Bearer sk-openai-123"
        );
    }

    #[test]
    fn anthropic_x_api_key_injected() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "anthropic".to_string());
        cs.write(state_keys::CREDS_REF_NAME, "e2e-anthropic".to_string());
        cs.write(state_keys::CREDS_REF_NAMESPACE, "bbr-e2e".to_string());

        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert_eq!(req.headers.get("x-api-key").unwrap(), "sk-ant-123");
    }

    #[test]
    fn azure_api_key_injected() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "azure-openai".to_string());
        cs.write(state_keys::CREDS_REF_NAME, "e2e-azure".to_string());
        cs.write(state_keys::CREDS_REF_NAMESPACE, "bbr-e2e".to_string());

        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert_eq!(req.headers.get("api-key").unwrap(), "azure-key-123");
    }

    #[test]
    fn no_provider_is_passthrough() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert!(!req.headers.contains_key("Authorization"));
    }

    #[test]
    fn missing_creds_ref_returns_error() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());
        // no CREDS_REF_NAME

        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        let err = plugin.process_request(&mut cs, &mut req).unwrap_err();
        assert_eq!(err.http_status_code(), 500);
        assert!(err.msg.contains("credentialRef"));
    }

    #[test]
    fn missing_secret_returns_error() {
        let plugin = ApiKeyInjectionPlugin::new(setup_store());
        let mut cs = CycleState::new();
        cs.write(state_keys::PROVIDER, "openai".to_string());
        cs.write(state_keys::CREDS_REF_NAME, "nonexistent".to_string());
        cs.write(state_keys::CREDS_REF_NAMESPACE, "bbr-e2e".to_string());

        let mut req = InferenceRequest::new();
        req.set_body(json!({}));

        let err = plugin.process_request(&mut cs, &mut req).unwrap_err();
        assert!(err.msg.contains("not found"));
    }
}
