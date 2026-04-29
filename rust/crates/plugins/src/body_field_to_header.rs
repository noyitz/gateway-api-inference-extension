use ipp_framework::cycle_state::CycleState;
use ipp_framework::error::PluginError;
use ipp_framework::inference_message::InferenceRequest;
use ipp_framework::plugin::RequestProcessor;

pub struct BodyFieldToHeaderPlugin {
    field_name: String,
    header_name: String,
}

impl BodyFieldToHeaderPlugin {
    pub fn new(field_name: &str, header_name: &str) -> Result<Self, PluginError> {
        if field_name.is_empty() {
            return Err(PluginError::bad_request(
                "fieldName is required for body-field-to-header plugin",
            ));
        }
        if header_name.is_empty() {
            return Err(PluginError::bad_request(
                "headerName is required for body-field-to-header plugin",
            ));
        }
        Ok(Self {
            field_name: field_name.to_string(),
            header_name: header_name.to_string(),
        })
    }
}

impl RequestProcessor for BodyFieldToHeaderPlugin {
    fn name(&self) -> &str {
        "body-field-to-header"
    }

    fn process_request(
        &self,
        _cycle_state: &mut CycleState,
        request: &mut InferenceRequest,
    ) -> Result<(), PluginError> {
        let field_value = match request.body.get(&self.field_name) {
            Some(v) => v,
            None => return Ok(()),
        };

        let field_str = match field_value.as_str() {
            Some(s) => s.to_string(),
            None => field_value.to_string().trim_matches('"').to_string(),
        };

        if field_str.is_empty() {
            return Ok(());
        }

        request.set_header(self.header_name.clone(), field_str);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_model_to_header() {
        let plugin = BodyFieldToHeaderPlugin::new("model", "X-Gateway-Model-Name").unwrap();
        let mut cs = CycleState::new();
        let mut req = InferenceRequest::new();
        req.set_body(json!({"model": "gpt-4o", "messages": []}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert_eq!(
            req.headers.get("X-Gateway-Model-Name").unwrap(),
            "gpt-4o"
        );
    }

    #[test]
    fn missing_field_is_noop() {
        let plugin = BodyFieldToHeaderPlugin::new("model", "X-Gateway-Model-Name").unwrap();
        let mut cs = CycleState::new();
        let mut req = InferenceRequest::new();
        req.set_body(json!({"messages": []}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert!(!req.headers.contains_key("X-Gateway-Model-Name"));
    }

    #[test]
    fn empty_field_name_rejected() {
        assert!(BodyFieldToHeaderPlugin::new("", "X-Header").is_err());
    }

    #[test]
    fn empty_header_name_rejected() {
        assert!(BodyFieldToHeaderPlugin::new("model", "").is_err());
    }

    #[test]
    fn numeric_value_converted() {
        let plugin = BodyFieldToHeaderPlugin::new("count", "X-Count").unwrap();
        let mut cs = CycleState::new();
        let mut req = InferenceRequest::new();
        req.set_body(json!({"count": 42}));

        plugin.process_request(&mut cs, &mut req).unwrap();
        assert_eq!(req.headers.get("X-Count").unwrap(), "42");
    }
}
