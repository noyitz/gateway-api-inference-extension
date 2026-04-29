use std::collections::HashMap;

use ipp_framework::error::PluginError;
use serde_json::Value;

#[derive(Debug)]
pub struct TranslateRequestResult {
    pub body: Option<Value>,
    pub headers_to_mutate: HashMap<String, String>,
    pub headers_to_remove: Vec<String>,
}

pub trait Translator: Send + Sync {
    fn translate_request(&self, body: &Value) -> Result<TranslateRequestResult, PluginError>;

    fn translate_response(&self, body: &mut Value, model: &str) -> Result<bool, PluginError>;
}
