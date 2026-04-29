use std::collections::HashMap;

const API_KEY_FIELD: &str = "api-key";

pub struct SimpleAuthGenerator {
    pub header_name: String,
    pub header_value_prefix: String,
}

impl SimpleAuthGenerator {
    pub fn generate_auth_headers(
        &self,
        credentials: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, String> {
        let api_key = credentials
            .get(API_KEY_FIELD)
            .ok_or_else(|| format!("credentials missing required field {}", API_KEY_FIELD))?;

        let mut headers = HashMap::new();
        headers.insert(
            self.header_name.clone(),
            format!("{}{}", self.header_value_prefix, api_key),
        );
        Ok(headers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_token() {
        let gen = SimpleAuthGenerator {
            header_name: "Authorization".to_string(),
            header_value_prefix: "Bearer ".to_string(),
        };
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "sk-123".to_string());

        let headers = gen.generate_auth_headers(&creds).unwrap();
        assert_eq!(headers["Authorization"], "Bearer sk-123");
    }

    #[test]
    fn raw_key() {
        let gen = SimpleAuthGenerator {
            header_name: "x-api-key".to_string(),
            header_value_prefix: String::new(),
        };
        let mut creds = HashMap::new();
        creds.insert("api-key".to_string(), "anthropic-key".to_string());

        let headers = gen.generate_auth_headers(&creds).unwrap();
        assert_eq!(headers["x-api-key"], "anthropic-key");
    }

    #[test]
    fn missing_api_key_field() {
        let gen = SimpleAuthGenerator {
            header_name: "Authorization".to_string(),
            header_value_prefix: "Bearer ".to_string(),
        };
        let creds = HashMap::new();
        assert!(gen.generate_auth_headers(&creds).is_err());
    }
}
