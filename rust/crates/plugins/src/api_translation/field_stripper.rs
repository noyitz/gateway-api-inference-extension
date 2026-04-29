use serde_json::Value;

pub struct ResponseFieldStripper {
    field_paths: Vec<FieldPath>,
}

type FieldPath = Vec<FieldSegment>;

struct FieldSegment {
    key: String,
    is_array: bool,
}

impl ResponseFieldStripper {
    pub fn new(fields_to_strip: &[&str]) -> Self {
        Self {
            field_paths: parse_field_paths(fields_to_strip),
        }
    }

    pub fn would_strip(&self, body: &Value) -> bool {
        let obj = match body.as_object() {
            Some(obj) => obj,
            None => return false,
        };
        self.field_paths
            .iter()
            .any(|fp| check_field_exists(obj, fp, 0))
    }

    pub fn strip(&self, body: &mut Value) -> bool {
        let obj = match body.as_object_mut() {
            Some(obj) => obj,
            None => return false,
        };

        let mut mutated = false;
        for fp in &self.field_paths {
            if strip_field(obj, fp, 0) {
                mutated = true;
            }
        }
        mutated
    }
}

fn parse_field_paths(raw: &[&str]) -> Vec<FieldPath> {
    raw.iter()
        .map(|r| {
            r.split('.')
                .map(|p| {
                    if let Some(key) = p.strip_suffix("[]") {
                        FieldSegment {
                            key: key.to_string(),
                            is_array: true,
                        }
                    } else {
                        FieldSegment {
                            key: p.to_string(),
                            is_array: false,
                        }
                    }
                })
                .collect()
        })
        .collect()
}

fn check_field_exists(
    obj: &serde_json::Map<String, Value>,
    path: &[FieldSegment],
    idx: usize,
) -> bool {
    if idx >= path.len() {
        return false;
    }
    let seg = &path[idx];
    let is_last = idx == path.len() - 1;

    if is_last {
        return obj.contains_key(&seg.key);
    }
    if seg.is_array {
        return match obj.get(&seg.key).and_then(Value::as_array) {
            Some(arr) => arr.iter().any(|elem| {
                elem.as_object()
                    .map(|m| check_field_exists(m, path, idx + 1))
                    .unwrap_or(false)
            }),
            None => false,
        };
    }
    match obj.get(&seg.key).and_then(Value::as_object) {
        Some(child) => check_field_exists(child, path, idx + 1),
        None => false,
    }
}

fn strip_field(
    obj: &mut serde_json::Map<String, Value>,
    path: &[FieldSegment],
    idx: usize,
) -> bool {
    if idx >= path.len() {
        return false;
    }

    let seg = &path[idx];
    let is_last = idx == path.len() - 1;

    if is_last {
        return obj.remove(&seg.key).is_some();
    }

    if seg.is_array {
        let arr = match obj.get_mut(&seg.key).and_then(Value::as_array_mut) {
            Some(arr) => arr,
            None => return false,
        };
        let mut mutated = false;
        for elem in arr.iter_mut() {
            if let Some(m) = elem.as_object_mut() {
                if strip_field(m, path, idx + 1) {
                    mutated = true;
                }
            }
        }
        return mutated;
    }

    if !obj.contains_key(&seg.key) {
        return false;
    }
    match obj.get_mut(&seg.key).and_then(Value::as_object_mut) {
        Some(child) => strip_field(child, path, idx + 1),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn strip_top_level_field() {
        let stripper = ResponseFieldStripper::new(&["prompt_filter_results"]);
        let mut body = json!({
            "id": "123",
            "prompt_filter_results": [{"index": 0}],
            "choices": []
        });

        assert!(stripper.strip(&mut body));
        assert!(body.get("prompt_filter_results").is_none());
        assert!(body.get("id").is_some());
    }

    #[test]
    fn strip_array_element_field() {
        let stripper = ResponseFieldStripper::new(&["choices[].content_filter_results"]);
        let mut body = json!({
            "choices": [
                {"index": 0, "content_filter_results": {"hate": "safe"}},
                {"index": 1, "content_filter_results": {"hate": "safe"}}
            ]
        });

        assert!(stripper.strip(&mut body));
        let choices = body["choices"].as_array().unwrap();
        for c in choices {
            assert!(c.get("content_filter_results").is_none());
            assert!(c.get("index").is_some());
        }
    }

    #[test]
    fn strip_nested_field() {
        let stripper = ResponseFieldStripper::new(&["usage.extra_properties"]);
        let mut body = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "extra_properties": {"cached": true}
            }
        });

        assert!(stripper.strip(&mut body));
        assert!(body["usage"].get("extra_properties").is_none());
        assert_eq!(body["usage"]["prompt_tokens"], 10);
    }

    #[test]
    fn no_mutation_when_field_absent() {
        let stripper = ResponseFieldStripper::new(&["nonexistent"]);
        let mut body = json!({"id": "123"});

        assert!(!stripper.strip(&mut body));
    }

    #[test]
    fn multiple_paths() {
        let stripper = ResponseFieldStripper::new(&[
            "prompt_filter_results",
            "choices[].content_filter_results",
        ]);
        let mut body = json!({
            "prompt_filter_results": [],
            "choices": [{"content_filter_results": {}, "index": 0}]
        });

        assert!(stripper.strip(&mut body));
        assert!(body.get("prompt_filter_results").is_none());
        assert!(body["choices"][0].get("content_filter_results").is_none());
    }

    #[test]
    fn empty_paths_is_noop() {
        let stripper = ResponseFieldStripper::new(&[]);
        let mut body = json!({"id": "123"});
        assert!(!stripper.strip(&mut body));
    }
}
