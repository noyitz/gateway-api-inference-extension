use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use ipp_framework::error::PluginError;
use serde_json::{json, Value};

use super::translator::{TranslateRequestResult, Translator};

const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_PATH: &str = "/v1/messages";
const DEFAULT_MAX_TOKENS: i64 = 4096;

pub struct AnthropicTranslator;

impl Translator for AnthropicTranslator {
    fn translate_request(&self, body: &Value) -> Result<TranslateRequestResult, PluginError> {
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("");
        if model.is_empty() {
            return Err(PluginError::bad_request("model field is required"));
        }

        let messages = extract_messages(body)?;
        let (system_prompt, anthropic_messages) = separate_system_messages(&messages)?;

        if anthropic_messages.is_empty() {
            return Err(PluginError::bad_request(
                "at least one non-system message is required",
            ));
        }

        let max_tokens = get_i64(body, "max_completion_tokens")
            .filter(|&v| v > 0)
            .or_else(|| get_i64(body, "max_tokens").filter(|&v| v > 0))
            .unwrap_or(DEFAULT_MAX_TOKENS);

        let mut translated = json!({
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens,
        });

        if !system_prompt.is_empty() {
            translated["system"] = json!(system_prompt);
        }
        if let Some(temp) = get_f64(body, "temperature") {
            translated["temperature"] = json!(temp);
        }
        if let Some(top_p) = get_f64(body, "top_p") {
            translated["top_p"] = json!(top_p);
        }
        if let Some(stop) = extract_stop_sequences(body) {
            translated["stop_sequences"] = json!(stop);
        }
        if let Some(tools) = translate_tool_definitions(body) {
            translated["tools"] = tools;
            if let Some(tc) = translate_tool_choice(body) {
                translated["tool_choice"] = tc;
            }
        }
        if let Some(stream) = body.get("stream").and_then(Value::as_bool) {
            translated["stream"] = json!(stream);
        }

        let mut headers = HashMap::new();
        headers.insert("anthropic-version".into(), ANTHROPIC_API_VERSION.into());
        headers.insert("content-type".into(), "application/json".into());
        headers.insert(":path".into(), ANTHROPIC_PATH.into());

        Ok(TranslateRequestResult {
            body: Some(translated),
            headers_to_mutate: headers,
            headers_to_remove: Vec::new(),
        })
    }

    fn translate_response(
        &self,
        body: &mut Value,
        model: &str,
    ) -> Result<bool, PluginError> {
        let body_type = body.get("type").and_then(Value::as_str).unwrap_or("");

        if body_type == "error" {
            *body = translate_error(&*body);
            return Ok(true);
        }

        let content = extract_text_content(&*body);
        let finish_reason = map_stop_reason(body.get("stop_reason").and_then(Value::as_str));
        let usage = map_usage(&*body);

        let id = body.get("id").and_then(Value::as_str).unwrap_or("");
        let model = if model.is_empty() {
            body.get("model").and_then(Value::as_str).unwrap_or("")
        } else {
            model
        };

        let mut message = json!({
            "role": "assistant",
            "content": content,
        });

        if finish_reason == "tool_calls" {
            let tool_calls = extract_tool_calls(&*body);
            if !tool_calls.is_empty() {
                message["tool_calls"] = json!(tool_calls);
            }
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        *body = json!({
            "id": id,
            "object": "chat.completion",
            "created": now,
            "model": model,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        });
        Ok(true)
    }
}

fn extract_messages(body: &Value) -> Result<Vec<&Value>, PluginError> {
    let arr = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| PluginError::bad_request("messages field is required"))?;
    if arr.is_empty() {
        return Err(PluginError::bad_request("messages field is required"));
    }
    Ok(arr.iter().collect())
}

fn separate_system_messages(
    messages: &[&Value],
) -> Result<(String, Vec<Value>), PluginError> {
    let mut system_parts: Vec<String> = Vec::new();
    let mut anthropic_messages: Vec<Value> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        let role = msg.get("role").and_then(Value::as_str).unwrap_or("");

        match role {
            "system" | "developer" => {
                system_parts.push(extract_content_string(msg));
            }
            "user" => {
                anthropic_messages.push(json!({
                    "role": "user",
                    "content": extract_content_string(msg),
                }));
            }
            "assistant" => {
                anthropic_messages.push(build_assistant_message(msg));
            }
            "tool" => {
                let tool_result = build_tool_result(msg, i)?;
                append_tool_result(&mut anthropic_messages, tool_result);
            }
            other => {
                return Err(PluginError::bad_request(format!(
                    "message at index {i} has unknown role '{other}'"
                )));
            }
        }
    }

    let system = system_parts.join("\n");
    Ok((system, anthropic_messages))
}

fn extract_content_str<'a>(msg: &'a Value) -> &'a str {
    match msg.get("content") {
        Some(Value::String(s)) => s.as_str(),
        Some(Value::Array(parts)) => {
            // For array content, we can't return &str without allocation,
            // so we fall back. This path is rare.
            ""
        }
        _ => "",
    }
}

fn extract_content_string(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(" ")
        }
        _ => String::new(),
    }
}

fn build_assistant_message(msg: &Value) -> Value {
    let tool_calls = msg.get("tool_calls").and_then(Value::as_array);
    if tool_calls.is_none_or(|tc| tc.is_empty()) {
        return json!({
            "role": "assistant",
            "content": extract_content_string(msg),
        });
    }

    let mut blocks: Vec<Value> = Vec::new();
    let text = extract_content_string(msg);
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }

    for tc in tool_calls.unwrap() {
        let id = tc.get("id").and_then(Value::as_str).unwrap_or("");
        let func = tc.get("function");
        let name = func
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let input = match func.and_then(|f| f.get("arguments")) {
            Some(Value::String(s)) => {
                serde_json::from_str(s).unwrap_or(json!({}))
            }
            Some(v) if !v.is_null() => v.clone(),
            _ => json!({}),
        };
        blocks.push(json!({
            "type": "tool_use",
            "id": id,
            "name": name,
            "input": input,
        }));
    }

    json!({
        "role": "assistant",
        "content": blocks,
    })
}

fn build_tool_result(msg: &Value, index: usize) -> Result<Value, PluginError> {
    let tool_call_id = msg
        .get("tool_call_id")
        .and_then(Value::as_str)
        .unwrap_or("");
    if tool_call_id.is_empty() {
        return Err(PluginError::bad_request(format!(
            "message at index {index}: tool message missing required 'tool_call_id' field"
        )));
    }
    let content = extract_content_string(msg);
    let mut result = json!({
        "type": "tool_result",
        "tool_use_id": tool_call_id,
    });
    if !content.is_empty() {
        result["content"] = json!(content);
    }
    Ok(result)
}

fn append_tool_result(messages: &mut Vec<Value>, tool_result: Value) {
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some("user") {
            if let Some(blocks) = last.get_mut("content").and_then(Value::as_array_mut) {
                blocks.push(tool_result);
                return;
            }
        }
    }
    messages.push(json!({
        "role": "user",
        "content": [tool_result],
    }));
}

fn extract_stop_sequences(body: &Value) -> Option<Vec<String>> {
    match body.get("stop")? {
        Value::String(s) if !s.is_empty() => Some(vec![s.clone()]),
        Value::Array(arr) => {
            let seqs: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            if seqs.is_empty() { None } else { Some(seqs) }
        }
        _ => None,
    }
}

fn translate_tool_definitions(body: &Value) -> Option<Value> {
    let tools = body.get("tools")?.as_array()?;
    if tools.is_empty() {
        return None;
    }

    let anthropic_tools: Vec<Value> = tools
        .iter()
        .filter_map(|t| {
            let func = t.get("function")?;
            let name = func.get("name")?.as_str()?;
            let mut tool = json!({"name": name});
            if let Some(desc) = func.get("description").and_then(Value::as_str) {
                if !desc.is_empty() {
                    tool["description"] = json!(desc);
                }
            }
            tool["input_schema"] = func
                .get("parameters")
                .cloned()
                .unwrap_or(json!({"type": "object"}));
            Some(tool)
        })
        .collect();

    if anthropic_tools.is_empty() {
        None
    } else {
        Some(json!(anthropic_tools))
    }
}

fn translate_tool_choice(body: &Value) -> Option<Value> {
    let tc = body.get("tool_choice")?;
    if let Some(s) = tc.as_str() {
        return match s {
            "auto" => Some(json!({"type": "auto"})),
            "required" => Some(json!({"type": "any"})),
            _ => None,
        };
    }
    if let Some(obj) = tc.as_object() {
        if let Some(name) = obj
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
        {
            return Some(json!({"type": "tool", "name": name}));
        }
    }
    None
}

fn translate_error(body: &Value) -> Value {
    let err = body.get("error");
    let err_type = err
        .and_then(|e| e.get("type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    let err_msg = err
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("");

    json!({
        "error": {
            "message": err_msg,
            "type": err_type,
            "param": null,
            "code": err_type,
        }
    })
}

fn extract_text_content(body: &Value) -> String {
    let blocks = match body.get("content").and_then(Value::as_array) {
        Some(b) => b,
        None => return String::new(),
    };
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("")
}

fn extract_tool_calls(body: &Value) -> Vec<Value> {
    let blocks = match body.get("content").and_then(Value::as_array) {
        Some(b) => b,
        None => return Vec::new(),
    };
    blocks
        .iter()
        .filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_use"))
        .enumerate()
        .map(|(idx, b)| {
            let args = b
                .get("input")
                .map(|v| serde_json::to_string(v).unwrap_or_else(|_| "{}".into()))
                .unwrap_or_else(|| "{}".into());
            json!({
                "id": b.get("id").and_then(Value::as_str).unwrap_or(""),
                "index": idx,
                "type": "function",
                "function": {
                    "name": b.get("name").and_then(Value::as_str).unwrap_or(""),
                    "arguments": args,
                }
            })
        })
        .collect()
}

fn map_stop_reason(reason: Option<&str>) -> &str {
    match reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        _ => "stop",
    }
}

fn map_usage(body: &Value) -> Value {
    match body.get("usage") {
        Some(u) => {
            let input = u.get("input_tokens").and_then(Value::as_i64).unwrap_or(0);
            let output = u.get("output_tokens").and_then(Value::as_i64).unwrap_or(0);
            json!({
                "prompt_tokens": input,
                "completion_tokens": output,
                "total_tokens": input + output,
            })
        }
        None => json!({
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }),
    }
}

fn get_f64(body: &Value, key: &str) -> Option<f64> {
    body.get(key)?.as_f64()
}

fn get_i64(body: &Value, key: &str) -> Option<i64> {
    body.get(key)?.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basic_chat_request() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        let translated = result.body.unwrap();

        assert_eq!(translated["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(translated["messages"][0]["role"], "user");
        assert_eq!(translated["messages"][0]["content"], "Hello");
        assert_eq!(translated["max_tokens"], DEFAULT_MAX_TOKENS);
        assert!(translated.get("system").is_none());
        assert_eq!(result.headers_to_mutate[":path"], "/v1/messages");
        assert_eq!(result.headers_to_mutate["anthropic-version"], "2023-06-01");
    }

    #[test]
    fn system_message_separation() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"}
            ]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        let translated = result.body.unwrap();

        assert_eq!(translated["system"], "You are helpful");
        assert_eq!(translated["messages"].as_array().unwrap().len(), 1);
        assert_eq!(translated["messages"][0]["role"], "user");
    }

    #[test]
    fn developer_role_maps_to_system() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "developer", "content": "Be concise"},
                {"role": "user", "content": "Hi"}
            ]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        let translated = result.body.unwrap();
        assert_eq!(translated["system"], "Be concise");
    }

    #[test]
    fn multiple_system_messages_concatenated() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "Part 1"},
                {"role": "developer", "content": "Part 2"},
                {"role": "user", "content": "Hi"}
            ]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        assert_eq!(result.body.unwrap()["system"], "Part 1\nPart 2");
    }

    #[test]
    fn max_completion_tokens_priority() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_completion_tokens": 100,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        assert_eq!(result.body.unwrap()["max_tokens"], 100);
    }

    #[test]
    fn max_tokens_fallback() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        assert_eq!(
            AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["max_tokens"],
            200
        );
    }

    #[test]
    fn optional_parameters_forwarded() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["END"],
            "messages": [{"role": "user", "content": "Hi"}]
        });
        let translated = AnthropicTranslator.translate_request(&body).unwrap().body.unwrap();
        assert_eq!(translated["temperature"], 0.7);
        assert_eq!(translated["top_p"], 0.9);
        assert_eq!(translated["stop_sequences"], json!(["END"]));
    }

    #[test]
    fn stream_forwarded() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "stream": true,
            "messages": [{"role": "user", "content": "Hi"}]
        });
        assert_eq!(
            AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["stream"],
            true
        );
    }

    #[test]
    fn tool_definitions_translated() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
                }
            }]
        });
        let translated = AnthropicTranslator.translate_request(&body).unwrap().body.unwrap();
        let tools = translated["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert!(tools[0].get("input_schema").is_some());
    }

    #[test]
    fn tool_choice_auto() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "auto"
        });
        assert_eq!(
            AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["tool_choice"]["type"],
            "auto"
        );
    }

    #[test]
    fn tool_choice_required() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "required"
        });
        assert_eq!(
            AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["tool_choice"]["type"],
            "any"
        );
    }

    #[test]
    fn tool_choice_specific_function() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        });
        let tc = &AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["tool_choice"];
        assert_eq!(tc["type"], "tool");
        assert_eq!(tc["name"], "get_weather");
    }

    #[test]
    fn assistant_with_tool_calls() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "72°F"}
            ]
        });
        let msgs = AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["messages"]
            .as_array()
            .unwrap()
            .clone();

        assert_eq!(msgs[1]["role"], "assistant");
        let blocks = msgs[1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "get_weather");

        assert_eq!(msgs[2]["role"], "user");
        let content = msgs[2]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_1");
    }

    #[test]
    fn multiple_tool_results_merged() {
        let body = json!({
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                        {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{}"}}
                    ]
                },
                {"role": "tool", "tool_call_id": "c1", "content": "r1"},
                {"role": "tool", "tool_call_id": "c2", "content": "r2"}
            ]
        });
        let msgs = AnthropicTranslator.translate_request(&body).unwrap().body.unwrap()["messages"]
            .as_array()
            .unwrap()
            .clone();

        assert_eq!(msgs.len(), 3);
        let tool_results = msgs[2]["content"].as_array().unwrap();
        assert_eq!(tool_results.len(), 2);
        assert_eq!(tool_results[0]["tool_use_id"], "c1");
        assert_eq!(tool_results[1]["tool_use_id"], "c2");
    }

    #[test]
    fn missing_model_error() {
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        assert!(AnthropicTranslator.translate_request(&body).is_err());
    }

    #[test]
    fn empty_messages_error() {
        let body = json!({"model": "claude", "messages": []});
        assert!(AnthropicTranslator.translate_request(&body).is_err());
    }

    #[test]
    fn unknown_role_error() {
        let body = json!({
            "model": "claude",
            "messages": [{"role": "custom_role", "content": "Hi"}]
        });
        let err = AnthropicTranslator.translate_request(&body).unwrap_err();
        assert!(err.msg.contains("unknown role"));
    }

    #[test]
    fn tool_missing_tool_call_id_error() {
        let body = json!({
            "model": "claude",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "tool", "content": "result"}
            ]
        });
        let err = AnthropicTranslator.translate_request(&body).unwrap_err();
        assert!(err.msg.contains("tool_call_id"));
    }

    #[test]
    fn basic_response_translation() {
        let mut body = json!({
            "id": "msg_123",
            "type": "message",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let mutated = AnthropicTranslator
            .translate_response(&mut body, "claude-3-5-sonnet-20241022")
            .unwrap();
        assert!(mutated);

        assert_eq!(body["id"], "msg_123");
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["model"], "claude-3-5-sonnet-20241022");
        assert_eq!(body["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["prompt_tokens"], 10);
        assert_eq!(body["usage"]["completion_tokens"], 5);
        assert_eq!(body["usage"]["total_tokens"], 15);
    }

    #[test]
    fn response_max_tokens_stop_reason() {
        let mut body = json!({
            "id": "msg_1", "type": "message",
            "content": [{"type": "text", "text": "Partial"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100}
        });
        AnthropicTranslator.translate_response(&mut body, "claude").unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn response_tool_use() {
        let mut body = json!({
            "id": "msg_1", "type": "message",
            "content": [
                {"type": "text", "text": "I'll check."},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "NYC"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        AnthropicTranslator.translate_response(&mut body, "claude").unwrap();
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        let tc = body["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0]["function"]["name"], "get_weather");
    }

    #[test]
    fn response_error_translation() {
        let mut body = json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "model not found"}
        });
        AnthropicTranslator.translate_response(&mut body, "claude").unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert_eq!(body["error"]["message"], "model not found");
    }

    #[test]
    fn response_model_from_body_when_empty() {
        let mut body = json!({
            "id": "msg_1", "type": "message", "model": "claude-3-haiku",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        });
        AnthropicTranslator.translate_response(&mut body, "").unwrap();
        assert_eq!(body["model"], "claude-3-haiku");
    }

    #[test]
    fn content_parts_array() {
        let body = json!({
            "model": "claude",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "World"}
                ]
            }]
        });
        let result = AnthropicTranslator.translate_request(&body).unwrap();
        let msgs = result.body.unwrap();
        assert_eq!(msgs["messages"][0]["content"], "Hello World");
    }
}
