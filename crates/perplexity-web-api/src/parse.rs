use crate::error::{Error, Result};
use crate::types::{SearchEvent, SearchWebResult};
use serde::Deserialize;
use serde_json::{Map, Value};

/// A step in the Perplexity response "text" array.
#[derive(Deserialize)]
struct TextStep {
    step_type: String,
    #[serde(default)]
    content: StepContent,
}

/// Content of a single response step.
#[derive(Deserialize, Default)]
struct StepContent {
    /// For FINAL steps, a JSON-encoded string containing the answer and web_results.
    answer: Option<String>,
}

/// The decoded payload of a FINAL step's "answer" JSON string.
#[derive(Deserialize)]
struct FinalAnswerData {
    answer: Option<String>,
    #[serde(default)]
    web_results: Vec<SearchWebResult>,
}

/// Parses an SSE event JSON string into a SearchEvent.
pub(crate) fn parse_sse_event(json_str: &str) -> Result<SearchEvent> {
    let mut content: Map<String, Value> =
        serde_json::from_str(json_str).map_err(Error::Json)?;

    // If the "text" field is a JSON string, expand it in-place so the full
    // parsed structure is available in `raw`.
    expand_text_field(&mut content);

    let (answer, web_results) = extract_answer_and_web_results(&content);
    let backend_uuid = extract_string(&content, "backend_uuid");
    let attachments = extract_string_array(&content, "attachments");
    let raw = Value::Object(content);

    Ok(SearchEvent { answer, web_results, backend_uuid, attachments, raw })
}

/// If the "text" field is a JSON string, replace it with the parsed value.
fn expand_text_field(content: &mut Map<String, Value>) {
    let parsed = match content.get("text").and_then(|v| v.as_str()) {
        Some(s) => serde_json::from_str::<Value>(s).ok(),
        None => None,
    };
    if let Some(v) = parsed {
        content.insert("text".to_string(), v);
    }
}

/// Extracts answer and web_results from the event content.
///
/// Tries the current live wire shape first (a top-level "blocks" array; see
/// `extract_from_blocks`), then the older "text" steps array (see
/// `extract_from_final_step`, kept for backward compatibility with any
/// endpoint that still emits it), then falls back to a bare top-level
/// "answer" field (which carries no web_results).
fn extract_answer_and_web_results(
    content: &Map<String, Value>,
) -> (Option<String>, Vec<SearchWebResult>) {
    if let Some(result) = extract_from_blocks(content) {
        return result;
    }
    if let Some(result) = extract_from_final_step(content) {
        return result;
    }
    (extract_string(content, "answer"), Vec::new())
}

/// Extracts answer and web_results from the live Perplexity wire format: a
/// top-level "blocks" array where each entry carries `intended_usage`, and
/// either `markdown_block.answer` (for `intended_usage == "ask_text"`, only
/// present once that block's `progress` reaches "DONE") or
/// `web_result_block.web_results` (for `intended_usage == "web_results"`).
///
/// VERIFIED LIVE 2026-09-16 (gap 2764) against the real Perplexity backend via
/// a standalone diagnostic capture: the SSE final event no longer contains any
/// "text" field at all, so `extract_from_final_step` always returned `None`
/// and every production `ask`/`search` response silently lost its citations
/// via the old top-level-"answer"-with-empty-Vec fallback -- even though the
/// answer text itself often contained inline citation markers like `[1][2]`
/// referencing a source list Perplexity was sending in this very `blocks`
/// array the whole time.
///
/// Returns `None` when the event has no "blocks" array or none of its entries
/// carry usable answer/web_results content, so callers can fall through to
/// the older extraction paths.
fn extract_from_blocks(
    content: &Map<String, Value>,
) -> Option<(Option<String>, Vec<SearchWebResult>)> {
    let blocks = content.get("blocks")?.as_array()?;

    let mut answer: Option<String> = None;
    let mut web_results: Vec<SearchWebResult> = Vec::new();
    let mut found_any = false;

    for block in blocks {
        match block.get("intended_usage").and_then(|v| v.as_str()) {
            Some("ask_text") => {
                if let Some(a) = block
                    .get("markdown_block")
                    .and_then(|m| m.get("answer"))
                    .and_then(|v| v.as_str())
                {
                    answer = Some(a.to_owned());
                    found_any = true;
                }
            }
            Some("web_results") => {
                if let Some(results) = block
                    .get("web_result_block")
                    .and_then(|w| w.get("web_results"))
                {
                    if let Ok(parsed) =
                        serde_json::from_value::<Vec<SearchWebResult>>(results.clone())
                    {
                        web_results = parsed;
                        found_any = true;
                    }
                }
            }
            _ => {}
        }
    }

    if found_any {
        Some((answer, web_results))
    } else {
        None
    }
}

/// Deserializes the "text" steps array and pulls answer + web_results from the
/// FINAL step. Returns `None` when no FINAL step exists or parsing fails.
fn extract_from_final_step(
    content: &Map<String, Value>,
) -> Option<(Option<String>, Vec<SearchWebResult>)> {
    let text_value = content.get("text")?;

    let steps: Vec<TextStep> = serde_json::from_value(text_value.clone()).ok()?;

    let final_step = steps.into_iter().find(|s| s.step_type == "FINAL")?;
    let answer_json = final_step.content.answer?;

    let data: FinalAnswerData = serde_json::from_str(&answer_json).ok()?;
    Some((data.answer, data.web_results))
}

/// Extracts a string value from the content map.
fn extract_string(content: &Map<String, Value>, key: &str) -> Option<String> {
    content.get(key).and_then(|v| v.as_str()).map(str::to_owned)
}

/// Extracts an array of strings from the content map.
fn extract_string_array(content: &Map<String, Value>, key: &str) -> Vec<String> {
    content
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(str::to_owned)).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_event() {
        let json = r#"{"answer": "Hello world"}"#;
        let event = parse_sse_event(json).unwrap();

        assert_eq!(event.answer, Some("Hello world".to_string()));
        assert!(event.web_results.is_empty());
        assert!(event.backend_uuid.is_none());
        assert!(event.attachments.is_empty());
    }

    #[test]
    fn test_parse_event_with_backend_uuid() {
        let json = r#"{"answer": "Test", "backend_uuid": "abc-123"}"#;
        let event = parse_sse_event(json).unwrap();

        assert_eq!(event.answer, Some("Test".to_string()));
        assert_eq!(event.backend_uuid, Some("abc-123".to_string()));
    }

    #[test]
    fn test_parse_event_with_attachments() {
        let json = r#"{"answer": "Test", "attachments": ["url1", "url2"]}"#;
        let event = parse_sse_event(json).unwrap();

        assert_eq!(event.attachments, vec!["url1", "url2"]);
    }

    #[test]
    fn test_parse_event_with_nested_text_json() {
        // Simulates the "text" field containing JSON string with steps
        let inner_answer = r#"{"answer": "Nested answer", "web_results": [{"name": "Source", "url": "https://example.com", "snippet": "Example"}]}"#;
        let text_content = serde_json::json!([
            {
                "step_type": "SEARCH",
                "content": {}
            },
            {
                "step_type": "FINAL",
                "content": {
                    "answer": inner_answer
                }
            }
        ]);
        let text_str = serde_json::to_string(&text_content).unwrap();

        let json = serde_json::json!({
            "text": text_str,
            "some_field": "value"
        });

        let event = parse_sse_event(&json.to_string()).unwrap();

        assert_eq!(event.answer, Some("Nested answer".to_string()));
        assert_eq!(event.web_results.len(), 1);
        assert_eq!(event.web_results[0].name, "Source");
        assert_eq!(event.web_results[0].url, "https://example.com");
        assert_eq!(event.web_results[0].snippet, "Example");
        // The "text" field should be parsed and stored in raw
        assert!(event.raw.get("text").is_some());
        assert!(event.raw.get("some_field").is_some());
    }

    #[test]
    fn test_parse_event_fallback_to_top_level() {
        // When text doesn't contain FINAL step, fall back to top-level
        let text_content = serde_json::json!([
            {
                "step_type": "SEARCH",
                "content": {}
            }
        ]);
        let text_str = serde_json::to_string(&text_content).unwrap();

        let json = serde_json::json!({
            "text": text_str,
            "answer": "Top level answer"
        });

        let event = parse_sse_event(&json.to_string()).unwrap();

        assert_eq!(event.answer, Some("Top level answer".to_string()));
        assert!(event.web_results.is_empty());
    }

    #[test]
    fn test_parse_event_raw_contains_all_keys() {
        let json = r#"{
            "answer": "Test",
            "backend_uuid": "uuid",
            "attachments": [],
            "extra_field": "should be in raw",
            "another": 123
        }"#;
        let event = parse_sse_event(json).unwrap();

        // All keys, including extracted ones, are present in raw
        assert!(event.raw.get("answer").is_some());
        assert!(event.raw.get("backend_uuid").is_some());
        assert!(event.raw.get("attachments").is_some());
        assert!(event.raw.get("extra_field").is_some());
        assert!(event.raw.get("another").is_some());
    }

    #[test]
    fn test_parse_event_empty_fields() {
        let json = r#"{}"#;
        let event = parse_sse_event(json).unwrap();

        assert!(event.answer.is_none());
        assert!(event.web_results.is_empty());
        assert!(event.backend_uuid.is_none());
        assert!(event.attachments.is_empty());
    }

    #[test]
    fn test_parse_invalid_json() {
        let result = parse_sse_event("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_event_live_blocks_shape_with_web_results() {
        // Trimmed real shape captured live 2026-09-16 (gap 2764) from the actual
        // Perplexity backend -- the current final SSE event has NO "text" field
        // at all; answer and citations arrive as separate entries in a
        // top-level "blocks" array instead.
        let json = serde_json::json!({
            "status": "COMPLETED",
            "final": true,
            "final_sse_message": true,
            "blocks": [
                {
                    "intended_usage": "ask_text",
                    "markdown_block": {
                        "progress": "DONE",
                        "answer": "Josh Morgan is the mayor of London, Ontario."
                    }
                },
                {
                    "intended_usage": "web_results",
                    "web_result_block": {
                        "progress": "DONE",
                        "web_results": [
                            {
                                "name": "Mayor Josh Morgan - City of London",
                                "url": "https://london.ca/government/mayor-josh-morgan",
                                "snippet": "Mayor Josh Morgan took office in 2022.",
                                "timestamp": "2022-10-24T00:00:00",
                                "meta_data": {"citation_domain_name": "london", "client": "web", "images": []},
                                "trust": {"level": 3, "name": "trusted", "description": "government", "type_of_trust": "government"}
                            }
                        ]
                    }
                }
            ]
        });

        let event = parse_sse_event(&json.to_string()).unwrap();

        assert_eq!(event.answer, Some("Josh Morgan is the mayor of London, Ontario.".to_string()));
        assert_eq!(event.web_results.len(), 1);
        assert_eq!(event.web_results[0].name, "Mayor Josh Morgan - City of London");
        assert_eq!(event.web_results[0].url, "https://london.ca/government/mayor-josh-morgan");
        assert_eq!(event.web_results[0].snippet, "Mayor Josh Morgan took office in 2022.");
    }

    #[test]
    fn test_parse_event_live_blocks_shape_intermediate_streaming_chunk() {
        // An in-progress streaming chunk: an "ask_text" block with partial
        // "chunks" but no "answer" yet, and no "web_results" block at all.
        // Must not panic or fabricate content -- both fields stay empty.
        let json = serde_json::json!({
            "status": "PENDING",
            "blocks": [
                {
                    "intended_usage": "ask_text",
                    "markdown_block": {
                        "progress": "IN_PROGRESS",
                        "chunks": ["Jo"],
                        "chunk_starting_offset": 0
                    }
                }
            ]
        });

        let event = parse_sse_event(&json.to_string()).unwrap();

        assert!(event.answer.is_none());
        assert!(event.web_results.is_empty());
    }
}
