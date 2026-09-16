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

/// An entry of the CURRENT Perplexity response "blocks" array.
///
/// Perplexity replaced the legacy `text` steps array with `blocks`. MEASURED
/// 2026-08-12 against a live signed-in browser request to
/// `/rest/sse/perplexity_ask`: the terminal message carries `status:"COMPLETED"`
/// and NO top-level `text` or `answer` key at all — the answer lives at
/// `blocks[].markdown_block.answer` (alongside `chunks`, `progress`,
/// `chunk_starting_offset`), with `intended_usage:"ask_text"`. Because both
/// legacy extraction paths key off fields that no longer exist, every call
/// returned `answer: null` even though the thread was created and answered
/// server-side (verified by opening the returned backend_uuid in the browser).
///
/// CORRECTED 2026-09-16 (gap 2764, live standalone diagnostic capture against
/// the real backend): citations do NOT live inside `markdown_block` -- that
/// was an unverified assumption in the original 2026-08-12 fix, and it left
/// `web_results` permanently empty even after this fix landed. They arrive in
/// a SEPARATE entry of the same `blocks` array, with `intended_usage:
/// "web_results"` and the real list at `web_result_block.web_results`.
#[derive(Deserialize)]
struct ResponseBlock {
    #[serde(default)]
    markdown_block: Option<MarkdownBlock>,
    #[serde(default)]
    web_result_block: Option<WebResultBlock>,
}

/// The markdown answer block of a `blocks` entry (`intended_usage: "ask_text"`).
#[derive(Deserialize)]
struct MarkdownBlock {
    answer: Option<String>,
}

/// The citation-list block of a `blocks` entry (`intended_usage: "web_results"`).
/// VERIFIED LIVE 2026-09-16 against the real backend -- see gap 2764.
#[derive(Deserialize)]
struct WebResultBlock {
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
/// Tries the CURRENT `blocks` shape first, then the legacy FINAL step inside
/// the "text" steps array, then the legacy top-level "answer" field. The legacy
/// paths are retained rather than replaced: they cost one failed lookup each on
/// the current shape, and keeping them means a thread or account still served
/// the old schema does not silently regress to a null answer.
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

/// Pulls answer + web_results from the current `blocks` array.
///
/// Answer and citations arrive as SEPARATE entries in the same array (one
/// `intended_usage: "ask_text"`, one `intended_usage: "web_results"`), so
/// both must be scanned for independently -- finding one is not a signal to
/// stop looking for the other. The first non-empty answer wins: intermediate
/// streamed messages carry the same `markdown_block` shape while still
/// filling in, so an empty string must not be treated as the final answer,
/// or a partial frame would win over the completed one. Web results are
/// similarly taken from the first block that actually carries any.
fn extract_from_blocks(
    content: &Map<String, Value>,
) -> Option<(Option<String>, Vec<SearchWebResult>)> {
    let blocks_value = content.get("blocks")?;
    let blocks: Vec<ResponseBlock> = serde_json::from_value(blocks_value.clone()).ok()?;

    let mut answer: Option<String> = None;
    let mut web_results: Vec<SearchWebResult> = Vec::new();
    let mut found_any = false;

    for block in blocks {
        if answer.is_none() {
            if let Some(markdown) = block.markdown_block {
                if markdown.answer.as_deref().is_some_and(|a| !a.is_empty()) {
                    answer = markdown.answer;
                    found_any = true;
                }
            }
        }
        if web_results.is_empty() {
            if let Some(web_result_block) = block.web_result_block {
                if !web_result_block.web_results.is_empty() {
                    web_results = web_result_block.web_results;
                    found_any = true;
                }
            }
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
    fn test_parse_current_blocks_shape_with_separate_web_results_block() {
        // Trimmed real shape captured live 2026-09-16 (gap 2764) via a standalone
        // diagnostic build against the actual Perplexity backend: citations arrive
        // as a SEPARATE blocks entry (intended_usage: "web_results"), not inside
        // the ask_text block's markdown_block -- the 2026-08-12 fix's own comment
        // ("not present on the ask_text block measured above") already flagged
        // this as unverified, and it was wrong.
        let json = serde_json::json!({
            "status": "COMPLETED",
            "final": true,
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
                                "snippet": "Mayor Josh Morgan took office in 2022."
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
    }

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
    fn test_parse_current_blocks_shape() {
        // VERBATIM SHAPE captured 2026-08-12 from a live signed-in browser POST to
        // /rest/sse/perplexity_ask. Note there is NO top-level "text" and NO top-level
        // "answer" — both legacy paths miss, which is exactly why every call returned
        // answer:null while Perplexity had in fact answered the thread.
        let json = r#"{"status":"COMPLETED","backend_uuid":"1c55f722","blocks":[
            {"intended_usage":"ask_text","markdown_block":{"progress":"DONE","chunks":["PONG"],
             "chunk_starting_offset":0,"answer":"PONG"}}]}"#;
        let event = parse_sse_event(json).unwrap();

        assert_eq!(event.answer, Some("PONG".to_string()));
        assert_eq!(event.backend_uuid, Some("1c55f722".to_string()));
    }

    #[test]
    fn test_partial_block_does_not_win_over_completed_answer() {
        // Streamed frames arrive with the same shape while still filling in. An empty
        // answer must not be accepted, or a partial frame beats the completed one.
        let json = r#"{"blocks":[
            {"intended_usage":"ask_text","markdown_block":{"progress":"IN_PROGRESS","answer":""}},
            {"intended_usage":"ask_text","markdown_block":{"progress":"DONE","answer":"Real answer"}}]}"#;
        let event = parse_sse_event(json).unwrap();

        assert_eq!(event.answer, Some("Real answer".to_string()));
    }

    #[test]
    fn test_legacy_shapes_still_parse() {
        // The legacy paths are retained, not replaced: a thread still served the old
        // schema must not regress to a null answer.
        let legacy = r#"{"answer": "Legacy top-level"}"#;
        assert_eq!(
            parse_sse_event(legacy).unwrap().answer,
            Some("Legacy top-level".to_string())
        );
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
}
