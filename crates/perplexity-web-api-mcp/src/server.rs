use base64::Engine as _;
use perplexity_web_api::{
    Client, FollowUpContext, ModelPreference, ReasonModel, SearchMode, SearchModel,
    SearchRequest, SearchResponse, SearchWebResult, Source, UploadFile,
};
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::wrapper::Parameters,
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

/// A file to attach to the query for document analysis.
/// Requires authentication tokens. Provide either `text` or `data`, not both.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FileAttachment {
    /// Filename with extension, e.g. "report.pdf" or "notes.txt".
    pub filename: String,

    /// Plain-text file content. Use for text files (.txt, .md, .csv, .json, source code).
    /// Mutually exclusive with `data`.
    #[serde(default)]
    pub text: Option<String>,

    /// Base64-encoded binary file content. Use for binary files (.pdf, .docx, images).
    /// Mutually exclusive with `text`.
    #[serde(default)]
    pub data: Option<String>,
}

/// Request parameters for `perplexity_search` (no file attachments).
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PerplexitySearchRequest {
    /// The search query or question to ask.
    pub query: String,

    /// Information sources to search. Valid values: "web", "scholar", "social".
    /// Defaults to ["web"] if not specified.
    #[serde(default)]
    pub sources: Option<Vec<String>>,

    /// Language code (ISO 639), e.g., "en-US". Defaults to "en-US".
    #[serde(default)]
    pub language: Option<String>,

    /// Backend UUID from a previous response, to continue that thread.
    #[serde(default)]
    pub follow_up_backend_uuid: Option<String>,
}

/// Request parameters for AI-powered tools that support file attachments.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PerplexityRequest {
    /// The search query or question to ask.
    pub query: String,

    /// Information sources to search. Valid values: "web", "scholar", "social".
    /// Defaults to ["web"] if not specified.
    #[serde(default)]
    pub sources: Option<Vec<String>>,

    /// Language code (ISO 639), e.g., "en-US". Defaults to "en-US".
    #[serde(default)]
    pub language: Option<String>,

    /// Optional file attachments for document analysis.
    /// Requires authentication tokens (PERPLEXITY_SESSION_TOKEN + PERPLEXITY_CSRF_TOKEN).
    /// Each entry needs `filename` and either `text` (plain text) or `data` (base64 binary).
    #[serde(default)]
    pub files: Option<Vec<FileAttachment>>,

    /// Backend UUID from a previous response, to continue that thread.
    #[serde(default)]
    pub follow_up_backend_uuid: Option<String>,
}

impl From<PerplexitySearchRequest> for PerplexityRequest {
    fn from(r: PerplexitySearchRequest) -> Self {
        Self {
            query: r.query,
            sources: r.sources,
            language: r.language,
            files: None,
            follow_up_backend_uuid: r.follow_up_backend_uuid,
        }
    }
}

/// Response from Perplexity tools.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct PerplexityResponse {
    /// The generated answer text.
    pub answer: Option<String>,

    /// Web search results/sources from the response.
    pub web_results: Vec<SearchWebResult>,

    /// Context for making follow-up queries.
    pub follow_up: FollowUpInfo,
}

/// Follow-up context information.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FollowUpInfo {
    /// Backend UUID for follow-up queries.
    pub backend_uuid: Option<String>,

    /// Attachment URLs from the response.
    pub attachments: Vec<String>,
}

/// Search-only response containing just links, titles, and snippets.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SearchOnlyResponse {
    /// Web search results with titles, URLs, and snippets.
    pub web_results: Vec<SearchWebResult>,
}

/// A generated image returned without exposing Perplexity's signed media URL.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct GeneratedImageResponse {
    pub answer: Option<String>,
    pub filename: String,
    pub mime_type: String,
    pub data: String,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub backend_uuid: Option<String>,
}

/// MCP server wrapping Perplexity AI client.
#[derive(Clone)]
pub struct PerplexityServer {
    client: Client,
    ask_model: Option<SearchModel>,
    reason_model: Option<ReasonModel>,
    tokenless: bool,
    incognito: bool,
    collection_uuid: Option<String>,
}

fn to_json_tool_result(value: &impl Serialize) -> Result<CallToolResult, McpError> {
    let json = serde_json::to_string_pretty(value).map_err(|e| {
        McpError::internal_error(format!("JSON serialization error: {}", e), None)
    })?;
    Ok(CallToolResult::success(vec![Content::text(json)]))
}

impl PerplexityServer {
    /// Creates a new server instance with the given Perplexity client.
    ///
    /// When `tokenless` is `true`, only `perplexity_search` and `perplexity_ask`
    /// (both with the `turbo` model) are registered. The `perplexity_research` and
    /// `perplexity_reason` tools require authenticated session cookies and are
    /// removed from the router.
    pub fn new(
        client: Client,
        ask_model: Option<SearchModel>,
        reason_model: Option<ReasonModel>,
        tokenless: bool,
        incognito: bool,
        collection_uuid: Option<String>,
    ) -> Self {
        Self { client, ask_model, reason_model, tokenless, incognito, collection_uuid }
    }

    /// Converts a `FileAttachment` from tool parameters into an `UploadFile`.
    fn convert_attachment(attachment: FileAttachment) -> Result<UploadFile, McpError> {
        match (attachment.text, attachment.data) {
            (Some(text), None) => Ok(UploadFile::from_text(attachment.filename, text)),
            (None, Some(b64)) => {
                let bytes =
                    base64::engine::general_purpose::STANDARD.decode(&b64).map_err(|e| {
                        McpError::invalid_params(
                            format!(
                                "Failed to decode base64 data for '{}': {}",
                                attachment.filename, e
                            ),
                            None,
                        )
                    })?;
                Ok(UploadFile::from_bytes(attachment.filename, bytes))
            }
            (Some(_), Some(_)) => Err(McpError::invalid_params(
                format!(
                    "File '{}' has both `text` and `data` set; provide only one.",
                    attachment.filename
                ),
                None,
            )),
            (None, None) => Err(McpError::invalid_params(
                format!(
                    "File '{}' has neither `text` nor `data` set; provide one.",
                    attachment.filename
                ),
                None,
            )),
        }
    }

    /// Helper to execute a search with the given mode.
    ///
    /// When `files_allowed` is `false`, the method rejects any request that
    /// contains file attachments with a clear error before doing anything else.
    async fn execute_search(
        &self,
        params: PerplexityRequest,
        mode: SearchMode,
        model_preference: Option<ModelPreference>,
        files_allowed: bool,
    ) -> Result<SearchResponse, McpError> {
        let files: Vec<UploadFile> = if let Some(attachments) = params.files {
            if !attachments.is_empty() {
                if !files_allowed {
                    return Err(McpError::invalid_params(
                        "This tool does not support file attachments. \
                         Use perplexity_ask, perplexity_research, or perplexity_reason instead.",
                        None,
                    ));
                }
                if self.tokenless {
                    return Err(McpError::invalid_params(
                        "File attachments require authentication tokens. \
                         Set PERPLEXITY_SESSION_TOKEN and PERPLEXITY_CSRF_TOKEN.",
                        None,
                    ));
                }
                attachments
                    .into_iter()
                    .map(Self::convert_attachment)
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let has_files = !files.is_empty();

        let effective_mode =
            if mode == SearchMode::Auto && (model_preference.is_some() || has_files) {
                SearchMode::Pro
            } else {
                mode
            };

        let mut request =
            SearchRequest::new(&params.query).mode(effective_mode).incognito(self.incognito);
        if let Some(ref uuid) = self.collection_uuid {
            request = request.collection_uuid(uuid.clone());
        }

        if let Some(model_preference) = model_preference {
            request = request.model(model_preference);
        }

        for file in files {
            request = request.file(file);
        }

        if let Some(sources) = params.sources
            && !sources.is_empty()
        {
            let parsed_sources: Vec<Source> =
                sources.iter().filter_map(|s| s.parse::<Source>().ok()).collect();
            if !parsed_sources.is_empty() {
                request = request.sources(parsed_sources);
            }
        }

        if let Some(language) = params.language {
            request = request.language(language);
        }

        if let Some(uuid) = params.follow_up_backend_uuid {
            request = request.follow_up(FollowUpContext {
                backend_uuid: Some(uuid),
                attachments: Vec::new(),
            });
        }

        self.client.search(request).await.map_err(|e| {
            McpError::internal_error(format!("Perplexity API error: {}", e), None)
        })
    }

    async fn do_search(
        &self,
        params: PerplexityRequest,
        mode: SearchMode,
        model_preference: Option<ModelPreference>,
        files_allowed: bool,
    ) -> Result<PerplexityResponse, McpError> {
        let response =
            self.execute_search(params, mode, model_preference, files_allowed).await?;
        let perplexity_web_api::SearchResponse { answer, web_results, follow_up, .. } =
            response;

        Ok(PerplexityResponse {
            answer,
            web_results,
            follow_up: FollowUpInfo {
                backend_uuid: follow_up.backend_uuid,
                attachments: follow_up.attachments,
            },
        })
    }

    fn generated_image_metadata(
        raw: &serde_json::Value,
    ) -> Result<(String, String, Option<u64>, Option<u64>), McpError> {
        let media = raw
            .get("media_items")
            .and_then(serde_json::Value::as_array)
            .and_then(|items| {
                items.iter().find(|item| {
                    item.get("medium").and_then(serde_json::Value::as_str) == Some("image")
                        && item.get("generated_media_metadata").is_some()
                })
            })
            .ok_or_else(|| {
                McpError::internal_error(
                    "Perplexity completed without a generated image asset".to_string(),
                    None,
                )
            })?;
        let url = media.get("image").and_then(serde_json::Value::as_str).ok_or_else(|| {
            McpError::internal_error(
                "Perplexity generated image metadata without an image URL".to_string(),
                None,
            )
        })?;
        let name = media
            .get("name")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("perplexity-generated-image");
        let filename = format!("{}.png", sanitize_filename(name));
        Ok((
            url.to_string(),
            filename,
            media.get("image_width").and_then(serde_json::Value::as_u64),
            media.get("image_height").and_then(serde_json::Value::as_u64),
        ))
    }
}

fn sanitize_filename(value: &str) -> String {
    let clean: String = value
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .collect();
    let clean = clean.trim_matches('-');
    if clean.is_empty() { "perplexity-generated-image".to_string() } else { clean.to_string() }
}

#[tool_router]
impl PerplexityServer {
    /// Quick web search returning only links, titles, and snippets.
    ///
    /// Always uses the turbo model. No generated answer is included.
    #[tool(
        name = "perplexity_search",
        description = "Search the web and return a ranked list of results with titles, URLs and snippets. \
                Best for: finding specific URLs, checking recent news, verifying facts, discovering sources. \
                Fastest and cheapest option. \
                Returns formatted results (title, URL, snippet) — no AI synthesis. \
                For AI-generated answers with citations, use perplexity_ask instead.",
        annotations(
            title = "Search the Web",
            read_only_hint = true,
            open_world_hint = true,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    pub async fn perplexity_search(
        &self,
        Parameters(params): Parameters<PerplexitySearchRequest>,
    ) -> Result<CallToolResult, McpError> {
        let response = self.do_search(params.into(), SearchMode::Auto, None, false).await?;
        to_json_tool_result(&SearchOnlyResponse { web_results: response.web_results })
    }

    /// Ask Perplexity AI a question and get an answer with sources.
    ///
    /// Uses the configured ask model.
    #[tool(
        name = "perplexity_ask",
        description = "Answer a question using web-grounded AI. \
                Best for: quick factual questions, summaries, explanations, and general Q&A. \
                Returns a text response with formatted results (title, URL, snippet). \
                For in-depth multi-source research, use perplexity_research instead. \
                For step-by-step reasoning and analysis, use perplexity_reason instead. \
                Supports optional file attachments via the `files` parameter (requires authentication token).",
        annotations(
            title = "Ask Perplexity",
            read_only_hint = true,
            open_world_hint = true,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    pub async fn perplexity_ask(
        &self,
        Parameters(params): Parameters<PerplexityRequest>,
    ) -> Result<CallToolResult, McpError> {
        let response = self
            .do_search(
                params,
                SearchMode::Auto,
                self.ask_model.map(ModelPreference::from),
                true,
            )
            .await?;
        to_json_tool_result(&response)
    }

    /// Generate an image with Perplexity and return its bytes without exposing signed URLs.
    #[tool(
        name = "perplexity_generate_image",
        description = "Generate an original image with Perplexity. Returns the generated PNG as base64 plus dimensions and thread metadata. Requires authenticated Perplexity session tokens.",
        annotations(
            title = "Generate Image",
            read_only_hint = false,
            open_world_hint = true,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    pub async fn perplexity_generate_image(
        &self,
        Parameters(params): Parameters<PerplexityRequest>,
    ) -> Result<CallToolResult, McpError> {
        if self.tokenless {
            return Err(McpError::invalid_params(
                "Image generation requires authenticated Perplexity session tokens.",
                None,
            ));
        }
        let response = self
            .execute_search(
                params,
                SearchMode::Pro,
                self.ask_model.map(ModelPreference::from),
                false,
            )
            .await?;
        let (url, filename, width, height) = Self::generated_image_metadata(&response.raw)?;
        let image = rquest::Client::builder()
            .build()
            .map_err(|e| McpError::internal_error(format!("image client error: {e}"), None))?
            .get(url)
            .send()
            .await
            .map_err(|e| McpError::internal_error(format!("image download error: {e}"), None))?
            .error_for_status()
            .map_err(|e| {
                McpError::internal_error(format!("image download status error: {e}"), None)
            })?
            .bytes()
            .await
            .map_err(|e| McpError::internal_error(format!("image body error: {e}"), None))?;
        if image.len() > 20 * 1024 * 1024 {
            return Err(McpError::internal_error(
                "generated image exceeds the 20 MiB response limit".to_string(),
                None,
            ));
        }
        to_json_tool_result(&GeneratedImageResponse {
            answer: response.answer,
            filename,
            mime_type: "image/png".to_string(),
            data: base64::engine::general_purpose::STANDARD.encode(image),
            width,
            height,
            backend_uuid: response.follow_up.backend_uuid,
        })
    }

    /// Deep, comprehensive research using Perplexity's sonar-deep-research model.
    ///
    /// Best for: Complex topics requiring detailed investigation, comprehensive reports,
    /// and in-depth analysis. Provides thorough analysis with citations.
    #[tool(
        name = "perplexity_research",
        description = "Conduct deep, multi-source research on a topic. \
                Best for: literature reviews, comprehensive overviews, investigative queries needing \
                many sources. Returns a detailed response with numbered citations. \
                Significantly slower than other tools (60+ seconds). \
                For quick factual questions, use perplexity_ask instead. \
                For logical analysis and reasoning, use perplexity_reason instead. \
                Supports optional file attachments via the `files` parameter (requires authentication token).",
        annotations(
            title = "Deep Research",
            read_only_hint = true,
            open_world_hint = true,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    pub async fn perplexity_research(
        &self,
        Parameters(params): Parameters<PerplexityRequest>,
    ) -> Result<CallToolResult, McpError> {
        to_json_tool_result(
            &self.do_search(params, SearchMode::DeepResearch, None, true).await?,
        )
    }

    /// Advanced reasoning and problem-solving using Perplexity's sonar-reasoning-pro model.
    ///
    /// Best for: Logical problems, complex analysis, decision-making,
    /// and tasks requiring step-by-step reasoning.
    #[tool(
        name = "perplexity_reason",
        description = "Analyze a question using step-by-step reasoning with web grounding. \
                Best for: math, logic, comparisons, complex arguments, and tasks requiring chain-of-thought. \
                Returns a reasoned response with numbered citations. \
                For quick factual questions, use perplexity_ask instead. \
                For comprehensive multi-source research, use perplexity_research instead. \
                Supports optional file attachments via the `files` parameter (requires authentication token).",
        annotations(
            title = "Advanced Reasoning",
            read_only_hint = true,
            open_world_hint = true,
            destructive_hint = false,
            idempotent_hint = false
        )
    )]
    pub async fn perplexity_reason(
        &self,
        Parameters(params): Parameters<PerplexityRequest>,
    ) -> Result<CallToolResult, McpError> {
        to_json_tool_result(
            &self
                .do_search(
                    params,
                    SearchMode::Reasoning,
                    self.reason_model.map(ModelPreference::from),
                    true,
                )
                .await?,
        )
    }
}

#[tool_handler]
impl ServerHandler for PerplexityServer {
    fn get_info(&self) -> ServerInfo {
        let mut instructions = String::from(
            "Perplexity AI server for web-grounded search. \
            All tools are read-only and access live web data. \
            Use perplexity_search for finding URLs, facts, and recent news. \
            Use perplexity_ask for quick AI-answered questions with citations.",
        );
        if !self.tokenless {
            instructions.push_str(
                " Use perplexity_research for in-depth multi-source investigation (slow, 60s+). \
                Use perplexity_reason for complex analysis requiring step-by-step logic. \
                All tools support an optional `files` parameter for document analysis: \
                pass an array of objects each with `filename` and either `text` (plain-text content) \
                or `data` (base64-encoded binary content, e.g. for PDFs).",
            );
        }

        let server_info =
            Implementation::new(env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(instructions)
            .with_server_info(server_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_image_metadata_selects_generated_media_without_exposing_it() {
        let raw = serde_json::json!({
            "media_items": [
                {"medium": "image", "image": "https://example.invalid/reference.png"},
                {
                    "medium": "image",
                    "name": "Green teapot / white background",
                    "image": "https://signed.example.invalid/generated.png?Signature=secret",
                    "image_width": 1254,
                    "image_height": 1254,
                    "generated_media_metadata": {"model_str": "gpt-4o-image"}
                }
            ]
        });
        let (url, filename, width, height) =
            PerplexityServer::generated_image_metadata(&raw).unwrap();
        assert_eq!(url, "https://signed.example.invalid/generated.png?Signature=secret");
        assert_eq!(filename, "Green-teapot---white-background.png");
        assert_eq!(width, Some(1254));
        assert_eq!(height, Some(1254));
    }

    #[test]
    fn generated_image_metadata_rejects_non_generated_images() {
        let raw = serde_json::json!({
            "media_items": [{"medium": "image", "image": "https://example.invalid/reference.png"}]
        });
        assert!(PerplexityServer::generated_image_metadata(&raw).is_err());
    }
}
