//! OpenAI-compatible HTTP API: `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
//!
//! One engine thread owns the GPU, the model and the tokenizer. Handlers queue a job on a
//! channel; the engine runs jobs one at a time (the runtime is a single sequence, so there is
//! no batching) and sends pieces of text back on a per-request channel. A dropped connection
//! drops that channel, which stops its generation.
//!
//! With thinking on, chat output is split at `</think>`: text before it is `reasoning_content`,
//! text after it is `content`. The `<think>` opener is part of the prompt and is not returned.

use crate::{generate, model::Model, sampler, tokenizer::Tokenizer};
use anyhow::Result;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, Sse},
    },
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::VecDeque,
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::mpsc as tokio_mpsc;

pub struct Options {
    pub addr: SocketAddr,
    /// Name reported by `/v1/models`. A request's `model` field is not checked.
    pub model_name: String,
    /// Whether chat prompts open a `<think>` block unless the request says otherwise.
    pub think: bool,
    /// Context capacity in tokens; `Model::new_state` is sized to it.
    pub ctx: usize,
    /// Defaults for every request: stop tokens, chunk size, MTP draft count, sampling.
    pub base: generate::Params,
}

/// A failed request, as the HTTP status and message it returns.
#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.status.is_client_error() {
            "invalid_request_error"
        } else {
            "server_error"
        };
        let body = json!({"error": {"message": self.message, "type": kind}});
        (self.status, Json(body)).into_response()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kind {
    Chat,
    Text,
}

struct Finish {
    reason: &'static str,
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Generated text: reasoning inside the `<think>` block, or the answer after it.
enum Piece {
    Reasoning(String),
    Content(String),
}

/// What the engine sends back for one request.
enum Msg {
    Piece(Piece),
    Done(Finish),
    Failed(ApiError),
}

struct Job {
    prompt: String,
    /// Whether the prompt opened a `<think>` block, so generated text starts as reasoning.
    think: bool,
    max_new: usize,
    sampling: sampler::Params,
    /// Text stop strings: generation ends before any of them is sent.
    stops: Vec<String>,
    reply: tokio_mpsc::UnboundedSender<Msg>,
}

/// The engine: everything that touches the GPU. Lives on its own thread.
struct Engine {
    model: Model,
    tok: Tokenizer,
    base: generate::Params,
    ctx: usize,
}

impl Engine {
    fn run(&mut self, job: &Job) -> Result<()> {
        let prompt = self.tok.encode(&job.prompt)?;
        if prompt.len() >= self.ctx {
            let msg = format!(
                "prompt is {} tokens; the context holds {}",
                prompt.len(),
                self.ctx
            );
            let _ = job.reply.send(Msg::Failed(ApiError::bad(msg)));
            return Ok(());
        }
        let params = generate::Params {
            max_new: job.max_new,
            chunk: self.base.chunk,
            sampling: job.sampling.clone(),
            spec_k: self.base.spec_k,
            stop: self.base.stop.clone(),
        };
        let mut st = self.model.new_state(self.ctx, params.spec_k);
        let mut text = Text::new(&job.stops, job.think);
        let mut hit_stop = false;
        let reply = &job.reply;
        let tok = &self.tok;
        let result = generate::generate(&mut self.model, &mut st, &prompt, &params, |id| {
            let (pieces, stop) = text.push(&tok.token_bytes(id));
            hit_stop = stop;
            for piece in pieces {
                // A closed channel means the client went away: stop generating.
                if reply.send(Msg::Piece(piece)).is_err() {
                    return Ok(true);
                }
            }
            Ok(stop)
        });
        match result {
            Ok(out) => {
                self.model.cache_sequence(&st);
                self.model.release_state(st);
                if !hit_stop {
                    for piece in text.finish() {
                        let _ = reply.send(Msg::Piece(piece));
                    }
                }
                let reason = if hit_stop || out.generated < job.max_new {
                    "stop"
                } else {
                    "length"
                };
                let _ = reply.send(Msg::Done(Finish {
                    reason,
                    prompt_tokens: prompt.len(),
                    completion_tokens: out.generated,
                }));
            }
            Err(e) => {
                self.model.release_state(st);
                let _ = reply.send(Msg::Failed(ApiError::internal(format!("{e:#}"))));
            }
        }
        Ok(())
    }
}

fn engine_loop(mut engine: Engine, jobs: mpsc::Receiver<Job>) {
    for job in jobs {
        // Errors that escape `run` (tokenizer failures) must still reach the client.
        if let Err(e) = engine.run(&job) {
            let _ = job
                .reply
                .send(Msg::Failed(ApiError::internal(format!("{e:#}"))));
        }
    }
}

const THINK_CLOSE: &str = "</think>";

/// Turns generated bytes into pieces for the client. Text that could still become a stop
/// string, or the `</think>` tag, is held back until it is decided. A stop string ends the
/// output, and nothing from it is sent. UTF-8 characters split across tokens wait for their
/// remaining bytes.
struct Text<'a> {
    bytes: Vec<u8>,
    held: String,
    stops: &'a [String],
    /// Inside the reasoning block: `</think>` ends it. Off when the prompt opened no block.
    reasoning: bool,
    /// Just past `</think>`: whitespace before the answer is dropped.
    trim: bool,
}

impl<'a> Text<'a> {
    fn new(stops: &'a [String], think: bool) -> Self {
        Self {
            bytes: Vec::new(),
            held: String::new(),
            stops,
            reasoning: think,
            trim: false,
        }
    }

    /// Adds one token's bytes. Returns the pieces safe to send now, and whether a stop string
    /// ended the output.
    fn push(&mut self, token: &[u8]) -> (Vec<Piece>, bool) {
        self.bytes.extend_from_slice(token);
        self.held.push_str(&take_utf8(&mut self.bytes));
        let mut out = Vec::new();
        loop {
            let stop_at = self
                .stops
                .iter()
                .filter_map(|s| self.held.find(s.as_str()))
                .min();
            let close_at = if self.reasoning {
                self.held.find(THINK_CLOSE)
            } else {
                None
            };
            match (stop_at, close_at) {
                (Some(s), close) if close.is_none_or(|c| s < c) => {
                    let text = self.held[..s].to_owned();
                    self.held.clear();
                    self.emit(&mut out, text);
                    return (out, true);
                }
                (_, Some(c)) => {
                    let text = self.held[..c].to_owned();
                    self.held.drain(..c + THINK_CLOSE.len());
                    self.emit(&mut out, text);
                    self.reasoning = false;
                    self.trim = true;
                }
                _ => break,
            }
        }
        let cut = (0..self.held.len())
            .find(|&i| self.held.is_char_boundary(i) && self.could_start(&self.held[i..]))
            .unwrap_or(self.held.len());
        let text: String = self.held.drain(..cut).collect();
        self.emit(&mut out, text);
        (out, false)
    }

    /// Whether `suffix` could be the start of a stop string or, while reasoning, the close tag.
    fn could_start(&self, suffix: &str) -> bool {
        self.stops.iter().any(|s| s.starts_with(suffix))
            || (self.reasoning && THINK_CLOSE.starts_with(suffix))
    }

    fn emit(&mut self, out: &mut Vec<Piece>, text: String) {
        let text = if self.trim {
            text.trim_start().to_owned()
        } else {
            text
        };
        if text.is_empty() {
            return;
        }
        self.trim = false;
        out.push(if self.reasoning {
            Piece::Reasoning(text)
        } else {
            Piece::Content(text)
        });
    }

    /// Everything still held, once generation ends without a stop string.
    fn finish(&mut self) -> Vec<Piece> {
        let mut text = std::mem::take(&mut self.held);
        text.push_str(&String::from_utf8_lossy(&self.bytes));
        self.bytes.clear();
        let mut out = Vec::new();
        self.emit(&mut out, text);
        out
    }
}

/// Splits off the longest valid UTF-8 prefix of `bytes`, leaving an incomplete tail in place.
fn take_utf8(bytes: &mut Vec<u8>) -> String {
    let valid = match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(e) => e.valid_up_to(),
    };
    let tail = bytes.split_off(valid);
    String::from_utf8(std::mem::replace(bytes, tail)).expect("validated prefix")
}

#[derive(Clone)]
struct AppState {
    jobs: mpsc::Sender<Job>,
    model_name: Arc<str>,
    think: bool,
    ctx: usize,
    defaults: sampler::Params,
}

#[derive(Deserialize, Default)]
struct Common {
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop: Option<Stop>,
    n: Option<usize>,
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    chat_template_kwargs: Option<TemplateKwargs>,
    #[serde(flatten)]
    common: Common,
}

#[derive(Deserialize)]
struct TemplateKwargs {
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: Option<Content>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
struct Part {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

impl ChatMessage {
    fn text(&self) -> Result<String, ApiError> {
        match &self.content {
            None => Ok(String::new()),
            Some(Content::Text(s)) => Ok(s.clone()),
            Some(Content::Parts(parts)) => {
                let mut out = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(ApiError::bad(format!(
                            "content part type {:?} is not supported",
                            part.kind
                        )));
                    }
                    out.push_str(part.text.as_deref().unwrap_or(""));
                }
                Ok(out)
            }
        }
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    prompt: String,
    #[serde(flatten)]
    common: Common,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Queues one request on the engine; the returned channel yields its pieces.
fn submit(
    state: &AppState,
    prompt: String,
    think: bool,
    c: &Common,
) -> Result<tokio_mpsc::UnboundedReceiver<Msg>, ApiError> {
    if c.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad("only n = 1 is supported"));
    }
    let stops = match &c.stop {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s.clone()],
        Some(Stop::Many(v)) => v.clone(),
    };
    if stops.iter().any(String::is_empty) {
        return Err(ApiError::bad("stop strings must not be empty"));
    }
    let d = &state.defaults;
    let sampling = sampler::Params {
        temperature: c.temperature.unwrap_or(d.temperature),
        top_k: c.top_k.unwrap_or(d.top_k),
        top_p: c.top_p.unwrap_or(d.top_p),
        seed: c.seed.unwrap_or(d.seed),
    };
    let (reply, rx) = tokio_mpsc::unbounded_channel();
    let job = Job {
        prompt,
        think,
        max_new: c
            .max_completion_tokens
            .or(c.max_tokens)
            .unwrap_or(state.ctx),
        sampling,
        stops,
        reply,
    };
    state
        .jobs
        .send(job)
        .map_err(|_| ApiError::internal("engine is not running"))?;
    Ok(rx)
}

#[derive(Default)]
struct Collected {
    content: String,
    reasoning: String,
}

/// Collects a whole response, or the error that ended it.
async fn collect(
    mut rx: tokio_mpsc::UnboundedReceiver<Msg>,
) -> Result<(Collected, Finish), ApiError> {
    let mut out = Collected::default();
    while let Some(msg) = rx.recv().await {
        match msg {
            Msg::Piece(Piece::Content(t)) => out.content.push_str(&t),
            Msg::Piece(Piece::Reasoning(t)) => out.reasoning.push_str(&t),
            Msg::Done(f) => return Ok((out, f)),
            Msg::Failed(e) => return Err(e),
        }
    }
    Err(ApiError::internal("engine stopped"))
}

fn usage(f: &Finish) -> Value {
    json!({
        "prompt_tokens": f.prompt_tokens,
        "completion_tokens": f.completion_tokens,
        "total_tokens": f.prompt_tokens + f.completion_tokens,
    })
}

/// Identity of one response, shared by its stream chunks.
#[derive(Clone)]
struct Meta {
    id: String,
    created: u64,
    model: Arc<str>,
    kind: Kind,
}

impl Meta {
    fn new(state: &AppState, kind: Kind) -> Self {
        let prefix = match kind {
            Kind::Chat => "chatcmpl",
            Kind::Text => "cmpl",
        };
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("{prefix}-{n}"),
            created: now_secs(),
            model: state.model_name.clone(),
            kind,
        }
    }

    fn object(&self) -> &'static str {
        match self.kind {
            Kind::Chat => "chat.completion.chunk",
            Kind::Text => "text_completion",
        }
    }

    /// One stream chunk. `piece` is `None` for the chunk that carries the finish reason.
    fn chunk(&self, piece: Option<&Piece>, finish: Value) -> String {
        let choice = match self.kind {
            Kind::Chat => {
                let mut delta = json!({"role": "assistant"});
                match piece {
                    Some(Piece::Reasoning(t)) => delta["reasoning_content"] = json!(t),
                    Some(Piece::Content(t)) => delta["content"] = json!(t),
                    None => {}
                }
                json!({"index": 0, "delta": delta, "finish_reason": finish})
            }
            Kind::Text => {
                let text = match piece {
                    Some(Piece::Content(t)) | Some(Piece::Reasoning(t)) => t.as_str(),
                    None => "",
                };
                json!({"index": 0, "text": text, "finish_reason": finish})
            }
        };
        json!({
            "id": self.id,
            "object": self.object(),
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        })
        .to_string()
    }

    fn usage_chunk(&self, f: &Finish) -> String {
        json!({
            "id": self.id,
            "object": self.object(),
            "created": self.created,
            "model": self.model,
            "choices": [],
            "usage": usage(f),
        })
        .to_string()
    }

    fn whole(&self, c: &Collected, f: &Finish) -> Value {
        let (object, choice) = match self.kind {
            Kind::Chat => (
                "chat.completion",
                json!({
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": c.content,
                        "reasoning_content": c.reasoning,
                    },
                    "finish_reason": f.reason,
                }),
            ),
            Kind::Text => (
                "text_completion",
                json!({"index": 0, "text": c.content, "finish_reason": f.reason}),
            ),
        };
        json!({
            "id": self.id,
            "object": object,
            "created": self.created,
            "model": self.model,
            "choices": [choice],
            "usage": usage(f),
        })
    }
}

/// Server-sent events for one response, ending with `data: [DONE]`.
struct Stream {
    rx: tokio_mpsc::UnboundedReceiver<Msg>,
    queue: VecDeque<String>,
    meta: Meta,
    include_usage: bool,
    ended: bool,
}

impl Stream {
    fn end(&mut self) {
        self.queue.push_back("[DONE]".into());
        self.ended = true;
    }
}

fn stream_response(
    rx: tokio_mpsc::UnboundedReceiver<Msg>,
    meta: Meta,
    include_usage: bool,
) -> Response {
    let state = Stream {
        rx,
        queue: VecDeque::new(),
        meta,
        include_usage,
        ended: false,
    };
    let events = futures_lite::stream::unfold(state, |mut s| async move {
        loop {
            if let Some(data) = s.queue.pop_front() {
                let event = SseEvent::default().data(data);
                return Some((Ok::<_, Infallible>(event), s));
            }
            if s.ended {
                return None;
            }
            match s.rx.recv().await {
                Some(Msg::Piece(p)) => s.queue.push_back(s.meta.chunk(Some(&p), Value::Null)),
                Some(Msg::Done(f)) => {
                    s.queue.push_back(s.meta.chunk(None, json!(f.reason)));
                    if s.include_usage {
                        s.queue.push_back(s.meta.usage_chunk(&f));
                    }
                    s.end();
                }
                Some(Msg::Failed(e)) => {
                    s.queue
                        .push_back(json!({"error": {"message": e.message}}).to_string());
                    s.end();
                }
                None => s.end(),
            }
        }
    });
    Sse::new(events).into_response()
}

/// Chat-ML prompt for a request's messages, and whether it opens a `<think>` block.
/// `enable_thinking` in the request overrides the server default.
fn chat_text(state: &AppState, req: &ChatRequest) -> Result<(String, bool), ApiError> {
    if req.messages.is_empty() {
        return Err(ApiError::bad("messages must not be empty"));
    }
    let messages = req
        .messages
        .iter()
        .map(|m| Ok((m.role.as_str(), m.text()?)))
        .collect::<Result<Vec<_>, ApiError>>()?;
    let think = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|k| k.enable_thinking)
        .unwrap_or(state.think);
    let prompt = generate::chat_prompt(messages.iter().map(|(r, c)| (*r, c.as_str())), think);
    Ok((prompt, think))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let (prompt, think) = chat_text(&state, &req)?;
    let rx = submit(&state, prompt, think, &req.common)?;
    let meta = Meta::new(&state, Kind::Chat);
    if req.common.stream {
        let include_usage = req
            .common
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage);
        return Ok(stream_response(rx, meta, include_usage));
    }
    let (collected, finish) = collect(rx).await?;
    Ok(Json(meta.whole(&collected, &finish)).into_response())
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Result<Response, ApiError> {
    // Legacy endpoint: the prompt is fed verbatim, without the chat template.
    let rx = submit(&state, req.prompt, false, &req.common)?;
    let meta = Meta::new(&state, Kind::Text);
    if req.common.stream {
        let include_usage = req
            .common
            .stream_options
            .as_ref()
            .is_some_and(|o| o.include_usage);
        return Ok(stream_response(rx, meta, include_usage));
    }
    let (collected, finish) = collect(rx).await?;
    Ok(Json(meta.whole(&collected, &finish)).into_response())
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": now_secs(),
            "owned_by": "burn-vllm",
        }],
    }))
}

/// Takes the loaded model and tokenizer, starts the engine thread, and serves HTTP until the
/// process is stopped.
pub fn serve(model: Model, tok: Tokenizer, opts: Options) -> Result<()> {
    let Options {
        addr,
        model_name,
        think,
        ctx,
        base,
    } = opts;
    let defaults = base.sampling.clone();
    let (jobs, rx) = mpsc::channel();
    let engine = Engine {
        model,
        tok,
        base,
        ctx,
    };
    std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_loop(engine, rx))?;

    let state = AppState {
        jobs,
        model_name: model_name.into(),
        think,
        ctx,
        defaults,
    };
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .with_state(state);
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(async move {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            eprintln!("serving OpenAI API on http://{addr}/v1");
            axum::serve(listener, app).await?;
            Ok(())
        })
}
