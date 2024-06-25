use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
#[derive(Error, Debug)]
pub enum GroqError {
    #[error("API request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),
    #[error("API error: {message}")]
    ApiError { message: String, type_: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRoles {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionMessage {
    pub role: ChatCompletionRoles,
    pub content: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<Choice>,
    pub created: u64,
    pub id: String,
    pub model: String,
    pub object: String,
    pub system_fingerprint: String,
    pub usage: Usage,
    pub x_groq: XGroq,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    pub finish_reason: String,
    pub index: u64,
    pub logprobs: Option<Value>,
    pub message: Message,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    pub content: String,
    pub role: ChatCompletionRoles,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub completion_time: f64,
    pub completion_tokens: u64,
    pub prompt_time: f64,
    pub prompt_tokens: u64,
    pub total_time: f64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct XGroq {
    pub id: String,
}

#[derive(Debug, Clone)]
pub struct SpeechToTextRequest {
    pub file: Vec<u8>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub language: Option<String>,
    pub english_text: bool,
}

impl SpeechToTextRequest {
    pub fn new(file: Vec<u8>) -> Self {
        Self {
            file,
            model: None,
            temperature: None,
            language: None,
            english_text: false,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    pub fn english_text(mut self, english_text: bool) -> Self {
        self.english_text = english_text;
        self
    }

    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpeechToTextResponse {
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
}

impl ChatCompletionRequest {
    pub fn new(model: &str, messages: Vec<ChatCompletionMessage>) -> Self {
        ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature: Some(1.0),
            max_tokens: Some(1024),
            top_p: Some(1.0),
            stream: Some(false),
            stop: None,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }
}
