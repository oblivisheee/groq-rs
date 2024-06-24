//! A Rust client for interacting with the Groq API.
//!
//! This crate provides a simple interface for making requests to the Groq API,
//! including chat completions and speech-to-text conversions.

use reqwest::blocking::multipart::{Form, Part};
use reqwest::blocking::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::error::Error;

/// Represents the roles in a chat completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChatCompletionRoles {
    /// The system role, typically used for setting up the conversation.
    System,
    /// The user role, representing the end-user's messages.
    User,
    /// The assistant role, representing the AI's responses.
    Assistant,
}

/// Represents a message in a chat completion.
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionMessage {
    /// The role of the message sender (System, User, or Assistant).
    pub role: ChatCompletionRoles,
    /// The content of the message.
    pub content: String,
    /// An optional name for the message sender.
    pub name: Option<String>,
}

/// Represents the response from a chat completion request.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionResponse {
    /// The generated choices from the model.
    pub choices: Vec<Choice>,
    /// The timestamp when the response was created.
    pub created: u64,
    /// The unique identifier for this completion.
    pub id: String,
    /// The model used for the completion.
    pub model: String,
    /// The object type, typically "chat.completion".
    pub object: String,
    /// The system fingerprint.
    pub system_fingerprint: String,
    /// Usage statistics for the request.
    pub usage: Usage,
    /// Additional Groq-specific information.
    pub x_groq: XGroq,
}

/// Represents a single choice in the chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct Choice {
    /// The reason why the model stopped generating.
    pub finish_reason: String,
    /// The index of this choice.
    pub index: u64,
    /// Log probabilities, if requested.
    pub logprobs: Option<Value>,
    /// The generated message.
    pub message: Message,
}

/// Represents a message in the chat completion response.
#[derive(Debug, Clone, Deserialize)]
pub struct Message {
    /// The content of the message.
    pub content: String,
    /// The role of the message sender.
    pub role: ChatCompletionRoles,
}

/// Represents usage statistics for the API request.
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    /// Time taken for completion in seconds.
    pub completion_time: f64,
    /// Number of tokens in the completion.
    pub completion_tokens: u64,
    /// Time taken for processing the prompt in seconds.
    pub prompt_time: f64,
    /// Number of tokens in the prompt.
    pub prompt_tokens: u64,
    /// Total time taken for the request in seconds.
    pub total_time: f64,
    /// Total number of tokens used.
    pub total_tokens: u64,
}

/// Represents Groq-specific information in the response.
#[derive(Debug, Clone, Deserialize)]
pub struct XGroq {
    /// A unique identifier for the Groq request.
    pub id: String,
}

/// Represents a request for speech-to-text conversion.
#[derive(Debug, Clone)]
pub struct SpeechToTextRequest {
    /// The audio file data.
    pub file: Vec<u8>,
    /// The model to use for speech recognition.
    pub model: Option<String>,
    /// The temperature setting for the model.
    pub temperature: Option<f64>,
    /// The language of the audio.
    pub language: Option<String>,
    /// Whether to translate the text to English.
    pub english_text: bool,
}

impl SpeechToTextRequest {
    /// Creates a new `SpeechToTextRequest` with the given audio file.
    pub fn new(file: Vec<u8>) -> Self {
        Self {
            file,
            model: None,
            temperature: None,
            language: None,
            english_text: false,
        }
    }

    /// Sets the temperature for the request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the language for the request.
    pub fn language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    /// Sets whether to translate to English text.
    pub fn english_text(mut self, english_text: bool) -> Self {
        self.english_text = english_text;
        self
    }

    /// Sets the model for the request.
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }
}

/// Represents the response from a speech-to-text conversion.
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechToTextResponse {
    /// The transcribed text.
    pub text: String,
}

/// Represents a request for chat completion.
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    /// The model to use for chat completion.
    pub model: String,
    /// The messages in the conversation.
    pub messages: Vec<ChatCompletionMessage>,
    /// The temperature setting for the model.
    pub temperature: Option<f64>,
    /// The maximum number of tokens to generate.
    pub max_tokens: Option<u32>,
    /// The top_p setting for the model.
    pub top_p: Option<f64>,
    /// Whether to stream the response.
    pub stream: Option<bool>,
    /// Optional stop sequences.
    pub stop: Option<Vec<String>>,
}

impl ChatCompletionRequest {
    /// Creates a new `ChatCompletionRequest` with the given parameters.
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

    /// Sets the temperature for the request.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the top_p value for the request.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets whether to stream the response.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(false);
        //self.stream =Some(stream);
        self
    }

    /// Sets the stop sequences for the request.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }
}

/// Client for interacting with the Groq API.
pub struct GroqClient {
    api_key: String,
    client: Client,
    endpoint: String,
}

impl GroqClient {
    /// Creates a new `GroqClient` with the given API key and optional endpoint.
    pub fn new(api_key: String, endpoint: Option<String>) -> Self {
        let ep: String = endpoint.unwrap_or_else(|| String::from("https://api.groq.com/"));
        Self {
            api_key,
            client: Client::new(),
            endpoint: ep,
        }
    }

    /// Sends a request to the Groq API.
    fn send_request(&self, body: Value, link: &str) -> Result<Response, Box<dyn Error>> {
        let res = self
            .client
            .post(link)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .json(&body)
            .send()?;
        Ok(res)
    }

    /// Performs speech-to-text conversion.
    pub fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, Box<dyn Error>> {
        let mut form = Form::new().part("file", Part::bytes(request.file).file_name("audio.wav"));
        if let Some(temperature) = request.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(language) = request.language {
            form = form.text("language", language);
        }
        let link_addition = if request.english_text {
            "openai/v1/audio/translations"
        } else {
            "openai/v1/audio/transcriptions"
        };
        if let Some(model) = request.model {
            form = form.text("model", model);
        }

        let link = format!("{}{}", self.endpoint, link_addition);
        let response = self
            .client
            .post(link)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()?;
        let speech_to_text_response: SpeechToTextResponse = response.json()?;
        Ok(speech_to_text_response)
    }

    /// Performs chat completion.
    pub fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, Box<dyn Error>> {
        let messages = request
            .messages
            .iter()
            .map(|m| {
                let mut message = json!({
                    "role": match m.role {
                        ChatCompletionRoles::System => "system",
                        ChatCompletionRoles::User => "user",
                        ChatCompletionRoles::Assistant => "assistant",
                    },
                    "content": m.content,
                });
                if let Some(name) = &m.name {
                    message["name"] = json!(name);
                }
                message
            })
            .collect::<Vec<Value>>();

        let mut body = json!({
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature.unwrap_or(1.0),
            "max_tokens": request.max_tokens.unwrap_or(1024),
            "top_p": request.top_p.unwrap_or(1.0),
            "stream": request.stream.unwrap_or(false),
        });
        if let Some(stop) = &request.stop {
            body["stop"] = json!(stop);
        }

        let response = self.send_request(
            body,
            &format!("{}openai/v1/chat/completions", self.endpoint),
        )?;
        let chat_completion_response: ChatCompletionResponse = response.json()?;
        Ok(chat_completion_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn test_chat_completion() {
        let api_key = "your_api_key";
        let client = GroqClient::new(api_key.to_string(), None);
        let messages = vec![ChatCompletionMessage {
            role: ChatCompletionRoles::User,
            content: "Hello".to_string(),
            name: None,
        }];
        let request = ChatCompletionRequest::new("llama3-70b-8192", messages);
        let response = client.chat_completion(request).unwrap();
        println!("{}", response.choices[0].message.content);
        assert!(!response.choices.is_empty());
    }

    #[test]
    fn test_speech_to_text() {
        let api_key = "your_api_key";
        let client = GroqClient::new(api_key.to_string(), None);
        let audio_file_path = "path_to_file.mp3";
        let mut file = File::open(audio_file_path).expect("Failed to open audio file");
        let mut audio_data = Vec::new();
        file.read_to_end(&mut audio_data)
            .expect("Failed to read audio file");
        let request = SpeechToTextRequest::new(audio_data)
            .temperature(0.7)
            .language("en")
            .model("whisper-large-v3");

        let response = client
            .speech_to_text(request)
            .expect("Failed to get response");
        println!("Speech to Text Response: {:?}", response);
        assert!(!response.text.is_empty());
    }
}
