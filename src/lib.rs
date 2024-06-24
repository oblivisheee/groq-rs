use reqwest::blocking::multipart::{Form, Part};
use reqwest::blocking::{Client, Response};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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

pub struct GroqClient {
    api_key: String,
    client: Client,
    endpoint: String,
}

impl GroqClient {
    pub fn new(api_key: String, endpoint: Option<String>) -> Self {
        let ep = endpoint.unwrap_or_else(|| String::from("https://api.groq.com/openai/v1"));
        Self {
            api_key,
            client: Client::new(),
            endpoint: ep,
        }
    }

    fn send_request(&self, body: Value, link: &str) -> Result<Value, GroqError> {
        let res = self
            .client
            .post(link)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .json(&body)
            .send()?;

        parse_response(res)
    }

    pub fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, Box<dyn std::error::Error>> {
        // Extract values from request
        let file = request.file;
        let temperature = request.temperature;
        let language = request.language;
        let english_text = request.english_text;
        let model = request.model;

        // Build the form
        let mut form = Form::new().part("file", Part::bytes(file).file_name("audio.wav"));

        if let Some(temp) = temperature {
            form = form.text("temperature", temp.to_string());
        }

        if let Some(lang) = language {
            form = form.text("language", lang);
        }

        let link_addition = if english_text {
            "/audio/translations"
        } else {
            "/audio/transcriptions"
        };

        if let Some(mdl) = model {
            form = form.text("model", mdl);
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

    pub fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GroqError> {
        let messages = request
            .messages
            .iter()
            .map(|m| {
                json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect::<Vec<_>>();

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

        let response = self.send_request(body, &format!("{}/chat/completions", self.endpoint))?;
        let chat_completion_response: ChatCompletionResponse = serde_json::from_value(response)?;
        Ok(chat_completion_response)
    }
}

fn parse_response(response: Response) -> Result<Value, GroqError> {
    let status = response.status();
    let body: Value = response.json()?;

    if !status.is_success() {
        if let Some(error) = body.get("error") {
            return Err(GroqError::ApiError {
                message: error["message"]
                    .as_str()
                    .unwrap_or("Unknown error")
                    .to_string(),
                type_: error["type"]
                    .as_str()
                    .unwrap_or("unknown_error")
                    .to_string(),
            });
        }
    }

    Ok(body)
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
        let audio_file_path = "onepiece_demo.mp4";
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
