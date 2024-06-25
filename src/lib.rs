pub mod message;
use message::*;
use reqwest::blocking::multipart::{Form, Part};
use reqwest::blocking::{Client, Response};
use reqwest::multipart::{Form as AForm, Part as APart};
use reqwest::{Client as AClient, Response as AResponse};
use serde_json::{json, Value};
use std::sync::Arc;

pub struct AsyncGroqClient {
    api_key: String,
    client: Arc<AClient>,
    endpoint: String,
}

impl AsyncGroqClient {
    pub fn new(api_key: String, endpoint: Option<String>) -> Self {
        let ep = endpoint.unwrap_or_else(|| String::from("https://api.groq.com/openai/v1"));
        Self {
            api_key,
            client: Arc::new(AClient::new()),
            endpoint: ep,
        }
    }

    async fn send_request(&self, body: Value, link: &str) -> Result<Value, GroqError> {
        let res = self
            .client
            .post(link)
            .header("Content-Type", "application/json")
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        self.parse_response(res).await
    }

    pub async fn speech_to_text(
        &self,
        request: SpeechToTextRequest,
    ) -> Result<SpeechToTextResponse, GroqError> {
        let file = request.file;
        let temperature = request.temperature;
        let language = request.language;
        let english_text = request.english_text;
        let model = request.model;

        let mut form = AForm::new().part("file", APart::bytes(file).file_name("audio.wav"));
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
            .post(&link)
            .header("Authorization", &format!("Bearer {}", self.api_key))
            .multipart(form)
            .send()
            .await?;

        let speech_to_text_response: SpeechToTextResponse = response.json().await?;
        Ok(speech_to_text_response)
    }

    pub async fn chat_completion(
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

        let response = self
            .send_request(body, &format!("{}/chat/completions", self.endpoint))
            .await?;
        let chat_completion_response: ChatCompletionResponse = serde_json::from_value(response)?;
        Ok(chat_completion_response)
    }

    async fn parse_response(&self, response: AResponse) -> Result<Value, GroqError> {
        let status = response.status();
        let body: Value = response.json().await?;

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
    use tokio;

    #[test]
    fn test_chat_completion() {
        let api_key = std::env::var("GROQ_API_KEY").unwrap();
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
        let api_key = std::env::var("GROQ_API_KEY").unwrap();
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

    #[tokio::test]
    async fn test_async_chat_completion() {
        let api_key = std::env::var("GROQ_API_KEY").unwrap();
        let client = AsyncGroqClient::new(api_key, None);

        let messages1 = vec![ChatCompletionMessage {
            role: ChatCompletionRoles::User,
            content: "Hello".to_string(),
            name: None,
        }];
        let request1 = ChatCompletionRequest::new("llama3-70b-8192", messages1);

        let messages2 = vec![ChatCompletionMessage {
            role: ChatCompletionRoles::User,
            content: "How are you?".to_string(),
            name: None,
        }];
        let request2 = ChatCompletionRequest::new("llama3-70b-8192", messages2);

        let (response1, response2) = tokio::join!(
            client.chat_completion(request1),
            client.chat_completion(request2)
        );

        let response1 = response1.expect("Failed to get response for request 1");
        let response2 = response2.expect("Failed to get response for request 2");

        println!("Response 1: {}", response1.choices[0].message.content);
        println!("Response 2: {}", response2.choices[0].message.content);

        assert!(!response1.choices.is_empty());
        assert!(!response2.choices.is_empty());
    }

    #[tokio::test]
    async fn test_async_speech_to_text() {
        let api_key = std::env::var("GROQ_API_KEY").unwrap();
        let client = AsyncGroqClient::new(api_key, None);

        let audio_file_path1 = "onepiece_demo.mp4";
        let audio_file_path2 = "save.ogg";

        let (audio_data1, audio_data2) = tokio::join!(
            tokio::fs::read(audio_file_path1),
            tokio::fs::read(audio_file_path2)
        );

        let audio_data1 = audio_data1.expect("Failed to read first audio file");
        let audio_data2 = audio_data2.expect("Failed to read second audio file");

        let (request1, request2) = (
            SpeechToTextRequest::new(audio_data1)
                .temperature(0.7)
                .language("en")
                .model("whisper-large-v3"),
            SpeechToTextRequest::new(audio_data2)
                .temperature(0.7)
                .language("en")
                .model("whisper-large-v3"),
        );
        let (response1, response2) = tokio::join!(
            client.speech_to_text(request1),
            client.speech_to_text(request2)
        );

        let response1 = response1.expect("Failed to get response for first audio");
        let response2 = response2.expect("Failed to get response for second audio");

        println!("Speech to Text Response 1: {:?}", response1);
        println!("Speech to Text Response 2: {:?}", response2);

        assert!(!response1.text.is_empty());
        assert!(!response2.text.is_empty());
    }
}
