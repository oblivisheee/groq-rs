use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
#[derive(Error, Debug)]
/// Represents errors that can occur when interacting with the GROQ API.
///
/// - `RequestFailed`: Indicates a failure in the underlying HTTP request.
/// - `JsonParseError`: Indicates a failure in parsing the JSON response from the API.
/// - `ApiError`: Indicates an error returned by the API, with a message and error type.
/// - `DeserializationError`: Indicates an error with deserialization, with a message and error type.
pub enum GroqError {
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    #[error("API request failed: {0}")]
    RequestFailed(#[from] reqwest::Error),
    #[error("Failed to parse JSON: {0}")]
    JsonParseError(#[from] serde_json::Error),
    #[error("API error: {message}")]
    ApiError { message: String, type_: String },
    #[error("Deserialization error: {message}")]
    DeserializationError { message: String, type_: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
/// Represents the different roles that can be used in a chat completion message.
///
/// - `System`: Indicates a message from the system.
/// - `User`: Indicates a message from the user.
/// - `Assistant`: Indicates a message from the assistant.
pub enum ChatCompletionRoles {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize)]
/// Represents a message in a chat completion response.
///
/// - `role`: The role of the message, such as `System`, `User`, or `Assistant`.
/// - `content`: The content of the message.
/// - `name`: An optional name associated with the message.
pub struct ChatCompletionMessage {
    pub role: ChatCompletionRoles,
    pub content: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents the response from a chat completion API request.
///
/// - `choices`: A vector of `Choice` objects, each representing a possible response.
/// - `created`: The timestamp (in seconds since the epoch) when the response was generated.
/// - `id`: The unique identifier for the response.
/// - `model`: The name of the model used to generate the response.
/// - `object`: The type of the response object.
/// - `system_fingerprint`: A unique identifier for the system that generated the response.
/// - `usage`: Usage statistics for the request, including token counts and processing times.
/// - `x_groq`: Additional metadata about the response, including the GROQ API ID.
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
/// Represents a single choice in a chat completion response.
///
/// - `finish_reason`: The reason the generation finished, such as "stop" or "length".
/// - `index`: The index of the choice within the list of choices.
/// - `logprobs`: Optional log probabilities for the tokens in the generated text.
/// - `message`: The message associated with this choice, containing the role, content, and optional name.
pub struct Choice {
    pub finish_reason: String,
    pub index: u64,
    pub logprobs: Option<Value>,
    pub message: Message,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents a message in a chat completion response.
///
/// - `content`: The content of the message.
/// - `role`: The role of the message, such as `System`, `User`, or `Assistant`.
pub struct Message {
    pub content: String,
    pub role: ChatCompletionRoles,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents the response from a chat completion API request.
///
/// - `choices`: A vector of `Choice` objects, each representing a possible response.
/// - `created`: The timestamp (in seconds since the epoch) when the response was generated.
/// - `id`: The unique identifier for the response.
/// - `model`: The name of the model used to generate the response.
/// - `object`: The type of the response object.
/// - `system_fingerprint`: A unique identifier for the system that generated the response.
/// - `usage`: Usage statistics for the request, including token counts and processing times.
/// - `x_groq`: Additional metadata about the response, including the GROQ API ID.
pub struct ChatCompletionDeltaResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: String,
    pub choices: Vec<ChoiceDelta>,
    pub x_groq: Option<XGroq>,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents a single choice in a chat completion response.
///
/// - `finish_reason`: The reason the generation finished, such as "stop" or "length".
/// - `index`: The index of the choice within the list of choices.
/// - `logprobs`: Optional log probabilities for the tokens in the generated text.
/// - `message`: The message associated with this choice, containing the role, content, and optional name.
pub struct ChoiceDelta {
    pub index: u64,
    pub delta: Delta,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents a message in a chat completion response.
///
/// - `content`: The content of the message.
/// - `role`: The role of the message, such as `System`, `User`, or `Assistant`.
pub struct Delta {
    pub role: Option<ChatCompletionRoles>,
    pub content: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents usage statistics for a chat completion request, including token counts and processing times.
///
/// - `completion_time`: The time (in seconds) it took to generate the completion.
/// - `completion_tokens`: The number of tokens in the generated completion.
/// - `prompt_time`: The time (in seconds) it took to process the prompt.
/// - `prompt_tokens`: The number of tokens in the prompt.
/// - `total_time`: The total time (in seconds) for the entire request.
/// - `total_tokens`: The total number of tokens used in the request.
pub struct Usage {
    pub completion_time: f64,
    pub completion_tokens: u64,
    pub prompt_time: f64,
    pub prompt_tokens: u64,
    pub total_time: f64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Deserialize)]
/// Represents a GROQ-related data structure.
///
/// - `id`: The unique identifier for this GROQ-related data.
pub struct XGroq {
    pub id: String,
}

#[derive(Debug, Clone)]
/// Represents a request to the speech-to-text API.
///
/// - `file`: The audio file to be transcribed.
/// - `model`: The speech recognition model to use.
/// - `temperature`: The temperature parameter to control the randomness of the transcription.
/// - `language`: The language of the audio file.
/// - `english_text`: If true, the API will use the translation endpoint instead of the transcription endpoint.
/// - `prompt`: An optional prompt to provide context for the transcription.
/// - `response_format`: The desired format of the transcription response, either "text" or "json".
pub struct SpeechToTextRequest {
    pub file: Vec<u8>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub language: Option<String>,
    /// If true, the API will use following path: `/audio/translations` instead of `/audio/transcriptions`
    pub english_text: bool,
    pub prompt: Option<String>,
    pub response_format: Option<String>,
}

/// Constructs a new `SpeechToTextRequest` with the given audio file.
///
/// # Arguments
/// * `file` - The audio file to be transcribed.
///
/// # Returns
/// A new `SpeechToTextRequest` instance with the given audio file and default values for other fields.
impl SpeechToTextRequest {
    pub fn new(file: Vec<u8>) -> Self {
        Self {
            file,
            model: None,
            temperature: None,
            language: None,
            english_text: false,
            prompt: None,
            response_format: None,
        }
    }

    /// Sets the temperature parameter for the speech recognition model.
    ///
    /// # Arguments
    /// * `temperature` - The temperature parameter to control the randomness of the transcription.
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated temperature.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the language of the audio file.
    ///
    /// # Arguments
    /// * `language` - The language of the audio file.
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated language.
    pub fn language(mut self, language: &str) -> Self {
        self.language = Some(language.to_string());
        self
    }

    /// Sets whether the API should use the translation endpoint instead of the transcription endpoint.
    ///
    /// # Arguments
    /// * `english_text` - If true, the API will use the translation endpoint.
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated `english_text` flag.
    pub fn english_text(mut self, english_text: bool) -> Self {
        self.english_text = english_text;
        self
    }

    /// Sets the speech recognition model to use.
    ///
    /// # Arguments
    /// * `model` - The speech recognition model to use.
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated model.
    pub fn model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Sets the prompt to provide context for the transcription.
    ///
    /// # Arguments
    /// * `prompt` - The prompt to provide context for the transcription.
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated prompt.
    pub fn prompt(mut self, prompt: &str) -> Self {
        self.prompt = Some(prompt.to_string());
        self
    }

    /// Sets the desired format of the transcription response.
    ///
    /// # Arguments
    /// * `response_format` - The desired format of the transcription response, either "text" or "json".
    ///
    /// # Returns
    /// The modified `SpeechToTextRequest` instance with the updated response format.
    pub fn response_format(mut self, response_format: &str) -> Self {
        // Currently only "text" and "json" are supported.
        self.response_format = Some(response_format.to_string());
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Represents the response from a speech-to-text transcription request.
///
/// The `text` field contains the transcribed text from the audio input.
pub struct SpeechToTextResponse {
    pub text: String,
}

/// Represents a request to the OpenAI chat completion API.
///
/// - `model`: The language model to use for the chat completion.
/// - `messages`: The messages to provide as context for the chat completion.
/// - `temperature`: The temperature parameter to control the randomness of the generated response.
/// - `max_tokens`: The maximum number of tokens to generate in the response.
/// - `top_p`: The top-p parameter to control the nucleus sampling.
/// - `stream`: Whether to stream the response or return it all at once.
/// - `stop`: A list of strings to stop the generation when encountered.
/// - `seed`: The seed value to use for the random number generator.
#[derive(Debug, Clone)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionMessage>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub seed: Option<u64>,
}

/// Represents a request to the OpenAI chat completion API.
///
/// This struct provides a builder-style API for constructing a `ChatCompletionRequest` with various optional parameters. The `new` method creates a new instance with default values, and the other methods allow modifying individual parameters.
///
/// - `model`: The language model to use for the chat completion.
/// - `messages`: The messages to provide as context for the chat completion.
/// - `temperature`: The temperature parameter to control the randomness of the generated response.
/// - `max_tokens`: The maximum number of tokens to generate in the response.
/// - `top_p`: The top-p parameter to control the nucleus sampling.
/// - `stream`: Whether to stream the response or return it all at once.
/// - `stop`: A list of strings to stop the generation when encountered.
/// - `seed`: The seed value to use for the random number generator.
impl ChatCompletionRequest {
    /// Creates a new `ChatCompletionRequest` instance with the given model and messages.
    ///
    /// # Arguments
    ///
    /// * `model` - The language model to use for the chat completion.
    /// * `messages` - The messages to provide as context for the chat completion.
    pub fn new(model: &str, messages: Vec<ChatCompletionMessage>) -> Self {
        ChatCompletionRequest {
            model: model.to_string(),
            messages,
            temperature: Some(1.0),
            max_tokens: Some(1024),
            top_p: Some(1.0),
            stream: Some(false),
            stop: None,
            seed: None,
        }
    }

    /// Sets the temperature parameter for the chat completion request.
    ///
    /// The temperature parameter controls the randomness of the generated response.
    /// Higher values (up to 1.0) make the output more random, while lower values make it more deterministic.
    ///
    /// # Arguments
    ///
    /// * `temperature` - The temperature value to use.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the maximum number of tokens to generate in the response.
    ///
    /// # Arguments
    ///
    /// * `max_tokens` - The maximum number of tokens to generate.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets the top-p parameter for the chat completion request.
    ///
    /// The top-p parameter controls the nucleus sampling, which is a technique for sampling from the most likely tokens.
    ///
    /// # Arguments
    ///
    /// * `top_p` - The top-p value to use.
    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets whether to stream the response or return it all at once.
    ///
    /// # Arguments
    ///
    /// * `stream` - Whether to stream the response or not.
    pub fn stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Sets the list of strings to stop the generation when encountered.
    ///
    /// # Arguments
    ///
    /// * `stop` - The list of stop strings.
    pub fn stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Sets the seed value to use for the random number generator.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed value to use.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}
