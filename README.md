# Unofficial Groq API Rust Client Library


This library provides the ability to interact with the Groq API. It allows you to send requests to the API and receive responses via the `GroqClient` interface.
## How to Install

To install the `groq-api-rust` crate, add it to your Rust project's dependencies by running the following command in your project's directory:

```
cargo add groq-api-rust
```

## Usage

1. Import the `groq-api-rust` crate into your Rust project.

2. Create an instance of the `GroqClient` struct by calling `GroqClient::new()` and providing your Groq API key and an optional endpoint URL.

3. Use the `chat_completion()` method on the `GroqClient` instance to perform chat completions. Create a `ChatCompletionRequest` struct with the desired model, messages, and other parameters, and pass it to the `chat_completion()` method.

4. Use the `speech_to_text()` method on the `GroqClient` instance to perform speech-to-text conversions. Create a `SpeechToTextRequest` struct with the audio file data and other parameters, and pass it to the `speech_to_text()` method.

## Examples

### Chat Completion

```rust
use groq_api_rust::{GroqClient, ChatCompletionMessage, ChatCompletionRoles, ChatCompletionRequest};
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
```

### Speech To Text

```rust
use groq_api_rust::{GroqClient, SpeechToTextRequest};
use std::{fs::File, io::Read};
let api_key = std::env::var("GROQ_API_KEY").unwrap();
let client = GroqClient::new(api_key, None);
let audio_file_path = "audio_file.mp3";
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
println!("Speech to Text Response: {}", response.text);
```

### Async Chat Completion

```rust
use groq_api_rust::{AsyncGroqClient, ChatCompletionMessage,ChatCompletionRoles, ChatCompletionRequest};
use tokio;
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
```

### Async Speech To Text

```rust
use groq_api_rust::{AsyncGroqClient, SpeechToTextRequest};
use tokio;
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
        .model("whisper-large-v3")
);
let (response1, response2) = tokio::join!(
    client.speech_to_text(request1),
    client.speech_to_text(request2)
);

let response1 = response1.expect("Failed to get response for first audio");
let response2 = response2.expect("Failed to get response for second audio");

println!("Speech to Text Response 1: {}", response1.text);
println!("Speech to Text Response 2: {}", response2.text);
```

## Contributing

Contributions are welcome! If you have an issue or want to suggest improvements, please open an issue or submit a pull request.

Special thanks to [terryluan12](https://github.com/terryluan12) for implemention of streaming requests.


## License

This library is licensed under the Apache License 2.0 License. See the LICENSE file for more information.
