<h1 align="center">jinfer-lfm3</h1>

<p align="center"><strong>Liquid AI LFM3 for the JVM</strong></p>

<div align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>

An implementation of Liquid AI's [LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) models for the JVM.
Small models that chat, see, listen and talk back, on an ordinary CPU.
No llama.cpp, no Python runtime.

</div>

## Highlights

- **Matches llama.cpp.** Greedy replies are identical to llama-server's, token for token.
- **Vision and audio.** LFM3-VL reads images; LFM3-Omni also listens.
- **Spoken replies.** LFM3-Omni answers out loud with a transcript, or in text, as the conversation asks.
- **Text to speech.** LFM3-Omni reads text aloud in 18 languages, in a female or a male voice.
- **Reasoning and tools.** Reasons through a problem and calls tools.
- **Decisions.** d1 answers yes/no, choice and rating questions about text, images, audio and video with probabilities, without generating text.

## Supported models

| Model | Input | Output | GGUF |
|-------|-------|--------|------|
| LFM3-1.7B | text | text | [LFM3-1.7B-GGUF](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF) |
| LFM3-VL-2B | text, images | text | [LFM3-VL-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-VL-2B-GGUF) |
| LFM3-Omni-2B | text, images, audio | text, speech | [LFM3-Omni-2B-GGUF](https://huggingface.co/LiquidAI/LFM3-Omni-2B-GGUF) |
| d1-2B | text, images, video | decisions | [d1-2B-GGUF](https://huggingface.co/LiquidAI/d1-2B-GGUF) |
| d1-Omni-2B | text, images, audio, video | decisions | [d1-Omni-2B-GGUF](https://huggingface.co/LiquidAI/d1-Omni-2B-GGUF) |

Images, audio and speech need the model's projector, the `mmproj-*.gguf` file in the same repository.
Models download on first use and stay cached.
The examples use `Q8_0`.

## Chat

Add `jinfer-lfm3` next to [`jinfer-langchain4j`](../jinfer-langchain4j/README.md) or [`jinfer-spring-ai`](../jinfer-spring-ai/README.md), as shown in [Use it from Java](../README.md#use-it-from-java).
With LangChain4j:

```java
try (var model = JinferChatModel.builder()
        .model("LiquidAI/LFM3-1.7B-GGUF:Q8_0")
        .build()) {
    System.out.println(model.chat("Name three uses for a hash table."));
}
```

## Images and audio

Attach the projector as the `media` companion and pass images and recordings as message content:

```java
try (var omni = JinferChatModel.builder()
        .model("LiquidAI/LFM3-Omni-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf")
        .build()) {

    ChatResponse seen = omni.chat(UserMessage.from(
            ImageContent.from(Path.of("photo.png").toUri()),
            TextContent.from("What is in this picture?")));

    ChatResponse heard = omni.chat(UserMessage.from(
            AudioContent.from(Path.of("question.wav").toUri()),
            TextContent.from("Answer the question in this recording.")));

    System.out.println(seen.aiMessage().text());
    System.out.println(heard.aiMessage().text());
}
```

The same with Spring AI and LFM3-VL:

```java
try (var vision = JinferChatModel.builder()
        .model("LiquidAI/LFM3-VL-2B-GGUF:Q8_0")
        .companion("media", "LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-F16.gguf")
        .build()) {

    ChatResponse response = vision.call(new Prompt(UserMessage.builder()
            .text("Describe this image in one sentence.")
            .media(new Media(MimeTypeUtils.IMAGE_PNG, new FileSystemResource("photo.png")))
            .build()));

    System.out.println(response.getResult().getOutput().getText());
}
```

## Spoken replies

Speech goes through jinfer's `ChatEngine`, from `jinfer-chat`.
A request names the outputs it allows, and LFM3-Omni speaks when the conversation asks for it:

```java
ModelStore hub = ModelStore.standard();
try (var omni = new ChatEngine(
        hub.resolve("LiquidAI/LFM3-Omni-2B-GGUF:Q8_0"),
        Map.of("media", hub.resolve("LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf")),
        PromptCache.Options.DEFAULTS)) {

    var messages = List.of(
            Message.system("Respond with interleaved text and audio."),
            Message.user("Tell me a fun fact about octopuses."));

    omni.complete(
            ChatEngine.Request.builder(messages, omni.loaded().samplingDefaults().resolve())
                    .output(Modality.TEXT, Modality.AUDIO)
                    .build(),
            new ChatEngine.ReplySink() {
                @Override public void on(ChatEngine.Delta delta) { System.out.print(delta.text()); }
                @Override public void onMedia(Media part) { play((Media.Audio) part); }
            });
}
```

The words arrive as they are spoken, and the speech in parts of 80 ms at 24 kHz.
With "Respond in text." as the system turn, the same request is answered in text.
A spoken chat reply comes in the model's one chat voice, the female one: a "using a male voice" clause leaves the voice as it is and cuts the speech short.
A language clause is a request the model may not honour: it can answer in that language, in English, or in text alone (Japanese replies came back as text in every run).
For speech in a set language, use the text-to-speech instruction, which takes the language explicitly.

## Text to speech

Reading text aloud is the same call: the GGUF's reading instruction as the system turn, the text as the user's, and audio as the only output.
It is the model's own mode, not a jinfer API: a dedicated speech model reads text for a fraction of the cost, so jinfer's text-to-speech faces serve those, and a chat model with a voice speaks its replies.

```java
String reading = "Perform TTS in French using a male voice.";

omni.complete(
        ChatEngine.Request.builder(
                        List.of(Message.system(reading), Message.user("Bonjour, comment allez-vous ?")),
                        omni.loaded().samplingDefaults().resolve())
                .output(Modality.AUDIO)
                .build(),
        new ChatEngine.ReplySink() {
            @Override public void onMedia(Media part) { play((Media.Audio) part); }
        });
```

The voice is `female` or `male`.
The language is one of `ar`, `da`, `de`, `el`, `en`, `en-gb`, `es`, `fi`, `fr`, `he`, `it`, `ja`, `ko`, `nl`, `pl`, `pt`, `sv`, `tr` and `zh`.
