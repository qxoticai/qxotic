<h1 align="center">jinfer CLI</h1>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache" alt="License: Apache 2.0"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
</p>

`jinfer-cli` is an executable fat JAR containing its runtime dependencies and model providers.
Do not add `com.qxotic:jinfer-cli` as a Maven or Gradle dependency. Use the libraries and adapters
described in the [jinfer guide](../README.md) for application integration.
GraalVM Native Image can compile the CLI into a standalone executable.

## Getting the command

- The release archive, `jinfer-<version>-<platform>.tar.gz` (a zip on Windows), an asset of the release on GitHub, built here by `make dist`: `bin/jinfer`, this README, the license and the [notice](../jinfer-native/NOTICE).
  Unpack it anywhere and run `bin/jinfer`; it needs no Java.
  Linux x86-64 and aarch64 (glibc 2.17 or later), macOS aarch64 and Windows x86-64; the x86-64 builds need x86-64-v3, any CPU since 2013.
- The executable JAR, or a native image for this machine, built from the source as below.

## Build and run

Run the build commands from the repository root. They require Maven 3.9+, CMake, and a C compiler.
The examples use a POSIX shell.

### Executable JAR

With JDK 25, build the JAR and define `jinfer` to run it:

```sh
mvn -pl jinfer/jinfer-cli -am package -DskipTests
JINFER_JAR="$PWD/jinfer/jinfer-cli/target/jinfer.jar"
jinfer() {
  java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
    -jar "$JINFER_JAR" "$@"
}
jinfer --help
```

### Native executable

With GraalVM 25.0.4 or newer, compile the CLI into a native executable:

```sh
make native   # from the repository root
JINFER_BIN="$PWD/bin/jinfer"
jinfer() { "$JINFER_BIN" "$@"; }
jinfer --help
```

The native executable runs without a JVM.

## Model selection

`--model` / `-m` accepts a local GGUF file or a hub reference.
Hub models are downloaded on first use and reused from the cache; like the hubs, a reference reads its owner and repository in any case.
These examples use a small language model:

```sh
LM=LiquidAI/LFM2.5-350M-GGUF:Q8_0
jinfer instruct -m "$LM" "Explain virtual threads in two sentences."
```

Options shared by several commands may precede or follow the command; an option of one command follows it.
Use `jinfer <command> --help` for the full option reference:

```sh
jinfer chat --help
jinfer speak --help
jinfer server --help
```

## Interactive chat: `chat`

```sh
jinfer chat -m "$LM"
```

Set instructions for the conversation with `--system-prompt`:

```sh
jinfer chat -m "$LM" --system-prompt "Answer concisely." --temp 0.3
```

`/help` lists the in-chat commands.
`/context` shows the tokens used out of the context capacity.
`/quit`, `/exit`, or EOF ends the session.

## Text generation: `instruct`

Pass a quoted prompt, or use `-` to read it from stdin. `prompt` is an alias for `instruct`.

```sh
jinfer instruct -m "$LM" --temp 0 -n 128 "Name three uses for a hash table."
printf '%s\n' 'Write a short greeting.' | jinfer prompt -m "$LM" -
```

`--temp 0` selects greedy generation; `-n` limits output tokens. Responses stream by default.
A reply cut short says so on stderr: a full context names `--context-capacity`, a spent budget names `-n`, and a capped reasoning span names `--max-reasoning-tokens`.
Use `--no-stream` to print the completed response:

```sh
jinfer instruct -m "$LM" --no-stream "Explain a binary search." > answer.txt
```

Use `--` before a prompt beginning with a dash:

```sh
jinfer instruct -m "$LM" -- "--help is a command-line option. Explain its purpose."
```

## Speech synthesis: `speak`

Speech plays after synthesis. `--stream` starts playback as audio becomes available;
`--output` saves a WAV instead of playing it.

```sh
TTS=remixerdec/Inflect-Nano-v2-GGUF:Q8_0
jinfer speak -m "$TTS" "Hello world."
jinfer speak -m "$TTS" --stream "Play the first sentence. Then the second."
jinfer speak -m "$TTS" --output speech.wav --speed 1.2 "Hello world."
```

Read text from another command, or write WAV bytes to stdout with `--output -`:

```sh
jinfer instruct -m "$LM" -n 64 "Write a short greeting." | jinfer speak -m "$TTS" -
printf '%s\n' 'Hello world.' | jinfer speak -m "$TTS" - --output - > speech.wav
```

Kokoro requires eSpeak on `PATH` and a voice file supplied through `--with`:

```sh
jinfer speak -m simonfxr/kokoro.cpp-GGUF/kokoro-82m-q8_0.gguf \
  --with voice=simonfxr/kokoro.cpp-GGUF/voices/kokoro-voice-af_heart.gguf \
  "Hello from Kokoro."
```

Playback uses `afplay` on macOS, PowerShell on Windows, or `aplay`/`ffplay` on Linux.

## Transcription: `transcribe`

Transcribe `speech.wav` from the previous examples, or substitute your own recording:

```sh
ASR=mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf
jinfer transcribe -m "$ASR" speech.wav
jinfer transcribe -m "$ASR" - < speech.wav > transcript.txt
```

The native executable requires `ffmpeg` on `PATH` to decode audio.
For raw PCM input, use `--raw-pcm` with 16 kHz mono signed 16-bit little-endian samples:

```sh
ffmpeg -nostdin -loglevel error -i speech.wav -ar 16000 -ac 1 -f s16le - \
  | jinfer transcribe -m "$ASR" - --raw-pcm
```

## HTTP server: `server`

Start an OpenAI-compatible API for the model: chat for a language model, embeddings, reranking, transcription or decisions for the others.
`serve` is an alias for `server`.

```sh
jinfer server -m "$LM" --port 8080 --temp 0 -n 256
```

Send a request from another terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the planets."}]}'
```

A model with a voice, LFM3-Omni with its projector attached through `--with media`, answers in speech when the request asks for it, in OpenAI's shape: `modalities` of `["text", "audio"]` and an `audio` object whose `format` is `wav` or `pcm16`.

```sh
jinfer server -m LiquidAI/LFM3-Omni-2B-GGUF:Q8_0 \
  --with media=LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf --port 8080
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Respond with interleaved text and audio."},
                   {"role":"user","content":"What is the capital of France?"}],
       "modalities":["text","audio"],"audio":{"format":"wav"}}'
```

The reply's `message.audio` carries the speech as base64 `data`, its `transcript`, an `id` and `expires_at`; `content` is null, as OpenAI's is.
To continue the conversation, send the assistant turn back as `{"role":"assistant","audio":{"id":"..."}}` and the server replays it exactly as spoken.
An id lives one hour on the server that minted it, among the last 4096; an unknown or expired one is refused.
A streamed spoken reply carries the transcript and the speech in `delta.audio` chunks naming that id, the speech as `pcm16` only, 16-bit mono at the voice's rate, and the `expires_at` chunk last.
`/v1/models` lists the `output_modalities` the served model has.
A spoken reply cannot offer tools or follow a grammar, and does not reason; the server never speaks unless the request allows audio.
`audio.voice`, which OpenAI's clients always send, is accepted and ignored: LFM3-Omni's spoken chat replies come in its one chat voice.
A language clause is a request the model may not honour: it can answer in that language, in English, or in text alone (Japanese replies came back as text in every run).
For speech in a set language, LFM3-Omni's text-to-speech instruction takes the language explicitly (see [jinfer-lfm3](../jinfer-lfm3/README.md#text-to-speech)).

To serve transcription instead, stop the language server and select the ASR model:

```sh
jinfer serve -m "$ASR" --port 8080
```

From another terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=text
```

An embedding or reranking model is served the same way, on the endpoint its kind takes:

```sh
jinfer server -m Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 --port 8080
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["The cat sat on the mat.","A kitten rests on a rug."],"dimensions":256}'
```

`/v1/embeddings` accepts a non-empty string or a batch of at most 2048 inputs, with `float` or `base64` output; token-ID inputs are not supported.
Each top-level string or content-part array counts as one input; larger batches return HTTP 400 and must be split across requests.
`dimensions` shortens a Matryoshka-trained model's vectors (Qwen3-Embedding: 32 to 1024, EmbeddingGemma 2: 128 to 768) and is refused on a fixed-width one (LFM2.5-Embedding).
Text is embedded as sent, so a retrieval prefix from the model card (LFM2.5-Embedding's `query: `) is the client's to add.
An input longer than the context is refused with its index, and the rest are packed into as few forward passes as the context allows.

A multimodal embedder (EmbeddingGemma 2) also embeds images, audio and video once its projector is attached with `--with media`.
An input is then either a string or an array of content parts in the chat endpoint's syntax, `text`, `image_url`, `input_audio` and `video_url` with base64 data URIs, and the parts of one input make one vector:

```sh
jinfer server -m unsloth/embeddinggemma-2-GGUF:Q8_0 \
  --with media=unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf --port 8080
curl -sS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"input":["task: search result | query: a cat",
       [{"type":"text","text":"A photo: "},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0..."}}]]}'
```

A video is sampled one frame a second, at most 32, as the model's reference does.
`/v1/models` lists the input modalities the server takes; a medium it does not take is refused naming them.

```sh
jinfer server -m mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0 --port 8080
curl -sS http://127.0.0.1:8080/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"query":"capital of France","documents":["Paris is in France.","Bananas are yellow."],"top_n":1,"return_documents":true}'
```

`/v1/rerank` answers `results` most relevant first, each with its `index`, `relevance_score` and, on request, `document`.
Qwen3-Reranker scores are probabilities in [0, 1].
LFM2.5-ColBERT scores are unbounded MaxSim sums that sit close together (about 28 to 30 out of 32), so rank by them and never threshold them.
`--context-capacity` sizes the server's state (default `0`: 4096, or the model's maximum when smaller; `max` or `-1` for the model's maximum); the language options are refused.

A decision model serves `POST /v1/systemone`, in the request and response shape llama.cpp uses:

```sh
jinfer server -m LiquidAI/d1-2B-GGUF:Q8_0 \
  --with media=LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf --port 8080
curl -sS http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice and I want my money back today.",
       "questions":{
         "refund":{"type":"noul","instructions":"Is the customer asking for a refund?"},
         "team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Charges and refunds","shipping":null}},
         "urgency":{"type":"score","instructions":"How urgent is this?","criteria":["Can wait","Today","Right now"]}}}'
```

`state` is a text, any other JSON value (read as JSON), or `null` when the files alone are judged.
`files` holds images, audio and video as base64 `data:` URIs, read before the state.
d1 reads images and video, d1-Omni also audio, vLLM's d3 images; `/v1/models` lists the modalities the loaded checkpoint really reads.
A `noul` answer is the probability of yes; a `choice` names the likeliest option with every option's probability; a `score` is the expected level (the probability-weighted mean of the levels) with each level's probability; `confidence` is the probability of the likeliest option, for every model.
Each question is answered as if asked alone; on d1 and d1-Omni, `"together": true` reads the state once for all of them, faster, an answer then free to shift with the questions beside it.
d3 is not trained to, and answers each question alone either way.

The default bind is `127.0.0.1:54154`. To listen on other interfaces, supply `--host` and `--api-key`:

```sh
jinfer server -m "$LM" --host 0.0.0.0 --port 8080 --api-key local-demo-key
```

For this example, clients send `Authorization: Bearer local-demo-key`.

Every server offers `GET /health`, `/v1/models`, `/props`, and `/metrics`, even while busy.
`/health` needs no API key; the other probes use the configured key.
The transcription server's `/props` reports the model name and input sample rate.
The retrieval server's `/props` reports the task, the context capacity and, for an embedder, its dimensions.
The decision server's `/props` reports the task and the context capacity.

## Model downloads: `pull`

Check upstream for one or more models and print their local paths:

```sh
jinfer pull "$LM" "$TTS" "$ASR"
jinfer pull --force "$LM"
```

`pull` checks for updates and reuses unchanged files when possible.
Use `--force` to download again, even if cached.
If an update fails, the previously usable model remains available.

Inference uses an installed model without checking upstream.
Set `JINFER_OFFLINE=1` to use cached models only:

```sh
JINFER_OFFLINE=1 jinfer instruct -m "$LM" "Hello."
```

`JINFER_OFFLINE` accepts `1`, `true`, `on`, or `yes` to enable offline mode, and
`0`, `false`, `off`, or `no` to disable it. Word values are case-insensitive.
On the JVM, `-Djinfer.offline` takes the same values and wins over the environment when set.

Set `JINFER_MODELS` to choose a different model-cache directory.
Update checks and forced downloads require network access.
An exact cached file pinned to a full commit can be pulled offline.

## Cached models: `list`

Show cached model references and their disk usage:

```sh
jinfer list
```

## Prompt caches: `cache-info`

Save a prompt cache with `--cache`, reuse it without modification with `--cache-ro`, and inspect it
with `cache-info`:

```sh
jinfer instruct -m "$LM" --cache prompts.jkv --temp 0 -n 64 "List the planets."
jinfer instruct -m "$LM" --cache-ro prompts.jkv --temp 0 -n 64 "List the planets."
jinfer cache-info prompts.jkv
```

## Scripting

Generated text and WAV bytes go to stdout; diagnostics and timings go to stderr.
For WAV pipelines on Windows, use `cmd` or PowerShell 7.4+.

Exit statuses: `0` success/help, `1` operational failure, `2` invalid invocation, `130` handled interruption.

## License

Apache 2.0 for the command's own code.
The executable in the release archive is built with Oracle GraalVM and carries its Native Image runtime under the [GraalVM Free Terms and Conditions](https://www.oracle.com/downloads/licenses/graal-free-license.html), spelled out in the archive's [NOTICE](../jinfer-native/NOTICE): use it as you like, commercially too; a product sold for a fee that bundles a copy of it is not covered by those terms.
For that case build it with GraalVM CE, `JAVA_HOME=<GraalVM CE> make image` in `jinfer/jinfer-cli`, which carries the runtime under GPLv2 with the Classpath Exception and no fee condition; the community compiler is slower, prefill most of all.
