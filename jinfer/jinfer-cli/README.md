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

With GraalVM 25.0.3 or newer, compile the CLI into a native executable:

```sh
make -C jinfer/jinfer-cli native
JINFER_BIN="$PWD/bin/jinfer"
jinfer() { "$JINFER_BIN" "$@"; }
jinfer --help
```

The native executable runs without a JVM.

## Model selection

`--model` / `-m` accepts a local GGUF file or a hub reference. Hub models are downloaded on first use
and reused from the cache. These examples use a small language model:

```sh
LM=LiquidAI/LFM2.5-350M-GGUF:Q8_0
jinfer instruct -m "$LM" "Explain virtual threads in two sentences."
```

Model and generation options can precede or follow the command. Command-specific options follow it.
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

`/context` shows context usage. `/quit`, `/exit`, or EOF ends the session.

## Text generation: `instruct`

Pass a quoted prompt, or use `-` to read it from stdin. `prompt` is an alias for `instruct`.

```sh
jinfer instruct -m "$LM" --temp 0 -n 128 "Name three uses for a hash table."
printf '%s\n' 'Write a short greeting.' | jinfer prompt -m "$LM" -
```

`--temp 0` selects greedy generation; `-n` limits output tokens. Responses stream by default.
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

Start an OpenAI-compatible language API. `serve` is an alias for `server`.

```sh
jinfer server -m "$LM" --port 8080 --temp 0 -n 256
```

Send a request from another terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name the planets."}]}'
```

To serve transcription instead, stop the language server and select the ASR model:

```sh
jinfer serve -m "$ASR" --port 8080
```

From another terminal:

```sh
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@speech.wav -F response_format=text
```

The default bind is `127.0.0.1:54154`. To listen on other interfaces, supply `--host` and `--api-key`:

```sh
jinfer server -m "$LM" --host 0.0.0.0 --port 8080 --api-key local-demo-key
```

For this example, clients send `Authorization: Bearer local-demo-key`.

## Model downloads: `pull`

Download one or more models and print their local paths:

```sh
jinfer pull "$LM" "$TTS" "$ASR"
```

Use `--force` to re-download a model. Set `JINFER_OFFLINE=1` to run using cached files only:

```sh
jinfer pull --force "$LM"
JINFER_OFFLINE=1 jinfer instruct -m "$LM" "Hello."
```

Set `JINFER_MODELS` to choose a different model-cache directory.

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
