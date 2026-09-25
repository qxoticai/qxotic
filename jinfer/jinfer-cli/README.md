# jinfer CLI

```text
jinfer [model options] <command> [options] [input]
```

Commands name applications; shared options configure the model.
Model options work before or after the command:

```sh
jinfer chat -m model.gguf --temp 0.3
jinfer -m model.gguf --temp 0.3 chat
jinfer -m model.gguf chat --temp 0.3
```

## Commands

| Command | Behavior |
|---------|----------|
| `chat` | Interactive conversation; `/quit`, `/exit`, `/context` |
| `instruct` (`prompt`) | Generate one response; text streams by default |
| `server` (`serve`) | Serve the model's language or transcription HTTP API automatically |
| `speak` | Synthesize and play speech by default |
| `transcribe` | Print the transcript of an audio file or stdin |
| `pull` | Download models and print their local paths |
| `list` | List cached model references and sizes |
| `cache-info` | Inspect a prompt/KV cache, including corruption diagnostics |

```sh
jinfer instruct -m model.gguf "Explain virtual threads."
jinfer -m model.gguf --system-prompt "Be concise." chat
jinfer server -m model.gguf --port 8080
jinfer speak -m inflect.gguf "Hello world."
jinfer transcribe -m parakeet.gguf recording.wav
jinfer pull LiquidAI/LFM2.5-350M-GGUF:Q8_0
jinfer list
jinfer cache-info prompts.jkv
```

`jinfer`, `jinfer --help`, `jinfer help speak`, and `jinfer speak --help` provide help without loading or downloading a model.
`jinfer --version` reports the packaged version.
Aliases use exactly the same implementation and help as their primary command.
Adding `--help` to a malformed invocation shows help without resolving a model; a literal `--help` used as an option value or after `--` remains input.

## Shared model settings

- `-m`, `--model`: a local file or `[host/]owner/repo[@revision][/file][:quant]` reference.
- `--with role=reference`: attach a companion; split at the first `=`, so paths can contain `=`.
- `--threads`: compute workers, distinct from HTTP concurrency.
- Language models share context and batch capacity, sampling, output-token limits, thinking, and speculation settings.
- `--system-prompt` applies to chat and instruct; server clients supply system messages in requests.

Scalar settings use the last supplied value.
Companion roles must be unique.
Select the model with `--model`; `--with tokenizer=...` selects a compatible tokenizer override.
Unspecified sampling settings use the model's recommendations before engine fallbacks.
`--raw-prompt` bypasses the conversation template, so explicitly supplied system prompts, thinking controls, and reasoning budgets are rejected together with it.
`--batch-capacity` sets the runtime's default prefill/scratch width; it is not server concurrency or a universal allocation limit.
Threads and batch capacity are configured before runtime initialization.

Application-specific options follow the command.
Unknown or inapplicable options fail before model resolution.
Model-dependent checks, such as companion support, run once the required model information is available.

## Input and output

Use one quoted positional input, or `-` for stdin.
Use `--` before input that starts with a dash:

```sh
jinfer instruct -m model.gguf -- "--help is useful."
jinfer instruct -m model.gguf "Write a greeting." | jinfer speak -m inflect.gguf -
```

Text stdin is UTF-8 and is read to EOF.
Boolean switches take no value in command syntax: use `--stream` / `--no-stream` and `--echo` / `--no-echo`.

Speech plays after full synthesis by default:

```sh
jinfer speak -m inflect.gguf --stream "Play clips as they are synthesized."
jinfer speak -m kokoro.gguf --with voice=af_heart.gguf --output hello.wav "Hello."
jinfer speak -m inflect.gguf --output - "Hello." > hello.wav
```

Explicit `--play`, `--stream`, and `--output` modes cannot be combined.
Audio playback uses the existing cross-platform backend: macOS `afplay`, Windows PowerShell SoundPlayer, Linux `aplay`/`ffplay`.
`--stream` streams generated audio, not incoming text.
Speech status goes to stderr. Playback and WAV output report synthesis time and RTFx, excluding
playback time; streaming reports time to first audio.

Transcription treats stdin as encoded audio unless raw PCM is explicit:

```sh
jinfer transcribe -m parakeet.gguf - < recording.wav
jinfer transcribe -m parakeet.gguf - --raw-pcm
```

Raw PCM must be 16 kHz, mono, signed 16-bit little-endian.
An odd trailing byte is an error.
Cancellation closes the transcription stream without requesting another final decode.
Final results go to stdout; prompts, progress, timings, and errors go to stderr.
Model loading starts with `Loading model ...`, appends a dot every half-second on terminals, and
leaves the line in place when finished. Redirected stderr and `TERM=dumb` get one static line.
Transcription reports when it reads audio, loads the model, and starts transcribing, then prints
audio duration, elapsed time, and RTFx (audio seconds / elapsed seconds; higher is faster).
File and encoded-stdin timings exclude audio decoding and model loading. Live raw-PCM timings
include time waiting for stdin, but exclude model loading and warm-up.
For binary pipelines on Windows, use `cmd` or PowerShell 7.4 or later; older PowerShell versions can convert native bytes to text.
Custom in-process streams remain caller-owned.
The live-input reader is interrupted during cleanup; a borrowed native stdin read may remain blocked until EOF or process exit, so the reader is a daemon.

## Server

`server` and `serve` automatically select the supported API from the model; there is no `--task` option.
The default bind is `127.0.0.1:54154`; `--port 0` selects a free port, reported on stderr.
Non-loopback binds require `--api-key`.

- `--concurrency N`: the existing language server admits up to `2*N` HTTP handlers; the transcription server uses `N` handler threads.
- `--queue-depth N`: waiting language-generation jobs; default 4, 0 disables waiting.
- Language generation currently runs one request at a time; concurrency is not parallel model generation.
- Transcription servers reject language-generation settings, queue depth, prompt caches, and grammar controls.
- Sampling and token budgets are request defaults, not HTTP resource limits.

## Invocation contract

A command is required; bare model options do not select an application implicitly.
Text and audio input are positional arguments.
Use `--with media=...` for projectors and `--with voice=...` for speech voices.
Switches such as `--stream` and `--echo` take no values; their negative forms disable them.
`--think` accepts `on`, `off`, or `inline`.
Unknown and inapplicable options are rejected.

Exit statuses: 0 success/help, 2 invalid invocation, 1 operational failure, 130 handled interruption.
A corrupt cache that can be inspected is reported by `cache-info` on stdout; inability to open the file is an operational failure.

Errors name the rejected value or the operation and file involved.
Unknown commands/options point to scoped help; range and conflict errors explain their correction directly.
Decoder and playback errors put returned backend details beneath a short summary and preserve the original cause.
Player processes may also emit directly to inherited stderr; the CLI does not redirect process-wide file descriptors to capture it.
Unexpected runtime failures include a stack trace, with causes and suppressed exceptions intact.
Chat recovers from refused prompts, but stops on an unexpected model failure.

## Build and test

From the repository root:

```sh
mvn -pl jinfer/jinfer-cli -am package -DskipTests
java --add-modules jdk.incubator.vector -jar jinfer/jinfer-cli/target/jinfer.jar --help
```

Model-free unit/component gate (real weightless engine, isolated cache sources, captured I/O and playback):

```sh
mvn -pl jinfer/jinfer-cli -am test \
  '-Dtest=com.qxotic.jinfer.cli.*Test,ProviderContractTest,PortArtifactsDriftTest' \
  -Dsurefire.failIfNoSpecifiedTests=false -Djinfer.test.noModels=true
```

Real-model gate; models must already be cached:

```sh
mvn -pl jinfer/jinfer-cli -am test \
  '-Dtest=com.qxotic.jinfer.cli.CliIT,com.qxotic.jinfer.cli.SpeakTest' \
  -Dsurefire.failIfNoSpecifiedTests=false -Dsurefire.excludedGroups=bench,driver \
  -Djinfer.test.requireModels=true
```

The gate uses LFM2.5-350M Q8, Inflect Nano v2 Q8, and Kokoro Q8 with `af_heart`.
The existing `-Djinfer.testModel.<filename>=<path>` override allows an explicitly selected local checkpoint.
Tests never implicitly download model fixtures.

Native executable:

```sh
make -C jinfer/jinfer-cli native
```

Run parser/help, packaged-JAR, native, redirection, and playback checks on Linux, macOS, and Windows.
Platform-selection tests and captured playback do not replace an audible smoke test on a real audio device.

### Coverage and distribution checks

The existing `coverage` profile generates `jinfer/jinfer-cli/target/site/jacoco/index.html`:

```sh
mvn -Pcoverage -pl jinfer/jinfer-cli -am test \
  '-Dtest=com.qxotic.jinfer.cli.*Test,ProviderContractTest,PortArtifactsDriftTest' \
  -Dsurefire.failIfNoSpecifiedTests=false -Djinfer.test.noModels=true
```

`ArgumentMatrixTest` covers public syntax and numeric boundaries.
`WorkflowTest` exercises `Main.run` through actual GGUF loading with weightless, test-only providers, including automatic server selection and multipart transcription.
Command tests cover output routing, HTTP/SSE/CORS, input failures, playback failures, and cleanup.
Forked Java checks inherit the JaCoCo agent when coverage is enabled.
Native executable behavior is smoke-tested separately; JaCoCo does not measure native execution.

After packaging, verify the actual artifacts:

```sh
mvn -Pcoverage -pl jinfer/jinfer-cli -am test \
  -Dtest=com.qxotic.jinfer.cli.DistributionIT \
  -Dsurefire.failIfNoSpecifiedTests=false -Dsurefire.excludedGroups=bench,driver \
  -Djinfer.test.executable="$PWD/jinfer/jinfer-cli/target/jinfer"
```

Omit `jinfer.test.executable` to test only the JAR, or point it at `jinfer.exe` on Windows.
This gate also checks that the executable JAR contains the production provider services and excludes the test provider.
Coverage includes platform-specific terminal code; headless tests cannot exercise every real-console or other-OS branch.
