# Release notes

## 0.3.1

A command-based CLI, embeddings and reranking over HTTP, faster prefill, and a release-wide round of fixes.
Every artifact whose code changed moves to 0.3.1, and so does everything that depends on one, so a published POM never names a stale version.
The one unchanged artifact is `json`, at 0.2.0.
`jinfer-bom` 0.3.1 names each artifact at its own version, so one import pins a coherent set.

### CLI

- **Commands, not mode flags.** Every invocation names its command: `chat`, `instruct`, `server`, `speak` and `transcribe` run a model; `pull`, `list` and `cache-info` manage the model cache.
  The legacy mode flags `--chat`, `--server`, `--transcribe` and `--speak` are gone, a bare `-m model.gguf` no longer implies `instruct`, and `--mmproj` is replaced by `--with media=<ref>`.
  Text and audio input are positional, switches such as `--stream` and `--echo` take no value, and `--think` accepts `on`, `off` or `inline`.
- **Help per command.** `jinfer <command> --help` and `jinfer help <command>` name only the options that command accepts, and an option that does not apply to it is refused rather than ignored; `/help` lists chat's own commands.
- **Live transcription.** `jinfer transcribe -m <model> - --raw-pcm` reads 16 kHz mono PCM from stdin and draws the live view; `-` alone reads an encoded audio file from stdin.
- **`speak` plays by default.** Without `--output`, speech is played after synthesis, so `--play` is gone; `--stream` starts playback with the first clip.
- **One server limit.** `--concurrency N` holds up to `N` requests and refuses the next one with a message naming the limit; `--queue-depth` is removed.
  Health, props, models and metrics answer outside that gate, and the transcription server has the same probes and metrics as the language server.
- **Embeddings and reranking over HTTP.** `jinfer server -m <embedding model>` serves OpenAI's `POST /v1/embeddings` (`float` or `base64`, Matryoshka `dimensions` where the model has them), and a reranker serves `POST /v1/rerank` (`query`, `documents` as strings, `top_n`, `return_documents`) with llama.cpp's response shape; the header tells the two apart, so the weights load once.
- **Chat keeps going when the context is full.** The oldest exchanges are dropped instead of every later turn being refused.
- **Safer pulls.** A failed download or refresh leaves the previously usable model in place, and `list` prints one line per reference.
- **One-line refusals.** A library refusal prints as one line with exit status 1; an invalid invocation exits with 2.
  An explicit `--speculation-depth` is refused where no draft head can use it.

### Performance

- **Prefill per row.** Residual adds, SwiGLU and KV commits run one row per job across the ports instead of serially over the batch, for up to 20% more prompt throughput at 16 threads, depending on the model.
- **AVX-512-VNNI prefill in `jam-native` 0.3.1.** A 32x4 tile shares every activation broadcast between two weight vectors and keeps K-quant sub-block scales integer; on Zen 5 the Gemma 4 E2B quants prefill faster than llama.cpp on the same machine.
- **Kernels that both compilers optimize.** The shipped libraries come from clang (Linux, macOS) and gcc (Windows), and some kernels compiled badly under one of them; their source now produces the fast loop under both, with bit-identical results.
  On the pre-AVX2 tiers prefill runs 2.3x to 3.3x faster (Gemma 4 E2B Q4_K 67 to 168 t/s on `sse3`), Q5_K and Q6_K gain 10% to 14% on AVX2 and AVX-VNNI, and the SSE3 tiers gain float, F16, BF16 and Q1_0 kernels.
- **`jam-vector` picks its band tile per JIT.** GraalVM CE 25.4 and newer allocate all 32 AVX-512 registers and take the 4x4 band (9% to 36% faster gemm, 16% to 23% faster prefill); OpenJDK C2, Oracle GraalVM and older GraalVM CE take 3x3, which on C2 is 14% to 24% faster than its former 4x4 default.

### Also

- **`gguf` 0.3.1.** `GGUFFormatException` extends `IllegalArgumentException`, so a malformed file reads as the refusal it is.
  Files are read and written little-endian on every platform, and the tensor name limit counts UTF-8 bytes; the writer stops at ggml's 63.
- **Responses API.** A deadline or a cancel ends a Responses reply as `incomplete`, with its reason.
- **`/v1/models` reports input modalities**, so a client can tell which models accept images or audio.
- **LangChain4j 1.21.0.** `jinfer-langchain4j` builds on it and passes its chat, streaming, AI-service, tools and JSON-schema TCKs.

### Behaviour changes

Coming from 0.3.0, these act differently:

- **GPT-2 style tokenizers in `toknroll-hf`.** A `ByteLevel` pre-tokenizer applies the GPT-2 split unless it says `use_regex: false`, as Hugging Face tokenizers does, so GPT-2, DistilGPT-2, Phi-2 and RoBERTa encode differently: now identical to `r50k_base` where they share its vocabulary.
- **Refusals instead of silent defaults.** A generative model loaded as an embedder throws `IncompatibleModelException`, and the positive-integer tuning properties (`jinfer.decodeBlockSize` and the like) refuse a value they cannot use instead of ignoring it; the undocumented `vis.squareResize` switch is gone.
- **Spring AI.** Sampling options are validated when they are built, naming the property, a blank rerank model is refused at boot, and `getReasoningBudget()` / `getReasoningBudgetMessage()` are deprecated for `getMaxReasoningTokens()` / `getReasoningCutoffMessage()`.
- **Stricter templates and grammars.** An unknown Jinja statement such as `{% endset %}` fails instead of rendering nothing, and malformed GBNF or unsatisfiable schema bounds (`minLength` over `maxLength`, `minItems` over `maxItems`) are refused instead of compiling to a different language.
- **`enable_thinking` decides.** A request's `chat_template_kwargs.enable_thinking` overrides the `thinking` flag for the whole request, in the library as on the server.
- **Offline precedence.** `-Djinfer.offline` decides when set, else `JINFER_OFFLINE`; both accept `1`/`true`/`on`/`yes` and `0`/`false`/`off`/`no`, so `-Djinfer.offline=false` overrides `JINFER_OFFLINE=1`.
- **CLI output.** Under 32 tokens the stats line reads `N tokens in X s` instead of a rate, a reply cut short says why on stderr, and `chat` and `instruct` refuse `--with media`, which only the server uses.
- **Server timings.** `timings.prompt_n` counts only the tokens evaluated, as llama.cpp does, so `prompt_per_second` no longer includes cached tokens.
- **`gguf` writer.** Tensor names longer than 63 UTF-8 bytes are refused, as ggml cannot load them.

### Fixes

- **Concurrent requests on one thread.** Two requests running inline (`jinfer.threads=1`) no longer share the same attention and matmul scratch.
- **Prompt cache.** A failed restore no longer leaves stale blocks that the next resume would trust.
- **Thinking.** Gemma 4 no longer shows its channel name `thought` as reasoning, and Laguna answers with thinking off instead of ending the turn at once.
- **Chat templates.** Jinja gains `{% raw %}`, `{% break %}` and `{% continue %}`, one-sided `strip` with characters, `split` with a limit, Python float formatting, and undefined (not None) missing macro arguments.
- **Grammars.** `maxItems: 0` means the empty array, and logits past the vocabulary are masked.
- **Server.** A Responses reply with text and tool calls streams both, the transcription server honours `--request-timeout 0`, and every request logs its status and duration.
- **Libraries.** `jam-scalar` refuses a weight stride under `k`; `jam-vector` reports itself unavailable without native access and refuses an unknown `jam.vector.tile`.
  `jota-memory` frees its staging buffers, and safetensors reports a corrupt shape as a format error.
  `Tokenizer.countBytes` handles tokens longer than 256 bytes, SentencePiece decodes a run of byte tokens into a nearly full buffer, and `toknroll-hf` caches a 404 only under a commit SHA.
  The undocumented `toknroll.fast.*` and `toknroll.spbpe.*` tuning properties are gone.

## 0.3.0

jinfer only: speech recognition, and the CLI on Maven Central.
Each project now releases on its own version, so gguf, json, safetensors, jota, jam and toknroll stay at 0.2.0, unchanged since that tag.
`jinfer-bom` names the whole set, jinfer at 0.3.0 and those at 0.2.0, so one import still pins a coherent release.

### Speech recognition

- **NVIDIA Parakeet.** `jinfer-parakeet` runs the Parakeet family: FastConformer encoder, token-and-duration transducer, cased and punctuated text with an audio span and a confidence per token.
  Five checkpoints, `parakeet-tdt-0.6b-v3` (multilingual, no language flag) through `parakeet-tdt_ctc-110m`, as the GGUF files jinfer and parakeet.cpp share.
- **Accuracy is the reference's.** 2.04% WER on the LibriSpeech test-clean set parakeet.cpp benchmarks, scored with its own normalization; at `F16` the transcript matches parakeet.cpp word for word, and the quantizations differ from it by 0.04% to 0.34%.
  Competitive on speed with both native engines: at `Q4_K`, 25.1x realtime against parakeet.cpp's 12.6x and sherpa-onnx's 18.6x on eight cores, and 81.2x with the 110M model. [How to reproduce it](jinfer/jinfer-parakeet/WER.md).
- **Streaming.** `TranscriptionStream` in `jinfer-core` takes audio as it arrives and returns final pieces that never change, plus a provisional tail for a responsive UI.
  Decoding follows NeMo's chunking, 10 s of left context, a 2 s chunk and 2 s of right context, with decoder state carried across chunks so words continue over the boundaries.
  Offline transcription runs the same path in 46 s chunks, so a long recording never holds a long attention matrix.
- **Every front end.** `JinferTranscriptionModel` for LangChain4j and for Spring AI, `POST /v1/audio/transcriptions` on the server (OpenAI-compatible multipart, `json`, `text` or `verbose_json`), and `jinfer --transcribe <file>` on the CLI.
- **Live from the microphone.** `--transcribe -` reads 16 kHz mono PCM from stdin and draws a live view on stderr: committed words settling into the scrollback, the draft tail behind them, a waveform on the voice and doubtful words flagged.
  `--theme mint|nord|catppuccin|ember|frost|mono` picks the palette, which degrades to 256 colors, to 16, and to plain attributes under `NO_COLOR`; the chrome falls back to ASCII outside UTF-8.
  Redirected, the same run prints plain lines, so it scripts.

### Published artifacts

- **`jinfer-cli` on Maven Central.** An executable fat jar with a dependency-free POM, so `jbang jinfer@qxoticai` is one download: chat, the server and transcription in one command.
- **`jinfer-parakeet` on Maven Central**, and in `jinfer-models-all` with the other providers.

### Also

- **Audio decoding has limits.** `jinfer-codecs` refuses input past `jinfer.codecs.maxAudioMinutes` (an hour by default) instead of buffering it, and clamps decoded PCM to [-1, 1] so a lossy decoder's overshoot is not an error.
- **`jinfer-bench` scores transcription.** `--dump` writes a transcript per utterance and `--gate <percent>` turns a WER run into a pass or fail for CI; `jinfer/scripts/score_asr.py` scores jinfer, parakeet.cpp and sherpa-onnx runs the same way.
- **Kernels.** `layerNorm` is vectorized and parallel, and `Workspace` moved to `jinfer-core`, where scratch is reused across windows rather than reallocated.

### Known limits

- Parakeet takes mono 16 kHz audio and refuses anything else rather than resampling it quietly, since a rate mismatch degrades recognition without an error.
- Short windows around digital silence can transcribe as nothing ([NVIDIA-NeMo/Speech#15757](https://github.com/NVIDIA-NeMo/Speech/issues/15757)), so a live microphone with a room noise floor does better than padded silence.

## 0.2.0

First release on Maven Central: `com.qxotic` artifacts for jota, jam, jinfer, toknroll, gguf, json and safetensors, with `jinfer-bom` managing the versions.

### Behaviour to know about

- **Thinking policy.** Every chat template states how its checkpoint reasons: `NONE`, `OPTIONAL` or `ALWAYS`.
  A model that always reasons, LFM2.5-8B-A1B and gpt-oss among them, refuses `thinking(false)` with a message naming the remedy instead of leaking its reasoning into the visible text.
  `maxReasoningTokens` caps the span on every model that has think markers: `--max-reasoning-tokens` in the CLI, `max_reasoning_tokens` on the server, and the same knob in langchain4j and Spring AI.
- **Structured output in langchain4j.** A JSON-schema response format is enforced by the grammar and, so that the model knows which fields exist, described in one line appended to the last user message.
  `describeSchema(false)` on the builder leaves the prompt untouched.
- **Reply scaffolding is guarded.** The family's reply language now masks control tokens wherever the language expects a specific one, so a model cannot derail its own tool-call header or channel scaffolding; free text stays free.
- **Batch embeddings.** A packed embedding group larger than the state's batch capacity is ingested in chunks; earlier builds failed the request.
- **Grammars with thinking off.** A completed grammar ends the turn cleanly on every family, and Qwen 3.5's thinking-off prefix no longer swallows a raw grammar.
- **Mellum 2.** JetBrains Mellum 2 (`jinfer-mellum`, architecture `mellum`) is a chat family: 64-expert MoE with sliding-window attention, ChatML with JSON tool calls, and the `mellum2` pre-tokenizer in toknroll.
  The Instruct checkpoint answers directly; the Thinking checkpoint reasons in `<think>` spans with the usual thinking switch.
- **Q5_0 weights.** The legacy `Q5_0` quantization loads and runs with jam kernels on every backend (llama.cpp's quantizer picks it for rows that are not a multiple of 256, such as the expert down-projections of every k-quant Mellum 2 mix).
- **Kokoro.** Kokoro 82M joins Inflect as a speech family (`jinfer-kokoro`): the model GGUF plus one voice pack as the `voice` companion, nine languages by voice, and espeak-ng on `PATH` for the phoneme front end, which speaks misaki's dialect - the one the model was trained on.
- **Speech companions.** The pronunciation lexicon of Inflect models is the `lexicon` companion, attachable on both speech builders and as `spring.ai.jinfer.speech.companions.lexicon`; a Kokoro voice attaches the same way.
- **Speech at the phoneme level.** `SpeechSynthesisModel.synthesize` takes phoneme ids and `phonemizer()` exposes the model's front end, a `Phonemizer` in `jinfer-core` that is to speech what a tokenizer is to text; `speak(text)` remains the text door.
  `Espeak` in `jinfer-codecs` drives espeak-ng for both speech families; the per-family symbol tables and espeak drivers are gone.
- **Spring Boot examples.** `mvn spring-boot:run` runs with full tiered compilation; its default `optimizedLaunch` pinned C1 and slowed the Vector API about a hundredfold.
- **CLI errors.** A bad `--cache` file, a read-only cache root and other wrapped IO failures print one `ERROR` line.
- **Tools with constrained output.** A request may offer tools together with a JSON schema or a grammar: the family's reply language then offers a tool call or the document, so langchain4j's tool-round-then-structured-answer loop works in one service call.
  A forced tool call with constrained output is still refused, and a family without a combined language refuses at request time.
- **Stringified arguments.** A small model that sends an array or object argument as a JSON string, Llama 3.2 1B does, gets it unwrapped where the tool's schema declares that shape.
- **Gemma 4 video.** A `VideoContent` (langchain4j), a video `Media` (Spring AI) or a `video_url` part (server) renders the way the Gemma 4 processor does: every sampled frame is a timestamped image block, `mm:ss <|image>...<image|>`, one space between frames.
  Qwen 3.5 still refuses video: its vision tower takes images only.
- **Browser URLs as model refs.** A repository page pasted from the browser (`https://huggingface.co/owner/repo`, its `tree`, `blob` and `resolve` views, ModelScope alike) is the ref it spells, so it lands in the same cache as `owner/repo`; `huggingface.co/owner/repo` is accepted as a host spelling.
  A plain URL that answers with a web page is refused and never kept in the cache.
- **Vector API check in the library.** A JVM started without `--add-modules jdk.incubator.vector` fails at model load with the one-line remedy, on every binding, instead of a NoClassDefFoundError inside a kernel.
- **Builder ranges.** The langchain4j builders refuse an out-of-range temperature, top-p, top-k, min-p, output limit, timeout or speech speed where it is set, with the range in the message.
- **`--raw-prompt` writes the start token.** The raw lane prepends the model's start tokens (BOS, where the family has one) unless the prompt already spells them, as llama.cpp's `add_bos_token` does; an LFM 2.5 raw prompt no longer decodes to noise.
- **Errors that name the mistake.** An unknown flag in last position is reported as unknown; the server answers 404 for a model name it does not serve and refuses `max_tokens: 0`; the always-reasoning refusal names the lever on every front end; a null embedding batch fails instead of returning nothing; the Narrate and Detect demos report a missing image in one line.
- **Vision prefill no longer collapses once a model has answered.** FlashAttention's pixel-value tiles took their vector species as a parameter, so the species was constant only while the JIT inlined them; once any prefill made the same kernels hot enough to compile standalone, every broadcast de-intrinsified and the tile allocated instead of using registers.
  One 512x512 image on LFM2.5-VL-3B went from 11 MB and 1.07 s to 162 GB and 5.3 s, on GraalVM after the first prefill and on C2 always.
  The tiles now read the constant species, so an image encode costs 11 MB and 1.06 s on GraalVM and 1.48 s on C2, with no JVM flags.
- **LFM2.5 thinking policy.** A checkpoint whose template never writes a think span (the 350M instruct) reports `NONE`; the ones that do keep `OPTIONAL` or `ALWAYS`.
  LFM2.5-VL-3B ships the same template as the 8B-A1B but does not reason, so it is read off the architecture rather than the template source: `thinking(false)` on the vision models works instead of being refused.

### Known limits

- The `Logic` gallery demo and the model-backed tests pin temperature 0 and a seed; small models still fail some puzzles, which the demo reports honestly.
- When a JSON schema's fields are all optional, LFM2.5-8B-A1B may leave out a field the text does state.
  Mark the fields you rely on as `required`, or check the extracted values; `describeSchema(false)` turns the description line off entirely.
