# Release notes

## 0.4.0-rc1

A release candidate for 0.4.0: Maven Central `0.4.0-rc1`, PyPI `0.4.0rc1` (`pip install --pre`), the native libraries and their archives `0.4.0-rc1`.
The libraries keep the soname of 0.4.0, `libtoknroll.so.0.4.0` and `libjinfer.so.0.4.0`, so a program built against the candidate runs on the release.
The native libraries, the jinfer command and the wheels are built with Oracle GraalVM and carry its runtime under the GraalVM Free Terms and Conditions: free to use, commercially too, but a product sold for a fee that bundles them is not covered; see the [libtoknroll](toknroll/toknroll-native/README.md#license) and [libjinfer](jinfer/jinfer-native/README.md#license) License sections and their NOTICE files.

- **Decisions.** A decision model answers yes/no, choice and rating questions about an input with probabilities, in one forward pass and without generating text: Liquid AI's d1 and d1-Omni (`jinfer-lfm3`) and vLLM Semantic Router's d3 (`jinfer-qwen35`).
  `Models.loadDecisionModel` loads one, `DecisionModel` answers in probabilities, and `jinfer-decision` puts typed questions over it (`Decisions`, `Question.predicate`, `choice` and `rating`).
  The input holds text, JSON fields, images and video, and audio on d1-Omni.
  Several questions are taken the way the model was trained to, with no isolation guaranteed: d1 reads the input once for up to 16 of them, and an answer can shift with the questions beside it.
  `decideIndependently` and `askIndependently` answer each question as if asked alone, as the adapters' `independent(true)` does; the server answers each alone unless vLLM's `"joint": true` asks otherwise.
  `jinfer server` serves `POST /v1/systemone` in llama.cpp's System One shape, and LangChain4j and Spring AI get a `JinferDecisionModel`, with `spring.ai.model.decision=jinfer` in the starter.
  d1 matches Liquid's reference and llama.cpp, and d3 matches llama.cpp.
- **json 0.4.0-rc1.** A Java string holding a lone surrogate prints it as an escape, `\uD800`, so the JSON can be encoded as UTF-8; a surrogate pair prints as itself.
  With `decimalsAsBigDecimal(false)`, a decimal past `double`'s range is refused at its first digit instead of read as `Infinity`.
  Errors count a line at `\r` and `\r\n` as at `\n`, and `unescapeString`'s errors carry line and column too.
  A duplicate key is reported at the key, and an invalid escape at its backslash.
  jinfer-jinja, which depends on json, moves to `0.4.0-rc1` with it.
- **safetensors 0.4.0-rc1.** A file read and written back keeps its own header layout, so its tensor offsets stay valid; writing one used to fail with a header size mismatch.
  A new file's tensor data starts at a multiple of 8 bytes, as upstream pads it.
  A header that is not UTF-8, and a shape or offset written as `1.0` or `1e0`, are refused, as upstream refuses them.
- **jinfer-fetch, new.** `com.qxotic:jinfer-fetch` is the HTTP GET jinfer-hub, toknroll-hf and toknroll-gguf download through, on Java 11 with no dependencies: the JDK's `HttpClient`, or `curl` as a process in native images, which then carry no TLS stack.
  It follows `HTTP_PROXY`, `HTTPS_PROXY` and `NO_PROXY`, and a redirect to another host drops the `Authorization` header.
- **PrismML Ternary Bonsai 2.** `Q2_0` and PrismML's `PQ2_0` and `PTQ1_0` load across gguf, jota, jam and jinfer, and the Qwen3.5 port applies Bonsai's `prism.hadamard` rotation, so Ternary Bonsai 2 runs in all three, images included.
  Checkpoints with an MTP head and a Hadamard rotation are refused.
- **Faster quantized matrix products in jam-native.** On AVX-512-VNNI CPUs prefill runs 8 to 13% faster end to end (LFM2.5-2.6B, 512 tokens, 16 threads), and calls of 3 to 7 tokens run twice as fast in the `Q4_0`, `Q5_0` and `MXFP4` kernels.
  On AVX2 and AVX-VNNI CPUs, Q4_K, Q5_K and Q6_K decode about three times as fast: their rows ran on SSE3 kernels before.
  `Q2_0`, `PQ2_0` and `PTQ1_0` decode up to 14 times as fast, and their row kernels no longer leave a buffer allocated on every thread that ran one.
  The 32-block formats' prefill now subtracts its bias exactly, which halves its rounding error, so outputs can differ from 0.3.1 in the last bits.
- **jam-native unpacks its library once.** The bundled `libjam` goes to `jinfer/jam/<sha256>/` in the user's cache directory (`$XDG_CACHE_HOME` or `~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows), and every later run reuses it.
  Each run used to unpack a copy of its own into the temporary directory, which Windows could not delete afterwards and a `/tmp` mounted `noexec` refused to run.
  A bundled library that does not load now prints a warning before the Java backends take over; it used to fall back silently, several times slower at prefill.
- **Qwen3.5 fixes.** RoPE rotates NEOX pairs, as llama.cpp's IMROPE does; the port had rotated adjacent pairs, a mild drift that changed some greedy replies against llama.cpp.
  Images are resized as the reference does: the grid rounds half to even, the image is stretched with bicubic weights instead of letterboxed, and the projector's own `image_min_pixels` and `image_max_pixels` bound it.
- **A GGUF's tensors must fill their slots.** The reader checks every tensor's size, from its shape and type, against the room the file gives it, and refuses a gap or two tensors that cross; a tensor inside another, such as Q and K inside a stacked QK, still loads.
  A type that sizes its tensors wrong, such as the first Ternary Bonsai's `Q2_0` files in a pre-release layout, now fails at load naming the tensors instead of decoding garbage.
- **gguf reads and writes version 3 only.** `Builder.setVersion` refuses any other version, so gguf no longer writes a file its own reader refuses; v2 is found only in files converted in 2023.
  A write the format refuses, such as a tensor name past 63 bytes or an uppercase key, now writes nothing: no bytes reach the channel and no file is created.
  The reader refuses tensor names past 63 bytes too, as llama.cpp does.
- **Model references read their repository in any case.** Hugging Face and ModelScope do, so `LiquidAI/d1-2B-gguf` now finds the cache of `LiquidAI/d1-2B-GGUF`, offline too, instead of downloading the model again.
- **LFM3-Omni answers text requests under its voice prompt.** A text-only reply under "Respond with interleaved text and audio." ended before its first word, reported as a normal stop.
  A text reply now bans every speech-lane token, not only the two lane openers, and answers in text.
- **libtoknroll survives running out of memory.** A batch is encoded a chunk at a time, so its heap use is bounded whatever its length, and a batch larger than the heap now succeeds.
  An out-of-memory or stack overflow the runtime raises where no handler runs fails the call with `TOKNROLL_ERROR` and words naming the knob, where a batch could hang or abort the process before.
  The tokenizer goes on working, since encoding takes no locks; batch workers are the library's own threads, with 8 MB stacks, in libjinfer and on Windows too.
- **A loaded tokenizer no longer stalls a batch's first collections.** libtoknroll, libjinfer and the jinfer command run the serial collector with one survivor space, so a tokenizer's tables (Gemma 4's: 230 MB) move to the old generation at the first young collection, instead of being copied at each one until they age out.
  Those copies were 50 ms pauses that held a batch's threads idle through its first few hundred MB of text.
- **A missing model has its own exception.** `ModelNotFoundException` (`jinfer-hub`), an `IllegalArgumentException`, is thrown wherever a source names nothing: a missing local file, a repository that does not exist or is gated, a file or quant it lacks, or an offline miss, which was an `UncheckedIOException` before.
  `source()` names it; a malformed or ambiguous ref stays a plain `IllegalArgumentException`, and libjinfer and libtoknroll report it as not found by its type.
- **A reranker no longer loads as an embedder.** `Models.loadEmbedder` refuses a checkpoint whose header says reranker, and `Models.loadReranker` one that says embedder, naming the right entry point; Qwen3-Reranker used to load as an embedder, with vectors that could not tell unrelated texts apart.
  A model path that does not exist or is a directory is refused with its absolute path and the reason, and the integrations' "failed to load model" message carries both.
- **Shorter calls.** `Sampling.GREEDY` is the most likely token, no seed, in place of `new Sampling(0f, 1f, 0, 0f, null)`, and `ChatEngine.ReplySink.of(delta -> ...)` takes a reply's text as a lambda.
- **The Python package is typed.** Every public function is annotated, and a reply without a schema is a `Reply[Any]` to a type checker; `Reply.reasoning` is `""`, like `.text`, when the model did not reason, where it was `None`.
  `load` returns the class the file calls for, so a type checker learns it from an annotation: `model: jinfer.LanguageModel = jinfer.load(ref)`.
- **A closed model lets go.** libjinfer frees a model's weights when its last handle goes, where the mapping stayed until some later collection, and a finished stream no longer holds its model: a conversation keeping its last reply no longer kept the prompt catalog from being written at `close()`.
  In Python, closing a reply from another thread waits for the step in progress and keeps the partial reply, a failed reply says why on every read, and a model refuses to be pickled or copied, as its handle means nothing in another process.
- **Smaller fixes in Python.** `schema=float` gives a `float` when the model writes `3`, and the OpenAI adapter streams tool calls.
  `transcribe` reads a float WAV on Python 3.15, whose `wave` module reads them, where its samples were taken as integers; an older Python refuses one with an `InvalidArgument` naming the file.
  Speech text with a NUL character is refused, where espeak stopped reading at it, and a file that is not GGUF says so instead of printing its magic number.
  A dataclass or TypedDict that contains itself works as `schema=`, through `$defs`, where it raised `RecursionError`.
  A conversation turn that fails, before it starts or in any tool round, leaves the history as it was and its reply says why, where a tool call without its results stayed and failed every later turn.
  libjinfer's errors name `context_capacity` as C and Python spell it, and a `prompt_cache` file that is not a cache is refused with "name another file", never advice to delete it.
- **Tool calls are matched one to one.** A request offering two tools of one name is refused, since a call names its tool, and each tool result in a conversation answers exactly one earlier call, so a memory trim that drops a call but keeps its result is refused for families without call ids too.
- **EmbeddingGemma 2.** Google EmbeddingGemma 2 (`jinfer-gemma-embedding2`, architecture `gemma-embedding2`) embeds text into 768 dimensions (Matryoshka down to 128), and images, audio and video through its mmproj with `GemmaEmbedding2.withMedia` and `embed`.
  It matches Google's official pipeline (sentence-transformers over `google/embeddinggemma-2`) for every task prompt on the model card, and for images, audio and video: cosine above 0.99999 with F32 weights.
  `GemmaEmbedding2` frames text for the card's seven tasks: `searchQuery`, `question`, `claim` and `codeQuery` against `document(title, text)`, and `classification`, `clustering` and `similarity`; images, audio and video take no prompt.
  The LangChain4j and Spring AI integrations keep to search, the one task their frameworks name.
  `VideoCodec.every(video, step, maxFrames)` samples a video the way the reference does (one frame a second, at most 32).
- **Media embeddings over HTTP.** `jinfer server` attaches an embedder's projector with `--with media=<mmproj>`, and `/v1/embeddings` then takes inputs that are arrays of content parts (`text`, `image_url`, `input_audio`, `video_url`, as in chat), each embedded as one vector; `/v1/models` lists the modalities served.
  In the library, `Models.loadEmbedder(path, arena, companions, tokenizer)` attaches companions and `LoadedEmbedder.embed(state, parts, dimension, consumer)` embeds one interleaved input (`EmbeddingModel.embed` underneath).
- **Media embeddings in LangChain4j and Spring AI.** Both `JinferEmbeddingModel` builders take `companion(capability, ref)` and `companionPath(capability, path)`, and the starter reads `spring.ai.jinfer.embedding.companions.media`.
  With EmbeddingGemma 2's projector attached, LangChain4j embeds image, audio and video `EmbeddingRequest` inputs (`supportedContentTypes()` says which), and Spring AI embeds media `Document`s.
  As the model card says, media embeds without a prompt; text keeps its query or document prompt.
  Spring AI's vector stores still refuse non-text documents at `add`; media vectors come from the `EmbeddingModel` directly.
- **One embedding width for every call.** Both `JinferEmbeddingModel` builders take `dimensions(int)`, and the starter reads `spring.ai.jinfer.embedding.dimensions`: every call that names no width embeds at it, and `dimension()` (LangChain4j) or `dimensions()` (Spring AI) reports it.
  Vector stores size their tables from the model and embed without options, so this is how a store holds shorter Matryoshka vectors, such as EmbeddingGemma 2 at 256.
  `0`, the default, is the model's own width; a request's own width still wins, and a width the model cannot produce is refused at build.
- **`fork()` is removed.** The `fork()` of the LangChain4j and Spring AI models carried some of the caller's objects into the new pipeline (listeners, observation registries, the video sampler) and not others, so what a fork shared with its origin was hard to predict.
  For parallel pipelines over one copy of the weights, load once with `Models.load`, `Models.loadEmbedder` or `Models.loadReranker` and build each model with `model(loaded)`: builds share the weights and nothing else.
  To start a pipeline from prompts another one prefilled, `saveCachedPrompts(path)` on the first and `promptCache(path)` on the next.
  In the core, a state copies into a fresh one with the model's `CheckpointCodec`, as the jinfer guide shows.
- **Spring AI's deprecated reasoning getters are removed.** `JinferChatOptions.getReasoningBudget()` and `getReasoningBudgetMessage()`, deprecated in 0.3.1, are gone: use `getMaxReasoningTokens()` and `getReasoningCutoffMessage()`, named like the builder and the properties.
- **The chat server no longer reads text aloud.** `POST /v1/audio/speech` is gone from a chat model's server, and `ChatTemplate.speechInstruction` with it: a model with a voice speaks its chat replies, and reading text aloud is a speech model's job, `jinfer speak` and the `JinferSpeechModel` faces, at a fraction of a 2B chat model's cost.
- **Gemma 4 media follows Google's reference, not llama.cpp.** Three places where llama.cpp departs from Google's JAX and Hugging Face implementations now follow the reference, for the Gemma 4 chat models as well as EmbeddingGemma 2.
  Images are resized to fill the token budget (280 by default) with their aspect kept, small images growing and nothing padded, where llama.cpp only shrinks them and pads with black.
  The vision tower's MLP uses tanh GELU, where llama.cpp runs quick-GELU.
  Audio features take `log(mel + 0.001)`, where llama.cpp clamps at 0.001.
  Against the official towers, Gemma 4 E2B's projected image and audio rows now agree to cosine 0.99998 and better.
  Gemma 4 chat encodes video frames at the reference's 70 tokens again, not at a still image's 280, so a 32-frame video fits a prompt about four times smaller.
  Audio handed in at another rate is resampled to 16 kHz with miniaudio's filter (`AudioCodec` still decodes files to 16 kHz itself).
- **The server finishes what it started.** A shutdown lets the reply in flight finish within the shutdown timeout, where it was cut at once; queued callers still get their 503 straight away.
  An upload far past the body limit (`--max-body-mb`) reads its 413, where the connection was reset before the client saw why.
  A reply cut short by the deadline finishes with `"length"`, OpenAI's value for a truncated reply, where `"other"` failed clients that read the field as an enum; the Responses API still says `timeout`.
  `HEAD` works wherever `GET` does, as uptime monitors probe, and the README shows `/v1/systemone`'s answers.
- **The server judges a parameter by its value.** A default that asks for nothing passes (`frequency_penalty: 0`, `logprobs: false`, `logit_bias: {}`), where it was refused and broke the stock LangChain4j OpenAI client.
  A request the server cannot honor is refused instead of answered without it: a `/v1/completions` array of several prompts (they were joined into one), `echo`, `suffix`, `best_of` above 1, the legacy `functions`, `grammar` together with `response_format`, and audio on `/v1/responses`.
  `tool_choice: required` works on a model that always reasons, LFM2.5-8B-A1B, where it was refused.
- **A JSON Schema is enforced whole, or refused.** `Grammar.fromSchema`, structured output in the server and the LangChain4j and Spring AI adapters refuse a schema using a constraint the grammar cannot enforce (a number's bounds, `pattern`, `allOf`, `uniqueItems`, `additionalProperties` other than `false`, …), naming the keyword; the server answers 400.
  These keywords were ignored before, so the model could produce documents the schema forbids.
  Annotations (`title`, `description`, an unknown `format`) and non-standard keywords are still ignored, as the JSON Schema spec asks.
  An integer's `minimum`, `maximum` and their exclusive forms are enforced, so a rating from 0 to 3 is a schema.
- **GBNF's `.` is one UTF-8 character,** as in llama.cpp, where it was one byte: a grammar written for llama.cpp means the same here, and `.{n}` cannot end inside a character.
- **`retainSessions(n)` is `retainedSessions(n)`** on both chat-model builders, the knob's name on every face (the core's `withRetainedSessions`, Boot's `spring.ai.jinfer.chat.retained-sessions`).
- **A first batch no longer allocates once per core.** Its first call built a SentencePiece tokenizer's 24 MB word cache, or a tiktoken tokenizer's fast engine, on every core at once and kept one; one thread builds it now, so a first batch fits the heap a single one does.
- **Cache management without the network: `list`, `find` and `remove`.**
  `jinfer list` names each cached model by the shortest reference that loads it (`LiquidAI/LFM2.5-350M-GGUF:Q8_0`), alphabetically, so a line pastes into `-m`, `pull`, `find` or `remove`.
  `jinfer find <ref>...` prints the file `-m` would load, without a request or a hash: `llama-server -m "$(jinfer find <ref>)"`.
  A model pulled at a branch or tag lists by it (`owner/repo@dev:Q8_0`), and is counted in the total.
  `jinfer remove <ref>...` deletes the one file `find` prints for each reference, a browser URL or a path `list` prints included, and names it with its size; a repository holding several cached files is refused with their names, never guessed.
  A model in the shared Hugging Face cache takes its blob along unless another snapshot links it, and a repository folder goes once it is empty.
  `ModelStore.evict(ref)` is `ModelStore.remove(ref)`, the command's name.
- **`jinfer prune` frees the space older versions take.** A pull that updates a model keeps the version it replaced in the Hugging Face cache; `list` now ends with how much that is, and `prune` deletes them, as `hf cache prune` does. `ModelStore.detached()` lists them with what removing each frees.
- **Errors say what went wrong, once.** A server that cannot be reached reads `cannot connect to <host>` or `cannot find host <host>`, where it read `ConnectException: ClosedChannelException`; an empty `:` or `@` quotes the reference; a bare word given as a model (`-m LiquidAI`) is shown the reference shape; and a cause is no longer repeated after a message that already says it.
- **Native images download behind an HTTPS proxy.** curl printed the proxy tunnel's own `200 Connection established` block ahead of the response, so headers read as body; it now skips it.
  A redirect to another scheme or port drops the `Authorization` header, as one to another host did.
- **Windows without symlinks shares the Hugging Face cache by hard links.** Outside Developer Mode a snapshot entry was a full copy of its blob: twice the disk, a full copy again on every up-to-date `pull` (refused while a server held the model), and a blob `remove` could not find, left behind for good. It is a hard link now, kept as it is when current, and `remove` frees its blob.
- **A throttled ModelScope reads as busy.** ModelScope answers a burst of requests with 403, which read as "does not exist, or is gated or private"; it reads `modelscope.cn is limiting requests from this machine: try again in a minute` now.
- **`ModelStore.cached()` lists GGUF files only,** as `jinfer list` showed them: the store's own bookkeeping (a remembered quant) no longer appears as a cached model.
- **`jinfer transcribe --raw-pcm` no longer warms the JIT before listening;** on the JVM the first second of audio may decode a little later.
- **`jinfer pull` of a cached model no longer reads it whole.** A blob in the shared Hugging Face cache is named by its verified SHA-256, so name and size say it is current: 0.75 s instead of 17 s for a 25 GB model.
  A fresh download is checked once, not twice. Bytes damaged after download are replaced by `pull --force`.

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
- **Speculation is opt-in.** A model with a draft head no longer drafts by default, since prose decodes at about half speed with it; `--speculation-depth 4` (`speculationDepth(4)` in Java) turns it on.

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
- **LangChain4j 1.23.0.** `jinfer-langchain4j` builds on it and passes its chat, streaming, AI-service, tools and JSON-schema TCKs.
- **Spring AI 2.1.** `jinfer-spring-ai` builds on 2.1 and its message parts: a reply's reasoning is a `ReasoningPart` (`AssistantMessage.getReasoning()`), before the text and the tool calls, and streams as its own part per delta, so `ChatClient` and `MessageAggregator` keep it off the content lane.
  The `thinking` metadata entry and the `isThought` chunk flag are gone.
  Prompts are read by parts too: a user message's text and media reach the model in the order written, and tool results are read from their parts (text; a media result is refused in words).
  `JinferEmbeddingModel.getEmbeddingContent` returns the framed document text, so Spring's default batched path embeds what the provider's own does.
  The starter configures 2.1's session memory (`SessionService`, `SessionAdvisors`) in place of the chat memory it deprecates for 3.0; it requires Spring Boot 4.2.
- **Spoken replies in Spring AI.** `JinferChatOptions.outputModalities` says what a reply may be made of: unset is text alone, `TEXT` and `AUDIO` let LFM3-Omni choose speech when the prompt asks, `AUDIO` alone makes it speak.
  The speech is a `MediaPart` of `audio/wav` beside the transcript, whole from `call` and a part per stretch from `stream`; `spring.ai.jinfer.chat.output-modalities` sets the default in Spring Boot.
- **Spoken replies in LangChain4j.** `JinferChatRequestParameters.outputModalities` says what a reply may be made of, the engine's own `Modality` enum: unset, text alone; text and audio, the model's choice; audio alone, a spoken reply.
  The speech is the `AiMessage`'s `audios()`, one `audio/wav` clip beside the transcript, whole from `chat`; the streaming model hands each stretch to `onUnmappedRawEvent` as the engine's `Media.Audio`.
  An unmodified reply echoed in history replays its speech exactly, and an `AudioContent` built from binary data is now decoded like one built from base64 or a file.

### Behaviour changes

Coming from 0.3.0, these act differently:

- **GPT-2 style tokenizers in `toknroll-hf`.** A `ByteLevel` pre-tokenizer applies the GPT-2 split unless it says `use_regex: false`, as Hugging Face tokenizers does, so GPT-2, DistilGPT-2, Phi-2 and RoBERTa encode differently: now identical to `r50k_base` where they share its vocabulary.
- **A reply remembers how it was said.** `Message` carries a `Verbatim` record: the turn's token ids from the prompt's reply prefix to its last token, a speech frame as `Verbatim.FRAME` with its codes aside, stamped with the model that generated it; `Verbatim.NONE` on a turn nobody generated.
  A native template writes the record in place of the turn (`PromptWriter.replay`), reasoning and all, so the model reads back exactly what it wrote and the prompt cache recognizes the stream it holds; a record from another model renders from its parts.
  The parts are plain data: `Content.Text`, `Content.Reasoning` and `Content.ToolCall` lose their `verbatim` ids, `Content.Media` holds an image, audio or video the caller gives the model and, in a reply, the model's own speech, captioned with the words it spoke; `Content.Speech` and `SpokenTurn` are gone.
  `ChatEngine.Completion.reasoningTokens` is a component, counted as the reply streams, and a thinking span the prompt seeded empty no longer appears as an empty `Content.Reasoning` part: prompt-owned tokens are the record's, not the reply's.
  A reply's parts are its stream folded, in emission order: `Content.Reasoning` is a run of reasoning text, and a call the model claimed while thinking is a `Content.ToolCall` between the runs it interrupted, no longer a part nested inside the reasoning; `Message.reasoning()` joins the runs as `text()` joins the text.
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
