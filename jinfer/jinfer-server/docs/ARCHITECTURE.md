# jinfer-server — class architecture

Package `com.qxotic.jinfer.server` (25 top-level classes, ~5 790 lines, one package). It is jinfer's
**HTTP transport**: an OpenAI-compatible (and llama.cpp-flavoured) server built on the JDK's
`com.sun.net.httpserver`, with no web framework. Four independent front doors share one plumbing layer:

* `Server` — the language server over a caller-owned `ChatEngine`: `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses`, `/v1/models`, `/tokenize`, `/detokenize`, `/health`, `/props`, `/metrics`;
* `RetrievalServer` — `/v1/embeddings` (OpenAI shape) or `/v1/rerank` (llama.cpp shape);
* `DecisionServer` — `/v1/systemone` (llama.cpp's System One shape) over a `DecisionModel`;
* `TranscriptionServer` — `/v1/audio/transcriptions` (OpenAI multipart) over a `TranscriptionModel`.

Public API: `Server`, `ServerConfig`, `OpenAiChat`, `RetrievalServer`, `DecisionServer`,
`TranscriptionServer` (and their nested `Running` handles). Everything else is package-private.
There is **no `META-INF/services` registration**: the only resource is
`META-INF/native-image/com.qxotic/jinfer-server/native-image.properties` (`--add-modules jdk.httpserver`).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
`*--` owns (composition); classes marked `<<external>>` live outside this module. Nested classes are
written `Outer_Nested` in class diagrams (Mermaid class names cannot contain dots).

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a public API, 2b language server, 2c task servers, 2d wire helpers
3. [Server start and shutdown](#3-server-start-and-shutdown)
4. [Request pipeline of a generation endpoint](#4-request-pipeline-of-a-generation-endpoint)
5. [Lowering and the reply path](#5-lowering-and-the-reply-path)
6. [Streaming (SSE)](#6-streaming-sse)
7. [Admission, queueing and concurrency](#7-admission-queueing-and-concurrency)
8. [Task servers: retrieval, decision, transcription](#8-task-servers)
9. [Errors, validation and observability](#9-errors-validation-and-observability)

| Class | Role |
|---|---|
| `Server` | Language transport over a `ChatEngine`: routes, the admission gate (`gated`), the shared generation scaffold (`handleGenerationPost`), streaming handlers; nested `Running` (public handle), `SpokenStream` (spoken SSE), `RequestJob` |
| `ServerConfig` | Public immutable record `(bind, Defaults, Limits, Access)`; `local(port)` factory and `with*` copies |
| `Generation` | Lowers OpenAI wire values onto `ChatEngine` (`chat`, `completion`), keeps the `SpokenTurns`, records metrics; nested `WireSink implements ChatEngine.ReplySink` |
| `OpenAiChat` | Public: the chat.completions dialect — `lower` (request → `ChatEngine.Request`), `reply` (completion → `Reply`), `completion` (JSON envelope). Also used in-process by libjinfer |
| `Worker` | One platform thread (`generation-worker`) + a bounded FIFO; `submitAndWait` returns `Result {COMPLETED, FULL, CLOSED, INTERRUPTED}` |
| `Http` | Transport plumbing: CORS/OPTIONS/bearer preamble, `AccessLog` filter, bounded body reads, JSON/text/bytes responses, OpenAI error envelope |
| `Sse` | Server-Sent Events: `Stream` (frame encode/flush, checked→unchecked bridge), `guarded` error framing, the stalled-write reaper thread |
| `Metrics` | Per-instance Prometheus counters + text exposition; `Outcome` enum |
| `Reply` | Record: protocol-independent result (generator result, token counts, text, reasoning, tool calls, finish reason, speculation, `Audio`) |
| `Sinks` | Record of live channels `(onText, onReasoning, onSpeech)`; `NONE` for non-streaming |
| `OpenAiSchema` | Pure JSON builders: chat/completion/responses envelopes and chunks, `usage`, llama.cpp `timings` |
| `Validation` | Request validation run on the handler thread; `UnknownModel` (→ 404), `require`/`requireParam` |
| `Values` | Typed coercion of parsed JSON maps; `InvalidParam` (names the field in `param`) |
| `Requests` | Request-shape adapters: model id echo, token budget, completion prompt, Responses-API → chat folding |
| `JsonCodec` | The single bridge to the external `com.qxotic.format.json.Json` library (null mapping, ordered objects) |
| `ToolUse` / `ToolCalls` | Tool-choice policy (`offered`, `forced`) / engine `Content.ToolCall` → OpenAI wire (+ streaming deltas) |
| `SpokenTurns` | Bounded (4096), expiring (1 h) LRU of spoken assistant turns referred to by `audio.id` |
| `MediaParts` | OpenAI media content parts (`image_url`, `input_audio`, `video_url`) → `Content.Media`; advertised modalities |
| `MediaSpill` | Streams a base64 video data URI to a temp file while hashing it (20 MB cap) |
| `TaskTransport` | Shared transport for the single-task servers: probes, gated `work` routes, 404 catch-all; nested `Running` |
| `RetrievalServer` | Embedding or rerank server; nested `Embedder`/`Ranker` seams adapted from `LoadedEmbedder`/`LoadedReranker` |
| `DecisionServer` | System One decision server over a `DecisionModel` and one `ContextState` |
| `TranscriptionServer` | Speech-to-text server; nested minimal RFC 7578 `Multipart` parser |
| `Log` | The `System.Logger` named `jinfer.server` |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph callers["Callers"]
        CLI["jinfer-cli: cli.Server<br/>chooses the server by model"]
        LIB["jinfer-native: libjinfer Stream/Model<br/>uses OpenAiChat + ServerConfig in-process"]
    end
    subgraph srv["jinfer-server"]
        S["Server"]
        RS["RetrievalServer"]
        DS["DecisionServer"]
        TS["TranscriptionServer"]
        OC["OpenAiChat"]
        PL["Http · Sse · Worker · Metrics · TaskTransport"]
    end
    subgraph chat["jinfer-chat"]
        CE["ChatEngine"]
        LM["LoadedModel / LoadedEmbedder / LoadedReranker"]
        DM["DecisionModel"]
    end
    subgraph core["jinfer-core"]
        TM["TranscriptionModel"]
        CS["ContextState"]
        TEL["telemetry: InferenceEvent, Telemetry"]
        MED["media: Media, Multimodal"]
    end
    CODEC["jinfer-codecs: AudioCodec, ImageCodec, VideoCodec"]
    CACHE["jinfer-cache: PromptCache"]
    JSON["json: com.qxotic.format.json.Json"]
    TOK["toknroll-core: IntSequence"]
    JDK["JDK: com.sun.net.httpserver"]

    CLI --> S & RS & DS & TS
    LIB --> OC
    S --> CE
    S --> OC
    RS --> LM
    DS --> DM
    TS --> TM
    S & RS & DS & TS --> PL
    PL --> JDK
    PL --> JSON
    CE --> LM
    S --> CACHE
    S --> TEL
    OC --> CODEC
    RS & DS --> CS
    S --> TOK
    S --> MED
```

Maven dependencies (compile): `jinfer-core`, `jinfer-chat`, `jinfer-cache`, `jinfer-codecs`, `json`,
`toknroll-core`. Test scope adds `jinfer-testkit`, `jam-native` and several model providers
(`jinfer-lfm2`, `jinfer-lfm3`, `jinfer-qwen3`, `jinfer-qwen35`, `jinfer-gemma-embedding2`) that the
integration tests load through `ModelProvider`'s `ServiceLoader`. The server itself never loads a
model: every `start` takes an already-loaded, caller-owned model or engine.

---

## 2. Class diagrams

### 2a. Public API and handles

```mermaid
classDiagram
    direction LR
    class AutoCloseable { <<interface, external>> }
    class ChatEngine { <<external>> }
    class Server {
        <<final>>
        +start(ChatEngine, ServerConfig)$ Server_Running
    }
    class Server_Running {
        <<final>>
        +address() InetSocketAddress
        +await()
        +close()
    }
    class RetrievalServer {
        <<final>>
        +start(LoadedEmbedder, int, String, ServerConfig)$ Running
        +start(LoadedReranker, int, String, ServerConfig)$ Running
    }
    class DecisionServer {
        <<final>>
        +start(DecisionModel, int, String, ServerConfig)$ Running
    }
    class TranscriptionServer {
        <<final>>
        +start(TranscriptionModel, String, ServerConfig)$ Running
    }
    class ServerConfig {
        <<record>>
        bind InetSocketAddress
        defaults Defaults
        limits Limits
        access Access
        +local(port)$ ServerConfig
        +withDefaults() withLimits() withAccess()
    }
    class ServerConfig_Defaults {
        <<record>>
        sampling, maxOutputTokens, think
        rawPrompt, maxReasoningTokens
        reasoningCutoffMessage
    }
    class ServerConfig_Limits {
        <<record>>
        threads, maxBodyBytes, grammar
        writeTimeout, requestTimeout, shutdownTimeout
        retryAfterSeconds() int
    }
    class ServerConfig_Access {
        <<record>>
        bearerToken
        allowedOrigins Set
    }
    class OpenAiChat {
        <<final>>
        +lower(request, servedModel, config, sampling)$ Lowered
        +modelId(request, servedModel)$ String
        +newId()$ String
        +completion(id, modelId, Completion, Lowered)$ String
    }
    class OpenAiChat_Lowered {
        <<record>>
        request ChatEngine_Request
        stops List
        inlineReasoning boolean
    }

    AutoCloseable <|.. Server_Running
    Server ..> Server_Running : serve
    Server --> ChatEngine : caller-owned
    ServerConfig *-- ServerConfig_Defaults
    ServerConfig *-- ServerConfig_Limits
    ServerConfig *-- ServerConfig_Access
    Server --> ServerConfig
    RetrievalServer --> ServerConfig
    DecisionServer --> ServerConfig
    TranscriptionServer --> ServerConfig
    OpenAiChat ..> OpenAiChat_Lowered : lower
```

Every server's `start` validates its arguments, binds, starts and returns at once. It **prints nothing,
reads no system properties, installs no shutdown hook**; the CLI owns those concerns. `RetrievalServer`,
`DecisionServer` and `TranscriptionServer` each nest their own public `Running implements AutoCloseable`
that delegates to a `TaskTransport.Running`.

`ServerConfig.Limits.DEFAULTS` = 16 threads, 32 MiB body, grammar on, 30 s write timeout, 300 s request
timeout, 30 s shutdown timeout; `retryAfterSeconds()` = `2 × threads`. `Access.LOCAL` = no bearer token,
origins `{"*"}`.

### 2b. The language server — `Server` and its collaborators

```mermaid
classDiagram
    direction TB
    class Server {
        <<final>>
        -Worker worker
        -ServerConfig config
        -Semaphore admissions
        -Metrics metrics
        -Generation generation
        -String servedModel
        -serve(engine, config) Server_Running
        ~gated(handler, admissions, limits)$ HttpHandler
        ~refuse(exchange, e)$
        -handleGenerationPost(exchange, path, idPrefix, validator, job)
        -handleChatCompletion() handleCompletion() handleResponse()
        -streamChatCompletion() streamCompletion() streamResponse()
        ~emitResponseItems(sse, id, mayCallTools, reply)$ List
        ~streamEndChunks(...)$ List
        -runQueued(exchange, work)
        -handleMetrics(exchange)
    }
    class Server_Running {
        -HttpServer http
        -Worker worker
        -Thread reaper
        +close()
    }
    class Server_SpokenStream {
        -StringBuilder held
        -boolean speaking
        text(String)
        speech(Media_Audio)
        finish()
    }
    class Generation {
        <<final>>
        -ChatEngine engine
        -Sampling defaults
        -SpokenTurns turns
        chat(request, messages, sinks, audioId) Reply
        completion(request, prompt, sinks) Reply
        forcedTool(request) String
        cacheSample() mediaCacheSample() blockCaching()
    }
    class Generation_WireSink {
        -Sinks sinks
        -boolean inline
        on(Delta)
        onMedia(Media)
        finish()
    }
    class ReplySink { <<interface, external>> }
    class ChatEngine { <<external>> }
    class Worker {
        <<final>>
        -BlockingQueue queue
        submitAndWait(Runnable) Result
        start() stopAdmission() close()
        queued() int
        busy() boolean
    }
    class Metrics {
        <<final>>
        record(Reply)
        record(Outcome)
        recordPromptCache(tier, restored)
        exposition(worker, cache, mediaCache) String
    }
    class SpokenTurns {
        <<final>>
        put(id, Message) long
        get(id) Message
        newId()$ String
    }
    class Reply { <<record>> }
    class Sinks { <<record>> }
    class OpenAiChat { <<final>> }

    Server *-- Worker
    Server *-- Metrics
    Server *-- Generation
    Server ..> Server_Running : creates
    Server ..> Server_SpokenStream : spoken stream
    Server_Running --> Worker
    Generation --> ChatEngine
    Generation --> Metrics
    Generation *-- SpokenTurns
    Generation ..> Generation_WireSink : per pass
    ReplySink <|.. Generation_WireSink
    Generation_WireSink --> Sinks
    Generation ..> OpenAiChat : lower, reply
    Generation ..> Reply : returns
    OpenAiChat ..> SpokenTurns : replays audio ids
```

### 2c. Task servers and their shared transport

```mermaid
classDiagram
    direction LR
    class TaskTransport {
        <<final>>
        -HttpServer http
        -Semaphore admissions
        probes(servedModel, modalities, props, metrics, queued)
        work(path, HttpHandler)
        start(serves, threadName, onClose) TaskTransport_Running
        inFlight() int
        uptimeSeconds() double
    }
    class TaskTransport_Running {
        address()
        await()
        close()
    }
    class RetrievalServer {
        <<final>>
        -ReentrantLock compute
        -embeddings(exchange, Embedder)
        -rerank(exchange, Ranker)
        -adapt(LoadedEmbedder, int)$ Embedder
        -adapt(LoadedReranker, int)$ Ranker
    }
    class RetrievalServer_Embedder {
        <<interface>>
        dimension() minimumDimension()
        contextCapacity()
        embedAll(texts, dim, first, sink) int
        embed(parts, dim, sink) int
        modalities() List
        close()
    }
    class RetrievalServer_Ranker {
        <<interface>>
        contextCapacity()
        score(query, docs, sink) int
        close()
    }
    class DecisionServer {
        <<final>>
        -DecisionModel model
        -ContextState state
        -ReentrantLock compute
        -handle(exchange)
    }
    class TranscriptionServer {
        <<final>>
        -TranscriptionModel model
        -ReentrantLock compute
        -handleTranscription(exchange)
    }
    class TranscriptionServer_Multipart {
        boundary(contentType)$ String
        parse(body, boundary)$ Map
    }
    class LoadedEmbedder~S~ { <<external>> }
    class LoadedReranker~S~ { <<external>> }
    class DecisionModel { <<interface, external>> }
    class TranscriptionModel { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class Server { <<final>> }

    RetrievalServer *-- TaskTransport
    DecisionServer *-- TaskTransport
    TranscriptionServer *-- TaskTransport
    TaskTransport ..> TaskTransport_Running : start
    TaskTransport ..> Server : gated, refuse
    RetrievalServer ..> RetrievalServer_Embedder : adapt
    RetrievalServer ..> RetrievalServer_Ranker : adapt
    RetrievalServer_Embedder ..> LoadedEmbedder : anonymous adapter
    RetrievalServer_Ranker ..> LoadedReranker : anonymous adapter
    DecisionServer --> DecisionModel
    DecisionServer *-- ContextState : newState
    TranscriptionServer --> TranscriptionModel
    TranscriptionServer ..> TranscriptionServer_Multipart
```

`Embedder` and `Ranker` are package-private seams ("a test fakes it"); the public `start` overloads
adapt a `LoadedEmbedder<S>` / `LoadedReranker<S>` with one `ContextState` of `contextCapacity`
tokens, created at start and closed with the server.

### 2d. Wire helpers (all package-private, stateless)

```mermaid
classDiagram
    direction LR
    class Http {
        preamble(exchange, access)$ boolean
        logged(HttpContext)$
        requireMethod(exchange, method)$ boolean
        readBody(exchange, max, timeout)$ byte[]
        readJsonObject(exchange, limits)$ Map
        sendJson() sendText() sendBytes()$
        sendError() sendErrorQuietly()$
        errorPayload(status, msg, param)$ Map
    }
    class Http_AccessLog
    class Filter { <<abstract, external>> }
    class Sse {
        begin(exchange, writeTimeout)$ Sse_Stream
        guarded() guardedResponses()$
        startReaper()$ Thread
    }
    class Sse_Stream {
        emit(Object)
        emit(event, Object)
        done()
        close()
    }
    class OpenAiSchema {
        usage() timings()$
        chatCompletionResponse() chatCompletionChunk()$
        completionResponse() completionChunk()$
        responseResponse() responseOutputItems()$
    }
    class Validation {
        validateChatRequest() validateChatOptions()$
        validateCompletionOptions() validateResponseOptions()$
        validateGenerationParams() validateModel()$
        audioFormat(request)$ String
    }
    class Validation_UnknownModel
    class Values {
        asObject() asArray() stringValue()$
        intValue() longValue() floatValue() booleanValue()$
        param(Throwable)$ String
    }
    class Values_InvalidParam { param String }
    class IllegalArgumentException { <<external>> }
    class Requests {
        modelId() budget() completionPrompt()$
        normalizeResponse() responseInputMessages()$
    }
    class JsonCodec {
        parse() stringify() object()$
    }
    class Json { <<external>> }
    class MediaParts {
        modality() text() modalities()$
        image() audio() video()$
    }
    class MediaSpill {
        base64Video(value, field)$ Spilled
    }
    class ToolUse { offered() forced()$ }
    class ToolCalls { toWire() toolCallDeltas()$ }

    Filter <|-- Http_AccessLog
    Http ..> Http_AccessLog : logged
    Sse ..> Sse_Stream : begin
    IllegalArgumentException <|-- Validation_UnknownModel
    IllegalArgumentException <|-- Values_InvalidParam
    JsonCodec --> Json
    Http ..> JsonCodec
    Sse_Stream ..> JsonCodec
    OpenAiSchema ..> ToolCalls
    MediaParts ..> MediaSpill : video
    Validation ..> Values
    Validation ..> Requests
```

---

## 3. Server start and shutdown

```mermaid
sequenceDiagram
    participant C as Caller "cli.Server"
    participant S as Server
    participant G as Generation
    participant W as Worker
    participant H as HttpServer "JDK"
    participant R as Sse reaper
    C->>S: start(engine, config)
    S->>G: new Generation(engine, config, metrics)
    Note over G: defaults = config sampling,<br/>else engine.loaded().samplingDefaults()
    S->>W: new Worker(threads - 1)
    S->>H: HttpServer.create(bind, 0)
    S->>H: createContext x N, each with AccessLog filter
    Note over S,H: work routes wrapped by gated(...),<br/>probes /health /props /metrics /v1/models ungated
    S->>W: start() "generation-worker"
    S->>R: Sse.startReaper()
    S->>H: setExecutor(newCachedThreadPool), start()
    S-->>C: Running(http, worker, reaper, stopDelay)
    C->>S: Running.close()
    S->>W: stopAdmission() — queued callers get 503 now
    S->>H: stop(shutdownTimeout seconds)
    S->>W: close() — interrupts a job still running
    S->>R: interrupt()
    S->>H: executor.shutdownNow()
```

Route table built by `Server.serve`:

| Path | Kind | Method | Handler |
|---|---|---|---|
| `/v1/chat/completions` | gated work | POST | `handleChatCompletion` |
| `/v1/completions` | gated work | POST | `handleCompletion` |
| `/v1/responses` | gated work | POST | `handleResponse` |
| `/tokenize`, `/v1/tokenize` | gated JSON | POST | tokenizer `encode` |
| `/detokenize`, `/v1/detokenize` | gated JSON | POST | tokenizer `decode` (ids range-checked) |
| `/v1/models`, `/v1/models/{id}` | probe | GET | model card (input/output modalities) |
| `/health` | probe, **no bearer key** | GET | `status`, `busy`, `queued` |
| `/props` | probe | GET | n_ctx, sampling, speculation, prompt/media cache stats |
| `/metrics` | probe | GET | `Metrics.exposition` (Prometheus text) |
| `/` | catch-all | any | JSON 404 |

The model card's `input_modalities` comes from `MediaParts.modalities(model, false)` (a `Multimodal`
`supports(...)` check); `output_modalities` adds `audio` when the model is a `SpeechOutput` that `speaks()`.

---

## 4. Request pipeline of a generation endpoint

All three generation routes go through `Server.handleGenerationPost`; parsing and validation happen on
the **handler thread**, only the job reaches the single generation worker.

```mermaid
flowchart TB
    A["HTTP exchange"] --> F["AccessLog filter"]
    F --> GT{"gated: admissions.tryAcquire"}
    GT -->|no permit| E503["503 + Retry-After"]
    GT -->|admitted| P["Http.preamble: CORS, OPTIONS 204, bearer 401"]
    P --> X{"exact path?"}
    X -->|no| E404["404 Not found"]
    X -->|yes| M{"POST?"}
    M -->|no| E405["405 + Allow"]
    M -->|yes| B["Http.readBody: maxBodyBytes, writeTimeout"]
    B -->|too big| E413["413"]
    B --> J["JsonCodec.parse + Values.asObject"]
    J -->|bad JSON| E400a["400"]
    J --> V["endpoint validator: Validation.*, Requests.normalizeResponse"]
    V -->|"IllegalArgumentException / UnsupportedOperationException"| RF["Server.refuse: 400, or 404 for UnknownModel"]
    V --> Q["runQueued: Worker.submitAndWait"]
    Q -->|"FULL / CLOSED / INTERRUPTED"| E503b["503 + InferenceEvent rejected"]
    Q -->|"on worker thread"| JOB["RequestJob: stream? SSE : JSON"]
    JOB --> GEN["Generation.chat / completion"]
    GEN --> OUT["OpenAiSchema envelope → Http.sendJson<br/>or SSE chunks"]
    JOB -->|IOException| DC["Outcome.CLIENT_DISCONNECTED"]
    JOB -->|other Throwable| E500["500 Internal server error, logged"]
```

| Endpoint | id prefix | Validator | Job |
|---|---|---|---|
| chat | `chatcmpl-` | `validateChatRequest`, `validateChatOptions`, `validateGenerationParams` | `generation.chat(request, messages, sinks)` |
| completions | `cmpl-` | `validateCompletionOptions`, `validateGenerationParams`, non-blank prompt | `generation.completion(request, prompt, sinks)` |
| responses | `resp_` | `validateResponseOptions`, `Requests.normalizeResponse`, `validateGenerationParams`, `validateChatRequest(request, folded)` | `generation.chat` over `Requests.responseInputMessages` |

---

## 5. Lowering and the reply path

```mermaid
sequenceDiagram
    participant W as worker thread
    participant G as Generation
    participant O as OpenAiChat
    participant E as ChatEngine "external"
    participant K as WireSink
    participant T as SpokenTurns
    participant M as Metrics
    W->>G: chat(request, messages, sinks, audioId)
    G->>O: lower(request, messages, config, defaults, turns)
    O->>O: tools, messages "MediaParts for image/audio/video", schema→system prompt
    O->>O: sampling, maxTokens, thinking, grammar, forced tool, stops, templateKwargs
    O-->>G: Lowered(ChatEngine.Request, stops, inlineReasoning)
    G->>E: prepare(request) → Prepared
    G->>E: complete(prepared, WireSink)
    loop each delta
        E->>K: on(Delta) / onMedia(Media)
        K->>K: REASONING → onReasoning, or inline "think" tags in onText
        K->>K: CONTENT → onText, Media.Audio → onSpeech
    end
    E-->>G: Completion
    G->>O: reply(completion, stops, inline) → Reply
    G->>M: recordPromptCache(tier, restoredTokens)
    opt reply has speech and was not cancelled
        G->>T: put(audioId, reply) → expiresAt
        G->>G: Reply.withAudio — base64 WAV/PCM16 unless already streamed
    end
    G->>M: record(Reply)
    G-->>W: Reply
```

`Generation.completion` skips the chat template: it tokenizes the prompt (`SpecialTokens.encode` when
`Defaults.rawPrompt`, else `tokenizer.encode`) and calls `ChatEngine.prepareRaw(ids, sampling, maxTokens,
requestTimeout, grammar, stops)`.

Key lowering rules in `OpenAiChat`:

* `thinking` = `chat_template_kwargs.enable_thinking`, else `reasoning_effort != "none"`, else
  `Defaults.think`; forced off when audio output is requested.
* `grammar` = request `grammar`, else `Grammar.jsonGbnf()` for `json_object`, else `Grammar.schemaGbnf(schema)`
  for `json_schema` (and `describeSchema` adds the schema to the system message).
* `forced` maps `ToolUse.forced` to `ChatEngine.ForcedTool.NONE / ANY / Named`.
* An assistant message carrying `audio.id` is replayed from `SpokenTurns.get(id)` (refused if unknown/expired).
* `reply()` collects `Content.Text / Reasoning / ToolCall`, applies `TextStops`, and maps
  `finish_reason` to `tool_calls` / `stop` / `length` (`cancelled` for a cancelled pass).

`OpenAiChat.lower(Map, String, ServerConfig, Sampling)` and `OpenAiChat.completion(...)` are public so
`jinfer-native`'s `libjinfer.Stream` can speak the same dialect in-process (with `turns == null`, so
spoken replay is refused there).

---

## 6. Streaming (SSE)

```mermaid
sequenceDiagram
    participant J as job "worker thread"
    participant S as Sse.Stream
    participant G as Generation
    participant C as client
    J->>S: Sse.begin(exchange, writeTimeout) — 200 text/event-stream
    J->>S: guarded(sse, metrics, body)
    S->>C: chunk delta.role = assistant
    J->>G: chat(request, messages, Sinks)
    loop deltas
        G-->>S: deltaSink → emit(chunk)
        S->>C: data: {...}
    end
    opt tool calls
        S->>C: one chunk with tool_calls deltas
    end
    J->>S: streamEndChunks → final chunk "+ usage-only chunk if include_usage"
    S->>C: data: [DONE]
```

* **Chat** chooses its `Sinks`: a `SpokenStream` (audio requested: text is held until the first speech,
  then sent as `delta.audio.transcript` + base64 PCM16 `delta.audio.data`; never-spoken text goes out
  as `content` at `finish()`), `Sinks.NONE` when `generation.forcedTool(request) != null`, otherwise
  content + `reasoning_content` delta sinks.
* **Completions** stream text chunks via `Sinks.text`.
* **Responses** emit named events (`response.created`, `response.output_item.added`,
  `response.content_part.added`, `response.output_text.delta`, …, `response.completed|incomplete`);
  `Stream.emit(event, map)` adds a `sequence_number`. When tools are offered nothing streams live and
  `emitResponseItems` replays the finished items; the final envelope reuses the same item list so
  tool-call ids agree.
* Timings ride the last chunk (`OpenAiSchema.timings`, llama.cpp's shape).
* `Sse.guarded` turns an `UncheckedIOException` from a frame write back into `IOException` (client
  gone), and other failures into an in-band error event + `[DONE]` (400 for validator types, 500 otherwise).
* The **reaper** thread (`sse-write-reaper`, one per `Server.Running`) scans `ACTIVE` streams each
  second and closes any whose in-flight write exceeded its own `writeTimeout` — the JDK server has no
  write timeout, and a dead client would otherwise wedge the single worker.

---

## 7. Admission, queueing and concurrency

```mermaid
flowchart LR
    subgraph handlers["cached thread pool: one thread per exchange"]
        H1["handler"] --- H2["handler"] --- H3["..."]
    end
    GATE["Semaphore admissions<br/>permits = Limits.threads"]
    Q["Worker queue<br/>capacity threads - 1<br/>SynchronousQueue if 0"]
    WT["generation-worker<br/>one platform daemon thread"]
    ENG["ChatEngine"]
    handlers --> GATE
    GATE -->|"tryAcquire fails"| R503["503 + Retry-After = 2 x threads"]
    GATE --> Q --> WT --> ENG
```

* `Server.requestExecutor()` is a cached pool so the gate — not the JDK — refuses excess load with a
  proper 503 instead of a dropped connection.
* `Worker.submitAndWait` blocks the handler on a `CountDownLatch`. If the handler is interrupted while
  its job is still queued, the job is removed (`INTERRUPTED`); if the worker already owns it, the
  handler waits for it to finish so only one writer touches the exchange.
* `Worker.Job.run` reports `Telemetry.queueWait(...)` before running.
* Probes are never gated, so `/health`, `/props`, `/metrics` answer even when saturated.
* Task servers use the same `Server.gated` gate via `TaskTransport.work`, but no `Worker`: model work is
  serialized by a fair `ReentrantLock compute`, whose `getQueueLength` feeds `/health`'s `queued`. The
  close hook takes the same lock so the `ContextState` closes after the request using it.

---

## 8. Task servers

```mermaid
flowchart TB
    subgraph tt["TaskTransport"]
        PR["probes: /health open, /props, /metrics, /v1/models"]
        WK["work(path): gated + preamble + exact path + POST<br/>IAE/UOE → Server.refuse, else 500"]
        CA["start: '/' JSON 404 naming what it serves"]
    end
    RS["RetrievalServer"] -->|"/v1/embeddings or /v1/rerank"| WK
    DS["DecisionServer"] -->|"/v1/systemone"| WK
    TS["TranscriptionServer"] -->|"/v1/audio/transcriptions"| WK
    RS & DS & TS --> PR
    RS --> EMB["Embedder → LoadedEmbedder.embedAll / embed"]
    RS --> RNK["Ranker → LoadedReranker.scoreAll"]
    DS --> DEC["DecisionModel.decide or decideIndependently"]
    TS --> MP["Multipart.parse → AudioCodec.decode → TranscriptionModel.transcribe"]
```

* **Embeddings**: `input` is a string or array (≤ 2048) of strings / content-part arrays (no token ids);
  runs of plain strings batch through `embedAll`, media inputs go one by one through `embed`;
  `dimensions` may go down to `minimumDimension()` (Matryoshka); `encoding_format` `float` or `base64`
  (little-endian float32). Video is sampled one frame per second, at most 32 (`EMBEDDING_FRAMES`).
* **Rerank**: `query` + `documents` scored with the reranker's `defaultInstruction()`, sorted stably by
  score, cut to `top_n`, optional `return_documents`.
* **Decision**: `state` (text or any JSON, or null with media) plus `files`/`images` data URIs; questions
  of type `noul` → `Predicate`, `choice` → `Choice`, `score` → `Rating` (2–10 levels); `joint: true`
  calls `decide`, otherwise `decideIndependently`. Answers carry probabilities, confidence, and for
  scores the probability-weighted mean level. Video uses `FRAMES` (2 fps, 4–768 frames, via `VideoCodec.ffmpeg()`).
* **Transcription**: multipart `file` + `response_format` (`json`, `text`, `verbose_json` with word timings).
* Each keeps its own `AtomicLong` counters and builds a small exposition with `Metrics.metric`.

---

## 9. Errors, validation and observability

```mermaid
flowchart LR
    V["Validation.require"] --> IAE["IllegalArgumentException"]
    RP["Validation.requireParam / Values.mistyped"] --> IP["Values.InvalidParam: param"]
    VM["Validation.validateModel"] --> UM["Validation.UnknownModel"]
    IAE & IP --> R400["Server.refuse → 400 invalid_request_error"]
    UOE["UnsupportedOperationException"] --> R400
    UM --> R404["Server.refuse → 404 not_found_error"]
    OTH["any other RuntimeException"] --> R500["500 server_error, logged, not echoed"]
    R400 & R404 & R500 --> ENV["Http.errorPayload: message, type, param, code"]
```

* The rule everywhere (JSON routes, generation scaffold, `Sse.guarded`, `TaskTransport.work`): only
  `IllegalArgumentException` and `UnsupportedOperationException` are the client's fault.
  `Http.errorMessage` keeps the first line, ≤ 512 chars.
* `Validation.validateGenerationParams` checks only fields the request carries (OpenAI ranges for
  temperature, top_p, top_k, min_p, budget, stops ≤ 4, `n == 1`), refuses unsupported options by value
  (logprobs, logit_bias, non-zero penalties), and refuses `grammar` + JSON `response_format` together or
  any grammar when `Limits.grammar` is off.
* `Metrics` is **per instance** (two servers in one JVM do not share counters); it records outcomes
  (`INVALID_REQUEST`, `REJECTED`, `CANCELLED`, `FAILED`, `CLIENT_DISCONNECTED`), tokens, prompt-cache tier
  hits (`PromptCache.Tier` SESSION/BLOCKS/FRESH), speculation, and exports the engine's
  `PromptCache.Sample` and `MediaEncodingCache.Sample` plus queue depth and worker busy.
* The per-inference JFR-style `InferenceEvent` is emitted by `ChatEngine`; the server only commits one
  itself for requests it sheds in `runQueued` (`errorType` `queue-full`, `interrupted`, `shutdown`).
* Logging goes through `Log.LOG` (`System.getLogger("jinfer.server")`); `Http.AccessLog` writes one INFO
  line per exchange (method, URI, status, ms, remote address).
