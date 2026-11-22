# jinfer-langchain4j — class architecture

Package `com.qxotic.jinfer.langchain4j` (13 top-level types + `package-info`, ~4560 lines). It is the
**LangChain4j provider** for jinfer: in-process CPU inference over local GGUF files, no server. Each
public class implements one LangChain4j model interface (`ChatModel`, `StreamingChatModel`, `EmbeddingModel`,
`ScoringModel`, `DecisionModel`, `TextToSpeechModel`, `AudioTranscriptionModel`) and maps LangChain4j types
onto jinfer's framework-neutral runtime (`ChatEngine`, `Models`, `LoadedEmbedder`, `LoadedReranker`, ...).
The module registers **no** `META-INF/services` entries: users build instances with each class's `builder()`;
SPI discovery happens one layer down, in `jinfer-chat`'s `Models` (`ServiceLoader<ModelProvider>`).

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package (LangChain4j or other jinfer modules).

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a public surface, 2b chat internals, 2c the other adapters
3. [Builders and model sources](#3-builders-and-model-sources)
4. [Blocking chat flow](#4-blocking-chat-flow)
5. [Streaming chat flow](#5-streaming-chat-flow)
6. [Cached-prompt views and reply records](#6-cached-prompt-views-and-reply-records)
7. [Parameter precedence and structured output](#7-parameter-precedence-and-structured-output)
8. [Embedding, scoring and decision flows](#8-embedding-scoring-and-decision-flows)
9. [Lifecycle, ownership and concurrency](#9-lifecycle-ownership-and-concurrency)

| Class | Visibility | Role |
|---|---|---|
| `JinferChatModel` | public final | `ChatModel` + `AutoCloseable`: owns/shares one `ChatEngine`; request preparation (tools, schema, grammar, sampling) shared with the streaming twin; `withCachedPrompt` views; `tokenCountEstimator()`; nested `Builder` |
| `JinferStreamingChatModel` | public final | `StreamingChatModel`: twin over the same `JinferChatModel`/engine; handler callbacks and `Flow.Publisher` of events; nested `Guarded` subscriber and `Relay` handler |
| `JinferChatRequestParameters` | public | extends `DefaultChatRequestParameters`: adds `grammar`, `seed`, `minP`, `maxReasoningTokens`, `thinking`, `reasoningCutoffMessage`, `timeout`, `outputModalities`; nested `Builder` |
| `JinferTokenUsage` | public final | extends `TokenUsage`: cached input tokens, reasoning tokens, `PromptCache.Tier`, prefill/decode nanos; overrides `add` |
| `JinferEmbeddingModel` | public final | `EmbeddingModel`: `LoadedEmbedder` + one reusable `ContextState`; query/document prefixes; Matryoshka dimensions; multimodal inputs |
| `JinferScoringModel` | public final | `ScoringModel`: `LoadedReranker` + one reusable state; instruction slot; shared-frame reranking |
| `JinferDecisionModel` | public final | LangChain4j `DecisionModel`: wraps jinfer `chat.DecisionModel`; maps inputs (text/fields/contents/messages) and questions (yes-no/choice/scale) |
| `JinferSpeechModel` | public final | `TextToSpeechModel`: `SpeechSynthesisModel`, WAV output, read/write lifecycle lock |
| `JinferTranscriptionModel` | public final | `AudioTranscriptionModel`: `TranscriptionModel`, read/write lifecycle lock |
| `Mappings` | package-private final | static mapping seam: LangChain4j messages/media/tools/JSON schema to jinfer types and back; finish reasons, timeouts |
| `Replies` | package-private final | LRU of recent replies' `Verbatim` token records keyed by `ContentKey` of the reply's visible content |
| `CachedPrompt` | package-private record | a view's prefix (messages + welded tools); tool precedence rule |
| `Estimators` | package-private final | `TokenCountEstimator` over the model's own tokenizer and media position plan |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    App["Application / LangChain4j AiServices, RAG, guardrails"]
    subgraph lc4j["langchain4j-core (external)"]
        IF["ChatModel / StreamingChatModel / EmbeddingModel<br/>ScoringModel / DecisionModel / TextToSpeechModel<br/>AudioTranscriptionModel"]
    end
    subgraph mod["jinfer-langchain4j"]
        CM["JinferChatModel"]
        SCM["JinferStreamingChatModel"]
        EM["JinferEmbeddingModel"]
        SM["JinferScoringModel"]
        DM["JinferDecisionModel"]
        TTS["JinferSpeechModel"]
        STT["JinferTranscriptionModel"]
        MAP["Mappings / Replies / CachedPrompt / Estimators"]
    end
    subgraph chat["jinfer-chat"]
        CE["ChatEngine"]
        MO["Models (ServiceLoader of ModelProvider)"]
        LD["LoadedModel / LoadedEmbedder / LoadedReranker / DecisionModel"]
        GR["llm.Grammar"]
    end
    subgraph other["other jinfer modules"]
        PC["jinfer-cache PromptCache"]
        HUB["jinfer-hub ModelStore"]
        COD["jinfer-codecs Image/Audio/VideoCodec, VideoSampler"]
        CORE["jinfer-core Arenas, ContextState, SpeechSynthesisModel, TranscriptionModel"]
        TOK["toknroll Tokenizer"]
    end
    Ports["jinfer-model ports (qwen3, gemma4, lfm2, ...) at runtime"]

    App --> IF
    CM -.->|implements| IF
    SCM -.->|implements| IF
    EM -.->|implements| IF
    SM -.->|implements| IF
    DM -.->|implements| IF
    TTS -.->|implements| IF
    STT -.->|implements| IF
    CM --> CE
    SCM --> CM
    CM --> MAP
    CE --> MO
    EM --> MO
    SM --> MO
    DM --> MO
    TTS --> MO
    STT --> MO
    MO --> LD
    MO -.->|SPI| Ports
    CE --> PC
    CM --> GR
    CM --> HUB
    MAP --> COD
    EM --> CORE
    MAP --> TOK
```

Maven dependencies (compile scope, from `pom.xml`): `langchain4j-core`, `slf4j-api`, `jinfer-core`, `jota-memory`,
`jinfer-chat`, `jinfer-codecs`, `jinfer-cache`, `json`, `jinfer-hub`, `toknroll-core`. Model ports
(`jinfer-models-all`), `jam-native`, `gguf`, `jinfer-kernels` etc. are **test-only**: an application picks its
ports explicitly. The jar carries `Automatic-Module-Name: com.qxotic.jinfer.langchain4j`.

---

## 2. Class diagrams

### 2a. Public surface — LangChain4j contracts

```mermaid
classDiagram
    direction LR
    class ChatModel { <<interface, external>> }
    class StreamingChatModel { <<interface, external>> }
    class EmbeddingModel { <<interface, external>> }
    class ScoringModel { <<interface, external>> }
    class LC4JDecisionModel { <<interface, external>> }
    class TextToSpeechModel { <<interface, external>> }
    class AudioTranscriptionModel { <<interface, external>> }
    class DefaultChatRequestParameters { <<external>> }
    class TokenUsage { <<external>> }
    class AutoCloseable { <<interface, external>> }

    class JinferChatModel { <<final>> }
    class JinferStreamingChatModel { <<final>> }
    class JinferEmbeddingModel { <<final>> }
    class JinferScoringModel { <<final>> }
    class JinferDecisionModel { <<final>> }
    class JinferSpeechModel { <<final>> }
    class JinferTranscriptionModel { <<final>> }
    class JinferChatRequestParameters
    class JinferTokenUsage { <<final>> }

    ChatModel <|.. JinferChatModel
    StreamingChatModel <|.. JinferStreamingChatModel
    EmbeddingModel <|.. JinferEmbeddingModel
    ScoringModel <|.. JinferScoringModel
    LC4JDecisionModel <|.. JinferDecisionModel
    TextToSpeechModel <|.. JinferSpeechModel
    AudioTranscriptionModel <|.. JinferTranscriptionModel
    AutoCloseable <|.. JinferChatModel
    AutoCloseable <|.. JinferStreamingChatModel
    AutoCloseable <|.. JinferEmbeddingModel
    AutoCloseable <|.. JinferScoringModel
    AutoCloseable <|.. JinferDecisionModel
    AutoCloseable <|.. JinferSpeechModel
    AutoCloseable <|.. JinferTranscriptionModel
    DefaultChatRequestParameters <|-- JinferChatRequestParameters
    TokenUsage <|-- JinferTokenUsage
```

`LC4JDecisionModel` stands for `dev.langchain4j.model.decision.DecisionModel` (the class refers to jinfer's
homonym by its fully qualified name `com.qxotic.jinfer.chat.DecisionModel`).

### 2b. Chat internals

```mermaid
classDiagram
    direction TB
    class JinferChatModel {
        <<final>>
        ~ChatEngine engine
        ~ChatRequestParameters defaults
        ~boolean thinking
        ~Integer maxReasoningTokens
        ~boolean describeSchema
        ~Duration timeout
        ~List listeners
        ~VideoSampler videoSampler
        ~Replies replies
        ~CachedPrompt prefix
        -AtomicBoolean warnedToolsOverride
        +builder()$ Builder
        +doChat(ChatRequest) ChatResponse
        +withCachedPrompt(messages, tools) JinferChatModel
        +saveCachedPrompts(Path)
        +streaming() JinferStreamingChatModel
        +tokenCountEstimator() TokenCountEstimator
        +thinkingPolicy() ThinkingPolicy
        +supportedCapabilities() Set
        +close()
        ~prepare(ChatRequest) Prepared
        ~offeredTools(ChatRequest) List
        ~validate(params, tools)
        ~resolveDefaults(model, recommended, b)$
    }
    class JinferStreamingChatModel {
        <<final>>
        -JinferChatModel model
        +blocking() JinferChatModel
        +doChat(ChatRequest, handler)
        +doChat(ChatRequest) Publisher
        -stream(Prepared, offered, handler)
    }
    class Guarded { <<JinferStreamingChatModel.Guarded>> }
    class Relay { <<record, JinferStreamingChatModel.Relay>> }
    class CachedPrompt {
        <<record>>
        messages List~Message~
        tools List~Tool~
        NONE$
        merge(moreMessages, moreTools) CachedPrompt
        conversation(thinking) Conversation
        resolveTools(stated) List
        serves(effective) boolean
    }
    class Replies {
        <<final>>
        -LinkedHashMap recent
        -long budget
        remember(AiMessage, Verbatim)
        recall(AiMessage) Verbatim
    }
    class Mappings {
        <<final, static>>
        toMessages(msgs, sampler, replies)$
        media(content, sampler)$
        toTools(specs)$
        toSchemaMap(element)$
        describeSchema(schema)$
        toAiMessage(Message)$
        unwrapStringifiedArguments(ai, tools)$
        requireInTime(done, timeout)$
        response(name, ai, promptTokens, done)$
        toFinishReason(reason, hasCalls)$
    }
    class Estimators {
        <<final>>
        -Tokenizer tokenizer
        -ToIntFunction mediaPositions
        -VideoSampler videoSampler
        -Replies replies
    }
    class JinferTokenUsage { <<final>> }
    class JinferChatRequestParameters
    class ChatEngine { <<external>> }
    class LoadedModel { <<record, external>> }
    class Grammar { <<external>> }
    class TokenCountEstimator { <<interface, external>> }

    JinferChatModel --> ChatEngine : engine shared by views/twin
    JinferChatModel --> CachedPrompt : prefix
    JinferChatModel --> Replies : replies shared
    JinferChatModel ..> Mappings : uses
    JinferChatModel ..> Grammar : schemaGbnf / jsonGbnf
    JinferChatModel ..> JinferChatRequestParameters : resolveDefaults
    JinferChatModel ..> Estimators : tokenCountEstimator
    JinferChatModel ..> JinferStreamingChatModel : streaming
    JinferStreamingChatModel --> JinferChatModel : model
    JinferStreamingChatModel ..> Guarded : creates
    JinferStreamingChatModel ..> Relay : creates
    Estimators ..|> TokenCountEstimator
    Estimators --> Replies
    Estimators ..> Mappings
    Mappings ..> Replies : recall
    Mappings ..> JinferTokenUsage : creates
    ChatEngine --> LoadedModel : loaded
```

### 2c. The other adapters and their jinfer delegates

```mermaid
classDiagram
    direction LR
    class JinferEmbeddingModel {
        <<final>>
        -LoadedEmbedder loaded
        -int dimension
        -ContextState state
        -Arena arena
        -ReentrantLock lock
        +doEmbed(EmbeddingRequest) EmbeddingResponse
        +dimension() int
        +tokenCountEstimator()
        ~closeStateThenArena(state, arena)$
    }
    class JinferScoringModel {
        <<final>>
        -LoadedReranker loaded
        -RuntimeState state
        -Arena arena
        -ReentrantLock lock
        -String instruction
        +scoreAll(segments, query) Response
        +doScore(ScoringRequest) ScoringResponse
    }
    class JinferDecisionModel {
        <<final>>
        -DecisionModel model
        -ContextState state
        -Arena arena
        -ReentrantLock lock
        -boolean independent
        -VideoSampler videoSampler
        +doDecide(DecisionRequest) DecisionResponse
    }
    class JinferSpeechModel {
        <<final>>
        -SpeechSynthesisModel model
        -Arena owned
        -ReentrantReadWriteLock lifecycle
        -SpeechOptions defaults
        -int maxInputChars
        +synthesize(TextToSpeechRequest) TextToSpeechResponse
    }
    class JinferTranscriptionModel {
        <<final>>
        -TranscriptionModel model
        -Arena owned
        -ReentrantReadWriteLock lifecycle
        +transcribe(AudioTranscriptionRequest) AudioTranscriptionResponse
        +transcribe(Path) Transcription
        +transcribe(byte[]) Transcription
    }
    class LoadedEmbedder { <<record, external>> }
    class LoadedReranker { <<record, external>> }
    class JDecisionModel { <<interface, external>> }
    class SpeechSynthesisModel { <<interface, external>> }
    class TranscriptionModel { <<interface, external>> }
    class Models { <<external>> }
    class Mappings { <<final>> }
    class Estimators { <<final>> }

    JinferEmbeddingModel --> LoadedEmbedder
    JinferScoringModel --> LoadedReranker
    JinferDecisionModel --> JDecisionModel
    JinferSpeechModel --> SpeechSynthesisModel
    JinferTranscriptionModel --> TranscriptionModel
    JinferEmbeddingModel ..> Models : loadEmbedder
    JinferScoringModel ..> Models : loadReranker
    JinferDecisionModel ..> Models : loadDecisionModel
    JinferSpeechModel ..> Models : loadSpeech
    JinferTranscriptionModel ..> Models : loadTranscription
    JinferEmbeddingModel ..> Mappings : media
    JinferDecisionModel ..> Mappings : parts / media
    JinferEmbeddingModel ..> Estimators : tokenCountEstimator
    JinferScoringModel ..> JinferEmbeddingModel : closeStateThenArena
    JinferDecisionModel ..> JinferEmbeddingModel : closeStateThenArena
```

`JDecisionModel` is `com.qxotic.jinfer.chat.DecisionModel` (questions `Predicate`, `Choice`, `Rating`, `Option`;
`decide` / `decideIndependently` returning one probability row per question).

---

## 3. Builders and model sources

Every adapter has a nested `Builder` whose model **source** is one field (`Object source`), last setter wins:
a local `Path` (`modelPath`), a model-ref `String` (`model("owner/repo:Q8_0")`, validated by
`ModelStore.requireRef`), or an already-loaded jinfer object (`LoadedModel`, `LoadedEmbedder`, `LoadedReranker`,
jinfer `DecisionModel`, `SpeechSynthesisModel`, `TranscriptionModel`). Companions (`companion` /
`companionPath`, e.g. `"media"` projector, `"lexicon"`, `"speculation"`) exist on chat, embedding, decision and
speech builders and are refused when the source is a pre-loaded model.

```mermaid
flowchart TD
    B["Builder.build()"] --> S{"source type"}
    S -->|"null"| E["IllegalArgumentException"]
    S -->|"Loaded* / model object"| L["use as-is<br/>companions must be empty<br/>weights stay caller-owned"]
    S -->|"Path"| P["modelPath = path"]
    S -->|"String ref"| R["ModelStore.standard()"]
    R -->|"resolveAll(ref + companion refs) in one batch"| P
    P --> C["companionPaths = local + resolved refs"]
    C --> K["new JinferXxxModel(this)"]
    L --> K
    K --> LOAD{"loaded given?"}
    LOAD -->|"no"| M["Models.loadXxx(path, arena, companions)<br/>or new ChatEngine(path, companions, cacheOptions)"]
    LOAD -->|"yes"| N["new ChatEngine(loaded, name, cacheOptions)<br/>or reuse loaded"]
```

`JinferScoringModel` and `JinferTranscriptionModel` take no companions and resolve a ref with
`ModelStore.standard().resolve(ref)`. Remote resolution happens only at `build()`; nothing on an inference
path downloads, and `Mappings` refuses non-`file://` media URLs.

---

## 4. Blocking chat flow

```mermaid
sequenceDiagram
    participant U as Caller
    participant LC as ChatModel.chat (langchain4j-core)
    participant CM as JinferChatModel
    participant CP as CachedPrompt
    participant MP as Mappings
    participant CE as ChatEngine
    participant R as Replies

    U->>LC: chat(ChatRequest)
    LC->>LC: defaults.overrideWith(request), listeners onRequest
    LC->>CM: doChat(request)
    CM->>CM: prepare(request)
    CM->>CP: resolveTools(statedTools)
    CM->>CM: validate, rejectUnsupported, rejectModelSwitch
    CM->>CP: serves(tools)? else warn once
    CM->>MP: toMessages(request.messages, videoSampler, replies)
    MP->>R: recall(AiMessage) per echoed reply
    CM->>CM: describeSchemaOnTheLastUserMessage (if schema)
    CM->>CE: prepare(ChatEngine.Request)
    CE-->>CM: Prepared
    CM->>CE: complete(prepared, ReplySink.NONE)
    CE-->>CM: Completion
    CM->>MP: requireInTime, toAiMessage, unwrapStringifiedArguments
    alt stop sequence hit
        CM->>CM: TextStops.apply - record not remembered
    else
        CM->>R: remember(ai, reply.verbatim)
    end
    CM->>MP: response(modelName, ai, promptTokens, done)
    MP-->>CM: ChatResponse with JinferTokenUsage
    CM-->>LC: ChatResponse
    LC-->>U: ChatResponse, listeners onResponse
```

`prepare` lowers a request into `ChatEngine.Request`: messages (view prefix first), effective tools, thinking
(off when `outputModalities` contains `AUDIO`), `maxOutputTokens`, reasoning cap and cutoff message, timeout,
sampling (`LoadedModel.samplingDefaults().resolve(temperature, topP, topK, minP, seed)`), GBNF grammar,
`ForcedTool.ANY` for `ToolChoice.REQUIRED`, stop sequences, output modalities. `UnsupportedOperationException`
from the engine is translated to LangChain4j's `UnsupportedFeatureException` by the private `framed` helpers.
`Mappings.toFinishReason` maps `Generator.FinishReason` (`STOP`, `LENGTH`, else `OTHER`; `TOOL_EXECUTION` when
calls exist); a timeout becomes `dev.langchain4j.exception.TimeoutException`.

---

## 5. Streaming chat flow

```mermaid
sequenceDiagram
    participant U as Caller
    participant S as JinferStreamingChatModel
    participant CM as JinferChatModel
    participant CE as ChatEngine
    participant H as StreamingChatResponseHandler

    U->>S: chat(request, handler)
    S->>CM: prepare(request) - synchronous, rejections throw
    S->>CM: offeredTools(request)
    S->>CE: stream(Runnable) - single lazy driver thread
    CE->>S: run stream(prepared, offered, handler)
    S->>CE: complete(prepared, ReplySink)
    loop each decoded Delta
        CE->>S: on(Delta)
        alt Channel.CONTENT
            S->>H: onPartialResponse
        else Channel.REASONING
            S->>H: onPartialThinking
        end
        CE->>S: onMedia(Media) - speech stretch
        S->>H: onUnmappedRawEvent(Media.Audio)
        CE->>S: cancelled()? via StreamingHandle
    end
    CE-->>S: Completion
    alt cancelled
        S-->>U: nothing more
    else
        S->>S: requireInTime, toAiMessage, remember record
        S->>H: onCompleteToolCall per call
        S->>H: onCompleteResponse(ChatResponse)
    end
```

User-callback exceptions are routed to `onError` per occurrence by `safely(...)` while generation continues.
The reactive `doChat(ChatRequest)` returns a `Flow.Publisher` that, per subscriber, creates a
`SubmissionPublisher` (delivered on the `CompletableFuture` default executor, buffer `1 << 20`), wraps the
subscriber in `Guarded` (cancels on a throwing `onNext`) and drives `doChat(request, new Relay(events))`;
`Relay` cancels the generation when no subscriber remains.

---

## 6. Cached-prompt views and reply records

```mermaid
flowchart LR
    Base["base JinferChatModel<br/>prefix = CachedPrompt.NONE"]
    V1["view A<br/>prefix = system + tools"]
    V2["view B<br/>prefix = A + addendum"]
    Tw["JinferStreamingChatModel"]
    CE["ChatEngine<br/>PromptCache: sessions, block tree, catalog"]
    R["Replies<br/>LRU ContentKey to Verbatim<br/>budget 2^24 token ids"]

    Base -->|"withCachedPrompt(msgs, tools)"| V1
    V1 -->|"withCachedPrompt(more, List.of())"| V2
    Base -->|"streaming()"| Tw
    Base --> CE
    V1 --> CE
    V2 --> CE
    Tw --> CE
    Base --> R
    V1 --> R
    V2 --> R
```

- `withCachedPrompt` converts messages/tools once (`Mappings.toMessages`, `Mappings.toTools`),
  `CachedPrompt.merge`s them onto the current prefix, calls `engine.definePrompt(conversation)` and returns a
  new view sharing every field except `prefix` and its own `warnedToolsOverride` flag.
- Tool precedence (`CachedPrompt.resolveTools`): a request that states tools (or `ToolChoice.NONE`) uses its own;
  unstated falls to the view's welded tools. `serves` tells whether the prepaid prefill applies.
- `saveCachedPrompts(Path)` calls `engine.freezePrompts`; `Builder.promptCache(Path)` mounts that artifact
  read-only through `PromptCache.Options.withCatalog(path, true)`; `retainedSessions` and `contextCapacity`
  also go into `PromptCache.Options`.
- `Replies` keys a reply by SHA-256 (`ContentKey.sha256`) of the JSON of its text, thinking and tool calls; an
  echoed `AiMessage` in later history is mapped to a jinfer `Message` carrying the original `Verbatim` token
  record, so the engine replays exact generated tokens. Stop-sequence-cut replies are never remembered.
- `JinferTokenUsage` exposes the effect per response: `cachedInputTokens` (`Completion.restoredTokens`),
  `servedFrom` (`PromptCache.Tier` `SESSION` / `BLOCKS` / `FRESH`), `promptNanos`, `predictedNanos`,
  `reasoningTokenCount`. Its `add` sums all counters and drops the tier.

---

## 7. Parameter precedence and structured output

```mermaid
flowchart TD
    E["port / GGUF recommendation<br/>LoadedModel.samplingDefaults"] --> D["defaultRequestParameters"]
    D --> X["explicit Builder setters<br/>temperature, topP, topK, minP, maxOutputTokens, seed"]
    X --> RD["resolved defaults<br/>always JinferChatRequestParameters"]
    RD --> M["ChatModel.chat merge<br/>defaults.overrideWith(request)"]
    Q["per-request parameters"] --> M
    M --> P["prepare()"]
    P --> G{"constraint"}
    G -->|"JSON schema"| G1["Mappings.toSchemaMap<br/>Grammar.schemaGbnf"]
    G -->|"ResponseFormat.JSON"| G2["Grammar.jsonGbnf"]
    G -->|"JinferChatRequestParameters.grammar"| G3["raw GBNF"]
    G1 --> DS["describeSchema line appended<br/>to last user message, unless disabled<br/>or tools offered before a tool round"]
```

`JinferChatRequestParameters.overrideWith` / `defaultedBy` and its `Builder.overrideWith` carry the jinfer-only
fields across merges, which is why defaults are always built jinfer-typed. `rejectUnsupported` refuses
`frequencyPenalty`, `presencePenalty`, and grammar combined with a JSON response format; `rejectModelSwitch`
refuses a different `modelName`. `supportedCapabilities()` returns `RESPONSE_FORMAT_JSON_SCHEMA`, which makes
AiServices use structured output for POJO extraction. `Mappings.unwrapStringifiedArguments` repairs tool
arguments that small models send as JSON strings where the declared schema says array/object.

---

## 8. Embedding, scoring and decision flows

```mermaid
sequenceDiagram
    participant U as Caller
    participant EM as JinferEmbeddingModel
    participant LE as LoadedEmbedder
    participant SM as JinferScoringModel
    participant LR as LoadedReranker
    participant DM as JinferDecisionModel
    participant JD as jinfer DecisionModel

    U->>EM: embed(EmbeddingRequest)
    EM->>EM: prefix from inputType QUERY/DOCUMENT, or warn once if prefix-trained
    alt all inputs text-only
        EM->>LE: embedAll(state, texts, dim, consumer) under lock
    else media present
        EM->>EM: Mappings.media(content, EMBEDDING_FRAMES) before lock
        EM->>LE: embed(state, parts, dim, consumer) per input
    end
    EM-->>U: EmbeddingResponse + TokenUsage

    U->>SM: scoreAll(segments, query) or score(ScoringRequest)
    SM->>SM: doScore - rejectModelSwitch
    SM->>LR: scoreAll(state, instruction, query, documents, sink) under lock
    SM-->>U: ScoringResponse scores + TokenUsage

    U->>DM: decide(DecisionRequest)
    DM->>DM: input(): Text / Fields / Contents / Messages to List of Content
    DM->>DM: question(): YesNo to Predicate, Choice to Choice, Scale to Rating
    DM->>JD: decide or decideIndependently under lock
    JD-->>DM: float rows
    DM-->>U: YesNoAnswer / ChoiceAnswer / ScaleAnswer
```

- Embedding: `dimension` resolves from `Builder.dimensions` against `LoadedEmbedder.minimumDimension..dimension`;
  `supportedParameters` adds `DIMENSIONS` only when `supportsCustomDimensions()`. `supportedContentTypes`
  adds IMAGE/VIDEO/AUDIO when the model is `Multimodal` and supports that media.
- Scoring: `instruction` defaults to `reranker().defaultInstruction()`; an explicit instruction on a reranker
  without `hasInstructionSlot()` is refused at build.
- Decision: a `MessagesInput` becomes a JSON transcript `{"messages": [...]}` with media markers
  (`[image 1]`, ...) and the media parts placed ahead of the JSON; token usage is the delta of
  `state.tokensRead()`.
- Speech and transcription (not drawn): `JinferSpeechModel.synthesize` refuses a named voice and over-long
  text, then `model.speak(text, SpeechOptions)` and `AudioCodec.wav`; `JinferTranscriptionModel` refuses prompt,
  temperature and language, decodes bytes with `AudioCodec` and calls `model.transcribe(audio)`.

---

## 9. Lifecycle, ownership and concurrency

| Adapter | Concurrency | State | `close()` | Weights freed |
|---|---|---|---|---|
| `JinferChatModel` (+ views, streaming twin) | serial; `ChatEngine` generation lock, streaming on its driver thread | engine-managed retained sessions + block tree | `engine.close()`; closes base, all views and twin | only if the engine loaded them (path/ref) |
| `JinferEmbeddingModel` | serial; fair `ReentrantLock` | one `ContextState` in a private cross-thread arena | lock, `closeStateThenArena` | arena closed; caller-loaded weights live in caller's arena |
| `JinferScoringModel` | serial; fair lock | one state | same as embedding | same |
| `JinferDecisionModel` | serial; fair lock | one `ContextState` from `model.newState` | same | same |
| `JinferSpeechModel` | parallel; read lock per request, state per call | none held | write lock waits for in-flight, idempotent | only an arena it created |
| `JinferTranscriptionModel` | parallel; read lock per request | none held | write lock waits for in-flight, idempotent | only an arena it created |

```mermaid
flowchart LR
    A["caller Arena<br/>Arenas.newCrossThread"] --> L["Models.loadEmbedder / load / loadReranker"]
    L --> P1["builder().model(loaded) pipeline 1"]
    L --> P2["builder().model(loaded) pipeline 2"]
    P1 -->|"close first"| X["caller closes Arena last"]
    P2 -->|"close first"| X
```

Constructors that allocate an arena or a live engine free it on any failure before rethrowing
(`Arenas.close`, `close(engine::close, e)` keeping the original exception primary). Parallel pipelines share
weights by building several adapters on one caller-owned loaded model, as documented in `package-info.java`.
