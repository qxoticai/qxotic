# jinfer-spring-ai-autoconfigure — class architecture

Package `com.qxotic.jinfer.spring.ai.autoconfigure` (15 classes, ~650 lines). It is the **Spring Boot
auto-configuration** layer over `jinfer-spring-ai`: seven `@AutoConfiguration` classes, each wiring one
Spring AI model bean (chat, embedding, decision, rerank, speech, transcription) or a session repository from
`spring.ai.jinfer.*` properties. It contains no inference code: every bean is built through the adapter's
`Builder` (`JinferChatModel.builder()`, ...), and the adapter resolves model refs through `jinfer-hub` and
loads GGUFs through `jinfer-chat`'s `Models` / `ChatEngine`. The `jinfer-spring-ai-spring-boot-starter` module
bundles this jar with `jinfer-spring-ai`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Registration and activation](#2-registration-and-activation) — 2a SPI imports, 2b conditions
3. [Class diagrams](#3-class-diagrams) — 3a configurations and properties, 3b built beans
4. [Bean creation sequence](#4-bean-creation-sequence)
5. [Model source resolution — `Sources`](#5-model-source-resolution--sources)
6. [Property reference](#6-property-reference)
7. [Build-time metadata](#7-build-time-metadata)

| Class | Role |
|---|---|
| `JinferChatAutoConfiguration` | `@AutoConfiguration` wiring a `JinferChatModel` bean; gated by `JinferChatSelected` |
| `JinferChatProperties` | `record` bound to `spring.ai.jinfer.chat`; `toOptions()` builds a `JinferChatOptions` |
| `JinferChatSelected` | package-private `Condition`: `spring.ai.model.chat=jinfer`, or no selection and `spring.ai.jinfer.chat.model` set |
| `JinferEmbeddingAutoConfiguration` | wires a `JinferEmbeddingModel` when `spring.ai.model.embedding=jinfer` and no `EmbeddingModel` bean exists |
| `JinferEmbeddingProperties` | `record` bound to `spring.ai.jinfer.embedding` (model, companions, contextCapacity, dimensions) |
| `JinferDecisionAutoConfiguration` | wires a `JinferDecisionModel` when `spring.ai.model.decision=jinfer` and no `DecisionModel` bean exists |
| `JinferDecisionProperties` | `record` bound to `spring.ai.jinfer.decision` (model, companions, contextCapacity, independent) |
| `JinferRerankAutoConfiguration` | wires a `JinferDocumentPostProcessor` (RAG post-retrieval) when `spring.ai.jinfer.rerank.model` is set |
| `JinferRerankProperties` | `record` bound to `spring.ai.jinfer.rerank` (model, contextCapacity, instruction, topK, minScore) |
| `JinferSpeechAutoConfiguration` | wires a `JinferSpeechModel` (TTS) when `spring.ai.jinfer.speech.model` is set and speech provider is jinfer/unset |
| `JinferSpeechProperties` | `record` bound to `spring.ai.jinfer.speech` (model, companions, speed, maxInputChars) |
| `JinferTranscriptionAutoConfiguration` | wires a `JinferTranscriptionModel` (STT) when `spring.ai.jinfer.transcription.model` is set and provider is jinfer/unset |
| `JinferTranscriptionProperties` | `record` bound to `spring.ai.jinfer.transcription` (model only) |
| `JinferSessionAutoConfiguration` | fallback `InMemorySessionRepository` bean when no `SessionRepository` exists |
| `Sources` | package-private helper: routes a property value to the builder's `model(String ref)` or `modelPath(Path)`, refuses URLs, expands `companions` maps |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    App["Spring Boot application<br/>application.properties: spring.ai.jinfer.*"]
    Starter["jinfer-spring-ai-spring-boot-starter<br/>(pom only)"]
    subgraph auto["jinfer-spring-ai-autoconfigure"]
        AC["7 x Jinfer*AutoConfiguration"]
        PR["6 x Jinfer*Properties records"]
        SRC["Sources"]
        SEL["JinferChatSelected"]
    end
    subgraph sai["jinfer-spring-ai"]
        CM["JinferChatModel / JinferChatOptions"]
        EM["JinferEmbeddingModel"]
        DM["JinferDecisionModel"]
        RR["JinferDocumentPostProcessor"]
        SM["JinferSpeechModel"]
        TM["JinferTranscriptionModel"]
    end
    Hub["jinfer-hub: ModelStore"]
    Chat["jinfer-chat: Models / ChatEngine"]
    Codecs["jinfer-codecs: VideoSampler"]
    Spring["Spring Boot / Spring AI<br/>spring-ai-model, spring-ai-rag, spring-ai-session"]

    App --> Starter --> auto
    Starter --> sai
    AC --> PR
    AC --> SRC
    AC -.->|builder| sai
    SRC --> Hub
    sai --> Hub
    sai --> Chat
    AC -.->|optional bean| Codecs
    AC -.->|annotations, conditions| Spring
```

All dependencies in `pom.xml` are `<optional>true</optional>` (jinfer-codecs, jinfer-hub, jinfer-spring-ai,
spring-ai-model, spring-ai-rag, spring-ai-session, micrometer-observation, spring-boot, spring-boot-autoconfigure,
spring-beans, spring-context, spring-core): the starter supplies the runtime classpath, and each
auto-configuration guards itself with `@ConditionalOnClass`. Test scope pulls real model ports (lfm2, lfm3,
gemma-embedding2, inflect2, parakeet), `jam-native`, `jinfer-testkit` and `spring-ai-autoconfigure-model-session`.

---

## 2. Registration and activation

### 2a. SPI registration

Boot discovers the configurations through
`src/main/resources/META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`
(no `META-INF/services` file in this module).

```mermaid
flowchart LR
    Imports["META-INF/spring/<br/>...AutoConfiguration.imports"]
    Imports --> C1["JinferChatAutoConfiguration"]
    Imports --> C2["JinferEmbeddingAutoConfiguration"]
    Imports --> C3["JinferDecisionAutoConfiguration"]
    Imports --> C4["JinferRerankAutoConfiguration"]
    Imports --> C5["JinferSpeechAutoConfiguration"]
    Imports --> C6["JinferTranscriptionAutoConfiguration"]
    Imports --> C7["JinferSessionAutoConfiguration"]
    C7 -.->|afterName| J["JdbcSessionRepositoryAutoConfiguration<br/>external"]
    C7 -.->|beforeName| S["SessionServiceAutoConfiguration<br/>external"]
```

`JinferSessionAutoConfiguration` is the only one with ordering: it runs after Spring AI's JDBC session
repository auto-configuration (so a JDBC repository wins) and before `SessionServiceAutoConfiguration` (which
needs a `SessionRepository` bean).

### 2b. Activation conditions

Each capability chooses its own switch. Chat, embedding and decision follow Spring AI's
`spring.ai.model.<type>` selector; rerank has no Spring AI model type, so the model property itself is the
switch; speech and transcription need both a model and a non-conflicting provider selection.

```mermaid
flowchart TB
    subgraph chat["Chat"]
        Q1{"spring.ai.model.chat set?"}
        Q1 -->|yes| Q1a{"== jinfer?"}
        Q1a -->|yes| ON1["active (missing model = boot error)"]
        Q1a -->|no| OFF1["off"]
        Q1 -->|no| Q1b{"spring.ai.jinfer.chat.model has text?"}
        Q1b -->|yes| ON1
        Q1b -->|no| OFF1
    end
    subgraph emb["Embedding / Decision"]
        Q2{"spring.ai.model.embedding == jinfer<br/>or spring.ai.model.decision == jinfer"}
        Q2 -->|yes| ON2["active, unless an EmbeddingModel / DecisionModel bean exists"]
        Q2 -->|no| OFF2["off"]
    end
    subgraph rr["Rerank"]
        Q3{"spring.ai.jinfer.rerank.model present?"}
        Q3 -->|yes| ON3["active"]
        Q3 -->|no| OFF3["off"]
    end
    subgraph audio["Speech / Transcription"]
        Q4{"spring.ai.jinfer.speech.model or<br/>.transcription.model present?"}
        Q4 -->|no| OFF4["off"]
        Q4 -->|yes| Q4a{"spring.ai.model.audio.speech / .transcription<br/>== jinfer or missing?"}
        Q4a -->|yes| ON4["active"]
        Q4a -->|no| OFF4
    end
```

| Configuration | `@ConditionalOnClass` | Property gate | Bean guard |
|---|---|---|---|
| Chat | `JinferChatModel` | `@Conditional(JinferChatSelected)` | `@ConditionalOnMissingBean` (type `JinferChatModel`) |
| Embedding | `JinferEmbeddingModel` | `spring.ai.model.embedding=jinfer` | `@ConditionalOnMissingBean(EmbeddingModel.class)` |
| Decision | `JinferDecisionModel`, `DecisionModel` | `spring.ai.model.decision=jinfer` | `@ConditionalOnMissingBean(DecisionModel.class)` |
| Rerank | `JinferDocumentPostProcessor`, `DocumentPostProcessor` | `spring.ai.jinfer.rerank.model` present | `@ConditionalOnMissingBean` (type `JinferDocumentPostProcessor`) |
| Speech | `JinferSpeechModel` | `spring.ai.model.audio.speech=jinfer` (matchIfMissing) **and** `spring.ai.jinfer.speech.model` | `@ConditionalOnMissingBean` |
| Transcription | `JinferTranscriptionModel` | `spring.ai.model.audio.transcription=jinfer` (matchIfMissing) **and** `spring.ai.jinfer.transcription.model` | `@ConditionalOnMissingBean` |
| Session | `SessionRepository` | — | `@ConditionalOnMissingBean(SessionRepository.class)` |

`JinferChatSelected` exists because a `matchIfMissing` default alone made an embeddings/rerank/speech-only app
fail with "spring.ai.jinfer.chat.model is required" (per its Javadoc).

---

## 3. Class diagrams

### 3a. Configurations, properties and helpers

```mermaid
classDiagram
    direction LR
    class Condition { <<interface, external>> }
    class JinferChatSelected {
        <<final, package-private>>
        +matches(context, metadata) boolean
    }
    class Sources {
        <<final, package-private>>
        +model(key, value, ref, path)$
        +companions(prefix, map, ref, path)$
    }
    class ModelStore { <<external>> }
    class JinferChatAutoConfiguration {
        +jinferChatModel(props, obsRegistry, obsConvention, videoSampler) JinferChatModel
    }
    class JinferEmbeddingAutoConfiguration {
        +jinferEmbeddingModel(props, obsRegistry, obsConvention) JinferEmbeddingModel
    }
    class JinferDecisionAutoConfiguration {
        +jinferDecisionModel(props, obsRegistry, obsConvention, videoSampler) JinferDecisionModel
    }
    class JinferRerankAutoConfiguration {
        +jinferDocumentPostProcessor(props) JinferDocumentPostProcessor
    }
    class JinferSpeechAutoConfiguration {
        +jinferSpeechModel(props, obsRegistry, obsConvention) JinferSpeechModel
    }
    class JinferTranscriptionAutoConfiguration {
        +jinferTranscriptionModel(props, obsRegistry, obsConvention) JinferTranscriptionModel
    }
    class JinferSessionAutoConfiguration {
        ~jinferInMemorySessionRepository() SessionRepository
    }
    class JinferChatProperties {
        <<record>>
        +toOptions() JinferChatOptions
    }
    class JinferEmbeddingProperties { <<record>> }
    class JinferDecisionProperties { <<record>> }
    class JinferRerankProperties { <<record>> }
    class JinferSpeechProperties { <<record>> }
    class JinferTranscriptionProperties { <<record>> }

    Condition <|.. JinferChatSelected
    JinferChatAutoConfiguration ..> JinferChatSelected : Conditional
    JinferChatAutoConfiguration ..> JinferChatProperties : EnableConfigurationProperties
    JinferEmbeddingAutoConfiguration ..> JinferEmbeddingProperties : EnableConfigurationProperties
    JinferDecisionAutoConfiguration ..> JinferDecisionProperties : EnableConfigurationProperties
    JinferRerankAutoConfiguration ..> JinferRerankProperties : EnableConfigurationProperties
    JinferSpeechAutoConfiguration ..> JinferSpeechProperties : EnableConfigurationProperties
    JinferTranscriptionAutoConfiguration ..> JinferTranscriptionProperties : EnableConfigurationProperties
    JinferChatAutoConfiguration --> Sources
    JinferEmbeddingAutoConfiguration --> Sources
    JinferDecisionAutoConfiguration --> Sources
    JinferRerankAutoConfiguration --> Sources
    JinferSpeechAutoConfiguration --> Sources
    JinferTranscriptionAutoConfiguration --> Sources
    Sources --> ModelStore : isRef / requireRef
```

All properties classes are `record`s annotated `@ConfigurationProperties` (constructor binding, defaults via
`@DefaultValue`). `JinferEmbeddingProperties` additionally marks its canonical constructor `@ConstructorBinding`
and keeps a two-argument `(model, contextCapacity)` convenience constructor.

### 3b. Beans built and their Spring AI contracts

```mermaid
classDiagram
    direction LR
    class ChatModel { <<interface, external>> }
    class EmbeddingModel { <<interface, external>> }
    class DecisionModel { <<interface, external>> }
    class DocumentPostProcessor { <<interface, external>> }
    class TextToSpeechModel { <<interface, external>> }
    class TranscriptionModel { <<interface, external>> }
    class SessionRepository { <<interface, external>> }
    class InMemorySessionRepository { <<external>> }
    class JinferChatModel { <<final, external>> }
    class JinferChatOptions { <<final, external>> }
    class JinferEmbeddingModel { <<final, external>> }
    class JinferDecisionModel { <<final, external>> }
    class JinferDocumentPostProcessor { <<final, external>> }
    class JinferSpeechModel { <<final, external>> }
    class JinferTranscriptionModel { <<final, external>> }

    ChatModel <|.. JinferChatModel
    EmbeddingModel <|.. JinferEmbeddingModel
    DecisionModel <|.. JinferDecisionModel
    DocumentPostProcessor <|.. JinferDocumentPostProcessor
    TextToSpeechModel <|.. JinferSpeechModel
    TranscriptionModel <|.. JinferTranscriptionModel
    SessionRepository <|.. InMemorySessionRepository
    JinferChatModel --> JinferChatOptions : default options

    JinferChatAutoConfiguration ..> JinferChatModel : builder().build()
    JinferChatProperties ..> JinferChatOptions : toOptions()
    JinferEmbeddingAutoConfiguration ..> JinferEmbeddingModel : builder().build()
    JinferDecisionAutoConfiguration ..> JinferDecisionModel : builder().build()
    JinferRerankAutoConfiguration ..> JinferDocumentPostProcessor : builder().build()
    JinferSpeechAutoConfiguration ..> JinferSpeechModel : builder().build()
    JinferTranscriptionAutoConfiguration ..> JinferTranscriptionModel : builder().build()
    JinferSessionAutoConfiguration ..> InMemorySessionRepository : builder().build()
```

All six `Jinfer*` adapters (in `jinfer-spring-ai`) also implement `AutoCloseable`, so Spring's inferred destroy
method closes them on context shutdown. Optional collaborators come in as `ObjectProvider`s and are applied with
`ifAvailable`:

| Configuration | `ObservationRegistry` | `*ObservationConvention` | `VideoSampler` |
|---|---|---|---|
| Chat | yes | `ChatModelObservationConvention` | yes (overrides UNIFORM) |
| Embedding | yes | `EmbeddingModelObservationConvention` | — |
| Decision | yes | `DecisionModelObservationConvention` | yes |
| Speech | yes | `TextToSpeechModelObservationConvention` | — |
| Transcription | yes | `TranscriptionModelObservationConvention` | — |
| Rerank | — | — | — |

---

## 4. Bean creation sequence

The chat bean is representative; the other model configurations follow the same shape (validate, builder,
`Sources.model`, `Sources.companions`, optional providers, `build()`).

```mermaid
sequenceDiagram
    autonumber
    participant Boot as Spring Boot context
    participant Sel as JinferChatSelected
    participant AC as JinferChatAutoConfiguration
    participant P as JinferChatProperties
    participant S as Sources
    participant MS as ModelStore (jinfer-hub)
    participant B as JinferChatModel.Builder
    participant E as ChatEngine (jinfer-chat)

    Boot->>Sel: matches(context, metadata)
    Sel-->>Boot: true
    Boot->>P: bind spring.ai.jinfer.chat.*
    Boot->>AC: jinferChatModel(props, providers...)
    AC->>AC: require properties.model() has text
    AC->>P: toOptions()
    P-->>AC: JinferChatOptions
    AC->>B: retainedSessions / options / speculationDepth / contextCapacity
    AC->>S: model("spring.ai.jinfer.chat.model", value, b::model, b::modelPath)
    S->>MS: requireRef(value) if value contains "://"
    S->>MS: isRef(value)
    alt model ref
        S->>B: model(String ref)
    else local path
        S->>B: modelPath(Path)
    end
    AC->>B: observationRegistry / observationConvention / videoSampler (ifAvailable)
    AC->>S: companions("spring.ai.jinfer.chat", map, b::companion, b::companionPath)
    opt promptCache set
        AC->>B: promptCache(Path)
    end
    AC->>B: build()
    B->>MS: standard().resolveAll(refs) — download when absent
    B->>E: new ChatEngine(modelPath, companionPaths, cacheOptions)
    B-->>AC: JinferChatModel
    AC-->>Boot: bean
```

Because `build()` resolves and loads at bean creation, a typo in a ref or an out-of-range option fails the
**boot**, not the first request (as documented in `JinferChatProperties` and tested by
`JinferChatAutoConfigurationTest`). The other adapters load via `Models.loadEmbedder`, `Models.loadDecisionModel`,
`Models.loadReranker`, `Models.loadSpeech` and `Models.loadTranscription` respectively.

Per-configuration validation done *before* the builder:

| Configuration | Checks in this module |
|---|---|
| Chat | model has text |
| Embedding | model has text |
| Decision | model has text |
| Rerank | model has text; `minScore` applied only when non-null (ColBERT MaxSim sums may be negative) ; `instruction` only when non-blank |
| Speech | model has text; `speed` positive and finite; `maxInputChars >= 0` (0 keeps the adapter default) |
| Transcription | model has text |

---

## 5. Model source resolution — `Sources`

```mermaid
flowchart TB
    In["property value"] --> U{"contains '://'?"}
    U -->|yes| R["ModelStore.requireRef(value)"]
    R -->|IllegalArgumentException| X["IllegalStateException<br/>'key: message'"]
    R -->|ok| IsRef
    U -->|no| IsRef{"ModelStore.isRef(value)?"}
    IsRef -->|yes| Ref["builder.model(String) / companion(cap, String)"]
    IsRef -->|no| Path["builder.modelPath(Path.of) / companionPath(cap, Path)"]

    M["companions map"] --> N{"null?"}
    N -->|yes| Done["nothing"]
    N -->|no| Each["for each capability -> value"]
    Each --> Blank{"blank value?"}
    Blank -->|yes| X2["IllegalStateException<br/>'prefix.companions.cap must not be blank'"]
    Blank -->|no| In
```

`Sources.model` never downloads from a URL: a URL goes through `ModelStore.requireRef`, whose error names the
ref a hub link spells, and the message is re-thrown prefixed with the property key. A value that is not a ref is
treated as a local path and never touches the network. Companions (`mmproj` media towers, Gemma 4's
`speculation` MTP sidecar, Inflect's `lexicon`, ...) reuse the same rule per capability key.

---

## 6. Property reference

| Prefix | Properties (default) |
|---|---|
| `spring.ai.jinfer.chat` | `model`, `companions.*`, `prompt-cache`, `retained-sessions` (1), `context-capacity` (0), `temperature`, `top-p`, `top-k`, `min-p`, `max-tokens`, `seed`, `thinking`, `max-reasoning-tokens`, `reasoning-cutoff-message`, `timeout` (Duration), `output-modalities` (Set of `Modality`), `speculation-depth` |
| `spring.ai.jinfer.embedding` | `model`, `companions.*`, `context-capacity` (0), `dimensions` (0 = model's own) |
| `spring.ai.jinfer.decision` | `model`, `companions.*`, `context-capacity` (0), `independent` (false) |
| `spring.ai.jinfer.rerank` | `model`, `context-capacity` (0), `instruction`, `top-k` (0 = keep all), `min-score` |
| `spring.ai.jinfer.speech` | `model`, `companions.*`, `speed`, `max-input-chars` (0 = 4096 adapter default) |
| `spring.ai.jinfer.transcription` | `model` |

`contextCapacity` everywhere: 0 = 4096 or the model's max when smaller, -1 = the model's max, other = exact
(refused at boot past the max). `JinferChatProperties.toOptions()` maps the sampling/reasoning/timeout/modality
fields one-to-one onto `JinferChatOptions.builder()`; null fields stay unset so the model's recommendations apply.

---

## 7. Build-time metadata

```mermaid
flowchart LR
    Rec["@ConfigurationProperties records"] --> AP["spring-boot-configuration-processor<br/>(annotationProcessorPaths)"]
    AP --> JSON["META-INF/spring-configuration-metadata.json"]
    JSON --> IDE["IDE completion for spring.ai.jinfer.*"]
    T["ConfigurationMetadataTest"] -.->|asserts keys present| JSON
```

The processor is declared in `maven-compiler-plugin`'s `annotationProcessorPaths` (JDK 23+ javac ignores
classpath-discovered processors, per `ConfigurationMetadataTest`), which also keeps it off the runtime classpath.
The jar manifest reserves `Automatic-Module-Name: com.qxotic.jinfer.spring.ai.autoconfigure`.
