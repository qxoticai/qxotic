# jinfer-example-local-rag — class architecture

Package `com.qxotic.jinfer.example.localrag` (1 main class, ~90 lines; 1 integration test). It is a **demo Spring Boot
application**, not a library: fully-local retrieval-augmented generation in one JVM. A Jinfer embedding model fills a
Spring AI `SimpleVectorStore`, and a Jinfer chat model answers through `ChatClient` + `RetrievalAugmentationAdvisor`. No
model server, no API key. The module is not published to Maven Central (see `pom.xml` description).

Almost all behaviour lives in dependencies: Spring AI (vector store, RAG advisor, chat client), the jinfer Spring AI
adapters (`jinfer-spring-ai`, `jinfer-spring-ai-autoconfigure`) and, below them, `jinfer-chat` with the model ports
`jinfer-lfm2` and `jinfer-qwen3` discovered through `ServiceLoader`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this module.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Module dependencies](#2-module-dependencies)
3. [Class diagram](#3-class-diagram)
4. [Bean wiring (auto-configuration)](#4-bean-wiring-auto-configuration)
5. [Model loading and SPI discovery](#5-model-loading-and-spi-discovery)
6. [Run flow](#6-run-flow) — 6a ingestion, 6b question answering
7. [Configuration](#7-configuration)
8. [Integration test](#8-integration-test)

| Class | Role |
|---|---|
| `LocalRagApplication` | `@SpringBootApplication`; holds the 4-document `CORPUS` and 3 `QUESTIONS`; `demo(...)` `@Bean` returns a `CommandLineRunner`; static `run(chat, embeddings)` ingests and answers, returning the answers |
| `LocalRagIT` (test) | `@Tag("integration")`; builds both models directly with builders and `TestModels.require(...)`, calls `LocalRagApplication.run`, asserts corpus facts appear in the answers |
| `JinferChatModel` <<external>> | Spring AI `ChatModel` over a jinfer `ChatEngine` (`jinfer-spring-ai`) |
| `JinferEmbeddingModel` <<external>> | Spring AI `EmbeddingModel` over a jinfer `LoadedEmbedder` (`jinfer-spring-ai`) |
| `JinferChatAutoConfiguration` / `JinferEmbeddingAutoConfiguration` <<external>> | Create the two model beans from `spring.ai.jinfer.*` properties |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph app["jinfer-example-local-rag"]
        LRA["LocalRagApplication"]
    end
    subgraph springai["Spring AI (external)"]
        CC["ChatClient"]
        RAA["RetrievalAugmentationAdvisor"]
        VSDR["VectorStoreDocumentRetriever"]
        SVS["SimpleVectorStore"]
        TTS["TokenTextSplitter"]
    end
    subgraph adapter["jinfer-spring-ai + autoconfigure"]
        JCM["JinferChatModel"]
        JEM["JinferEmbeddingModel"]
        AC["JinferChat/EmbeddingAutoConfiguration"]
    end
    subgraph chat["jinfer-chat"]
        CE["ChatEngine"]
        M["Models (ServiceLoader)"]
        LE["LoadedEmbedder"]
        LM["LoadedModel"]
    end
    subgraph ports["model ports"]
        L2["jinfer-lfm2: Lfm2Provider"]
        Q3["jinfer-qwen3: Qwen3Provider"]
    end
    JV["jam-vector: VectorJAMProvider (runtime)"]

    LRA --> CC
    LRA --> SVS
    LRA --> TTS
    CC --> RAA --> VSDR --> SVS
    CC --> JCM
    SVS --> JEM
    AC -.->|creates beans| JCM
    AC -.->|creates beans| JEM
    JCM --> CE --> M
    JEM --> M
    M -->|loadEmbedder| LE
    M -->|load| LM
    M --> L2
    M --> Q3
    L2 -.-> JV
    Q3 -.-> JV
```

The default chat model in the README is LFM2.5 (`LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0`, architecture `lfm2`/`lfm2moe`
served by `Lfm2Provider`) and the embedder is `Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` (architecture `qwen3`, served by
`Qwen3Provider`). That is why exactly these two ports are on the classpath. Any GGUF whose architecture one of them
supports would work; others need their port added to `pom.xml`.

---

## 2. Module dependencies

```mermaid
flowchart LR
    APP["jinfer-example-local-rag"]
    STARTER["jinfer-spring-ai-spring-boot-starter"]
    SAI["jinfer-spring-ai"]
    AUTO["jinfer-spring-ai-autoconfigure"]
    LFM2["jinfer-lfm2"]
    QW3["jinfer-qwen3"]
    JAMV["jam-vector (runtime)"]
    VS["spring-ai-vector-store"]
    RAG["spring-ai-rag"]
    TK["jinfer-testkit (test)"]
    CLIENT["spring-ai-client-chat + chat-client autoconfig"]

    APP --> STARTER
    APP --> LFM2
    APP --> QW3
    APP --> JAMV
    APP --> VS
    APP --> RAG
    APP --> TK
    STARTER --> SAI
    STARTER --> AUTO
    STARTER --> CLIENT
```

- Spring AI and Spring Boot versions are imported as BOMs (`spring-ai-bom`, `spring-boot-dependencies`).
- `spring-boot-maven-plugin` gets its version and the jinfer JVM flags (`--add-modules jdk.incubator.vector,...
  --enable-native-access=ALL-UNNAMED`) from the parent `jinfer/pom.xml`, so `mvn spring-boot:run` starts with them.
- `jam-vector` is runtime-only: it registers `com.qxotic.jam.vector.VectorJAMProvider` as a `JAM$Provider` service,
  the Java Vector API backend for the kernels.

---

## 3. Class diagram

```mermaid
classDiagram
    direction LR
    class LocalRagApplication {
        <<SpringBootApplication>>
        ~List CORPUS$
        ~List QUESTIONS$
        +main(String[] args)$
        ~demo(JinferChatModel, JinferEmbeddingModel) CommandLineRunner
        ~run(JinferChatModel, JinferEmbeddingModel)$ List
    }
    class LocalRagIT {
        <<test>>
        ~answersAreGroundedInTheCorpus()
    }
    class CommandLineRunner { <<interface, external>> }
    class ChatModel { <<interface, external>> }
    class EmbeddingModel { <<interface, external>> }
    class VectorStore { <<interface, external>> }
    class SimpleVectorStore { <<external>> }
    class TokenTextSplitter { <<external>> }
    class ChatClient { <<interface, external>> }
    class RetrievalAugmentationAdvisor { <<external>> }
    class VectorStoreDocumentRetriever { <<external>> }
    class Document { <<external>> }
    class JinferChatModel {
        <<external>>
        +call(Prompt) ChatResponse
        +stream(Prompt) Flux
        +close()
    }
    class JinferEmbeddingModel {
        <<external>>
        +embed(String) float[]
        +embed(List, EmbeddingOptions, BatchingStrategy) List
        +getEmbeddingContent(Document) String
        +call(EmbeddingRequest) EmbeddingResponse
        +close()
    }
    class ChatEngine { <<external>> }
    class LoadedEmbedder~S~ { <<record, external>> }

    JinferChatModel ..|> ChatModel
    JinferEmbeddingModel ..|> EmbeddingModel
    SimpleVectorStore ..|> VectorStore
    LocalRagApplication ..> CommandLineRunner : demo() returns lambda
    LocalRagApplication --> Document : CORPUS
    LocalRagApplication ..> SimpleVectorStore : builder(embeddings)
    LocalRagApplication ..> TokenTextSplitter : split(CORPUS)
    LocalRagApplication ..> ChatClient : builder(chat)
    LocalRagApplication ..> RetrievalAugmentationAdvisor : builder()
    LocalRagApplication ..> VectorStoreDocumentRetriever : builder() topK 2
    SimpleVectorStore --> JinferEmbeddingModel
    VectorStoreDocumentRetriever --> SimpleVectorStore
    RetrievalAugmentationAdvisor --> VectorStoreDocumentRetriever
    ChatClient --> RetrievalAugmentationAdvisor : default advisor
    ChatClient --> JinferChatModel
    JinferChatModel --> ChatEngine : engine
    JinferEmbeddingModel --> LoadedEmbedder : loaded
    LocalRagIT ..> LocalRagApplication : run()
    LocalRagIT ..> JinferChatModel : builder()
    LocalRagIT ..> JinferEmbeddingModel : builder()
```

`run` is `static` and takes the two models as parameters, so the same code runs both as the Boot `CommandLineRunner`
(beans injected) and from `LocalRagIT` (models built by hand, no Spring context). The application never names
`ChatClient.Builder` as a bean; it calls `ChatClient.builder(chat)` directly.

---

## 4. Bean wiring (auto-configuration)

The starter's `jinfer-spring-ai-autoconfigure` lists its auto-configurations in
`META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports` (Chat, Embedding, Decision, Rerank,
Speech, Transcription, Session). Two of them produce the beans this app injects:

```mermaid
flowchart TB
    P["application.properties"]
    SEL{"JinferChatSelected:<br/>spring.ai.model.chat unset<br/>and spring.ai.jinfer.chat.model set"}
    EMB{"spring.ai.model.embedding=jinfer"}
    CAC["JinferChatAutoConfiguration<br/>@ConditionalOnMissingBean"]
    EAC["JinferEmbeddingAutoConfiguration<br/>@ConditionalOnMissingBean EmbeddingModel"]
    CP["JinferChatProperties<br/>spring.ai.jinfer.chat.*"]
    EP["JinferEmbeddingProperties<br/>spring.ai.jinfer.embedding.*"]
    SRC["Sources.model: ref or Path"]
    JCM["JinferChatModel bean"]
    JEM["JinferEmbeddingModel bean"]
    DEMO["LocalRagApplication.demo(chat, embeddings)"]

    P --> SEL --> CAC
    P --> EMB --> EAC
    CAC --> CP
    EAC --> EP
    CAC --> SRC
    EAC --> SRC
    CAC -->|"builder().contextCapacity.options.build()"| JCM
    EAC -->|"builder().contextCapacity.dimensions.build()"| JEM
    JCM --> DEMO
    JEM --> DEMO
```

- Chat is enabled by `JinferChatSelected` (a `Condition`): if `spring.ai.model.chat` is set it must equal `jinfer`,
  otherwise `spring.ai.jinfer.chat.model` must have text. This app relies on the second branch.
- Embedding requires explicit selection, so the app sets `spring.ai.model.embedding=jinfer`.
- `Sources.model` rejects `://` URLs that are not valid refs, then sends a model ref (`ModelStore.isRef`) to
  `builder.model(String)` and anything else to `builder.modelPath(Path)`.
- `JinferChatProperties.toOptions()` maps the flat properties (here `max-tokens=256`) onto `JinferChatOptions`.

---

## 5. Model loading and SPI discovery

```mermaid
sequenceDiagram
    participant AC as Jinfer*AutoConfiguration
    participant EB as JinferEmbeddingModel.Builder
    participant CB as JinferChatModel.Builder
    participant MS as ModelStore
    participant CE as ChatEngine
    participant M as Models
    participant SL as ServiceLoader
    participant P as Lfm2Provider / Qwen3Provider

    Note over M,SL: class init: PROVIDERS = discover()
    M->>SL: load(ModelProvider.class)
    SL-->>M: Lfm2Provider, Qwen3Provider (META-INF/services)

    AC->>EB: build()
    EB->>MS: standard().resolveAll(refs) when model is a ref
    MS-->>EB: local Path(s)
    EB->>M: loadEmbedder(path, arena, companions, null)
    M->>P: dispatch on general.architecture (qwen3)
    P-->>M: LoadedEmbedder
    M-->>EB: LoadedEmbedder (state allocated in private arena)

    AC->>CB: build()
    CB->>CE: new ChatEngine(modelPath, companions, cacheOptions)
    CE->>M: load(modelPath, weights[, companions])
    M->>P: dispatch on general.architecture (lfm2 / lfm2moe)
    P-->>M: LoadedModel
```

`jinfer-lfm2` registers `com.qxotic.jinfer.models.lfm2.Lfm2Provider` (architectures `lfm2`, `lfm2moe`) and
`jinfer-qwen3` registers `com.qxotic.jinfer.models.qwen3.Qwen3Provider` (architecture `qwen3`, retrieval family only)
in `META-INF/services/com.qxotic.jinfer.chat.ModelProvider`. This module registers no services of its own.

---

## 6. Run flow

### 6a. Ingestion

```mermaid
sequenceDiagram
    participant R as LocalRagApplication.run
    participant S as SimpleVectorStore
    participant T as TokenTextSplitter
    participant E as JinferEmbeddingModel
    participant L as LoadedEmbedder

    R->>S: SimpleVectorStore.builder(embeddings).build()
    R->>T: new TokenTextSplitter().split(CORPUS)
    T-->>R: chunks (List of Document)
    R->>S: add(chunks)
    S->>E: embed(documents, options, batchingStrategy)
    loop per batch
        E->>E: getEmbeddingContent(doc) = documentPrefix + text
        E->>E: call(EmbeddingRequest) observed
        E->>L: embedAll(state, inputs, dim, consumer) under lock
    end
    E-->>S: float[] per document
    R->>R: print ingested N documents in Xs
```

### 6b. Question answering

```mermaid
sequenceDiagram
    participant R as LocalRagApplication.run
    participant C as ChatClient
    participant A as RetrievalAugmentationAdvisor
    participant V as VectorStoreDocumentRetriever
    participant S as SimpleVectorStore
    participant E as JinferEmbeddingModel
    participant M as JinferChatModel
    participant G as ChatEngine

    loop each of QUESTIONS
        R->>C: prompt(question).call().content()
        C->>A: advise request
        A->>V: retrieve(query)
        V->>S: similaritySearch(topK = 2)
        S->>E: embed(String) = queryPrefix + text
        E-->>S: query vector
        S-->>V: 2 nearest Documents
        A-->>C: request augmented with context
        C->>M: call(Prompt)
        M->>G: complete(prepared, ReplySink.NONE)
        G-->>M: Completion
        M-->>C: ChatResponse
        C-->>R: answer text
        R->>R: collect + print Q / A
    end
```

The query and document sides frame text differently: `JinferEmbeddingModel.embed(String)` prepends the embedder's
`queryPrefix`, while `getEmbeddingContent(Document)` prepends its `documentPrefix` (both fields of `LoadedEmbedder`,
supplied by the model port).

---

## 7. Configuration

`src/main/resources/application.properties`:

| Property | Value | Effect |
|---|---|---|
| `spring.ai.jinfer.chat.model` | `${JINFER_CHAT_MODEL:models/chat.gguf}` | Chat GGUF: a hub ref or local path; also enables chat via `JinferChatSelected` |
| `spring.ai.jinfer.chat.context-capacity` | `4096` | Context bound passed to `JinferChatModel.Builder.contextCapacity` |
| `spring.ai.jinfer.chat.max-tokens` | `256` | Default `JinferChatOptions.maxTokens` |
| `spring.ai.model.embedding` | `jinfer` | Activates `JinferEmbeddingAutoConfiguration` |
| `spring.ai.jinfer.embedding.model` | `${JINFER_EMBEDDING_MODEL:models/embeddings.gguf}` | Embedding GGUF: ref or path |

---

## 8. Integration test

```mermaid
flowchart LR
    TM["TestModels.require(hf.co/... refs)"] --> EB["JinferEmbeddingModel.builder()<br/>contextCapacity 2048"]
    TM --> CB["JinferChatModel.builder()<br/>contextCapacity 4096, maxTokens 256"]
    EB --> RUN["LocalRagApplication.run"]
    CB --> RUN
    RUN --> AS["assert: store credit / next business day or monday / two-year"]
```

`LocalRagIT` uses try-with-resources on both models (both are `AutoCloseable`), so weights and states are freed after
the test. It runs only in the `integration` group and only when `TestModels` finds the models.
