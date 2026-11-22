# jinfer-example-judge-advisor — class architecture

Package `com.qxotic.jinfer.example.judgeadvisor` (3 main classes + 1 nested record + 1 nested builder, ~325 lines;
3 test classes). A **Spring Boot demo**, not a library and not published to Maven Central: a remote OpenAI-compatible
generator (Kimi, via `spring-ai-starter-model-openai`) answers a question, and a **local in-JVM jinfer judge**
(`JinferChatModel` over a GGUF) grades each answer. A Spring AI `CallAdvisor` loops generation until the judge's
verdict passes. The verdict is *grammar-pinned*: its JSON schema is compiled to GBNF and enforced at sampling time,
so it always parses and its `rating` is always one of 1–4.

Format: **Mermaid**. Legend: `..|>` implements; `-->` holds/uses; `..>` creates/depends; `*--` nested/composed;
classes marked `<<external>>` live outside this module.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Module dependencies](#2-module-dependencies)
3. [Class diagrams](#3-class-diagrams) — 3a module classes, 3b external collaborators
4. [Bean wiring and startup](#4-bean-wiring-and-startup)
5. [The self-refine loop](#5-the-self-refine-loop) — 5a flowchart, 5b sequence
6. [Verdict decoding and grammar pinning](#6-verdict-decoding-and-grammar-pinning)
7. [Tests](#7-tests)

| Class | Role |
|---|---|
| `JudgeAdvisorApplication` | `@SpringBootApplication`; holds the judge `RUBRIC` and `JUDGE_OPTIONS`; a `CommandLineRunner` bean calls `run(generator, judgeBase)`, which builds the cached-prompt judge and the `ChatClient` and asks one question |
| `GrammarPinnedEvaluationAdvisor` | Spring AI `CallAdvisor`: the self-refine loop — call the chain, judge the text answer, append feedback to the user message and retry up to `maxRepeatAttempts` |
| `GrammarPinnedEvaluationAdvisor.Verdict` | record `(rating, evaluation, feedback)` parsed from the judge's JSON |
| `GrammarPinnedEvaluationAdvisor.Builder` | fluent builder: `judge(ChatModel, ChatOptions)` (required), `maxRepeatAttempts` (default 3), `successRating` (default 3) |
| `WeatherTools` | `@Tool weather(location)`: a deterministic cycling queue of readings `-255`, `15` so the first answer is physically impossible and must fail |

Resources: `src/main/resources/application.properties` disables the auto-configured `ChatClient`
(`spring.ai.chat.client.enabled=false`, since two `ChatModel` beans exist) and configures both models
(`spring.ai.jinfer.chat.*` for the judge, `spring.ai.openai.*` for Kimi). The module declares no
`META-INF/services` entries of its own; it relies on the SPI and auto-configuration registrations of its dependencies
(see §2).

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    subgraph app["jinfer-example-judge-advisor"]
        App["JudgeAdvisorApplication"]
        Adv["GrammarPinnedEvaluationAdvisor"]
        WT["WeatherTools"]
    end
    subgraph springai["Spring AI (external)"]
        CC["ChatClient + CallAdvisorChain"]
        OAI["OpenAiChatModel"]
    end
    subgraph jsa["jinfer-spring-ai / -autoconfigure"]
        AC["JinferChatAutoConfiguration"]
        JCM["JinferChatModel"]
        JCO["JinferChatOptions"]
    end
    subgraph chat["jinfer-chat"]
        CE["ChatEngine"]
        G["Grammar (schemaGbnf)"]
        MP["ModelProvider SPI"]
    end
    LFM["jinfer-lfm2: Lfm2Provider"]
    Kimi["Kimi OpenAI-compatible endpoint"]
    JSON["json: Json.parseMap"]

    App --> CC
    CC --> Adv
    CC --> OAI --> Kimi
    CC --> WT
    Adv -->|"judge.call"| JCM
    Adv --> JSON
    AC -->|"@Bean"| JCM
    JCM --> CE
    JCM --> G
    CE --> MP
    LFM -.->|"implements"| MP
```

The generator runs remotely; only the judge runs in-process. In the offline integration test, one `JinferChatModel`
plays both roles (§7).

---

## 2. Module dependencies

From `pom.xml` (versions managed by `jinfer-parent`; Jackson, Spring AI and Spring Boot BOMs imported):

| Dependency | Scope | Why |
|---|---|---|
| `jinfer-spring-ai-spring-boot-starter` | compile | brings `jinfer-spring-ai` (`JinferChatModel`, `JinferChatOptions`) and `jinfer-spring-ai-autoconfigure` (`JinferChatAutoConfiguration`, registered in `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`) plus Spring AI chat-client/tool auto-configuration |
| `jinfer-lfm2` | compile | one architecture port on the runtime classpath for the default judge (LFM2.5); registers `Lfm2Provider` under `META-INF/services/com.qxotic.jinfer.chat.ModelProvider` |
| `json` | compile | `com.qxotic.format.json.Json.parseMap` to read the verdict |
| `spring-ai-starter-model-openai` | compile | `OpenAiChatModel` bean for the remote generator |
| `spring-boot-starter-test` | test | JUnit 5, AssertJ, `ApplicationContextRunner` |
| `jinfer-testkit` | test | `TestModels.require(...)` model fixtures (gates model-backed tests) |

Build: `spring-boot-maven-plugin` (run with `KIMI_API_KEY=... mvn spring-boot:run`; jinfer JVM flags managed by the
parent).

---

## 3. Class diagrams

### 3a. Module classes

```mermaid
classDiagram
    direction LR
    class CallAdvisor { <<interface, external>> }
    class ChatModel { <<interface, external>> }
    class ChatOptions { <<interface, external>> }

    class JudgeAdvisorApplication {
        <<SpringBootApplication>>
        ~String RUBRIC$
        -JinferChatOptions JUDGE_OPTIONS$
        +main(String[] args)$
        ~demo(OpenAiChatModel, JinferChatModel) CommandLineRunner
        ~run(ChatModel generator, JinferChatModel judgeBase)$ String
    }
    class GrammarPinnedEvaluationAdvisor {
        ~String VERDICT_SCHEMA$
        -ChatModel judge
        -ChatOptions judgeOptions
        -int maxRepeatAttempts
        -int successRating
        +adviseCall(ChatClientRequest, CallAdvisorChain) ChatClientResponse
        -verdictOf(ChatResponse) Verdict
        -transcript(...)$ String
        -preview(String)$ String
        -questionOf(Prompt)$ String
        -addFeedback(ChatClientRequest, Verdict)$ ChatClientRequest
        +getName() String
        +getOrder() int
        +builder()$ Builder
    }
    class Verdict {
        <<record>>
        +int rating
        +String evaluation
        +String feedback
    }
    class Builder {
        -ChatModel judge
        -ChatOptions judgeOptions
        -int maxRepeatAttempts = 3
        -int successRating = 3
        +judge(ChatModel, ChatOptions) Builder
        +maxRepeatAttempts(int) Builder
        +successRating(int) Builder
        +build() GrammarPinnedEvaluationAdvisor
    }
    class WeatherTools {
        -Queue~Integer~ readings
        +weather(String location) String
    }

    GrammarPinnedEvaluationAdvisor ..|> CallAdvisor
    GrammarPinnedEvaluationAdvisor *-- Verdict
    GrammarPinnedEvaluationAdvisor *-- Builder
    Builder ..> GrammarPinnedEvaluationAdvisor : creates
    GrammarPinnedEvaluationAdvisor --> ChatModel : judge
    GrammarPinnedEvaluationAdvisor --> ChatOptions : judgeOptions
    GrammarPinnedEvaluationAdvisor ..> Verdict : creates
    JudgeAdvisorApplication ..> GrammarPinnedEvaluationAdvisor : builds
    JudgeAdvisorApplication ..> WeatherTools : creates
```

`getOrder()` returns `0`; `getName()` returns `"GrammarPinnedEvaluationAdvisor"`. `build()` throws
`IllegalArgumentException` when `judge` or `judgeOptions` is missing.

### 3b. External collaborators

```mermaid
classDiagram
    direction LR
    class ChatModel { <<interface, external>> }
    class AutoCloseable { <<interface, external>> }
    class DefaultToolCallingChatOptions { <<external>> }
    class JinferChatModel {
        <<external>>
        ~ChatEngine engine
        +withCachedPrompt(List prefixMessages, List tools) JinferChatModel
        +call(Prompt) ChatResponse
        ~grammar(JinferChatOptions)$ String
    }
    class JinferChatOptions {
        <<external>>
        +getOutputSchema() String
        +getGrammar() String
    }
    class ChatEngine {
        <<external>>
        +definePrompt(Conversation)
        +prepare(Request) Prepared
    }
    class Grammar {
        <<external>>
        +schemaGbnf(Map)$ String
    }
    class OpenAiChatModel { <<external>> }
    class ChatClient { <<interface, external>> }
    class CallAdvisorChain {
        <<interface, external>>
        +nextCall(ChatClientRequest) ChatClientResponse
        +copy(CallAdvisor) CallAdvisorChain
    }
    class Json {
        <<external>>
        +parseMap(String)$ Map
    }
    class JudgeAdvisorApplication
    class GrammarPinnedEvaluationAdvisor

    JinferChatModel ..|> ChatModel
    JinferChatModel ..|> AutoCloseable
    OpenAiChatModel ..|> ChatModel
    JinferChatOptions --|> DefaultToolCallingChatOptions
    JinferChatModel --> ChatEngine
    JinferChatModel ..> Grammar : schema to GBNF
    JinferChatModel ..> JinferChatOptions : reads
    JudgeAdvisorApplication ..> JinferChatModel : withCachedPrompt
    JudgeAdvisorApplication ..> JinferChatOptions : JUDGE_OPTIONS
    JudgeAdvisorApplication ..> ChatClient : builds
    JudgeAdvisorApplication ..> OpenAiChatModel : generator bean
    GrammarPinnedEvaluationAdvisor --> CallAdvisorChain : copy(this).nextCall
    GrammarPinnedEvaluationAdvisor ..> Json : parseMap
```

---

## 4. Bean wiring and startup

```mermaid
sequenceDiagram
    autonumber
    participant Boot as SpringApplication
    participant JAC as JinferChatAutoConfiguration
    participant OAC as OpenAI auto-config
    participant App as JudgeAdvisorApplication
    participant Base as JinferChatModel base
    participant Judge as JinferChatModel view
    participant CC as ChatClient

    Boot->>JAC: spring.ai.jinfer.chat.model set
    JAC->>Base: JinferChatModel.builder()...build()
    Boot->>OAC: spring.ai.openai.* set
    OAC-->>Boot: OpenAiChatModel bean
    Boot->>App: demo(generator, judgeBase)
    App-->>Boot: CommandLineRunner
    Boot->>App: runner.run -> run(generator, judgeBase)
    App->>Base: withCachedPrompt([SystemMessage RUBRIC], [])
    Base->>Base: engine.definePrompt(prefix) - rubric prefilled once
    Base-->>App: judge view
    App->>CC: ChatClient.builder(generator).defaultTools(new WeatherTools()).defaultAdvisors(advisor).build()
    App->>CC: prompt("What is the weather in Paris?").call().content()
    CC-->>App: final answer
    App->>App: print ">>> FINAL: ..."
```

- `JUDGE_OPTIONS`: `temperature(0.0)`, `maxTokens(384)`, `outputSchema(VERDICT_SCHEMA)`.
- The rubric carries no format instructions: structure is enforced by the grammar alone (the spring-ai adapter never
  restates the schema in the prompt).
- `run(...)` is static and takes a plain `ChatModel` generator, so tests can pass any model (including the same
  `JinferChatModel` as both roles).

---

## 5. The self-refine loop

### 5a. `adviseCall` control flow

```mermaid
flowchart TD
    S(["adviseCall(request, chain)"]) --> L["attempt = 1 .. maxRepeatAttempts+1"]
    L --> N["response = chain.copy(this).nextCall(current)"]
    N --> T{"chatResponse null or hasToolCalls?"}
    T -->|yes| R1(["return response - never judged"])
    T -->|no| J["judge.call(Prompt('Question: ...\nAnswer: ...', judgeOptions))"]
    J --> V["verdict = verdictOf(judgeResponse)"]
    V --> P{"1 <= rating <= 4 and rating >= successRating?"}
    P -->|"pass"| R2(["return response"])
    P -->|"fail"| M{"attempt > maxRepeatAttempts?"}
    M -->|yes| R3(["print 'max attempts reached', return last response"])
    M -->|no| F["current = addFeedback(current, verdict)"]
    F --> L
```

- `questionOf(prompt)` joins the SYSTEM and USER messages of the current prompt as `TYPE: text` lines.
- `addFeedback` uses `Prompt.augmentUserMessage` to append "The previous answer failed evaluation with feedback: …
  Please answer again, addressing the feedback." and rebuilds the request with `request.mutate().prompt(...)`.
- Each attempt prints one transcript line: pass/fail, rating, judge completion tokens, `cacheRead` (from
  `Usage.getCacheReadInputTokens()`, when non-null), `decode` (metadata key `eval-duration`), feedback on failure, and an
  80-char preview of the answer.

### 5b. One run with the weather gimmick

```mermaid
sequenceDiagram
    autonumber
    participant CC as ChatClient
    participant Adv as GrammarPinnedEvaluationAdvisor
    participant Chain as CallAdvisorChain
    participant Gen as Generator - Kimi
    participant WT as WeatherTools
    participant Judge as JinferChatModel view

    CC->>Adv: adviseCall(request, chain)
    Adv->>Chain: copy(this).nextCall(request)
    Chain->>Gen: call
    Gen->>WT: weather("Paris") -> -255C
    Gen-->>Chain: final text answer
    Chain-->>Adv: response
    Adv->>Judge: call(Question+Answer, JUDGE_OPTIONS)
    Judge-->>Adv: grammar-pinned JSON, rating 1
    Adv->>Adv: fail -> addFeedback
    Adv->>Chain: copy(this).nextCall(augmented)
    Chain->>Gen: call
    Gen->>WT: weather("Paris") -> 15C
    Gen-->>Chain: final text answer
    Chain-->>Adv: response
    Adv->>Judge: call(Question+Answer, JUDGE_OPTIONS)
    Judge-->>Adv: rating 4
    Adv-->>CC: response (passed)
```

`WeatherTools` keeps an `ArrayDeque` of `[-255, 15]`; each call polls the head and re-appends it, so readings
alternate forever.

---

## 6. Verdict decoding and grammar pinning

```mermaid
flowchart LR
    VS["VERDICT_SCHEMA - rating enum 1..4, evaluation, feedback"] --> JO["JinferChatOptions.outputSchema"]
    JO --> GR["JinferChatModel.grammar(options)"]
    GR --> SG["Grammar.schemaGbnf(jsonMap(schema))"]
    SG --> ENG["ChatEngine.Request grammar - logits masked at sampling"]
    ENG --> RESP["ChatResponse text + finishReason"]
    RESP --> FR{"finishReason == length?"}
    FR -->|yes| TRUNC["Verdict(1, 'judge verdict truncated...', 'Be more concise.')"]
    FR -->|no| PARSE["Json.parseMap(text) -> Verdict"]
```

- `VERDICT_SCHEMA` is hand-written (not record-generated) so `rating` is pinned by `enum [1,2,3,4]`; all three fields are
  `required`.
- The grammar guarantees no invalid token, not completion within the budget: a `length` finish yields truncated JSON,
  which `verdictOf` degrades to a failed verdict instead of throwing.
- The range check in `adviseCall` (`1 <= rating <= 4`) is defensive; under the grammar an out-of-range rating cannot be
  produced.
- `JinferChatModel.grammar` refuses `outputSchema` and `grammar` set together.

---

## 7. Tests

| Test | Kind | What it covers |
|---|---|---|
| `GrammarPinnedEvaluationAdvisorTest` | unit (stubs `StubJudge implements ChatModel`, `StubChain implements CallAdvisorChain`) | pass on first good verdict; failed verdict retries with feedback in the user message; max attempts returns last response; tool-call responses skip evaluation; out-of-range rating never passes; truncated verdict degrades to a failed evaluation |
| `JudgeAdvisorOfflineIT` | `@Tag("integration")`, model-gated via `TestModels` | LFM2.5-8B-A1B Q8_0 plays both generator and judge: `JudgeAdvisorApplication.run(base, base)`; assumes the final answer contains `15` |
| `JudgeAdvisorKimiIT` | `@Tag("integration")`, gated on `KIMI_API_KEY` and model | `ApplicationContextRunner` with `JinferChatAutoConfiguration`, `ToolCallingAutoConfiguration`, `OpenAiChatAutoConfiguration`; runs the full hybrid loop |
