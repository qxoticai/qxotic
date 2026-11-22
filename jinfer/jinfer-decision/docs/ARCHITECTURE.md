# jinfer-decision — class architecture

Package `com.qxotic.jinfer.decision` (4 top-level types, ~440 lines). It is a thin, **typed** layer over the
`DecisionModel` interface of `jinfer-chat`: predicate (yes/no), choice and rating questions about an input, each
answered with the model's probabilities over the question's outcomes — **no text is generated**. It owns no model,
no kernel and no state; every computation is delegated to a `DecisionModel` loaded by `Models.loadDecisionModel`
(implemented today by `Lfm3Decision` in `jinfer-lfm3`, Liquid d1 / d1-Omni, and `Qwen35Decision` in `jinfer-qwen35`,
vLLM's d3). Compile dependencies: `jinfer-core`, `jinfer-chat` only; no `META-INF/services` registration (the module
is not a `ModelProvider`). JPMS automatic module name: `com.qxotic.jinfer.decision`.

Format: **Mermaid**. Legend: `..|>` implements; `--|>` extends; `-->` holds/uses; `..>` creates/depends;
classes marked `<<external>>` live outside this package.

1. [Position in the stack](#1-position-in-the-jinfer-stack)
2. [Class diagrams](#2-class-diagrams) — 2a public API, 2b mapping to `DecisionModel`
3. [Question factories](#3-question-factories)
4. [Call flows](#4-call-flows) — 4a one question, 4b several questions
5. [Answer derivations](#5-answer-derivations)
6. [Invariants and validation](#6-invariants-and-validation)

| Class | Role |
|---|---|
| `Decisions` | Entry point (final, immutable, thread-safe): wraps a `DecisionModel`; `ask` (one or many questions) and `askIndependently`; lowers typed questions to `DecisionModel.Question` |
| `Question<A extends Answer>` | Sealed interface of typed questions with static factories `predicate` / `choice` / `rating`; nested records `Predicate`, `Choice<E>`, `Rating<E>` |
| `Answer` | Sealed interface: `probabilities()`, default `selectedIndex()` / `confidence()`; nested records `Predicate`, `Choice<E>`, `Rating<E>` |
| `Answers` | Final, immutable map from question to typed answer for multi-question calls; `get(Question<A>) : A`; package-private `answer(question, float[])` builds one typed answer |

---

## 1. Position in the jinfer stack

```mermaid
flowchart TB
    App["Application code / examples/scripts/Triage.java"]
    subgraph dec["jinfer-decision"]
        D["Decisions"]
        Q["Question: Predicate / Choice / Rating"]
        A["Answer / Answers"]
    end
    subgraph chat["jinfer-chat"]
        Models["Models.loadDecisionModel"]
        MP["ModelProvider.loadDecisionModel"]
        DM["DecisionModel<br/>Question / Predicate / Choice / Rating / Option"]
        Content["Content: Text / Media"]
    end
    subgraph core["jinfer-core"]
        CS["ContextState"]
    end
    subgraph impls["model families"]
        L["Lfm3Decision (jinfer-lfm3, d1 / d1-Omni)"]
        Q35["Qwen35Decision (jinfer-qwen35, d3)"]
    end

    App --> Models --> MP
    MP -.->|creates| L
    MP -.->|creates| Q35
    L -.->|implements| DM
    Q35 -.->|implements| DM
    App --> D
    D --> DM
    D --> Content
    D --> CS
    D --> Q
    D --> A
```

Other integrations (`jinfer-langchain4j`, `jinfer-spring-ai`, `jinfer-server`, `jinfer-cli`) talk to
`DecisionModel` directly; none of them depends on this module. It is listed in `jinfer-bom`.

---

## 2. Class diagrams

### 2a. Public API

```mermaid
classDiagram
    direction LR
    class Decisions {
        <<final>>
        -DecisionModel model
        +of(DecisionModel) Decisions$
        +ask(ContextState, String, Question~A~) A
        +ask(ContextState, List~Content~, Question~A~) A
        +ask(ContextState, String, Question...) Answers
        +ask(ContextState, List~Content~, Question...) Answers
        +askIndependently(ContextState, String, Question...) Answers
        +askIndependently(ContextState, List~Content~, Question...) Answers
        -text(String) List~Content~$
        -require(ContextState, List~Content~)$
        -lower(Question[]) List$
        -lower(Question) DecisionModel.Question$
    }
    class Question~A~ {
        <<sealed interface>>
        +text() String
        +predicate(text)$
        +predicate(text, yesMeans, noMeans)$
        +choice(text, Class)$
        +choice(text, String...)$
        +choice(text, SequencedMap)$
        +choice(text, List, Function)$
        +rating(text, String...)$
        +rating(text, Class)$
        +rating(text, SequencedMap)$
    }
    class QPredicate["Question.Predicate"] {
        <<record>>
        text, yesMeans, noMeans
    }
    class QChoice["Question.Choice"] {
        <<record>>
        text, List options, List~Option~ labels
    }
    class QRating["Question.Rating"] {
        <<record>>
        text, List levels, List~Option~ labels
    }
    class Answer {
        <<sealed interface>>
        +probabilities() List~Float~
        +selectedIndex() int
        +confidence() float
    }
    class APredicate["Answer.Predicate"] {
        <<record>>
        float yes
        +no() float
        +isYes() boolean
    }
    class AChoice["Answer.Choice"] {
        <<record>>
        options, probabilities
        +selectedOption() E
    }
    class ARating["Answer.Rating"] {
        <<record>>
        levels, probabilities
        +selectedLevel() E
        +medianLevel() E
        +mean() float
        +medianIndex() int
    }
    class Answers {
        <<final>>
        -Map answers
        ~Answers(Question[], float[][])
        +get(Question~A~) A
        ~answer(Question~A~, float[]) A$
    }
    class DecisionModel { <<interface, external>> }
    class ContextState { <<abstract, external>> }
    class Option["DecisionModel.Option"] { <<record, external>> }

    Question <|.. QPredicate
    Question <|.. QChoice
    Question <|.. QRating
    Answer <|.. APredicate
    Answer <|.. AChoice
    Answer <|.. ARating
    QPredicate ..> APredicate : answered by
    QChoice ..> AChoice : answered by
    QRating ..> ARating : answered by
    QChoice --> Option : labels
    QRating --> Option : labels
    Decisions --> DecisionModel : model
    Decisions ..> ContextState : caller's state
    Decisions ..> Answers : creates
    Decisions ..> Answers : answer()
    Answers *-- Answer : by question
```

The type parameter of `Question<A>` ties each question kind to its answer kind: `Question.Predicate implements
Question<Answer.Predicate>`, `Question.Choice<E> implements Question<Answer.Choice<E>>`, `Question.Rating<E>
implements Question<Answer.Rating<E>>`. So `decisions.ask(state, input, Question.choice("…", Team.class))` returns an
`Answer.Choice<Team>` and `selectedOption()` is a `Team`, with no cast at the call site.

### 2b. Mapping to `DecisionModel` (lowering)

`Decisions.lower` turns a typed question into the untyped one the model takes. The caller's values (enum constants,
strings, arbitrary objects) stay in the typed question; only their `Option` labels reach the model.

```mermaid
classDiagram
    direction LR
    class QPredicate["Question.Predicate"] { <<record>> }
    class QChoice["Question.Choice"] { <<record>> }
    class QRating["Question.Rating"] { <<record>> }
    class DMQuestion["DecisionModel.Question"] {
        <<sealed interface, external>>
        text() String
        options() List~Option~
    }
    class DMPredicate["DecisionModel.Predicate"] {
        <<record, external>>
        options() = no, yes
    }
    class DMChoice["DecisionModel.Choice"] { <<record, external>> }
    class DMRating["DecisionModel.Rating"] {
        <<record, external>>
        options() = levels
    }
    class Option["DecisionModel.Option"] {
        <<record, external>>
        label, description
        of(String...)$
    }
    DMQuestion <|.. DMPredicate
    DMQuestion <|.. DMChoice
    DMQuestion <|.. DMRating
    QPredicate ..> DMPredicate : lower text, yesMeans, noMeans
    QChoice ..> DMChoice : lower text, labels
    QRating ..> DMRating : lower text, labels
    DMChoice --> Option
    DMRating --> Option
```

---

## 3. Question factories

All factories are static methods on `Question`. Choice and rating keep two parallel, equal-sized lists: the caller's
**values** (`options` / `levels`, type `E`) and the model's **labels** (`List<Option>`).

```mermaid
flowchart LR
    subgraph P["predicate"]
        P1["predicate(text)"] --> PR["Predicate(text, null, null)"]
        P2["predicate(text, yesMeans, noMeans)"] --> PR2["Predicate(text, yesMeans, noMeans)"]
    end
    subgraph C["choice → Choice of E"]
        C1["choice(text, Class of enum)"] -->|"values = enum constants<br/>labels = readable(name)"| CC["Choice"]
        C2["choice(text, String...)"] -->|"values = strings<br/>labels = Option.of(strings)"| CC
        C3["choice(text, SequencedMap)"] -->|"values = keys<br/>labels = Option(key, description)"| CC
        C4["choice(text, List, Function label)"] -->|"values = list<br/>labels = Option(label(v))"| CC
    end
    subgraph R["rating → Rating of E, lowest first"]
        R1["rating(text, String...)"] --> RR["Rating"]
        R2["rating(text, Class of enum)"] -->|"labels = readable(name)"| RR
        R3["rating(text, SequencedMap)"] -->|"Option(key, description)"| RR
    end
```

- `readable(Enum)`: `name().toLowerCase(Locale.ROOT).replace('_', ' ').strip()` — `TECH_SUPPORT` reads as
  "tech support"; the `strip()` keeps a leading/trailing underscore from becoming a stray space.
- Option order is part of the question: it is the order the model reads them in, and the order of the answer's
  probabilities. Enums use declaration order.

---

## 4. Call flows

### 4a. One question — `ask(state, input, question)`

```mermaid
sequenceDiagram
    participant C as caller
    participant D as Decisions
    participant M as DecisionModel (external)
    participant As as Answers

    C->>D: ask(state, "ticket text", Question.choice(..., Team.class))
    D->>D: text(input) → List.of(Content.Text)
    D->>D: require(state, input) — null checks only
    D->>D: lower(question) → DecisionModel.Choice(text, labels)
    D->>M: decide(state, input, loweredQuestion)
    M-->>D: float[] probabilities (option order, sum 1)
    D->>As: Answers.answer(question, probabilities)
    As-->>D: Answer.Choice of Team
    D-->>C: typed answer
```

### 4b. Several questions — `ask` vs `askIndependently`

```mermaid
sequenceDiagram
    participant C as caller
    participant D as Decisions
    participant M as DecisionModel (external)
    participant As as Answers

    C->>D: ask(state, input, refund, urgency)
    D->>D: require, then lower(questions[]) — null array or element refused
    alt ask — the model's own way
        D->>M: decide(state, input, List of lowered)
        Note over M: default: decideIndependently<br/>Lfm3Decision: packs up to GROUP=16 questions<br/>after one copy of the input — no isolation
    else askIndependently
        D->>M: decideIndependently(state, input, List of lowered)
        Note over M: row i equals decide(..., question i) alone
    end
    M-->>D: float[][] rows, one per question
    D->>As: new Answers(questions, rows)
    loop i over questions
        As->>As: put(questions[i], answer(questions[i], rows[i]))
    end
    D-->>C: Answers
    C->>As: get(refund) → Answer.Predicate
    C->>As: get(urgency) → Answer.Rating of Urgency
```

`Qwen35Decision` (d3) answers `decide(list)` and `decideIndependently` alike (one prompt per question).
`Decisions` itself adds no state, locking or caching: one call at a time per `ContextState`, one state per thread
over a shared model.

---

## 5. Answer derivations

`Answers.answer(question, float[])` is a pattern-matching switch over the sealed `Question`:

```mermaid
flowchart TB
    In["float[] p from the model,<br/>in the question's option order"] --> Sw{"question kind"}
    Sw -->|Predicate| AP["Answer.Predicate(p[1])<br/>options are no, yes → p[1] is yes"]
    Sw -->|Choice| AC["Answer.Choice(q.options(), list(p))"]
    Sw -->|Rating| AR["Answer.Rating(q.levels(), list(p))"]
    AP --> Der1["no() = 1 - yes<br/>isYes(): yes > 0.5<br/>probabilities() = [no, yes]"]
    AC --> Der2["selectedOption() = options[selectedIndex]"]
    AR --> Der3["selectedLevel(), medianLevel()<br/>mean() = Σ i·p[i]<br/>medianIndex(): first i with cumulative ≥ 0.5"]
    Der1 --> Common["Answer defaults:<br/>selectedIndex() = argmax, earlier wins a tie<br/>confidence() = p[selectedIndex]"]
    Der2 --> Common
    Der3 --> Common
```

- `isYes()` is strictly `yes > 0.5f`, so an even split is "no" — consistent with `selectedIndex()` choosing the
  earlier outcome (`no`) on a tie.
- `Rating.mean()` is a 0-based position on the scale; `medianIndex()` always names a level the model gave weight to,
  even when the mass is split between distant levels.
- The answer never re-normalises; calibration is the model's.

---

## 6. Invariants and validation

| Where | Check | Failure |
|---|---|---|
| `Decisions.of` | model non-null | `NullPointerException("model")` |
| `Decisions.require` | state, input non-null | `NullPointerException` |
| `Decisions.text` | input string non-null | `NullPointerException("input")` |
| `Decisions.lower(Question[])` | array and each element non-null | `NullPointerException` |
| `Question.*` compact ctors | text not null/blank | `IllegalArgumentException("blank text")` |
| `Question.Choice` / `Rating` | values and labels same size; ≥ 2 entries; lists copied (`List.copyOf`) | `IllegalArgumentException` |
| `DecisionModel.Option` (external) | label not blank | `IllegalArgumentException("blank label")` |
| `Answer.Choice` / `Rating` | outcomes and probabilities same size; lists copied | `IllegalArgumentException` |
| `Answers.get` | question was asked (by `equals`) | `IllegalArgumentException("this question was not asked")` |

- `Answers` is keyed by a `HashMap<Question<?>, Answer>`: since questions are records, an *equal* question finds the
  answer, and a question asked twice in one call holds the last of its answers.
- Everything else (input too long for the state, too many options, unsupported media part, empty input) is left to
  the model to refuse — "a caller's null is named here; anything else is the model's to refuse".
- All types are immutable and safe to share across threads.

Tests (`src/test`): `DecisionsTest`, `DecisionApiEdgeTest`, `DecisionsFailureTest` use a canned in-test
`DecisionModel`; `D1DecisionsIT` / `D1DecisionsRobustnessIT` run against LFM3 d1 and `D3DecisionsIT` against
Qwen3.5 d3 (test-scope dependencies `jinfer-lfm3`, `jinfer-qwen35`).
