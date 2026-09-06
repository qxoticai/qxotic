# Release notes

## 0.2.0

First release on Maven Central: `com.qxotic` artifacts for jota, jam, jinfer, toknroll, gguf, json and safetensors, with `jinfer-bom` managing the versions.

### Behaviour to know about

- **Thinking policy.** Every chat template states how its checkpoint reasons: `NONE`, `OPTIONAL` or `ALWAYS`.
  A model that always reasons, LFM2.5-8B-A1B and gpt-oss among them, refuses `thinking(false)` with a message naming the remedy instead of leaking its reasoning into the visible text.
  `reasoningBudget` caps the span on every model that has think markers, in the CLI, the server, langchain4j and Spring AI.
- **Structured output in langchain4j.** A JSON-schema response format is enforced by the grammar and, so that the model knows which fields exist, described in one line appended to the last user message.
  `describeSchema(false)` on the builder leaves the prompt untouched.
- **Reply scaffolding is guarded.** The family's reply language now masks control tokens wherever the language expects a specific one, so a model cannot derail its own tool-call header or channel scaffolding; free text stays free.
- **Batch embeddings.** A packed embedding group larger than the state's batch capacity is ingested in chunks; earlier builds failed the request.
- **Grammars with thinking off.** A completed grammar ends the turn cleanly on every family, and Qwen 3.5's thinking-off prefix no longer swallows a raw grammar.
- **Speech companions.** The pronunciation lexicon of Inflect models is the `lexicon` companion, attachable on both speech builders and as `spring.ai.jinfer.speech.companions.lexicon`.
- **Spring Boot examples.** `mvn spring-boot:run` runs with full tiered compilation; its default `optimizedLaunch` pinned C1 and slowed the Vector API about a hundredfold.
- **CLI errors.** A bad `--cache` file, a read-only cache root and other wrapped IO failures print one `ERROR` line.

- **Tools with constrained output.** A request may offer tools together with a JSON schema or a grammar: the family's reply language then offers a tool call or the document, so langchain4j's tool-round-then-structured-answer loop works in one service call.
  A forced tool call with constrained output is still refused, and a family without a combined language refuses at request time.
- **Stringified arguments.** A small model that sends an array or object argument as a JSON string, Llama 3.2 1B does, gets it unwrapped where the tool's schema declares that shape.

### Known limits

- The `Logic` gallery demo and the model-backed tests pin temperature 0 and a seed; small models still fail some puzzles, which the demo reports honestly.
