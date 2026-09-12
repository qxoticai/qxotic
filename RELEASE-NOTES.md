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
- **Mellum 2.** JetBrains Mellum 2 (`jinfer-mellum`, architecture `mellum`) is a chat family: 64-expert MoE with sliding-window attention, ChatML with JSON tool calls, and the `mellum2` pre-tokenizer in toknroll.
- **Kokoro.** Kokoro 82M joins Inflect as a speech family (`jinfer-kokoro`): the model GGUF plus one voice pack as the `voice` companion, nine languages by voice, and espeak-ng on `PATH` for the phoneme front end, which speaks misaki's dialect - the one the model was trained on.
- **Speech companions.** The pronunciation lexicon of Inflect models is the `lexicon` companion, attachable on both speech builders and as `spring.ai.jinfer.speech.companions.lexicon`; a Kokoro voice attaches the same way.
- **Speech at the phoneme level.** `SpeechSynthesisModel.synthesize` takes phoneme ids and `phonemizer()` exposes the model's front end, a `Phonemizer` in `jinfer-core` that is to speech what a tokenizer is to text; `speak(text)` remains the text door.
  `Espeak` in `jinfer-codecs` drives espeak-ng for both speech families; the per-family symbol tables and espeak drivers are gone.
- **Spring Boot examples.** `mvn spring-boot:run` runs with full tiered compilation; its default `optimizedLaunch` pinned C1 and slowed the Vector API about a hundredfold.
- **CLI errors.** A bad `--cache` file, a read-only cache root and other wrapped IO failures print one `ERROR` line.

- **Tools with constrained output.** A request may offer tools together with a JSON schema or a grammar: the family's reply language then offers a tool call or the document, so langchain4j's tool-round-then-structured-answer loop works in one service call.
  A forced tool call with constrained output is still refused, and a family without a combined language refuses at request time.
- **Stringified arguments.** A small model that sends an array or object argument as a JSON string, Llama 3.2 1B does, gets it unwrapped where the tool's schema declares that shape.
- **Gemma 4 video.** A `VideoContent` (langchain4j), a video `Media` (Spring AI) or a `video_url` part (server) renders the way the Gemma 4 processor does: every sampled frame is a timestamped image block, `mm:ss <|image>...<image|>`, one space between frames. Qwen 3.5 still refuses video: its vision tower takes images only.
- **Browser URLs as model refs.** A repository page pasted from the browser (`https://huggingface.co/owner/repo`, its `tree`, `blob` and `resolve` views, ModelScope alike) is the ref it spells, so it lands in the same cache as `owner/repo`; `huggingface.co/owner/repo` is accepted as a host spelling. A plain URL that answers with a web page is refused and never kept in the cache.
- **Vector API check in the library.** A JVM started without `--add-modules jdk.incubator.vector` fails at model load with the one-line remedy, on every binding, instead of a NoClassDefFoundError inside a kernel.
- **Builder ranges.** The langchain4j builders refuse an out-of-range temperature, top-p, top-k, min-p, output limit, timeout or speech speed where it is set, with the range in the message.
- **`--raw-prompt` writes the start token.** The raw lane prepends the model's start tokens (BOS, where the family has one) unless the prompt already spells them, as llama.cpp's `add_bos_token` does; an LFM 2.5 raw prompt no longer decodes to noise.
- **Errors that name the mistake.** An unknown flag in last position is reported as unknown; the server answers 404 for a model name it does not serve and refuses `max_tokens: 0`; the always-reasoning refusal names the lever on every front end; a null embedding batch fails instead of returning nothing; the Narrate and Detect demos report a missing image in one line.
- **Vision prefill no longer collapses once a model has answered.** FlashAttention's pixel-value tiles took their vector species as a parameter, so the species was constant only while the JIT inlined them; once any prefill made the same kernels hot enough to compile standalone, every broadcast de-intrinsified and the tile allocated instead of using registers.
  One 512x512 image on LFM2.5-VL-3B went from 11 MB and 1.07 s to 162 GB and 5.3 s, on GraalVM after the first prefill and on C2 always. The tiles now read the constant species, so an image encode costs 11 MB and 1.06 s on GraalVM and 1.48 s on C2, with no JVM flags.
- **LFM2.5 thinking policy.** A checkpoint whose template never writes a think span (the 350M instruct) reports `NONE`; the ones that do keep `OPTIONAL` or `ALWAYS`.
  LFM2.5-VL-3B ships the same template as the 8B-A1B but does not reason, so it is read off the architecture rather than the template source: `thinking(false)` on the vision models works instead of being refused.

### Known limits

- The `Logic` gallery demo and the model-backed tests pin temperature 0 and a seed; small models still fail some puzzles, which the demo reports honestly.
- When a JSON schema's fields are all optional, LFM2.5-8B-A1B may leave out a field the text does state. Mark the fields you rely on as `required`, or check the extracted values; `describeSchema(false)` turns the description line off entirely.
