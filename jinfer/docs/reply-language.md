# The Reply Language

Design document, 2026-08-09.
Status: ALL NINE families parse through the walk (MiniCPM5 and Gemma4 joined via the claimed-span rule); gpt-oss/SmolLM3/Granite/Mistral force through schema-bound selections (Mistral gained forcing it NEVER had); the reply lifecycle laws landed (`beginReply` - prompt bytes are not reply bytes; `ended` - generation stops when the language says the reply is over).
GBNF-OPENING REGIONS landed too: a region may open on its payload's own tokens (`content(gbnf(schema))`), which makes tools + a JSON-schema response format ONE selection - calls stay the family's syntax, visible text can only be the schema; proven E2E on LFM2.5 (`ToolsWithSchemaPrototypeIT`), authored via the `spans` preset's content-hole overload, driven by `Walk.sampler`.
Remaining wiring for the tools+schema capability: a `ChatTemplate` hook, `prepare()` where the loud rejection sits, and the TCK capability flip.
ALL NINE families now parse through the walk: MiniCPM5 and Gemma4 joined once the claimed-span rule landed (a marker-pair call span claims interior control tokens as their spellings - the old span parsers' exact semantics, derived instead of designed; open question 4 is resolved).
The `ReplyLanguage.spans(...)` preset (mirroring `ReplyParser.spans`) is the one-line authoring form the span families share; a duplicate opener of an open THINK/CONTENT region is scaffold-inert (the span parsers' nested-open behavior, needed by prompt-opened think seeds).
The gpt-oss REQUIRED defect gate passed 5/5 E2E where the seed/pin/epilogue recipe failed ~1 in 3.
The hardening pass added: same-opener CANDIDACY (Harmony's four `<|channel|>` messages), shared-span CALL regions carrying several calls (LFM2's bracket list), close-less regions exiting on payload ACCEPTANCE not exhaustion (every whitespace-tolerant schema tail), whole-program build-time ambiguity validation, pinned-NORMAL ids as control everywhere, and the UTF-8 flush discipline.

## 1. Why

A model family's reply structure is described three times today.
`TokenRuns` code in each template renders call turns back into history (prescriptive, encode side).
`ReplyParser` implementations recognize what the model emitted (descriptive, decode side).
`callGrammar`/`callSeed`/`callEpilogue`/`replySeed` and the sampler wrappers constrain what the model may emit (prescriptive, decode side).

Every recorded forced-call lesson is a patch over gaps between these descriptions.
The three field failures we have collected are one defect class: a constrained region that ends mid-call hands an off-policy state to free sampling.

- LFM2.5 hallucinated a `toolbench_rapidapi_key` argument when the pin ended at the delimiter.
- Mistral derailed after a name-only pin and returned an empty forced reply.
- gpt-oss malforms its free argument JSON after the forced epilogue in roughly one run in three.

This document replaces the three descriptions with ONE definition per family, from which render, parse, and constrain are derived.
The prompt side already works this way: `ChatTemplate`'s javadoc calls the template "a codec with two directions over one grammar".
The reply language is that promise, kept on the reply side.

## 2. The object

Each family defines its reply language once, as a tree of nodes over the model's full token alphabet.

```
Node := Seq(nodes...)
      | Alt(nodes...)
      | Repeat(node, min, max)          # max -1 = unbounded
      | Bytes(literal)                  # plain content bytes, tokenized plainly
      | Mark(spelling)                  # ONE control token, matched by identity
      | Free                            # pass-through: no mask, boundary-watched
      | Region(kind, node)              # THINK | CONTENT | CALL: the event/budget unit
```

Seven node types; there are no hole types.
A family authors its CALL body as a function `(tool, args) -> Node`: the engine applies it once per offered tool, passing the policy-bound argument subtree (the tool's schema grammar under `REQUIRED`/named choice, the bare syntax under `AUTO`), so the name-to-schema binding lives in ordinary function application and policy never leaks into family code.
A response format is a substitution on the `Free` inside CONTENT regions.
In the family trees below, `name` and `args(S)` are notation for the function's parameters, not node types.

Three faces derive from the tree, and none is authored separately:

- **render**: structure to tokens, for history echoes (marks emit as trusted ids, bytes plain-encode, arguments serialize via their `Syntax`; recorded verbatim ids splice preferentially, unchanged law).
- **parse**: tokens to structure events, one walk (see section 5).
- **constrain**: a policy selects within the tree and the same walk masks logits.

### Notation used in this document

```
A B         sequence          A | B     alternation        A?  A*  A+   repetition
"text"      Bytes             %marker   Mark               free = Free
name, args(S) = the CALL body function's parameters;  content = the Free of a CONTENT region
think{...}  content{...}  call{...}     regions
```

## 3. The control rule

`Free` admits plain tokens only.
Every control token either matches a `Mark` the language expects at the current state, or ends the reply.

This single rule derives, with no per-family code:

- the stop-token set (the accept boundary of the language),
- SmolLM3's turn-fabrication stop (`<|im_start|>` is no Mark of the reply language, so it ends the reply),
- Gemma's `<|tool_response>` handoff stop (same),
- Harmony's interior `<|end|>` NOT stopping the reply (it is a Mark inside the analysis message),
- the reply-side mirror of the prompt scrub law (content can neither mint nor absorb control tokens).

### Mark resolution

The tree is built programmatically, so a `Mark` resolves by spelling through the specials table at bind time, in the tree layer.
The `Grammar` engine gains only a programmatic compile entry with a token-identity element; the GBNF text parser is untouched (payload syntaxes that compile through GBNF are pure bytes).
A family may pin an explicit id for a token the container mistypes as NORMAL (the Gemma4 `<eos>` id 1 case); the resolution is the port author's assertion either way.
A Mark whose spelling is absent from the checkpoint's vocabulary removes every alternative containing it (see capability derivation, section 4).

## 4. Selection: policy as a subset, never as machinery

```
ReplyPolicy := (tools, toolChoice, responseSchema, thinking)
Selection.of(language, policy, tokenizer) -> Selection   (or throws, see section 6)
```

Selection performs, in order:

1. **Prune and select.** Alternatives whose Marks do not resolve in this vocabulary are removed (an older NemotronH checkpoint without `<tool_call>` loses its CALL alternatives).
   `NONE` drops CALL regions; `REQUIRED` drops the no-call alternatives and sets the CALL repeat minimum to 1; a named choice restricts the per-tool alternatives to one; `thinking=false` drops THINK regions (a language without THINK ignores the flag); a response schema substitutes the `Free` inside CONTENT regions.
   The per-tool CALL alternatives are built here by applying the family's `(tool, args) -> Node` body function, so name-to-schema binding is ordinary function application; there is no hole machinery and no single-tool restriction.
2. **Liveness check.** The selected automaton must have an accepting path; otherwise the request is rejected now, before any token (section 6).
3. **Forced prefix extraction.** The longest prefix of the selection with exactly one admissible path is tokenized canonically and returned for prompt injection.
   This derives `callSeed`, `replySeed`, and the epilogue as one concept: forced regions are wherever the automaton has one path, whether at the front or in the interior (interior forcing happens in the walk, front forcing in the prompt for prefill efficiency).

### Prompt-opened regions

Some prompts end inside a region (LFM2 and Qwen3.5 generation prompts pre-open `<think>`).
The walk is initialized by consuming the prompt's reply tail, exactly the existing `Prompt.replySeed` contract; the concept transfers unchanged.

### Region budgets (deferred)

`Thinking.capBudget` keeps working unchanged alongside the walk.
Budgets as region annotations (forcing the region's close Mark on exhaustion) are a later policy feature; the named trigger is a migrated family whose THINK region conflicts with the marker-based cap.

## 5. The walk: one object, streaming included

```
Walk.mask(logits)            # free regions: no-op; constrained regions: grammar mask
Walk.advance(token, events)  # the parse: emits committed structure
Walk.accepted()              # the reply may end here
```

Event emission needs no consensus machinery, because the control rule makes it deterministic.
Streaming deltas come only from `Free` regions, where every token is decided alone: plain tokens stay in the region, control tokens are a boundary.
Structured events are atomic at region exit, and region exits are single Marks or, for close-less regions, the pushdown's balance point; both are unambiguous.
The only holdback is the existing UTF-8 assembly (`PendingUtf8`), unchanged.

There is no new event interface: the walk feeds the contracts that already exist.
Content and thinking deltas go to `ChatEngine.ReplySink`; calls and spans are produced as `Part`s through the `ReplyParser` interface, which the walk implements per migrated family, so `ChatEngine` and both providers keep their contracts throughout.

## 6. Errors: two situations, both on existing seams

No new exception hierarchy; the two things that can happen ride mechanisms every consumer already maps.

- **Unsatisfiable selection** (request time): the selection has no accepting path, and `Selection.of` throws `UnsupportedOperationException` with the structural reason.
  The existing seam maps it: `UnsupportedFeatureException` in both providers, an error response in the server.
  Examples: `REQUIRED` on a family whose language has no CALL region, or whose checkpoint lacks the call Marks; a response schema the syntax cannot host.
- **Budget cut mid-structure** (generation time): the completion finishes with reason `LENGTH`, the ecosystem's exact contract, carrying which region was open and which tool if in a CALL.
  A partial call is never returned as prose; recovery is the caller's, with full information: raise the budget, retry.

Nothing else can happen in production.
Parse-versus-generation disagreement is unrepresentable (same walk), and mid-walk dead ends are excluded by construction (the mask runs every step, so every sampled token is admissible; a state admitting no token is an authoring bug, caught at build/test time by Law 2's round trips, not an error kind).

## 7. The three laws, and the language as its own fuzzer

1. **Totality.** Every selection yields a live automaton or a typed rejection before the first generated token.
2. **Duality.** `parse(render(s)) = s` for every structure `s`, and `render(parse(w))` is byte-identical for canonical wires (verbatim splice included).
   Containment (every render is admitted by the language, today's wire law) is implied: the round trip cannot hold for a render the language rejects.
3. **Preservation.** Masks only exclude: a generation that is compliant today is unchanged (greedy decode bit-identical; sampled decode identical up to renormalizing mass that sat on illegal tokens).
   Behavior changes only in runs that are failures today.

Law 2 makes the language self-testing: walk the automaton with a random driver to SAMPLE a valid wire, parse it back, compare.
Every family gets generative property testing derived from its own definition, plus replay of recorded E2E transcripts through the parse face as the migration acceptance gate.

## 8. The nine family languages

These are the soundness proof: every family below is expressed with the section 2 vocabulary and nothing else.
Argument syntaxes are shared: `JSON` (exists as schema-to-GBNF), `XML` (Qwen dialect), `PY` (pythonic literals), `CPT` (Gemma compact).
Where a family stresses the design, the note says what it proved.

### 8.1 SmolLM3 (ChatML, JSON envelope)

```
reply := think? content{content}?
         call{ %<tool_call> "\n{\"name\": \"" name "\", \"arguments\": " args(JSON) "}\n" %</tool_call> }*
         %<|im_end|>
think := %<think> free %</think>
```

Notes: no-think selections force-close the pair in the prompt (prompt-opened region, empty).
The content region is OPTIONAL: a zero-token free-opening region can never be entered-and-exited, so a reply that goes straight from thinking to a call needs the `?` (this applies to every family below).
The turn-fabrication stop derives from the control rule.

### 8.2 Granite (ChatML roles, JSON envelope, no reasoning scaffold)

```
reply := content{content}
         call{ %<tool_call> "\n{\"name\": \"" name "\", \"arguments\": " args(JSON) "}\n" %</tool_call> }*
         %<|end_of_text|>
```

Notes: the template ignores the thinking flag; the language simply has no THINK region, and selection with `thinking=true` is a no-op.
Proves: the JSON envelope is one shared shape (identical to SmolLM3's CALL body).

### 8.3 LFM2 / LFM2.5 (ChatML, pythonic calls)

```
reply := think? content{content}?
         call{ %<|tool_call_start|> "[" callExpr ("," " " callExpr)* "]" %<|tool_call_end|> }?
         %<|im_end|>
callExpr := name "(" args(PY) ")"         # inside the ONE region's body, per-tool alternation
think := %<think> free %</think>          # generation prompt pre-opens <think>: walk starts inside
```

Notes: ONE bracket span is ONE CALL region carrying the whole parallel-call list; its `calls` parser (the region's payload parser, `Function<String, List<Part.ToolCall>>` - exactly today's span-parser shape) yields every call in the span.
Per-tool schema binding still lives in the body's alternation; the separators and brackets are ordinary body bytes.
A sole parsed call carries the span's verbatim ids; several parse without per-call verbatim (attribution needs offsets nobody records).
`args(PY)` is keyword-argument form `k=v, ...` with Python literals.
The original delimiter lesson (the `(` merge) is gone: the grammar continues through `(` into the schema, so `(city`-style merges stay admissible.
Proves: prompt-opened regions (the walk-seeding contract), a non-JSON syntax, and a shared-span region carrying several calls.

### 8.4 Qwen3.5 (ChatML, function-XML calls)

```
reply := think? content{content}
         call{ %<tool_call> "\n<function=" name ">\n" param* "</function>\n" %</tool_call> }*
         %<|im_end|>
param := "<parameter=" key ">\n" value "\n</parameter>\n"
think := %<think> free %</think>          # generation prompt pre-opens <think>\n
```

Notes: `param*` expands from the tool schema: `key` alternates over schema properties, `value` is that property's grammar; free-text values use until-close exclusion emitted by the XML syntax mapper.
Proves: schema-driven expansion below the call level (per-parameter, not just per-tool).

### 8.5 NemotronH (same dialect as Qwen3.5)

```
reply := think? content{content}
         call{ %<tool_call> "\n<function=" name ">\n" param* "</function>\n" %</tool_call> }*
         %<|im_end|>
```

Notes: older checkpoints lack the `<tool_call>` Marks; capability pruning removes the CALL alternatives and `REQUIRED` becomes `UNSATISFIABLE` at request time, replacing today's silent degradation.
Proves: capability detection derives from Mark resolution.

### 8.6 MiniCPM5 (ChatML, attribute-XML with CDATA)

```
reply := think? content{content}
         call{ %<function " name=\"" name "\">" param* %</function> }*
         %<|im_end|>
param := "<param name=\"" key "\">" value %</param>
```

Notes: `<function`, `</param>`, `</function>` are control tokens but `<param` opens as PLAIN BYTES; the tree mixes Marks and Bytes freely, which no span-pair parser can express cleanly.
`value` is the property grammar or a CDATA-wrapped form for multi-line strings (both alternatives in the syntax mapper, matching `MiniCpmToolSyntax`).
Proves: mixed special/plain structure inside one call.

### 8.7 Mistral / Ministral (v13 wire, close-less calls)

```
reply := content{content}
         call{ %[TOOL_CALLS] name %[ARGS] args(JSON) }*
         %</s>
```

Notes: a call has NO close marker; the CALL region exits when the args grammar ACCEPTS - not when it exhausts, because every whitespace-tolerant schema tail keeps optional continuations pending after the balanced payload.
A control token arriving at an accepting close-less payload exits the region and dispatches; a generation that simply ends there commits the balanced call in `finish()` (llama.cpp's end-of-generation behavior).
Today's `SpanToolCallDetector` self-closing heuristic derives from grammar balance.
The empty-prefix pin failure is structurally gone: `%[ARGS]` and the schema-bound args leave no free region to derail in.
Proves: close-less regions, and a Mark interior to the call (`[ARGS]`), which the old pin could not span.

### 8.8 Gemma4 (turn markers, thought channel, compact calls with a quote token)

```
reply := thought? content{content}
         call{ %<|tool_call> "call:" name "{" args(CPT) "}" %<tool_call|> }*
         ( %<turn|> | %<end_of_turn> | %<eos:1> )        # <eos> pinned by id: GGUF mistypes it NORMAL
thought := %<|channel> "thought\n" free %<channel|>
```

Notes: the channel NAME is plain bytes after a Mark (mixed again).
`CPT` is the compact syntax: unquoted keys, dictsorted, and string values delimited by the `<|"|>` QUOTE TOKEN, so the args syntax itself contains Marks.
A model-emitted `<|tool_response>` ends the reply via the control rule (the handoff stop, derived).
Proves: Marks inside a payload syntax, and explicit-id Mark pinning.

### 8.9 gpt-oss (Harmony channels)

```
reply := analysis* preamble* ( final | call )
analysis := think{ %<|channel|> "analysis" %<|message|> free %<|end|> }
preamble := content{ %<|channel|> "commentary" %<|message|> free %<|end|> }
final    := content{ %<|channel|> "final" %<|message|> content %<|return|> }
call     := call{ %<|channel|> "commentary to=functions." name " " %<|constrain|> "json" %<|message|> args(JSON) %<|call|> }
```

Notes: all four message shapes open on the SAME `<|channel|>` mark - the walk disambiguates them through candidacy on the channel-name bytes that follow (scaffold in every branch, so nothing streams speculatively).
Interior `<|end|>` is a Mark, so the historical stop-set truncation bug is unrepresentable.
`REQUIRED` selects `analysis* preamble* call`, which also makes REQUIRED-with-thinking expressible for the first time (the current default, thinking off, remains a selection).
The argument JSON is schema-bound per tool: the release defect dies here.
Proves: multi-message replies, channel routing, and interior forced scaffold (` <|constrain|>json<|message|>`) as plain tree structure.

## 9. Interface refinements the nine languages forced

Writing all nine families changed the interfaces; this is why the exercise preceded implementation.

1. **`Syntax` produces nodes, not GBNF text.**
   Gemma4's quote token lives INSIDE the argument syntax, so a payload grammar must be able to contain Marks.
   `Syntax.node(schema) -> Node`; GBNF remains an internal front-end for pure-byte syntaxes (JSON, XML, PY compile through it).
   `Syntax` keeps its second face, `serialize(arguments) -> parts`, emitting text runs and quote Marks for render (the existing `Gemma4ToolSyntax.Sink` generalized).
2. **Marks may pin an explicit id.**
   Vocabulary mistyping exists in the wild (Gemma4 `<eos>`); spelling-based resolution needs the escape hatch.
3. **Prompt-opened regions are a first-class walk initialization**, not a special case (LFM2, Qwen3.5, and Gemma4's no-think closed-channel prefix all use it).
4. **Close-less regions are legal**: a region may exit on payload completion (Mistral) rather than a Mark.
5. **AUTO strictness is an engine-owned policy knob**: the `args` subtree binds to the tool schema under `REQUIRED`/named choice, and to the bare syntax under `AUTO` (well-formedness enforced, extra arguments tolerated), preserving today's permissiveness while making malformed calls unrepresentable.

## 10. What this replaces

| today | becomes |
| --- | --- |
| `callSeed`, `replySeed` | forced-prefix extraction from the selection |
| `callGrammar`, `callPrefix` | the CALL region and its body function |
| `callEpilogue` | interior forced regions (one admissible path) |
| "the pin ends AT the name" | meaningless: no interior grammar edges exist |
| `SpansReplyParser`, `HarmonyReplyParser`, `SpanToolCallDetector`, `ToolCallDetector` | the walk's parse face |
| `ChannelConstrainedSampler`, `withPrefixGrammar` | the walk's mask face |
| stop-token sets, `RequestPolicy.endTurn` | the accept boundary plus the control rule |
| `Thinking.capBudget`, think floor | region budgets (deferred; capBudget stays until then) |
| the server's bare-call string scan | unrepresentable input |
| per-family stop additions (`<|im_start|>`, `<|tool_response>`) | the control rule |
| wire-law triangle (render vs grammar vs parser tests) | Law 2 over one artifact |

`ChatTemplate`'s reply surface becomes one hook: `replyLanguage()`.

## 11. Migration plan

External contracts (`ReplyParser`, `ChatEngine.prepare`, provider APIs) hold throughout; families migrate one at a time.

1. **Bricks**: a token-identity element in the `Grammar` engine (feasibility verified against its internals: new `T_TOKEN` slot kind plus a programmatic compile entry, roughly sixty lines, the GBNF parser, stack machinery and mask cache unchanged) and the region-tagged walk layered above it in jinfer-chat.
2. **gpt-oss first**: full language, `ReplyParser` implemented by the walk, Laws 1-3 plus transcript replay as the gate.
   The REQUIRED release defect dies here as a corollary.
3. **SmolLM3 and Granite next** (shared JSON envelope), then LFM2, Mistral, Qwen3.5, NemotronH, MiniCPM5, Gemma4 (hardest: compact syntax with quote Marks).
4. Per migrated family, delete the derived-away code from section 10.
5. Policy features on top, each with a named trigger: parallel-call bounds, region budgets (replacing `capBudget`), `REQUIRED`-with-thinking where the language allows it.

## 12. Open questions

1. **Echo canon fidelity.** `Syntax.serialize` must reproduce each template's existing echo bytes exactly (SmolLM3 echoes Jinja-spaced JSON) or cached-prefix bytes shift; Law 2 catches drift, but the canon must be captured per syntax dialect during migration.
2. **AUTO content inside think spans.** Families that call tools while thinking (Harmony analysis-with-calls variants) need the CALL region admissible inside THINK; the tree expresses it, but per-family corpus replay must confirm each shape before the language claims it.
3. **Schema breadth.** `args(JSON)` inherits the current schema-to-GBNF subset; llama.cpp's converter (same GBNF dialect) is the port source when broader features are demanded.
4. **RESOLVED - payload-interior marks needed no capture-rule design.** The old span parsers already defined the right semantics: a marker-pair call span claims EVERYTHING to its closer, interior control tokens included AS THEIR SPELLINGS (that is the exact decoded text `MiniCpmToolSyntax` and `Gemma4ToolSyntax` always parsed).
The walk derives it in one rule on CALL free holes: closer advances, the region's own opener self-closes the span (the old chained-span behavior), any other control token is payload text + verbatim id; region-final (close-less) frees keep their exit-on-control rule.
The per-mark structural-vs-payload distinction was over-design; the only remaining sliver is FORCED schema-bound CPT arguments (a constrained segment whose grammar contains quote-token terminals would need mark spellings in its captured text) - relevant only if Gemma4 ever gets schema-bound forcing, its named trigger.
5. **Mask cost at scale.** Hundred-tool alternations multiply automaton states; the existing per-state mask cache and first-byte filter carry today's shapes, and xgrammar-style context-independent precomputation is the named upgrade path if profiling ever demands it.
