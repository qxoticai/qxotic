# jinfer-example-judge-advisor

Remote genius + local judge: **Kimi** (OpenAI-compatible API) generates answers, a **local GGUF
running in-JVM via jinfer** judges them, and a self-refine advisor loops until the judge passes.

This is the [spring-ai-examples](https://github.com/spring-projects/spring-ai-examples/tree/main/advisors/evaluation-recursive-advisor-demo)
`evaluation-recursive-advisor-demo` pattern, hardened where the original is fragile:

1. **The judge can never emit unparseable output** - the verdict schema is compiled to a GBNF
   grammar that masks logits at decode time (the original crashes the loop on a judge parse
   error; a budget-truncated verdict degrades to a failed evaluation, not a crash).
2. **The rating is always 1-4** - the schema pins `rating` to `enum [1,2,3,4]` (the original
   silently accepts `5`, which passes its `>= successRating` check).
3. **The rubric is prefilled once** via jinfer's cached prompts - every verdict restores it from
   the KV block tree (`cacheRead` in the transcript) instead of re-prefilling.

## What a run looks like

```
>>> tool: weather(Paris) -> -255C
>>> judge: attempt 1 failed (rating 1, 249 tokens, cacheRead 163, decode PT6.7S): The response
    is factually wrong... [answer: The current weather in Paris is sunny with a temperature of -255°C.]
>>> tool: weather(Paris) -> 15C
>>> judge: attempt 2 passed (rating 4, 233 tokens, cacheRead 163, decode PT5.6S)
    [answer: The current weather in Paris is sunny with a temperature of 15C.]
>>> FINAL: The current weather in Paris is sunny with a temperature of 15C.
```

The weather tool returns -255C (below absolute zero) on alternating calls, so the judge must
fail at least the first answer.

## Run

```
export KIMI_API_KEY=sk-kimi-...      # kimi.com (Kimi for Coding); api.moonshot.ai keys work too
export JINFER_JUDGE_MODEL=/path/to/judge.gguf   # e.g. LFM2.5-8B-A1B-Q8_0.gguf
mvn spring-boot:run
```

Defaults: `https://api.kimi.com/coding/v1`, model `kimi-for-coding` (override with
`KIMI_BASE_URL` / `KIMI_MODEL`; for the Moonshot platform use `https://api.moonshot.ai/v1`
and e.g. `kimi-k2-0711-preview`).

## Tuning lessons baked in (measured)

- **The judge keeps its think span ON** - `thinking(false)` halves verdict cost but the judge
  stopped catching the impossible claim it exists to catch. Cheap and wrong loses.
- **The rubric is terse and gives honest hedges a pass path** - verdicts stay ~250 tokens
  (~6s at ~40 tok/s decode on a 32-core CPU) and honest "no real-time data" answers pass.
- **Judge latency is decode-bound** - the cached rubric keeps prefill at ~0.3s; the decode is
  the verdict text itself.
- **An 8B judge handles clear cases** - for stricter judging point `JINFER_JUDGE_MODEL` at a
  bigger GGUF; the loop mechanics don't change.

## Tests

- Unit: `mvn test` - the loop with stub models (pass/retry/feedback/max-attempts/tool-skip/
  out-of-range/truncation), no GGUF, no API key
- Offline (both models local): `mvn test -Dsurefire.excludedGroups= -Dgroups=integration`
- Kimi: same, with `KIMI_API_KEY` set (auto-skips otherwise)
