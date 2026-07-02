# Prompt cache (attic - reference only, not built)

`PromptCache.java` + `PromptCacheSupport.java` were the LFM2.5 prompt/prefix cache from the legacy
production engine. They are preserved here as reference for a future re-port; they are **not part of
the build** and will not compile as-is (they reference the now-deleted legacy `Llama`/`Engine`).

## Why it's here and not in a module

The cache is not a model-agnostic add-on - it reaches directly into the legacy `Llama.State`'s
internals:

- `PromptCache.Harvest implements Llama.ConvHarvest` - harvests the short-conv state (LFM2.5's conv layers)
- `PromptCache.CacheRun implements GenerationHooks` - hooks into the prefill/decode loop for prefix resume
- `beginGeneration(InferenceState)` casts to `Llama.State`; `restore`/`snapshotConv`/`copyStateRows`/
  `commitSpan` read and write the KV rows + conv memory of that concrete state.

The code says it: *"The cache is only ever created for LFM2.5 ... the engine's opaque state is a
Llama.State here."*

## Re-porting to the new API

To bring it back, re-implement against the new `Lfm2.State` (jinfer-lfm2): its `keyCache`/`valueCache`
(F16 KV rings) + `shortConvState` (rolling dConv history). The new-API generation driver is
`Generator` (jinfer-core), which currently has **no** prefix-resume mechanism - `Generator.generate`
does one `Batch.prefill` then decodes. A re-port needs:

1. A resume/prefix hook on the `Batch` ingest path (ingest at a non-zero start offset, reusing cached KV),
   replacing the old `GenerationHooks` (resumePosition / clampChunk / afterIngest / afterPrefill).
2. Save/restore of `Lfm2.State`'s KV + short-conv memory (the `restore`/`snapshotConv`/`copyStateRows`
   equivalents against the new layout).
3. Wiring into `Generator` + the server (`Generation` dropped its cache; `Generation.cache()` and the
   `/metrics` + `/props` cache stats were removed - re-add them when the cache returns).
