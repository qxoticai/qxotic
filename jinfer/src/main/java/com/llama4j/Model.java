// The model abstraction: each supported architecture (LFM2.5, Gemma4, ...) is one Model
// implementation. The engine drives generation purely through this seam and never sees an
// architecture's config, weights or state. Tensors, kernels, samplers, the GGUF tokenizer and
// chat formatting are shared infrastructure underneath.
package com.llama4j;

import java.util.Optional;
import java.util.Set;

/** A loadable language model behind a uniform inference seam. {@link Engine} prefills and decodes
 *  through these methods alone; architecture specifics (sliding-window attention, MoE, per-layer
 *  embeddings, short convolutions, ...) live entirely inside the implementation and its
 *  {@link InferenceState}. */
interface Model {

    /** GGUF-loaded tokenizer: vocabulary, special tokens and the (optional) chat template. */
    LFMTokenizer tokenizer();

    /** Maximum number of context positions a state can hold. */
    int contextLength();

    int vocabularySize();

    /** Largest token chunk a single {@link #ingest} call accepts (>= 1). Batched models prefill
     *  in chunks of this size; single-token models return 1. */
    int batchCapacity();

    /** A fresh inference state (KV cache + scratch), positioned before the first token. */
    InferenceState createNewState();

    /** Ingest {@code tokens[tokenOffset, tokenOffset+sequenceLength)} at context positions
     *  {@code [startPosition, startPosition+sequenceLength)}, extending the KV cache. */
    void ingest(InferenceState state, int[] tokens, int tokenOffset, int startPosition, int sequenceLength);

    /** Logits for the last ingested token. Idempotent between ingests. */
    FloatTensor computeLogits(InferenceState state);

    /** Tokens that terminate generation for this model's chat format (end-of-turn / eos / ...). */
    Set<Integer> stopTokens();

    /** Builds prompt tokens for a chat request in this model's format (Jinja template, ChatML or a
     *  hand-written format). Internal — never exposed by the OpenAI layer. */
    ChatFormat chatFormat();

    /** This model's incremental prompt-cache support, or empty if it has none. When present, the
     *  server drives the returned cache opaquely; only models whose KV/conv layout the cache
     *  understands provide one, so the engine never needs to know the concrete model type. */
    default Optional<PromptCacheSupport> promptCacheSupport() {
        return Optional.empty();
    }
}
