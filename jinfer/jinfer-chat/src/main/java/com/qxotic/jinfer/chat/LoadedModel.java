package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;

/**
 * A loaded model and everything a generation needs around it, bound at load: the tokenizer, the
 * family's stop tokens, the cache-identity seed, the native chat-template codec when the family has
 * one, and the container's sampling recommendations. Data, not behaviour - a caller that wants
 * different stop tokens builds another record.
 *
 * <p>Lifetime: no {@code close()} - the arena given to {@code Models.load} owns the weights, and
 * who provided it owns its lifetime.
 */
public record LoadedModel<S extends ContextState>(
        LanguageModel<?, ?, S> model,
        Tokenizer tokenizer,
        String chatTemplateSource,
        Set<Integer> stopTokens,
        ContentKey seed,
        Optional<ChatTemplate> template,
        SamplingDefaults samplingDefaults) {

    public LoadedModel {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (chatTemplateSource == null) throw new IllegalArgumentException("null template source");
        if (seed == null) throw new IllegalArgumentException("null seed");
        if (template == null) throw new IllegalArgumentException("null template optional");
        if (samplingDefaults == null) throw new IllegalArgumentException("null sampling defaults");
        // an ORDER-PRESERVING immutable copy, never Set.copyOf: the set's iteration order carries
        // meaning - the family's own end-of-turn comes first, and a decode ended from outside
        // emits iterator().next(). Set.copyOf is salt-randomized per JVM run.
        stopTokens = Collections.unmodifiableSet(new LinkedHashSet<>(stopTokens));
    }

    /**
     * Returns the same loaded model with a caller-supplied Jinja chat template.
     *
     * <p>This is the narrow escape hatch for a GGUF whose embedded template is missing or wrong.
     * The native template is deliberately dropped: a model-family codec was written for the
     * original wire format and must not frame a replacement one. Rendering therefore falls back to
     * {@code source}; replies use the generic parser. The model's prompt start is retained because
     * it also frames raw completions and is independent of the chat-template source.
     *
     * <p>The cache identity is unchanged. Cache entries also fingerprint the rendered token ids, so
     * a changed template naturally misses old prefixes without invalidating compatible ones.
     */
    public LoadedModel<S> withChatTemplateSource(String source) {
        if (source == null || source.isBlank()) {
            throw new IllegalArgumentException("blank chat template source");
        }
        IntSequence promptStart =
                template.map(ChatTemplate::promptStart).orElse(IntSequence.empty());
        Optional<ChatTemplate> fallback =
                promptStart.isEmpty()
                        ? Optional.empty()
                        : Optional.of(promptStartOnly(promptStart));
        return new LoadedModel<>(
                model, tokenizer, source, stopTokens, seed, fallback, samplingDefaults);
    }

    private static ChatTemplate promptStartOnly(IntSequence promptStart) {
        return new ChatTemplate() {
            @Override
            public IntSequence promptStart() {
                return promptStart;
            }

            @Override
            public ReplyState encode(
                    Conversation conversation, int batchCapacity, Consumer<Batch> sink) {
                throw new UnsupportedConversation(
                        "native framing disabled by custom chat template source");
            }
        };
    }

    /**
     * The container's recommended sampling parameters, layered:
     *
     * <ol>
     *   <li>the GGUF's {@code general.sampling.*} metadata (llama.cpp's convention),
     *   <li>the model author's recommendation, declared by the port,
     *   <li>the engine baseline (0.8 / 0.95 / 40 / 0.05 - llama.cpp's defaults).
     * </ol>
     *
     * Only the knobs jinfer's sampler stack implements are carried; a container's penalties or
     * mirostat settings are ignored.
     */
    public record SamplingDefaults(Float temperature, Float topP, Integer topK, Float minP) {

        private static final float DEFAULT_TEMPERATURE = 0.8f;
        private static final float DEFAULT_TOP_P = 0.95f;
        private static final int DEFAULT_TOP_K = 40;
        private static final float DEFAULT_MIN_P = 0.05f;

        /** No recommendations - every lookup falls through to the engine baseline. */
        public static final SamplingDefaults NONE = new SamplingDefaults(null, null, null, null);

        static SamplingDefaults fromGGUF(GGUF gguf) {
            return new SamplingDefaults(
                    floatValue(gguf, "general.sampling.temp"),
                    floatValue(gguf, "general.sampling.top_p"),
                    topKValue(gguf),
                    floatValue(gguf, "general.sampling.min_p"));
        }

        /**
         * Reads {@code general.sampling.top_k} following llama.cpp's convention: a non-positive
         * value is an explicit "top-k disabled" statement (normalized to {@code 0}, which {@link
         * com.qxotic.jinfer.llm.Sampling} accepts and the sampler treats as pass-through), not an
         * absent opinion. Only a missing key yields {@code null}, letting the layering fall through
         * to the port's recommendation or the engine baseline.
         */
        private static Integer topKValue(GGUF gguf) {
            if (!gguf.containsKey("general.sampling.top_k")) {
                return null;
            }
            return Math.max(0, gguf.getValue(Integer.class, "general.sampling.top_k"));
        }

        private static Float floatValue(GGUF gguf, String key) {
            return gguf.containsKey(key) ? gguf.getValue(Float.class, key) : null;
        }

        /**
         * Field-wise precedence merge: this record's values where present, {@code fallback}'s where
         * not.
         */
        SamplingDefaults withFallback(SamplingDefaults fallback) {
            return new SamplingDefaults(
                    temperature != null ? temperature : fallback.temperature,
                    topP != null ? topP : fallback.topP,
                    topK != null ? topK : fallback.topK,
                    minP != null ? minP : fallback.minP);
        }

        /**
         * The layered chain resolved into one {@link Sampling}: each non-null argument (a request
         * field or a CLI flag) wins, then this record's recommendation, then the engine baseline.
         * {@code seed} passes through untouched: no layer below a caller has an opinion about
         * randomness, and null means fresh randomness per call.
         */
        public Sampling resolve(
                Float temperature, Float topP, Integer topK, Float minP, Long seed) {
            return new Sampling(
                    temperature != null ? temperature : effectiveTemperature(),
                    topP != null ? topP : effectiveTopP(),
                    topK != null ? topK : effectiveTopK(),
                    minP != null ? minP : effectiveMinP(),
                    seed);
        }

        /** The recommended temperature, or the engine baseline when no layer has one. */
        public float effectiveTemperature() {
            return temperature != null ? temperature : DEFAULT_TEMPERATURE;
        }

        /** The recommended top-p, or the engine baseline when no layer has one. */
        public float effectiveTopP() {
            return topP != null ? topP : DEFAULT_TOP_P;
        }

        /** The recommended top-k, or the engine baseline when no layer has one. */
        public int effectiveTopK() {
            return topK != null ? topK : DEFAULT_TOP_K;
        }

        /** The recommended min-p, or the engine baseline when no layer has one. */
        public float effectiveMinP() {
            return minP != null ? minP : DEFAULT_MIN_P;
        }
    }
}
