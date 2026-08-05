package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.toknroll.Tokenizer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Optional;
import java.util.Set;

/**
 * A {@link LanguageModel} together with the token-level facts its container carries about text: the
 * tokenizer, the raw chat-template source (transport only - compiling it is the Jinja engine's
 * job), the ids that end a turn, and the model's cache identity ({@code seed} - the GGUF-derived
 * key every persisted prompt-cache artifact is rooted in), and the model's chat framing (the native
 * {@link ChatTemplate} codec, when the port ships one).
 *
 * <p>These are data, not behaviour, so they are a record rather than an interface on the model. A
 * caller that wants different stop tokens simply builds another record; a caller that only needs
 * logits passes {@link #model()} and never sees the rest. {@code Generator} takes the model
 * explicitly for exactly that reason.
 *
 * <p>Produced by each model class ({@code loaded()}); the architecture-dispatching loaders (the
 * server's {@code Models.load}) bundle it into a {@code ChatModel}.
 *
 * <p>Lifetime: there is no close() on this record - the arena given to {@code Models.load} owns the
 * weights (one READ_ONLY mmap shared by every tensor; {@code ofAuto} = GC-unmapped, {@code global}
 * = process, scoped = deterministic and must outlive every model sharing them). States own or
 * borrow their memory per {@code newState}; see the lifetime note on {@link
 * com.qxotic.jinfer.Model}.
 */
public record LoadedModel<S extends RuntimeState>(
        LanguageModel<?, ?, S> model,
        Tokenizer tokenizer,
        String chatTemplateSource,
        Set<Integer> stopTokens,
        byte[] seed,
        Optional<ChatTemplate> template,
        SamplingDefaults samplingDefaults) {

    public LoadedModel {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (chatTemplateSource == null) throw new IllegalArgumentException("null template source");
        if (seed == null) throw new IllegalArgumentException("null seed");
        if (template == null) throw new IllegalArgumentException("null template optional");
        if (samplingDefaults == null) throw new IllegalArgumentException("null sampling defaults");
        stopTokens = Set.copyOf(stopTokens);
        seed = seed.clone();
    }

    /** The same model with the container's recommended sampling parameters attached. */
    public LoadedModel<S> withSamplingDefaults(SamplingDefaults samplingDefaults) {
        return new LoadedModel<>(
                model, tokenizer, chatTemplateSource, stopTokens, seed, template, samplingDefaults);
    }

    /**
     * The same model with a different tokenizer - the supported way to override the one the
     * container carries, for a GGUF whose vocabulary metadata is wrong or a caller who has a better
     * one.
     *
     * <p>{@link #seed} is RE-ROOTED, not copied. Prompt-cache artifacts are keyed by the seed and
     * hold token ids, so one built with the container's tokenizer must not mount under a
     * replacement that encodes differently - and must still mount under one that encodes the same.
     * Only behaviour can say which, so the seed folds in a probe: the vocabulary size and the ids
     * of a fixed sentence.
     */
    public LoadedModel<S> withTokenizer(Tokenizer tokenizer) {
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        return new LoadedModel<>(
                model,
                tokenizer,
                chatTemplateSource,
                stopTokens,
                reseed(seed, tokenizer),
                template,
                samplingDefaults);
    }

    /**
     * The same model with a caller-supplied Jinja chat template replacing the container's - for a
     * fine-tune whose GGUF carries the wrong template, or a wire format you wrote yourself. The
     * native per-turn codec is DROPPED along with it: every port was written against the
     * container's template, and framing a custom wire with one would encode conversations the model
     * never saw. Chat then renders through the whole-render Jinja path with {@code source}; reply
     * parsing degrades to plain text (no think/tool lanes) - implement {@link
     * #withTemplate(ChatTemplate)} instead when the custom wire has markers worth parsing.
     *
     * <p>The cache {@link #seed} is NOT re-rooted (unlike {@link #withTokenizer}): a different
     * template renders different token streams, so cached prefixes from the old wire miss cold -
     * never corrupt.
     */
    public LoadedModel<S> withChatTemplateSource(String source) {
        if (source == null || source.isBlank()) {
            throw new IllegalArgumentException("blank chat template source");
        }
        return new LoadedModel<>(
                model, tokenizer, source, stopTokens, seed, Optional.empty(), samplingDefaults);
    }

    /**
     * The same model with a caller-implemented {@link ChatTemplate} - full control of the wire: the
     * encoding AND the reply grammar ({@link ChatTemplate#parser}). It takes the native slot, so it
     * is preferred over the Jinja fallback exactly like a shipped port; the container's Jinja
     * template remains the fallback when the implementation punts with {@code
     * UnsupportedConversation} or unknown template kwargs arrive. The cache seed is not re-rooted -
     * see {@link #withChatTemplateSource} for why that is sound.
     */
    public LoadedModel<S> withTemplate(ChatTemplate template) {
        if (template == null) throw new IllegalArgumentException("null template");
        return new LoadedModel<>(
                model,
                tokenizer,
                chatTemplateSource,
                stopTokens,
                seed,
                Optional.of(template),
                samplingDefaults);
    }

    /**
     * Letters, digits, whitespace and non-Latin script: enough to separate any two real
     * vocabularies.
     */
    private static final String SEED_PROBE = "The quick brown fox jumps over 0123456789 éß中文";

    private static byte[] reseed(byte[] seed, Tokenizer tokenizer) {
        StringBuilder probe = new StringBuilder().append(tokenizer.vocabulary().size());
        for (int id : tokenizer.encodeToArray(SEED_PROBE)) probe.append(' ').append(id);
        try {
            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            sha.update(seed);
            return sha.digest(probe.toString().getBytes(StandardCharsets.UTF_8));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    /**
     * The state codec, required: throws when this model declares none (every shipped chat family
     * has one - fine, or coarse for the large-recurrence hybrids). Use {@code model().stateCodec()}
     * for the capability query.
     */
    public StateCodec<S> codec() {
        return model.stateCodec()
                .orElseThrow(
                        () ->
                                new IllegalStateException(
                                        model.getClass().getSimpleName()
                                                + " does not support block caching (no state"
                                                + " codec)"));
    }

    /**
     * The sampling parameters recommended for a model, resolved field-wise through a chain of
     * opinions - {@code null} where a layer has none:
     *
     * <ol>
     *   <li>the GGUF's {@code general.sampling.*} metadata (llama.cpp's convention),
     *   <li>the model author's documented recommendation, declared by the port in {@code loaded()},
     *   <li>the engine baseline (0.8, top-p 0.95 - llama.cpp's defaults).
     * </ol>
     *
     * <p>{@code Models.load} layers 1 over 2 with {@link #withFallback}; consumers finish the chain
     * with {@link #effectiveTemperature()}/{@link #effectiveTopP()} - an explicit request or
     * configuration value always wins before this record is consulted at all. Only the knobs
     * jinfer's sampler stack implements are carried; a container's {@code top_k}, penalties or
     * mirostat settings are ignored.
     */
    public record SamplingDefaults(Float temperature, Float topP) {

        // the engine baseline, OK-ish for any chat model (llama.cpp's defaults)
        private static final float DEFAULT_TEMPERATURE = 0.8f;
        private static final float DEFAULT_TOP_P = 0.95f;

        /** No recommendations - every lookup falls through to the engine baseline. */
        public static final SamplingDefaults NONE = new SamplingDefaults(null, null);

        static SamplingDefaults fromGGUF(GGUF gguf) {
            return new SamplingDefaults(
                    floatValue(gguf, "general.sampling.temp"),
                    floatValue(gguf, "general.sampling.top_p"));
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
                    topP != null ? topP : fallback.topP);
        }

        /** The recommended temperature, or the engine baseline when no layer has one. */
        public float effectiveTemperature() {
            return temperature != null ? temperature : DEFAULT_TEMPERATURE;
        }

        /** The recommended top-p, or the engine baseline when no layer has one. */
        public float effectiveTopP() {
            return topP != null ? topP : DEFAULT_TOP_P;
        }
    }
}
