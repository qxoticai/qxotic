package com.qxotic.jinfer.chat;

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
        Optional<ChatTemplate> template) {

    public LoadedModel {
        if (model == null) throw new IllegalArgumentException("null model");
        if (tokenizer == null) throw new IllegalArgumentException("null tokenizer");
        if (chatTemplateSource == null) throw new IllegalArgumentException("null template source");
        if (seed == null) throw new IllegalArgumentException("null seed");
        if (template == null) throw new IllegalArgumentException("null template optional");
        stopTokens = Set.copyOf(stopTokens);
        seed = seed.clone();
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
                template);
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
     * The state codec, required: throws when this model does not support block caching (large
     * recurrent state). Use {@code model().stateCodec()} for the capability query.
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
}
