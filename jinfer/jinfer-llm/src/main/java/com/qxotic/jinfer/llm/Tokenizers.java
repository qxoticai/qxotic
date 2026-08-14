package com.qxotic.jinfer.llm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.util.function.UnaryOperator;

/**
 * The model-loader-owned slice of GGUF tokenizer handling. Everything about SCHEMES - the builtin
 * table, per-port registrations, and the {@code -Dtoknroll.gguf.pre.<name>=alias:|regex:|file:}
 * escape hatch (priority: builtin &lt; registration &lt; property) - lives in toknroll's {@link
 * GGUFTokenizerLoader}; what stays here is what only a model loader can know: that a supplied
 * tokenizer's ids must match the embedding table ({@link #requireSameIdSpace}), and where the chat
 * template sits in the header ({@link #chatTemplateSource}).
 */
public final class Tokenizers {

    private static final String MOVED_PREFIX = "jinfer.preTokenizer.";

    private Tokenizers() {}

    public static Tokenizer fromGGUF(GGUF gguf) {
        return fromGGUF(gguf, b -> b);
    }

    /**
     * Whether any pre-tokenizer override is configured - the FACT, for callers whose policy depends
     * on it (the AOT preload rebuilds instead of using its preloaded tokenizer, so the escape hatch
     * keeps outranking it). Both namespaces count: the current {@code toknroll.gguf.pre.*} and the
     * moved {@code jinfer.preTokenizer.*} (which {@link #fromGGUF} then refuses loudly).
     */
    public static boolean hasPropertyOverrides() {
        for (String key : System.getProperties().stringPropertyNames()) {
            if (key.startsWith(GGUFTokenizerLoader.OVERRIDE_PREFIX)
                    || key.startsWith(MOVED_PREFIX)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Refuses a caller-SUPPLIED tokenizer whose id space cannot match this GGUF: the model's
     * embedding rows and the header's stop-token ids are indexed by token id, so a supplied
     * tokenizer may change surface behaviour (regexes, merges, normalization) but never the ids.
     * Same SIZE is the checkable half of that contract; same-ids-for-same-tokens is the caller's
     * oath.
     */
    public static void requireSameIdSpace(GGUF gguf, Tokenizer tokenizer) {
        if (!gguf.containsKey("tokenizer.ggml.tokens")) {
            return; // no vocabulary in the header: nothing checkable
        }
        String[] tokens = gguf.getValue(String[].class, "tokenizer.ggml.tokens");
        if (tokenizer.vocabulary().size() != tokens.length) {
            throw new IllegalArgumentException(
                    "the supplied tokenizer has "
                            + tokenizer.vocabulary().size()
                            + " tokens but this GGUF's vocabulary has "
                            + tokens.length
                            + " - token ids index the embedding table, so this tokenizer cannot"
                            + " serve this model");
        }
    }

    /**
     * As {@link #fromGGUF(GGUF)} with the caller's own registrations - the entry for a model port
     * whose family needs a pre-tokenizer, normalizer or tokenization model the builtins lack. The
     * port owns its load path, so it passes the pieces right here; no registry:
     *
     * <pre>{@code
     * Tokenizers.fromGGUF(gguf, b ->
     *         b.registerPreTokenizer("myfamily", g -> Splitter.regex(MY_PATTERN)));
     * }</pre>
     *
     * <p>Registrations apply after the builtins, so they can override. An unregistered {@code
     * tokenizer.ggml.pre} fails loudly with the no-rebuild remedy ({@code
     * -Dtoknroll.gguf.pre.<name>=...}) in the message - nothing silently mis-tokenizes.
     */
    public static Tokenizer fromGGUF(
            GGUF gguf, UnaryOperator<GGUFTokenizerLoader.Builder> registrations) {
        rejectMovedFlags();
        return registrations
                .apply(GGUFTokenizerLoader.createBuilderWithBuiltins())
                .build()
                .fromGGUF(gguf);
    }

    /**
     * The old {@code jinfer.preTokenizer.*} namespace, refused loudly rather than silently ignored:
     * the hatch moved into toknroll, so the flag did too.
     */
    private static void rejectMovedFlags() {
        for (String key : System.getProperties().stringPropertyNames()) {
            if (key.startsWith(MOVED_PREFIX)) {
                throw new IllegalArgumentException(
                        "-D"
                                + key
                                + ": the pre-tokenizer escape hatch moved into toknroll - rename"
                                + " it to -D"
                                + GGUFTokenizerLoader.OVERRIDE_PREFIX
                                + key.substring(MOVED_PREFIX.length())
                                + "=<same value>");
            }
        }
    }

    /** The GGUF's raw Jinja chat-template source, or {@code ""} when it carries none. */
    public static String chatTemplateSource(GGUF gguf) {
        return gguf.getStringOrDefault("tokenizer.chat_template", "");
    }
}
