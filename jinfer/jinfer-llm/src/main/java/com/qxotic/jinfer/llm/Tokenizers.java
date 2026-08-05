package com.qxotic.jinfer.llm;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.util.regex.Pattern;

/**
 * The one place GGUF tokenizer knowledge lives: builds a toknroll {@link Tokenizer} from a GGUF's
 * {@code tokenizer.ggml.*} metadata, with the model-family pre-tokenizers toknroll's builtins lack
 * registered here. Everything above this consumes the container-blind {@link Tokenizer}.
 */
public final class Tokenizers {

    private static final String LFM2_PRE_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    + "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"
                    + "|\\p{N}{1,3}"
                    + "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*"
                    + "|\\s*[\\r\\n]+"
                    + "|\\s+(?!\\S)"
                    + "|\\s+";

    private Tokenizers() {}

    public static Tokenizer fromGGUF(GGUF gguf) {
        return fromGGUF(gguf, b -> b);
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
     * <p>Registrations apply after the builtins and the bundled families, so they can override. An
     * unregistered {@code tokenizer.ggml.pre} still fails loudly with the register-it remedy -
     * nothing silently mis-tokenizes.
     */
    public static Tokenizer fromGGUF(
            GGUF gguf,
            java.util.function.UnaryOperator<GGUFTokenizerLoader.Builder> registrations) {
        GGUFTokenizerLoader.Builder builder =
                GGUFTokenizerLoader.createBuilderWithBuiltins()
                        .registerPreTokenizer(
                                "lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                        .registerNormalizer("lfm2", g -> Normalizer.identity());
        return registrations.apply(builder).build().fromGGUF(gguf);
    }

    /** The GGUF's raw Jinja chat-template source, or {@code ""} when it carries none. */
    public static String chatTemplateSource(GGUF gguf) {
        return gguf.getStringOrDefault("tokenizer.chat_template", "");
    }
}
