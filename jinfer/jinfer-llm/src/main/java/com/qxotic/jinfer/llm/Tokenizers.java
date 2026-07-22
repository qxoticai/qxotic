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
        return GGUFTokenizerLoader.createBuilderWithBuiltins()
                .registerPreTokenizer(
                        "lfm2", g -> Splitter.regex(Pattern.compile(LFM2_PRE_PATTERN)))
                .registerNormalizer("lfm2", g -> Normalizer.identity())
                .build()
                .fromGGUF(gguf);
    }

    /** The GGUF's raw Jinja chat-template source, or {@code ""} when it carries none. */
    public static String chatTemplateSource(GGUF gguf) {
        return gguf.getStringOrDefault("tokenizer.chat_template", "");
    }
}
