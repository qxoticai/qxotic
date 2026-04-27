package com.qxotic.toknroll.gguf;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import java.util.function.Function;
import java.util.regex.Pattern;

final class GGUFTokenizerDefaults {
    private static final String GPT2_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    private static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String QWEN2_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String QWEN35_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String TEKKEN_PATTERN =
            "[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                + "\\n"
                + "/]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+";

    private static final String GPT4O_PATTERN =
            "[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}|"
                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                + "\\n"
                + "/]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+";

    private static final String GEMMA4_PATTERN = "[^\\n]+|[\\n]+";

    private static final String KIMI_K2_PATTERN =
            "[\\p{IsHan}]+|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}|"
                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                + "\\n"
                + "]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+";

    private static final Normalizer IDENTITY_NORMALIZER = Normalizer.identity();
    private static final Function<GGUF, Normalizer> IDENTITY_NORMALIZER_FACTORY =
            gguf -> IDENTITY_NORMALIZER;

    private static final Normalizer METASPACE_NORMALIZER =
            text -> '\u2581' + text.toString().replace(' ', '\u2581');
    private static final Function<GGUF, Normalizer> METASPACE_NORMALIZER_FACTORY =
            gguf -> METASPACE_NORMALIZER;

    private GGUFTokenizerDefaults() {}

    static void applyTo(GGUFTokenizerLoader.Builder builder) {
        builder.registerModelFactory("gpt2", GGUFTokenizerModelFactory::buildTiktokenModel);
        builder.registerModelFactory("llama", GGUFTokenizerModelFactory::buildSentencePieceModel);
        builder.registerModelFactory("gemma4", GGUFTokenizerModelFactory::buildSentencePieceModel);

        registerPreTokenizers(builder, GPT2_PATTERN, "gpt-2", "gpt2");
        registerPreTokenizers(
                builder,
                LLAMA3_PATTERN,
                "llama3",
                "llama-v3",
                "llama-bpe",
                "pixtral",
                "smollm3",
                "llama4",
                "glm4",
                "dbrx",
                "smaug-bpe");
        registerPreTokenizers(builder, QWEN2_PATTERN, "qwen2");
        registerPreTokenizers(builder, QWEN35_PATTERN, "qwen35");
        registerPreTokenizers(builder, TEKKEN_PATTERN, "tekken");
        registerPreTokenizers(builder, GPT4O_PATTERN, "gpt-4o", "kanana2", "minimax-m2");
        registerPreTokenizers(builder, KIMI_K2_PATTERN, "kimi-k2");
        registerPreTokenizers(builder, GEMMA4_PATTERN, "gemma4");

        // SPM models with "default" pre-tokenizer need identity splitter + metaspace normalizer.
        builder.registerPreTokenizer("default", gguf -> Splitter.identity());
        builder.registerNormalizer("default", METASPACE_NORMALIZER_FACTORY);

        registerNormalizers(
                builder,
                "gpt-2",
                "gpt2",
                "llama3",
                "llama-v3",
                "llama-bpe",
                "pixtral",
                "smollm3",
                "llama4",
                "glm4",
                "dbrx",
                "smaug-bpe",
                "qwen2",
                "qwen35",
                "tekken",
                "gpt-4o",
                "kanana2",
                "minimax-m2",
                "kimi-k2",
                "gemma4");

        builder.registerPreFallback("gemma4", "gemma4");
        builder.registerNormalizerFallback("gemma4", "gemma4");
    }

    private static Splitter regexSplitter(String pattern) {
        return Splitter.regex(Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS));
    }

    private static void registerPreTokenizers(
            GGUFTokenizerLoader.Builder builder, String pattern, String... keys) {
        Splitter splitter = regexSplitter(pattern);
        for (String key : keys) {
            builder.registerPreTokenizer(key, gguf -> splitter);
        }
    }

    private static void registerNormalizers(GGUFTokenizerLoader.Builder builder, String... keys) {
        for (String key : keys) {
            builder.registerNormalizer(key, IDENTITY_NORMALIZER_FACTORY);
        }
    }
}
