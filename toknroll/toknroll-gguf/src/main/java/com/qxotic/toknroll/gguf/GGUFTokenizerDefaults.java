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

    private static final String DEEPSEEK_V3_MAIN =
            "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    private static final String CJK_RANGE = "[一-龥぀-ゟ゠-ヿ]+";

    // minicpm5 main pass (after the 1-3 digit stage): qwen2 with unbounded digit runs.
    private static final String MINICPM5_MAIN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r"
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

        // Name groups mirror llama.cpp's llama-vocab.cpp: names listed together share a
        // byte-identical regex set there, so here they share one splitter factory.
        registerPreTokenizers(
                builder,
                GPT2_PATTERN,
                "gpt-2",
                "gpt2",
                "granite-docling",
                "exaone4",
                "modern-bert");
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
                "smaug-bpe",
                "falcon3",
                "falcon-h1",
                "jina-v5-nano");
        registerPreTokenizers(
                builder,
                QWEN2_PATTERN,
                "qwen2",
                "solar-open",
                "hunyuan",
                "grok-2",
                "deepseek-r1-qwen");
        registerPreTokenizers(builder, QWEN35_PATTERN, "qwen35");
        registerPreTokenizers(builder, TEKKEN_PATTERN, "tekken");
        registerPreTokenizers(builder, GPT4O_PATTERN, "gpt-4o", "kanana2", "minimax-m2");
        registerPreTokenizers(builder, KIMI_K2_PATTERN, "kimi-k2");
        registerPreTokenizers(builder, GEMMA4_PATTERN, "gemma4", "granite-embed-multi-311m");

        registerSequencePreTokenizers(
                builder, new String[] {"\\p{N}{1,3}", CJK_RANGE, DEEPSEEK_V3_MAIN}, "deepseek-v3");
        // Digit-first stacks: every digit split apart, then the GPT-2 word-level pass.
        registerSequencePreTokenizers(
                builder, new String[] {"\\p{N}", GPT2_PATTERN}, "smollm", "command-r", "exaone");
        registerSequencePreTokenizers(
                builder, new String[] {"\\p{N}{1,3}", MINICPM5_MAIN}, "minicpm5");

        // SPM models with "default" pre-tokenizer need identity splitter + metaspace normalizer.
        builder.registerPreTokenizer("default", gguf -> Splitter.identity());
        builder.registerNormalizer("default", METASPACE_NORMALIZER_FACTORY);

        builder.registerPreFallback("gemma4", "gemma4");
        builder.registerNormalizerFallback("gemma4", "gemma4");
    }

    private static Splitter regexSplitter(String pattern) {
        return Splitter.regex(Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS));
    }

    /**
     * Registers the splitter factory under every key, each with the identity normalizer. Factories
     * must capture pattern STRINGS, never compiled {@link Pattern}s: resolution invokes the one
     * factory the GGUF's pre name selects, so patterns compile once per load - and a native image
     * that build-time-initializes this table bakes only strings into its heap, not compiled pattern
     * node trees.
     */
    private static void register(
            GGUFTokenizerLoader.Builder builder, Function<GGUF, Splitter> factory, String... keys) {
        for (String key : keys) {
            builder.registerPreTokenizer(key, factory);
            builder.registerNormalizer(key, IDENTITY_NORMALIZER_FACTORY);
        }
    }

    private static void registerPreTokenizers(
            GGUFTokenizerLoader.Builder builder, String pattern, String... keys) {
        register(builder, gguf -> regexSplitter(pattern), keys);
    }

    private static void registerSequencePreTokenizers(
            GGUFTokenizerLoader.Builder builder, String[] patterns, String... keys) {
        register(
                builder,
                gguf -> {
                    Splitter[] stages = new Splitter[patterns.length];
                    for (int i = 0; i < patterns.length; i++) {
                        stages[i] = regexSplitter(patterns[i]);
                    }
                    return Splitter.sequence(stages);
                },
                keys);
    }
}
