package com.qxotic.toknroll.gguf;

import com.qxotic.toknroll.Splitter;
import java.util.Arrays;
import java.util.regex.Pattern;

/**
 * Model-specific text splitter patterns used by GGUF tokenizer recreation tests.
 *
 * <p>Regexes are ported from llama.cpp tokenizer pre-tokenizer definitions: src/llama-vocab.cpp @
 * 08f21453aec846867b39878500d725a05bd32683.
 */
public final class ModelTextSplitters {

    private ModelTextSplitters() {
        // Utility class
    }

    /**
     * Llama 3 BPE pattern - used by Llama 3, Mistral, and other Llama-based models.
     *
     * <p>Pattern breakdown:
     *
     * <ul>
     *   <li>(?i:'s|'t|'re|'ve|'m|'ll|'d) - Contractions (case insensitive)
     *   <li>[^\r\n\p{L}\p{N}]?\p{L}+ - Optional non-letter/number prefix + letters
     *   <li>\p{N}{1,3} - 1-3 digits
     *   <li>?[^\s\p{L}\p{N}]+[\r\n]* - Optional space + non-letter/number chars
     *   <li>\s*[\r\n]+ - Newlines
     *   <li>\s+(?!\S) - Trailing whitespace
     *   <li>\s+ - Other whitespace
     * </ul>
     */
    public static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    /**
     * Qwen2 pattern - used by Qwen2 and Qwen3 models. Similar to Llama3 but handles single digits
     * differently.
     */
    public static final String QWEN2_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    /** Qwen 3.5 pattern from llama.cpp pre-tokenizer definitions. */
    public static final String QWEN35_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    /** Tekken pattern from llama.cpp pre-tokenizer definitions. */
    public static final String TEKKEN_PATTERN =
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

    /** SmolLM2 patterns - used by SmolLM models. Uses multiple patterns applied sequentially. */
    public static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

    /** DeepSeek V3 pre-tokenizer sequence from tokenizer.json backend config. */
    public static final String[] DEEPSEEK_V3_PATTERNS = {
        "\\p{N}{1,3}",
        "[一-龥\\u3040-ゟ゠-ヿ]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r"
                + "\\n"
                + "\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r"
                + "\\n"
                + "]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+"
    };

    /** GPT-2 style byte-level BPE pattern (used by Granite 4.0 family). */
    public static final String GPT2_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    /** Kimi K2 pre-tokenizer pattern (gpt2 + custom Han-aware letter handling). */
    public static final String KIMI_K2_PATTERN =
            "[\\p{IsHan}]+|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{IsHan}]]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    /**
     * Gemma pattern - used by Gemma models (SentencePiece-based). Gemma uses a different
     * tokenization approach but we use Llama3 pattern as a reasonable approximation for testing.
     */
    public static final String GEMMA_PATTERN = LLAMA3_PATTERN;

    /** Phi pattern - used by Phi models (similar to Llama). */
    public static final String PHI_PATTERN = LLAMA3_PATTERN;

    /**
     * Creates a Splitter for the specified model type.
     *
     * @param modelType the model architecture/type (e.g., "llama", "qwen2", "gemma")
     * @return the appropriate Splitter for that model
     */
    public static Splitter createSplitter(String modelType) {
        if (modelType == null) {
            return Splitter.identity();
        }

        String normalizedType = modelType.toLowerCase();
        if ("llama".equals(normalizedType)
                || "mistral".equals(normalizedType)
                || "mixtral".equals(normalizedType)
                || "phi".equals(normalizedType)
                || "phi3".equals(normalizedType)
                || "phi4".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(LLAMA3_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("qwen".equals(normalizedType)
                || "qwen2".equals(normalizedType)
                || "qwen3".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(QWEN2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("qwen3.5".equals(normalizedType) || "qwen3_5".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("deepseek-v3".equals(normalizedType)
                || "deepseek_v3".equals(normalizedType)
                || "deepseek".equals(normalizedType)) {
            return createDeepSeekV3Splitter();
        }
        if ("tekken".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(TEKKEN_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("gemma".equals(normalizedType)
                || "gemma2".equals(normalizedType)
                || "gemma3".equals(normalizedType)) {
            return createGemmaSplitter();
        }
        if ("smollm".equals(normalizedType) || "smollm2".equals(normalizedType)) {
            return createSmolLMSplitter();
        }
        if ("smollm3".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(LLAMA3_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("granite".equals(normalizedType)
                || "granite4".equals(normalizedType)
                || "granite4.0".equals(normalizedType)
                || "gpt2".equals(normalizedType)) {
            return Splitter.regex(Pattern.compile(GPT2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if ("kimi-k2".equals(normalizedType) || "kimi_k2".equals(normalizedType)) {
            return Splitter.regex(
                    Pattern.compile(KIMI_K2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        return Splitter.identity();
    }

    /**
     * Creates a Splitter for a specific test model.
     *
     * @param model the test model enum value
     * @return the appropriate Splitter
     */
    public static Splitter createSplitter(TestDataManager.TestModel model) {
        if (model == TestDataManager.TestModel.QWEN3_0_6B
                || model == TestDataManager.TestModel.QWEN2_5_0_5B) {
            return Splitter.regex(Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if (model == TestDataManager.TestModel.GEMMA_3_4B_UNSLOTH) {
            return createGemmaSplitter();
        }
        if (model == TestDataManager.TestModel.MISTRAL_3_3B_BARTOWSKI) {
            return Splitter.regex(Pattern.compile(TEKKEN_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        return Splitter.identity();
    }

    /**
     * Creates a Gemma-specific splitter. Gemma uses SentencePiece but we can approximate with a
     * modified pattern.
     */
    private static Splitter createGemmaSplitter() {
        // Gemma uses SentencePiece which is different from BPE
        // For testing purposes, we use IDENTITY since SentencePiece
        // doesn't use regex pre-tokenization like BPE does
        return Splitter.identity();
    }

    /** Creates a SmolLM-specific splitter using multiple patterns. */
    private static Splitter createSmolLMSplitter() {
        Splitter[] splitters =
                Arrays.stream(SMOLLM2_PATTERNS)
                        .map(
                                pattern ->
                                        Splitter.regex(
                                                Pattern.compile(
                                                        pattern, Pattern.UNICODE_CHARACTER_CLASS)))
                        .toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }

    private static Splitter createDeepSeekV3Splitter() {
        Splitter[] splitters =
                Arrays.stream(DEEPSEEK_V3_PATTERNS)
                        .map(
                                pattern ->
                                        Splitter.regex(
                                                Pattern.compile(
                                                        pattern, Pattern.UNICODE_CHARACTER_CLASS)))
                        .toArray(Splitter[]::new);
        return Splitter.sequence(splitters);
    }
}
