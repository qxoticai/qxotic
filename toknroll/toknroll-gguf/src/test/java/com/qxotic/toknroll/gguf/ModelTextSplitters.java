package com.qxotic.toknroll.gguf;

import com.qxotic.toknroll.Splitter;
import java.util.Arrays;
import java.util.regex.Pattern;

/** Model-specific text splitter patterns used by GGUF tokenizer recreation tests. */
public final class ModelTextSplitters {

    private ModelTextSplitters() {}

    public static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    public static final String QWEN2_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    public static final String QWEN35_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

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

    public static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };

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

    public static final String GPT2_PATTERN =
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    public static final String KIMI_K2_PATTERN =
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
            return Splitter.identity();
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

    public static Splitter createSplitter(TestDataManager.TestModel model) {
        if (model == TestDataManager.TestModel.QWEN3_6_35B_A3B) {
            return Splitter.regex(Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if (model == TestDataManager.TestModel.GEMMA_4_26B_A4B) {
            return Splitter.identity();
        }
        if (model == TestDataManager.TestModel.MISTRAL_SMALL_4_119B) {
            return Splitter.regex(Pattern.compile(TEKKEN_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if (model == TestDataManager.TestModel.GRANITE_4_1_3B) {
            return Splitter.regex(Pattern.compile(GPT2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        if (model == TestDataManager.TestModel.KIMI_K2_6) {
            return Splitter.regex(
                    Pattern.compile(KIMI_K2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        }
        return Splitter.identity();
    }

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
