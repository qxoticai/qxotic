package com.qxotic.tokenizers.gguf;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.Arrays;
import java.util.regex.Pattern;

final class GGUFSplitters {

    private static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String QWEN2_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };
    private static final String[] REFACT_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };
    private static final String[] TEKKEN_PATTERNS = {
        "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    };
    private static final String[] DEFAULT_BPE_SPLITS = {
        "[\\p{P}\\$\\+<=>\\^~\\|]+",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        "\\p{N}+",
        "[0-9][0-9][0-9]"
    };

    private static final Pattern LLAMA3_COMPILED = Pattern.compile(LLAMA3_PATTERN);
    private static final Pattern QWEN2_COMPILED = Pattern.compile(QWEN2_PATTERN);

    private static final Splitter LLAMA3_SPLITTER = Splitter.regex(LLAMA3_COMPILED);
    private static final Splitter QWEN2_SPLITTER = Splitter.regex(QWEN2_COMPILED);
    private static final Splitter SMOLLM2_SPLITTER = sequence(SMOLLM2_PATTERNS);
    private static final Splitter REFACT_SPLITTER = sequence(REFACT_PATTERNS);
    private static final Splitter TEKKEN_SPLITTER = sequence(TEKKEN_PATTERNS);
    private static final Splitter DEFAULT_BPE_SPLITTER = sequence(DEFAULT_BPE_SPLITS);
    private static final Splitter IDENTITY_SPLITTER = Splitter.identity();

    private GGUFSplitters() {}

    static void registerDefaults(GGUFPreTokenizerRegistry registry) {
        registry.register("llama", LLAMA3_SPLITTER);
        registry.register("llama-bpe", LLAMA3_SPLITTER);
        registry.register("mistral", LLAMA3_SPLITTER);
        registry.register("mixtral", LLAMA3_SPLITTER);
        registry.register("phi", LLAMA3_SPLITTER);
        registry.register("phi3", LLAMA3_SPLITTER);
        registry.register("dbrx", LLAMA3_SPLITTER);

        registry.register("qwen", QWEN2_SPLITTER);
        registry.register("qwen2", QWEN2_SPLITTER);
        registry.register("qwen3", QWEN2_SPLITTER);

        registry.register("smollm", SMOLLM2_SPLITTER);
        registry.register("smollm2", SMOLLM2_SPLITTER);

        registry.register("tekken", TEKKEN_SPLITTER);
        registry.register("refact", REFACT_SPLITTER);
        registry.register("granite", REFACT_SPLITTER);
        registry.register("default", DEFAULT_BPE_SPLITTER);

        registry.register("gemma", IDENTITY_SPLITTER);
        registry.register("gemma2", IDENTITY_SPLITTER);
        registry.register("gemma3", IDENTITY_SPLITTER);
    }

    private static Splitter sequence(String[] patterns) {
        return Splitter.sequence(
                Arrays.stream(patterns).map(Splitter::regex).toArray(Splitter[]::new));
    }
}
