package com.qxotic.tokenizers.loaders;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.Arrays;
import java.util.regex.Pattern;

/**
 * Model-specific text splitters (pre-tokenizers) for various LLM architectures.
 *
 * <p>These splitters are used to break text into initial units before BPE tokenization. Different
 * models use different splitting strategies based on their training data and tokenization approach.
 *
 * <p>This class provides pre-configured splitters for popular model families including:
 *
 * <ul>
 *   <li>Llama/Mistral family (llama3 pattern)
 *   <li>Qwen family (qwen2 pattern)
 *   <li>SmolLM family
 *   <li>Tekken (Mistral's custom tokenizer)
 *   <li>And others
 * </ul>
 */
public final class ModelSplitters {

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

    public static final Splitter LLAMA3 = Splitter.regex(LLAMA3_COMPILED);
    public static final Splitter QWEN2 = Splitter.regex(QWEN2_COMPILED);
    public static final Splitter SMOLLM2 = sequence(SMOLLM2_PATTERNS);
    public static final Splitter REFACT = sequence(REFACT_PATTERNS);
    public static final Splitter TEKKEN = sequence(TEKKEN_PATTERNS);
    public static final Splitter DEFAULT_BPE = sequence(DEFAULT_BPE_SPLITS);
    public static final Splitter IDENTITY = Splitter.identity();

    private ModelSplitters() {}

    /**
     * Returns a splitter for the given model type name.
     *
     * @param modelType the model type name (e.g., "llama", "qwen2", "mistral")
     * @return the appropriate splitter, or null if not found
     */
    public static Splitter forModel(String modelType) {
        if (modelType == null || modelType.isEmpty()) {
            return null;
        }
        String normalized = modelType.trim().toLowerCase();
        return switch (normalized) {
            case "llama", "llama-bpe", "mistral", "mixtral", "phi", "phi3", "dbrx" -> LLAMA3;
            case "qwen", "qwen2", "qwen3" -> QWEN2;
            case "smollm", "smollm2" -> SMOLLM2;
            case "tekken" -> TEKKEN;
            case "refact", "granite" -> REFACT;
            case "gemma", "gemma2", "gemma3" -> IDENTITY;
            case "default" -> DEFAULT_BPE;
            default -> null;
        };
    }

    private static Splitter sequence(String[] patterns) {
        return Splitter.sequence(
                Arrays.stream(patterns).map(Splitter::regex).toArray(Splitter[]::new));
    }
}
