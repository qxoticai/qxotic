package com.qxotic.toknroll.loaders;

import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.impl.FastTekkenSplitter;
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

    // Regexes below are ported from llama.cpp tokenizer pre-tokenizer definitions:
    // src/llama-vocab.cpp @ 08f21453aec846867b39878500d725a05bd32683
    // - LLAMA_VOCAB_PRE_TYPE_LLAMA3
    // - LLAMA_VOCAB_PRE_TYPE_QWEN2
    // - LLAMA_VOCAB_PRE_TYPE_QWEN35
    // - LLAMA_VOCAB_PRE_TYPE_TEKKEN
    //
    // Notes on Java port:
    // - Escaping adjusted for Java string literals.
    // - The expressions are kept semantically equivalent to the C++ definitions.

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
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";
    private static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };
    private static final String[] REFACT_PATTERNS = {
        "\\p{N}", "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
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
    public static final Splitter QWEN35 = FastQwen35Splitter.INSTANCE;
    public static final Splitter SMOLLM2 = sequence(SMOLLM2_PATTERNS);
    public static final Splitter REFACT = sequence(REFACT_PATTERNS);
    public static final Splitter TEKKEN = FastTekkenSplitter.INSTANCE;
    public static final Splitter MISTRAL_TEKKEN = TEKKEN;
    public static final Splitter DEEPSEEK_LATEST = QWEN35;
    public static final Splitter KIMI_25 = QWEN35;
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
        switch (normalized) {
            case "llama":
            case "llama3":
            case "llama-bpe":
            case "mistral":
            case "mixtral":
            case "mistral-llama":
            case "phi":
            case "phi3":
            case "phi4":
            case "dbrx":
                return LLAMA3;
            case "mistral-tekken":
            case "mistral_nemo":
            case "ministral":
                return MISTRAL_TEKKEN;
            case "qwen":
            case "qwen2":
            case "qwen3":
                return QWEN2;
            case "qwen3.5":
            case "qwen3_5":
                return QWEN35;
            case "deepseek":
            case "deepseek-v3":
            case "deepseek_r1":
            case "deepseek-r1":
                return DEEPSEEK_LATEST;
            case "kimi":
            case "kimi-2.5":
            case "kimi_2_5":
                return KIMI_25;
            case "smollm":
            case "smollm2":
            case "smollm3":
                return SMOLLM2;
            case "tekken":
                return TEKKEN;
            case "refact":
            case "granite":
            case "granite3":
            case "granite4":
                return REFACT;
            case "gemma":
            case "gemma2":
            case "gemma3":
            case "gemma4":
                return IDENTITY;
            case "default":
                return DEFAULT_BPE;
            default:
                return null;
        }
    }

    private static Splitter sequence(String[] patterns) {
        return Splitter.sequence(
                Arrays.stream(patterns).map(Splitter::regex).toArray(Splitter[]::new));
    }
}
