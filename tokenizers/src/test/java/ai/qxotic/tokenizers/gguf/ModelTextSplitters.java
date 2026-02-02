package ai.qxotic.tokenizers.gguf;

import ai.qxotic.tokenizers.TextSplitter;
import ai.qxotic.tokenizers.impl.RegexSplitter;
import java.util.Arrays;

/**
 * Model-specific text splitter patterns copied from the scratch project.
 * 
 * <p>These patterns ensure that tokenization matches exactly what the models expect.
 * Using the correct splitter is essential for 100% accurate tokenization.
 * 
 * <p>Patterns sourced from: ai.qxotic.model.llm.*TextSplitterFactory classes
 */
public final class ModelTextSplitters {
    
    private ModelTextSplitters() {
        // Utility class
    }
    
    /**
     * Llama 3 BPE pattern - used by Llama 3, Mistral, and other Llama-based models.
     * 
     * <p>Pattern breakdown:
     * <ul>
     *   <li>(?i:'s|'t|'re|'ve|'m|'ll|'d) - Contractions (case insensitive)</li>
     *   <li>[^\r\n\p{L}\p{N}]?\p{L}+ - Optional non-letter/number prefix + letters</li>
     *   <li>\p{N}{1,3} - 1-3 digits</li>
     *   <li> ?[^\s\p{L}\p{N}]+[\r\n]* - Optional space + non-letter/number chars</li>
     *   <li>\s*[\r\n]+ - Newlines</li>
     *   <li>\s+(?!\S) - Trailing whitespace</li>
     *   <li>\s+ - Other whitespace</li>
     * </ul>
     */
    public static final String LLAMA3_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    
    /**
     * Qwen2 pattern - used by Qwen2 and Qwen3 models.
     * Similar to Llama3 but handles single digits differently.
     */
    public static final String QWEN2_PATTERN =
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    
    /**
     * SmolLM2 patterns - used by SmolLM models.
     * Uses multiple patterns applied sequentially.
     */
    public static final String[] SMOLLM2_PATTERNS = {
        "\\p{N}",
        "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
    };
    
    /**
     * Gemma pattern - used by Gemma models (SentencePiece-based).
     * Gemma uses a different tokenization approach but we use Llama3 pattern
     * as a reasonable approximation for testing.
     */
    public static final String GEMMA_PATTERN = LLAMA3_PATTERN;
    
    /**
     * Phi pattern - used by Phi models (similar to Llama).
     */
    public static final String PHI_PATTERN = LLAMA3_PATTERN;
    
    /**
     * Creates a TextSplitter for the specified model type.
     * 
     * @param modelType the model architecture/type (e.g., "llama", "qwen2", "gemma")
     * @return the appropriate TextSplitter for that model
     */
    public static TextSplitter createSplitter(String modelType) {
        if (modelType == null) {
            return TextSplitter.IDENTITY;
        }
        
        String normalizedType = modelType.toLowerCase();
        
        return switch (normalizedType) {
            case "llama", "mistral", "mixtral", "phi", "phi3" -> 
                RegexSplitter.create(LLAMA3_PATTERN);
            case "qwen", "qwen2", "qwen3" -> 
                RegexSplitter.create(QWEN2_PATTERN);
            case "gemma", "gemma2", "gemma3" -> 
                createGemmaSplitter();
            case "smollm", "smollm2" -> 
                createSmolLMSplitter();
            default -> 
                TextSplitter.IDENTITY;
        };
    }
    
    /**
     * Creates a TextSplitter for a specific test model.
     * 
     * @param model the test model enum value
     * @return the appropriate TextSplitter
     */
    public static TextSplitter createSplitter(TestDataManager.TestModel model) {
        return switch (model) {
            case GEMMA_3_1B -> createGemmaSplitter();
            case QWEN3_0_6B -> RegexSplitter.create(QWEN2_PATTERN);
            case MISTRAL_SMALL_3_1 -> RegexSplitter.create(LLAMA3_PATTERN);
        };
    }
    
    /**
     * Creates a Gemma-specific splitter.
     * Gemma uses SentencePiece but we can approximate with a modified pattern.
     */
    private static TextSplitter createGemmaSplitter() {
        // Gemma uses SentencePiece which is different from BPE
        // For testing purposes, we use IDENTITY since SentencePiece
        // doesn't use regex pre-tokenization like BPE does
        return TextSplitter.IDENTITY;
    }
    
    /**
     * Creates a SmolLM-specific splitter using multiple patterns.
     */
    private static TextSplitter createSmolLMSplitter() {
        TextSplitter[] splitters = Arrays.stream(SMOLLM2_PATTERNS)
                .map(RegexSplitter::create)
                .toArray(TextSplitter[]::new);
        return TextSplitter.compose(splitters);
    }
}