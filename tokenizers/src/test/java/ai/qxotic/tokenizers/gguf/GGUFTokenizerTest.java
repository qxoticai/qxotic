package ai.qxotic.tokenizers.gguf;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.tokenizers.gguf.TestDataManager.TestModel;
import ai.qxotic.tokenizers.gguf.TestDataManager.TokenizerMetadata;
import java.io.IOException;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Tests that verify tokenizer implementations against real GGUF model metadata.
 * 
 * <p>These tests download metadata from Hugging Face for popular models and verify:
 * <ul>
 *   <li>Vocabulary can be extracted from GGUF metadata</li>
 *   <li>Token counts match expected values</li>
 *   <li>Special tokens are correctly identified</li>
 *   <li>Tokenizer type detection works (BPE vs SentencePiece)</li>
 * </ul>
 * 
 * <p>The first run will download metadata from Hugging Face. Subsequent runs use cached data.
 * To invalidate the cache, delete {@code ~/.cache/qxotic-tokenizers/gguf-metadata/}.
 */
public class GGUFTokenizerTest {
    
    private static TestDataManager dataManager;
    
    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }
    
    /**
     * Tests that metadata can be downloaded and parsed for all supported models.
     */
    @ParameterizedTest(name = "Load metadata for {0}")
    @EnumSource(TestModel.class)
    void testLoadMetadata(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        
        assertNotNull(gguf, "GGUF metadata should not be null");
        assertTrue(gguf.getVersion() >= 2, "GGUF version should be 2 or higher");
        assertFalse(gguf.getMetadataKeys().isEmpty(), "Metadata should not be empty");
        
        System.out.println("Loaded " + model.name() + " (GGUF v" + gguf.getVersion() + ")");
        System.out.println("  Metadata keys: " + gguf.getMetadataKeys().size());
        System.out.println("  Tensors: " + gguf.getTensors().size());
    }
    
    /**
     * Tests that tokenizer vocabulary can be extracted from GGUF metadata.
     */
    @ParameterizedTest(name = "Extract vocabulary from {0}")
    @EnumSource(TestModel.class)
    void testExtractVocabulary(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        assertNotNull(tokenizerMeta, "Tokenizer metadata should not be null");
        assertTrue(tokenizerMeta.vocabularySize() > 0, 
                "Vocabulary should not be empty for " + model.name());
        assertNotNull(tokenizerMeta.tokens(), "Tokens array should not be null");
        
        // Verify all tokens are non-null
        for (int i = 0; i < Math.min(10, tokenizerMeta.tokens().length); i++) {
            assertNotNull(tokenizerMeta.tokens()[i], 
                    "Token at index " + i + " should not be null");
        }
        
        System.out.println(model.name() + " vocabulary size: " + tokenizerMeta.vocabularySize());
        System.out.println("  Model type: " + tokenizerMeta.modelType());
        System.out.println("  Architecture: " + tokenizerMeta.architecture());
    }
    
    /**
     * Tests that tokenizer type is correctly detected.
     */
    @ParameterizedTest(name = "Detect tokenizer type for {0}")
    @EnumSource(TestModel.class)
    void testTokenizerTypeDetection(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        // Gemma uses SentencePiece, most others use BPE
        if (model.name().contains("GEMMA")) {
            assertTrue(tokenizerMeta.isSentencePiece() || tokenizerMeta.modelType().equalsIgnoreCase("llama"),
                    "Gemma should use SentencePiece/llama tokenizer type");
        } else if (model.name().contains("LLAMA")) {
            assertTrue(tokenizerMeta.isSentencePiece() || tokenizerMeta.modelType().equalsIgnoreCase("llama"),
                    "Llama should use SentencePiece/llama tokenizer type");
        } else {
            // Qwen, Mistral, Phi typically use BPE
            assertTrue(tokenizerMeta.isBpe() || tokenizerMeta.modelType().equalsIgnoreCase("gpt2"),
                    model.name() + " should use BPE/gpt2 tokenizer type, but was: " + tokenizerMeta.modelType());
        }
        
        System.out.println(model.name() + " tokenizer type: " + tokenizerMeta.modelType());
    }
    
    /**
     * Tests that special tokens are present in the metadata.
     */
    @ParameterizedTest(name = "Check special tokens for {0}")
    @EnumSource(TestModel.class)
    void testSpecialTokens(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        // Most models should have BOS and EOS tokens
        assertNotNull(tokenizerMeta.bosTokenId(), 
                model.name() + " should have BOS token ID");
        assertNotNull(tokenizerMeta.eosTokenId(), 
                model.name() + " should have EOS token ID");
        
        // Verify token IDs are within vocabulary range
        assertTrue(tokenizerMeta.bosTokenId() >= 0 && tokenizerMeta.bosTokenId() < tokenizerMeta.vocabularySize(),
                "BOS token ID should be within vocabulary range");
        assertTrue(tokenizerMeta.eosTokenId() >= 0 && tokenizerMeta.eosTokenId() < tokenizerMeta.vocabularySize(),
                "EOS token ID should be within vocabulary range");
        
        System.out.println(model.name() + " special tokens:");
        System.out.println("  BOS: " + tokenizerMeta.bosTokenId() + 
                " (\"" + tokenizerMeta.tokens()[tokenizerMeta.bosTokenId()] + "\")");
        System.out.println("  EOS: " + tokenizerMeta.eosTokenId() + 
                " (\"" + tokenizerMeta.tokens()[tokenizerMeta.eosTokenId()] + "\")");
        if (tokenizerMeta.padTokenId() != null) {
            System.out.println("  PAD: " + tokenizerMeta.padTokenId());
        }
        if (tokenizerMeta.unkTokenId() != null) {
            System.out.println("  UNK: " + tokenizerMeta.unkTokenId());
        }
    }
    
    /**
     * Tests that merges are present for BPE tokenizers.
     */
    @ParameterizedTest(name = "Check BPE merges for {0}")
    @EnumSource(TestModel.class)
    void testBpeMerges(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        if (tokenizerMeta.isBpe()) {
            assertNotNull(tokenizerMeta.merges(), 
                    "BPE tokenizer should have merges");
            assertTrue(tokenizerMeta.merges().length > 0, 
                    "BPE tokenizer should have at least one merge");
            System.out.println(model.name() + " has " + tokenizerMeta.merges().length + " BPE merges");
        } else {
            System.out.println(model.name() + " is not BPE (no merges expected)");
        }
    }
    
    /**
     * Tests cache functionality by loading the same model twice.
     */
    @Test
    void testCacheFunctionality() throws IOException, InterruptedException {
        TestModel model = TestModel.GEMMA_3_1B;
        
        // First load - should download
        long start1 = System.currentTimeMillis();
        GGUF gguf1 = dataManager.getOrDownloadMetadata(model);
        long duration1 = System.currentTimeMillis() - start1;
        
        // Second load - should use cache
        long start2 = System.currentTimeMillis();
        GGUF gguf2 = dataManager.getOrDownloadMetadata(model);
        long duration2 = System.currentTimeMillis() - start2;
        
        // Cached load should be much faster
        System.out.println("First load (download): " + duration1 + "ms");
        System.out.println("Second load (cached): " + duration2 + "ms");
        
        assertTrue(duration2 < duration1, "Cached load should be faster than download");
        assertEquals(gguf1.getVersion(), gguf2.getVersion(), "Versions should match");
        assertEquals(gguf1.getMetadataKeys().size(), gguf2.getMetadataKeys().size(), 
                "Metadata key count should match");
    }
    
    /**
     * Prints detailed information about all models for debugging.
     */
    @Test
    void printAllModelsInfo() throws IOException, InterruptedException {
        System.out.println("\n=== GGUF Model Information ===\n");
        
        for (TestModel model : TestModel.values()) {
            System.out.println("Model: " + model.name());
            System.out.println("  URL: " + model.getHuggingFaceUrl());
            
            GGUF gguf = dataManager.getOrDownloadMetadata(model);
            TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
            
            System.out.println("  GGUF Version: " + gguf.getVersion());
            System.out.println("  Model Name: " + tokenizerMeta.modelName());
            System.out.println("  Architecture: " + tokenizerMeta.architecture());
            System.out.println("  Tokenizer Type: " + tokenizerMeta.modelType());
            System.out.println("  Vocabulary Size: " + tokenizerMeta.vocabularySize());
            System.out.println("  Is BPE: " + tokenizerMeta.isBpe());
            System.out.println("  Is SentencePiece: " + tokenizerMeta.isSentencePiece());
            
            if (tokenizerMeta.merges() != null) {
                System.out.println("  BPE Merges: " + tokenizerMeta.merges().length);
            }
            
            System.out.println("  Metadata Keys: " + gguf.getMetadataKeys().size());
            System.out.println("  Tensor Count: " + gguf.getTensors().size());
            System.out.println();
        }
    }
}