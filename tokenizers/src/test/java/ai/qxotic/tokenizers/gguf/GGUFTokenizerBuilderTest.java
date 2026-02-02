package ai.qxotic.tokenizers.gguf;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.tokenizers.*;
import ai.qxotic.tokenizers.gguf.TestDataManager.TestModel;
import ai.qxotic.tokenizers.gguf.TestDataManager.TokenizerMetadata;
import ai.qxotic.tokenizers.impl.*;
import java.io.IOException;
import java.util.*;
import java.util.regex.Pattern;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Tests that build and verify tokenizers from real GGUF model metadata.
 * 
 * <p>These tests extract vocabulary and configuration from downloaded GGUF files
 * and use them to instantiate tokenizer implementations for testing.
 */
public class GGUFTokenizerBuilderTest {
    
    private static TestDataManager dataManager;
    
    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }
    
    /**
     * Builds a vocabulary from GGUF metadata and verifies basic operations.
     */
    @ParameterizedTest(name = "Build vocabulary from {0}")
    @EnumSource(TestModel.class)
    void testBuildVocabulary(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        // Build vocabulary from GGUF tokens
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }
        
        Vocabulary vocabulary = new VocabularyImpl(tokenToId);
        
        // Verify vocabulary size
        assertEquals(tokenizerMeta.vocabularySize(), vocabulary.size(),
                "Vocabulary size should match");
        
        // Verify we can look up tokens
        for (int i = 0; i < Math.min(100, tokenizerMeta.tokens().length); i++) {
            String token = tokenizerMeta.tokens()[i];
            assertTrue(vocabulary.contains(token), 
                    "Vocabulary should contain token: " + token);
            assertEquals(i, vocabulary.id(token),
                    "Token ID should match for: " + token);
            assertEquals(token, vocabulary.token(i),
                    "Token string should match for ID: " + i);
        }
        
        System.out.println(model.name() + " vocabulary built successfully");
        System.out.println("  Size: " + vocabulary.size());
    }
    
    /**
     * Tests building a BPE tokenizer from GGUF metadata for BPE-based models.
     */
    @ParameterizedTest(name = "Build BPE tokenizer from {0}")
    @EnumSource(TestModel.class)
    void testBuildBpeTokenizer(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        if (!tokenizerMeta.isBpe()) {
            System.out.println(model.name() + " is not BPE, skipping BPE tokenizer test");
            return;
        }
        
        // Build vocabulary
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }
        
        // Build special tokens map
        Map<String, Integer> specialTokens = new HashMap<>();
        if (tokenizerMeta.bosTokenId() != null) {
            specialTokens.put(tokenizerMeta.tokens()[tokenizerMeta.bosTokenId()], tokenizerMeta.bosTokenId());
        }
        if (tokenizerMeta.eosTokenId() != null) {
            specialTokens.put(tokenizerMeta.tokens()[tokenizerMeta.eosTokenId()], tokenizerMeta.eosTokenId());
        }
        if (tokenizerMeta.padTokenId() != null) {
            specialTokens.put(tokenizerMeta.tokens()[tokenizerMeta.padTokenId()], tokenizerMeta.padTokenId());
        }
        if (tokenizerMeta.unkTokenId() != null) {
            specialTokens.put(tokenizerMeta.tokens()[tokenizerMeta.unkTokenId()], tokenizerMeta.unkTokenId());
        }
        
        // Create vocabulary (special tokens are included in the main vocabulary for GGUF)
        Vocabulary vocabulary = new VocabularyImpl(tokenToId);
        
        // Build merge ranks from merges
        Map<IntPair, GPT2Tokenizer.MergeRank> merges = new HashMap<>();
        if (tokenizerMeta.merges() != null) {
            for (int rank = 0; rank < tokenizerMeta.merges().length; rank++) {
                String merge = tokenizerMeta.merges()[rank];
                String[] parts = merge.split(" ");
                if (parts.length == 2) {
                    Integer leftId = tokenToId.get(parts[0]);
                    Integer rightId = tokenToId.get(parts[1]);
                    if (leftId != null && rightId != null) {
                        String merged = parts[0] + parts[1];
                        Integer mergedId = tokenToId.get(merged);
                        if (mergedId != null) {
                            merges.put(new IntPair(leftId, rightId), 
                                    new GPT2Tokenizer.MergeRank(mergedId, rank));
                        }
                    }
                }
            }
        }
        
        // Create GPT2-style tokenizer with model-specific text splitter
        TextSplitter textSplitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer = new GPT2Tokenizer(
                vocabulary,
                Normalizer.IDENTITY,
                textSplitter,
                merges
        );
        
        assertNotNull(tokenizer, "Tokenizer should be created");
        assertNotNull(tokenizer.vocabulary(), "Tokenizer should have vocabulary");
        assertEquals(vocabulary.size(), tokenizer.vocabulary().size(),
                "Tokenizer vocabulary size should match");
        
        System.out.println(model.name() + " BPE tokenizer built successfully");
        System.out.println("  Vocabulary: " + vocabulary.size());
        System.out.println("  Merges: " + merges.size());
        System.out.println("  Special tokens: " + specialTokens.size());
    }
    
    /**
     * Tests basic encoding/decoding with a simple text for models where we can build tokenizers.
     */
    @ParameterizedTest(name = "Test basic encoding with {0}")
    @EnumSource(TestModel.class)
    void testBasicEncoding(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        // Skip if we can't build a tokenizer for this model type
        if (!tokenizerMeta.isBpe()) {
            System.out.println(model.name() + " uses " + tokenizerMeta.modelType() + 
                    " tokenizer - skipping encoding test (BPE only for now)");
            return;
        }
        
        // Build simple vocabulary
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }
        
        Vocabulary vocabulary = new VocabularyImpl(tokenToId);
        
        // Create a simple tokenizer with model-specific text splitter
        TextSplitter textSplitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer = new GPT2Tokenizer(
                vocabulary,
                Normalizer.IDENTITY,
                textSplitter,
                Collections.emptyMap() // No merges for basic test
        );
        
        // Test encoding a simple string character by character
        String testText = "Hi";
        IntSequence tokens = tokenizer.encode(testText);
        
        assertNotNull(tokens, "Encoded tokens should not be null");
        assertTrue(tokens.length() > 0, "Should produce at least one token");
        
        // Verify all tokens are in vocabulary
        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            assertTrue(tokenizer.vocabulary().contains(tokenId),
                    "Token ID " + tokenId + " should be in vocabulary");
        }
        
        System.out.println(model.name() + " encoding test passed");
        System.out.println("  Input: \"" + testText + "\"");
        System.out.println("  Tokens: " + tokens);
    }
    
    /**
     * Verifies that special token IDs from GGUF match the vocabulary.
     */
    @ParameterizedTest(name = "Verify special tokens for {0}")
    @EnumSource(TestModel.class)
    void testSpecialTokenVerification(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
        
        // Build vocabulary
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }
        
        Vocabulary vocabulary = new VocabularyImpl(tokenToId);
        
        // Verify BOS token
        if (tokenizerMeta.bosTokenId() != null) {
            int bosId = tokenizerMeta.bosTokenId();
            assertTrue(vocabulary.contains(bosId), "BOS token ID should be in vocabulary");
            String bosToken = vocabulary.token(bosId);
            assertNotNull(bosToken, "BOS token string should not be null");
            System.out.println(model.name() + " BOS: \"" + bosToken + "\" (ID: " + bosId + ")");
        }
        
        // Verify EOS token
        if (tokenizerMeta.eosTokenId() != null) {
            int eosId = tokenizerMeta.eosTokenId();
            assertTrue(vocabulary.contains(eosId), "EOS token ID should be in vocabulary");
            String eosToken = vocabulary.token(eosId);
            assertNotNull(eosToken, "EOS token string should not be null");
            System.out.println(model.name() + " EOS: \"" + eosToken + "\" (ID: " + eosId + ")");
        }
        
        // Verify PAD token if present
        if (tokenizerMeta.padTokenId() != null) {
            int padId = tokenizerMeta.padTokenId();
            assertTrue(vocabulary.contains(padId), "PAD token ID should be in vocabulary");
            String padToken = vocabulary.token(padId);
            System.out.println(model.name() + " PAD: \"" + padToken + "\" (ID: " + padId + ")");
        }
    }
}