package com.qxotic.tokenizers.gguf;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.tokenizers.*;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.gguf.TestDataManager.TestModel;
import com.qxotic.tokenizers.gguf.TestDataManager.TokenizerMetadata;
import com.qxotic.tokenizers.impl.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Tests that build and verify tokenizers from real GGUF model metadata.
 *
 * <p>These tests extract vocabulary and configuration from downloaded GGUF files and use them to
 * instantiate tokenizer implementations for testing.
 */
@Tag("network")
public class GGUFTokenizerBuilderTest {

    private static TestDataManager dataManager;

    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }

    /** Builds a vocabulary from GGUF metadata and verifies basic operations. */
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
        assertEquals(
                tokenizerMeta.vocabularySize(), vocabulary.size(), "Vocabulary size should match");

        // Verify we can look up tokens
        for (int i = 0; i < Math.min(100, tokenizerMeta.tokens().length); i++) {
            String token = tokenizerMeta.tokens()[i];
            assertTrue(vocabulary.contains(token), "Vocabulary should contain token: " + token);
            assertEquals(i, vocabulary.id(token), "Token ID should match for: " + token);
            assertEquals(token, vocabulary.token(i), "Token string should match for ID: " + i);
        }
    }

    /** Tests building a BPE tokenizer from GGUF metadata for BPE-based models. */
    @ParameterizedTest(name = "Build BPE tokenizer from {0}")
    @EnumSource(TestModel.class)
    void testBuildBpeTokenizer(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        if (!tokenizerMeta.isBpe()) {
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
            specialTokens.put(
                    tokenizerMeta.tokens()[tokenizerMeta.bosTokenId()], tokenizerMeta.bosTokenId());
        }
        if (tokenizerMeta.eosTokenId() != null) {
            specialTokens.put(
                    tokenizerMeta.tokens()[tokenizerMeta.eosTokenId()], tokenizerMeta.eosTokenId());
        }
        if (tokenizerMeta.padTokenId() != null) {
            specialTokens.put(
                    tokenizerMeta.tokens()[tokenizerMeta.padTokenId()], tokenizerMeta.padTokenId());
        }
        if (tokenizerMeta.unkTokenId() != null) {
            specialTokens.put(
                    tokenizerMeta.tokens()[tokenizerMeta.unkTokenId()], tokenizerMeta.unkTokenId());
        }

        // Create vocabulary (special tokens are included in the main vocabulary for GGUF)
        Vocabulary vocabulary = new VocabularyImpl(tokenToId);

        // Build merge ranks from merges
        List<long[]> keyValuePairs = new ArrayList<>();
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
                            keyValuePairs.add(
                                    new long[] {
                                        IntPair.of(leftId, rightId), IntPair.of(mergedId, rank)
                                    });
                        }
                    }
                }
            }
        }
        long[] keys = new long[keyValuePairs.size()];
        long[] values = new long[keyValuePairs.size()];
        for (int i = 0; i < keyValuePairs.size(); i++) {
            keys[i] = keyValuePairs.get(i)[0];
            values[i] = keyValuePairs.get(i)[1];
        }
        LongLongMap merges = new LongLongMap(keys, values);

        // Create GPT2-style tokenizer with model-specific pre-tokenizer
        Splitter splitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer =
                new GPT2Tokenizer(vocabulary, Normalizer.identity(), splitter, merges);

        assertNotNull(tokenizer, "Tokenizer should be created");
        assertNotNull(tokenizer.vocabulary(), "Tokenizer should have vocabulary");
        assertEquals(
                vocabulary.size(),
                tokenizer.vocabulary().size(),
                "Tokenizer vocabulary size should match");
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
            return;
        }

        // Build simple vocabulary
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }

        Vocabulary vocabulary = new VocabularyImpl(tokenToId);

        // Create a simple tokenizer with model-specific pre-tokenizer
        Splitter splitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer =
                new GPT2Tokenizer(
                        vocabulary,
                        Normalizer.identity(),
                        splitter,
                        new LongLongMap(new long[0], new long[0]) // No merges for basic test
                        );

        // Test encoding a simple string character by character
        String testText = "Hi";
        IntSequence tokens = tokenizer.encode(testText);

        assertNotNull(tokens, "Encoded tokens should not be null");
        assertTrue(tokens.length() > 0, "Should produce at least one token");

        // Verify all tokens are in vocabulary
        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            assertTrue(
                    tokenizer.vocabulary().contains(tokenId),
                    "Token ID " + tokenId + " should be in vocabulary");
        }
    }

    /** Verifies that special token IDs from GGUF match the vocabulary. */
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
        }

        // Verify EOS token
        if (tokenizerMeta.eosTokenId() != null) {
            int eosId = tokenizerMeta.eosTokenId();
            assertTrue(vocabulary.contains(eosId), "EOS token ID should be in vocabulary");
            String eosToken = vocabulary.token(eosId);
            assertNotNull(eosToken, "EOS token string should not be null");
        }

        // Verify PAD token if present
        if (tokenizerMeta.padTokenId() != null) {
            int padId = tokenizerMeta.padTokenId();
            assertTrue(vocabulary.contains(padId), "PAD token ID should be in vocabulary");
            assertNotNull(vocabulary.token(padId), "PAD token string should not be null");
        }
    }
}
