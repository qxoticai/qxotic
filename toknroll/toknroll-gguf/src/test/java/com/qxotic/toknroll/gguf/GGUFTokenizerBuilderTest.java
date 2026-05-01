package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.gguf.TestDataManager.TestModel;
import com.qxotic.toknroll.gguf.TestDataManager.TokenizerMetadata;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@Tag("network")
@Tag("local-external")
public class GGUFTokenizerBuilderTest {

    private static TestDataManager dataManager;

    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }

    @ParameterizedTest(name = "Build vocabulary from {0}")
    @EnumSource(TestModel.class)
    void testBuildVocabulary(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }

        Vocabulary vocabulary = ImplAccessor.createVocabulary(tokenToId);

        assertEquals(
                tokenizerMeta.vocabularySize(), vocabulary.size(), "Vocabulary size should match");

        for (int i = 0; i < Math.min(100, tokenizerMeta.tokens().length); i++) {
            String token = tokenizerMeta.tokens()[i];
            assertTrue(vocabulary.contains(token), "Vocabulary should contain token: " + token);
            assertEquals(i, vocabulary.id(token), "Token ID should match for: " + token);
            assertEquals(token, vocabulary.token(i), "Token string should match for ID: " + i);
        }
    }

    @ParameterizedTest(name = "Build BPE tokenizer from {0}")
    @EnumSource(TestModel.class)
    void testBuildBpeTokenizer(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        if (!tokenizerMeta.isBpe()) {
            return;
        }

        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }

        Vocabulary vocabulary = ImplAccessor.createVocabulary(tokenToId);

        List<Toknroll.MergeRule> merges = buildMergeRules(tokenizerMeta, tokenToId);

        Splitter splitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer =
                buildTokenizer(vocabulary, merges, splitter, tokenizerMeta.isSentencePiece());

        assertNotNull(tokenizer, "Tokenizer should be created");
        assertNotNull(tokenizer.vocabulary(), "Tokenizer should have vocabulary");
        assertEquals(
                vocabulary.size(),
                tokenizer.vocabulary().size(),
                "Tokenizer vocabulary size should match");
    }

    @ParameterizedTest(name = "Test basic encoding with {0}")
    @EnumSource(TestModel.class)
    void testBasicEncoding(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        if (!tokenizerMeta.isBpe()) {
            return;
        }

        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }

        Vocabulary vocabulary = ImplAccessor.createVocabulary(tokenToId);

        Splitter splitter = ModelTextSplitters.createSplitter(model);
        Tokenizer tokenizer =
                buildTokenizer(vocabulary, List.of(), splitter, tokenizerMeta.isSentencePiece());

        String testText = "Hi";
        IntSequence tokens = tokenizer.encode(testText);

        assertNotNull(tokens, "Encoded tokens should not be null");
        assertTrue(tokens.length() > 0, "Should produce at least one token");

        for (int i = 0; i < tokens.length(); i++) {
            int tokenId = tokens.intAt(i);
            assertTrue(
                    tokenizer.vocabulary().contains(tokenId),
                    "Token ID " + tokenId + " should be in vocabulary");
        }
    }

    @ParameterizedTest(name = "Verify special tokens for {0}")
    @EnumSource(TestModel.class)
    void testSpecialTokenVerification(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        for (int i = 0; i < tokenizerMeta.tokens().length; i++) {
            tokenToId.put(tokenizerMeta.tokens()[i], i);
        }

        Vocabulary vocabulary = ImplAccessor.createVocabulary(tokenToId);

        if (tokenizerMeta.bosTokenId() != null) {
            int bosId = tokenizerMeta.bosTokenId();
            assertTrue(vocabulary.contains(bosId), "BOS token ID should be in vocabulary");
            assertNotNull(vocabulary.token(bosId), "BOS token string should not be null");
        }

        if (tokenizerMeta.eosTokenId() != null) {
            int eosId = tokenizerMeta.eosTokenId();
            assertTrue(vocabulary.contains(eosId), "EOS token ID should be in vocabulary");
            assertNotNull(vocabulary.token(eosId), "EOS token string should not be null");
        }

        if (tokenizerMeta.padTokenId() != null) {
            int padId = tokenizerMeta.padTokenId();
            assertTrue(vocabulary.contains(padId), "PAD token ID should be in vocabulary");
            assertNotNull(vocabulary.token(padId), "PAD token string should not be null");
        }
    }

    private static List<Toknroll.MergeRule> buildMergeRules(
            TokenizerMetadata tokenizerMeta, Map<String, Integer> tokenToId) {
        List<Toknroll.MergeRule> merges = new ArrayList<>();
        if (tokenizerMeta.merges() == null) {
            return merges;
        }
        for (int rank = 0; rank < tokenizerMeta.merges().length; rank++) {
            String[] parts = tokenizerMeta.merges()[rank].split(" ");
            if (parts.length != 2) {
                continue;
            }
            Integer leftId = tokenToId.get(parts[0]);
            Integer rightId = tokenToId.get(parts[1]);
            Integer mergedId = tokenToId.get(parts[0] + parts[1]);
            if (leftId != null && rightId != null && mergedId != null) {
                merges.add(Toknroll.MergeRule.of(leftId, rightId, rank));
            }
        }
        return merges;
    }

    private static Tokenizer buildTokenizer(
            Vocabulary vocabulary,
            List<Toknroll.MergeRule> merges,
            Splitter splitter,
            boolean sentencePiecePreferred) {
        try {
            if (sentencePiecePreferred) {
                return Toknroll.pipeline(
                        splitter, Toknroll.sentencePieceBpeModel(vocabulary, merges));
            }
            return Toknroll.pipeline(splitter, Toknroll.tiktokenModel(vocabulary, merges));
        } catch (IllegalArgumentException e) {
            return Toknroll.pipeline(splitter, Toknroll.sentencePieceBpeModel(vocabulary, merges));
        }
    }
}
