package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.gguf.TestDataManager.TestModel;
import com.qxotic.toknroll.gguf.TestDataManager.TokenizerMetadata;
import com.qxotic.toknroll.impl.ImplAccessor;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@Tag("network")
@Tag("local-external")
class GGUFFamilyRecreationTest {

    private static final List<String> SMOKE_TEXTS =
            List.of(
                    "Hello world",
                    "Tokenizer family validation",
                    "Whitespace\n\tand unicode 😀",
                    "cafe and symbols <>[]{}",
                    "ไทยภาษาไทย without spaces",
                    "العربية mixed English 123");

    private final TestDataManager data = new TestDataManager();

    @ParameterizedTest(name = "recreate tokenizer from GGUF metadata {0}")
    @EnumSource(TestModel.class)
    void recreateTokenizerFromMetadata(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = data.getOrDownloadMetadata(model);
        TokenizerMetadata metadata = TestDataManager.extractTokenizerMetadata(gguf);

        Tokenizer tokenizer = buildTokenizer(model, metadata);
        assertNotNull(tokenizer);
        assertTrue(tokenizer.vocabulary().size() > 0);

        for (String text : SMOKE_TEXTS) {
            IntSequence tokens = tokenizer.encode(text);
            assertEquals(
                    tokens.length(), tokenizer.countTokens(text), model.name() + " count parity");
            assertFalse(
                    tokens.isEmpty(), model.name() + " produced empty tokens for non-empty text");
            for (int i = 0; i < tokens.length(); i++) {
                assertTrue(
                        tokenizer.vocabulary().contains(tokens.intAt(i)),
                        model.name() + " unknown token id at index " + i);
            }
            assertNotNull(tokenizer.decode(tokens), model.name() + " decode result");
        }
    }

    private static Tokenizer buildTokenizer(TestModel model, TokenizerMetadata metadata) {
        Map<String, Integer> tokenToId = new LinkedHashMap<>();
        String[] tokens = metadata.tokens();
        for (int i = 0; i < tokens.length; i++) {
            tokenToId.put(tokens[i], i);
        }

        Vocabulary vocabulary =
                ImplAccessor.createVocabulary(metadata.tokens(), metadata.tokenTypes());
        List<Tokenizers.MergeRule> merges = buildMergeRules(metadata, tokenToId);

        TokenizationModel tokenizationModel =
                createModel(vocabulary, merges, metadata.isSentencePiece());

        Splitter splitter = ModelTextSplitters.createSplitter(model);
        return TokenizationPipeline.builder(tokenizationModel).splitter(splitter).build();
    }

    private static TokenizationModel createModel(
            Vocabulary vocabulary,
            List<Tokenizers.MergeRule> merges,
            boolean sentencePiecePreferred) {
        if (sentencePiecePreferred) {
            return Tokenizers.sentencePieceBpeModel(vocabulary, merges);
        }
        try {
            return Tokenizers.tiktokenModel(vocabulary, merges);
        } catch (IllegalArgumentException e) {
            return Tokenizers.sentencePieceBpeModel(vocabulary, merges);
        }
    }

    private static List<Tokenizers.MergeRule> buildMergeRules(
            TokenizerMetadata metadata, Map<String, Integer> tokenToId) {
        List<Tokenizers.MergeRule> merges = new ArrayList<>();
        String[] mergeSpecs = metadata.merges();
        if (mergeSpecs == null) {
            return merges;
        }
        for (int rank = 0; rank < mergeSpecs.length; rank++) {
            String[] parts = mergeSpecs[rank].split(" ");
            if (parts.length != 2) {
                continue;
            }
            Integer leftId = tokenToId.get(parts[0]);
            Integer rightId = tokenToId.get(parts[1]);
            Integer mergedId = tokenToId.get(parts[0] + parts[1]);
            if (leftId != null && rightId != null && mergedId != null) {
                merges.add(new Tokenizers.MergeRule(leftId, rightId, rank));
            }
        }
        return merges;
    }
}
