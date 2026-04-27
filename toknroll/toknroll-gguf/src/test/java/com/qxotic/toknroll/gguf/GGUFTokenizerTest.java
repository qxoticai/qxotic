package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.toknroll.gguf.TestDataManager.TestModel;
import com.qxotic.toknroll.gguf.TestDataManager.TokenizerMetadata;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

@Tag("network")
@Tag("local-external")
public class GGUFTokenizerTest {

    private static TestDataManager dataManager;

    @BeforeAll
    static void setUp() {
        dataManager = new TestDataManager();
    }

    @ParameterizedTest(name = "Load metadata for {0}")
    @EnumSource(TestModel.class)
    void testLoadMetadata(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);

        assertNotNull(gguf, "GGUF metadata should not be null");
        assertTrue(gguf.getVersion() >= 2, "GGUF version should be 2 or higher");
        assertFalse(gguf.getMetadataKeys().isEmpty(), "Metadata should not be empty");
    }

    @ParameterizedTest(name = "Extract vocabulary from {0}")
    @EnumSource(TestModel.class)
    void testExtractVocabulary(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        assertNotNull(tokenizerMeta, "Tokenizer metadata should not be null");
        assertTrue(
                tokenizerMeta.vocabularySize() > 0,
                "Vocabulary should not be empty for " + model.name());
        assertNotNull(tokenizerMeta.tokens(), "Tokens array should not be null");

        for (int i = 0; i < Math.min(10, tokenizerMeta.tokens().length); i++) {
            assertNotNull(tokenizerMeta.tokens()[i], "Token at index " + i + " should not be null");
        }
    }

    @ParameterizedTest(name = "Detect tokenizer type for {0}")
    @EnumSource(TestModel.class)
    void testTokenizerTypeDetection(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        if (model.name().contains("GEMMA")) {
            assertTrue(
                    tokenizerMeta.isSentencePiece()
                            || tokenizerMeta.modelType().toLowerCase().contains("gemma")
                            || tokenizerMeta.modelType().equalsIgnoreCase("spm"),
                    model.name()
                            + " should use SentencePiece tokenizer type, but was: "
                            + tokenizerMeta.modelType());
        } else if (model.name().contains("MISTRAL")) {
            assertTrue(
                    tokenizerMeta.isBpe()
                            || tokenizerMeta.modelType().equalsIgnoreCase("llama")
                            || tokenizerMeta.modelType().equalsIgnoreCase("gpt2"),
                    model.name()
                            + " should use BPE/llama tokenizer type, but was: "
                            + tokenizerMeta.modelType());
        } else {
            assertTrue(
                    tokenizerMeta.isBpe()
                            || tokenizerMeta.modelType().equalsIgnoreCase("gpt2")
                            || tokenizerMeta.modelType().equalsIgnoreCase("qwen"),
                    model.name()
                            + " should use BPE/gpt2/qwen tokenizer type, but was: "
                            + tokenizerMeta.modelType());
        }
    }

    @ParameterizedTest(name = "Check special tokens for {0}")
    @EnumSource(TestModel.class)
    void testSpecialTokens(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        assertNotNull(tokenizerMeta.bosTokenId(), model.name() + " should have BOS token ID");
        assertNotNull(tokenizerMeta.eosTokenId(), model.name() + " should have EOS token ID");

        assertTrue(
                tokenizerMeta.bosTokenId() >= 0
                        && tokenizerMeta.bosTokenId() < tokenizerMeta.vocabularySize(),
                "BOS token ID should be within vocabulary range");
        assertTrue(
                tokenizerMeta.eosTokenId() >= 0
                        && tokenizerMeta.eosTokenId() < tokenizerMeta.vocabularySize(),
                "EOS token ID should be within vocabulary range");
    }

    @ParameterizedTest(name = "Check BPE merges for {0}")
    @EnumSource(TestModel.class)
    void testBpeMerges(TestModel model) throws IOException, InterruptedException {
        GGUF gguf = dataManager.getOrDownloadMetadata(model);
        TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);

        if (tokenizerMeta.isBpe()) {
            assertNotNull(tokenizerMeta.merges(), "BPE tokenizer should have merges");
            assertTrue(
                    tokenizerMeta.merges().length > 0,
                    "BPE tokenizer should have at least one merge");
        }
    }

    @Test
    void testCacheFunctionality() throws IOException, InterruptedException {
        TestModel model = TestModel.QWEN3_0_6B;

        GGUF gguf1 = dataManager.getOrDownloadMetadata(model);

        GGUF gguf2 = dataManager.getOrDownloadMetadata(model);

        assertEquals(gguf1.getVersion(), gguf2.getVersion(), "Versions should match");
        assertEquals(
                gguf1.getMetadataKeys().size(),
                gguf2.getMetadataKeys().size(),
                "Metadata key count should match");
        Path expectedCacheFile =
                dataManager
                        .getCachePath()
                        .resolve(TestDataManager.cacheFileNameForUrl(model.getHuggingFaceUrl()));
        assertTrue(Files.exists(expectedCacheFile), "Expected metadata file in cache");
    }

    @Test
    void printAllModelsInfo() throws IOException, InterruptedException {
        for (TestModel model : TestModel.values()) {
            GGUF gguf = dataManager.getOrDownloadMetadata(model);
            TokenizerMetadata tokenizerMeta = TestDataManager.extractTokenizerMetadata(gguf);
            assertNotNull(tokenizerMeta.modelName());
            assertTrue(tokenizerMeta.vocabularySize() > 0);
            assertTrue(gguf.getMetadataKeys().size() > 0);
        }
    }
}
