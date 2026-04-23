package com.qxotic.toknroll.testkit;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import java.util.Map;
import java.util.Optional;

/**
 * Single entry point for tokenizer construction in tests.
 *
 * <p>This keeps model-family and tiktoken creation logic centralized so tests do not each wire
 * tokenizers slightly differently.
 */
public final class TestTokenizers {

    private TestTokenizers() {}

    public static Optional<Tokenizer> modelFamily(String familyId) {
        return ModelFamilyTokenizers.create(familyId);
    }

    public static Optional<Tokenizer> modelFamilyFast(String familyId) {
        return ModelFamilyTokenizers.createFast(familyId);
    }

    public static Optional<Tokenizer> modelFamilyFromHf(
            String familyId, String hfModelRef, String hfRevision) {
        return ModelFamilyTokenizers.createFromHfFiles(familyId, hfModelRef, hfRevision);
    }

    public static Tokenizer tiktokenReference(String encoding) {
        return TiktokenFixtures.createJtokkitTokenizer(encoding);
    }

    public static Tokenizer tiktoken(String encoding) {
        return TiktokenFixtures.createTiktokenTokenizer(encoding);
    }

    public static Tokenizer tiktoken(String encoding, Splitter splitter) {
        return TiktokenFixtures.createTiktokenTokenizer(encoding, splitter);
    }

    public static Tokenizer tiktoken(
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens,
            Splitter splitter) {
        return TiktokenFixtures.createTiktokenTokenizer(mergeableRanks, specialTokens, splitter);
    }
}
