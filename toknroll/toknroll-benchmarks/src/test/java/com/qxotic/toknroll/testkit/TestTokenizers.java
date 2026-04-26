package com.qxotic.toknroll.testkit;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import com.qxotic.toknroll.testkit.FamilyGoldenFixture.Family;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Single entry point for tokenizer construction in tests.
 *
 * <p>This keeps model-family and tiktoken creation logic centralized so tests do not each wire
 * tokenizers slightly differently.
 */
public final class TestTokenizers {

    private static final FamilyGoldenFixture FAMILY_FIXTURE = FamilyGoldenFixture.load();
    private static final Map<String, HfSpec> HF_ALIAS = buildAliasMap();

    private TestTokenizers() {}

    public static Optional<Tokenizer> modelFamily(String familyId) {
        HfSpec spec = resolveFamilySpec(familyId);
        if (spec == null) {
            return Optional.empty();
        }
        return loadFromHf(spec);
    }

    public static Optional<Tokenizer> modelFamilyFast(String familyId) {
        return modelFamily(familyId);
    }

    public static Optional<Tokenizer> modelFamilyFromHf(
            String familyId, String hfModelRef, String hfRevision) {
        String modelRef = hfModelRef;
        String revision = hfRevision;
        if (modelRef == null || modelRef.isBlank()) {
            HfSpec fallback = resolveFamilySpec(familyId);
            if (fallback == null) {
                return Optional.empty();
            }
            modelRef = fallback.modelRef();
            if (revision == null || revision.isBlank()) {
                revision = fallback.revision();
            }
        }

        HfSpec spec = fromModelRef(modelRef, revision);
        if (spec == null) {
            return Optional.empty();
        }
        return loadFromHf(spec);
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

    private static Optional<Tokenizer> loadFromHf(HfSpec spec) {
        try {
            return Optional.of(
                    HuggingFaceTokenizerLoader.fromHuggingFace(
                            spec.user(), spec.repository(), spec.revision(), false, false));
        } catch (RuntimeException e) {
            return Optional.empty();
        }
    }

    private static HfSpec resolveFamilySpec(String familyId) {
        String normalized = normalizeFamilyId(familyId);
        Family family = FAMILY_FIXTURE.families().get(normalized);
        if (family != null && family.modelRef() != null && !family.modelRef().isBlank()) {
            HfSpec fromFixture = fromModelRef(family.modelRef(), family.revision());
            if (fromFixture != null) {
                return fromFixture;
            }
        }
        return HF_ALIAS.get(normalized);
    }

    private static String normalizeFamilyId(String familyId) {
        if (familyId == null) {
            return null;
        }
        if ("mistral-v03-spbpe".equals(familyId)) {
            return "mistral.v0_3_spbpe";
        }
        if ("gpt-oss".equals(familyId)) {
            return "openai.gpt-oss";
        }
        if ("mistral-tekken".equals(familyId)) {
            return "mistral.tekken";
        }
        if ("gemma4".equals(familyId)) {
            return "google.gemma4";
        }
        if ("llama3".equals(familyId)) {
            return "meta.llama3";
        }
        if ("qwen35".equals(familyId)) {
            return "alibaba.qwen3_5";
        }
        return familyId;
    }

    private static HfSpec fromModelRef(String modelRef, String revision) {
        if (modelRef == null || modelRef.isBlank()) {
            return null;
        }
        String[] parts = modelRef.split("/", 2);
        if (parts.length != 2 || parts[0].isBlank() || parts[1].isBlank()) {
            return null;
        }
        String resolvedRevision = revision == null || revision.isBlank() ? "main" : revision;
        return new HfSpec(parts[0], parts[1], resolvedRevision);
    }

    private static Map<String, HfSpec> buildAliasMap() {
        Map<String, HfSpec> map = new LinkedHashMap<>();
        map.put(
                "mistral.tekken",
                Objects.requireNonNull(
                        fromModelRef(
                                "mistralai/ministral-8b-instruct-2410",
                                "2f494a194c5b980dfb9772cb92d26cbb671fce5a")));
        map.put(
                "openai.gpt-oss",
                Objects.requireNonNull(fromModelRef("openai/gpt-oss-20b", "main")));
        map.put(
                "mistral.v0_3_spbpe",
                Objects.requireNonNull(fromModelRef("mistralai/Mistral-7B-Instruct-v0.3", "main")));
        return map;
    }

    private record HfSpec(String user, String repository, String revision) {
        private HfSpec {
            Objects.requireNonNull(user, "user");
            Objects.requireNonNull(repository, "repository");
            Objects.requireNonNull(revision, "revision");
        }

        private String modelRef() {
            return user + "/" + repository;
        }
    }
}
