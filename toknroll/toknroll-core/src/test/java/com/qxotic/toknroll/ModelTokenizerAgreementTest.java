package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.TestCorpora;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TokenizerAdapters;
import java.text.Normalizer.Form;
import java.util.List;
import org.junit.jupiter.api.Test;

class ModelTokenizerAgreementTest {

    private static final String LLAMA3_HF_MODEL_REF = "unsloth/Llama-3.2-1B-Instruct";
    private static final String LLAMA3_HF_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c";
    private static final String QWEN35_HF_MODEL_REF = "Qwen/Qwen3.5-0.8B";
    private static final String MISTRAL_TEKKEN_HF_MODEL_REF =
            "mistralai/ministral-8b-instruct-2410";
    private static final String MISTRAL_TEKKEN_HF_REVISION =
            "2f494a194c5b980dfb9772cb92d26cbb671fce5a";
    private static final List<String> MODELS =
            List.of("gpt2", "llama3", "qwen35", "mistral-tekken");

    @Test
    void comparedTokenizersAgreeOnEncoding() {
        for (String model : MODELS) {
            ComparedTokenizers compared = comparedTokenizersForModel(model);
            for (String text : TestCorpora.MODEL_TOKENIZER_AGREEMENT_TEXTS) {
                int[] reference = compared.reference().encodeToArray(text);
                int[] fidelity = compared.fidelity().encodeToArray(text);
                int[] fast = compared.fast().encodeToArray(text);

                assertArrayEquals(
                        reference, fidelity, "reference!=fidelity for " + model + " on: " + text);
                assertArrayEquals(reference, fast, "reference!=fast for " + model + " on: " + text);
                assertEquals(
                        reference.length,
                        compared.reference().countTokens(text),
                        "countTokens mismatch reference");
                assertEquals(
                        fidelity.length,
                        compared.fidelity().countTokens(text),
                        "countTokens mismatch fidelity");
                assertEquals(
                        fast.length,
                        compared.fast().countTokens(text),
                        "countTokens mismatch fast");
            }
        }
    }

    private static ComparedTokenizers comparedTokenizersForModel(String model) {
        if ("gpt2".equals(model)) {
            return new ComparedTokenizers(
                    TestTokenizers.tiktoken("r50k_base"),
                    TestTokenizers.tiktoken("r50k_base"),
                    TestTokenizers.tiktoken("r50k_base", FastSplitters.r50k()));
        }
        if ("llama3".equals(model)) {
            return modelNative(
                    "meta.llama3",
                    LLAMA3_HF_MODEL_REF,
                    LLAMA3_HF_REVISION,
                    Normalizer.identity(),
                    FastSplitters.llama3());
        }
        if ("qwen35".equals(model)) {
            return modelNative(
                    "alibaba.qwen3_5",
                    QWEN35_HF_MODEL_REF,
                    null,
                    Normalizer.unicode(Form.NFC),
                    FastSplitters.qwen35());
        }
        if ("mistral-tekken".equals(model)) {
            return modelNative(
                    "mistral.tekken",
                    MISTRAL_TEKKEN_HF_MODEL_REF,
                    MISTRAL_TEKKEN_HF_REVISION,
                    Normalizer.identity(),
                    ModelSplitters.TEKKEN);
        }
        throw new IllegalArgumentException("Unsupported model: " + model);
    }

    private static ComparedTokenizers modelNative(
            String familyId,
            String hfModelRef,
            String hfRevision,
            Normalizer normalizer,
            Splitter fastSplitter) {
        Tokenizer fidelity =
                TestTokenizers.modelFamily(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Missing GGUF tokenizer for " + familyId));
        Tokenizer hf =
                TestTokenizers.modelFamilyFromHf(familyId, hfModelRef, hfRevision)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Missing HF tokenizer for " + familyId));
        Tokenizer fast =
                TokenizerAdapters.withSplitter(
                        TokenizerAdapters.withNormalizer(fidelity, normalizer), fastSplitter);
        return new ComparedTokenizers(hf, fidelity, fast);
    }

    private static final class ComparedTokenizers {
        private final Tokenizer reference;
        private final Tokenizer fidelity;
        private final Tokenizer fast;

        private ComparedTokenizers(Tokenizer reference, Tokenizer fidelity, Tokenizer fast) {
            this.reference = reference;
            this.fidelity = fidelity;
            this.fast = fast;
        }

        private Tokenizer reference() {
            return reference;
        }

        private Tokenizer fidelity() {
            return fidelity;
        }

        private Tokenizer fast() {
            return fast;
        }
    }
}
