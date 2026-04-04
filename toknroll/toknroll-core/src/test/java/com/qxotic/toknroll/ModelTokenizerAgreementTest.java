package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.gguf.ModelFamilyTokenizers;
import com.qxotic.toknroll.impl.FastLlama3Splitter;
import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import com.qxotic.toknroll.impl.RegexSplitter;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.TestCorpora;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.text.Normalizer.Form;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ModelTokenizerAgreementTest {

    private static final String LLAMA3_HF_MODEL_REF = "unsloth/Llama-3.2-1B-Instruct";
    private static final String LLAMA3_HF_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c";
    private static final String QWEN35_HF_MODEL_REF = "Qwen/Qwen3-0.6B";
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
                int[] jtokkit = compared.jtokkit().encodeToArray(text);
                int[] classic = compared.classic().encodeToArray(text);
                int[] fast = compared.fast().encodeToArray(text);

                assertArrayEquals(
                        jtokkit, classic, "jtokkit!=classic for " + model + " on: " + text);
                assertArrayEquals(jtokkit, fast, "jtokkit!=fast for " + model + " on: " + text);
                assertEquals(
                        jtokkit.length,
                        compared.jtokkit().countTokens(text),
                        "countTokens mismatch jtokkit");
                assertEquals(
                        classic.length,
                        compared.classic().countTokens(text),
                        "countTokens mismatch classic");
                assertEquals(
                        fast.length,
                        compared.fast().countTokens(text),
                        "countTokens mismatch fast");
            }
        }
    }

    private static ComparedTokenizers comparedTokenizersForModel(String model) {
        if ("gpt2".equals(model)) {
            Map<String, Integer> ranks = TiktokenFixtures.mergeableRanks("r50k_base");
            Map<String, Integer> specials = TiktokenFixtures.specialTokens("r50k_base");
            String pattern = TiktokenFixtures.encoding("r50k_base").pattern();
            return new ComparedTokenizers(
                    TiktokenFixtures.createJtokkitTokenizer("r50k_base"),
                    Tokenizers.classicBpe(
                            ranks, specials, Normalizer.identity(), RegexSplitter.create(pattern)),
                    Tokenizers.fastBpe(
                            ranks, specials, Normalizer.identity(), FastR50kSplitter.INSTANCE));
        }
        if ("llama3".equals(model)) {
            return modelNative(
                    "meta.llama3",
                    LLAMA3_HF_MODEL_REF,
                    LLAMA3_HF_REVISION,
                    Normalizer.identity(),
                    FastLlama3Splitter.INSTANCE);
        }
        if ("qwen35".equals(model)) {
            return modelNative(
                    "alibaba.qwen3_5",
                    QWEN35_HF_MODEL_REF,
                    null,
                    Normalizer.unicode(Form.NFC),
                    FastQwen35Splitter.INSTANCE);
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
                ModelFamilyTokenizers.create(familyId)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Missing GGUF tokenizer for " + familyId));
        Tokenizer hf =
                ModelFamilyTokenizers.createFromHfFiles(familyId, hfModelRef, hfRevision)
                        .orElseThrow(
                                () ->
                                        new IllegalStateException(
                                                "Missing HF tokenizer for " + familyId));
        Tokenizer fast =
                Tokenizers.pipeline(fidelity).normalizer(normalizer).splitter(fastSplitter).build();
        return new ComparedTokenizers(hf, fidelity, fast);
    }

    private static final class ComparedTokenizers {
        private final Tokenizer jtokkit;
        private final Tokenizer classic;
        private final Tokenizer fast;

        private ComparedTokenizers(Tokenizer jtokkit, Tokenizer classic, Tokenizer fast) {
            this.jtokkit = jtokkit;
            this.classic = classic;
            this.fast = fast;
        }

        private Tokenizer jtokkit() {
            return jtokkit;
        }

        private Tokenizer classic() {
            return classic;
        }

        private Tokenizer fast() {
            return fast;
        }
    }
}
