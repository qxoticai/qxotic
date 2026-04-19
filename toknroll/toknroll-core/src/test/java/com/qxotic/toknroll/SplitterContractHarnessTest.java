package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.FastCl100kSplitter;
import com.qxotic.toknroll.impl.FastLlama3Splitter;
import com.qxotic.toknroll.impl.FastO200kSplitter;
import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.testkit.SplitterContractHarness;
import com.qxotic.toknroll.testkit.TestCorpora;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class SplitterContractHarnessTest {

    @ParameterizedTest(name = "{0}")
    @MethodSource("strictCases")
    void splittersRespectStrictContract(SplitterCase splitterCase) {
        for (String sample : TestCorpora.SPLITTER_CONTRACT_SAMPLES) {
            SplitterContractHarness.assertConformsOnText(
                    splitterCase.name, splitterCase.splitter, sample);
        }
    }

    @Test
    void identityRespectsStrictContract() {
        for (String sample : TestCorpora.SPLITTER_CONTRACT_SAMPLES) {
            SplitterContractHarness.assertConformsOnText("identity", Splitter.identity(), sample);
        }
    }

    private static Stream<SplitterCase> strictCases() {
        return Stream.of(
                new SplitterCase(
                        "regex",
                        Splitter.regex(
                                Pattern.compile("\\s+|[,.!?]", Pattern.UNICODE_CHARACTER_CLASS))),
                new SplitterCase("fast-r50k", FastR50kSplitter.INSTANCE),
                new SplitterCase("fast-llama3", FastLlama3Splitter.INSTANCE),
                new SplitterCase("fast-cl100k", FastCl100kSplitter.INSTANCE),
                new SplitterCase("fast-qwen35", FastQwen35Splitter.INSTANCE),
                new SplitterCase("fast-o200k", FastO200kSplitter.INSTANCE),
                new SplitterCase("model-llama3", ModelSplitters.LLAMA3),
                new SplitterCase("model-qwen2", ModelSplitters.QWEN2),
                new SplitterCase("model-qwen35", ModelSplitters.QWEN35),
                new SplitterCase("model-smollm2", ModelSplitters.SMOLLM2),
                new SplitterCase("model-tekken", ModelSplitters.TEKKEN),
                new SplitterCase("model-mistral-tekken", ModelSplitters.MISTRAL_TEKKEN),
                new SplitterCase("model-deepseek-latest", ModelSplitters.DEEPSEEK_LATEST),
                new SplitterCase("model-kimi25", ModelSplitters.KIMI_25),
                new SplitterCase("model-refact", ModelSplitters.REFACT),
                new SplitterCase("model-default-bpe", ModelSplitters.DEFAULT_BPE),
                new SplitterCase("model-identity", ModelSplitters.IDENTITY),
                new SplitterCase("sequence-empty", Splitter.sequence()));
    }

    private static final class SplitterCase {
        final String name;
        final Splitter splitter;

        SplitterCase(String name, Splitter splitter) {
            this.name = name;
            this.splitter = splitter;
        }

        @Override
        public String toString() {
            return name;
        }
    }
}
