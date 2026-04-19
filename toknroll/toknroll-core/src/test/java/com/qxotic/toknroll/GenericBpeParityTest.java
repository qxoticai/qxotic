package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.TiktokenReconstruction;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.List;
import org.junit.jupiter.api.Test;

class GenericBpeParityTest {

    @Test
    void genericEngineMatchesJtokkitOnRepresentativeSamples() {
        assertEncodingParity("r50k_base");
        assertEncodingParity("cl100k_base");
        assertEncodingParity("o200k_base");
    }

    private static void assertEncodingParity(String encoding) {
        Tokenizer reference = TiktokenFixtures.createJtokkitTokenizer(encoding);
        Splitter splitter = Splitter.regex(TiktokenFixtures.splitPattern(encoding));
        Vocabulary vocabulary =
                TiktokenReconstruction.vocabulary(
                        TiktokenFixtures.mergeableRanks(encoding),
                        TiktokenFixtures.specialTokens(encoding));
        List<Tokenizers.MergeRule> merges =
                TiktokenReconstruction.mergeRules(TiktokenFixtures.mergeableRanks(encoding));
        Tokenizer generic =
                TokenizationPipeline.builder(Tokenizers.tikTokenModel(vocabulary, merges))
                        .splitter(splitter)
                        .build();

        List<String> samples =
                List.of(
                        "Hello world!",
                        "The quick brown fox jumps over the lazy dog.",
                        "Emoji: 😀😅🔥 and CJK: 你好 世界",
                        "RTL mix: مرحبا بالعالم version 2.5",
                        "Numbers 1234567890 and symbols <>[]{}()/*+-=_",
                        "Whitespace\n\t multiple   spaces\r\n",
                        "A longer paragraph with mixed content to stress reusable BPE parity.");

        for (String sample : samples) {
            IntSequence expected = reference.encode(sample);
            IntSequence actual = generic.encode(sample);
            assertEquals(expected, actual, () -> "encoding=" + encoding + " text=" + sample);
            assertEquals(expected.length(), generic.countTokens(sample));
            assertEquals(reference.decode(expected), generic.decode(actual));
        }
    }
}
