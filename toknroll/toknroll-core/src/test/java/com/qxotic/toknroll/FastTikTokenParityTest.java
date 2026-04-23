package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastSplitters;
import com.qxotic.toknroll.testkit.TestTokenizers;
import java.util.List;
import org.junit.jupiter.api.Test;

class FastTiktokenParityTest {

    @Test
    void matchesJtokkitForRepresentativeSamples() {
        assertEncodingParity("r50k_base", FastSplitters.r50k());
        assertEncodingParity("cl100k_base", FastSplitters.cl100k());
        assertEncodingParity("o200k_base", FastSplitters.o200k());
    }

    private static void assertEncodingParity(String encoding, Splitter splitter) {
        Tokenizer reference = TestTokenizers.tiktokenReference(encoding);
        Tokenizer fast = TestTokenizers.tiktoken(encoding, splitter);

        List<String> samples =
                List.of(
                        "Hello world!",
                        "The quick brown fox jumps over the lazy dog.",
                        "Emoji: 😀😅🔥 and CJK: 你好 世界",
                        "Numbers 1234567890 and symbols <>[]{}()/*+-=_",
                        "Whitespace\n\t multiple   spaces\r\n",
                        "A longer paragraph with mixed content to stress the merge loop and ensure"
                                + " parity.");

        for (String sample : samples) {
            IntSequence expected = reference.encode(sample);
            IntSequence actual = fast.encode(sample);
            assertEquals(expected, actual, () -> "encoding=" + encoding + " text=" + sample);
            assertEquals(expected.length(), fast.countTokens(sample));
            assertEquals(reference.decode(expected), fast.decode(actual));
        }
    }
}
