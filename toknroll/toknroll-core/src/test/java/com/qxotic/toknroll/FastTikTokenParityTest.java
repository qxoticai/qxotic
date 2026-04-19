package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastCl100kSplitter;
import com.qxotic.toknroll.impl.FastO200kSplitter;
import com.qxotic.toknroll.impl.FastR50kSplitter;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.List;
import org.junit.jupiter.api.Test;

class FastTikTokenParityTest {

    @Test
    void matchesJtokkitForRepresentativeSamples() {
        assertEncodingParity("r50k_base", FastR50kSplitter.INSTANCE);
        assertEncodingParity("cl100k_base", FastCl100kSplitter.INSTANCE);
        assertEncodingParity("o200k_base", FastO200kSplitter.INSTANCE);
    }

    private static void assertEncodingParity(String encoding, Splitter splitter) {
        Tokenizer reference = TiktokenFixtures.createJtokkitTokenizer(encoding);
        Tokenizer fast = TiktokenFixtures.createTikTokenTokenizer(encoding, splitter);

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
