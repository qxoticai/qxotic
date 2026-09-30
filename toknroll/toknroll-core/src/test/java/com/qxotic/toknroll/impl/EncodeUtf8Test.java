package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Toknroll;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** The UTF-8 entry gives the string entry's ids: on the byte path and on the fallback. */
class EncodeUtf8Test {

    @Test
    void bytesGiveTheStringsIds() {
        for (String encoding : new String[] {"r50k_base", "o200k_base"}) {
            Tokenizer tokenizer =
                    TiktokenFixtures.createTiktokenTokenizer(
                            encoding, Splitter.regex(TiktokenFixtures.splitPattern(encoding)));
            check(tokenizer, encoding);
        }
    }

    @Test
    void aNormalizerTakesTheFallback() {
        TokenizationPipeline base =
                (TokenizationPipeline)
                        TiktokenFixtures.createTiktokenTokenizer(
                                "o200k_base",
                                Splitter.regex(TiktokenFixtures.splitPattern("o200k_base")));
        Tokenizer normalized =
                Toknroll.pipeline(
                        Normalizer.unicode(java.text.Normalizer.Form.NFC),
                        base.splitter().get(),
                        base.model());
        check(normalized, "o200k with NFC");
    }

    private static void check(Tokenizer tokenizer, String name) {
        Random random = new Random(0xB17E5L);
        for (int round = 0; round < 300; round++) {
            StringBuilder text = new StringBuilder();
            for (int n = random.nextInt(40); n > 0; n--) {
                text.append(
                        TiktokenSplitterTest.PIECES[
                                random.nextInt(TiktokenSplitterTest.PIECES.length)]);
            }
            String s = text.toString();
            IntSequence expected = tokenizer.encode(s);
            // the pieces include lone surrogates, which String.getBytes turns into '?' and the
            // tokenizer into U+FFFD: encode as the tokenizer does
            byte[] padded = TiktokenSplitter.utf8(s, 0, s.length());
            byte[] utf8 = Arrays.copyOf(padded, padded.length - TiktokenSplitter.PADDING);
            assertEquals(expected, tokenizer.encodeUtf8(utf8), name + " on " + s);
            // a range inside a larger array, with and without slack after it
            byte[] slack = Arrays.copyOf(utf8, utf8.length + 40);
            IntSequence.Builder out = IntSequence.newBuilder(8);
            tokenizer.encodeUtf8Into(slack, 0, utf8.length, out);
            assertEquals(expected, out.build(), name + " with slack");
            byte[] shifted = new byte[utf8.length + 3];
            System.arraycopy(utf8, 0, shifted, 3, utf8.length);
            out = IntSequence.newBuilder(8);
            tokenizer.encodeUtf8Into(shifted, 3, shifted.length, out);
            assertEquals(expected, out.build(), name + " shifted");
        }
    }
}
