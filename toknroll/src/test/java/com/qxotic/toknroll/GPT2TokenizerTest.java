package com.qxotic.toknroll;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.TokenizerAssertions;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class GPT2TokenizerTest {
    private Tokenizer createGPT2Tokenizer() {
        return TiktokenFixtures.createClassicR50kTokenizer();
    }

    private Tokenizer createJTokkitTokenizer() {
        return TiktokenFixtures.createJtokkitTokenizer("r50k_base");
    }

    @Test
    void testDaughterPortrayingCharliesMother() {
        Tokenizer tokenizer = createGPT2Tokenizer();
        String text = "daughter, portraying Charlie's mother";

        IntSequence tokens = tokenizer.encode(text);

        Assertions.assertEquals(6, tokens.length(), "Should produce 6 tokens");

        int[] expectedTokens = {29642, 11, 42458, 11526, 338, 2802};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(
                    expectedTokens[i],
                    tokens.intAt(i),
                    "Token at position " + i + " should be " + expectedTokens[i]);
        }

        TokenizerAssertions.assertRoundTrip(tokenizer, text, "classic r50k known sample");
        TokenizerAssertions.assertCountMatchesEncode(tokenizer, text, "classic r50k known sample");
        TokenizerAssertions.assertTokensInVocabulary(
                tokenizer, tokens, "classic r50k known sample");
    }

    @Test
    void testDaughterPortrayingCharliesMotherWithJTokkit() {
        Tokenizer tokenizer = createJTokkitTokenizer();
        String text = "daughter, portraying Charlie's mother";

        IntSequence tokens = tokenizer.encode(text);

        Assertions.assertEquals(6, tokens.length(), "Should produce 6 tokens");

        int[] expectedTokens = {29642, 11, 42458, 11526, 338, 2802};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(
                    expectedTokens[i],
                    tokens.intAt(i),
                    "Token at position " + i + " should be " + expectedTokens[i]);
        }

        TokenizerAssertions.assertRoundTrip(tokenizer, text, "jtokkit r50k known sample");
        TokenizerAssertions.assertCountMatchesEncode(tokenizer, text, "jtokkit r50k known sample");
        TokenizerAssertions.assertTokensInVocabulary(
                tokenizer, tokens, "jtokkit r50k known sample");
    }

    @Test
    void testBothTokenizersProduceSameTokens() {
        Tokenizer classicTokenizer = createGPT2Tokenizer();
        Tokenizer jtokkitTokenizer = createJTokkitTokenizer();
        String text = "daughter, portraying Charlie's mother";

        IntSequence classicTokens = classicTokenizer.encode(text);
        IntSequence jtokkitTokens = jtokkitTokenizer.encode(text);

        Assertions.assertEquals(
                classicTokens.length(),
                jtokkitTokens.length(),
                "Both tokenizers should produce same number of tokens");

        for (int i = 0; i < classicTokens.length(); i++) {
            Assertions.assertEquals(
                    classicTokens.intAt(i),
                    jtokkitTokens.intAt(i),
                    "Token at position " + i + " should match");
        }
    }

    @Test
    void testPortrayingWithAndWithoutSpace() {
        Tokenizer tokenizer = createJTokkitTokenizer();

        String textWithSpace = " portraying";
        IntSequence tokensWithSpace = tokenizer.encode(textWithSpace);

        Assertions.assertEquals(
                1,
                tokensWithSpace.length(),
                "' portraying' (with leading space) should be 1 token");
        Assertions.assertEquals(
                42458, tokensWithSpace.intAt(0), "' portraying' should be token 42458");

        String textWithoutSpace = "portraying";
        IntSequence tokensWithoutSpace = tokenizer.encode(textWithoutSpace);

        Assertions.assertEquals(
                3, tokensWithoutSpace.length(), "'portraying' (without space) should be 3 tokens");
        int[] expectedTokens = {634, 2433, 278};
        for (int i = 0; i < expectedTokens.length; i++) {
            Assertions.assertEquals(
                    expectedTokens[i],
                    tokensWithoutSpace.intAt(i),
                    "Token at position " + i + " for 'portraying' should be " + expectedTokens[i]);
        }
    }
}
