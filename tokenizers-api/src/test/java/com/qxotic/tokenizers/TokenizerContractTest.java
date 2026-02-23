package com.qxotic.tokenizers;

import com.qxotic.tokenizers.testkit.TiktokenFixtures;
import com.qxotic.tokenizers.testkit.TokenizerAssertions;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerContractTest {

    private static final List<TokenizerSpec> TOKENIZERS = buildTokenizers();

    private static List<TokenizerSpec> buildTokenizers() {
        List<TokenizerSpec> specs = new ArrayList<>();
        Tokenizer classic = TiktokenFixtures.createClassicR50kTokenizer();
        Map<String, Integer> classicSpecials =
                TiktokenFixtures.encoding("r50k_base").specialTokens();
        specs.add(new TokenizerSpec("classic-r50k", classic, classicSpecials));

        for (TiktokenFixtures.EncodingFixture fixture : TiktokenFixtures.encodings()) {
            specs.add(
                    new TokenizerSpec(
                            "jtokkit-" + fixture.name(),
                            TiktokenFixtures.createJtokkitTokenizer(fixture.name()),
                            fixture.specialTokens()));
        }
        return List.copyOf(specs);
    }

    private static Stream<TokenizerSpec> tokenizerSpecs() {
        return TOKENIZERS.stream();
    }

    private static Stream<String> roundTripTexts() {
        return Stream.of(
                "",
                "Hello world",
                "Intërnâtiônàlizætiøn",
                "こんにちは世界",
                "Whitespace\tand\nnewlines",
                "Symbols: !@#$%^&*()",
                "Emoji: 😀🎉✨",
                "Flags: 🇺🇸🇯🇵",
                "Family emoji: 👨‍👩‍👧‍👦");
    }

    private static Stream<Arguments> roundTripCases() {
        return tokenizerSpecs()
                .flatMap(spec -> roundTripTexts().map(text -> Arguments.of(spec, text)));
    }

    @ParameterizedTest(name = "roundTrip {0}")
    @MethodSource("roundTripCases")
    void encodeDecodeRoundTrip(TokenizerSpec spec, String text) {
        TokenizerAssertions.assertRoundTrip(spec.tokenizer(), text, spec.name());
    }

    @ParameterizedTest(name = "countTokens {0}")
    @MethodSource("tokenizerSpecs")
    void countTokensMatchesEncodeLength(TokenizerSpec spec) {
        TokenizerAssertions.assertCountMatchesEncode(
                spec.tokenizer(), "Token count check", spec.name());
    }

    @ParameterizedTest(name = "tokensInVocab {0}")
    @MethodSource("tokenizerSpecs")
    void tokensAreInVocabulary(TokenizerSpec spec) {
        IntSequence tokens = spec.tokenizer().encode("Tokens should exist");
        TokenizerAssertions.assertTokensInVocabulary(spec.tokenizer(), tokens, spec.name());
    }

    @ParameterizedTest(name = "specialTokens {0}")
    @MethodSource("tokenizerSpecs")
    void specialTokenHandling(TokenizerSpec spec) {
        String specialToken = spec.specialTokens().keySet().iterator().next();
        int specialId = spec.specialTokens().get(specialToken);

        Assertions.assertTrue(spec.tokenizer().vocabulary().contains(specialId), spec.name());
        Assertions.assertEquals(
                specialToken, spec.tokenizer().vocabulary().token(specialId), spec.name());
    }

    @ParameterizedTest(name = "decodeBytes {0}")
    @MethodSource("tokenizerSpecs")
    void decodeBytesMatchUtf8(TokenizerSpec spec) {
        TokenizerAssertions.assertDecodeBytesUtf8(
                spec.tokenizer(), "Byte-level check", spec.name());
    }

    private record TokenizerSpec(
            String name, Tokenizer tokenizer, Map<String, Integer> specialTokens) {}
}
