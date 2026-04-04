package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.SymbolCodec;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.charset.StandardCharsets;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerFidelityTest {

    @ParameterizedTest(name = "fidelity {0}")
    @MethodSource("tokenizers")
    void encodeDecodeIsFidelityPreservingForRegularText(String name, Tokenizer tokenizer) {
        String text = "  Keep all whitespace\\n\\tand symbols <>[]{} and emoji 😀  ";
        IntSequence tokens = tokenizer.encode(text);

        assertEquals(text, tokenizer.decode(tokens), name + " decode(encode(text))");
        assertEquals(tokens.length(), tokenizer.countTokens(text), name + " countTokens");
        assertArrayEquals(
                text.getBytes(StandardCharsets.UTF_8),
                tokenizer.decodeBytes(tokens),
                name + " decodeBytes(encode(text))");
    }

    @ParameterizedTest(name = "deterministic {0}")
    @MethodSource("tokenizers")
    void encodeIsDeterministic(String name, Tokenizer tokenizer) {
        String text = "Determinism check: same input, same tokens.";
        IntSequence first = tokenizer.encode(text);
        IntSequence second = tokenizer.encode(text);
        assertArrayEquals(first.toArray(), second.toArray(), name + " deterministic");
    }

    @ParameterizedTest(name = "invalid utf8 bytes {0}")
    @MethodSource("tokenizers")
    void decodeBytesIsAuthoritativeForNonUtf8TokenSequences(String name, Tokenizer tokenizer) {
        String byteSymbol = SymbolCodec.BYTE_LEVEL.encodeBytes(new byte[] {(byte) 0xFF});
        Assumptions.assumeTrue(
                tokenizer.vocabulary().contains(byteSymbol),
                name + " does not expose direct 0xFF byte token");

        int id = tokenizer.vocabulary().id(byteSymbol);
        IntSequence token = IntSequence.of(id);

        assertArrayEquals(new byte[] {(byte) 0xFF}, tokenizer.decodeBytes(token), name + " bytes");
        assertDoesNotThrow(() -> tokenizer.decode(token), name + " decode should not crash");
    }

    private static Stream<Arguments> tokenizers() {
        Stream<Arguments> classic =
                Stream.of(
                        Arguments.of(
                                "classic-r50k", TiktokenFixtures.createClassicR50kTokenizer()));
        Stream<String> encodingNames =
                Stream.of("r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base");
        Stream<Arguments> jtokkit =
                encodingNames.map(
                        name ->
                                Arguments.of(
                                        "jtokkit-" + name,
                                        TiktokenFixtures.createJtokkitTokenizer(name)));
        return Stream.concat(classic, jtokkit);
    }
}
