package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

@Tag("slow")
@Tag("local-external")
class HuggingFaceTokenizerAdditionalSupportTest {

    @ParameterizedTest(name = "{0}:{1}")
    @MethodSource("additionalSupportModels")
    void supportsCoreTokenizerContracts(String user, String repo, String sampleText) {
        Tokenizer tokenizer =
                HuggingFaceTokenizerLoader.fromHuggingFace(user, repo, "main", false, false);

        IntSequence encoded = tokenizer.encode(sampleText);
        assertFalse(encoded.isEmpty());
        assertEquals(encoded.length(), tokenizer.countTokens(sampleText));

        IntSequence.Builder builder = IntSequence.newBuilder(encoded.length() + 8);
        tokenizer.encodeInto(sampleText, builder);
        assertArrayEquals(encoded.toArray(), builder.build().toArray());

        int[] encodedArray = tokenizer.encodeToArray(sampleText);
        assertEquals(encoded.length(), encodedArray.length);
        assertArrayEquals(encoded.toArray(), encodedArray);

        String decoded = tokenizer.decode(encoded);
        assertFalse(decoded.isEmpty());

        IntSequence reencoded = tokenizer.encode(decoded);
        assertEquals(reencoded.length(), tokenizer.countTokens(decoded));
    }

    private static Stream<Arguments> additionalSupportModels() {
        return Stream.of(
                Arguments.of("microsoft", "phi-4", "Hello Phi-4\nLine 2"),
                Arguments.of("ibm-granite", "granite-4.0-h-small", "Hello Granite 4\tTab"),
                Arguments.of("mistralai", "Mistral-7B-Instruct-v0.3", "Hello Mistral: 123 !"));
    }
}
