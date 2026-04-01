package ai.llm4j.test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.model.llm.ChatFormat;
import com.qxotic.model.llm.llama.Llama3ChatFormat;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

@Disabled
class Llama3TokenizerTest extends TokenizerTest {

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testSpecialTokens(Tokenizer tokenizer) {
        assertEquals(128000, tokenizer.vocabulary().id("<|begin_of_text|>"));
    }

    private IntSequence encode(
            Llama3ChatFormat chatFormat, String text, boolean prependBOS, boolean appendEOS) {
        IntSequence.Builder builder = IntSequence.newBuilder();
        if (prependBOS) {
            builder.add(chatFormat.beginOfText);
        }
        builder.addAll(chatFormat.tokenizer().encode(text));
        if (appendEOS) {
            builder.add(chatFormat.endOfText);
        }
        return builder;
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testEncode(Tokenizer tokenizer) {
        IntSequence expected = IntSequence.of(2028, 374, 264, 1296, 11914, 13);
        assertEquals(expected, tokenizer.encode("This is a test sentence."));
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testDecode(Tokenizer tokenizer) {
        IntSequence input = IntSequence.of(2028, 374, 264, 1296, 11914, 13);
        assertEquals("This is a test sentence.", tokenizer.decode(input));
    }

    @ParameterizedTest
    @MethodSource("chatFormatProvider")
    void testEncodeMessage(Llama3ChatFormat chatFormat) {
        Llama3ChatFormat.Message message =
                new Llama3ChatFormat.Message(ChatFormat.USER, "This is a test sentence.");

        IntSequence expected =
                IntSequence.of(
                        128006, // <|start_header_id|>
                        882, // "user"
                        128007, // <|end_of_header|>
                        271, // "\n\n"
                        2028,
                        374,
                        264,
                        1296,
                        11914,
                        13, // This is a test sentence.
                        128009 // <|eot_id|>
                        );

        IntSequence encodedMessage = chatFormat.encodeMessage(message);
        assertEquals(expected, encodedMessage);
    }

    @ParameterizedTest
    @MethodSource("chatFormatProvider")
    void testEncodeDialog(Llama3ChatFormat chatFormat) {
        List<Llama3ChatFormat.Message> messages =
                List.of(
                        new Llama3ChatFormat.Message(ChatFormat.SYSTEM, "This is a test sentence."),
                        new Llama3ChatFormat.Message(ChatFormat.USER, "This is a response."));

        IntSequence expected =
                IntSequence.of(
                        128000, // <|begin_of_text|>
                        128006, // <|start_header_id|>
                        9125, // "system"
                        128007, // <|end_of_header|>
                        271, // "\n\n"
                        2028,
                        374,
                        264,
                        1296,
                        11914,
                        13, // This is a test sentence.
                        128009, // <|eot_id|>
                        128006, // <|start_header_id|>
                        882, // "user"
                        128007, // <|end_of_header|>
                        271, // "\n\n"
                        2028,
                        374,
                        264,
                        2077,
                        13, // This is a response.
                        128009, // <|eot_id|>
                        128006, // <|start_header_id|>
                        78191, // "assistant"
                        128007, // <|end_of_header|>
                        271 // "\n\n"
                        );

        IntSequence dialogPrompt = chatFormat.encodeDialog(messages);
        assertEquals(expected, dialogPrompt);
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testEncodeEmpty(Tokenizer tokenizer) {
        assertTrue(tokenizer.encode("").isEmpty());
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testEncodeSurrogatePairs(Tokenizer tokenizer) {
        assertEquals(IntSequence.of(9468, 239, 235), tokenizer.encode("👍"));

        // surrogate pair gets converted to codepoint
        assertEquals(IntSequence.of(9468, 239, 235), tokenizer.encode("\ud83d\udc4d"));

        // lone surrogate just gets replaced
        IntSequence loneSurrogate = tokenizer.encode("\ud83d");

        assertTrue(
                // Use replacement character \ufffd , this is the default behavior in Llama 3
                // tokenizer.
                tokenizer.encode("�").equals(loneSurrogate)
                        ||
                        // Java default UTF8 replacement; both behaviors are accepted.
                        tokenizer.encode("?").equals(loneSurrogate));
    }

    @ParameterizedTest
    @MethodSource("tokenizerProvider")
    void testCatastrophicallyRepetitive(Tokenizer tokenizer) {
        for (String c : List.of("^", "0", "a", "'s", " ", "\n")) {
            String bigValue = c.repeat(10_000);
            assertEquals(bigValue, tokenizer.decode(tokenizer.encode(bigValue)));

            bigValue = " " + bigValue;
            assertEquals(bigValue, tokenizer.decode(tokenizer.encode(bigValue)));

            bigValue = bigValue + "\n";
            assertEquals(bigValue, tokenizer.decode(tokenizer.encode(bigValue)));
        }
    }
}
