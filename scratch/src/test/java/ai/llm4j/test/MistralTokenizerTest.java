package ai.llm4j.test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MistralTokenizerTest extends TokenizerTest {
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testSpecialTokens(BaseTokenizer tokenizer) {
//        assertEquals(128000, tokenizer.vocabulary().getTokenIndex("<|begin_of_text|>"));
//    }
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testEncode(BaseTokenizer tokenizer) {
//        IntSequence expected = IntSequence.of(2028, 374, 264, 1296, 11914, 13);
//        assertEquals(expected,
//                tokenizer.encode("This is a test sentence."));
//    }
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testDecode(BaseTokenizer tokenizer) {
//        IntSequence input = IntSequence.of(2028, 374, 264, 1296, 11914, 13);
//        assertEquals(
//                "This is a test sentence.",
//                tokenizer.decodeString(input)
//        );
//    }
//
//    @ParameterizedTest
//    @MethodSource("chatFormatProvider")
//    void testEncodeMessage(MistralChatFormat chatFormat) {
//        MistralChatFormat.Message message = new MistralChatFormat.Message(MistralChatFormat.Role.USER, "This is a test sentence.");
//
//        IntSequence expected = IntSequence.of(
//                128006,  // <|start_header_id|>
//                882,     // "user"
//                128007,  // <|end_of_header|>
//                271,     // "\n\n"
//                2028, 374, 264, 1296, 11914, 13,  // This is a test sentence.
//                128009   // <|eot_id|>
//        );
//
//        IntSequence encodedMessage = chatFormat.encodeHeaderAndMessage(message);
//        assertEquals(expected, encodedMessage);
//    }
//
//    @ParameterizedTest
//    @MethodSource("chatFormatProvider")
//    void testEncodeDialog(MistralChatFormat chatFormat) {
//        List<MistralChatFormat.Message> messages = List.of(
//                new MistralChatFormat.Message(MistralChatFormat.Role.SYSTEM, "This is a test sentence."),
//                new MistralChatFormat.Message(MistralChatFormat.Role.USER, "This is a response.")
//        );
//
//        IntSequence expected = IntSequence.of(
//                128000,  // <|begin_of_text|>
//                128006,  // <|start_header_id|>
//                9125,    // "system"
//                128007,  // <|end_of_header|>
//                271,     // "\n\n"
//                2028, 374, 264, 1296, 11914, 13,  // This is a test sentence.
//                128009,  // <|eot_id|>
//                128006,  // <|start_header_id|>
//                882,     // "user"
//                128007,  // <|end_of_header|>
//                271,     // "\n\n"
//                2028, 374, 264, 2077, 13,  // This is a response.
//                128009,  // <|eot_id|>
//                128006,  // <|start_header_id|>
//                78191,   // "assistant"
//                128007,  // <|end_of_header|>
//                271      // "\n\n"
//        );
//
//        IntSequence dialogPrompt = chatFormat.encodeDialogPrompt(true, messages);
//        assertEquals(expected, dialogPrompt);
//    }
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testEncodeEmpty(BaseTokenizer tokenizer) {
//        assertTrue(tokenizer.encode("").isEmpty());
//    }
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testEncodeSurrogatePairs(BaseTokenizer tokenizer) {
//        assertEquals(IntSequence.of(9468, 239, 235), tokenizer.encode("👍"));
//
//        // surrogate pair gets converted to codepoint
//        assertEquals(IntSequence.of(9468, 239, 235), tokenizer.encode("\ud83d\udc4d"));
//
//        // lone surrogate just gets replaced
//        IntSequence loneSurrogate = tokenizer.encode("\ud83d");
//
//        assertTrue(
//                // Use replacement character \ufffd , this is the default behavior in Llama 3 tokenizer.
//                tokenizer.encode("�").equals(loneSurrogate) ||
//                        // Java default UTF8 replacement; both behaviors are accepted.
//                        tokenizer.encode("?").equals(loneSurrogate)
//        );
//    }
//
//    @ParameterizedTest
//    @MethodSource("tokenizerProvider")
//    void testCatastrophicallyRepetitive(BaseTokenizer tokenizer) {
//        for (String c : List.of("^", "0", "a", "'s", " ", "\n")) {
//            String bigValue = c.repeat(10_000);
//            assertEquals(bigValue, tokenizer.decodeString(tokenizer.encode(bigValue)));
//
//            bigValue = " " + bigValue;
//            assertEquals(bigValue, tokenizer.decodeString(tokenizer.encode(bigValue)));
//
//            bigValue = bigValue + "\n";
//            assertEquals(bigValue, tokenizer.decodeString(tokenizer.encode(bigValue)));
//        }
//    }
}
