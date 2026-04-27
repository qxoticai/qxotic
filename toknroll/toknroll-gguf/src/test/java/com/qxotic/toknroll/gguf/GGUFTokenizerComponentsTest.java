package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Splitter;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class GGUFTokenizerComponentsTest {

    @Test
    void qwenAndLlamaPatternsSplitNumbersDifferently() {
        Splitter qwen =
                Splitter.regex(
                        Pattern.compile(
                                ModelTextSplitters.QWEN2_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        Splitter llama =
                Splitter.regex(
                        Pattern.compile(
                                ModelTextSplitters.LLAMA3_PATTERN,
                                Pattern.UNICODE_CHARACTER_CLASS));

        String text = "v 1234";
        List<String> qwenTokens =
                qwen.splitAllToListEagerly(text).stream()
                        .map(CharSequence::toString)
                        .collect(Collectors.toList());
        List<String> llamaTokens =
                llama.splitAllToListEagerly(text).stream()
                        .map(CharSequence::toString)
                        .collect(Collectors.toList());

        assertTrue(qwenTokens.stream().anyMatch(t -> t.contains("1")));
        assertTrue(qwenTokens.stream().anyMatch(t -> t.contains("4")));
        assertTrue(llamaTokens.stream().anyMatch(t -> t.contains("123") || t.contains(" 123")));
    }

    @Test
    void tekkenLlamaPatternHandlesContractions() {
        Splitter splitter =
                Splitter.regex(
                        Pattern.compile(
                                ModelTextSplitters.LLAMA3_PATTERN,
                                Pattern.UNICODE_CHARACTER_CLASS));
        List<String> tokens =
                splitter.splitAllToListEagerly("I'm ready").stream()
                        .map(CharSequence::toString)
                        .collect(Collectors.toList());
        assertTrue(tokens.stream().anyMatch(t -> t.contains("'m")));
    }

    @Test
    void preTokenizerAdapterUsesModelSplitter() {
        Splitter splitter = ModelTextSplitters.createSplitter("qwen3");

        String input = "Hello, world";
        List<CharSequence> tokens = splitter.splitAllToListEagerly(input);

        assertTrue(tokens.size() >= 2);
        String rebuilt = tokens.stream().map(CharSequence::toString).collect(Collectors.joining());
        assertEquals(input, rebuilt);
    }
}
