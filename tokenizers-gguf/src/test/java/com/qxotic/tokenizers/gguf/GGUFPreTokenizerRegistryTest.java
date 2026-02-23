package com.qxotic.tokenizers.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.advanced.Splitter;
import java.util.List;
import org.junit.jupiter.api.Test;

class GGUFPreTokenizerRegistryTest {

    private final GGUFPreTokenizerRegistry registry = GGUFPreTokenizerRegistry.defaults();

    @Test
    void qwenPatternSplitsSingleDigits() {
        Splitter splitter = registry.resolve("qwen2");
        List<String> tokens =
                splitter.split("v 1234").stream().map(CharSequence::toString).toList();
        assertTrue(tokens.stream().anyMatch(t -> t.equals("1")), "qwen2 splits single digits");
    }

    @Test
    void llamaPatternSplitsInChunks() {
        Splitter splitter = registry.resolve("llama-bpe");
        List<String> tokens =
                splitter.split("v 1234").stream().map(CharSequence::toString).toList();
        assertTrue(tokens.stream().anyMatch(t -> t.equals("123")), "llama-bpe uses 1..3 chunks");
    }

    @Test
    void unknownResolveReturnsNull() {
        assertNull(registry.resolve("unknown"));
    }

    @Test
    void unknownRequireThrows() {
        assertThrows(GGUFTokenizerException.class, () -> registry.require("unknown"));
    }

    @Test
    void customSplitterCanBeRegisteredByName() {
        registry.register("custom", text -> List.of("x", "y"));
        List<String> tokens =
                registry.resolve("custom").split("anything").stream()
                        .map(CharSequence::toString)
                        .toList();
        assertEquals(List.of("x", "y"), tokens);
    }
}
