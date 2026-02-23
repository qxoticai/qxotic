package com.qxotic.tokenizers.gguf;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GGUFTokenizerRegistryTest {

    @Test
    void defaultsContainGpt2AndLlama() {
        GGUFTokenizerRegistry registry = GGUFTokenizerRegistry.defaults();
        assertTrue(registry.contains("gpt2"));
        assertTrue(registry.contains("llama"));
    }

    @Test
    void customFactoryCanBeRegisteredByName() {
        GGUFTokenizerRegistry registry = GGUFTokenizerRegistry.defaults();
        registry.register("custom", registry.resolve("gpt2"));
        assertNotNull(registry.resolve("custom"));
    }

    @Test
    void unknownResolveReturnsNullAndRequireThrows() {
        GGUFTokenizerRegistry registry = GGUFTokenizerRegistry.defaults();
        assertFalse(registry.contains("unknown"));
        assertThrows(GGUFTokenizerException.class, () -> registry.require("unknown"));
    }

    @Test
    void canCheckBothEntriesViaTokenizersFacade() {
        assertTrue(GGUFTokenizers.isRegistered("qwen2", "gpt2"));
        assertFalse(GGUFTokenizers.isRegistered("missing", "gpt2"));
        assertFalse(GGUFTokenizers.isRegistered("qwen2", "missing"));
    }
}
