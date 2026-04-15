package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.toknroll.loaders.ModelSplitters;
import com.qxotic.toknroll.loaders.SplitterRegistry;
import org.junit.jupiter.api.Test;

class ModelSplitterFamilyTest {

    @Test
    void modelSplittersSupportModernFamilyAliases() {
        assertSame(ModelSplitters.QWEN35, ModelSplitters.forModel("qwen3.5"));
        assertSame(ModelSplitters.QWEN35, ModelSplitters.forModel("qwen3_5"));
        assertSame(ModelSplitters.LLAMA3, ModelSplitters.forModel("llama3"));
        assertSame(ModelSplitters.LLAMA3, ModelSplitters.forModel("phi4"));
        assertSame(ModelSplitters.IDENTITY, ModelSplitters.forModel("gemma3"));
        assertSame(ModelSplitters.IDENTITY, ModelSplitters.forModel("gemma4"));
        assertSame(ModelSplitters.SMOLLM2, ModelSplitters.forModel("smollm3"));
        assertSame(ModelSplitters.QWEN35, ModelSplitters.forModel("deepseek-r1"));
        assertSame(ModelSplitters.QWEN35, ModelSplitters.forModel("kimi-2.5"));
        assertSame(ModelSplitters.TEKKEN, ModelSplitters.forModel("mistral-tekken"));
        assertSame(ModelSplitters.REFACT, ModelSplitters.forModel("granite4"));
        assertSame(ModelSplitters.TEKKEN, ModelSplitters.forModel("tekken"));
    }

    @Test
    void splitterRegistryResolvesModernFamilies() {
        SplitterRegistry registry = SplitterRegistry.defaults();
        Splitter qwen = registry.require("qwen3.5");
        Splitter phi = registry.require("phi4");
        Splitter gemma = registry.require("gemma3");
        Splitter tekken = registry.require("tekken");
        Splitter deepseek = registry.require("deepseek-r1");
        Splitter kimi = registry.require("kimi-2.5");
        Splitter mistralTekken = registry.require("mistral-tekken");
        Splitter granite = registry.require("granite4");

        assertSame(ModelSplitters.QWEN35, qwen);
        assertSame(ModelSplitters.LLAMA3, phi);
        assertSame(ModelSplitters.IDENTITY, gemma);
        assertSame(ModelSplitters.TEKKEN, tekken);
        assertSame(ModelSplitters.QWEN35, deepseek);
        assertSame(ModelSplitters.QWEN35, kimi);
        assertSame(ModelSplitters.TEKKEN, mistralTekken);
        assertSame(ModelSplitters.REFACT, granite);
        assertNotNull(qwen);
        assertNotNull(phi);
        assertNotNull(gemma);
        assertNotNull(tekken);
        assertNotNull(deepseek);
        assertNotNull(kimi);
        assertNotNull(mistralTekken);
        assertNotNull(granite);
    }
}
