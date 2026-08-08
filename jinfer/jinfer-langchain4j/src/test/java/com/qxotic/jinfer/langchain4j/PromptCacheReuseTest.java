package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import jdk.jfr.Recording;
import org.junit.jupiter.api.Test;

/**
 * ChatEngine used to RESUME from the block tree but never commit to it - only withCachedPrompt
 * populated the tree, so an ordinary second turn reused nothing. This pins the commit flow: the
 * pool ingests through CachedSession and adopts the decode, so a repeated prompt is served from
 * blocks rather than prefilled again.
 */
class PromptCacheReuseTest {
    @Test
    void aSecondTurnReusesTheFirst() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path jfr = Files.createTempFile("cacheproof", ".jfr");
        try (var m = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(8).build()) {
            try (Recording r = new Recording()) {
                r.enable("jinfer.Inference");
                r.start();
                m.chat("The capital of France is Paris. Tell me one fact about it.");
                m.chat("The capital of France is Paris. Tell me one fact about it.");
                r.stop();
                r.dump(jfr);
            }
        }
        var events = TelemetryEmissionTest.eventsOf(jfr, "jinfer.Inference");
        assertEquals(2, events.size());
        assertTrue(
                events.get(1).getInt("cachedTokens") > 0,
                "the second identical prompt must reuse the first");
    }
}
