package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Path;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordingFile;
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
        Path jfr = java.nio.file.Files.createTempFile("cacheproof", ".jfr");
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
        var events =
                RecordingFile.readAllEvents(jfr).stream()
                        .filter(e -> e.getEventType().getName().equals("jinfer.Inference"))
                        .toList();
        assertEquals(2, events.size());
        for (var e : events)
            System.out.println(
                    "  cachedTokens="
                            + e.getInt("cachedTokens")
                            + " tier="
                            + e.getString("cacheTier")
                            + " input="
                            + e.getInt("inputTokens"));
        assertTrue(
                events.get(1).getInt("cachedTokens") > 0,
                "the second identical prompt must reuse the first");
    }
}
