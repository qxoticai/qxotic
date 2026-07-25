package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Codec-less models (Nemotron-H: no StateCodec, no block caching) must still load and chat; the
 * codec throw belongs to the first CACHED-feature use, not to construction. Model-gated (30B - slow
 * to load): assume-skips when the file is absent. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class JinferCodeclessIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelCodecless",
                            "/home/mukel/Desktop/playground/models/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf"));

    @Test
    void codeclessModelLoadsAndChatsButRejectsCachedPrompts() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        JinferChatModel m =
                JinferChatModel.builder().modelPath(MODEL).contextLength(2048).maxTokens(8).build();
        try {
            ChatResponse r = m.call(new Prompt(new UserMessage("hi")));
            assertNotNull(r.getResult().getOutput().getText());
            IllegalStateException e =
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    m.withCachedPrompt(
                                            List.of(new SystemMessage("You are terse.")),
                                            List.of()));
            assertTrue(e.getMessage().contains("no state codec"), e.getMessage());
        } finally {
            m.close();
        }
    }
}
