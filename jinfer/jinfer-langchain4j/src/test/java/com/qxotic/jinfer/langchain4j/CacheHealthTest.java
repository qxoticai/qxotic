package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.SystemMessage;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * The cache health reading exists from CONSTRUCTION and is refreshed by define() - not only by
 * traffic. An empty cache is a reading, not an absence: /props and the JFR gauge report it from the
 * first moment, and a define-only server (mounted views, no chats yet) must already show its
 * blocks. Deleting the constructor snapshot or the define() refresh fails exactly this.
 */
class CacheHealthTest {

    @Test
    void theReadingExistsBeforeAnyTrafficAndDefineRefreshesIt() {
        var gguf = ModelFixture.LLAMA32_1B_Q8.require();
        try (var model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(4).build()) {
            var fresh = model.engine.cacheSample();
            assertNotNull(fresh, "a codec model has a reading from construction");
            assertEquals(0, fresh.blocks(), "an empty cache reads as zero, not null");
            assertTrue(fresh.budgetBytes() > 0, "the budget is part of the reading");

            model.withCachedPrompt(List.of(SystemMessage.from("You are terse.")), List.of());
            assertTrue(
                    model.engine.cacheSample().blocks() > 0,
                    "define() changes blocks and must refresh the reading without any chat");
        }
    }
}
