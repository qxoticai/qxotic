package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Prompt-opened think spans end-to-end against MiniCPM5 (its generation prompt OPENS {@code
 * <think>} - the reply starts inside the span, so the parser must be pre-fed the template's reply
 * seed or every reasoning token lands on the content lane). Model-gated via {@link TestModels}.
 * Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class MiniCpm5ThinkingIT {

    static Path MODEL; // resolved in @BeforeAll: a missing model then skips

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        MODEL = TestModels.require("hf.co/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf");
        model = JinferChatModel.builder().modelPath(MODEL).contextCapacity(4096).build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void promptOpenedThinkSpanParsesOnTheReasoningLane() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("What is 17 + 25? Answer briefly."),
                                JinferChatOptions.builder().maxTokens(512).build()));
        AssistantMessage out = r.getResult().getOutput();
        String thinking = JinferMappings.reasoning(out);
        assertTrue(
                !thinking.isBlank(),
                "reasoning must land in its part, not content - the reply seed was not pre-fed?"
                        + " text: "
                        + out.getText());
        assertTrue(
                out.getText() != null && !out.getText().isBlank(),
                "no content after the think span");
    }

    @Test
    void streamingFlagsThoughtChunksFromTheFirstToken() {
        List<ChatResponse> chunks =
                model.stream(
                                new Prompt(
                                        new UserMessage("What is 17 + 25? Answer briefly."),
                                        JinferChatOptions.builder().maxTokens(512).build()))
                        .collectList()
                        .block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        StringBuilder thoughts = new StringBuilder();
        for (ChatResponse c : chunks.subList(0, chunks.size() - 1)) {
            AssistantMessage out = c.getResult().getOutput();
            thoughts.append(JinferMappings.reasoning(out));
        }
        assertTrue(
                !thoughts.isEmpty(),
                "no thought chunks streamed - the prompt-opened think span leaked to content");
        // the seed's scaffold newline aside, which the parser gives to reasoning while the chunks,
        // generated after the seed, start fresh
        AssistantMessage streamed = ChatFixtures.aggregate(chunks);
        assertTrue(
                JinferMappings.reasoning(streamed).endsWith(thoughts.toString()), "the reasoning");
        assertFalse(
                model.replies.recall(streamed).isEmpty(),
                "the streamed reply and the parsed one disagree:\n" + streamed);
    }

    @Test
    void thinkingOffStaysContentOnly() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("One word: ok?"),
                                JinferChatOptions.builder().thinking(false).maxTokens(32).build()));
        assertTrue(r.getResult().getOutput().getReasoning().isEmpty());
        assertTrue(!r.getResult().getOutput().getText().isBlank());
    }
}
