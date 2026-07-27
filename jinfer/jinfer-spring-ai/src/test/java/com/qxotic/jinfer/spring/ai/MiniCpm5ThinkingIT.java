package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
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
 * seed or every reasoning token lands on the content lane). Model-gated: assume-skips when the file
 * is absent. Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-spring-ai}
 */
@Tag("integration")
class MiniCpm5ThinkingIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelMiniCpm5",
                            ModelFixture.MINICPM5_1B_Q8.path().toString()));

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        model = JinferChatModel.builder().modelPath(MODEL).contextLength(4096).build();
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
        String thinking = (String) out.getMetadata().get(JinferMappings.THINKING_KEY);
        assertNotNull(
                thinking,
                "reasoning must land in metadata, not content - the reply seed was not pre-fed?"
                        + " text: "
                        + out.getText());
        assertTrue(!thinking.isBlank());
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
            if (Boolean.TRUE.equals(out.getMetadata().get(JinferChatModel.IS_THOUGHT_KEY))) {
                thoughts.append(out.getText());
            }
        }
        assertTrue(
                !thoughts.isEmpty(),
                "no thought chunks streamed - the prompt-opened think span leaked to content");
        String thinking =
                (String)
                        chunks.get(chunks.size() - 1)
                                .getResult()
                                .getOutput()
                                .getMetadata()
                                .get(JinferMappings.THINKING_KEY);
        // the finished thinking is the streamed thoughts plus the seed's scaffold newline (the
        // parser attributes the prompt-opened span's "\n" to reasoning; the chunks, generated
        // after the seed, start fresh)
        assertTrue(
                thinking.endsWith(thoughts.toString()),
                "thinking metadata and streamed thought chunks disagree:\n"
                        + thinking
                        + "\n---\n"
                        + thoughts);
    }

    @Test
    void thinkingOffStaysContentOnly() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("One word: ok?"),
                                JinferChatOptions.builder().thinking(false).maxTokens(32).build()));
        assertNull(r.getResult().getOutput().getMetadata().get(JinferMappings.THINKING_KEY));
        assertTrue(!r.getResult().getOutput().getText().isBlank());
    }
}
