package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Path;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The context-overflow taxonomy, blocking lanes (the streaming wall pin lives in {@link
 * StreamingContractTest#hittingTheContextWallMidStreamKeepsThePartialsAndFinishesLength}). Three
 * distinct fates, three distinct signals: a prompt that cannot fit is refused BEFORE any state is
 * touched, with the counts and the remedy in the message; a generation that runs out of room ends
 * gracefully as LENGTH (what a hosted provider answers at max_tokens); a prompt that leaves zero
 * room for even one output token is the same refusal as any other over-capacity prompt.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ContextWallContractTest {

    private static JinferChatModel model;

    @BeforeAll
    void load() {
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        model =
                JinferChatModel.builder()
                        .modelPath(gguf)
                        .contextLength(64) // small enough that walls are a few dozen tokens away
                        .temperature(0.0)
                        .build();
    }

    @AfterAll
    void close() {
        if (model != null) model.close();
    }

    @Test
    void promptOverCapacityIsRefusedWithTheCountsAndTheRemedy() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(
                                                        UserMessage.from("elaborate ".repeat(500)))
                                                .maxOutputTokens(8)
                                                .build()));
        assertTrue(e.getMessage().contains("context capacity"), e.getMessage());
        assertTrue(e.getMessage().contains("64 available"), e.getMessage());
        assertTrue(e.getMessage().contains("raise the context capacity"), e.getMessage());
    }

    @Test
    void promptLeavingNoRoomForOneTokenIsTheSameRefusal() {
        // 55 repeats land the prompt a token past the wall: no room to answer at all
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(UserMessage.from("word ".repeat(55)))
                                                .maxOutputTokens(8)
                                                .build()));
        assertTrue(e.getMessage().contains("context capacity"), e.getMessage());
    }

    @Test
    void generationHittingTheWallFinishesLengthNeverErrors() {
        // fits: prompt ~22 tokens, wall at 42 output tokens - the count runs into it
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Count from 1 to 500, separated by commas."))
                                .maxOutputTokens(2000) // the knob allows more than the wall does
                                .build());
        assertEquals(FinishReason.LENGTH, r.finishReason());
        String text = r.aiMessage().text();
        assertTrue(text.startsWith("1") && text.contains("2"), text);
        assertTrue(
                r.tokenUsage().inputTokenCount() + r.tokenUsage().outputTokenCount() <= 64,
                "the wall is the context: " + r.tokenUsage());
    }
}
