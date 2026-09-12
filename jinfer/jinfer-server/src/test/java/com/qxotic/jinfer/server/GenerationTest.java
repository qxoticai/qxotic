package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GenerationTest {

    @Test
    void schemaInstructionsPreserveTheSchemaAndStayStableAcrossTurns() {
        Map<String, Object> schema =
                Map.of(
                        "$defs",
                        Map.of("answer", Map.of("enum", List.of("yes", "no"))),
                        "$ref",
                        "#/$defs/answer");
        Message system = Message.system("Be concise.");
        Message user =
                new Message(
                        Role.USER, List.of(new Content.Text("Question"), new Content.Text("?")));
        List<Message> messages = new ArrayList<>(List.of(system, user));
        Generation.describeSchema(messages, schema);
        assertEquals("Be concise.", system.text(), "the caller's message is immutable");
        assertEquals(user, messages.getLast());
        assertEquals(
                "Be concise.\n\nYour final answer must be JSON matching this schema: "
                        + JsonCodec.stringify(schema),
                messages.getFirst().text());
        List<Message> next =
                new ArrayList<>(
                        List.of(system, user, Message.assistant("yes"), Message.user("Next?")));
        Generation.describeSchema(next, schema);
        assertEquals(messages.getFirst(), next.getFirst());

        List<Message> noSystem = new ArrayList<>(List.of(user));
        Generation.describeSchema(noSystem, schema);
        assertEquals(Role.SYSTEM, noSystem.getFirst().role());
        assertEquals(user, noSystem.getLast());
    }

    @Test
    void anOmittedSeedStaysUnseeded() {
        Sampling defaults = new Sampling(0.8f, 0.95f, 40, 0.05f, null);
        assertNull(Generation.sampling(Map.of(), defaults).seed());
        assertEquals(42L, Generation.sampling(Map.of("seed", 42), defaults).seed());
    }

    @Test
    void reasoningKnobsAreNullUnlessGiven() {
        assertNull(Generation.reasoningMax(Map.of()));
        assertNull(Generation.reasoningCutoffMessage(Map.of()));
        assertEquals(64, Generation.reasoningMax(Map.of("max_reasoning_tokens", 64)));
        assertEquals(
                "... Let me wrap up.",
                Generation.reasoningCutoffMessage(
                        Map.of("reasoning_cutoff_message", "... Let me wrap up.")));
    }

    @Test
    void malformedEchoedArgumentsAreTheClientsFault() {
        // an assistant turn echoed with a model's broken JSON: a 400 through the worker's
        // IllegalArgumentException mapping, never a 500 with a stack trace
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Generation.parseArguments("{\"city\": Paris"));
        assertTrue(e.getMessage().contains("arguments"), e.getMessage());
    }

    @Test
    void reasoningEffortIsTheOpenAiSpellingOfThinking() {
        assertEquals(false, Generation.thinking(Map.of("reasoning_effort", "none"), true));
        assertEquals(true, Generation.thinking(Map.of("reasoning_effort", "low"), false));
        assertEquals(
                false,
                Generation.thinking(
                        Map.of(
                                "reasoning_effort",
                                "high",
                                "chat_template_kwargs",
                                Map.of("enable_thinking", false)),
                        true),
                "the explicit kwarg wins");
        assertEquals(true, Generation.thinking(Map.of(), true));
    }

    @Test
    void finishReasonSaysWhatEndedTheReply() {
        assertEquals(
                "tool_calls", Generation.finishReason(Generator.FinishReason.STOP, true, false));
        assertEquals("stop", Generation.finishReason(Generator.FinishReason.LENGTH, false, true));
        assertEquals("stop", Generation.finishReason(Generator.FinishReason.STOP, false, false));
        assertEquals(
                "length", Generation.finishReason(Generator.FinishReason.LENGTH, false, false));
        // a deadline is neither the model's end nor the token budget: "other", never "stop"
        assertEquals(
                "other", Generation.finishReason(Generator.FinishReason.TIMEOUT, false, false));
        assertEquals("other", Generation.finishReason(Generator.FinishReason.ABORT, false, false));
    }
}
