package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class OpenAiChatTest {

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
        OpenAiChat.describeSchema(messages, schema);
        assertEquals("Be concise.", system.text(), "the caller's message is immutable");
        assertEquals(user, messages.getLast());
        assertEquals(
                "Be concise.\n\nYour final answer must be JSON matching this schema: "
                        + JsonCodec.stringify(schema),
                messages.getFirst().text());
        List<Message> next =
                new ArrayList<>(
                        List.of(system, user, Message.assistant("yes"), Message.user("Next?")));
        OpenAiChat.describeSchema(next, schema);
        assertEquals(messages.getFirst(), next.getFirst());

        List<Message> noSystem = new ArrayList<>(List.of(user));
        OpenAiChat.describeSchema(noSystem, schema);
        assertEquals(Role.SYSTEM, noSystem.getFirst().role());
        assertEquals(user, noSystem.getLast());
    }

    @Test
    void aCancelledPassKeepsTheWordsItProduced() {
        // the engine drops a cancelled reply; the words a streaming caller saw come back through
        // `produced`, with the finish reason still "cancelled" and the tokens still counted
        Generator.GenerationResult result =
                new Generator.GenerationResult(
                        new int[] {1, 2, 3},
                        OptionalInt.empty(),
                        Generator.FinishReason.ABORT,
                        Duration.ofMillis(5),
                        Duration.ofMillis(7));
        ChatEngine.Completion cancelled =
                new ChatEngine.Completion(null, result, false, 10, 0, PromptCache.Tier.FRESH, null);
        Reply kept =
                OpenAiChat.reply(cancelled, Message.assistant("Once upon a"), List.of(), false);
        assertEquals("cancelled", kept.finishReason());
        assertEquals("Once upon a", kept.text());
        assertEquals(3, kept.completionTokens());
        Reply silent = OpenAiChat.reply(cancelled, null, List.of(), false);
        assertEquals("cancelled", silent.finishReason());
        assertEquals("", silent.text());
        // a finished pass answers for itself, whatever the caller saw streamed
        ChatEngine.Completion done =
                new ChatEngine.Completion(
                        Message.assistant("Paris"),
                        result,
                        true,
                        10,
                        0,
                        PromptCache.Tier.FRESH,
                        null);
        assertEquals(
                "Paris",
                OpenAiChat.reply(done, Message.assistant("ignored"), List.of(), false).text());
    }

    @Test
    void anOmittedSeedStaysUnseeded() {
        Sampling defaults = new Sampling(0.8f, 0.95f, 40, 0.05f, null);
        assertNull(OpenAiChat.sampling(Map.of(), defaults).seed());
        assertEquals(42L, OpenAiChat.sampling(Map.of("seed", 42), defaults).seed());
    }

    @Test
    void reasoningKnobsAreNullUnlessGiven() {
        assertNull(OpenAiChat.reasoningMax(Map.of()));
        assertNull(OpenAiChat.reasoningCutoffMessage(Map.of()));
        assertEquals(64, OpenAiChat.reasoningMax(Map.of("max_reasoning_tokens", 64)));
        assertEquals(
                "... Let me wrap up.",
                OpenAiChat.reasoningCutoffMessage(
                        Map.of("reasoning_cutoff_message", "... Let me wrap up.")));
    }

    @Test
    void malformedEchoedArgumentsAreTheClientsFault() {
        // an assistant turn echoed with a model's broken JSON: a 400 through the worker's
        // IllegalArgumentException mapping, never a 500 with a stack trace
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> OpenAiChat.parseArguments("{\"city\": Paris"));
        assertTrue(e.getMessage().contains("arguments"), e.getMessage());
    }

    @Test
    void reasoningEffortIsTheOpenAiSpellingOfThinking() {
        assertEquals(false, OpenAiChat.thinking(Map.of("reasoning_effort", "none"), true));
        assertEquals(true, OpenAiChat.thinking(Map.of("reasoning_effort", "low"), false));
        assertEquals(
                false,
                OpenAiChat.thinking(
                        Map.of(
                                "reasoning_effort",
                                "high",
                                "chat_template_kwargs",
                                Map.of("enable_thinking", false)),
                        true),
                "the explicit kwarg wins");
        assertEquals(true, OpenAiChat.thinking(Map.of(), true));
    }

    @Test
    void finishReasonSaysWhatEndedTheReply() {
        assertEquals(
                "tool_calls", OpenAiChat.finishReason(Generator.FinishReason.STOP, true, false));
        assertEquals("stop", OpenAiChat.finishReason(Generator.FinishReason.LENGTH, false, true));
        assertEquals("stop", OpenAiChat.finishReason(Generator.FinishReason.STOP, false, false));
        assertEquals(
                "length", OpenAiChat.finishReason(Generator.FinishReason.LENGTH, false, false));
        // a deadline is neither the model's end nor the token budget: "other", never "stop"
        assertEquals(
                "other", OpenAiChat.finishReason(Generator.FinishReason.TIMEOUT, false, false));
        assertEquals("other", OpenAiChat.finishReason(Generator.FinishReason.ABORT, false, false));
    }
}
