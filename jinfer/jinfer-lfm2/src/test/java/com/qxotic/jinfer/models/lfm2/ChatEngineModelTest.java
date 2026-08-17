package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.boundary.Arenas;
import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/**
 * The engine end to end on a real (small) model: blocking and streaming agree, a cancelled pass
 * reports no reply, a follow-up turn rides the session cache, and a defined prompt is served from
 * blocks. Skipped when the model cache has no LFM2.5-350M (see {@link TestModels}).
 */
final class ChatEngineModelTest {

    private static final String REF = "hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0";
    private static final Sampling GREEDY = new Sampling(0f, 1f, 0, 0f, null);

    private static ChatEngine.Request request(List<Message> messages) {
        return request(messages, Duration.ZERO);
    }

    private static ChatEngine.Request request(List<Message> messages, Duration timeout) {
        return new ChatEngine.Request(
                messages, List.of(), false, 32, null, null, timeout, GREEDY, null, null, List.of(),
                null);
    }

    @Test
    void blockingStreamingCancellationAndCacheReuse() throws Exception {
        Path path = TestModels.require(REF);
        Arena weights = Arenas.newCrossThread();
        ChatEngine engine =
                new ChatEngine(
                        Models.load(path, weights), "lfm2-test", PromptCache.Options.DEFAULTS);
        try {
            int bos = SpecialTokens.require(engine.loaded().tokenizer(), "<|startoftext|>");
            int[] bare = engine.loaded().tokenizer().encode("hello").toArray();
            int[] explicit = IntSequence.of(bos).concat(IntSequence.of(bare)).toArray();
            try (ChatEngine.Prepared prepared =
                            engine.prepareRaw(bare, GREEDY, 1, Duration.ZERO, null, List.of());
                    ChatEngine.Prepared alreadyStarted =
                            engine.prepareRaw(
                                    explicit, GREEDY, 1, Duration.ZERO, null, List.of())) {
                assertArrayEquals(explicit, Batch.tokenIds(prepared.encoded().prompt()));
                assertArrayEquals(explicit, Batch.tokenIds(alreadyStarted.encoded().prompt()));
                assertEquals(explicit.length, prepared.promptTokens());
                assertEquals(explicit.length, alreadyStarted.promptTokens());
            }

            ChatEngine.Request request = request(List.of(Message.user("Name one color.")));

            // blocking: a fresh prompt computes every position
            ChatEngine.Completion blocking = engine.complete(request, ChatEngine.ReplySink.NONE);
            assertFalse(blocking.cancelled());
            assertFalse(blocking.reply().text().isBlank());
            assertTrue(blocking.result().completionTokens() > 0);
            assertEquals(PromptCache.Tier.FRESH, blocking.tier());
            assertEquals(0, blocking.restoredTokens());

            // streaming: the deltas concatenate to the finished reply's text, and their tokens
            // re-encode it exactly (Prepared is single-use: the parser is stateful)
            List<ChatEngine.Delta> deltas = new ArrayList<>();
            ChatEngine.Completion streaming =
                    engine.complete(
                            request,
                            new ChatEngine.ReplySink() {
                                @Override
                                public void on(ChatEngine.Delta delta) {
                                    deltas.add(delta);
                                }
                            });
            assertFalse(deltas.isEmpty());
            String streamedText =
                    deltas.stream().map(ChatEngine.Delta::text).collect(Collectors.joining());
            assertEquals(streaming.reply().text(), streamedText);
            IntSequence.Builder ids = IntSequence.newBuilder();
            deltas.forEach(d -> d.tokens().forEachInt(ids::add));
            assertTrue(ids.build().length() > 0);
            IntSequence.Builder all = IntSequence.newBuilder();
            deltas.forEach(d -> d.tokens().forEachInt(all::add));
            assertEquals(streamedText, engine.loaded().tokenizer().decode(all.build().toArray()));

            // a follow-up turn strictly extends the retained session: nothing recomputes
            ChatEngine.Request followUp =
                    request(
                            List.of(
                                    Message.user("Name one color."),
                                    Message.assistant(blocking.reply().text()),
                                    Message.user("And another?")));
            ChatEngine.Completion second = engine.complete(followUp, ChatEngine.ReplySink.NONE);
            assertFalse(second.cancelled());
            assertTrue(
                    second.restoredTokens() > 0,
                    "the follow-up must reuse the session, got " + second.tier());

            // a defined prompt is served from blocks on a fresh root
            Conversation prefix =
                    new Conversation(
                            List.of(
                                    Message.system("Answer in one word."),
                                    Message.user("Name one animal.")),
                            List.of(),
                            false,
                            "");
            engine.definePrompt(prefix);
            ChatEngine.Completion fromBlocks =
                    engine.complete(request(prefix.messages()), ChatEngine.ReplySink.NONE);
            assertEquals(PromptCache.Tier.BLOCKS, fromBlocks.tier());
            assertTrue(fromBlocks.restoredTokens() > 0);

            // cancellation: the pass ends silently, no reply and no result
            ChatEngine.Completion cancelled =
                    engine.complete(
                            request,
                            new ChatEngine.ReplySink() {
                                @Override
                                public boolean cancelled() {
                                    return true;
                                }
                            });
            assertTrue(cancelled.cancelled());
            assertNull(cancelled.reply());
            assertNull(cancelled.result());
        } finally {
            engine.close();
            engine.close(); // idempotent
        }
        assertThrows(
                IllegalStateException.class,
                () ->
                        engine.complete(
                                request(List.of(Message.user("hi"))), ChatEngine.ReplySink.NONE));
        weights.close();
    }

    @Test
    void expiredDeadlineStopsPrefillBeforeAnyToken() throws Exception {
        Path path = TestModels.require(REF);
        Arena weights = Arenas.newCrossThread();
        ChatEngine engine =
                new ChatEngine(
                        Models.load(path, weights), "lfm2-test", PromptCache.Options.DEFAULTS);
        try {
            List<ChatEngine.Delta> deltas = new ArrayList<>();
            ChatEngine.Completion completion =
                    engine.complete(
                            request(List.of(Message.user("Name one color.")), Duration.ofNanos(1)),
                            new ChatEngine.ReplySink() {
                                @Override
                                public void on(ChatEngine.Delta delta) {
                                    deltas.add(delta);
                                }
                            });
            assertTrue(deltas.isEmpty(), "no token is sampled past the deadline");
            assertFalse(completion.cancelled());
            assertEquals(Generator.FinishReason.TIMEOUT, completion.result().finishReason());
            assertEquals(0, completion.result().completionTokens());
            assertTrue(
                    completion.result().promptTime().toNanos() > 0,
                    "the interrupted prefill still reports its time");

            // the interrupted session was discarded and committed nothing: the same prompt
            // computes fresh, then generates normally
            ChatEngine.Completion retry =
                    engine.complete(
                            request(List.of(Message.user("Name one color."))),
                            ChatEngine.ReplySink.NONE);
            assertEquals(PromptCache.Tier.FRESH, retry.tier());
            assertFalse(retry.reply().text().isBlank());
        } finally {
            engine.close();
        }
        weights.close();
    }
}
