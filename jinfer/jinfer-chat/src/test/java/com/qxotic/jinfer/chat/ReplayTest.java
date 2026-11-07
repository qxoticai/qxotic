package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * A generated turn carries its record, the engine replays it into the model that wrote it, and a
 * record from another model renders from its parts.
 */
final class ReplayTest {
    private static final ContentKey SEED = new ContentKey("replay-test");
    private static final int PREFIX = '>';

    /** Turns as {@code r:text\n}, the reply opened by {@code A:>}: the prefix is the prompt's. */
    private static final ChatTemplate TEMPLATE =
            (conversation, batchCapacity, sink) -> {
                PromptWriter out =
                        new PromptWriter(TestLanguageModel.TOKENIZER, batchCapacity, sink);
                for (Message message : conversation.messages()) {
                    out.text(message.role().name().substring(0, 1) + ":");
                    if (!out.replay(message)) out.text(message.text());
                    out.text("\n");
                }
                out.text("A:").id(PREFIX);
                out.finish();
                return new ChatTemplate.ReplyState(
                        IntSequence.of(PREFIX), ReplyParser.spans(TestLanguageModel.TOKENIZER));
            };

    @Test
    void aReplyRecordsItsPrefixAndEveryGeneratedId() throws Exception {
        try (ChatEngine engine = engine()) {
            ChatEngine.Completion done = engine.complete(request(List.of(Message.user("hi")), 3));
            Verbatim record = done.reply().verbatim();
            assertEquals(SEED, record.model());
            assertEquals(0, record.frames());
            assertArrayEquals(
                    IntSequence.concatAll(
                                    IntSequence.of(PREFIX), IntSequence.of(done.result().tokens()))
                            .toArray(),
                    record.positions().toArray());
            assertEquals("xxx", done.reply().text());
        }
    }

    @Test
    void theModelsOwnRecordReplaysAndAnothersRendersFromParts() throws Exception {
        List<Content> parts = List.of(new Content.Text("zzz"));
        Message own =
                new Message(
                        Role.ASSISTANT,
                        parts,
                        new Verbatim(SEED, IntSequence.of(PREFIX, 'y'), IntSequence.empty()));
        Message foreign =
                new Message(
                        Role.ASSISTANT,
                        parts,
                        new Verbatim(
                                new ContentKey("another-model"),
                                IntSequence.of(PREFIX, 'y'),
                                IntSequence.empty()));
        try (ChatEngine engine = engine()) {
            assertEquals(
                    "a:>y\nA:>",
                    prompt(engine, own),
                    "the model's own record stands in for the turn");
            assertEquals(
                    "a:zzz\nA:>",
                    prompt(engine, foreign),
                    "another model's record is dropped, the parts render");
        }
    }

    private static String prompt(ChatEngine engine, Message history) {
        try (ChatEngine.Prepared prepared = engine.prepare(request(List.of(history), 1))) {
            return TestLanguageModel.TOKENIZER.decode(
                    IntSequence.of(Batch.tokenIds(prepared.encoded().prompt())));
        }
    }

    private static ChatEngine engine() {
        var loaded =
                new LoadedModel<>(
                        new TestLanguageModel(),
                        TestLanguageModel.TOKENIZER,
                        "",
                        Set.of(),
                        SEED,
                        Optional.of(TEMPLATE),
                        LoadedModel.SamplingDefaults.NONE);
        return new ChatEngine(loaded, "test", PromptCache.Options.DEFAULTS.withBlockBudget(0));
    }

    private static ChatEngine.Request request(List<Message> messages, int maxOutputTokens) {
        return new ChatEngine.Request(
                messages,
                List.of(),
                false,
                maxOutputTokens,
                null,
                null,
                Duration.ZERO,
                new Sampling(0, 1, 0, 0, 1L),
                null,
                null,
                List.of(),
                null);
    }
}
