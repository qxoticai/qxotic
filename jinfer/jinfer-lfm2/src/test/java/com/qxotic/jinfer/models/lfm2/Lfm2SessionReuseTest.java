package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Tag("integration")
final class Lfm2SessionReuseTest {
    private static final Sampling GREEDY = new Sampling(0, 1, 0, 0, 42L);

    @ParameterizedTest
    @CsvSource({"false, 0", "true, 0", "true, 96"})
    void verbatimReplyResumesWithoutBlocks(boolean thinking, int reasoningBudget) {
        Path path = TestModels.require("hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q8_0");
        PromptCache.Options options =
                PromptCache.Options.DEFAULTS
                        .withContextCapacity(512)
                        .withRetainedSessions(1)
                        .withBlockBudget(0);
        try (ChatEngine engine = new ChatEngine(path, Map.of(), options)) {
            List<Message> messages =
                    new ArrayList<>(
                            List.of(
                                    Message.system("Be concise."),
                                    Message.user("What is the capital of France?")));
            ChatEngine.Completion first;
            IntSequence retained;
            try (ChatEngine.Prepared prepared =
                    engine.prepare(request(messages, thinking, reasoningBudget))) {
                IntSequence prompt = IntSequence.of(Batch.tokenIds(prepared.encoded().prompt()));
                first = engine.complete(prepared, ChatEngine.ReplySink.NONE);
                assertEquals(Generator.FinishReason.STOP, first.result().finishReason());
                assertFalse(first.reply().text().isBlank());
                assertEquals(thinking && reasoningBudget > 0, first.reasoningTokens() > 0);
                retained = prompt.concat(IntSequence.of(first.result().tokens()));
            }

            messages.add(first.reply());
            messages.add(Message.user("And the capital of Germany?"));
            ChatEngine.Request followup = request(messages, thinking, reasoningBudget);
            ChatEngine.Completion warm;
            try (ChatEngine.Prepared prepared = engine.prepare(followup)) {
                assertTrue(
                        IntSequence.of(Batch.tokenIds(prepared.encoded().prompt()))
                                .startsWith(retained),
                        "the appended conversation must preserve every ingested token");
                warm = engine.complete(prepared, ChatEngine.ReplySink.NONE);
            }
            assertEquals(PromptCache.Tier.SESSION, warm.tier());
            assertEquals(Generator.FinishReason.STOP, warm.result().finishReason());
            assertFalse(warm.reply().text().isBlank());
            assertEquals(retained.length(), warm.restoredTokens());
            assertEquals(1, engine.cacheSample().stateAllocations());
            assertEquals(0, engine.cacheSample().blocks());

            try (ChatEngine cold =
                    new ChatEngine(path, Map.of(), options.withRetainedSessions(0))) {
                ChatEngine.Completion fresh = cold.complete(followup);
                assertEquals(PromptCache.Tier.FRESH, fresh.tier());
                assertEquals(fresh.result().finishReason(), warm.result().finishReason());
                assertArrayEquals(fresh.result().tokens(), warm.result().tokens());
            }
        }
    }

    private static ChatEngine.Request request(
            List<Message> messages, boolean thinking, int reasoningBudget) {
        return ChatEngine.Request.builder(messages, GREEDY)
                .thinking(thinking)
                .maxReasoningTokens(reasoningBudget)
                .maxOutputTokens(192)
                .build();
    }
}
