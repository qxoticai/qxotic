package com.qxotic.jinfer.models.inkling2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Inkling 2 Mini Q8_0 and its vision tower end to end. */
@Tag("integration")
final class Inkling2IntegrationTest {
    static final String MMPROJ_REF =
            "hf.co/ggml-org/Inkling-2-Mini-GGUF/mmproj-Inkling-2-Mini-Q8_0.gguf";

    private static final int PARIS = 152_717; // " Paris"

    private static Arena arena;
    private static LoadedModel<?> loaded;
    private static Inkling2 model;

    @BeforeAll
    static void load() throws Exception {
        arena = Arenas.newCrossThread();
        loaded =
                Models.load(
                        TestModels.require(Inkling2ChatTemplateTest.MODEL_REF),
                        arena,
                        Map.of("media", TestModels.require(MMPROJ_REF)));
        model = assertInstanceOf(Inkling2.class, loaded.model());
    }

    @AfterAll
    static void close() {
        Arenas.close(arena);
    }

    @Test
    void loadsThroughTheProviderWithTheContainersSampling() {
        assertInstanceOf(Inkling2ChatTemplate.class, loaded.template().orElseThrow());
        assertEquals(1f, loaded.samplingDefaults().temperature());
        assertEquals(1f, loaded.samplingDefaults().topP());
        assertEquals(50, loaded.samplingDefaults().topK());
        assertEquals(0f, loaded.samplingDefaults().minP());
        assertEquals(200_006, loaded.stopTokens().iterator().next(), "the end of the turn");

        Inkling2.Configuration config = model.configuration();
        assertEquals(24, config.numberOfLayers());
        assertEquals(2_304, config.embeddingLength());
        assertEquals(200_058, config.vocabularySize());
        assertEquals(201_024, config.paddedVocabularySize());
        assertEquals(1_048_576, config.maxContextLength());
        assertEquals(24, config.numberOfHeads());
        assertEquals(8, config.numberOfKeyValueHeads());
        assertEquals(128, config.headSize());
        assertEquals(128, config.expertCount());
        assertEquals(8, config.expertUsedCount());
        assertEquals(1_152, config.expertFeedForwardLength());
        assertEquals(1_024, config.slidingWindow());
        assertEquals(1_024, config.relativeExtent());
        for (int layer = 0; layer < 24; layer++)
            assertEquals(layer % 6 != 5, config.isSwa()[layer], "layer " + layer);
        assertEquals(6, config.loopStart());
        assertEquals(18, config.loopEnd());
        assertEquals(3, config.loopCount());
        assertEquals(48, config.slots());

        int[] prompt = model.tokenizer().encodeToArray("The capital of France is");
        assertArrayEquals(new int[] {371, 151_451, 149_879, 151_194, 149_884}, prompt);
        try (Inkling2.State state = model.newState(64, prompt.length)) {
            model.ingest(state, Batch.prefill(prompt));
            assertEquals(PARIS, Inkling2Test.argmax(logits(state)));
        }
    }

    @Test
    void batchedAndIncrementalInferenceAgreeAndResetReplays() {
        int[] tokens = model.tokenizer().encodeToArray("The quick brown fox jumps over the");
        float[] batched;
        try (Inkling2.State state = model.newState(64, tokens.length)) {
            model.ingest(state, Batch.prefill(tokens));
            batched = logits(state);
        }
        try (Inkling2.State state = model.newState(64, 1)) {
            for (int token : tokens) model.ingest(state, Batch.step(token));
            float[] incremental = logits(state);
            System.out.printf(
                    "Inkling 2 batch/step parity: cosine=%.9f%n",
                    Inkling2Test.cosine(batched, incremental));
            assertEquals(Inkling2Test.argmax(batched), Inkling2Test.argmax(incremental));
            // a step reads its history from the FP16 caches and runs the int8 gemv where the
            // batch runs the gemm
            assertTrue(Inkling2Test.cosine(batched, incremental) > .995);

            state.reset();
            for (int token : tokens) model.ingest(state, Batch.step(token));
            assertArrayEquals(incremental, logits(state), .01f);
        }
    }

    /** A defined prompt longer than the window and the bias extent restores from its blocks. */
    @Test
    void aDefinedPromptRestoresFromItsBlocks() {
        int[] tokens = new Random(7).ints(1_200, 1_000, 190_000).toArray();
        List<Batch> prompt =
                List.of(
                        Batch.prefill(Arrays.copyOfRange(tokens, 0, 400)),
                        Batch.prefill(Arrays.copyOfRange(tokens, 400, 800)),
                        Batch.prefill(Arrays.copyOfRange(tokens, 800, tokens.length - 1)),
                        Batch.step(tokens[tokens.length - 1]));
        PromptCache.Options options =
                PromptCache.Options.DEFAULTS
                        .withRetainedSessions(0)
                        .withContextCapacity(2_048)
                        .withBlockBudget(512L << 20);
        try (PromptCache<Inkling2.State> cache = PromptCache.of(model, loaded.seed(), options)) {
            CacheResult fresh = cachedLogits(cache, prompt);
            cache.define(prompt);
            CacheResult restored = cachedLogits(cache, prompt);

            assertEquals(PromptCache.Tier.FRESH, fresh.tier());
            assertEquals(PromptCache.Tier.BLOCKS, restored.tier());
            assertEquals(tokens.length - 1, restored.restored());
            assertEquals(
                    Inkling2Test.argmax(fresh.logits()), Inkling2Test.argmax(restored.logits()));
            assertTrue(Inkling2Test.cosine(fresh.logits(), restored.logits()) > .999);
        }
    }

    @Test
    void repliesWithAndWithoutReasoning() throws Exception {
        try (ChatEngine engine =
                new ChatEngine(loaded, "inkling2-test", PromptCache.Options.DEFAULTS)) {
            for (boolean thinking : new boolean[] {false, true}) {
                // the mark that closes a reasoning block closes the answer too: masked with
                // thinking off, the reply ran on to the token budget
                Message reply =
                        stopped(
                                engine,
                                ask("What is the capital of France?").thinking(thinking).build());
                assertTrue(reply.text().contains("Paris"), reply.text());
                assertEquals(thinking, !reply.reasoning().isEmpty(), "thinking " + thinking);
            }
        }
    }

    /** Thinking off, a spent reasoning budget, a grammar and a forced call all end their turn. */
    @Test
    void everyKindOfReplyEndsItsTurn() throws Exception {
        Tool weather =
                Tool.of(
                        "get_weather",
                        "Get the current weather for a city",
                        Map.of(
                                "type", "object",
                                "properties", Map.of("city", Map.of("type", "string")),
                                "required", List.of("city")));
        ChatEngine.Request spent =
                ask("Name three primary colors.").thinking(true).maxReasoningTokens(8).build();
        ChatEngine.Request stated =
                ask("Is 17 a prime number?").grammar("root ::= \"yes\" | \"no\"").build();
        ChatEngine.Request forced =
                ask("Hello.")
                        .tools(List.of(weather))
                        .forcedTool(new ChatEngine.ForcedTool.Named("get_weather"))
                        .build();
        // made-up call names followed the block header here before unoffered calls were masked
        ChatEngine.Request french = ask("Answer in French: What is the capital of Italy?").build();
        try (ChatEngine engine =
                new ChatEngine(loaded, "inkling2-test", PromptCache.Options.DEFAULTS)) {
            assertFalse(stopped(engine, spent).text().isBlank());
            assertEquals("yes", stopped(engine, stated).text());
            assertTrue(stopped(engine, french).text().contains("Rome"));
            Message call = stopped(engine, forced);
            assertEquals(
                    "get_weather",
                    assertInstanceOf(Content.ToolCall.class, call.content().getFirst()).name());
        }
    }

    /** A greedy request, thinking off unless asked. */
    private static ChatEngine.Request.Builder ask(Message message) {
        return ChatEngine.Request.builder(List.of(message), Sampling.GREEDY)
                .thinking(false)
                .maxOutputTokens(256);
    }

    private static ChatEngine.Request.Builder ask(String prompt) {
        return ask(Message.user(prompt));
    }

    private static Message stopped(ChatEngine engine, ChatEngine.Request request) {
        ChatEngine.Completion completion = engine.complete(request, ChatEngine.ReplySink.NONE);
        assertEquals(
                Generator.FinishReason.STOP,
                completion.result().finishReason(),
                completion.reply().text());
        return completion.reply();
    }

    /** A red image of 64 x 48 pixels: one padded cell of 40 pixels more each way, four rows. */
    @Test
    void readsAnImage() throws Exception {
        float[] values = new float[64 * 48 * 3];
        for (int pixel = 0; pixel < 64 * 48; pixel++) values[pixel * 3] = 1f;
        Media.Image red = new Media.Image(values, 48, 64, 3);
        MediaProjector<Media.Image> vision = model.projector(Media.Image.class).orElseThrow();
        assertEquals(4, vision.positions(red));

        Message question =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Media(red),
                                new Content.Text("What color is this image? One word.")));
        try (ChatEngine engine =
                new ChatEngine(loaded, "inkling2-test", PromptCache.Options.DEFAULTS)) {
            String reply = stopped(engine, ask(question).build()).text();
            assertTrue(reply.toLowerCase(Locale.ROOT).contains("red"), reply);
        }
    }

    private static float[] logits(Inkling2.State state) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state), "logits"), "logits");
    }

    private static CacheResult cachedLogits(PromptCache<Inkling2.State> cache, List<Batch> prompt) {
        return cache.serve(
                prompt,
                (state, serving) ->
                        new CacheResult(serving.tier(), serving.restored(), logits(state)));
    }

    private record CacheResult(PromptCache.Tier tier, int restored, float[] logits) {}
}
