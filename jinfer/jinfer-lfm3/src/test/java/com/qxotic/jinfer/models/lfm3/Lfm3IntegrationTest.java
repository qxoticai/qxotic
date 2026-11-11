package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** LFM3 the way an application meets it: the provider, the engine, and the block cache. */
@Tag("integration")
final class Lfm3IntegrationTest {

    static final String TEXT = "hf.co/LiquidAI/LFM3-1.7B-GGUF:Q8_0";
    static final String OMNI = "hf.co/LiquidAI/LFM3-Omni-2B-GGUF:Q8_0";
    static final String OMNI_MMPROJ =
            "hf.co/LiquidAI/LFM3-Omni-2B-GGUF/mmproj-LFM3-Omni-2B-F16.gguf";
    private static final Sampling GREEDY = new Sampling(0f, 1f, 0, 0f, null);

    @Test
    void omniLoadsThroughTheProviderWithBothTowers() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> loaded =
                    Models.load(
                            TestModels.require(OMNI),
                            arena,
                            Map.of("media", TestModels.require(OMNI_MMPROJ)));
            Lfm3 model = assertInstanceOf(Lfm3.class, loaded.model());
            assertInstanceOf(Lfm3ChatTemplate.class, loaded.template().orElseThrow());
            assertTrue(model.projector(Media.Image.class).isPresent());
            assertTrue(model.projector(Media.Audio.class).isPresent());
            assertTrue(model.projector(Media.Video.class).isEmpty());
            assertEquals(
                    model.tokenizer().vocabulary().id("<|im_end|>"),
                    loaded.stopTokens().iterator().next());
            // the GGUF's general.sampling.*: Omni ships greedy
            assertEquals(0f, loaded.samplingDefaults().temperature());
            assertEquals(1, loaded.samplingDefaults().topK());

            Lfm3.Configuration c = model.configuration();
            assertEquals(24, c.numberOfLayers());
            assertEquals(2048, c.embeddingLength());
            assertEquals(128_000, c.vocabularySize());
            assertEquals(256, c.slidingWindow());
            assertEquals(7, c.shortConvLCache());
            for (int layer = 0; layer < 24; layer++) {
                boolean attention = layer >= 2 && layer % 2 == 0;
                assertEquals(!attention, c.isShortConv(layer), "layer " + layer);
                assertEquals(attention && layer % 4 == 0, c.isSwa(layer), "layer " + layer);
            }
        }
    }

    @Test
    void theEngineStreamsCallsToolsReasonsAndSeesImages() throws Exception {
        Arena weights = Arenas.newCrossThread();
        try {
            ChatEngine engine =
                    new ChatEngine(
                            Models.load(
                                    TestModels.require(OMNI),
                                    weights,
                                    Map.of("media", TestModels.require(OMNI_MMPROJ))),
                            "lfm3-test",
                            PromptCache.Options.DEFAULTS);
            // streaming and blocking agree
            ChatEngine.Request capital =
                    request(
                            List.of(Message.user("What is the capital of France? One word.")),
                            List.of(),
                            false);
            StringBuilder streamed = new StringBuilder();
            ChatEngine.Completion blocking = engine.complete(capital, ChatEngine.ReplySink.NONE);
            ChatEngine.Completion streaming =
                    engine.complete(
                            capital,
                            new ChatEngine.ReplySink() {
                                @Override
                                public void on(ChatEngine.Delta delta) {
                                    if (delta.channel().equals(Channel.CONTENT))
                                        streamed.append(delta.text());
                                }
                            });
            assertEquals("Paris", blocking.reply().text().strip());
            assertEquals(blocking.reply().text(), streaming.reply().text());
            assertEquals(blocking.reply().text(), streamed.toString());

            // a tool call parses from the model's own XML wire
            Tool weather =
                    new Tool(
                            "get_weather",
                            Map.of(
                                    "type",
                                    "function",
                                    "function",
                                    Map.of(
                                            "name",
                                            "get_weather",
                                            "parameters",
                                            Map.of(
                                                    "type",
                                                    "object",
                                                    "properties",
                                                    Map.of("city", Map.of("type", "string"))))));
            ChatEngine.Completion call =
                    engine.complete(
                            request(
                                    List.of(Message.user("What's the weather in Paris?")),
                                    List.of(weather),
                                    false),
                            ChatEngine.ReplySink.NONE);
            Content.ToolCall toolCall =
                    (Content.ToolCall)
                            call.reply().content().stream()
                                    .filter(Content.ToolCall.class::isInstance)
                                    .findFirst()
                                    .orElseThrow();
            assertEquals("get_weather", toolCall.name());
            assertEquals(Map.of("city", "Paris"), toolCall.arguments());

            // an image, through the whole engine
            Media.Image cat = ImageCodec.load(Oracle.ROOT.resolve("../images/cat.png"));
            ChatEngine.Completion image =
                    engine.complete(
                            request(
                                    List.of(
                                            new Message(
                                                    Role.USER,
                                                    List.of(
                                                            new Content.Media(cat),
                                                            new Content.Text(
                                                                    "What animal is this? One"
                                                                            + " word.")))),
                                    List.of(),
                                    false),
                            ChatEngine.ReplySink.NONE);
            assertTrue(image.reply().text().toLowerCase().contains("cat"), image.reply().text());
            assertTrue(image.promptTokens() > 300, "the image rows are context positions");
        } finally {
            Arenas.close(weights);
        }
    }

    /**
     * Asked for audio, Omni opens a spoken reply ({@code <|mixed_start|>}), which the control rule
     * used to cut before a word; a text reply forbids every speech-lane token and answers in words,
     * under LiquidAI's voice-chat prompt too ({@code Lfm3SpokenChatTest}).
     */
    @Test
    void aTextReplyNeverSwitchesToSpeech() throws Exception {
        Arena weights = Arenas.newCrossThread();
        try {
            LoadedModel<?> loaded =
                    Models.load(
                            TestModels.require(OMNI),
                            weights,
                            Map.of("media", TestModels.require(OMNI_MMPROJ)));
            ChatEngine engine = new ChatEngine(loaded, "lfm3-test", PromptCache.Options.DEFAULTS);
            ChatEngine.Completion reply =
                    engine.complete(
                            request(
                                    List.of(
                                            Message.system("Respond with audio."),
                                            Message.user("Hi! How are you?")),
                                    List.of(),
                                    false),
                            ChatEngine.ReplySink.NONE);
            assertTrue(reply.reply().text().contains("well"), reply.reply().text());
            Set<Integer> lanes = loaded.template().orElseThrow().replyForbidden();
            for (int token : reply.result().tokens()) assertTrue(!lanes.contains(token));
        } finally {
            Arenas.close(weights);
        }
    }

    @Test
    void theTextModelReasonsWhenAsked() throws Exception {
        Arena weights = Arenas.newCrossThread();
        try {
            ChatEngine engine =
                    new ChatEngine(
                            Models.load(TestModels.require(TEXT), weights),
                            "lfm3-test",
                            PromptCache.Options.DEFAULTS);
            ChatEngine.Completion reply =
                    engine.complete(
                            request(
                                    List.of(Message.user("What is 12 + 30?")),
                                    List.of(),
                                    true,
                                    400),
                            ChatEngine.ReplySink.NONE);
            assertTrue(reply.reasoningTokens() > 0, "a think span");
            assertTrue(reply.reply().text().contains("42"), reply.reply().text());
        } finally {
            Arenas.close(weights);
        }
    }

    /**
     * Two blocks - the second longer than the sliding window, so its ring wraps - restored into a
     * fresh state resume bit for bit where the original continues.
     */
    @Test
    void checkpointsRestoreTheRingsAndTheConvTaps() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model = Lfm3.loadModel(TestModels.require(TEXT), arena);
            int[] tokens = Oracle.ints(Oracle.dir("text-f32").resolve("tokens.i32"));
            int cut = 200, end = 600;
            CheckpointCodec<Lfm3.State> codec = model.checkpointCodec().orElseThrow();
            try (Lfm3.State original = model.newState(1024, 512);
                    Lfm3.State restored = model.newState(1024, 512);
                    Arena blobs = Arena.ofConfined()) {
                MemorySegment first = blobs.allocate(codec.byteSize(cut));
                MemorySegment second = blobs.allocate(codec.byteSize(end - cut));
                model.ingest(original, Batch.prefill(Arrays.copyOfRange(tokens, 0, cut)));
                codec.capture(original, 0, cut, first);
                model.ingest(original, Batch.prefill(Arrays.copyOfRange(tokens, cut, end)));
                codec.capture(original, cut, end, second);
                codec.restore(restored, 0, cut, first);
                codec.restore(restored, cut, end, second);
                restored.resumeAt(end);
                for (int step = end; step < tokens.length; step++) {
                    model.ingest(original, Batch.step(tokens[step]));
                    model.ingest(restored, Batch.step(tokens[step]));
                    assertArrayEquals(
                            logits(model, original), logits(model, restored), "step " + step);
                }
            }
        }
    }

    @Test
    void mediaWithoutItsTowerAndOverlongIngestsAreRefused() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model = Lfm3.loadModel(TestModels.require(TEXT), arena);
            try (Lfm3.State state = model.newState(8, 8)) {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> model.ingest(state, Batch.prefill(new int[9])));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> model.ingest(state, Batch.prefill(new int[] {-1})));
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.ingest(
                                        state,
                                        Batch.embeddings(
                                                Views.allocateF32(
                                                        com.qxotic.jota.memory.MemoryAllocators
                                                                .ofArena(arena),
                                                        1,
                                                        2048),
                                                1,
                                                false)));
            }
        }
    }

    private static float[] logits(Lfm3 model, Lfm3.State state) {
        int vocab = model.configuration().vocabularySize();
        return Oracle.rows(Views.castToSegmentBacked(model.logits(state, 0), "logits"), 1, vocab);
    }

    private static ChatEngine.Request request(
            List<Message> messages, List<Tool> tools, boolean thinking) {
        return request(messages, tools, thinking, 48);
    }

    private static ChatEngine.Request request(
            List<Message> messages, List<Tool> tools, boolean thinking, int max) {
        return new ChatEngine.Request(
                messages,
                tools,
                thinking,
                max,
                null,
                null,
                Duration.ZERO,
                GREEDY,
                null,
                null,
                List.of(),
                null);
    }
}
