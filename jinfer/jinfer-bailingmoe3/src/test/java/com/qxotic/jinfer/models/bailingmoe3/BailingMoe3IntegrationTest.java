package com.qxotic.jinfer.models.bailingmoe3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class BailingMoe3IntegrationTest {
    private static final String MODEL_PROPERTY = "jinfer.bailingmoe3.model";
    private static final String LLAMA_LOGITS_PROPERTY = "jinfer.bailingmoe3.llamaLogits";
    private static final String LLAMA_LONG_LOGITS_PROPERTY = "jinfer.bailingmoe3.llamaLongLogits";
    private static final String LONG_PARITY_PROMPT =
            "The quick brown fox jumps over the lazy dog while a curious cat watches from the"
                    + " window.";
    private static final Sampling GREEDY = new Sampling(0f, 1f, 0, 0f, 1L);
    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    Map.of(
                            "name",
                            "get_weather",
                            "description",
                            "Get the weather for a city.",
                            "parameters",
                            Map.of(
                                    "type",
                                    "object",
                                    "properties",
                                    Map.of(
                                            "city",
                                            Map.of("type", "string", "description", "City name")),
                                    "required",
                                    List.of("city"))));

    @Test
    @Tag("integration")
    void batchedAndIncrementalInferenceAgreeAndResetReplays() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            BailingMoe3 model = BailingMoe3.loadModel(requiredFile(MODEL_PROPERTY), arena);
            int[] tokens = model.tokenizer().encodeToArray("Hello world");
            assertTrue(tokens.length > 1, "fixture must exercise positions after zero");

            float[] batched;
            try (BailingMoe3.State state = model.newState(32, tokens.length)) {
                model.ingest(state, Batch.prefill(tokens));
                batched = logits(model, state);
            }

            try (BailingMoe3.State state = model.newState(32, 1)) {
                for (int token : tokens) model.ingest(state, Batch.step(token));
                float[] incremental = logits(model, state);
                assertEquals(argmax(batched), argmax(incremental));
                assertTrue(maxAbsDifference(batched, incremental) < .1f);

                state.reset();
                for (int token : tokens) model.ingest(state, Batch.step(token));
                float[] replay = logits(model, state);
                assertEquals(argmax(incremental), argmax(replay));
                assertArrayEquals(incremental, replay, .001f);
            }
        }
    }

    @Test
    @Tag("integration")
    void fullVocabularyLogitsMatchLlamaCpp() throws Exception {
        assertLlamaCppParity("Hello", requiredFile(LLAMA_LOGITS_PROPERTY), 1, false);
    }

    @Test
    @Tag("integration")
    void fullVocabularyLogitsMatchLlamaCppAfterMultiplePositions() throws Exception {
        assertLlamaCppParity(LONG_PARITY_PROMPT, requiredFile(LLAMA_LONG_LOGITS_PROPERTY), 8, true);
    }

    private static void assertLlamaCppParity(
            String prompt, Path reference, int minimumTokens, boolean incremental)
            throws Exception {
        float[] expected = readFloats(reference);
        try (Arena arena = Arena.ofShared()) {
            BailingMoe3 model = BailingMoe3.loadModel(requiredFile(MODEL_PROPERTY), arena);
            int[] tokens = model.tokenizer().encodeToArray(prompt);
            assertTrue(
                    tokens.length >= minimumTokens,
                    "fixture must exercise at least " + minimumTokens + " tokens");
            try (BailingMoe3.State state = model.newState(64, incremental ? 1 : tokens.length)) {
                float[] actual;
                if (incremental) {
                    actual = null;
                    for (int token : tokens) {
                        model.ingest(state, Batch.step(token));
                        actual = logits(model, state);
                    }
                } else {
                    model.ingest(state, Batch.prefill(tokens));
                    actual = logits(model, state);
                }
                assertEquals(expected.length, actual.length);
                double rmse = rmse(expected, actual);
                float max = maxAbsDifference(expected, actual);
                double cosine = cosineSimilarity(expected, actual);
                System.out.printf(
                        "llama.cpp parity: rmse=%.6f max=%.6f cosine=%.9f%n", rmse, max, cosine);
                assertEquals(argmax(expected), argmax(actual));
                // Q8 matmuls use different kernels and recurrent state amplifies their small
                // first-token differences. The longer bound is still tight enough to reject a
                // wrong KDA recurrence (cosine drops below .97 in the row/column decay case).
                assertTrue(rmse < (incremental ? .6 : .05), "RMSE " + rmse);
                assertTrue(max < (incremental ? 3f : .25f), "max difference " + max);
                assertTrue(cosine > (incremental ? .99 : .99995), "cosine similarity " + cosine);
            }
        }
    }

    @Test
    @Tag("integration")
    void nativePromptMatchesGgufTemplateAndReplyParserUsesBailingV3WireFormat() throws Exception {
        try (Arena weights = Arenas.newCrossThread()) {
            LoadedModel<?> loaded = Models.load(requiredFile(MODEL_PROPERTY), weights);
            assertInstanceOf(BailingMoe3ChatTemplate.class, loaded.template().orElseThrow());
            assertEquals(1f, loaded.samplingDefaults().temperature());
            assertEquals(.95f, loaded.samplingDefaults().topP());
            assertEquals(20, loaded.samplingDefaults().topK());
            try (ChatEngine engine =
                    new ChatEngine(loaded, "bailing-tools", PromptCache.Options.DEFAULTS)) {
                List<Message> history =
                        List.of(
                                Message.system("Answer briefly."),
                                Message.user("What is the weather in Paris?"),
                                new Message(
                                        Role.ASSISTANT,
                                        List.of(
                                                new Content.Reasoning(
                                                        List.of(
                                                                new Content.Text(
                                                                        "I should check the"
                                                                                + " weather.")),
                                                        IntSequence.empty()),
                                                new Content.Text("I will check."),
                                                new Content.ToolCall(
                                                        "call-1",
                                                        WEATHER.name(),
                                                        Map.of("city", "Paris")))),
                                new Message(
                                        Role.TOOL,
                                        List.of(
                                                new Content.ToolResult(
                                                        "call-1", "18 C and sunny"))));
                Conversation conversation = new Conversation(history, List.of(WEATHER), false, "");
                ChatEngine.Encoded encoded = engine.encode(conversation, null);
                ChatEngine.Encoded jinja =
                        engine.encode(conversation, Map.of("parity_check", true));
                int[] jinjaIds = Batch.tokenIds(jinja.prompt());
                int[] nativeIds = Batch.tokenIds(encoded.prompt());
                assertEquals(
                        loaded.tokenizer().decode(jinjaIds), loaded.tokenizer().decode(nativeIds));
                assertArrayEquals(jinjaIds, nativeIds);
                Conversation thinking = new Conversation(history, List.of(WEATHER), true, "");
                assertArrayEquals(
                        Batch.tokenIds(
                                engine.encode(thinking, Map.of("parity_check", true)).prompt()),
                        Batch.tokenIds(engine.encode(thinking, null).prompt()));
                String prompt = loaded.tokenizer().decode(Batch.tokenIds(encoded.prompt()));
                assertTrue(prompt.contains("# Tools"));
                assertTrue(prompt.contains("<tool_call>get_weather"));
                assertTrue(
                        prompt.contains("<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>"));
                assertTrue(
                        prompt.contains(
                                "<role>OBSERVATION</role>\n"
                                        + "<tool_response>\n"
                                        + "18 C and sunny\n"
                                        + "</tool_response>"));

                IntSequence.Builder reply = IntSequence.newBuilder();
                reply.add(SpecialTokens.require(loaded.tokenizer(), "<tool_call>"));
                reply.addAll(
                        loaded.tokenizer()
                                .encode(
                                        "get_weather\n<arg_key>city</arg_key>\n"
                                                + "<arg_value>Paris</arg_value>"));
                reply.add(SpecialTokens.require(loaded.tokenizer(), "</tool_call>"));
                reply.add(SpecialTokens.require(loaded.tokenizer(), "<|role_end|>"));
                Message parsed =
                        ReplyParser.parse(
                                loaded.template().orElseThrow().parser(loaded.tokenizer()),
                                reply.build());
                Content.ToolCall call =
                        parsed.content().stream()
                                .filter(Content.ToolCall.class::isInstance)
                                .map(Content.ToolCall.class::cast)
                                .findFirst()
                                .orElseThrow();
                assertEquals(WEATHER.name(), call.name());
                assertEquals(Map.of("city", "Paris"), call.arguments());
            }
        }
    }

    @Test
    @Tag("integration")
    void automaticAndForcedToolCallsRunOnTheRealModelAndReturnStructuredArguments()
            throws Exception {
        try (Arena weights = Arenas.newCrossThread();
                ChatEngine engine =
                        new ChatEngine(
                                Models.load(requiredFile(MODEL_PROPERTY), weights),
                                "bailing-tools-real",
                                PromptCache.Options.DEFAULTS)) {
            for (ChatEngine.ForcedTool choice :
                    List.of(
                            ChatEngine.ForcedTool.NONE,
                            new ChatEngine.ForcedTool.Named(WEATHER.name()))) {
                ChatEngine.Request request =
                        new ChatEngine.Request(
                                List.of(
                                        Message.user(
                                                "Use get_weather for Paris. The city argument must"
                                                        + " be exactly Paris.")),
                                List.of(WEATHER),
                                false,
                                64,
                                null,
                                null,
                                Duration.ZERO,
                                GREEDY,
                                null,
                                choice,
                                List.of(),
                                null);
                ChatEngine.Completion completion =
                        engine.complete(request, ChatEngine.ReplySink.NONE);
                assertFalse(completion.cancelled());
                Content.ToolCall call =
                        completion.reply().content().stream()
                                .filter(Content.ToolCall.class::isInstance)
                                .map(Content.ToolCall.class::cast)
                                .findFirst()
                                .orElseThrow();
                assertEquals(WEATHER.name(), call.name(), choice.toString());
                assertTrue(
                        String.valueOf(call.arguments().get("city")).contains("Paris"),
                        choice.toString());
            }
        }
    }

    private static Path requiredFile(String property) {
        String configured = System.getProperty(property, "");
        assumeTrue(!configured.isBlank(), "set -D" + property + "=/path/to/file");
        Path path = Path.of(configured);
        assumeTrue(Files.isRegularFile(path), path + " is not a file");
        return path;
    }

    private static float[] readFloats(Path path) throws Exception {
        byte[] bytes = Files.readAllBytes(path);
        assertEquals(0, bytes.length % Float.BYTES, "reference logits must be raw FP32");
        float[] values = new float[bytes.length / Float.BYTES];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        return values;
    }

    private static float[] logits(BailingMoe3 model, BailingMoe3.State state) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state), "logits"), "logits");
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    private static float maxAbsDifference(float[] left, float[] right) {
        float max = 0f;
        for (int i = 0; i < left.length; i++) max = Math.max(max, Math.abs(left[i] - right[i]));
        return max;
    }

    private static double rmse(float[] left, float[] right) {
        double squareSum = 0;
        for (int i = 0; i < left.length; i++) {
            double difference = left[i] - right[i];
            squareSum += difference * difference;
        }
        return Math.sqrt(squareSum / left.length);
    }

    private static double cosineSimilarity(float[] left, float[] right) {
        double dot = 0, leftSquareSum = 0, rightSquareSum = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftSquareSum += (double) left[i] * left[i];
            rightSquareSum += (double) right[i] * right[i];
        }
        return dot / Math.sqrt(leftSquareSum * rightSquareSum);
    }
}
