package com.qxotic.jinfer.models.laguna;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.jinja.JinjaRenderer;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jota.DataType;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class LagunaIntegrationTest {
    private static final int BOS = 2;
    private static final String MODEL_PROPERTY = "jinfer.laguna.model";
    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    Map.of(
                            "type",
                            "function",
                            "function",
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
                                            Map.of("city", Map.of("type", "string")),
                                            "required",
                                            List.of("city")))));

    @Test
    void checkpointMetadataMatchesLagunaXs21() throws Exception {
        Checkpoint checkpoint = checkpoint();
        Laguna.Configuration config =
                Laguna.loadConfiguration(
                        checkpoint.gguf(), checkpoint.tokenizer().vocabulary().size(), "laguna");

        assertEquals(40, config.numberOfLayers());
        assertEquals(2_048, config.embeddingLength());
        assertEquals(100_352, config.vocabularySize());
        assertEquals(256, config.expertCount());
        assertEquals(8, config.expertUsedCount());
        assertEquals(1, config.denseLeadingLayers());
        assertEquals(512, config.slidingWindow());
        for (int layer = 0; layer < config.numberOfLayers(); layer++) {
            assertEquals(layer % 4 == 0 ? 48 : 64, config.headCount()[layer]);
            assertEquals(layer % 4 != 0, config.isSwa()[layer]);
        }
    }

    @Test
    void nativePromptMatchesCheckpointJinjaForThinkingAndTools() throws Exception {
        Checkpoint checkpoint = checkpoint();
        LagunaChatTemplate nativeTemplate =
                new LagunaChatTemplate(
                        checkpoint.tokenizer(),
                        checkpoint.gguf().getValue(int.class, "tokenizer.ggml.bos_token_id"));
        assertEquals(-1, nativeTemplate.defaultReasoningBudget(128));
        int[] plainThinking =
                encode(nativeTemplate, new Conversation(List.of(Message.user("Who are you?"))));
        assertArrayEquals(
                new int[] {
                    2, 97, 6453, 55620, 515, 330, 6408, 81, 12123, 1009, 8286, 10167, 18263, 2637,
                    565, 30810, 638, 83, 1239, 515, 1973, 367, 445, 6408, 367, 1667, 1388, 5882,
                    2930, 22746, 4187, 6453, 99, 268, 97, 1437, 99, 23910, 515, 453, 23638, 1437,
                    99, 268, 23, 18
                },
                plainThinking,
                "llama.cpp tokenization of the embedded template's plain thinking prompt");
        int[] plainDirect =
                encode(
                        nativeTemplate,
                        new Conversation(
                                List.of(Message.user("Who are you?")), List.of(), false, ""));
        assertEquals(19, plainDirect[plainDirect.length - 1], "</think> scaffold token");
        assertArrayEquals(
                java.util.Arrays.copyOf(plainThinking, plainThinking.length - 1),
                java.util.Arrays.copyOf(plainDirect, plainDirect.length - 1));
        List<Message> messages =
                List.of(
                        Message.system("Answer briefly."),
                        Message.user("What is the weather in Paris?"),
                        new Message(
                                Role.ASSISTANT,
                                List.of(
                                        new Content.Reasoning(
                                                List.of(new Content.Text("I should check.")),
                                                com.qxotic.toknroll.IntSequence.empty()),
                                        new Content.Text("I will check."),
                                        new Content.ToolCall(
                                                "call-1", "get_weather", Map.of("city", "Paris")))),
                        new Message(
                                Role.TOOL,
                                List.of(new Content.ToolResult("call-1", "18 C and sunny"))));
        for (boolean thinking : new boolean[] {false, true}) {
            Conversation conversation = new Conversation(messages, List.of(WEATHER), thinking, "");
            int[] expected = render(checkpoint, messages, thinking);
            int[] actual = encode(nativeTemplate, conversation);
            assertEquals(
                    checkpoint.tokenizer().decode(expected), checkpoint.tokenizer().decode(actual));
            assertArrayEquals(expected, actual);
        }
    }

    @Test
    void nativeToolSyntaxIsTextWhenTheRequestOffersNoTools() throws Exception {
        Checkpoint checkpoint = checkpoint();
        Tokenizer tokenizer = checkpoint.tokenizer();
        LagunaChatTemplate template =
                new LagunaChatTemplate(
                        tokenizer,
                        checkpoint.gguf().getValue(int.class, "tokenizer.ggml.bos_token_id"));
        String payload = "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>";
        IntSequence wire =
                IntSequence.of(SpecialTokens.require(tokenizer, "<tool_call>"))
                        .concat(tokenizer.encode(payload))
                        .concat(IntSequence.of(SpecialTokens.require(tokenizer, "</tool_call>")));

        Content.ToolCall call =
                assertInstanceOf(
                        Content.ToolCall.class,
                        ReplyParser.parse(template.parser(tokenizer), wire).content().getFirst());
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());

        ReplyParser disabled = template.parser(tokenizer);
        disabled.disableToolCalls();
        Message text = ReplyParser.parse(disabled, wire);
        assertEquals(payload, text.text());
        assertEquals(wire, ((Content.Text) text.content().getFirst()).verbatim());
        assertTrue(text.content().stream().noneMatch(Content.ToolCall.class::isInstance));
    }

    @Test
    void batchedAndIncrementalInferenceAgreeAndResetReplays() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> loaded = Models.load(requiredFile(MODEL_PROPERTY), arena);
            Laguna model = assertInstanceOf(Laguna.class, loaded.model());
            assertInstanceOf(LagunaChatTemplate.class, loaded.template().orElseThrow());
            assertEquals(20, loaded.samplingDefaults().topK());
            int[] tokens = promptTokens(model.tokenizer(), "Hello world");
            assertArrayEquals(new int[] {2, 6352, 3078}, tokens);

            float[] batched;
            try (Laguna.State state = model.newState(32, tokens.length)) {
                assertEquals(DataType.FP16, state.keyCache[0].dataType());
                assertEquals(DataType.FP16, state.valueCache[1].dataType());
                assertEquals(DataType.FP32, state.batchK.dataType());
                model.ingest(state, Batch.prefill(tokens));
                batched = logits(model, state);
            }

            try (Laguna.State state = model.newState(32, 1)) {
                for (int token : tokens) model.ingest(state, Batch.step(token));
                float[] incremental = logits(model, state);
                double drift = rmse(batched, incremental);
                System.out.printf(
                        "Laguna batch/step parity: rmse=%.6f max=%.6f cosine=%.9f%n",
                        drift,
                        maxAbsDifference(batched, incremental),
                        cosineSimilarity(batched, incremental));
                assertEquals(argmax(batched), argmax(incremental));
                // Prefill and decode use different attention reductions. Near-tied routes in the
                // 256-expert top-k router amplify that ordinary floating-point difference.
                assertTrue(drift < .45, "batched/incremental logits RMSE " + drift);
                assertTrue(maxAbsDifference(batched, incremental) < 2f);
                assertTrue(cosineSimilarity(batched, incremental) > .985);

                state.reset();
                for (int token : tokens) model.ingest(state, Batch.step(token));
                assertArrayEquals(incremental, logits(model, state), .001f);
            }
        }
    }

    @Test
    void promptCacheRestoresFp16FullAndSlidingWindowLayers() throws Exception {
        Path modelPath = requiredFile(MODEL_PROPERTY);
        try (Arena arena = Arena.ofShared()) {
            Laguna model = Laguna.loadModel(modelPath, arena);
            int[] tokens = promptTokens(model.tokenizer(), "Hello world");
            List<Batch> prompt =
                    List.of(
                            Batch.prefill(java.util.Arrays.copyOf(tokens, tokens.length - 1)),
                            Batch.step(tokens[tokens.length - 1]));
            PromptCache.Options options =
                    PromptCache.Options.DEFAULTS
                            .withRetainedSessions(0)
                            .withContextCapacity(32)
                            .withBlockBudget(2L << 20);
            try (PromptCache<Laguna.State> cache =
                    PromptCache.of(model, Models.modelSeed(modelPath), options)) {
                CacheResult fresh = cachedLogits(cache, model, prompt);
                CacheResult restored = cachedLogits(cache, model, prompt);

                assertEquals(PromptCache.Tier.FRESH, fresh.tier());
                assertEquals(PromptCache.Tier.BLOCKS, restored.tier());
                assertEquals(tokens.length - 1, restored.restored());
                assertArrayEquals(fresh.logits(), restored.logits(), 1e-4f);
            }
        }
    }

    private static int[] render(Checkpoint checkpoint, List<Message> messages, boolean thinking) {
        List<Map<String, Object>> mapped = new ArrayList<>();
        mapped.add(Map.of("role", "system", "content", messages.get(0).text()));
        mapped.add(Map.of("role", "user", "content", messages.get(1).text()));
        Map<String, Object> assistant = new LinkedHashMap<>();
        assistant.put("role", "assistant");
        assistant.put("content", "I will check.");
        assistant.put("reasoning", "I should check.");
        assistant.put(
                "tool_calls",
                List.of(
                        Map.of(
                                "type",
                                "function",
                                "function",
                                Map.of(
                                        "name",
                                        "get_weather",
                                        "arguments",
                                        Map.of("city", "Paris")))));
        mapped.add(assistant);
        mapped.add(Map.of("role", "tool", "content", "18 C and sunny"));
        String rendered =
                JinjaRenderer.template(checkpoint.gguf().getString("tokenizer.chat_template"))
                        .render(
                                Map.of(
                                        "messages",
                                        mapped,
                                        "tools",
                                        List.of(WEATHER.definition()),
                                        "add_generation_prompt",
                                        true,
                                        "enable_thinking",
                                        thinking));
        String bos = "〈|EOS|〉";
        assertTrue(rendered.startsWith(bos));
        int[] body =
                SpecialTokens.encode(checkpoint.tokenizer(), rendered.substring(bos.length()))
                        .toArray();
        int[] result = new int[body.length + 1];
        result[0] = checkpoint.gguf().getValue(int.class, "tokenizer.ggml.bos_token_id");
        System.arraycopy(body, 0, result, 1, body.length);
        return result;
    }

    private static int[] encode(ChatTemplate template, Conversation conversation) {
        List<Batch> batches = new ArrayList<>();
        template.encode(conversation, 7, batches::add);
        return Batch.tokenIds(batches);
    }

    private static int[] promptTokens(Tokenizer tokenizer, String prompt) {
        int[] text = tokenizer.encodeToArray(prompt);
        int[] tokens = new int[text.length + 1];
        tokens[0] = BOS;
        System.arraycopy(text, 0, tokens, 1, text.length);
        return tokens;
    }

    private static Checkpoint checkpoint() throws Exception {
        Path path = requiredFile(MODEL_PROPERTY);
        try (FileChannel file = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            return new Checkpoint(gguf, tokenizer);
        }
    }

    private static Path requiredFile(String property) {
        String configured = System.getProperty(property, "");
        assumeTrue(!configured.isBlank(), "set -D" + property + "=/path/to/file");
        Path path = Path.of(configured);
        assumeTrue(Files.isRegularFile(path), path + " is not a file");
        return path;
    }

    private static float[] logits(Laguna model, Laguna.State state) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state), "logits"), "logits");
    }

    private static CacheResult cachedLogits(
            PromptCache<Laguna.State> cache, Laguna model, List<Batch> prompt) {
        return cache.serve(
                prompt,
                (state, serving) ->
                        new CacheResult(serving.tier(), serving.restored(), logits(model, state)));
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

    private record Checkpoint(GGUF gguf, Tokenizer tokenizer) {}

    private record CacheResult(PromptCache.Tier tier, int restored, float[] logits) {}
}
