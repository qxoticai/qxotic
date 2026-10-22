package com.qxotic.jinfer.libjinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The library's Java face on the JVM, against the cached LFM2.5 models: what the Python QA pass
 * found and fixed, pinned where the C entry points are too thin to hold a bug.
 */
@Tag("integration")
class ModelTest {

    private static final String LM = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";
    private static final String EMB =
            "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf";
    private static final String SYSTEM = "You are a terse assistant. ".repeat(10);

    private static Path lmPath;
    private static Model lm;

    @BeforeAll
    static void load() throws IOException {
        lmPath = TestModels.require(LM);
        lm = Model.load(lmPath.toString(), Options.Values.DEFAULTS);
    }

    @AfterAll
    static void free() {
        if (lm != null) lm.release();
    }

    @Test
    void aMissingPathIsNotFoundByName() {
        NoSuchFileException e =
                assertThrows(
                        NoSuchFileException.class,
                        () -> Model.load("/nonexistent/model.gguf", Options.Values.DEFAULTS));
        assertTrue(e.getMessage().contains("no such file"), e.getMessage());
        assertEquals(Errors.ENOTFOUND, Errors.code(e));
    }

    @Test
    void aCancelledStreamKeepsTheWordsItProduced() throws Exception {
        Stream s = Stream.chat(lm, chat("Write a long story about the sea.", 300));
        assertEquals(Errors.OK, s.step());
        String first = s.text;
        s.cancel();
        try {
            while (s.step() == Errors.OK) {}
            fail("a cancelled stream ends with ECANCELLED");
        } catch (Errors.Failure e) {
            assertEquals(Errors.ECANCELLED, e.code);
        }
        Map<String, Object> message = message(s.result);
        assertEquals("cancelled", finishReason(s.result));
        assertTrue(((String) message.get("content")).startsWith(first), s.result);
        assertEquals(Errors.DONE, s.step(), "finished after the cancellation");
        s.close();
    }

    @Test
    void twoThreadsSteppingOneStreamBothFinish() throws Exception {
        // against the header's contract, but the second must see DONE, not block forever
        Stream s = Stream.chat(lm, chat("Count from one to twenty.", 40));
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Callable<Integer> drain =
                    () -> {
                        int n = 0;
                        try {
                            while (s.step() == Errors.OK) n++;
                        } catch (Errors.Failure ignored) {
                            // not expected; the count below says what happened
                        }
                        return n;
                    };
            Future<Integer> a = pool.submit(drain);
            Future<Integer> b = pool.submit(drain);
            int chunks = a.get(60, TimeUnit.SECONDS) + b.get(60, TimeUnit.SECONDS);
            assertTrue(chunks > 0);
            assertTrue(List.of("stop", "length").contains(finishReason(s.result)), s.result);
        } finally {
            pool.shutdownNow();
            s.close();
        }
    }

    @Test
    void aPromptCacheFileSurvivesTheModel() throws Exception {
        Path catalog = Files.createTempDirectory("jinfer-native").resolve("prompts.bin");
        Options.Values options = new Options.Values(Map.of(), catalog.toString(), 0);
        Model first = Model.load(lmPath.toString(), options);
        Stream.prepare(first, "{\"messages\":[" + system() + "]}");
        first.release(); // the catalog is "read and extended": the extension happens here
        assertTrue(Files.size(catalog) > 0, "the catalog stayed empty");

        Model second = Model.load(lmPath.toString(), options);
        Stream s =
                Stream.chat(
                        second,
                        "{\"messages\":["
                                + system()
                                + ",{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":2}");
        while (s.step() == Errors.OK) {}
        Map<String, Object> timings = part(s.result, "timings");
        assertTrue(((Number) timings.get("cache_n")).intValue() > 0, s.result);
        s.close();
        second.release();
    }

    @Test
    void anEmbedderTakesATextLongerThanTheDefaultBatch() throws IOException {
        Model emb = Model.load(TestModels.require(EMB).toString(), Options.Values.DEFAULTS);
        try {
            int dimension = emb.property(3);
            assertEquals(dimension, emb.embed(List.of("word ".repeat(2000)), 0).length);
            assertEquals(
                    3 * dimension, emb.embed(List.of("a", "word ".repeat(1500), "b"), 0).length);
            Errors.Failure e =
                    assertThrows(Errors.Failure.class, () -> emb.embed(List.of("a"), -1));
            assertEquals(Errors.EINVAL, e.code);
        } finally {
            emb.release();
        }
    }

    @Test
    void theContextCapacityIsTheCallersUpToTheModelsMaximum() throws IOException {
        int max = lm.property(5);
        assertEquals(
                Math.min(4096, max),
                lm.property(2),
                "the default is 4096, or the maximum when smaller");
        Model small = Model.load(lmPath.toString(), new Options.Values(Map.of(), null, 1024));
        try {
            assertEquals(1024, small.property(2));
            assertEquals(max, small.property(5));
        } finally {
            small.release();
        }
        Model whole = Model.load(lmPath.toString(), new Options.Values(Map.of(), null, -1));
        try {
            assertEquals(max, whole.property(2), "-1 is the maximum");
        } finally {
            whole.release();
        }
        RuntimeException above =
                assertThrows(
                        RuntimeException.class,
                        () ->
                                Model.load(
                                        lmPath.toString(),
                                        new Options.Values(Map.of(), null, max + 1)));
        assertEquals(Errors.EINVAL, Errors.code(above));
        assertTrue(above.getMessage().contains(Integer.toString(max)), above.getMessage());
        Errors.Failure negative =
                assertThrows(
                        Errors.Failure.class,
                        () ->
                                Model.load(
                                        lmPath.toString(), new Options.Values(Map.of(), null, -2)));
        assertEquals(Errors.EINVAL, negative.code);
        Model emb =
                Model.load(
                        TestModels.require(EMB).toString(),
                        new Options.Values(Map.of(), null, 600));
        try {
            assertEquals(600, emb.property(2), "a retrieval model's state is capped too");
            assertTrue(emb.property(5) >= 600);
        } finally {
            emb.release();
        }
    }

    private static String chat(String prompt, int maxTokens) {
        return "{\"messages\":[{\"role\":\"user\",\"content\":\""
                + prompt
                + "\"}],\"max_tokens\":"
                + maxTokens
                + ",\"temperature\":0}";
    }

    private static String system() {
        return "{\"role\":\"system\",\"content\":\"" + SYSTEM + "\"}";
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> part(String json, String key) {
        return (Map<String, Object>) ((Map<String, Object>) Json.parse(json)).get(key);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> choice(String json) {
        return (Map<String, Object>)
                ((List<Object>) ((Map<String, Object>) Json.parse(json)).get("choices")).get(0);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> message(String json) {
        return (Map<String, Object>) choice(json).get("message");
    }

    private static String finishReason(String json) {
        return (String) choice(json).get("finish_reason");
    }
}
