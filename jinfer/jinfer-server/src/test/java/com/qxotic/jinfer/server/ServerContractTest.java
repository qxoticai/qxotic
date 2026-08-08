package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The HTTP contract, over the smallest fixture, in a DEFAULT build. The battery that covers this
 * module properly is {@code @Tag("integration")} and excluded unless asked for, so a normal build
 * exercised almost none of the transport layer - these are the parts worth having either way:
 * status codes, error shape, and the endpoints a client probes before it trusts a server.
 *
 * <p>One model load for the whole class; nothing here generates more than a handful of tokens.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ServerContractTest {

    private static Arena arena;
    private static Server.Running server;
    private static HttpClient client;
    private static String base;
    private static String modelId;

    @BeforeAll
    void start() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        modelId = gguf.getFileName().toString();
        arena = Arena.ofShared();
        LoadedModel<?> model = Models.load(gguf, arena);
        server = Server.start(model, ServerTestSupport.config(gguf));
        base = ServerTestSupport.baseUrl(server);
        client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    @AfterAll
    void stop() {
        if (server != null) server.close();
        if (arena != null) arena.close();
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return ServerTestSupport.post(client, base + path, body);
    }

    private HttpResponse<String> get(String path) throws Exception {
        return ServerTestSupport.get(client, base + path);
    }

    private String chatBody(String content) {
        return ServerTestSupport.chatBody(modelId, content, 4);
    }

    @Test
    void theProbeEndpointsAnswerBeforeAnyGeneration() throws Exception {
        assertEquals(200, get("/health").statusCode());
        assertEquals(200, get("/props").statusCode());
        assertEquals(200, get("/v1/models").statusCode());
        assertEquals(200, get("/metrics").statusCode());
        assertEquals(404, get("/nope").statusCode());

        Map<String, Object> models = json(get("/v1/models").body());
        assertTrue(models.get("data") instanceof List<?> d && !((List<?>) d).isEmpty());
    }

    @Test
    void aChatCompletionRoundTrips() throws Exception {
        HttpResponse<String> response = post("/v1/chat/completions", chatBody("Say hi"));
        assertEquals(200, response.statusCode(), response.body());
        Map<String, Object> body = json(response.body());
        assertEquals("chat.completion", body.get("object"));
        assertTrue(body.get("usage") instanceof Map<?, ?>, "usage is part of the contract");
    }

    @Test
    void malformedRequestsAreClientErrors() throws Exception {
        assertEquals(400, post("/v1/chat/completions", "{not json").statusCode());
        assertEquals(
                400,
                post("/v1/chat/completions", "{\"model\":\"" + modelId + "\",\"messages\":[]}")
                        .statusCode());
        assertEquals(
                400,
                post("/v1/chat/completions", chatBody("hi").replace("\"max_tokens\":4", "\"n\":2"))
                        .statusCode());
    }

    /**
     * Client-error bodies explain the request, never the server: no exception class, no stack, no
     * package name.
     *
     * <p>SCOPE, measured rather than assumed: this passes against the old handler too, which mapped
     * every RuntimeException to 400 and echoed its message. Every request below fails VALIDATION,
     * and those messages were always clean. The case that mapping got wrong - an internal fault
     * answered as a client error quoting a null field - cannot be induced from outside without a
     * fault-injection seam, so that branch stays unexercised. Good sign, untested branch; do not
     * read this as covering it.
     */
    @Test
    void clientErrorBodiesNeverLeakInternals() throws Exception {
        List<HttpResponse<String>> errors =
                List.of(
                        post("/v1/chat/completions", "{not json"),
                        post("/v1/chat/completions", "{\"model\":\"x\",\"messages\":[]}"),
                        post("/v1/chat/completions", "{\"messages\":[]}"),
                        get("/nope"));
        for (HttpResponse<String> error : errors) {
            assertTrue(error.statusCode() >= 400, "expected an error, got " + error.statusCode());
            String body = error.body();
            for (String leak :
                    List.of("Exception", "Cannot invoke", "com.qxotic", "java.lang", "\tat ")) {
                assertTrue(!body.contains(leak), "error body leaked '" + leak + "': " + body);
            }
        }
    }

    /**
     * /props reports the block tree's real health, not a hardcoded flag: a served request commits
     * blocks and counts a lookup, and an operator polling /props must see that. (The June-era
     * warm_tokens/dense_hits keys died with the machinery they described; these are the block-cache
     * vocabulary.)
     */
    @Test
    void propsReportTheCacheTreesHealth() throws Exception {
        post("/v1/chat/completions", chatBody("warm the tree"));
        Map<String, Object> cache =
                (Map<String, Object>) json(get("/props").body()).get("prompt_cache");
        assertEquals(Boolean.TRUE, cache.get("enabled"), String.valueOf(cache));
        assertTrue(
                ((Number) cache.get("blocks")).longValue() > 0, "a served pass commits: " + cache);
        assertTrue(((Number) cache.get("budget_bytes")).longValue() > 0, String.valueOf(cache));
        long lookups =
                ((Number) cache.get("hits")).longValue()
                        + ((Number) cache.get("misses")).longValue();
        assertTrue(lookups > 0, "a served pass consults the tree: " + cache);
    }

    @Test
    void tokenizeAndDetokenizeRoundTrip() throws Exception {
        Map<String, Object> tokens =
                json(post("/tokenize", "{\"content\":\"Hello, world!\"}").body());
        assertTrue(tokens.get("tokens") instanceof List<?> t && !((List<?>) t).isEmpty());
        String ids = JsonCodec.stringify(tokens.get("tokens"));
        Map<String, Object> text = json(post("/detokenize", "{\"tokens\":" + ids + "}").body());
        assertEquals("Hello, world!", text.get("content"));
    }

    /**
     * These two helpers took whatever arrived and answered 200. A typo'd field, a null, a float id
     * and a string id each produced a confident wrong answer - {@code {"content": null}} tokenized
     * the literal four characters "null", {@code [1.5]} silently truncated to token 1 - and the one
     * input that did fail answered 400 quoting the JVM ("class java.lang.String cannot be cast to
     * class java.lang.Number..."). Every case below is a REQUEST-shaped mistake and must be named
     * as one.
     */
    @Test
    void tokenizeHelpersRefuseMalformedInputByName() throws Exception {
        record Case(String path, String body, String mustName) {}
        List<Case> cases =
                List.of(
                        new Case("/tokenize", "{}", "content"),
                        new Case("/tokenize", "{\"text\":\"hello\"}", "content"),
                        new Case("/tokenize", "{\"content\":null}", "content"),
                        new Case("/tokenize", "{\"content\":42}", "content"),
                        new Case("/detokenize", "{}", "tokens"),
                        new Case("/detokenize", "{\"tokens\":\"nope\"}", "tokens"),
                        new Case("/detokenize", "{\"tokens\":[\"a\"]}", "tokens[0]"),
                        new Case("/detokenize", "{\"tokens\":[1.5]}", "tokens[0]"),
                        new Case("/detokenize", "{\"tokens\":[-1]}", "tokens[0]"),
                        new Case("/detokenize", "{\"tokens\":[999999999]}", "vocabulary"));
        for (Case c : cases) {
            HttpResponse<String> response = post(c.path(), c.body());
            assertEquals(
                    400,
                    response.statusCode(),
                    c.path() + " " + c.body() + " -> " + response.body());
            assertTrue(
                    response.body().contains(c.mustName()),
                    c.body() + " must name " + c.mustName() + ", said: " + response.body());
            assertFalse(
                    response.body().contains("java.lang"),
                    "a client error must not quote the JVM: " + response.body());
        }
        // present-but-empty is a legitimate question with a legitimate answer
        assertEquals(200, post("/tokenize", "{\"content\":\"\"}").statusCode());
        assertEquals(200, post("/detokenize", "{\"tokens\":[]}").statusCode());
    }

    /**
     * A streaming request that FORCES a tool call installs no live sinks - the turn is seeded into
     * the call block and the calls are emitted once at the end. But whether forcing actually
     * happens is the model's business: a model whose template has no call seed ignores tool_choice
     * and generates ordinary prose. The transport used to decide that separately from the
     * generation ({@code ToolUse.forced(request) != null} vs {@code nativeForcedOk() && ...}) and
     * the two disagreed on exactly those models: the stream installed no sinks, the pass ran
     * unforced, and the client received a role chunk, an empty terminal chunk, and NONE of the text
     * the model produced. Llama 3.2 is seedless, which is why this fixture is the right one.
     */
    @Test
    void aForcedToolStreamOnASeedlessModelStillDeliversItsReply() throws Exception {
        String body =
                """
                {"model":"%s","stream":true,"max_tokens":48,"temperature":0,
                 "tool_choice":"required",
                 "tools":[{"type":"function","function":{"name":"get_weather",
                   "description":"Get weather for a city",
                   "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                     "required":["city"]}}}],
                 "messages":[{"role":"user","content":"What is the weather in Paris?"}]}
                """
                        .formatted(modelId);
        HttpResponse<String> response = post("/v1/chat/completions", body);
        assertEquals(200, response.statusCode(), response.body());

        StringBuilder streamed = new StringBuilder();
        int chunks = 0;
        for (String line : response.body().split("\n")) {
            if (!line.startsWith("data:")) continue;
            String payload = line.substring(5).trim();
            if ("[DONE]".equals(payload)) continue;
            chunks++;
            Map<String, Object> chunk = json(payload);
            List<?> choices = (List<?>) chunk.get("choices");
            if (choices.isEmpty()) continue;
            Object delta = ((Map<?, ?>) choices.get(0)).get("delta");
            Object content = ((Map<?, ?>) delta).get("content");
            Object calls = ((Map<?, ?>) delta).get("tool_calls");
            if (content != null) streamed.append(content);
            if (calls != null) streamed.append("<tool_calls>");
        }
        assertTrue(chunks >= 2, "expected a real stream, got " + chunks + " chunks");
        assertFalse(
                streamed.isEmpty(),
                "the pass ran unforced and produced tokens, so the stream must carry them - it"
                        + " delivered nothing at all");
    }

    /**
     * Llama has no native tool codec, so a tools request falls back to the Jinja whole-render over
     * the GGUF's own template - and Meta's template opens with {@code {{- bos_token }}}. That
     * variable was bound by looking for {@code <bos>} / {@code <|startoftext|>} only, so for Llama
     * 3 (whose BOS is {@code <|begin_of_text|>}) it resolved to null and Jinja rendered the literal
     * four characters "None" at the very front of every prompt. Off-distribution, the model never
     * emitted a turn end: it produced a call, opened a fresh assistant header, and repeated until
     * the budget ran out - finish_reason "length", no call parsed, trailing garbage.
     *
     * <p>This is the E2E shape of that bug, and of the argument-spelling one behind it: the reply
     * must be a CALL, not a truncated essay, and it must carry the argument the user gave.
     */
    @Test
    void aToolsRequestThroughTheWholeRenderCallsAndStops() throws Exception {
        String body =
                """
                {"model":"%s","max_tokens":200,"temperature":0,
                 "tools":[{"type":"function","function":{"name":"get_weather",
                   "description":"Get the current weather for a city",
                   "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                     "required":["city"]}}}],
                 "messages":[{"role":"user","content":"What is the weather in Paris?"}]}
                """
                        .formatted(modelId);
        Map<String, Object> reply = json(post("/v1/chat/completions", body).body());
        Map<?, ?> choice = (Map<?, ?>) ((List<?>) reply.get("choices")).get(0);
        assertEquals(
                "tool_calls",
                choice.get("finish_reason"),
                "the turn must END on a call, not run to the budget: " + reply);

        List<?> calls = (List<?>) ((Map<?, ?>) choice.get("message")).get("tool_calls");
        assertTrue(calls != null && !calls.isEmpty(), "a call must be parsed: " + reply);
        Map<?, ?> function = (Map<?, ?>) ((Map<?, ?>) calls.get(0)).get("function");
        assertEquals("get_weather", function.get("name"));
        String arguments = String.valueOf(function.get("arguments"));
        assertTrue(
                arguments.contains("Paris"),
                "the argument the user actually gave must survive: " + arguments);
    }

    /** One model served, so naming it is optional - naming the WRONG one is not. */
    @Test
    void theModelFieldIsOptionalOverTheWire() throws Exception {
        HttpResponse<String> response =
                post(
                        "/v1/chat/completions",
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":4}");
        assertEquals(200, response.statusCode(), response.body());
        assertEquals(
                modelId, json(response.body()).get("model"), "the reply names the served model");
    }

    @Test
    void metricsCountTheRequestsItServed() throws Exception {
        String before = get("/metrics").body();
        assertTrue(before.contains("jinfer_requests_total"), before);
        post("/v1/chat/completions", chatBody("one more"));
        String after = get("/metrics").body();
        assertTrue(
                ServerTestSupport.counter(after, "jinfer_requests_total")
                        > ServerTestSupport.counter(before, "jinfer_requests_total"),
                "a served request must move the counter");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> json(String body) {
        return (Map<String, Object>) JsonCodec.parse(body);
    }

    /**
     * The DEFAULT server has no seed: every request draws its own randomness. This is the common
     * case and it was broken - Sampling.seed() is a Long, and a conditional that mixed it with a
     * long unboxed it, so an unseeded server answered 500 to everything.
     */
    @Test
    void anUnseededServerAnswersRequests() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
            LoadedModel<?> model = Models.load(gguf, weights);
            try (Server.Running unseeded =
                    Server.start(model, ServerTestSupport.configUnseeded(gguf))) {
                HttpResponse<String> response =
                        ServerTestSupport.post(
                                client,
                                ServerTestSupport.baseUrl(unseeded) + "/v1/chat/completions",
                                ServerTestSupport.chatBody(gguf.getFileName().toString(), "hi", 4));
                assertEquals(200, response.statusCode(), response.body());
            }
        }
    }
}
