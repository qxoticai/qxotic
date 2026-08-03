package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
        LoadedModel<?> model = Models.load(gguf, 2048, arena);
        server = Server.start(model, ServerTestSupport.options(gguf));
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

    @Test
    void tokenizeAndDetokenizeRoundTrip() throws Exception {
        Map<String, Object> tokens =
                json(post("/tokenize", "{\"content\":\"Hello, world!\"}").body());
        assertTrue(tokens.get("tokens") instanceof List<?> t && !((List<?>) t).isEmpty());
        String ids = JsonCodec.stringify(tokens.get("tokens"));
        Map<String, Object> text = json(post("/detokenize", "{\"tokens\":" + ids + "}").body());
        assertEquals("Hello, world!", text.get("content"));
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
}
