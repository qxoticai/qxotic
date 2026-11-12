package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/**
 * The decision transport's resource limits and routing edges over a stub model: the body bound, the
 * admission gate with its 503 and Retry-After while a probe still answers, the unknown-model and
 * wrong-path 404s, malformed files, unusual question names, every state shape, an unreadable
 * modality refused before it is decoded, and the smallest legal questions. No weights.
 */
final class DecisionServerLimitsTest {

    /** Answers immediately with a uniform row, or blocks until released. */
    static class Stub implements DecisionModel {
        final CountDownLatch entered = new CountDownLatch(1);
        final CountDownLatch release = new CountDownLatch(1);
        final List<Question> questions = new CopyOnWriteArrayList<>();
        volatile boolean slow;
        volatile List<Content> input;

        @Override
        public ContextState newState(int contextCapacity) {
            return new ContextState(16, 16, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                @Override
                protected void clearHistory() {}
            };
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            this.input = input;
            questions.add(question);
            if (slow) {
                entered.countDown();
                try {
                    release.await(60, TimeUnit.SECONDS);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            return uniform(question.options().size());
        }

        private static float[] uniform(int n) {
            float[] p = new float[n];
            Arrays.fill(p, 1f / n);
            return p;
        }
    }

    private final Stub stub = new Stub();
    private final HttpClient client = HttpClient.newHttpClient();

    private static final String ONE =
            "\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"?\"}}";

    private DecisionServer.Running start(ServerConfig.Limits limits) throws Exception {
        return DecisionServer.start(stub, 0, "d1", ServerConfig.local(0).withLimits(limits));
    }

    private static String base(DecisionServer.Running server) {
        return "http://127.0.0.1:" + server.address().getPort();
    }

    private HttpResponse<String> send(HttpRequest request) throws Exception {
        return client.send(request, HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> post(String base, String json) throws Exception {
        return send(
                HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                        .POST(HttpRequest.BodyPublishers.ofString(json))
                        .header("Content-Type", "application/json")
                        .timeout(Duration.ofSeconds(120))
                        .build());
    }

    /** A body over the configured bound is a 413 naming the bound, and the model never runs. */
    @Test
    void aBodyOverTheLimitIsA413() throws Exception {
        ServerConfig.Limits tight =
                new ServerConfig.Limits(
                        4,
                        1024,
                        true,
                        Duration.ofSeconds(30),
                        Duration.ofSeconds(300),
                        Duration.ofSeconds(30));
        try (var server = start(tight)) {
            String body = "{\"state\": \"" + "x".repeat(2048) + "\", " + ONE + "}";
            HttpResponse<String> reply = post(base(server), body);
            assertEquals(413, reply.statusCode(), reply.body());
            assertTrue(reply.body().contains("1024-byte limit"), reply.body());
            assertEquals(null, stub.input, "nothing reached the model");
        }
    }

    /**
     * One permit: the request that holds it runs, the next is a 503 with Retry-After, and the
     * probes still answer while both are true - a saturated server stays observable.
     */
    @Test
    void aSaturatedGateIsA503WithRetryAfterAndTheProbesStillAnswer() throws Exception {
        try (var server =
                start(
                        new ServerConfig.Limits(
                                1,
                                32L << 20,
                                true,
                                Duration.ofSeconds(30),
                                Duration.ofSeconds(300),
                                Duration.ofSeconds(30)))) {
            String base = base(server);
            stub.slow = true;
            ExecutorService pool = Executors.newFixedThreadPool(2);
            try {
                Future<HttpResponse<String>> busy =
                        pool.submit(() -> post(base, "{\"state\": \"x\", " + ONE + "}"));
                assertTrue(stub.entered.await(30, TimeUnit.SECONDS), "the first call is running");

                HttpResponse<String> refused = post(base, "{\"state\": \"y\", " + ONE + "}");
                assertEquals(503, refused.statusCode(), refused.body());
                assertEquals("2", refused.headers().firstValue("Retry-After").orElse(""));
                assertTrue(refused.body().contains("Server busy"), refused.body());

                // the probes are not gated: a load balancer must still see the busy server
                assertEquals(200, send(get(base + "/health")).statusCode());
                assertTrue(send(get(base + "/health")).body().contains("\"busy\":true"));
                assertEquals(200, send(get(base + "/props")).statusCode());
                assertEquals(200, send(get(base + "/metrics")).statusCode());

                stub.release.countDown();
                assertEquals(200, busy.get(120, TimeUnit.SECONDS).statusCode());
            } finally {
                stub.release.countDown();
                pool.shutdownNow();
            }
        }
    }

    private static HttpRequest get(String url) {
        return HttpRequest.newBuilder(URI.create(url)).GET().build();
    }

    /** The routing edges: an unknown model, a prefixed probe path, a wrong method. */
    @Test
    void theRoutingEdgesSayWhy() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            String base = base(server);

            HttpResponse<String> unknown = send(get(base + "/v1/models/nope.gguf"));
            assertEquals(404, unknown.statusCode(), unknown.body());
            assertTrue(unknown.body().contains("Unknown model: nope.gguf"), unknown.body());

            // a context matches by PREFIX: /healthXYZ is a wrong path, not the probe
            assertEquals(404, send(get(base + "/healthXYZ")).statusCode());
            assertEquals(404, send(get(base + "/propsXYZ")).statusCode());
            // /v1/models itself is an exact match; a prefixed one is not
            assertEquals(200, send(get(base + "/v1/models/d1")).statusCode());
            assertEquals(404, send(get(base + "/v1/modelsXYZ")).statusCode());

            // PUT is not a method this route takes
            HttpResponse<String> put =
                    send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                    .PUT(HttpRequest.BodyPublishers.ofString("{}"))
                                    .header("Content-Type", "application/json")
                                    .build());
            assertEquals(405, put.statusCode(), put.body());
            assertEquals("POST, OPTIONS", put.headers().firstValue("Allow").orElse(""));

            // the catch-all names what this server serves
            HttpResponse<String> wrong = send(get(base + "/v1/chat/completions"));
            assertEquals(404, wrong.statusCode(), wrong.body());
            assertTrue(wrong.body().contains("/v1/systemone"), wrong.body());
        }
    }

    /** The smallest legal questions answer; a whitespace state is a text. */
    @Test
    void theSmallestQuestionsAndAWhitespaceStateAnswer() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            String base = base(server);
            HttpResponse<String> two =
                    post(
                            base,
                            "{\"state\": \"   \", \"questions\": {"
                                    + "\"s\": {\"type\": \"score\", \"instructions\": \"?\","
                                    + " \"criteria\": [\"lo\", \"hi\"]},"
                                    + "\"c\": {\"type\": \"choice\", \"instructions\": \"?\","
                                    + " \"criteria\": {\"a\": null, \"b\": null}}}}");
            assertEquals(200, two.statusCode(), two.body());
            assertEquals(List.of(new Content.Text("   ")), stub.input);
            assertTrue(two.body().contains("\"probabilities\":{\"0\":0.5,\"1\":0.5}"), two.body());
            Map<String, Object> body = parseMap(two.body());
            Map<String, Object> answers = castMap(body.get("answers"));
            assertEquals(
                    0.5,
                    ((Number) ((Map<?, ?>) answers.get("c")).get("confidence")).doubleValue(),
                    1e-9);
            Map<?, ?> levels = (Map<?, ?>) ((Map<?, ?>) answers.get("s")).get("probabilities");
            assertEquals(2, levels.size());
        }
    }

    /** A JSON string can carry a NUL; the transport passes it through as the text it is. */
    @Test
    void aNulInTheStateIsTheTextItIs() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            HttpResponse<String> reply =
                    post(base(server), "{\"state\": \"a\\u0000b\", " + ONE + "}");
            assertEquals(200, reply.statusCode(), reply.body());
            assertEquals(List.of(new Content.Text("a\u0000b")), stub.input);
        }
    }

    /**
     * A modality the model cannot read is refused before it is decoded: an invalid video sent to a
     * text-only model is a "takes no video", not an ffmpeg complaint, and no decoder runs.
     */
    @Test
    void anUnreadModalityIsRefusedBeforeDecoding() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            // the stub reads no media at all
            for (String kind : List.of("video/mp4", "audio/wav", "image/png")) {
                HttpResponse<String> refused =
                        post(
                                base(server),
                                "{\"state\": \"x\", \"files\": [\"data:"
                                        + kind
                                        + ";base64,AAAA\"], "
                                        + ONE
                                        + "}");
                assertEquals(400, refused.statusCode(), refused.body());
                String modality = kind.substring(0, kind.indexOf('/'));
                assertTrue(refused.body().contains("takes no " + modality), refused.body());
            }
        }
    }

    /** A key on the routes that need one: 401 with WWW-Authenticate, and /health stays open. */
    @Test
    void theApiKeyGuardsTheWorkRoute() throws Exception {
        ServerConfig.Access keyed =
                new ServerConfig.Access("secret-token", java.util.Set.of("https://app.example"));
        try (var server =
                DecisionServer.start(stub, 0, "d1", ServerConfig.local(0).withAccess(keyed))) {
            String base = base(server);
            String body = "{\"state\": \"x\", " + ONE + "}";

            // no key: 401 on the work route, with the challenge a client acts on
            HttpResponse<String> anonymous =
                    send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                    .POST(HttpRequest.BodyPublishers.ofString(body))
                                    .header("Content-Type", "application/json")
                                    .build());
            assertEquals(401, anonymous.statusCode(), anonymous.body());
            assertEquals("Bearer", anonymous.headers().firstValue("WWW-Authenticate").orElse(""));

            // a wrong key is the same refusal
            assertEquals(401, send(withKey(base, body, "wrong")).statusCode());

            // the right key runs the model
            assertEquals(200, send(withKey(base, body, "secret-token")).statusCode());
            assertTrue(stub.input != null);

            // /health needs no key: a load balancer must reach it
            assertEquals(200, send(get(base + "/health")).statusCode());
            // the other probes do when a key is set
            assertEquals(401, send(get(base + "/props")).statusCode());
            assertEquals(
                    200,
                    send(HttpRequest.newBuilder(URI.create(base + "/props"))
                                    .header("Authorization", "Bearer secret-token")
                                    .GET()
                                    .build())
                            .statusCode());

            // a CORS origin not listed is a 403 that says so
            HttpResponse<String> offOrigin =
                    send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                    .POST(HttpRequest.BodyPublishers.ofString(body))
                                    .header("Content-Type", "application/json")
                                    .header("Authorization", "Bearer secret-token")
                                    .header("Origin", "https://evil.example")
                                    .build());
            assertEquals(403, offOrigin.statusCode(), offOrigin.body());
            assertTrue(offOrigin.body().contains("Origin is not allowed"), offOrigin.body());

            // the listed origin passes
            HttpResponse<String> onOrigin =
                    send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                                    .POST(HttpRequest.BodyPublishers.ofString(body))
                                    .header("Content-Type", "application/json")
                                    .header("Authorization", "Bearer secret-token")
                                    .header("Origin", "https://app.example")
                                    .build());
            assertEquals(200, onOrigin.statusCode(), onOrigin.body());
        }
    }

    private static HttpRequest withKey(String base, String body, String key) {
        return HttpRequest.newBuilder(URI.create(base + "/v1/systemone"))
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + key)
                .build();
    }

    /** A question's name is the client's key: unusual spellings travel to the answer unchanged. */
    @Test
    void questionNamesAreTheCallersKeys() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            HttpResponse<String> reply =
                    post(
                            base(server),
                            "{\"state\": \"x\", \"questions\": {"
                                    + "\"\": {\"type\": \"noul\", \"instructions\": \"?\"},"
                                    + "\"a.b/c\": {\"type\": \"noul\", \"instructions\": \"?\"},"
                                    + "\"\u00e9t\u00e9 \ud83c\udf89\": {\"type\": \"noul\","
                                    + " \"instructions\": \"?\"}}}");
            assertEquals(200, reply.statusCode(), reply.body());
            Map<String, Object> answers = castMap(parseMap(reply.body()).get("answers"));
            assertEquals(
                    List.of("", "a.b/c", "\u00e9t\u00e9 \ud83c\udf89"),
                    List.copyOf(answers.keySet()));
        }
    }

    /** A state is text, any JSON value, or null with files: each form is read, none is refused. */
    @Test
    void everyStateShapeIsRead() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            for (String state : List.of("42", "true", "false", "[1,2]", "1.5")) {
                HttpResponse<String> reply =
                        post(base(server), "{\"state\": " + state + ", " + ONE + "}");
                assertEquals(200, reply.statusCode(), state + ": " + reply.body());
                // a non-string state is its JSON text; a scalar is the scalar
                assertEquals(1, stub.input.size(), state);
                assertTrue(
                        stub.input.getFirst() instanceof Content.Text,
                        state + ": " + stub.input.getFirst());
            }
        }
    }

    /** A malformed file list or payload is a 400 that names the field, never a hang. */
    @Test
    void malformedFilesAreRefused() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            String base = base(server);
            assertRefused(
                    post(base, "{\"state\": \"x\", \"files\": \"not an array\", " + ONE + "}"),
                    "must be an array");
            assertRefused(
                    post(
                            base,
                            "{\"state\": \"x\", \"files\": [\"https://example.org/a.png\"], "
                                    + ONE
                                    + "}"),
                    "does not fetch remote URLs");
            assertRefused(
                    post(
                            base,
                            "{\"state\": \"x\", \"files\": [\"data:image/png,AAAA\"], "
                                    + ONE
                                    + "}"),
                    "must be base64-encoded");
            assertRefused(
                    post(
                            base,
                            "{\"state\": \"x\", \"files\": [\"data:text/plain;base64,aGk=\"], "
                                    + ONE
                                    + "}"),
                    "not data:text/plain");
        }
    }

    private static void assertRefused(HttpResponse<String> reply, String says) {
        assertEquals(400, reply.statusCode(), reply.body());
        assertTrue(reply.body().contains(says), "expected '" + says + "' in " + reply.body());
    }

    /** Every question kind's option labels reach the model as the client wrote them. */
    @Test
    void labelsReachTheModelVerbatim() throws Exception {
        try (var server = start(ServerConfig.Limits.DEFAULTS)) {
            HttpResponse<String> reply =
                    post(
                            base(server),
                            "{\"state\": \"x\", \"questions\": {"
                                    + "\"c\": {\"type\": \"choice\", \"instructions\": \"Pick\","
                                    + " \"criteria\": {\"a \\\"quoted\\\" label\": \"with a"
                                    + " \\\"description\\\"\", \"b\": null}}}}");
            assertEquals(200, reply.statusCode(), reply.body());
            // the client's own spelling, quotes and all: the server adds no escaping of its own
            assertEquals(
                    List.of(
                            new Option("a \"quoted\" label", "with a \"description\""),
                            new Option("b")),
                    stub.questions.getLast().options());
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseMap(String json) {
        return (Map<String, Object>) JsonCodec.parse(json);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object value) {
        return (Map<String, Object>) value;
    }
}
