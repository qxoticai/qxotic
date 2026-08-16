package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import jdk.jfr.Recording;
import jdk.jfr.consumer.RecordedEvent;
import jdk.jfr.consumer.RecordingFile;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class ServerIntegrationTest {

    private static final String MODEL = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";

    @Test
    void openAiTransportRunsAgainstARealMemoryViewModel() throws Exception {
        Path path = TestModels.require(MODEL);
        try (ChatEngine engine =
                        new ChatEngine(
                                path,
                                Map.of(),
                                PromptCache.Options.DEFAULTS.withContextCapacity(256));
                Server.Running server = Server.start(engine, ServerConfig.local(0))) {
            String base = "http://127.0.0.1:" + server.address().getPort();
            HttpClient client = HttpClient.newHttpClient();

            assertEquals(200, get(client, base + "/health").statusCode());
            String props = get(client, base + "/props").body();
            assertTrue(props.contains("\"speculation\""));
            assertTrue(props.contains("\"n_ctx\":256"), props);
            assertTrue(props.contains("\"retained_sessions\":"), props);
            assertTrue(props.contains("\"retained_session_limit\":4"), props);
            assertTrue(props.contains("\"state_allocations\":"), props);
            assertTrue(props.contains("\"block_hits\":"), props);
            assertFalse(props.contains("\"hot_sessions\":"), props);
            assertFalse(props.contains("\"hot_hits\":"), props);
            assertEquals(
                    200, post(client, base + "/tokenize", "{\"content\":\"hello\"}").statusCode());
            assertEquals(400, post(client, base + "/tokenize", "{}").statusCode());
            assertEquals(
                    400, post(client, base + "/detokenize", "{\"tokens\":[1.5]}").statusCode());

            HttpResponse<String> completion =
                    post(
                            client,
                            base + "/v1/completions",
                            "{\"prompt\":\"Once upon a time\",\"max_tokens\":2,"
                                    + "\"temperature\":0}");
            assertEquals(200, completion.statusCode(), completion.body());
            assertTrue(completion.body().contains("\"text_completion\""), completion.body());
            assertTrue(completion.body().contains("\"prompt_tokens\""), completion.body());

            HttpResponse<String> constrained =
                    post(
                            client,
                            base + "/v1/completions",
                            "{\"prompt\":\"Once\",\"max_tokens\":4,\"temperature\":0,"
                                    + "\"grammar\":\"root ::= \\\"OK\\\"\"}");
            assertEquals(200, constrained.statusCode(), constrained.body());
            assertTrue(constrained.body().contains("\"text\":\"OK\""), constrained.body());

            HttpResponse<String> stream =
                    post(
                            client,
                            base + "/v1/completions",
                            "{\"prompt\":\"Once upon a time\",\"max_tokens\":2,"
                                    + "\"temperature\":0,\"stream\":true}");
            assertEquals(200, stream.statusCode(), stream.body());
            assertTrue(stream.body().contains("data: [DONE]"), stream.body());

            HttpResponse<String> toolStream =
                    post(
                            client,
                            base + "/v1/responses",
                            "{\"input\":\"Use the weather tool for Zurich\",\"stream\":true,"
                                    + "\"temperature\":0,\"max_output_tokens\":32,"
                                    + "\"tools\":[{\"type\":\"function\",\"name\":\"weather\","
                                    + "\"parameters\":{\"type\":\"object\",\"properties\":{"
                                    + "\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}],"
                                    + "\"tool_choice\":{\"type\":\"function\","
                                    + "\"name\":\"weather\"}}");
            assertEquals(200, toolStream.statusCode(), toolStream.body());
            assertEquals(
                    eventItemIds(toolStream.body(), "response.output_item.added"),
                    eventItemIds(toolStream.body(), "response.output_item.done"),
                    toolStream.body());
            assertTrue(
                    toolStream.body().contains("event: response.function_call_arguments.done"),
                    toolStream.body());
            assertEquals(
                    eventResponseCreatedAt(toolStream.body(), "response.created"),
                    eventResponseCreatedAt(toolStream.body(), "response.completed"),
                    toolStream.body());

            HttpResponse<String> responseStream =
                    post(
                            client,
                            base + "/v1/responses",
                            "{\"input\":\"Say OK\",\"stream\":true,\"temperature\":0,"
                                    + "\"max_output_tokens\":4}");
            assertEquals(200, responseStream.statusCode(), responseStream.body());
            assertTrue(
                    responseStream.body().contains("event: response.content_part.added"),
                    responseStream.body());
            assertTrue(
                    responseStream.body().contains("event: response.content_part.done"),
                    responseStream.body());
            assertTrue(responseStream.body().contains("\"sequence_number\":0"));

            HttpResponse<String> failedResponseStream =
                    post(
                            client,
                            base + "/v1/responses",
                            JsonCodec.stringify(
                                    Map.of(
                                            "input",
                                            "x".repeat(10_000),
                                            "stream",
                                            true,
                                            "max_output_tokens",
                                            1)));
            assertEquals(200, failedResponseStream.statusCode(), failedResponseStream.body());
            assertTrue(
                    failedResponseStream.body().contains("event: error"),
                    failedResponseStream.body());
            assertTrue(
                    failedResponseStream.body().contains("\"type\":\"error\""),
                    failedResponseStream.body());

            String metrics = get(client, base + "/metrics").body();
            assertTrue(metrics.contains("jinfer_generations_completed_total 5"), metrics);
            assertTrue(metrics.contains("jinfer_generation_requests_invalid_total 1"), metrics);
            assertTrue(metrics.contains("jinfer_speculation_accepted_tokens_total 0"), metrics);
        }
    }

    @Test
    void grammarRefusalAndTransportRestartAreRealLifecycleBoundaries() throws Exception {
        Path path = TestModels.require(MODEL);
        try (ChatEngine engine = engine(path, PromptCache.Options.DEFAULTS)) {
            ServerConfig local = ServerConfig.local(0);
            ServerConfig noGrammar = local.withLimits(local.limits().withGrammar(false));
            try (Server.Running first = Server.start(engine, noGrammar)) {
                HttpResponse<String> refused =
                        post(
                                HttpClient.newHttpClient(),
                                base(first) + "/v1/completions",
                                "{\"prompt\":\"x\",\"max_tokens\":1,"
                                        + "\"grammar\":\"root ::= \\\"x\\\"\"}");
                assertEquals(400, refused.statusCode(), refused.body());
            }
            // Running owns only the transport: the same engine starts a fresh listener and works.
            try (Server.Running second = Server.start(engine, ServerConfig.local(0))) {
                assertEquals(
                        200,
                        post(
                                        HttpClient.newHttpClient(),
                                        base(second) + "/v1/completions",
                                        "{\"prompt\":\"Once\",\"max_tokens\":1,"
                                                + "\"temperature\":0}")
                                .statusCode());
            }
        }
    }

    @Test
    void writableCatalogRestoresAfterRestart() throws Exception {
        Path path = TestModels.require(MODEL);
        Path catalog = Files.createTempDirectory("jinfer-server-cache").resolve("prompts.jkvf");
        PromptCache.Options options =
                PromptCache.Options.DEFAULTS.withContextCapacity(256).withCatalog(catalog, false);
        String body =
                "{\"messages\":[{\"role\":\"user\",\"content\":"
                        + "\"The capital of France is Paris. Reply with one word.\"}],"
                        + "\"max_tokens\":2,\"temperature\":0}";
        HttpClient client = HttpClient.newHttpClient();

        try (ChatEngine first = engine(path, options)) {
            try (Server.Running server = Server.start(first, ServerConfig.local(0))) {
                assertEquals(
                        200,
                        post(client, base(server) + "/v1/chat/completions", body).statusCode());
            }
            first.savePrompts();
        }
        assertTrue(Files.size(catalog) > 0, "writable catalog stayed empty");

        try (ChatEngine second = engine(path, options.withCatalog(catalog, true));
                Server.Running server = Server.start(second, ServerConfig.local(0))) {
            HttpResponse<String> response =
                    post(client, base(server) + "/v1/chat/completions", body);
            assertEquals(200, response.statusCode(), response.body());
            assertTrue(cachedTokens(response.body()) > 0, response.body());
        }
    }

    @Test
    void metricsArePerServerAndTelemetryIsEmittedOnce() throws Exception {
        Path path = TestModels.require(MODEL);
        Path recordingFile = Files.createTempFile("jinfer-server", ".jfr");
        try (ChatEngine engine = engine(path, PromptCache.Options.DEFAULTS);
                Server.Running busy = Server.start(engine, ServerConfig.local(0));
                Server.Running idle = Server.start(engine, ServerConfig.local(0));
                Recording recording = new Recording()) {
            recording.enable("jinfer.Inference");
            recording.start();
            assertEquals(
                    200,
                    post(
                                    HttpClient.newHttpClient(),
                                    base(busy) + "/v1/completions",
                                    "{\"prompt\":\"Once\",\"max_tokens\":1," + "\"temperature\":0}")
                            .statusCode());
            recording.stop();
            recording.dump(recordingFile);

            assertEquals(
                    1,
                    counter(
                            get(base(busy) + "/metrics").body(),
                            "jinfer_generations_completed_total"));
            assertEquals(
                    0,
                    counter(
                            get(base(idle) + "/metrics").body(),
                            "jinfer_generations_completed_total"));
        }
        List<RecordedEvent> events = new ArrayList<>();
        try (RecordingFile recording = new RecordingFile(recordingFile)) {
            while (recording.hasMoreEvents()) {
                RecordedEvent event = recording.readEvent();
                if (event.getEventType().getName().equals("jinfer.Inference")) events.add(event);
            }
        }
        assertEquals(1, events.size());
        RecordedEvent event = events.getFirst();
        assertTrue(event.getLong("timeToFirstToken") > 0);
        assertTrue(event.getLong("timeToFirstToken") <= event.getDuration().toNanos());
    }

    @Test
    void seedlessJinjaFallbackAnswers() throws Exception {
        Path path = TestModels.require(MODEL);
        try (Arena weights = Arena.ofShared()) {
            LoadedModel<?> nativeModel = Models.load(path, weights);
            LoadedModel<?> jinjaModel =
                    new LoadedModel<>(
                            nativeModel.model(),
                            nativeModel.tokenizer(),
                            nativeModel.chatTemplateSource(),
                            nativeModel.stopTokens(),
                            nativeModel.seed(),
                            Optional.empty(),
                            nativeModel.samplingDefaults());
            try (ChatEngine engine =
                            new ChatEngine(
                                    jinjaModel,
                                    path.getFileName().toString(),
                                    PromptCache.Options.DEFAULTS.withContextCapacity(256));
                    Server.Running server = Server.start(engine, ServerConfig.local(0))) {
                HttpResponse<String> response =
                        post(
                                HttpClient.newHttpClient(),
                                base(server) + "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\","
                                        + "\"content\":\"Say hi\"}],\"max_tokens\":2}");
                assertEquals(200, response.statusCode(), response.body());
                assertFalse(response.body().isBlank());
            }
        }
    }

    private static ChatEngine engine(Path path, PromptCache.Options options) {
        return new ChatEngine(path, Map.of(), options.withContextCapacity(256));
    }

    private static String base(Server.Running server) {
        return "http://127.0.0.1:" + server.address().getPort();
    }

    @SuppressWarnings("unchecked")
    private static long cachedTokens(String body) {
        Map<String, Object> usage =
                (Map<String, Object>) ((Map<String, Object>) JsonCodec.parse(body)).get("usage");
        return usage.get("prompt_tokens_details") instanceof Map<?, ?> details
                        && details.get("cached_tokens") instanceof Number cached
                ? cached.longValue()
                : 0;
    }

    private static long counter(String exposition, String name) {
        for (String line : exposition.split("\\n")) {
            if (line.startsWith(name + " ")) {
                return (long) Double.parseDouble(line.substring(name.length() + 1));
            }
        }
        throw new AssertionError("missing " + name + " in:\n" + exposition);
    }

    private static List<String> eventItemIds(String body, String event) {
        List<String> ids = new ArrayList<>();
        for (String frame : body.split("\\n\\n")) {
            if (!frame.startsWith("event: " + event + "\n")) continue;
            int data = frame.indexOf("data: ");
            Map<String, Object> payload =
                    Values.asObject(JsonCodec.parse(frame.substring(data + 6)), "event");
            Map<String, Object> item = Values.asObject(payload.get("item"), "event.item");
            ids.add(Values.stringValue(item.get("id"), ""));
        }
        return ids;
    }

    private static long eventResponseCreatedAt(String body, String event) {
        for (String frame : body.split("\\n\\n")) {
            if (!frame.startsWith("event: " + event + "\n")) continue;
            int data = frame.indexOf("data: ");
            Map<String, Object> payload =
                    Values.asObject(JsonCodec.parse(frame.substring(data + 6)), "event");
            Map<String, Object> response =
                    Values.asObject(payload.get("response"), "event.response");
            return Values.longValue(response.get("created_at"), -1);
        }
        throw new AssertionError("Missing event " + event);
    }

    private static HttpResponse<String> get(HttpClient client, String uri) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(uri)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static HttpResponse<String> get(String uri) throws Exception {
        return get(HttpClient.newHttpClient(), uri);
    }

    private static HttpResponse<String> post(HttpClient client, String uri, String body)
            throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(uri))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
