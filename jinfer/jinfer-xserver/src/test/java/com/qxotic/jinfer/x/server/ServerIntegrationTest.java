package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.ChatEngine;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class ServerIntegrationTest {

    private static final String MODEL =
            "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";

    @Test
    void openAiTransportRunsAgainstARealMemoryViewModel() throws Exception {
        Path path = TestModels.require(MODEL);
        try (ChatEngine engine =
                        new ChatEngine(
                                path,
                                Map.of(),
                                PromptCache.Options.DEFAULTS.withContextCapacity(256));
                Server.Running server =
                        Server.start(
                                engine,
                                ServerConfig.local(0))) {
            String base = "http://127.0.0.1:" + server.address().getPort();
            HttpClient client = HttpClient.newHttpClient();

            assertEquals(200, get(client, base + "/health").statusCode());
            String props = get(client, base + "/props").body();
            assertTrue(props.contains("\"speculation\""));
            assertTrue(props.contains("\"n_ctx\":256"), props);
            assertEquals(
                    200,
                    post(client, base + "/tokenize", "{\"content\":\"hello\"}")
                            .statusCode());

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
            assertTrue(metrics.contains("jinfer_requests_total 5"), metrics);
            assertTrue(metrics.contains("jinfer_speculation_accepted_tokens_total 0"), metrics);
        }
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
