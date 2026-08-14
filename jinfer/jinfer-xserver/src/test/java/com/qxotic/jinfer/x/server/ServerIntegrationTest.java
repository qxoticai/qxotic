package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.llm.Sampling;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class ServerIntegrationTest {

    private static final String MODEL = "hf.co/ggml-org/stories15M_MOE:Q8_0";

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
                                new ServerConfig(
                                        ServerConfig.local(0).bind(),
                                        new ServerConfig.Defaults(
                                                new Sampling(0f, 1f, 0, 0f, 42L),
                                                4,
                                                false,
                                                false),
                                        ServerConfig.Limits.DEFAULTS,
                                        ServerConfig.Access.LOCAL))) {
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

            String metrics = get(client, base + "/metrics").body();
            assertTrue(metrics.contains("jinfer_requests_total 3"), metrics);
            assertTrue(metrics.contains("jinfer_speculation_accepted_tokens_total 0"), metrics);
        }
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
