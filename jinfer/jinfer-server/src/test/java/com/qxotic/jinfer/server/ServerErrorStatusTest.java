package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The status a refused request gets is decided before it is queued, on every generation endpoint: a
 * model this server does not serve is a 404 {@code not_found_error} (what OpenAI's SDKs expect and
 * retry on), any other client fault a 400. No model file: a test language model behind a real
 * transport, and the refusals never reach the engine.
 */
class ServerErrorStatusTest {

    private static ChatEngine engine;
    private static Server.Running server;
    private static final HttpClient CLIENT = HttpClient.newHttpClient();

    @BeforeAll
    static void start() throws Exception {
        var loaded =
                new LoadedModel<>(
                        new TestLanguageModel(),
                        TestLanguageModel.TOKENIZER,
                        "",
                        Set.of(),
                        new ContentKey("server-error-status-test"),
                        Optional.empty(),
                        LoadedModel.SamplingDefaults.NONE);
        engine = new ChatEngine(loaded, "served-model", PromptCache.Options.DEFAULTS);
        server = Server.start(engine, ServerConfig.local(0));
    }

    @AfterAll
    static void stop() {
        server.close();
        engine.close();
    }

    @ParameterizedTest
    @ValueSource(strings = {"/v1/chat/completions", "/v1/completions", "/v1/responses"})
    void anUnknownModelIsA404OnEveryGenerationEndpoint(String endpoint) throws Exception {
        HttpResponse<String> refused = post(endpoint, body(endpoint, "\"model\":\"nope\","));
        assertEquals(404, refused.statusCode(), refused.body());
        assertTrue(refused.body().contains("not_found_error"), refused.body());
        assertTrue(refused.body().contains("served-model"), refused.body());
    }

    @ParameterizedTest
    @ValueSource(strings = {"/v1/chat/completions", "/v1/completions", "/v1/responses"})
    void anyOtherClientFaultStaysA400(String endpoint) throws Exception {
        HttpResponse<String> refused = post(endpoint, body(endpoint, "\"stream\":\"yes\","));
        assertEquals(400, refused.statusCode(), refused.body());
        assertTrue(refused.body().contains("invalid_request_error"), refused.body());
    }

    @ParameterizedTest
    @ValueSource(strings = {"/v1/chat/completions", "/v1/completions", "/v1/responses"})
    void theServedModelSpelledAnyCaseIsNotUnknown(String endpoint) throws Exception {
        // the same request with the served name: whatever else happens, it is not a 404
        HttpResponse<String> answer =
                post(endpoint, body(endpoint, "\"model\":\"SERVED-MODEL\",\"stream\":\"yes\","));
        assertEquals(400, answer.statusCode(), answer.body());
        assertTrue(answer.body().contains("stream"), answer.body());
    }

    /** A minimal valid body per endpoint, with {@code extra} spliced in front. */
    private static String body(String endpoint, String extra) {
        return switch (endpoint) {
            case "/v1/chat/completions" ->
                    "{" + extra + "\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}";
            case "/v1/completions" -> "{" + extra + "\"prompt\":\"hi\"}";
            case "/v1/responses" -> "{" + extra + "\"input\":\"hi\"}";
            default -> throw new IllegalArgumentException(endpoint);
        };
    }

    private static HttpResponse<String> post(String endpoint, String json) throws Exception {
        URI uri = URI.create("http://127.0.0.1:" + server.address().getPort() + endpoint);
        return CLIENT.send(
                HttpRequest.newBuilder(uri)
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(json))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
