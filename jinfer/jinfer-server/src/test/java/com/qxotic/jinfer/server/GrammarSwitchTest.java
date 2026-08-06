package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.net.http.HttpClient;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Grammar has ONE switch. It used to have two that disagreed: {@code --no-grammar} refused the
 * request with a 400, while {@code -Djinfer.grammar=false} silently returned 200 with unconstrained
 * output - a client that asked for JSON got prose and no indication that its constraint had been
 * dropped. The property is gone; this pins the behaviour that replaced it. The other half of the
 * switch - that --no-grammar is refused outside --server, where it did nothing at all - is a
 * command-line rule, and lives in jinfer-cli's OptionsTest.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GrammarSwitchTest {

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
        server = Server.start(model, ServerTestSupport.config(gguf));
        base = ServerTestSupport.baseUrl(server);
        client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    @AfterAll
    void stop() {
        if (server != null) server.close();
        if (arena != null) arena.close();
    }

    /** A grammar request on a server that allows grammars is honoured, not merely accepted. */
    @Test
    void jsonObjectIsConstrainedWhenGrammarIsOn() throws Exception {
        HttpResponse<String> response =
                ServerTestSupport.post(
                        client,
                        base + "/v1/chat/completions",
                        "{\"model\":\""
                                + modelId
                                + "\",\"messages\":[{\"role\":\"user\",\"content\":\"a fruit, as"
                                + " json\"}],\"max_tokens\":24,\"response_format\":{\"type\":"
                                + " \"json_object\"}}");
        assertEquals(200, response.statusCode(), response.body());
        String content = ServerTestSupport.messageContent(response.body());
        // any RFC 8259 document, not just an object: the grammar is full JSON, so a bare string
        // like "apples" is a correct constrained answer. What it cannot be is prose
        assertDoesNotThrow(
                () -> com.qxotic.jinfer.chat.JsonCodec.parse(content),
                "grammar-constrained output must be JSON, got: " + content);
    }

    /**
     * With grammars refused, the request FAILS. The whole point of the switch: a dropped constraint
     * that answers 200 is indistinguishable from a model that ignored the schema.
     */
    @Test
    void aRefusedGrammarIs400NotUnconstrained200() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
            LoadedModel<?> model = Models.load(gguf, 2048, weights);
            try (Server.Running refusing =
                    Server.start(model, ServerTestSupport.configNoGrammar(gguf))) {
                String url = ServerTestSupport.baseUrl(refusing) + "/v1/chat/completions";
                String id = gguf.getFileName().toString();
                for (String constraint :
                        new String[] {
                            "\"response_format\":{\"type\": \"json_object\"}",
                            "\"grammar\":\"root ::= \\\"a\\\"\""
                        }) {
                    HttpResponse<String> response =
                            ServerTestSupport.post(
                                    client,
                                    url,
                                    "{\"model\":\""
                                            + id
                                            + "\",\"messages\":[{\"role\":\"user\",\"content\":\"hi,"
                                            + " as json\"}],\"max_tokens\":8,"
                                            + constraint
                                            + "}");
                    assertEquals(400, response.statusCode(), response.body());
                    assertTrue(response.body().contains("--no-grammar"), response.body());
                }
            }
        }
    }
}
