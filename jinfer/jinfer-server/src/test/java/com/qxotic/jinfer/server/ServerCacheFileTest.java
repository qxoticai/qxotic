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
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * {@code --server --cache <file>}: the catalog a server accumulates survives its restarts. The
 * June-era server had a persistent cache file; the July block redesign kept the flag for instruct
 * mode only and the server's tree became memory-only, dying with the process - this pins the ported
 * behavior end to end: serve, shut down (append), boot a second server on the file, and the echoed
 * prompt must resume from the artifact instead of re-prefilling.
 */
class ServerCacheFileTest {

    @Test
    void theCatalogSurvivesARestart() throws Exception {
        Path gguf = ModelFixture.LLAMA32_1B_Q8.require();
        Path catalog = Files.createTempDirectory("jinfer-cache").resolve("catalog.jkvf");
        String prompt = "The capital of France is Paris. Answer in one word: what is it?";
        HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
        try (Arena arena = Arena.ofShared()) {
            LoadedModel<?> model = Models.load(gguf, 2048, arena);

            try (Server.Running first =
                    Server.start(model, ServerTestSupport.config(gguf, catalog, false))) {
                assertEquals(200, chat(client, first, gguf, prompt).statusCode());
            } // close appends the tree to the catalog
            assertTrue(Files.exists(catalog), "shutdown must write the catalog");
            assertTrue(Files.size(catalog) > 0, "an empty artifact cached nothing");

            try (Server.Running second =
                    Server.start(model, ServerTestSupport.config(gguf, catalog, false))) {
                HttpResponse<String> echoed = chat(client, second, gguf, prompt);
                assertEquals(200, echoed.statusCode(), echoed.body());
                assertTrue(
                        cachedTokens(echoed.body()) > 0,
                        "the echoed prompt must resume from the mounted artifact: "
                                + echoed.body());
            }
        }
    }

    private static HttpResponse<String> chat(
            HttpClient client, Server.Running server, Path gguf, String content) throws Exception {
        return ServerTestSupport.post(
                client,
                ServerTestSupport.baseUrl(server) + "/v1/chat/completions",
                ServerTestSupport.chatBody(gguf.getFileName().toString(), content, 4));
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
}
