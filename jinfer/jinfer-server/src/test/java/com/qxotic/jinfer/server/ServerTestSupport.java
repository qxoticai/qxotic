package com.qxotic.jinfer.server;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;

/**
 * The server test package's single copy of the plumbing every HTTP test was duplicating: the
 * 20-positional-argument default {@link LLMOptions} (a transposed flag in any copy silently changes
 * what that test runs against), the chat-completions body, and the Prometheus counter parse.
 */
final class ServerTestSupport {

    private ServerTestSupport() {}

    /** The all-defaults server config the tests run against: think off, ephemeral port, seed 42. */
    static LLMOptions options(Path gguf) {
        return options(gguf, null, false);
    }

    /** As {@link #options(Path)} with a {@code --cache} / {@code --cache-ro} file. */
    static LLMOptions options(Path gguf, Path promptCache, boolean readOnly) {
        return new LLMOptions(
                gguf,
                null,
                null,
                false,
                true,
                "127.0.0.1",
                0,
                1f,
                0.95f,
                42L,
                2048,
                true,
                false,
                true,
                false,
                false,
                false,
                false,
                promptCache,
                readOnly);
    }

    static String baseUrl(Server.Running server) {
        return "http://127.0.0.1:" + server.address().getPort();
    }

    static HttpResponse<String> get(HttpClient client, String url) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(url)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    static HttpResponse<String> post(HttpClient client, String url, String body) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(url))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    /** A minimal non-streaming chat-completions body. */
    static String chatBody(String modelId, String content, int maxTokens) {
        return "{\"model\":\""
                + modelId
                + "\",\"messages\":[{\"role\":\"user\",\"content\":\""
                + content
                + "\"}],\"max_tokens\":"
                + maxTokens
                + "}";
    }

    /** The value of one counter in a Prometheus exposition; fails loudly when absent. */
    static long counter(String exposition, String name) {
        for (String line : exposition.split("\n")) {
            if (line.startsWith(name + " ")) {
                return (long) Double.parseDouble(line.substring(name.length() + 1).trim());
            }
        }
        throw new AssertionError("no " + name + " in:\n" + exposition);
    }
}
