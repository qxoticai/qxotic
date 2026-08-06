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
        return options(gguf, promptCache, readOnly, false, true);
    }

    /** As {@link #options(Path)} with {@code --no-grammar}: every grammar request is refused. */
    static LLMOptions optionsNoGrammar(Path gguf) {
        return optionsNoGrammar(gguf, true);
    }

    /** {@code --no-grammar} in a chosen mode; outside {@code --server} it must not be accepted. */
    static LLMOptions optionsNoGrammar(Path gguf, boolean server) {
        return options(gguf, null, false, true, server);
    }

    private static LLMOptions options(
            Path gguf, Path promptCache, boolean readOnly, boolean noGrammar, boolean server) {
        return new LLMOptions(
                gguf,
                null, // companions
                server ? null : "hi", // a prompt is required outside server mode
                null,
                false,
                server,
                "127.0.0.1",
                0,
                1f,
                0.95f,
                40,
                0.05f,
                42L,
                2048,
                true,
                false,
                true,
                false,
                false,
                false,
                noGrammar,
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

    /** The assistant message content of a chat-completions response. */
    static String messageContent(String body) {
        Object parsed = com.qxotic.jinfer.chat.JsonCodec.parse(body);
        Object choice =
                com.qxotic.jinfer.chat.Values.asArray(
                                com.qxotic.jinfer.chat.Values.asObject(parsed, "response")
                                        .get("choices"),
                                "choices")
                        .get(0);
        return com.qxotic.jinfer.chat.Values.stringValue(
                com.qxotic.jinfer.chat.Values.asObject(
                                com.qxotic.jinfer.chat.Values.asObject(choice, "choice")
                                        .get("message"),
                                "message")
                        .get("content"),
                "");
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
