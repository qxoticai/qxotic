package com.qxotic.jinfer.server;

import com.qxotic.jinfer.llm.Sampling;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;

/**
 * The server test package's single copy of the plumbing every HTTP test was duplicating: the
 * default {@link ServerConfig}, the chat-completions body, and the Prometheus counter parse. It
 * used to build a 20-positional-argument LLMOptions, in which a transposed flag in any copy
 * silently changed what that test ran against.
 */
final class ServerTestSupport {

    private ServerTestSupport() {}

    /** The all-defaults server config the tests run against: think off, ephemeral port, seed 42. */
    static ServerConfig config(Path gguf) {
        return config(gguf, null, false);
    }

    /** As {@link #config(Path)} with a {@code --cache} / {@code --cache-ro} file. */
    static ServerConfig config(Path gguf, Path promptCache, boolean readOnly) {
        return new ServerConfig(
                gguf.getFileName().toString(),
                new java.net.InetSocketAddress("127.0.0.1", 0),
                new ServerConfig.Defaults(
                        new Sampling(1f, 0.95f, 40, 0.05f, 42L), 2048, true, false),
                ServerConfig.Limits.DEFAULTS,
                com.qxotic.jinfer.cache.PromptCache.Options.DEFAULTS.withCatalog(
                        promptCache, readOnly));
    }

    /**
     * As {@link #config(Path)} with NO seed, which is the real default: fresh randomness per
     * request. The seeded fixture hid a NullPointerException on this path for every server started
     * without --seed, so it is worth its own factory.
     */
    static ServerConfig configUnseeded(Path gguf) {
        ServerConfig base = config(gguf);
        Sampling s = base.defaults().sampling();
        return new ServerConfig(
                base.modelName(),
                base.bind(),
                new ServerConfig.Defaults(
                        new Sampling(s.temperature(), s.topP(), s.topK(), s.minP(), null),
                        base.defaults().maxTokens(),
                        base.defaults().think(),
                        base.defaults().rawPrompt()),
                base.limits(),
                base.cache());
    }

    /** As {@link #config(Path)} with grammar requests refused. */
    static ServerConfig configNoGrammar(Path gguf) {
        ServerConfig base = config(gguf);
        return new ServerConfig(
                base.modelName(),
                base.bind(),
                base.defaults(),
                base.limits().withGrammar(false),
                base.cache());
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
