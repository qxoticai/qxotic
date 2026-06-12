package com.llama4j;

import com.sun.net.httpserver.HttpServer;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Stream;

/**
 * End-to-end test of the OpenAI-compatible server, run IN PROCESS on an ephemeral port (one
 * model load, no subprocess management). Guards the bug classes seen in the wild: requests
 * that hang instead of failing fast (validation must answer 400 on the handler thread even
 * while the generation worker is busy), streaming regressions (reasoning_content deltas,
 * usage chunks, [DONE] termination), stop-string truncation, chat-session resume, and
 * prompt-cache cold/warm byte-identity. Requires the model file; skips cleanly when absent.
 */
public final class ServerIntegrationTest {

    private static int failures = 0;
    private static HttpClient client;
    private static String base;

    public static void main(String[] args) throws Exception {
        Path model = Path.of(args.length > 0 ? args[0] : "../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("ServerIntegrationTest: model not found (" + model + "), skipping");
            return;
        }
        Llama llama = ModelLoader.loadModel(model, 2048);
        LFM25.Options options = new LFM25.Options(model, null, null, null, false, true, "127.0.0.1", 0,
                1f, 0.95f, 42L, 2048, true, false, true, false, false, false, false);
        HttpServer server = Server.run(llama, options);
        base = "http://127.0.0.1:" + server.getAddress().getPort();
        client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

        try {
            plumbing();
            validation();
            tokenizeRoundTrip();
            chatNonStreaming();
            chatStreaming();
            completionsAndStops();
            responsesEndpoint();
            promptCacheColdWarm();
            fast400WhileBusy();
            sessionResume();
        } catch (Throwable t) {
            // always exit: the server executor pool is non-daemon, an escaped exception
            // would otherwise leave the JVM (and the make invocation) hanging forever
            t.printStackTrace();
            failures++;
        }
        System.out.println("ServerIntegrationTest: failures=" + failures);
        System.exit(failures > 0 ? 1 : 0);
    }

    // --- request plumbing: health, models, method/path errors ---

    private static void plumbing() throws Exception {
        check(get("/health").statusCode() == 200, "health 200");
        Map<String, Object> models = json(get("/v1/models"));
        check("list".equals(models.get("object")) && models.get("data") instanceof List<?> l && !l.isEmpty(),
                "models list non-empty");
        check(post("/v1/models", "{}").statusCode() == 405, "POST /v1/models -> 405");
        check(get("/v1/chat/completions").statusCode() == 405, "GET chat -> 405");
        check(get("/no/such/path").statusCode() == 404, "unknown path -> 404");
    }

    // --- malformed requests must 400 instantly, never enter the queue ---

    private static void validation() throws Exception {
        expect400("/v1/chat/completions", "{bad json", "invalid JSON");
        expect400("/v1/chat/completions", "{\"messages\":[]}", "empty messages");
        expect400("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":2}", "n=2");
        expect400("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":2,\"stream\":true}", "stream + n=2 pre-SSE");
        expect400("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":-1}", "negative temperature");
        expect400("/v1/completions", "{\"prompt\":\"\"}", "empty prompt");
        expect400("/v1/responses", "{}", "responses without input");
    }

    private static void expect400(String path, String body, String what) throws Exception {
        HttpResponse<String> response = post(path, body);
        check(response.statusCode() == 400, what + " -> 400 (got " + response.statusCode() + ")");
        check(response.body().contains("error"), what + " carries an error payload");
    }

    private static void tokenizeRoundTrip() throws Exception {
        Map<String, Object> tokens = json(post("/tokenize", "{\"content\":\"Hello, world!\"}"));
        check(List.of(35808L, 20L, 1530L, 9L).equals(tokens.get("tokens")), "tokenize ground truth");
        Map<String, Object> text = json(post("/detokenize", "{\"tokens\":[35808,20,1530,9]}"));
        check("Hello, world!".equals(text.get("content")), "detokenize round trip");
    }

    // --- generation endpoints ---

    private static void chatNonStreaming() throws Exception {
        HttpResponse<String> response = post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Briefly, why is the sky blue?\"}],\"temperature\":0,\"max_tokens\":48}");
        check(response.statusCode() == 200, "chat 200");
        check(response.headers().firstValue("X-LFM2-Timing").isPresent(), "timing header present");
        Map<String, Object> chat = json(response);
        Map<String, Object> message = path(chat, "choices", 0, "message");
        String content = (String) message.get("content");
        String reasoning = (String) message.get("reasoning_content");
        check((content != null && !content.isEmpty()) || (reasoning != null && !reasoning.isEmpty()),
                "chat produced content or reasoning");
        Map<String, Object> usage = path(chat, "usage");
        long prompt = (long) usage.get("prompt_tokens"), completion = (long) usage.get("completion_tokens");
        check(prompt > 0 && completion > 0 && (long) usage.get("total_tokens") == prompt + completion,
                "usage arithmetic consistent");
        Object finish = path(chat, "choices", 0).get("finish_reason");
        check("stop".equals(finish) || "length".equals(finish), "finish_reason valid");
    }

    private static void chatStreaming() throws Exception {
        HttpRequest request = HttpRequest.newBuilder(URI.create(base + "/v1/chat/completions"))
                .POST(HttpRequest.BodyPublishers.ofString(
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"temperature\":0,\"max_tokens\":48,"
                        + "\"stream\":true,\"stream_options\":{\"include_usage\":true}}"))
                .timeout(Duration.ofSeconds(60)).build();
        HttpResponse<Stream<String>> response = client.send(request, HttpResponse.BodyHandlers.ofLines());
        check(response.statusCode() == 200, "stream 200");
        boolean[] sawReasoning = {false}, sawUsage = {false}, sawDone = {false};
        response.body().forEach(line -> {
            if (line.equals("data: [DONE]")) {
                sawDone[0] = true;
            } else if (line.startsWith("data: ")) {
                if (line.contains("\"reasoning_content\"")) sawReasoning[0] = true;
                if (line.contains("\"completion_tokens\"")) sawUsage[0] = true;
            }
        });
        check(sawReasoning[0], "stream carried reasoning_content deltas");
        check(sawUsage[0], "stream carried usage");
        check(sawDone[0], "stream terminated with [DONE]");
    }

    private static void completionsAndStops() throws Exception {
        Map<String, Object> completion = json(post("/v1/completions",
                "{\"prompt\":\"Count: one, two, three, four\",\"max_tokens\":24,\"temperature\":0,\"stop\":[\", six\"]}"));
        String text = (String) path(completion, "choices", 0).get("text");
        check(text != null && text.contains("five") && !text.contains("six"),
                "stop string truncated the completion (got: " + text + ")");
        check("stop".equals(path(completion, "choices", 0).get("finish_reason")), "stop finish_reason");
    }

    private static void responsesEndpoint() throws Exception {
        Map<String, Object> response = json(post("/v1/responses",
                "{\"input\":\"Reply with one word: ok\",\"max_output_tokens\":48,"
                + "\"chat_template_kwargs\":{\"enable_thinking\":false}}"));
        check("completed".equals(response.get("status")), "responses status completed");
        String text = (String) path(response, "output", 0, "content", 0).get("text");
        check(text != null && !text.isEmpty(), "responses output_text non-empty");
    }

    // --- prompt cache: identical long-prompt requests; warm run resumes and output is identical ---

    private static void promptCacheColdWarm() throws Exception {
        String body = "{\"prompt\":\"" + "The quick brown fox jumps over the lazy dog. ".repeat(80)
                + "\",\"max_tokens\":4,\"temperature\":0}";
        HttpResponse<String> cold = post("/v1/completions", body);
        HttpResponse<String> warm = post("/v1/completions", body);
        long cachedCold = cachedTokens(cold), cachedWarm = cachedTokens(warm);
        check(cachedCold == 0, "cold run uncached (got " + cachedCold + ")");
        check(cachedWarm >= 512, "warm run resumed from the cache (cached " + cachedWarm + ")");
        String coldText = (String) path(json(cold), "choices", 0).get("text");
        String warmText = (String) path(json(warm), "choices", 0).get("text");
        check(coldText.equals(warmText), "cold and warm outputs identical");
    }

    private static long cachedTokens(HttpResponse<String> response) {
        Object cached = path(json(response), "usage", "prompt_tokens_details").get("cached_tokens");
        return ((Number) cached).longValue();
    }

    // --- a garbage request must be answered while the worker is mid-generation ---

    private static void fast400WhileBusy() throws Exception {
        HttpRequest slow = HttpRequest.newBuilder(URI.create(base + "/v1/chat/completions"))
                .POST(HttpRequest.BodyPublishers.ofString(
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a long story.\"}],\"temperature\":0,\"max_tokens\":200}"))
                .timeout(Duration.ofSeconds(60)).build();
        CompletableFuture<HttpResponse<String>> running =
                client.sendAsync(slow, HttpResponse.BodyHandlers.ofString());
        Thread.sleep(400); // let it reach the generation worker
        long start = System.nanoTime();
        HttpResponse<String> garbage = post("/v1/chat/completions", "{not json at all");
        long millis = (System.nanoTime() - start) / 1_000_000;
        check(garbage.statusCode() == 400, "garbage 400 while busy");
        check(millis < 2_000, "garbage answered in " + millis + "ms while the worker is busy");
        check(running.get().statusCode() == 200, "the long generation still completed");
    }

    // --- chat-session resume: turn 2 extends turn 1 in place ---

    private static void sessionResume() throws Exception {
        String turn1 = "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: hello\"}],"
                + "\"temperature\":0,\"max_tokens\":400,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        Map<String, Object> first = json(post("/v1/chat/completions", turn1));
        check("stop".equals(path(first, "choices", 0).get("finish_reason")), "turn 1 stopped cleanly");
        String reply = (String) path(first, "choices", 0, "message").get("content");
        String turn2 = "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: hello\"},"
                + "{\"role\":\"assistant\",\"content\":" + Server.Json.stringify(reply) + "},"
                + "{\"role\":\"user\",\"content\":\"now say goodbye\"}],"
                + "\"temperature\":0,\"max_tokens\":400,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        HttpResponse<String> second = post("/v1/chat/completions", turn2);
        check(second.statusCode() == 200, "turn 2 200");
        check(cachedTokens(second) > 0, "turn 2 resumed the session (cached " + cachedTokens(second) + ")");
    }

    // --- helpers ---

    private static HttpResponse<String> get(String path) throws Exception {
        return client.send(HttpRequest.newBuilder(URI.create(base + path)).timeout(Duration.ofSeconds(30)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static HttpResponse<String> post(String path, String body) throws Exception {
        return client.send(HttpRequest.newBuilder(URI.create(base + path))
                        .POST(HttpRequest.BodyPublishers.ofString(body)).timeout(Duration.ofSeconds(60)).build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> json(HttpResponse<String> response) {
        return (Map<String, Object>) Server.Json.parse(response.body());
    }

    /** Walks maps by key and lists by index: path(m, "choices", 0, "message"). */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> path(Map<String, Object> root, Object... steps) {
        Object current = root;
        for (Object step : steps) {
            current = step instanceof Integer i ? ((List<Object>) current).get(i) : ((Map<String, Object>) current).get(step);
        }
        return (Map<String, Object>) current;
    }

    private static void check(boolean condition, String what) {
        if (!condition) {
            failures++;
            System.err.println("FAIL: " + what);
        } else {
            System.out.println("ok: " + what);
        }
    }
}
