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
    private static Path warmFile;

    public static void main(String[] args) throws Exception {
        toolCallParser(); // pure parser tests: no model required
        Path model = Path.of(args.length > 0 ? args[0] : "../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        if (!Files.exists(model)) {
            System.out.println("ServerIntegrationTest: model not found (" + model + "), skipping");
            System.exit(failures > 0 ? 1 : 0);
            return;
        }
        Llama llama = ModelLoader.loadModel(model, 2048);
        StringBuilder manual = new StringBuilder("Agent operating manual.");
        for (int i = 1; i <= 50; i++) {
            manual.append(" Directive ").append(i).append(": when handling case ").append(i)
                    .append(", consult registry entry ").append(i).append(" and apply policy ")
                    .append(i).append(" before responding;");
        }
        warmFile = Files.createTempFile("lfm25-warm", ".txt");
        warmFile.toFile().deleteOnExit();
        Files.writeString(warmFile, manual.toString());
        LFM25.Options options = new LFM25.Options(model, null, null, null, false, true, "127.0.0.1", 0,
                1f, 0.95f, 42L, 2048, true, false, true, false, false, false, false,
                List.of(warmFile.toString()));
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
            promptCacheStrictPrefix();
            promptCacheBranchPoint();
            warmPromptInstant();
            strideAndTailResume();
            toolChoiceForced();
            reasoningBudget();
            stopStringsIgnoreReasoning();
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
        String servedId = (String) path(models, "data", 0).get("id");
        Map<String, Object> card = json(get("/v1/models/" + servedId));
        check("model".equals(card.get("object")) && servedId.equals(card.get("id")), "GET /v1/models/{id} -> card");
        check(get("/v1/models/garbage").statusCode() == 404, "GET /v1/models/{unknown} -> 404");
        check(post("/v1/models", "{}").statusCode() == 405, "POST /v1/models -> 405");
        check(get("/v1/chat/completions").statusCode() == 405, "GET chat -> 405");
        check(get("/no/such/path").statusCode() == 404, "unknown path -> 404");
        check(get("/health/garbage").statusCode() == 404, "prefix-matched subpath -> 404");
    }

    // --- malformed requests must 400 instantly, never enter the queue ---

    private static void validation() throws Exception {
        expect400("/v1/chat/completions", "{bad json", "invalid JSON");
        expect400("/v1/chat/completions", "{\"messages\":[]}", "empty messages");
        expect400("/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}", "all-empty message content");
        expect400("/v1/chat/completions",
                "{\"model\":\"garbage\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}", "unknown model name");
        expect400("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_max_tokens\":-2}", "reasoning_max_tokens=-2");
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
        Map<String, Object> alias = json(post("/v1/tokenize", "{\"content\":\"Hello, world!\"}"));
        check(tokens.get("tokens").equals(alias.get("tokens")), "/v1/tokenize alias");
        check("Hello, world!".equals(json(post("/v1/detokenize", "{\"tokens\":[35808,20,1530,9]}")).get("content")),
                "/v1/detokenize alias");
        HttpResponse<String> metrics = get("/metrics");
        check(metrics.statusCode() == 200 && metrics.body().contains("lfm25_requests_total")
                && metrics.body().contains("lfm25_uptime_seconds"), "/metrics exposition");
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
        // earlier requests may have left a checkpoint near the stream start (divergence right
        // after BOS), so "cold" can legitimately resume a few tokens — but no more
        check(cachedCold <= 4, "cold run essentially uncached (got " + cachedCold + ")");
        long promptTokens = ((Number) path(json(warm), "usage").get("prompt_tokens")).longValue();
        check(cachedWarm == promptTokens, "warm run resumed token-exact at L-1 (cached "
                + cachedWarm + " of " + promptTokens + ")");
        String coldText = (String) path(json(cold), "choices", 0).get("text");
        String warmText = (String) path(json(warm), "choices", 0).get("text");
        check(coldText.equals(warmText), "cold and warm outputs identical");
    }

    private static long cachedTokens(HttpResponse<String> response) {
        Object cached = path(json(response), "usage", "prompt_tokens_details").get("cached_tokens");
        return ((Number) cached).longValue();
    }

    /** A strict PREFIX of already-cached content: the end-of-prompt checkpoint splits inside a
     *  cached node, and the end-of-generation commit dedups through existing nodes (pin-walk).
     *  The repeat request must then resume token-exact at its own L-1. */
    private static void promptCacheStrictPrefix() throws Exception {
        String body = "{\"prompt\":\"" + "The quick brown fox jumps over the lazy dog. ".repeat(40)
                + "\",\"max_tokens\":4,\"temperature\":0}";
        HttpResponse<String> first = post("/v1/completions", body); // prefix of the 80-rep stream above
        HttpResponse<String> second = post("/v1/completions", body);
        long promptTokens = ((Number) path(json(second), "usage").get("prompt_tokens")).longValue();
        check(cachedTokens(second) == promptTokens, "strict-prefix re-request resumed token-exact (cached "
                + cachedTokens(second) + " of " + promptTokens + ")");
        String firstText = (String) path(json(first), "choices", 0).get("text");
        String secondText = (String) path(json(second), "choices", 0).get("text");
        check(firstText.equals(secondText), "strict-prefix outputs identical");
    }

    /** Requests sharing only a SYSTEM PROMPT prefix. A populates the tree; the short stream
     *  lies entirely inside the dense tail, so B and C resume token-exact at the divergence
     *  on their FIRST visit (bx rows — no checkpoint round-trip needed anymore). */
    private static void promptCacheBranchPoint() throws Exception {
        String system = "You are the dedicated test oracle for the lfm25 integration battery, a most particular "
                + "and verbose persona that always answers with at most one short sentence and no preamble.";
        String template = "{\"messages\":[{\"role\":\"system\",\"content\":\"" + system + "\"},"
                + "{\"role\":\"user\",\"content\":\"%s\"}],\"temperature\":0,\"max_tokens\":24,"
                + "\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        long cachedA = cachedTokens(post("/v1/chat/completions", template.formatted("What color is the sea?")));
        long cachedB = cachedTokens(post("/v1/chat/completions", template.formatted("What color is grass?")));
        long cachedC = cachedTokens(post("/v1/chat/completions", template.formatted("What color is the sun?")));
        check(cachedB >= 20, "second variant resumed at the shared prefix via dense rows (cached " + cachedB + ")");
        check(cachedC >= cachedB, "third variant resumed at least as deep (cached " + cachedC + ")");
    }

    /** --warm-prompt: the file was pre-ingested FULLY DENSE at startup, so chat requests
     *  using it as the system prompt resume token-exact at ANY divergence inside it: full
     *  match, truncation, and a mid-prompt word edit. */
    private static void warmPromptInstant() throws Exception {
        String warm = Files.readString(warmFile);
        HttpResponse<String> full = post("/v1/chat/completions",
                chatBody(warm, "Summarize directive 7 in five words."));
        long ptFull = promptTokens(full), cachedFull = cachedTokens(full);
        check(cachedFull >= ptFull - 40, "full warmed prompt resumed (cached " + cachedFull + " of " + ptFull + ")");
        String cut = warm.substring(0, (int) (warm.length() * 0.6));
        cut = cut.substring(0, cut.lastIndexOf(' '));
        HttpResponse<String> truncated = post("/v1/chat/completions",
                chatBody(cut, "Summarize directive 7 in five words."));
        long ptCut = promptTokens(truncated), cachedCut = cachedTokens(truncated);
        check(cachedCut >= ptCut - 40 && cachedCut < ptCut,
                "truncated warmed prompt resumed token-exact mid-warm (cached " + cachedCut + " of " + ptCut + ")");
        HttpResponse<String> edited = post("/v1/chat/completions",
                chatBody(warm.replace("Directive 25:", "Directive xxv:"), "Summarize directive 7 in five words."));
        long ptMid = promptTokens(edited), cachedMid = cachedTokens(edited);
        check(cachedMid > ptMid / 4 && cachedMid < ptMid - 100,
                "mid-edited warmed prompt resumed at the edit (cached " + cachedMid + " of " + ptMid + ")");
        Map<String, Object> stats = path(json(get("/props")), "prompt_cache");
        check(((Number) stats.get("warm_tokens")).longValue() > 0, "warm_tokens reported");
        check(((Number) stats.get("dense_hits")).longValue() >= 2, "dense hits recorded");
    }

    /** Regular traffic keeps stride bx pairs (K=64) over the body plus a dense tail: an edit
     *  OUTSIDE the tail resumes at the stride point just below the divergence, and the F32
     *  checkpoint attached during that pass upgrades the next divergent request beyond it. */
    private static void strideAndTailResume() throws Exception {
        StringBuilder sb = new StringBuilder("You are a rules engine.");
        for (int i = 1; i <= 80; i++) {
            sb.append(" Rule ").append(i).append(" says topic ").append(i).append(" must cite section ").append(i).append(";");
        }
        String system = sb.toString();
        post("/v1/chat/completions", chatBody(system, "What does rule 3 say? One sentence."));
        HttpResponse<String> tailEdit = post("/v1/chat/completions",
                chatBody(system, "What does rule 9 say? One sentence."));
        long ptTail = promptTokens(tailEdit), cachedTail = cachedTokens(tailEdit);
        check(cachedTail >= ptTail - 40,
                "edited user message resumed token-exact via the dense tail (cached " + cachedTail + " of " + ptTail + ")");
        String edited = system.replace("Rule 20 says", "Rule twenty says");
        HttpResponse<String> midEdit = post("/v1/chat/completions",
                chatBody(edited, "What does rule 3 say? One sentence."));
        long cachedMid = cachedTokens(midEdit);
        check(cachedMid >= 64 && cachedMid % 64 == 0,
                "mid-body edit resumed at a stride point (cached " + cachedMid + ")");
        HttpResponse<String> again = post("/v1/chat/completions",
                chatBody(edited, "What does rule 30 say? One sentence."));
        check(cachedTokens(again) > cachedMid,
                "second divergent request resumed beyond the stride point (cached " + cachedTokens(again) + ")");
    }

    private static String chatBody(String system, String user) {
        return "{\"messages\":[{\"role\":\"system\",\"content\":\"" + system + "\"},"
                + "{\"role\":\"user\",\"content\":\"" + user + "\"}],\"temperature\":0,\"max_tokens\":12,"
                + "\"chat_template_kwargs\":{\"enable_thinking\":false}}";
    }

    private static long promptTokens(HttpResponse<String> response) {
        return ((Number) path(json(response), "usage").get("prompt_tokens")).longValue();
    }

    // --- tool-call parsing (SGLang Lfm2Detector reference semantics; no model needed) ---

    @SuppressWarnings("unchecked")
    private static void toolCallParser() {
        java.util.Set<String> tools = java.util.Set.of("get_weather", "get_time", "book_hotel", "noop");
        // multi-call Pythonic list in one block
        List<Map<String, Object>> calls = Server.parseToolCalls(
                "<|tool_call_start|>[get_weather(city=\"Paris\", unit=\"c\"), get_time(timezone='UTC')]<|tool_call_end|>", tools);
        check(calls.size() == 2, "pythonic list parses both calls (got " + calls.size() + ")");
        check("get_weather".equals(fn(calls, 0).get("name")) && "get_time".equals(fn(calls, 1).get("name")),
                "multi-call names preserved");
        check("{\"city\":\"Paris\",\"unit\":\"c\"}".equals(fn(calls, 0).get("arguments")), "string args -> JSON object");
        // typed literals: numbers, negatives, booleans, None, nested list/dict
        calls = Server.parseToolCalls("<|tool_call_start|>[book_hotel(city='NYC', guests=2, rating=-4.5, "
                + "amenities=['gym', 'pool'], smoking=False, note=None, meta={'floor': 3, 'view': True})]<|tool_call_end|>", tools);
        check(calls.size() == 1, "typed-literal call parses");
        Map<String, Object> args = (Map<String, Object>) Server.Json.parse((String) fn(calls, 0).get("arguments"));
        check(Long.valueOf(2).equals(args.get("guests")) && Double.valueOf(-4.5).equals(args.get("rating")),
                "numbers stay numeric (guests=" + args.get("guests") + ", rating=" + args.get("rating") + ")");
        check(List.of("gym", "pool").equals(args.get("amenities")), "list literal -> JSON array");
        check(Boolean.FALSE.equals(args.get("smoking")) && args.containsKey("note") && args.get("note") == null,
                "False/None map to JSON false/null");
        check(args.get("meta") instanceof Map<?, ?> meta && Long.valueOf(3).equals(meta.get("floor"))
                && Boolean.TRUE.equals(meta.get("view")), "dict literal -> JSON object");
        // single bare call, empty args
        calls = Server.parseToolCalls("<|tool_call_start|>noop()<|tool_call_end|>", tools);
        check(calls.size() == 1 && "{}".equals(fn(calls, 0).get("arguments")), "bare call with no args");
        // JSON block format
        calls = Server.parseToolCalls(
                "<|tool_call_start|>[{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}]<|tool_call_end|>", tools);
        check(calls.size() == 1 && "get_time".equals(fn(calls, 0).get("name")), "JSON block format");
        // unknown tool dropped; malformed yields nothing
        calls = Server.parseToolCalls("<|tool_call_start|>[hack_the_planet(x=1)]<|tool_call_end|>", tools);
        check(calls.isEmpty(), "unknown tool dropped");
        calls = Server.parseToolCalls("<|tool_call_start|>[get_time(timezone=]<|tool_call_end|>", tools);
        check(calls.isEmpty(), "malformed block yields no calls");
        // two separate blocks accumulate
        calls = Server.parseToolCalls("<|tool_call_start|>[get_time(timezone=\"UTC\")]<|tool_call_end|> and "
                + "<|tool_call_start|>[noop()]<|tool_call_end|>", tools);
        check(calls.size() == 2, "separate blocks accumulate");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> fn(List<Map<String, Object>> calls, int index) {
        return (Map<String, Object>) calls.get(index).get("function");
    }

    // --- tool_choice required/named forces a call by seeding <|tool_call_start|> ---

    private static void toolChoiceForced() throws Exception {
        String tools = "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\","
                + "\"description\":\"Current weather for a city\",\"parameters\":{\"type\":\"object\","
                + "\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},"
                + "{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Current time\","
                + "\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
        Map<String, Object> required = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris?\"}]," + tools
                        + ",\"tool_choice\":\"required\",\"temperature\":0,\"max_tokens\":128}"));
        Map<String, Object> message = path(required, "choices", 0, "message");
        check("tool_calls".equals(path(required, "choices", 0).get("finish_reason")),
                "tool_choice required -> finish_reason tool_calls");
        check(message.get("tool_calls") instanceof List<?> calls && !calls.isEmpty(),
                "tool_choice required produced a call");
        Map<String, Object> named = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"What time is it?\"}]," + tools
                        + ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}},"
                        + "\"temperature\":0,\"max_tokens\":128}"));
        Object name = path(named, "choices", 0, "message", "tool_calls", 0, "function").get("name");
        check("get_time".equals(name), "named tool_choice pinned the function (got " + name + ")");
        // system prompt + tools at DEFAULT settings (thinking on): two regressions hid here —
        // a separate tools-only system turn, and COMPACT tool-list JSON (training data is
        // json.dumps-spaced) — each made the model disown its tools ("I don't have access...")
        Map<String, Object> merged = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
                        + "{\"role\":\"user\",\"content\":\"What's the weather in Paris?\"}]," + tools
                        + ",\"temperature\":0,\"max_tokens\":400}"));
        check("tool_calls".equals(path(merged, "choices", 0).get("finish_reason")),
                "system prompt + tools at defaults: model uses the tool instead of refusing");
    }

    // --- thinking budget: content survives tight max_tokens ---

    private static void reasoningBudget() throws Exception {
        // default budget (half of max_tokens): the old behavior burned all 64 on reasoning
        Map<String, Object> response = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"How many letters r are in the word strawberry?\"}],"
                        + "\"temperature\":0,\"max_tokens\":64}"));
        String content = (String) path(response, "choices", 0, "message").get("content");
        check(content != null && !content.isBlank(), "default reasoning budget leaves room for content");
        // explicit cap
        response = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 12 + 13?\"}],"
                        + "\"temperature\":0,\"max_tokens\":80,\"reasoning_max_tokens\":8}"));
        content = (String) path(response, "choices", 0, "message").get("content");
        check(content != null && !content.isBlank(), "explicit reasoning_max_tokens leaves room for content");
        long completion = ((Number) path(response, "usage").get("completion_tokens")).longValue();
        check(completion <= 80, "completion stayed within max_tokens (" + completion + ")");
    }

    // --- stop strings must match CONTENT only, never the think span ---

    private static void stopStringsIgnoreReasoning() throws Exception {
        // "the" is a single token: the old token-stop promotion killed generation at the first
        // "the" INSIDE reasoning, leaving content empty
        Map<String, Object> response = json(post("/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the sky is blue in two sentences.\"}],"
                        + "\"temperature\":0,\"max_tokens\":300,\"reasoning_max_tokens\":60,\"stop\":[\"the\"]}"));
        String content = (String) path(response, "choices", 0, "message").get("content");
        check(content != null && !content.isBlank(), "content produced despite stop word appearing in reasoning");
        check(!content.contains("the"), "stop string still truncates content");
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
