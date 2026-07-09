package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
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
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Stream;

/**
 * End-to-end test of the OpenAI-compatible server, run IN PROCESS on an ephemeral port (one model
 * load, no subprocess management). Guards the bug classes seen in the wild: requests that hang
 * instead of failing fast (validation must answer 400 on the handler thread even while the
 * generation worker is busy), streaming regressions (reasoning_content deltas, usage chunks, [DONE]
 * termination), stop-string truncation, chat-session resume, and prompt-cache cold/warm
 * byte-identity. Requires the model file; skips cleanly when absent.
 */
public final class ServerIntegrationTest {

    private static int failures = 0;
    private static HttpClient client;
    private static String base;
    private static Path warmFile;
    private static String modelId;

    public static void main(String[] args) throws Exception {
        toolCallParser(); // pure parser tests: no model required
        Path model =
                Path.of(args.length > 0 ? args[0] : "../models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf");
        modelId = model.getFileName().toString();
        if (!Files.exists(model)) {
            System.out.println("ServerIntegrationTest: model not found (" + model + "), skipping");
            System.exit(failures > 0 ? 1 : 0);
            return;
        }
        LanguageModel<?, ?, ?> llama = Models.load(model, 2048);
        StringBuilder manual = new StringBuilder("Agent operating manual.");
        for (int i = 1; i <= 50; i++) {
            manual.append(" Directive ")
                    .append(i)
                    .append(": when handling case ")
                    .append(i)
                    .append(", consult registry entry ")
                    .append(i)
                    .append(" and apply policy ")
                    .append(i)
                    .append(" before responding;");
        }
        warmFile = Files.createTempFile("jinfer-warm", ".txt");
        warmFile.toFile().deleteOnExit();
        Files.writeString(warmFile, manual.toString());
        LLMOptions options =
                new LLMOptions(
                        model,
                        null,
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
                        List.of(warmFile.toString()),
                        false,
                        null);
        HttpServer server = Server.start(llama, options);
        base = "http://127.0.0.1:" + server.getAddress().getPort();
        client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

        // "ceiling" mode runs ONLY the token-ceiling check, under a small -Dllama.serverMaxTokens
        // (the two generation limits race — whichever is tighter fires first — so the ceiling needs
        // its own server config; the deadline is tested Engine-level in the normal battery).
        boolean ceilingMode = args.length > 1 && "ceiling".equals(args[1]);
        try {
            if (ceilingMode) {
                tokenCeiling();
            } else {
                plumbing();
                validation();
                complianceValidation();
                tokenizeRoundTrip();
                chatNonStreaming();
                chatStreaming();
                completionsAndStops();
                responsesEndpoint();
                promptCacheReuse();
                sessionPoolReuse();
                toolChoiceForced();
                reasoningBudget();
                stopStringsIgnoreReasoning();
                fast400WhileBusy();
                generationDeadline(llama);
            }
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
        check(
                "list".equals(models.get("object"))
                        && models.get("data") instanceof List<?> l
                        && !l.isEmpty(),
                "models list non-empty");
        String servedId = (String) path(models, "data", 0).get("id");
        Map<String, Object> card = json(get("/v1/models/" + servedId));
        check(
                "model".equals(card.get("object")) && servedId.equals(card.get("id")),
                "GET /v1/models/{id} -> card");
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
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}",
                "all-empty message content");
        expect400(
                "/v1/chat/completions",
                "{\"model\":\"garbage\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                "unknown model name");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"reasoning_max_tokens\":-2}",
                "reasoning_max_tokens=-2");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":2}",
                "n=2");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"n\":2,\"stream\":true}",
                "stream + n=2 pre-SSE");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":-1}",
                "negative temperature");
        expect400("/v1/completions", "{\"prompt\":\"\"}", "empty prompt");
        expect400("/v1/responses", "{}", "responses without input");
    }

    // --- OpenAI-spec compliance: parameters that MUST 400 ---

    private static void complianceValidation() throws Exception {
        // model: required per spec
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                "missing model");
        expect400(
                "/v1/chat/completions",
                "{\"model\":\"\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
                "blank model");

        // temperature: spec says 0–2 (negative already tested in validation())
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"temperature\":3}",
                "temperature > 2");

        // unsupported parameters: must 400, not silently ignore
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"logprobs\":true}",
                "logprobs rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"top_logprobs\":5}",
                "top_logprobs rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"logit_bias\":{\"1234\":5}}",
                "logit_bias rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"frequency_penalty\":0.5}",
                "frequency_penalty rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"presence_penalty\":0.5}",
                "presence_penalty rejected");

        // response_format: only json_object and text supported
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"json_schema\"}}",
                "json_schema rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"garbage\"}}",
                "garbage response_format type rejected");

        // response_format json_object: requires "json" keyword in system/user message
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"response_format\":{\"type\":\"json_object\"}}",
                "json_object without json hint rejected");

        // messages role: must be system/user/assistant/tool
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"hacker\",\"content\":\"hi\"}]}",
                "invalid role rejected");
        expect400(
                "/v1/chat/completions",
                "{\"messages\":[{\"role\":\"\",\"content\":\"hi\"}]}",
                "blank role rejected");
    }

    private static void expect400(String path, String body, String what) throws Exception {
        HttpResponse<String> response = send(path, body);
        check(response.statusCode() == 400, what + " -> 400 (got " + response.statusCode() + ")");
        check(response.body().contains("error"), what + " carries an error payload");
    }

    private static void tokenizeRoundTrip() throws Exception {
        Map<String, Object> tokens = json(post("/tokenize", "{\"content\":\"Hello, world!\"}"));
        check(
                List.of(35808L, 20L, 1530L, 9L).equals(tokens.get("tokens")),
                "tokenize ground truth");
        Map<String, Object> text = json(post("/detokenize", "{\"tokens\":[35808,20,1530,9]}"));
        check("Hello, world!".equals(text.get("content")), "detokenize round trip");
        Map<String, Object> alias = json(post("/v1/tokenize", "{\"content\":\"Hello, world!\"}"));
        check(tokens.get("tokens").equals(alias.get("tokens")), "/v1/tokenize alias");
        check(
                "Hello, world!"
                        .equals(
                                json(post("/v1/detokenize", "{\"tokens\":[35808,20,1530,9]}"))
                                        .get("content")),
                "/v1/detokenize alias");
        HttpResponse<String> metrics = get("/metrics");
        check(
                metrics.statusCode() == 200
                        && metrics.body().contains("jinfer_requests_total")
                        && metrics.body().contains("jinfer_uptime_seconds"),
                "/metrics exposition");
    }

    // --- generation endpoints ---

    private static void chatNonStreaming() throws Exception {
        HttpResponse<String> response =
                post(
                        "/v1/chat/completions",
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"Briefly, why is the sky"
                                + " blue?\"}],\"temperature\":0,\"max_tokens\":48}");
        check(response.statusCode() == 200, "chat 200");
        check(response.headers().firstValue("X-LFM2-Timing").isPresent(), "timing header present");
        Map<String, Object> chat = json(response);
        Map<String, Object> message = path(chat, "choices", 0, "message");
        String content = (String) message.get("content");
        String reasoning = (String) message.get("reasoning_content");
        check(
                (content != null && !content.isEmpty())
                        || (reasoning != null && !reasoning.isEmpty()),
                "chat produced content or reasoning");
        Map<String, Object> usage = path(chat, "usage");
        long prompt = (long) usage.get("prompt_tokens"),
                completion = (long) usage.get("completion_tokens");
        check(
                prompt > 0
                        && completion > 0
                        && (long) usage.get("total_tokens") == prompt + completion,
                "usage arithmetic consistent");
        Object finish = path(chat, "choices", 0).get("finish_reason");
        check("stop".equals(finish) || "length".equals(finish), "finish_reason valid");
    }

    /**
     * Prompt cache on the TurnTemplate path: turn 2 echoes turn 1's history, so the server must
     * resume the shared prefix - reported as usage.prompt_tokens_details.cached_tokens > 0 - and
     * still answer coherently from the restored KV.
     */
    private static void promptCacheReuse() throws Exception {
        String noThink =
                ",\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        String userTurn =
                "{\"role\":\"user\",\"content\":\"Remember the codeword PELICAN. Reply with just"
                        + " OK.\"}";
        Map<String, Object> r1 =
                json(post("/v1/chat/completions", "{\"messages\":[" + userTurn + "]" + noThink));
        String a1 = ((String) ((Map<?, ?>) path(r1, "choices", 0).get("message")).get("content"));
        long cached1 = cachedTokens(r1);
        long prompt1 = (long) ((Map<String, Object>) path(r1, "usage")).get("prompt_tokens");
        // turn 1 may resume only the shared <bos>/preamble cached by earlier requests - a few
        // tokens
        check(cached1 < prompt1, "turn 1 mostly cold (cached " + cached1 + " of " + prompt1 + ")");

        // turn 2 echoes the full history (SAME first user turn + assistant + new user) - the shared
        // prefix (bos + that first user turn) must resume from turn 1's committed blocks
        String turn2 =
                "{\"messages\":["
                        + userTurn
                        + ","
                        + "{\"role\":\"assistant\",\"content\":"
                        + JsonCodec.stringify(a1)
                        + "},"
                        + "{\"role\":\"user\",\"content\":\"What was the codeword? One word.\"}"
                        + "]"
                        + noThink;
        Map<String, Object> r2 = json(post("/v1/chat/completions", turn2));
        long cached2 = cachedTokens(r2);
        String a2 = ((String) ((Map<?, ?>) path(r2, "choices", 0).get("message")).get("content"));
        // turn 2 must reuse turn 1's whole prompt (bos + the first user turn), well beyond turn 1's
        // hit
        check(
                cached2 > cached1 + 8,
                "turn 2 reused turn 1's KV (cached " + cached2 + " vs turn-1 " + cached1 + ")");
        check(
                a2 != null && a2.toUpperCase().contains("PELICAN"),
                "turn 2 recalled the codeword from restored KV: " + a2);
    }

    /**
     * Tier-1 session pool: a 3-turn conversation must continue APPEND-ONLY on the pooled live
     * session for turns 2 and 3 (jinfer_session_pool_hits_total increments and cached_tokens covers
     * the whole prior stream incl. the previous reply); an interleaved second conversation coexists
     * in the pool; and an identical repeat request (no delta) falls to the tier-2 block restore yet
     * answers identically - the pool-vs-restore identity check.
     */
    private static void sessionPoolReuse() throws Exception {
        String noThink =
                ",\"temperature\":0,\"max_tokens\":48,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        String u1 =
                "{\"role\":\"user\",\"content\":\"The secret color is CRIMSON. Reply with just"
                        + " OK.\"}";
        Map<String, Object> r1 =
                json(post("/v1/chat/completions", "{\"messages\":[" + u1 + "]" + noThink));
        String a1 = (String) ((Map<?, ?>) path(r1, "choices", 0).get("message")).get("content");
        long hits0 = metricValue("jinfer_session_pool_hits_total");

        // Turn 2: echo turn 1 + the reply verbatim -> the pooled session's whole stream (prompt +
        // adopted decode tokens) strictly prefixes the request -> tier-1 append-only.
        String turn2 =
                "{\"messages\":["
                        + u1
                        + ","
                        + "{\"role\":\"assistant\",\"content\":"
                        + JsonCodec.stringify(a1)
                        + "},"
                        + "{\"role\":\"user\",\"content\":\"What is the secret color? One word.\"}"
                        + "]"
                        + noThink;
        Map<String, Object> r2 = json(post("/v1/chat/completions", turn2));
        String a2 = (String) ((Map<?, ?>) path(r2, "choices", 0).get("message")).get("content");
        long hits2 = metricValue("jinfer_session_pool_hits_total");
        check(
                hits2 == hits0 + 1,
                "turn 2 continued append-only on the pooled session (hits "
                        + hits0
                        + " -> "
                        + hits2
                        + ")");
        check(
                cachedTokens(r2) > cachedTokens(r1),
                "turn 2 reused the whole live stream (cached " + cachedTokens(r2) + ")");
        check(
                a2 != null && a2.toUpperCase().contains("CRIMSON"),
                "turn 2 recalled from the live KV: " + a2);

        // An interleaved, unrelated conversation coexists in the pool (capacity > 1)...
        json(
                post(
                        "/v1/chat/completions",
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"Say BLUE.\"}]" + noThink));

        // ...so turn 3 still finds conversation A's session and appends again.
        String turn3 =
                "{\"messages\":["
                        + u1
                        + ","
                        + "{\"role\":\"assistant\",\"content\":"
                        + JsonCodec.stringify(a1)
                        + "},"
                        + "{\"role\":\"user\",\"content\":\"What is the secret color? One word.\"},"
                        + "{\"role\":\"assistant\",\"content\":"
                        + JsonCodec.stringify(a2)
                        + "},{\"role\":\"user\",\"content\":\"Repeat the secret color once more."
                        + " One word.\"}]"
                        + noThink;
        Map<String, Object> r3 = json(post("/v1/chat/completions", turn3));
        String a3 = (String) ((Map<?, ?>) path(r3, "choices", 0).get("message")).get("content");
        long hits3 = metricValue("jinfer_session_pool_hits_total");
        check(
                hits3 == hits2 + 1,
                "turn 3 hit tier 1 despite the interleaved conversation (hits " + hits3 + ")");
        check(
                a3 != null && a3.toUpperCase().contains("CRIMSON"),
                "turn 3 recalled from the live KV: " + a3);

        // Identity: the SAME turn-3 request repeated has no delta (not a strict prefix), so it must
        // fall to the tier-2 block restore on a fresh state - and produce the identical reply.
        Map<String, Object> r4 = json(post("/v1/chat/completions", turn3));
        String a4 = (String) ((Map<?, ?>) path(r4, "choices", 0).get("message")).get("content");
        long hits4 = metricValue("jinfer_session_pool_hits_total");
        check(hits4 == hits3, "identical repeat is not append-only (falls to tier-2 restore)");
        check(
                a4 != null && a4.equals(a3),
                "tier-2 restore reply identical to the tier-1 reply: " + a4);
    }

    /** Reads one counter from the Prometheus /metrics exposition. */
    private static long metricValue(String name) throws Exception {
        for (String line : get("/metrics").body().split("\n")) {
            if (line.startsWith(name + " "))
                return (long) Double.parseDouble(line.substring(name.length() + 1));
        }
        return -1;
    }

    private static long cachedTokens(Map<String, Object> chat) {
        Map<String, Object> usage = path(chat, "usage");
        Object details = usage.get("prompt_tokens_details");
        if (details instanceof Map<?, ?> d && d.get("cached_tokens") instanceof Number n)
            return n.longValue();
        return 0;
    }

    private static void chatStreaming() throws Exception {
        HttpRequest request =
                HttpRequest.newBuilder(URI.create(base + "/v1/chat/completions"))
                        .POST(
                                HttpRequest.BodyPublishers.ofString(
                                        "{\"model\":\""
                                                + modelId
                                                + "\",\"messages\":[{\"role\":\"user\",\"content\":\"What"
                                                + " is 2+2?\"}],\"temperature\":0,\"max_tokens\":48,"
                                                + "\"stream\":true,\"stream_options\":{\"include_usage\":true}}"))
                        .timeout(Duration.ofSeconds(60))
                        .build();
        HttpResponse<Stream<String>> response =
                client.send(request, HttpResponse.BodyHandlers.ofLines());
        check(response.statusCode() == 200, "stream 200");
        boolean[] sawReasoning = {false}, sawUsage = {false}, sawDone = {false};
        response.body()
                .forEach(
                        line -> {
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
        Map<String, Object> completion =
                json(
                        post(
                                "/v1/completions",
                                "{\"prompt\":\"Count: one, two, three,"
                                    + " four\",\"max_tokens\":24,\"temperature\":0,\"stop\":[\","
                                    + " six\"]}"));
        String text = (String) path(completion, "choices", 0).get("text");
        check(
                text != null && text.contains("five") && !text.contains("six"),
                "stop string truncated the completion (got: " + text + ")");
        check(
                "stop".equals(path(completion, "choices", 0).get("finish_reason")),
                "stop finish_reason");
    }

    private static void responsesEndpoint() throws Exception {
        Map<String, Object> response =
                json(
                        post(
                                "/v1/responses",
                                "{\"input\":\"Reply with one word: ok\",\"max_output_tokens\":48,"
                                        + "\"chat_template_kwargs\":{\"enable_thinking\":false}}"));
        check("completed".equals(response.get("status")), "responses status completed");
        String text = (String) path(response, "output", 0, "content", 0).get("text");
        check(text != null && !text.isEmpty(), "responses output_text non-empty");
    }

    // --- prompt cache: identical long-prompt requests; warm run resumes and output is identical
    // ---

    private static void promptCacheColdWarm() throws Exception {
        String body =
                "{\"prompt\":\""
                        + "The quick brown fox jumps over the lazy dog. ".repeat(80)
                        + "\",\"max_tokens\":4,\"temperature\":0}";
        HttpResponse<String> cold = post("/v1/completions", body);
        HttpResponse<String> warm = post("/v1/completions", body);
        long cachedCold = cachedTokens(cold), cachedWarm = cachedTokens(warm);
        // earlier requests may have left a checkpoint near the stream start (divergence right
        // after BOS), so "cold" can legitimately resume a few tokens — but no more
        check(cachedCold <= 4, "cold run essentially uncached (got " + cachedCold + ")");
        long promptTokens = ((Number) path(json(warm), "usage").get("prompt_tokens")).longValue();
        check(
                cachedWarm == promptTokens,
                "warm run resumed token-exact at L-1 (cached "
                        + cachedWarm
                        + " of "
                        + promptTokens
                        + ")");
        String coldText = (String) path(json(cold), "choices", 0).get("text");
        String warmText = (String) path(json(warm), "choices", 0).get("text");
        check(coldText.equals(warmText), "cold and warm outputs identical");
    }

    private static long cachedTokens(HttpResponse<String> response) {
        Object cached = path(json(response), "usage", "prompt_tokens_details").get("cached_tokens");
        return ((Number) cached).longValue();
    }

    /**
     * A strict PREFIX of already-cached content: the end-of-prompt checkpoint splits inside a
     * cached node, and the end-of-generation commit dedups through existing nodes (pin-walk). The
     * repeat request must then resume token-exact at its own L-1.
     */
    private static void promptCacheStrictPrefix() throws Exception {
        String body =
                "{\"prompt\":\""
                        + "The quick brown fox jumps over the lazy dog. ".repeat(40)
                        + "\",\"max_tokens\":4,\"temperature\":0}";
        HttpResponse<String> first =
                post("/v1/completions", body); // prefix of the 80-rep stream above
        HttpResponse<String> second = post("/v1/completions", body);
        long promptTokens = ((Number) path(json(second), "usage").get("prompt_tokens")).longValue();
        check(
                cachedTokens(second) == promptTokens,
                "strict-prefix re-request resumed token-exact (cached "
                        + cachedTokens(second)
                        + " of "
                        + promptTokens
                        + ")");
        String firstText = (String) path(json(first), "choices", 0).get("text");
        String secondText = (String) path(json(second), "choices", 0).get("text");
        check(firstText.equals(secondText), "strict-prefix outputs identical");
    }

    /**
     * Requests sharing only a SYSTEM PROMPT prefix. A populates the tree; the short stream lies
     * entirely inside the dense tail, so B and C resume token-exact at the divergence on their
     * FIRST visit (bx rows — no checkpoint round-trip needed anymore).
     */
    private static void promptCacheBranchPoint() throws Exception {
        String system =
                "You are the dedicated test oracle for the jinfer integration battery, a most"
                    + " particular and verbose persona that always answers with at most one short"
                    + " sentence and no preamble.";
        String template =
                "{\"messages\":[{\"role\":\"system\",\"content\":\""
                        + system
                        + "\"},{\"role\":\"user\",\"content\":\"%s\"}],\"temperature\":0,\"max_tokens\":24,"
                        + "\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        long cachedA =
                cachedTokens(
                        post("/v1/chat/completions", template.formatted("What color is the sea?")));
        long cachedB =
                cachedTokens(
                        post("/v1/chat/completions", template.formatted("What color is grass?")));
        long cachedC =
                cachedTokens(
                        post("/v1/chat/completions", template.formatted("What color is the sun?")));
        check(
                cachedB >= 20,
                "second variant resumed at the shared prefix via dense rows (cached "
                        + cachedB
                        + ")");
        check(
                cachedC >= cachedB,
                "third variant resumed at least as deep (cached " + cachedC + ")");
    }

    /**
     * --warm-prompt: the file was pre-ingested FULLY DENSE at startup, so chat requests using it as
     * the system prompt resume token-exact at ANY divergence inside it: full match, truncation, and
     * a mid-prompt word edit.
     */
    private static void warmPromptInstant() throws Exception {
        String warm = Files.readString(warmFile);
        HttpResponse<String> full =
                post(
                        "/v1/chat/completions",
                        chatBody(warm, "Summarize directive 7 in five words."));
        long ptFull = promptTokens(full), cachedFull = cachedTokens(full);
        check(
                cachedFull >= ptFull - 40,
                "full warmed prompt resumed (cached " + cachedFull + " of " + ptFull + ")");
        String cut = warm.substring(0, (int) (warm.length() * 0.6));
        cut = cut.substring(0, cut.lastIndexOf(' '));
        HttpResponse<String> truncated =
                post("/v1/chat/completions", chatBody(cut, "Summarize directive 7 in five words."));
        long ptCut = promptTokens(truncated), cachedCut = cachedTokens(truncated);
        check(
                cachedCut >= ptCut - 40 && cachedCut < ptCut,
                "truncated warmed prompt resumed token-exact mid-warm (cached "
                        + cachedCut
                        + " of "
                        + ptCut
                        + ")");
        HttpResponse<String> edited =
                post(
                        "/v1/chat/completions",
                        chatBody(
                                warm.replace("Directive 25:", "Directive xxv:"),
                                "Summarize directive 7 in five words."));
        long ptMid = promptTokens(edited), cachedMid = cachedTokens(edited);
        check(
                cachedMid > ptMid / 4 && cachedMid < ptMid - 100,
                "mid-edited warmed prompt resumed at the edit (cached "
                        + cachedMid
                        + " of "
                        + ptMid
                        + ")");
        Map<String, Object> stats = path(json(get("/props")), "prompt_cache");
        check(((Number) stats.get("warm_tokens")).longValue() > 0, "warm_tokens reported");
        check(((Number) stats.get("dense_hits")).longValue() >= 2, "dense hits recorded");
    }

    /**
     * Regular traffic keeps stride bx pairs (K=64) over the body plus a dense tail: an edit OUTSIDE
     * the tail resumes at the stride point just below the divergence, and the F32 checkpoint
     * attached during that pass upgrades the next divergent request beyond it.
     */
    private static void strideAndTailResume() throws Exception {
        StringBuilder sb = new StringBuilder("You are a rules engine.");
        for (int i = 1; i <= 80; i++) {
            sb.append(" Rule ")
                    .append(i)
                    .append(" says topic ")
                    .append(i)
                    .append(" must cite section ")
                    .append(i)
                    .append(";");
        }
        String system = sb.toString();
        post("/v1/chat/completions", chatBody(system, "What does rule 3 say? One sentence."));
        HttpResponse<String> tailEdit =
                post(
                        "/v1/chat/completions",
                        chatBody(system, "What does rule 9 say? One sentence."));
        long ptTail = promptTokens(tailEdit), cachedTail = cachedTokens(tailEdit);
        check(
                cachedTail >= ptTail - 40,
                "edited user message resumed token-exact via the dense tail (cached "
                        + cachedTail
                        + " of "
                        + ptTail
                        + ")");
        String edited = system.replace("Rule 20 says", "Rule twenty says");
        HttpResponse<String> midEdit =
                post(
                        "/v1/chat/completions",
                        chatBody(edited, "What does rule 3 say? One sentence."));
        long cachedMid = cachedTokens(midEdit);
        check(
                cachedMid >= 64 && cachedMid % 64 == 0,
                "mid-body edit resumed at a stride point (cached " + cachedMid + ")");
        HttpResponse<String> again =
                post(
                        "/v1/chat/completions",
                        chatBody(edited, "What does rule 30 say? One sentence."));
        check(
                cachedTokens(again) > cachedMid,
                "second divergent request resumed beyond the stride point (cached "
                        + cachedTokens(again)
                        + ")");
    }

    private static String chatBody(String system, String user) {
        return "{\"messages\":[{\"role\":\"system\",\"content\":\""
                + system
                + "\"},"
                + "{\"role\":\"user\",\"content\":\""
                + user
                + "\"}],\"temperature\":0,\"max_tokens\":12,"
                + "\"chat_template_kwargs\":{\"enable_thinking\":false}}";
    }

    private static long promptTokens(HttpResponse<String> response) {
        return ((Number) path(json(response), "usage").get("prompt_tokens")).longValue();
    }

    // --- tool-call parsing (SGLang Lfm2Detector reference semantics; no model needed) ---

    @SuppressWarnings("unchecked")
    private static void toolCallParser() {
        Set<String> tools = Set.of("get_weather", "get_time", "book_hotel", "noop");
        // multi-call Pythonic list in one block
        List<Map<String, Object>> calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[get_weather(city=\"Paris\", unit=\"c\"),"
                                + " get_time(timezone='UTC')]<|tool_call_end|>",
                        tools);
        check(calls.size() == 2, "pythonic list parses both calls (got " + calls.size() + ")");
        check(
                "get_weather".equals(fn(calls, 0).get("name"))
                        && "get_time".equals(fn(calls, 1).get("name")),
                "multi-call names preserved");
        check(
                "{\"city\":\"Paris\",\"unit\":\"c\"}".equals(fn(calls, 0).get("arguments")),
                "string args -> JSON object");
        // typed literals: numbers, negatives, booleans, None, nested list/dict
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[book_hotel(city='NYC', guests=2, rating=-4.5,"
                            + " amenities=['gym', 'pool'], smoking=False, note=None, meta={'floor':"
                            + " 3, 'view': True})]<|tool_call_end|>",
                        tools);
        check(calls.size() == 1, "typed-literal call parses");
        Map<String, Object> args =
                (Map<String, Object>) JsonCodec.parse((String) fn(calls, 0).get("arguments"));
        check(
                Long.valueOf(2).equals(args.get("guests"))
                        && Double.valueOf(-4.5).equals(args.get("rating")),
                "numbers stay numeric (guests="
                        + args.get("guests")
                        + ", rating="
                        + args.get("rating")
                        + ")");
        check(List.of("gym", "pool").equals(args.get("amenities")), "list literal -> JSON array");
        check(
                Boolean.FALSE.equals(args.get("smoking"))
                        && args.containsKey("note")
                        && args.get("note") == null,
                "False/None map to JSON false/null");
        check(
                args.get("meta") instanceof Map<?, ?> meta
                        && Long.valueOf(3).equals(meta.get("floor"))
                        && Boolean.TRUE.equals(meta.get("view")),
                "dict literal -> JSON object");
        // single bare call, empty args
        calls = ToolCalls.parseToolCalls("<|tool_call_start|>noop()<|tool_call_end|>", tools);
        check(
                calls.size() == 1 && "{}".equals(fn(calls, 0).get("arguments")),
                "bare call with no args");
        // JSON block format
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[{\"name\":\"get_time\",\"arguments\":{\"timezone\":\"UTC\"}}]<|tool_call_end|>",
                        tools);
        check(
                calls.size() == 1 && "get_time".equals(fn(calls, 0).get("name")),
                "JSON block format");
        // unknown tool dropped; malformed yields nothing
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[hack_the_planet(x=1)]<|tool_call_end|>", tools);
        check(calls.isEmpty(), "unknown tool dropped");
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[get_time(timezone=]<|tool_call_end|>", tools);
        check(calls.isEmpty(), "malformed block yields no calls");
        // two separate blocks accumulate
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[get_time(timezone=\"UTC\")]<|tool_call_end|> and "
                                + "<|tool_call_start|>[noop()]<|tool_call_end|>",
                        tools);
        check(calls.size() == 2, "separate blocks accumulate");
        // bare pythonic (no markers) — the model sometimes emits these
        calls = ToolCalls.parseToolCalls("[noop()]", tools);
        check(
                calls.size() == 1 && "noop".equals(fn(calls, 0).get("name")),
                "bare pythonic noop (got " + calls.size() + ")");
        // bare pythonic embedded in prose
        calls = ToolCalls.parseToolCalls("<think>I should call noop.</think>\n[noop()]", tools);
        check(
                calls.size() == 1 && "noop".equals(fn(calls, 0).get("name")),
                "bare pythonic after thinking (got " + calls.size() + ")");
        // bare pythonic with prose BEFORE and AFTER
        calls =
                ToolCalls.parseToolCalls(
                        "Some text before\n[get_weather(city=\"Paris\")]\nSome text after", tools);
        check(
                calls.size() == 1 && "get_weather".equals(fn(calls, 0).get("name")),
                "bare pythonic in prose (got " + calls.size() + ")");
        // bare call without brackets
        calls = ToolCalls.parseToolCalls("noop()", tools);
        check(
                calls.size() == 1 && "noop".equals(fn(calls, 0).get("name")),
                "bare call without brackets (got " + calls.size() + ")");
        // markdown link NOT parsed as tool call
        calls = ToolCalls.parseToolCalls("[Google](https://google.com)", tools);
        check(calls.isEmpty(), "markdown link not parsed as tool call");
        // code example NOT parsed (func not in known tools)
        calls = ToolCalls.parseToolCalls("[calc(x=1, y=2)]", tools);
        check(calls.isEmpty(), "unknown function not parsed");

        // --- bullet-proofing: string-aware scanning of the bare (marker-less) form ---
        // brackets/parens/commas inside a quoted argument must NOT terminate detection early
        calls = ToolCalls.parseToolCalls("[get_weather(city=\"a ] b ) c , d\")]", tools);
        check(
                calls.size() == 1
                        && "{\"city\":\"a ] b ) c , d\"}".equals(fn(calls, 0).get("arguments")),
                "bracket/paren/comma inside string arg preserved (got " + calls.size() + ")");
        // same hostile content but with native markers
        calls =
                ToolCalls.parseToolCalls(
                        "<|tool_call_start|>[get_weather(note=\"close ) bracket"
                                + " ]\")]<|tool_call_end|>",
                        tools);
        check(
                calls.size() == 1
                        && "{\"note\":\"close ) bracket ]\"}".equals(fn(calls, 0).get("arguments")),
                "string with brackets in native block (got " + calls.size() + ")");
        // multiple bare calls in a list, no markers
        calls =
                ToolCalls.parseToolCalls(
                        "[get_weather(city=\"Paris\"), get_time(timezone=\"UTC\")]", tools);
        check(
                calls.size() == 2
                        && "get_weather".equals(fn(calls, 0).get("name"))
                        && "get_time".equals(fn(calls, 1).get("name")),
                "multi bare-list calls (got " + calls.size() + ")");
        // dotted function name (qualified, e.g. Calendar.create_event)
        Set<String> dotted = Set.of("Calendar.create_event");
        calls = ToolCalls.parseToolCalls("[Calendar.create_event(title=\"Sync\")]", dotted);
        check(
                calls.size() == 1 && "Calendar.create_event".equals(fn(calls, 0).get("name")),
                "dotted function name (got " + calls.size() + ")");
        // JSON-cased literals inside pythonic args (true/false/null lowercase)
        calls =
                ToolCalls.parseToolCalls(
                        "[book_hotel(city=\"NYC\", smoking=false, note=null)]", tools);
        check(calls.size() == 1, "json-cased literals parse (got " + calls.size() + ")");
        Map<String, Object> jsonArgs =
                (Map<String, Object>) JsonCodec.parse((String) fn(calls, 0).get("arguments"));
        check(
                Boolean.FALSE.equals(jsonArgs.get("smoking"))
                        && jsonArgs.containsKey("note")
                        && jsonArgs.get("note") == null,
                "lowercase false/null map correctly");
        // single-quoted strings in the marker-less form
        calls = ToolCalls.parseToolCalls("[get_weather(city='Berlin')]", tools);
        check(
                calls.size() == 1 && "{\"city\":\"Berlin\"}".equals(fn(calls, 0).get("arguments")),
                "single-quoted string in bare call");
        // identifier suffix must not false-match a known tool (xget_weather != get_weather)
        calls = ToolCalls.parseToolCalls("[xget_weather(city=\"Paris\")]", tools);
        check(calls.isEmpty(), "longer identifier containing a tool name is not a match");
        // bare call embedded mid-prose, no brackets
        calls =
                ToolCalls.parseToolCalls(
                        "Sure, let me check: get_time(timezone=\"UTC\") now.", tools);
        check(
                calls.size() == 1 && "get_time".equals(fn(calls, 0).get("name")),
                "bare call without brackets mid-prose (got " + calls.size() + ")");
        // empty bracketed list is not a call
        calls = ToolCalls.parseToolCalls("[]", tools);
        check(calls.isEmpty(), "empty list is not a tool call");
        // function reference in prose without a call is not parsed
        calls = ToolCalls.parseToolCalls("You can use get_weather to look that up.", tools);
        check(calls.isEmpty(), "bare tool name without parens is not a call");
        // no tools defined -> never treat text as a call
        calls = ToolCalls.parseToolCalls("[get_weather(city=\"Paris\")]", java.util.Set.of());
        check(calls.isEmpty(), "no tools defined -> no bare detection");
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> fn(List<Map<String, Object>> calls, int index) {
        return (Map<String, Object>) calls.get(index).get("function");
    }

    // --- tool_choice required/named forces a call by seeding <|tool_call_start|> ---

    private static void toolChoiceForced() throws Exception {
        String tools =
                "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Current"
                    + " weather for a city\",\"parameters\":{\"type\":\"object\","
                    + "\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"get_time\",\"description\":\"Current"
                    + " time\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]";
        Map<String, Object> required =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is the"
                                        + " weather in Paris?\"}],"
                                        + tools
                                        + ",\"tool_choice\":\"required\",\"temperature\":0,\"max_tokens\":128}"));
        Map<String, Object> message = path(required, "choices", 0, "message");
        check(
                "tool_calls".equals(path(required, "choices", 0).get("finish_reason")),
                "tool_choice required -> finish_reason tool_calls");
        check(
                message.get("tool_calls") instanceof List<?> calls && !calls.isEmpty(),
                "tool_choice required produced a call");
        Map<String, Object> named =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\",\"content\":\"What time is"
                                        + " it?\"}],"
                                        + tools
                                        + ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"get_time\"}},"
                                        + "\"temperature\":0,\"max_tokens\":128}"));
        Object name = path(named, "choices", 0, "message", "tool_calls", 0, "function").get("name");
        check("get_time".equals(name), "named tool_choice pinned the function (got " + name + ")");
        // system prompt + tools at DEFAULT settings (thinking on): two regressions hid here —
        // a separate tools-only system turn, and COMPACT tool-list JSON (training data is
        // json.dumps-spaced) — each made the model disown its tools ("I don't have access...")
        Map<String, Object> merged =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"system\",\"content\":\"You are a"
                                    + " helpful"
                                    + " assistant.\"},{\"role\":\"user\",\"content\":\"What's the"
                                    + " weather in Paris?\"}],"
                                        + tools
                                        + ",\"temperature\":0,\"max_tokens\":400}"));
        check(
                "tool_calls".equals(path(merged, "choices", 0).get("finish_reason")),
                "system prompt + tools at defaults: model uses the tool instead of refusing");
    }

    // --- thinking budget: content survives tight max_tokens ---

    private static void reasoningBudget() throws Exception {
        // default budget (half of max_tokens): the old behavior burned all 64 on reasoning
        Map<String, Object> response =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\",\"content\":\"How many letters r"
                                        + " are in the word strawberry?\"}],"
                                        + "\"temperature\":0,\"max_tokens\":64}"));
        String content = (String) path(response, "choices", 0, "message").get("content");
        check(
                content != null && !content.isBlank(),
                "default reasoning budget leaves room for content");
        // explicit cap
        response =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\",\"content\":\"What is 12 +"
                                    + " 13?\"}],"
                                    + "\"temperature\":0,\"max_tokens\":80,\"reasoning_max_tokens\":8}"));
        content = (String) path(response, "choices", 0, "message").get("content");
        check(
                content != null && !content.isBlank(),
                "explicit reasoning_max_tokens leaves room for content");
        long completion = ((Number) path(response, "usage").get("completion_tokens")).longValue();
        check(completion <= 80, "completion stayed within max_tokens (" + completion + ")");
    }

    // --- stop strings must match CONTENT only, never the think span ---

    private static void stopStringsIgnoreReasoning() throws Exception {
        // "the" is a single token: the old token-stop promotion killed generation at the first
        // "the" INSIDE reasoning, leaving content empty
        Map<String, Object> response =
                json(
                        post(
                                "/v1/chat/completions",
                                "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the"
                                    + " sky is blue in two sentences.\"}],"
                                    + "\"temperature\":0,\"max_tokens\":300,\"reasoning_max_tokens\":60,\"stop\":[\"the\"]}"));
        String content = (String) path(response, "choices", 0, "message").get("content");
        check(
                content != null && !content.isBlank(),
                "content produced despite stop word appearing in reasoning");
        check(!content.contains("the"), "stop string still truncates content");
    }

    // --- a garbage request must be answered while the worker is mid-generation ---

    private static void fast400WhileBusy() throws Exception {
        HttpRequest slow =
                HttpRequest.newBuilder(URI.create(base + "/v1/chat/completions"))
                        .POST(
                                HttpRequest.BodyPublishers.ofString(
                                        "{\"model\":\""
                                                + modelId
                                                + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Write"
                                                + " a long"
                                                + " story.\"}],\"temperature\":0,\"max_tokens\":200}"))
                        .timeout(Duration.ofSeconds(60))
                        .build();
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
        String turn1 =
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word:"
                    + " hello\"}],"
                    + "\"temperature\":0,\"max_tokens\":400,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        Map<String, Object> first = json(post("/v1/chat/completions", turn1));
        check(
                "stop".equals(path(first, "choices", 0).get("finish_reason")),
                "turn 1 stopped cleanly");
        String reply = (String) path(first, "choices", 0, "message").get("content");
        String turn2 =
                "{\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word:"
                        + " hello\"},{\"role\":\"assistant\",\"content\":"
                        + JsonCodec.stringify(reply)
                        + "},{\"role\":\"user\",\"content\":\"now say goodbye\"}],"
                        + "\"temperature\":0,\"max_tokens\":400,\"chat_template_kwargs\":{\"enable_thinking\":false}}";
        HttpResponse<String> second = post("/v1/chat/completions", turn2);
        check(second.statusCode() == 200, "turn 2 200");
        check(
                cachedTokens(second) > 0,
                "turn 2 resumed the session (cached " + cachedTokens(second) + ")");
    }

    // --- generation limits ---

    /**
     * Wall-clock deadline: Params.timeoutNanos aborts a long generation through the per-token abort
     * path, reporting finish_reason "length". Engine-level so it needs no global flag.
     */
    private static <S extends RuntimeState> void generationDeadline(LanguageModel<?, ?, S> model) {
        List<Integer> prompt = model.tokenizer().encode("Write a very long, detailed story.");
        Generator.Params params =
                new Generator.Params(
                        Sampler.ARGMAX,
                        100_000,
                        TimeUnit.MILLISECONDS.toNanos(300),
                        new Generator.StopSpec(Set.of(), List.of()),
                        false);
        long start = System.nanoTime();
        S state = model.newState(model.config().contextLength(), Math.max(prompt.size(), 16));
        Generator.GenerationResult result =
                Generator.generate(
                        model,
                        state,
                        prompt,
                        params,
                        new Generator.Listener(null, null, null, null));
        long elapsedMs = (System.nanoTime() - start) / 1_000_000;
        check(
                "length".equals(result.finishReason()),
                "deadline -> finish_reason length (" + result.finishReason() + ")");
        check(
                result.completionTokens() > 0 && result.completionTokens() < 100_000,
                "deadline aborted mid-generation (" + result.completionTokens() + " tokens)");
        check(elapsedMs < 30_000, "deadline honored promptly (" + elapsedMs + "ms)");
    }

    /**
     * Server-side completion ceiling: an oversized request is clamped to llama.serverMaxTokens (set
     * small for this isolated run), reporting finish_reason "length".
     */
    private static void tokenCeiling() throws Exception {
        int ceiling = ServerFlags.SERVER_MAX_TOKENS;
        check(ceiling > 0, "ceiling mode needs -Dllama.serverMaxTokens > 0 (" + ceiling + ")");
        HttpResponse<String> response =
                post(
                        "/v1/chat/completions",
                        "{\"messages\":[{\"role\":\"user\",\"content\":\"Write a very long,"
                                + " detailed story.\"}],\"temperature\":0,\"max_tokens\":100000}");
        check(response.statusCode() == 200, "ceiling request 200");
        Map<String, Object> chat = json(response);
        long completion = ((Number) path(chat, "usage").get("completion_tokens")).longValue();
        check(
                completion <= ceiling,
                "completion clamped to serverMaxTokens (" + completion + " <= " + ceiling + ")");
        check(
                "length".equals(path(chat, "choices", 0).get("finish_reason")),
                "ceiling -> finish_reason length");
    }

    // --- helpers ---

    private static HttpResponse<String> get(String path) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(base + path))
                        .timeout(Duration.ofSeconds(30))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    /**
     * POST a positive request: generation endpoints require a "model" field (OpenAI-compatible
     * validation), so inject the served model when the body omits it. Negative tests use {@link
     * #send} directly to keep their bodies verbatim (including the deliberate "missing model"
     * case).
     */
    private static HttpResponse<String> post(String path, String body) throws Exception {
        if ((path.contains("completions") || path.contains("responses"))
                && body.startsWith("{")
                && body.length() > 1
                && body.charAt(1) != '}'
                && !body.contains("\"model\"")) {
            body = "{\"model\":\"" + modelId + "\"," + body.substring(1);
        }
        return send(path, body);
    }

    private static HttpResponse<String> send(String path, String body) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(base + path))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .timeout(Duration.ofSeconds(60))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> json(HttpResponse<String> response) {
        return (Map<String, Object>) JsonCodec.parse(response.body());
    }

    /** Walks maps by key and lists by index: path(m, "choices", 0, "message"). */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> path(Map<String, Object> root, Object... steps) {
        Object current = root;
        for (Object step : steps) {
            current =
                    step instanceof Integer i
                            ? ((List<Object>) current).get(i)
                            : ((Map<String, Object>) current).get(step);
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
