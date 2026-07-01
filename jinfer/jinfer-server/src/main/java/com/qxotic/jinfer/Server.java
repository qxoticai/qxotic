package com.qxotic.jinfer;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.Function;

import com.qxotic.jinfer.Engine.GenerationResult;

/**
 * The OpenAI-compatible HTTP server, and the module's sole public entry point: {@link #start}.
 * The transport/protocol layer — it registers the routes (/v1/chat/completions, /v1/completions,
 * /v1/responses, /v1/models, /health, /props, /metrics, /tokenize, /detokenize), parses + validates
 * requests, and translates between the wire (JSON, SSE event sequences) and the inference service.
 * The plumbing it builds on lives in {@link Http} (responses/CORS/errors), {@link Sse} (streaming),
 * {@link Worker} (the generation queue), and {@link Metrics}; all inference goes through
 * {@link Generation}. Everything but {@code start} is package-private.
 */
public final class Server {

    private Server() {
    }

    /**
     * Starts the server for an already-loaded {@code model} and returns the running instance (it
     * serves on its own executor; this call does not block). Host/port come from {@code options};
     * port 0 binds an ephemeral port, readable from {@link HttpServer#getAddress()}. Prompt-cache
     * warming, if configured, completes before this returns. This is the only public API of the
     * module — load a model (jinfer-core), then hand it here to serve it.
     */
    public static HttpServer start(ModelLegacy model, LLMOptions options) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(options.host(), options.port()), 0);
        String servedId = options.modelPath().getFileName().toString();
        Map<String, Object> modelCard = Map.of(
                "id", servedId, "object", "model", "created", 0, "owned_by", "jinfer");
        server.createContext("/v1/models", exchange -> { // also serves /v1/models/{id} -> card or 404
            if (Http.preamble(exchange)) return;
            if (Http.requireMethod(exchange, "GET")) return;
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/v1/models")) {
                Http.sendJson(exchange, 200, Map.of("object", "list", "data", List.of(modelCard)));
            } else if (path.equals("/v1/models/" + servedId)) {
                Http.sendJson(exchange, 200, modelCard);
            } else {
                Http.sendError(exchange, 404, "Unknown model: " + path.substring("/v1/models/".length())
                        + " (this server serves " + servedId + ")");
            }
        });
        server.createContext("/v1/chat/completions", exchange -> handleChatCompletion(exchange, options));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, options));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, options));
        jsonRoute(server, "/health", null, request ->
                Map.of("status", "ok", "busy", WORKER.busy(), "queued", WORKER.queueDepth()));
        jsonRoute(server, "/props", null, request -> Map.of(
                "model", options.modelPath().getFileName().toString(),
                "n_ctx", model.contextLength(),
                "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                "n_vocab", model.vocabularySize(),
                "prompt_cache", GENERATION.cache() == null ? Map.of("enabled", false) : GENERATION.cache().stats()));
        Function<Map<String, Object>, Object> tokenize = request ->
                Map.of("tokens", model.tokenizer().encode(String.valueOf(request.getOrDefault("content", ""))));
        Function<Map<String, Object>, Object> detokenize = request -> {
            List<Integer> tokens = request.get("tokens") instanceof List<?> list
                    ? list.stream().map(v -> ((Number) v).intValue()).toList()
                    : List.<Integer>of();
            return Map.of("content", model.tokenizer().decode(tokens));
        };
        jsonRoute(server, "/tokenize", "POST", tokenize);       // llama.cpp paths and the
        jsonRoute(server, "/v1/tokenize", "POST", tokenize);    // /v1-prefixed aliases
        jsonRoute(server, "/detokenize", "POST", detokenize);
        jsonRoute(server, "/v1/detokenize", "POST", detokenize);
        server.createContext("/metrics", Server::handleMetrics);
        server.createContext("/", exchange -> {
            if (Http.preamble(exchange)) return;
            Http.sendError(exchange, 404, "Not found");
        });
        GENERATION = new Generation(model, options, WORKER);
        WORKER.start();
        GENERATION.warm(); // blocks until warmed: instant resume from request one
        Sse.startReaper();
        // bounded pool: handlers only parse/validate and block on the generation queue latch,
        // so a fixed pool also caps the threads slow-loris connections can pin
        server.setExecutor(Executors.newFixedThreadPool(RuntimeFlags.SERVER_THREADS));
        server.start();
        Runtime.getRuntime().addShutdownHook(new Thread(() -> server.stop(1)));
        System.out.printf("OpenAI-compatible server listening on http://%s:%d%n",
                options.host(), server.getAddress().getPort());
        return server;
    }

    /** Registers a JSON endpoint with the shared preamble (request log, CORS headers, OPTIONS
     *  preflight), an optional method restriction, the parsed JSON body for POST routes, and
     *  the uniform 400 error envelope. */
    private static void jsonRoute(HttpServer server, String path, String method,
                                  Function<Map<String, Object>, Object> body) {
        server.createContext(path, exchange -> {
            if (Http.preamble(exchange)) return;
            // contexts match by longest PREFIX: /v1/models/garbage would land here — 404 it
            if (!exchange.getRequestURI().getPath().equals(path)) {
                Http.sendError(exchange, 404, "Not found");
                return;
            }
            if (method != null && Http.requireMethod(exchange, method)) return;
            Map<String, Object> request = Map.of();
            if ("POST".equals(method)) {
                byte[] raw = Http.readBody(exchange);
                if (raw == null) return;
                try {
                    request = Values.asObject(JsonCodec.parse(new String(raw, StandardCharsets.UTF_8)), "request");
                } catch (RuntimeException e) {
                    Http.sendError(exchange, 400, Http.errorMessage(e));
                    return;
                }
            }
            try {
                Http.sendJson(exchange, 200, body.apply(request));
            } catch (RuntimeException e) {
                Http.sendError(exchange, 400, Http.errorMessage(e));
            }
        });
    }

    private interface RequestJob {
        void run(Map<String, Object> request, String id) throws IOException;
    }

    /** Shared scaffold for the generation POST endpoints: preflight/method checks, bounded body
     *  read, then parse + validation on the HANDLER thread — malformed requests get an instant
     *  400 and never occupy the generation worker, even while it is busy with a long generation.
     *  The job runs on the worker via the bounded queue with uniform error handling. */
    private static void handleGenerationPost(HttpExchange exchange, String idPrefix,
                                             Consumer<Map<String, Object>> validator, RequestJob job) throws IOException {
        if (Http.preamble(exchange)) return;
        if (Http.requireMethod(exchange, "POST")) return;
        byte[] body = Http.readBody(exchange); // read on the handler thread: a stalled upload must not block the generation worker
        if (body == null) return;
        Map<String, Object> request;
        try {
            request = Values.asObject(JsonCodec.parse(new String(body, StandardCharsets.UTF_8)), "request");
            validator.accept(request);
        } catch (RuntimeException e) {
            Http.sendError(exchange, 400, Http.errorMessage(e));
            return;
        }
        String id = idPrefix + Long.toUnsignedString(System.nanoTime(), 36);
        runQueued(exchange, () -> {
            try {
                job.run(request, id);
            } catch (RuntimeException e) {
                Http.sendErrorQuietly(exchange, 400, Http.errorMessage(e));
            } catch (IOException e) {
                System.err.println("client connection lost: " + e);
            } catch (Throwable t) {
                Http.sendErrorQuietly(exchange, 500, t.toString());
            }
        });
    }

    private static void handleChatCompletion(HttpExchange exchange, LLMOptions options) throws IOException {
        handleGenerationPost(exchange, "chatcmpl-", request -> {
            Validation.validateChatRequest(request);
            Validation.validateGenerationParams(request, options);
        }, (request, id) -> {
            List<Object> messages = Values.asArray(request.get("messages"), "messages");
            String modelId = Requests.modelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamChatCompletion(exchange, request, messages, modelId, id);
            } else {
                GenerationResult result = GENERATION.chat(request, messages, Sinks.NONE); // non-streaming, no tools
                respond(exchange, result, OpenAiSchema.chatCompletionResponse(id, modelId, result));
            }
        });
    }

    private static void handleCompletion(HttpExchange exchange, LLMOptions options) throws IOException {
        handleGenerationPost(exchange, "cmpl-", request -> {
            Validation.validateGenerationParams(request, options);
            LLMOptions.require(!Requests.completionPrompt(request).isBlank(), "prompt must not be empty");
        }, (request, id) -> {
            String prompt = Requests.completionPrompt(request);
            String modelId = Requests.modelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamCompletion(exchange, request, prompt, modelId, id);
            } else {
                GenerationResult result = GENERATION.completion(request, prompt, Sinks.NONE); // non-streaming
                respond(exchange, result, OpenAiSchema.completionResponse(id, modelId, result));
            }
        });
    }

    private static void handleResponse(HttpExchange exchange, LLMOptions options) throws IOException {
        handleGenerationPost(exchange, "resp-", request -> {
            Requests.normalizeResponse(request);
            Validation.validateGenerationParams(request, options);
            LLMOptions.require(!Requests.responseInputMessages(request).isEmpty(), "input must not be empty");
        }, (request, id) -> {
            List<Object> messages = Requests.responseInputMessages(request);
            String modelId = Requests.modelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamResponse(exchange, request, messages, modelId, id);
            } else {
                GenerationResult result = GENERATION.chat(request, messages, Sinks.NONE); // non-streaming, no tools
                respond(exchange, result, OpenAiSchema.responseResponse(id, modelId, result));
            }
        });
    }

    private static void streamChatCompletion(HttpExchange exchange, Map<String, Object> request,
                                             List<Object> messages, String modelId, String id) throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(sse, () -> {
                sse.emit(OpenAiSchema.chatCompletionChunk(id, modelId, Map.of("role", "assistant"), null));
                boolean forcedTool = ToolUse.forced(request) != null;
                boolean hasTools = ToolUse.offered(request);
                OpenAiSchema.Usage usage = forcedTool ? null : new OpenAiSchema.Usage();
                // A forced tool call streams no content (the turn is seeded straight into the
                // tool-call block); otherwise content and reasoning stream live. Tool-call body
                // text always goes to a discard sink so it never leaks into the content stream —
                // the calls themselves are parsed from the result and emitted once below.
                Consumer<String> contentSink = forcedTool ? DISCARD
                        : deltaSink(sse, usage, t -> OpenAiSchema.chatCompletionChunk(id, modelId, Map.of("content", t), null));
                Consumer<String> reasoningSink = forcedTool ? null
                        : deltaSink(sse, usage, t -> OpenAiSchema.chatCompletionChunk(id, modelId, Map.of("reasoning_content", t), null));
                Consumer<String> toolCallSink = hasTools || forcedTool ? DISCARD : null;
                GenerationResult result = GENERATION.chat(request, messages, new Sinks(contentSink, reasoningSink, toolCallSink, usage));
                if (!result.toolCalls().isEmpty()) {
                    sse.emit(OpenAiSchema.chatCompletionChunk(id, modelId, Map.of("tool_calls", ToolCalls.toolCallDeltas(result.toolCalls())), null));
                }
                endStream(sse, request, result,
                        OpenAiSchema.chatCompletionChunk(id, modelId, Map.of(), result.finishReason()),
                        OpenAiSchema.chatCompletionChunk(id, modelId, Map.of(), null));
            });
        }
    }

    /** Final stream sequence shared by chat and completions: finish chunk carrying usage, the
     *  stream_options usage-only chunk when requested, then [DONE]. */
    private static void endStream(Sse.Stream sse, Map<String, Object> request, GenerationResult result,
                                  Map<String, Object> finalChunk, Map<String, Object> usageOnlyChunk) {
        Map<String, Object> usage = OpenAiSchema.usage(result);
        finalChunk.put("usage", usage);
        sse.emit(finalChunk);
        if (includeUsage(request)) {
            usageOnlyChunk.put("choices", List.of());
            usageOnlyChunk.put("usage", usage);
            sse.emit(usageOnlyChunk);
        }
        sse.done();
    }

    /** OpenAI stream_options: {"include_usage": true} requests an extra usage-only chunk after the final chunk. */
    @SuppressWarnings("unchecked")
    private static boolean includeUsage(Map<String, Object> request) {
        return request.get("stream_options") instanceof Map<?, ?> so && Boolean.TRUE.equals(((Map<String, Object>) so).get("include_usage"));
    }

    private static void streamCompletion(HttpExchange exchange, Map<String, Object> request,
                                         String prompt, String modelId, String id) throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(sse, () -> {
                OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                Consumer<String> sink = deltaSink(sse, usage, t -> OpenAiSchema.completionChunk(id, modelId, t, null));
                GenerationResult result = GENERATION.completion(request, prompt, Sinks.text(sink, usage));
                endStream(sse, request, result,
                        OpenAiSchema.completionChunk(id, modelId, "", result.finishReason()),
                        OpenAiSchema.completionChunk(id, modelId, "", null));
            });
        }
    }

    private static void streamResponse(HttpExchange exchange, Map<String, Object> request,
                                       List<Object> messages, String modelId, String id) throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(sse, () -> {
                String itemId = "msg_" + id;
                sse.emit("response.created", Map.of(
                        "type", "response.created",
                        "response", OpenAiSchema.responseEnvelope(id, modelId, "in_progress", List.of(), null)));
                sse.emit("response.output_item.added", Map.of(
                        "type", "response.output_item.added",
                        "output_index", 0,
                        "item", OpenAiSchema.responseMessageItem(itemId, "in_progress", "")));
                OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                Consumer<String> sink = deltaSink(sse, usage, "response.output_text.delta",
                        t -> OpenAiSchema.responseTextDelta(itemId, t));
                GenerationResult result = GENERATION.chat(request, messages, Sinks.text(sink, usage));
                sse.emit("response.output_text.done", Map.of(
                        "type", "response.output_text.done",
                        "item_id", itemId,
                        "output_index", 0,
                        "content_index", 0,
                        "text", result.text()));
                sse.emit("response.output_item.done", Map.of(
                        "type", "response.output_item.done",
                        "output_index", 0,
                        "item", OpenAiSchema.responseMessageItem(itemId, "completed", result.text())));
                sse.emit("response.completed", Map.of(
                        "type", "response.completed",
                        "response", OpenAiSchema.responseResponse(id, modelId, result)));
                sse.done();
            });
        }
    }

    /** A no-op sink: consumes generated text without emitting it. Used to suppress a stream
     *  (a forced tool call streams no content) and to route tool-call body text away from the
     *  content stream — a non-null tool-call sink keeps it out of {@code content}. */
    private static final Consumer<String> DISCARD = text -> {};

    /** A streaming text sink: each chunk of generated text becomes one {@code data:} SSE frame
     *  built by {@code chunkOf}, with running usage attached when tracked. */
    private static Consumer<String> deltaSink(Sse.Stream sse, OpenAiSchema.Usage usage,
                                              Function<String, Map<String, Object>> chunkOf) {
        return deltaSink(sse, usage, null, chunkOf);
    }

    /** As {@link #deltaSink(Sse.Stream, OpenAiSchema.Usage, Function)}, but emitted as a named SSE event (the
     *  Responses API) when {@code event} is non-null. */
    private static Consumer<String> deltaSink(Sse.Stream sse, OpenAiSchema.Usage usage, String event,
                                              Function<String, Map<String, Object>> chunkOf) {
        return text -> {
            Map<String, Object> chunk = chunkOf.apply(text);
            if (usage != null) chunk.put("usage", OpenAiSchema.chunkUsage(usage));
            if (event == null) sse.emit(chunk);
            else sse.emit(event, chunk);
        };
    }

    /** Prometheus text exposition (llama.cpp-style /metrics): request/token totals, queue and
     *  worker gauges, prompt-cache stats. */
    private static void handleMetrics(HttpExchange exchange) throws IOException {
        if (Http.preamble(exchange)) return;
        if (!exchange.getRequestURI().getPath().equals("/metrics")) {
            Http.sendError(exchange, 404, "Not found");
            return;
        }
        if (Http.requireMethod(exchange, "GET")) return;
        Http.sendText(exchange, 200, Metrics.CONTENT_TYPE, Metrics.exposition(WORKER, GENERATION.cache()));
    }

    private static void setTimingHeader(HttpExchange exchange, GenerationResult result) {
        exchange.getResponseHeaders().set("X-LFM2-Timing", JsonCodec.stringify(OpenAiSchema.timings(result)));
    }

    /** Non-streaming reply: attach the timing header, then send the schema body as JSON. */
    private static void respond(HttpExchange exchange, GenerationResult result, Object body) throws IOException {
        setTimingHeader(exchange, result);
        Http.sendJson(exchange, 200, body);
    }

    private static final Worker WORKER = new Worker();
    private static Generation GENERATION;

    /** Enqueues the request for the generation worker (FIFO) and waits for it to finish;
     *  rejects with 503 + Retry-After when the queue is full. */
    private static void runQueued(HttpExchange exchange, Runnable work) throws IOException {
        if (!WORKER.submitAndWait(work)) {
            exchange.getResponseHeaders().set("Retry-After", String.valueOf(Worker.retryAfterSeconds()));
            Http.sendError(exchange, 503, "Server busy: " + RuntimeFlags.SERVER_QUEUE + " requests already queued");
            return;
        }
        // a job that finished without ever answering (escaped exception) must not hang the client
        if (exchange.getResponseCode() == -1) {
            Http.sendErrorQuietly(exchange, 500, "Internal server error");
        }
    }
}
