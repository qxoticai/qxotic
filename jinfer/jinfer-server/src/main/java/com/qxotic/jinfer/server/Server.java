package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.telemetry.InferenceEvent;
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

/**
 * The OpenAI-compatible HTTP server, and the module's sole public entry point: {@link #start}. The
 * transport/protocol layer — it registers the routes (/v1/chat/completions, /v1/completions,
 * /v1/responses, /v1/models, /health, /props, /metrics, /tokenize, /detokenize), parses + validates
 * requests, and translates between the wire (JSON, SSE event sequences) and the inference service.
 * The plumbing it builds on lives in {@link Http} (responses/CORS/errors), {@link Sse} (streaming),
 * {@link Worker} (the generation queue), and {@link Metrics}; all inference goes through {@link
 * Generation}. Everything but {@code start} is package-private.
 */
public final class Server {

    private final Worker worker = new Worker();
    private final Generation generation;

    private final String servedModel;

    private Server(LoadedModel<?> model, LLMOptions options) {
        this.generation = new Generation(model, options);
        this.servedModel = options.modelPath().getFileName().toString();
    }

    /**
     * Starts the server for an already-loaded {@code model} and returns the running instance (it
     * serves on its own executor; this call does not block). Host/port come from {@code options};
     * port 0 binds an ephemeral port, readable from {@link HttpServer#getAddress()}. This is the
     * only public API of the module - load a model (jinfer-core), then hand it here to serve it.
     * Each call serves an independent instance (own worker queue, own generation state); the SSE
     * write-stall watchdog and {@link Metrics} counters are deliberately process-wide.
     */
    public static HttpServer start(LoadedModel<?> model, LLMOptions options) throws IOException {
        return new Server(model, options).serve(model, options);
    }

    private HttpServer serve(LoadedModel<?> model, LLMOptions options) throws IOException {
        HttpServer server =
                HttpServer.create(new InetSocketAddress(options.host(), options.port()), 0);
        String servedId = options.modelPath().getFileName().toString();
        Map<String, Object> modelCard =
                Map.of("id", servedId, "object", "model", "created", 0, "owned_by", "jinfer");
        server.createContext(
                "/v1/models",
                exchange -> { // also serves /v1/models/{id} -> card or 404
                    if (Http.preamble(exchange)) return;
                    if (Http.requireMethod(exchange, "GET")) return;
                    String path = exchange.getRequestURI().getPath();
                    if (path.equals("/v1/models")) {
                        Http.sendJson(
                                exchange,
                                200,
                                Map.of("object", "list", "data", List.of(modelCard)));
                    } else if (path.equals("/v1/models/" + servedId)) {
                        Http.sendJson(exchange, 200, modelCard);
                    } else {
                        Http.sendError(
                                exchange,
                                404,
                                "Unknown model: "
                                        + path.substring("/v1/models/".length())
                                        + " (this server serves "
                                        + servedId
                                        + ")");
                    }
                });
        server.createContext(
                "/v1/chat/completions", exchange -> handleChatCompletion(exchange, options));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, options));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, options));
        jsonRoute(
                server,
                "/health",
                null,
                request ->
                        Map.of(
                                "status",
                                "ok",
                                "busy",
                                worker.busy(),
                                "queued",
                                worker.queueDepth()));
        jsonRoute(
                server,
                "/props",
                null,
                request ->
                        Map.of(
                                "model", options.modelPath().getFileName().toString(),
                                "n_ctx", model.model().config().contextLength(),
                                "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                                "n_vocab", model.model().config().vocabularySize(),
                                "prompt_cache", Map.of("enabled", false)));
        Function<Map<String, Object>, Object> tokenize =
                request ->
                        Map.of(
                                "tokens",
                                model.tokenizer()
                                        .encode(
                                                String.valueOf(
                                                        request.getOrDefault("content", ""))));
        Function<Map<String, Object>, Object> detokenize =
                request -> {
                    List<Integer> tokens =
                            request.get("tokens") instanceof List<?> list
                                    ? list.stream().map(v -> ((Number) v).intValue()).toList()
                                    : List.<Integer>of();
                    return Map.of(
                            "content",
                            model.tokenizer().decode(com.qxotic.toknroll.IntSequence.wrap(tokens)));
                };
        jsonRoute(server, "/tokenize", "POST", tokenize); // llama.cpp paths and the
        jsonRoute(server, "/v1/tokenize", "POST", tokenize); // /v1-prefixed aliases
        jsonRoute(server, "/detokenize", "POST", detokenize);
        jsonRoute(server, "/v1/detokenize", "POST", detokenize);
        server.createContext("/metrics", this::handleMetrics);
        server.createContext(
                "/",
                exchange -> {
                    if (Http.preamble(exchange)) return;
                    Http.sendError(exchange, 404, "Not found");
                });
        worker.start();
        Sse.startReaper();
        // bounded pool: handlers only parse/validate and block on the generation queue latch,
        // so a fixed pool also caps the threads slow-loris connections can pin
        server.setExecutor(Executors.newFixedThreadPool(ServerFlags.SERVER_THREADS));
        server.start();
        Runtime.getRuntime()
                .addShutdownHook(
                        new Thread(
                                () -> {
                                    server.stop(1);
                                    // the engine owns states and cache blobs now; free them after
                                    // the listener stops rather than leaving it to process exit
                                    generation.close();
                                }));
        System.out.printf(
                "OpenAI-compatible server listening on http://%s:%d%n",
                options.host(), server.getAddress().getPort());
        return server;
    }

    /**
     * Registers a JSON endpoint with the shared preamble (request log, CORS headers, OPTIONS
     * preflight), an optional method restriction, the parsed JSON body for POST routes, and the
     * uniform 400 error envelope.
     */
    private static void jsonRoute(
            HttpServer server,
            String path,
            String method,
            Function<Map<String, Object>, Object> body) {
        server.createContext(
                path,
                exchange -> {
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
                            request =
                                    Values.asObject(
                                            JsonCodec.parse(
                                                    new String(raw, StandardCharsets.UTF_8)),
                                            "request");
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

    /**
     * Shared scaffold for the generation POST endpoints: preflight/method checks, bounded body
     * read, then parse + validation on the HANDLER thread — malformed requests get an instant 400
     * and never occupy the generation worker, even while it is busy with a long generation. The job
     * runs on the worker via the bounded queue with uniform error handling.
     */
    private void handleGenerationPost(
            HttpExchange exchange,
            String idPrefix,
            Consumer<Map<String, Object>> validator,
            RequestJob job)
            throws IOException {
        if (Http.preamble(exchange)) return;
        if (Http.requireMethod(exchange, "POST")) return;
        byte[] body =
                Http.readBody(
                        exchange); // read on the handler thread: a stalled upload must not block
        // the generation worker
        if (body == null) return;
        Map<String, Object> request;
        try {
            request =
                    Values.asObject(
                            JsonCodec.parse(new String(body, StandardCharsets.UTF_8)), "request");
            validator.accept(request);
        } catch (RuntimeException e) {
            Http.sendError(exchange, 400, Http.errorMessage(e));
            return;
        }
        String id = idPrefix + Long.toUnsignedString(System.nanoTime(), 36);
        runQueued(
                exchange,
                () -> {
                    try {
                        job.run(request, id);
                    } catch (IllegalArgumentException | UnsupportedOperationException e) {
                        // the request is genuinely at fault: a bad parameter, or input this model
                        // cannot frame (media on a text-only model, a shape with no codec)
                        Http.sendErrorQuietly(exchange, 400, Http.errorMessage(e));
                    } catch (IOException e) {
                        System.err.println("client connection lost: " + e);
                    } catch (Throwable t) {
                        // Anything else is OURS. Catching RuntimeException as 400 here reported
                        // server defects as client errors - an NPE came back as a 400 quoting the
                        // null field - so they neither showed up as failures nor were actionable.
                        System.err.println("request " + id + " failed:");
                        t.printStackTrace();
                        Http.sendErrorQuietly(exchange, 500, "Internal server error");
                    }
                });
    }

    private void handleChatCompletion(HttpExchange exchange, LLMOptions options)
            throws IOException {
        handleGenerationPost(
                exchange,
                "chatcmpl-",
                request -> {
                    Validation.validateChatRequest(request);
                    Validation.validateGenerationParams(request, options);
                },
                (request, id) -> {
                    List<Object> messages = Values.asArray(request.get("messages"), "messages");
                    String modelId = Requests.modelId(request, options);
                    if (Values.booleanValue(request.get("stream"), false)) {
                        streamChatCompletion(exchange, request, messages, modelId, id);
                    } else {
                        Reply result =
                                generation.chat(
                                        request, messages, Sinks.NONE); // non-streaming, no tools
                        respond(
                                exchange,
                                result,
                                OpenAiSchema.chatCompletionResponse(id, modelId, result));
                    }
                });
    }

    private void handleCompletion(HttpExchange exchange, LLMOptions options) throws IOException {
        handleGenerationPost(
                exchange,
                "cmpl-",
                request -> {
                    Validation.validateGenerationParams(request, options);
                    LLMOptions.require(
                            !Requests.completionPrompt(request).isBlank(),
                            "prompt must not be empty");
                },
                (request, id) -> {
                    String prompt = Requests.completionPrompt(request);
                    String modelId = Requests.modelId(request, options);
                    if (Values.booleanValue(request.get("stream"), false)) {
                        streamCompletion(exchange, request, prompt, modelId, id);
                    } else {
                        Reply result =
                                generation.completion(request, prompt, Sinks.NONE); // non-streaming
                        respond(
                                exchange,
                                result,
                                OpenAiSchema.completionResponse(id, modelId, result));
                    }
                });
    }

    private void handleResponse(HttpExchange exchange, LLMOptions options) throws IOException {
        handleGenerationPost(
                exchange,
                "resp-",
                request -> {
                    Requests.normalizeResponse(request);
                    Validation.validateGenerationParams(request, options);
                    LLMOptions.require(
                            !Requests.responseInputMessages(request).isEmpty(),
                            "input must not be empty");
                },
                (request, id) -> {
                    List<Object> messages = Requests.responseInputMessages(request);
                    String modelId = Requests.modelId(request, options);
                    if (Values.booleanValue(request.get("stream"), false)) {
                        streamResponse(exchange, request, messages, modelId, id);
                    } else {
                        Reply result =
                                generation.chat(
                                        request, messages, Sinks.NONE); // non-streaming, no tools
                        respond(
                                exchange,
                                result,
                                OpenAiSchema.responseResponse(id, modelId, result));
                    }
                });
    }

    private void streamChatCompletion(
            HttpExchange exchange,
            Map<String, Object> request,
            List<Object> messages,
            String modelId,
            String id)
            throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(
                    sse,
                    () -> {
                        sse.emit(
                                OpenAiSchema.chatCompletionChunk(
                                        id, modelId, Map.of("role", "assistant"), null));
                        // A forced tool call streams no live channels (the turn is seeded
                        // straight into the tool-call block; the calls are parsed from the result
                        // and emitted once below); otherwise content and reasoning stream live.
                        OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                        Sinks sinks =
                                ToolUse.forced(request) != null
                                        ? Sinks.NONE
                                        : new Sinks(
                                                deltaSink(
                                                        sse,
                                                        usage,
                                                        t ->
                                                                OpenAiSchema.chatCompletionChunk(
                                                                        id,
                                                                        modelId,
                                                                        Map.of("content", t),
                                                                        null)),
                                                deltaSink(
                                                        sse,
                                                        usage,
                                                        t ->
                                                                OpenAiSchema.chatCompletionChunk(
                                                                        id,
                                                                        modelId,
                                                                        Map.of(
                                                                                "reasoning_content",
                                                                                t),
                                                                        null)),
                                                usage);
                        Reply result = generation.chat(request, messages, sinks);
                        if (!result.toolCalls().isEmpty()) {
                            sse.emit(
                                    OpenAiSchema.chatCompletionChunk(
                                            id,
                                            modelId,
                                            Map.of(
                                                    "tool_calls",
                                                    ToolCalls.toolCallDeltas(
                                                            ToolCalls.toWire(result.toolCalls()))),
                                            null));
                        }
                        endStream(
                                sse,
                                request,
                                result,
                                OpenAiSchema.chatCompletionChunk(
                                        id, modelId, Map.of(), result.finishReason()),
                                OpenAiSchema.chatCompletionChunk(id, modelId, Map.of(), null));
                    });
        }
    }

    /**
     * Final stream sequence shared by chat and completions: finish chunk carrying usage, the
     * stream_options usage-only chunk when requested, then [DONE].
     */
    private static void endStream(
            Sse.Stream sse,
            Map<String, Object> request,
            Reply result,
            Map<String, Object> finalChunk,
            Map<String, Object> usageOnlyChunk) {
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

    /**
     * OpenAI stream_options: {"include_usage": true} requests an extra usage-only chunk after the
     * final chunk.
     */
    @SuppressWarnings("unchecked")
    private static boolean includeUsage(Map<String, Object> request) {
        return request.get("stream_options") instanceof Map<?, ?> so
                && Boolean.TRUE.equals(((Map<String, Object>) so).get("include_usage"));
    }

    private void streamCompletion(
            HttpExchange exchange,
            Map<String, Object> request,
            String prompt,
            String modelId,
            String id)
            throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(
                    sse,
                    () -> {
                        OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                        Consumer<String> sink =
                                deltaSink(
                                        sse,
                                        usage,
                                        t -> OpenAiSchema.completionChunk(id, modelId, t, null));
                        Reply result =
                                generation.completion(request, prompt, Sinks.text(sink, usage));
                        endStream(
                                sse,
                                request,
                                result,
                                OpenAiSchema.completionChunk(
                                        id, modelId, "", result.finishReason()),
                                OpenAiSchema.completionChunk(id, modelId, "", null));
                    });
        }
    }

    private void streamResponse(
            HttpExchange exchange,
            Map<String, Object> request,
            List<Object> messages,
            String modelId,
            String id)
            throws IOException {
        try (Sse.Stream sse = Sse.begin(exchange)) {
            Sse.guarded(
                    sse,
                    () -> {
                        String itemId = "msg_" + id;
                        sse.emit(
                                "response.created",
                                Map.of(
                                        "type",
                                        "response.created",
                                        "response",
                                        OpenAiSchema.responseEnvelope(
                                                id, modelId, "in_progress", List.of(), null)));
                        sse.emit(
                                "response.output_item.added",
                                Map.of(
                                        "type",
                                        "response.output_item.added",
                                        "output_index",
                                        0,
                                        "item",
                                        OpenAiSchema.responseMessageItem(
                                                itemId, "in_progress", "")));
                        OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                        Consumer<String> sink =
                                deltaSink(
                                        sse,
                                        usage,
                                        "response.output_text.delta",
                                        t -> OpenAiSchema.responseTextDelta(itemId, t));
                        Reply result = generation.chat(request, messages, Sinks.text(sink, usage));
                        sse.emit(
                                "response.output_text.done",
                                Map.of(
                                        "type",
                                        "response.output_text.done",
                                        "item_id",
                                        itemId,
                                        "output_index",
                                        0,
                                        "content_index",
                                        0,
                                        "text",
                                        result.text()));
                        sse.emit(
                                "response.output_item.done",
                                Map.of(
                                        "type",
                                        "response.output_item.done",
                                        "output_index",
                                        0,
                                        "item",
                                        OpenAiSchema.responseMessageItem(
                                                itemId, "completed", result.text())));
                        sse.emit(
                                "response.completed",
                                Map.of(
                                        "type",
                                        "response.completed",
                                        "response",
                                        OpenAiSchema.responseResponse(id, modelId, result)));
                        sse.done();
                    });
        }
    }

    /**
     * A streaming text sink: each chunk of generated text becomes one {@code data:} SSE frame built
     * by {@code chunkOf}, with running usage attached when tracked.
     */
    private static Consumer<String> deltaSink(
            Sse.Stream sse,
            OpenAiSchema.Usage usage,
            Function<String, Map<String, Object>> chunkOf) {
        return deltaSink(sse, usage, null, chunkOf);
    }

    /**
     * As {@link #deltaSink(Sse.Stream, OpenAiSchema.Usage, Function)}, but emitted as a named SSE
     * event (the Responses API) when {@code event} is non-null.
     */
    private static Consumer<String> deltaSink(
            Sse.Stream sse,
            OpenAiSchema.Usage usage,
            String event,
            Function<String, Map<String, Object>> chunkOf) {
        return text -> {
            Map<String, Object> chunk = chunkOf.apply(text);
            if (usage != null) chunk.put("usage", OpenAiSchema.chunkUsage(usage));
            if (event == null) sse.emit(chunk);
            else sse.emit(event, chunk);
        };
    }

    /**
     * Prometheus text exposition (llama.cpp-style /metrics): request/token totals, queue and worker
     * gauges, prompt-cache stats.
     */
    private void handleMetrics(HttpExchange exchange) throws IOException {
        if (Http.preamble(exchange)) return;
        if (!exchange.getRequestURI().getPath().equals("/metrics")) {
            Http.sendError(exchange, 404, "Not found");
            return;
        }
        if (Http.requireMethod(exchange, "GET")) return;
        Http.sendText(exchange, 200, Metrics.CONTENT_TYPE, Metrics.exposition(worker));
    }

    private static void setTimingHeader(HttpExchange exchange, Reply result) {
        exchange.getResponseHeaders()
                .set("X-Jinfer-Timing", JsonCodec.stringify(OpenAiSchema.timings(result)));
    }

    /** Non-streaming reply: attach the timing header, then send the schema body as JSON. */
    private static void respond(HttpExchange exchange, Reply result, Object body)
            throws IOException {
        setTimingHeader(exchange, result);
        Http.sendJson(exchange, 200, body);
    }

    /**
     * Enqueues the request for the generation worker (FIFO) and waits for it to finish; rejects
     * with 503 + Retry-After when the queue is full.
     */
    private void runQueued(HttpExchange exchange, Runnable work) throws IOException {
        if (!worker.submitAndWait(work)) {
            // a shed request never reaches the engine, so nothing else would report it - and going
            // silent exactly when the server is saturated is the worst possible time to do so
            InferenceEvent rejected =
                    InferenceEvent.started(servedModel, InferenceEvent.CHAT, InferenceEvent.TEXT);
            rejected.errorType = "queue-full";
            rejected.end();
            rejected.commit();
            exchange.getResponseHeaders()
                    .set("Retry-After", String.valueOf(Worker.retryAfterSeconds()));
            Http.sendError(
                    exchange,
                    503,
                    "Server busy: " + ServerFlags.SERVER_QUEUE + " requests already queued");
            return;
        }
        // a job that finished without ever answering (escaped exception) must not hang the client
        if (exchange.getResponseCode() == -1) {
            Http.sendErrorQuietly(exchange, 500, "Internal server error");
        }
    }
}
