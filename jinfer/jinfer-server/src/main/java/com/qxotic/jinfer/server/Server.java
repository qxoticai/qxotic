package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Sampling;
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
 * Generation}. Everything but {@code start} and the {@link Running} handle it returns is
 * package-private.
 */
public final class Server {

    private final Worker worker;
    private final ServerConfig config;
    private final Metrics metrics = new Metrics();
    private final Generation generation;

    private final String servedModel;

    private Server(LoadedModel<?> model, ServerConfig config) {
        this.generation = new Generation(model, config, metrics);
        this.servedModel = config.modelName();
        this.worker = new Worker(config.limits().queueCapacity());
        this.config = config;
    }

    /**
     * A running server. {@link #close} stops the listener and then frees the engine's states and
     * cache blobs - an embedder that started a server can actually shut it down; stopping the raw
     * {@link HttpServer} alone would leak the engine until process exit.
     */
    public static final class Running implements AutoCloseable {
        private final HttpServer http;
        private final Generation generation;

        private Running(HttpServer http, Generation generation) {
            this.http = http;
            this.generation = generation;
        }

        /** The bound address; port 0 in the options binds an ephemeral port readable here. */
        public InetSocketAddress address() {
            return http.getAddress();
        }

        /** Listener first (no new requests), then the engine; idempotent via the engine's close. */
        @Override
        public void close() {
            http.stop(1);
            // the fixed handler pool is non-daemon and stop() does not touch it - without this an
            // embedder's JVM never exits
            if (http.getExecutor() instanceof java.util.concurrent.ExecutorService pool) {
                pool.shutdownNow();
            }
            generation.close();
        }
    }

    /**
     * Starts the server for an already-loaded {@code model} and returns the running instance (it
     * serves on its own executor; this call does not block). This is the only public API of the
     * module - load a model (jinfer-core), then hand it here to serve it. Each call serves an
     * independent instance: own worker queue, own generation state, own {@link Metrics}; only the
     * SSE write-stall watchdog is process-wide.
     *
     * <p>Prints NOTHING, reads no system properties or environment, and installs no shutdown hook:
     * what a start is worth announcing, where its settings come from, and who owns the process's
     * exit are all the caller's to decide - see {@link Main} for the CLI's answers.
     */
    public static Running start(LoadedModel<?> model, ServerConfig config) throws IOException {
        return new Server(model, config).serve(model, config);
    }

    /** A float for humans and JSON: {@code 0.2}, not {@code 0.20000000298023224}. */
    private static double trim(float value) {
        return Math.round(value * 1000.0) / 1000.0;
    }

    /**
     * The block tree's health for {@code /props}, in the block-cache vocabulary (the June-era
     * {@code warm_tokens}/{@code dense_hits} keys described machinery the 2026-07 redesign
     * deleted): sizes now, counters cumulative since the engine was built.
     */
    private Map<String, Object> promptCacheProps() {
        var sample = generation.cacheSample();
        Map<String, Object> props = new java.util.LinkedHashMap<>();
        props.put("enabled", generation.blockCaching());
        props.put("hot_sessions", sample.hotSessions());
        props.put("hot_hits", sample.hotHits());
        props.put("allocations", sample.statesAllocated());
        props.put("snapshot_bytes", sample.snapshotBytes());
        props.put("blocks", sample.blocks());
        props.put("bytes", sample.bytes());
        props.put("budget_bytes", sample.budgetBytes());
        props.put("hits", sample.hits());
        props.put("misses", sample.misses());
        props.put("evictions", sample.evictions());
        props.put("refusals", sample.refusals());
        return props;
    }

    private Running serve(LoadedModel<?> model, ServerConfig config) throws IOException {
        Sampling sampling = config.defaults().sampling();
        HttpServer server = HttpServer.create(config.bind(), 0);
        String servedId = servedModel;
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
                        // contexts match by PREFIX, so this also catches /v1/modelsXYZ - which
                        // substring("/v1/models/".length()) mangled into "YZ" (or "", or an
                        // exception) because it assumed a separator that is not there
                        String requested =
                                path.startsWith("/v1/models/")
                                        ? path.substring("/v1/models/".length())
                                        : path.substring("/v1/models".length());
                        Http.sendError(
                                exchange,
                                404,
                                "Unknown model: "
                                        + requested
                                        + " (this server serves "
                                        + servedId
                                        + ")");
                    }
                });
        server.createContext(
                "/v1/chat/completions", exchange -> handleChatCompletion(exchange, config));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, config));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, config));
        jsonRoute(
                server,
                "/health",
                "GET",
                request ->
                        Map.of("status", "ok", "busy", worker.busy(), "queued", worker.queued()));
        jsonRoute(
                server,
                "/props",
                "GET",
                request ->
                        Map.of(
                                "model", config.modelName(),
                                // what a request may actually use, not what the model was
                                // trained for: a client sizing to the latter gets refused
                                "n_ctx", config.cache().contextCapacity(),
                                "n_ctx_train", model.model().config().contextLength(),
                                "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                                "n_vocab", model.model().config().vocabularySize(),
                                "sampling",
                                        Map.of(
                                                "temperature", trim(sampling.temperature()),
                                                "top_p", trim(sampling.topP()),
                                                "top_k", sampling.topK(),
                                                "min_p", trim(sampling.minP())),
                                "prompt_cache", promptCacheProps()));
        Function<Map<String, Object>, Object> tokenize =
                request -> {
                    // present-but-empty is a legitimate question (the answer is []); ABSENT is a
                    // mistake, and so is a non-string - both used to return [] with a 200, so a
                    // client that sent {"text": ...} or a null got a confident wrong answer
                    Validation.require(
                            request.containsKey("content"),
                            "Invalid argument: content is required");
                    Validation.require(
                            request.get("content") instanceof String,
                            "Invalid argument: content must be a string");
                    return Map.of(
                            "tokens", model.tokenizer().encode((String) request.get("content")));
                };
        int vocabularySize = model.tokenizer().vocabulary().size();
        Function<Map<String, Object>, Object> detokenize =
                request -> {
                    Validation.require(
                            request.containsKey("tokens"), "Invalid argument: tokens is required");
                    List<Object> values = Values.asArray(request.get("tokens"), "tokens");
                    int[] tokens = new int[values.size()];
                    for (int i = 0; i < tokens.length; i++) {
                        Object value = values.get(i);
                        // every element checked by hand: the stream cast this used to do turned a
                        // string element into a ClassCastException, which the route reported as a
                        // 400 quoting raw JVM text ("class java.lang.String cannot be cast to..."),
                        // and a float element into a SILENT truncation
                        Validation.require(
                                value instanceof Number number
                                        && number.doubleValue() == Math.rint(number.doubleValue()),
                                "Invalid argument: tokens[%d] must be an integer",
                                i);
                        long id = ((Number) value).longValue();
                        Validation.require(
                                0 <= id && id < vocabularySize,
                                "Invalid argument: tokens[%d] is %d, outside this model's"
                                        + " vocabulary [0, %d)",
                                i,
                                id,
                                vocabularySize);
                        tokens[i] = (int) id;
                    }
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
        server.setExecutor(Executors.newFixedThreadPool(config.limits().threads()));
        server.start();
        // no shutdown hook here: one per start() is a hook the embedder cannot unregister, and it
        // pins the engine for the life of the JVM. The CLI, which never closes the handle,
        // registers its own; every other caller closes the handle it was given.
        return new Running(server, generation);
    }

    /**
     * Registers a JSON endpoint with the shared preamble (request log, CORS headers, OPTIONS
     * preflight), an optional method restriction, the parsed JSON body for POST routes, and the
     * uniform 400 error envelope.
     */
    private void jsonRoute(
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
                        byte[] raw = Http.readBody(exchange, config.limits().maxBodyBytes());
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
                    } catch (IllegalArgumentException | UnsupportedOperationException e) {
                        // the SAME rule the generation endpoints use: only the two types a
                        // validator throws are the client's fault. A blanket RuntimeException ->
                        // 400 here told clients their request was malformed whenever this server
                        // had a defect, and echoed the JVM's own text while doing it
                        Http.sendError(exchange, 400, Http.errorMessage(e));
                    } catch (RuntimeException e) {
                        Log.LOG.log(
                                System.Logger.Level.ERROR, "unhandled fault serving " + path, e);
                        Http.sendErrorQuietly(exchange, 500, "Internal server error");
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
            String path,
            String idPrefix,
            Consumer<Map<String, Object>> validator,
            RequestJob job)
            throws IOException {
        if (Http.preamble(exchange)) return;
        // contexts match by PREFIX: without this, POST /v1/chat/completionsXYZ and
        // /v1/completions/foo were served as if they were the canonical endpoint. jsonRoute and
        // /metrics already checked; these three did not.
        if (!exchange.getRequestURI().getPath().equals(path)) {
            Http.sendError(exchange, 404, "Not found");
            return;
        }
        if (Http.requireMethod(exchange, "POST")) return;
        // read on the handler thread: a stalled upload must not block the generation worker
        byte[] body = Http.readBody(exchange, config.limits().maxBodyBytes());
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
                        Log.LOG.log(System.Logger.Level.DEBUG, "client connection lost", e);
                    } catch (Throwable t) {
                        // Anything else is OURS. Catching RuntimeException as 400 here reported
                        // server defects as client errors - an NPE came back as a 400 quoting the
                        // null field - so they neither showed up as failures nor were actionable.
                        Log.LOG.log(System.Logger.Level.ERROR, "request " + id + " failed", t);
                        Http.sendErrorQuietly(exchange, 500, "Internal server error");
                    }
                });
    }

    private void handleChatCompletion(HttpExchange exchange, ServerConfig config)
            throws IOException {
        handleGenerationPost(
                exchange,
                "/v1/chat/completions",
                "chatcmpl-",
                request -> {
                    Validation.validateChatRequest(request);
                    Validation.validateGenerationParams(request, config);
                },
                (request, id) -> {
                    List<Object> messages = Values.asArray(request.get("messages"), "messages");
                    String modelId = Requests.modelId(request, config);
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

    private void handleCompletion(HttpExchange exchange, ServerConfig config) throws IOException {
        handleGenerationPost(
                exchange,
                "/v1/completions",
                "cmpl-",
                request -> {
                    Validation.validateGenerationParams(request, config);
                    Validation.require(
                            !Requests.completionPrompt(request).isBlank(),
                            "prompt must not be empty");
                },
                (request, id) -> {
                    String prompt = Requests.completionPrompt(request);
                    String modelId = Requests.modelId(request, config);
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

    private void handleResponse(HttpExchange exchange, ServerConfig config) throws IOException {
        handleGenerationPost(
                exchange,
                "/v1/responses",
                "resp-",
                request -> {
                    Requests.normalizeResponse(request);
                    Validation.validateGenerationParams(request, config);
                    List<Object> folded = Requests.responseInputMessages(request);
                    Validation.require(!folded.isEmpty(), "input must not be empty");
                    // the SAME shape rules as chat, over the folded turns: roles, substance,
                    // response_format and tool declarations. This endpoint used to skip them
                    // entirely, so a bad role or a malformed tool reached the engine here and
                    // was refused there - or not at all.
                    Map<String, Object> asChat = new java.util.HashMap<>(request);
                    asChat.put("messages", folded);
                    Validation.validateChatRequest(asChat);
                },
                (request, id) -> {
                    List<Object> messages = Requests.responseInputMessages(request);
                    String modelId = Requests.modelId(request, config);
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
        try (Sse.Stream sse = Sse.begin(exchange, config.limits().writeTimeout())) {
            Sse.guarded(
                    sse,
                    () -> {
                        sse.emit(
                                OpenAiSchema.chatCompletionChunk(
                                        id, modelId, Map.of("role", "assistant"), null));
                        // A forced tool call streams no live channels (the turn is seeded
                        // straight into the tool-call block; the calls are parsed from the result
                        // and emitted once below); otherwise content and reasoning stream live.
                        // Ask GENERATION whether forcing really happens - a model with no call
                        // seed ignores tool_choice and generates ordinary prose, which silently
                        // went nowhere while this asked ToolUse.forced(request) on its own.
                        Sinks sinks =
                                generation.forcedTool(request) != null
                                        ? Sinks.NONE
                                        : new Sinks(
                                                deltaSink(
                                                        sse,
                                                        t ->
                                                                OpenAiSchema.chatCompletionChunk(
                                                                        id,
                                                                        modelId,
                                                                        Map.of("content", t),
                                                                        null)),
                                                deltaSink(
                                                        sse,
                                                        t ->
                                                                OpenAiSchema.chatCompletionChunk(
                                                                        id,
                                                                        modelId,
                                                                        Map.of(
                                                                                "reasoning_content",
                                                                                t),
                                                                        null)));
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
        try (Sse.Stream sse = Sse.begin(exchange, config.limits().writeTimeout())) {
            Sse.guarded(
                    sse,
                    () -> {
                        Consumer<String> sink =
                                deltaSink(
                                        sse,
                                        t -> OpenAiSchema.completionChunk(id, modelId, t, null));
                        Reply result = generation.completion(request, prompt, Sinks.text(sink));
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
        try (Sse.Stream sse = Sse.begin(exchange, config.limits().writeTimeout())) {
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
                        Consumer<String> sink =
                                deltaSink(
                                        sse,
                                        "response.output_text.delta",
                                        t -> OpenAiSchema.responseTextDelta(itemId, t));
                        Reply result = generation.chat(request, messages, Sinks.text(sink));
                        // a tool-call turn produced no text, so there is no text item to finish -
                        // emitting one anyway announced a COMPLETED message holding "" and left
                        // the call visible only in the final envelope
                        if (result.toolCalls().isEmpty()) {
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
                        }
                        List<Map<String, Object>> items =
                                OpenAiSchema.responseOutputItems(id, result);
                        for (int i = 0; i < items.size(); i++) {
                            sse.emit(
                                    "response.output_item.done",
                                    Map.of(
                                            "type",
                                            "response.output_item.done",
                                            "output_index",
                                            i,
                                            "item",
                                            items.get(i)));
                        }
                        sse.emit(
                                "response.completed",
                                Map.of(
                                        "type",
                                        "response.completed",
                                        "response",
                                        // the SAME items just emitted, not a rebuild: see the
                                        // overload's note on clock-minted call ids
                                        OpenAiSchema.responseResponse(id, modelId, result, items)));
                        sse.done();
                    });
        }
    }

    /**
     * A streaming text sink: each chunk of generated text becomes one {@code data:} SSE frame built
     * by {@code chunkOf}, with running usage attached when tracked.
     */
    private static Consumer<String> deltaSink(
            Sse.Stream sse, Function<String, Map<String, Object>> chunkOf) {
        return deltaSink(sse, null, chunkOf);
    }

    /**
     * As {@link #deltaSink(Sse.Stream, Function)}, but emitted as a named SSE event (the Responses
     * API) when {@code event} is non-null.
     */
    private static Consumer<String> deltaSink(
            Sse.Stream sse, String event, Function<String, Map<String, Object>> chunkOf) {
        return text -> {
            Map<String, Object> chunk = chunkOf.apply(text);
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
        Http.sendText(exchange, 200, Metrics.CONTENT_TYPE, metrics.exposition(worker));
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
                    .set("Retry-After", String.valueOf(config.limits().retryAfterSeconds()));
            Http.sendError(
                    exchange,
                    503,
                    "Server busy: " + config.limits().queueCapacity() + " requests already queued");
            return;
        }
        // a job that finished without ever answering (escaped exception) must not hang the client
        if (exchange.getResponseCode() == -1) {
            Http.sendErrorQuietly(exchange, 500, "Internal server error");
        }
    }
}
