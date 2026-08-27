package com.qxotic.jinfer.server;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.telemetry.InferenceEvent;
import com.qxotic.toknroll.IntSequence;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The OpenAI-compatible HTTP transport over a caller-configured {@link ChatEngine}. The
 * transport/protocol layer — it registers the routes (/v1/chat/completions, /v1/completions,
 * /v1/responses, /v1/models, /health, /props, /metrics, /tokenize, /detokenize), parses + validates
 * requests, and translates between the wire (JSON, SSE event sequences) and the inference service.
 * The plumbing it builds on lives in {@link Http} (responses/CORS/errors), {@link Sse} (streaming),
 * {@link Worker} (the generation queue), and {@link Metrics}; all inference goes through {@link
 * Generation}.
 */
public final class Server {

    private final Worker worker;
    private final ServerConfig config;
    private final Metrics metrics = new Metrics();
    private final Generation generation;

    private final String servedModel;

    private Server(ChatEngine engine, ServerConfig config) {
        this.generation = new Generation(engine, config, metrics);
        this.servedModel = engine.modelName();
        this.worker = new Worker(config.limits().queueCapacity());
        this.config = config;
    }

    /**
     * A running transport. The caller retains ownership of the engine and may reuse it after this
     * handle closes.
     */
    public static final class Running implements AutoCloseable {
        private final HttpServer http;
        private final Worker worker;
        private final int stopDelaySeconds;
        private final CountDownLatch stopped = new CountDownLatch(1);
        private volatile boolean closed;

        private Running(HttpServer http, Worker worker, int stopDelaySeconds) {
            this.http = http;
            this.worker = worker;
            this.stopDelaySeconds = stopDelaySeconds;
        }

        /** The bound address; port 0 in the options binds an ephemeral port readable here. */
        public InetSocketAddress address() {
            return http.getAddress();
        }

        /** Blocks until {@link #close()} is called. Useful to process-owning launchers. */
        public void await() throws InterruptedException {
            stopped.await();
        }

        /** Stops admission, releases queued callers, and terminates handler threads. Idempotent. */
        @Override
        public synchronized void close() {
            if (closed) return;
            closed = true;
            // queued callers first: their handlers write the 503 while the server still delivers
            // it. Stopping HTTP first parked them for the whole stop delay and then cut the
            // connections, so the "shutting down" answer went to a closed socket.
            worker.close();
            http.stop(stopDelaySeconds);
            // the fixed handler pool is non-daemon and stop() does not touch it - without this an
            // embedder's JVM never exits
            if (http.getExecutor() instanceof ExecutorService pool) {
                pool.shutdownNow();
            }
            stopped.countDown();
        }
    }

    /**
     * Starts serving an already-configured engine without reloading it or changing its cache or MTP
     * policy. The call does not block. Each server has its own queue and metrics; the engine stays
     * caller-owned.
     *
     * <p>Prints NOTHING, reads no system properties or environment, and installs no shutdown hook:
     * what a start is worth announcing, where its settings come from, and who owns the process's
     * exit are all the caller's to decide.
     */
    public static Running start(ChatEngine engine, ServerConfig config) throws IOException {
        if (engine == null) throw new IllegalArgumentException("engine is required");
        if (config == null) throw new IllegalArgumentException("config is required");
        return new Server(engine, config).serve(engine, config);
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
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("retained_sessions", sample.retainedSessions());
        props.put("retained_session_limit", sample.retainedSessionLimit());
        props.put("session_hits", sample.sessionHits());
        props.put("state_allocations", sample.stateAllocations());
        props.put("session_snapshot_bytes", sample.sessionSnapshotBytes());
        props.put("blocks_enabled", generation.blockCaching());
        props.put("blocks", sample.blocks());
        props.put("bytes", sample.bytes());
        props.put("budget_bytes", sample.budgetBytes());
        props.put("block_hits", sample.blockHits());
        props.put("block_misses", sample.blockMisses());
        props.put("block_evictions", sample.blockEvictions());
        props.put("block_discards", sample.blockDiscards());
        props.put("block_refusals", sample.blockRefusals());
        return props;
    }

    /** The projected-media cache's health for {@code /props} - same counters /metrics exports. */
    private Map<String, Object> mediaCacheProps() {
        var sample = generation.mediaCacheSample();
        Map<String, Object> props = new LinkedHashMap<>();
        props.put("entries", sample.entries());
        props.put("bytes", sample.bytes());
        props.put("budget_bytes", sample.budgetBytes());
        props.put("hits", sample.hits());
        props.put("misses", sample.misses());
        props.put("refusals", sample.refusals());
        return props;
    }

    private Running serve(ChatEngine engine, ServerConfig config) throws IOException {
        LoadedModel<?> model = engine.loaded();
        Sampling sampling = generation.defaults();
        HttpServer server = HttpServer.create(config.bind(), 0);
        String servedId = servedModel;
        Map<String, Object> modelCard =
                Map.of("id", servedId, "object", "model", "created", 0, "owned_by", "jinfer");
        server.createContext(
                "/v1/models",
                exchange -> { // also serves /v1/models/{id} -> card or 404
                    if (Http.preamble(exchange, config.access())) return;
                    if (Http.requireMethod(exchange, "GET")) return;
                    String path = exchange.getRequestURI().getPath();
                    if (path.equals("/v1/models")) {
                        Http.sendJson(
                                exchange,
                                200,
                                Map.of("object", "list", "data", List.of(modelCard)));
                    } else if (path.equals("/v1/models/" + servedId)) {
                        Http.sendJson(exchange, 200, modelCard);
                    } else if (path.startsWith("/v1/models/")) {
                        Http.sendError(
                                exchange,
                                404,
                                "Unknown model: "
                                        + path.substring("/v1/models/".length())
                                        + " (this server serves "
                                        + servedId
                                        + ")");
                    } else {
                        // contexts match by PREFIX, so /v1/modelsXYZ lands here too. That is a
                        // wrong PATH, not an unknown model - and the substring that called it one
                        // assumed a separator that is not there, reporting "XYZ" as "YZ".
                        Http.sendError(exchange, 404, "Not found");
                    }
                });
        server.createContext(
                "/v1/chat/completions", exchange -> handleChatCompletion(exchange, config));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, config));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, config));
        // liveness probes carry no key: /health is open (llama.cpp's is too); it says nothing
        // a probe should not know (up, busy, queue depth)
        jsonRoute(
                server,
                "/health",
                "GET",
                request -> Map.of("status", "ok", "busy", worker.busy(), "queued", worker.queued()),
                new ServerConfig.Access(null, config.access().allowedOrigins()));
        jsonRoute(
                server,
                "/props",
                "GET",
                request ->
                        Map.of(
                                "model", servedModel,
                                // what a request may actually use, not what the model was
                                // trained for: a client sizing to the latter gets refused
                                "n_ctx", engine.contextCapacity(),
                                "n_ctx_train", model.model().configuration().contextLength(),
                                "n_vocab", model.model().configuration().vocabularySize(),
                                "speculation",
                                        Map.of(
                                                "ready", engine.speculationReady(),
                                                "enabled",
                                                        engine.speculationReady()
                                                                && engine.speculationDepth() > 0,
                                                "depth", engine.speculationDepth()),
                                "sampling",
                                        Map.of(
                                                "temperature", trim(sampling.temperature()),
                                                "top_p", trim(sampling.topP()),
                                                "top_k", sampling.topK(),
                                                "min_p", trim(sampling.minP())),
                                "prompt_cache", promptCacheProps(),
                                "media_cache", mediaCacheProps()));
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
                    return Map.of("content", model.tokenizer().decode(IntSequence.wrap(tokens)));
                };
        jsonRoute(server, "/tokenize", "POST", tokenize); // llama.cpp paths and the
        jsonRoute(server, "/v1/tokenize", "POST", tokenize); // /v1-prefixed aliases
        jsonRoute(server, "/detokenize", "POST", detokenize);
        jsonRoute(server, "/v1/detokenize", "POST", detokenize);
        server.createContext("/metrics", this::handleMetrics);
        server.createContext(
                "/",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    Http.sendError(exchange, 404, "Not found");
                });
        worker.start();
        Sse.startReaper();
        server.setExecutor(requestExecutor(config.limits().threads()));
        server.start();
        // no shutdown hook here: one per start() is a hook the embedder cannot unregister, and it
        // pins the engine for the life of the JVM. The CLI, which never closes the handle,
        // registers its own; every other caller closes the handle it was given.
        long stopDelay = Math.ceilDiv(config.limits().shutdownTimeout().toNanos(), 1_000_000_000L);
        return new Running(server, worker, (int) Math.min(Integer.MAX_VALUE, stopDelay));
    }

    /** Bounds both active handlers and requests waiting to enter one. */
    static ExecutorService requestExecutor(int threads) {
        return new ThreadPoolExecutor(
                threads,
                threads,
                0,
                TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(threads),
                new ThreadPoolExecutor.AbortPolicy());
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
        jsonRoute(server, path, method, body, config.access());
    }

    private void jsonRoute(
            HttpServer server,
            String path,
            String method,
            Function<Map<String, Object>, Object> body,
            ServerConfig.Access access) {
        server.createContext(
                path,
                exchange -> {
                    if (Http.preamble(exchange, access)) return;
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
                            request = Values.asObject(JsonCodec.parse(raw), "request");
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
        if (Http.preamble(exchange, config.access())) return;
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
        if (body == null) {
            metrics.record(Metrics.Outcome.INVALID_REQUEST);
            return;
        }
        Map<String, Object> request;
        try {
            request = Values.asObject(JsonCodec.parse(body), "request");
            validator.accept(request);
        } catch (RuntimeException e) {
            metrics.record(Metrics.Outcome.INVALID_REQUEST);
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
                        metrics.record(Metrics.Outcome.INVALID_REQUEST);
                        Http.sendErrorQuietly(exchange, 400, Http.errorMessage(e));
                    } catch (IOException e) {
                        metrics.record(Metrics.Outcome.CLIENT_DISCONNECTED);
                        Log.LOG.log(System.Logger.Level.DEBUG, "client connection lost", e);
                    } catch (Throwable t) {
                        // Anything else is OURS. Catching RuntimeException as 400 here reported
                        // server defects as client errors - an NPE came back as a 400 quoting the
                        // null field - so they neither showed up as failures nor were actionable.
                        metrics.record(Metrics.Outcome.FAILED);
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
                    Validation.validateChatOptions(request);
                    Validation.validateGenerationParams(request, servedModel, config);
                },
                (request, id) -> {
                    List<Object> messages = Values.asArray(request.get("messages"), "messages");
                    String modelId = Requests.modelId(request, servedModel);
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
                    Validation.validateGenerationParams(request, servedModel, config);
                    Validation.require(
                            !Requests.completionPrompt(request).isBlank(),
                            "prompt must not be empty");
                },
                (request, id) -> {
                    String prompt = Requests.completionPrompt(request);
                    String modelId = Requests.modelId(request, servedModel);
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
                "resp_",
                request -> {
                    Validation.validateResponseOptions(request);
                    Requests.normalizeResponse(request);
                    Validation.validateGenerationParams(request, servedModel, config);
                    List<Object> folded = Requests.responseInputMessages(request);
                    Validation.require(!folded.isEmpty(), "input must not be empty");
                    // the SAME shape rules as chat, over the folded turns: roles, substance,
                    // response_format and tool declarations. This endpoint used to skip them
                    // entirely, so a bad role or a malformed tool reached the engine here and
                    // was refused there - or not at all.
                    Validation.validateChatRequest(request, folded);
                },
                (request, id) -> {
                    List<Object> messages = Requests.responseInputMessages(request);
                    String modelId = Requests.modelId(request, servedModel);
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
                    metrics,
                    () -> {
                        long created = System.currentTimeMillis() / 1000;
                        sse.emit(
                                OpenAiSchema.chatCompletionChunk(
                                        id, modelId, created, Map.of("role", "assistant"), null));
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
                                                                        created,
                                                                        Map.of("content", t),
                                                                        null)),
                                                deltaSink(
                                                        sse,
                                                        t ->
                                                                OpenAiSchema.chatCompletionChunk(
                                                                        id,
                                                                        modelId,
                                                                        created,
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
                                            created,
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
                                        id, modelId, created, Map.of(), result.finishReason()),
                                OpenAiSchema.chatCompletionChunk(
                                        id, modelId, created, Map.of(), null));
                    });
        }
    }

    /** Final stream sequence shared by chat and completions, followed by {@code [DONE]}. */
    private static void endStream(
            Sse.Stream sse,
            Map<String, Object> request,
            Reply result,
            Map<String, Object> finalChunk,
            Map<String, Object> usageOnlyChunk) {
        for (Map<String, Object> chunk :
                streamEndChunks(request, result, finalChunk, usageOnlyChunk)) sse.emit(chunk);
        sse.done();
    }

    /** OpenAI's optional final usage-only chunk; all preceding chunks carry null usage. */
    static List<Map<String, Object>> streamEndChunks(
            Map<String, Object> request,
            Reply result,
            Map<String, Object> finalChunk,
            Map<String, Object> usageOnlyChunk) {
        // The SSE response headers are committed by Sse.begin before generation, so per-phase
        // timings ride the LAST chunk (llama.cpp's server does the same: deltas.back() carries
        // timings). With include_usage the last chunk is the usage-only chunk; without it, the
        // final content chunk.
        if (!includeUsage(request)) {
            finalChunk.put("timings", OpenAiSchema.timings(result));
            return List.of(finalChunk);
        }
        finalChunk.put("usage", null);
        usageOnlyChunk.put("choices", List.of());
        usageOnlyChunk.put("usage", OpenAiSchema.usage(result));
        usageOnlyChunk.put("timings", OpenAiSchema.timings(result));
        return List.of(finalChunk, usageOnlyChunk);
    }

    /**
     * OpenAI stream_options: {"include_usage": true} requests an extra usage-only chunk after the
     * final chunk.
     */
    private static boolean includeUsage(Map<String, Object> request) {
        return request.get("stream_options") instanceof Map<?, ?> so
                && Boolean.TRUE.equals(so.get("include_usage"));
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
                    metrics,
                    () -> {
                        long created = System.currentTimeMillis() / 1000;
                        Consumer<String> sink =
                                deltaSink(
                                        sse,
                                        t ->
                                                OpenAiSchema.completionChunk(
                                                        id, modelId, created, t, null));
                        Reply result = generation.completion(request, prompt, Sinks.text(sink));
                        endStream(
                                sse,
                                request,
                                result,
                                OpenAiSchema.completionChunk(
                                        id, modelId, created, "", result.finishReason()),
                                OpenAiSchema.completionChunk(id, modelId, created, "", null));
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
            Sse.guardedResponses(
                    sse,
                    metrics,
                    () -> {
                        long created = System.currentTimeMillis() / 1000;
                        String itemId = "msg_" + id;
                        sse.emit(
                                "response.created",
                                Map.of(
                                        "type",
                                        "response.created",
                                        "response",
                                        OpenAiSchema.responseEnvelope(
                                                id,
                                                modelId,
                                                created,
                                                "in_progress",
                                                List.of(),
                                                null)));
                        boolean mayCallTools = ToolUse.offered(request);
                        if (!mayCallTools) responseMessageAdded(sse, itemId);
                        Consumer<String> sink =
                                mayCallTools
                                        ? null
                                        : deltaSink(
                                                sse,
                                                "response.output_text.delta",
                                                t -> OpenAiSchema.responseTextDelta(itemId, t));
                        Reply result =
                                generation.chat(
                                        request,
                                        messages,
                                        sink == null ? Sinks.NONE : Sinks.text(sink));
                        if (result.toolCalls().isEmpty()) {
                            if (mayCallTools) {
                                responseMessageAdded(sse, itemId);
                                if (!result.text().isEmpty()) {
                                    sse.emit(
                                            "response.output_text.delta",
                                            OpenAiSchema.responseTextDelta(itemId, result.text()));
                                }
                            }
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
                                    "response.content_part.done",
                                    Map.of(
                                            "type",
                                            "response.content_part.done",
                                            "item_id",
                                            itemId,
                                            "output_index",
                                            0,
                                            "content_index",
                                            0,
                                            "part",
                                            OpenAiSchema.outputText(result.text())));
                        }
                        List<Map<String, Object>> items =
                                OpenAiSchema.responseOutputItems(id, result);
                        for (int i = 0; i < items.size(); i++) {
                            if (!result.toolCalls().isEmpty()) {
                                Map<String, Object> started = new LinkedHashMap<>(items.get(i));
                                started.put("status", "in_progress");
                                started.put("arguments", "");
                                sse.emit(
                                        "response.output_item.added",
                                        Map.of(
                                                "type",
                                                "response.output_item.added",
                                                "output_index",
                                                i,
                                                "item",
                                                started));
                                sse.emit(
                                        "response.function_call_arguments.done",
                                        Map.of(
                                                "type",
                                                "response.function_call_arguments.done",
                                                "item_id",
                                                Values.stringValue(started.get("id"), ""),
                                                "output_index",
                                                i,
                                                "name",
                                                Values.stringValue(started.get("name"), ""),
                                                "arguments",
                                                Values.stringValue(
                                                        items.get(i).get("arguments"), "{}")));
                            }
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
                                        "timings",
                                        OpenAiSchema.timings(result),
                                        "response",
                                        // the SAME items just emitted, not a rebuild: see the
                                        // overload's note on clock-minted call ids
                                        OpenAiSchema.responseResponse(
                                                id, modelId, created, result, items)));
                        sse.done();
                    });
        }
    }

    private static void responseMessageAdded(Sse.Stream sse, String itemId) {
        sse.emit(
                "response.output_item.added",
                Map.of(
                        "type",
                        "response.output_item.added",
                        "output_index",
                        0,
                        "item",
                        OpenAiSchema.responseMessageItem(itemId, "in_progress", "")));
        sse.emit(
                "response.content_part.added",
                Map.of(
                        "type",
                        "response.content_part.added",
                        "item_id",
                        itemId,
                        "output_index",
                        0,
                        "content_index",
                        0,
                        "part",
                        OpenAiSchema.outputText("")));
    }

    /**
     * A streaming text sink: each chunk of generated text becomes one {@code data:} SSE frame built
     * by {@code chunkOf}.
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
        if (Http.preamble(exchange, config.access())) return;
        if (!exchange.getRequestURI().getPath().equals("/metrics")) {
            Http.sendError(exchange, 404, "Not found");
            return;
        }
        if (Http.requireMethod(exchange, "GET")) return;
        Http.sendText(
                exchange,
                200,
                Metrics.CONTENT_TYPE,
                metrics.exposition(
                        worker, generation.cacheSample(), generation.mediaCacheSample()));
    }

    /** Non-streaming reply: send the schema body as JSON. Timings ride in the body. */
    private static void respond(HttpExchange exchange, Reply result, Object body)
            throws IOException {
        Http.sendJson(exchange, 200, body);
    }

    /**
     * Enqueues the request for the generation worker (FIFO) and waits for it to finish; rejects
     * with 503 + Retry-After when the queue is full.
     */
    private void runQueued(HttpExchange exchange, Runnable work) throws IOException {
        Worker.Result result = worker.submitAndWait(work);
        if (result != Worker.Result.COMPLETED) {
            metrics.record(
                    result == Worker.Result.INTERRUPTED
                            ? Metrics.Outcome.CANCELLED
                            : Metrics.Outcome.REJECTED);
            // a shed request never reaches the engine, so nothing else would report it - and going
            // silent exactly when the server is saturated is the worst possible time to do so
            InferenceEvent rejected =
                    InferenceEvent.started(servedModel, InferenceEvent.CHAT, InferenceEvent.TEXT);
            rejected.errorType =
                    switch (result) {
                        case FULL -> "queue-full";
                        case INTERRUPTED -> "interrupted";
                        default -> "shutdown";
                    };
            rejected.end();
            rejected.commit();
            if (result == Worker.Result.FULL) {
                exchange.getResponseHeaders()
                        .set("Retry-After", String.valueOf(config.limits().retryAfterSeconds()));
            }
            String message =
                    switch (result) {
                        case FULL ->
                                "Server busy: "
                                        + config.limits().queueCapacity()
                                        + " requests already queued";
                        case INTERRUPTED -> "Request interrupted";
                        default -> "Server is shutting down";
                    };
            Http.sendError(exchange, 503, message);
            return;
        }
        // a job that finished without ever answering (escaped exception) must not hang the client
        if (exchange.getResponseCode() == -1) {
            metrics.record(Metrics.Outcome.FAILED);
            Http.sendErrorQuietly(exchange, 500, "Internal server error");
        }
    }
}
