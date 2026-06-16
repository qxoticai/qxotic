package com.llama4j;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.SynchronousQueue;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.IntConsumer;
import java.util.stream.Collectors;

import com.llama4j.Engine.GenerationResult;
import com.llama4j.Engine.StopSpec;
import com.llama4j.LFM25.Options;

/**
 * OpenAI-compatible HTTP server (--server): endpoint handlers (/v1/chat/completions,
 * /v1/completions, /v1/responses, /v1/models, /health, /props, /tokenize, /detokenize),
 * SSE streaming, request validation, tool-call parsing, the single-thread generation
 * worker with its bounded queue, chat-session resume, and the prompt cache wiring.
 * Pure transport and orchestration: all inference goes through {@link Engine#generate}.
 */
final class Server {

    private Server() {
    }

    /** Boots the server and returns it (port 0 binds an ephemeral port — the integration
     *  test reads the actual one from the returned server). */
    static HttpServer run(Model model, Options options) throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(options.host(), options.port()), 0);
        String servedId = options.modelPath().getFileName().toString();
        Map<String, Object> modelCard = Map.of(
                "id", servedId, "object", "model", "created", 0, "owned_by", "lfm25.java");
        server.createContext("/v1/models", exchange -> { // also serves /v1/models/{id} -> card or 404
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.getResponseHeaders().set("Allow", "GET, OPTIONS");
                sendError(exchange, 405, "Method not allowed");
                return;
            }
            String path = exchange.getRequestURI().getPath();
            if (path.equals("/v1/models")) {
                sendJson(exchange, 200, Map.of("object", "list", "data", List.of(modelCard)));
            } else if (path.equals("/v1/models/" + servedId)) {
                sendJson(exchange, 200, modelCard);
            } else {
                sendError(exchange, 404, "Unknown model: " + path.substring("/v1/models/".length())
                        + " (this server serves " + servedId + ")");
            }
        });
        server.createContext("/v1/chat/completions", exchange -> handleChatCompletion(exchange, model, options));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, model, options));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, model, options));
        jsonRoute(server, "/health", null, request ->
                Map.of("status", "ok", "busy", workerBusy, "queued", GENERATION_QUEUE.size()));
        jsonRoute(server, "/props", null, request -> Map.of(
                "model", options.modelPath().getFileName().toString(),
                "n_ctx", model.contextLength(),
                "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                "n_vocab", model.vocabularySize(),
                "prompt_cache", promptCache == null ? Map.of("enabled", false) : promptCache.stats()));
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
        server.createContext("/metrics", exchange -> handleMetrics(exchange));
        server.createContext("/", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            sendError(exchange, 404, "Not found");
        });
        // The prompt cache is an LFM2.5-specific optimization (its tree and bx tiers know that
        // architecture's KV/conv layout); other models run the plain, un-cached generation path.
        if (RuntimeFlags.PROMPT_CACHE && model instanceof Llama llamaModel) {
            Llama.Configuration config = llamaModel.configuration();
            CacheStore store;
            String cacheFile = RuntimeFlags.PROMPT_CACHE_FILE;
            if (cacheFile != null) {
                long kvBytesPerToken = 0;
                for (int l = 0; l < config.nLayerKvFromStart; l++) {
                    int kvDim = config.kvDim(l);
                    if (kvDim > 0) kvBytesPerToken += 2L * kvDim * Float16.BYTES;
                }
                store = CacheStore.mmap(Path.of(cacheFile),
                        RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES,
                        RuntimeFlags.PROMPT_CACHE_BLOCK_TOKENS, kvBytesPerToken);
            } else {
                store = CacheStore.inMemory();
            }
            promptCache = new PromptCache(config, store);
            System.out.printf("Prompt cache enabled: budget=%d MB %s%n",
                    RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES >> 20,
                    cacheFile != null ? "file=" + cacheFile : "(in memory)");
        }
        startGenerationWorker();
        warmPromptCache(model, options); // blocks until warmed: instant resume from request one
        startStreamReaper();
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
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            // contexts match by longest PREFIX: /v1/models/garbage would land here — 404 it
            if (!exchange.getRequestURI().getPath().equals(path)) {
                sendError(exchange, 404, "Not found");
                return;
            }
            if (method != null && !method.equals(exchange.getRequestMethod())) {
                exchange.getResponseHeaders().set("Allow", method + ", OPTIONS");
                sendError(exchange, 405, "Method not allowed");
                return;
            }
            Map<String, Object> request = Map.of();
            if ("POST".equals(method)) {
                byte[] raw = readBody(exchange);
                if (raw == null) return;
                try {
                    request = Values.asObject(JsonCodec.parse(new String(raw, StandardCharsets.UTF_8)), "request");
                } catch (RuntimeException e) {
                    sendError(exchange, 400, errorMessage(e));
                    return;
                }
            }
            try {
                sendJson(exchange, 200, body.apply(request));
            } catch (RuntimeException e) {
                sendError(exchange, 400, errorMessage(e));
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
        logRequest(exchange);
        addCommonHeaders(exchange);
        if (handleOptions(exchange)) return;
        if (!"POST".equals(exchange.getRequestMethod())) {
            exchange.getResponseHeaders().set("Allow", "POST, OPTIONS");
            sendError(exchange, 405, "Method not allowed");
            return;
        }
        byte[] body = readBody(exchange); // read on the handler thread: a stalled upload must not block the generation worker
        if (body == null) return;
        Map<String, Object> request;
        try {
            request = Values.asObject(JsonCodec.parse(new String(body, StandardCharsets.UTF_8)), "request");
            validator.accept(request);
        } catch (RuntimeException e) {
            sendError(exchange, 400, errorMessage(e));
            return;
        }
        String id = idPrefix + Long.toUnsignedString(System.nanoTime(), 36);
        runQueued(exchange, () -> {
            try {
                job.run(request, id);
            } catch (RuntimeException e) {
                sendErrorQuietly(exchange, 400, errorMessage(e));
            } catch (IOException e) {
                System.err.println("client connection lost: " + e);
            } catch (Throwable t) {
                sendErrorQuietly(exchange, 500, t.toString());
            }
        });
    }

    private static void handleChatCompletion(HttpExchange exchange, Model model, Options options) throws IOException {
        handleGenerationPost(exchange, "chatcmpl-", request -> {
            Validation.validateChatRequest(request);
            Validation.validateGenerationParams(request, options);
        }, (request, id) -> {
            List<Object> messages = Values.asArray(request.get("messages"), "messages");
            String modelId = requestModelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamChatCompletion(exchange, model, options, request, messages, modelId, id);
            } else {
                GenerationResult result = generateChat(model, options, request, messages, null, null, null, null); // non-streaming, no tools
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, OpenAiSchema.chatCompletionResponse(id, modelId, result));
            }
        });
    }

    private static void handleCompletion(HttpExchange exchange, Model model, Options options) throws IOException {
        handleGenerationPost(exchange, "cmpl-", request -> {
            Validation.validateGenerationParams(request, options);
            Options.require(!completionPrompt(request).isBlank(), "prompt must not be empty");
        }, (request, id) -> {
            String prompt = completionPrompt(request);
            String modelId = requestModelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamCompletion(exchange, model, options, request, prompt, modelId, id);
            } else {
                GenerationResult result = generateCompletion(model, options, request, prompt, null, null, null, null); // non-streaming
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, OpenAiSchema.completionResponse(id, modelId, result));
            }
        });
    }

    private static String requestModelId(Map<String, Object> request, Options options) {
        return Values.stringValue(request.get("model"), options.modelPath().getFileName().toString());
    }

    private static String completionPrompt(Map<String, Object> request) {
        Object promptValue = request.get("prompt");
        return promptValue instanceof List<?> prompts
                ? prompts.stream().map(String::valueOf).collect(Collectors.joining("\n"))
                : Values.stringValue(promptValue, "");
    }

    private static void handleResponse(HttpExchange exchange, Model model, Options options) throws IOException {
        handleGenerationPost(exchange, "resp-", request -> {
            normalizeResponseRequest(request);
            Validation.validateGenerationParams(request, options);
            Options.require(!responseInputMessages(request).isEmpty(), "input must not be empty");
        }, (request, id) -> {
            List<Object> messages = responseInputMessages(request);
            String modelId = requestModelId(request, options);
            if (Values.booleanValue(request.get("stream"), false)) {
                streamResponse(exchange, model, options, request, messages, modelId, id);
            } else {
                GenerationResult result = generateChat(model, options, request, messages, null, null, null, null); // non-streaming, no tools
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, OpenAiSchema.responseResponse(id, modelId, result));
            }
        });
    }

    /** Runs an SSE body. A failure after the headers were sent is delivered as a terminal in-band
     *  error event + [DONE] so clients stop instead of hanging on a dead stream; a lost connection
     *  ({@link UncheckedIOException} from a frame write) propagates as IOException for the handler
     *  to log. */
    private static void guarded(SseStream sse, Runnable body) throws IOException {
        try {
            body.run();
        } catch (UncheckedIOException e) {
            throw e.getCause(); // client connection lost mid-stream
        } catch (RuntimeException e) {
            sse.emit(Map.of("error", errorPayload(400, errorMessage(e))));
            sse.done();
        }
    }

    private static void streamChatCompletion(HttpExchange exchange, Model model, Options options, Map<String, Object> request,
                                             List<Object> messages, String modelId, String id) throws IOException {
        try (SseStream sse = beginStream(exchange)) {
            guarded(sse, () -> {
                sse.emit(OpenAiSchema.chatCompletionChunk(id, modelId, Map.of("role", "assistant"), null));
                boolean forcedTool = forcedToolChoice(request) != null;
                boolean hasTools = hasUsableTools(request);
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
                GenerationResult result = generateChat(model, options, request, messages, contentSink, reasoningSink, toolCallSink, usage);
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
    private static void endStream(SseStream sse, Map<String, Object> request, GenerationResult result,
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

    private static void streamCompletion(HttpExchange exchange, Model model, Options options, Map<String, Object> request,
                                         String prompt, String modelId, String id) throws IOException {
        try (SseStream sse = beginStream(exchange)) {
            guarded(sse, () -> {
                OpenAiSchema.Usage usage = new OpenAiSchema.Usage();
                Consumer<String> sink = deltaSink(sse, usage, t -> OpenAiSchema.completionChunk(id, modelId, t, null));
                GenerationResult result = generateCompletion(model, options, request, prompt, sink, null, null, usage);
                endStream(sse, request, result,
                        OpenAiSchema.completionChunk(id, modelId, "", result.finishReason()),
                        OpenAiSchema.completionChunk(id, modelId, "", null));
            });
        }
    }

    private static void streamResponse(HttpExchange exchange, Model model, Options options, Map<String, Object> request,
                                       List<Object> messages, String modelId, String id) throws IOException {
        try (SseStream sse = beginStream(exchange)) {
            guarded(sse, () -> {
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
                GenerationResult result = generateChat(model, options, request, messages, sink, null, null, usage);
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
    private static Consumer<String> deltaSink(SseStream sse, OpenAiSchema.Usage usage,
                                              Function<String, Map<String, Object>> chunkOf) {
        return deltaSink(sse, usage, null, chunkOf);
    }

    /** As {@link #deltaSink(SseStream, OpenAiSchema.Usage, Function)}, but emitted as a named SSE event (the
     *  Responses API) when {@code event} is non-null. */
    private static Consumer<String> deltaSink(SseStream sse, OpenAiSchema.Usage usage, String event,
                                              Function<String, Map<String, Object>> chunkOf) {
        return text -> {
            Map<String, Object> chunk = chunkOf.apply(text);
            if (usage != null) chunk.put("usage", OpenAiSchema.chunkUsage(usage));
            if (event == null) sse.emit(chunk);
            else sse.emit(event, chunk);
        };
    }

    private static void normalizeResponseRequest(Map<String, Object> request) {
        if (!request.containsKey("max_tokens") && request.containsKey("max_output_tokens")) {
            request.put("max_tokens", request.get("max_output_tokens"));
        }
        Object tools = request.get("tools");
        if (tools instanceof List<?> values) {
            List<Object> normalized = new ArrayList<>();
            for (Object value : values) normalized.add(normalizeResponseTool(value));
            request.put("tools", normalized);
        }
    }

    private static Object normalizeResponseTool(Object value) {
        Map<String, Object> tool = Values.asObject(value, "tool");
        if (tool.get("function") != null) return tool;
        if ("function".equals(tool.get("type")) && tool.get("name") != null) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", tool.get("name"));
            if (tool.get("description") != null) function.put("description", tool.get("description"));
            function.put("parameters", tool.getOrDefault("parameters", Map.of()));
            return Map.of("type", "function", "function", function);
        }
        return tool;
    }

    private static List<Object> responseInputMessages(Map<String, Object> request) {
        List<Object> messages = new ArrayList<>();
        String instructions = Values.stringValue(request.get("instructions"), null);
        if (instructions != null && !instructions.isBlank()) {
            messages.add(Map.of("role", "system", "content", instructions));
        }
        Object input = request.get("input");
        if (input instanceof String s) {
            messages.add(Map.of("role", "user", "content", s));
        } else if (input instanceof List<?> list) {
            for (Object item : list) addResponseInputMessage(messages, item);
        } else if (input != null) {
            addResponseInputMessage(messages, input);
        } else {
            throw new IllegalArgumentException("input is required");
        }
        return messages;
    }

    private static void addResponseInputMessage(List<Object> messages, Object item) {
        if (item instanceof String s) {
            messages.add(Map.of("role", "user", "content", s));
            return;
        }
        Map<String, Object> map = Values.asObject(item, "input item");
        String type = Values.stringValue(map.get("type"), "message");
        if ("function_call_output".equals(type)) {
            messages.add(Map.of(
                    "role", "tool",
                    "name", Values.stringValue(map.get("call_id"), "tool"),
                    "content", Values.stringValue(map.get("output"), "")));
            return;
        }
        String role = Values.stringValue(map.get("role"), "user");
        messages.add(Map.of("role", role, "content", responseInputText(map.get("content"))));
    }

    private static String responseInputText(Object content) {
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object part : parts) {
                if (part instanceof String s) {
                    sb.append(s);
                } else if (part instanceof Map<?, ?> map) {
                    Object text = map.get("text");
                    if (text == null) text = map.get("input_text");
                    if (text == null) text = map.get("output_text");
                    if (text != null) sb.append(text);
                }
            }
            return sb.toString();
        }
        return Values.stringValue(content, "");
    }

    private static GenerationResult generateChat(Model model, Options options, Map<String, Object> request,
                                                   List<Object> messages,
                                                   Consumer<String> onText, Consumer<String> onReasoning,
                                                   Consumer<String> onToolCall,
                                                   OpenAiSchema.Usage usageCounts) {
        LFMTokenizer tokenizer = model.tokenizer();
        ChatContext chatContext = new ChatContext(
                messages,
                hasUsableTools(request) ? Values.asArray(request.get("tools"), "tools") : null,
                request.get("tool_choice"),
                true,
                requestThink(request, options),
                request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs ? (Map<String, Object>) kwargs : null);
        List<Integer> promptTokens = new ArrayList<>(model.chatFormat().encode(chatContext));
        seedForcedToolCall(tokenizer, request, promptTokens);
        if (System.getProperty("llama.debugPrompt") != null) {
            System.err.println("[prompt] " + tokenizer.decode(promptTokens));
        }
        // Incremental in-place session resume is LFM2.5-only (ChatML delta encoding); other models
        // re-encode the full prompt each turn.
        Ingestion resumed = model instanceof Llama
                ? matchChatSession(request, messages, new LFMChatFormat(tokenizer), tokenizer, options) : null;
        Ingestion ingestion = resumed != null ? resumed : Ingestion.of(model.createNewState(), 0, promptTokens);
        GenerationResult result = generateResponse(model, options, request, promptTokens, ingestion, model.stopTokens(), onText, onReasoning, onToolCall, usageCounts);
        if (model instanceof Llama) saveChatSession(model, request, messages, ingestion, result);
        return hasUsableTools(request) ? withParsedToolCalls(model, result, request) : result;
    }

    /** Remember the live state for instant resume of the next turn; only clean stop-terminated
     *  text replies are resumable (tool calls and aborted/length-capped replies are not). */
    private static void saveChatSession(Model model, Map<String, Object> request, List<Object> messages,
                                        Ingestion ingestion, GenerationResult result) {
        if (!"stop".equals(result.finishReason()) || !result.toolCalls().isEmpty() || result.text().isBlank()) {
            return;
        }
        int position = ingestion.prefillPositions() + result.completionTokens();
        List<String> keys = new ArrayList<>(messages.size());
        for (Object message : messages) keys.add(messageKey(message));
        chatSession = new ChatSession(ingestion.state(), position, keys, result.text(), toolsKey(request));
    }

    private static GenerationResult generateCompletion(Model model, Options options, Map<String, Object> request, String prompt,
                                                        Consumer<String> onText, Consumer<String> onReasoning,
                                                        Consumer<String> onToolCall,
                                                        OpenAiSchema.Usage usageCounts) {
        List<Integer> promptTokens = options.rawPrompt() ? model.tokenizer().encodeWithSpecialTokens(prompt) : new ArrayList<>(model.tokenizer().encode(prompt));
        Ingestion ingestion = Ingestion.of(model.createNewState(), 0, promptTokens);
        return generateResponse(model, options, request, promptTokens, ingestion, model.stopTokens(), onText, onReasoning, onToolCall, usageCounts);
    }


    /** Prompt cache instance; created in runServer when enabled, null in CLI modes. */
    private static PromptCache promptCache;

    /** What to feed the generation loop: which state, from which position, with which tokens.
     *  Fresh requests: new state, position 0, the full prompt. Resumed chat sessions: the live
     *  state mid-context with only the delta (turn close + new messages + generation prompt). */
    private record Ingestion(InferenceState state, int startPosition, List<Integer> tokens, int prefillPositions) {
        /** prefillPositions: see {@link Engine#prefillPositions}. */
        static Ingestion of(InferenceState state, int startPosition, List<Integer> tokens) {
            return new Ingestion(state, startPosition, tokens, Engine.prefillPositions(state, startPosition, tokens));
        }
    }

    /**
     * The live state of the most recent chat generation. A follow-up request resumes it IN
     * PLACE when its messages extend the session's conversation (same prefix, the assistant
     * echo equal to what we returned, same tools): only the delta is ingested — no re-prefill,
     * no cache restore, token-exact even with thinking/surrogate tokens in the stream, because
     * history is never re-encoded. Single generation worker thread, so a plain field suffices;
     * an unrelated request simply replaces the session.
     */
    private record ChatSession(InferenceState state, int position, List<String> messageKeys, String reply, String toolsKey) {}
    private static ChatSession chatSession;

    private static String messageKey(Object message) {
        Map<String, Object> map = Values.asObject(message, "message");
        return Values.stringValue(map.get("role"), "user") + "\u0000" + ChatFormats.chatMessageContent(map);
    }

    private static String toolsKey(Map<String, Object> request) {
        if (!hasUsableTools(request)) return "";
        return JsonCodec.stringify(request.get("tools")) + "|" + JsonCodec.stringify(request.get("tool_choice"));
    }

    /** Delta tokens to resume the session with this request, or null when it does not extend it. */
    private static Ingestion matchChatSession(Map<String, Object> request, List<Object> messages,
                                              LFMChatFormat chatFormat, LFMTokenizer tokenizer, Options options) {
        ChatSession s = chatSession;
        if (s == null || !toolsKey(request).equals(s.toolsKey())) return null;
        int n = s.messageKeys().size();
        if (messages.size() < n + 2) return null;
        for (int i = 0; i < n; i++) {
            if (!messageKey(messages.get(i)).equals(s.messageKeys().get(i))) return null;
        }
        Map<String, Object> echo = Values.asObject(messages.get(n), "message");
        if (!"assistant".equals(Values.stringValue(echo.get("role"), ""))
                || !ChatFormats.chatMessageContent(echo).strip().equals(s.reply().strip())) return null;
        chatSession = null; // taken: a failure mid-resume must not leave a stale session behind
        // the session's un-ingested stop token (state.latestToken) closes the assistant turn;
        // generate() prepends it, we add the template newline and the new turns
        List<Integer> delta = new ArrayList<>(tokenizer.encode("\n"));
        for (int i = n + 1; i < messages.size(); i++) {
            delta.addAll(ChatFormats.encodeChatTurn(chatFormat, messages.get(i)));
        }
        delta.addAll(chatFormat.encodeGenerationPrompt());
        if (!requestThink(request, options)) chatFormat.appendThinkSurrogate(delta);
        seedForcedToolCall(tokenizer, request, delta);
        return Ingestion.of(s.state(), s.position(), delta);
    }

    /**
     * Server generation driver over {@link Engine#generate}. The ingestion plan says where to
     * start: a fresh state at position 0 (with the prompt cache plugged in through
     * {@link GenerationHooks}), or a resumed chat session mid-context — in which case the
     * cache is bypassed (its tree is keyed by full-stream positions). cachedOut[0] tracks the
     * resumed-prefix length for the streaming usage counters.
     */
    private static GenerationResult runServerGeneration(Model model, Ingestion ingestion, Engine.Params params,
                                                        Engine.Listener listener, int[] cachedOut, boolean warm) {
        PromptCache cache = promptCache;
        InferenceState state = ingestion.state();
        if (cache == null || ingestion.startPosition() > 0) {
            return Engine.generate(model, state, ingestion.startPosition(), ingestion.tokens(), params, listener, GenerationHooks.NONE);
        }
        // The prompt cache owns its own resume/commit policy; we just drive Engine.generate with
        // it as the hooks, commit the final frontier on success, and release per-request state.
        // The cache is only ever created for LFM2.5 (see run()), so the state is a Llama.State.
        PromptCache.CacheRun run = cache.beginGeneration((Llama.State) state, cachedOut, warm);
        try {
            GenerationResult result = Engine.generate(model, state, 0, ingestion.tokens(), params, listener, run);
            run.commitFinal();
            return result;
        } finally {
            run.cleanup();
        }
    }

    /**
     * Pre-ingests --warm-prompt / -Dllama.promptCacheWarm files into the prompt cache with
     * FULLY DENSE bx retention and sticky (eviction-exempt) nodes: requests diverging at ANY
     * position inside a warmed prompt resume token-exact, with zero re-ingest. Each file is
     * warmed in two stream forms — chat-template system message and raw completion prompt.
     * Runs on the generation worker (cache blobs are confined to it) and blocks until done.
     */
    private static void warmPromptCache(Model model, Options options) {
        List<String> files = new ArrayList<>(options.warmPrompts());
        if (RuntimeFlags.PROMPT_CACHE_WARM != null) {
            for (String f : RuntimeFlags.PROMPT_CACHE_WARM.split(",")) {
                if (!f.isBlank()) files.add(f.strip());
            }
        }
        if (files.isEmpty()) return;
        if (promptCache == null) {
            System.err.println("warm-prompt ignored: prompt cache is disabled");
            return;
        }
        CountDownLatch done = new CountDownLatch(1);
        boolean queued = GENERATION_QUEUE.offer(() -> {
            try {
                for (String file : files) {
                    warmFile(model, file);
                }
            } catch (Exception e) {
                System.err.println("warm-prompt failed: " + e);
            } finally {
                done.countDown();
            }
        });
        if (!queued) throw new IllegalStateException("generation queue full at startup");
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static void warmFile(Model model, String file) throws IOException {
        String text = Files.readString(Path.of(file));
        LFMTokenizer tokenizer = model.tokenizer();
        ChatContext warm = new ChatContext(
                List.of(Map.of("role", "system", "content", text)), null, null, true, false, null);
        List<Integer> chatForm = ChatFormats.forModel(tokenizer).encode(warm);
        List<Integer> rawForm = tokenizer.encode(text);
        Engine.Params params = new Engine.Params(Sampler.ARGMAX, 0, 0, new StopSpec(Set.of(), List.of()), false); // warm: no deadline
        for (List<Integer> tokens : List.of(chatForm, rawForm)) {
            long startNanos = System.nanoTime();
            Ingestion ingestion = Ingestion.of(model.createNewState(), 0, tokens);
            runServerGeneration(model, ingestion, params, new Engine.Listener(null, null, null, null), new int[1], true);
            System.out.printf("warm-prompt %s: %d tokens in %.1f s%n",
                    file, ingestion.prefillPositions(), (System.nanoTime() - startNanos) / 1e9);
        }
    }

    /** Builds a grammar cursor from request params: {@code grammar} (GBNF string) or
     *  {@code response_format: {type: "json_object"}}. Returns null when no constraint. */
    private static Grammar.Cursor buildGrammarCursor(LFMTokenizer tokenizer, Map<String, Object> request) {
        if (!RuntimeFlags.GRAMMAR) return null;
        Object gbnf = request.get("grammar");
        if (gbnf instanceof String s && !s.isBlank()) {
            Grammar.Spec spec = Grammar.of(s, tokenizer);
            return spec.cursor();
        }
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?,?> f && "json_object".equals(f.get("type"))) {
            Grammar.Spec spec = Grammar.json(tokenizer);
            return spec.cursor();
        }
        return null;
    }

    /** Request fields to {@link Engine.Params}/{@link Engine.Listener}, then one engine pass
     *  through {@link #runServerGeneration}. Streaming counters mirror Engine's final usage:
     *  generated tokens are counted unless they are the trailing token stop removed from the
     *  result. */
    private static GenerationResult generateResponse(Model model, Options options, Map<String, Object> request, List<Integer> promptTokens,
                                           Ingestion ingestion, Set<Integer> baseStopTokens, Consumer<String> onText,
                                           Consumer<String> onReasoning, Consumer<String> onToolCall,
                                           OpenAiSchema.Usage usageCounts) {
        float temperature = Values.floatValue(request.get("temperature"), options.temperature());
        float topp = Values.floatValue(request.get("top_p"), options.topp());
        long seed = Values.longValue(request.get("seed"), options.seed());
        int maxTokens = Values.intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens());
        // server-side completion-token ceiling: an unbounded (or oversized) request can never run
        // the worker past llama.serverMaxTokens; hitting it reports finish_reason "length"
        if (RuntimeFlags.SERVER_MAX_TOKENS > 0)
            maxTokens = maxTokens < 0 ? RuntimeFlags.SERVER_MAX_TOKENS : Math.min(maxTokens, RuntimeFlags.SERVER_MAX_TOKENS);
        // defense in depth: server requests were already checked by validateGenerationParams on
        // the handler thread; these guards keep the method safe for any future non-HTTP caller
        Options.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        Options.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        StopSpec stops = stopSpec(request.get("stop"), baseStopTokens);
        boolean think = requestThink(request, options);
        Sampler sampler = Engine.configuredSampler(model, think, temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span,
            // by default at half the completion budget (request reasoning_max_tokens overrides;
            // -1 = uncapped)
            int reasoningBudget = Values.intValue(request.get("reasoning_max_tokens"),
                    maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1);
            sampler = Engine.withThinkBudget(sampler, model.tokenizer(), reasoningBudget);
        }
        Grammar.Cursor grammarCursor = buildGrammarCursor(model.tokenizer(), request);
        if (grammarCursor != null) {
            Map<String, Integer> specials = model.tokenizer().getSpecialTokens();
            int eosToken = specials.getOrDefault("<eos>", specials.getOrDefault("<|endoftext|>", 2));
            sampler = Sampler.withGrammar(sampler, grammarCursor, eosToken);
        }
        int consumedPromptTokens = Engine.consumedPromptTokens(model.tokenizer(), promptTokens); // client-facing usage counts
        int[] cachedOut = {Math.min(ingestion.startPosition(), consumedPromptTokens)};
        IntConsumer onToken = usageCounts == null ? null : token -> {
            usageCounts.cachedTokens = Math.min(cachedOut[0], consumedPromptTokens);
            if (!stops.tokenStops().contains(token)) usageCounts.completionTokens++;
        };
        if (usageCounts != null) usageCounts.promptTokens = consumedPromptTokens;
        Engine.Params params = new Engine.Params(sampler, maxTokens, RuntimeFlags.SERVER_REQUEST_TIMEOUT_NANOS, stops, inlineReasoning(request));
        Engine.Listener listener = new Engine.Listener(onToken, onText, onReasoning, onToolCall);
        GenerationResult result = runServerGeneration(model, ingestion, params, listener, cachedOut, false);
        // /metrics counters: single-writer (generation worker), volatile for handler reads
        generationRequests++;
        promptTokensTotal += result.promptTokens();
        completionTokensTotal += result.completionTokens();
        return result;
    }

    private static final long START_NANOS = System.nanoTime();
    private static volatile long generationRequests, promptTokensTotal, completionTokensTotal;

    /** Prometheus text exposition (llama.cpp-style /metrics): request/token totals, queue and
     *  worker gauges, prompt-cache stats. */
    private static void handleMetrics(HttpExchange exchange) throws IOException {
        logRequest(exchange);
        addCommonHeaders(exchange);
        if (handleOptions(exchange)) return;
        if (!exchange.getRequestURI().getPath().equals("/metrics")) {
            sendError(exchange, 404, "Not found");
            return;
        }
        if (!"GET".equals(exchange.getRequestMethod())) {
            exchange.getResponseHeaders().set("Allow", "GET, OPTIONS");
            sendError(exchange, 405, "Method not allowed");
            return;
        }
        StringBuilder sb = new StringBuilder();
        metric(sb, "lfm25_uptime_seconds", "gauge", (System.nanoTime() - START_NANOS) / 1e9);
        metric(sb, "lfm25_requests_total", "counter", generationRequests);
        metric(sb, "lfm25_prompt_tokens_total", "counter", promptTokensTotal);
        metric(sb, "lfm25_completion_tokens_total", "counter", completionTokensTotal);
        metric(sb, "lfm25_queue_depth", "gauge", GENERATION_QUEUE.size());
        metric(sb, "lfm25_worker_busy", "gauge", workerBusy ? 1 : 0);
        PromptCache cache = promptCache;
        if (cache != null) {
            for (Map.Entry<String, Object> entry : cache.stats().entrySet()) {
                if (entry.getValue() instanceof Number n) {
                    String kind = entry.getKey().endsWith("bytes") || entry.getKey().equals("nodes") ? "gauge" : "counter";
                    metric(sb, "lfm25_prompt_cache_" + entry.getKey(), kind, n);
                }
            }
        }
        byte[] body = sb.toString().getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
        exchange.sendResponseHeaders(200, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }

    private static void metric(StringBuilder sb, String name, String type, Number value) {
        sb.append("# TYPE ").append(name).append(' ').append(type).append('\n')
          .append(name).append(' ').append(value).append('\n');
    }

    private static boolean hasUsableTools(Map<String, Object> request) {
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "none".equals(s)) return false;
        Object tools = request.get("tools");
        return tools instanceof List<?> list && !list.isEmpty();
    }

    /** The function name a tool_choice forces ("" = any function via "required"), or null when
     *  the request does not force a call. */
    private static String forcedToolChoice(Map<String, Object> request) {
        if (!hasUsableTools(request)) return null;
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "required".equals(s)) return "";
        if (choice instanceof Map<?, ?> map && map.get("function") instanceof Map<?, ?> fn
                && fn.get("name") instanceof String name) {
            return name;
        }
        return null;
    }

    /** The assistant-turn text seeded by {@link #seedForcedToolCall}; re-attached to the reply
     *  before parsing so the seeded call parses whole. */
    private static String forcedToolCallPrefix(Map<String, Object> request) {
        String choice = forcedToolChoice(request);
        if (choice == null) return "";
        return choice.isEmpty() ? ToolCalls.TC_START : ToolCalls.TC_START + "[" + choice;
    }

    /** tool_choice "required"/named function: seed the assistant turn with <|tool_call_start|>
     *  (plus "[name" for a named choice) so the model can only complete a tool call instead of
     *  merely being asked to make one. The open paren is deliberately NOT seeded: a bare
     *  trailing "(" lands on a tokenization boundary the model never saw (it merges with the
     *  first argument in training data) and greedy decoding stops dead. No-op when the
     *  vocabulary lacks the marker (the prompted fallback keeps its text hint). */
    private static void seedForcedToolCall(LFMTokenizer tokenizer, Map<String, Object> request, List<Integer> promptTokens) {
        String choice = forcedToolChoice(request);
        if (choice == null) return;
        Integer marker = tokenizer.getSpecialTokens().get("<|tool_call_start|>");
        if (marker == null) return;
        promptTokens.add(marker);
        if (!choice.isEmpty()) promptTokens.addAll(tokenizer.encode("[" + choice));
    }

    private static GenerationResult withParsedToolCalls(Model model, GenerationResult result, Map<String, Object> request) {
        // Parse from the FULL generated text (think span included). result.text() is the
        // think-STRIPPED content, so a call the model emits before it closes </think> (or in an
        // unterminated think span) would be deleted before we ever see it. Decoding the raw
        // response tokens renders special tokens (<|tool_call_start|>, <think>) as literal text.
        String text = forcedToolCallPrefix(request);
        String decoded = model.tokenizer().decode(result.tokens());
        text += !decoded.strip().isEmpty() ? decoded
                : (result.reasoning() != null ? result.reasoning() + "\n" : "") + result.text();
        boolean debug = System.getProperty("llama.debugToolCalls") != null;
        List<Map<String, Object>> toolCalls = ToolCalls.parseToolCalls(text, toolNames(request));
        if (toolCalls.isEmpty()) {
            // A reply that smells like an attempted call (markers or a "name" key) but parsed to
            // nothing is the diagnostic we care about; surface it even without the debug flag.
            String t = text.strip();
            if (t.contains(ToolCalls.TC_START) || t.contains("\"name\"")) {
                System.err.println("[tool-parse] tools offered but parsed 0 calls from reply: " + t.replace("\n", "\\n"));
            }
            return result;
        }
        if (debug) System.err.println("[tool-parse] found " + toolCalls.size() + " call(s) in: " + text.strip().replace("\n", "\\n"));
        int marker = text.indexOf(ToolCalls.TC_START);
        return result.asToolCalls(toolCalls, marker > 0 ? text.substring(0, marker).strip() : "");
    }

    /** The function names a request offers; calls naming anything else are dropped. */
    private static Set<String> toolNames(Map<String, Object> request) {
        if (!(request.get("tools") instanceof List<?> tools)) return Set.of();
        Set<String> names = new HashSet<>();
        for (Object tool : tools) {
            if (tool instanceof Map<?, ?> t && t.get("function") instanceof Map<?, ?> fn
                    && fn.get("name") instanceof String name) {
                names.add(name);
            }
        }
        return names;
    }

    /** User stop strings stay TEXT stops only: token stops end generation anywhere, including
     *  inside the think span, while text stops are matched against content alone. (A former
     *  single-token shortcut here made stop strings fire on reasoning.) */
    private static StopSpec stopSpec(Object value, Set<Integer> baseStopTokens) {
        List<String> textStops = new ArrayList<>();
        if (value instanceof String s) {
            if (!s.isEmpty()) textStops.add(s);
        } else if (value instanceof List<?> values) {
            for (Object item : values) {
                String stop = Values.stringValue(item, "");
                if (!stop.isEmpty()) textStops.add(stop);
            }
        } else if (value != null) {
            throw new IllegalArgumentException("stop must be a string or an array of strings");
        }
        return new StopSpec(Collections.unmodifiableSet(baseStopTokens), List.copyOf(textStops));
    }

    /** Effective thinking switch for a server request: chat_template_kwargs.enable_thinking
     *  (llama.cpp convention) overrides the CLI --think flag. Forced tool calls never think —
     *  the call marker is seeded as the first assistant token. */
    private static boolean requestThink(Map<String, Object> request, Options options) {
        if (forcedToolChoice(request) != null) {
            return false;
        }
        if (request.get("chat_template_kwargs") instanceof Map<?, ?> kwargs
                && kwargs.get("enable_thinking") instanceof Boolean enabled) {
            return enabled;
        }
        return options.think();
    }

    /** llama.cpp-compatible reasoning_format: "none" = leave thinking inline in content (with
     *  literal <think> markers) instead of routing it to the reasoning_content channel — lets
     *  vanilla OpenAI clients that only render content show thinking live. */
    private static boolean inlineReasoning(Map<String, Object> request) {
        return "none".equals(Values.stringValue(request.get("reasoning_format"), null));
    }

    private static void setTimingHeader(HttpExchange exchange, GenerationResult result) {
        exchange.getResponseHeaders().set("X-LFM2-Timing", JsonCodec.stringify(OpenAiSchema.timings(result)));
    }

    private static void sendJson(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] bytes = JsonCodec.stringify(value).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    /** Generation requests run one at a time on a dedicated worker thread, fed by a bounded
     *  FIFO queue of llama.serverQueue waiting requests (default 4; 0 = reject unless idle).
     *  When the queue is full the request is rejected with 503 + Retry-After so clients back
     *  off instead of piling up. */
    private static final BlockingQueue<Runnable> GENERATION_QUEUE =
            RuntimeFlags.SERVER_QUEUE == 0 ? new SynchronousQueue<>()
                                    : new ArrayBlockingQueue<>(RuntimeFlags.SERVER_QUEUE);

    private static volatile boolean workerBusy;

    private static void startGenerationWorker() {
        Thread.ofPlatform().name("generation-worker").daemon(true).start(() -> {
            while (true) {
                try {
                    Runnable job = GENERATION_QUEUE.take();
                    workerBusy = true;
                    try {
                        job.run();
                    } finally {
                        workerBusy = false;
                    }
                } catch (InterruptedException e) {
                    return;
                } catch (Throwable t) {
                    System.err.println("generation worker:");
                    t.printStackTrace();
                }
            }
        });
    }

    /** Enqueues the request for the generation worker (FIFO) and waits for it to finish;
     *  rejects with 503 + Retry-After when the queue is full. */
    private static void runQueued(HttpExchange exchange, Runnable work) throws IOException {
        CountDownLatch done = new CountDownLatch(1);
        Runnable job = () -> {
            try {
                work.run();
            } finally {
                done.countDown();
            }
        };
        if (!GENERATION_QUEUE.offer(job)) {
            exchange.getResponseHeaders().set("Retry-After", String.valueOf(Math.max(1, 2 * (RuntimeFlags.SERVER_QUEUE + 1))));
            sendError(exchange, 503, "Server busy: " + RuntimeFlags.SERVER_QUEUE + " requests already queued");
            return;
        }
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        if (exchange.getResponseCode() == -1) {
            // the job finished without ever answering (escaped exception): never leave the
            // client waiting on an unanswered exchange
            sendErrorQuietly(exchange, 500, "Internal server error");
        }
    }


    /** Reads the request body, bounded by llama.serverMaxBodyMB; returns null after sending 413
     *  when the body exceeds the limit (callers must return immediately on null). */
    private static byte[] readBody(HttpExchange exchange) throws IOException {
        byte[] body = exchange.getRequestBody().readNBytes((int) RuntimeFlags.SERVER_MAX_BODY_BYTES + 1);
        if (body.length > RuntimeFlags.SERVER_MAX_BODY_BYTES) {
            sendError(exchange, 413, "Request body exceeds " + (RuntimeFlags.SERVER_MAX_BODY_BYTES >> 20) + " MB");
            return null;
        }
        return body;
    }

    private static void sendErrorQuietly(HttpExchange exchange, int status, String message) {
        try {
            sendError(exchange, status, message);
        } catch (IOException e) {
            System.err.println("client connection lost: " + e);
        } catch (RuntimeException e) {
            // response headers already sent (e.g. mid-stream failure) — nothing more we can do here
            System.err.println("response already committed, dropping error (" + status + " " + message + "): " + e);
        }
    }

    private static void sendError(HttpExchange exchange, int status, String message) throws IOException {
        sendJson(exchange, status, Map.of("error", errorPayload(status, message)));
    }

    private static String errorMessage(Throwable e) {
        return e.getMessage() == null ? e.toString() : e.getMessage();
    }

    private static Map<String, Object> errorPayload(int status, String message) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", status == 404 ? "not_found_error" : status >= 500 ? "internal_error" : "invalid_request_error");
        error.put("param", null);
        error.put("code", null);
        return error;
    }

    private static void logRequest(HttpExchange exchange) {
        System.err.printf("%s %s from %s%n",
                exchange.getRequestMethod(),
                exchange.getRequestURI(),
                exchange.getRemoteAddress());
    }

    private static void addCommonHeaders(HttpExchange exchange) {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Headers", "authorization, content-type, x-request-id");
        headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        String requestId = exchange.getRequestHeaders().getFirst("X-Request-ID");
        if (requestId != null) headers.set("X-Request-ID", requestId);
    }

    private static boolean handleOptions(HttpExchange exchange) throws IOException {
        if (!"OPTIONS".equals(exchange.getRequestMethod())) return false;
        exchange.sendResponseHeaders(204, -1);
        exchange.close();
        return true;
    }

    /** A slow (or stopped) streaming client blocks the worker's SSE write once the TCP window
     *  fills; the JDK server has no write timeout, so one such client would wedge the single
     *  generation worker forever. Every streaming write is tracked here, and a reaper closes
     *  the exchange when one write stalls past llama.serverWriteTimeout seconds — the blocked
     *  write then fails with an IOException, which aborts that generation cleanly. */
    private static final Set<StreamWatch> ACTIVE_STREAMS = ConcurrentHashMap.newKeySet();

    private static final class StreamWatch {
        final HttpExchange exchange;
        volatile long writeStartNanos; // 0 = no write in flight

        StreamWatch(HttpExchange exchange) {
            this.exchange = exchange;
        }
    }

    private static void startStreamReaper() {
        Thread.ofPlatform().name("sse-write-reaper").daemon(true).start(() -> {
            while (true) {
                try {
                    Thread.sleep(5_000);
                } catch (InterruptedException e) {
                    return;
                }
                long now = System.nanoTime();
                for (StreamWatch watch : ACTIVE_STREAMS) {
                    long start = watch.writeStartNanos;
                    if (start != 0 && now - start > RuntimeFlags.SERVER_WRITE_STALL_NANOS) {
                        System.err.println("closing stalled streaming client " + watch.exchange.getRemoteAddress());
                        ACTIVE_STREAMS.remove(watch);
                        watch.exchange.close();
                    }
                }
            }
        });
    }

    /** A live Server-Sent-Events response. Owns the byte encoding, the per-frame flush, and the
     *  checked-to-unchecked exception bridge so callers — including streaming sinks invoked deep
     *  in the generation loop — just call {@link #emit}/{@link #done}. Registered with the stall
     *  reaper for the life of the stream. */
    private static final class SseStream implements AutoCloseable {
        private final OutputStream out;
        private final StreamWatch watch;

        private SseStream(OutputStream out, StreamWatch watch) {
            this.out = out;
            this.watch = watch;
        }

        /** A {@code data:} frame carrying one JSON value. */
        void emit(Object value) {
            frame("data: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** A named SSE event ({@code event:} line + {@code data:} frame) — the Responses API. */
        void emit(String event, Object value) {
            frame("event: " + event + "\ndata: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** The terminal {@code [DONE]} sentinel. */
        void done() {
            frame("data: [DONE]\n\n");
        }

        private void frame(String text) {
            try {
                out.write(text.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                throw new UncheckedIOException(e); // client gone; unwound by guarded()
            }
        }

        @Override
        public void close() throws IOException {
            ACTIVE_STREAMS.remove(watch);
            out.close();
        }
    }

    private static SseStream beginStream(HttpExchange exchange) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0);
        StreamWatch watch = new StreamWatch(exchange);
        ACTIVE_STREAMS.add(watch);
        OutputStream watched = new FilterOutputStream(exchange.getResponseBody()) {
            @Override
            public void write(byte[] b, int off, int len) throws IOException {
                watch.writeStartNanos = System.nanoTime();
                try {
                    out.write(b, off, len);
                } finally {
                    watch.writeStartNanos = 0;
                }
            }
        };
        return new SseStream(watched, watch);
    }
}
