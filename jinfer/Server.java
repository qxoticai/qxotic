package com.llama4j;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

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
import java.util.concurrent.Executors;
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
 * Pure transport and orchestration: all inference goes through {@link Llama#generate}.
 */
final class Server {

    private Server() {
    }

    /** Boots the server and returns it (port 0 binds an ephemeral port — the integration
     *  test reads the actual one from the returned server). */
    static HttpServer run(Llama model, Options options) throws IOException {
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
        jsonRoute(server, "/props", null, request -> {
            Llama.Configuration config = model.configuration();
            return Map.of(
                    "model", options.modelPath().getFileName().toString(),
                    "n_ctx", config.contextLength,
                    "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                    "n_embd", config.embeddingLength,
                    "n_vocab", config.vocabularySize,
                    "n_layer", config.numberOfLayers,
                    "prompt_cache", promptCache == null ? Map.of("enabled", false) : promptCache.stats());
        });
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
        if (RuntimeFlags.PROMPT_CACHE) {
            promptCache = new PromptCache(model.configuration(), CacheStore.inMemory());
            System.out.printf("Prompt cache enabled: budget=%d MB%n",
                    RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES >> 20);
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
                    request = asObject(Json.parse(new String(raw, StandardCharsets.UTF_8)), "request");
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
            request = asObject(Json.parse(new String(body, StandardCharsets.UTF_8)), "request");
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

    private static void handleChatCompletion(HttpExchange exchange, Llama model, Options options) throws IOException {
        handleGenerationPost(exchange, "chatcmpl-", request -> {
            validateChatRequest(request);
            validateGenerationParams(request, options);
        }, (request, id) -> {
            List<Object> messages = asArray(request.get("messages"), "messages");
            String modelId = requestModelId(request, options);
            if (booleanValue(request.get("stream"), false)) {
                streamChatCompletion(exchange, model, options, request, messages, modelId, id);
            } else {
                GenerationResult result = generateChat(model, options, request, messages, null, null, null);
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, chatCompletionResponse(id, modelId, result));
            }
        });
    }

    private static void handleCompletion(HttpExchange exchange, Llama model, Options options) throws IOException {
        handleGenerationPost(exchange, "cmpl-", request -> {
            validateGenerationParams(request, options);
            Options.require(!completionPrompt(request).isBlank(), "prompt must not be empty");
        }, (request, id) -> {
            String prompt = completionPrompt(request);
            String modelId = requestModelId(request, options);
            if (booleanValue(request.get("stream"), false)) {
                streamCompletion(exchange, model, options, request, prompt, modelId, id);
            } else {
                GenerationResult result = generateCompletion(model, options, request, prompt, null, null, null);
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, completionResponse(id, modelId, result));
            }
        });
    }

    private static String requestModelId(Map<String, Object> request, Options options) {
        return stringValue(request.get("model"), options.modelPath().getFileName().toString());
    }

    private static String completionPrompt(Map<String, Object> request) {
        Object promptValue = request.get("prompt");
        return promptValue instanceof List<?> prompts
                ? prompts.stream().map(String::valueOf).collect(Collectors.joining("\n"))
                : stringValue(promptValue, "");
    }

    private static void handleResponse(HttpExchange exchange, Llama model, Options options) throws IOException {
        handleGenerationPost(exchange, "resp-", request -> {
            normalizeResponseRequest(request);
            validateGenerationParams(request, options);
            Options.require(!responseInputMessages(request).isEmpty(), "input must not be empty");
        }, (request, id) -> {
            List<Object> messages = responseInputMessages(request);
            String modelId = requestModelId(request, options);
            if (booleanValue(request.get("stream"), false)) {
                streamResponse(exchange, model, options, request, messages, modelId, id);
            } else {
                GenerationResult result = generateChat(model, options, request, messages, null, null, null);
                setTimingHeader(exchange, result);
                sendJson(exchange, 200, responseResponse(id, modelId, result));
            }
        });
    }

    private interface SseBody {
        void run() throws IOException;
    }

    /** Runs an SSE stream body; a failure after the SSE headers were already sent is delivered
     *  as an in-band error event followed by [DONE], so clients terminate instead of waiting on
     *  a silently dead stream. */
    private static void streamGuarded(OutputStream out, SseBody body) throws IOException {
        try {
            body.run();
        } catch (UncheckedIOException e) {
            throw e.getCause(); // client connection lost mid-stream
        } catch (RuntimeException e) {
            writeSse(out, Map.of("error", errorPayload(400, errorMessage(e))));
            out.write(SSE_DONE);
        }
    }

    private static void streamChatCompletion(HttpExchange exchange, Llama model, Options options, Map<String, Object> request,
                                             List<Object> messages, String modelId, String id) throws IOException {
        try (OutputStream out = beginStream(exchange)) {
            streamGuarded(out, () -> {
                writeSse(out, chatCompletionChunk(id, modelId, Map.of("role", "assistant"), null));
                boolean forcedTool = forcedToolChoice(request) != null;
                // Any tool-enabled turn can emit a call inline as ordinary content tokens (the
                // <|tool_call_start|>/<|tool_call_end|> markers are dropped from the stream, but
                // the `[name(args)]` body is plain text), and whether it WAS a call is only known
                // once generation ends. So we defer content for every tool-enabled turn: hold it
                // back, then emit structured tool_calls if any were detected, else flush the
                // buffered content as a single delta. Reasoning still streams live.
                boolean deferContent = forcedTool || hasUsableTools(request);
                Usage usageCounts = forcedTool ? null : new Usage();
                Consumer<String> reasoningSink = forcedTool ? null : sseDeltaSink(out, id, modelId, "reasoning_content", usageCounts);
                StringBuilder buffered = deferContent && !forcedTool ? new StringBuilder() : null;
                Consumer<String> contentSink;
                if (!deferContent) contentSink = sseDeltaSink(out, id, modelId, "content", usageCounts);
                else if (buffered != null) contentSink = buffered::append; // auto turn: hold content, resolve at end
                else contentSink = null;                                   // forced turn: suppress content entirely
                GenerationResult result = generateChat(model, options, request, messages, contentSink, reasoningSink, usageCounts);
                if (deferContent) {
                    Map<String, Object> delta = !result.toolCalls().isEmpty()
                            ? Map.of("tool_calls", toolCallDeltas(result.toolCalls()))
                            : Map.of("content", buffered != null ? buffered.toString() : result.text());
                    writeSse(out, chatCompletionChunk(id, modelId, delta, null));
                }
                endStream(out, request, result,
                        chatCompletionChunk(id, modelId, Map.of(), result.finishReason()),
                        chatCompletionChunk(id, modelId, Map.of(), null));
            });
        }
    }

    /** Final stream sequence shared by chat and completions: finish chunk carrying usage, the
     *  stream_options usage-only chunk when requested, then [DONE]. */
    private static void endStream(OutputStream out, Map<String, Object> request, GenerationResult result,
                                  Map<String, Object> finalChunk, Map<String, Object> usageOnlyChunk) throws IOException {
        finalChunk.put("usage", usage(result));
        writeSse(out, finalChunk);
        if (includeUsage(request)) {
            usageOnlyChunk.put("choices", List.of());
            usageOnlyChunk.put("usage", usage(result));
            writeSse(out, usageOnlyChunk);
        }
        out.write(SSE_DONE);
    }

    /** OpenAI stream_options: {"include_usage": true} requests an extra usage-only chunk after the final chunk. */
    @SuppressWarnings("unchecked")
    private static boolean includeUsage(Map<String, Object> request) {
        return request.get("stream_options") instanceof Map<?, ?> so && Boolean.TRUE.equals(((Map<String, Object>) so).get("include_usage"));
    }

    private static void streamCompletion(HttpExchange exchange, Llama model, Options options, Map<String, Object> request,
                                         String prompt, String modelId, String id) throws IOException {
        try (OutputStream out = beginStream(exchange)) {
            streamGuarded(out, () -> {
                Usage usageCounts = new Usage();
                GenerationResult result = generateCompletion(model, options, request, prompt, text -> {
                    try {
                        Map<String, Object> chunk = completionChunk(id, modelId, text, null);
                        chunk.put("usage", chunkUsage(usageCounts));
                        writeSse(out, chunk);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }, null, usageCounts);
                endStream(out, request, result,
                        completionChunk(id, modelId, "", result.finishReason()),
                        completionChunk(id, modelId, "", null));
            });
        }
    }

    private static void streamResponse(HttpExchange exchange, Llama model, Options options, Map<String, Object> request,
                                       List<Object> messages, String modelId, String id) throws IOException {
        try (OutputStream out = beginStream(exchange)) {
            streamGuarded(out, () -> {
                writeSseEvent(out, "response.created", Map.of(
                        "type", "response.created",
                        "response", responseEnvelope(id, modelId, "in_progress", List.of(), null)));
                String itemId = "msg_" + id;
                writeSseEvent(out, "response.output_item.added", Map.of(
                        "type", "response.output_item.added",
                        "output_index", 0,
                        "item", responseMessageItem(itemId, "in_progress", "")));
                Usage usageCounts = new Usage();
                GenerationResult result = generateChat(model, options, request, messages, text -> {
                    try {
                        Map<String, Object> delta = new LinkedHashMap<>();
                        delta.put("type", "response.output_text.delta");
                        delta.put("item_id", itemId);
                        delta.put("output_index", 0);
                        delta.put("content_index", 0);
                        delta.put("delta", text);
                        delta.put("usage", chunkUsage(usageCounts));
                        writeSseEvent(out, "response.output_text.delta", delta);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                }, null, usageCounts);
                writeSseEvent(out, "response.output_text.done", Map.of(
                        "type", "response.output_text.done",
                        "item_id", itemId,
                        "output_index", 0,
                        "content_index", 0,
                        "text", result.text()));
                writeSseEvent(out, "response.output_item.done", Map.of(
                        "type", "response.output_item.done",
                        "output_index", 0,
                        "item", responseMessageItem(itemId, "completed", result.text())));
                writeSseEvent(out, "response.completed", Map.of(
                        "type", "response.completed",
                        "response", responseResponse(id, modelId, result)));
                out.write(SSE_DONE);
            });
        }
    }

    private static Map<String, Object> chatCompletionChunk(String id, String modelId, Map<String, Object> delta, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("delta", delta);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "chat.completion.chunk");
        chunk.put("created", System.currentTimeMillis() / 1000);
        chunk.put("model", modelId);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    private static Consumer<String> sseDeltaSink(OutputStream out, String id, String modelId, String deltaKey, Usage usageCounts) {
        return text -> {
            try {
                Map<String, Object> chunk = chatCompletionChunk(id, modelId, Map.of(deltaKey, text), null);
                if (usageCounts != null) chunk.put("usage", chunkUsage(usageCounts));
                writeSse(out, chunk);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        };
    }

    /** Mutable per-request token counters, updated by the generation pipeline and read by the
     *  streaming sinks to attach running usage to delta chunks. */
    private static final class Usage {
        int promptTokens;
        int completionTokens;
        int cachedTokens;
    }

    private static Map<String, Object> chunkUsage(Usage usage) {
        return Map.of(
                "prompt_tokens", usage.promptTokens,
                "completion_tokens", usage.completionTokens,
                "total_tokens", usage.promptTokens + usage.completionTokens,
                "prompt_tokens_details", Map.of("cached_tokens", usage.cachedTokens));
    }

    private static Map<String, Object> chatCompletionResponse(String id, String modelId, GenerationResult result) {
        Map<String, Object> message = new LinkedHashMap<>();
        message.put("role", "assistant");
        message.put("content", result.toolCalls().isEmpty() ? result.text() : null);
        if (result.reasoning() != null) message.put("reasoning_content", result.reasoning());
        if (!result.toolCalls().isEmpty()) message.put("tool_calls", result.toolCalls());
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put("message", message);
        choice.put("finish_reason", result.finishReason());
        return Map.of(
                "id", id,
                "object", "chat.completion",
                "created", System.currentTimeMillis() / 1000,
                "model", modelId,
                "choices", List.of(choice),
                "usage", usage(result));
    }

    private static Map<String, Object> completionResponse(String id, String modelId, GenerationResult result) {
        return Map.of(
                "id", id,
                "object", "text_completion",
                "created", System.currentTimeMillis() / 1000,
                "model", modelId,
                "choices", List.of(Map.of("text", result.text(), "index", 0, "finish_reason", result.finishReason())),
                "usage", usage(result));
    }

    private static Map<String, Object> completionChunk(String id, String modelId, String text, String finishReason) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("text", text);
        choice.put("index", 0);
        choice.put("finish_reason", finishReason);
        Map<String, Object> chunk = new LinkedHashMap<>();
        chunk.put("id", id);
        chunk.put("object", "text_completion");
        chunk.put("created", System.currentTimeMillis() / 1000);
        chunk.put("model", modelId);
        chunk.put("choices", List.of(choice));
        return chunk;
    }

    private static Map<String, Object> usage(GenerationResult result) {
        return Map.of(
                "prompt_tokens", result.promptTokens(),
                "completion_tokens", result.completionTokens(),
                "total_tokens", result.promptTokens() + result.completionTokens(),
                "prompt_tokens_details", Map.of("cached_tokens", result.cachedTokens()));
    }

    /** llama.cpp-compatible timings extension: per-phase durations and rates. */
    private static Map<String, Object> timings(GenerationResult result) {
        Map<String, Object> timings = new LinkedHashMap<>();
        timings.put("prompt_n", result.promptTokens());
        timings.put("prompt_ms", Math.round(result.promptMillis() * 100.0) / 100.0);
        timings.put("prompt_per_second", result.promptMillis() > 0 ? Math.round(result.promptTokens() / result.promptMillis() * 100_000.0) / 100.0 : 0.0);
        timings.put("predicted_n", result.completionTokens());
        timings.put("predicted_ms", Math.round(result.predictedMillis() * 100.0) / 100.0);
        timings.put("predicted_per_second", result.predictedMillis() > 0 ? Math.round(result.completionTokens() / result.predictedMillis() * 100_000.0) / 100.0 : 0.0);
        timings.put("cached_n", result.cachedTokens());
        return timings;
    }

    private static Map<String, Object> responseUsage(GenerationResult result) {
        return Map.of(
                "input_tokens", result.promptTokens(),
                "output_tokens", result.completionTokens(),
                "total_tokens", result.promptTokens() + result.completionTokens());
    }

    private static Map<String, Object> responseResponse(String id, String modelId, GenerationResult result) {
        List<Map<String, Object>> output = result.toolCalls().isEmpty()
                ? List.of(responseMessageItem("msg_" + id, "completed", result.text()))
                : responseToolCallItems(result.toolCalls());
        Map<String, Object> response = responseEnvelope(id, modelId, "completed", output, responseUsage(result));
        return response;
    }

    private static Map<String, Object> responseEnvelope(String id, String modelId, String status,
                                                       List<Map<String, Object>> output, Map<String, Object> usage) {
        Map<String, Object> response = new LinkedHashMap<>();
        response.put("id", id);
        response.put("object", "response");
        response.put("created_at", System.currentTimeMillis() / 1000);
        response.put("status", status);
        response.put("model", modelId);
        response.put("output", output);
        response.put("parallel_tool_calls", false);
        response.put("tool_choice", "auto");
        response.put("usage", usage);
        return response;
    }

    private static Map<String, Object> responseMessageItem(String id, String status, String text) {
        return Map.of(
                "id", id,
                "type", "message",
                "status", status,
                "role", "assistant",
                "content", List.of(Map.of(
                        "type", "output_text",
                        "text", text,
                        "annotations", List.of())));
    }

    private static List<Map<String, Object>> responseToolCallItems(List<Map<String, Object>> toolCalls) {
        List<Map<String, Object>> output = new ArrayList<>();
        for (Map<String, Object> toolCall : toolCalls) {
            Map<String, Object> function = asObject(toolCall.get("function"), "tool_call.function");
            output.add(Map.of(
                    "id", stringValue(toolCall.get("id"), ""),
                    "type", "function_call",
                    "status", "completed",
                    "call_id", stringValue(toolCall.get("id"), ""),
                    "name", stringValue(function.get("name"), ""),
                    "arguments", stringValue(function.get("arguments"), "{}")));
        }
        return output;
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
        Map<String, Object> tool = asObject(value, "tool");
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
        String instructions = stringValue(request.get("instructions"), null);
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
        Map<String, Object> map = asObject(item, "input item");
        String type = stringValue(map.get("type"), "message");
        if ("function_call_output".equals(type)) {
            messages.add(Map.of(
                    "role", "tool",
                    "name", stringValue(map.get("call_id"), "tool"),
                    "content", stringValue(map.get("output"), "")));
            return;
        }
        String role = stringValue(map.get("role"), "user");
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
        return stringValue(content, "");
    }

    private static void validateChatRequest(Map<String, Object> request) {
        List<Object> messages = asArray(request.get("messages"), "messages");
        Options.require(!messages.isEmpty(), "messages must not be empty");
        boolean substance = false;
        for (Object message : messages) {
            Map<String, Object> m = asObject(message, "message");
            String role = stringValue(m.get("role"), "");
            Options.require(List.of("system", "user", "assistant", "tool").contains(role),
                    "Invalid role: %s (must be system, user, assistant, or tool)", role);
            substance |= !messageContent(m.get("content")).isBlank()
                    || (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty());
        }
        Options.require(substance, "messages must contain at least one non-empty message");
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?,?> m) {
            String type = stringValue(m.get("type"), "");
            Options.require("json_object".equals(type) || "text".equals(type),
                    "Unsupported response_format type: %s (only json_object and text are supported)", type);
            if ("json_object".equals(type)) {
                boolean hasJsonHint = false;
                for (Object message : messages) {
                    Map<String, Object> msg = asObject(message, "message");
                    String role = stringValue(msg.get("role"), "");
                    String content = messageContent(msg.get("content"));
                    if (("system".equals(role) || "user".equals(role)) && content.toLowerCase().contains("json"))
                        hasJsonHint = true;
                }
                Options.require(hasJsonHint,
                        "response_format json_object requires the word 'json' in a system or user message");
            }
        }
        Object tools = request.get("tools");
        if (tools != null) {
            List<Object> toolList = asArray(tools, "tools");
            for (Object value : toolList) validateTool(value);
        }
        Object toolChoice = request.get("tool_choice");
        if (toolChoice instanceof String s) {
            Options.require(List.of("auto", "none", "required").contains(s), "tool_choice must be auto, none, required, or a function choice object");
        } else if (toolChoice instanceof Map<?, ?> map) {
            Options.require("function".equals(map.get("type")), "Only function tool_choice objects are supported");
            Object function = map.get("function");
            Options.require(function instanceof Map<?, ?> fn && fn.get("name") instanceof String, "tool_choice.function.name is required");
        } else if (toolChoice != null) {
            throw new IllegalArgumentException("tool_choice must be a string or object");
        }
    }

    private static void validateTool(Object value) {
        Map<String, Object> tool = asObject(value, "tool");
        Options.require("function".equals(stringValue(tool.get("type"), "function")), "Only function tools are supported");
        Map<String, Object> function = asObject(tool.get("function"), "tool.function");
        Options.require(function.get("name") instanceof String name && !name.isBlank(), "tool.function.name is required");
    }
    private static GenerationResult generateChat(Llama model, Options options, Map<String, Object> request,
                                                  List<Object> messages,
                                                  Consumer<String> onText, Consumer<String> onReasoning,
                                                  Usage usageCounts) {
        LFMTokenizer tokenizer = model.tokenizer();
        LFMChatFormat chatFormat = new LFMChatFormat(tokenizer);
        List<Integer> promptTokens;
        String template = tokenizer.chatTemplate();
        if (!template.isEmpty()) {
            // Use the model's Jinja chat template
            var ctx = new LinkedHashMap<String,Object>();
            ctx.put("messages", messages);
            ctx.put("add_generation_prompt", true);
            String bos = specialTokenString(tokenizer, "<bos>");
            String eos = specialTokenString(tokenizer, "<eos>");
            ctx.put("bos_token", bos != null ? bos : specialTokenString(tokenizer, "<|startoftext|>"));
            ctx.put("eos_token", eos != null ? eos : specialTokenString(tokenizer, "<|endoftext|>"));
            // Always bind `tools` (null when absent), matching HuggingFace apply_chat_template:
            // many templates guard with `if tools is not none` WITHOUT an `is defined` check, and
            // an undefined variable is "not none", so leaving it unset makes them emit a spurious
            // empty tool block (e.g. Mistral's [AVAILABLE_TOOLS]).
            ctx.put("tools", hasUsableTools(request) ? asArray(request.get("tools"), "tools") : null);
            boolean thinking = requestThink(request, options);
            ctx.put("enable_thinking", thinking);
            ctx.put("preserve_thinking", false);
            if (request.get("chat_template_kwargs") instanceof Map<?,?> kwargs) {
                ctx.putAll((Map<String,Object>) kwargs);
            }
            String rendered = JinjaRenderer.render(template, ctx);
            promptTokens = tokenizer.encodeWithSpecialTokens(rendered);
            seedForcedToolCall(tokenizer, request, promptTokens);
            if (System.getProperty("llama.debugPrompt") != null) {
                System.err.println("[prompt] " + tokenizer.decode(promptTokens));
            }
        } else {
            // Fallback: hardcoded ChatML format for models without a Jinja template
            promptTokens = new ArrayList<>();
            promptTokens.add(chatFormat.beginOfSentence);
            List<Object> turns = messages;
            String systemText = null;
            if (!messages.isEmpty()) {
                Map<String, Object> first = asObject(messages.getFirst(), "message");
                if ("system".equals(stringValue(first.get("role"), ""))) {
                    systemText = chatMessageContent(first);
                    turns = messages.subList(1, messages.size());
                }
            }
            if (hasUsableTools(request)) {
                promptTokens.addAll(encodeToolsSystemMessage(tokenizer, chatFormat, request, systemText));
            } else if (systemText != null) {
                promptTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, systemText)));
            }
            for (Object value : turns) {
                Map<String, Object> message = asObject(value, "message");
                LFMChatFormat.Role role = LFMChatFormat.Role.of(stringValue(message.get("role"), "user"));
                promptTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(role, chatMessageContent(message))));
            }
            promptTokens.addAll(chatFormat.encodeGenerationPrompt());
            if (!requestThink(request, options)) chatFormat.appendThinkSurrogate(promptTokens);
            seedForcedToolCall(tokenizer, request, promptTokens);
            if (System.getProperty("llama.debugPrompt") != null) {
                System.err.println("[prompt] " + tokenizer.decode(promptTokens));
            }
        }
        Ingestion resumed = matchChatSession(request, messages, chatFormat, model.tokenizer(), options);
        Ingestion ingestion = resumed != null ? resumed : Ingestion.of(model.createNewState(), 0, promptTokens);
        GenerationResult result = generateResponse(model, options, request, promptTokens, ingestion, chatFormat.getStopTokens(), onText, onReasoning, usageCounts);
        saveChatSession(model, request, messages, ingestion, result);
        return hasUsableTools(request) ? withParsedToolCalls(model, result, request) : result;
    }

    /** Remember the live state for instant resume of the next turn; only clean stop-terminated
     *  text replies are resumable (tool calls and aborted/length-capped replies are not). */
    private static void saveChatSession(Llama model, Map<String, Object> request, List<Object> messages,
                                        Ingestion ingestion, GenerationResult result) {
        if (!"stop".equals(result.finishReason()) || !result.toolCalls().isEmpty() || result.text().isBlank()) {
            return;
        }
        int position = ingestion.prefillPositions() + result.completionTokens();
        List<String> keys = new ArrayList<>(messages.size());
        for (Object message : messages) keys.add(messageKey(message));
        chatSession = new ChatSession(ingestion.state(), position, keys, result.text(), toolsKey(request));
    }

    private static GenerationResult generateCompletion(Llama model, Options options, Map<String, Object> request, String prompt,
                                                        Consumer<String> onText, Consumer<String> onReasoning,
                                                        Usage usageCounts) {
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> promptTokens = options.rawPrompt() ? model.tokenizer().encodeWithSpecialTokens(prompt) : new ArrayList<>(model.tokenizer().encode(prompt));
        Ingestion ingestion = Ingestion.of(model.createNewState(), 0, promptTokens);
        return generateResponse(model, options, request, promptTokens, ingestion, chatFormat.getStopTokens(), onText, onReasoning, usageCounts);
    }


    /** Prompt cache instance; created in runServer when enabled, null in CLI modes. */
    private static PromptCache promptCache;

    /** What to feed the generation loop: which state, from which position, with which tokens.
     *  Fresh requests: new state, position 0, the full prompt. Resumed chat sessions: the live
     *  state mid-context with only the delta (turn close + new messages + generation prompt). */
    private record Ingestion(Llama.State state, int startPosition, List<Integer> tokens, int prefillPositions) {
        /** prefillPositions: see {@link Llama#prefillPositions}. */
        static Ingestion of(Llama.State state, int startPosition, List<Integer> tokens) {
            return new Ingestion(state, startPosition, tokens, Llama.prefillPositions(state, startPosition, tokens));
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
    private record ChatSession(Llama.State state, int position, List<String> messageKeys, String reply, String toolsKey) {}
    private static ChatSession chatSession;

    private static String messageKey(Object message) {
        Map<String, Object> map = asObject(message, "message");
        return stringValue(map.get("role"), "user") + "\u0000" + chatMessageContent(map);
    }

    private static String toolsKey(Map<String, Object> request) {
        if (!hasUsableTools(request)) return "";
        return Json.stringify(request.get("tools")) + "|" + Json.stringify(request.get("tool_choice"));
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
        Map<String, Object> echo = asObject(messages.get(n), "message");
        if (!"assistant".equals(stringValue(echo.get("role"), ""))
                || !chatMessageContent(echo).strip().equals(s.reply().strip())) return null;
        chatSession = null; // taken: a failure mid-resume must not leave a stale session behind
        // the session's un-ingested stop token (state.latestToken) closes the assistant turn;
        // generate() prepends it, we add the template newline and the new turns
        List<Integer> delta = new ArrayList<>(tokenizer.encode("\n"));
        for (int i = n + 1; i < messages.size(); i++) {
            Map<String, Object> message = asObject(messages.get(i), "message");
            LFMChatFormat.Role role = LFMChatFormat.Role.of(stringValue(message.get("role"), "user"));
            delta.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(role, chatMessageContent(message))));
        }
        delta.addAll(chatFormat.encodeGenerationPrompt());
        if (!requestThink(request, options)) chatFormat.appendThinkSurrogate(delta);
        seedForcedToolCall(tokenizer, request, delta);
        return Ingestion.of(s.state(), s.position(), delta);
    }

    /**
     * Server generation driver over {@link Engine#generate}. The ingestion plan says where to
     * start: a fresh state at position 0 (with the prompt cache plugged in through
     * {@link Llama.GenerationHooks}), or a resumed chat session mid-context — in which case the
     * cache is bypassed (its tree is keyed by full-stream positions). cachedOut[0] tracks the
     * resumed-prefix length for the streaming usage counters.
     */
    private static GenerationResult runServerGeneration(Llama model, Ingestion ingestion, Engine.Params params,
                                                        Engine.Listener listener, int[] cachedOut, boolean warm) {
        PromptCache cache = promptCache;
        Llama.State state = ingestion.state();
        if (cache == null || ingestion.startPosition() > 0) {
            return Engine.generate(model, state, ingestion.startPosition(), ingestion.tokens(), params, listener, Llama.GenerationHooks.NONE);
        }

        /** Wires the prompt cache into the generation loop: lookup + restore on resume (a
         *  sparse F32 checkpoint or a dense bx landing anywhere rows are retained), the bx
         *  harvest installed around prompt-ingest chunks (cleared before decode), chunks
         *  clamped so the frontier lands exactly on checkpoint positions (end of prompt L-1,
         *  and the divergence point found by lookup — checkpointing it once makes repeats
         *  bit-exact), commits per ingested chunk past the matched region, and an
         *  end-of-generation commit with a conv checkpoint so multi-turn resume is exact. */
        class CacheHooks implements Llama.GenerationHooks {
            boolean caching = true;
            int prefillLength;
            int matchedPos;
            int committedTo;       // commits cover (matchedPos-or-later, committedTo]
            final int[] checkpoints = new int[2];
            int checkpointCount;
            int[] stream;
            int frontier;

            @Override
            public int resumePosition(int[] stream, int prefillLength) {
                this.prefillLength = prefillLength;
                PromptCache.Match match = cache.lookup(stream, prefillLength);
                this.matchedPos = match.matchedPos();
                this.committedTo = match.matchedPos();
                int resume = match.resumePos();
                cachedOut[0] = resume;
                cache.restore(match, state);
                cache.pin(match); // even on a cold resume: checkpoints attach into matched nodes
                state.convHarvest = cache.beginHarvest(prefillLength, warm);
                for (int p : new int[]{Math.min(matchedPos, prefillLength - 1), prefillLength - 1}) {
                    if (p > resume && p > 0 && (checkpointCount == 0 || checkpoints[checkpointCount - 1] != p)) {
                        checkpoints[checkpointCount++] = p;
                    }
                }
                return resume;
            }

            @Override
            public int clampChunk(int position, int chunkLength) {
                int clamp = chunkLength;
                for (int i = 0; i < checkpointCount; i++) {
                    if (checkpoints[i] > position) {
                        clamp = Math.min(clamp, checkpoints[i] - position);
                        break;
                    }
                }
                if (caching && cache.anySwaKv) {
                    clamp = Math.min(clamp, Math.max(matchedPos, committedTo) + cache.swaStride - position);
                }
                return clamp;
            }

            private boolean isCheckpoint(int position) {
                for (int i = 0; i < checkpointCount; i++) {
                    if (checkpoints[i] == position) return true;
                }
                return false;
            }

            @Override
            public void afterIngest(int[] stream, int position) {
                this.stream = stream;
                this.frontier = position;
                if (!caching) {
                    return;
                }
                if (position <= matchedPos) {
                    // re-ingesting cached tokens (resume < divergence): KV is already in the
                    // tree, only the conv checkpoint at the divergence is new
                    if (isCheckpoint(position)) {
                        cache.attachCheckpoint(stream, position, state);
                    }
                    return;
                }
                boolean commit = position <= prefillLength                                       // prefill chunk
                        || (cache.anySwaKv && position - committedTo >= cache.swaStride);        // SWA ring pressure
                if (commit) {
                    if (cache.commitSpan(stream, committedTo, position, state, isCheckpoint(position)) == null) {
                        caching = false;
                        return;
                    }
                    committedTo = position;
                }
            }

            @Override
            public void afterPrefill() {
                state.convHarvest = null; // decode chunks are never harvested
                cache.endHarvest();
            }
        }
        CacheHooks hooks = new CacheHooks();
        try {
            GenerationResult result = Engine.generate(model, state, 0, ingestion.tokens(), params, listener, hooks);
            // end-of-generation commit: the frontier sits at the last ingested token (stop
            // tokens are never ingested), checkpointed so the next turn resumes exactly there
            if (hooks.caching && hooks.frontier > hooks.committedTo) {
                cache.commitSpan(hooks.stream, hooks.committedTo, hooks.frontier, state, true);
            }
            return result;
        } finally {
            state.convHarvest = null; // error paths may skip afterPrefill
            cache.endHarvest();
            cache.unpinCurrent();
        }
    }

    /**
     * Pre-ingests --warm-prompt / -Dllama.promptCacheWarm files into the prompt cache with
     * FULLY DENSE bx retention and sticky (eviction-exempt) nodes: requests diverging at ANY
     * position inside a warmed prompt resume token-exact, with zero re-ingest. Each file is
     * warmed in two stream forms — chat-template system message and raw completion prompt.
     * Runs on the generation worker (cache blobs are confined to it) and blocks until done.
     */
    private static void warmPromptCache(Llama model, Options options) {
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
        java.util.concurrent.CountDownLatch done = new java.util.concurrent.CountDownLatch(1);
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

    private static void warmFile(Llama model, String file) throws IOException {
        String text = Files.readString(Path.of(file));
        LFMTokenizer tokenizer = model.tokenizer();
        List<Integer> chatForm;
        String template = tokenizer.chatTemplate();
        if (!template.isEmpty()) {
            var ctx = new LinkedHashMap<String,Object>();
            ctx.put("messages", List.of(Map.of("role", "system", "content", text)));
            ctx.put("add_generation_prompt", true);
            String bos = specialTokenString(tokenizer, "<bos>");
            String eos = specialTokenString(tokenizer, "<eos>");
            ctx.put("bos_token", bos != null ? bos : specialTokenString(tokenizer, "<|startoftext|>"));
            ctx.put("eos_token", eos != null ? eos : specialTokenString(tokenizer, "<|endoftext|>"));
            // bind tools=null (no tools on this path) so `if tools is not none` guards see a
            // defined value rather than an undefined that reads as "not none" — see the chat path.
            ctx.put("tools", null);
            ctx.put("enable_thinking", false);
            ctx.put("preserve_thinking", false);
            String rendered = JinjaRenderer.render(template, ctx);
            chatForm = tokenizer.encodeWithSpecialTokens(rendered);
        } else {
            LFMChatFormat chatFormat = new LFMChatFormat(tokenizer);
            chatForm = new ArrayList<>();
            chatForm.add(chatFormat.beginOfSentence);
            chatForm.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, text)));
        }
        List<Integer> rawForm = tokenizer.encode(text);
        Engine.Params params = new Engine.Params(Sampler.ARGMAX, 0, 0, new StopSpec(Set.of(), List.of()), false); // warm: no deadline
        for (List<Integer> tokens : List.of(chatForm, rawForm)) {
            long startNanos = System.nanoTime();
            Ingestion ingestion = Ingestion.of(model.createNewState(), 0, tokens);
            runServerGeneration(model, ingestion, params, new Engine.Listener(null, null, null), new int[1], true);
            System.out.printf("warm-prompt %s: %d tokens in %.1f s%n",
                    file, ingestion.prefillPositions(), (System.nanoTime() - startNanos) / 1e9);
        }
    }

    /** Builds a grammar cursor from request params: {@code grammar} (GBNF string) or
     *  {@code response_format: {type: "json_object"}}. Returns null when no constraint. */
    private static Grammar.Cursor buildGrammarCursor(LFMTokenizer tokenizer, Map<String, Object> request) {
        if (!Grammar.ENABLED) return null;
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

    /** Sampling-parameter validation shared by all endpoints; called on the HTTP handler thread
     *  (before queueing, and before any SSE headers) so invalid requests fail fast with a 400. */
    private static void validateGenerationParams(Map<String, Object> request, Options options) {
        Options.require(request.get("model") instanceof String name && !name.isBlank(), "model is required");
        if (request.get("model") instanceof String name) {
            String served = options.modelPath().getFileName().toString();
            Options.require(name.equalsIgnoreCase(served), "Unknown model: %s (this server serves %s)", name, served);
        }
        if ((request.containsKey("grammar") || request.containsKey("response_format"))
                && options.noGrammar()) {
            Options.require(false, "Grammar constraints disabled (--no-grammar)");
        }
        Options.require(intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        float temperature = floatValue(request.get("temperature"), options.temperature());
        Options.require(Float.isFinite(temperature) && 0 <= temperature && temperature <= 2, "Invalid argument: temperature must be within [0, 2]");
        float topp = floatValue(request.get("top_p"), options.topp());
        Options.require(Float.isFinite(topp) && 0 <= topp && topp <= 1, "Invalid argument: top_p must be within [0, 1]");
        Options.require(0 <= intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens()), "Invalid argument: max_tokens must be non-negative");
        Options.require(-1 <= intValue(request.get("reasoning_max_tokens"), -1), "Invalid argument: reasoning_max_tokens must be -1 (uncapped) or non-negative");
        longValue(request.get("seed"), 0); // type check only
        Options.require(!request.containsKey("logprobs") && !request.containsKey("top_logprobs"),
                "logprobs is not supported");
        Options.require(!request.containsKey("logit_bias"), "logit_bias is not supported");
        Options.require(!request.containsKey("frequency_penalty") && !request.containsKey("presence_penalty"),
                "frequency_penalty and presence_penalty are not supported");
    }

    /** Request fields to {@link Engine.Params}/{@link Engine.Listener}, then one engine pass
     *  through {@link #runServerGeneration}. Streaming counters mirror Engine's final usage:
     *  generated tokens are counted unless they are the trailing token stop removed from the
     *  result. */
    private static GenerationResult generateResponse(Llama model, Options options, Map<String, Object> request, List<Integer> promptTokens,
                                           Ingestion ingestion, Set<Integer> baseStopTokens, Consumer<String> onText,
                                           Consumer<String> onReasoning, Usage usageCounts) {
        float temperature = floatValue(request.get("temperature"), options.temperature());
        float topp = floatValue(request.get("top_p"), options.topp());
        long seed = longValue(request.get("seed"), options.seed());
        int maxTokens = intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens());
        // server-side completion-token ceiling: an unbounded (or oversized) request can never run
        // the worker past llama.serverMaxTokens; hitting it reports finish_reason "length"
        if (RuntimeFlags.SERVER_MAX_TOKENS > 0)
            maxTokens = maxTokens < 0 ? RuntimeFlags.SERVER_MAX_TOKENS : Math.min(maxTokens, RuntimeFlags.SERVER_MAX_TOKENS);
        // defense in depth: server requests were already checked by validateGenerationParams on
        // the handler thread; these guards keep the method safe for any future non-HTTP caller
        Options.require(intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        Options.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        StopSpec stops = stopSpec(request.get("stop"), baseStopTokens);
        boolean think = requestThink(request, options);
        Sampler sampler = Engine.configuredSampler(model, think, temperature, topp, seed);
        if (think) {
            // thinking models starve the answer under tight budgets: cap the think span,
            // by default at half the completion budget (request reasoning_max_tokens overrides;
            // -1 = uncapped)
            int reasoningBudget = intValue(request.get("reasoning_max_tokens"),
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
        Engine.Listener listener = new Engine.Listener(onToken, onText, onReasoning);
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

    /** Per the model's Jinja chat template: tools are rendered as a JSON array in the system
     *  message — {@code List of tools: [{...}, {...}]} — without any special-token markers. */
    private static List<Integer> encodeToolsSystemMessage(LFMTokenizer tokenizer, LFMChatFormat chatFormat,
                                                           Map<String, Object> request, String systemText) {
        String lead = (systemText == null || systemText.isBlank()) ? "" : systemText + "\n\n";
        StringBuilder sb = new StringBuilder();
        sb.append(lead).append("List of tools: [");
        List<Object> tools = asArray(request.get("tools"), "tools");
        for (int i = 0; i < tools.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append(modelFacingJson(tools.get(i)));
        }
        sb.append("]");
        String hints = toolChoiceHints(request);
        if (!hints.isEmpty()) sb.append("\n").append(hints);
        return chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, sb.toString()));
    }

    /**
     * json.dumps-style JSON (", " and ": " separators) for text the MODEL reads. LFM2.5's
     * training data renders tool lists through Python/Jinja tojson, which spaces separators —
     * feeding the same content COMPACT puts it out of distribution and the model disowns its
     * tools ("I don't have access to real-time data"). API responses stay compact (Json).
     */
    private static String modelFacingJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeModelFacingJson(sb, value);
        return sb.toString();
    }

    private static void writeModelFacingJson(StringBuilder sb, Object value) {
        switch (value) {
            case null -> sb.append("null");
            case String s -> sb.append(Json.stringify(s)); // a bare string stringifies quoted+escaped
            case Number n -> sb.append(n);
            case Boolean b -> sb.append(b);
            case Map<?, ?> map -> {
                sb.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) sb.append(", ");
                    first = false;
                    writeModelFacingJson(sb, String.valueOf(entry.getKey()));
                    sb.append(": ");
                    writeModelFacingJson(sb, entry.getValue());
                }
                sb.append('}');
            }
            case Iterable<?> iterable -> {
                sb.append('[');
                boolean first = true;
                for (Object item : iterable) {
                    if (!first) sb.append(", ");
                    first = false;
                    writeModelFacingJson(sb, item);
                }
                sb.append(']');
            }
            default -> writeModelFacingJson(sb, String.valueOf(value));
        }
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
        return choice.isEmpty() ? TC_START : TC_START + "[" + choice;
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

    private static String toolChoiceHints(Map<String, Object> request) {
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "required".equals(s)) {
            return "A tool call is required.";
        }
        if (choice instanceof Map<?, ?> map && map.get("function") instanceof Map<?, ?> fn && fn.get("name") != null) {
            return "Call the tool named \"" + fn.get("name") + "\".";
        }
        return "";
    }

    private static String chatMessageContent(Map<String, Object> message) {
        String role = stringValue(message.get("role"), "user");
        if ("tool".equals(role)) {
            String name = stringValue(message.get("name"), stringValue(message.get("tool_call_id"), "tool"));
            return "Tool result from " + name + ":\n" + messageContent(message.get("content"));
        }
        String content = messageContent(message.get("content"));
        Object toolCalls = message.get("tool_calls");
        if (toolCalls instanceof List<?> calls && !calls.isEmpty()) {
            String callsText = "Tool calls made:\n" + modelFacingJson(calls);
            return content.isEmpty() ? callsText : content + "\n" + callsText;
        }
        Object functionCall = message.get("function_call");
        if (functionCall instanceof Map<?, ?> call) {
            String callText = "Function call made:\n" + modelFacingJson(call);
            return content.isEmpty() ? callText : content + "\n" + callText;
        }
        return content;
    }

    private static GenerationResult withParsedToolCalls(Llama model, GenerationResult result, Map<String, Object> request) {
        // Parse from the FULL generated text (think span included). result.text() is the
        // think-STRIPPED content, so a call the model emits before it closes </think> (or in an
        // unterminated think span) would be deleted before we ever see it. Decoding the raw
        // response tokens renders special tokens (<|tool_call_start|>, <think>) as literal text.
        String text = forcedToolCallPrefix(request);
        String decoded = model.tokenizer().decode(result.tokens());
        text += !decoded.strip().isEmpty() ? decoded
                : (result.reasoning() != null ? result.reasoning() + "\n" : "") + result.text();
        boolean debug = System.getProperty("llama.debugToolCalls") != null;
        List<Map<String, Object>> toolCalls = parseToolCalls(text, toolNames(request));
        if (toolCalls.isEmpty()) {
            // A reply that smells like an attempted call (markers or a "name" key) but parsed to
            // nothing is the diagnostic we care about; surface it even without the debug flag.
            String t = text.strip();
            if (t.contains(TC_START) || t.contains("\"name\"")) {
                System.err.println("[tool-parse] tools offered but parsed 0 calls from reply: " + t.replace("\n", "\\n"));
            }
            return result;
        }
        if (debug) System.err.println("[tool-parse] found " + toolCalls.size() + " call(s) in: " + text.strip().replace("\n", "\\n"));
        int marker = text.indexOf(TC_START);
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

    private static final String TC_START = "<|tool_call_start|>";
    private static final String TC_END = "<|tool_call_end|>";

    /** All {@code <|tool_call_start|>...<|tool_call_end|>} blocks parsed per the format LFM2.5
     *  was trained on (reference: SGLang Lfm2Detector): each block holds either a Pythonic
     *  call list {@code [f(a=1), g(b='x')]} (or a single bare call) or a JSON array/object of
     *  {@code {name, arguments}}. Calls naming a function absent from {@code knownTools} are
     *  dropped (a non-empty set validates; empty = accept all). */
    private static List<Map<String, Object>> parseNativeToolCalls(String text, Set<String> knownTools) {
        List<Map<String, Object>> calls = new ArrayList<>();
        int pos = 0;
        while (true) {
            int start = text.indexOf(TC_START, pos);
            if (start < 0) break;
            int end = text.indexOf(TC_END, start + TC_START.length());
            if (end < 0) break; // truncated mid-call: nothing reliable to parse
            String content = text.substring(start + TC_START.length(), end).strip();
            pos = end + TC_END.length();
            for (Map<String, Object> call : parseToolCallBlock(content)) {
                if (!knownTools.isEmpty() && !knownTools.contains(stringValue(call.get("name"), ""))) {
                    System.err.println("dropping tool call to undefined function: " + call.get("name"));
                    continue;
                }
                Map<String, Object> normalized = normalizeToolCall(call, calls.size());
                if (normalized != null) calls.add(normalized);
            }
        }
        return calls;
    }

    /** One block's content as raw {@code {name, arguments}} maps: JSON-looking content parses
     *  as JSON, anything else as the Pythonic form; malformed content yields no calls. */
    private static List<Map<String, Object>> parseToolCallBlock(String content) {
        if (content.startsWith("{") || content.startsWith("[{")) {
            try {
                Object parsed = Json.parse(content);
                List<?> list = parsed instanceof List<?> l ? l : List.of(parsed);
                List<Map<String, Object>> calls = new ArrayList<>();
                for (Object value : list) calls.add(asObject(value, "tool call"));
                return calls;
            } catch (RuntimeException e) {
                // fall through: '{' can also open a Pythonic dict literal
            }
        }
        try {
            return new PythonicCalls(content).parse();
        } catch (RuntimeException e) {
            System.err.println("unparseable tool call block: " + e.getMessage());
            return List.of();
        }
    }

    /**
     * Recursive-descent parser for the Pythonic tool-call syntax: {@code [name(k=v, ...), ...]}
     * or a single bare call; values are Python literals — strings (either quote, backslash
     * escapes), numbers, True/False/None, and nested lists/tuples/dicts — converted to their
     * JSON-ready Java shapes. Positional arguments are skipped (matching SGLang); anything
     * else malformed throws.
     */
    private static final class PythonicCalls {
        private final String s;
        private int i;

        PythonicCalls(String s) {
            this.s = s;
        }

        /** Parse a call sequence — either a bracketed list {@code [f(..), g(..)]} or a single
         *  bare call {@code f(..)} — starting at the current offset, leaving {@code i} just past
         *  the closing bracket (or the call). String contents are honored, so brackets, parens,
         *  or commas inside quoted argument values never terminate the scan early. */
        List<Map<String, Object>> parseCallSequence() {
            List<Map<String, Object>> calls = new ArrayList<>();
            skipWs();
            if (peek() == '[') {
                i++;
                skipWs();
                if (peek() == ']') { i++; return calls; }
                while (true) {
                    calls.add(call());
                    skipWs();
                    char c = next();
                    if (c == ']') break;
                    if (c != ',') throw err("',' or ']'");
                }
            } else {
                calls.add(call());
            }
            return calls;
        }

        /** Parse the entire input as exactly one call sequence; trailing junk is an error. */
        List<Map<String, Object>> parse() {
            List<Map<String, Object>> calls = parseCallSequence();
            skipWs();
            if (i < s.length()) throw err("end of input");
            return calls;
        }

        private Map<String, Object> call() {
            skipWs();
            String name = identifier();
            skipWs();
            if (next() != '(') throw err("'('");
            Map<String, Object> arguments = new LinkedHashMap<>();
            skipWs();
            if (peek() == ')') {
                i++;
            } else {
                while (true) {
                    skipWs();
                    int mark = i;
                    String key = identifier();
                    skipWs();
                    if (peek() == '=') {
                        i++;
                        arguments.put(key, literal());
                    } else {
                        i = mark;
                        literal(); // positional argument: parse and skip (SGLang behavior)
                    }
                    skipWs();
                    char c = next();
                    if (c == ')') break;
                    if (c != ',') throw err("',' or ')'");
                }
            }
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("name", name);
            call.put("arguments", arguments);
            return call;
        }

        private Object literal() {
            skipWs();
            char c = peek();
            if (c == '"' || c == '\'') return string();
            if (c == '[') return sequence('[', ']');
            if (c == '(') return sequence('(', ')'); // tuple -> JSON array
            if (c == '{') return dict();
            if (c == '-' || c == '+' || Character.isDigit(c) || c == '.') return number();
            String word = identifier();
            return switch (word) {
                case "True", "true" -> Boolean.TRUE;
                case "False", "false" -> Boolean.FALSE;
                case "None", "null" -> null;
                default -> throw err("literal");
            };
        }

        private String string() {
            char quote = next();
            StringBuilder out = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw err("closing quote");
                char c = s.charAt(i++);
                if (c == quote) return out.toString();
                if (c == '\\' && i < s.length()) {
                    char esc = s.charAt(i++);
                    out.append(switch (esc) {
                        case 'n' -> '\n';
                        case 't' -> '\t';
                        case 'r' -> '\r';
                        case '0' -> '\0';
                        default -> esc; // \' \" \\ and anything exotic pass through
                    });
                } else {
                    out.append(c);
                }
            }
        }

        private Object number() {
            int from = i;
            if (peek() == '-' || peek() == '+') i++;
            boolean floating = false;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (Character.isDigit(c)) i++;
                else if (c == '.' || c == 'e' || c == 'E') { floating = true; i++; }
                else if ((c == '-' || c == '+') && (s.charAt(i - 1) == 'e' || s.charAt(i - 1) == 'E')) i++;
                else break;
            }
            String token = s.substring(from, i);
            if (floating) return Double.parseDouble(token);
            return Long.parseLong(token);
        }

        private List<Object> sequence(char open, char close) {
            if (next() != open) throw err("'" + open + "'");
            List<Object> out = new ArrayList<>();
            skipWs();
            if (peek() == close) {
                i++;
                return out;
            }
            while (true) {
                out.add(literal());
                skipWs();
                char c = next();
                if (c == close) return out;
                if (c != ',') throw err("',' or '" + close + "'");
                skipWs();
                if (peek() == close) { i++; return out; } // trailing comma (and 1-tuples)
            }
        }

        private Map<String, Object> dict() {
            if (next() != '{') throw err("'{'");
            Map<String, Object> out = new LinkedHashMap<>();
            skipWs();
            if (peek() == '}') {
                i++;
                return out;
            }
            while (true) {
                Object key = literal();
                skipWs();
                if (next() != ':') throw err("':'");
                out.put(String.valueOf(key), literal());
                skipWs();
                char c = next();
                if (c == '}') return out;
                if (c != ',') throw err("',' or '}'");
                skipWs();
                if (peek() == '}') { i++; return out; }
            }
        }

        private String identifier() {
            skipWs();
            int from = i;
            while (i < s.length() && isIdentifierPart(s.charAt(i))) i++;
            if (i == from) throw err("identifier");
            return s.substring(from, i);
        }

        private void skipWs() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        private char peek() {
            return i < s.length() ? s.charAt(i) : '\0';
        }

        private char next() {
            if (i >= s.length()) throw err("more input");
            return s.charAt(i++);
        }

        private IllegalArgumentException err(String expected) {
            return new IllegalArgumentException("expected " + expected + " at offset " + i + " in: " + s);
        }
    }

    /** Parse tool calls from a model reply, trying the three shapes LFM2.5 is known to emit, in
     *  descending order of confidence: native {@code <|tool_call_start|>...<|tool_call_end|>}
     *  blocks, a JSON tool-call envelope, then bare Pythonic {@code [name(args)]} text. */
    static List<Map<String, Object>> parseToolCalls(String text, Set<String> knownTools) {
        List<Map<String, Object>> nativeCalls = parseNativeToolCalls(text, knownTools);
        if (!nativeCalls.isEmpty()) return nativeCalls;
        String stripped = text.strip(); // both fallbacks work on the trimmed reply; strip once
        List<Map<String, Object>> jsonCalls = parseJsonToolCalls(stripped);
        if (!jsonCalls.isEmpty()) return jsonCalls;
        return parseBarePythonic(stripped, knownTools);
    }

    /** Lenient JSON fallback: an OpenAI-style {@code {"tool_calls":[...]}} envelope, a single
     *  {@code {"function_call":{...}}} or {@code {"name":.., "arguments":..}} object, or a bare
     *  JSON array of call objects — extracted from anywhere in the (already-stripped) text,
     *  fenced code included. Yields no calls when the text holds no parseable JSON of a
     *  recognized shape. */
    private static List<Map<String, Object>> parseJsonToolCalls(String stripped) {
        String json = extractJson(stripped);
        if (json.isEmpty()) return List.of();
        try {
            List<?> calls = jsonCallList(Json.parse(json));
            if (calls == null) return List.of();
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object value : calls) {
                Map<String, Object> normalized = normalizeToolCall(asObject(value, "tool call"), out.size());
                if (normalized != null) out.add(normalized);
            }
            return out;
        } catch (RuntimeException e) {
            return List.of(); // malformed JSON or an unexpected element shape
        }
    }

    /** The raw call objects carried by a recognized JSON tool-call shape, or null if none. */
    private static List<?> jsonCallList(Object parsed) {
        if (parsed instanceof List<?> list) return list;
        if (parsed instanceof Map<?, ?> map) {
            if (map.get("tool_calls") instanceof List<?> list) return list;
            if (map.get("function_call") instanceof Map<?, ?> call) return List.of(call);
            if (map.get("name") instanceof String && (map.containsKey("arguments") || map.containsKey("parameters"))) {
                return List.of(map);
            }
        }
        return null;
    }

    /** Fallback: scan the whole text for a bare pythonic tool call — a bracketed list
     *  {@code [name(args), ...]} or a single {@code name(args)} — emitted without the
     *  {@code <|tool_call_start|>} markers (a documented LFM2.5 behavior, see llama.cpp #24178).
     *  The scan tries to parse a call sequence at every plausible start offset; matching is
     *  fully string-aware (it uses {@link PythonicCalls}, not bracket counting), so quoted
     *  argument values may contain brackets, parens, or commas without breaking detection.
     *  Requires a non-empty {@code knownTools}, and accepts a run only when every call names a
     *  known tool — this keeps ordinary prose ({@code [text](url)}, {@code print()}) from being
     *  mistaken for a tool call. */
    private static List<Map<String, Object>> parseBarePythonic(String text, Set<String> knownTools) {
        if (text.isEmpty() || knownTools.isEmpty()) return List.of();
        for (int p = 0; p < text.length(); p++) {
            char c = text.charAt(p);
            // A call sequence starts either with '[' (list form) or with the first character of
            // an identifier at a word boundary (bare-call form, e.g. `get_weather(...)`).
            boolean listStart = c == '[';
            boolean callStart = (Character.isLetter(c) || c == '_')
                    && (p == 0 || !isIdentifierPart(text.charAt(p - 1)));
            if (!listStart && !callStart) continue;
            List<Map<String, Object>> calls = tryParseCallsAt(text, p, knownTools);
            if (calls != null) return calls;
        }
        return List.of();
    }

    /** Attempt to parse a call sequence starting exactly at {@code from}. Returns the normalized
     *  calls when parsing succeeds, the run is non-empty, and every name is a known tool;
     *  otherwise {@code null} (so the caller keeps scanning). */
    private static List<Map<String, Object>> tryParseCallsAt(String text, int from, Set<String> knownTools) {
        PythonicCalls parser = new PythonicCalls(text);
        parser.i = from;
        List<Map<String, Object>> calls;
        try {
            calls = parser.parseCallSequence();
        } catch (RuntimeException e) {
            return null; // not a well-formed call sequence at this offset
        }
        if (calls.isEmpty()) return null;
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> call : calls) {
            if (!knownTools.contains(stringValue(call.get("name"), ""))) return null;
            Map<String, Object> normalized = normalizeToolCall(call, out.size());
            if (normalized != null) out.add(normalized);
        }
        return out.isEmpty() ? null : out;
    }

    /** Whether {@code c} can appear inside a (dotted) function identifier. */
    private static boolean isIdentifierPart(char c) {
        return Character.isLetterOrDigit(c) || c == '_' || c == '.';
    }

    private static String extractJson(String text) {
        if (text.startsWith("```")) {
            int firstNewline = text.indexOf('\n');
            int lastFence = text.lastIndexOf("```");
            if (firstNewline >= 0 && lastFence > firstNewline) return text.substring(firstNewline + 1, lastFence).strip();
        }
        int objectStart = text.indexOf('{');
        int arrayStart = text.indexOf('[');
        int start = objectStart < 0 ? arrayStart : arrayStart < 0 ? objectStart : Math.min(objectStart, arrayStart);
        if (start < 0) return "";
        int end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']'));
        return end >= start ? text.substring(start, end + 1).strip() : "";
    }

    private static Map<String, Object> normalizeToolCall(Map<String, Object> call, int index) {
        Object functionValue = call.get("function");
        String name;
        Object arguments;
        if (functionValue instanceof Map<?, ?> function) {
            name = stringValue(function.get("name"), null);
            arguments = function.get("arguments");
        } else {
            name = stringValue(call.get("name"), null);
            arguments = call.get("arguments");
        }
        if (name == null || name.isBlank()) return null;
        String argumentString = arguments instanceof String s ? s : Json.stringify(arguments == null ? Map.of() : arguments);
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", name);
        function.put("arguments", argumentString);
        Map<String, Object> normalized = new LinkedHashMap<>();
        normalized.put("id", stringValue(call.get("id"), "call_" + Long.toUnsignedString(System.nanoTime(), 36) + "_" + index));
        normalized.put("type", "function");
        normalized.put("function", function);
        return normalized;
    }

    private static List<Map<String, Object>> toolCallDeltas(List<Map<String, Object>> toolCalls) {
        List<Map<String, Object>> deltas = new ArrayList<>();
        for (int i = 0; i < toolCalls.size(); i++) {
            Map<String, Object> call = toolCalls.get(i);
            Map<String, Object> delta = new LinkedHashMap<>(call);
            delta.put("index", i);
            deltas.add(delta);
        }
        return deltas;
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
                String stop = stringValue(item, "");
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
        return "none".equals(stringValue(request.get("reasoning_format"), null));
    }

    private static String messageContent(Object content) {
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object part : parts) {
                if (part instanceof Map<?, ?> map && "text".equals(map.get("type"))) {
                    Object text = map.get("text");
                    if (text != null) sb.append(text);
                }
            }
            return sb.toString();
        }
        return stringValue(content, "");
    }

    private static void setTimingHeader(HttpExchange exchange, GenerationResult result) {
        exchange.getResponseHeaders().set("X-LFM2-Timing", Json.stringify(timings(result)));
    }

    private static void sendJson(HttpExchange exchange, int status, Object value) throws IOException {
        byte[] bytes = Json.stringify(value).getBytes(StandardCharsets.UTF_8);
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
    private static final java.util.concurrent.BlockingQueue<Runnable> GENERATION_QUEUE =
            RuntimeFlags.SERVER_QUEUE == 0 ? new java.util.concurrent.SynchronousQueue<>()
                                    : new java.util.concurrent.ArrayBlockingQueue<>(RuntimeFlags.SERVER_QUEUE);

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
        java.util.concurrent.CountDownLatch done = new java.util.concurrent.CountDownLatch(1);
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

    private static final byte[] SSE_DONE = "data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8);

    /** A slow (or stopped) streaming client blocks the worker's SSE write once the TCP window
     *  fills; the JDK server has no write timeout, so one such client would wedge the single
     *  generation worker forever. Every streaming write is tracked here, and a reaper closes
     *  the exchange when one write stalls past llama.serverWriteTimeout seconds — the blocked
     *  write then fails with an IOException, which aborts that generation cleanly. */
    private static final Set<StreamWatch> ACTIVE_STREAMS = java.util.concurrent.ConcurrentHashMap.newKeySet();

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

    private static OutputStream beginStream(HttpExchange exchange) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0);
        OutputStream responseBody = exchange.getResponseBody(); // before registering: a throw here must not leak the watch
        StreamWatch watch = new StreamWatch(exchange);
        ACTIVE_STREAMS.add(watch);
        return new java.io.FilterOutputStream(responseBody) {
            @Override
            public void write(byte[] b, int off, int len) throws IOException {
                watch.writeStartNanos = System.nanoTime();
                try {
                    out.write(b, off, len);
                } finally {
                    watch.writeStartNanos = 0;
                }
            }

            @Override
            public void close() throws IOException {
                ACTIVE_STREAMS.remove(watch);
                super.close();
            }
        };
    }

    private static void writeSse(OutputStream out, Object value) throws IOException {
        out.write(("data: " + Json.stringify(value) + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static void writeSseEvent(OutputStream out, String event, Object value) throws IOException {
        out.write(("event: " + event + "\n").getBytes(StandardCharsets.UTF_8));
        writeSse(out, value);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asObject(Object value, String name) {
        if (value instanceof Map<?, ?> map) return (Map<String, Object>) map;
        throw new IllegalArgumentException(name + " must be an object");
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asArray(Object value, String name) {
        if (value instanceof List<?> list) return (List<Object>) list;
        throw new IllegalArgumentException(name + " must be an array");
    }

    private static String stringValue(Object value, String defaultValue) {
        return value == null ? defaultValue : String.valueOf(value);
    }

    /** The text representation of a special token, or null if absent. */
    private static String specialTokenString(LFMTokenizer t, String name) {
        Integer id = t.getSpecialTokens().get(name);
        return id != null ? t.decode(id) : null;
    }

    private static boolean booleanValue(Object value, boolean defaultValue) {
        return value instanceof Boolean b ? b : defaultValue;
    }

    private static int intValue(Object value, int defaultValue) {
        return Math.toIntExact(longValue(value, defaultValue));
    }

    private static long longValue(Object value, long defaultValue) {
        if (value instanceof Number n) {
            return n.longValue();
        }
        if (value instanceof String s) { // tolerate string-encoded numbers (e.g. "seed": "42")
            try {
                return Long.parseLong(s.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid argument: '" + s + "' is not an integer");
            }
        }
        return defaultValue;
    }

    private static float floatValue(Object value, float defaultValue) {
        if (value instanceof Number n) {
            return n.floatValue();
        }
        if (value instanceof String s) {
            try {
                return Float.parseFloat(s.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid argument: '" + s + "' is not a number");
            }
        }
        return defaultValue;
    }

    /**
     * JSON facade over com.qxotic:json keeping this server's conventions: explicit JSON null
     * parses to Java null (treated like an absent field), Java null prints as JSON null (tool
     * arguments carry Python None), decimals parse as Double (integers stay Long, huge ones
     * promote to BigInteger), any Iterable prints as an array and unknown types print as their
     * string form. Parse failures stay RuntimeExceptions (handler threads turn them into 400s).
     */
    static final class Json {
        private static final com.qxotic.format.json.Json.ParseOptions OPTIONS =
                com.qxotic.format.json.Json.ParseOptions.defaults().decimalsAsBigDecimal(false);

        static Object parse(String text) {
            return fromLibrary(com.qxotic.format.json.Json.parse(text, OPTIONS));
        }

        static String stringify(Object value) {
            return com.qxotic.format.json.Json.stringify(toLibrary(value));
        }

        /** Json.NULL -> Java null, in place (the parser's containers are mutable). */
        private static Object fromLibrary(Object value) {
            if (value == com.qxotic.format.json.Json.NULL) {
                return null;
            }
            if (value instanceof Map<?, ?> map) {
                @SuppressWarnings("unchecked")
                Map<String, Object> object = (Map<String, Object>) map;
                for (Map.Entry<String, Object> entry : object.entrySet()) {
                    entry.setValue(fromLibrary(entry.getValue()));
                }
            } else if (value instanceof List<?> list) {
                @SuppressWarnings("unchecked")
                List<Object> array = (List<Object>) list;
                array.replaceAll(Json::fromLibrary);
            }
            return value;
        }

        /** Java null -> Json.NULL, plus the lenient coercions the previous printer had. */
        private static Object toLibrary(Object value) {
            if (value == null) {
                return com.qxotic.format.json.Json.NULL;
            }
            if (value instanceof String || value instanceof Number || value instanceof Boolean) {
                return value;
            }
            if (value instanceof Map<?, ?> map) {
                Map<String, Object> object = LinkedHashMap.newLinkedHashMap(map.size());
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    object.put(String.valueOf(entry.getKey()), toLibrary(entry.getValue()));
                }
                return object;
            }
            if (value instanceof Iterable<?> iterable) {
                List<Object> array = new ArrayList<>();
                for (Object item : iterable) {
                    array.add(toLibrary(item));
                }
                return array;
            }
            return String.valueOf(value);
        }
    }
}
