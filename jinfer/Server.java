package com.llama4j;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
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
        server.createContext("/v1/models", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.getResponseHeaders().set("Allow", "GET, OPTIONS");
                sendError(exchange, 405, "Method not allowed");
                return;
            }
            String modelId = options.modelPath().getFileName().toString();
            sendJson(exchange, 200, Map.of(
                    "object", "list",
                    "data", List.of(Map.of(
                            "id", modelId,
                            "object", "model",
                            "created", 0,
                            "owned_by", "lfm25.java"))));
        });
        server.createContext("/v1/chat/completions", exchange -> handleChatCompletion(exchange, model, options));
        server.createContext("/v1/completions", exchange -> handleCompletion(exchange, model, options));
        server.createContext("/v1/responses", exchange -> handleResponse(exchange, model, options));
        server.createContext("/health", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            sendJson(exchange, 200, Map.of("status", "ok", "busy", workerBusy, "queued", GENERATION_QUEUE.size()));
        });
        server.createContext("/props", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            Llama.Configuration config = model.configuration();
            sendJson(exchange, 200, Map.of(
                    "model", options.modelPath().getFileName().toString(),
                    "n_ctx", config.contextLength,
                    "n_batch", RuntimeFlags.MAX_PROMPT_SEQUENCE_LENGTH,
                    "n_embd", config.embeddingLength,
                    "n_vocab", config.vocabularySize,
                    "n_layer", config.numberOfLayers,
                    "prompt_cache", promptCache == null ? Map.of("enabled", false) : promptCache.stats()));
        });
        server.createContext("/tokenize", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            try {
                byte[] body = readBody(exchange);
                if (body == null) return;
                Map<String, Object> request = asObject(Json.parse(new String(body, StandardCharsets.UTF_8)), "request");
                String content = String.valueOf(request.getOrDefault("content", ""));
                sendJson(exchange, 200, Map.of("tokens", model.tokenizer().encode(content)));
            } catch (Exception e) {
                sendError(exchange, 400, errorMessage(e));
            }
        });
        server.createContext("/detokenize", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            try {
                byte[] body = readBody(exchange);
                if (body == null) return;
                Map<String, Object> request = asObject(Json.parse(new String(body, StandardCharsets.UTF_8)), "request");
                Object raw = request.get("tokens");
                List<Integer> tokens = raw instanceof List<?> list
                        ? list.stream().map(v -> ((Number) v).intValue()).toList()
                        : List.of();
                sendJson(exchange, 200, Map.of("content", model.tokenizer().decode(tokens)));
            } catch (Exception e) {
                sendError(exchange, 400, errorMessage(e));
            }
        });
        server.createContext("/", exchange -> {
            logRequest(exchange);
            addCommonHeaders(exchange);
            if (handleOptions(exchange)) return;
            sendError(exchange, 404, "Not found");
        });
        if (RuntimeFlags.PROMPT_CACHE) {
            promptCache = new PromptCache(model.configuration());
            System.out.printf("Prompt cache enabled: page=%d tokens, budget=%d MB%n",
                    promptCache.page, RuntimeFlags.PROMPT_CACHE_BUDGET_BYTES >> 20);
        }
        startGenerationWorker();
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
                boolean toolRequest = hasUsableTools(request);
                Usage usageCounts = toolRequest ? null : new Usage();
                Consumer<String> contentSink = toolRequest ? null : sseDeltaSink(out, id, modelId, "content", usageCounts);
                Consumer<String> reasoningSink = toolRequest ? null : sseDeltaSink(out, id, modelId, "reasoning_content", usageCounts);
                GenerationResult result = generateChat(model, options, request, messages, contentSink, reasoningSink, usageCounts);
                if (toolRequest) {
                    Map<String, Object> delta = result.toolCalls().isEmpty()
                            ? Map.of("content", result.text())
                            : Map.of("tool_calls", toolCallDeltas(result.toolCalls()));
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
        for (Object message : messages) {
            asObject(message, "message");
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

    private static GenerationResult generateChat(Llama model, Options options, Map<String, Object> request, List<Object> messages,
                                                  Consumer<String> onText, Consumer<String> onReasoning,
                                                  Usage usageCounts) {
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        promptTokens.add(chatFormat.beginOfSentence);
        if (hasUsableTools(request)) {
            promptTokens.addAll(encodeToolsSystemMessage(model.tokenizer(), chatFormat, request));
        }
        for (Object value : messages) {
            Map<String, Object> message = asObject(value, "message");
            String role = stringValue(message.get("role"), "user");
            String content = chatMessageContent(message);
            LFMChatFormat.Role lfmRole = switch (role) {
                case "system" -> LFMChatFormat.Role.SYSTEM;
                case "assistant" -> LFMChatFormat.Role.ASSISTANT;
                case "tool" -> LFMChatFormat.Role.TOOL; // native tool turn, not flattened into user
                default -> LFMChatFormat.Role.USER;
            };
            promptTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(lfmRole, content)));
        }
        promptTokens.addAll(chatFormat.encodeGenerationPrompt());
        if (!requestThink(request, options)) chatFormat.appendThinkSurrogate(promptTokens);
        Ingestion resumed = matchChatSession(request, messages, chatFormat, model.tokenizer(), options);
        Ingestion ingestion = resumed != null ? resumed : Ingestion.of(model.createNewState(), 0, promptTokens);
        GenerationResult result = generateResponse(model, options, request, promptTokens, ingestion, chatFormat.getStopTokens(), onText, onReasoning, usageCounts);
        saveChatSession(model, request, messages, ingestion, result);
        return hasUsableTools(request) ? withParsedToolCalls(result) : result;
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
        /** prefillPositions = positions occupied once the prompt is ingested, computed with
         *  buildPrefillTokens' exact rule against the PRE-generation latestToken (no BOS-name
         *  guessing — the model may call it <bos> or <|startoftext|>). */
        static Ingestion of(Llama.State state, int startPosition, List<Integer> tokens) {
            int skip = startPosition == 0 && !tokens.isEmpty() && tokens.getFirst() == state.latestToken ? 1 : 0;
            return new Ingestion(state, startPosition, tokens, startPosition + 1 + tokens.size() - skip);
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
            String role = stringValue(message.get("role"), "user");
            LFMChatFormat.Role lfmRole = switch (role) {
                case "system" -> LFMChatFormat.Role.SYSTEM;
                case "assistant" -> LFMChatFormat.Role.ASSISTANT;
                case "tool" -> LFMChatFormat.Role.TOOL;
                default -> LFMChatFormat.Role.USER;
            };
            delta.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(lfmRole, chatMessageContent(message))));
        }
        delta.addAll(chatFormat.encodeGenerationPrompt());
        if (!requestThink(request, options)) chatFormat.appendThinkSurrogate(delta);
        return Ingestion.of(s.state(), s.position(), delta);
    }

    /**
     * Server generation driver over {@link Engine#generate}. The ingestion plan says where to
     * start: a fresh state at position 0 (with the prompt cache plugged in through
     * {@link Llama.GenerationHooks}), or a resumed chat session mid-context — in which case the
     * cache is bypassed (its page indexes are full-stream positions). cachedOut[0] tracks the
     * resumed-prefix length for the streaming usage counters.
     */
    private static GenerationResult runServerGeneration(Llama model, Ingestion ingestion, Engine.Params params,
                                                        Engine.Listener listener, int[] cachedOut) {
        PromptCache cache = promptCache;
        Llama.State state = ingestion.state();
        if (cache == null || ingestion.startPosition() > 0) {
            return Engine.generate(model, state, ingestion.startPosition(), ingestion.tokens(), params, listener, Llama.GenerationHooks.NONE);
        }
        class CacheHooks implements Llama.GenerationHooks {
            PromptCache.Node cursor = cache.root;
            boolean caching = true;

            @Override
            public int resumePosition(int[] stream, int prefillLength) {
                PromptCache.Match match = cache.lookup(stream, prefillLength);
                int cached = match.positions();
                cachedOut[0] = cached;
                if (cached > 0) {
                    cache.restore(match, state);
                    cache.pin(match.path());
                    cursor = match.path().getLast();
                }
                return cached;
            }

            @Override
            public int clampChunk(int position, int chunkLength) {
                int pageEnd = (position / cache.page + 1) * cache.page;
                return Math.min(chunkLength, pageEnd - position);
            }

            @Override
            public void afterIngest(int[] stream, int position) {
                while (caching && (long) (cursor.depth + 2) * cache.page <= position) {
                    PromptCache.Node node = cache.commitPage(cursor, stream, state);
                    if (node == null) {
                        caching = false;
                        break;
                    }
                    cursor = node;
                }
            }
        }
        CacheHooks hooks = new CacheHooks();
        try {
            return Engine.generate(model, state, 0, ingestion.tokens(), params, listener, hooks);
        } finally {
            cache.unpin(hooks.cursor);
        }
    }

    /** Sampling-parameter validation shared by all endpoints; called on the HTTP handler thread
     *  (before queueing, and before any SSE headers) so invalid requests fail fast with a 400. */
    private static void validateGenerationParams(Map<String, Object> request, Options options) {
        Options.require(intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        float temperature = floatValue(request.get("temperature"), options.temperature());
        Options.require(Float.isFinite(temperature) && 0 <= temperature, "Invalid argument: temperature must be a finite non-negative number");
        float topp = floatValue(request.get("top_p"), options.topp());
        Options.require(Float.isFinite(topp) && 0 <= topp && topp <= 1, "Invalid argument: top_p must be within [0, 1]");
        Options.require(0 <= intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens()), "Invalid argument: max_tokens must be non-negative");
        longValue(request.get("seed"), 0); // type check only
    }

    /** Request fields to {@link Engine.Params}/{@link Engine.Listener}, then one engine pass
     *  through {@link #runServerGeneration}. Usage counters for streaming chunks are fed by the
     *  token listener (cached prefix + non-special completion tokens, matching the final usage
     *  the non-streaming path reports). */
    private static GenerationResult generateResponse(Llama model, Options options, Map<String, Object> request, List<Integer> promptTokens,
                                           Ingestion ingestion, Set<Integer> baseStopTokens, Consumer<String> onText,
                                           Consumer<String> onReasoning, Usage usageCounts) {
        float temperature = floatValue(request.get("temperature"), options.temperature());
        float topp = floatValue(request.get("top_p"), options.topp());
        long seed = longValue(request.get("seed"), options.seed());
        int maxTokens = intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens());
        // defense in depth: server requests were already checked by validateGenerationParams on
        // the handler thread; these guards keep the method safe for any future non-HTTP caller
        Options.require(intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        Options.require(0 <= maxTokens, "Invalid argument: max_tokens must be non-negative");
        StopSpec stops = stopSpec(model.tokenizer(), request.get("stop"), baseStopTokens);
        Sampler sampler = Engine.configuredSampler(model, requestThink(request, options), temperature, topp, seed);
        int consumedPromptTokens = Engine.consumedPromptTokens(model.tokenizer(), promptTokens); // client-facing usage counts
        int[] cachedOut = {Math.min(ingestion.startPosition(), consumedPromptTokens)};
        IntConsumer onToken = usageCounts == null ? null : token -> {
            usageCounts.cachedTokens = Math.min(cachedOut[0], consumedPromptTokens);
            if (!model.tokenizer().isSpecialToken(token)) usageCounts.completionTokens++;
        };
        if (usageCounts != null) usageCounts.promptTokens = consumedPromptTokens;
        Engine.Params params = new Engine.Params(sampler, maxTokens, stops, inlineReasoning(request));
        Engine.Listener listener = new Engine.Listener(onToken, onText, onReasoning);
        return runServerGeneration(model, ingestion, params, listener, cachedOut);
    }

    private static boolean hasUsableTools(Map<String, Object> request) {
        Object choice = request.get("tool_choice");
        if (choice instanceof String s && "none".equals(s)) return false;
        Object tools = request.get("tools");
        return tools instanceof List<?> list && !list.isEmpty();
    }

    /** Tools system turn. Prefers the model's native protocol — the tool list as a JSON array
     *  between <|tool_list_start|> and <|tool_list_end|> (LFM2.5 was trained on this shape);
     *  plain encode() does not map special-token strings, so the markers are inserted as token
     *  ids directly. Falls back to a plain-text instruction when the vocabulary lacks them. */
    private static List<Integer> encodeToolsSystemMessage(LFMTokenizer tokenizer, LFMChatFormat chatFormat, Map<String, Object> request) {
        Map<String, Integer> specialTokens = tokenizer.getSpecialTokens();
        Integer listStart = specialTokens.get("<|tool_list_start|>");
        Integer listEnd = specialTokens.get("<|tool_list_end|>");
        if (listStart == null || listEnd == null) {
            return chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, toolsPrompt(request)));
        }
        List<Object> functions = new ArrayList<>();
        for (Object tool : asArray(request.get("tools"), "tools")) {
            Map<String, Object> toolObject = asObject(tool, "tool");
            functions.add(toolObject.getOrDefault("function", toolObject));
        }
        List<Integer> tokens = chatFormat.encodeHeader(new LFMChatFormat.Message(LFMChatFormat.Role.SYSTEM, ""));
        tokens.addAll(tokenizer.encode("List of tools: "));
        tokens.add(listStart);
        tokens.addAll(tokenizer.encode(Json.stringify(functions)));
        tokens.add(listEnd);
        String hints = toolChoiceHints(request);
        if (!hints.isEmpty()) {
            tokens.addAll(tokenizer.encode("\n" + hints));
        }
        if (chatFormat.endOfTurn >= 0) {
            tokens.add(chatFormat.endOfTurn);
        }
        tokens.addAll(tokenizer.encode("\n"));
        return tokens;
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

    private static String toolsPrompt(Map<String, Object> request) {
        StringBuilder sb = new StringBuilder();
        sb.append("You may call tools to help answer the user.\n");
        sb.append("When calling tools, respond only with valid JSON and no extra text.\n");
        sb.append("Use this exact shape: {\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n");
        sb.append("If no tool is needed, answer normally.\n");
        String hints = toolChoiceHints(request);
        if (!hints.isEmpty()) {
            sb.append(hints).append('\n');
        }
        sb.append("Available tools:\n");
        for (Object tool : asArray(request.get("tools"), "tools")) {
            Map<String, Object> toolObject = asObject(tool, "tool");
            Object function = toolObject.get("function");
            sb.append(Json.stringify(function == null ? toolObject : function)).append('\n');
        }
        return sb.toString();
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
            String callsText = "Tool calls made:\n" + Json.stringify(calls);
            return content.isEmpty() ? callsText : content + "\n" + callsText;
        }
        Object functionCall = message.get("function_call");
        if (functionCall instanceof Map<?, ?> call) {
            String callText = "Function call made:\n" + Json.stringify(call);
            return content.isEmpty() ? callText : content + "\n" + callText;
        }
        return content;
    }

    private static GenerationResult withParsedToolCalls(GenerationResult result) {
        List<Map<String, Object>> toolCalls = parseToolCalls(result.text());
        if (toolCalls.isEmpty()) return result;
        return result.asToolCalls(toolCalls);
    }

    private static final String TC_START = "<|tool_call_start|>";
    private static final String TC_END = "<|tool_call_end|>";
    private static final String TC_ARGS = "<|tool_call_args|>";
    private static final String TC_ARGS_END = "<|tool_call_args_end|>";

    private static List<Map<String, Object>> parseNativeToolCalls(String text) {
        List<Map<String, Object>> calls = new ArrayList<>();
        int pos = 0;
        while (true) {
            int start = text.indexOf(TC_START, pos);
            if (start < 0) break;
            int nameEnd = text.indexOf(TC_END, start + TC_START.length());
            if (nameEnd < 0) break;
            String rawName = text.substring(start + TC_START.length(), nameEnd).strip();
            if (rawName.isEmpty()) {
                pos = nameEnd + TC_END.length();
                continue;
            }
            int parenOpen = rawName.indexOf('(');
            if (parenOpen >= 0) {
                int parenClose = rawName.lastIndexOf(')');
                if (parenClose > parenOpen) {
                    String funcName = rawName.substring(0, parenOpen).strip();
                    if (funcName.startsWith("[")) funcName = funcName.substring(1).strip();
                    String args = rawName.substring(parenOpen + 1, parenClose).strip();
                    Map<String, Object> parsed = parseCallExpressionArgs(args);
                    addNativeCall(calls, funcName, Json.stringify(parsed));
                    pos = nameEnd + TC_END.length();
                    continue;
                }
            }
            int argsStart = text.indexOf(TC_ARGS, nameEnd);
            if (argsStart >= 0) {
                int jsonStart = argsStart + TC_ARGS.length();
                int jsonEnd = text.indexOf(TC_ARGS_END, jsonStart);
                if (jsonEnd >= 0) {
                    addNativeCall(calls, rawName, text.substring(jsonStart, jsonEnd).strip());
                    pos = jsonEnd + TC_ARGS_END.length();
                    continue;
                }
                pos = jsonStart;
            } else {
                pos = nameEnd + TC_END.length();
            }
            int nextCall = text.indexOf(TC_START, pos);
            String raw = nextCall >= 0 ? text.substring(pos, nextCall).strip() : text.substring(pos).strip();
            String json = extractJson(raw);
            addNativeCall(calls, rawName, !json.isEmpty() ? json : raw);
            pos = nextCall >= 0 ? nextCall : text.length();
        }
        return calls;
    }

    private static Map<String, Object> parseCallExpressionArgs(String args) {
        Map<String, Object> result = new LinkedHashMap<>();
        args = args.strip();
        if (args.isEmpty()) return result;
        List<String> pairs = splitArgPairs(args);
        for (String pair : pairs) {
            int eq = pair.indexOf('=');
            if (eq < 0) continue;
            String key = pair.substring(0, eq).strip();
            String value = pair.substring(eq + 1).strip();
            if ((value.startsWith("\"") && value.endsWith("\""))
                    || (value.startsWith("'") && value.endsWith("'"))) {
                value = value.substring(1, value.length() - 1);
            }
            result.put(key, value);
        }
        return result;
    }

    private static List<String> splitArgPairs(String s) {
        List<String> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inQuote = false;
        char quoteChar = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (!inQuote && (c == '"' || c == '\'')) {
                inQuote = true;
                quoteChar = c;
                current.append(c);
            } else if (inQuote && c == quoteChar) {
                inQuote = false;
                quoteChar = 0;
                current.append(c);
            } else if (!inQuote && c == ',') {
                if (!current.isEmpty()) parts.add(current.toString().strip());
                current.setLength(0);
            } else {
                current.append(c);
            }
        }
        if (!current.isEmpty()) parts.add(current.toString().strip());
        return parts;
    }

    private static void addNativeCall(List<Map<String, Object>> calls, String name, String arguments) {
        Map<String, Object> call = new LinkedHashMap<>();
        call.put("name", name);
        call.put("arguments", arguments);
        // normalize into the OpenAI envelope: {id, type:"function", function:{name, arguments:"<json string>"}}
        Map<String, Object> normalized = normalizeToolCall(call, calls.size());
        if (normalized != null) calls.add(normalized);
    }

    private static List<Map<String, Object>> parseToolCalls(String text) {
        List<Map<String, Object>> nativeCalls = parseNativeToolCalls(text);
        if (!nativeCalls.isEmpty()) return nativeCalls;
        String json = extractJson(text.strip());
        if (json.isEmpty()) return List.of();
        try {
            Object parsed = Json.parse(json);
            List<Object> calls;
            if (parsed instanceof Map<?, ?> map && map.get("tool_calls") instanceof List<?> list) {
                calls = new ArrayList<>(list);
            } else if (parsed instanceof Map<?, ?> map && map.get("function_call") instanceof Map<?, ?> call) {
                calls = List.of(call);
            } else if (parsed instanceof Map<?, ?> map && map.get("name") instanceof String
                    && (map.containsKey("arguments") || map.containsKey("parameters"))) {
                // bare {"name": ...} alone is too loose: ordinary JSON answers with a "name"
                // field would be misread as tool calls and their text discarded
                calls = List.of(map);
            } else if (parsed instanceof List<?> list) {
                calls = new ArrayList<>(list);
            } else {
                return List.of();
            }
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object value : calls) {
                Map<String, Object> call = asObject(value, "tool call");
                Map<String, Object> normalized = normalizeToolCall(call, out.size());
                if (normalized != null) out.add(normalized);
            }
            return out;
        } catch (RuntimeException e) {
            return List.of();
        }
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

    private static StopSpec stopSpec(LFMTokenizer tokenizer, Object value, Set<Integer> baseStopTokens) {
        Set<Integer> tokenStops = new HashSet<>(baseStopTokens);
        List<String> textStops = new ArrayList<>();
        if (value instanceof String s) {
            addStop(tokenizer, tokenStops, textStops, s);
        } else if (value instanceof List<?> values) {
            for (Object item : values) addStop(tokenizer, tokenStops, textStops, stringValue(item, ""));
        } else if (value != null) {
            throw new IllegalArgumentException("stop must be a string or an array of strings");
        }
        return new StopSpec(Collections.unmodifiableSet(tokenStops), List.copyOf(textStops));
    }

    private static void addStop(LFMTokenizer tokenizer, Set<Integer> tokenStops, List<String> textStops, String stop) {
        if (stop.isEmpty()) return;
        textStops.add(stop);
        List<Integer> tokens = tokenizer.encode(stop);
        if (tokens.size() == 1) tokenStops.add(tokens.getFirst());
    }

    /** Effective thinking switch for a server request: chat_template_kwargs.enable_thinking
     *  (llama.cpp convention) overrides the CLI --think flag. */
    private static boolean requestThink(Map<String, Object> request, Options options) {
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

    static final class Json {
        static Object parse(String text) {
            return new Parser(text).parse();
        }

        static String stringify(Object value) {
            StringBuilder sb = new StringBuilder();
            writeJson(sb, value);
            return sb.toString();
        }

        private static void writeJson(StringBuilder sb, Object value) {
            if (value == null) {
                sb.append("null");
            } else if (value instanceof String s) {
                sb.append('"');
                for (int i = 0; i < s.length(); i++) {
                    char c = s.charAt(i);
                    switch (c) {
                        case '"' -> sb.append("\\\"");
                        case '\\' -> sb.append("\\\\");
                        case '\b' -> sb.append("\\b");
                        case '\f' -> sb.append("\\f");
                        case '\n' -> sb.append("\\n");
                        case '\r' -> sb.append("\\r");
                        case '\t' -> sb.append("\\t");
                        default -> {
                            if (c < 0x20) sb.append("\\u%04x".formatted((int) c));
                            else sb.append(c);
                        }
                    }
                }
                sb.append('"');
            } else if (value instanceof Number || value instanceof Boolean) {
                sb.append(value);
            } else if (value instanceof Map<?, ?> map) {
                sb.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) sb.append(',');
                    first = false;
                    writeJson(sb, String.valueOf(entry.getKey()));
                    sb.append(':');
                    writeJson(sb, entry.getValue());
                }
                sb.append('}');
            } else if (value instanceof Iterable<?> iterable) {
                sb.append('[');
                boolean first = true;
                for (Object item : iterable) {
                    if (!first) sb.append(',');
                    first = false;
                    writeJson(sb, item);
                }
                sb.append(']');
            } else {
                writeJson(sb, String.valueOf(value));
            }
        }

        private static final class Parser {
            private final String text;
            private int index;

            Parser(String text) {
                this.text = text;
            }

            Object parse() {
                Object value = value();
                skipWhitespace();
                if (index != text.length()) throw error("Unexpected trailing data");
                return value;
            }

            private Object value() {
                skipWhitespace();
                if (index >= text.length()) throw error("Unexpected end of input");
                return switch (text.charAt(index)) {
                    case '{' -> object();
                    case '[' -> array();
                    case '"' -> string();
                    case 't' -> literal("true", Boolean.TRUE);
                    case 'f' -> literal("false", Boolean.FALSE);
                    case 'n' -> literal("null", null);
                    default -> number();
                };
            }

            private Map<String, Object> object() {
                index++;
                Map<String, Object> map = new LinkedHashMap<>();
                skipWhitespace();
                if (consume('}')) return map;
                do {
                    skipWhitespace();
                    if (index >= text.length() || text.charAt(index) != '"') throw error("Expected object key");
                    String key = string();
                    skipWhitespace();
                    if (!consume(':')) throw error("Expected ':'");
                    map.put(key, value());
                    skipWhitespace();
                } while (consume(','));
                if (!consume('}')) throw error("Expected '}'");
                return map;
            }

            private List<Object> array() {
                index++;
                List<Object> list = new ArrayList<>();
                skipWhitespace();
                if (consume(']')) return list;
                do {
                    list.add(value());
                    skipWhitespace();
                } while (consume(','));
                if (!consume(']')) throw error("Expected ']'");
                return list;
            }

            private String string() {
                index++;
                StringBuilder sb = new StringBuilder();
                while (index < text.length()) {
                    char c = text.charAt(index++);
                    if (c == '"') return sb.toString();
                    if (c == '\\') {
                        if (index >= text.length()) throw error("Unexpected escape");
                        char e = text.charAt(index++);
                        switch (e) {
                            case '"' -> sb.append('"');
                            case '\\' -> sb.append('\\');
                            case '/' -> sb.append('/');
                            case 'b' -> sb.append('\b');
                            case 'f' -> sb.append('\f');
                            case 'n' -> sb.append('\n');
                            case 'r' -> sb.append('\r');
                            case 't' -> sb.append('\t');
                            case 'u' -> {
                                if (index + 4 > text.length()) throw error("Invalid unicode escape");
                                sb.append((char) Integer.parseInt(text.substring(index, index + 4), 16));
                                index += 4;
                            }
                            default -> throw error("Invalid escape");
                        }
                    } else {
                        sb.append(c);
                    }
                }
                throw error("Unterminated string");
            }

            private Object number() {
                int start = index;
                if (consume('-')) {}
                while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                boolean floating = false;
                if (consume('.')) {
                    floating = true;
                    while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                }
                if (index < text.length() && (text.charAt(index) == 'e' || text.charAt(index) == 'E')) {
                    floating = true;
                    index++;
                    if (index < text.length() && (text.charAt(index) == '+' || text.charAt(index) == '-')) index++;
                    while (index < text.length() && Character.isDigit(text.charAt(index))) index++;
                }
                if (start == index) throw error("Expected value");
                String number = text.substring(start, index);
                // no ternary: double/long branches would promote BOTH to double, silently
                // turning every integer into a Double (and corrupting longs above 2^53)
                if (floating) {
                    return Double.parseDouble(number);
                }
                return Long.parseLong(number);
            }

            private Object literal(String literal, Object value) {
                if (!text.startsWith(literal, index)) throw error("Expected " + literal);
                index += literal.length();
                return value;
            }

            private boolean consume(char c) {
                if (index < text.length() && text.charAt(index) == c) {
                    index++;
                    return true;
                }
                return false;
            }

            private void skipWhitespace() {
                while (index < text.length() && Character.isWhitespace(text.charAt(index))) index++;
            }

            private IllegalArgumentException error(String message) {
                return new IllegalArgumentException(message + " at character " + index);
            }
        }
    }
}
