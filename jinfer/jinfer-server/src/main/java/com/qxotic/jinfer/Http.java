package com.qxotic.jinfer;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * HTTP transport plumbing shared by every endpoint: the request preamble (access log, CORS,
 * OPTIONS preflight), bounded body reads, JSON responses, and the uniform error envelope. Pure
 * transport — it knows nothing about inference, so handlers read top-to-bottom as
 * {@code preamble → parse → do work → respond}.
 */
final class Http {

    private Http() {}

    /** Per-request preamble: access log, CORS headers, and OPTIONS preflight. Returns {@code true}
     *  when the request was a preflight already answered (204) — the caller should then return. */
    static boolean preamble(HttpExchange exchange) throws IOException {
        log(exchange);
        cors(exchange);
        if (!"OPTIONS".equals(exchange.getRequestMethod())) return false;
        exchange.sendResponseHeaders(204, -1);
        exchange.close();
        return true;
    }

    static void log(HttpExchange exchange) {
        System.err.printf("%s %s from %s%n",
                exchange.getRequestMethod(), exchange.getRequestURI(), exchange.getRemoteAddress());
    }

    static void cors(HttpExchange exchange) {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Access-Control-Allow-Origin", "*");
        headers.set("Access-Control-Allow-Headers", "authorization, content-type, x-request-id");
        headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        String requestId = exchange.getRequestHeaders().getFirst("X-Request-ID");
        if (requestId != null) headers.set("X-Request-ID", requestId);
    }

    /** Rejects a wrong method with 405 + Allow; returns {@code true} when it did so (caller returns). */
    static boolean requireMethod(HttpExchange exchange, String method) throws IOException {
        if (method.equals(exchange.getRequestMethod())) return false;
        exchange.getResponseHeaders().set("Allow", method + ", OPTIONS");
        sendError(exchange, 405, "Method not allowed");
        return true;
    }

    /** Reads the request body, bounded by {@code llama.serverMaxBodyMB}; returns null after sending
     *  413 when the body exceeds the limit (callers must return immediately on null). */
    static byte[] readBody(HttpExchange exchange) throws IOException {
        byte[] body = exchange.getRequestBody().readNBytes((int) RuntimeFlags.SERVER_MAX_BODY_BYTES + 1);
        if (body.length > RuntimeFlags.SERVER_MAX_BODY_BYTES) {
            sendError(exchange, 413, "Request body exceeds " + (RuntimeFlags.SERVER_MAX_BODY_BYTES >> 20) + " MB");
            return null;
        }
        return body;
    }

    static void sendJson(HttpExchange exchange, int status, Object value) throws IOException {
        sendText(exchange, status, "application/json; charset=utf-8", JsonCodec.stringify(value));
    }

    /** Sends a body with the given content type (JSON via {@link #sendJson}, Prometheus text, …). */
    static void sendText(HttpExchange exchange, int status, String contentType, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", contentType);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    static void sendError(HttpExchange exchange, int status, String message) throws IOException {
        sendJson(exchange, status, Map.of("error", errorPayload(status, message)));
    }

    /** Best-effort error send for paths where the response may already be (partly) committed —
     *  a connection loss or an already-sent header is logged, never thrown. */
    static void sendErrorQuietly(HttpExchange exchange, int status, String message) {
        try {
            sendError(exchange, status, message);
        } catch (IOException e) {
            System.err.println("client connection lost: " + e);
        } catch (RuntimeException e) {
            System.err.println("response already committed, dropping error (" + status + " " + message + "): " + e);
        }
    }

    static Map<String, Object> errorPayload(int status, String message) {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("message", message);
        error.put("type", status == 404 ? "not_found_error" : status >= 500 ? "internal_error" : "invalid_request_error");
        error.put("param", null);
        error.put("code", null);
        return error;
    }

    static String errorMessage(Throwable e) {
        return e.getMessage() == null ? e.toString() : e.getMessage();
    }
}
