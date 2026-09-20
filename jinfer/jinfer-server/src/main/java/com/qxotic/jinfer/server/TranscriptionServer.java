package com.qxotic.jinfer.server;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The speech-to-text transport: {@code POST /v1/audio/transcriptions} (OpenAI-compatible
 * multipart), {@code /v1/models} and {@code /health}, over one {@link TranscriptionModel}. Started
 * by the CLI when {@code --server} is given a transcription-only checkpoint; a chat server serves
 * chat models, this serves listeners.
 *
 * <p>Transcriptions run one at a time - the compute pool is the process-wide one, so two concurrent
 * utterances would only fight over it - while admission and parsing stay concurrent.
 */
public final class TranscriptionServer {

    private final TranscriptionModel<?, ?, ?> model;
    private final String servedModel;
    private final ServerConfig config;
    private final ReentrantLock compute = new ReentrantLock(true);

    private TranscriptionServer(
            TranscriptionModel<?, ?, ?> model, String servedModel, ServerConfig config) {
        this.model = model;
        this.servedModel = servedModel;
        this.config = config;
    }

    /** A running transport; the caller retains ownership of the model. */
    public static final class Running implements AutoCloseable {
        private final HttpServer http;
        private final int stopDelaySeconds;
        private final CountDownLatch stopped = new CountDownLatch(1);
        private volatile boolean closed;

        private Running(HttpServer http, int stopDelaySeconds) {
            this.http = http;
            this.stopDelaySeconds = stopDelaySeconds;
        }

        public InetSocketAddress address() {
            return http.getAddress();
        }

        /** Blocks until {@link #close()} is called. */
        public void await() throws InterruptedException {
            stopped.await();
        }

        @Override
        public synchronized void close() {
            if (closed) return;
            closed = true;
            http.stop(stopDelaySeconds);
            if (http.getExecutor() instanceof ExecutorService pool) pool.shutdownNow();
            stopped.countDown();
        }
    }

    /** Starts serving; does not block, prints nothing, owns no shutdown hook. */
    public static Running start(
            TranscriptionModel<?, ?, ?> model, String servedModel, ServerConfig config)
            throws IOException {
        if (model == null) throw new IllegalArgumentException("model is required");
        if (config == null) throw new IllegalArgumentException("config is required");
        return new TranscriptionServer(model, servedModel, config).serve();
    }

    private Running serve() throws IOException {
        HttpServer server = HttpServer.create(config.bind(), 0);
        Map<String, Object> modelCard =
                Map.of("id", servedModel, "object", "model", "created", 0, "owned_by", "jinfer");
        server.createContext(
                "/health",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    Http.sendJson(exchange, 200, Map.of("status", "ok"));
                });
        server.createContext(
                "/v1/models",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    String path = exchange.getRequestURI().getPath();
                    if (path.equals("/v1/models")) {
                        Http.sendJson(
                                exchange,
                                200,
                                Map.of("object", "list", "data", List.of(modelCard)));
                    } else if (path.equals("/v1/models/" + servedModel)) {
                        Http.sendJson(exchange, 200, modelCard);
                    } else {
                        Http.sendError(exchange, 404, "unknown path " + path);
                    }
                });
        server.createContext(
                "/v1/audio/transcriptions",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    if (!"/v1/audio/transcriptions".equals(exchange.getRequestURI().getPath())) {
                        Http.sendError(
                                exchange,
                                404,
                                "unknown path " + exchange.getRequestURI().getPath());
                        return;
                    }
                    if (Http.requireMethod(exchange, "POST")) return;
                    try {
                        handleTranscription(exchange);
                    } catch (IllegalArgumentException | IllegalStateException e) {
                        Http.sendErrorQuietly(exchange, 400, Http.errorMessage(e));
                    } catch (RuntimeException e) {
                        Http.sendErrorQuietly(exchange, 500, Http.errorMessage(e));
                    }
                });
        server.setExecutor(
                Executors.newFixedThreadPool(
                        config.limits().threads(),
                        runnable -> {
                            Thread thread = new Thread(runnable, "jinfer-transcription-handler");
                            thread.setDaemon(true);
                            return thread;
                        }));
        server.start();
        return new Running(server, Math.toIntExact(config.limits().shutdownTimeout().toSeconds()));
    }

    private void handleTranscription(HttpExchange exchange) throws IOException {
        String contentType =
                Objects.requireNonNullElse(
                        exchange.getRequestHeaders().getFirst("Content-Type"), "");
        String boundary = Multipart.boundary(contentType);
        if (boundary == null) {
            Http.sendError(
                    exchange,
                    400,
                    "expected multipart/form-data with a boundary (OpenAI audio API); got '"
                            + contentType
                            + "'");
            return;
        }
        byte[] body =
                Http.readBody(
                        exchange, config.limits().maxBodyBytes(), config.limits().requestTimeout());
        if (body == null) return; // readBody already answered
        Map<String, Multipart.Part> parts = Multipart.parse(body, boundary);
        Multipart.Part file = parts.get("file");
        if (file == null) {
            Http.sendError(exchange, 400, "multipart field 'file' is required");
            return;
        }
        String format =
                parts.containsKey("response_format") ? parts.get("response_format").text() : "json";
        if (!format.equals("json") && !format.equals("text") && !format.equals("verbose_json")) {
            Http.sendError(
                    exchange,
                    400,
                    "response_format must be json, text or verbose_json; got '" + format + "'");
            return;
        }

        Media.Audio audio;
        try {
            audio = AudioCodec.decode(file.content());
        } catch (IOException | IllegalArgumentException e) {
            Http.sendError(exchange, 400, "cannot decode audio: " + Http.errorMessage(e));
            return;
        }
        Transcription transcription;
        compute.lock();
        try {
            transcription = model.transcribe(audio.pcm());
        } finally {
            compute.unlock();
        }
        switch (format) {
            case "text" -> Http.sendText(exchange, 200, "text/plain", transcription.text() + "\n");
            case "json" -> Http.sendJson(exchange, 200, Map.of("text", transcription.text()));
            default -> {
                double duration = (double) audio.pcm().length / model.sampleRate();
                Map<String, Object> payload = new LinkedHashMap<>();
                payload.put("task", "transcribe");
                payload.put("duration", duration);
                payload.put("text", transcription.text());
                payload.put("words", words(transcription));
                Http.sendJson(exchange, 200, payload);
            }
        }
    }

    static List<Map<String, Object>> words(Transcription transcription) {
        List<Map<String, Object>> words = new ArrayList<>();
        for (Transcription.Word word : transcription.words()) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("word", word.text());
            entry.put("start", word.start());
            entry.put("end", word.end());
            entry.put("confidence", Math.round(word.confidence() * 1000) / 1000.0);
            words.add(entry);
        }
        return words;
    }

    /** The minimal RFC 7578 subset OpenAI clients emit: no nesting, one value per name. */
    static final class Multipart {

        private Multipart() {}

        record Part(byte[] content) {
            String text() {
                return new String(content, StandardCharsets.UTF_8).strip();
            }
        }

        static String boundary(String contentType) {
            String lower = contentType.toLowerCase(Locale.ROOT);
            if (!lower.startsWith("multipart/form-data")) return null;
            for (String parameter : contentType.split(";")) {
                String trimmed = parameter.strip();
                if (trimmed.toLowerCase(Locale.ROOT).startsWith("boundary=")) {
                    String value = trimmed.substring("boundary=".length()).strip();
                    if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\""))
                        value = value.substring(1, value.length() - 1);
                    return value.isEmpty() ? null : value;
                }
            }
            return null;
        }

        static Map<String, Part> parse(byte[] body, String boundary) {
            byte[] delimiter = ("--" + boundary).getBytes(StandardCharsets.UTF_8);
            Map<String, Part> parts = new LinkedHashMap<>();
            int at = indexOf(body, delimiter, 0);
            while (at >= 0) {
                int lineEnd = at + delimiter.length;
                // the final delimiter is "--boundary--"
                if (lineEnd + 1 < body.length && body[lineEnd] == '-' && body[lineEnd + 1] == '-')
                    break;
                int headersFrom = skipCrlf(body, lineEnd);
                int headersEnd =
                        indexOf(body, "\r\n\r\n".getBytes(StandardCharsets.UTF_8), headersFrom);
                if (headersEnd < 0) break;
                String headers =
                        new String(
                                body,
                                headersFrom,
                                headersEnd - headersFrom,
                                StandardCharsets.UTF_8);
                String name = dispositionName(headers);
                int contentFrom = headersEnd + 4;
                int next = indexOf(body, delimiter, contentFrom);
                if (next < 0) break;
                int contentEnd = next;
                // the CRLF before the next delimiter belongs to the framing, not the content
                if (contentEnd >= 2 && body[contentEnd - 2] == '\r' && body[contentEnd - 1] == '\n')
                    contentEnd -= 2;
                if (name != null) {
                    byte[] content = new byte[contentEnd - contentFrom];
                    System.arraycopy(body, contentFrom, content, 0, content.length);
                    parts.put(name, new Part(content));
                }
                at = next;
            }
            return parts;
        }

        private static int skipCrlf(byte[] body, int at) {
            if (at + 1 < body.length && body[at] == '\r' && body[at + 1] == '\n') return at + 2;
            return at;
        }

        private static String dispositionName(String headers) {
            for (String header : headers.split("\r\n")) {
                String lower = header.toLowerCase(Locale.ROOT);
                if (!lower.startsWith("content-disposition:")) continue;
                for (String parameter : header.split(";")) {
                    String trimmed = parameter.strip();
                    if (trimmed.toLowerCase(Locale.ROOT).startsWith("name=")) {
                        String value = trimmed.substring("name=".length()).strip();
                        if (value.length() >= 2 && value.startsWith("\"") && value.endsWith("\""))
                            value = value.substring(1, value.length() - 1);
                        return value;
                    }
                }
            }
            return null;
        }

        private static int indexOf(byte[] haystack, byte[] needle, int from) {
            outer:
            for (int i = Math.max(from, 0); i <= haystack.length - needle.length; i++) {
                for (int j = 0; j < needle.length; j++) {
                    if (haystack[i + j] != needle[j]) continue outer;
                }
                return i;
            }
            return -1;
        }
    }
}
