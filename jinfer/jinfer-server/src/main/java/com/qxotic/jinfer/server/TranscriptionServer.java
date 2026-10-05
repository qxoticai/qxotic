package com.qxotic.jinfer.server;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The speech-to-text transport: {@code POST /v1/audio/transcriptions} (OpenAI-compatible
 * multipart), {@code /v1/models}, {@code /health}, {@code /props} and {@code /metrics}, over one
 * {@link TranscriptionModel}. Started by the CLI when {@code server} is given a transcription-only
 * checkpoint; a chat server serves chat models, this serves listeners.
 *
 * <p>Transcriptions run one at a time - the compute pool is the process-wide one, so two concurrent
 * utterances would only fight over it - while admission and parsing stay concurrent.
 */
public final class TranscriptionServer {

    private final TranscriptionModel<?, ?, ?> model;
    private final String servedModel;
    private final ServerConfig config;
    private final ReentrantLock compute = new ReentrantLock(true);
    private final TaskTransport transport;
    private final AtomicLong transcriptions = new AtomicLong(), audioMillis = new AtomicLong();

    private TranscriptionServer(
            TranscriptionModel<?, ?, ?> model, String servedModel, ServerConfig config)
            throws IOException {
        this.model = model;
        this.servedModel = servedModel;
        this.config = config;
        this.transport = new TaskTransport(config);
    }

    /** The scrape's view, in the language server's format and names where the meaning is shared. */
    private String exposition() {
        StringBuilder sb = new StringBuilder();
        Metrics.metric(sb, "jinfer_uptime_seconds", "gauge", transport.uptimeSeconds());
        Metrics.metric(
                sb, "jinfer_transcriptions_completed_total", "counter", transcriptions.get());
        Metrics.metric(
                sb, "jinfer_transcribed_audio_seconds_total", "counter", audioMillis.get() / 1e3);
        Metrics.metric(sb, "jinfer_transcriptions_in_flight", "gauge", transport.inFlight());
        return sb.toString();
    }

    /** A running transport; the caller retains ownership of the model. */
    public static final class Running implements AutoCloseable {
        private final TaskTransport.Running running;

        private Running(TaskTransport.Running running) {
            this.running = running;
        }

        public InetSocketAddress address() {
            return running.address();
        }

        /** Blocks until {@link #close()} is called. */
        public void await() throws InterruptedException {
            running.await();
        }

        @Override
        public void close() {
            running.close();
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

    private Running serve() {
        transport.probes(
                servedModel,
                "audio",
                () -> JsonCodec.object("model", servedModel, "sample_rate", model.sampleRate()),
                this::exposition);
        transport.work("/v1/audio/transcriptions", this::handleTranscription);
        return new Running(
                transport.start(
                        "POST /v1/audio/transcriptions", "jinfer-transcription-handler", () -> {}));
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
                        exchange, config.limits().maxBodyBytes(), config.limits().writeTimeout());
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
            // the decoder's own words (ffmpeg's stderr, a pointer address in it) are for the
            // operator's log; the client learns what it can act on
            Log.LOG.log(
                    System.Logger.Level.WARNING,
                    () -> "cannot decode an uploaded audio file: " + Http.errorMessage(e));
            Http.sendError(exchange, 400, "cannot decode audio: unsupported or corrupt file");
            return;
        }
        Transcription transcription;
        compute.lock();
        try {
            transcription = model.transcribe(audio.pcm());
        } finally {
            compute.unlock();
        }
        transcriptions.incrementAndGet();
        audioMillis.addAndGet(audio.pcm().length * 1000L / model.sampleRate());
        switch (format) {
            case "text" -> Http.sendText(exchange, 200, "text/plain", transcription.text() + "\n");
            case "json" ->
                    Http.sendJson(exchange, 200, JsonCodec.object("text", transcription.text()));
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
            entry.put("start", word.start().toNanos() / 1e9); // OpenAI's float seconds
            entry.put("end", word.end().toNanos() / 1e9);
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
