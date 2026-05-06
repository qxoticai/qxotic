package com.qxotic.jota.examples.tokenizer;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import com.qxotic.toknroll.hf.HuggingFaceTokenizerLoader;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public final class TokenizerDemoServer {
    private static final int PORT = 8080;

    private final Map<String, TokenizerEntry> tokenizers = new ConcurrentHashMap<>();

    private static final class TokenizerEntry {
        private final String name;
        private final String description;
        private final Callable<Tokenizer> factory;
        private volatile Tokenizer tokenizer;
        private volatile String loadError;

        TokenizerEntry(String name, String description, Callable<Tokenizer> factory) {
            this.name = name;
            this.description = description;
            this.factory = factory;
        }

        String name() { return name; }
        String description() { return description; }

        Tokenizer tokenizer() throws Exception {
            if (loadError != null) throw new IllegalStateException(loadError);
            Tokenizer t = tokenizer;
            if (t == null) {
                synchronized (this) {
                    if (loadError != null) throw new IllegalStateException(loadError);
                    t = tokenizer;
                    if (t == null) {
                        try {
                            tokenizer = t = factory.call();
                        } catch (Exception e) {
                            loadError = e.getMessage();
                            throw e;
                        }
                    }
                }
            }
            return t;
        }
    }

    private TokenizerDemoServer() {
        initializeTokenizers();
    }

    public static void main(String[] args) throws Exception {
        new TokenizerDemoServer().start();
    }

    // -- preset registration -------------------------------------------------

    private void initializeTokenizers() {
        register("hf_kimi_2_6", "HF Kimi 2.6", "moonshotai/Kimi-K2.6",
                () -> HuggingFaceTokenizerLoader.fromHuggingFace("moonshotai", "Kimi-K2.6"));

        register("hf_deepseek_v4_pro", "HF DeepSeek V4 Pro", "deepseek-ai/DeepSeek-V4-Pro",
                () -> HuggingFaceTokenizerLoader.fromHuggingFace(
                        "deepseek-ai", "DeepSeek-V4-Pro"));

        register("hf_gemma_4", "HF Gemma 4", "google/gemma-4-e2b-it",
                () -> HuggingFaceTokenizerLoader.fromHuggingFace("google", "gemma-4-e2b-it"));

        register("hf_mistral_tekken", "HF Mistral Tekken", "mistralai/ministral-8b-instruct-2410",
                () -> HuggingFaceTokenizerLoader.fromHuggingFace(
                        "mistralai", "ministral-8b-instruct-2410"));

        GGUFTokenizerLoader gguf = GGUFTokenizerLoader.createBuilderWithBuiltins().build();
        register("gguf_llama3_2_1b", "GGUF Llama-3.2-1B", "bartowski/Llama-3.2-1B-Instruct-GGUF",
                () -> gguf.fromHuggingFace(
                        "bartowski", "Llama-3.2-1B-Instruct-GGUF",
                        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"));

        register("gguf_qwen3_6", "GGUF Qwen3.6", "unsloth/Qwen3.6-35B-A3B-GGUF",
                () -> gguf.fromHuggingFace(
                        "unsloth", "Qwen3.6-35B-A3B-GGUF",
                        "Qwen3.6-35B-A3B-Q8_0.gguf"));

        register("gguf_granite_4_1", "GGUF Granite 4.1", "unsloth/granite-4.1-3b-GGUF",
                () -> gguf.fromHuggingFace(
                        "unsloth", "granite-4.1-3b-GGUF",
                        "granite-4.1-3b-Q8_0.gguf"));
    }

    private void register(
            String id, String name, String description, Callable<Tokenizer> factory) {
        tokenizers.put(id, new TokenizerEntry(name, description, factory));
    }

    // -- server --------------------------------------------------------------

    private void start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress(PORT), 0);
        server.createContext("/", this::handleStatic);
        server.createContext("/tokenizer.js", this::handleStatic);
        server.createContext("/tokenizer.css", this::handleStatic);
        server.createContext("/api/tokenizers", this::handleListTokenizers);
        server.createContext("/api/tokenize", this::handleTokenize);
        server.createContext("/api/decode", this::handleDecode);
        server.setExecutor(Executors.newFixedThreadPool(4));
        server.start();
        System.out.println("Tokenizer demo running on http://localhost:" + PORT);
    }

    // -- static file handlers ------------------------------------------------

    private static final Map<String, String> STATIC_TYPES = Map.of(
            "/", "/tokenizer/tokenizer.html;text/html; charset=utf-8",
            "/tokenizer.js", "/tokenizer/tokenizer.js;application/javascript; charset=utf-8",
            "/tokenizer.css", "/tokenizer/tokenizer.css;text/css; charset=utf-8");

    private void handleStatic(HttpExchange exchange) throws IOException {
        String entry = STATIC_TYPES.get(exchange.getRequestURI().getPath());
        if (entry == null) {
            sendText(exchange, 404, "Not found");
            return;
        }
        int semi = entry.indexOf(';');
        String resource = entry.substring(0, semi);
        String contentType = entry.substring(semi + 1);
        try (InputStream stream = getClass().getResourceAsStream(resource)) {
            if (stream == null) {
                sendText(exchange, 404, "Not found: " + resource);
                return;
            }
            writeBody(exchange, 200, stream.readAllBytes(),
                    h -> h.set("Content-Type", contentType));
        }
    }

    // -- API handlers --------------------------------------------------------

    private void handleListTokenizers(HttpExchange exchange) throws IOException {
        if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendText(exchange, 405, "Method not allowed");
            return;
        }
        StringBuilder json = new StringBuilder("[");
        boolean first = true;
        for (var e : tokenizers.entrySet()) {
            if (!first) json.append(',');
            first = false;
            json.append("{\"id\":\"").append(escapeJson(e.getKey())).append("\",");
            json.append("\"name\":\"").append(escapeJson(e.getValue().name())).append("\",");
            json.append("\"description\":\"")
                    .append(escapeJson(e.getValue().description()))
                    .append("\"}");
        }
        json.append(']');
        sendJson(exchange, json.toString());
    }

    private void handleTokenize(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendText(exchange, 405, "Method not allowed");
            return;
        }
        try {
            Map<String, String> p = parseForm(exchange);
            String tokenizerId = p.get("tokenizer");
            String text = p.get("text");
            if (tokenizerId == null || text == null) {
                sendJson(exchange, "{\"error\":\"Missing tokenizer or text\"}");
                return;
            }
            TokenizerEntry entry = tokenizers.get(tokenizerId);
            if (entry == null) {
                sendJson(exchange,
                        "{\"error\":\"Unknown tokenizer: " + escapeJson(tokenizerId) + "\"}");
                return;
            }
            Tokenizer tok = entry.tokenizer();
            IntSequence tokens = tok.encode(text);
            Vocabulary vocab = tok.vocabulary();

            StringBuilder json = new StringBuilder();
            json.append("{\"text\":\"").append(escapeJson(text)).append("\",");
            json.append("\"tokenCount\":").append(tokens.length()).append(',');
            json.append("\"characters\":").append(text.length()).append(',');
            json.append("\"tokens\":[");
            for (int i = 0; i < tokens.length(); i++) {
                if (i > 0) json.append(',');
                int id = tokens.intAt(i);
                String surface = vocab.token(id);
                json.append("{\"id\":").append(id).append(',');
                json.append("\"text\":\"")
                        .append(escapeJson(surface != null ? surface : "<?>"))
                        .append("\",");
                json.append("\"decoded\":\"").append(escapeJson(decodeToken(tok, id))).append("\"}");
            }
            json.append("]}");
            sendJson(exchange, json.toString());
        } catch (Exception e) {
            System.err.println("[tokenize] " + e);
            sendJson(exchange, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
        }
    }

    private void handleDecode(HttpExchange exchange) throws IOException {
        if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
            sendText(exchange, 405, "Method not allowed");
            return;
        }
        try {
            Map<String, String> p = parseForm(exchange);
            String tokenizerId = p.get("tokenizer");
            String tokensParam = p.get("tokens");
            if (tokenizerId == null || tokensParam == null) {
                sendJson(exchange, "{\"error\":\"Missing tokenizer or tokens\"}");
                return;
            }
            TokenizerEntry entry = tokenizers.get(tokenizerId);
            if (entry == null) {
                sendJson(exchange, "{\"error\":\"Unknown tokenizer\"}");
                return;
            }
            String[] parts = tokensParam.split(",");
            int[] ids = new int[parts.length];
            for (int i = 0; i < parts.length; i++) ids[i] = Integer.parseInt(parts[i].trim());
            String decoded = entry.tokenizer().decode(IntSequence.wrap(ids));
            sendJson(exchange, "{\"text\":\"" + escapeJson(decoded) + "\"}");
        } catch (Exception e) {
            System.err.println("[decode] " + e);
            sendJson(exchange, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
        }
    }

    // -- helpers -------------------------------------------------------------

    private static String decodeToken(Tokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.decode(IntSequence.wrap(new int[] {tokenId}));
        } catch (Exception e) {
            return "<?>";
        }
    }

    private static Map<String, String> parseForm(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        Map<String, String> params = new HashMap<>();
        for (String pair : body.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0) {
                params.put(
                        URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8),
                        URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8));
            }
        }
        return params;
    }

    // -- response writing ----------------------------------------------------

    private static void sendJson(HttpExchange exchange, String json) throws IOException {
        writeBody(exchange, 200, json.getBytes(StandardCharsets.UTF_8), h -> {
            h.set("Content-Type", "application/json; charset=utf-8");
            h.set("Access-Control-Allow-Origin", "*");
        });
    }

    private static void sendText(HttpExchange exchange, int code, String message)
            throws IOException {
        writeBody(exchange, code, message.getBytes(StandardCharsets.UTF_8), null);
    }

    private static void writeBody(
            HttpExchange exchange, int code, byte[] body, Consumer<Headers> headerConfig)
            throws IOException {
        if (headerConfig != null) headerConfig.accept(exchange.getResponseHeaders());
        exchange.sendResponseHeaders(code, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
    }

    // -- JSON escaping -------------------------------------------------------

    private static String escapeJson(String value) {
        if (value == null) return "";
        return value.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                .replace("\b", "\\b")
                .replace("\f", "\\f");
    }
}
