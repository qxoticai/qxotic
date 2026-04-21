package com.qxotic.jota.examples.tokenizer;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.impl.Tiktoken;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.regex.Pattern;

public final class TokenizerDemoServer {
    private static final int PORT = 8080;

    private final Map<String, TokenizerEntry> tokenizers = new ConcurrentHashMap<>();

    private record TokenizerEntry(String name, String description, Tokenizer tokenizer) {}

    private TokenizerDemoServer() {
        initializeTokenizers();
    }

    public static void main(String[] args) throws Exception {
        new TokenizerDemoServer().start();
    }

    private void initializeTokenizers() {
        try {
            // GPT-2 / GPT-3 tokenizer (gpt2)
            tokenizers.put(
                    "gpt2",
                    new TokenizerEntry(
                            "GPT-2",
                            "GPT-2 tokenizer - 50,257 vocab",
                            loadTiktokenTokenizer(
                                    "gpt2",
                                    "tiktoken/r50k_base.tiktoken",
                                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
                                            + " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
                                    Map.of("<|endoftext|>", 50256))));

            // CL100k (GPT-4, GPT-3.5)
            tokenizers.put(
                    "cl100k_base",
                    new TokenizerEntry(
                            "CL100k",
                            "GPT-4 / GPT-3.5 tokenizer - 100,256 vocab",
                            loadTiktokenTokenizer(
                                    "cl100k_base",
                                    "tiktoken/cl100k_base.tiktoken",
                                    "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                                            + "\\n"
                                            + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+|"
                                            + " ?[^\\s\\p{L}\\p{N}]++[\\r"
                                            + "\\n"
                                            + "]*+|\\s++$|\\s*[\\r"
                                            + "\\n"
                                            + "]|\\s+(?!\\S)|\\s",
                                    Map.of(
                                            "<|endoftext|>", 100257,
                                            "<|fim_prefix|>", 100258,
                                            "<|fim_middle|>", 100259,
                                            "<|fim_suffix|>", 100260,
                                            "<|endofprompt|>", 100276))));

            // o200k (GPT-4o)
            tokenizers.put(
                    "o200k_base",
                    new TokenizerEntry(
                            "O200k",
                            "GPT-4o tokenizer - 200,019 vocab",
                            loadTiktokenTokenizer(
                                    "o200k_base",
                                    "tiktoken/o200k_base.tiktoken",
                                    "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                                            + "\\n"
                                            + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+|"
                                            + " ?[^\\s\\p{L}\\p{N}]++[\\r"
                                            + "\\n"
                                            + "]*+|\\s++$|\\s*[\\r"
                                            + "\\n"
                                            + "]|\\s+(?!\\S)|\\s",
                                    Map.of(
                                            "<|endoftext|>", 199999,
                                            "<|endofprompt|>", 200018))));

            // p50k (text-davinci-003)
            tokenizers.put(
                    "p50k_base",
                    new TokenizerEntry(
                            "P50k",
                            "text-davinci-003 tokenizer - 50,281 vocab",
                            loadTiktokenTokenizer(
                                    "p50k_base",
                                    "tiktoken/p50k_base.tiktoken",
                                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
                                            + " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
                                    Map.of("<|endoftext|>", 50256))));

        } catch (Exception e) {
            System.err.println("Warning: Failed to initialize some tokenizers: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private Tokenizer loadTiktokenTokenizer(
            String name, String resourcePath, String pattern, Map<String, Integer> specialTokens)
            throws IOException {

        InputStream stream = getClass().getClassLoader().getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new RuntimeException("Tokenizer resource not found: " + resourcePath);
        }

        Map<String, Integer> mergeableRanks;
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            mergeableRanks = Tiktoken.loadMergeableRanks(reader);
        }

        return Tiktoken.createFromTiktoken(
                name, mergeableRanks, Pattern.compile(pattern), specialTokens);
    }

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

    private void handleStatic(HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        if (path.equals("/")) {
            serveResource(exchange, "/web/tokenizer.html", "text/html; charset=utf-8");
            return;
        }
        if (path.equals("/tokenizer.js")) {
            serveResource(exchange, "/web/tokenizer.js", "application/javascript; charset=utf-8");
            return;
        }
        if (path.equals("/tokenizer.css")) {
            serveResource(exchange, "/web/tokenizer.css", "text/css; charset=utf-8");
            return;
        }
        send(exchange, 404, "Not found");
    }

    private void handleListTokenizers(HttpExchange exchange) throws IOException {
        if (!"GET".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }

        StringBuilder json = new StringBuilder();
        json.append('[');
        boolean first = true;
        for (Map.Entry<String, TokenizerEntry> entry : tokenizers.entrySet()) {
            if (!first) json.append(',');
            first = false;
            json.append('{');
            json.append("\"id\":\"").append(escapeJson(entry.getKey())).append("\",");
            json.append("\"name\":\"").append(escapeJson(entry.getValue().name())).append("\",");
            json.append("\"description\":\"")
                    .append(escapeJson(entry.getValue().description()))
                    .append("\"");
            json.append('}');
        }
        json.append(']');
        sendJson(exchange, json.toString());
    }

    private void handleTokenize(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }

        try {
            Map<String, String> params = parseFormData(exchange);
            String tokenizerId = params.get("tokenizer");
            String text = params.get("text");

            if (tokenizerId == null || text == null) {
                sendJson(exchange, "{\"error\":\"Missing tokenizer or text parameter\"}");
                return;
            }

            TokenizerEntry entry = tokenizers.get(tokenizerId);
            if (entry == null) {
                sendJson(
                        exchange,
                        "{\"error\":\"Unknown tokenizer: " + escapeJson(tokenizerId) + "\"}");
                return;
            }

            IntSequence tokens = entry.tokenizer().encode(text);
            Vocabulary vocab = entry.tokenizer().vocabulary();

            StringBuilder json = new StringBuilder();
            json.append('{');
            json.append("\"text\":\"").append(escapeJson(text)).append("\",");
            json.append("\"tokenCount\":").append(tokens.length()).append(',');
            json.append("\"characters\":").append(text.length()).append(',');
            json.append("\"tokens\":[");

            for (int i = 0; i < tokens.length(); i++) {
                if (i > 0) json.append(',');
                int tokenId = tokens.intAt(i);
                String tokenStr = vocab.token(tokenId);

                json.append('{');
                json.append("\"id\":").append(tokenId).append(',');
                json.append("\"text\":\"")
                        .append(escapeJson(tokenStr != null ? tokenStr : "<?>"))
                        .append("\",");
                json.append("\"decoded\":\"")
                        .append(escapeJson(decodeToken(entry.tokenizer(), tokenId)))
                        .append("\"");
                json.append('}');
            }
            json.append("]}");

            sendJson(exchange, json.toString());
        } catch (Exception e) {
            e.printStackTrace();
            sendJson(exchange, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
        }
    }

    private void handleDecode(HttpExchange exchange) throws IOException {
        if (!"POST".equals(exchange.getRequestMethod())) {
            send(exchange, 405, "Method not allowed");
            return;
        }

        try {
            Map<String, String> params = parseFormData(exchange);
            String tokenizerId = params.get("tokenizer");
            String tokensParam = params.get("tokens");

            if (tokenizerId == null || tokensParam == null) {
                sendJson(exchange, "{\"error\":\"Missing tokenizer or tokens parameter\"}");
                return;
            }

            TokenizerEntry entry = tokenizers.get(tokenizerId);
            if (entry == null) {
                sendJson(exchange, "{\"error\":\"Unknown tokenizer\"}");
                return;
            }

            String[] tokenStrings = tokensParam.split(",");
            int[] tokenIds = new int[tokenStrings.length];
            for (int i = 0; i < tokenStrings.length; i++) {
                tokenIds[i] = Integer.parseInt(tokenStrings[i].trim());
            }

            String decoded = entry.tokenizer().decode(IntSequence.wrap(tokenIds));

            StringBuilder json = new StringBuilder();
            json.append('{');
            json.append("\"text\":\"").append(escapeJson(decoded)).append("\"");
            json.append('}');

            sendJson(exchange, json.toString());
        } catch (Exception e) {
            e.printStackTrace();
            sendJson(exchange, "{\"error\":\"" + escapeJson(e.getMessage()) + "\"}");
        }
    }

    private String decodeToken(Tokenizer tokenizer, int tokenId) {
        try {
            return tokenizer.decode(IntSequence.wrap(new int[] {tokenId}));
        } catch (Exception e) {
            return "<?>";
        }
    }

    private Map<String, String> parseFormData(HttpExchange exchange) throws IOException {
        String body = new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
        Map<String, String> params = new HashMap<>();
        for (String pair : body.split("&")) {
            int eq = pair.indexOf('=');
            if (eq > 0) {
                String key = URLDecoder.decode(pair.substring(0, eq), StandardCharsets.UTF_8);
                String value = URLDecoder.decode(pair.substring(eq + 1), StandardCharsets.UTF_8);
                params.put(key, value);
            }
        }
        return params;
    }

    private void serveResource(HttpExchange exchange, String resource, String contentType)
            throws IOException {
        try (InputStream stream = getClass().getResourceAsStream(resource)) {
            if (stream == null) {
                send(exchange, 404, "Not found: " + resource);
                return;
            }
            byte[] bytes = stream.readAllBytes();
            Headers headers = exchange.getResponseHeaders();
            headers.set("Content-Type", contentType);
            exchange.sendResponseHeaders(200, bytes.length);
            try (OutputStream out = exchange.getResponseBody()) {
                out.write(bytes);
            }
        }
    }

    private void sendJson(HttpExchange exchange, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private void send(HttpExchange exchange, int code, String message) throws IOException {
        byte[] bytes = message.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(code, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private String escapeJson(String value) {
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
