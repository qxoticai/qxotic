package com.qxotic.jota.examples.tokenizer;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.jtokkit.JTokkitTokenizers;
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
import java.util.Base64;
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
            mergeableRanks = loadMergeableRanks(reader);
        }

        return JTokkitTokenizers.fromTiktoken(
                name, mergeableRanks, Pattern.compile(pattern), specialTokens);
    }

    private static final char[] BYTE_SYMBOLS = {
        '\u0100', '\u0101', '\u0102', '\u0103', '\u0104', '\u0105', '\u0106', '\u0107', '\u0108',
                '\u0109', '\u010a', '\u010b', '\u010c', '\u010d', '\u010e', '\u010f', '\u0110', '\u0111', '\u0112', '\u0113', '\u0114',
                '\u0115', '\u0116', '\u0117', '\u0118', '\u0119', '\u011a', '\u011b', '\u011c', '\u011d', '\u011e', '\u011f', '\u0120',
                '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';',
        '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
        'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
        '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
        'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
        'x', 'y', 'z', '{', '|', '}', '~', '\u0121', '\u0122', '\u0123', '\u0124', '\u0125',
        '\u0126', '\u0127', '\u0128', '\u0129', '\u012a', '\u012b', '\u012c', '\u012d', '\u012e',
                '\u012f', '\u0130', '\u0131', '\u0132', '\u0133', '\u0134', '\u0135', '\u0136', '\u0137', '\u0138', '\u0139', '\u013a',
        '\u013b', '\u013c', '\u013d', '\u013e', '\u013f', '\u0140', '\u0141', '\u0142', '\u00a1',
                '\u00a2', '\u00a3', '\u00a4', '\u00a5', '\u00a6', '\u00a7', '\u00a8', '\u00a9', '\u00aa', '\u00ab', '\u00ac', '\u0143',
        '\u00ae', '\u00af', '\u00b0', '\u00b1', '\u00b2', '\u00b3', '\u00b4', '\u00b5', '\u00b6',
                '\u00b7', '\u00b8', '\u00b9', '\u00ba', '\u00bb', '\u00bc', '\u00bd', '\u00be', '\u00bf', '\u00c0', '\u00c1', '\u00c2',
        '\u00c3', '\u00c4', '\u00c5', '\u00c6', '\u00c7', '\u00c8', '\u00c9', '\u00ca', '\u00cb',
                '\u00cc', '\u00cd', '\u00ce', '\u00cf', '\u00d0', '\u00d1', '\u00d2', '\u00d3', '\u00d4', '\u00d5', '\u00d6', '\u00d7',
        '\u00d8', '\u00d9', '\u00da', '\u00db', '\u00dc', '\u00dd', '\u00de', '\u00df', '\u00e0',
                '\u00e1', '\u00e2', '\u00e3', '\u00e4', '\u00e5', '\u00e6', '\u00e7', '\u00e8', '\u00e9', '\u00ea', '\u00eb', '\u00ec',
        '\u00ed', '\u00ee', '\u00ef', '\u00f0', '\u00f1', '\u00f2', '\u00f3', '\u00f4', '\u00f5',
                '\u00f6', '\u00f7', '\u00f8', '\u00f9', '\u00fa', '\u00fb', '\u00fc', '\u00fd', '\u00fe', '\u00ff'
    };

    private static Map<String, Integer> loadMergeableRanks(BufferedReader reader) {
        Map<String, Integer> mergeableRanks = new HashMap<>();
        reader.lines().forEachOrdered(line -> {
            String[] parts = line.split(" ");
            byte[] bytes = Base64.getDecoder().decode(parts[0]);
            StringBuilder key = new StringBuilder(bytes.length);
            for (byte b : bytes) {
                key.append(BYTE_SYMBOLS[b & 0xFF]);
            }
            mergeableRanks.put(key.toString(), Integer.parseInt(parts[1]));
        });
        return mergeableRanks;
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
