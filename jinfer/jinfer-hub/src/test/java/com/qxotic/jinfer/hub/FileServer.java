package com.qxotic.jinfer.hub;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A local HTTP file server for download tests: honors single {@code Range} requests the way every
 * model host does (206 + Content-Range), records hits and the last Range and query per path, and
 * can be told to IGNORE Range for a path - the "some servers answer 200 with the whole file"
 * downgrade {@code Fetch.sizeOf} is built around.
 */
final class FileServer implements AutoCloseable {

    private final HttpServer server;
    private final Map<String, byte[]> files = new ConcurrentHashMap<>();
    private final Map<String, AtomicInteger> hits = new ConcurrentHashMap<>();
    private final Map<String, String> lastRange = new ConcurrentHashMap<>();
    private final Map<String, String> lastQuery = new ConcurrentHashMap<>();
    private final List<String> noRange = new CopyOnWriteArrayList<>();

    private FileServer(HttpServer server) {
        this.server = server;
    }

    static FileServer start() throws IOException {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(
                Executors.newCachedThreadPool(
                        r -> {
                            Thread t = new Thread(r, "file-test-server");
                            t.setDaemon(true);
                            return t;
                        }));
        FileServer files = new FileServer(server);
        server.createContext("/", files::handle);
        server.start();
        return files;
    }

    FileServer serve(String path, byte[] payload) {
        files.put(path, payload);
        return this;
    }

    FileServer serve(String path, String payload) {
        return serve(path, payload.getBytes(StandardCharsets.UTF_8));
    }

    /** This path answers every GET, ranged or not, with 200 and the WHOLE body. */
    FileServer ignoringRange(String path) {
        noRange.add(path);
        return this;
    }

    String url(String path) {
        return "http://127.0.0.1:" + server.getAddress().getPort() + path;
    }

    int hits(String path) {
        return hits.getOrDefault(path, new AtomicInteger()).get();
    }

    String lastRange(String path) {
        return lastRange.get(path);
    }

    String lastQuery(String path) {
        return lastQuery.get(path);
    }

    private void handle(com.sun.net.httpserver.HttpExchange exchange) throws IOException {
        String path = exchange.getRequestURI().getPath();
        String query = exchange.getRequestURI().getQuery();
        if (query != null) {
            lastQuery.put(path, query);
        }
        byte[] payload = files.get(path);
        if (payload == null) {
            exchange.sendResponseHeaders(404, -1);
            exchange.close();
            return;
        }
        hits.computeIfAbsent(path, p -> new AtomicInteger()).incrementAndGet();
        String range = exchange.getRequestHeaders().getFirst("Range");
        byte[] body = payload;
        int status = 200;
        if (range != null && range.startsWith("bytes=") && !noRange.contains(path)) {
            lastRange.put(path, range);
            String[] ends = range.substring("bytes=".length()).split("-", 2);
            long start = Long.parseLong(ends[0]);
            long end =
                    ends.length > 1 && !ends[1].isEmpty()
                            ? Math.min(Long.parseLong(ends[1]), payload.length - 1)
                            : payload.length - 1;
            body = Arrays.copyOfRange(payload, (int) start, (int) end + 1);
            exchange.getResponseHeaders()
                    .set("Content-Range", "bytes " + start + "-" + end + "/" + payload.length);
            status = 206;
        }
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(body);
        }
        exchange.close();
    }

    @Override
    public void close() {
        server.stop(0);
    }
}
