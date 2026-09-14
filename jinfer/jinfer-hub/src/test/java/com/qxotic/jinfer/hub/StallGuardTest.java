package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The stall guard against a live server that goes SILENT mid-body - the failure mode no timeout
 * covers: the connection stays open, no RST arrives, and without the guard the read parks until the
 * kernel's TCP timeout (minutes). This also proves the load-bearing assumption that closing the
 * HttpClient response stream from another thread actually unblocks a parked read.
 *
 * <p>Takes a few seconds by nature: three transfer attempts each have to be DETECTED as stalled.
 */
class StallGuardTest {

    private static HttpServer server;
    private static ExecutorService executor;
    private static CountDownLatch stop;
    private static final byte[] PAYLOAD = new byte[(int) Fetch.PARALLEL_FLOOR + 1000];

    /** Range starts whose FIRST request has already been swallowed without headers. */
    private static final Set<String> SWALLOWED = ConcurrentHashMap.newKeySet();

    @BeforeAll
    static void start() throws IOException {
        stop = new CountDownLatch(1);
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        executor =
                Executors.newCachedThreadPool(
                        Thread.ofPlatform().name("stall-test-server-", 0).daemon(true).factory());
        server.setExecutor(executor);
        server.createContext(
                "/stall.bin",
                exchange -> {
                    exchange.sendResponseHeaders(200, 200_000);
                    try (OutputStream out = exchange.getResponseBody()) {
                        out.write(new byte[1000]);
                        out.flush();
                        try {
                            stop.await(); // keep the response open without sending the rest
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    }
                });
        new Random(7).nextBytes(PAYLOAD);
        // accepts each distinct request once and never answers it - no status line, no headers -
        // then serves the retry normally. Seen from a CDN under a burst of 429s.
        server.createContext(
                "/silent-headers/",
                exchange -> {
                    String name = exchange.getRequestURI().getPath();
                    int size = name.endsWith("big.bin") ? PAYLOAD.length : 200_000;
                    String range = exchange.getRequestHeaders().getFirst("Range");
                    long start = 0, end = size - 1;
                    if (range != null) {
                        String[] bounds = range.substring("bytes=".length()).split("-", -1);
                        start = Long.parseLong(bounds[0]);
                        if (!bounds[1].isEmpty()) end = Math.min(end, Long.parseLong(bounds[1]));
                    }
                    boolean probe = start == end; // sizeOf-style 0-0 probes answer at once
                    if (!probe && SWALLOWED.add(name + "@" + start)) {
                        try {
                            stop.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        return;
                    }
                    exchange.getResponseHeaders().add("ETag", "\"silent-1\"");
                    exchange.getResponseHeaders().add("Accept-Ranges", "bytes");
                    long length = end - start + 1;
                    if (range != null) {
                        exchange.getResponseHeaders()
                                .add("Content-Range", "bytes " + start + "-" + end + "/" + size);
                        exchange.sendResponseHeaders(206, length);
                    } else {
                        exchange.sendResponseHeaders(200, length);
                    }
                    try (OutputStream out = exchange.getResponseBody()) {
                        out.write(PAYLOAD, (int) start, (int) length);
                    }
                });
        server.start();
    }

    @AfterAll
    static void shutdown() {
        stop.countDown();
        server.stop(0);
        executor.shutdownNow();
    }

    @Test
    void aSilentServerIsAbandonedInSecondsNotMinutes(@TempDir Path dir)
            throws InterruptedException {
        String previous = System.setProperty("jinfer.downloadStallSeconds", "1");
        try {
            String url = "http://127.0.0.1" + ":" + server.getAddress().getPort() + "/stall.bin";
            Path dest = dir.resolve("stall.bin");
            long t0 = System.nanoTime();
            assertThrows(
                    IOException.class, () -> Fetch.download(url, dest, 200_000, null, Map.of()));
            double seconds = (System.nanoTime() - t0) / 1e9;
            // 3 attempts, each detected within ~2s of silence; minutes would mean the guard is
            // dead and we are back to waiting for the kernel
            assertTrue(seconds < 30, "gave up after " + seconds + "s - the stall guard is dead");
            assertTrue(Files.exists(dir.resolve("stall.bin.part")), "the partial must survive");
            // the guard is a helper thread with a job, not a resident: it leaves once nothing is
            // watched, and the next transfer starts a fresh one
            long deadline = System.nanoTime() + 5_000_000_000L;
            while (helperAlive("jinfer-stall-guard") && System.nanoTime() < deadline)
                Thread.sleep(10);
            assertTrue(!helperAlive("jinfer-stall-guard"), "the stall guard outlived its work");
        } finally {
            if (previous == null) System.clearProperty("jinfer.downloadStallSeconds");
            else System.setProperty("jinfer.downloadStallSeconds", previous);
        }
    }

    @Test
    void aServerThatNeverSendsHeadersIsRetriedNotAwaitedForever(@TempDir Path dir)
            throws IOException {
        String previous = System.setProperty("jinfer.downloadStallSeconds", "1");
        try {
            String base = "http://127.0.0.1:" + server.getAddress().getPort() + "/silent-headers/";
            // one stream (below the parallel floor) and chunked (at it): both used to call
            // HttpClient.send with no timeout, so a swallowed request parked the pull for good
            for (String name : new String[] {"small.bin", "big.bin"}) {
                int size = name.equals("big.bin") ? PAYLOAD.length : 200_000;
                Path dest = dir.resolve(name);
                long t0 = System.nanoTime();
                Fetch.download(base + name, dest, size, null, Map.of());
                double seconds = (System.nanoTime() - t0) / 1e9;
                assertTrue(
                        seconds < 30,
                        name + " took " + seconds + "s - the header wait is unbounded");
                byte[] expected = java.util.Arrays.copyOf(PAYLOAD, size);
                assertArrayEquals(
                        expected, Files.readAllBytes(dest), name + " resumed wrong bytes");
            }
        } finally {
            if (previous == null) System.clearProperty("jinfer.downloadStallSeconds");
            else System.setProperty("jinfer.downloadStallSeconds", previous);
        }
    }

    private static boolean helperAlive(String name) {
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (t.getName().equals(name) && t.isAlive()) return true;
        return false;
    }
}
