package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
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

    private static boolean helperAlive(String name) {
        for (Thread t : Thread.getAllStackTraces().keySet())
            if (t.getName().equals(name) && t.isAlive()) return true;
        return false;
    }
}
