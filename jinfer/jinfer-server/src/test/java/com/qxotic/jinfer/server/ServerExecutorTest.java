package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ServerExecutorTest {

    @Test
    void saturationAnswers503WithRetryAfterInsteadOfDroppingTheConnection() throws Exception {
        // the bounded executor queue rejected the excess inside the JDK server, which closed
        // the socket with no status; the gate answers like every other overload path
        AtomicInteger served = new AtomicInteger();
        Semaphore admissions = new Semaphore(1);
        var gated =
                Server.gated(
                        exchange -> served.incrementAndGet(),
                        admissions,
                        ServerConfig.Limits.DEFAULTS.withThreads(1));

        TestExchange ok = new TestExchange(new byte[0]);
        gated.handle(ok);
        assertTrue(ok.closed());
        assertEquals(1, served.get());
        assertEquals(1, admissions.availablePermits(), "the permit comes back");

        admissions.acquire();
        TestExchange busy = new TestExchange(new byte[0]);
        gated.handle(busy);
        assertEquals(503, busy.getResponseCode());
        assertEquals(
                String.valueOf(ServerConfig.Limits.DEFAULTS.withThreads(1).retryAfterSeconds()),
                busy.getResponseHeaders().getFirst("Retry-After"));
        assertTrue(busy.closed());
        assertEquals(
                0, admissions.availablePermits(), "a refused request owns no permit to release");
        assertEquals(1, served.get(), "a refused request never reaches the handler");
        assertTrue(Server.requestExecutor().getClass().getSimpleName().contains("ThreadPool"));
    }

    @Test
    void failedHandlersCloseTheirExchangeAndReturnThePermit() {
        Semaphore admissions = new Semaphore(1);
        AtomicInteger permitsOnClose = new AtomicInteger(-1);
        TestExchange exchange =
                new TestExchange(
                        InputStream.nullInputStream(),
                        new ByteArrayOutputStream() {
                            @Override
                            public void close() {
                                permitsOnClose.set(admissions.availablePermits());
                            }
                        });
        var gated =
                Server.gated(
                        request -> {
                            throw new IOException("response failed");
                        },
                        admissions,
                        ServerConfig.Limits.DEFAULTS.withThreads(1));
        assertThrows(IOException.class, () -> gated.handle(exchange));
        assertTrue(exchange.closed());
        assertEquals(0, permitsOnClose.get(), "cleanup still belongs to the admitted request");
        assertEquals(1, admissions.availablePermits());
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 16384})
    void interruptedResponseDoesNotConsumeTheShutdownGracePeriod(int headerBytes) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        var executor = Executors.newSingleThreadExecutor();
        CountDownLatch finished = new CountDownLatch(1);
        AtomicReference<IOException> failure = new AtomicReference<>();
        server.setExecutor(executor);
        server.createContext(
                "/",
                Server.gated(
                        exchange -> {
                            try {
                                if (headerBytes > 0)
                                    exchange.getResponseHeaders()
                                            .set("X-Padding", "x".repeat(headerBytes));
                                // Simulate worker shutdown interrupting the response writer.
                                Thread.currentThread().interrupt();
                                Http.sendText(exchange, 200, "text/plain", "finished");
                            } catch (IOException e) {
                                failure.set(e);
                            } finally {
                                Thread.interrupted();
                                finished.countDown();
                            }
                        },
                        new Semaphore(1),
                        ServerConfig.Limits.DEFAULTS.withThreads(1)));
        server.start();
        try (Socket client = new Socket("127.0.0.1", server.getAddress().getPort())) {
            client.getOutputStream()
                    .write(
                            "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
                                    .getBytes(StandardCharsets.US_ASCII));
            client.getOutputStream().flush();
            assertTrue(finished.await(5, TimeUnit.SECONDS), "handler did not finish");
            assertNotNull(failure.get(), "response write must fail to exercise shutdown cleanup");
            long start = System.nanoTime();
            server.stop(5);
            Duration elapsed = Duration.ofNanos(System.nanoTime() - start);
            assertTrue(
                    elapsed.compareTo(Duration.ofSeconds(3)) < 0,
                    "finished request held shutdown open for " + elapsed);
        } finally {
            server.stop(0);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}
