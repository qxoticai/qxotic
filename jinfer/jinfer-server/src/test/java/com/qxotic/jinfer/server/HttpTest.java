package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class HttpTest {

    @Test
    void bodyLimitAcceptsTheBoundaryAndRejectsTheNextByte() throws Exception {
        TestExchange exact = new TestExchange("1234".getBytes(StandardCharsets.UTF_8));
        assertArrayEquals(
                "1234".getBytes(StandardCharsets.UTF_8),
                Http.readBody(exact, 4, Duration.ofSeconds(1)));
        assertEquals(-1, exact.getResponseCode());

        TestExchange oversized = new TestExchange("12345".getBytes(StandardCharsets.UTF_8));
        assertNull(Http.readBody(oversized, 4, Duration.ofSeconds(1)));
        assertEquals(413, oversized.getResponseCode());
        assertTrue(
                new String(oversized.responseBytes(), StandardCharsets.UTF_8).contains("4-byte"));
    }

    @Test
    @Timeout(2)
    void bodyReadDeadlineClosesAStalledExchange() throws Exception {
        CountDownLatch closed = new CountDownLatch(1);
        InputStream stalled =
                new InputStream() {
                    @Override
                    public int read() throws IOException {
                        try {
                            closed.await();
                            return -1;
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IOException(e);
                        }
                    }

                    @Override
                    public void close() {
                        closed.countDown();
                    }
                };
        TestExchange exchange = new TestExchange(stalled, new ByteArrayOutputStream());

        assertNull(Http.readBody(exchange, 4, Duration.ofMillis(20)));
        assertTrue(exchange.closed());
    }

    @Test
    @Timeout(2)
    void completedBodyReadCancelsItsDeadline() throws Exception {
        TestExchange exchange = new TestExchange("{}".getBytes(StandardCharsets.UTF_8));

        assertArrayEquals(
                "{}".getBytes(StandardCharsets.UTF_8),
                Http.readBody(exchange, 4, Duration.ofMillis(100)));
        Thread.sleep(200);

        assertFalse(exchange.closed());
    }

    @Test
    void errorEnvelopeTypesFollowTheOpenAiSpelling() {
        assertEquals("authentication_error", Http.errorPayload(401, "no").get("type"));
        assertEquals("rate_limit_error", Http.errorPayload(429, "slow").get("type"));
        assertEquals("server_error", Http.errorPayload(503, "busy").get("type"));
        assertEquals("invalid_request_error", Http.errorPayload(400, "bad").get("type"));
    }
}
