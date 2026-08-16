package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class HttpTest {

    @Test
    void bodyLimitAcceptsTheBoundaryAndRejectsTheNextByte() throws Exception {
        TestExchange exact = new TestExchange("1234".getBytes(StandardCharsets.UTF_8));
        assertArrayEquals("1234".getBytes(StandardCharsets.UTF_8), Http.readBody(exact, 4));
        assertEquals(-1, exact.getResponseCode());

        TestExchange oversized = new TestExchange("12345".getBytes(StandardCharsets.UTF_8));
        assertNull(Http.readBody(oversized, 4));
        assertEquals(413, oversized.getResponseCode());
        assertTrue(
                new String(oversized.responseBytes(), StandardCharsets.UTF_8).contains("4-byte"));
    }
}
