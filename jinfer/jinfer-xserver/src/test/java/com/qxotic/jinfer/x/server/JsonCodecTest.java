package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

class JsonCodecTest {

    @Test
    void rejectsAmbiguousOrMalformedInput() {
        assertThrows(
                RuntimeException.class,
                () -> JsonCodec.parse("{\"max_tokens\":1,\"max_tokens\":2}"));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        JsonCodec.parse(
                                new byte[] {'{', '"', 'x', '"', ':', '"', (byte) 0xff, '"', '}'}));
        JsonCodec.parse("{}".getBytes(StandardCharsets.UTF_8));
    }

    @Test
    void doesNotReflectRequestBodiesInErrors() {
        RuntimeException failure =
                assertThrows(
                        RuntimeException.class,
                        () -> JsonCodec.parse("{\"secret\":1,\"secret\":2}"));

        String message = Http.errorMessage(failure);

        assertEquals(-1, message.indexOf('\n'));
        assertTrue(message.length() <= 512);
        assertTrue(message.contains("Duplicate key"));
    }
}
