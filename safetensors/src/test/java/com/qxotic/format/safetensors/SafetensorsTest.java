package com.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.List;

public abstract class SafetensorsTest {

    static byte[] writeToBytes(Safetensors safetensors) throws IOException {
        try (ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
                WritableByteChannel byteChannel = Channels.newChannel(bytesOut)) {
            Safetensors.write(safetensors, byteChannel);
            return bytesOut.toByteArray();
        }
    }

    static Safetensors readFromBytes(byte[] bytes) throws IOException {
        try (ByteArrayInputStream bytesIn = new ByteArrayInputStream(bytes);
                ReadableByteChannel byteChannel = Channels.newChannel(bytesIn)) {
            return Safetensors.read(byteChannel);
        }
    }

    static void testSerialization(Safetensors expected) throws IOException {
        Safetensors read = readFromBytes(writeToBytes(expected));
        assertEqualsSafetensors(expected, read, true);
    }

    static void assertEqualsSafetensors(Safetensors a, Safetensors b, boolean strictOrder) {
        assertEquals(a.getAlignment(), b.getAlignment());

        if (strictOrder) {
            assertEquals(
                    List.copyOf(a.getMetadata().keySet()), List.copyOf(b.getMetadata().keySet()));
            assertEquals(List.copyOf(a.getTensors()), List.copyOf(b.getTensors()));
        } else {
            assertEquals(a.getMetadata(), b.getMetadata());
            assertEquals(a.getTensors(), b.getTensors());
        }
    }
}
