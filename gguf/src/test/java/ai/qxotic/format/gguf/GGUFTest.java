package ai.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;
import java.util.List;
import java.util.Objects;
import org.junit.jupiter.api.Assertions;

abstract class GGUFTest {

    static byte[] writeToBytes(GGUF gguf) throws IOException {
        try (ByteArrayOutputStream bytesOut = new ByteArrayOutputStream();
                WritableByteChannel byteChannel = Channels.newChannel(bytesOut)) {
            GGUF.write(gguf, byteChannel);
            return bytesOut.toByteArray();
        }
    }

    static GGUF readFromBytes(byte[] bytes) throws IOException {
        try (ByteArrayInputStream bytesIn = new ByteArrayInputStream(bytes);
                ReadableByteChannel byteChannel = Channels.newChannel(bytesIn)) {
            return GGUF.read(byteChannel);
        }
    }

    static void testSerialization(GGUF expected) throws IOException {
        GGUF read = readFromBytes(writeToBytes(expected));
        assertEqualsGGUF(expected, read, true);
    }

    static void assertEqualsGGUF(GGUF a, GGUF b, boolean strictOrder) {
        assertEquals(a.getVersion(), b.getVersion());
        assertEquals(a.getAlignment(), b.getAlignment());

        if (strictOrder) {
            assertEquals(List.copyOf(a.getMetadataKeys()), List.copyOf(b.getMetadataKeys()));
            assertEquals(List.copyOf(a.getTensors()), List.copyOf(b.getTensors()));
        } else {
            assertEquals(a.getMetadataKeys(), b.getMetadataKeys());
            assertEquals(a.getTensors(), b.getTensors());
        }

        for (String key : a.getMetadataKeys()) {
            assertEquals(a.getType(key), b.getType(key));

            if (a.getType(key) == MetadataValueType.ARRAY) {
                assertEquals(a.getComponentType(key), b.getComponentType(key));
            } else {
                assertNull(a.getComponentType(key));
                assertNull(b.getComponentType(key));
            }

            Object aValue = a.getValue(Object.class, key);
            Object bValue = b.getValue(Object.class, key);
            Assertions.assertTrue(Objects.deepEquals(aValue, bValue));
        }
    }

    static Builder putValues(Builder builder) {
        return builder.putString("string", "bar")
                .putBoolean("bool", true)
                .putByte("int8", (byte) 123)
                .putUnsignedByte("uint8", (byte) -1)
                .putShort("int16", (short) 123)
                .putUnsignedShort("uint16", (short) -1)
                .putInteger("int32", 12345678)
                .putUnsignedInteger("uint32", -1)
                .putLong("int64", 1234567890123456L)
                .putUnsignedLong("uint64", -1L)
                .putFloat("float32", (float) Math.PI)
                .putDouble("float64", Math.E);
    }

    static Builder putArrays(Builder builder) {
        return builder.putArrayOfString("string", new String[] {"bar", "baz"})
                .putArrayOfBoolean("bool", new boolean[] {true, false})
                .putArrayOfByte("int8", new byte[] {123, 42})
                .putArrayOfUnsignedByte("uint8", new byte[] {123, -1})
                .putArrayOfShort("int16", new short[] {12345, -1})
                .putArrayOfUnsignedShort("uint16", new short[] {12345, -1})
                .putArrayOfInteger("int32", new int[] {12345678})
                .putArrayOfUnsignedInteger("uint32", new int[] {12345678, -1})
                .putArrayOfLong("int64", new long[] {1234567890123456L})
                .putArrayOfUnsignedLong("uint64", new long[] {1234567890123456L, -1})
                .putArrayOfFloat("float32", new float[] {(float) Math.PI})
                .putArrayOfDouble("float64", new double[] {Math.E});
    }
}
