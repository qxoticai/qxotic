package com.llm4j.gguf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

public class ReadWriteTest extends GGUFTest {

    @Test
    public void testValues() throws IOException {
        GGUFTest.testSerialization(
                Builder.newBuilder()
                        .putString("string", "bar")
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
                        .putDouble("float64", Math.E)
                        .build()
        );
    }

    @Test
    public void testArrays() throws IOException {
        GGUFTest.testSerialization(
                Builder.newBuilder()
                        .putArrayOfString("string", new String[]{"bar", "baz"})
                        .putArrayOfBoolean("bool", new boolean[]{true, false})
                        .putArrayOfByte("int8", new byte[]{123, 42})
                        .putArrayOfUnsignedByte("uint8", new byte[]{123, -1})
                        .putArrayOfShort("int16", new short[]{12345, -1})
                        .putArrayOfUnsignedShort("uint16", new short[]{12345, -1})
                        .putArrayOfInteger("int32", new int[]{12345678})
                        .putArrayOfUnsignedInteger("uint32", new int[]{12345678, -1})
                        .putArrayOfLong("int64", new long[]{1234567890123456L})
                        .putArrayOfUnsignedLong("uint64", new long[]{1234567890123456L, -1})
                        .putArrayOfFloat("float32", new float[]{(float) Math.PI})
                        .putArrayOfDouble("float64", new double[]{Math.E})
                        .build()
        );
    }

    @Test
    public void testNulls() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.putArrayOfString("string", null));
        assertThrows(NullPointerException.class, () -> builder.putString("string", null));
    }

    @Test
    public void testTensors() {
        GGUF gguf = Builder.newBuilder()
                .putTensor(TensorInfo.create("foo", new long[]{1, 2}, GGMLType.F32, -1))
                .putTensor(TensorInfo.create("bar", new long[]{3, 4}, GGMLType.I8, -1))
                .putTensor(TensorInfo.create("baz", new long[]{5, 6}, GGMLType.F16, -1))
                .build();
        assertEquals(3, gguf.getTensors().size());
        // Order must be preserved.
        assertEquals(List.of("foo", "bar", "baz"),
                gguf.getTensors().stream().map(TensorInfo::name).collect(Collectors.toList()));
        TensorInfo foo = gguf.getTensor("foo");
        assertEquals("foo", foo.name());
        assertEquals(GGMLType.F32, foo.ggmlType());
    }

    @Test
    public void testPutNullTensor() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.putTensor(null));
    }

    @Test
    public void testWriteFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("model.gguf");
        assertFalse(Files.exists(modelPath));
        GGUF gguf = Builder.newBuilder().putString("hello", "world").build();
        GGUF.write(gguf, modelPath);
        assertTrue(Files.exists(modelPath));
        assertTrue(Files.size(modelPath) > 0);
    }

    @Test
    public void testWriteToExistingFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("model.gguf");
        Files.createFile(modelPath);
        // Writing to existing file throws, to prevent accidental overwrite.
        GGUF gguf = Builder.newBuilder().putString("hello", "world").build();
        assertThrows(FileAlreadyExistsException.class, () -> GGUF.write(gguf, modelPath));
    }

    @Test
    public void testReadEmptyFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("empty.gguf");
        Files.createFile(modelPath);
        assertThrows(IOException.class, () -> GGUF.read(modelPath));
    }

    @Test
    public void testInvalidMagic() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid MAGIC header.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder()).putInt(0, 0xBADBEEF);

        assertThrows(IllegalArgumentException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testInvalidVersion() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid version.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder()).putInt(0, 0xBADBEEF);

        assertThrows(IllegalArgumentException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }
}
