package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

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
                        .build());
    }

    @Test
    public void testArrays() throws IOException {
        GGUFTest.testSerialization(
                Builder.newBuilder()
                        .putArrayOfString("string", new String[] {"bar", "baz"})
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
                        .putArrayOfDouble("float64", new double[] {Math.E})
                        .build());
    }

    @Test
    public void testNulls() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.putArrayOfString("string", null));
        assertThrows(NullPointerException.class, () -> builder.putString("string", null));
    }

    @Test
    public void testTensors() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("foo", new long[] {1, 2}, GGMLType.F32, -1))
                        .putTensor(TensorEntry.create("bar", new long[] {3, 4}, GGMLType.I8, -1))
                        .putTensor(TensorEntry.create("baz", new long[] {5, 6}, GGMLType.F16, -1))
                        .build();
        assertEquals(3, gguf.getTensors().size());
        // Order must be preserved.
        assertEquals(
                List.of("foo", "bar", "baz"),
                gguf.getTensors().stream().map(TensorEntry::name).collect(Collectors.toList()));
        TensorEntry foo = gguf.getTensor("foo");
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
    public void testHeaderIsLittleEndianRegardlessOfPlatform() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // "GGUF" magic, then uint32 version 3, then uint64 tensor count 0 and kv count 1.
        assertArrayEquals(
                new byte[] {'G', 'G', 'U', 'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
                Arrays.copyOf(ggufBytes, 20));
    }

    @Test
    public void testInvalidMagic() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid MAGIC header.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN).putInt(0, 0xBADBEEF);

        assertThrows(
                GGUFFormatException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testInvalidVersion() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid version.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, 0xBADBEEF);

        assertThrows(
                GGUFFormatException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testUnsupportedVersion() throws IOException {
        // v2 too: only version 3 is read
        for (int version : new int[] {1, 2, 4, 99}) {
            byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
            ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN).putInt(4, version);
            GGUFFormatException e =
                    assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
            assertTrue(e.getMessage().contains("version " + version), e.getMessage());
        }
    }

    @Test
    public void testInvalidMetadataValueType() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());

        // Find the position of the metadata value type (after magic, version, tensor_count,
        // metadata_kv_count, and key string)
        // GGUF format: 4 bytes magic + 4 bytes version + 8 bytes tensor_count + 8 bytes
        // metadata_kv_count
        // Then: 8 bytes key length + key bytes + 4 bytes value type
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN);

        // Skip: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        int pos = 24;
        // Read key length
        long keyLength = buffer.getLong(pos);
        pos += 8 + (int) keyLength; // Skip key length and key bytes

        // one past the last type, far past it, and 0xFFFFFFFF read as -1
        for (int bad : new int[] {MetadataValueType.values().length, 999, -1}) {
            buffer.putInt(pos, bad);
            GGUFFormatException e =
                    assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
            assertEquals("Invalid metadata value type index: " + bad, e.getMessage());
            assertNull(e.getCause());
        }
    }

    @Test
    public void testInvalidMetadataKeyFormatFails() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString("Bad.Key", "v")});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTooLongMetadataKeyFails() {
        String tooLongKey = "a".repeat(1 << 16);
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString(tooLongKey, "v")});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testInvalidArrayComponentType() throws IOException {
        byte[] ggufBytes =
                writeToBytes(
                        Builder.newBuilder()
                                .putArrayOfFloat("arr", new float[] {1.0f, 2.0f})
                                .build());

        // Find the position of the array component type
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN);

        // Skip: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        int pos = 24;
        // Read key length
        long keyLength = buffer.getLong(pos);
        pos += 8 + (int) keyLength; // Skip key length and key bytes
        pos += 4; // Skip the ARRAY value type

        // Write invalid array component type (888)
        buffer.putInt(pos, 888);

        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testNestedArrays() throws IOException {
        byte[] ggufBytes =
                writeToBytes(
                        Builder.newBuilder().putArrayOfInteger("arr", new int[] {1, 2, 3}).build());

        // Find the position of the array component type and change it to ARRAY (nested arrays not
        // supported)
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.LITTLE_ENDIAN);

        // Skip: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        int pos = 24;
        // Read key length
        long keyLength = buffer.getLong(pos);
        pos += 8 + (int) keyLength; // Skip key length and key bytes
        pos += 4; // Skip the ARRAY value type

        // Write ARRAY as component type (value 9 - ARRAY is the 10th enum value, so index 9)
        buffer.putInt(pos, 9);

        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testDuplicateMetadataKeyFails() {
        byte[] ggufBytes =
                rawGguf(
                        0,
                        new byte[][] {
                            metadataString("dup", "first"), metadataString("dup", "second")
                        });
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testDuplicateTensorNameFails() {
        byte[] ggufBytes =
                rawGguf(
                        2,
                        new byte[0][],
                        new byte[][] {
                            tensorInfo("dup", new long[] {1}, GGMLType.F32, 0),
                            tensorInfo("dup", new long[] {2}, GGMLType.F32, 32)
                        });
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testUnknownTensorTypeIdIsAFormatError() {
        for (int bad : new int[] {999, -1}) {
            byte[] info = tensorInfo("w", new long[] {1}, GGMLType.F32, 0);
            // the type id sits right before the trailing 8-byte offset
            ByteBuffer.wrap(info).order(ByteOrder.LITTLE_ENDIAN).putInt(info.length - 12, bad);
            byte[] ggufBytes = rawGguf(1, new byte[0][], new byte[][] {info});
            GGUFFormatException e =
                    assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
            // the id said once: fromId's own message is not chained after it
            assertEquals("Tensor w has unknown GGML type ID " + bad, e.getMessage());
            assertNull(e.getCause());
        }
    }

    @Test
    public void testTensorRankWithTheHighBitSetIsAFormatError() {
        byte[] info = tensorInfo("w", new long[] {1}, GGMLType.F32, 0);
        // uint32 n_dimensions sits right after the 8-byte length + 1-byte name
        ByteBuffer.wrap(info).order(ByteOrder.LITTLE_ENDIAN).putInt(9, 0x80000001);
        byte[] ggufBytes = rawGguf(1, new byte[0][], new byte[][] {info});
        GGUFFormatException e =
                assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
        assertTrue(e.getMessage().contains("dimension count"), e.getMessage());
    }

    @Test
    public void testNonPositiveTensorDimensionsAreFormatErrors() {
        for (long bad : new long[] {0, -1, Long.MIN_VALUE}) {
            byte[] ggufBytes =
                    rawGguf(
                            1,
                            new byte[0][],
                            new byte[][] {tensorInfo("w", new long[] {2, bad}, GGMLType.F32, 0)});
            GGUFFormatException e =
                    assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
            assertTrue(e.getMessage().contains("dimension 1 is not positive"), e.getMessage());
        }
    }

    @Test
    public void testTensorElementCountOverflowIsAFormatError() {
        byte[] ggufBytes =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {
                            tensorInfo("w", new long[] {1L << 40, 1L << 40}, GGMLType.F32, 0)
                        });
        GGUFFormatException e =
                assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
        assertTrue(e.getMessage().contains("more elements"), e.getMessage());
    }

    @Test
    public void testInvalidBooleanEncodingTreatsNonZeroAsTrue() throws IOException {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataBoolRaw("flag", (byte) 2)});
        GGUF gguf = readFromBytes(ggufBytes);
        assertEquals(MetadataValueType.BOOL, gguf.getType("flag"));
        assertTrue(gguf.getValue(boolean.class, "flag"));
    }

    @Test
    public void testTruncatedHeaderThrows() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        for (int len = 0; len < 24; len++) {
            byte[] truncated = Arrays.copyOf(ggufBytes, len);
            assertThrows(IOException.class, () -> readFromBytes(truncated));
        }
    }

    @Test
    public void testTruncatedMetadataPayloadThrows() throws IOException {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString("foo", "bar")});
        byte[] truncated = Arrays.copyOf(ggufBytes, ggufBytes.length - 1);
        assertThrows(IOException.class, () -> readFromBytes(truncated));
    }

    @Test
    public void testTruncatedArrayPayloadThrows() throws IOException {
        byte[] ggufBytes =
                rawGguf(
                        0,
                        new byte[][] {
                            metadataArrayHeaderWithLength("arr", MetadataValueType.INT32, 3),
                            rawBytes(1, 2)
                        });
        assertThrows(IOException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTruncatedTensorInfoThrows() throws IOException {
        byte[] full =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo("w", new long[] {2, 3}, GGMLType.F32, 0)});
        byte[] truncated = Arrays.copyOf(full, full.length - 1);
        assertThrows(IOException.class, () -> readFromBytes(truncated));
    }

    @Test
    public void testOversizedStringLengthIsAFormatError() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataStringWithLength("s", Long.MAX_VALUE)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testOversizedArrayLengthIsAFormatError() {
        byte[] ggufBytes =
                rawGguf(
                        0,
                        new byte[][] {
                            metadataArrayHeaderWithLength(
                                    "arr", MetadataValueType.INT32, Long.MAX_VALUE)
                        });
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentWrongTypeFailsWhenRequested() throws IOException {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString("general.alignment", "bad")});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentZeroFailsWhileReading() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataInt("general.alignment", 0)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentNonPowerOfTwoFailsWhileReading() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataInt("general.alignment", 3)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentNegativeFailsWhileReading() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataInt("general.alignment", -1)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorNameLongerThan63Fails() {
        String longName = "a".repeat(64);
        byte[] ggufBytes =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo(longName, new long[] {1}, GGMLType.F32, 0)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorNameLimitCountsUtf8Bytes() throws IOException {
        // 32 two-byte chars: 32 UTF-16 units but 64 UTF-8 bytes.
        String wide = "é".repeat(32);
        byte[] tooLong =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo(wide, new long[] {1}, GGMLType.F32, 0)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(tooLong));

        // The reader accepts llama.cpp's bound, GGML_MAX_NAME less its NUL: 63 bytes.
        String atLimit = "a".repeat(63);
        byte[] ok =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo(atLimit, new long[] {1}, GGMLType.F32, 0)});
        assertNotNull(readFromBytes(ok).getTensor(atLimit));
    }

    @Test
    public void testTensorWithMoreThan4DimensionsFails() {
        byte[] ggufBytes =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {
                            tensorInfo("t", new long[] {1, 2, 3, 4, 5}, GGMLType.F32, 0)
                        });
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorWithMisalignedOffsetFails() {
        byte[] ggufBytes =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo("t", new long[] {1}, GGMLType.F32, 1)});
        assertThrows(GGUFFormatException.class, () -> readFromBytes(ggufBytes));
    }

    /**
     * A tensor's size from its shape and type must match the room the file gives it: up to the next
     * tensor, padding only. Nested tensors (Q and K as views of a stacked QK) and shared bytes are
     * legal; a gap or a crossing means a type that sizes its tensors wrong.
     */
    @Test
    public void testTensorsFillTheirSlots() throws IOException {
        byte[] padded = tensors(f32("a", 1, 0), f32("b", 8, 32)); // 4 bytes, then padding to 32
        assertEquals(2, readFromBytes(padded).getTensors().size());
        byte[] stacked =
                tensors(f32("qk", 16, 0), f32("q", 8, 0), f32("k", 8, 32), f32("next", 1, 64));
        assertEquals(4, readFromBytes(stacked).getTensors().size());
        byte[] shared = tensors(f32("a", 8, 0), f32("tied", 8, 0));
        assertEquals(2, readFromBytes(shared).getTensors().size());

        var gap =
                assertThrows(
                        GGUFFormatException.class,
                        () -> readFromBytes(tensors(f32("a", 8, 0), f32("b", 8, 64))));
        assertTrue(
                gap.getMessage()
                        .contains(
                                "32 unused bytes lie between tensor a (32 bytes of F32 at offset 0)"
                                        + " and tensor b"),
                gap.getMessage());
        var late =
                assertThrows(
                        GGUFFormatException.class, () -> readFromBytes(tensors(f32("a", 8, 32))));
        assertTrue(
                late.getMessage()
                        .contains(
                                "32 unused bytes lie between the tensor data's start and tensor a"),
                late.getMessage());
        var crossing =
                assertThrows(
                        GGUFFormatException.class,
                        () -> readFromBytes(tensors(f32("a", 16, 0), f32("b", 16, 32))));
        assertTrue(
                crossing.getMessage()
                        .contains(
                                "overlap without one holding the other, tensor a (64 bytes of F32"
                                    + " at offset 0) and tensor b (64 bytes of F32 at offset 32)"),
                crossing.getMessage());
    }

    private static byte[] f32(String name, long elements, long offset) {
        return tensorInfo(name, new long[] {elements}, GGMLType.F32, offset);
    }

    private static byte[] tensors(byte[]... infos) {
        return rawGguf(infos.length, new byte[0][], infos);
    }

    private static byte[] rawGguf(int tensorCount, byte[][] metadataEntries) {
        return rawGguf(tensorCount, metadataEntries, new byte[0][]);
    }

    private static byte[] rawGguf(int tensorCount, byte[][] metadataEntries, byte[][] tensorInfos) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeInt(out, 0x46554747);
            writeInt(out, 3);
            writeLong(out, tensorCount);
            writeLong(out, metadataEntries.length);
            for (byte[] entry : metadataEntries) {
                out.write(entry);
            }
            for (byte[] tensorInfo : tensorInfos) {
                out.write(tensorInfo);
            }
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] metadataString(String key, String value) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, key);
            writeInt(out, MetadataValueType.STRING.ordinal());
            writeString(out, value);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] metadataBoolRaw(String key, byte rawBool) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, key);
            writeInt(out, MetadataValueType.BOOL.ordinal());
            out.write(rawBool);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] metadataInt(String key, int value) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, key);
            writeInt(out, MetadataValueType.UINT32.ordinal());
            writeInt(out, value);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] metadataStringWithLength(String key, long len) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, key);
            writeInt(out, MetadataValueType.STRING.ordinal());
            writeLong(out, len);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] metadataArrayHeaderWithLength(
            String key, MetadataValueType componentType, long len) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, key);
            writeInt(out, MetadataValueType.ARRAY.ordinal());
            writeInt(out, componentType.ordinal());
            writeLong(out, len);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static byte[] tensorInfo(String name, long[] shape, GGMLType type, long offset) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            writeString(out, name);
            writeInt(out, shape.length);
            for (long d : shape) {
                writeLong(out, d);
            }
            writeInt(out, type.getId());
            writeLong(out, offset);
            return out.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void writeString(ByteArrayOutputStream out, String value) throws IOException {
        byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
        writeLong(out, bytes.length);
        out.write(bytes);
    }

    private static void writeInt(ByteArrayOutputStream out, int value) throws IOException {
        out.write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array());
    }

    private static void writeLong(ByteArrayOutputStream out, long value) throws IOException {
        out.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(value).array());
    }

    private static byte[] rawBytes(int... values) {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte) values[i];
        }
        return bytes;
    }

    @Test
    public void aCorruptCountIsAFormatError() throws IOException {
        // a truncated download whose kv_count field lands in tensor data used to surface as an
        // ArithmeticException or a multi-GB allocation, never as the format error it is
        byte[] bytes = GGUFTest.writeToBytes(Builder.newBuilder().putString("k", "v").build());
        for (int i = 16; i < 24; i++) bytes[i] = (byte) 0x7F; // metadata_kv_count
        Assertions.assertThrows(GGUFFormatException.class, () -> GGUFTest.readFromBytes(bytes));
    }
}
