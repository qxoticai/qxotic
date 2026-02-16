package ai.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.stream.Collectors;
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
    public void testInvalidMagic() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid MAGIC header.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder()).putInt(0, 0xBADBEEF);

        assertThrows(
                GGUFFormatException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testInvalidVersion() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write invalid version.
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder()).putInt(4, 0xBADBEEF);

        assertThrows(
                GGUFFormatException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testUnsupportedVersion() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        // Write unsupported version (version 99).
        ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder()).putInt(4, 99);

        assertThrows(
                GGUFFormatException.class,
                () -> GGUF.read(Channels.newChannel(new ByteArrayInputStream(ggufBytes))));
    }

    @Test
    public void testInvalidMetadataValueType() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());

        // Find the position of the metadata value type (after magic, version, tensor_count,
        // metadata_kv_count, and key string)
        // GGUF format: 4 bytes magic + 4 bytes version + 8 bytes tensor_count + 8 bytes
        // metadata_kv_count
        // Then: 8 bytes key length + key bytes + 4 bytes value type
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder());

        // Skip: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        int pos = 24;
        // Read key length
        long keyLength = buffer.getLong(pos);
        pos += 8 + (int) keyLength; // Skip key length and key bytes

        // Write invalid metadata value type (999)
        buffer.putInt(pos, 999);

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
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder());

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
        ByteBuffer buffer = ByteBuffer.wrap(ggufBytes).order(ByteOrder.nativeOrder());

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
                            metadataString("dup", "first"),
                            metadataString("dup", "second")
                        });
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
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
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testInvalidBooleanEncodingTreatsNonZeroAsTrue() throws IOException {
        byte[] ggufBytes =
                rawGguf(0, new byte[][] {metadataBoolRaw("flag", (byte) 2)});
        GGUF gguf = readFromBytes(ggufBytes);
        assertEquals(MetadataValueType.BOOL, gguf.getType("flag"));
        assertTrue(gguf.getValue(boolean.class, "flag"));
    }

    @Test
    public void testTruncatedHeaderThrows() throws IOException {
        byte[] ggufBytes = writeToBytes(Builder.newBuilder().putString("foo", "bar").build());
        for (int len = 0; len < 24; len++) {
            byte[] truncated = java.util.Arrays.copyOf(ggufBytes, len);
            assertThrows(IOException.class, () -> readFromBytes(truncated));
        }
    }

    @Test
    public void testTruncatedMetadataPayloadThrows() throws IOException {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString("foo", "bar")});
        byte[] truncated = java.util.Arrays.copyOf(ggufBytes, ggufBytes.length - 1);
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
        byte[] full = rawGguf(1, new byte[0][], new byte[][] {tensorInfo("w", new long[] {2, 3}, GGMLType.F32, 0)});
        byte[] truncated = java.util.Arrays.copyOf(full, full.length - 1);
        assertThrows(IOException.class, () -> readFromBytes(truncated));
    }

    @Test
    public void testOversizedStringLengthThrowsArithmeticException() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataStringWithLength("s", Long.MAX_VALUE)});
        assertThrows(ArithmeticException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testOversizedArrayLengthThrowsArithmeticException() {
        byte[] ggufBytes =
                rawGguf(
                        0,
                        new byte[][] {
                            metadataArrayHeaderWithLength("arr", MetadataValueType.INT32, Long.MAX_VALUE)
                        });
        assertThrows(ArithmeticException.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentWrongTypeFailsWhenRequested() throws IOException {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataString("general.alignment", "bad")});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentZeroFailsWhileReading() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataInt("general.alignment", 0)});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testAlignmentNonPowerOfTwoFailsWhileReading() {
        byte[] ggufBytes = rawGguf(0, new byte[][] {metadataInt("general.alignment", 3)});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorNameLongerThan64Fails() {
        String longName = "a".repeat(65);
        byte[] ggufBytes = rawGguf(1, new byte[0][], new byte[][] {tensorInfo(longName, new long[] {1}, GGMLType.F32, 0)});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorWithMoreThan4DimensionsFails() {
        byte[] ggufBytes =
                rawGguf(
                        1,
                        new byte[0][],
                        new byte[][] {tensorInfo("t", new long[] {1, 2, 3, 4, 5}, GGMLType.F32, 0)});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
    }

    @Test
    public void testTensorWithMisalignedOffsetFails() {
        byte[] ggufBytes = rawGguf(1, new byte[0][], new byte[][] {tensorInfo("t", new long[] {1}, GGMLType.F32, 1)});
        assertThrows(AssertionError.class, () -> readFromBytes(ggufBytes));
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
            writeInt(out, type.ordinal());
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
        out.write(ByteBuffer.allocate(4).order(ByteOrder.nativeOrder()).putInt(value).array());
    }

    private static void writeLong(ByteArrayOutputStream out, long value) throws IOException {
        out.write(ByteBuffer.allocate(8).order(ByteOrder.nativeOrder()).putLong(value).array());
    }

    private static byte[] rawBytes(int... values) {
        byte[] bytes = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            bytes[i] = (byte) values[i];
        }
        return bytes;
    }
}
