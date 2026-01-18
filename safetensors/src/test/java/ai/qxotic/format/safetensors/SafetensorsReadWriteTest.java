package ai.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class SafetensorsReadWriteTest extends SafetensorsTest {

    @Test
    public void testMetadata() throws IOException {
        SafetensorsTest.testSerialization(
                Builder.newBuilder()
                        .putMetadataKey("format", "pt")
                        .putMetadataKey("model_type", "llama")
                        .putMetadataKey("version", "1.0")
                        .build());
    }

    @Test
    public void testNulls() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.putMetadataKey("key", null));
        assertThrows(NullPointerException.class, () -> builder.putMetadataKey(null, "value"));
    }

    @Test
    public void testTensors() {
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("foo", DType.F32, new long[] {1, 2}, 0))
                        .putTensor(TensorEntry.create("bar", DType.I8, new long[] {3, 4}, 0))
                        .putTensor(TensorEntry.create("baz", DType.F16, new long[] {5, 6}, 0))
                        .build();
        assertEquals(3, st.getTensors().size());
        assertEquals(
                List.of("foo", "bar", "baz"),
                st.getTensors().stream().map(TensorEntry::name).collect(Collectors.toList()));
        TensorEntry foo = st.getTensor("foo");
        assertEquals("foo", foo.name());
        assertEquals(DType.F32, foo.dtype());
    }

    @Test
    public void testPutNullTensor() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.putTensor(null));
    }

    @Test
    public void testWriteFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("model.safetensors");
        assertFalse(Files.exists(modelPath));
        Safetensors st = Builder.newBuilder().putMetadataKey("hello", "world").build();
        Safetensors.write(st, modelPath);
        assertTrue(Files.exists(modelPath));
        assertTrue(Files.size(modelPath) > 0);
    }

    @Test
    public void testWriteToExistingFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("model.safetensors");
        Files.createFile(modelPath);
        Safetensors st = Builder.newBuilder().putMetadataKey("hello", "world").build();
        assertThrows(FileAlreadyExistsException.class, () -> Safetensors.write(st, modelPath));
    }

    @Test
    public void testReadEmptyFile(@TempDir Path tempDir) throws IOException {
        Path modelPath = tempDir.resolve("empty.safetensors");
        Files.createFile(modelPath);
        assertThrows(IOException.class, () -> Safetensors.read(modelPath));
    }

    @Test
    public void testInvalidHeaderStart() throws IOException {
        byte[] stBytes = writeToBytes(Builder.newBuilder().putMetadataKey("foo", "bar").build());
        ByteBuffer headerBuffer = ByteBuffer.wrap(stBytes).order(ByteOrder.LITTLE_ENDIAN);
        long headerSize = headerBuffer.getLong(0);
        stBytes[8] = (byte) 'X';

        assertThrows(
                SafetensorsFormatException.class,
                () -> Safetensors.read(Channels.newChannel(new ByteArrayInputStream(stBytes))));
    }

    @Test
    public void testInvalidJson() {
        String invalidJson =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]";
        byte[] bytes = createSafetensorsBytes(invalidJson);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testAlignment() throws IOException {
        Safetensors st =
                Builder.newBuilder()
                        .setAlignment(64)
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {10, 10}, 0))
                        .putTensor(TensorEntry.create("tensor2", DType.F16, new long[] {20, 20}, 0))
                        .build();

        assertEquals(64, st.getAlignment());
        Safetensors roundTrip = readFromBytes(writeToBytes(st));
        assertEquals(64, roundTrip.getAlignment());
    }

    @Test
    public void testEmptyTensors() throws IOException {
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("empty", DType.F32, new long[] {0, 10}, 0))
                        .build();

        TensorEntry empty = st.getTensor("empty");
        assertEquals(0, empty.totalNumberOfElements());
        assertEquals(0, empty.byteSize());
    }

    @Test
    public void testScalarTensor() throws IOException {
        Safetensors st =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("scalar", DType.F32, new long[] {}, 0))
                        .build();

        TensorEntry scalar = st.getTensor("scalar");
        assertEquals(1, scalar.totalNumberOfElements());
        assertEquals(4, scalar.dtype().byteSizeForShape(scalar.shape()));
    }

    @Test
    public void testInvalidHeaderSize() {
        byte[] invalidHeader = new byte[8];
        ByteBuffer.wrap(invalidHeader).order(ByteOrder.LITTLE_ENDIAN).putLong(-1);

        assertThrows(
                SafetensorsFormatException.class,
                () ->
                        Safetensors.read(
                                Channels.newChannel(new ByteArrayInputStream(invalidHeader))));
    }

    @Test
    public void testOverlappingTensors() throws IOException {
        byte[] validBytes =
                writeToBytes(
                        Builder.newBuilder()
                                .putTensor(TensorEntry.create("t1", DType.F32, new long[] {2}, 0))
                                .putTensor(TensorEntry.create("t2", DType.F32, new long[] {2}, 4))
                                .build(false));

        String validJson =
                new String(
                        validBytes,
                        8,
                        (int) ByteBuffer.wrap(validBytes).order(ByteOrder.LITTLE_ENDIAN).getLong(0),
                        StandardCharsets.UTF_8);
        String invalidJson =
                validJson.replace("\"data_offsets\":[4,12]", "\"data_offsets\":[2,10]");

        byte[] headerBytes = invalidJson.getBytes(StandardCharsets.UTF_8);
        byte[] invalidBytes = new byte[8 + headerBytes.length];
        ByteBuffer.wrap(invalidBytes).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.length);
        System.arraycopy(headerBytes, 0, invalidBytes, 8, headerBytes.length);

        assertThrows(
                SafetensorsFormatException.class,
                () ->
                        Safetensors.read(
                                Channels.newChannel(new ByteArrayInputStream(invalidBytes))));
    }

    @Test
    public void testInvalidDtypeType() {
        String json = "{\"tensor\":{\"dtype\":123,\"shape\":[10],\"data_offsets\":[0,40]}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testInvalidShapeType() {
        String json =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":\"invalid\",\"data_offsets\":[0,40]}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testInvalidShapeElementType() {
        String json =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10,\"not_a_number\"],\"data_offsets\":[0,40]}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testInvalidDataOffsetsType() {
        String json =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":\"invalid\"}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testInvalidDataOffsetsSize() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0]}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testInvalidDataOffsetsElementType() {
        String json =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[\"invalid\",40]}}";
        byte[] bytes = createSafetensorsBytes(json);

        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    private byte[] createSafetensorsBytes(String json) {
        byte[] headerBytes = json.getBytes(StandardCharsets.UTF_8);
        byte[] bytes = new byte[8 + headerBytes.length];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).putLong(headerBytes.length);
        System.arraycopy(headerBytes, 0, bytes, 8, headerBytes.length);
        return bytes;
    }
}
