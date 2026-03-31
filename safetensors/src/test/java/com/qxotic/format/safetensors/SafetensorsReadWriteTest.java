package com.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
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
    public void testPublicApiNullContracts(@TempDir Path tempDir) throws IOException {
        Safetensors st = Builder.newBuilder().build();
        Path out = tempDir.resolve("x.safetensors");

        assertThrows(NullPointerException.class, () -> Safetensors.read((Path) null));
        assertThrows(
                NullPointerException.class,
                () -> Safetensors.read((java.nio.channels.ReadableByteChannel) null));
        assertThrows(NullPointerException.class, () -> Safetensors.write(null, out));
        assertThrows(NullPointerException.class, () -> Safetensors.write(st, (Path) null));
        assertThrows(
                NullPointerException.class,
                () -> Safetensors.write(st, (java.nio.channels.WritableByteChannel) null));
        assertThrows(
                NullPointerException.class,
                () -> TensorEntry.create(null, DType.F32, new long[] {1}, 0));
        assertThrows(
                NullPointerException.class, () -> TensorEntry.create("t", null, new long[] {1}, 0));
        assertThrows(NullPointerException.class, () -> TensorEntry.create("t", DType.F32, null, 0));
        assertThrows(NullPointerException.class, () -> st.absoluteOffset(null));

        // sanity check: non-null writable channel still works
        Safetensors.write(st, Channels.newChannel(new ByteArrayOutputStream()));
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
    public void testTensorAbsoluteOffset() throws IOException {
        // Create a Safetensors with tensor data offset
        Safetensors st =
                Builder.newBuilder()
                        .putMetadataKey("format", "pt")
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {1}, 0))
                        .putTensor(TensorEntry.create("tensor2", DType.F32, new long[] {1}, 128))
                        .build();

        TensorEntry tensor1 = st.getTensor("tensor1");
        TensorEntry tensor2 = st.getTensor("tensor2");

        // absoluteOffset = tensorDataOffset + tensor.byteOffset()
        long expectedAbsolute1 = st.getTensorDataOffset() + tensor1.byteOffset();
        long expectedAbsolute2 = st.getTensorDataOffset() + tensor2.byteOffset();

        assertEquals(expectedAbsolute1, st.absoluteOffset(tensor1));
        assertEquals(expectedAbsolute2, st.absoluteOffset(tensor2));

        // Verify absolute offset is correct by checking offset difference
        assertEquals(
                tensor2.byteOffset() - tensor1.byteOffset(),
                st.absoluteOffset(tensor2) - st.absoluteOffset(tensor1));
    }

    @Test
    public void testTensorAbsoluteOffsetRoundTrip() throws IOException {
        // Build and write Safetensors
        Safetensors original =
                Builder.newBuilder()
                        .putMetadataKey("key", "value")
                        .putTensor(TensorEntry.create("weights", DType.F32, new long[] {100}, 0))
                        .build();

        // Write and read back
        Safetensors read = readFromBytes(writeToBytes(original));

        TensorEntry tensor = read.getTensor("weights");

        // absoluteOffset should work correctly on deserialized Safetensors
        long expectedAbsolute = read.getTensorDataOffset() + tensor.byteOffset();
        assertEquals(expectedAbsolute, read.absoluteOffset(tensor));
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

    @Test
    public void testTensorEntryMustBeObject() {
        String json = "{\"tensor\":123}";
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(json)));
    }

    @Test
    public void testDataOffsetsBeginGreaterThanEndRejected() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[40,0]}}";
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(json)));
    }

    @Test
    public void testEmptyHeaderIsValid() throws IOException {
        Safetensors st = readFromBytes(createSafetensorsBytes("{}"));
        assertTrue(st.getMetadata().isEmpty());
        assertTrue(st.getTensors().isEmpty());
    }

    @Test
    public void testUnknownTensorFieldRejected() {
        String json =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40],\"extra\":1}}";
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(json)));
    }

    @Test
    public void testMissingTensorFieldsRejected() {
        String missingDType = "{\"tensor\":{\"shape\":[10],\"data_offsets\":[0,40]}}";
        String missingShape = "{\"tensor\":{\"dtype\":\"F32\",\"data_offsets\":[0,40]}}";
        String missingOffsets = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10]}}";

        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(missingDType)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(missingShape)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(missingOffsets)));
    }

    @Test
    public void testUnsupportedDTypeRejected() {
        String json = "{\"tensor\":{\"dtype\":\"F128\",\"shape\":[10],\"data_offsets\":[0,40]}}";
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(json)));
    }

    @Test
    public void testNonIntegerNumericValuesRejected() {
        String nonIntegerShape =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10.5],\"data_offsets\":[0,40]}}";
        String nonIntegerOffsets =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40.5]}}";

        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(nonIntegerShape)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(nonIntegerOffsets)));
    }

    @Test
    public void testSizeTBoundsRejected() {
        String negativeShape =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[-1],\"data_offsets\":[0,40]}}";
        String tooLargeShape =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[281474976710656],\"data_offsets\":[0,40]}}";
        String negativeOffset =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[-1,40]}}";
        String tooLargeOffset =
                "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,281474976710656]}}";

        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(negativeShape)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(tooLargeShape)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(negativeOffset)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(tooLargeOffset)));
    }

    @Test
    public void testMetadataMustBeStringMap() {
        String metadataNotObject =
                "{\"__metadata__\":\"bad\",\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}}";
        String metadataNonStringValue =
                "{\"__metadata__\":{\"x\":1},\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}}";

        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(metadataNotObject)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(metadataNonStringValue)));
    }

    @Test
    public void testMetadataInvalidAlignmentRejected() {
        String metadataInvalidAlignment =
                "{\"__metadata__\":{\"__alignment__\":\"abc\"},\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}}";
        String metadataNonPowerOfTwoAlignment =
                "{\"__metadata__\":{\"__alignment__\":\"3\"},\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}}";
        String metadataZeroAlignment =
                "{\"__metadata__\":{\"__alignment__\":\"0\"},\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}}";

        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(metadataInvalidAlignment)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(metadataNonPowerOfTwoAlignment)));
        assertThrows(
                SafetensorsFormatException.class,
                () -> readFromBytes(createSafetensorsBytes(metadataZeroAlignment)));
    }

    @Test
    public void testHeaderDoesNotPersistDefaultAlignment() throws IOException {
        Safetensors st = Builder.newBuilder().setAlignment(32).build();
        String header = readHeaderJson(writeToBytes(st));
        assertFalse(header.contains("\"__alignment__\""));
    }

    @Test
    public void testHeaderPersistsCustomAlignment() throws IOException {
        Safetensors st = Builder.newBuilder().setAlignment(64).build();
        String header = readHeaderJson(writeToBytes(st));
        assertTrue(header.contains("\"__alignment__\""));
        assertTrue(header.contains("\"64\""));
    }

    @Test
    public void testToStringIncludesMetadataAndTensors() {
        Safetensors st =
                Builder.newBuilder()
                        .putMetadataKey("format", "pt")
                        .putTensor(TensorEntry.create("weight", DType.F32, new long[] {2, 2}, 0))
                        .build();

        String text = st.toString();
        assertTrue(text.contains("Safetensors {"));
        assertTrue(text.contains("\"__metadata__\""));
        assertTrue(text.contains("format"));
        assertTrue(text.contains("weight"));
        assertTrue(text.contains("F32"));
    }

    @Test
    public void testDuplicateMetadataKeyLastWins() throws IOException {
        String json =
                "{"
                        + "\"__metadata__\":{\"format\":\"pt\",\"format\":\"tf\"},"
                        + "\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}"
                        + "}";

        Safetensors st = readFromBytes(createSafetensorsBytes(json));
        assertEquals("tf", st.getMetadata().get("format"));
    }

    @Test
    public void testDuplicateTensorKeyLastWins() throws IOException {
        String json =
                "{"
                        + "\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
                        + "\"tensor\":{\"dtype\":\"F32\",\"shape\":[10],\"data_offsets\":[0,40]}"
                        + "}";

        Safetensors st = readFromBytes(createSafetensorsBytes(json));
        TensorEntry tensor = st.getTensor("tensor");
        assertArrayEquals(new long[] {10}, tensor.shape());
        assertEquals(40, tensor.byteSize());
    }

    @Test
    public void testPartialHeaderRead() throws IOException {
        // Create channel that returns fewer bytes than header size
        byte[] data = new byte[8]; // Just the size prefix
        ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).putLong(1000L); // Claim 1000 bytes
        ByteArrayInputStream bais = new ByteArrayInputStream(data);
        ReadableByteChannel channel = Channels.newChannel(bais);

        assertThrows(IOException.class, () -> Safetensors.read(channel));
    }

    @Test
    public void testTensorSizeMismatch() {
        // F32[2,2] = 16 bytes, but data_offsets says 10 bytes
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,10]}}";
        byte[] bytes = createSafetensorsBytes(json);
        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testNegativeEndOffset() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,-1]}}";
        byte[] bytes = createSafetensorsBytes(json);
        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testBeginGreaterThanEnd() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[10,5]}}";
        byte[] bytes = createSafetensorsBytes(json);
        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testNegativeBeginOffset() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[-5,4]}}";
        byte[] bytes = createSafetensorsBytes(json);
        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testDoubleInShape() {
        // JSON with Double values in shape array
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,4]}}";
        byte[] bytes = createSafetensorsBytes(json);
        assertThrows(SafetensorsFormatException.class, () -> readFromBytes(bytes));
    }

    @Test
    public void testNaNInShape() {
        String json = "{\"tensor\":{\"dtype\":\"F32\",\"shape\":[null],\"data_offsets\":[0,4]}}";
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

    private String readHeaderJson(byte[] safetensorsBytes) {
        int headerSize =
                Math.toIntExact(
                        ByteBuffer.wrap(safetensorsBytes).order(ByteOrder.LITTLE_ENDIAN).getLong());
        return new String(safetensorsBytes, 8, headerSize, StandardCharsets.UTF_8);
    }
}
