package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

final class WriterImpl {
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.nativeOrder());

    private final Safetensors safetensors;
    private long totalBytesWritten;

    private WriterImpl(Safetensors safetensors) {
        this.safetensors = safetensors;
    }

    static long computeTensorDataOffset(
            Map<String, String> metadata, Collection<TensorEntry> tensorEntries, int alignment) {
        byte[] headerBytes = getHeaderBytes(metadata, tensorEntries);
        int paddingSpaces = Math.toIntExact(padding(8 + headerBytes.length, alignment));
        return 8 + headerBytes.length + paddingSpaces;
    }

    static void writeImpl(Safetensors safetensors, WritableByteChannel byteChannel)
            throws IOException {
        WriterImpl writer = new WriterImpl(safetensors);
        byte[] headerBytes = getHeaderBytes(safetensors.getMetadata(), safetensors.getTensors());
        int paddingSpaces =
                Math.toIntExact(padding(8 + headerBytes.length, safetensors.getAlignment()));
        // The length of the header + padding, in bytes.
        writer.writeLong(byteChannel, headerBytes.length + paddingSpaces);
        // The string as a UTF-8 non-null-terminated string.
        writer.writeBytes(byteChannel, headerBytes);
        // Always align, even if there are no tensors.
        writer.writePaddingWithSpaces(byteChannel, paddingSpaces);
        assert writer.totalBytesWritten == safetensors.getTensorDataOffset();
    }

    private static byte[] getHeaderBytes(
            Map<String, String> metadata, Collection<TensorEntry> tensorEntries) {
        Map<String, Object> json = toJSON(metadata, tensorEntries);
        String header = JSON.stringify(json);
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        return headerBytes;
    }

    private void writeFully(WritableByteChannel byteChannel, ByteBuffer byteBuffer)
            throws IOException {
        while (byteBuffer.hasRemaining()) {
            this.totalBytesWritten += byteChannel.write(byteBuffer);
        }
    }

    private void writePaddingWithSpaces(WritableByteChannel byteChannel, int padding)
            throws IOException {
        if (padding == 0) {
            return;
        }
        byte[] paddingSpaces = new byte[padding];
        Arrays.fill(paddingSpaces, (byte) ' ');
        writeFully(byteChannel, ByteBuffer.wrap(paddingSpaces));
    }

    private void writeBytes(WritableByteChannel byteChannel, byte[] bytes) throws IOException {
        writeFully(byteChannel, ByteBuffer.wrap(bytes));
    }

    private void writeLong(WritableByteChannel byteChannel, long value) throws IOException {
        writeFully(byteChannel, BB_8.clear().putLong(value).flip());
    }

    private static Map<String, Object> toJSON(
            Map<String, String> metadata, Collection<TensorEntry> tensorEntries) {
        Map<String, Object> json = new LinkedHashMap<>();
        if (!metadata.isEmpty()) {
            json.put(ReaderImpl.METADATA_KEY, metadata);
        }
        for (TensorEntry tensorEntry : tensorEntries) {
            List<Long> shape =
                    Arrays.stream(tensorEntry.shape()).boxed().collect(Collectors.toList());
            long startOffset = tensorEntry.byteOffset();
            long endOffset = startOffset + tensorEntry.byteSize();
            json.put(
                    tensorEntry.name(),
                    Map.of(
                            "dtype", tensorEntry.dtype().toString(),
                            "shape", shape,
                            "data_offsets", List.of(startOffset, endOffset)));
        }
        return json;
    }

    static long padding(long position, long alignment) {
        long nextAlignedPosition = (position + alignment - 1) / alignment * alignment;
        return nextAlignedPosition - position;
    }
}
