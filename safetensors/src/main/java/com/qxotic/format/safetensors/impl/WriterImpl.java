package com.qxotic.format.safetensors.impl;

import com.qxotic.format.json.Json;
import com.qxotic.format.safetensors.Safetensors;
import com.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class WriterImpl {
    private final ByteBuffer headerSizeBuffer =
            ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

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
        if (writer.totalBytesWritten != safetensors.getTensorDataOffset()) {
            throw new IllegalStateException("header size mismatch while writing safetensors");
        }
    }

    private static byte[] getHeaderBytes(
            Map<String, String> metadata, Collection<TensorEntry> tensorEntries) {
        Map<String, Object> json = new LinkedHashMap<>();
        if (metadata != null && !metadata.isEmpty()) {
            json.put(ReaderImpl.METADATA_KEY, metadata);
        }
        for (TensorEntry entry : tensorEntries) {
            long start = entry.byteOffset();
            long end = start + entry.byteSize();
            json.put(
                    entry.name(),
                    Map.of(
                            "dtype", entry.dtype().toString(),
                            "shape", toList(entry.shape()),
                            "data_offsets", List.of(start, end)));
        }
        String header = Json.stringify(json, false);
        byte[] headerBytes = header.getBytes(StandardCharsets.UTF_8);
        return headerBytes;
    }

    private static List<Long> toList(long[] values) {
        List<Long> list = new java.util.ArrayList<>(values.length);
        for (long value : values) {
            list.add(value);
        }
        return list;
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
        writeFully(byteChannel, headerSizeBuffer.clear().putLong(value).flip());
    }

    static long padding(long position, long alignment) {
        long nextAlignedPosition = (position + alignment - 1) / alignment * alignment;
        return nextAlignedPosition - position;
    }
}
