package com.qxotic.format.safetensors.impl;

import com.qxotic.format.json.Json;
import com.qxotic.format.safetensors.Safetensors;
import com.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class WriterImpl {

    private WriterImpl() {}

    /**
     * Where the tensor data of a new file starts: past the header, padded to a multiple of 8 as
     * upstream pads it (or of the alignment, when larger), so the tensor data is aligned.
     */
    static long computeTensorDataOffset(
            Map<String, String> metadata, Collection<TensorEntry> tensorEntries, int alignment) {
        long headerEnd = 8 + getHeaderBytes(metadata, tensorEntries).length;
        return headerEnd + padding(headerEnd, Math.max(8, alignment));
    }

    /**
     * Writes the header padded with spaces up to {@link Safetensors#getTensorDataOffset()}, so a
     * file read back is written with its own layout and its tensor offsets stay valid.
     */
    static void writeImpl(Safetensors safetensors, WritableByteChannel byteChannel)
            throws IOException {
        byte[] headerBytes = getHeaderBytes(safetensors.getMetadata(), safetensors.getTensors());
        long headerSize = safetensors.getTensorDataOffset() - 8; // the JSON and its padding
        if (headerSize < headerBytes.length) {
            throw new IllegalArgumentException(
                    "header of "
                            + headerBytes.length
                            + " bytes does not fit before the tensor data at offset "
                            + safetensors.getTensorDataOffset());
        }
        ByteBuffer buffer =
                ByteBuffer.allocate(Math.toIntExact(8 + headerSize))
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .putLong(headerSize)
                        .put(headerBytes);
        while (buffer.hasRemaining()) {
            buffer.put((byte) ' ');
        }
        buffer.flip();
        while (buffer.hasRemaining()) {
            byteChannel.write(buffer);
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
            long end = Math.addExact(start, entry.byteSize());
            // insertion-ordered: Map.of iterates in a per-JVM random order, and the header bytes
            // must not change from one run to the next
            Map<String, Object> header = new LinkedHashMap<>();
            header.put("dtype", entry.dtype().toString());
            header.put("shape", box(entry.shape()));
            header.put("data_offsets", Arrays.asList(start, end));
            json.put(entry.name(), header);
        }
        return Json.stringify(json, false).getBytes(StandardCharsets.UTF_8);
    }

    private static List<Long> box(long[] shape) {
        List<Long> list = new ArrayList<>(shape.length);
        for (long value : shape) {
            list.add(value);
        }
        return list;
    }

    static long padding(long position, long alignment) {
        long nextAlignedPosition = (position + alignment - 1) / alignment * alignment;
        return nextAlignedPosition - position;
    }
}
