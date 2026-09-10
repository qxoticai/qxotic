package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelLoaderTest {

    @Test
    void diagnosesConfinedWeightsBeforeMapping(@TempDir Path dir) throws Exception {
        Path file = Files.write(dir.resolve("weights.bin"), new byte[Float.BYTES]);
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofConfined()) {
            var failure =
                    assertThrows(
                            AssertionError.class,
                            () -> ModelLoader.loadTensors(channel, 0, java.util.List.of(), arena));
            assertTrue(failure.getMessage().contains("Confined arenas"));
            assertTrue(arena.scope().isAlive());
        }
    }

    /**
     * A GGUF whose tensor table promises more bytes than the file holds fails by name, not deep in
     * a view.
     */
    @Test
    void aTruncatedFileNamesTheTensorAndTheSizes(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("header-only.gguf");
        GGUF.write(
                Builder.newBuilder()
                        .putTensor(
                                TensorEntry.create(
                                        "blk.0.attn_q.weight", new long[] {4, 4}, GGMLType.F32, 0))
                        .build(),
                file);
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            GGUF gguf = GGUF.read(file);
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> ModelLoader.loadTensors(channel, gguf, arena));
            assertTrue(failure.getMessage().contains("blk.0.attn_q.weight"), failure.getMessage());
            assertTrue(failure.getMessage().contains("truncated"), failure.getMessage());
            assertTrue(failure.getMessage().contains("64"), failure.getMessage()); // 4*4 floats
        }
    }

    @Test
    void rejectsUnalignedEmbeddedTensorData(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("model.gguf");
        GGUF.write(Builder.newBuilder().build(), file);
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> ModelLoader.loadTensors(channel, 1, java.util.List.of(), arena));
            assertTrue(failure.getMessage().contains("4-byte aligned"), failure.getMessage());
        }
    }

    @Test
    void mapsAnEmbeddedGgufThroughItsRelocatedHeader(@TempDir Path dir) throws Exception {
        // a GGUF sitting at byte 4096 of a larger file, as an entry of a self-archive does
        Path standalone = dir.resolve("model.gguf");
        GGUF.write(
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("w", new long[] {2}, GGMLType.F32, 0))
                        .build(),
                standalone);
        byte[] header = Files.readAllBytes(standalone);
        long dataOffset = GGUF.read(standalone).getTensorDataOffset();
        byte[] payload =
                ByteBuffer.allocate(2 * Float.BYTES)
                        .order(ByteOrder.LITTLE_ENDIAN)
                        .putFloat(1.5f)
                        .putFloat(-2.5f)
                        .array();
        int base = 4096;
        byte[] archive = new byte[base + (int) dataOffset + payload.length];
        System.arraycopy(header, 0, archive, base, header.length);
        System.arraycopy(payload, 0, archive, base + (int) dataOffset, payload.length);
        Path file = Files.write(dir.resolve("archive"), archive);

        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            channel.position(base);
            GGUF embedded = GGUF.read(channel).at(base);
            var tensors = ModelLoader.loadTensors(channel, embedded, arena);
            MemorySegment w = tensors.get("w").memory().base();
            assertEquals(1.5f, w.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
            assertEquals(-2.5f, w.get(ValueLayout.JAVA_FLOAT_UNALIGNED, Float.BYTES));
        }
    }

    @Test
    void anEmbeddedGgufWhoseFileEndsEarlyNamesTheTensor(@TempDir Path dir) throws Exception {
        Path standalone = dir.resolve("model.gguf");
        GGUF.write(
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("w", new long[] {2}, GGMLType.F32, 0))
                        .build(),
                standalone);
        byte[] header = Files.readAllBytes(standalone);
        int base = 4096;
        byte[] archive = new byte[base + header.length]; // the tensor bytes never made it in
        System.arraycopy(header, 0, archive, base, header.length);
        Path file = Files.write(dir.resolve("archive"), archive);

        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            channel.position(base);
            GGUF embedded = GGUF.read(channel).at(base);
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> ModelLoader.loadTensors(channel, embedded, arena));
            assertTrue(failure.getMessage().contains("w"), failure.getMessage());
        }
    }

    @Test
    void anUnalignedRelocationIsRejected(@TempDir Path dir) throws Exception {
        Path standalone = dir.resolve("model.gguf");
        GGUF.write(Builder.newBuilder().build(), standalone);
        GGUF gguf = GGUF.read(standalone);
        try (FileChannel channel = FileChannel.open(standalone, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> ModelLoader.loadTensors(channel, gguf.at(3), arena));
            assertTrue(failure.getMessage().contains("4-byte aligned"), failure.getMessage());
        }
    }

    @Test
    void mapsOnlyTensorData(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("model.gguf");
        GGUF.write(
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("test", new long[] {1}, GGMLType.F32, 0))
                        .build(),
                file);
        GGUF gguf = GGUF.read(file);
        Files.write(file, new byte[68], StandardOpenOption.APPEND);

        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            var tensors = ModelLoader.loadTensors(channel, gguf, arena);
            assertEquals(Float.BYTES, tensors.get("test").memory().base().byteSize());
        }
    }

    @Test
    void validatesBoundedTensorData() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("weight", new long[] {4}, GGMLType.F32, 8))
                        .build();
        TensorEntry weight = gguf.getTensor("weight");
        long required = gguf.getTensorDataOffset() + weight.offset() + weight.byteSize();

        ModelLoader.requireComplete(gguf, required, "model.gguf");
        assertThrows(
                IllegalArgumentException.class,
                () -> ModelLoader.requireComplete(gguf, required - 1, "model.gguf"));
    }

    @Test
    void rejectsNegativeTensorOffsets() {
        GGUF gguf =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("weight", new long[] {1}, GGMLType.F32, -1))
                        .build(false);

        assertThrows(
                IllegalArgumentException.class,
                () -> ModelLoader.requireComplete(gguf, Long.MAX_VALUE, "model.gguf"));
    }
}
