package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelLoaderTest {

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
                Arena arena = Arena.ofConfined()) {
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
}
