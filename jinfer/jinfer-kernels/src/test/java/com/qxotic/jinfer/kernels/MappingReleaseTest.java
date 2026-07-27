package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * The weights-lifetime law: {@link ModelLoader#loadTensors} maps tensor data into an automatic
 * arena, so the mapping lives while any tensor slice is reachable and is UNMAPPED by GC once the
 * graph is dropped - loading models can never leak mappings for the life of the process. Verified
 * against the ground truth ({@code /proc/self/maps}), so it runs on Linux only.
 */
class MappingReleaseTest {

    @Test
    void droppedTensorGraphUnmaps() throws Exception {
        Assumptions.assumeTrue(
                Files.isReadable(Path.of("/proc/self/maps")), "needs /proc/self/maps (Linux)");
        Path file = Files.createTempFile("jinfer-mapping-release", ".bin");
        try {
            Files.write(file, new byte[1 << 20]);
            String marker = file.getFileName().toString();
            Map<String, GGMLTensorEntry> entries = load(file);
            assertTrue(mapped(marker), "mapping present while tensors are reachable");
            entries = null; // drop the whole tensor graph
            boolean unmapped = false;
            for (int i = 0; i < 50 && !unmapped; i++) { // Cleaner runs post-GC, not instantly
                System.gc();
                Thread.sleep(50);
                unmapped = !mapped(marker);
            }
            assertTrue(unmapped, "dropped tensor graph must be unmapped by GC");
        } finally {
            Files.deleteIfExists(file);
        }
    }

    private static Map<String, GGMLTensorEntry> load(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            return ModelLoader.loadTensors(
                    ch,
                    0,
                    List.of(TensorEntry.create("t", new long[] {256}, GGMLType.F32, 0)),
                    java.lang.foreign.Arena.ofAuto());
        }
    }

    private static boolean mapped(String marker) throws IOException {
        return Files.readAllLines(Path.of("/proc/self/maps")).stream()
                .anyMatch(line -> line.contains(marker));
    }
}
