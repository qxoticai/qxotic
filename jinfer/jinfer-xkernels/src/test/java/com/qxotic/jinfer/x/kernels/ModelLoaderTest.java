package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.kernels.GGMLTensorEntry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Differential oracle over the REAL cycle-1 model (LFM2.5-2.6B-Q8_0.gguf, pulled in Phase 0): the x
 * loader vs jinfer-kernels ModelLoader on the same file — same tensor set, dtype mapping, logical
 * shapes (GGUF dims reversed), byte sizes, and spot-checked dequantized values. Skipped when the
 * model is not in the HF cache.
 */
class ModelLoaderTest {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");

    private static Path model;

    @BeforeAll
    static void findModel() throws IOException {
        Path repo = HF_CACHE.resolve("models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots");
        if (Files.isDirectory(repo)) {
            try (Stream<Path> snaps = Files.list(repo)) {
                model =
                        snaps.flatMap(
                                        s -> {
                                            try {
                                                return Files.list(s);
                                            } catch (IOException e) {
                                                return Stream.empty();
                                            }
                                        })
                                .filter(
                                        p ->
                                                p.getFileName()
                                                        .toString()
                                                        .equals("LFM2.5-2.6B-Q8_0.gguf"))
                                .findFirst()
                                .orElse(null);
            }
        }
    }

    private void withLoaders(LoaderCheck check) throws IOException {
        assumeTrue(model != null, "LFM2.5-2.6B-Q8_0.gguf not in the HF cache");
        Arena oldArena = Arena.ofAuto(), newArena = Arena.ofAuto(); // GC-owned, not closeable
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5");
            Map<String, GGMLTensorEntry> oldTensors =
                    com.qxotic.jinfer.kernels.ModelLoader.loadTensors(channel, gguf, oldArena);
            Map<String, MemoryView<MemorySegment>> newTensors =
                    ModelLoader.loadTensors(channel, gguf, newArena);
            check.run(oldTensors, newTensors);
        }
    }

    @FunctionalInterface
    interface LoaderCheck {
        void run(
                Map<String, GGMLTensorEntry> oldTensors,
                Map<String, MemoryView<MemorySegment>> newTensors)
                throws IOException;
    }

    @Test
    void sameTensorSetAndDtypes() throws IOException {
        withLoaders(
                (oldTensors, newTensors) -> {
                    assertEquals(oldTensors.keySet(), newTensors.keySet(), "tensor names");
                    for (var e : oldTensors.entrySet()) {
                        MemoryView<MemorySegment> view = newTensors.get(e.getKey());
                        GGMLType g = e.getValue().ggmlType();
                        DataType expected =
                                g == GGMLType.F32
                                        ? DataType.FP32
                                        : g == GGMLType.F16 ? DataType.FP16 : DataType.Q8_0;
                        assertEquals(expected, view.dataType(), e.getKey());
                    }
                });
    }

    @Test
    void logicalShapesMatchGgufReversed() throws IOException {
        withLoaders(
                (oldTensors, newTensors) -> {
                    for (var e : oldTensors.entrySet()) {
                        int[] ggufShape = e.getValue().shape();
                        Shape logical = newTensors.get(e.getKey()).logicalShape();
                        // GGUF fastest-first vs jota row-major slowest-first
                        for (int i = 0; i < ggufShape.length; i++) {
                            long expected = ggufShape[ggufShape.length - 1 - i];
                            assertEquals(expected, logical.size(i), e.getKey() + " dim " + i);
                        }
                    }
                });
    }

    @Test
    void byteSizesMatch() throws IOException {
        withLoaders(
                (oldTensors, newTensors) -> {
                    for (var e : oldTensors.entrySet()) {
                        GGMLTensorEntry old = e.getValue();
                        long elements = 1;
                        for (int d : old.shape()) elements *= d;
                        long ggufBytes = old.ggmlType().byteSizeFor(elements);
                        MemoryView<MemorySegment> view = newTensors.get(e.getKey());
                        assertEquals(
                                ggufBytes,
                                view.dataType().byteSizeFor(view.shape()),
                                e.getKey() + " byte size");
                    }
                });
    }

    @Test
    void dequantizedValuesMatch() throws IOException {
        withLoaders(
                (oldTensors, newTensors) -> {
                    // Q8_0 weight: old virtual getFloat vs raw block reads through the view
                    GGMLTensorEntry q8 = oldTensors.get("blk.0.ffn_gate.weight");
                    assumeTrue(q8 != null && q8.ggmlType() == GGMLType.Q8_0);
                    FloatTensor oldTensor = com.qxotic.jinfer.kernels.ModelLoader.loadQuantized(q8);
                    MemoryView<MemorySegment> view = newTensors.get("blk.0.ffn_gate.weight");
                    MemorySegment base = view.memory().base();
                    long vOff = view.byteOffset();
                    for (long idx = 0; idx < 32 * 40; idx += 997) {
                        long block = idx / 32, lane = idx % 32;
                        long bo = vOff + block * 34;
                        float scale =
                                Float.float16ToFloat(
                                        base.get(ValueLayout.JAVA_SHORT_UNALIGNED, bo));
                        float expected = base.get(ValueLayout.JAVA_BYTE, bo + 2 + lane) * scale;
                        assertEquals(oldTensor.getFloat(idx), expected, "q8 element " + idx);
                    }

                    // F32 norm: raw floats through the view equal the old tensor's
                    GGMLTensorEntry norm = oldTensors.get("blk.0.attn_norm.weight");
                    assumeTrue(norm != null && norm.ggmlType() == GGMLType.F32);
                    FloatTensor oldNorm = com.qxotic.jinfer.kernels.ModelLoader.toF32Tensor(norm);
                    MemoryView<MemorySegment> normView = newTensors.get("blk.0.attn_norm.weight");
                    MemorySegment nBase = normView.memory().base();
                    long nOff = normView.byteOffset();
                    for (long i = 0; i < normView.logicalSize(); i += 131) {
                        assertEquals(
                                oldNorm.getFloat(i),
                                nBase.get(ValueLayout.JAVA_FLOAT_UNALIGNED, nOff + i * 4),
                                "f32 element " + i);
                    }
                });
    }

    @Test
    void ropeFreqFactorsAbsent() throws IOException {
        // LFM2 uses plain RoPE: no rope_freqs.weight in this GGUF
        withLoaders(
                (oldTensors, newTensors) -> assertNull(ModelLoader.ropeFreqFactors(newTensors)));
    }

    @Test
    void firstPresentAndViewOrNull() throws IOException {
        withLoaders(
                (oldTensors, newTensors) -> {
                    assertNotNull(
                            ModelLoader.firstPresent(
                                    newTensors, "nope.weight", "token_embd.weight"));
                    assertNull(ModelLoader.firstPresent(newTensors, "nope.a", "nope.b"));
                    assertNull(ModelLoader.viewOrNull(newTensors, "nope.weight"));
                    assertTrue(
                            ModelLoader.viewOrNull(newTensors, "token_embd.weight").dataType()
                                    == DataType.Q8_0);
                });
    }
}
