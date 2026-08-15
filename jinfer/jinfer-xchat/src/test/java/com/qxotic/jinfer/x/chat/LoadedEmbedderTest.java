package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.EmbeddingModel;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Proxy;
import org.junit.jupiter.api.Test;

final class LoadedEmbedderTest {

    @Test
    void validatesAndResolvesVariableDimensions() {
        LoadedEmbedder<?> loaded = loaded(4, 2);

        assertTrue(loaded.supportsCustomDimensions());
        assertEquals(4, loaded.resolveDimension(null));
        assertEquals(2, loaded.resolveDimension(2));
        assertEquals(3, loaded.resolveDimension(3));
        assertEquals(4, loaded.resolveDimension(4));
        assertThrows(IllegalArgumentException.class, () -> loaded.resolveDimension(-1));
        assertThrows(IllegalArgumentException.class, () -> loaded.resolveDimension(0));
        assertThrows(IllegalArgumentException.class, () -> loaded.resolveDimension(1));
        assertThrows(IllegalArgumentException.class, () -> loaded.resolveDimension(5));
    }

    @Test
    void fixedWidthRejectsAnExplicitDimension() {
        LoadedEmbedder<?> loaded = loaded(4, 4);

        assertFalse(loaded.supportsCustomDimensions());
        assertEquals(4, loaded.resolveDimension(null));
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> loaded.resolveDimension(4));
        assertTrue(failure.getMessage().contains("fixed embedding dimension"));
    }

    @Test
    @SuppressWarnings({"rawtypes", "unchecked"})
    void legacyConstructorDefaultsToFixedWidth() {
        LoadedEmbedder<?> loaded =
                new LoadedEmbedder<>(
                        (EmbeddingModel) proxy(EmbeddingModel.class),
                        proxy(Tokenizer.class),
                        new int[0],
                        new int[0],
                        4,
                        "fixed",
                        "",
                        "");

        assertEquals(4, loaded.minimumDimension());
        assertFalse(loaded.supportsCustomDimensions());
    }

    @Test
    void constructorRejectsAnInvalidMinimum() {
        assertThrows(IllegalArgumentException.class, () -> loaded(4, 0));
        assertThrows(IllegalArgumentException.class, () -> loaded(4, 5));
    }

    @Test
    void nativeCopyIsExactAndTruncatedCopyIsNormalized() {
        LoadedEmbedder<?> loaded = loaded(4, 2);
        float[] source = {3, 4, 12, 7};
        MemoryView<MemorySegment> view =
                Views.wrap(MemorySegment.ofArray(source), DataType.FP32, Shape.flat(4));

        assertArrayEquals(source, loaded.copyEmbedding(view, 4));
        assertArrayEquals(new float[] {0.6f, 0.8f}, loaded.copyEmbedding(view, 2), 1e-6f);
        assertArrayEquals(new float[] {3, 4, 12, 7}, source, "source must stay unchanged");
    }

    @Test
    void zeroPrefixStaysFinite() {
        LoadedEmbedder<?> loaded = loaded(4, 2);
        MemoryView<MemorySegment> view =
                Views.wrap(
                        MemorySegment.ofArray(new float[] {0, 0, 1, 0}),
                        DataType.FP32,
                        Shape.flat(4));

        assertArrayEquals(new float[] {0, 0}, loaded.copyEmbedding(view, 2));
    }

    @Test
    void copyRejectsAnInvalidView() {
        LoadedEmbedder<?> loaded = loaded(4, 2);
        MemoryView<MemorySegment> shortView =
                Views.wrap(MemorySegment.ofArray(new float[3]), DataType.FP32, Shape.flat(3));
        MemoryView<MemorySegment> wrongType =
                Views.wrap(MemorySegment.ofArray(new byte[8]), DataType.FP16, Shape.flat(4));
        MemoryView<MemorySegment> dense =
                Views.wrap(MemorySegment.ofArray(new float[4]), DataType.FP32, Shape.flat(2, 2));
        MemoryView<MemorySegment> transposed =
                MemoryView.of(
                        dense.memory(),
                        0,
                        DataType.FP32,
                        Layout.of(Shape.flat(2, 2), Stride.flat(1, 2)));

        assertThrows(IllegalArgumentException.class, () -> loaded.copyEmbedding(shortView, 4));
        assertThrows(IllegalArgumentException.class, () -> loaded.copyEmbedding(wrongType, 4));
        assertThrows(IllegalArgumentException.class, () -> loaded.copyEmbedding(transposed, 4));
        assertThrows(IllegalArgumentException.class, () -> loaded.copyEmbedding(dense, 1));
        assertThrows(IllegalArgumentException.class, () -> loaded.copyEmbedding(dense, 5));
    }

    @Test
    void copyRejectsClosedMemory() {
        LoadedEmbedder<?> loaded = loaded(4, 2);
        Arena arena = Arena.ofShared();
        MemoryView<MemorySegment> view =
                Views.wrap(arena.allocate(4L * Float.BYTES), DataType.FP32, Shape.flat(4));
        arena.close();

        assertThrows(IllegalStateException.class, () -> loaded.copyEmbedding(view, 4));
    }

    @SuppressWarnings({"rawtypes", "unchecked"})
    private static LoadedEmbedder<?> loaded(int dimension, int minimumDimension) {
        EmbeddingModel model = proxy(EmbeddingModel.class);
        Tokenizer tokenizer = proxy(Tokenizer.class);
        return new LoadedEmbedder<>(
                model,
                tokenizer,
                new int[0],
                new int[0],
                dimension,
                minimumDimension,
                "test-embedder",
                "",
                "");
    }

    @SuppressWarnings("unchecked")
    private static <T> T proxy(Class<T> type) {
        return (T)
                Proxy.newProxyInstance(
                        type.getClassLoader(),
                        new Class<?>[] {type},
                        (proxy, method, args) -> null);
    }
}
