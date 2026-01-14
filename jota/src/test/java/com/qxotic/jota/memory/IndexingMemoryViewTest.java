
package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class IndexingMemoryViewTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void linearToOffsetRowMajorMatchesPhysicalIndex(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3);
        Layout layout = Layout.rowMajor(shape);
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.F32, 0, shape.size(), 1);
        MemoryView<B> view = MemoryView.of(base.memory(), 0L, DataType.F32, layout);

        for (long linear = 0; linear < shape.size(); linear++) {
            long expectedOffset = expectedOffset(layout, linear, DataType.F32, view.byteOffset());
            float actual = memoryAccess.readFloat(view.memory(), expectedOffset);
            long[] coord = Indexing.linearToCoord(shape, linear);
            assertEquals(expectedOffset / DataType.F32.byteSize(), actual);
            assertEquals(expectedOffset, Indexing.linearToOffset(view, linear));
            assertEquals(expectedOffset, Indexing.coordToOffset(view, coord));
        }
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void linearToOffsetColumnMajorUsesLayout(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3);
        Layout layout = Layout.columnMajor(shape);
        long byteOffset = DataType.F32.byteSize() * 2;
        MemoryView<B> base = MemoryHelpers.arange(context, DataType.F32, 0, shape.size() + 2, 1);
        MemoryView<B> view = MemoryView.of(base.memory(), byteOffset, DataType.F32, layout);

        for (long linear = 0; linear < shape.size(); linear++) {
            long expectedOffset = expectedOffset(layout, linear, DataType.F32, view.byteOffset());
            float actual = memoryAccess.readFloat(view.memory(), expectedOffset);
            long[] coord = Indexing.linearToCoord(shape, linear);
            assertEquals(expectedOffset / DataType.F32.byteSize(), actual);
            assertEquals(expectedOffset, Indexing.linearToOffset(view, linear));
            assertEquals(expectedOffset, Indexing.coordToOffset(view, coord));
        }
    }

    static long[] reverse(long[] in) {
        long[] out = new long[in.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = in[in.length - i - 1];
        }
        return out;
    }

    @Test
    void testLinearIsColexicographicOrder() {
        Shape shape = Shape.of(2, 3);
        List<long[]> linearOrdering = IntStream.range(0, (int) shape.size())
                .mapToObj(linear -> Indexing.linearToCoord(shape, linear))
                .toList();

        Comparator<long[]> byColexicographicOrder = Arrays::compare;

        List<long[]> colexOrdering = linearOrdering.stream()
                .sorted(byColexicographicOrder)
                .toList();

        for (int i = 0; i < linearOrdering.size(); i++) {
            assertArrayEquals(colexOrdering.get(i), linearOrdering.get(i));
        }
    }

    private long expectedOffset(Layout layout, long linear, DataType dataType, long baseOffset) {
        long[] dims = layout.shape().toArray();
        long remaining = linear;
        long offset = baseOffset;
        for (int i = dims.length - 1; i >= 0; i--) {
            long dim = dims[i];
            long coord = remaining % dim;
            remaining /= dim;
            offset += coord * layout.stride().flatAt(i) * dataType.byteSize();
        }
        return offset;
    }
}
