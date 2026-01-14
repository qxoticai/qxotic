package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryViewToStringTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void metadataOnlyIncludesLayoutAndMemory(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, shape.size())
                .view(shape);

        String text = view.toString();
        assertTrue(text.startsWith("MemoryView{"));
        assertTrue(text.contains("layout="));
        assertTrue(text.contains("dataType=f32"));
        assertTrue(text.contains("memory="));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void valuesIncludeElision(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(10, 10);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, shape.size())
                .view(shape);

        String text = view.toString(memoryAccess);
        assertTrue(text.contains("..."));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void valuesOnlyOmitsMetadata(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 2);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, shape.size())
                .view(shape);

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertFalse(text.contains("MemoryView{"));
        assertFalse(text.contains("layout="));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void scalarValuesPrintInline(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.scalar();
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, shape.size())
                .view(shape);

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertTrue(text.startsWith("["));
        assertTrue(text.endsWith("]"));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void threeDimensionalAddsBlankLineBetweenBlocks(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 2, 2);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.F32, shape.size())
                .view(shape);

        String text = view.toString(memoryAccess);
        String lineBreak = System.lineSeparator();
        assertTrue(text.contains(lineBreak + lineBreak + "  ["));
    }
}
