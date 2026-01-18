package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class MemoryViewToStringTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void metadataOnlyIncludesLayoutAndMemory(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);

        String text = view.toString();
        assertTrue(text.startsWith("MemoryView{"));
        assertTrue(text.contains("layout="));
        assertTrue(text.contains("dataType=fp32"));
        assertTrue(text.contains("memory="));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void valuesIncludeElision(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(10, 10);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);

        String text = view.toString(memoryAccess);
        assertTrue(text.contains("..."));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void valuesOnlyOmitsMetadata(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 2);
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);

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
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);

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
        MemoryView<B> view = MemoryHelpers.arange(context, DataType.FP32, shape.size()).view(shape);

        String text = view.toString(memoryAccess);
        String lineBreak = System.lineSeparator();
        assertTrue(text.contains(lineBreak + lineBreak + "  ["));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void booleanScalarPrintsCorrectly(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.scalar();

        // Test true
        MemoryView<B> trueView = MemoryHelpers.full(context, DataType.BOOL, shape, 1);
        String trueText = trueView.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertTrue(trueText.contains("true"));

        // Test false
        MemoryView<B> falseView = MemoryHelpers.full(context, DataType.BOOL, shape, 0);
        String falseText = falseView.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertTrue(falseText.contains("false"));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void boolean1DArrayPrintsCorrectly(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(5);
        // Create array: [true, false, true, true, false]
        MemoryView<B> view = MemoryHelpers.zeros(context, DataType.BOOL, shape);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 0, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 1, (byte) 0);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 2, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 3, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 4, (byte) 0);

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        assertTrue(text.contains("true"));
        assertTrue(text.contains("false"));
        assertTrue(text.startsWith("["));
        assertTrue(text.endsWith("]"));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void boolean2DArrayPrintsCorrectly(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(2, 3);
        // Create 2x3 array:
        // [[true, false, true],
        //  [false, true, false]]
        MemoryView<B> view = MemoryHelpers.zeros(context, DataType.BOOL, shape);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 0, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 1, (byte) 0);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 2, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 3, (byte) 0);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 4, (byte) 1);
        memoryAccess.writeByte(view.memory(), view.byteOffset() + 5, (byte) 0);

        String text = view.toString(memoryAccess);
        assertTrue(text.contains("true"));
        assertTrue(text.contains("false"));
        // Should have nested brackets with proper formatting
        String lineBreak = System.lineSeparator();
        assertTrue(text.contains("[" + lineBreak + "  ["));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void booleanMetadataIncludesBoolType(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(3);
        MemoryView<B> view = MemoryHelpers.ones(context, DataType.BOOL, shape);

        String text = view.toString();
        assertTrue(text.startsWith("MemoryView{"));
        assertTrue(text.contains("dataType=bool"));
        assertTrue(text.contains("layout="));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void booleanAllTruesPrintsCorrectly(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(4);
        MemoryView<B> view = MemoryHelpers.ones(context, DataType.BOOL, shape);

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        // Count occurrences of "true"
        int trueCount = text.split("true", -1).length - 1;
        assertEquals(4, trueCount);
        // Should not contain "false"
        assertFalse(text.contains("false"));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void booleanAllFalsesPrintsCorrectly(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(4);
        MemoryView<B> view = MemoryHelpers.zeros(context, DataType.BOOL, shape);

        String text = view.toString(memoryAccess, ViewPrintOptions.valuesOnly());
        // Count occurrences of "false"
        int falseCount = text.split("false", -1).length - 1;
        assertEquals(4, falseCount);
        // Should not contain "true"
        assertFalse(text.contains("true"));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingBool")
    <B> void booleanLargeArrayIncludesElision(MemoryContext<B> context) {
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        Assumptions.assumeTrue(memoryAccess != null, "memory access required");

        Shape shape = Shape.of(100);
        MemoryView<B> view = MemoryHelpers.ones(context, DataType.BOOL, shape);

        String text = view.toString(memoryAccess);
        // Should have elision marker
        assertTrue(text.contains("..."));
    }
}
