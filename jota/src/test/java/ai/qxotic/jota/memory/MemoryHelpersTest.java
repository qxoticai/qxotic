package ai.qxotic.jota.memory;

import static ai.qxotic.jota.memory.MemoryHelpers.arange;
import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;

public class MemoryHelpersTest extends AbstractMemoryTest {

    // ============================================================
    // Byte (I8) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI8")
    <B> void testArangeByteThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(2, access.readByte(view.memory(), 2));
        assertEquals(3, access.readByte(view.memory(), 3));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI8")
    <B> void testArangeByteTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI8")
    <B> void testArangeByteOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    // ============================================================
    // Short (I16) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI16")
    <B> void testArangeShortThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I16, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I16, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readShort(view.memory(), 0));
        assertEquals(1, access.readShort(view.memory(), 2));
        assertEquals(9, access.readShort(view.memory(), 18));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI16")
    <B> void testArangeShortTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I16, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I16, view.dataType());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI16")
    <B> void testArangeShortOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I16, 7);
        assertEquals(7, view.shape().size());
        assertEquals(DataType.I16, view.dataType());
    }

    // ============================================================
    // Int (I32) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeIntThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 20);
        assertEquals(20, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(19, access.readInt(view.memory(), 76));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeIntTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(9, access.readInt(view.memory(), 36));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeIntOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 8);
        assertEquals(8, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(7, access.readInt(view.memory(), 28));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeIntNegativeStep(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(4, access.readInt(view.memory(), 16));
    }

    // ============================================================
    // Long (I64) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI64")
    <B> void testArangeLongThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I64, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I64, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0L, access.readLong(view.memory(), 0));
        assertEquals(1L, access.readLong(view.memory(), 8));
        assertEquals(9L, access.readLong(view.memory(), 72));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI64")
    <B> void testArangeLongTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I64, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I64, view.dataType());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI64")
    <B> void testArangeLongOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I64, 6);
        assertEquals(6, view.shape().size());
        assertEquals(DataType.I64, view.dataType());
    }

    // ============================================================
    // Float (FP32) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testArangeFloatThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP32, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(1.0f, access.readFloat(view.memory(), 4), 1e-6);
        assertEquals(9.0f, access.readFloat(view.memory(), 36), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testArangeFloatTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP32, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(3.0f, access.readFloat(view.memory(), 12), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testArangeFloatOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP32, 3);
        assertEquals(3, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(1.0f, access.readFloat(view.memory(), 4), 1e-6);
        assertEquals(2.0f, access.readFloat(view.memory(), 8), 1e-6);
    }

    // ============================================================
    // Double (FP64) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingF64")
    <B> void testArangeDoubleThreeArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP64, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(1.0, access.readDouble(view.memory(), 8), 1e-10);
        assertEquals(3.0, access.readDouble(view.memory(), 24), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF64")
    <B> void testArangeDoubleTwoArgs(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP64, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(4.0, access.readDouble(view.memory(), 32), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF64")
    <B> void testArangeDoubleOneArg(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP64, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(3.0, access.readDouble(view.memory(), 24), 1e-10);
    }

    // ============================================================
    // Explicit DataType tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeExplicitDataType(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(4, access.readInt(view.memory(), 16));
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingF32")
    <B> void testArangeExplicitDataTypeFP32(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.FP32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());
    }

    // ============================================================
    // Edge cases
    // ============================================================

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeEmptyRange(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 0);
        assertEquals(0, view.shape().size());
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeSingleElement(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 1);
        assertEquals(1, view.shape().size());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
    }

    @Test
    void testArangeUnsupportedDataType() {
        // FloatsContext has 4-byte granularity, cannot support I8 (1 byte)
        var context =
                AbstractMemoryTest.contextsSupportingF32()
                        .filter(
                                c ->
                                        c.memoryGranularity() == Float.BYTES
                                                && !c.supportsDataType(DataType.I8))
                        .findFirst()
                        .orElseThrow();

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    arange(context, DataType.I8, 10);
                });
    }

    @ParameterizedTest
    @MethodSource("contextsSupportingI32")
    <B> void testArangeNegativeStepDescending(MemoryContext<B> context) {
        MemoryView<B> view = arange(context, DataType.I32, 4);
        assertEquals(4, view.shape().size());

        MemoryAccess<B> access = context.memoryAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(2, access.readInt(view.memory(), 8));
        assertEquals(3, access.readInt(view.memory(), 12));
    }

    static final List<DataType> PRIMITIVES =
            List.of(
                    DataType.BOOL,
                    DataType.I8,
                    DataType.I16,
                    DataType.I32,
                    DataType.I64,
                    DataType.FP16,
                    DataType.BF16,
                    DataType.FP32,
                    DataType.FP64);

    @ParameterizedTest
    @MethodSource("allContexts")
    <B> void nestingPreservesLinearOrdering(MemoryContext<B> context) {

        Shape flatShape = Shape.flat(2, 3, 5, 7);
        Layout flatLayout = Layout.rowMajor(flatShape);

        for (DataType dataType : PRIMITIVES) {
            if (!context.supportsDataType(dataType)) {
                continue;
            }

            Memory<B> memory = context.memoryAllocator().allocateMemory(dataType, flatShape.size());
            MemoryView<B> flatView = MemoryView.of(memory, dataType, flatLayout);

            for (String nesting : List.of(
                    "((_,_),_,_)",
                    "(_,(_,_),_)",
                    "(_,_,(_,_))",
                    "(((_,_),_),_)",
                    "((_,(_,_)),_)",
                    "(_,(_,(_,_)))",
                    "(_,((_,_),_))",
                    "((_,_,_),_)",
                    "(_,(_,_,_))",
                    "((_,_),(_,_))"
            )) {
                Shape nestedShape = Shape.pattern(nesting, flatShape.toArray());
                Layout nestedLayout = Layout.rowMajor(nestedShape);
                MemoryView<B> nestedView = MemoryView.of(memory, dataType, nestedLayout);

                assertTrue(flatLayout.shape().isFlat());
                assertFalse(nestedLayout.shape().isFlat());

                for (int i = 0; i < flatView.shape().size(); ++i) {
                    long flatOffset = Indexing.linearToOffset(flatView, i);
                    long nestedOffset = Indexing.linearToOffset(nestedView, i);
                    assertEquals(flatOffset, nestedOffset);
                }
            }
        }
    }
}
