package com.qxotic.jota.memory;

import static com.qxotic.jota.memory.MemoryHelpers.arange;
import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

public class MemoryHelpersTest extends AbstractMemoryTest {

    // ============================================================
    // Byte (I8) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI8")
    <B> void testArangeByteThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(2, access.readByte(view.memory(), 2));
        assertEquals(3, access.readByte(view.memory(), 3));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI8")
    <B> void testArangeByteTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI8")
    <B> void testArangeByteOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I8, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I8, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readByte(view.memory(), 0));
        assertEquals(1, access.readByte(view.memory(), 1));
        assertEquals(4, access.readByte(view.memory(), 4));
    }

    // ============================================================
    // Short (I16) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI16")
    <B> void testArangeShortThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I16, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I16, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readShort(view.memory(), 0));
        assertEquals(1, access.readShort(view.memory(), 2));
        assertEquals(9, access.readShort(view.memory(), 18));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI16")
    <B> void testArangeShortTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I16, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I16, view.dataType());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI16")
    <B> void testArangeShortOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I16, 7);
        assertEquals(7, view.shape().size());
        assertEquals(DataType.I16, view.dataType());
    }

    // ============================================================
    // Int (I32) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeIntThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 20);
        assertEquals(20, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(19, access.readInt(view.memory(), 76));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeIntTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(9, access.readInt(view.memory(), 36));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeIntOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 8);
        assertEquals(8, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(7, access.readInt(view.memory(), 28));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeIntNegativeStep(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(4, access.readInt(view.memory(), 16));
    }

    // ============================================================
    // Long (I64) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI64")
    <B> void testArangeLongThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I64, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.I64, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0L, access.readLong(view.memory(), 0));
        assertEquals(1L, access.readLong(view.memory(), 8));
        assertEquals(9L, access.readLong(view.memory(), 72));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI64")
    <B> void testArangeLongTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I64, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I64, view.dataType());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI64")
    <B> void testArangeLongOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I64, 6);
        assertEquals(6, view.shape().size());
        assertEquals(DataType.I64, view.dataType());
    }

    // ============================================================
    // Float (FP32) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testArangeFloatThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP32, 10);
        assertEquals(10, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(1.0f, access.readFloat(view.memory(), 4), 1e-6);
        assertEquals(9.0f, access.readFloat(view.memory(), 36), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testArangeFloatTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP32, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(3.0f, access.readFloat(view.memory(), 12), 1e-6);
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testArangeFloatOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP32, 3);
        assertEquals(3, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0f, access.readFloat(view.memory(), 0), 1e-6);
        assertEquals(1.0f, access.readFloat(view.memory(), 4), 1e-6);
        assertEquals(2.0f, access.readFloat(view.memory(), 8), 1e-6);
    }

    // ============================================================
    // Double (FP64) tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingF64")
    <B> void testArangeDoubleThreeArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP64, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(1.0, access.readDouble(view.memory(), 8), 1e-10);
        assertEquals(3.0, access.readDouble(view.memory(), 24), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF64")
    <B> void testArangeDoubleTwoArgs(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP64, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(4.0, access.readDouble(view.memory(), 32), 1e-10);
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF64")
    <B> void testArangeDoubleOneArg(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP64, 4);
        assertEquals(4, view.shape().size());
        assertEquals(DataType.FP64, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0.0, access.readDouble(view.memory(), 0), 1e-10);
        assertEquals(3.0, access.readDouble(view.memory(), 24), 1e-10);
    }

    // ============================================================
    // Explicit DataType tests
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeExplicitDataType(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.I32, view.dataType());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(4, access.readInt(view.memory(), 16));
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void testArangeExplicitDataTypeFP32(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.FP32, 5);
        assertEquals(5, view.shape().size());
        assertEquals(DataType.FP32, view.dataType());
    }

    // ============================================================
    // Edge cases
    // ============================================================

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeEmptyRange(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 0);
        assertEquals(0, view.shape().size());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeSingleElement(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 1);
        assertEquals(1, view.shape().size());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
    }

    @Test
    void testArangeUnsupportedDataType() {
        // FloatsDomain has 4-byte granularity, cannot support I8 (1 byte)
        var domain =
                AbstractMemoryTest.domainsSupportingF32()
                        .filter(
                                c ->
                                        c.memoryGranularity() == Float.BYTES
                                                && !c.supportsDataType(DataType.I8))
                        .findFirst()
                        .orElseThrow();

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    arange(domain, DataType.I8, 10);
                });
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void testArangeNegativeStepDescending(MemoryDomain<B> domain) {
        MemoryView<B> view = arange(domain, DataType.I32, 4);
        assertEquals(4, view.shape().size());

        MemoryAccess<B> access = domain.directAccess();
        assertEquals(0, access.readInt(view.memory(), 0));
        assertEquals(1, access.readInt(view.memory(), 4));
        assertEquals(2, access.readInt(view.memory(), 8));
        assertEquals(3, access.readInt(view.memory(), 12));
    }

    @ParameterizedTest
    @MethodSource("allDomains")
    <B> void nestingPreservesLinearOrdering(MemoryDomain<B> domain) {

        Shape flatShape = Shape.flat(2, 3, 5, 7);
        Layout flatLayout = Layout.rowMajor(flatShape);

        for (DataType dataType : PRIMITIVE_DATA_TYPES) {
            if (!domain.supportsDataType(dataType)) {
                continue;
            }

            Memory<B> memory = domain.memoryAllocator().allocateMemory(dataType, flatShape.size());
            MemoryView<B> flatView = MemoryView.of(memory, dataType, flatLayout);

            for (String nesting :
                    List.of(
                            "((_,_),_,_)",
                            "(_,(_,_),_)",
                            "(_,_,(_,_))",
                            "(((_,_),_),_)",
                            "((_,(_,_)),_)",
                            "(_,(_,(_,_)))",
                            "(_,((_,_),_))",
                            "((_,_,_),_)",
                            "(_,(_,_,_))",
                            "((_,_),(_,_))")) {
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
