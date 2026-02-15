package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class DomainHelpersTest extends AbstractMemoryTest {

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void fullFillsWithValue(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }
        Shape shape = Shape.of(2, 3);
        MemoryView<B> view = MemoryHelpers.full(domain, DataType.FP32, shape, 2.5);
        long byteStride = DataType.FP32.byteSize();
        for (long i = 0; i < shape.size(); i++) {
            float actual =
                    memoryAccess.readFloat(view.memory(), view.byteOffset() + i * byteStride);
            assertEquals(2.5f, actual);
        }

        MemoryView<B> flat = MemoryHelpers.full(domain, DataType.FP32, 6, 3.5);
        for (long i = 0; i < flat.shape().size(); i++) {
            float actual =
                    memoryAccess.readFloat(flat.memory(), flat.byteOffset() + i * byteStride);
            assertEquals(3.5f, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void onesAndZerosFill(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }
        Shape shape = Shape.of(2, 2);
        MemoryView<B> ones = MemoryHelpers.ones(domain, DataType.FP32, shape);
        MemoryView<B> zeros = MemoryHelpers.zeros(domain, DataType.FP32, shape);
        long byteStride = DataType.FP32.byteSize();
        for (long i = 0; i < shape.size(); i++) {
            long offset = i * byteStride;
            assertEquals(1.0f, memoryAccess.readFloat(ones.memory(), ones.byteOffset() + offset));
            assertEquals(0.0f, memoryAccess.readFloat(zeros.memory(), zeros.byteOffset() + offset));
        }

        MemoryView<B> flatOnes = MemoryHelpers.ones(domain, DataType.FP32, 4);
        MemoryView<B> flatZeros = MemoryHelpers.zeros(domain, DataType.FP32, 4);
        for (long i = 0; i < flatOnes.shape().size(); i++) {
            long offset = i * byteStride;
            assertEquals(
                    1.0f,
                    memoryAccess.readFloat(flatOnes.memory(), flatOnes.byteOffset() + offset));
            assertEquals(
                    0.0f,
                    memoryAccess.readFloat(flatZeros.memory(), flatZeros.byteOffset() + offset));
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void arangeBuildsSequence(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }
        MemoryView<B> view = MemoryHelpers.arange(domain, DataType.FP32, 3);
        long byteStride = DataType.FP32.byteSize();
        for (long i = 0; i < view.shape().size(); i++) {
            float actual =
                    memoryAccess.readFloat(view.memory(), view.byteOffset() + i * byteStride);
            assertEquals((float) i, actual);
        }

        MemoryView<B> simple = MemoryHelpers.arange(domain, DataType.FP32, 10);
        for (long i = 0; i < simple.shape().size(); i++) {
            float actual =
                    memoryAccess.readFloat(simple.memory(), simple.byteOffset() + i * byteStride);
            assertEquals((float) i, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI32")
    <B> void arangeIntegralTypes(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test I32 with explicit DataType
        MemoryView<B> i32View = MemoryHelpers.arange(domain, DataType.I32, 5);
        assertEquals(DataType.I32, i32View.dataType());
        assertEquals(5, i32View.shape().size());
        for (long i = 0; i < i32View.shape().size(); i++) {
            int actual =
                    memoryAccess.readInt(
                            i32View.memory(), i32View.byteOffset() + i * DataType.I32.byteSize());
            assertEquals(i, actual);
        }

        // Test convenience int version (infers I32)
        MemoryView<B> intView = MemoryHelpers.arange(domain, DataType.I32, 5);
        assertEquals(DataType.I32, intView.dataType());
        assertEquals(5, intView.shape().size());

        // Test int end-only version
        MemoryView<B> simpleInt = MemoryHelpers.arange(domain, DataType.I32, 5);
        assertEquals(DataType.I32, simpleInt.dataType());
        assertEquals(5, simpleInt.shape().size());
        for (long i = 0; i < simpleInt.shape().size(); i++) {
            int actual =
                    memoryAccess.readInt(
                            simpleInt.memory(),
                            simpleInt.byteOffset() + i * DataType.I32.byteSize());
            assertEquals(i, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI64")
    <B> void arangeLongTypes(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test convenience long version (infers I64)
        MemoryView<B> longView = MemoryHelpers.arange(domain, DataType.I64, 5);
        assertEquals(DataType.I64, longView.dataType());
        assertEquals(5, longView.shape().size());
        for (long i = 0; i < longView.shape().size(); i++) {
            long actual =
                    memoryAccess.readLong(
                            longView.memory(), longView.byteOffset() + i * DataType.I64.byteSize());
            assertEquals(i, actual);
        }

        // Test long end-only version
        MemoryView<B> simpleLong = MemoryHelpers.arange(domain, DataType.I64, 5);
        assertEquals(DataType.I64, simpleLong.dataType());
        assertEquals(5, simpleLong.shape().size());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF32")
    <B> void arangeFloatTypes(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test convenience float version (infers FP32)
        MemoryView<B> floatView = MemoryHelpers.arange(domain, DataType.FP32, 4);
        assertEquals(DataType.FP32, floatView.dataType());
        assertEquals(4, floatView.shape().size());
        for (long i = 0; i < floatView.shape().size(); i++) {
            float actual =
                    memoryAccess.readFloat(
                            floatView.memory(),
                            floatView.byteOffset() + i * DataType.FP32.byteSize());
            assertEquals((float) i, actual, 1e-6f);
        }

        // Test float end-only version
        MemoryView<B> simpleFloat = MemoryHelpers.arange(domain, DataType.FP32, 5);
        assertEquals(DataType.FP32, simpleFloat.dataType());
        assertEquals(5, simpleFloat.shape().size());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingF64")
    <B> void arangeDoubleTypes(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test convenience double version (infers FP64)
        MemoryView<B> doubleView = MemoryHelpers.arange(domain, DataType.FP64, 4);
        assertEquals(DataType.FP64, doubleView.dataType());
        assertEquals(4, doubleView.shape().size());
        for (long i = 0; i < doubleView.shape().size(); i++) {
            double actual =
                    memoryAccess.readDouble(
                            doubleView.memory(),
                            doubleView.byteOffset() + i * DataType.FP64.byteSize());
            assertEquals((double) i, actual, 1e-12);
        }

        // Test double end-only version
        MemoryView<B> simpleDouble = MemoryHelpers.arange(domain, DataType.FP64, 5);
        assertEquals(DataType.FP64, simpleDouble.dataType());
        assertEquals(5, simpleDouble.shape().size());
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI16")
    <B> void arangeI16(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test I16 with explicit DataType
        MemoryView<B> i16View = MemoryHelpers.arange(domain, DataType.I16, 10);
        assertEquals(DataType.I16, i16View.dataType());
        assertEquals(10, i16View.shape().size());
        for (long i = 0; i < i16View.shape().size(); i++) {
            short actual =
                    memoryAccess.readShort(
                            i16View.memory(), i16View.byteOffset() + i * DataType.I16.byteSize());
            assertEquals(i, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingI8")
    <B> void arangeI8(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test I8 with explicit DataType
        MemoryView<B> i8View = MemoryHelpers.arange(domain, DataType.I8, 10);
        assertEquals(DataType.I8, i8View.dataType());
        assertEquals(10, i8View.shape().size());
        for (long i = 0; i < i8View.shape().size(); i++) {
            byte actual =
                    memoryAccess.readByte(
                            i8View.memory(), i8View.byteOffset() + i * DataType.I8.byteSize());
            assertEquals(i, actual);
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolTrueFalse(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test true
        MemoryView<B> trueView = MemoryHelpers.full(domain, 5, true);
        assertEquals(DataType.BOOL, trueView.dataType());
        for (long i = 0; i < trueView.shape().size(); i++) {
            byte value = memoryAccess.readByte(trueView.memory(), trueView.byteOffset() + i);
            assertEquals((byte) 1, value, "true should be stored as 1");
        }

        // Test false
        MemoryView<B> falseView = MemoryHelpers.full(domain, 5, false);
        for (long i = 0; i < falseView.shape().size(); i++) {
            byte value = memoryAccess.readByte(falseView.memory(), falseView.byteOffset() + i);
            assertEquals((byte) 0, value, "false should be stored as 0");
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolZeros(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        Shape shape = Shape.of(2, 3);

        // Test +0.0
        MemoryView<B> posZero = MemoryHelpers.full(domain, DataType.BOOL, shape, +0.0);
        for (long i = 0; i < shape.size(); i++) {
            byte value = memoryAccess.readByte(posZero.memory(), posZero.byteOffset() + i);
            assertEquals((byte) 0, value, "+0.0 should be false (0)");
        }

        // Test -0.0
        MemoryView<B> negZero = MemoryHelpers.full(domain, DataType.BOOL, shape, -0.0);
        for (long i = 0; i < shape.size(); i++) {
            byte value = memoryAccess.readByte(negZero.memory(), negZero.byteOffset() + i);
            assertEquals((byte) 0, value, "-0.0 should be false (0)");
        }

        // Test integer 0
        MemoryView<B> intZero = MemoryHelpers.full(domain, DataType.BOOL, shape, 0);
        for (long i = 0; i < shape.size(); i++) {
            byte value = memoryAccess.readByte(intZero.memory(), intZero.byteOffset() + i);
            assertEquals((byte) 0, value, "0 should be false (0)");
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolNaN(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test NaN - should be true (1) per NumPy convention
        MemoryView<B> nanView = MemoryHelpers.full(domain, DataType.BOOL, 3, Double.NaN);
        for (long i = 0; i < nanView.shape().size(); i++) {
            byte value = memoryAccess.readByte(nanView.memory(), nanView.byteOffset() + i);
            assertEquals((byte) 1, value, "NaN should be true (1)");
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolSmallDoubles(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test various small non-zero doubles
        double[] smallValues = {
            Double.MIN_VALUE, // Smallest positive double
            Double.MIN_NORMAL, // Smallest normal positive double
            -Double.MIN_VALUE, // Smallest negative double
            1e-308, // Very small positive
            -1e-308, // Very small negative
            0.0001, // Small but not tiny
            -0.0001 // Small negative
        };

        for (double value : smallValues) {
            MemoryView<B> view = MemoryHelpers.full(domain, DataType.BOOL, 1, value);
            byte byteValue = memoryAccess.readByte(view.memory(), view.byteOffset());
            assertEquals(
                    (byte) 1, byteValue, "Small non-zero double " + value + " should be true (1)");
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolLargeValues(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test large integer values
        Number[] largeValues = {
            Long.MAX_VALUE,
            Long.MIN_VALUE,
            Integer.MAX_VALUE,
            Integer.MIN_VALUE,
            Double.MAX_VALUE,
            -Double.MAX_VALUE,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY
        };

        for (Number value : largeValues) {
            MemoryView<B> view = MemoryHelpers.full(domain, DataType.BOOL, 1, value);
            byte byteValue = memoryAccess.readByte(view.memory(), view.byteOffset());
            assertEquals((byte) 1, byteValue, "Large value " + value + " should be true (1)");
        }
    }

    @ParameterizedTest
    @MethodSource("domainsSupportingBool")
    <B> void boolIntegerValues(MemoryDomain<B> domain) {
        MemoryAccess<B> memoryAccess = domain.directAccess();
        if (memoryAccess == null) {
            return;
        }

        // Test 1
        MemoryView<B> oneView = MemoryHelpers.full(domain, DataType.BOOL, 3, 1);
        for (long i = 0; i < oneView.shape().size(); i++) {
            byte value = memoryAccess.readByte(oneView.memory(), oneView.byteOffset() + i);
            assertEquals((byte) 1, value, "1 should be true (1)");
        }

        // Test -1
        MemoryView<B> negOneView = MemoryHelpers.full(domain, DataType.BOOL, 3, -1);
        for (long i = 0; i < negOneView.shape().size(); i++) {
            byte value = memoryAccess.readByte(negOneView.memory(), negOneView.byteOffset() + i);
            assertEquals((byte) 1, value, "-1 should be true (1)");
        }

        // Test 42
        MemoryView<B> fortyTwoView = MemoryHelpers.full(domain, DataType.BOOL, 3, 42);
        for (long i = 0; i < fortyTwoView.shape().size(); i++) {
            byte value =
                    memoryAccess.readByte(fortyTwoView.memory(), fortyTwoView.byteOffset() + i);
            assertEquals((byte) 1, value, "42 should be true (1)");
        }
    }
}
