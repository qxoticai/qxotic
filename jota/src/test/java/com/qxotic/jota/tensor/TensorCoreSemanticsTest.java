package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class TensorCoreSemanticsTest {

    private static MemoryDomain<MemorySegment> domain;

    @BeforeAll
    static void setUpDomain() {
        domain = DomainFactory.ofMemorySegment();
    }

    @Test
    void lazyWhereSupportsTrueScalarSemantics() {
        Tensor values = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor condition = values.greaterThan(Tensor.scalar(2.0f));
        Tensor scalarTrue = Tensor.scalar(10.0f);

        Tensor result = condition.where(scalarTrue, values);
        MemoryView<?> output = result.materialize();

        assertEquals(Shape.of(2, 3), output.shape());
        assertEquals(0.0f, readFloat(output, 0), 1e-4f);
        assertEquals(1.0f, readFloat(output, 1), 1e-4f);
        assertEquals(2.0f, readFloat(output, 2), 1e-4f);
        assertEquals(10.0f, readFloat(output, 3), 1e-4f);
        assertEquals(10.0f, readFloat(output, 4), 1e-4f);
        assertEquals(10.0f, readFloat(output, 5), 1e-4f);
    }

    @Test
    void lazyReductionsHandleKeepDimsAndMean() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor sumKeepDims = input.sum(DataType.FP32, true, 1);
        Tensor mean = input.mean(1);

        MemoryView<?> sumView = sumKeepDims.materialize();
        MemoryView<?> meanView = mean.materialize();

        assertEquals(Shape.of(2, 1), sumView.shape());
        assertEquals(3.0f, readFloat(sumView, 0), 1e-4f);
        assertEquals(12.0f, readFloat(sumView, 1), 1e-4f);

        assertEquals(Shape.of(2), meanView.shape());
        assertEquals(1.0f, readFloat(meanView, 0), 1e-4f);
        assertEquals(4.0f, readFloat(meanView, 1), 1e-4f);
    }

    @Test
    void lazyReductionsNormalizeDuplicateAxes() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor reduced = input.sum(DataType.FP32, true, 1, -1);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(3.0f, readFloat(output, 0), 1e-4f);
        assertEquals(12.0f, readFloat(output, 1), 1e-4f);
    }

    @Test
    void lazyReductionsForwardAxisAndKeepDims() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor reduced = input.max(true, 1);
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(2.0f, readFloat(output, 0), 1e-4f);
        assertEquals(5.0f, readFloat(output, 1), 1e-4f);
    }

    @Test
    void lazyMeanSupportsFloatingInputs() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor mean = input.mean(true, 1);
        MemoryView<?> output = mean.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(1.0f, readFloat(output, 0), 1e-4f);
        assertEquals(4.0f, readFloat(output, 1), 1e-4f);
    }

    @Test
    void lazyMeanReducesAllAxesToScalar() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor mean = input.mean();
        MemoryView<?> output = mean.materialize();

        assertEquals(Shape.scalar(), output.shape());
        assertEquals(2.5f, readFloat(output, 0), 1e-4f);
    }

    @Test
    void lazyMeanSupportsMultiAxisWithWrapAround() {
        Tensor input = Tensor.iota(8, DataType.FP32).view(Shape.of(2, 2, 2));
        Tensor mean = input.mean(false, 1, -1);
        MemoryView<?> output = mean.materialize();

        assertEquals(Shape.of(2), output.shape());
        assertEquals(1.5f, readFloat(output, 0), 1e-4f);
        assertEquals(5.5f, readFloat(output, 1), 1e-4f);
    }

    @Test
    void lazyMeanRejectsNonFloatingInputs() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor input = Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));
                    input.mean(1).materialize();
                });
    }

    @Test
    void anyAndAllReduceBoolGlobally() {
        Tensor condition =
                Tensor.iota(4, DataType.I32)
                        .view(Shape.of(2, 2))
                        .lessThan(Tensor.full(3L, DataType.I32, Shape.scalar()));

        Tensor any = condition.any().cast(DataType.I32);
        Tensor all = condition.all().cast(DataType.I32);
        MemoryView<?> anyView = any.materialize();
        MemoryView<?> allView = all.materialize();

        assertEquals(Shape.scalar(), anyView.shape());
        assertEquals(Shape.scalar(), allView.shape());
        assertEquals(1, readInt(anyView, 0));
        assertEquals(0, readInt(allView, 0));
    }

    @Test
    void scalarOverloadArithmeticProducesExpectedValues() {
        Tensor intBase = Tensor.iota(3, DataType.I32);
        MemoryView<?> addInt = intBase.add(2).materialize();
        MemoryView<?> addLong = intBase.add(2L).materialize();
        assertEquals(2, readInt(addInt, 0));
        assertEquals(3, readInt(addInt, 1));
        assertEquals(2, readInt(addLong, 0));

        MemoryView<?> sub = intBase.subtract(1).materialize();
        MemoryView<?> mul = intBase.multiply(2).materialize();
        assertEquals(-1, readInt(sub, 0));
        assertEquals(0, readInt(sub, 1));
        assertEquals(4, readInt(mul, 2));

        Tensor fpBase = Tensor.iota(3, DataType.FP32);
        MemoryView<?> addFloat = fpBase.add(2.0f).materialize();
        assertEquals(2.0f, readFloat(addFloat, 0), 1e-4f);
        MemoryView<?> div = fpBase.divide(2.0f).materialize();
        assertEquals(0.5f, readFloat(div, 1), 1e-4f);

        Tensor fp64Base = Tensor.iota(3, DataType.FP64);
        MemoryView<?> addDouble = fp64Base.add(2.0d).materialize();
        assertEquals(2.0d, readDouble(addDouble, 0), 1e-8);

        assertThrows(IllegalArgumentException.class, () -> fpBase.add(2).materialize());
    }

    @Test
    void whereRejectsMismatchedBranchTypes() {
        Tensor condition = Tensor.iota(3, DataType.I32).lessThan(Tensor.scalar(2L));
        Tensor fp = Tensor.iota(3, DataType.FP32);
        Tensor ints = Tensor.iota(3, DataType.I32);
        assertThrows(IllegalArgumentException.class, () -> condition.where(fp, ints));
    }

    @Test
    void realizeAliasReturnsSameTensorAndMaterializes() {
        Tensor tensor = Tensor.iota(4, DataType.FP32);
        Tensor realized = tensor.realize();
        assertEquals(tensor, realized);
        assertTrue(tensor.isMaterialized());
    }

    @Test
    void lazyArithmeticPromotesWithStrictTypeRules() {
        Tensor left = Tensor.full(2L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1.5d, DataType.FP32, Shape.of(2));

        Tensor result = left.add(right);
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(3.5f, readFloat(output, 0), 1e-4f);
        assertEquals(3.5f, readFloat(output, 1), 1e-4f);

        Tensor i64 = Tensor.full(1L, DataType.I64, Shape.of(2));
        Tensor fp32 = Tensor.full(1.0f, DataType.FP32, Shape.of(2));
        assertThrows(IllegalArgumentException.class, () -> i64.add(fp32).materialize());
    }

    @Test
    void bitwiseRequiresSameDtype() {
        Tensor left = Tensor.full(3L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1L, DataType.I32, Shape.of(2));

        assertThrows(IllegalArgumentException.class, () -> left.bitwiseAnd(right));
    }

    @Test
    void numericOpsRejectBoolOperands() {
        Tensor left = Tensor.full(1L, DataType.BOOL, Shape.of(2));
        Tensor right = Tensor.full(0L, DataType.BOOL, Shape.of(2));

        assertThrows(IllegalArgumentException.class, () -> left.add(right).materialize());

        Tensor compare = left.equal(right);
        assertEquals(DataType.BOOL, compare.dataType());
    }

    @Test
    void reductionsValidateAccumulatorTypeStrictly() {
        Tensor boolInput = Tensor.full(1L, DataType.BOOL, Shape.of(2, 2));
        Tensor i32Input = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));

        assertThrows(
                IllegalArgumentException.class,
                () -> boolInput.sum(DataType.BOOL, 1).materialize());

        assertThrows(
                IllegalArgumentException.class, () -> i32Input.sum(DataType.FP32, 1).materialize());

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Tensor lazyInput = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));
                    lazyInput.sum(DataType.FP32, 1).materialize();
                });
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readFloat(typedView.memory(), offset);
    }

    private static int readInt(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readInt(typedView.memory(), offset);
    }

    private static double readDouble(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readDouble(typedView.memory(), offset);
    }
}
