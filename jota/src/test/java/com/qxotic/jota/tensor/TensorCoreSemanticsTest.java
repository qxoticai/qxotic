package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.ExecutionMode;
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
    void eagerWhereSupportsTrueScalarSemantics() {
        Tensor values = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
        Tensor condition = values.greaterThan(Tensor.scalar(2.0f));
        Tensor scalarTrue = Tensor.scalar(10.0f);

        Tensor result =
                TensorOpsContext.with(
                        new EagerTensorOps(domain), () -> condition.select(scalarTrue, values));
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
    void eagerReductionsHandleKeepDimsAndMean() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor sumKeepDims =
                TensorOpsContext.with(
                        new EagerTensorOps(domain), () -> input.sum(DataType.FP32, true, 1));
        Tensor mean =
                TensorOpsContext.with(
                        new EagerTensorOps(domain),
                        () -> TensorOpsContext.require().mean(input, 1, false));

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
    void eagerReductionsNormalizeDuplicateAxes() {
        Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));

        Tensor reduced =
                TensorOpsContext.with(
                        new EagerTensorOps(domain), () -> input.sum(DataType.FP32, true, 1, -1));
        MemoryView<?> output = reduced.materialize();

        assertEquals(Shape.of(2, 1), output.shape());
        assertEquals(3.0f, readFloat(output, 0), 1e-4f);
        assertEquals(12.0f, readFloat(output, 1), 1e-4f);
    }

    @Test
    void lazyReductionsForwardAxisAndKeepDims() {
        Environment current = Environment.current();
        Environment lazyEnv =
                new Environment(
                        current.defaultDevice(),
                        current.defaultFloat(),
                        current.runtimes(),
                        ExecutionMode.LAZY);

        Environment.with(
                lazyEnv,
                () -> {
                    Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
                    Tensor reduced = input.max(true, 1);
                    MemoryView<?> output = reduced.materialize();

                    assertEquals(Shape.of(2, 1), output.shape());
                    assertEquals(2.0f, readFloat(output, 0), 1e-4f);
                    assertEquals(5.0f, readFloat(output, 1), 1e-4f);
                    return null;
                });
    }

    @Test
    void lazyMeanSupportsFloatingInputs() {
        Environment current = Environment.current();
        Environment lazyEnv =
                new Environment(
                        current.defaultDevice(),
                        current.defaultFloat(),
                        current.runtimes(),
                        ExecutionMode.LAZY);

        Environment.with(
                lazyEnv,
                () -> {
                    Tensor input = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3));
                    Tensor mean = TensorOpsContext.require().mean(input, 1, true);
                    MemoryView<?> output = mean.materialize();

                    assertEquals(DataType.FP32, output.dataType());
                    assertEquals(Shape.of(2, 1), output.shape());
                    assertEquals(1.0f, readFloat(output, 0), 1e-4f);
                    assertEquals(4.0f, readFloat(output, 1), 1e-4f);
                    return null;
                });
    }

    @Test
    void lazyMeanRejectsNonFloatingInputs() {
        Environment current = Environment.current();
        Environment lazyEnv =
                new Environment(
                        current.defaultDevice(),
                        current.defaultFloat(),
                        current.runtimes(),
                        ExecutionMode.LAZY);

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                lazyEnv,
                                () -> {
                                    Tensor input =
                                            Tensor.iota(6, DataType.I32).view(Shape.of(2, 3));
                                    return TensorOpsContext.require().mean(input, 1, false);
                                }));
    }

    @Test
    void eagerArithmeticPromotesWithStrictTypeRules() {
        Tensor left = Tensor.full(2L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1.5d, DataType.FP32, Shape.of(2));

        Tensor result = TensorOpsContext.with(new EagerTensorOps(domain), () -> left.add(right));
        MemoryView<?> output = result.materialize();

        assertEquals(DataType.FP32, output.dataType());
        assertEquals(3.5f, readFloat(output, 0), 1e-4f);
        assertEquals(3.5f, readFloat(output, 1), 1e-4f);

        Tensor i64 = Tensor.full(1L, DataType.I64, Shape.of(2));
        Tensor fp32 = Tensor.full(1.0f, DataType.FP32, Shape.of(2));
        assertThrows(
                IllegalArgumentException.class,
                () -> TensorOpsContext.with(new EagerTensorOps(domain), () -> i64.add(fp32)));
    }

    @Test
    void bitwiseRequiresSameDtype() {
        Tensor left = Tensor.full(3L, DataType.I16, Shape.of(2));
        Tensor right = Tensor.full(1L, DataType.I32, Shape.of(2));

        assertThrows(IllegalArgumentException.class, () -> left.bitwiseAnd(right));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TensorOpsContext.with(
                                new EagerTensorOps(domain), () -> left.bitwiseAnd(right)));
    }

    @Test
    void numericOpsRejectBoolOperands() {
        Tensor left = Tensor.full(1L, DataType.BOOL, Shape.of(2));
        Tensor right = Tensor.full(0L, DataType.BOOL, Shape.of(2));

        assertThrows(
                IllegalArgumentException.class,
                () -> TensorOpsContext.with(new EagerTensorOps(domain), () -> left.add(right)));

        Tensor compare = TensorOpsContext.with(new EagerTensorOps(domain), () -> left.equal(right));
        assertEquals(DataType.BOOL, compare.dataType());
    }

    @Test
    void reductionsValidateAccumulatorTypeStrictly() {
        Tensor boolInput = Tensor.full(1L, DataType.BOOL, Shape.of(2, 2));
        Tensor i32Input = Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TensorOpsContext.with(
                                new EagerTensorOps(domain), () -> boolInput.sum(DataType.BOOL, 1)));

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        TensorOpsContext.with(
                                new EagerTensorOps(domain), () -> i32Input.sum(DataType.FP32, 1)));

        Environment current = Environment.current();
        Environment lazyEnv =
                new Environment(
                        current.defaultDevice(),
                        current.defaultFloat(),
                        current.runtimes(),
                        ExecutionMode.LAZY);

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                lazyEnv,
                                () -> {
                                    Tensor lazyInput =
                                            Tensor.iota(4, DataType.I32).view(Shape.of(2, 2));
                                    return lazyInput.sum(DataType.FP32, 1);
                                }));
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = domain.directAccess();
        return access.readFloat(typedView.memory(), offset);
    }
}
