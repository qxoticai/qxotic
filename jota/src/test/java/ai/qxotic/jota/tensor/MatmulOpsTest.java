package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class MatmulOpsTest {

    @Test
    void matmulWorksInLazyMode() {
        MemoryView<MemorySegment> output = runMatmul(ExecutionMode.LAZY);
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
    }

    @Test
    void matmulWorksInEagerMode() {
        MemoryView<MemorySegment> output = runMatmul(ExecutionMode.EAGER);
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
    }

    @Test
    void batchedMatmulWorksInLazyMode() {
        MemoryView<MemorySegment> output = runBatchedMatmul(ExecutionMode.LAZY);
        assertEquals(Shape.of(2, 2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
        assertClose(readFloat(output, 4), 220f);
        assertClose(readFloat(output, 5), 244f);
        assertClose(readFloat(output, 6), 301f);
        assertClose(readFloat(output, 7), 334f);
    }

    @Test
    void batchedMatmulWorksInEagerMode() {
        MemoryView<MemorySegment> output = runBatchedMatmul(ExecutionMode.EAGER);
        assertEquals(Shape.of(2, 2, 2), output.shape());
        assertClose(readFloat(output, 0), 22f);
        assertClose(readFloat(output, 1), 28f);
        assertClose(readFloat(output, 2), 49f);
        assertClose(readFloat(output, 3), 64f);
        assertClose(readFloat(output, 4), 220f);
        assertClose(readFloat(output, 5), 244f);
        assertClose(readFloat(output, 6), 301f);
        assertClose(readFloat(output, 7), 334f);
    }

    @Test
    void matmulHandlesNonContiguousInputInLazyMode() {
        MemoryView<MemorySegment> output = runMatmulNonContiguousLeft(ExecutionMode.LAZY);
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 35f);
        assertClose(readFloat(output, 1), 44f);
        assertClose(readFloat(output, 2), 44f);
        assertClose(readFloat(output, 3), 56f);
    }

    @Test
    void matmulHandlesNonContiguousInputInEagerMode() {
        MemoryView<MemorySegment> output = runMatmulNonContiguousLeft(ExecutionMode.EAGER);
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 35f);
        assertClose(readFloat(output, 1), 44f);
        assertClose(readFloat(output, 2), 44f);
        assertClose(readFloat(output, 3), 56f);
    }

    @Test
    void tracedMatmulHandlesNonContiguousInputInLazyMode() {
        MemoryView<MemorySegment> output = runTracedMatmulNonContiguousLeft(ExecutionMode.LAZY);
        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFloat(output, 0), 35f);
        assertClose(readFloat(output, 1), 44f);
        assertClose(readFloat(output, 2), 44f);
        assertClose(readFloat(output, 3), 56f);
    }

    @Test
    void vectorTimesMatrixIsRejected() {
        assertMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return vector.matmul(matrix);
                });
        assertMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return vector.matmul(matrix);
                });
    }

    @Test
    void matrixTimesVectorIsRejected() {
        assertMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    return matrix.matmul(vector);
                });
        assertMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                    return matrix.matmul(vector);
                });
    }

    @Test
    void scalarMatmulIsRejected() {
        assertMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor scalar = Tensor.scalar(2f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return scalar.matmul(matrix);
                });
        assertMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor scalar = Tensor.scalar(2f);
                    Tensor matrix = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return scalar.matmul(matrix);
                });
    }

    @Test
    void incompatibleInnerDimensionsAreRejected() {
        assertMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(4, 2));
                    return a.matmul(b);
                });
        assertMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(8, DataType.FP32).add(1f).view(Shape.of(4, 2));
                    return a.matmul(b);
                });
    }

    @Test
    void incompatibleDtypesAreRejected() {
        assertMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.I64).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(3, 2));
                    return a.matmul(b);
                });
        assertMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.I64).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(3, 2));
                    return a.matmul(b);
                });
    }

    @Test
    void batchedMatmulIncompatibleBatchIsRejected() {
        assertBatchedMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(18, DataType.FP32).view(Shape.of(3, 3, 2));
                    return a.batchedMatmul(b);
                });
        assertBatchedMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(18, DataType.FP32).view(Shape.of(3, 3, 2));
                    return a.batchedMatmul(b);
                });
    }

    @Test
    void batchedMatmulIncompatibleDtypesAreRejected() {
        assertBatchedMatmulFails(
                ExecutionMode.LAZY,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.I64).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 3, 2));
                    return a.batchedMatmul(b);
                });
        assertBatchedMatmulFails(
                ExecutionMode.EAGER,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.I64).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(12, DataType.FP32).view(Shape.of(2, 3, 2));
                    return a.batchedMatmul(b);
                });
    }

    @SuppressWarnings("unchecked")
    private MemoryView<MemorySegment> runMatmul(ExecutionMode mode) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        return Environment.with(
                env,
                () -> {
                    Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                    Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return (MemoryView<MemorySegment>) a.matmul(b).materialize();
                });
    }

    @SuppressWarnings("unchecked")
    private MemoryView<MemorySegment> runBatchedMatmul(ExecutionMode mode) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        return Environment.with(
                env,
                () -> {
                    Tensor a = Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 2, 3));
                    Tensor b = Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 3, 2));
                    return (MemoryView<MemorySegment>) a.batchedMatmul(b).materialize();
                });
    }

    @SuppressWarnings("unchecked")
    private MemoryView<MemorySegment> runMatmulNonContiguousLeft(ExecutionMode mode) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        return Environment.with(
                env,
                () -> {
                    Tensor a =
                            Tensor.iota(6, DataType.FP32)
                                    .add(1f)
                                    .view(Shape.of(3, 2))
                                    .transpose(-2, -1);
                    Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return (MemoryView<MemorySegment>) a.matmul(b).materialize();
                });
    }

    @SuppressWarnings("unchecked")
    private MemoryView<MemorySegment> runTracedMatmulNonContiguousLeft(ExecutionMode mode) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        return Environment.with(
                env,
                () -> {
                    Tensor a =
                            Tensor.iota(6, DataType.FP32)
                                    .add(1f)
                                    .view(Shape.of(3, 2))
                                    .transpose(-2, -1);
                    Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                    return (MemoryView<MemorySegment>) Tracer.trace(a, b, Tensor::matmul).materialize();
                });
    }

    private static float readFloat(MemoryView<MemorySegment> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryAccess<MemorySegment> access =
                (MemoryAccess<MemorySegment>)
                        Environment.current().nativeRuntime().memoryDomain().directAccess();
        long offset = Indexing.linearToOffset(view, linearIndex);
        return access.readFloat(view.memory(), offset);
    }

    private void assertMatmulFails(ExecutionMode mode, java.util.function.Supplier<Tensor> op) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        assertThrows(IllegalArgumentException.class, () -> Environment.with(env, op::get));
    }

    private void assertBatchedMatmulFails(
            ExecutionMode mode, java.util.function.Supplier<Tensor> op) {
        Environment env =
                new Environment(
                        Device.PANAMA, DataType.FP32, Environment.current().runtimes(), mode);
        assertThrows(IllegalArgumentException.class, () -> Environment.with(env, op::get));
    }

    private static void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-4f);
    }
}
