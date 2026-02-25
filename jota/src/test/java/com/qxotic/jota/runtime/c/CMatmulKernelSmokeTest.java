package com.qxotic.jota.runtime.c;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.testutil.ExternalToolChecks;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class CMatmulKernelSmokeTest {

    @Test
    void runsTracedMatmulKernel() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor a = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(2, 3));
                            Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                            return Tracer.trace(a, b, Tensor::matmul).materialize();
                        });

        assertEquals(Shape.of(2, 2), output.shape());
        assertClose(readFp32(output, 0), 22f);
        assertClose(readFp32(output, 1), 28f);
        assertClose(readFp32(output, 2), 49f);
        assertClose(readFp32(output, 3), 64f);
    }

    @Test
    void runsTracedBatchedMatmulKernel() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor a =
                                    Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 2, 3));
                            Tensor b =
                                    Tensor.iota(12, DataType.FP32).add(1f).view(Shape.of(2, 3, 2));
                            return Tracer.trace(a, b, Tensor::batchedMatmul).materialize();
                        });

        assertEquals(Shape.of(2, 2, 2), output.shape());
        assertClose(readFp32(output, 0), 22f);
        assertClose(readFp32(output, 1), 28f);
        assertClose(readFp32(output, 2), 49f);
        assertClose(readFp32(output, 3), 64f);
        assertClose(readFp32(output, 4), 220f);
        assertClose(readFp32(output, 5), 244f);
        assertClose(readFp32(output, 6), 301f);
        assertClose(readFp32(output, 7), 334f);
    }

    @Test
    void materializesTransposedInputCorrectly() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();

        MemoryView<?> transposed =
                Environment.with(
                        cEnv,
                        () ->
                                Tensor.iota(6, DataType.FP32)
                                        .add(1f)
                                        .view(Shape.of(3, 2))
                                        .transpose(-2, -1)
                                        .materialize());

        assertEquals(Shape.of(2, 3), transposed.shape());
        assertClose(readFp32(transposed, 0), 1f);
        assertClose(readFp32(transposed, 1), 3f);
        assertClose(readFp32(transposed, 2), 5f);
        assertClose(readFp32(transposed, 3), 2f);
        assertClose(readFp32(transposed, 4), 4f);
        assertClose(readFp32(transposed, 5), 6f);
    }

    @Test
    void materializesRightHandMatmulInputCorrectly() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();

        MemoryView<?> rhs =
                Environment.with(
                        cEnv,
                        () ->
                                Tensor.iota(6, DataType.FP32)
                                        .add(1f)
                                        .view(Shape.of(3, 2))
                                        .materialize());

        assertEquals(Shape.of(3, 2), rhs.shape());
        assertClose(readFp32(rhs, 0), 1f);
        assertClose(readFp32(rhs, 1), 2f);
        assertClose(readFp32(rhs, 2), 3f);
        assertClose(readFp32(rhs, 3), 4f);
        assertClose(readFp32(rhs, 4), 5f);
        assertClose(readFp32(rhs, 5), 6f);
    }

    @Test
    void matmulHandlesNonContiguousInput() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();

        MemoryView<?> output =
                Environment.with(
                        cEnv,
                        () -> {
                            Tensor a =
                                    Tensor.iota(6, DataType.FP32)
                                            .add(1f)
                                            .view(Shape.of(3, 2))
                                            .transpose(-2, -1);
                            Tensor b = Tensor.iota(6, DataType.FP32).add(1f).view(Shape.of(3, 2));
                            return Tracer.trace(a, b, Tensor::matmul).materialize();
                        });

        assertEquals(Shape.of(2, 2), output.shape());
        System.out.println(
                "C noncontig matmul out="
                        + readFp32(output, 0)
                        + ","
                        + readFp32(output, 1)
                        + ","
                        + readFp32(output, 2)
                        + ","
                        + readFp32(output, 3));
        assertClose(readFp32(output, 0), 35f);
        assertClose(readFp32(output, 1), 44f);
        assertClose(readFp32(output, 2), 44f);
        assertClose(readFp32(output, 3), 56f);
    }

    @Test
    void rejectsVectorTimesMatrix() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                                    Tensor matrix =
                                            Tensor.iota(6, DataType.FP32)
                                                    .add(1f)
                                                    .view(Shape.of(3, 2));
                                    return vector.matmul(matrix).materialize();
                                }));
    }

    @Test
    void rejectsMatrixTimesVector() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor matrix =
                                            Tensor.iota(6, DataType.FP32)
                                                    .add(1f)
                                                    .view(Shape.of(2, 3));
                                    Tensor vector = Tensor.iota(3, DataType.FP32).add(1f);
                                    return matrix.matmul(vector).materialize();
                                }));
    }

    @Test
    void rejectsScalarMatmul() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor scalar = Tensor.scalar(2f);
                                    Tensor matrix =
                                            Tensor.iota(6, DataType.FP32)
                                                    .add(1f)
                                                    .view(Shape.of(3, 2));
                                    return scalar.matmul(matrix).materialize();
                                }));
    }

    @Test
    void rejectsInnerDimensionMismatch() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor a =
                                            Tensor.iota(6, DataType.FP32)
                                                    .add(1f)
                                                    .view(Shape.of(2, 3));
                                    Tensor b =
                                            Tensor.iota(8, DataType.FP32)
                                                    .add(1f)
                                                    .view(Shape.of(4, 2));
                                    return a.matmul(b).materialize();
                                }));
    }

    @Test
    void rejectsIncompatibleDtypes() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor a = Tensor.iota(6, DataType.I64).view(Shape.of(2, 3));
                                    Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(3, 2));
                                    return a.matmul(b).materialize();
                                }));
    }

    @Test
    void rejectsBatchedMatmulBatchMismatch() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor a =
                                            Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
                                    Tensor b =
                                            Tensor.iota(18, DataType.FP32).view(Shape.of(3, 3, 2));
                                    return a.batchedMatmul(b).materialize();
                                }));
    }

    @Test
    void rejectsBatchedMatmulInnerMismatch() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor a =
                                            Tensor.iota(12, DataType.FP32).view(Shape.of(2, 2, 3));
                                    Tensor b =
                                            Tensor.iota(16, DataType.FP32).view(Shape.of(2, 4, 2));
                                    return a.batchedMatmul(b).materialize();
                                }));
    }

    @Test
    void rejectsBatchedMatmulIncompatibleDtypes() {
        assumeCBackendAvailable();
        Environment cEnv = cEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                cEnv,
                                () -> {
                                    Tensor a =
                                            Tensor.iota(12, DataType.I64).view(Shape.of(2, 2, 3));
                                    Tensor b =
                                            Tensor.iota(12, DataType.FP32).view(Shape.of(2, 3, 2));
                                    return a.batchedMatmul(b).materialize();
                                }));
    }

    private static Environment cEnvironment() {
        Environment current = Environment.current();
        return new Environment(Device.C, current.defaultFloat(), current.runtimes());
    }

    @SuppressWarnings("unchecked")
    private static float readFp32(MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> typed = (MemoryView<MemorySegment>) view;
        MemoryDomain<MemorySegment> domain =
                (MemoryDomain<MemorySegment>)
                        Environment.current().runtimeFor(Device.C).memoryDomain();
        MemoryAccess<MemorySegment> access = domain.directAccess();
        long offset = Indexing.linearToOffset(typed, linearIndex);
        return access.readFloat(typed.memory(), offset);
    }

    private static void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-4f);
    }

    private static void assumeCBackendAvailable() {
        Assumptions.assumeTrue(CNative.isAvailable(), "C JNI runtime not available");
        Assumptions.assumeTrue(ExternalToolChecks.hasVersionCommand("gcc"), "gcc not available");
    }
}
