package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class HipMatmulKernelSmokeTest {

    @Test
    void runsTracedMatmulKernel() {
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();

        MemoryView<?> output =
                Environment.with(
                        hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();

        MemoryView<?> output =
                Environment.with(
                        hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();

        MemoryView<?> transposed =
                Environment.with(
                        hipEnv,
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
    void matmulHandlesNonContiguousInput() {
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();

        MemoryView<?> output =
                Environment.with(
                        hipEnv,
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
                "HIP noncontig matmul out="
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
                                () -> {
                                    Tensor a = Tensor.iota(6, DataType.I64).view(Shape.of(2, 3));
                                    Tensor b = Tensor.iota(6, DataType.FP32).view(Shape.of(3, 2));
                                    return a.matmul(b).materialize();
                                }));
    }

    @Test
    void rejectsBatchedMatmulBatchMismatch() {
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
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
        assumeHipAvailable();
        Environment hipEnv = hipEnvironment();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        Environment.with(
                                hipEnv,
                                () -> {
                                    Tensor a =
                                            Tensor.iota(12, DataType.I64).view(Shape.of(2, 2, 3));
                                    Tensor b =
                                            Tensor.iota(12, DataType.FP32).view(Shape.of(2, 3, 2));
                                    return a.batchedMatmul(b).materialize();
                                }));
    }

    private static Environment hipEnvironment() {
        Environment current = Environment.current();
        return new Environment(
                new Device(DeviceType.HIP, 0), current.defaultFloat(), current.runtimes());
    }

    private static float readFp32(MemoryView<?> view, long linearIndex) {
        MemoryView<MemorySegment> hostView = toHost(view);
        MemoryAccess<MemorySegment> access = DomainFactory.ofMemorySegment().directAccess();
        long offset = Indexing.linearToOffset(hostView, linearIndex);
        return access.readFloat(hostView.memory(), offset);
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<MemorySegment> toHost(MemoryView<?> view) {
        return (MemoryView<MemorySegment>)
                Tensor.of(view).to(new Device(DeviceType.PANAMA, 0)).materialize();
    }

    private static void assertClose(float actual, float expected) {
        assertEquals(expected, actual, 1e-4f);
    }

    private static void assumeHipAvailable() {
        HipTestAssumptions.assumeHipReady();
    }
}
