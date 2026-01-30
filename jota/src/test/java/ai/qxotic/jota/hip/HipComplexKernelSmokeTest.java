package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipComplexKernelSmokeTest {

    @Test
    void runsSigmoidKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 64) * 0.05f);
        MemoryView<MemorySegment> output = runOnHip(hostInput, t -> t.sigmoid(), DataType.FP32, n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long lastOffset = Indexing.linearToOffset(output, n - 1);
        float input = (n - 1 - 64) * 0.05f;
        float expected = 1.0f / (1.0f + (float) Math.exp(-input));
        assertEquals(expected, access.readFloat(output.memory(), lastOffset), 0.0005f);
    }

    @Test
    void runsSiluKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 64) * 0.05f);
        MemoryView<MemorySegment> output = runOnHip(hostInput, t -> t.silu(), DataType.FP32, n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long lastOffset = Indexing.linearToOffset(output, n - 1);
        float input = (n - 1 - 64) * 0.05f;
        float sigmoid = 1.0f / (1.0f + (float) Math.exp(-input));
        float expected = input * sigmoid;
        assertEquals(expected, access.readFloat(output.memory(), lastOffset), 0.0005f);
    }

    @Test
    void runsReluKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 8) * 0.5f);
        MemoryView<MemorySegment> output = runOnHip(hostInput, t -> t.relu(), DataType.FP32, n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long offset0 = Indexing.linearToOffset(output, 0);
        long offset15 = Indexing.linearToOffset(output, 15);
        assertEquals(0.0f, access.readFloat(output.memory(), offset0), 0.0001f);
        assertEquals(3.5f, access.readFloat(output.memory(), offset15), 0.0001f);
    }

    @Test
    void runsWhereKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 8) * 0.5f);
        MemoryView<MemorySegment> output =
                runOnHip(
                        hostInput,
                        t ->
                                Tensor.where(
                                        t.lessThan(Tensor.scalar(0f, t.dataType())), t.negate(), t),
                        DataType.FP32,
                        n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long offset0 = Indexing.linearToOffset(output, 0);
        long offset15 = Indexing.linearToOffset(output, 15);
        assertEquals(4.0f, access.readFloat(output.memory(), offset0), 0.0001f);
        assertEquals(3.5f, access.readFloat(output.memory(), offset15), 0.0001f);
    }

    @Test
    void runsFusedMathChainKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i + 1) * 0.01f);
        MemoryView<MemorySegment> output =
                runOnHip(
                        hostInput,
                        t -> t.exp().add(t.add(Tensor.scalar(1f, t.dataType())).log()).tanh(),
                        DataType.FP32,
                        n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long lastOffset = Indexing.linearToOffset(output, n - 1);
        float input = (n - 1 + 1) * 0.01f;
        float expected = (float) Math.tanh(Math.exp(input) + Math.log(input + 1.0f));
        assertEquals(expected, access.readFloat(output.memory(), lastOffset), 0.0008f);
    }

    @Test
    void runsGeluSiluFusionKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 64) * 0.03f);
        MemoryView<MemorySegment> output =
                runOnHip(hostInput, t -> t.gelu().add(t.silu()), DataType.FP32, n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long lastOffset = Indexing.linearToOffset(output, n - 1);
        float input = (n - 1 - 64) * 0.03f;
        float expected = geluApprox(input) + siluApprox(input);
        assertEquals(expected, access.readFloat(output.memory(), lastOffset), 0.0008f);
    }

    @Test
    void runsMixedWhereCompareKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 32) * 0.25f);
        MemoryView<MemorySegment> output =
                runOnHip(
                        hostInput,
                        t ->
                                Tensor.where(
                                        t.lessThan(Tensor.scalar(0f, t.dataType())),
                                        t.negate().square(),
                                        t.square()),
                        DataType.FP32,
                        n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long offset0 = Indexing.linearToOffset(output, 0);
        long offset63 = Indexing.linearToOffset(output, 63);
        float in0 = (0 - 32) * 0.25f;
        float in63 = (63 - 32) * 0.25f;
        assertEquals(in0 * in0, access.readFloat(output.memory(), offset0), 0.0001f);
        assertEquals(in63 * in63, access.readFloat(output.memory(), offset63), 0.0001f);
    }

    @Test
    void runsFp64FusedKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        MemoryView<MemorySegment> hostInput = hostArrayFp64(n, i -> 0.1 + i * 0.02);
        MemoryView<MemorySegment> output =
                runOnHip(hostInput, t -> t.exp().add(t.log()).tanh(), DataType.FP64, n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long lastOffset = Indexing.linearToOffset(output, n - 1);
        double input = 0.1 + (n - 1) * 0.02;
        double expected = Math.tanh(Math.exp(input) + Math.log(input));
        assertEquals(expected, access.readDouble(output.memory(), lastOffset), 0.0000005);
    }

    @Test
    void runsMixedLogicalKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        MemoryView<MemorySegment> hostInput = hostArray(n, i -> (i - 32) * 0.25f);
        MemoryView<MemorySegment> output =
                runOnHip(
                        hostInput,
                        t ->
                                Tensor.where(
                                        t.lessThan(Tensor.scalar(0f, t.dataType()))
                                                .logicalOr(
                                                        t.greaterThan(
                                                                Tensor.scalar(1f, t.dataType()))),
                                        t.negate(),
                                        t),
                        DataType.FP32,
                        n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long offset0 = Indexing.linearToOffset(output, 0);
        long offset63 = Indexing.linearToOffset(output, 63);
        assertEquals(8.0f, access.readFloat(output.memory(), offset0), 0.0001f);
        assertEquals(-7.75f, access.readFloat(output.memory(), offset63), 0.0001f);
    }

    @Test
    void runsBitwiseKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryView<MemorySegment> hostInput = hostArrayI32(n, i -> i + 1);
        MemoryView<MemorySegment> output =
                runOnHip(
                        hostInput,
                        t ->
                                t.bitwiseAnd(Tensor.scalar(3, t.dataType()))
                                        .bitwiseXor(Tensor.scalar(1, t.dataType())),
                        DataType.I32,
                        n);

        MemoryAccess<MemorySegment> access = hostAccess();
        long offset0 = Indexing.linearToOffset(output, 0);
        long offset1 = Indexing.linearToOffset(output, 1);
        assertEquals(((1 & 3) ^ 1), access.readInt(output.memory(), offset0));
        assertEquals(((2 & 3) ^ 1), access.readInt(output.memory(), offset1));
    }

    @Test
    void runsRangeFusionKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        Environment env = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP, env.defaultFloat(), env.backends(), env.executionMode());
        MemoryView<?> deviceOut =
                Environment.with(
                        hipEnv,
                        () -> Tensor.iota(n, DataType.FP32).add(1.0f).tanh().materialize());

        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> hipOut = (MemoryView<HipDevicePtr>) deviceOut;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostOutMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostOut =
                MemoryView.of(hostOutMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(HipMemoryContext.instance(), hipOut, host, hostOut);

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        float expected = (float) Math.tanh(n);
        assertEquals(expected, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    private static MemoryView<MemorySegment> hostArray(int n, IndexValue supplier) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static MemoryView<MemorySegment> hostArrayFp64(int n, DoubleIndexValue supplier) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP64, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP64, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeDouble(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static MemoryView<MemorySegment> hostArrayI32(int n, IntIndexValue supplier) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.I32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.I32, Layout.rowMajor(Shape.flat(n)));
        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeInt(hostView.memory(), offset, supplier.value(i));
        }
        return hostView;
    }

    private static float geluApprox(float x) {
        float cubic = x * x * x;
        float inner = cubic * 0.044715f + x;
        float scaled = inner * 0.7978845608f;
        float tanh = (float) Math.tanh(scaled);
        return x * 0.5f * (1.0f + tanh);
    }

    private static float siluApprox(float x) {
        float sigmoid = 1.0f / (1.0f + (float) Math.exp(-x));
        return x * sigmoid;
    }

    private static MemoryView<MemorySegment> runOnHip(
            MemoryView<MemorySegment> hostInput, TensorOp op, DataType dataType, int n) {
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        HipMemoryContext hipContext = HipMemoryContext.instance();

        MemoryView<HipDevicePtr> devInput =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(dataType, n),
                        dataType,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostInput, hipContext, devInput);

        Environment env = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP, env.defaultFloat(), env.backends(), env.executionMode());
        MemoryView<?> deviceOut =
                Environment.with(hipEnv, () -> op.apply(Tensor.of(devInput)).materialize());

        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> hipOut = (MemoryView<HipDevicePtr>) deviceOut;
        var hostOutMem = host.memoryAllocator().allocateMemory(dataType, n);
        MemoryView<MemorySegment> hostOut =
                MemoryView.of(hostOutMem, dataType, Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(hipContext, hipOut, host, hostOut);
        return hostOut;
    }

    private static MemoryAccess<MemorySegment> hostAccess() {
        return ContextFactory.ofMemorySegment().memoryAccess();
    }

    private static void assumeHipccAvailable() {
        try {
            Process process = new ProcessBuilder("hipcc", "--version").start();
            int code = process.waitFor();
            Assumptions.assumeTrue(code == 0);
        } catch (Exception e) {
            Assumptions.assumeTrue(false, "hipcc not available");
        }
    }

    @FunctionalInterface
    private interface TensorOp {
        Tensor apply(Tensor t);
    }

    @FunctionalInterface
    private interface IndexValue {
        float value(int index);
    }

    @FunctionalInterface
    private interface DoubleIndexValue {
        double value(int index);
    }

    @FunctionalInterface
    private interface IntIndexValue {
        int value(int index);
    }
}
