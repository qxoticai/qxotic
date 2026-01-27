package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
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

class HipUnaryKernelSmokeTest {

    @Test
    void runsSqrtKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 256;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).sqrt().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(
                (float) Math.sqrt(n - 1), access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsUnaryExpKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 128;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i * 0.01f);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).exp().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        float expected = (float) Math.exp((n - 1) * 0.01f);
        assertEquals(expected, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsLogicalNotKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.BOOL, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.BOOL, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeByte(hostView.memory(), offset, (byte) ((i % 2 == 0) ? 1 : 0));
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.BOOL, n),
                        DataType.BOOL,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).logicalNot().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOut, i);
            byte value = access.readByte(hostOut.memory(), offset);
            byte expected = (byte) ((i % 2 == 0) ? 0 : 1);
            assertEquals(expected, value);
        }
    }

    @Test
    void runsBitwiseNotKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.I32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.I32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeInt(hostView.memory(), offset, i + 1);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.I32, n),
                        DataType.I32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).bitwiseNot().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOut, i);
            int value = access.readInt(hostOut.memory(), offset);
            int expected = ~(i + 1);
            assertEquals(expected, value);
        }
    }

    @Test
    void runsAbsKernelOnInt() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 8;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.I32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.I32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeInt(hostView.memory(), offset, i % 2 == 0 ? -(i + 1) : (i + 1));
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.I32, n),
                        DataType.I32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).abs().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostOut, i);
            int value = access.readInt(hostOut.memory(), offset);
            assertEquals(i + 1, value);
        }
    }

    @Test
    void runsReciprocalKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 16;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i + 1.0f);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> output = Tensor.of(dev).reciprocal().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostOut = (MemoryView<MemorySegment>) output;
        long lastOffset = Indexing.linearToOffset(hostOut, n - 1);
        assertEquals(1.0f / n, access.readFloat(hostOut.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsSinCosKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, i * 0.05f);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> sinOut = Tensor.of(dev).sin().materialize();
        MemoryView<?> cosOut = Tensor.of(dev).cos().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> sinHost = (MemoryView<MemorySegment>) sinOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> cosHost = (MemoryView<MemorySegment>) cosOut;

        long lastOffset = Indexing.linearToOffset(sinHost, n - 1);
        float expected = (float) Math.sin((n - 1) * 0.05f);
        assertEquals(expected, access.readFloat(sinHost.memory(), lastOffset), 0.0001f);
        expected = (float) Math.cos((n - 1) * 0.05f);
        assertEquals(expected, access.readFloat(cosHost.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsTanhLogKernel() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 64;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, 0.1f + i * 0.01f);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> tanhOut = Tensor.of(dev).tanh().materialize();
        MemoryView<?> logOut = Tensor.of(dev).log().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> tanhHost = (MemoryView<MemorySegment>) tanhOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> logHost = (MemoryView<MemorySegment>) logOut;

        long lastOffset = Indexing.linearToOffset(tanhHost, n - 1);
        float expected = (float) Math.tanh(0.1f + (n - 1) * 0.01f);
        assertEquals(expected, access.readFloat(tanhHost.memory(), lastOffset), 0.0001f);
        expected = (float) Math.log(0.1f + (n - 1) * 0.01f);
        assertEquals(expected, access.readFloat(logHost.memory(), lastOffset), 0.0001f);
    }

    @Test
    void runsFp64UnaryKernels() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        assumeHipccAvailable();

        int n = 32;
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        var hostMem = host.memoryAllocator().allocateMemory(DataType.FP64, n);
        MemoryView<MemorySegment> hostView =
                MemoryView.of(hostMem, DataType.FP64, Layout.rowMajor(Shape.flat(n)));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeDouble(hostView.memory(), offset, 0.1 + i * 0.01);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP64, n),
                        DataType.FP64,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        MemoryView<?> sinOut = Tensor.of(dev).sin().materialize();
        MemoryView<?> cosOut = Tensor.of(dev).cos().materialize();
        MemoryView<?> sqrtOut = Tensor.of(dev).sqrt().materialize();
        MemoryView<?> expOut = Tensor.of(dev).exp().materialize();
        MemoryView<?> logOut = Tensor.of(dev).log().materialize();
        MemoryView<?> tanhOut = Tensor.of(dev).tanh().materialize();

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> sinHost = (MemoryView<MemorySegment>) sinOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> cosHost = (MemoryView<MemorySegment>) cosOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> sqrtHost = (MemoryView<MemorySegment>) sqrtOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> expHost = (MemoryView<MemorySegment>) expOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> logHost = (MemoryView<MemorySegment>) logOut;
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> tanhHost = (MemoryView<MemorySegment>) tanhOut;

        long lastOffset = Indexing.linearToOffset(sinHost, n - 1);
        double input = 0.1 + (n - 1) * 0.01;
        assertEquals(Math.sin(input), access.readDouble(sinHost.memory(), lastOffset), 0.0000001);
        assertEquals(Math.cos(input), access.readDouble(cosHost.memory(), lastOffset), 0.0000001);
        assertEquals(Math.sqrt(input), access.readDouble(sqrtHost.memory(), lastOffset), 0.0000001);
        assertEquals(Math.exp(input), access.readDouble(expHost.memory(), lastOffset), 0.0000001);
        assertEquals(Math.log(input), access.readDouble(logHost.memory(), lastOffset), 0.0000001);
        assertEquals(Math.tanh(input), access.readDouble(tanhHost.memory(), lastOffset), 0.0000001);
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
}
