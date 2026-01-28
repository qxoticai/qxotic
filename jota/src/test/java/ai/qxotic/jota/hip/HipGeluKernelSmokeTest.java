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

class HipGeluKernelSmokeTest {

    @Test
    void runsGeluKernel() {
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
            access.writeFloat(hostView.memory(), offset, (i - 64) * 0.05f);
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> dev =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(host, hostView, hipContext, dev);

        Environment env = Environment.current();
        Environment hipEnv =
                new Environment(
                        Device.HIP, env.defaultFloat(), env.backends(), env.executionMode());

        MemoryView<?> output = Environment.with(hipEnv, () -> Tensor.of(dev).gelu().materialize());

        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> deviceOut = (MemoryView<HipDevicePtr>) output;
        var hostOutMem = host.memoryAllocator().allocateMemory(DataType.FP32, n);
        MemoryView<MemorySegment> hostOut =
                MemoryView.of(hostOutMem, DataType.FP32, Layout.rowMajor(Shape.flat(n)));
        MemoryContext.copy(hipContext, deviceOut, host, hostOut);
        long offset = Indexing.linearToOffset(hostOut, n - 1);
        float input = (n - 1 - 64) * 0.05f;
        float expected = geluApprox(input);
        assertEquals(expected, access.readFloat(hostOut.memory(), offset), 0.0005f);
    }

    private static float geluApprox(float x) {
        float cubic = x * x * x;
        float inner = cubic * 0.044715f + x;
        float scaled = inner * 0.7978845608f;
        float tanh = (float) Math.tanh(scaled);
        return x * 0.5f * (1.0f + tanh);
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
