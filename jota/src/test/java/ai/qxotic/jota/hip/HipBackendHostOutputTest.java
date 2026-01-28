package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipBackendHostOutputTest {

    @Test
    void hipComputeReturnsHostOutputWhenDefaultDeviceIsCpu() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());
        HipRuntime.requireAvailable();
        String hsacoPath = System.getenv("HIP_HSACO_PATH");
        Assumptions.assumeTrue(hsacoPath != null && !hsacoPath.isBlank());

        Environment env = Environment.current();
        Environment cpuEnv =
                new Environment(
                        Device.PANAMA, env.defaultFloat(), env.backends(), env.executionMode());

        MemoryView<?> output =
                Environment.with(
                        cpuEnv,
                        () -> {
                            int n = 256;
                            MemoryContext<MemorySegment> hostContext = cpuEnv.panamaContext();
                            HipMemoryContext hipContext = HipMemoryContext.instance();

                            var hostMemA =
                                    hostContext.memoryAllocator().allocateMemory(DataType.FP32, n);
                            var hostMemB =
                                    hostContext.memoryAllocator().allocateMemory(DataType.FP32, n);
                            MemoryView<MemorySegment> hostA =
                                    MemoryView.of(
                                            hostMemA,
                                            DataType.FP32,
                                            ai.qxotic.jota.Layout.rowMajor(Shape.flat(n)));
                            MemoryView<MemorySegment> hostB =
                                    MemoryView.of(
                                            hostMemB,
                                            DataType.FP32,
                                            ai.qxotic.jota.Layout.rowMajor(Shape.flat(n)));

                            MemoryAccess<MemorySegment> hostAccess = hostContext.memoryAccess();
                            for (int i = 0; i < n; i++) {
                                long offset = Indexing.linearToOffset(hostA, i);
                                hostAccess.writeFloat(hostA.memory(), offset, i);
                                hostAccess.writeFloat(hostB.memory(), offset, 1.0f);
                            }

                            MemoryView<HipDevicePtr> devA =
                                    MemoryView.of(
                                            hipContext
                                                    .memoryAllocator()
                                                    .allocateMemory(DataType.FP32, n),
                                            DataType.FP32,
                                            ai.qxotic.jota.Layout.rowMajor(Shape.flat(n)));
                            MemoryView<HipDevicePtr> devB =
                                    MemoryView.of(
                                            hipContext
                                                    .memoryAllocator()
                                                    .allocateMemory(DataType.FP32, n),
                                            DataType.FP32,
                                            ai.qxotic.jota.Layout.rowMajor(Shape.flat(n)));

                            MemoryContext.copy(hostContext, hostA, hipContext, devA);
                            MemoryContext.copy(hostContext, hostB, hipContext, devB);

                            Tensor a = Tensor.of(devA);
                            Tensor b = Tensor.of(devB);
                            return a.add(b).materialize();
                        });

        assertSame(Device.PANAMA, output.memory().device());

        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> hostView = (MemoryView<MemorySegment>) output;
        MemoryContext<MemorySegment> hostContext = cpuEnv.panamaContext();
        MemoryAccess<MemorySegment> access = hostContext.memoryAccess();
        long lastOffset = Indexing.linearToOffset(hostView, 255);
        assertEquals(256.0f, access.readFloat(hostView.memory(), lastOffset), 0.0001f);
    }
}
