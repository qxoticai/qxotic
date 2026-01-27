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

class HipSumReductionTest {

    @Test
    void reducesSumOnHipBackend() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());

        Shape shape = Shape.of(2, 3);
        int n = Math.toIntExact(shape.size());
        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        host.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(shape));

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < n; i++) {
            long offset = Indexing.linearToOffset(hostView, i);
            access.writeFloat(hostView.memory(), offset, (float) (i + 1));
        }

        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryView<HipDevicePtr> deviceView =
                MemoryView.of(
                        hipContext.memoryAllocator().allocateMemory(DataType.FP32, n),
                        DataType.FP32,
                        Layout.rowMajor(shape));
        MemoryContext.copy(host, hostView, hipContext, deviceView);

        Environment env = Environment.current();
        Environment hipEnv =
                new Environment(Device.HIP, env.defaultFloat(), env.registry(), env.executionMode());

        MemoryView<MemorySegment> axis1 =
                reduceToHost(host, hipContext, hipEnv, deviceView, DataType.FP32, false, 1);
        assertEquals(6.0f, access.readFloat(axis1.memory(), Indexing.linearToOffset(axis1, 0)), 0.0001f);
        assertEquals(15.0f, access.readFloat(axis1.memory(), Indexing.linearToOffset(axis1, 1)), 0.0001f);

        MemoryView<MemorySegment> axis0 =
                reduceToHost(host, hipContext, hipEnv, deviceView, DataType.FP32, false, 0);
        assertEquals(5.0f, access.readFloat(axis0.memory(), Indexing.linearToOffset(axis0, 0)), 0.0001f);
        assertEquals(7.0f, access.readFloat(axis0.memory(), Indexing.linearToOffset(axis0, 1)), 0.0001f);
        assertEquals(9.0f, access.readFloat(axis0.memory(), Indexing.linearToOffset(axis0, 2)), 0.0001f);

        MemoryView<MemorySegment> keepDims =
                reduceToHost(host, hipContext, hipEnv, deviceView, DataType.FP32, true, 1);
        assertEquals(6.0f, access.readFloat(keepDims.memory(), Indexing.linearToOffset(keepDims, 0)), 0.0001f);
        assertEquals(15.0f, access.readFloat(keepDims.memory(), Indexing.linearToOffset(keepDims, 1)), 0.0001f);
    }

    private static MemoryView<MemorySegment> reduceToHost(
            MemoryContext<MemorySegment> host,
            HipMemoryContext hipContext,
            Environment hipEnv,
            MemoryView<HipDevicePtr> deviceView,
            DataType accumulatorType,
            boolean keepDims,
            int axis) {
        MemoryView<?> output =
                Environment.with(
                        hipEnv,
                        () ->
                                Tensor.of(deviceView)
                                        .sum(accumulatorType, keepDims, axis)
                                        .materialize());
        @SuppressWarnings("unchecked")
        MemoryView<HipDevicePtr> deviceOut = (MemoryView<HipDevicePtr>) output;
        MemoryView<MemorySegment> hostOut =
                MemoryView.of(
                        host.memoryAllocator().allocateMemory(
                                accumulatorType, deviceOut.shape().size()),
                        accumulatorType,
                        Layout.rowMajor(deviceOut.shape()));
        MemoryContext.copy(hipContext, deviceOut, host, hostOut);
        return hostOut;
    }
}
