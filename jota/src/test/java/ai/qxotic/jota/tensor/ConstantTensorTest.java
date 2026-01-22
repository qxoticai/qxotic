package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class ConstantTensorTest {

    private static final MemoryContext<byte[]> CONTEXT = ContextFactory.ofBytes();

    private static final ComputeEngine ENGINE =
            new ComputeEngine() {
                @Override
                public ComputeBackend backendFor(Device device) {
                    throw new UnsupportedOperationException("No backend for " + device);
                }

                @Override
                public KernelCache cache() {
                    return DiskKernelCache.defaultCache();
                }
            };

    @BeforeAll
    static void registerContext() {
        DeviceRegistry.global().register(Device.JAVA, CONTEXT, ENGINE);
    }

    @Test
    void broadcastedFloatStaysLazy() {
        Tensor tensor = Tensor.broadcasted(3.5f, Shape.of(2, 3));

        assertTrue(tensor.isLazy());
        assertFalse(tensor.isMaterialized());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @Test
    void materializesBroadcastedFloat() {
        Tensor tensor = Tensor.broadcasted(1.25f, Shape.of(2, 3));
        MemoryView<?> view = tensor.materialize();

        assertEquals(Shape.of(2, 3), view.shape());
        assertEquals(DataType.FP32, view.dataType());
        assertTrue(view.isBroadcasted());

        MemoryAccess<byte[]> access = CONTEXT.memoryAccess();
        @SuppressWarnings("unchecked")
        MemoryView<byte[]> byteView = (MemoryView<byte[]>) view;
        long firstOffset = Indexing.linearToOffset(view, 0);
        long lastOffset = Indexing.linearToOffset(view, 5);
        assertEquals(1.25f, access.readFloat(byteView.memory(), firstOffset), 0.0001f);
        assertEquals(1.25f, access.readFloat(byteView.memory(), lastOffset), 0.0001f);
    }

    @Test
    void materializesBroadcastedLong() {
        Tensor tensor = Tensor.broadcasted(42L, Shape.of(2, 3));
        MemoryView<?> view = tensor.materialize();

        assertEquals(Shape.of(2, 3), view.shape());
        assertEquals(DataType.I64, view.dataType());
        assertTrue(view.isBroadcasted());

        MemoryAccess<byte[]> access = CONTEXT.memoryAccess();
        @SuppressWarnings("unchecked")
        MemoryView<byte[]> byteView = (MemoryView<byte[]>) view;
        long firstOffset = Indexing.linearToOffset(view, 0);
        long lastOffset = Indexing.linearToOffset(view, 5);
        assertEquals(42L, access.readLong(byteView.memory(), firstOffset));
        assertEquals(42L, access.readLong(byteView.memory(), lastOffset));
    }
}
