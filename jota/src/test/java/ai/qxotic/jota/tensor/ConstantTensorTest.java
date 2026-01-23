package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class ConstantTensorTest {

    private static ComputeEngine dummyEngine() {
        return new ComputeEngine() {
            @Override
            public ComputeBackend backendFor(Device device) {
                throw new UnsupportedOperationException("No backend for " + device);
            }

            @Override
            public KernelCache cache() {
                return DiskKernelCache.defaultCache();
            }
        };
    }

    @Test
    void broadcastedFloatStaysLazy() {
        Tensor tensor = Tensor.broadcasted(3.5f, Shape.of(2, 3));

        assertTrue(tensor.isLazy());
        assertFalse(tensor.isMaterialized());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void materializesBroadcastedFloat(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.broadcasted(1.25f, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertEquals(DataType.FP32, view.dataType());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long firstOffset = Indexing.linearToOffset(view, 0);
                    long lastOffset = Indexing.linearToOffset(view, 5);
                    assertEquals(1.25f, access.readFloat(typedView.memory(), firstOffset), 0.0001f);
                    assertEquals(
                            1.25f, access.readFloat(typedView.memory(), lastOffset), 0.0001f);
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingI64")
    <B> void materializesBroadcastedLong(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.broadcasted(42L, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertEquals(DataType.I64, view.dataType());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long firstOffset = Indexing.linearToOffset(view, 0);
                    long lastOffset = Indexing.linearToOffset(view, 5);
                    assertEquals(42L, access.readLong(typedView.memory(), firstOffset));
                    assertEquals(42L, access.readLong(typedView.memory(), lastOffset));
                    return null;
                });
    }

    @Test
    void scalarFloatStaysLazy() {
        Tensor tensor = Tensor.scalar(3.5f);

        assertTrue(tensor.isLazy());
        assertFalse(tensor.isMaterialized());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(1L, tensor.size());
        assertEquals(0, tensor.stride().flatRank());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF64")
    <B> void materializesScalarDouble(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.scalar(2.125);
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.scalar(), view.shape());
                    assertEquals(1L, view.shape().size());
                    assertEquals(0, view.stride().flatRank());
                    assertEquals(DataType.FP64, view.dataType());
                    assertFalse(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(2.125, access.readDouble(typedView.memory(), offset), 0.000001);
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingI64")
    <B> void materializesScalarLong(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.scalar(42L);
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.scalar(), view.shape());
                    assertEquals(1L, view.shape().size());
                    assertEquals(0, view.stride().flatRank());
                    assertEquals(DataType.I64, view.dataType());
                    assertFalse(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(42L, access.readLong(typedView.memory(), offset));
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#allContexts")
    <B> void scalarUsesEnvironmentDefaultDevice(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.scalar(1.0f);
                    assertEquals(context.device(), tensor.device());
                    return null;
                });
    }
}
