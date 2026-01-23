package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
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
                    assertEquals(1.25f, access.readFloat(typedView.memory(), lastOffset), 0.0001f);
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

    // ========== zeros tests ==========

    @Test
    void zerosUsesDefaultFloat() {
        Tensor tensor = Tensor.zeros(Shape.of(2, 3));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.defaultFloat(), tensor.dataType());
        assertEquals(Shape.of(2, 3), tensor.shape());
    }

    @Test
    void zerosWithExplicitDtype() {
        Tensor tensor = Tensor.zeros(DataType.I32, Shape.of(4, 5));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.I32, tensor.dataType());
        assertEquals(Shape.of(4, 5), tensor.shape());
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void materializesZeros(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.zeros(DataType.FP32, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(0.0f, access.readFloat(typedView.memory(), offset), 0.0001f);
                    return null;
                });
    }

    // ========== ones tests ==========

    @Test
    void onesUsesDefaultFloat() {
        Tensor tensor = Tensor.ones(Shape.of(3, 4));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.defaultFloat(), tensor.dataType());
        assertEquals(Shape.of(3, 4), tensor.shape());
    }

    @Test
    void onesWithExplicitDtype() {
        Tensor tensor = Tensor.ones(DataType.I64, Shape.of(2, 2));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.I64, tensor.dataType());
        assertEquals(Shape.of(2, 2), tensor.shape());
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void materializesOnes(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.ones(DataType.FP32, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(1.0f, access.readFloat(typedView.memory(), offset), 0.0001f);
                    return null;
                });
    }

    // ========== full tests ==========

    @Test
    void fullFloatInfersDtype() {
        Tensor tensor = Tensor.full(3.14f, Shape.of(2, 2));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @Test
    void fullDoubleInfersDtype() {
        Tensor tensor = Tensor.full(2.718, Shape.of(3, 3));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.FP64, tensor.dataType());
    }

    @Test
    void fullLongInfersDtype() {
        Tensor tensor = Tensor.full(42L, Shape.of(4, 4));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.I64, tensor.dataType());
    }

    @Test
    void fullIntInfersDtype() {
        Tensor tensor = Tensor.full(7, Shape.of(5, 5));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.I32, tensor.dataType());
    }

    @Test
    void fullWithExplicitDtype() {
        Tensor tensor = Tensor.full(Integer.valueOf(99), DataType.I16, Shape.of(2, 3));

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(DataType.I16, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void materializesFull(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.full(7.5f, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long firstOffset = Indexing.linearToOffset(view, 0);
                    long lastOffset = Indexing.linearToOffset(view, 5);
                    assertEquals(7.5f, access.readFloat(typedView.memory(), firstOffset), 0.0001f);
                    assertEquals(7.5f, access.readFloat(typedView.memory(), lastOffset), 0.0001f);
                    return null;
                });
    }

    // ========== Array creation tests ==========

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void ofFloatArray(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(tensor.isMaterialized());
                    assertEquals(DataType.FP32, tensor.dataType());
                    assertEquals(Shape.flat(4), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;

                    for (int i = 0; i < data.length; i++) {
                        long offset = Indexing.linearToOffset(view, i);
                        assertEquals(
                                data[i], access.readFloat(typedView.memory(), offset), 0.0001f);
                    }
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF32")
    <B> void ofFloatArrayWithShape(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
                    Tensor tensor = Tensor.of(data, Shape.of(2, 3));

                    assertTrue(tensor.isMaterialized());
                    assertEquals(Shape.of(2, 3), tensor.shape());
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingF64")
    <B> void ofDoubleArray(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    double[] data = {1.0, 2.0, 3.0};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(tensor.isMaterialized());
                    assertEquals(DataType.FP64, tensor.dataType());
                    assertEquals(Shape.flat(3), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;

                    for (int i = 0; i < data.length; i++) {
                        long offset = Indexing.linearToOffset(view, i);
                        assertEquals(
                                data[i], access.readDouble(typedView.memory(), offset), 0.000001);
                    }
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingI32")
    <B> void ofIntArray(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    int[] data = {10, 20, 30, 40};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(tensor.isMaterialized());
                    assertEquals(DataType.I32, tensor.dataType());
                    assertEquals(Shape.flat(4), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;

                    for (int i = 0; i < data.length; i++) {
                        long offset = Indexing.linearToOffset(view, i);
                        assertEquals(data[i], access.readInt(typedView.memory(), offset));
                    }
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#contextsSupportingI64")
    <B> void ofLongArray(MemoryContext<B> context) {
        DeviceRegistry registry = DeviceRegistry.builder().register(context, dummyEngine()).build();
        Environment environment =
                new Environment(context.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    long[] data = {100L, 200L, 300L};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(tensor.isMaterialized());
                    assertEquals(DataType.I64, tensor.dataType());
                    assertEquals(Shape.flat(3), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = context.memoryAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;

                    for (int i = 0; i < data.length; i++) {
                        long offset = Indexing.linearToOffset(view, i);
                        assertEquals(data[i], access.readLong(typedView.memory(), offset));
                    }
                    return null;
                });
    }

    @Test
    void ofArrayShapeMismatchThrows() {
        float[] data = {1.0f, 2.0f, 3.0f};
        assertThrows(
                IllegalArgumentException.class, () -> Tensor.of(data, Shape.of(2, 3))); // 3 != 6
    }
}
