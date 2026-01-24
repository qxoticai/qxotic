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
import ai.qxotic.jota.memory.MemoryViewPrinter;
import ai.qxotic.jota.memory.impl.ContextFactory;
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

    @Test
    void canary() {
        Tensor scalar =
                Tensor.scalar(3.14f); // Tensor.of(new float[] {3.14f}).view(Shape.scalar());
        Tensor reciprocal1 = scalar.reciprocal();
        Tensor reciprocal =
                TensorOpsContext.with(
                        new EagerTensorOps(ContextFactory.ofMemorySegment()), scalar::reciprocal);
        MemoryAccess access = Environment.current().panamaContext().memoryAccess();
        System.out.println(MemoryViewPrinter.toString(reciprocal1.materialize(), access));
    }

    // ========== Typed scalar tests (carrier methods) ==========

    @Test
    void scalarWithDoubleCarrierCreatesFP16() {
        Tensor tensor = Tensor.scalar(3.14, DataType.FP16);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP16, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesBF16() {
        Tensor tensor = Tensor.scalar(2.718, DataType.BF16);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.BF16, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesFP32() {
        Tensor tensor = Tensor.scalar(1.5, DataType.FP32);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesFP64() {
        Tensor tensor = Tensor.scalar(1.5, DataType.FP64);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP64, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI8() {
        Tensor tensor = Tensor.scalar(42L, DataType.I8);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I8, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI16() {
        Tensor tensor = Tensor.scalar(1000L, DataType.I16);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I16, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI32() {
        Tensor tensor = Tensor.scalar(100000L, DataType.I32);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I32, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI64() {
        Tensor tensor = Tensor.scalar(9999999999L, DataType.I64);

        assertTrue(tensor.isLazy());
        assertTrue(tensor.isScalarBroadcast());
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I64, tensor.dataType());
    }

    // ========== Constant folding tests (binary ops) ==========

    private static ConstantComputation getConstant(Tensor tensor) {
        return tensor.computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .orElseThrow(() -> new AssertionError("Expected ConstantComputation"));
    }

    @Test
    void foldsBinaryAdd() {
        Tensor result = Tensor.scalar(2.0f).add(3.0f);

        assertTrue(result.isLazy());
        assertTrue(result.isScalarBroadcast());
        assertEquals(Shape.scalar(), result.shape());
        assertEquals(DataType.FP32, result.dataType());

        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinarySubtract() {
        Tensor result = Tensor.scalar(10.0f).subtract(3.0f);

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(7.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryMultiply() {
        Tensor result = Tensor.scalar(4.0f).multiply(5.0f);

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(20.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryDivide() {
        Tensor result = Tensor.scalar(15.0f).divide(3.0f);

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryWithTypePromotion() {
        Tensor result = Tensor.scalar(2.0f).add(Tensor.scalar(3.0)); // FP32 + FP64 -> FP64

        assertTrue(result.isLazy());
        assertEquals(DataType.FP64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsChainedBinaryOps() {
        // (2 + 3) * 4 = 20
        Tensor result = Tensor.scalar(2.0f).add(3.0f).multiply(4.0f);

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(20.0, constant.value().doubleValue(), 0.0001);
    }

    // ========== Constant folding tests (unary ops) ==========

    @Test
    void foldsUnaryNegate() {
        Tensor result = Tensor.scalar(5.0f).negate();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(-5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryAbs() {
        Tensor result = Tensor.scalar(-7.0f).abs();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(7.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryExp() {
        Tensor result = Tensor.scalar(1.0f).exp();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(Math.E, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryLog() {
        Tensor result = Tensor.scalar((float) Math.E).log();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(1.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySqrt() {
        Tensor result = Tensor.scalar(16.0f).sqrt();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(4.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySin() {
        Tensor result = Tensor.scalar(0.0f).sin();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryCos() {
        Tensor result = Tensor.scalar(0.0f).cos();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(1.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryTanh() {
        Tensor result = Tensor.scalar(0.0f).tanh();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryReciprocal() {
        Tensor result = Tensor.scalar(4.0f).reciprocal();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.25, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySigmoid() {
        Tensor result = Tensor.scalar(0.0f).sigmoid();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.5, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsChainedUnaryOps() {
        // abs(-5) = 5, then negate = -5
        Tensor result = Tensor.scalar(-5.0f).abs().negate();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(-5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryPreservesDataType() {
        Tensor result = Tensor.scalar(4.0, DataType.FP32).negate();

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void foldsMixedUnaryAndBinaryOps() {
        // negate(2) + 5 = -2 + 5 = 3
        Tensor result = Tensor.scalar(2.0f).negate().add(5.0f);

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(3.0, constant.value().doubleValue(), 0.0001);
    }
}
