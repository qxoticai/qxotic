package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.backend.Backend;
import ai.qxotic.jota.backend.DefaultBackendRegistry;
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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
        DefaultBackendRegistry registry = new DefaultBackendRegistry();
        registry.register(new StubBackend<>(context, dummyEngine()));
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

    private static final class StubBackend<B> implements Backend {
        private final MemoryContext<B> context;
        private final ComputeEngine engine;

        private StubBackend(MemoryContext<B> context, ComputeEngine engine) {
            this.context = context;
            this.engine = engine;
        }

        @Override
        public Device device() {
            return context.device();
        }

        @Override
        public MemoryContext<?> memoryContext() {
            return context;
        }

        @Override
        public ComputeEngine computeEngine() {
            return engine;
        }

        @Override
        public java.util.Optional<ai.qxotic.jota.backend.KernelService> kernels() {
            return java.util.Optional.empty();
        }
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

    // ========== min/max constant folding tests ==========

    @Test
    void foldsBinaryMin() {
        Tensor result = Tensor.scalar(7.0f).min(Tensor.scalar(3.0f));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(3.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryMax() {
        Tensor result = Tensor.scalar(7.0f).max(Tensor.scalar(3.0f));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(7.0, constant.value().doubleValue(), 0.0001);
    }

    // ========== Activation functions constant folding tests ==========

    @Test
    void foldsReluPositive() {
        Tensor result = Tensor.scalar(5.0f).relu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsReluNegative() {
        Tensor result = Tensor.scalar(-3.0f).relu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsReluZero() {
        Tensor result = Tensor.scalar(0.0f).relu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsSilu() {
        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        Tensor result = Tensor.scalar(0.0f).silu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsSiluPositive() {
        // silu(x) = x * sigmoid(x)
        // For x=1: sigmoid(1) ≈ 0.7311, silu(1) ≈ 0.7311
        Tensor result = Tensor.scalar(1.0f).silu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        double expected = 1.0 / (1.0 + Math.exp(-1.0)); // sigmoid(1)
        assertEquals(expected, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsGelu() {
        // gelu(0) = 0 (since tanh(0) = 0, and 0 * anything = 0)
        Tensor result = Tensor.scalar(0.0f).gelu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsGeluPositive() {
        // GELU(1) ≈ 0.8413 (approximation)
        Tensor result = Tensor.scalar(1.0f).gelu();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        double x = 1.0;
        double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        double expected = 0.5 * x * (1 + Math.tanh(inner));
        assertEquals(expected, constant.value().doubleValue(), 0.01);
    }

    // ========== Integer constant folding tests ==========

    @Test
    void foldsIntegerAdd() {
        Tensor result = Tensor.scalar(100L).add(Tensor.scalar(200L));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(300L, constant.value().longValue());
    }

    @Test
    void foldsIntegerMultiply() {
        Tensor result = Tensor.scalar(12L).multiply(Tensor.scalar(11L));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(132L, constant.value().longValue());
    }

    @Test
    void foldsIntegerNegate() {
        Tensor result = Tensor.scalar(42L).negate();

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(-42L, constant.value().longValue());
    }

    @Test
    void foldsIntegerAbs() {
        Tensor result = Tensor.scalar(-99L).abs();

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(99L, constant.value().longValue());
    }

    @Test
    void foldsIntegerMinMax() {
        Tensor minResult = Tensor.scalar(50L).min(Tensor.scalar(30L));
        Tensor maxResult = Tensor.scalar(50L).max(Tensor.scalar(30L));

        assertTrue(minResult.isLazy());
        assertTrue(maxResult.isLazy());
        assertEquals(30L, getConstant(minResult).value().longValue());
        assertEquals(50L, getConstant(maxResult).value().longValue());
    }

    @Test
    void foldsIntegerDivision() {
        // Integer division: 17 / 5 = 3
        Tensor result = Tensor.scalar(17L).divide(Tensor.scalar(5L));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(3L, constant.value().longValue());
    }

    @Test
    void preservesI32DataType() {
        Tensor result = Tensor.scalar(10L, DataType.I32).add(Tensor.scalar(5L, DataType.I32));

        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(15L, constant.value().longValue());
    }

    // ========== Activation functions preserve dataType ==========

    @Test
    void sigmoidPreservesDataType() {
        Tensor result = Tensor.scalar(0.0, DataType.FP32).sigmoid();

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void reluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).relu();

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void siluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).silu();

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void geluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).gelu();

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
    }

    // ========== Cast constant folding tests ==========

    @Test
    void foldsCastFloatToDouble() {
        Tensor result = Tensor.scalar(3.14f).cast(DataType.FP64);

        assertTrue(result.isLazy());
        assertEquals(DataType.FP64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(3.14, constant.value().doubleValue(), 0.001);
    }

    @Test
    void foldsCastDoubleToFloat() {
        Tensor result = Tensor.scalar(2.718).cast(DataType.FP32);

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(2.718f, constant.value().floatValue(), 0.001f);
    }

    @Test
    void foldsCastIntToFloat() {
        Tensor result = Tensor.scalar(42L, DataType.I32).cast(DataType.FP32);

        assertTrue(result.isLazy());
        assertEquals(DataType.FP32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(42.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsCastFloatToInt() {
        Tensor result = Tensor.scalar(7.9f).cast(DataType.I32);

        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(7L, constant.value().longValue());
    }

    @Test
    void foldsCastI32ToI64() {
        Tensor result = Tensor.scalar(100L, DataType.I32).cast(DataType.I64);

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(100L, constant.value().longValue());
    }

    @Test
    void castSameTypeReturnsThis() {
        Tensor original = Tensor.scalar(5.0f);
        Tensor result = original.cast(DataType.FP32);

        assertSame(original, result);
    }

    // ========== Comparison constant folding tests ==========

    @Test
    void foldsEqualTrue() {
        Tensor result = Tensor.scalar(5.0f).equal(Tensor.scalar(5.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue());
    }

    @Test
    void foldsEqualFalse() {
        Tensor result = Tensor.scalar(5.0f).equal(Tensor.scalar(3.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue());
    }

    @Test
    void foldsLessThanTrue() {
        Tensor result = Tensor.scalar(3.0f).lessThan(Tensor.scalar(5.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue());
    }

    @Test
    void foldsLessThanFalse() {
        Tensor result = Tensor.scalar(7.0f).lessThan(Tensor.scalar(5.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue());
    }

    @Test
    void foldsNotEqual() {
        // notEqual uses equal().logicalNot()
        Tensor result = Tensor.scalar(5.0f).notEqual(Tensor.scalar(3.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue()); // 5 != 3 is true
    }

    @Test
    void foldsGreaterThan() {
        // greaterThan uses other.lessThan(this)
        Tensor result = Tensor.scalar(7.0f).greaterThan(Tensor.scalar(3.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue()); // 7 > 3 is true
    }

    @Test
    void foldsLessThanOrEqual() {
        // lessThanOrEqual uses other.lessThan(this).logicalNot()
        Tensor result = Tensor.scalar(3.0f).lessThanOrEqual(Tensor.scalar(5.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue()); // 3 <= 5 is true
    }

    @Test
    void foldsGreaterThanOrEqual() {
        // greaterThanOrEqual uses lessThan(other).logicalNot()
        Tensor result = Tensor.scalar(5.0f).greaterThanOrEqual(Tensor.scalar(5.0f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue()); // 5 >= 5 is true
    }

    @Test
    void foldsIntegerComparison() {
        Tensor result = Tensor.scalar(10L).lessThan(Tensor.scalar(20L));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(1L, constant.value().longValue());
    }

    @Test
    void foldsLogicalNot() {
        // Create a boolean constant (via comparison) and negate it
        Tensor boolTrue = Tensor.scalar(5.0f).equal(Tensor.scalar(5.0f)); // true (1)
        Tensor result = boolTrue.logicalNot();

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue()); // !true = false
    }

    // ========== Bitwise operations constant folding tests ==========

    @Test
    void foldsBitwiseNot() {
        // ~0x0F = 0xFFFFFFFFFFFFFFF0 for I64
        Tensor result = Tensor.scalar(0x0FL).bitwiseNot();

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(~0x0FL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseNotAllOnes() {
        // ~(-1) = 0
        Tensor result = Tensor.scalar(-1L).bitwiseNot();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseAnd() {
        // 0xFF & 0x0F = 0x0F
        Tensor result = Tensor.scalar(0xFFL).bitwiseAnd(Tensor.scalar(0x0FL));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0x0FL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseAndWithZero() {
        // anything & 0 = 0
        Tensor result = Tensor.scalar(0x12345678L).bitwiseAnd(Tensor.scalar(0L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseAndWithAllOnes() {
        // x & (-1) = x (all bits set)
        Tensor result = Tensor.scalar(0xABCDL).bitwiseAnd(Tensor.scalar(-1L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0xABCDL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseOr() {
        // 0xF0 | 0x0F = 0xFF
        Tensor result = Tensor.scalar(0xF0L).bitwiseOr(Tensor.scalar(0x0FL));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0xFFL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseOrWithZero() {
        // x | 0 = x
        Tensor result = Tensor.scalar(0x12345678L).bitwiseOr(Tensor.scalar(0L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0x12345678L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseOrWithAllOnes() {
        // x | (-1) = -1 (all bits set)
        Tensor result = Tensor.scalar(0xABCDL).bitwiseOr(Tensor.scalar(-1L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(-1L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseXor() {
        // 0xFF ^ 0x0F = 0xF0
        Tensor result = Tensor.scalar(0xFFL).bitwiseXor(Tensor.scalar(0x0FL));

        assertTrue(result.isLazy());
        assertEquals(DataType.I64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0xF0L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseXorSameValue() {
        // x ^ x = 0
        Tensor result = Tensor.scalar(0x12345678L).bitwiseXor(Tensor.scalar(0x12345678L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0L, constant.value().longValue());
    }

    @Test
    void foldsBitwiseXorWithZero() {
        // x ^ 0 = x
        Tensor result = Tensor.scalar(0xABCDL).bitwiseXor(Tensor.scalar(0L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0xABCDL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseXorWithAllOnes() {
        // x ^ (-1) = ~x
        Tensor result = Tensor.scalar(0x0FL).bitwiseXor(Tensor.scalar(-1L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(~0x0FL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseI32() {
        // Test with I32 datatype
        Tensor result =
                Tensor.scalar(0xFFL, DataType.I32).bitwiseAnd(Tensor.scalar(0x0FL, DataType.I32));

        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(0x0FL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseChained() {
        // (0xFF & 0xF0) | 0x0F = 0xF0 | 0x0F = 0xFF
        Tensor result =
                Tensor.scalar(0xFFL)
                        .bitwiseAnd(Tensor.scalar(0xF0L))
                        .bitwiseOr(Tensor.scalar(0x0FL));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0xFFL, constant.value().longValue());
    }

    @Test
    void foldsBitwiseNotTwice() {
        // ~~x = x
        Tensor result = Tensor.scalar(0x12345678L).bitwiseNot().bitwiseNot();

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(0x12345678L, constant.value().longValue());
    }

    @Test
    void foldsDeMorgansLaw() {
        // ~(a & b) = ~a | ~b
        long a = 0xF0L;
        long b = 0x0FL;

        Tensor leftSide = Tensor.scalar(a).bitwiseAnd(Tensor.scalar(b)).bitwiseNot();
        Tensor rightSide = Tensor.scalar(a).bitwiseNot().bitwiseOr(Tensor.scalar(b).bitwiseNot());

        ConstantComputation leftConst = getConstant(leftSide);
        ConstantComputation rightConst = getConstant(rightSide);
        assertEquals(leftConst.value().longValue(), rightConst.value().longValue());
    }

    // ========== Comprehensive edge case tests ==========

    @Test
    void foldsLargeIntegerValues() {
        // Test with large values near I64 limits
        long largeValue = Long.MAX_VALUE - 100;
        Tensor result = Tensor.scalar(largeValue).add(Tensor.scalar(50L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(largeValue + 50L, constant.value().longValue());
    }

    @Test
    void foldsNegativeValues() {
        Tensor result = Tensor.scalar(-100L).multiply(Tensor.scalar(-5L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(500L, constant.value().longValue());
    }

    @Test
    void foldsIntegerOverflow() {
        // Java integer overflow behavior
        Tensor result = Tensor.scalar(Long.MAX_VALUE).add(Tensor.scalar(1L));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(Long.MIN_VALUE, constant.value().longValue()); // Overflow wraps around
    }

    @Test
    void foldsFloatSpecialValues() {
        // Test with infinity
        Tensor infResult = Tensor.scalar(1.0).divide(Tensor.scalar(0.0));
        assertTrue(infResult.isLazy());
        assertEquals(Double.POSITIVE_INFINITY, getConstant(infResult).value().doubleValue());

        // Test with negative infinity
        Tensor negInfResult = Tensor.scalar(-1.0).divide(Tensor.scalar(0.0));
        assertTrue(negInfResult.isLazy());
        assertEquals(Double.NEGATIVE_INFINITY, getConstant(negInfResult).value().doubleValue());
    }

    @Test
    void foldsNaNPropagation() {
        // NaN propagates through operations
        Tensor nanResult = Tensor.scalar(0.0).divide(Tensor.scalar(0.0));
        assertTrue(nanResult.isLazy());
        assertTrue(Double.isNaN(getConstant(nanResult).value().doubleValue()));

        // NaN + anything = NaN
        Tensor nanAdd = nanResult.add(Tensor.scalar(5.0));
        assertTrue(Double.isNaN(getConstant(nanAdd).value().doubleValue()));
    }

    @Test
    void foldsComplexExpression() {
        // ((a + b) * c - d) / e
        // ((2 + 3) * 4 - 10) / 2 = (5 * 4 - 10) / 2 = (20 - 10) / 2 = 10 / 2 = 5
        Tensor result =
                Tensor.scalar(2.0f)
                        .add(Tensor.scalar(3.0f))
                        .multiply(Tensor.scalar(4.0f))
                        .subtract(Tensor.scalar(10.0f))
                        .divide(Tensor.scalar(2.0f));

        assertTrue(result.isLazy());
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsComparisonChain() {
        // (a < b) && (b < c) expressed as: a < b, b < c, then logicalAnd (if available)
        // For now, test that individual comparisons fold
        Tensor aLtB = Tensor.scalar(1.0f).lessThan(Tensor.scalar(2.0f));
        Tensor bLtC = Tensor.scalar(2.0f).lessThan(Tensor.scalar(3.0f));

        assertEquals(1L, getConstant(aLtB).value().longValue());
        assertEquals(1L, getConstant(bLtC).value().longValue());
    }

    @Test
    void foldsCastChain() {
        // FP32 -> FP64 -> I64 -> I32
        Tensor result =
                Tensor.scalar(3.7f).cast(DataType.FP64).cast(DataType.I64).cast(DataType.I32);

        assertTrue(result.isLazy());
        assertEquals(DataType.I32, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(3L, constant.value().longValue()); // Truncated from 3.7
    }

    @Test
    void preservesDataTypeInBitwiseOps() {
        Tensor result =
                Tensor.scalar(0xFFL, DataType.I16).bitwiseOr(Tensor.scalar(0x100L, DataType.I16));

        assertTrue(result.isLazy());
        assertEquals(DataType.I16, result.dataType());
    }

    @Test
    void foldsMixedIntAndFloatComparison() {
        // When comparing I32 with FP32, should promote to FP32
        Tensor result = Tensor.scalar(5L, DataType.I8).lessThan(Tensor.scalar(5.5f));

        assertTrue(result.isLazy());
        assertEquals(DataType.BOOL, result.dataType());
        assertEquals(1L, getConstant(result).value().longValue()); // 5 < 5.5 is true
    }
}
