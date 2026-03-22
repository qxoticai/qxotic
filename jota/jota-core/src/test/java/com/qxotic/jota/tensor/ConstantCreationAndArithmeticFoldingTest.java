package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.MemoryViewPrinter;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DefaultRuntimeRegistry;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import com.qxotic.jota.testutil.RunOnAllAvailableBackends;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

@RunOnAllAvailableBackends
class ConstantCreationAndArithmeticFoldingTest {

    private static ComputeEngine dummyBackend() {
        return new ComputeEngine() {
            @Override
            public Device device() {
                return DeviceType.PANAMA.deviceIndex(0);
            }

            @Override
            public MemoryView<?> execute(TIRGraph graph, List<Tensor> inputs) {
                throw new UnsupportedOperationException("No backend execution in this test");
            }
        };
    }

    @Test
    void broadcastedFloatStaysLazy() {
        Tensor tensor = Tensor.full(3.5f, Shape.of(2, 3));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertFalse(TensorTestInternals.isMaterialized(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void materializesBroadcastedFloat(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.full(1.25f, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertEquals(DataType.FP32, view.dataType());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = domain.directAccess();
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
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingI64")
    <B> void materializesBroadcastedLong(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.full(42L, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertEquals(DataType.I64, view.dataType());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = domain.directAccess();
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

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertFalse(TensorTestInternals.isMaterialized(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(1L, tensor.size());
        assertEquals(0, tensor.stride().flatRank());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF64")
    <B> void materializesScalarDouble(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

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

                    MemoryAccess<B> access = domain.directAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(2.125, access.readDouble(typedView.memory(), offset), 0.000001);
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingI64")
    <B> void materializesScalarLong(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

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

                    MemoryAccess<B> access = domain.directAccess();
                    @SuppressWarnings("unchecked")
                    MemoryView<B> typedView = (MemoryView<B>) view;
                    long offset = Indexing.linearToOffset(view, 0);
                    assertEquals(42L, access.readLong(typedView.memory(), offset));
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#allDomains")
    <B> void scalarUsesEnvironmentDefaultDevice(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.scalar(1.0f);
                    assertEquals(domain.device(), tensor.device());
                    return null;
                });
    }

    // ========== zeros tests ==========

    @Test
    void zerosUsesDefaultFloat() {
        Tensor tensor = Tensor.zeros(Shape.of(2, 3));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.defaultFloat(), tensor.dataType());
        assertEquals(Shape.of(2, 3), tensor.shape());
    }

    @Test
    void zerosWithExplicitDtype() {
        Tensor tensor = Tensor.zeros(DataType.I32, Shape.of(4, 5));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.I32, tensor.dataType());
        assertEquals(Shape.of(4, 5), tensor.shape());
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void materializesZeros(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.zeros(DataType.FP32, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = domain.directAccess();
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

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.defaultFloat(), tensor.dataType());
        assertEquals(Shape.of(3, 4), tensor.shape());
    }

    @Test
    void onesWithExplicitDtype() {
        Tensor tensor = Tensor.ones(DataType.I64, Shape.of(2, 2));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.I64, tensor.dataType());
        assertEquals(Shape.of(2, 2), tensor.shape());
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void materializesOnes(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.ones(DataType.FP32, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = domain.directAccess();
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

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @Test
    void fullDoubleInfersDtype() {
        Tensor tensor = Tensor.full(2.718, Shape.of(3, 3));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.FP64, tensor.dataType());
    }

    @Test
    void fullLongInfersDtype() {
        Tensor tensor = Tensor.full(42L, Shape.of(4, 4));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.I64, tensor.dataType());
    }

    @Test
    void fullIntInfersDtype() {
        Tensor tensor = Tensor.full(7, Shape.of(5, 5));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.I32, tensor.dataType());
    }

    @Test
    void fullWithExplicitDtype() {
        Tensor tensor = Tensor.full(Integer.valueOf(99), DataType.I16, Shape.of(2, 3));

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(DataType.I16, tensor.dataType());
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void materializesFull(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    Tensor tensor = Tensor.full(7.5f, Shape.of(2, 3));
                    MemoryView<?> view = tensor.materialize();

                    assertEquals(Shape.of(2, 3), view.shape());
                    assertTrue(view.isBroadcasted());

                    MemoryAccess<B> access = domain.directAccess();
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
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void ofFloatArray(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    float[] data = {1.0f, 2.0f, 3.0f, 4.0f};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(TensorTestInternals.isMaterialized(tensor));
                    assertEquals(DataType.FP32, tensor.dataType());
                    assertEquals(Shape.flat(4), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = domain.directAccess();
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
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void ofFloatArrayWithShape(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    float[] data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
                    Tensor tensor = Tensor.of(data, Shape.of(2, 3));

                    assertTrue(TensorTestInternals.isMaterialized(tensor));
                    assertEquals(Shape.of(2, 3), tensor.shape());
                    return null;
                });
    }

    @ParameterizedTest
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF64")
    <B> void ofDoubleArray(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    double[] data = {1.0, 2.0, 3.0};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(TensorTestInternals.isMaterialized(tensor));
                    assertEquals(DataType.FP64, tensor.dataType());
                    assertEquals(Shape.flat(3), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = domain.directAccess();
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
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingI32")
    <B> void ofIntArray(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    int[] data = {10, 20, 30, 40};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(TensorTestInternals.isMaterialized(tensor));
                    assertEquals(DataType.I32, tensor.dataType());
                    assertEquals(Shape.flat(4), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = domain.directAccess();
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
    @MethodSource("com.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingI64")
    <B> void ofLongArray(MemoryDomain<B> domain) {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.registerFactory(
                domain.device(), d -> new StubDeviceRuntime<>(domain, dummyBackend()));
        Environment environment = Environment.of(domain.device(), DataType.FP32, registry);

        Environment.with(
                environment,
                () -> {
                    long[] data = {100L, 200L, 300L};
                    Tensor tensor = Tensor.of(data);

                    assertTrue(TensorTestInternals.isMaterialized(tensor));
                    assertEquals(DataType.I64, tensor.dataType());
                    assertEquals(Shape.flat(3), tensor.shape());

                    MemoryView<?> view = tensor.materialize();
                    MemoryAccess<B> access = domain.directAccess();
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
        Tensor scalar = Tensor.scalar(3.14f);
        Tensor reciprocal = scalar.reciprocal();
        MemoryAccess access = Environment.nativeMemoryDomain().directAccess();
        MemoryView<?> printable =
                reciprocal.to(Environment.nativeRuntime().device()).materialize();
        System.out.println(MemoryViewPrinter.toString(printable, access));
    }

    // ========== Typed scalar tests (carrier methods) ==========

    @Test
    void scalarWithDoubleCarrierCreatesFP16() {
        Tensor tensor = Tensor.scalar(3.14, DataType.FP16);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP16, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesBF16() {
        Tensor tensor = Tensor.scalar(2.718, DataType.BF16);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.BF16, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesFP32() {
        Tensor tensor = Tensor.scalar(1.5, DataType.FP32);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP32, tensor.dataType());
    }

    @Test
    void scalarWithDoubleCarrierCreatesFP64() {
        Tensor tensor = Tensor.scalar(1.5, DataType.FP64);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.FP64, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI8() {
        Tensor tensor = Tensor.scalar(42L, DataType.I8);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I8, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI16() {
        Tensor tensor = Tensor.scalar(1000L, DataType.I16);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I16, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI32() {
        Tensor tensor = Tensor.scalar(100000L, DataType.I32);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I32, tensor.dataType());
    }

    @Test
    void scalarWithLongCarrierCreatesI64() {
        Tensor tensor = Tensor.scalar(9999999999L, DataType.I64);

        assertTrue(TensorTestInternals.isLazy(tensor));
        assertTrue(TensorTestInternals.isScalarBroadcast(tensor));
        assertEquals(Shape.scalar(), tensor.shape());
        assertEquals(DataType.I64, tensor.dataType());
    }

    // ========== Constant folding tests (binary ops) ==========

    private static ConstantComputation getConstant(Tensor tensor) {
        return TensorTestInternals.computation(tensor)
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast)
                .orElseThrow(() -> new AssertionError("Expected ConstantComputation"));
    }

    private static final class StubDeviceRuntime<B> implements DeviceRuntime {
        private final MemoryDomain<B> memoryDomain;
        private final ComputeEngine computeEngine;

        private StubDeviceRuntime(MemoryDomain<B> memoryDomain, ComputeEngine computeEngine) {
            this.memoryDomain = memoryDomain;
            this.computeEngine = computeEngine;
        }

        @Override
        public Device device() {
            return memoryDomain.device();
        }

        @Override
        public MemoryDomain<?> memoryDomain() {
            return memoryDomain;
        }

        @Override
        public ComputeEngine computeEngine() {
            return computeEngine;
        }

        @Override
        public Optional<KernelService> kernelService() {
            return Optional.empty();
        }
    }

    @Test
    void foldsBinaryAdd() {
        Tensor result = Tensor.scalar(2.0f).add(3.0f);

        assertTrue(TensorTestInternals.isLazy(result));
        assertTrue(TensorTestInternals.isScalarBroadcast(result));
        assertEquals(Shape.scalar(), result.shape());
        assertEquals(DataType.FP32, result.dataType());

        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinarySubtract() {
        Tensor result = Tensor.scalar(10.0f).subtract(3.0f);

        assertTrue(TensorTestInternals.isLazy(result));
        ConstantComputation constant = getConstant(result);
        assertEquals(7.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryMultiply() {
        Tensor result = Tensor.scalar(4.0f).multiply(5.0f);

        assertTrue(TensorTestInternals.isLazy(result));
        ConstantComputation constant = getConstant(result);
        assertEquals(20.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryDivide() {
        Tensor result = Tensor.scalar(15.0f).divide(3.0f);

        assertTrue(TensorTestInternals.isLazy(result));
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryWithTypePromotion() {
        Tensor result = Tensor.scalar(2.0f).add(Tensor.scalar(3.0)); // FP32 + FP64 -> FP64

        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP64, result.dataType());
        ConstantComputation constant = getConstant(result);
        assertEquals(5.0, constant.value().doubleValue(), 0.0001);
    }

    @Test
    void foldsChainedBinaryOps() {
        // (2 + 3) * 4 = 20
        Tensor result = Tensor.scalar(2.0f).add(3.0f).multiply(4.0f);

        assertTrue(TensorTestInternals.isLazy(result));
        ConstantComputation constant = getConstant(result);
        assertEquals(20.0, constant.value().doubleValue(), 0.0001);
    }

    // ========== Constant folding tests (unary ops) ==========

    @Test
    void foldsUnaryNegate() {
        Tensor result = Tensor.scalar(5.0f).negate();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(-5.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryAbs() {
        Tensor result = Tensor.scalar(-7.0f).abs();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(7.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryExp() {
        Tensor result = Tensor.scalar(1.0f).exp();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(Math.E, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryLog() {
        Tensor result = Tensor.scalar((float) Math.E).log();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(1.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySqrt() {
        Tensor result = Tensor.scalar(16.0f).sqrt();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(4.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryRsqrt() {
        Tensor result = Tensor.scalar(16.0f).rsqrt();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.25, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySin() {
        Tensor result = Tensor.scalar(0.0f).sin();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryCos() {
        Tensor result = Tensor.scalar(0.0f).cos();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(1.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryTanh() {
        Tensor result = Tensor.scalar(0.0f).tanh();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryReciprocal() {
        Tensor result = Tensor.scalar(4.0f).reciprocal();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.25, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsClamp() {
        // Test clip with scalar bounds
        Tensor result = Tensor.scalar(10.0f).clip(0.0, 5.0);
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(5.0, getConstant(result).value().doubleValue(), 0.0001);

        // Test value below min
        result = Tensor.scalar(-3.0f).clip(0.0, 5.0);
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);

        // Test value within range
        result = Tensor.scalar(3.0f).clip(0.0, 5.0);
        assertEquals(3.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnarySigmoid() {
        Tensor result = Tensor.scalar(0.0f).sigmoid();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.5, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsChainedUnaryOps() {
        Tensor result = Tensor.scalar(-5.0f).abs().negate();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(-5.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsUnaryPreservesDataType() {
        Tensor result = Tensor.scalar(4.0, DataType.FP32).negate();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void foldsMixedUnaryAndBinaryOps() {
        Tensor result = Tensor.scalar(2.0f).negate().add(5.0f);
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(3.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    // ========== min/max constant folding tests ==========

    @Test
    void foldsBinaryMin() {
        Tensor result = Tensor.scalar(7.0f).min(Tensor.scalar(3.0f));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(3.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsBinaryMax() {
        Tensor result = Tensor.scalar(7.0f).max(Tensor.scalar(3.0f));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(7.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    // ========== Activation functions constant folding tests ==========

    @Test
    void foldsReluPositive() {
        Tensor result = Tensor.scalar(5.0f).relu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(5.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsReluNegative() {
        Tensor result = Tensor.scalar(-3.0f).relu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsReluZero() {
        Tensor result = Tensor.scalar(0.0f).relu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsSilu() {
        Tensor result = Tensor.scalar(0.0f).silu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsSiluPositive() {
        Tensor result = Tensor.scalar(1.0f).silu();
        assertTrue(TensorTestInternals.isLazy(result));
        double expected = 1.0 / (1.0 + Math.exp(-1.0));
        assertEquals(expected, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsGelu() {
        Tensor result = Tensor.scalar(0.0f).gelu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(0.0, getConstant(result).value().doubleValue(), 0.0001);
    }

    @Test
    void foldsGeluPositive() {
        Tensor result = Tensor.scalar(1.0f).gelu();
        assertTrue(TensorTestInternals.isLazy(result));
        double x = 1.0;
        double inner = Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x);
        double expected = 0.5 * x * (1 + Math.tanh(inner));
        assertEquals(expected, getConstant(result).value().doubleValue(), 0.01);
    }

    // ========== Integer constant folding tests ==========

    @Test
    void foldsIntegerAdd() {
        Tensor result = Tensor.scalar(100L).add(Tensor.scalar(200L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.I64, result.dataType());
        assertEquals(300L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerMultiply() {
        Tensor result = Tensor.scalar(12L).multiply(Tensor.scalar(11L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(132L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerNegate() {
        Tensor result = Tensor.scalar(42L).negate();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(-42L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerAbs() {
        Tensor result = Tensor.scalar(-99L).abs();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(99L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerMinMax() {
        Tensor minResult = Tensor.scalar(50L).min(Tensor.scalar(30L));
        Tensor maxResult = Tensor.scalar(50L).max(Tensor.scalar(30L));
        assertEquals(30L, getConstant(minResult).value().longValue());
        assertEquals(50L, getConstant(maxResult).value().longValue());
    }

    @Test
    void foldsIntegerDivision() {
        Tensor result = Tensor.scalar(17L).divide(Tensor.scalar(5L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(3L, getConstant(result).value().longValue());
    }

    @Test
    void preservesI32DataType() {
        Tensor result = Tensor.scalar(10L, DataType.I32).add(Tensor.scalar(5L, DataType.I32));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.I32, result.dataType());
        assertEquals(15L, getConstant(result).value().longValue());
    }

    // ========== Activation functions preserve dataType ==========

    @Test
    void sigmoidPreservesDataType() {
        Tensor result = Tensor.scalar(0.0, DataType.FP32).sigmoid();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void reluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).relu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void siluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).silu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP32, result.dataType());
    }

    @Test
    void geluPreservesDataType() {
        Tensor result = Tensor.scalar(1.0, DataType.FP32).gelu();
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(DataType.FP32, result.dataType());
    }

    // ========== Comprehensive edge case tests ==========

    @Test
    void foldsLargeIntegerValues() {
        long largeValue = Long.MAX_VALUE - 100;
        Tensor result = Tensor.scalar(largeValue).add(Tensor.scalar(50L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(largeValue + 50L, getConstant(result).value().longValue());
    }

    @Test
    void foldsNegativeValues() {
        Tensor result = Tensor.scalar(-100L).multiply(Tensor.scalar(-5L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(500L, getConstant(result).value().longValue());
    }

    @Test
    void foldsIntegerOverflow() {
        Tensor result = Tensor.scalar(Long.MAX_VALUE).add(Tensor.scalar(1L));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(Long.MIN_VALUE, getConstant(result).value().longValue());
    }

    @Test
    void foldsFloatSpecialValues() {
        Tensor infResult = Tensor.scalar(1.0).divide(Tensor.scalar(0.0));
        assertTrue(TensorTestInternals.isLazy(infResult));
        assertEquals(Double.POSITIVE_INFINITY, getConstant(infResult).value().doubleValue());

        Tensor negInfResult = Tensor.scalar(-1.0).divide(Tensor.scalar(0.0));
        assertTrue(TensorTestInternals.isLazy(negInfResult));
        assertEquals(Double.NEGATIVE_INFINITY, getConstant(negInfResult).value().doubleValue());
    }

    @Test
    void foldsNaNPropagation() {
        Tensor nanResult = Tensor.scalar(0.0).divide(Tensor.scalar(0.0));
        assertTrue(TensorTestInternals.isLazy(nanResult));
        assertTrue(Double.isNaN(getConstant(nanResult).value().doubleValue()));

        Tensor nanAdd = nanResult.add(Tensor.scalar(5.0));
        assertTrue(Double.isNaN(getConstant(nanAdd).value().doubleValue()));
    }

    @Test
    void foldsComplexExpression() {
        Tensor result =
                Tensor.scalar(2.0f)
                        .add(Tensor.scalar(3.0f))
                        .multiply(Tensor.scalar(4.0f))
                        .subtract(Tensor.scalar(10.0f))
                        .divide(Tensor.scalar(2.0f));
        assertTrue(TensorTestInternals.isLazy(result));
        assertEquals(5.0, getConstant(result).value().doubleValue(), 0.0001);
    }
}
