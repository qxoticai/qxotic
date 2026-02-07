package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.DomainFactory;
import ai.qxotic.jota.runtime.DefaultRuntimeRegistry;
import ai.qxotic.jota.runtime.DeviceRuntime;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ExecutionModeTest {

    private static final MemoryDomain<float[]> CONTEXT = DomainFactory.ofFloats();

    private static final ComputeEngine COMPUTE_BACKEND =
            new ComputeEngine() {
                @Override
                public Device device() {
                    return Device.PANAMA;
                }

                @Override
                public MemoryView<?> execute(
                        ai.qxotic.jota.ir.tir.TIRGraph graph, java.util.List<Tensor> inputs) {
                    throw new UnsupportedOperationException("No backend execution in this test");
                }
            };

    @Test
    void eagerModeSelectsEagerOps() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.register(new StubDeviceRuntime(CONTEXT, COMPUTE_BACKEND));
        Environment environment =
                new Environment(CONTEXT.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    TensorOps ops = TensorOpsContext.require();
                    assertInstanceOf(EagerTensorOps.class, ops);
                    assertSame(CONTEXT, ops.memoryDomain());
                    return null;
                });
    }

    @Test
    void lazyModeReturnsLazyTensor() {
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.register(new StubDeviceRuntime(CONTEXT, COMPUTE_BACKEND));
        Environment environment =
                new Environment(Device.PANAMA, DataType.FP32, registry, ExecutionMode.LAZY);

        Environment.with(
                environment,
                () -> {
                    Tensor input = Tensor.of(range(Shape.of(2)));
                    Tensor output = input.add(input);
                    assertTrue(output.isLazy());
                    assertFalse(output.isMaterialized());
                    assertTrue(output.computation().isPresent());
                    assertInstanceOf(LazyTensorOps.class, TensorOpsContext.require());
                    return null;
                });
    }

    @Test
    void eagerOpsMaterializeScalarSqrt() {
        MemoryDomain<MemorySegment> segmentDomain = DomainFactory.ofMemorySegment();
        DefaultRuntimeRegistry registry = new DefaultRuntimeRegistry();
        registry.register(new StubDeviceRuntime(segmentDomain, COMPUTE_BACKEND));
        Environment environment =
                new Environment(
                        segmentDomain.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    TensorOps ops = TensorOpsContext.require();
                    assertInstanceOf(EagerTensorOps.class, ops);
                    Tensor result =
                            TensorOpsContext.with(ops, () -> Tensor.scalar((float) Math.PI).sqrt());

                    assertTrue(result.isLazy());
                    assertFalse(result.isMaterialized());

                    MemoryView<?> output = result.materialize();
                    assertEquals(DataType.FP32, output.dataType());
                    assertEquals(Shape.scalar(), output.shape());
                    assertEquals(
                            (float) Math.sqrt(Math.PI), readFloat(segmentDomain, output), 0.0001f);
                    return null;
                });
    }

    private static MemoryView<float[]> range(Shape shape) {
        return MemoryHelpers.arange(CONTEXT, DataType.FP32, shape.size()).view(shape);
    }

    private static float readFloat(MemoryDomain<MemorySegment> domain, MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, 0);
        return domain.directAccess().readFloat(typedView.memory(), offset);
    }

    private static final class StubDeviceRuntime implements DeviceRuntime {
        private final MemoryDomain<?> memoryDomain;
        private final ComputeEngine computeEngine;

        private StubDeviceRuntime(MemoryDomain<?> memoryDomain, ComputeEngine computeEngine) {
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
        public java.util.Optional<ai.qxotic.jota.runtime.KernelService> kernelService() {
            return java.util.Optional.empty();
        }
    }
}
