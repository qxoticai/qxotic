package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.ExecutionMode;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class ExecutionModeTest {

    private static final MemoryContext<float[]> CONTEXT = ContextFactory.ofFloats();

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

    @Test
    void eagerModeSelectsEagerOps() {
        DeviceRegistry registry = DeviceRegistry.builder().register(CONTEXT, ENGINE).build();
        Environment environment =
                new Environment(CONTEXT.device(), DataType.FP32, registry, ExecutionMode.EAGER);

        Environment.with(
                environment,
                () -> {
                    TensorOps ops = TensorOpsContext.require();
                    assertInstanceOf(EagerTensorOps.class, ops);
                    assertSame(CONTEXT, ops.context());
                    return null;
                });
    }

    @Test
    void lazyModeReturnsLazyTensor() {
        DeviceRegistry registry = DeviceRegistry.builder().register(CONTEXT, ENGINE).build();
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
        MemoryContext<MemorySegment> segmentContext = ContextFactory.ofMemorySegment();
        DeviceRegistry registry = DeviceRegistry.builder().register(segmentContext, ENGINE).build();
        Environment environment =
                new Environment(
                        segmentContext.device(), DataType.FP32, registry, ExecutionMode.EAGER);

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
                            (float) Math.sqrt(Math.PI), readFloat(segmentContext, output), 0.0001f);
                    return null;
                });
    }

    private static MemoryView<float[]> range(Shape shape) {
        return MemoryHelpers.arange(CONTEXT, DataType.FP32, shape.size()).view(shape);
    }

    private static float readFloat(MemoryContext<MemorySegment> context, MemoryView<?> view) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, 0);
        return context.memoryAccess().readFloat(typedView.memory(), offset);
    }
}
