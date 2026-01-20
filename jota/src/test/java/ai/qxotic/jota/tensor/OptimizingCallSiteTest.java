package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.DeviceRegistry;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryHelpers;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class OptimizingCallSiteTest {

    private static MemoryContext<MemorySegment> context;

    @BeforeAll
    static void setUp() {
        context = ContextFactory.ofMemorySegment();
        try {
            DeviceRegistry.engine(context.device());
        } catch (IllegalStateException ignored) {
            DeviceRegistry.register(context.device(), context, new JavaComputeEngine(context));
        }
    }

    @Test
    void returnsTracedTensorWhenInputIsTraced() {
        OptimizingCallSite callSite = Tracer.createOptimizingCallSite(t -> t.add(1));
        TraceTensor tracedInput = new TraceTensor(
                new InputNode(0, DataType.FP32, Layout.rowMajor(4), context.device()));

        Tensor result = callSite.apply(tracedInput);

        assertInstanceOf(TraceTensor.class, result);
        assertTrue(result.isLazy());
        assertFalse(result.isMaterialized());
    }

    @Test
    void returnsMaterializedTensorWhenInputIsEager() {
        OptimizingCallSite add1 = Tracer.createOptimizingCallSite(t -> t.add(1));
        MemoryView<MemorySegment> view =
                MemoryHelpers.arange(context, DataType.FP32, 4).view(Shape.of(4));
        Tensor input = Tensor.of(view);
        Tensor result = add1.apply(input);

        assertTrue(result.isMaterialized());
        assertFalse(result.isLazy());
        MemoryView<?> output = result.materialize();
        assertEquals(1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(2.0f, readFloat(output, 1), 0.0001f);
        assertEquals(3.0f, readFloat(output, 2), 0.0001f);
        assertEquals(4.0f, readFloat(output, 3), 0.0001f);
    }

    private float readFloat(MemoryView<?> view, long index) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> castView = (MemoryView<MemorySegment>) view;
        long offset = castView.byteOffset() + index * DataType.FP32.byteSize();
        return context.memoryAccess().readFloat(castView.memory(), offset);
    }
}
