package ai.qxotic.jota.ir.interpreter;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.memory.AbstractMemoryTest;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.IRTracer;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.Test;

class IRTracerTest {

    private static final MemoryContext<MemorySegment> CONTEXT = ContextFactory.ofMemorySegment();

    private float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        return AbstractMemoryTest.readFloat(CONTEXT.memoryAccess(), typedView, linearIndex);
    }

    @Test
    void testUnaryNegateViaTracer() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor result = IRTracer.trace(input, Tensor::negate);
        MemoryView<?> output = result.materialize();

        assertEquals(-1.0f, readFloat(output, 0), 0.0001f);
        assertEquals(-2.0f, readFloat(output, 1), 0.0001f);
        assertEquals(-3.0f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void testBinaryAddViaTracer() {
        Tensor input1 = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor input2 = Tensor.of(new float[] {4.0f, 5.0f, 6.0f});
        Tensor result =
                IRTracer.trace(
                        List.of(input1, input2), tensors -> tensors.get(0).add(tensors.get(1)));
        MemoryView<?> output = result.materialize();

        assertEquals(5.0f, readFloat(output, 0), 0.0001f);
        assertEquals(7.0f, readFloat(output, 1), 0.0001f);
        assertEquals(9.0f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void testScalarConstantAddViaTracer() {
        Tensor input = Tensor.of(new float[] {1.0f, 2.0f, 3.0f});
        Tensor scalar = Tensor.full(10.0f, input.dataType(), input.shape());
        Tensor result =
                IRTracer.trace(
                        List.of(input, scalar), tensors -> tensors.get(0).add(tensors.get(1)));
        MemoryView<?> output = result.materialize();

        assertEquals(11.0f, readFloat(output, 0), 0.0001f);
        assertEquals(12.0f, readFloat(output, 1), 0.0001f);
        assertEquals(13.0f, readFloat(output, 2), 0.0001f);
    }

    @Test
    void testGeluViaTracer() {
        Tensor input = Tensor.of(new float[] {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f});
        Tensor result = IRTracer.trace(input, Tensor::gelu);
        MemoryView<?> output = result.materialize();

        float delta = 0.001f;
        assertEquals(-0.045402f, readFloat(output, 0), delta);
        assertEquals(-0.158808f, readFloat(output, 1), delta);
        assertEquals(0.0f, readFloat(output, 2), delta);
        assertEquals(0.841192f, readFloat(output, 3), delta);
        assertEquals(1.9545977f, readFloat(output, 4), delta);
    }
}
