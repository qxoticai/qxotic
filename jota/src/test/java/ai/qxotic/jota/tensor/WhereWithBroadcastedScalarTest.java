package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Regression tests for Tensor.where() with broadcasted scalar inputs.
 *
 * <p>This tests the fix for the issue where where() would not properly update values when inputs
 * are broadcasted scalars (e.g., tensors created with Tensor.zeros() or Tensor.broadcasted()).
 */
class WhereWithBroadcastedScalarTest {

    @SuppressWarnings("unchecked")
    private static final MemoryContext<MemorySegment> CONTEXT =
            (MemoryContext<MemorySegment>) Environment.current().nativeBackend().memoryContext();

    @Test
    void whereWithBroadcastedScalarFalseValue() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.arange(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.greaterThan(Tensor.scalar(2.0f)); // [F,F,F,T,T,T]

        // Use broadcasted scalar as false value
        Tensor trueValue = Tensor.broadcasted(100.0f, shape);
        Tensor falseValue = Tensor.zeros(DataType.FP32, shape); // broadcasted scalar

        Tensor result = Tensor.where(condition, trueValue, falseValue);
        MemoryView<?> view = result.materialize();

        // Expected: [0,0,0,100,100,100]
        assertEquals(0.0f, readFloat(view, 0), 0.001f);
        assertEquals(0.0f, readFloat(view, 1), 0.001f);
        assertEquals(0.0f, readFloat(view, 2), 0.001f);
        assertEquals(100.0f, readFloat(view, 3), 0.001f);
        assertEquals(100.0f, readFloat(view, 4), 0.001f);
        assertEquals(100.0f, readFloat(view, 5), 0.001f);
    }

    @Test
    void whereWithBroadcastedScalarTrueValue() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.arange(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.lessThan(Tensor.scalar(3.0f)); // [T,T,T,F,F,F]

        // Use broadcasted scalar as true value
        Tensor trueValue = Tensor.broadcasted(-1.0f, shape); // broadcasted scalar
        Tensor falseValue = values;

        Tensor result = Tensor.where(condition, trueValue, falseValue);
        MemoryView<?> view = result.materialize();

        // Expected: [-1,-1,-1,3,4,5]
        assertEquals(-1.0f, readFloat(view, 0), 0.001f);
        assertEquals(-1.0f, readFloat(view, 1), 0.001f);
        assertEquals(-1.0f, readFloat(view, 2), 0.001f);
        assertEquals(3.0f, readFloat(view, 3), 0.001f);
        assertEquals(4.0f, readFloat(view, 4), 0.001f);
        assertEquals(5.0f, readFloat(view, 5), 0.001f);
    }

    @Test
    void whereWithBothBroadcastedScalars() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.arange(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.greaterThan(Tensor.scalar(2.5f)); // [F,F,F,T,T,T]

        // Both true and false values are broadcasted scalars
        Tensor trueValue = Tensor.broadcasted(1.0f, shape);
        Tensor falseValue = Tensor.broadcasted(0.0f, shape);

        Tensor result = Tensor.where(condition, trueValue, falseValue);
        MemoryView<?> view = result.materialize();

        // Expected: [0,0,0,1,1,1]
        assertEquals(0.0f, readFloat(view, 0), 0.001f);
        assertEquals(0.0f, readFloat(view, 1), 0.001f);
        assertEquals(0.0f, readFloat(view, 2), 0.001f);
        assertEquals(1.0f, readFloat(view, 3), 0.001f);
        assertEquals(1.0f, readFloat(view, 4), 0.001f);
        assertEquals(1.0f, readFloat(view, 5), 0.001f);
    }

    @Test
    void whereUpdateInLoop() {
        // This simulates the Mandelbrot use case: updating a tensor in a loop
        Shape shape = Shape.of(2, 3);
        Tensor iterations = Tensor.zeros(DataType.FP32, shape); // broadcasted scalar

        // Simulate 3 iterations
        for (int i = 0; i < 3; i++) {
            // Create a condition that becomes true for different elements each iteration
            // i=0: elements 0,1 escape; i=1: element 2 escapes; i=2: elements 3,4,5 escape
            Tensor threshold = Tensor.scalar((float) (i * 2));
            Tensor indices = Tensor.arange(6, DataType.FP32).view(shape);
            Tensor shouldUpdate =
                    indices.greaterThanOrEqual(threshold)
                            .logicalAnd(indices.lessThan(Tensor.scalar((float) (i * 2 + 2))));

            Tensor newValue = Tensor.broadcasted((float) i, shape);
            iterations = Tensor.where(shouldUpdate, newValue, iterations);
        }

        MemoryView<?> view = iterations.materialize();

        // Expected: [0,0,1,1,2,2] - each pair of elements gets the iteration number
        // when they "escaped"
        assertEquals(0.0f, readFloat(view, 0), 0.001f);
        assertEquals(0.0f, readFloat(view, 1), 0.001f);
        assertEquals(1.0f, readFloat(view, 2), 0.001f);
        assertEquals(1.0f, readFloat(view, 3), 0.001f);
        assertEquals(2.0f, readFloat(view, 4), 0.001f);
        assertEquals(2.0f, readFloat(view, 5), 0.001f);
    }

    @Test
    void mandelbrotStyleUpdateLoop() {
        // More closely simulates the Mandelbrot use case
        Shape shape = Shape.of(2, 3);

        // Values that will "escape" at different iterations
        Tensor values = Tensor.of(new float[] {10f, 5f, 3f, 2f, 1f, 0.5f}).view(shape);

        Tensor iterations = Tensor.zeros(DataType.FP32, shape);
        Tensor escaped = Tensor.zeros(DataType.FP32, shape);

        for (int i = 0; i < 5; i++) {
            // Points "escape" when value > threshold (decreasing threshold each iteration)
            float threshold = 8f - i * 2; // 8, 6, 4, 2, 0
            Tensor hasEscaped = values.greaterThan(Tensor.scalar(threshold));
            Tensor notYetEscaped = escaped.lessThan(Tensor.scalar(0.5f));
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            Tensor iterValue = Tensor.broadcasted((float) i, shape);
            iterations = Tensor.where(justEscaped, iterValue, iterations);

            Tensor one = Tensor.broadcasted(1.0f, shape);
            escaped = Tensor.where(justEscaped, one, escaped);

            // Debug: print intermediate state
            MemoryView<?> escView = escaped.materialize();
            int escapedCount = 0;
            for (int j = 0; j < 6; j++) {
                if (readFloat(escView, j) > 0.5f) escapedCount++;
            }
            System.out.println(
                    "Iteration "
                            + i
                            + " (threshold="
                            + threshold
                            + "): escapedCount="
                            + escapedCount);
        }

        MemoryView<?> view = iterations.materialize();
        System.out.println("Final iterations:");
        for (int j = 0; j < 6; j++) {
            System.out.println("  [" + j + "] = " + readFloat(view, j));
        }

        // values = [10, 5, 3, 2, 1, 0.5]
        // threshold progression: 8, 6, 4, 2, 0
        // i=0 (threshold=8): 10 > 8, so element 0 escapes at iteration 0
        // i=1 (threshold=6): 5 < 6, nothing new
        // i=2 (threshold=4): 5 > 4, so element 1 escapes at iteration 2
        // i=3 (threshold=2): 3 > 2, so element 2 escapes at iteration 3
        // i=4 (threshold=0): 2,1,0.5 > 0, so elements 3,4,5 escape at iteration 4
        assertEquals(0.0f, readFloat(view, 0), 0.001f); // escaped at i=0
        assertEquals(2.0f, readFloat(view, 1), 0.001f); // escaped at i=2
        assertEquals(3.0f, readFloat(view, 2), 0.001f); // escaped at i=3
        assertEquals(4.0f, readFloat(view, 3), 0.001f); // escaped at i=4
        assertEquals(4.0f, readFloat(view, 4), 0.001f); // escaped at i=4
        assertEquals(4.0f, readFloat(view, 5), 0.001f); // escaped at i=4
    }

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        MemoryAccess<MemorySegment> access = CONTEXT.memoryAccess();
        return access.readFloat(typedView.memory(), offset);
    }
}
