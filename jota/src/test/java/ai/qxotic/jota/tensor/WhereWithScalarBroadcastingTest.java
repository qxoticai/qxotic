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
 * Tests for Tensor.where() with scalar broadcasting.
 *
 * <p>Tensor.where() broadcasts scalar values to match the shape of other inputs. Only true scalar
 * tensors (where shape.isScalar() == true) can be broadcast. Broadcasted tensors (with non-scalar
 * shapes and zero strides) are NOT automatically broadcast.
 *
 * <p>Use Tensor.scalar(value) to create scalar values that will broadcast.
 */
class WhereWithScalarBroadcastingTest {

    @SuppressWarnings("unchecked")
    private static final MemoryContext<MemorySegment> CONTEXT =
            (MemoryContext<MemorySegment>) Environment.current().nativeBackend().memoryContext();

    @Test
    void whereWithScalarFalseValue() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.iota(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.greaterThan(Tensor.scalar(2.0f)); // [F,F,F,T,T,T]

        // Use true scalar as false value (broadcasted to shape)
        Tensor trueValue = Tensor.scalar(100.0f);
        Tensor falseValue = Tensor.scalar(0.0f); // scalar that will be broadcast

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
    void whereWithScalarTrueValue() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.iota(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.lessThan(Tensor.scalar(3.0f)); // [T,T,T,F,F,F]

        // Use true scalar as true value (broadcasted to shape)
        Tensor trueValue = Tensor.scalar(-1.0f); // scalar that will be broadcast
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
    void whereWithBothScalars() {
        // Create a condition where some elements are true
        Shape shape = Shape.of(2, 3);
        Tensor values = Tensor.iota(6, DataType.FP32).view(shape); // [0,1,2,3,4,5]
        Tensor condition = values.greaterThan(Tensor.scalar(2.5f)); // [F,F,F,T,T,T]

        // Both true and false values are true scalars (will be broadcast)
        Tensor trueValue = Tensor.scalar(1.0f);
        Tensor falseValue = Tensor.scalar(0.0f);

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
        // Initialize with a real tensor (not a broadcasted scalar)
        Tensor iterations = Tensor.of(new float[6]).view(shape); // [0,0,0,0,0,0]

        // Simulate 3 iterations
        for (int i = 0; i < 3; i++) {
            // Create a condition that becomes true for different elements each iteration
            // i=0: elements 0,1 escape; i=1: element 2 escapes; i=2: elements 3,4,5 escape
            Tensor threshold = Tensor.scalar((float) (i * 2));
            Tensor indices = Tensor.iota(6, DataType.FP32).view(shape);
            Tensor shouldUpdate =
                    indices.greaterThanOrEqual(threshold)
                            .logicalAnd(indices.lessThan(Tensor.scalar((float) (i * 2 + 2))));

            // Use scalar for new value (will be broadcast)
            Tensor newValue = Tensor.scalar((float) i);
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

        // Initialize with real tensors (not broadcasted scalars)
        Tensor iterations = Tensor.of(new float[6]).view(shape); // [0,0,0,0,0,0]
        Tensor escaped = Tensor.of(new float[6]).view(shape); // [0,0,0,0,0,0]

        for (int i = 0; i < 5; i++) {
            // Points "escape" when value > threshold (decreasing threshold each iteration)
            float threshold = 8f - i * 2; // 8, 6, 4, 2, 0
            Tensor hasEscaped = values.greaterThan(Tensor.scalar(threshold));
            Tensor notYetEscaped = escaped.lessThan(Tensor.scalar(0.5f));
            Tensor justEscaped = hasEscaped.logicalAnd(notYetEscaped);

            // Use scalars for values (will be broadcast)
            Tensor iterValue = Tensor.scalar((float) i);
            iterations = Tensor.where(justEscaped, iterValue, iterations);

            Tensor one = Tensor.scalar(1.0f);
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
