package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryViewFactory;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

class JavaKernelCompilerTest {

    @AutoClose
    private final MemoryContext<MemorySegment> context = ContextFactory.ofMemorySegment();

    private static float gelu(float value) {
        float cubic = value * value * value;
        float inner = (cubic * 0.044715f + value) * 0.79788456f;
        return ((float) Math.tanh(inner) + 1.0f) * value * 0.5f;
    }

    private static Tensor tensorGelu(Tensor value) {
        Tensor cubic = value.multiply(value).multiply(value);
        Tensor inner = cubic.multiply(0.044715f).add(value).multiply(0.79788456f);
        return inner.tanh().add(1f).multiply(value).multiply(0.5f);
    }

    @FunctionalInterface
    interface FloatUnaryOperator {
        float applyAsFloat(float value);
    }

    static float[] map(float[] in, FloatUnaryOperator mapper) {
        float[] out = new float[in.length];
        for (int i = 0; i < in.length; i++) {
            out[i] = mapper.applyAsFloat(in[i]);
        }
        return out;
    }

    @Test
    void compilesAndRunsNonTrivialKernel() {
        MemoryView<MemorySegment> input = range(Shape.of(2, 3));
        Tensor inputTensor = Tensor.of(input);
        Tensor traced = Tracer.trace(inputTensor, JavaKernelCompilerTest::tensorGelu);

        MemoryView<?> output = traced.materialize();

        float[] values = readFloatValues(output, 6);
        float[] expected = map(new float[] {0, 1, 2, 3, 4, 5}, JavaKernelCompilerTest::gelu);
        assertArrayEquals(expected, values, 0.0001f);
    }

    static float[] pepe(float[] in, float[] out) {
        for (int i = 0; i < 100_000_000; ++i) {
            out[i] = gelu(in[i]);
        }
        return out;
    }

    @Disabled
    @Test
    void benchmark() {
        MemoryView<MemorySegment> input = range(Shape.of(100, 1_000_000));
        Tensor inputTensor = Tensor.of(input);

        //        float[] in = new float[100_000_000];
        //        for (int i = 0; i < in.length; ++i) {
        //            in[i] = i;
        //        }

        for (int i = 0; i < 10; ++i) {
            Tensor traced = Tracer.trace(inputTensor, JavaKernelCompilerTest::tensorGelu);
            long startNanos = System.nanoTime();
            //            float[] out = new float[100_000_000];
            MemoryView<?> output = traced.materialize();
            //          pepe(in, out);
            long elapsedNanos = System.nanoTime() - startNanos;
            long elapsedMillis = TimeUnit.NANOSECONDS.toMillis(elapsedNanos);
            System.out.println("Kernel execution time: " + elapsedMillis + " ms");

            float[] values =
                    //                Arrays.copyOf(out, 6); //
                    readFloatValues(output, 6);
            float[] expected = map(new float[] {0, 1, 2, 3, 4, 5}, JavaKernelCompilerTest::gelu);
            assertArrayEquals(expected, values, 0.0001f);
        }
    }

    @Test
    void canary() {
        Tensor tensor0 = Tensor.of(range(Shape.of(2, 3)));
        Tensor tensor1 = Tensor.of(range(Shape.of(2, 3)));
        Tensor traced = Tracer.trace(tensor0, tensor1, (t0, t1) -> t0.multiply(t1));

        MemoryView<?> output = traced.materialize();

        float[] values = readFloatValues(output, 6);
        assertArrayEquals(new float[] {0, 1, 4, 9, 16, 25}, values, 0.0001f);
    }

    @Test
    void compilesAndRunsContiguousKernel() {
        MemoryView<MemorySegment> input = range(Shape.of(2, 3));
        Tensor inputTensor = Tensor.of(input);
        Tensor traced = Tracer.trace(inputTensor, t -> t.add(t));

        MemoryView<?> output = traced.materialize();

        float[] values = readFloatValues(output, 6);
        assertArrayEquals(new float[] {0, 2, 4, 6, 8, 10}, values, 0.0001f);
    }

    @Test
    void compilesAndRunsStridedKernel() {
        MemoryView<MemorySegment> input = range(Shape.of(2, 3)).transpose(0, 1);
        Tensor inputTensor = Tensor.of(input);
        Tensor traced = Tracer.trace(inputTensor, t -> t.square().add(1f));

        MemoryView<?> output = traced.materialize();

        float[] values = readFloatValues(output, 6);
        assertArrayEquals(new float[] {1, 10, 2, 17, 5, 26}, values, 0.0001f);
        assertEquals(Layout.rowMajor(output.shape()), output.layout());
    }

    @Test
    void compilesAndRunsIntKernel() {
        MemoryView<MemorySegment> input = rangeInt(Shape.of(2, 2));
        Tensor inputTensor = Tensor.of(input);
        Tensor traced = Tracer.trace(inputTensor, t -> t.add(3).square());

        MemoryView<?> output = traced.materialize();

        int[] values = readIntValues(output, 4);
        assertArrayEquals(new int[] {9, 16, 25, 36}, values);
    }

    @Test
    void kernelCacheIsPersisted() {
        MemoryView<MemorySegment> input = range(Shape.of(1, 4));
        Tensor traced = Tracer.trace(Tensor.of(input), t -> t.add(t));

        MemoryView<?> first = traced.materialize();
        assertNotNull(first);

        assertTrue(traced.computation().isPresent());

        MemoryView<?> second = traced.materialize();
        assertNotNull(second);
    }

    @Test
    void compilesAndRunsTriInputKernel() {
        MemoryView<MemorySegment> input0 = range(Shape.of(2, 2));
        MemoryView<MemorySegment> input1 = range(Shape.of(2, 2));
        MemoryView<MemorySegment> input2 = range(Shape.of(2, 2));
        Tensor traced =
                Tracer.trace(
                        Tensor.of(input0),
                        Tensor.of(input1),
                        Tensor.of(input2),
                        (a, b, c) -> a.add(b).multiply(c));

        MemoryView<?> output = traced.materialize();

        float[] values = readFloatValues(output, 4);
        assertArrayEquals(new float[] {0, 2, 8, 18}, values, 0.0001f);
    }

    private MemoryView<MemorySegment> range(Shape shape) {
        int size = Math.toIntExact(shape.size());
        var memory = context.memoryAllocator().allocateMemory(DataType.FP32, shape);
        MemorySegment segment = memory.base();
        for (int i = 0; i < size; i++) {
            long offset = (long) i * DataType.FP32.byteSize();
            segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, offset, (float) i);
        }
        return MemoryViewFactory.of(DataType.FP32, memory, ai.qxotic.jota.Layout.rowMajor(shape));
    }

    private MemoryView<MemorySegment> rangeInt(Shape shape) {
        int size = Math.toIntExact(shape.size());
        var memory = context.memoryAllocator().allocateMemory(DataType.I32, shape);
        MemorySegment segment = memory.base();
        for (int i = 0; i < size; i++) {
            long offset = (long) i * DataType.I32.byteSize();
            segment.set(ValueLayout.JAVA_INT_UNALIGNED, offset, i);
        }
        return MemoryViewFactory.of(DataType.I32, memory, ai.qxotic.jota.Layout.rowMajor(shape));
    }

    private float[] readFloatValues(MemoryView<?> view, int count) {
        float[] values = new float[count];
        MemorySegment segment = (MemorySegment) view.memory().base();
        long base = view.byteOffset();
        long[] shape = view.shape().toArray();
        long[] stride = view.byteStride().toArray();
        for (int i = 0; i < count; i++) {
            long offset = offsetForIndex(i, base, shape, stride);
            values[i] = segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, offset);
        }
        return values;
    }

    private int[] readIntValues(MemoryView<?> view, int count) {
        int[] values = new int[count];
        MemorySegment segment = (MemorySegment) view.memory().base();
        for (int i = 0; i < count; i++) {
            long offset = Indexing.linearToOffset(view, i);
            values[i] = segment.get(ValueLayout.JAVA_INT_UNALIGNED, offset);
        }
        return values;
    }

    private long offsetForIndex(int index, long baseOffset, long[] shape, long[] stride) {
        long offset = baseOffset;
        long remaining = index;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            long size = shape[dim];
            if (size == 0) {
                return baseOffset;
            }
            long coord = remaining % size;
            remaining /= size;
            offset += coord * stride[dim];
        }
        return offset;
    }
}
