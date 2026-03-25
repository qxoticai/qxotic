package com.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.testutil.TensorTestReads;
import com.qxotic.jota.testutil.TestKernels;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import org.junit.jupiter.api.Test;

class TraceFunctionTest {

    @Test
    void singleInputDynamic() {
        Tensor sample = Tensor.of(new float[] {1, 2, 3});
        Function<Tensor, Tensor> square = Tracer.traceFunction(sample, t -> t.multiply(t));

        Tensor r1 = square.apply(Tensor.of(new float[] {2, 3, 4}));
        assertEquals(4f, TensorTestReads.readFloat(r1, 0), 1e-4f);
        assertEquals(9f, TensorTestReads.readFloat(r1, 1), 1e-4f);
        assertEquals(16f, TensorTestReads.readFloat(r1, 2), 1e-4f);

        Tensor r2 = square.apply(Tensor.of(new float[] {5, 6, 7}));
        assertEquals(25f, TensorTestReads.readFloat(r2, 0), 1e-4f);
        assertEquals(36f, TensorTestReads.readFloat(r2, 1), 1e-4f);
        assertEquals(49f, TensorTestReads.readFloat(r2, 2), 1e-4f);
    }

    @Test
    void biInputDynamic() {
        Tensor a = Tensor.of(new float[] {1, 2});
        Tensor b = Tensor.of(new float[] {3, 4});
        BiFunction<Tensor, Tensor, Tensor> add = Tracer.traceFunction(a, b, Tensor::add);

        Tensor r1 = add.apply(Tensor.of(new float[] {10, 20}), Tensor.of(new float[] {1, 2}));
        assertEquals(11f, TensorTestReads.readFloat(r1, 0), 1e-4f);
        assertEquals(22f, TensorTestReads.readFloat(r1, 1), 1e-4f);

        Tensor r2 = add.apply(Tensor.of(new float[] {100, 200}), Tensor.of(new float[] {5, 6}));
        assertEquals(105f, TensorTestReads.readFloat(r2, 0), 1e-4f);
        assertEquals(206f, TensorTestReads.readFloat(r2, 1), 1e-4f);
    }

    @Test
    void supplierBakesConstants() {
        Shape shape = Shape.of(3);
        Supplier<Tensor> constant = Tracer.traceFunction(() -> Tensor.full(42f, shape));

        Tensor r1 = constant.get();
        Tensor r2 = constant.get();
        for (int i = 0; i < 3; i++) {
            assertEquals(42f, TensorTestReads.readFloat(r1, i), 1e-4f);
            assertEquals(42f, TensorTestReads.readFloat(r2, i), 1e-4f);
        }
    }

    @Test
    void listInputDynamic() {
        Tensor a = Tensor.of(new float[] {1, 2});
        Tensor b = Tensor.of(new float[] {3, 4});
        Tensor c = Tensor.of(new float[] {5, 6});
        Function<List<Tensor>, Tensor> fn =
                Tracer.traceFunction(
                        List.of(a, b, c), ts -> ts.get(0).add(ts.get(1)).multiply(ts.get(2)));

        // (10+1)*2 = 22, (20+2)*3 = 66
        Tensor r =
                fn.apply(
                        List.of(
                                Tensor.of(new float[] {10, 20}),
                                Tensor.of(new float[] {1, 2}),
                                Tensor.of(new float[] {2, 3})));
        assertEquals(22f, TensorTestReads.readFloat(r, 0), 1e-4f);
        assertEquals(66f, TensorTestReads.readFloat(r, 1), 1e-4f);
    }

    @Test
    void constantsBakedAtTraceTime() {
        Tensor sample = Tensor.of(new float[] {1, 2, 3});
        // The scalar 5.0f is a constant captured at trace time.
        Function<Tensor, Tensor> addFive = Tracer.traceFunction(sample, t -> t.add(5.0f));

        Tensor r = addFive.apply(Tensor.of(new float[] {10, 20, 30}));
        assertEquals(15f, TensorTestReads.readFloat(r, 0), 1e-4f);
        assertEquals(25f, TensorTestReads.readFloat(r, 1), 1e-4f);
        assertEquals(35f, TensorTestReads.readFloat(r, 2), 1e-4f);
    }

    @Test
    void graphReusedAcrossCalls() {
        Tensor sample = Tensor.of(new float[] {1, 2, 3});
        Function<Tensor, Tensor> fn = Tracer.traceFunction(sample, t -> t.multiply(2.0f));

        Tensor r1 = fn.apply(Tensor.of(new float[] {1, 2, 3}));
        Tensor r2 = fn.apply(Tensor.of(new float[] {4, 5, 6}));

        TIRGraph g1 = TensorTestInternals.tracedGraph(r1).orElseThrow();
        TIRGraph g2 = TensorTestInternals.tracedGraph(r2).orElseThrow();
        assertSame(g1, g2, "traceFunction should reuse the same TIRGraph instance");
    }

    // --- More interesting kernels ---

    private static Tensor tensorGelu(Tensor value) {
        Tensor cubic = value.multiply(value).multiply(value);
        Tensor inner = cubic.multiply(0.044715f).add(value).multiply(0.79788456f);
        return inner.tanh().add(1f).multiply(value).multiply(0.5f);
    }

    @Test
    void geluTracedFunctionMatchesReference() {
        Tensor sample = Tensor.of(new float[] {0, 1, 2, 3, 4, 5});
        Function<Tensor, Tensor> gelu = Tracer.traceFunction(sample, TraceFunctionTest::tensorGelu);

        // First call — triggers compilation
        Tensor r1 = gelu.apply(Tensor.of(new float[] {0, 1, 2, 3, 4, 5}));
        for (int i = 0; i < 6; i++) {
            assertEquals(TestKernels.gelu(i), TensorTestReads.readFloat(r1, i), 1e-4f);
        }

        // Second call — reuses compiled kernel, different data
        float[] data2 = {-2, -1, 0.5f, 1.5f, 2.5f, 3.5f};
        Tensor r2 = gelu.apply(Tensor.of(data2));
        for (int i = 0; i < 6; i++) {
            assertEquals(TestKernels.gelu(data2[i]), TensorTestReads.readFloat(r2, i), 1e-4f);
        }

        // Graph is the same object
        assertSame(
                TensorTestInternals.tracedGraph(r1).orElseThrow(),
                TensorTestInternals.tracedGraph(r2).orElseThrow());
    }

    @Test
    void mandelbrotTracedSupplier() {
        int width = 80;
        int height = 60;
        int iterations = 32;

        Supplier<Tensor> mandelbrot =
                Tracer.traceFunction(() -> TestKernels.mandelbrotTensor(width, height, iterations));

        // First call — traces and compiles
        Tensor r1 = mandelbrot.get();
        // Second call — reuses compiled kernels
        Tensor r2 = mandelbrot.get();

        // Verify correctness at several sample points
        int[][] samplePoints = {{0, 0}, {height / 2, width / 2}, {height - 1, width - 1}};
        for (int[] pt : samplePoints) {
            int row = pt[0];
            int col = pt[1];
            long idx = (long) row * width + col;
            float expected = TestKernels.mandelbrotIter(row, col, width, height, iterations);
            assertEquals(expected, TensorTestReads.readFloat(r1, idx), 1e-3f);
            assertEquals(expected, TensorTestReads.readFloat(r2, idx), 1e-3f);
        }

        // Both calls share the same graph
        assertSame(
                TensorTestInternals.tracedGraph(r1).orElseThrow(),
                TensorTestInternals.tracedGraph(r2).orElseThrow());
    }

    @Test
    void geluRepeatedCallsProduceCorrectResults() {
        Tensor sample = Tensor.of(new float[] {0, 0, 0, 0});
        Function<Tensor, Tensor> gelu = Tracer.traceFunction(sample, TraceFunctionTest::tensorGelu);

        // Call many times with different data — all should be correct
        for (int round = 0; round < 10; round++) {
            float base = round * 0.5f - 2.0f;
            float[] data = {base, base + 0.1f, base + 0.2f, base + 0.3f};
            Tensor result = gelu.apply(Tensor.of(data));
            for (int i = 0; i < 4; i++) {
                assertEquals(
                        TestKernels.gelu(data[i]),
                        TensorTestReads.readFloat(result, i),
                        1e-4f,
                        "round=" + round + " i=" + i);
            }
        }
    }

    @Test
    void tracedBiFunction2DMatrixOps() {
        Shape shape = Shape.of(2, 3);
        Tensor sampleA = Tensor.of(new float[] {1, 1, 1, 1, 1, 1}, shape);
        Tensor sampleB = Tensor.of(new float[] {1, 1, 1, 1, 1, 1}, shape);

        // Trace: (a, b) -> (a + b) * (a - b), i.e. a^2 - b^2
        BiFunction<Tensor, Tensor, Tensor> diffOfSquares =
                Tracer.traceFunction(sampleA, sampleB, (a, b) -> a.add(b).multiply(a.subtract(b)));

        Tensor a = Tensor.of(new float[] {5, 4, 3, 2, 1, 0}, shape);
        Tensor b = Tensor.of(new float[] {3, 2, 1, 1, 1, 1}, shape);
        Tensor r = diffOfSquares.apply(a, b);

        // 5^2-3^2=16, 4^2-2^2=12, 3^2-1^2=8, 2^2-1^2=3, 1^2-1^2=0, 0^2-1^2=-1
        float[] expected = {16f, 12f, 8f, 3f, 0f, -1f};
        for (int i = 0; i < 6; i++) {
            assertEquals(expected[i], TensorTestReads.readFloat(r, i), 1e-4f);
        }
    }
}
