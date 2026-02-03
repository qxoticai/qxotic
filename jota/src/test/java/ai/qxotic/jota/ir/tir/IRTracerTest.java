package ai.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.TIRToLIRLowerer;
import ai.qxotic.jota.ir.lir.LIRGraph;
import ai.qxotic.jota.ir.lir.LIRInterpreter;
import ai.qxotic.jota.ir.lir.LIRStandardPipeline;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.tensor.IRTracer;
import ai.qxotic.jota.tensor.LazyComputation;
import ai.qxotic.jota.tensor.Tensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class IRTracerTest {

    private static final MemoryContext<MemorySegment> CONTEXT = ContextFactory.ofMemorySegment();

    private static float readFloat(MemoryView<?> view, long linearIndex) {
        @SuppressWarnings("unchecked")
        MemoryView<MemorySegment> typedView = (MemoryView<MemorySegment>) view;
        long offset = Indexing.linearToOffset(typedView, linearIndex);
        return CONTEXT.memoryAccess().readFloat(typedView.memory(), offset);
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

    /**
     * Matrix multiplication test using einsum-style broadcasting and reduction.
     *
     * <p>Equivalent to tinygrad: (a.reshape(N, 1, N) * b.T.reshape(1, N, N)).sum(axis=2)
     *
     * <p>This tests the complete chain: reshape -> transpose -> broadcasted multiply -> reduction
     */
    @Test
    void testMatrixMultiplicationViaTracer() {
        int N = 4; // Small for fast tests

        // Create two NxN identity-like matrices with actual data
        // A is all ones, B is identity (1 on diagonal)
        float[] aData = new float[N * N];
        float[] bData = new float[N * N];
        for (int i = 0; i < N * N; i++) {
            aData[i] = 1.0f;
            bData[i] = (i % (N + 1) == 0) ? 1.0f : 0.0f; // Identity matrix
        }

        Tensor a = Tensor.of(aData, Shape.of(N, N));
        Tensor b = Tensor.of(bData, Shape.of(N, N));

        Tensor result =
                IRTracer.trace(
                        List.of(a, b),
                        tensors -> {
                            Tensor x = tensors.get(0); // (N, N)
                            Tensor y = tensors.get(1); // (N, N)

                            // x.reshape(N, 1, N)
                            Tensor xReshaped = x.reshape(Shape.of(N, 1, N));

                            // y.T (transpose last two axes) -> reshape to (1, N, N)
                            Tensor yTransposed = y.transpose(0, 1);
                            Tensor yReshaped = yTransposed.reshape(Shape.of(1, N, N));

                            // Explicitly broadcast to (N, N, N) before multiply
                            Shape broadcastShape = Shape.of(N, N, N);
                            Tensor xBroadcasted = xReshaped.broadcast(broadcastShape);
                            Tensor yBroadcasted = yReshaped.broadcast(broadcastShape);

                            // Multiply: (N, N, N) * (N, N, N) -> (N, N, N)
                            Tensor multiplied = xBroadcasted.multiply(yBroadcasted);

                            // Sum over axis 2 -> (N, N) result
                            return multiplied.sum(DataType.FP32, 2);
                        });

        // Verify shape (N, N)
        assertEquals(Shape.of(N, N), result.shape());
        MemoryView<?> output = result.materialize();

        // When A is all ones and B is identity, the result should be all ones
        // (each row of A is [1,1,1,1], each col of B is [1,0,0,0], [0,1,0,0], etc.)
        // So A * B = all ones matrix since A has all ones
        for (int i = 0; i < N * N; i++) {
            assertEquals(1.0f, readFloat(output, i), 0.0001f);
        }
    }

    /**
     * Matrix multiplication test for non-square matrices.
     *
     * <p>Tests (N, K) x (K, M) matmul where N, K, M are all different.
     *
     * <p>Equivalent to tinygrad: (a.reshape(N, 1, K) * b.T.reshape(1, M, K)).sum(axis=2)
     */
    @Test
    void testMatrixMultiplicationNonSquareViaTracer() {
        int N = 2;
        int K = 3;
        int M = 5;

        Tensor a = Tensor.iota(N * K, DataType.FP32).view(Shape.of(N, K));
        Tensor b = Tensor.iota(K * M, DataType.FP32).view(Shape.of(K, M));

        // Debug: Print b values before tracing
        MemoryView<?> bView = b.materialize();
        System.out.println("DEBUG: Original b (y input) shape: " + bView.shape());
        System.out.println("DEBUG: Original b (y input) stride: " + bView.stride());
        System.out.println("DEBUG: b values at row 0 (K=0, positions [0,0] to [0,4]):");
        for (int i = 0; i < M; i++) {
            System.out.println("  b[0," + i + "] (linear " + i + ") = " + readFloat(bView, i));
        }
        System.out.println("DEBUG: b values at row 1 (K=1, positions [1,0] to [1,4]):");
        for (int i = 0; i < M; i++) {
            System.out.println(
                    "  b[1," + i + "] (linear " + (M + i) + ") = " + readFloat(bView, M + i));
        }
        System.out.println("DEBUG: b values at row 2 (K=2, positions [2,0] to [2,4]):");
        for (int i = 0; i < M; i++) {
            System.out.println(
                    "  b[2,"
                            + i
                            + "] (linear "
                            + (2 * M + i)
                            + ") = "
                            + readFloat(bView, 2 * M + i));
        }

        Tensor result =
                IRTracer.trace(List.of(a, b), tensors -> matmul(tensors.get(0), tensors.get(1)));

        // Verify shape (N, M)
        assertEquals(Shape.of(N, M), result.shape());
        MemoryView<MemorySegment> output = (MemoryView<MemorySegment>) result.materialize();

        // Expected result: since A is all ones and B's rows are [1, 2, ..., M],
        // each element in result[i, j] = sum_k 1.0 * (j+1) = K * (j+1)
        // So each row should be [4, 8, 12, 16, 20] (since K=4)

        MemoryAccess<MemorySegment> access = Environment.current().panamaContext().memoryAccess();
        MemoryView<MemorySegment> aView = (MemoryView<MemorySegment>) a.materialize();
        MemoryView<MemorySegment> bView2 = (MemoryView<MemorySegment>) b.materialize();
        System.out.println("DEBUG: aView shape=" + aView.shape() + ", stride=" + aView.stride());
        System.out.println("DEBUG: bView2 shape=" + bView2.shape() + ", stride=" + bView2.stride());
        for (int n = 0; n < N; n++) {
            for (int m = 0; m < M; m++) {
                float acc = 0;
                for (int k = 0; k < K; ++k) {
                    float fa =
                            access.readFloat(aView.memory(), Indexing.coordToOffset(aView, n, k));
                    float fb =
                            access.readFloat(bView2.memory(), Indexing.coordToOffset(bView2, k, m));
                    acc += fa * fb;
                    if (n == 0 && m == 0) {
                        System.out.println("DEBUG: n=0,m=0,k=" + k + ": fa=" + fa + ", fb=" + fb);
                    }
                }
                float expected = acc;
                float actual =
                        access.readFloat(output.memory(), Indexing.coordToOffset(output, n, m));
                System.out.println(
                        "DEBUG: result["
                                + n
                                + ","
                                + m
                                + "]: expected="
                                + expected
                                + ", actual="
                                + actual);
                assertEquals(expected, actual, 0.0001f);
            }
        }
    }

    private static Tensor matmul(Tensor x, Tensor y) {
        // int N, int K, int M
        // Tensor x = tensors.get(0); // (N, K)
        // Tensor y = tensors.get(1); // (K, M)
        long N = x.shape().size(0);
        long K = x.shape().size(1);
        long M = y.shape().size(1);

        System.out.println("DEBUG: Input shapes: x=" + x.shape() + ", y=" + y.shape());

        // x.reshape(N, 1, K)
        Tensor xReshaped = x.reshape(Shape.of(N, 1, K));
        System.out.println("DEBUG: After x.reshape(" + N + ", 1, " + K + "): " + xReshaped.shape());

        // y.T gives (M, K) -> reshape to (1, M, K)
        Tensor yTransposed = y.transpose(0, 1);
        System.out.println("DEBUG: After y.transpose(0, 1): " + yTransposed.shape());
        System.out.println("DEBUG: yTransposed stride: " + yTransposed.stride());
        Tensor yReshaped = yTransposed.reshape(Shape.of(1, M, K));
        System.out.println("DEBUG: After y.reshape(1, " + M + ", " + K + "): " + yReshaped.shape());
        System.out.println("DEBUG: yReshaped stride: " + yReshaped.stride());

        // Explicitly broadcast to (N, M, K) before multiply
        Shape broadcastShape = Shape.of(N, M, K);
        Tensor xBroadcasted = xReshaped.broadcast(broadcastShape);
        Tensor yBroadcasted = yReshaped.broadcast(broadcastShape);
        System.out.println(
                "DEBUG: After broadcast to ("
                        + N
                        + ", "
                        + M
                        + ", "
                        + K
                        + "): x="
                        + xBroadcasted.shape()
                        + ", y="
                        + yBroadcasted.shape());
        System.out.println("DEBUG: yBroadcasted stride: " + yBroadcasted.stride());

        // yReshaped is (1, 5, 3) with stride (15, 3, 1)
        // yBroadcasted with stride (0, 3, 1) should read from yReshaped at:
        // [0,0,0] -> offset 0*0 + 0*3 + 0*1 = 0 -> yReshaped[0,0,0] = original y[0,0]
        // [0,0,1] -> offset 0*0 + 0*3 + 1*1 = 1 -> yReshaped[0,0,1] = original y[1,0]
        // [0,0,2] -> offset 0*0 + 0*3 + 2*1 = 2 -> yReshaped[0,0,2] = original y[2,0]
        // Since yReshaped has stride (15, 3, 1):
        // yReshaped[0,0,k] -> offset 0*15 + 0*3 + k*1 = k -> original y[k,0] (column 0)

        // Multiply: (N, M, K) * (N, M, K) -> (N, M, K)
        Tensor multiplied = xBroadcasted.multiply(yBroadcasted);
        System.out.println("DEBUG: After multiply: " + multiplied.shape());

        // Sum over axis 2 (the K dimension) -> (N, M)
        Tensor result = multiplied.sum(DataType.FP32, 2);
        System.out.println("DEBUG: After sum(axis=2): " + result.shape());
        return result;
    }

    /**
     * Full pipeline test for matmul with broadcasted scalar inputs using Tensor API.
     *
     * <p>This test traces a matmul operation using the high-level Tensor API with constant-filled
     * tensors, then runs the complete optimization pipeline:
     *
     * <ol>
     *   <li>Trace matmul using Tensor API (IRTracer)
     *   <li>Extract TIR from lazy tensor
     *   <li>Apply TIR constant folding (graph inputs are preserved as scalar parameters)
     *   <li>Lower optimized TIR to LIR
     *   <li>Execute via LIRInterpreter with bound scalar inputs
     * </ol>
     *
     * <p>The test verifies that scalar constants from graph inputs are properly preserved as
     * runtime scalar parameters (not folded away), allowing the values to be passed at execution.
     */
    @Test
    void testMatmulFoldingFullPipeline() {
        int M = 2;
        int K = 3;
        int N = 5;

        // Create constant-filled tensors using Tensor.full()
        // These are materialized tensors filled with constant values
        Tensor aFilled = Tensor.scalar(2.0f).broadcast(Shape.of(M, K));
        Tensor bFilled = Tensor.scalar(3.0f).broadcast(Shape.of(K, N));

        // Trace matmul operation using high-level Tensor API
        Tensor result =
                IRTracer.trace(
                        List.of(aFilled, bFilled),
                        tensors -> {
                            Tensor a = tensors.get(0);
                            Tensor b = tensors.get(1);

                            // Reshape for matmul: a to (M, 1, K), b.T to (1, N, K)
                            Tensor aReshaped = a.reshape(Shape.of(M, 1, K));
                            Tensor bReshaped = b.transpose(0, 1).reshape(Shape.of(1, N, K));

                            // Broadcast to (M, N, K) and multiply
                            Shape broadcastShape = Shape.of(M, N, K);
                            Tensor aBroadcasted = aReshaped.broadcast(broadcastShape);
                            Tensor bBroadcasted = bReshaped.broadcast(broadcastShape);
                            Tensor multiplied = aBroadcasted.multiply(bBroadcasted);

                            // Sum over K dimension -> (M, N)
                            return multiplied.sum(DataType.FP32, 2);
                        });

        // Extract TIRGraph from lazy tensor
        LazyComputation comp = result.computation().orElseThrow();
        Map<String, Object> attrs = comp.attributes();
        TIRGraph tirGraph = (TIRGraph) attrs.get("graph");

        // Apply constant folding - should fold entire matmul computation
        TIRConstantFoldingPass foldingPass = new TIRConstantFoldingPass();
        TIRGraph optimizedTir = foldingPass.run(tirGraph);

        // Lower optimized TIR to LIR
        TIRToLIRLowerer lowerer = new TIRToLIRLowerer();
        LIRGraph lirGraph = lowerer.lower(optimizedTir);
        lirGraph = new LIRStandardPipeline().run(lirGraph);

        // Execute with LIRInterpreter
        // Scalar inputs are passed at runtime (not folded since they're graph inputs)
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment output = arena.allocate(M * N * Float.BYTES);

            // Execute - bind output buffer and scalar inputs
            LIRInterpreter interpreter = new LIRInterpreter();
            // Scalar inputs: id 0 = 2.0f, id 1 = 3.0f
            interpreter.bindScalarInput(0, Float.floatToRawIntBits(2.0f), DataType.FP32);
            interpreter.bindScalarInput(1, Float.floatToRawIntBits(3.0f), DataType.FP32);
            // Output buffer at id 2
            interpreter.bindBuffer(2, output);
            interpreter.execute(lirGraph);

            // Verify: each element should be 2.0 * 3.0 * K = 18.0
            float expected = 2.0f * 3.0f * K;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float actual = output.getAtIndex(ValueLayout.JAVA_FLOAT, i * N + j);
                    assertEquals(expected, actual, 1e-5f, "Mismatch at [" + i + "," + j + "]");
                }
            }
        }
    }

    @Test
    void testMatmulFoldingFullIotasPipeline() {
        int M = 2;
        int K = 3;
        int N = 5;

        // Create tensors with iota values [0, 1, 2, ...]
        // aFilled = [[0, 1, 2], [3, 4, 5]] (M x K)
        // bFilled = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]] (K x N)
        Tensor aFilled = Tensor.iota(M * K).view(Shape.of(M, K));
        Tensor bFilled = Tensor.iota(K * N).view(Shape.of(K, N));

        // Trace matmul operation using high-level Tensor API
        Tensor result =
                IRTracer.trace(
                        List.of(aFilled, bFilled),
                        tensors -> {
                            Tensor a = tensors.get(0);
                            Tensor b = tensors.get(1);

                            // Reshape for matmul: a to (M, 1, K), b.T to (1, N, K)
                            Tensor aReshaped = a.reshape(Shape.of(M, 1, K));
                            Tensor bReshaped = b.transpose(0, 1).reshape(Shape.of(1, N, K));

                            // Broadcast to (M, N, K) and multiply
                            Shape broadcastShape = Shape.of(M, N, K);
                            Tensor aBroadcasted = aReshaped.broadcast(broadcastShape);
                            Tensor bBroadcasted = bReshaped.broadcast(broadcastShape);
                            Tensor multiplied = aBroadcasted.multiply(bBroadcasted);

                            // Sum over K dimension -> (M, N)
                            return multiplied.sum(DataType.FP32, 2);
                        });

        // Materialize the result
        MemoryView<?> output = result.materialize();

        // Verify the matmul result
        // aFilled = [[0, 1, 2], [3, 4, 5]]
        // bFilled = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]]
        // bFilled.T = [[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8, 13], [4, 9, 14]]
        //
        // For element [0,0]: dot([0, 1, 2], [0, 5, 10]) = 0*0 + 1*5 + 2*10 = 25
        // For element [0,1]: dot([0, 1, 2], [1, 6, 11]) = 0*1 + 1*6 + 2*11 = 28
        // For element [0,2]: dot([0, 1, 2], [2, 7, 12]) = 0*2 + 1*7 + 2*12 = 31
        // For element [0,3]: dot([0, 1, 2], [3, 8, 13]) = 0*3 + 1*8 + 2*13 = 34
        // For element [0,4]: dot([0, 1, 2], [4, 9, 14]) = 0*4 + 1*9 + 2*14 = 37
        //
        // For element [1,0]: dot([3, 4, 5], [0, 5, 10]) = 3*0 + 4*5 + 5*10 = 70
        // For element [1,1]: dot([3, 4, 5], [1, 6, 11]) = 3*1 + 4*6 + 5*11 = 82
        // For element [1,2]: dot([3, 4, 5], [2, 7, 12]) = 3*2 + 4*7 + 5*12 = 94
        // For element [1,3]: dot([3, 4, 5], [3, 8, 13]) = 3*3 + 4*8 + 5*13 = 106
        // For element [1,4]: dot([3, 4, 5], [4, 9, 14]) = 3*4 + 4*9 + 5*14 = 118
        float delta = 0.001f;
        assertEquals(25.0f, readFloat(output, 0), delta, "Mismatch at [0,0]");
        assertEquals(28.0f, readFloat(output, 1), delta, "Mismatch at [0,1]");
        assertEquals(31.0f, readFloat(output, 2), delta, "Mismatch at [0,2]");
        assertEquals(34.0f, readFloat(output, 3), delta, "Mismatch at [0,3]");
        assertEquals(37.0f, readFloat(output, 4), delta, "Mismatch at [0,4]");
        assertEquals(70.0f, readFloat(output, 5), delta, "Mismatch at [1,0]");
        assertEquals(82.0f, readFloat(output, 6), delta, "Mismatch at [1,1]");
        assertEquals(94.0f, readFloat(output, 7), delta, "Mismatch at [1,2]");
        assertEquals(106.0f, readFloat(output, 8), delta, "Mismatch at [1,3]");
        assertEquals(118.0f, readFloat(output, 9), delta, "Mismatch at [1,4]");
    }
}
