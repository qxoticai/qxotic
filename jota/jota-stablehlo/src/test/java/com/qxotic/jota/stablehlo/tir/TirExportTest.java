package com.qxotic.jota.stablehlo.tir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.RandomUniformOp;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.random.RandomAlgorithms;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.StructuralEquality;
import com.qxotic.jota.stablehlo.portable.PortableArtifact;
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.text.MlirParser;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.transform.CanonicalizeDefaults;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TracedGraphs;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

/**
 * Traces real jota tensor code into TIR, exports it to StableHLO, converts it back to TIR, and runs
 * both the original and the round-tripped graph through jota's execution pipeline: results must
 * agree. The exported program must also print / parse and serialize / deserialize losslessly.
 */
class TirExportTest {

    /**
     * Traces {@code fn} over {@code inputs} and checks the export round trip; returns the program.
     */
    static Program check(List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        return check(inputs, fn, null);
    }

    /**
     * As {@link #check(List, Function)}; when {@code reference} is given it replaces executing the
     * original traced graph (see {@link #softmaxOfMatmul()}).
     */
    static Program check(
            List<Tensor> inputs, Function<List<Tensor>, Tensor> fn, DenseElementsAttr reference) {
        Tensor traced = Tracer.trace(inputs, fn);
        TIRGraph graph = TracedGraphs.graphOf(traced);
        Program program = TirToStableHlo.export(graph);
        assertEquals(List.of(), com.qxotic.jota.stablehlo.verify.Verifier.verify(program));

        Program reparsed = MlirParser.parse(MlirPrinter.print(program));
        assertTrue(StructuralEquality.diff(program, reparsed).isEmpty(), "text round trip");
        Program deserialized =
                PortableArtifact.deserialize(
                        PortableArtifact.serialize(program, new StableHloVersion(1, 0, 0)));
        assertTrue(
                StructuralEquality.diff(
                                CanonicalizeDefaults.apply(program),
                                CanonicalizeDefaults.apply(deserialized))
                        .isEmpty(),
                "portable artifact round trip");

        TirConversion back = StableHloToTirConverter.convert(program, "main");
        List<Tensor> boundInputs = new ArrayList<>();
        for (TirConversion.InputBinding b : back.inputs()) {
            switch (b) {
                case TirConversion.Argument a -> boundInputs.add(inputs.get(a.index()));
                case TirConversion.Constant c ->
                        boundInputs.add(
                                Tensor.of(
                                        TirConversion.materialize(
                                                c.value(), Environment.nativeMemoryDomain())));
            }
        }
        TensorType resultType = back.resultTypes().getFirst();
        DenseElementsAttr expected =
                reference != null ? reference : dense(traced.materialize(), resultType);
        MemoryView<?> actual =
                TracedGraphs.execute(
                        back.graph(),
                        boundInputs,
                        traced.dataType(),
                        Shape.flat(resultType.dims()));
        String mismatch = NumericComparison.compare(dense(actual, resultType), expected);
        if (mismatch != null) {
            fail(mismatch + "\n" + MlirPrinter.print(program));
        }
        return program;
    }

    @SuppressWarnings("unchecked")
    private static DenseElementsAttr dense(MemoryView<?> view, TensorType type) {
        return TirConversion.toDense((MemoryView<MemorySegment>) view, type);
    }

    private static Tensor f32(Shape shape, float... values) {
        return Tensor.of(values, shape);
    }

    /**
     * The reference is computed independently: jota cannot execute this traced graph itself (its
     * panama kernel for {@code l.subtract(l.max(true, 1))} reads out of bounds), while the
     * round-tripped graph, whose broadcasts are explicit, runs correctly.
     */
    @Test
    void softmaxOfMatmul() {
        Tensor x = f32(Shape.of(2, 3), 1f, -2f, 0.5f, 0.25f, 3f, -1f);
        Tensor w = f32(Shape.of(3, 4), 0.5f, 1f, -1f, 2f, 1.5f, -0.5f, 0f, 1f, 2f, 0.25f, 1f, -1f);
        check(
                List.of(x, w),
                t -> {
                    Tensor logits = t.get(0).matmul(t.get(1));
                    Tensor e = logits.subtract(logits.max(true, 1)).exp();
                    return e.divide(e.sum(DataType.FP32, true, 1));
                },
                softmaxReference());
    }

    /** softmax(x · w) in double precision for the inputs of {@link #softmaxOfMatmul()}. */
    private static DenseElementsAttr softmaxReference() {
        double[][] x = {{1, -2, 0.5}, {0.25, 3, -1}};
        double[][] w = {{0.5, 1, -1, 2}, {1.5, -0.5, 0, 1}, {2, 0.25, 1, -1}};
        StringBuilder sb = new StringBuilder("dense<[");
        for (int i = 0; i < 2; i++) {
            double[] l = new double[4];
            double max = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 3; k++) {
                    l[j] += x[i][k] * w[k][j];
                }
                max = Math.max(max, l[j]);
            }
            double sum = 0;
            for (int j = 0; j < 4; j++) {
                l[j] = Math.exp(l[j] - max);
                sum += l[j];
            }
            sb.append(i > 0 ? ", [" : "[");
            for (int j = 0; j < 4; j++) {
                sb.append(j > 0 ? ", " : "").append((float) (l[j] / sum));
            }
            sb.append(']');
        }
        return (DenseElementsAttr)
                MlirParser.parseAttribute(sb.append("]> : tensor<2x4xf32>").toString());
    }

    @Test
    void activations() {
        Tensor x = f32(Shape.of(5), -3f, -0.5f, 0f, 0.75f, 4f);
        check(
                List.of(x),
                t -> {
                    Tensor v = t.get(0);
                    return v.gelu()
                            .add(v.silu())
                            .add(v.sigmoid())
                            .add(v.relu())
                            .add(v.abs().add(1f).rsqrt());
                });
    }

    @Test
    void integerBitsAndShiftsKeepJavaSemantics() {
        Tensor a = Tensor.of(new int[] {1, -8, 7, 1024}, Shape.of(4));
        Tensor s = Tensor.of(new int[] {1, 33, -1, 2}, Shape.of(4));
        check(
                List.of(a, s),
                t ->
                        t.get(0)
                                .leftShift(t.get(1))
                                .bitwiseXor(t.get(0).rightShift(t.get(1)))
                                .bitwiseAnd(t.get(0).bitwiseNot()));
    }

    @Test
    void castsAndComparisons() {
        Tensor x = f32(Shape.of(5), 0.5f, -1.5f, 0f, 2.75f, -0.5f);
        check(
                List.of(x),
                t -> {
                    Tensor asBool = t.get(0).cast(DataType.BOOL); // x != 0, as jota executes it
                    Tensor asInt = t.get(0).cast(DataType.I32);
                    Tensor cmp = t.get(0).lessThan(Tensor.scalar(1f)).logicalNot();
                    return asBool.cast(DataType.I32)
                            .add(asInt)
                            .add(cmp.cast(DataType.I32))
                            .cast(DataType.FP64);
                });
    }

    @Test
    void viewsSlicesAndBroadcasts() {
        long[] data = new long[15];
        for (int i = 0; i < data.length; i++) {
            data[i] = i;
        }
        Tensor x = Tensor.of(data, Shape.of(3, 5));
        check(
                List.of(x),
                t -> {
                    Tensor x0 = t.get(0);
                    Tensor strided = x0.slice(1, 0, 5, 2); // [3, 3]
                    Tensor reversed = x0.slice(0, 2, -1, -1); // rows in reverse order
                    Tensor tr = reversed.transpose(0, 1).slice(0, 1, 4); // [3, 3]
                    Tensor row =
                            x0.slice(0, 1, 2)
                                    .view(5)
                                    .unsqueeze(0)
                                    .broadcast(Shape.of(3, 5))
                                    .slice(1, 0, 3);
                    return strided.multiply(tr).add(row);
                });
    }

    @Test
    void gatherConcatAndReductions() {
        Tensor table = f32(Shape.of(3, 2), 1f, 2f, 3f, 4f, 5f, 6f);
        Tensor idx = Tensor.of(new int[] {2, 0, 1, 2}, Shape.of(4));
        check(
                List.of(table, idx),
                t -> {
                    Tensor rows = t.get(0).gather(t.get(1), 0); // [4, 2]
                    Tensor both = Tensor.concat(0, rows, t.get(0)); // [7, 2]
                    return both.max(false, 1).add(both.mean(1)).add(Tensor.iota(7, DataType.FP32));
                });
    }

    /** The exported LCG arithmetic reproduces jota's generator bit for bit. */
    @Test
    void randomUniformIsReproducedExactly() {
        for (DataType dt : List.of(DataType.FP32, DataType.FP64)) {
            RandomUniformOp rand = new RandomUniformOp(Shape.of(4, 3), dt, 42, -7);
            Program program = TirToStableHlo.export(new TIRGraph(List.of(), rand));
            assertEquals(List.of(), com.qxotic.jota.stablehlo.verify.Verifier.verify(program));
            TirConversion back = StableHloToTirConverter.convert(program, "main");
            List<Tensor> constants = new ArrayList<>();
            for (TirConversion.InputBinding b : back.inputs()) {
                constants.add(
                        Tensor.of(
                                TirConversion.materialize(
                                        ((TirConversion.Constant) b).value(),
                                        Environment.nativeMemoryDomain())));
            }
            TensorType type = back.resultTypes().getFirst();
            DenseElementsAttr got =
                    dense(TracedGraphs.execute(back.graph(), constants, dt, Shape.of(4, 3)), type);
            for (int i = 0; i < 12; i++) {
                double want =
                        dt == DataType.FP32
                                ? RandomAlgorithms.uniformFp32(i, 42, -7)
                                : RandomAlgorithms.uniformFp64(i, 42, -7);
                assertEquals(want, got.getDouble(i), 0.0, dt + " element " + i);
            }
            assertTrue(type.elementType() instanceof FloatType);
        }
    }

    /**
     * A schedule (jota's kernel partitioning) exports as one function per kernel step plus a
     * {@code @main} calling them; inlined back to TIR it computes what jota computes.
     */
    @Test
    void scheduledProgramExportsAsCallingFunctions() {
        Tensor x = f32(Shape.of(2, 3), 1f, -2f, 0.5f, 0.25f, 3f, -1f);
        Tensor w = f32(Shape.of(3, 4), 0.5f, 1f, -1f, 2f, 1.5f, -0.5f, 0f, 1f, 2f, 0.25f, 1f, -1f);
        Tensor traced =
                Tracer.trace(
                        List.of(x, w),
                        t -> t.get(0).matmul(t.get(1)).tanh().sum(DataType.FP32, 1).exp().sqrt());
        TIRGraph graph = TracedGraphs.graphOf(traced);
        com.qxotic.jota.ir.tir.ScheduledProgram schedule =
                new com.qxotic.jota.ir.tir.TIRSchedulePass().run(graph);
        Program program = TirToStableHlo.export(schedule, graph);
        assertEquals(List.of(), com.qxotic.jota.stablehlo.verify.Verifier.verify(program));
        assertEquals(schedule.steps().size() + 1, program.functions().size());
        assertTrue(schedule.steps().size() >= 2, "expected several kernel steps");

        Program reparsed = MlirParser.parse(MlirPrinter.print(program));
        assertTrue(StructuralEquality.diff(program, reparsed).isEmpty());
        PortableArtifact.deserialize(
                PortableArtifact.serialize(program, new StableHloVersion(1, 0, 0)));

        TirConversion back = StableHloToTirConverter.convert(program, "main");
        List<Tensor> boundInputs = new ArrayList<>();
        for (TirConversion.InputBinding b : back.inputs()) {
            boundInputs.add(
                    switch (b) {
                        case TirConversion.Argument a -> List.of(x, w).get(a.index());
                        case TirConversion.Constant c ->
                                Tensor.of(
                                        TirConversion.materialize(
                                                c.value(), Environment.nativeMemoryDomain()));
                    });
        }
        TensorType type = back.resultTypes().getFirst();
        DenseElementsAttr expected = dense(traced.materialize(), type);
        DenseElementsAttr actual =
                dense(
                        TracedGraphs.execute(
                                back.graph(), boundInputs, DataType.FP32, Shape.flat(type.dims())),
                        type);
        assertEquals(null, NumericComparison.compare(actual, expected));
    }
}
