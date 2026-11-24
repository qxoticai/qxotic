package com.qxotic.jota.stablehlo.tir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.RandomUniformOp;
import com.qxotic.jota.ir.tir.TIRGraph;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirPrinter;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.Type;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.TracedGraphs;
import com.qxotic.jota.tensor.Tracer;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.junit.jupiter.api.io.TempDir;

/**
 * Runs exported TIR graphs on XLA (through jaxlib) and compares with jota: an independent
 * implementation executing the StableHLO this module produces. Enabled with {@code
 * -Dstablehlo.jax.python=/path/to/python} (a Python with {@code jax} installed).
 */
@EnabledIfSystemProperty(named = "stablehlo.jax.python", matches = ".+")
class XlaExecutionTest {

    private static final String RUNNER =
            """
            import json, sys, numpy as np, jax
            from jax._src import xla_bridge as xb
            from jax._src.lib import xla_client as xc
            from jax._src.interpreters import mlir as jmlir
            from jax._src.lib.mlir import ir
            DTYPES = {'i1': np.bool_, 'i8': np.int8, 'i16': np.int16, 'i32': np.int32, 'i64': np.int64,
                      'f16': np.float16, 'f32': np.float32, 'f64': np.float64}
            backend = xb.get_backend('cpu')
            for case in json.load(open(sys.argv[1])):
                with jmlir.make_ir_context():
                    module = ir.Module.parse(open(case['program']).read())
                    exe = backend.compile_and_load(str(module), backend.devices()[:1], xc.CompileOptions())
                args = [jax.device_put(np.fromfile(i['file'], DTYPES[i['dtype']]).reshape(i['shape']))
                        for i in case['inputs']]
                out = np.asarray(exe.execute(args)[0])
                out.astype(DTYPES[case['output_dtype']]).tofile(case['output'])
            """;

    private record Case(
            String name, TIRGraph graph, List<DenseElementsAttr> inputs, DenseElementsAttr jota) {}

    private final List<Case> cases = new ArrayList<>();

    @SuppressWarnings("unchecked")
    private static DenseElementsAttr dense(MemoryView<?> view, Shape shape, DataType dt) {
        return TirConversion.toDense(
                (MemoryView<MemorySegment>) view, TirToStableHlo.tensorType(shape, dt));
    }

    private void add(String name, List<Tensor> inputs, Function<List<Tensor>, Tensor> fn) {
        Tensor traced = Tracer.trace(inputs, fn);
        List<DenseElementsAttr> in = new ArrayList<>();
        for (Tensor t : inputs) {
            in.add(dense(t.materialize(), t.shape(), t.dataType()));
        }
        cases.add(
                new Case(
                        name,
                        TracedGraphs.graphOf(traced),
                        in,
                        dense(traced.materialize(), traced.shape(), traced.dataType())));
    }

    @Test
    void xlaAgreesWithJota(@TempDir Path dir) throws Exception {
        Tensor x = Tensor.of(new float[] {-3f, -0.5f, 0f, 0.75f, 4f, 1.25f}, Shape.of(2, 3));
        Tensor w = Tensor.of(new float[] {0.5f, 1f, -1f, 2f, 1.5f, -0.5f}, Shape.of(3, 2));
        Tensor ints = Tensor.of(new int[] {1, -8, 7, 1024, 5, -3}, Shape.of(2, 3));
        Tensor shifts = Tensor.of(new int[] {1, 33, 31, 2, 0, 65}, Shape.of(2, 3));
        Tensor idx = Tensor.of(new int[] {2, 0, 1}, Shape.of(3));
        add("matmul", List.of(x, w), t -> t.get(0).matmul(t.get(1)).tanh());
        add(
                "activations",
                List.of(x),
                t -> t.get(0).gelu().add(t.get(0).silu()).add(t.get(0).sigmoid()));
        add(
                "reductions",
                List.of(x),
                t -> t.get(0).sum(DataType.FP32, 1).add(t.get(0).max(1)).add(t.get(0).mean(1)));
        add(
                "shifts",
                List.of(ints, shifts),
                t -> t.get(0).leftShift(t.get(1)).bitwiseXor(t.get(0).rightShift(t.get(1))));
        add(
                "casts",
                List.of(x),
                t ->
                        t.get(0)
                                .cast(DataType.BOOL)
                                .where(t.get(0).cast(DataType.I32), Tensor.scalar(7))
                                .cast(DataType.FP64));
        add(
                "views",
                List.of(x),
                t ->
                        t.get(0)
                                .transpose(0, 1)
                                .slice(0, 2, -1, -1)
                                .contiguous()
                                .view(6)
                                .slice(0, 0, 6, 2));
        add("gather", List.of(w, idx), t -> t.get(0).gather(t.get(1), 0).exp());
        RandomUniformOp rand = new RandomUniformOp(Shape.of(3, 4), DataType.FP32, 5, 9);
        cases.add(
                new Case(
                        "random",
                        new TIRGraph(List.of(), rand),
                        List.of(),
                        dense(
                                TracedGraphs.execute(
                                        new TIRGraph(List.of(), rand),
                                        List.of(),
                                        DataType.FP32,
                                        Shape.of(3, 4)),
                                Shape.of(3, 4),
                                DataType.FP32)));

        StringBuilder manifest = new StringBuilder("[");
        Map<Case, Path> outputs = new LinkedHashMap<>();
        for (Case c : cases) {
            Program program = TirToStableHlo.export(c.graph);
            Path programFile = dir.resolve(c.name + ".mlir");
            Files.writeString(programFile, MlirPrinter.printGeneric(program));
            Path out = dir.resolve(c.name + ".out");
            outputs.put(c, out);
            StringBuilder inputs = new StringBuilder("[");
            for (int i = 0; i < c.inputs.size(); i++) {
                DenseElementsAttr in = c.inputs.get(i);
                Path file = dir.resolve(c.name + ".in" + i);
                Files.write(file, expand(in));
                inputs.append(i > 0 ? "," : "")
                        .append("{\"file\":\"")
                        .append(file)
                        .append("\",\"dtype\":\"")
                        .append(in.type().elementType().mlir())
                        .append("\",\"shape\":")
                        .append(java.util.Arrays.toString(in.type().dims()))
                        .append('}');
            }
            Type resultType = c.jota.type();
            manifest.append(manifest.length() > 1 ? "," : "")
                    .append("{\"program\":\"")
                    .append(programFile)
                    .append("\",\"inputs\":")
                    .append(inputs)
                    .append("]")
                    .append(",\"output\":\"")
                    .append(out)
                    .append("\",\"output_dtype\":\"")
                    .append(((TensorType) resultType).elementType().mlir())
                    .append("\"}");
        }
        Path manifestFile = dir.resolve("cases.json");
        Files.writeString(manifestFile, manifest.append(']').toString());
        Path script = dir.resolve("run.py");
        Files.writeString(script, RUNNER);
        Process p =
                new ProcessBuilder(
                                System.getProperty("stablehlo.jax.python"),
                                "-I",
                                script.toString(),
                                manifestFile.toString())
                        .redirectErrorStream(true)
                        .start();
        String log = new String(p.getInputStream().readAllBytes());
        assertTrue(p.waitFor(10, TimeUnit.MINUTES));
        assertEquals(0, p.exitValue(), log);

        for (Case c : cases) {
            DenseElementsAttr xla =
                    new DenseElementsAttr(c.jota.type(), Files.readAllBytes(outputs.get(c)), false);
            String mismatch = NumericComparison.compare(xla, c.jota);
            assertEquals(null, mismatch, c.name + ": XLA vs jota");
        }
    }

    /** Raw row-major bytes of a dense attribute (splats expanded). */
    private static byte[] expand(DenseElementsAttr d) {
        int bytes = d.elementBytes();
        byte[] raw = d.rawData();
        byte[] out = new byte[(int) (d.numElements() * bytes)];
        for (int i = 0; i < d.numElements(); i++) {
            System.arraycopy(raw, d.isSplat() ? 0 : i * bytes, out, i * bytes, bytes);
        }
        return out;
    }
}
