package com.qxotic.jota.stablehlo.op;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.stablehlo.attr.ComparisonDirection;
import com.qxotic.jota.stablehlo.attr.ComparisonType;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.text.MlirParser;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

class OpVisitorTest {

    private static final String PROGRAM =
            """
            module @m {
              func.func public @main(%x: tensor<4x3xf32>, %idx: tensor<2xi32>) -> tensor<i1> {
                %cst = stablehlo.constant dense<0.0> : tensor<f32>
                %g = "stablehlo.gather"(%x, %idx) <{
                    dimension_numbers = #stablehlo.gather<offset_dims = [1],
                      collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>,
                    slice_sizes = array<i64: 1, 3>}> : (tensor<4x3xf32>, tensor<2xi32>) -> tensor<2x3xf32>
                %s = stablehlo.reduce(%g init: %cst) applies stablehlo.add across dimensions = [0, 1]
                    : (tensor<2x3xf32>, tensor<f32>) -> tensor<f32>
                %w:2 = stablehlo.while(%i = %cst, %acc = %s) : tensor<f32>, tensor<f32>
                 cond {
                  %lt = stablehlo.compare LT, %i, %acc, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
                  stablehlo.return %lt : tensor<i1>
                } do {
                  %n = stablehlo.add %i, %acc : tensor<f32>
                  stablehlo.return %n, %acc : tensor<f32>, tensor<f32>
                }
                %r = stablehlo.compare GE, %w#0, %w#1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
                return %r : tensor<i1>
              }
            }
            """;

    @Test
    void typedAccessors() {
        Program program = MlirParser.parse(PROGRAM);
        assertEquals("m", program.name().orElseThrow());
        FuncOp main = program.function("main").orElseThrow();
        List<Operation> body = main.body().block().operations();

        GatherOp gather = assertInstanceOf(GatherOp.class, body.get(1).asOp());
        assertArrayEquals(new long[] {1, 3}, gather.sliceSizes());
        assertArrayEquals(new long[] {0}, gather.dimensionNumbers().collapsedSliceDims());
        assertEquals(main.body().block().argument(0), gather.operand());

        ReduceOp reduce = assertInstanceOf(ReduceOp.class, body.get(2).asOp());
        assertArrayEquals(new long[] {0, 1}, reduce.dimensions());
        assertEquals(1, reduce.inputs().size());
        assertEquals(body.get(0).result(0), reduce.initValues().getFirst());
        Operation combiner = reduce.body().block().operations().getFirst();
        assertInstanceOf(AddOp.class, combiner.asOp());
        assertEquals(2, reduce.body().block().arguments().size());

        WhileOp loop = assertInstanceOf(WhileOp.class, body.get(3).asOp());
        assertEquals(2, loop.operand().size());
        CompareOp lt = (CompareOp) loop.cond().block().operations().getFirst().asOp();
        assertEquals(ComparisonDirection.LT, lt.comparisonDirection());
        assertEquals(ComparisonType.FLOAT, lt.compareType().orElseThrow());
        // region arguments are scoped: %i in cond is the cond block argument
        assertEquals(loop.cond().block().argument(0), lt.lhs());

        CompareOp ge = (CompareOp) body.get(4).asOp();
        assertTrue(ge.compareType().isEmpty());
        assertEquals(loop.operation().result(1), ge.rhs());
    }

    @Test
    void visitorDispatchesOnTypedRecords() {
        Program program = MlirParser.parse(PROGRAM);
        Map<String, Integer> counts = new TreeMap<>();
        OpVisitor<String> visitor =
                new OpVisitor<>() {
                    @Override
                    public String visitDefault(Op op) {
                        return "other";
                    }

                    @Override
                    public String visitCompare(CompareOp op) {
                        return "compare:" + op.comparisonDirection();
                    }

                    @Override
                    public String visitAdd(AddOp op) {
                        return "add";
                    }
                };
        program.operation().walk(op -> counts.merge(op.asOp().accept(visitor), 1, Integer::sum));
        assertEquals(2, counts.get("add")); // reducer body + while body
        assertEquals(1, counts.get("compare:LT"));
        assertEquals(1, counts.get("compare:GE"));
    }

    @Test
    void defaultVisitorRejectsUnsupportedOps() {
        Program program = MlirParser.parse(PROGRAM);
        OpVisitor<Void> none = new OpVisitor<>() {};
        Operation first = program.function("main").orElseThrow().body().block().operations().get(0);
        assertThrows(UnsupportedOperationException.class, () -> first.asOp().accept(none));
    }

    /** Every typed record is produced by {@link Op#wrap} and has its own visit method. */
    @Test
    void typedRecordsAreWiredIntoWrapAndVisitor() throws Exception {
        List<String> problems = new ArrayList<>();
        for (Class<?> c : Op.class.getPermittedSubclasses()) {
            if (c == GenericOp.class) {
                continue;
            }
            String name = (String) c.getField("NAME").get(null);
            Operation op =
                    Operation.create(name, List.of(), List.of(), List.of(), List.of(), List.of());
            if (op.asOp().getClass() != c) {
                problems.add(name + " wraps to " + op.asOp().getClass().getSimpleName());
            }
            String visit = "visit" + c.getSimpleName().substring(0, c.getSimpleName().length() - 2);
            Method m = OpVisitor.class.getMethod(visit, c);
            if (m == null) {
                problems.add("missing " + visit);
            }
        }
        assertTrue(problems.isEmpty(), problems.toString());
        assertTrue(Op.class.getPermittedSubclasses().length > 120);
        assertInstanceOf(
                GenericOp.class,
                Operation.create("chlo.erf", List.of(), List.of(), List.of(), List.of(), List.of())
                        .asOp());
    }
}
