package ai.qxotic.jota.tensor;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.tir.BinaryOp;
import ai.qxotic.jota.ir.tir.CastOp;
import ai.qxotic.jota.ir.tir.Contiguous;
import ai.qxotic.jota.ir.tir.GatherOp;
import ai.qxotic.jota.ir.tir.ReductionOp;
import ai.qxotic.jota.ir.tir.ScheduledOutputRef;
import ai.qxotic.jota.ir.tir.ScheduledProgram;
import ai.qxotic.jota.ir.tir.TIRGraph;
import ai.qxotic.jota.ir.tir.TIRNode;
import ai.qxotic.jota.ir.tir.TIRSchedulePass;
import ai.qxotic.jota.ir.tir.TernaryOp;
import ai.qxotic.jota.ir.tir.UnaryOp;
import ai.qxotic.jota.ir.tir.ViewTransform;
import java.util.List;
import org.junit.jupiter.api.Test;

class TensorScheduleIntegrationTest {

    @Test
    void schedulesNonTrivialTracedTensorProgram() {
        Tensor lhs = Tensor.of(range(12), Shape.of(3, 4));
        Tensor rhs = Tensor.of(reverseRange(12), Shape.of(3, 4));
        Tensor table = Tensor.of(range(24), Shape.of(6, 4));
        Tensor indices = Tensor.of(new int[] {2, 0, 5}, Shape.of(3));

        Tensor traced =
                Tracer.trace(
                        List.of(lhs, rhs, table, indices),
                        ts -> {
                            Tensor x = ts.get(0).multiply(ts.get(1)).add(1f).relu();
                            Tensor y = ts.get(2).gather(ts.get(3), 0);
                            Tensor z = x.add(y);
                            return z.sum(DataType.FP32, 1).add(2f).square();
                        });

        assertTrue(traced.isLazy());
        IRComputation computation = assertInstanceOf(IRComputation.class, traced.computation().orElseThrow());
        TIRGraph optimized = computation.optimizeGraph(computation.graph());

        ScheduledProgram schedule = new TIRSchedulePass().run(optimized);

        assertTrue(!schedule.steps().isEmpty(), "Expected at least one scheduled kernel");
        assertInstanceOf(ScheduledOutputRef.ValueOutput.class, schedule.output());

        boolean hasGather =
                schedule.steps().stream().anyMatch(step -> containsNodeType(step.graph().outputs().getFirst(), GatherOp.class));
        boolean hasReduction =
                schedule.steps().stream().anyMatch(step -> containsNodeType(step.graph().outputs().getFirst(), ReductionOp.class));

        assertTrue(hasGather, "Expected schedule to include at least one gather kernel");
        assertTrue(hasReduction, "Expected schedule to include at least one reduction kernel");
    }

    private static boolean containsNodeType(TIRNode node, Class<? extends TIRNode> type) {
        if (type.isInstance(node)) {
            return true;
        }
        return switch (node) {
            case UnaryOp op -> containsNodeType(op.input(), type);
            case BinaryOp op -> containsNodeType(op.left(), type) || containsNodeType(op.right(), type);
            case TernaryOp op ->
                    containsNodeType(op.cond(), type)
                            || containsNodeType(op.trueExpr(), type)
                            || containsNodeType(op.falseExpr(), type);
            case CastOp op -> containsNodeType(op.input(), type);
            case ReductionOp op -> containsNodeType(op.input(), type);
            case GatherOp op -> containsNodeType(op.input(), type) || containsNodeType(op.indices(), type);
            case ViewTransform vt -> containsNodeType(vt.input(), type);
            case Contiguous contig -> containsNodeType(contig.input(), type);
            default -> false;
        };
    }

    private static float[] range(int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = i;
        }
        return values;
    }

    private static float[] reverseRange(int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = size - 1 - i;
        }
        return values;
    }
}
