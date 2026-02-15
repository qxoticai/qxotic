package ai.qxotic.jota.ir.tir;

import static org.junit.jupiter.api.Assertions.*;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class TIRSchedulePassTest {

    @Test
    void schedulesSimpleChainIntoKernelSteps() {
        Layout layout = Layout.rowMajor(Shape.of(4, 4));
        TensorInput x = new TensorInput(0, DataType.FP32, layout);
        TensorInput y = new TensorInput(1, DataType.FP32, layout);

        BinaryOp mul = new BinaryOp(BinaryOperator.MULTIPLY, x, y, layout.shape());
        BinaryOp add = new BinaryOp(BinaryOperator.ADD, mul, y, layout.shape());
        TIRGraph graph = new TIRGraph(List.of(x, y), List.of(add));

        ScheduledProgram schedule = new TIRSchedulePass().run(graph);

        assertEquals(2, schedule.steps().size(), "Expected one kernel per compute stage");
        KernelStep first = schedule.steps().get(0);
        KernelStep second = schedule.steps().get(1);

        assertTrue(first.inputs().contains(new ScheduleInputRef.TensorInputRef(0)));
        assertTrue(first.inputs().contains(new ScheduleInputRef.TensorInputRef(1)));
        assertTrue(second.inputs().contains(new ScheduleInputRef.ProducedValueRef(first.output())));
        assertTrue(second.inputs().contains(new ScheduleInputRef.TensorInputRef(1)));
        assertEquals(
                new ScheduledOutputRef.ValueOutput(second.output()),
                schedule.output(),
                "Final output should come from last kernel step");
    }

    @Test
    void schedulesGatherKernelStep() {
        Layout tableLayout = Layout.rowMajor(Shape.of(8, 4));
        Layout indicesLayout = Layout.rowMajor(Shape.of(2, 3));
        TensorInput table = new TensorInput(0, DataType.FP32, tableLayout);
        TensorInput indices = new TensorInput(1, DataType.I32, indicesLayout);
        GatherOp gather =
                new GatherOp(
                        table,
                        indices,
                        0,
                        GatherOp.computeOutputShape(table.shape(), indices.shape(), 0));
        TIRGraph graph = new TIRGraph(List.of(table, indices), List.of(gather));

        ScheduledProgram schedule = new TIRSchedulePass().run(graph);

        assertEquals(1, schedule.steps().size());
        KernelStep step = schedule.steps().getFirst();
        assertTrue(step.inputs().contains(new ScheduleInputRef.TensorInputRef(0)));
        assertTrue(step.inputs().contains(new ScheduleInputRef.TensorInputRef(1)));
        assertEquals(new ScheduledOutputRef.ValueOutput(step.output()), schedule.output());
    }

    @Test
    void splitsKernelWhenGraphContainsTwoReductions() {
        Layout layout = Layout.rowMajor(Shape.of(3, 4));
        TensorInput input = new TensorInput(0, DataType.FP32, layout);

        ReductionOp first =
                new ReductionOp(
                        ReductionOperator.SUM,
                        input,
                        new int[] {1},
                        false,
                        DataType.FP32);
        ReductionOp second =
                new ReductionOp(
                        ReductionOperator.SUM,
                        first,
                        new int[] {0},
                        false,
                        DataType.FP32);
        TIRGraph graph = new TIRGraph(List.of(input), List.of(second));

        ScheduledProgram schedule = new TIRSchedulePass().run(graph);

        assertEquals(2, schedule.steps().size(), "Expected split because each kernel supports at most one reduction");
        KernelStep firstStep = schedule.steps().get(0);
        KernelStep secondStep = schedule.steps().get(1);
        assertTrue(secondStep.inputs().contains(new ScheduleInputRef.ProducedValueRef(firstStep.output())));
    }
}
