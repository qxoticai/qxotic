package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.tir.KernelStep;
import ai.qxotic.jota.ir.tir.ScheduleInputRef;
import ai.qxotic.jota.ir.tir.ScheduledOutputRef;
import ai.qxotic.jota.ir.tir.ScheduledProgram;
import ai.qxotic.jota.ir.tir.ValueId;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryView;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class ScheduledExecutor {

    MemoryView<?> execute(
            ScheduledProgram program, ComputeEngine computeEngine, List<Tensor> graphInputs) {
        Objects.requireNonNull(program, "program");
        Objects.requireNonNull(graphInputs, "graphInputs");

        if (!program.steps().isEmpty() && computeEngine == null) {
            throw new IllegalStateException(
                    "No compute engine available for scheduled execution with "
                            + program.steps().size()
                            + " kernel steps");
        }

        Map<ValueId, MemoryView<?>> produced = new HashMap<>();
        for (int i = 0; i < program.steps().size(); i++) {
            KernelStep step = program.steps().get(i);
            List<Tensor> stepInputs = resolveStepInputs(step.inputs(), graphInputs, produced);
            MemoryView<?> output = computeEngine.execute(step.graph(), stepInputs);
            boolean isIntermediate = i < program.steps().size() - 1;
            if (isIntermediate) {
                output = ensureDeviceView(output, computeEngine.device());
            }
            produced.put(step.output(), output);
        }

        return switch (program.output()) {
            case ScheduledOutputRef.ValueOutput value -> getProduced(produced, value.valueId());
            case ScheduledOutputRef.TensorInputOutput input ->
                    inputAt(graphInputs, input.inputId()).materialize();
            case ScheduledOutputRef.ScalarInputOutput input ->
                    inputAt(graphInputs, input.inputId()).materialize();
        };
    }

    private static List<Tensor> resolveStepInputs(
            List<ScheduleInputRef> refs,
            List<Tensor> graphInputs,
            Map<ValueId, MemoryView<?>> produced) {
        List<Tensor> resolved = new ArrayList<>(refs.size());
        for (ScheduleInputRef ref : refs) {
            Tensor tensor =
                    switch (ref) {
                        case ScheduleInputRef.TensorInputRef input ->
                                inputAt(graphInputs, input.inputId());
                        case ScheduleInputRef.ScalarInputRef input ->
                                inputAt(graphInputs, input.inputId());
                        case ScheduleInputRef.ProducedValueRef value ->
                                Tensor.of(getProduced(produced, value.valueId()));
                    };
            resolved.add(tensor);
        }
        return resolved;
    }

    private static Tensor inputAt(List<Tensor> graphInputs, int inputId) {
        if (inputId < 0 || inputId >= graphInputs.size()) {
            throw new IllegalStateException(
                    "Scheduled input id "
                            + inputId
                            + " is out of bounds for "
                            + graphInputs.size()
                            + " provided inputs");
        }
        return graphInputs.get(inputId);
    }

    private static MemoryView<?> getProduced(Map<ValueId, MemoryView<?>> produced, ValueId id) {
        MemoryView<?> view = produced.get(id);
        if (view == null) {
            throw new IllegalStateException("Missing produced value for " + id);
        }
        return view;
    }

    @SuppressWarnings("unchecked")
    private static MemoryView<?> ensureDeviceView(MemoryView<?> view, Device targetDevice) {
        if (view.memory().device().equals(targetDevice)) {
            return view;
        }

        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>) Environment.current().runtimeFor(view.memory().device()).memoryDomain();
        MemoryDomain<Object> dstDomain =
                (MemoryDomain<Object>) Environment.current().runtimeFor(targetDevice).memoryDomain();
        MemoryView<Object> srcView = (MemoryView<Object>) view;
        Memory<Object> dstMemory = dstDomain.memoryAllocator().allocateMemory(view.dataType(), view.shape());
        MemoryView<Object> dstView =
                (MemoryView<Object>)
                        MemoryView.of(
                                dstMemory,
                                view.dataType(),
                                Layout.rowMajor(view.shape()));
        MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
        return dstView;
    }
}
