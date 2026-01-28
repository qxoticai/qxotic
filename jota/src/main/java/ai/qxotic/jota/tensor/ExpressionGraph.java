package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import java.util.List;
import java.util.Optional;

public record ExpressionGraph(
        ExprNode root, List<InputNode> inputs, java.util.Map<Integer, Tensor> inputTensorMap) {

    public ExpressionGraph {
        inputs = List.copyOf(inputs);
        inputTensorMap = java.util.Map.copyOf(inputTensorMap);
    }

    public java.util.Map<Integer, Tensor> inputTensorMap() {
        return inputTensorMap;
    }

    public Optional<ReductionInfo> reductionRoot() {
        if (root instanceof ReductionNode reduction) {
            return Optional.of(
                    new ReductionInfo(
                            reduction.op(),
                            reduction.axis(),
                            reduction.keepDims(),
                            reduction.input(),
                            reduction.layout(),
                            reduction.dataType(),
                            reduction.device()));
        }
        return Optional.empty();
    }

    public record ReductionInfo(
            ReductionOp op,
            int axis,
            boolean keepDims,
            ExprNode input,
            Layout layout,
            DataType dataType,
            Device device) {}
}
