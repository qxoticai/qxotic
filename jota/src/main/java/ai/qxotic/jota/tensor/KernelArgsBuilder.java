package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;

public final class KernelArgsBuilder {

    public KernelArgs build(ExpressionGraph graph, List<Tensor> inputs, MemoryView<?> output) {
        KernelArgs args = new KernelArgs();
        List<InputNode> inputNodes = graph.inputs();
        for (InputNode inputNode : inputNodes) {
            Tensor inputTensor = inputs.get(inputNode.index());
            ConstantComputation constant =
                    inputTensor
                            .computation()
                            .filter(ConstantComputation.class::isInstance)
                            .map(ConstantComputation.class::cast)
                            .orElse(null);
            if (constant != null && inputTensor.isScalar()) {
                args.addScalarBits(constant.rawBits(), inputTensor.dataType());
                continue;
            }
            MemoryView<?> view =
                    inputTensor.tryGetMaterialized().orElseGet(inputTensor::materialize);
            args.addBuffer(view);
        }
        args.addBuffer(output);
        long n = output.shape().size();
        if (n > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException(
                    "Kernel args builder supports int32 sizes only");
        }
        args.addScalar((int) n, DataType.I32);
        return args;
    }

    public DataType[] buildSignature(
            ExpressionGraph graph,
            List<Tensor> inputs,
            java.util.Map<Integer, Tensor> inputTensorMap) {
        List<InputNode> inputNodes = graph.inputs();
        DataType[] signature = new DataType[inputNodes.size() + 2];
        for (InputNode inputNode : inputNodes) {
            Tensor inputTensor = inputTensorMap.get(inputNode.index());
            ConstantComputation constant =
                    inputTensor
                            .computation()
                            .filter(ConstantComputation.class::isInstance)
                            .map(ConstantComputation.class::cast)
                            .orElse(null);
            if (constant != null && inputTensor.isScalar()) {
                signature[inputNode.index()] = inputTensor.dataType();
            }
        }
        signature[signature.length - 2] = null;
        signature[signature.length - 1] = DataType.I32;
        return signature;
    }
}
