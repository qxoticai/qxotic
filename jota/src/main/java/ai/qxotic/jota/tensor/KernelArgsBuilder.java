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
            MemoryView<?> view = inputTensor.tryGetMaterialized().orElseGet(inputTensor::materialize);
            args.addBuffer(view);
        }
        args.addBuffer(output);
        long n = output.shape().size();
        if (n > Integer.MAX_VALUE) {
            throw new UnsupportedOperationException("Kernel args builder supports int32 sizes only");
        }
        args.addScalar((int) n, DataType.I32);
        return args;
    }
}
