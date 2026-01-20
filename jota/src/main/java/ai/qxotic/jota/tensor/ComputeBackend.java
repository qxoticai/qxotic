package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryView;
import java.util.List;

public interface ComputeBackend {

    Device device();

    MemoryView<?> execute(ExpressionGraph graph, List<Tensor> inputs);
}
