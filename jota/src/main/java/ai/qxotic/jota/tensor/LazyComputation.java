package ai.qxotic.jota.tensor;

import ai.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;

public sealed interface LazyComputation
        permits ExpressionComputation, ConstantComputation, RangeComputation, IRComputation {

    Op operation();

    List<Tensor> inputs();

    Map<String, Object> attributes();

    MemoryView<?> execute();
}
