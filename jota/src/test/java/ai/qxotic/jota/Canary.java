package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;

public class Canary {
    static void main() {
        Tensor x = Tensor.iota(6, DataType.FP32).view(Shape.of(2, 3)); // [2x3]
        Tensor result =
                Tracer.trace(x, Tensor.scalar(2f), (a, b) -> a.add(b).add(Tensor.scalar(3.14f)));
        // Trigger execution
        var out = result.materialize();
        MemoryAccess access = Environment.current().nativeMemoryDomain().directAccess();
        System.out.println(out.toString(access));
    }
}
