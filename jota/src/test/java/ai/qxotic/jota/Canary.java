package ai.qxotic.jota;

import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;

public class Canary {
    static void main() {

        //        Environment environment =
        //                new Environment(
        //                        Device.PANAMA, DataType.FP32, Environment.global().backends(),
        // ExecutionMode.LAZY);
        //        MemoryContext<MemorySegment> context = (MemoryContext<MemorySegment>)
        // Environment.current().nativeBackend().memoryContext();
        //
        //        var two = Tensor.scalar(2f);
        //        var result =
        //                Environment.with(
        //                        environment,
        //                        () -> {
        //                            //var t = Tensor.broadcasted(3.14f, Shape.of(30, 30));
        //                            //t = t.multiply(t);
        //                            return Tensor.of(new
        // float[]{3.14f}).view(Shape.scalar()).multiply(two);
        //                        });
        //
        //        Tensor x = Tensor.of(new float[]{3.14f, 5.1f});
        //        var x2 = add2(x);
        //
        //        MemoryView<MemorySegment> view =
        //                (MemoryView<MemorySegment>) x2.materialize();
        //
        //        System.out.println(view.toString(context.memoryAccess()));
        //
        //        EagerTensorOps ops = new EagerTensorOps(context);
        //
        //        Tensor PI = Tensor.scalar((float) Math.PI);
        //        Tensor result2 = TensorOpsContext.with(ops, () -> PI.sqrt());
        //
        //        System.out.println(result2); // view.toString(conte0xt.memoryAccess()));

        Tensor x = Tensor.arange(6, DataType.FP32).view(Shape.of(2, 3)); // [2x3]
        Tensor result =
                Tracer.trace(x, Tensor.scalar(2f), (a, b) -> a.add(b).add(Tensor.scalar(3.14f)));
        // Trigger execution
        var out = result.materialize();
        MemoryAccess access = Environment.current().panamaContext().memoryAccess();
        System.out.println(out.toString(access));
    }
}
