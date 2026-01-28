package ai.qxotic.jota;

import ai.qxotic.jota.tensor.*;

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
        //                (MemoryView<MemorySegment>) ComputeEngineContext.with(
        //                        new JavaComputeEngine((MemoryContext<MemorySegment>) context),
        //                        x2::materialize);
        //
        //        System.out.println(view.toString(context.memoryAccess()));
        //
        //        EagerTensorOps ops = new EagerTensorOps(context);
        //
        //        Tensor PI = Tensor.scalar((float) Math.PI);
        //        Tensor result2 = TensorOpsContext.with(ops, () -> PI.sqrt());
        //
        //        System.out.println(result2); // view.toString(conte0xt.memoryAccess()));

        Tensor x = Tensor.of(new float[] {0, 1, 2, 3, 4, 5}).view(Shape.of(2, 3)); // [2x3]
        Tensor y = x.add(x).sqrt(); // lazy ops chain
        System.out.println("isLazy=" + y.isLazy());
        // Trigger execution
        var out = y.materialize();
        System.out.println(out);
    }

    static Tensor add2(Tensor in) {
        return in.add(Tensor.scalar(2f));
    }
}
