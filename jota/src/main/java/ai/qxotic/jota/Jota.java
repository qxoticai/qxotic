package ai.qxotic.jota;

import ai.qxotic.jota.tensor.OptimizingCallSite;
import ai.qxotic.jota.tensor.Tracer;
import ai.qxotic.jota.tensor.Tensor;
import java.util.function.Function;

public final class Jota {

    private Jota() {}

    public static OptimizingCallSite createOptimizingCallSite(Function<Tensor, Tensor> function) {
        return Tracer.createOptimizingCallSite(function);
    }
}
