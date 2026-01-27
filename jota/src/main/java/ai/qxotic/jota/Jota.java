package ai.qxotic.jota;

import ai.qxotic.jota.tensor.ExecutionContext;
import ai.qxotic.jota.tensor.ExecutionContexts;
import ai.qxotic.jota.tensor.KernelRegistry;
import ai.qxotic.jota.tensor.OptimizingCallSite;
import ai.qxotic.jota.tensor.Tensor;
import ai.qxotic.jota.tensor.Tracer;
import java.util.function.Function;

public final class Jota {

    private Jota() {}

    public static OptimizingCallSite createOptimizingCallSite(Function<Tensor, Tensor> function) {
        return Tracer.createOptimizingCallSite(function);
    }

    public static ExecutionContext defaultExecutionContext() {
        return ExecutionContexts.defaultContext();
    }

    public static KernelRegistry kernelRegistry() {
        return ExecutionContexts.globalRegistry();
    }
}
