package ai.qxotic.jota;

import ai.qxotic.jota.tensor.ExecutionContext;
import ai.qxotic.jota.tensor.ExecutionContexts;
import ai.qxotic.jota.tensor.KernelRegistry;

public final class Jota {

    private Jota() {}

    public static ExecutionContext defaultExecutionContext() {
        return ExecutionContexts.defaultContext();
    }

    public static KernelRegistry kernelRegistry() {
        return ExecutionContexts.globalRegistry();
    }
}
