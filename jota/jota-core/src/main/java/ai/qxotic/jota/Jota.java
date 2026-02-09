package ai.qxotic.jota;

import ai.qxotic.jota.tensor.ExecutionContext;
import ai.qxotic.jota.tensor.ExecutionContexts;

public final class Jota {

    private Jota() {}

    public static ExecutionContext defaultExecutionContext() {
        return ExecutionContexts.defaultContext();
    }
}
