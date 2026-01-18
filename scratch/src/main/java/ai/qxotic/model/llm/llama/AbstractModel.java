package ai.qxotic.model.llm.llama;

import ai.qxotic.model.llm.Model;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;

public abstract class AbstractModel<Configuration, Weights, State>
        implements Model<Configuration, Weights, State> {

    protected final Configuration configuration;

    protected final KernelOps<FloatSpan, FloatMatrixView> kernelOps;
    protected final FloatSpanFactory<? extends FloatSpan> spanFactory;

    protected AbstractModel(
            Configuration configuration,
            KernelOps<FloatSpan, FloatMatrixView> kernelOps,
            FloatSpanFactory<? extends FloatSpan> spanFactory) {
        this.configuration = configuration;
        this.kernelOps = kernelOps;
        this.spanFactory = spanFactory;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }
}
