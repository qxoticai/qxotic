package com.qxotic.model.llm.llama;

import com.qxotic.model.llm.Model;
import com.qxotic.span.FloatMatrixView;
import com.qxotic.span.FloatSpan;
import com.qxotic.span.KernelOps;

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
