package com.llm4j.model.llama;

import com.llm4j.api.model.Model;
import com.llm4j.span.FloatMatrixView;
import com.llm4j.span.FloatSpan;
import com.llm4j.span.KernelOps;

public abstract class AbstractModel<Configuration, Weights, State> implements Model<Configuration, Weights, State> {

    protected final Configuration configuration;

    protected final KernelOps<FloatSpan, FloatMatrixView> kernelOps;
    protected final FloatSpanFactory<? extends FloatSpan> spanFactory;

    protected AbstractModel(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        this.configuration = configuration;
        this.kernelOps = kernelOps;
        this.spanFactory = spanFactory;
    }

    @Override
    public Configuration configuration() {
        return configuration;
    }
}
