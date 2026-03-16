package com.qxotic.model.llm.phi3;

import com.qxotic.model.llm.llama.FloatSpanFactory;
import com.qxotic.model.llm.llama.Llama;
import com.qxotic.span.FloatMatrixView;
import com.qxotic.span.FloatSpan;
import com.qxotic.span.KernelOps;

public class Phi3 extends Llama {
    public Phi3(
            Configuration configuration,
            KernelOps<FloatSpan, FloatMatrixView> kernelOps,
            FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
