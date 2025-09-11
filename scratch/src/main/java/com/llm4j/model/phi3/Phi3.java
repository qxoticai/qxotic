package com.llm4j.model.phi3;

import com.llm4j.model.llama.FloatSpanFactory;
import com.llm4j.model.llama.Llama;
import com.llm4j.span.FloatMatrixView;
import com.llm4j.span.FloatSpan;
import com.llm4j.span.KernelOps;

public class Phi3 extends Llama {
    public Phi3(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
