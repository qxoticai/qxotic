package com.llm4j.model.granite;

import com.llm4j.model.llama.FloatSpanFactory;
import com.llm4j.model.llama.Llama;
import com.llm4j.span.FloatMatrixView;
import com.llm4j.span.FloatSpan;
import com.llm4j.span.KernelOps;

public class Granite extends Llama {
    public Granite(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
