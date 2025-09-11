package com.llm4j.model.gemma3;

import com.llm4j.model.llama.FloatSpanFactory;
import com.llm4j.model.llama.Llama;
import com.llm4j.span.FloatMatrixView;
import com.llm4j.span.FloatSpan;
import com.llm4j.span.KernelOps;

public class Gemma3 extends Llama {
    public Gemma3(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
