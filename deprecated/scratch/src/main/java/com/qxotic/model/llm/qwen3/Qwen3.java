package com.qxotic.model.llm.qwen3;

import com.qxotic.model.llm.llama.FloatSpanFactory;
import com.qxotic.model.llm.llama.Llama;
import com.qxotic.span.FloatMatrixView;
import com.qxotic.span.FloatSpan;
import com.qxotic.span.KernelOps;

public class Qwen3 extends Llama {
    public Qwen3(
            Configuration configuration,
            KernelOps<FloatSpan, FloatMatrixView> kernelOps,
            FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
