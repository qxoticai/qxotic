package ai.qxotic.model.llm.apertus;

import ai.qxotic.model.llm.llama.FloatSpanFactory;
import ai.qxotic.model.llm.llama.Llama;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
import ai.qxotic.span.KernelOps;

public class Apertus extends Llama {
    public Apertus(Configuration configuration, KernelOps<FloatSpan, FloatMatrixView> kernelOps, FloatSpanFactory<? extends FloatSpan> spanFactory) {
        super(configuration, kernelOps, spanFactory);
    }
}
